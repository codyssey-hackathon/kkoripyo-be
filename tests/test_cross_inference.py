import copy
import json
from pathlib import Path

import pytest

from app.cross_inference import (
    ContractError,
    agent_view,
    build_result,
    combine_detections,
    inference_rules,
    review_detected_items,
    validate_and_submit,
)
from app.cross_inference.rules import verify_candidates
from app.cross_inference.schema import ensure_detection

PRIMARY = json.loads(
    (Path(__file__).parent / "fixtures" / "detection_gildong_hong.json").read_text(encoding="utf-8")
)
IMAGE_POSTS = {"gildong_hong": {"fake_card.png": "FAKE001"}}


def item(
    id_, typ, value, input_id, conf, category="일반개인정보", normalized=None, modality="text"
):
    return {
        "id": id_,
        "type": typ,
        "category": category,
        "value": value,
        "normalized": normalized if normalized is not None else value,
        "evidence": f"{value} 근거",
        "source": {"input_id": input_id, "modality": modality, "method": "regex", "bbox": None},
        "confidence": conf,
    }


def entity(id_, typ, value, item_ids, conf):
    return {
        "entity_id": id_,
        "type": typ,
        "value": value,
        "item_ids": item_ids,
        "sources": [],
        "match_reasons": ["single"],
        "confidence": conf,
    }


def detection(subject, items, entities):
    return {
        "schema_version": "json2/v1",
        "subject": subject,
        "llm_used": False,
        "items": items,
        "entities": entities,
    }


# 가상의 주변 계정: NB001 게시물에서 본인을 언급, 나머지 게시물·프로필은 본인과 연결 없음
NEIGHBOR = detection(
    "chulsoo_kim",
    [
        item("I1", "sns_account", "chulsoo_kim", "profile.username", 1.0),
        item(
            "I2",
            "sns_account",
            "@gildong_hong",
            "post:NB001.caption",
            0.9,
            normalized="gildong_hong",
        ),
        item("I3", "location", "가평", "post:NB001.caption", 0.7, category="준식별자"),
        item("I4", "phone", "010-1234-5678", "post:NB002.caption", 0.9, normalized="01012345678"),
        item(
            "I5",
            "affiliation",
            "서울대학교 경영학과",
            "profile.biography",
            0.7,
            category="준식별자",
        ),
        item("I6", "sensitive", "병원 진료", "post:NB001.caption", 0.8, category="민감정보"),
        item("I7", "location", "부산", "post:NB003.caption", 0.7, category="준식별자"),
        item("I8", "location", "가평", "post:NB005.caption", 0.7, category="준식별자"),
        item(
            "I9",
            "affiliation",
            "한국대학교 컴퓨터공학과",
            "post:NB001.caption",
            0.7,
            category="준식별자",
            normalized="한국대학교컴퓨터공학과",
        ),
    ],
    [
        entity("E1", "sns_account", "chulsoo_kim", ["I1"], 1.0),
        entity("E2", "sns_account", "@gildong_hong", ["I2"], 0.9),
        entity("E3", "location", "가평", ["I3", "I8"], 0.7),
        entity("E4", "phone", "010-1234-5678", ["I4"], 0.9),
        entity("E5", "affiliation", "서울대학교 경영학과", ["I5"], 0.7),
        entity("E6", "sensitive", "병원 진료", ["I6"], 0.8),
        entity("E7", "location", "부산", ["I7"], 0.7),
        entity("E8", "affiliation", "한국대학교 컴퓨터공학과", ["I9"], 0.7),
    ],
)


def cand(rule_id, typ, value, status, info_ids):
    return {"rule_id": rule_id, "type": typ, "value": value, "status": status, "info_ids": info_ids}


# 본인 소속(프로필)을 뒷받침으로, 본인을 언급한 주변 계정 게시물의 장소를 추론
GAPYEONG = cand("R03", "location", "가평", "inferred", ["gildong_hong:I6", "chulsoo_kim:I3"])


def make_bundle(*extra, links=None, primary=None, **review):
    return combine_detections(
        copy.deepcopy(primary or PRIMARY),
        copy.deepcopy(NEIGHBOR),
        *extra,
        links=links,
        image_posts=IMAGE_POSTS,
        **review,
    )


@pytest.fixture()
def bundle():
    return make_bundle()


def find(result, typ, value):
    return [f for f in result["findings"] if f["type"] == typ and f["value"] == value]


def verify(bundle, *candidates):
    return verify_candidates(bundle, list(candidates))


def test_sample_detection_is_valid():
    ensure_detection(PRIMARY)


def test_combine_namespaces_ids_and_source_ids(bundle):
    assert bundle["subject"] == "gildong_hong"
    assert [a["scope"] for a in bundle["accounts"]] == ["primary", "secondary"]
    by_id = {it["info_id"]: it for it in bundle["items"]}
    assert by_id["gildong_hong:I4"]["source_id"] == "gildong_hong:profile:biography:0"
    assert by_id["gildong_hong:I8"]["source_id"] == "gildong_hong:FAKE001:comment:0"
    assert by_id["gildong_hong:I11"]["source_id"] == "gildong_hong:FAKE001:image:fake_card.png"
    assert by_id["gildong_hong:I11"]["unit"] == "post:FAKE001"
    assert by_id["chulsoo_kim:I3"]["entity_id"] == "chulsoo_kim:E3"
    assert bundle["links"] == [
        {
            "account_id": "chulsoo_kim",
            "kind": "mentioned",
            "unit": "post:NB001",
            "source_id": "chulsoo_kim:NB001:caption:0",
        }
    ]


def test_combine_rejects_bad_input():
    with pytest.raises(ContractError):
        combine_detections(copy.deepcopy(PRIMARY), copy.deepcopy(PRIMARY))
    with pytest.raises(ContractError):
        combine_detections(
            copy.deepcopy(PRIMARY),
            links=[{"account_id": "nobody", "post_id": "X", "kind": "tagged"}],
        )
    with pytest.raises(ContractError):
        make_bundle(links=[{"account_id": "chulsoo_kim", "kind": "liked"}])
    with pytest.raises(ContractError):
        combine_detections({**PRIMARY, "schema_version": "1.0"})
    with pytest.raises(ContractError):
        combine_detections({**PRIMARY, "subject": None})
    with pytest.raises(ContractError):
        combine_detections({**PRIMARY, "subject": 5})
    with pytest.raises(ContractError):
        make_bundle(links=["chulsoo_kim"])
    bad = copy.deepcopy(PRIMARY)
    bad["items"][0]["normalized"] = 123
    with pytest.raises(ContractError):
        combine_detections(bad)


def test_subject_verified_by_profile_and_id_card(bundle):
    result = build_result(bundle)
    values = {(f["type"], f["value"]) for f in result["findings"]}
    assert {
        ("sns_account", "gildong_hong"),
        ("name", "홍길동"),
        ("phone", "010.1234.5678"),
        ("email", "gildong.hong@example.com"),
        ("affiliation", "한국대학교 컴퓨터공학과"),
        # 학생증 이미지에서 본인 이름과 함께 나온 값
        ("student_id", "20261234"),
    } <= values
    # 같은 이미지에 기타 단서가 둘(재학, 2 학년)이라 누구 것인지 확정하지 않는다
    assert all(f["type"] != "other_clue" for f in result["findings"])
    # 이름 규칙 오탐은 confidence와 무관하게 본인 정보로 승격되지 않는다
    assert ("name", "길동아") not in values
    assert ("name", "연락처") not in values
    unverified = {u["info_id"] for u in result["unverified"]}
    assert {"gildong_hong:I9", "gildong_hong:I18", "gildong_hong:I14"} <= unverified
    assert all("value" not in u for u in result["unverified"])
    # 주변 계정에만 있는 정보는 본인 정보로 집계하지 않는다
    assert ("location", "가평") not in values
    assert ("affiliation", "서울대학교 경영학과") not in values
    assert all(f["status"] == "confirmed" for f in result["findings"])
    # 탐지 단계에서 로마자 표기로 묶은 이름은 같은 finding으로 합쳐진다
    [name] = find(result, "name", "홍길동")
    assert "gildong_hong:I7" in name["evidence_ids"]


def test_regex_or_comment_alone_does_not_confirm():
    primary = copy.deepcopy(PRIMARY)
    primary["items"] += [
        # 다른 사람이 단 댓글의 처음 보는 번호
        item(
            "I20", "phone", "010-9999-0000", "post:FAKE001.comment1", 0.95, normalized="01099990000"
        ),
        # 본인 게시물 본문의 다른 사람 이름
        item("I22", "name", "김영희", "post:FAKE002.caption", 0.9),
    ]
    primary["entities"] += [
        entity("E20", "phone", "010-9999-0000", ["I20"], 0.95),
        entity("E22", "name", "김영희", ["I22"], 0.9),
    ]
    result = build_result(make_bundle(primary=primary))
    assert not find(result, "phone", "010-9999-0000")
    assert not find(result, "name", "김영희")
    assert {"gildong_hong:I20", "gildong_hong:I22"} <= {u["info_id"] for u in result["unverified"]}


def test_roster_image_values_are_not_attributed():
    primary = copy.deepcopy(PRIMARY)
    primary["items"] += [
        item("I30", "name", "홍길동", "image:roster.png", 0.5, modality="image"),
        item("I31", "student_id", "20260001", "image:roster.png", 0.6, modality="image"),
        item("I32", "student_id", "20260002", "image:roster.png", 0.6, modality="image"),
    ]
    primary["entities"] += [
        entity("E30", "name", "홍길동", ["I30"], 0.5),
        entity("E31", "student_id", "20260001", ["I31"], 0.6),
        entity("E32", "student_id", "20260002", ["I32"], 0.6),
    ]
    result = build_result(make_bundle(primary=primary))
    assert [f["value"] for f in result["findings"] if f["type"] == "student_id"] == ["20261234"]


def test_similar_handle_is_not_subject_account():
    neighbor = copy.deepcopy(NEIGHBOR)
    neighbor["items"].append(
        item(
            "I40",
            "sns_account",
            "@gildong.hong",
            "post:NB009.caption",
            0.9,
            normalized="gildong.hong",
        )
    )
    neighbor["entities"].append(entity("E40", "sns_account", "@gildong.hong", ["I40"], 0.9))
    result = build_result(combine_detections(copy.deepcopy(PRIMARY), neighbor))
    [account] = [f for f in result["findings"] if f["type"] == "sns_account"]
    assert "chulsoo_kim:I40" not in account["evidence_ids"]


def test_same_value_on_neighbor_is_added_as_source(bundle):
    result = build_result(bundle)
    [phone] = find(result, "phone", "010.1234.5678")
    assert {
        "gildong_hong:profile:biography:0",
        "gildong_hong:FAKE001:comment:0",
        "gildong_hong:FAKE001:image:fake_card.png",
        # 전화번호는 본인과 연결 없는 게시물이어도 같은 값이면 출처로 추가
        "chulsoo_kim:NB002:caption:0",
    } <= set(phone["source_ids"])
    assert set(phone["evidence_ids"]) >= {"gildong_hong:I4", "chulsoo_kim:I4"}
    # 소속처럼 흔한 값은 본인과 연결된 게시물(NB001)일 때만
    [aff] = find(result, "affiliation", "한국대학교 컴퓨터공학과")
    assert "chulsoo_kim:I9" in aff["evidence_ids"]


def test_sensitive_only_flagged_for_subject(bundle):
    result = build_result(bundle)
    assert result["sensitive_detected"] is False
    assert result["sensitive_flags"] == []
    assert "병원" not in json.dumps(result, ensure_ascii=False)


def test_result_has_no_score_fields(bundle):
    result = build_result(bundle)
    assert not {"score", "scores", "risk"} & set(result)
    assert result["findings"][0]["finding_id"] == "finding_0001"


def test_inferred_location_from_linked_neighbor_post(bundle):
    accepted, rejected = verify(bundle, GAPYEONG)
    assert rejected == []
    result = build_result(bundle, accepted)
    [f] = find(result, "location", "가평")
    assert f["status"] == "inferred" and f["source_scope"] == "cross_account"
    assert f["evidence_ids"] == ["gildong_hong:I6", "chulsoo_kim:I3"]
    assert f["source_ids"] == ["gildong_hong:profile:biography:0", "chulsoo_kim:NB001:caption:0"]


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        # 본인 언급이 없는 주변 계정 게시물
        (
            cand("R03", "location", "부산", "inferred", ["gildong_hong:I6", "chulsoo_kim:I7"]),
            "no_subject_link",
        ),
        # 주변 계정 주인의 소속을 본인 것으로 옮기기
        (
            cand(
                "R02",
                "affiliation",
                "서울대학교 경영학과",
                "inferred",
                ["gildong_hong:I6", "chulsoo_kim:I5"],
            ),
            "no_subject_link",
        ),
        (
            cand(
                "R04", "phone", "010-1234-5678", "inferred", ["gildong_hong:I1", "chulsoo_kim:I4"]
            ),
            "not_inferable_type",
        ),
        (
            cand("R03", "location", "제주", "inferred", ["gildong_hong:I6", "chulsoo_kim:I3"]),
            "value_not_in_evidence",
        ),
        (cand("R01", "name", "길동아", "confirmed", ["gildong_hong:I9"]), "not_subject_verified"),
        (
            cand("R01", "name", "병원 진료", "inferred", ["gildong_hong:I1", "chulsoo_kim:I6"]),
            "sensitive_info",
        ),
        (
            cand("R03", "location", "가평", "confirmed", ["chulsoo_kim:I3"]),
            "not_subject_verified",
        ),
        # 계정 ID는 장소 추론의 뒷받침 근거가 아니다
        (
            cand("R03", "location", "가평", "inferred", ["gildong_hong:I1", "chulsoo_kim:I3"]),
            "support_gate",
        ),
        (
            cand("R01", "location", "가평", "inferred", ["gildong_hong:I6", "chulsoo_kim:I3"]),
            "rule_type_mismatch",
        ),
        (cand("R01", "name", "홍길동", "confirmed", ["I2"]), "unknown_info_id"),
        (cand("R01", "name", "홍길동", "confirmed", ["gildong_hong:E2"]), "unknown_info_id"),
        (cand("R01", "name", "홍길동", "confirmed", "gildong_hong:I2"), "invalid_info_ids"),
        ({**GAPYEONG, "value": ["가평"]}, "invalid_value"),
        (cand("R05", "rrn", "x", "confirmed", ["gildong_hong:I1"]), "hidden_type"),
        # 본인 계정에서 이미 확인된 소속을 다시 '추론'으로 제출
        (
            cand(
                "R02",
                "affiliation",
                "한국대학교 컴퓨터공학과",
                "inferred",
                ["gildong_hong:I6", "chulsoo_kim:I9"],
            ),
            "already_confirmed",
        ),
        (cand("R08", "relationship", "nobody", "inferred", ["gildong_hong:I1"]), "unknown_target"),
        (
            cand("R08", "relationship", "chulsoo_kim", "confirmed", ["gildong_hong:I1"]),
            "invalid_status",
        ),
        (
            cand(
                "R07",
                "activity_pattern",
                "매주 가평 방문",
                "inferred",
                ["gildong_hong:I1", "chulsoo_kim:I3", "chulsoo_kim:I8"],
            ),
            # NB005는 본인과 연결되지 않아 반복 횟수에 들어가지 않는다
            "pattern_needs_repeat",
        ),
        (
            cand("R07", "activity_pattern", "매주 부산 방문", "inferred", ["chulsoo_kim:I7"]),
            "no_subject_link",
        ),
    ],
)
def test_invalid_candidates_are_rejected(bundle, candidate, reason):
    assert verify(bundle, candidate) == ([], [{"index": 0, "reason": reason}])
    assert build_result(bundle, [candidate]) == build_result(bundle)


def test_candidate_matching_confirmed_value_is_merged(bundle):
    # 로마자 표기는 이미 확인된 '홍길동'과 같은 정보이므로 별도 finding을 만들지 않는다
    roman = cand("R01", "name", "Hong Gildong", "confirmed", ["gildong_hong:I13"])
    result = build_result(bundle, [roman])
    assert [f["value"] for f in result["findings"] if f["type"] == "name"] == ["홍길동"]
    inferred = cand("R01", "name", "Hong Gildong", "inferred", ["gildong_hong:I13"])
    assert verify(bundle, inferred)[1] == [{"index": 0, "reason": "already_confirmed"}]


def test_unrelated_evidence_is_dropped(bundle):
    padded = {**GAPYEONG, "info_ids": [*GAPYEONG["info_ids"], "chulsoo_kim:I8", "chulsoo_kim:I5"]}
    [f] = find(build_result(bundle, [padded]), "location", "가평")
    assert f["evidence_ids"] == ["gildong_hong:I6", "chulsoo_kim:I3"]


def test_duplicate_candidate_is_rejected(bundle):
    accepted, rejected = verify(bundle, GAPYEONG, GAPYEONG)
    assert len(accepted) == 1
    assert rejected == [{"index": 1, "reason": "duplicate"}]


def test_backend_link_enables_inference():
    linked = make_bundle(
        links=[{"account_id": "chulsoo_kim", "post_id": "NB003", "kind": "tagged"}]
    )
    busan = cand("R03", "location", "부산", "inferred", ["gildong_hong:I6", "chulsoo_kim:I7"])
    result = build_result(linked, [busan])
    assert find(result, "location", "부산")[0]["status"] == "inferred"


def test_links_accept_only_subject_link_kinds():
    # 팔로워·팔로잉은 탐색 단서일 뿐 본인 연결(links)로 받지 않는다
    with pytest.raises(ContractError):
        make_bundle(links=[{"account_id": "chulsoo_kim", "post_id": "NB003", "kind": "follower"}])
    # post_id에 본인 게시물을 넣는 실수를 막는다
    with pytest.raises(ContractError):
        make_bundle(links=[{"account_id": "chulsoo_kim", "post_id": "FAKE001", "kind": "tagged"}])
    # 본인 게시물에 단 댓글처럼 주변 게시물이 없는 관계는 source_id만으로 받는다
    commented = make_bundle(
        links=[
            {
                "account_id": "chulsoo_kim",
                "kind": "commented",
                "source_id": "gildong_hong:FAKE001:comment:0",
            }
        ]
    )
    rel = cand("R08", "relationship", "chulsoo_kim", "inferred", ["chulsoo_kim:I1"])
    [f] = find(build_result(commented, [rel]), "relationship", "chulsoo_kim")
    assert f["relation_kinds"] == ["commented", "mentioned"]
    assert "gildong_hong:FAKE001:comment:0" in f["source_ids"]


def test_activity_pattern_needs_repeated_linked_posts():
    linked = make_bundle(
        links=[{"account_id": "chulsoo_kim", "post_id": "NB005", "kind": "tagged"}]
    )
    pattern = cand(
        "R07",
        "activity_pattern",
        "매주 가평 방문",
        "inferred",
        ["chulsoo_kim:I3", "chulsoo_kim:I8"],
    )
    result = build_result(linked, [pattern])
    [f] = find(result, "activity_pattern", "매주 가평 방문")
    assert f["rule_id"] == "R07" and f["source_scope"] == "cross_account"
    assert set(f["source_ids"]) == {"chulsoo_kim:NB001:caption:0", "chulsoo_kim:NB005:caption:0"}


def test_relationship_value_is_target_account(bundle):
    rel = cand("R08", "relationship", "@Chulsoo_Kim", "inferred", ["chulsoo_kim:I2"])
    accepted, rejected = verify(bundle, rel)
    assert rejected == []
    result = build_result(bundle, accepted)
    [f] = find(result, "relationship", "chulsoo_kim")
    assert f["relation_kinds"] == ["mentioned"]
    assert f["source_ids"] == ["chulsoo_kim:NB001:caption:0"]


def test_relationship_rejects_third_party_evidence():
    third = detection(
        "younghee",
        [item("I1", "sns_account", "younghee", "profile.username", 1.0)],
        [entity("E1", "sns_account", "younghee", ["I1"], 1.0)],
    )
    linked = make_bundle(third)
    rel = cand("R08", "relationship", "chulsoo_kim", "inferred", ["younghee:I1"])
    assert verify(linked, rel)[1] == [{"index": 0, "reason": "foreign_evidence"}]


def test_build_result_reverifies_candidates(bundle):
    forged = {**GAPYEONG, "info_ids": ["chulsoo_kim:I3"], "source_scope": "cross_account"}
    rejected = []
    result = build_result(bundle, [forged], rejected=rejected)
    assert rejected == [{"index": 0, "reason": "support_gate"}]
    assert not find(result, "location", "가평")


def test_validate_and_submit_returns_counts_only(bundle):
    bad = cand("R03", "location", "제주", "inferred", ["gildong_hong:I6", "chulsoo_kim:I3"])
    response, record = validate_and_submit(
        bundle, {"candidates": [GAPYEONG, bad]}, validation_ref="validation_0001"
    )
    assert response == {
        "validation_ref": "validation_0001",
        "accepted_count": 1,
        "rejected_count": 1,
        "status": "ok",
    }
    assert record["rejected"] == [{"index": 1, "reason": "value_not_in_evidence"}]
    assert "제주" not in json.dumps(record["rejected"], ensure_ascii=False)
    assert find(record["analysis"], "location", "가평")


def test_validate_and_submit_accumulates_previous(bundle):
    _, first = validate_and_submit(
        bundle, {"candidates": [GAPYEONG]}, validation_ref="validation_0001"
    )
    rel = cand("R08", "relationship", "chulsoo_kim", "inferred", ["chulsoo_kim:I2"])
    response, second = validate_and_submit(
        bundle,
        {"candidates": [GAPYEONG, rel]},
        validation_ref="validation_0002",
        previous=first["accepted"],
    )
    assert response["accepted_count"] == 1
    assert second["rejected"] == [{"index": 0, "reason": "duplicate"}]
    assert find(second["analysis"], "location", "가평")
    assert find(second["analysis"], "relationship", "chulsoo_kim")
    # 누적 후보에는 호출당 30개 제한을 적용하지 않는다
    _, many = validate_and_submit(
        bundle, {"candidates": []}, validation_ref="v3", previous=[GAPYEONG] * 40
    )
    assert len(many["accepted"]) == 1


def test_validate_and_submit_bad_args(bundle):
    for args in ({"candidates": "x"}, {"candidates": [GAPYEONG] * 31}, None):
        response, record = validate_and_submit(bundle, args, validation_ref="v")
        assert response == {"status": "error", "error_code": "INVALID_CANDIDATES"}
        assert record is None


def test_agent_view_sends_minimum(bundle):
    view = agent_view(bundle)
    text = json.dumps(view, ensure_ascii=False)
    assert "chulsoo_kim:I3" in text
    assert "병원" not in text
    # 연락처는 추론하지 않으므로 값도 근거 문맥도 보내지 않는다
    assert "5678" not in text and "example.com" not in text
    # 학번처럼 추론 값으로 쓰지 않는 유형은 값을 가린다
    assert "20261234" not in text
    assert not {"phone", "email", "address", "sensitive"} & {it["type"] for it in view["items"]}
    # 숨긴 항목 수: 본인 전화 3개·이메일 1개, 주변 계정 전화·민감정보 2개
    assert view["omitted_count"] == 6
    neighbor = copy.deepcopy(NEIGHBOR)
    neighbor["items"][2]["evidence"] = "길동이랑 가평 병원 진료 같이 감"
    linked = combine_detections(copy.deepcopy(PRIMARY), neighbor)
    # 민감정보가 함께 탐지된 원문은 근거 문맥을 보내지 않는다
    assert "병원" not in json.dumps(agent_view(linked), ensure_ascii=False)
    rules = inference_rules("gildong_hong")
    assert "gildong_hong" in rules and "{subject}" not in rules


def test_review_records_decisions(bundle):
    response, record = review_detected_items(
        bundle,
        {
            "bundle_ref": "bundle_0001",
            "decisions": [
                {"info_id": "gildong_hong:I18", "decision": "exclude", "reason": "항목 제목"},
                {"info_id": "chulsoo_kim:I7", "decision": "uncertain", "reason": "지명 여부 불명"},
                {"info_id": "gildong_hong:I99", "decision": "exclude", "reason": "x"},
                {"info_id": "chulsoo_kim:I6", "decision": "exclude", "reason": "x"},
                {"info_id": "gildong_hong:I9", "decision": "delete", "reason": "x"},
                {"info_id": "gildong_hong:I9", "decision": "exclude", "reason": " "},
            ],
        },
    )
    assert response == {
        "status": "ok",
        "applied_count": 2,
        "rejected_count": 4,
        "excluded_count": 1,
        "uncertain_count": 1,
    }
    assert [r["reason"] for r in record["rejected"]] == [
        "unknown_info_id",
        "not_reviewable",
        "invalid_decision",
        "missing_reason",
    ]
    assert record["excluded_info_ids"] == ["gildong_hong:I18"]
    assert record["uncertain_info_ids"] == ["chulsoo_kim:I7"]
    for args in ({"decisions": []}, {"decisions": "x"}, None):
        assert review_detected_items(bundle, args) == (
            {"status": "error", "error_code": "INVALID_DECISIONS"},
            None,
        )


def test_review_keep_restores_previous_state():
    reviewed = make_bundle(excluded_info_ids=["chulsoo_kim:I3"])
    _, record = review_detected_items(
        reviewed,
        {
            "bundle_ref": "b",
            "decisions": [{"info_id": "chulsoo_kim:I3", "decision": "keep", "reason": "재확인"}],
        },
    )
    assert record["excluded_info_ids"] == []


def test_excluded_item_is_not_evidence_and_revokes_previous():
    _, first = validate_and_submit(make_bundle(), {"candidates": [GAPYEONG]}, validation_ref="v1")
    reviewed = make_bundle(excluded_info_ids=["chulsoo_kim:I3"])
    assert verify(reviewed, GAPYEONG)[1] == [{"index": 0, "reason": "excluded_info"}]
    _, second = validate_and_submit(
        reviewed,
        {"candidates": []},
        validation_ref="v2",
        previous=first["accepted"],
        bundle_ref="bundle_0002",
    )
    assert second["revoked"] == [{"index": 0, "reason": "excluded_info"}]
    assert second["bundle_ref"] == "bundle_0002"
    assert not find(second["analysis"], "location", "가평")
    with pytest.raises(ContractError):
        make_bundle(excluded_info_ids=["nobody:I1"])


def test_excluded_detection_leaves_unverified_and_findings():
    reviewed = make_bundle(excluded_info_ids=["gildong_hong:I18", "gildong_hong:I12"])
    result = build_result(reviewed)
    assert "gildong_hong:I18" not in {u["info_id"] for u in result["unverified"]}
    # 학생증 이름을 오탐으로 제외하면 그 이미지의 학번도 본인 정보로 확인되지 않는다
    assert not [f for f in result["findings"] if f["type"] == "student_id"]
    view = agent_view(reviewed, "bundle_0002")
    assert view["bundle_ref"] == "bundle_0002"
    reviews = {it["info_id"]: it["review"] for it in view["items"]}
    assert reviews["gildong_hong:I18"] == "excluded"
