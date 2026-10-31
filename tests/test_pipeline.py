"""교차 추론과 위험도 계산을 이어서 돌리는 통합 테스트.

가상 계정 두 개(본인 minji_lee, 본인을 언급한 주변 계정 jisoo_park)의 수집 원문과 탐지 결과로
combine_detections → validate_and_submit → calculate_risk 흐름을 확인한다. 실제 사용자 데이터는 쓰지 않는다.
"""

import pytest

from app.cross_inference import (
    agent_view,
    combine_detections,
    review_detected_items,
    validate_and_submit,
)
from app.risk import calculate_risk

SUBJECT = "minji_lee"
NEIGHBOR = "jisoo_park"

# 백엔드가 보관하는 수집 원문 (collect_account 결과를 줄인 형태)
COLLECTED = {
    SUBJECT: {
        "profile": {
            "username": "minji_lee",
            "full_name": "이민지",
            "biography": "한빛대학교 경영학과 23학번",
        },
        "posts": [
            {
                "post_id": "P1",
                "text": "주말에 가평 다녀옴 with @jisoo_park",
                "comments": [{"username": "stranger", "text": "민지야 번호 010-9999-8888 맞아?"}],
            },
            {"post_id": "P2", "text": "학생증 재발급 완료", "images": ["card.png"]},
        ],
    },
    NEIGHBOR: {
        "profile": {"username": "jisoo_park", "full_name": "박지수", "biography": "서울대학교"},
        "posts": [{"post_id": "N1", "text": "@minji_lee 랑 또 가평 왔다"}],
    },
}
IMAGE_POSTS = {SUBJECT: {"card.png": "P2"}}


def source_texts(collected: dict) -> dict[str, str]:
    """수집 원문을 source_id -> 텍스트로 펼친다(get_collected_context가 조회하는 저장소 역할)."""
    texts = {}
    for account, data in collected.items():
        for field, value in data["profile"].items():
            texts[f"{account}:profile:{field}:0"] = value
        for post in data["posts"]:
            texts[f"{account}:{post['post_id']}:caption:0"] = post["text"]
            for n, comment in enumerate(post.get("comments", [])):
                texts[f"{account}:{post['post_id']}:comment:{n}"] = comment["text"]
    return texts


TEXTS = source_texts(COLLECTED)


def item(id_, typ, value, input_id, category="일반개인정보", normalized=None, modality="text"):
    return {
        "id": id_,
        "type": typ,
        "category": category,
        "value": value,
        "normalized": normalized if normalized is not None else value,
        "evidence": value,
        "source": {
            "input_id": input_id,
            "modality": modality,
            "method": "ocr+regex" if modality == "image" else "regex",
            "bbox": None,
        },
        "confidence": 0.8,
    }


def detection(subject, items, groups):
    """groups: [[item id, ...], ...] 같은 정보로 묶인 항목"""
    by_id = {i["id"]: i for i in items}
    return {
        "schema_version": "json2/v1",
        "subject": subject,
        "llm_used": False,
        "items": items,
        "entities": [
            {
                "entity_id": f"E{n}",
                "type": by_id[ids[0]]["type"],
                "value": by_id[ids[0]]["value"],
                "item_ids": ids,
                "sources": [],
                "match_reasons": ["normalized" if len(ids) > 1 else "single"],
                "confidence": 0.8,
            }
            for n, ids in enumerate(groups, 1)
        ],
    }


PRIMARY = detection(
    SUBJECT,
    [
        item("I1", "sns_account", "minji_lee", "profile.username"),
        item("I2", "name", "이민지", "profile.full_name"),
        item(
            "I3",
            "affiliation",
            "한빛대학교 경영학과",
            "profile.biography",
            "준식별자",
            "한빛대학교경영학과",
        ),
        item("I4", "location", "가평", "post:P1.caption", "준식별자"),
        item("I5", "sns_account", "@jisoo_park", "post:P1.caption", normalized="jisoo_park"),
        # 학생증 OCR: 본인 이름과 함께 나온 번호, 항목 제목 '연락처'를 이름으로 잘못 잡은 오탐
        item("I6", "name", "이민지", "image:card.png", modality="image"),
        item(
            "I7",
            "phone",
            "010-2222-3333",
            "image:card.png",
            normalized="01022223333",
            modality="image",
        ),
        item("I8", "name", "연락처", "image:card.png", modality="image"),
        # 남이 단 댓글의 번호. 작성자를 알 수 없어 본인 정보가 아니다
        item(
            "I9",
            "phone",
            "010-9999-8888",
            "post:P1.comment0",
            normalized="01099998888",
        ),
    ],
    [["I1"], ["I2", "I6"], ["I3"], ["I4"], ["I5"], ["I7"], ["I8"], ["I9"]],
)
NEIGHBOR_DETECTION = detection(
    NEIGHBOR,
    [
        item("I1", "sns_account", "jisoo_park", "profile.username"),
        item("I2", "sns_account", "@minji_lee", "post:N1.caption", normalized="minji_lee"),
        item("I3", "location", "가평", "post:N1.caption", "준식별자"),
        # 본인과 연결되지 않은 주변 계정 프로필
        item("I4", "affiliation", "서울대학교", "profile.biography", "준식별자"),
    ],
    [["I1"], ["I2"], ["I3"], ["I4"]],
)


def cand(rule_id, typ, value, info_ids):
    return {
        "rule_id": rule_id,
        "type": typ,
        "value": value,
        "status": "inferred",
        "info_ids": info_ids,
    }


# Agent가 제출한다고 가정한 후보. 마지막 소속은 본인과 연결되지 않은 근거라 거절되어야 한다
CANDIDATES = [
    cand("R03", "location", "가평", ["minji_lee:I3", "minji_lee:I4"]),
    cand("R07", "activity_pattern", "매주 가평 방문", ["minji_lee:I4", "jisoo_park:I3"]),
    cand("R08", "relationship", "jisoo_park", ["jisoo_park:I2"]),
    cand("R02", "affiliation", "서울대학교", ["minji_lee:I3", "jisoo_park:I4"]),
]


def combine(**review):
    return combine_detections(PRIMARY, NEIGHBOR_DETECTION, image_posts=IMAGE_POSTS, **review)


def submit(bundle, candidates, previous=None, ref="validation_0001", bundle_ref="bundle_0001"):
    response, record = validate_and_submit(
        bundle,
        {"candidates": candidates},
        validation_ref=ref,
        previous=previous,
        bundle_ref=bundle_ref,
    )
    assert response["status"] == "ok"
    return response, record


def score(record, coverage="complete", sources=TEXTS.get):
    return calculate_risk(
        record,
        {"validation_ref": record["validation_ref"]},
        risk_ref="risk_0001",
        sources=sources,
        coverage_status=coverage,
    )


def by_type(stored):
    return {b["type"]: b["points"] for b in stored["breakdown"]}


def test_source_ids_match_collected_data():
    """교차 추론이 만드는 출처 ID로 수집 원문을 찾을 수 있어야 장소 판정이 동작한다."""
    bundle = combine()
    for it in bundle["items"]:
        if it["part"] != "image":
            assert it["source_id"] in TEXTS, it["source_id"]
    # 이미지는 image_posts로 게시물이 연결된다
    image = next(it for it in bundle["items"] if it["info_id"] == "minji_lee:I7")
    assert image["source_id"] == "minji_lee:P2:image:card.png"


def test_full_pipeline_score():
    _, record = submit(combine(), CANDIDATES)
    response, stored = score(record)
    assert by_type(stored) == {
        "sns_account": 5.0,
        "name": 8.0,
        "affiliation": 14.0,
        "phone": 25.0,
        # '주말에 가평 다녀옴' 문장으로 활동 장소 확인, 추론이라 12 × 0.6
        "location": 7.2,
        "activity_pattern": 8.4,
        "relationship": 3.6,
    }
    # 이름+소속 8, 소속+활동 장소 8 × 0.6, 계정+연락처 6
    assert [c["points"] for c in stored["combinations"]] == [8.0, 4.8, 6.0]
    assert response == {
        "risk_ref": "risk_0001",
        "score": 90.0,
        "base_score": 71.2,
        "combination_score": 18.8,
        "status": "calculated",
    }


def test_only_verified_information_is_scored():
    response, record = submit(combine(), CANDIDATES)
    _, stored = score(record)
    assert response["rejected_count"] == 1
    values = {b["value"] for b in stored["breakdown"]}
    # 거절된 후보, 댓글 속 남의 번호, OCR 오탐 이름은 점수에 없다
    assert "서울대학교" not in values
    assert "010-9999-8888" not in values
    assert "연락처" not in values
    finding_ids = {f["finding_id"] for f in record["analysis"]["findings"]}
    assert {b["finding_id"] for b in stored["breakdown"]} <= finding_ids


def test_agent_view_hides_contact_before_scoring():
    view = agent_view(combine(), "bundle_0001")
    assert not any(it["type"] == "phone" for it in view["items"])
    assert "010-2222-3333" not in str(view)


def test_excluding_false_positive_recalculates_score():
    bundle = combine()
    _, first = submit(bundle, CANDIDATES)
    first_score = score(first)[0]["score"]

    # Agent가 P1 본문의 '가평'을 오탐으로 판단했다고 가정
    _, review = review_detected_items(
        bundle,
        {
            "bundle_ref": "bundle_0001",
            "decisions": [
                {"info_id": "minji_lee:I4", "decision": "exclude", "reason": "카페 이름 '가평'"}
            ],
        },
    )
    rebuilt = combine(
        excluded_info_ids=review["excluded_info_ids"],
        uncertain_info_ids=review["uncertain_info_ids"],
    )
    _, second = submit(
        rebuilt, [], previous=first["accepted"], ref="validation_0002", bundle_ref="bundle_0002"
    )
    # 제외한 항목을 근거로 쓰던 장소·생활 패턴은 철회된다
    assert [r["reason"] for r in second["revoked"]] == ["excluded_info", "excluded_info"]
    response, stored = score(second)
    assert "location" not in by_type(stored)
    assert "activity_pattern" not in by_type(stored)
    assert stored["validation_ref"] == "validation_0002"
    assert stored["bundle_ref"] == "bundle_0002"
    # 71.2 - 7.2 - 8.4 = 55.6, 소속+활동 장소 가점도 빠져 8 + 6
    assert response["score"] == 69.6 < first_score


def test_place_without_source_text_is_not_scored():
    """백엔드가 원문을 찾지 못하면 장소는 점수에서 빠지고 남겨 둔다."""
    _, record = submit(combine(), CANDIDATES)
    _, stored = score(record, sources=lambda source_id: None)
    assert "location" not in by_type(stored)
    assert [u["type"] for u in stored["unmapped"]] == ["location"]


def test_collection_failure_returns_null():
    _, record = submit(combine(), CANDIDATES)
    response, _ = score(record, coverage="unavailable")
    assert response["status"] == "insufficient_data"
    assert response["score"] is None


@pytest.mark.parametrize(("coverage", "expected"), [("partial", None), ("complete", 0)])
def test_account_without_information(coverage, expected):
    """비공개 계정처럼 탐지된 정보가 없을 때: 일부만 수집됐으면 null, 다 수집했으면 0점."""
    empty = detection(SUBJECT, [], [])
    _, record = submit(combine_detections(empty), [])
    assert record["analysis"]["findings"] == []
    response, _ = score(record, coverage=coverage)
    assert response["score"] == expected
