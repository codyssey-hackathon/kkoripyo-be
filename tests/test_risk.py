import json
from pathlib import Path

import pytest

from app.cross_inference import combine_detections, validate_and_submit
from app.risk import calculate_risk, check_context

PRIMARY = json.loads(
    (Path(__file__).parent / "fixtures" / "detection_gildong_hong.json").read_text(encoding="utf-8")
)
IMAGE_POSTS = {"gildong_hong": {"fake_card.png": "FAKE001"}}
NO_SOURCES = {}.get


def finding(id_, typ, value, status="confirmed", source_ids=()):
    return {
        "finding_id": id_,
        "type": typ,
        "value": value,
        "status": status,
        "source_ids": list(source_ids),
    }


def record(*findings, bundle_ref="bundle_0001"):
    return {
        "validation_ref": "validation_0001",
        "bundle_ref": bundle_ref,
        "subject": "gildong_hong",
        "analysis": {"findings": list(findings), "sensitive_flags": [], "unverified": []},
    }


def run(rec, sources=NO_SOURCES, coverage="complete"):
    return calculate_risk(
        rec,
        {"validation_ref": "validation_0001"},
        risk_ref="risk_0001",
        sources=sources,
        coverage_status=coverage,
    )


PLACE = finding(
    "finding_0001", "location", "가평", "inferred", ["me:P1:caption:0", "nb:P2:caption:0"]
)
HOME = finding("finding_0002", "address", "서울시 마포구 연남동", source_ids=["me:P3:caption:0"])
TEXTS = {
    "me:P1:caption:0": "주말에 가평 다녀옴",
    "nb:P2:caption:0": "가평 가고 싶다",
    "me:P3:caption:0": "서울시 마포구 연남동 우리 집 앞 카페",
}


def test_scores_validation_record_from_cross_inference():
    bundle = combine_detections(PRIMARY, image_posts=IMAGE_POSTS)
    _, rec = validate_and_submit(
        bundle, {"candidates": []}, validation_ref="validation_0001", bundle_ref="bundle_0001"
    )
    response, stored = run(rec)
    # 계정 5 + 이름 8 + 전화 25 + 이메일 25 + 소속 14, 이름+소속 8 + 계정+연락처 6
    assert response == {
        "risk_ref": "risk_0001",
        "score": 91.0,
        "base_score": 77.0,
        "combination_score": 14.0,
        "status": "calculated",
    }
    assert stored["unscored_types"] == ["student_id"]
    assert stored["coverage_status"] == "complete"


def test_inferred_uses_status_multiplier():
    _, stored = run(record(finding("finding_0001", "affiliation", "예시 기관명", "inferred")))
    assert stored["breakdown"] == [
        {
            "finding_id": "finding_0001",
            "type": "affiliation",
            "value": "예시 기관명",
            "item": "affiliation",
            "base_points": 14,
            "status_multiplier": 0.6,
            "points": 8.4,
        }
    ]
    assert stored["score"] == 8.4


def test_combination_once_with_smaller_multiplier():
    rec = record(
        finding("finding_0001", "name", "홍길동"),
        finding("finding_0002", "affiliation", "한국대학교", "inferred"),
        finding("finding_0003", "affiliation", "서울대학교", "inferred"),
    )
    _, stored = run(rec)
    # 소속 값이 둘이어도 이름+소속 가점은 한 번, 계수는 작은 쪽(0.6)
    assert stored["combinations"] == [
        {"items": ["name", "affiliation"], "bonus": 8, "status_multiplier": 0.6, "points": 4.8}
    ]
    assert stored["base_score"] == 24.8
    assert stored["score"] == 29.6


def test_score_capped_at_100():
    rec = record(*(finding(f"finding_{i:04d}", "phone", f"010-0000-000{i}") for i in range(5)))
    response, _ = run(rec)
    assert response["base_score"] == 125.0
    assert response["score"] == 100


def test_place_with_visit_and_residence_context_is_scored():
    _, stored = run(record(PLACE, HOME), TEXTS.get)
    assert [(b["item"], b["points"]) for b in stored["breakdown"]] == [
        ("activity_location", 7.2),
        ("home_address", 30.0),
    ]
    assert stored["unmapped"] == []


def test_place_without_context_is_not_scored():
    texts = {
        "me:P1:caption:0": "가평 날씨 좋다",
        "nb:P2:caption:0": "가평 가고 싶다",
        "me:P3:caption:0": "서울시 마포구 연남동 맛집 다녀옴",
    }
    _, stored = run(record(PLACE, HOME), texts.get)
    assert stored["breakdown"] == []
    assert stored["unmapped"] == [
        {"finding_id": "finding_0001", "type": "location"},
        {"finding_id": "finding_0002", "type": "address"},
    ]
    assert stored["score"] == 0


def test_place_without_source_text_is_not_scored():
    _, stored = run(record(PLACE, HOME))
    assert stored["breakdown"] == []
    assert len(stored["unmapped"]) == 2


@pytest.mark.parametrize(
    ("kind", "value", "text", "expected"),
    [
        ("activity_location", "가평", "매주 가평 방문", None),
        ("activity_location", "가평", "가평 못 갔어", "no_context"),
        ("activity_location", "가평", "가평 날씨 좋다", "no_context"),
        ("activity_location", "가평", "주말에 춘천 다녀옴", "value_not_in_source"),
        ("home_address", "연남동", "연남동 자취 3년차", None),
        ("home_address", "연남동", "연남동 맛집 다녀옴", "no_context"),
        ("home_address", "연남동", "연남동 이사님 미팅", "no_context"),
        # 장소가 나온 문장 안의 표현만 본다
        ("activity_location", "가평", "가평 날씨 좋다. 부산 다녀옴", "no_context"),
        ("activity_location", "부산", "가평 날씨 좋다. 부산 다녀옴", None),
        ("activity_location", "가평", "가평 다녀옴ㅋㅋ 부산은 못 갔어", None),
        ("activity_location", "부산", "가평 다녀옴ㅋㅋ 부산은 못 갔어", "no_context"),
        ("activity_location", "가평", "가평\n3.5km 걸어서 도착", "no_context"),
        ("home_address", "연남동", "연남동 맛집 감\n우리 집은 망원동", "no_context"),
    ],
)
def test_check_context(kind, value, text, expected):
    assert check_context(kind, value, text) == expected


def test_unavailable_coverage_is_not_zero():
    response, stored = run(
        record(finding("finding_0001", "name", "홍길동")), coverage="unavailable"
    )
    assert response["status"] == "insufficient_data" and response["score"] is None
    assert stored["coverage_status"] == "unavailable"


def test_partial_coverage():
    response, _ = run(record(), coverage="partial")
    assert response["status"] == "insufficient_data" and response["score"] is None
    response, _ = run(record(finding("finding_0001", "name", "홍길동")), coverage="partial")
    assert response["status"] == "calculated" and response["score"] == 8


def test_complete_coverage_without_findings_is_zero():
    response, _ = run(record())
    assert response["status"] == "calculated" and response["score"] == 0


def test_errors():
    rec = record(finding("finding_0001", "name", "홍길동"))
    assert run(None)[0] == {"status": "error", "error_code": "INVALID_VALIDATION_REF"}
    wrong = calculate_risk(
        rec,
        {"validation_ref": "validation_9999"},
        risk_ref="r",
        sources=NO_SOURCES,
        coverage_status="complete",
    )
    assert wrong == ({"status": "error", "error_code": "INVALID_VALIDATION_REF"}, None)
    with pytest.raises(ValueError):
        run(rec, coverage="done")


def test_same_record_same_score():
    rec = record(PLACE, HOME, finding("finding_0003", "name", "홍길동"))
    assert run(rec, TEXTS.get) == run(rec, TEXTS.get)
