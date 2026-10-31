"""calculate_risk 실행. 교차 추론 검증 기록(validate_and_submit의 record)으로 위험도 점수를 계산한다.

1. Agent가 calculate_risk를 호출하면 백엔드가 인자의 validation_ref로 검증 기록을 찾는다.
2. 백엔드가 새 risk_ref를 만들어 calculate_risk를 실행한다. sources는 source_id로 원문을 돌려주는 함수
   (get_collected_context와 같은 저장소), coverage_status는 수집 결과로 백엔드가 정한다.
   Agent에는 응답(점수·참조 ID)만 돌려주고, 기록(record)은 서버에 저장한다.

점수·확인 상태·근거는 Agent 인자로 받지 않는다. 장소·주소가 활동 장소·집 주소인지도 코드가 원문을 읽어 정한다.
"""

from __future__ import annotations

from collections.abc import Callable

from .context import check_context
from .table import (
    BASE_POINTS,
    COMBINATIONS,
    CONTEXT_ITEM_OF,
    ITEM_OF,
    MAX_SCORE,
    STATUS_MULTIPLIERS,
)

CALCULATE_TOOL = "calculate_risk"
COVERAGE_STATUSES = {"complete", "partial", "unavailable"}

CALCULATE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": CALCULATE_TOOL,
        "description": "검증을 마친 교차 추론 결과로 위험도 점수를 계산한다. 점수는 서버가 정하며 바꿀 수 없다.",
        "parameters": {
            "type": "object",
            "properties": {"validation_ref": {"type": "string"}},
            "required": ["validation_ref"],
            "additionalProperties": False,
        },
    },
}


def _context_item(finding: dict, sources: Callable[[str], str | None]) -> str | None:
    """장소·주소의 출처 원문 중 하나라도 활동·거주 맥락이 확인되면 점수표 항목, 아니면 None.

    출처는 교차 추론 검증에서 본인과 연결된 것으로 확인된 것만 들어 있다.
    """
    item = CONTEXT_ITEM_OF[finding["type"]]
    for source_id in finding["source_ids"]:
        text = sources(source_id)
        if text is not None and check_context(item, finding["value"], text) is None:
            return item
    return None


def _error(code: str) -> tuple[dict, None]:
    return {"status": "error", "error_code": code}, None


def calculate_risk(
    record: dict | None,
    args: dict,
    *,
    risk_ref: str,
    sources: Callable[[str], str | None],
    coverage_status: str,
) -> tuple[dict, dict | None]:
    """calculate_risk 실행. (Agent에 돌려줄 응답, 서버에 저장할 기록)을 반환한다.

    record: args의 validation_ref로 찾은 검증 기록(없으면 None).
    점수 = Σ(기본점수 × 상태계수) + Σ(조합 가점 × 두 항목 중 작은 상태계수), 최대 100점, 소수 첫째 자리.
    수집이 불가능했거나(unavailable), 일부만 수집됐는데 점수 항목이 하나도 없으면 0점 대신 score=None.
    """
    if coverage_status not in COVERAGE_STATUSES:
        raise ValueError("coverage_status는 complete/partial/unavailable 중 하나입니다")
    if (
        not isinstance(args, dict)
        or not isinstance(record, dict)
        or args.get("validation_ref") != record.get("validation_ref")
        or not isinstance(record.get("analysis"), dict)
    ):
        return _error("INVALID_VALIDATION_REF")

    breakdown: list[dict] = []
    unscored_types: set[str] = set()
    unmapped: list[dict] = []
    for f in record["analysis"]["findings"]:
        if f["type"] in CONTEXT_ITEM_OF:
            item = _context_item(f, sources)
            if item is None:
                # 단순 언급된 장소·주소. 점수에서 빼고 남겨 둔다
                unmapped.append({"finding_id": f["finding_id"], "type": f["type"]})
                continue
        elif f["type"] in ITEM_OF:
            item = ITEM_OF[f["type"]]
        else:
            unscored_types.add(f["type"])
            continue
        multiplier = STATUS_MULTIPLIERS[f["status"]]
        breakdown.append(
            {
                "finding_id": f["finding_id"],
                "type": f["type"],
                "value": f["value"],
                "item": item,
                "base_points": BASE_POINTS[item],
                "status_multiplier": multiplier,
                "points": round(BASE_POINTS[item] * multiplier, 1),
            }
        )

    best = {}
    for b in breakdown:
        best[b["item"]] = max(best.get(b["item"], 0), b["status_multiplier"])
    combinations = []
    for first, second, bonus in COMBINATIONS:
        if first in best and second in best:
            multiplier = min(best[first], best[second])
            combinations.append(
                {
                    "items": [first, second],
                    "bonus": bonus,
                    "status_multiplier": multiplier,
                    "points": round(bonus * multiplier, 1),
                }
            )

    base_score = round(sum(b["points"] for b in breakdown), 1)
    combination_score = round(sum(c["points"] for c in combinations), 1)
    # 수집 실패·부족을 '안전한 0점'으로 표시하지 않는다
    if coverage_status == "unavailable" or (coverage_status == "partial" and not breakdown):
        status, score = "insufficient_data", None
    else:
        status, score = "calculated", round(min(MAX_SCORE, base_score + combination_score), 1)

    stored = {
        "risk_ref": risk_ref,
        "validation_ref": record["validation_ref"],
        "bundle_ref": record.get("bundle_ref"),
        "status": status,
        "score": score,
        "base_score": base_score,
        "combination_score": combination_score,
        "breakdown": breakdown,
        "combinations": combinations,
        "unscored_types": sorted(unscored_types),
        "unmapped": unmapped,
        "coverage_status": coverage_status,
    }
    response = {
        "risk_ref": risk_ref,
        "score": score,
        "base_score": base_score,
        "combination_score": combination_score,
        "status": status,
    }
    return response, stored
