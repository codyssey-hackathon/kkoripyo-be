"""검증을 통과한 정보로 개인정보 분석 결과를 만든다. 위험 점수는 calculate_risk가 따로 계산한다.

findings·sensitive_flags는 최종 보고서(report/v1)의 analysis로 그대로 옮길 수 있는 형식이다.
"""

from __future__ import annotations

from .rules import DERIVED_TYPES, blocked, direct_findings, unverified_items, value_key
from .schema import RESULT_VERSION

LABELS = {
    "name": "이름",
    "sns_account": "SNS 계정",
    "affiliation": "소속",
    "student_id": "학번",
    # 활동 장소인지는 맥락이 확인돼야 하므로(위험도 계산 담당) 여기서는 '장소'로만 표시
    "location": "장소",
    "phone": "전화번호",
    "email": "이메일",
    # 거주지인지는 별도 근거가 필요하므로 '집 주소'로 표시하지 않는다
    "address": "주소",
    "rrn": "주민등록번호",
    "passport": "여권번호",
    "driver_license": "운전면허번호",
    "alien_reg": "외국인등록번호",
    "card_number": "카드번호",
    "bank_account": "계좌번호",
    "birth_date": "생년월일",
    "vehicle_plate": "차량번호",
    "other_clue": "기타 단서",
    "activity_pattern": "생활 패턴",
    "relationship": "인간관계",
}


def merge_candidates(candidates: list[dict]) -> list[dict]:
    """같은 유형·값의 후보를 하나로 합친다. 확인된 정보는 추론 정보로 강등하지 않는다."""
    result: dict[tuple, dict] = {}
    for c in candidates:
        k = (c["type"], value_key(c["type"], c["value"]))
        if k not in result:
            result[k] = dict(c)
            continue
        old = result[k]
        old["info_ids"] = list(dict.fromkeys(old["info_ids"] + c["info_ids"]))
        if c["status"] == "confirmed":
            old["status"] = "confirmed"
        if c["source_scope"] == "direct":
            old["source_scope"] = "direct"
        if "relation_kinds" in c:
            old["relation_kinds"] = sorted({*old.get("relation_kinds", []), *c["relation_kinds"]})
    return list(result.values())


def _description(typ: str, value: str, status: str) -> str:
    label = LABELS[typ]
    if typ == "relationship":
        return f"본인과 주변 계정({value})이 태그·언급·댓글로 연결되어 있어 인간관계가 드러날 수 있습니다."
    if status == "confirmed":
        return f"{label}에 해당하는 '{value}' 정보가 SNS 콘텐츠에서 확인됩니다."
    return f"본인과 주변 계정의 정보를 함께 보면 {label}이(가) '{value}'일 가능성이 있습니다. 확정된 사실은 아니며 근거를 확인해야 합니다."


def render(bundle: dict, candidates: list[dict]) -> dict:
    """본인 확인 정보 + 검증된 후보로 결과를 만든다. candidates는 verify_candidates를 통과한 것이어야 한다."""
    by_id = {it["info_id"]: it for it in bundle["items"]}
    findings = []
    for index, c in enumerate(merge_candidates(direct_findings(bundle) + candidates), 1):
        typ = c["type"]
        items = [by_id[i] for i in c["info_ids"]]
        if typ in DERIVED_TYPES:
            category = "준식별자"
        else:
            category = next(it["category"] for it in items if it["type"] == typ)
        source_ids = [it["source_id"] for it in items]
        if typ == "relationship":
            source_ids += [
                link["source_id"]
                for link in bundle["links"]
                if link["account_id"] == c["value"]
                and link["kind"] in c["relation_kinds"]
                and link["source_id"]
            ]
        finding = {
            "finding_id": f"finding_{index:04d}",
            "rule_id": c["rule_id"],
            "type": typ,
            "category": category,
            "value": c["value"],
            "status": c["status"],
            "source_scope": c["source_scope"],
            "evidence_ids": c["info_ids"],
            "source_ids": list(dict.fromkeys(source_ids)),
            "description": _description(typ, c["value"], c["status"]),
        }
        if typ == "relationship":
            finding["relation_kinds"] = c["relation_kinds"]
        findings.append(finding)

    # 민감정보 원문·추론은 결과에 쓰지 않고 본인 계정의 탐지 위치만 표시
    sensitive = [
        {
            "source_id": it["source_id"],
            "modality": it["modality"],
            "note": "민감정보 명시적 노출 탐지(내용 미수록)",
        }
        for it in bundle["items"]
        if it["scope"] == "primary" and blocked(it)
    ]
    return {
        "schema_version": RESULT_VERSION,
        "subject": bundle["subject"],
        "analysis_scope": {
            "secondary_accounts": sum(a["scope"] == "secondary" for a in bundle["accounts"]),
            "detector_llm_used": any(a["llm_used"] for a in bundle["accounts"]),
        },
        "findings": findings,
        "unverified": unverified_items(bundle),
        "sensitive_detected": bool(sensitive),
        "sensitive_flags": sensitive,
    }
