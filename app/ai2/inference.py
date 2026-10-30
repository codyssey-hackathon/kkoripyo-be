"""개인정보 추론 후보와 검증. 점수 계산 / 얼굴 매칭 / 민감정보 추론 없음."""

from __future__ import annotations

import re

from .contracts import scopes
from .graph import EvidenceGraph

RULE_MAP = {
    "name": "R01",
    "account_id": "R01",
    "face": "R01",
    "affiliation": "R02",
    "location": "R03",
    "activity_pattern": "R04",
    "relationship": "R05",
    "phone": "R06",
    "email": "R06",
    "home_address": "R06",
}
LABELS = {
    "name": "이름",
    "account_id": "SNS 계정 ID",
    "face": "식별 가능한 얼굴",
    "affiliation": "소속",
    "location": "활동 장소",
    "activity_pattern": "생활 패턴",
    "relationship": "관계",
    "phone": "전화번호",
    "email": "이메일",
    "home_address": "거주지 주소",
}
# 제출 type별로 근거가 될 수 있는 JSON2 info.type
EVIDENCE_TYPES = {
    "affiliation": {"affiliation", "organization"},
    "activity_pattern": {"activity_pattern", "activity_time"},
}
# JSON2 info.type -> 보고서 type. 직접 확인 경로도 모델 제출 경로와 같은 유형으로 묶는다.
REPORT_TYPE = {src: typ for typ, sources in EVIDENCE_TYPES.items() for src in sources}
BLOCKED = {"sensitive_flag"}


def clean(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").casefold())


def text_of(post: dict) -> str:
    # 캡션 없는 게시물은 text가 null일 수 있음
    return post.get("text") or ""


def relation_to_subject(data: dict, post: dict) -> bool:
    subject = data["subject_account_id"]
    if subject in post.get("tagged_account_ids", []):
        return True
    return any(
        r.get("post_id") == post["post_id"]
        and r.get("target_account_id") == subject
        and r.get("type") in {"tagged", "mentioned"}
        for r in data["relations"]
    )


def direct_findings(data: dict) -> list[dict]:
    subject = data["subject_account_id"]
    result: list[dict] = []
    # 친구 계정에 포함된 정보를 직접 개인정보로 잘못 집계하지 않음
    for info in data["extracted_info"]:
        typ = REPORT_TYPE.get(info.get("type"), info.get("type"))
        if typ not in RULE_MAP or info.get("subject_account_id") != subject:
            continue
        if not info.get("subject_verified", False):
            # 확정된 대상자 연결 없는 정보는 내 개인정보로 확정하지 않음
            continue
        if info.get("classification") == "inferred":
            continue
        result.append(
            {
                "rule_id": RULE_MAP[typ],
                "type": typ,
                "value": info["value"],
                "status": "confirmed",
                "info_ids": [info["info_id"]],
                "source_scope": "direct" if info["source_account_id"] == subject else "cross",
            }
        )
    return result


def deterministic_candidates(data: dict, graph: EvidenceGraph) -> list[dict]:
    """로컬 LLM이 없어도 검증 가능한 R02 사례만 보수적으로 추론."""
    subject = data["subject_account_id"]
    info = data["extracted_info"]
    posts = {p["post_id"]: p for p in data["posts"]}
    accounts = scopes(data)
    primary_org = [
        i
        for i in info
        if i["type"] == "organization"
        and i["source_account_id"] == subject
        and i.get("source_post_id")
    ]
    secondary_org = [
        i
        for i in info
        if i["type"] == "organization"
        and accounts[i["source_account_id"]] == "secondary"
        and i.get("source_post_id")
    ]
    findings = []
    for a in primary_org:
        p1 = posts[a["source_post_id"]]
        if "출근" not in text_of(p1) and "근무" not in text_of(p1):
            continue
        for b in secondary_org:
            p2 = posts[b["source_post_id"]]
            if clean(a["value"]) != clean(b["value"]):
                continue
            if a["evidence_group_id"] == b["evidence_group_id"]:
                continue
            if not relation_to_subject(data, p2):
                continue
            if not any(term in text_of(p2) for term in ("선생님", "직원", "동료", "함께 근무")):
                continue
            if graph.path("account:" + subject, "info:" + b["info_id"]) is None:
                continue
            findings.append(
                {
                    "rule_id": "R02",
                    "type": "affiliation",
                    "value": a["value"],
                    "status": "inferred",
                    "info_ids": [a["info_id"], b["info_id"]],
                    "source_scope": "cross",
                }
            )
    return findings


def check_candidate(data: dict, c: dict) -> tuple[dict | None, str | None]:
    """모델이 제안한 개인정보 후보를 코드에서 근거·대상자·관계로 재검증. (결과, 거절 사유)를 반환."""
    info_by_id = {i["info_id"]: i for i in data["extracted_info"]}
    posts = {p["post_id"]: p for p in data["posts"]}
    role = scopes(data)
    subject = data["subject_account_id"]
    typ = c.get("type")
    ids = c.get("info_ids", [])
    value = c.get("value")
    if not isinstance(typ, str) or typ not in RULE_MAP:
        return None, "unknown_type"
    if c.get("rule_id") != RULE_MAP[typ]:
        return None, "rule_type_mismatch"
    if not isinstance(value, str) or not value.strip():
        return None, "empty_value"
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        return None, "invalid_info_ids"
    if any(x not in info_by_id for x in ids):
        return None, "unknown_info_id"
    infos = [info_by_id[x] for x in dict.fromkeys(ids)]
    if any(i["type"] in BLOCKED for i in infos):
        return None, "sensitive_info"
    if any(i.get("subject_account_id") not in (None, subject) for i in infos):
        return None, "other_subject"
    # 값뿐 아니라 유형까지 일치하는 근거만 인정. 원문 어디에도 없는 정보는 LLM이 새로 만들어 냈을 가능성이 있음
    matched = [
        i
        for i in infos
        if i["type"] in EVIDENCE_TYPES.get(typ, {typ}) and clean(i["value"]) == clean(value)
    ]
    if not matched:
        return None, "value_not_in_evidence"
    if c.get("status") == "confirmed":
        # 값이 일치하는 그 근거 자체가 본인으로 검증된 경우만 확인으로 취급. 다른 info의 검증으로 승격 금지.
        if not any(
            i.get("subject_account_id") == subject and i.get("subject_verified") is True
            for i in matched
        ):
            return None, "subject_not_verified"
        status = "confirmed"
    elif c.get("status") == "inferred":
        status = "inferred"
        if len({i["evidence_group_id"] for i in infos}) < 2:
            return None, "single_evidence_group"
        if not any(i["source_account_id"] == subject for i in infos):
            return None, "no_subject_source"
        secondary = [i for i in infos if role[i["source_account_id"]] == "secondary"]
        if not secondary:
            return None, "no_secondary_source"
        if not any(
            i.get("source_post_id") and relation_to_subject(data, posts[i["source_post_id"]])
            for i in secondary
        ):
            return None, "no_subject_link"
        # 유형별 안전 게이트: 모델이 임의로 관계·주소·소속을 단정하지 못하도록 제한
        if typ == "affiliation":
            own_org = [
                i
                for i in infos
                if i["type"] in {"organization", "affiliation"}
                and i["source_account_id"] == subject
                and i.get("source_post_id")
            ]
            neighbor_org = [
                i
                for i in secondary
                if i["type"] in {"organization", "affiliation"} and i.get("source_post_id")
            ]
            if not any(
                clean(a["value"]) == clean(b["value"]) == clean(value)
                and any(w in text_of(posts[a["source_post_id"]]) for w in ("출근", "근무"))
                and any(
                    w in text_of(posts[b["source_post_id"]])
                    for w in ("선생님", "직원", "동료", "함께 근무")
                )
                for a in own_org
                for b in neighbor_org
            ):
                return None, "affiliation_gate"
        elif typ in {"phone", "email", "home_address", "face"}:
            # 주소/연락처·얼굴을 외부 포스트에서 새로 '추측'하도록 허용하지 않음
            return None, "not_inferable_type"
        elif typ == "activity_pattern":
            if not any(
                w in text_of(posts[i["source_post_id"]])
                for i in infos
                if i.get("source_post_id")
                for w in ("매주", "매일", "주마다", "정기")
            ):
                return None, "activity_pattern_gate"
        elif typ == "relationship":
            if not any(
                i["type"] == "relationship" and i.get("subject_account_id") == subject
                for i in infos
            ):
                return None, "relationship_gate"
        elif typ == "name":
            if not any(
                i["type"] == "name" and i.get("subject_account_id") == subject for i in infos
            ):
                return None, "name_gate"
        elif typ == "account_id":
            # 계정 ID는 식별 사실이므로 교차 '추론' 대상이 아님
            return None, "not_inferable_type"
    else:
        return None, "invalid_status"
    return {
        "rule_id": c["rule_id"],
        "type": typ,
        "value": value.strip(),
        "status": status,
        "info_ids": [i["info_id"] for i in infos],
        "source_scope": "cross"
        if any(role[i["source_account_id"]] == "secondary" for i in infos)
        else "direct",
    }, None


def validate_candidate(data: dict, c: dict) -> dict | None:
    return check_candidate(data, c)[0]


def merge_candidates(candidates: list[dict]) -> list[dict]:
    result = {}
    for c in candidates:
        key = (c["type"], clean(c["value"]))
        if key not in result:
            result[key] = dict(c)
        else:
            old = result[key]
            old["info_ids"] = list(dict.fromkeys(old["info_ids"] + c["info_ids"]))
            if c["status"] == "confirmed":
                old["status"] = "confirmed"
            if old["source_scope"] == "direct":
                # 직접 확인된 개인정보는 주변 근거 때문에 교차 추론 정보로 강등/재분류하지 않음
                pass
            elif c["source_scope"] == "direct":
                old["source_scope"] = "direct"
    return list(result.values())


def render_json3(
    data: dict, candidates: list[dict], mode: str, tool_log: list[dict] | None = None
) -> dict:
    subject = data["subject_account_id"]
    role = scopes(data)
    alias = {subject: "me"}
    secondary = [a["account_id"] for a in data["accounts"] if a["scope"] == "secondary"]
    alias.update({account_id: f"neighbor_{k:02d}" for k, account_id in enumerate(secondary, 1)})
    info_map = {i["info_id"]: i for i in data["extracted_info"]}
    findings = []
    for index, candidate in enumerate(merge_candidates(candidates), 1):
        evidences = []
        seen = set()
        for info_id in candidate["info_ids"]:
            info = info_map[info_id]
            dedup = (info["evidence_group_id"], info["source_account_id"])
            if dedup in seen:
                continue
            seen.add(dedup)
            evidences.append(
                {
                    "info_id": info_id,
                    "account_ref": alias[info["source_account_id"]],
                    "account_scope": role[info["source_account_id"]],
                    "post_id": info.get("source_post_id"),
                    "source_kind": info.get("source_kind", "post"),
                    "evidence_group_id": info["evidence_group_id"],
                    "excerpt": str(info["evidence"])[:160],
                }
            )
        value = candidate["value"]
        label = LABELS.get(candidate["type"], "개인정보")
        if candidate["status"] == "confirmed":
            description = f"{label}에 해당하는 '{value}' 정보가 SNS 콘텐츠에서 확인됩니다. 아래 출처에서 확인할 수 있습니다."
        else:
            description = f"본인과 주변 계정의 정보를 함께 보면 {label}이(가) '{value}'일 가능성이 있습니다. 확정된 사실은 아니며 아래 근거를 확인해야 합니다."
        findings.append(
            {
                "finding_id": f"F{index:03d}",
                "rule_id": candidate["rule_id"],
                "type": candidate["type"],
                "value": value,
                "source_scope": candidate["source_scope"],
                "status": candidate["status"],
                "description": description,
                "evidence": evidences,
            }
        )
    # 민감정보 원문·추론은 결과에 쓰지 않고 탐지 플래그만 표시
    sensitive = []
    for info in data["extracted_info"]:
        if info["type"] == "sensitive_flag" and info["subject_account_id"] == subject:
            sensitive.append(
                {
                    "category": info.get("category", "other"),
                    "account_scope": role[info["source_account_id"]],
                    "post_id": info.get("source_post_id"),
                    "note": "민감정보 명시적 노출 탐지(내용 미수록)",
                }
            )
    from .contracts import analysis_scope

    return {
        "schema_version": "1.0",
        "subject_account_id": subject,
        "analysis_scope": analysis_scope(data, mode, tool_log),
        "findings": findings,
        "sensitive_flags": sensitive,
    }
