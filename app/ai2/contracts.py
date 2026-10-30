"""JSON2 -> JSON3 공통 계약과 입력 검증. 외부 라이브러리 불필요."""

from __future__ import annotations

TYPES = {
    "name",
    "account_id",
    "face",
    "organization",
    "affiliation",
    "location",
    "activity_time",
    "activity_pattern",
    "relationship",
    "phone",
    "email",
    "home_address",
    "sensitive_flag",
}
ROLES = {"primary", "secondary"}
RELATIONS = {"follows", "likes", "commented", "tagged", "mentioned", "mutual_follow"}
STATUS = {"confirmed", "inferred", "pending"}
MODES = {
    "backend_agent",
    "backend_agent_fallback",
    "offline_mock",
    "backend_mock",
    "backend_mock_fallback",
}


class ContractError(ValueError):
    pass


def ensure_json2(data: dict) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ContractError("JSON2.schema_version은 '1.0'이어야 합니다")
    for key in ("subject_account_id", "accounts", "posts", "relations", "extracted_info"):
        if key not in data:
            raise ContractError(f"JSON2 필수 필드 누락: {key}")
    if (
        not isinstance(data["accounts"], list)
        or not isinstance(data["posts"], list)
        or not isinstance(data["extracted_info"], list)
        or not isinstance(data["relations"], list)
    ):
        raise ContractError("accounts/posts/relations/extracted_info는 배열이어야 합니다")
    ids = [a.get("account_id") for a in data["accounts"]]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ContractError("account_id는 중복되지 않는 문자열이어야 합니다")
    by_id = {a["account_id"]: a for a in data["accounts"]}
    subject = data["subject_account_id"]
    if subject not in by_id or by_id[subject].get("scope") != "primary":
        raise ContractError("subject_account_id는 scope=primary 계정이어야 합니다")
    if any(a.get("scope") not in ROLES for a in data["accounts"]):
        raise ContractError("accounts.scope는 primary/secondary 중 하나입니다")
    post_ids = [p.get("post_id") for p in data["posts"]]
    if len(post_ids) != len(set(post_ids)) or any(not x for x in post_ids):
        raise ContractError("post_id가 누락되거나 중복되었습니다")
    # 태그·관계·정보 대상자는 선정되지 않은 계정을 가리킬 수 있다(실제 게시물에서 흔함).
    # 출처(게시물 작성자·정보 출처)만 분석 범위 안이면 된다.
    for p in data["posts"]:
        if p.get("account_id") not in by_id:
            raise ContractError("알 수 없는 게시물 작성자: " + str(p.get("account_id")))
        if p.get("text") is not None and not isinstance(p["text"], str):
            raise ContractError(
                "posts.text는 문자열 또는 null이어야 합니다: " + str(p.get("post_id"))
            )
        if not isinstance(p.get("tagged_account_ids", []), list):
            raise ContractError("tagged_account_ids는 배열이어야 합니다")
    info_ids = [i.get("info_id") for i in data["extracted_info"]]
    if len(info_ids) != len(set(info_ids)) or any(not x for x in info_ids):
        raise ContractError("info_id가 누락되거나 중복되었습니다")
    post_authors = {p["post_id"]: p["account_id"] for p in data["posts"]}
    for info in data["extracted_info"]:
        if info.get("type") not in TYPES:
            raise ContractError("허용되지 않은 개인정보 유형: " + str(info.get("type")))
        if info.get("source_account_id") not in by_id:
            raise ContractError("출처 계정이 분석 범위 밖입니다: " + str(info.get("info_id")))
        if not isinstance(info.get("value"), str):
            raise ContractError("info.value는 문자열이어야 합니다: " + str(info.get("info_id")))
        if info.get("source_post_id") is not None and info["source_post_id"] not in post_ids:
            raise ContractError("알 수 없는 출처 게시물: " + str(info.get("source_post_id")))
        if (
            info.get("source_post_id") is not None
            and post_authors[info["source_post_id"]] != info["source_account_id"]
        ):
            raise ContractError("개인정보 출처 계정과 게시물 작성자가 일치하지 않습니다")
        if not info.get("evidence_group_id") or not info.get("evidence"):
            raise ContractError("info마다 실제 evidence, evidence_group_id가 필요합니다")
    for rel in data["relations"]:
        if rel.get("type") not in RELATIONS:
            raise ContractError("허용되지 않은 관계 유형: " + str(rel.get("type")))
        if not rel.get("source_account_id") or not rel.get("target_account_id"):
            raise ContractError("관계에는 source_account_id, target_account_id가 필요합니다")


def scopes(data: dict) -> dict:
    return {a["account_id"]: a["scope"] for a in data["accounts"]}


def analysis_scope(data: dict, model_mode: str, tool_log: list[dict] | None = None) -> dict:
    selected = [a for a in data["accounts"] if a["scope"] == "secondary"]
    return {
        "primary_accounts": 1,
        "secondary_accounts": len(selected),
        "analyzed_accounts": len(selected) + 1,
        "coverage": data.get("analysis_scope", {}).get("coverage", "provided_data_only"),
        "model_mode": model_mode,
        "tool_calls": len(tool_log or []),
    }


def check_json3(data: dict) -> None:
    if data.get("schema_version") != "1.0":
        raise ContractError("JSON3 버전 누락")
    if data.get("analysis_scope", {}).get("model_mode") not in MODES:
        raise ContractError(
            "model_mode 허용값 오류: " + str(data.get("analysis_scope", {}).get("model_mode"))
        )
    if not isinstance(data.get("findings"), list):
        raise ContractError("findings는 배열이어야 합니다")
    for f in data["findings"]:
        for required in (
            "finding_id",
            "rule_id",
            "type",
            "value",
            "source_scope",
            "status",
            "description",
            "evidence",
        ):
            if required not in f:
                raise ContractError("JSON3 findings 필수 필드 누락: " + required)
        if f["source_scope"] not in {"direct", "cross"} or f["status"] not in STATUS:
            raise ContractError("source_scope/status 허용값 오류")
        if not isinstance(f["evidence"], list) or not f["evidence"]:
            raise ContractError("각 개인정보에는 최소 한 개의 출처가 필요합니다")
        for e in f["evidence"]:
            if e.get("account_scope") not in ROLES or not e.get("info_id"):
                raise ContractError("출처에는 account_scope와 info_id가 필요합니다")
    if "scores" in data or "risk" in data or "combinations" in data:
        raise ContractError("현재 버전은 위험 점수를 생성하지 않습니다")
