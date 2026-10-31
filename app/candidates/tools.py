"""list_account_candidates 실행. 저장된 수집 결과에서 주변 계정 후보와 연결 근거를 페이지 단위로 돌려준다.

1. Agent가 list_account_candidates를 호출하면 백엔드가 이번 분석 실행의 subject와,
   result_ref로 저장된 수집 결과({source_tool, account_id, data[, post_id]})를 돌려주는 함수를 넘긴다.
   다른 실행의 result_ref나 없는 result_ref는 None을 돌려줘야 한다.
2. 새 SNS 스크래핑은 하지 않는다. 기록할 것이 없어 응답만 반환한다.

priority_score는 조사 우선순위용 보조 신호이며 최종 위험도와 별개다.
follower·following은 탐색 후보일 뿐 combine_detections의 links로 넘기지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable

from .extract import extract_candidates

LIST_TOOL = "list_account_candidates"
DEFAULT_LIMIT = 50
# 한 페이지 최대 후보 수(응답 크기 한도)
MAX_LIMIT = 50

LIST_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": LIST_TOOL,
        "description": (
            "저장된 수집 결과에서 본인과 연결된 주변 계정 후보와 근거 source_id를 조회한다. "
            "새로 수집하지 않는다. priority_score는 조사 우선순위일 뿐 위험도가 아니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_refs": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string"},
                    "description": "수집 도구 응답의 result_ref",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "default": DEFAULT_LIMIT,
                },
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["result_refs"],
            "additionalProperties": False,
        },
    },
}


def _error(code: str) -> dict:
    return {"status": "error", "error_code": code}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def list_account_candidates(
    args: dict, *, subject: str, results: Callable[[str], dict | None]
) -> dict:
    """list_account_candidates 실행. Agent에 돌려줄 응답을 반환한다.

    result_refs가 비었거나 하나라도 찾을 수 없으면 INVALID_RESULT_REF,
    limit(1~50)·offset(0 이상)이 잘못되면 INVALID_PAGINATION.
    """
    refs = args.get("result_refs") if isinstance(args, dict) else None
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
        return _error("INVALID_RESULT_REF")
    stored = [results(r) for r in dict.fromkeys(refs)]
    if any(s is None for s in stored):
        return _error("INVALID_RESULT_REF")
    limit, offset = args.get("limit", DEFAULT_LIMIT), args.get("offset", 0)
    if not _is_int(limit) or not 1 <= limit <= MAX_LIMIT or not _is_int(offset) or offset < 0:
        return _error("INVALID_PAGINATION")

    candidates = extract_candidates(subject, stored)
    end = offset + limit
    return {
        "candidates": candidates[offset:end],
        "total_count": len(candidates),
        "next_offset": end if end < len(candidates) else None,
        "status": "ok",
    }
