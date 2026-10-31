"""중앙 Agent가 쓰는 교차 추론 도구(review_detected_items, validate_and_submit_inferences).
모델 호출 루프는 Agent(백엔드)가 소유한다.

1. combine_detections로 수집 결과를 합친다(수집·검토가 늘어날 때마다 다시 합침).
2. extract_personal_info 응답에 agent_view(bundle, bundle_ref)를 넣고, inference_rules를 시스템 프롬프트에 넣는다.
   Agent가 review_detected_items를 호출하면 review_detected_items를 실행하고, 기록의 excluded_info_ids /
   uncertain_info_ids를 저장했다가 combine_detections에 넘긴다.
3. Agent가 validate_and_submit_inferences를 호출하면 백엔드가 새 validation_ref를 만들어 validate_and_submit을 실행한다.
   Agent에는 응답(개수·참조 ID)만 돌려주고, 기록(record)은 서버에 저장한다.
   다음 호출 때는 직전 record의 accepted를 previous로 넘긴다. 이전 후보도 현재 bundle 기준으로 다시 검증한다.
4. Agent가 한 번도 호출하지 않았어도 백엔드는 candidates=[]로 validate_and_submit을 실행해 validation_ref를 만든다.
   record["analysis"]가 calculate_risk와 최종 보고서의 입력이다.
"""

from __future__ import annotations

import re
from pathlib import Path

from .report import render
from .rules import (
    CONTACT_TYPES,
    HIDDEN_TYPES,
    RULE_MAP,
    SUPPORT_TYPES,
    blocked,
    verify_candidates,
)
from .schema import REVIEW_DECISIONS, ContractError, ensure_bundle

# 도구 한 번 호출에 받는 후보 수. 누적 후보(previous)에는 적용하지 않는다
MAX_CANDIDATES = 30
VALIDATE_TOOL = "validate_and_submit_inferences"
_SUBMIT_TYPES = sorted(set(RULE_MAP) - HIDDEN_TYPES)
REVIEW_TOOL = "review_detected_items"
# 도구 한 번 호출에 받는 검토 결정 수
MAX_DECISIONS = 50
# 후보 value로 써야 하는 유형만 값을 보낸다. 나머지(학번·생년월일 등)는 근거 ID로만 쓰므로 값을 가린다
_VALUE_TYPES = set(SUPPORT_TYPES) | {"sns_account"}

VALIDATE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": VALIDATE_TOOL,
        "description": "교차 추론 후보를 제출한다. 서버가 근거·연결을 검증해 통과한 결과를 저장하고 개수만 돌려준다.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": MAX_CANDIDATES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": _SUBMIT_TYPES},
                            "value": {
                                "type": "string",
                                "description": "추론 값. relationship이면 대상 주변 계정 아이디",
                            },
                            "rule_id": {
                                "type": "string",
                                "enum": sorted({RULE_MAP[t] for t in _SUBMIT_TYPES}),
                            },
                            "info_ids": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {"type": "string"},
                                "description": "근거 항목 ID('<계정>:I1')",
                            },
                            "status": {"type": "string", "enum": ["confirmed", "inferred"]},
                        },
                        "required": ["type", "value", "rule_id", "info_ids", "status"],
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    },
}

REVIEW_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": REVIEW_TOOL,
        "description": "OCR·정규식 탐지 항목의 오탐 여부를 기록한다. 원본은 지우지 않으며 나중에 되돌릴 수 있다.",
        "parameters": {
            "type": "object",
            "properties": {
                "bundle_ref": {"type": "string"},
                "decisions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_DECISIONS,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "info_id": {"type": "string"},
                            "decision": {"type": "string", "enum": sorted(REVIEW_DECISIONS)},
                            "reason": {
                                "type": "string",
                                "description": "이 결정을 뒷받침하는 실제 문맥 근거",
                            },
                        },
                        "required": ["info_id", "decision", "reason"],
                    },
                },
            },
            "required": ["bundle_ref", "decisions"],
            "additionalProperties": False,
        },
    },
}

_MASKS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}"),
    re.compile(r"\d{6,}"),
]


def _mask(text: str) -> str:
    """근거 문맥에 섞인 연락처·번호를 가린다."""
    for pattern in _MASKS:
        text = pattern.sub("[가림]", text)
    return text


def inference_rules(subject: str) -> str:
    """중앙 Agent 시스템 프롬프트에 넣을 교차 추론 규칙 설명."""
    text = (Path(__file__).parent / "prompts" / "rules.txt").read_text(encoding="utf-8")
    return text.replace("{subject}", subject)


def _visible(it: dict) -> bool:
    return (
        not blocked(it)
        and it["type"] in RULE_MAP
        and it["type"] not in HIDDEN_TYPES | CONTACT_TYPES
    )


def agent_view(bundle: dict, bundle_ref: str | None = None) -> dict:
    """Agent 컨텍스트에 넣을 정보(extract_personal_info 응답에 포함). 외부 모델에는 최소 정보만 보낸다.

    민감정보·고유식별/금융정보·얼굴·연락처/주소 항목은 빼고(omitted_count), 추론 값으로 쓰지 않는 유형은 값을 가린다.
    검토에서 제외·불확실로 표시한 항목은 review로 알려 준다(새 근거가 나오면 되돌릴 수 있도록 계속 보여 준다).
    근거 문맥은 번호·이메일을 가리고, 민감정보가 함께 탐지된 원문이면 아예 보내지 않는다.
    원문에는 악의적인 명령 문장이 섞일 수 있으므로, 백엔드는 이를 지시와 분리된 데이터로 전달해야 한다.
    """
    ensure_bundle(bundle)
    sensitive_inputs = {(it["account_id"], it["input_id"]) for it in bundle["items"] if blocked(it)}
    shown = [it for it in bundle["items"] if _visible(it)]
    return {
        "subject": bundle["subject"],
        "bundle_ref": bundle_ref,
        "accounts": [
            {"account_id": a["account_id"], "scope": a["scope"]} for a in bundle["accounts"]
        ],
        "links": [
            {"account_id": link["account_id"], "kind": link["kind"], "source_id": link["source_id"]}
            for link in bundle["links"]
        ],
        "items": [
            {
                "info_id": it["info_id"],
                "entity_id": it["entity_id"],
                "account_id": it["account_id"],
                "scope": it["scope"],
                "type": it["type"],
                "value": it["value"] if it["type"] in _VALUE_TYPES else "[가림]",
                "source_id": it["source_id"],
                "related": it["related"],
                "subject_verified": it["subject_verified"],
                "review": it["review"],
                "confidence": it["confidence"],
                "evidence": ""
                if (it["account_id"], it["input_id"]) in sensitive_inputs
                else _mask(it["evidence"][:120]),
            }
            for it in shown
        ],
        "omitted_count": len(bundle["items"]) - len(shown),
    }


def _candidates(candidates) -> list:
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
        raise ContractError(f"candidates는 최대 {MAX_CANDIDATES}개 배열이어야 합니다")
    return candidates


def build_result(
    bundle: dict, candidates: list[dict] | None = None, *, rejected: list[dict] | None = None
) -> dict:
    """본인 확인 정보 + 검증을 다시 통과한 후보로 개인정보 분석 결과를 만든다.

    candidates 수는 제한하지 않는다(여러 번 제출해 누적된 후보). 검증을 건너뛴 후보는 들어가지 않는다.
    rejected에 리스트를 넘기면 재검증에서 버려진 후보의 {index, reason}이 추가된다(백엔드 로그용).
    """
    ensure_bundle(bundle)
    if not isinstance(candidates or [], list):
        raise ContractError("candidates는 배열이어야 합니다")
    accepted, dropped = verify_candidates(bundle, candidates or [])
    if rejected is not None:
        rejected.extend(dropped)
    return render(bundle, accepted)


def review_detected_items(bundle: dict, args: dict) -> tuple[dict, dict | None]:
    """review_detected_items 실행. (Agent에 돌려줄 응답, 서버에 저장할 기록)을 반환한다.

    기록의 excluded_info_ids / uncertain_info_ids는 현재 bundle의 검토 상태에 이번 결정을 반영한 전체 목록이다.
    keep은 이전 제외·불확실 표시를 해제한다. Agent에 보이지 않는 항목(민감정보 등)은 검토할 수 없다.
    bundle_ref가 현재 bundle과 같은지는 백엔드가 확인한다. 인자 형식이 잘못되면 기록은 None.
    """
    ensure_bundle(bundle)
    decisions = args.get("decisions") if isinstance(args, dict) else None
    if not isinstance(decisions, list) or not decisions or len(decisions) > MAX_DECISIONS:
        return {"status": "error", "error_code": "INVALID_DECISIONS"}, None
    by_id = {it["info_id"]: it for it in bundle["items"]}
    state = {it["info_id"]: it["review"] for it in bundle["items"] if it["review"]}
    applied: list[dict] = []
    rejected: list[dict] = []
    for index, d in enumerate(decisions):
        if not isinstance(d, dict) or d.get("decision") not in REVIEW_DECISIONS:
            reason = "invalid_decision"
        elif not isinstance(d.get("info_id"), str) or d["info_id"] not in by_id:
            reason = "unknown_info_id"
        elif not _visible(by_id[d["info_id"]]):
            reason = "not_reviewable"
        elif not isinstance(d.get("reason"), str) or not d["reason"].strip():
            reason = "missing_reason"
        else:
            reason = None
        if reason:
            rejected.append({"index": index, "reason": reason})
            continue
        info_id, decision = d["info_id"], d["decision"]
        if decision == "keep":
            state.pop(info_id, None)
        else:
            state[info_id] = "excluded" if decision == "exclude" else "uncertain"
        applied.append({"info_id": info_id, "decision": decision, "reason": d["reason"].strip()})
    record = {
        "decisions": applied,
        "rejected": rejected,
        "excluded_info_ids": sorted(i for i, s in state.items() if s == "excluded"),
        "uncertain_info_ids": sorted(i for i, s in state.items() if s == "uncertain"),
    }
    response = {
        "status": "ok",
        "applied_count": len(applied),
        "rejected_count": len(rejected),
        "excluded_count": len(record["excluded_info_ids"]),
        "uncertain_count": len(record["uncertain_info_ids"]),
    }
    return response, record


def validate_and_submit(
    bundle: dict,
    args: dict,
    *,
    validation_ref: str,
    previous: list[dict] | None = None,
    bundle_ref: str | None = None,
) -> tuple[dict, dict | None]:
    """validate_and_submit_inferences 실행. (Agent에 돌려줄 응답, 서버에 저장할 기록)을 반환한다.

    validation_ref는 백엔드가 만든다(LLM이 지정하지 않음). 인자가 잘못되면 기록은 None.
    bundle_ref를 기록해 두면 점수 계산 전에 최신 bundle에서 나온 검증인지 확인할 수 있다.
    기록의 rejected·revoked에는 {index, reason}만 있고 개인정보 원문은 없다.
    """
    ensure_bundle(bundle)
    try:
        candidates = _candidates(args.get("candidates") if isinstance(args, dict) else None)
    except ContractError:
        return {"status": "error", "error_code": "INVALID_CANDIDATES"}, None
    kept, revoked = verify_candidates(bundle, previous or [])
    accepted, rejected = verify_candidates(bundle, candidates, prior=kept)
    record = {
        "validation_ref": validation_ref,
        "subject": bundle["subject"],
        "bundle_ref": bundle_ref,
        "accepted": kept + accepted,
        "rejected": rejected,
        "revoked": revoked,
        "analysis": render(bundle, kept + accepted),
    }
    response = {
        "validation_ref": validation_ref,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "status": "ok",
    }
    return response, record
