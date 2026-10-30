"""1차/2차에 같은 AI1 추출 파이프라인을 실행한 JSON2 결과를 합친다."""

from __future__ import annotations

import json

from .contracts import ContractError, ensure_json2


def combine_json2(*batches: dict) -> dict:
    if not batches:
        raise ContractError("최소 1개 JSON2가 필요합니다")
    subject = batches[0].get("subject_account_id")
    result = {
        "schema_version": "1.0",
        "subject_account_id": subject,
        "analysis_scope": {"coverage": "provided_batches"},
        "accounts": [],
        "posts": [],
        "relations": [],
        "extracted_info": [],
    }
    account_index = {}
    post_index = {}
    info_index = {}
    rel_keys = set()
    for batch in batches:
        ensure_json2(batch)
        if batch["subject_account_id"] != subject:
            raise ContractError("1차/2차의 분석 대상 계정 ID가 달라 병합할 수 없습니다")
        for account in batch["accounts"]:
            key = account["account_id"]
            if key in account_index:
                if account_index[key] != account["scope"]:
                    raise ContractError("동일한 계정 ID의 scope가 서로 다릅니다")
            else:
                account_index[key] = account["scope"]
                result["accounts"].append(account)
        for src, index, key in (
            ("posts", post_index, "post_id"),
            ("extracted_info", info_index, "info_id"),
        ):
            for obj in batch[src]:
                ident = obj[key]
                if ident in index:
                    if index[ident] != obj:
                        raise ContractError(f"{src}의 ID {ident}가 중복되지만 데이터가 다릅니다")
                else:
                    index[ident] = obj
                    result[src].append(obj)
        for rel in batch["relations"]:
            key = json.dumps(rel, sort_keys=True, ensure_ascii=False)
            if key not in rel_keys:
                rel_keys.add(key)
                result["relations"].append(rel)
    ensure_json2(result)
    return result
