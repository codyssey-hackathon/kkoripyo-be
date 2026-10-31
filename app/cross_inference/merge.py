"""본인과 주변 계정의 개인정보 탐지 결과를 하나로 합치고, 계정 간 관계(태그·언급·댓글·팔로우)를 정리한다.

탐지 결과는 계정마다 따로 나오고 I1·E1이 파일 안에서만 고유하므로 '<계정>:I1', '<계정>:E1'처럼 계정을 앞에 붙인다.
원본 탐지 결과의 필드는 수정하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable

from .rules import annotate, value_key
from .schema import (
    BUNDLE_VERSION,
    LINK_KINDS,
    ContractError,
    account_key,
    ensure_bundle,
    ensure_detection,
    locate,
)


def _items(account: str, scope: str, data: dict, image_posts: dict[str, str]) -> list[dict]:
    entity_of: dict[str, str] = {}
    sensitive_entities = set()
    by_id = {i["id"]: i for i in data["items"]}
    for e in data["entities"]:
        for i in e["item_ids"]:
            entity_of.setdefault(i, e["entity_id"])
            if by_id[i]["type"] == "sensitive" or by_id[i]["category"] == "민감정보":
                sensitive_entities.add(e["entity_id"])
    result = []
    for m in data["items"]:
        entity_id = entity_of.get(m["id"])
        result.append(
            {
                "info_id": f"{account}:{m['id']}",
                "entity_id": f"{account}:{entity_id}" if entity_id else None,
                "account_id": account,
                "scope": scope,
                "type": m["type"],
                "category": m["category"],
                "value": m["value"],
                "norm": value_key(m["type"], m.get("normalized") or m["value"]),
                "handle": account_key(m.get("normalized") or m["value"])
                if m["type"] == "sns_account"
                else None,
                "confidence": m["confidence"],
                "input_id": m["source"]["input_id"],
                "modality": m["source"]["modality"],
                "method": m["source"].get("method"),
                "bbox": m["source"].get("bbox"),
                "evidence": str(m.get("evidence") or ""),
                # 하나라도 민감정보로 분류된 묶음은 전체를 민감정보로 취급
                "sensitive": m["type"] == "sensitive"
                or m["category"] == "민감정보"
                or entity_id in sensitive_entities,
                **locate(account, m["source"]["input_id"], image_posts),
            }
        )
    return result


def combine_detections(
    primary: dict,
    *neighbors: dict,
    links: list[dict] | None = None,
    image_posts: dict[str, dict[str, str]] | None = None,
    excluded_info_ids: Iterable[str] = (),
    uncertain_info_ids: Iterable[str] = (),
) -> dict:
    """primary: 본인 계정 탐지 결과, neighbors: 추가 조사한 주변 계정 탐지 결과.

    links: 수집 데이터에서 확인한 계정 관계 [{account_id, kind, post_id?, source_id?}].
      kind는 tagged/mentioned/commented(팔로워·팔로잉은 받지 않음). post_id는 그 관계가 확인된
      '주변 계정의' 게시물이며, 그 게시물에서 나온 정보가 본인과 연결된 근거가 된다. 생략 가능.
    image_posts: {account_id: {이미지 파일명: post_id}}. 탐지 결과의 image:<파일명>이 어느 게시물 것인지 알려 준다.
    excluded_info_ids / uncertain_info_ids: review_detected_items로 기록된 검토 상태.
      제외 항목은 원본에 남기되 본인 판별·근거·결과에서 빠진다. 불확실 항목은 표시만 한다.

    다음 관계는 탐지 결과에서 자동으로 만든다(mentioned). 댓글은 작성자를 알 수 없어 제외한다.
      - 주변 계정의 본문·프로필에 본인 아이디가 sns_account로 잡힌 위치(그 게시물이 본인과 연결됨)
      - 본인의 본문·프로필에 주변 계정 아이디가 sns_account로 잡힌 경우(관계만, 주변 계정 게시물은 연결 안 됨)
    """
    image_posts = image_posts or {}
    ensure_detection(primary)
    subject = account_key(primary["subject"])
    accounts: list[dict] = []
    entities: list[dict] = []
    items: list[dict] = []

    for data, scope in [(primary, "primary"), *((n, "secondary") for n in neighbors)]:
        ensure_detection(data)
        account = account_key(data["subject"])
        if any(a["account_id"] == account for a in accounts):
            raise ContractError("같은 계정의 탐지 결과가 두 번 들어왔습니다: " + account)
        accounts.append(
            {"account_id": account, "scope": scope, "llm_used": bool(data.get("llm_used"))}
        )
        items.extend(_items(account, scope, data, image_posts.get(account, {})))
        entities.extend(
            {
                "entity_id": f"{account}:{e['entity_id']}",
                "account_id": account,
                "type": e["type"],
                "value": e.get("value"),
                "info_ids": [f"{account}:{i}" for i in e["item_ids"]],
                "match_reasons": list(e.get("match_reasons") or []),
            }
            for e in data["entities"]
        )

    neighbor_ids = {a["account_id"] for a in accounts if a["scope"] == "secondary"}
    found: dict[tuple, None] = {}  # (account, kind, unit, source_id), 순서 유지용
    excluded, uncertain = set(excluded_info_ids), set(uncertain_info_ids)
    known = {it["info_id"] for it in items}
    if (excluded | uncertain) - known:
        raise ContractError("검토 상태에 없는 info_id가 있습니다")
    for it in items:
        it["review"] = (
            "excluded"
            if it["info_id"] in excluded
            else "uncertain"
            if it["info_id"] in uncertain
            else None
        )

    own_posts = {it["post_id"] for it in items if it["scope"] == "primary" and it["post_id"]}
    for it in items:
        if it["type"] != "sns_account" or it["part"] == "comment" or it["review"] == "excluded":
            continue
        if it["scope"] == "secondary" and it["handle"] == subject:
            found[(it["account_id"], "mentioned", it["unit"], it["source_id"])] = None
        elif it["scope"] == "primary" and it["handle"] in neighbor_ids:
            found[(it["handle"], "mentioned", None, it["source_id"])] = None

    for link in links or []:
        if not isinstance(link, dict):
            raise ContractError("links 항목은 객체여야 합니다")
        account = account_key(str(link.get("account_id", "")))
        if account not in neighbor_ids:
            raise ContractError("links.account_id는 주변 계정이어야 합니다: " + account)
        if link.get("kind") not in LINK_KINDS:
            raise ContractError("links.kind 허용값 오류: " + str(link.get("kind")))
        post_id, source_id = link.get("post_id"), link.get("source_id")
        if not isinstance(post_id, str | None) or not isinstance(source_id, str | None):
            raise ContractError("links.post_id/source_id는 문자열이어야 합니다")
        if post_id in own_posts:
            raise ContractError("links.post_id는 주변 계정의 게시물이어야 합니다: " + post_id)
        found[(account, link["kind"], f"post:{post_id}" if post_id else None, source_id)] = None

    bundle = {
        "schema_version": BUNDLE_VERSION,
        "subject": subject,
        "accounts": accounts,
        "items": items,
        "entities": entities,
        "links": [{"account_id": a, "kind": k, "unit": u, "source_id": s} for a, k, u, s in found],
    }
    ensure_bundle(bundle)
    annotate(bundle)
    return bundle
