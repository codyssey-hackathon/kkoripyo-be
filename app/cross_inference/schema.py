"""입력 계약: 개인정보 탐지 결과(items·entities)와 이를 합친 교차 추론 입력(bundle), 근거 위치(source_id) 규칙."""

from __future__ import annotations

import re

# 개인정보 탐지 모듈이 내보내는 schema_version 값(외부 계약이라 그대로 둔다)
DETECTION_VERSION = "json2/v1"
BUNDLE_VERSION = "cross-inference-input/v2"
RESULT_VERSION = "cross-inference-result/v1"

# 개인정보 탐지 AI 명세의 items[].type / category
TYPES = {
    "name",
    "phone",
    "email",
    "address",
    "birth_date",
    "student_id",
    "rrn",
    "passport",
    "driver_license",
    "alien_reg",
    "card_number",
    "bank_account",
    "vehicle_plate",
    "sns_account",
    "affiliation",
    "location",
    "sensitive",
    "other_clue",
}
CATEGORIES = {"고유식별정보", "민감정보", "금융정보", "일반개인정보", "준식별자"}
MODALITIES = {"text", "image"}
# 본인과의 연결로 인정하는 관계. 팔로워·팔로잉은 탐색 단서일 뿐 links로 받지 않는다
LINK_KINDS = {"tagged", "mentioned", "commented"}
# Agent의 탐지 항목 검토(review_detected_items) 결정
REVIEW_DECISIONS = {"keep", "exclude", "uncertain"}

# post:<게시물 ID>.<caption|commentN|...>
_POST_INPUT = re.compile(r"^post:([^.]+)\.([a-z_]+?)(\d*)$")


class ContractError(ValueError):
    pass


def clean(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").casefold())


def account_key(value: str) -> str:
    return (value or "").strip().lstrip("@").casefold()


def locate(account: str, input_id: str, image_posts: dict[str, str]) -> dict:
    """탐지 결과의 source.input_id를 출처 위치로 바꾼다.

    part: profile | caption | comment | image | 그 밖의 게시물 필드 | other
    unit: 본인 연결을 판단하는 단위(post:<ID> 또는 profile). 이미지는 image_posts로 게시물을 알 때만 post:<ID>.
    source_id: '{account_id}:{post_id}:{kind}:{index}'. 프로필은 '{account_id}:profile:{필드}:0',
      이미지는 index 자리에 파일명을 쓴다(게시물을 모르면 post_id 자리가 '_').
    """
    m = _POST_INPUT.match(input_id)
    if m:
        post_id, kind, index = m.group(1), m.group(2), m.group(3) or "0"
        return {
            "part": kind,
            "post_id": post_id,
            "unit": "post:" + post_id,
            "source_id": f"{account}:{post_id}:{kind}:{index}",
        }
    if input_id.startswith("profile."):
        field = input_id.removeprefix("profile.")
        return {
            "part": "profile",
            "post_id": None,
            "unit": "profile",
            "source_id": f"{account}:profile:{field}:0",
        }
    if input_id.startswith("image:"):
        name = input_id.removeprefix("image:")
        post_id = image_posts.get(name)
        return {
            "part": "image",
            "post_id": post_id,
            "unit": "post:" + post_id if post_id else input_id,
            "source_id": f"{account}:{post_id or '_'}:image:{name}",
        }
    return {
        "part": "other",
        "post_id": None,
        "unit": input_id,
        "source_id": f"{account}:{input_id}",
    }


def scopes(bundle: dict) -> dict[str, str]:
    return {a["account_id"]: a["scope"] for a in bundle["accounts"]}


def ensure_detection(data: dict) -> None:
    """계정 하나에 대한 탐지 결과를 검증한다."""
    if not isinstance(data, dict) or data.get("schema_version") != DETECTION_VERSION:
        raise ContractError(f"탐지 결과 schema_version은 '{DETECTION_VERSION}'이어야 합니다")
    if not isinstance(data.get("subject"), str) or not data["subject"].strip():
        raise ContractError("탐지 결과 subject(분석 계정 아이디)가 필요합니다")
    if not isinstance(data.get("items"), list) or not isinstance(data.get("entities"), list):
        raise ContractError("items/entities는 배열이어야 합니다")
    item_ids = set()
    for item in data["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ContractError("items[].id는 문자열이어야 합니다")
        if item["id"] in item_ids:
            raise ContractError("items[].id 중복: " + item["id"])
        item_ids.add(item["id"])
        if item.get("type") not in TYPES:
            raise ContractError("허용되지 않은 개인정보 유형: " + str(item.get("type")))
        if item.get("category") not in CATEGORIES:
            raise ContractError("허용되지 않은 category: " + str(item.get("category")))
        if not isinstance(item.get("value"), str):
            raise ContractError("items[].value는 문자열이어야 합니다: " + item["id"])
        if not isinstance(item.get("normalized"), str | None):
            raise ContractError("items[].normalized는 문자열이어야 합니다: " + item["id"])
        source = item.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("input_id"), str):
            raise ContractError("items[].source.input_id가 필요합니다: " + item["id"])
        if source.get("modality") not in MODALITIES:
            raise ContractError("source.modality는 text/image 중 하나입니다: " + item["id"])
        if not isinstance(item.get("confidence"), int | float):
            raise ContractError("items[].confidence는 숫자여야 합니다: " + item["id"])
    entity_ids = set()
    for entity in data["entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("entity_id"), str):
            raise ContractError("entities[].entity_id는 문자열이어야 합니다")
        if entity["entity_id"] in entity_ids:
            raise ContractError("entity_id 중복: " + entity["entity_id"])
        entity_ids.add(entity["entity_id"])
        if entity.get("type") not in TYPES:
            raise ContractError("허용되지 않은 개인정보 유형: " + str(entity.get("type")))
        refs = entity.get("item_ids")
        if not isinstance(refs, list) or not refs or any(r not in item_ids for r in refs):
            raise ContractError("entity의 item_ids가 비었거나 없는 item을 가리킵니다")


def ensure_bundle(bundle: dict) -> None:
    """combine_detections 결과인지 검증한다."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != BUNDLE_VERSION:
        raise ContractError(
            f"교차 추론 입력은 combine_detections로 만든 '{BUNDLE_VERSION}'이어야 합니다"
        )
    for key in ("subject", "accounts", "items", "entities", "links"):
        if key not in bundle:
            raise ContractError("교차 추론 입력 필수 필드 누락: " + key)
    roles = scopes(bundle)
    if roles.get(bundle["subject"]) != "primary" or list(roles.values()).count("primary") != 1:
        raise ContractError("subject는 유일한 primary 계정이어야 합니다")
    if any(it.get("account_id") not in roles for it in bundle["items"]):
        raise ContractError("알 수 없는 계정의 item이 있습니다")
