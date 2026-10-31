"""본인 정보 판별과 교차 추론 후보 검증. 점수 계산 / 민감정보 추론 없음.

탐지 방식(regex·OCR)과 confidence는 본인 확인 근거로 쓰지 않는다. 출처 위치와 값 일치로만 판단한다.
"""

from __future__ import annotations

from .schema import LINK_KINDS, account_key, clean, scopes

RULE_MAP = {
    "name": "R01",
    "sns_account": "R01",
    "affiliation": "R02",
    "student_id": "R02",
    "location": "R03",
    "phone": "R04",
    "email": "R04",
    "address": "R04",
    "rrn": "R05",
    "passport": "R05",
    "driver_license": "R05",
    "alien_reg": "R05",
    "card_number": "R05",
    "bank_account": "R05",
    "birth_date": "R06",
    "vehicle_plate": "R06",
    "other_clue": "R06",
    "activity_pattern": "R07",
    "relationship": "R08",
}
# 고유식별·금융정보는 외부 LLM에 보내지 않고 후보로도 받지 않는다. 본인 확인되면 직접 노출로만 보고한다
HIDDEN_TYPES = {t for t, r in RULE_MAP.items() if r == "R05"}
# 연락처·주소는 추론하지 않으므로 외부 LLM에 원문을 보낼 필요가 없다
CONTACT_TYPES = {t for t, r in RULE_MAP.items() if r == "R04"}
# 개별 탐지 결과에는 없고 여러 근거를 연결해야만 나오는 유형
DERIVED_TYPES = {"activity_pattern", "relationship"}
# 교차 '추론'을 허용하는 유형과, 그때 함께 제출해야 하는 본인 확인 근거 유형
SUPPORT_TYPES = {
    "name": {"name", "sns_account"},
    "affiliation": {"affiliation", "student_id", "other_clue"},
    "location": {"location", "affiliation", "other_clue"},
    "other_clue": {"affiliation", "student_id", "other_clue", "location"},
}
# 같은 값이 흔한 유형은 주변 계정에서 같은 값이 나와도 본인과 연결된 게시물일 때만 같은 정보로 본다
COMMON_TYPES = {"name", "affiliation", "location", "other_clue", "birth_date"}
# 프로필 소개글에는 다른 사람 이름이 들어갈 수 있으므로 이름은 이름 칸에서만 본인 것으로 본다
NAME_FIELDS = {"profile.full_name"}
# 생활 패턴은 서로 다른 게시물에서 같은 장소가 이만큼 반복될 때만 인정
PATTERN_MIN_POSTS = 2


def blocked(it: dict) -> bool:
    return it["sensitive"]


def excluded(it: dict) -> bool:
    """Agent 검토에서 오탐으로 제외된 항목. 원본은 남기되 판별·근거에 쓰지 않는다."""
    return it["review"] == "excluded"


def key(it: dict) -> tuple[str, str]:
    return it["type"], it["norm"]


def linked_units(bundle: dict) -> dict[str, set[str]]:
    """주변 계정별로 본인과 태그·언급·댓글로 연결된 위치."""
    units: dict[str, set[str]] = {}
    for link in bundle["links"]:
        if link["kind"] in LINK_KINDS and link["unit"]:
            units.setdefault(link["account_id"], set()).add(link["unit"])
    return units


def _profile_anchor(it: dict, subject: str) -> bool:
    if it["scope"] != "primary" or it["part"] != "profile" or not it["norm"]:
        return False
    if blocked(it) or excluded(it):
        return False
    if it["type"] == "sns_account":
        return it["handle"] == subject
    if it["type"] == "name":
        return it["input_id"] in NAME_FIELDS
    return True


def annotate(bundle: dict) -> None:
    """항목마다 본인 관련 여부를 표시한다(명세 4-2의 subject_verified를 규칙으로 생성).

    related: 본인 계정의 본문·프로필·이미지, 또는 본인과 연결된 주변 계정 게시물에서 나옴.
      댓글은 작성자를 알 수 없어 제외한다.
    subject_verified: 본인의 정보로 볼 근거가 있음
      1) 본인 프로필에 직접 쓴 값(이름은 이름 칸, 계정은 본인 아이디만)
      2) 본인 이미지에서 1)의 이름과 함께 나온 값(예: 학생증의 학번·소속).
         그 이미지에 같은 유형의 값이 여럿이면(명단 등) 그 유형은 제외
      3) 1)·2)와 유형·값이 같거나, 탐지 단계에서 그와 같은 정보로 묶은(entity) 항목.
         단 이름·소속처럼 흔한 값은 주변 계정에서는 본인과 연결된 게시물일 때만
    판단할 수 없는 항목은 subject_verified=False로 남긴다.
    """
    subject = bundle["subject"]
    units = linked_units(bundle)
    items = bundle["items"]
    for it in items:
        it["related"] = (
            not excluded(it)
            and it["part"] != "comment"
            and (it["scope"] == "primary" or it["unit"] in units.get(it["account_id"], ()))
        )
    anchors = {key(it) for it in items if _profile_anchor(it, subject)}
    named = {
        it["input_id"]
        for it in items
        if it["scope"] == "primary"
        and it["modality"] == "image"
        and it["type"] == "name"
        and key(it) in anchors
        and not excluded(it)
    }
    # 명단·단체 사진처럼 같은 유형의 값이 여럿이면 누구 것인지 알 수 없으므로 하나뿐인 유형만 인정
    values: dict[tuple, set[str]] = {}
    for it in items:
        if it["scope"] == "primary" and it["input_id"] in named and it["norm"] and not excluded(it):
            values.setdefault((it["input_id"], it["type"]), set()).add(it["norm"])
    anchors |= {
        (typ, *norms)
        for (_, typ), norms in values.items()
        if typ not in ("name", "sns_account") and len(norms) == 1
    }

    def allowed(it: dict) -> bool:
        return (
            not blocked(it)
            and not excluded(it)
            and (it["scope"] == "primary" or it["type"] not in COMMON_TYPES or it["related"])
        )

    entities = {it["entity_id"] for it in items if key(it) in anchors and allowed(it)} - {None}
    for it in items:
        it["subject_verified"] = allowed(it) and (key(it) in anchors or it["entity_id"] in entities)


def _groups(items: list[dict]) -> list[list[dict]]:
    """같은 유형·값이거나 탐지 단계에서 같은 entity로 묶은 항목을 하나로 모은다."""
    parent: dict[tuple, tuple] = {}

    def root(x: tuple) -> tuple:
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for it in items:
        a = root(("key", *key(it)))
        b = root(("entity", it["entity_id"] or it["info_id"]))
        parent[b] = a
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        groups.setdefault(root(("key", *key(it))), []).append(it)
    return list(groups.values())


def direct_findings(bundle: dict) -> list[dict]:
    """본인 정보로 확인된 항목(subject_verified). LLM 없이 코드가 확정한다."""
    verified = [it for it in bundle["items"] if it["subject_verified"] and it["type"] in RULE_MAP]
    result = []
    for group in _groups(verified):
        head = next((it for it in group if it["part"] == "profile"), group[0])
        result.append(
            {
                "rule_id": RULE_MAP[head["type"]],
                "type": head["type"],
                "value": head["value"],
                "status": "confirmed",
                "info_ids": [it["info_id"] for it in group],
                "source_scope": "direct",
            }
        )
    return result


def confirmed_values(bundle: dict) -> dict[tuple[str, str], str]:
    """본인 확인 정보의 (유형, 비교값) -> 대표 값. 로마자 표기 등 같은 정보로 묶인 값도 포함한다."""
    by_id = {it["info_id"]: it for it in bundle["items"]}
    result = {}
    for d in direct_findings(bundle):
        for it in (by_id[i] for i in d["info_ids"]):
            result[key(it)] = d["value"]
            result[(it["type"], value_key(it["type"], it["value"]))] = d["value"]
    return result


def unverified_items(bundle: dict) -> list[dict]:
    """본인 계정에서 탐지됐지만 본인 정보인지 판단할 수 없는 항목. 값은 내보내지 않는다."""
    return [
        {"info_id": it["info_id"], "type": it["type"], "source_id": it["source_id"]}
        for it in bundle["items"]
        if it["scope"] == "primary"
        and not it["subject_verified"]
        and not blocked(it)
        and not excluded(it)
        and it["type"] in RULE_MAP
    ]


def value_key(typ: str, value: str) -> str:
    # 계정 아이디는 '.'·'_'만 달라도 다른 계정이므로 기호를 지우지 않는다
    return account_key(value) if typ == "sns_account" else clean(value)


def has_value(it: dict, value: str) -> bool:
    v = value_key(it["type"], value)
    return bool(v) and v in (it["norm"], value_key(it["type"], it["value"]))


def mentions_value(it: dict, text: str) -> bool:
    """자유 서술 안에 근거 값이 들어 있는지(예: '매주 가평 방문' ⊃ '가평')."""
    t = clean(text)
    return any(n and n in t for n in (it["norm"], clean(it["value"])))


def _check_pattern(items: list[dict], value: str) -> str | None:
    """같은 장소가 본인 관련 게시물 여러 곳에서 반복되는지 확인한다."""
    places = [it for it in items if it["type"] == "location" and mentions_value(it, value)]
    if not places:
        return "value_not_in_evidence"
    places = [it for it in places if it["related"]]
    if not places:
        return "no_subject_link"
    norms = {places[0]["norm"]}
    if any(it["norm"] not in norms for it in places):
        return "pattern_place_mismatch"
    posts = {(it["account_id"], it["unit"]) for it in places if it["unit"].startswith("post:")}
    if len(posts) < PATTERN_MIN_POSTS:
        return "pattern_needs_repeat"
    return None


def _evidence(items: list[dict], typ: str, value: str) -> list[dict]:
    """통과한 후보의 근거에서 본인과 무관한 항목(연결 안 된 게시물 등)을 뺀다."""
    if typ == "relationship":
        return items
    if typ == "activity_pattern":
        return [
            it
            for it in items
            if it["type"] == "location" and it["related"] and mentions_value(it, value)
        ]
    return [
        it
        for it in items
        if (it["related"] and it["type"] == typ and has_value(it, value))
        or (it["scope"] == "primary" and it["subject_verified"])
    ]


def _check_relationship(bundle: dict, items: list[dict], target: str) -> str | None:
    """관계는 태그·언급·댓글 연결이 확인된 주변 계정에 대해서만. 팔로우만으로는 인정하지 않는다."""
    if scopes(bundle).get(target) != "secondary":
        return "unknown_target"
    if not any(
        link["account_id"] == target and link["kind"] in LINK_KINDS for link in bundle["links"]
    ):
        return "no_subject_link"
    if any(it["account_id"] not in (target, bundle["subject"]) for it in items):
        # 제3자 계정의 정보로 본인과 대상 계정의 관계를 만들지 않는다
        return "foreign_evidence"
    return None


def _check_inferred(items: list[dict], typ: str, matched: list[dict]) -> str | None:
    if typ not in SUPPORT_TYPES:
        # 연락처·주소·계정 ID·학번 등은 새로 '추측'하지 않고 직접 확인된 것만 보고한다
        return "not_inferable_type"
    if not any(it["related"] for it in matched):
        return "no_subject_link"
    if not any(
        it["scope"] == "primary" and it["subject_verified"] and it["type"] in SUPPORT_TYPES[typ]
        for it in items
    ):
        return "support_gate"
    return None


def _scope(items: list[dict]) -> str:
    return "cross_account" if any(it["scope"] == "secondary" for it in items) else "direct"


def check_candidate(bundle: dict, c: dict) -> tuple[dict | None, str | None]:
    """모델이 제안한 후보를 근거·계정·연결로 재검증한다. (통과한 후보, 거절 사유)를 반환."""
    by_id = {it["info_id"]: it for it in bundle["items"]}
    typ, ids, value, status = (c.get(k) for k in ("type", "info_ids", "value", "status"))
    if not isinstance(typ, str) or typ not in RULE_MAP:
        return None, "unknown_type"
    if c.get("rule_id") != RULE_MAP[typ]:
        return None, "rule_type_mismatch"
    if status not in {"confirmed", "inferred"}:
        return None, "invalid_status"
    if not isinstance(value, str) or not value.strip():
        return None, "invalid_value"
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        return None, "invalid_info_ids"
    if any(x not in by_id for x in ids):
        return None, "unknown_info_id"
    items = [by_id[x] for x in dict.fromkeys(ids)]
    if any(excluded(it) for it in items):
        return None, "excluded_info"
    if any(blocked(it) for it in items):
        return None, "sensitive_info"
    if typ in HIDDEN_TYPES or any(it["type"] in HIDDEN_TYPES for it in items):
        return None, "hidden_type"

    result = {
        "rule_id": c["rule_id"],
        "type": typ,
        "value": value.strip(),
        "status": status,
        "info_ids": [it["info_id"] for it in items],
        "source_scope": _scope(items),
    }
    if typ in DERIVED_TYPES:
        if status != "inferred":
            return None, "invalid_status"
        if typ == "activity_pattern":
            error = _check_pattern(items, value)
        else:
            target = account_key(value)
            error = _check_relationship(bundle, items, target)
            # 값은 LLM이 붙인 관계 이름이 아니라 대상 계정과 확인된 연결 종류로 정한다
            result["value"] = target
            result["relation_kinds"] = sorted(
                {link["kind"] for link in bundle["links"] if link["account_id"] == target}
                & LINK_KINDS
            )
        if error:
            return None, error
        result["info_ids"] = [it["info_id"] for it in _evidence(items, typ, value)]
        return result, None

    # 값뿐 아니라 유형까지 일치하는 근거만 인정. 원문 어디에도 없는 정보는 LLM이 새로 만들어 냈을 가능성이 있음
    matched = [it for it in items if it["type"] == typ and has_value(it, value)]
    if not matched:
        return None, "value_not_in_evidence"
    if status == "confirmed":
        verified = [it for it in matched if it["subject_verified"]]
        if not verified:
            return None, "not_subject_verified"
        result["info_ids"] = [it["info_id"] for it in verified]
        result["source_scope"] = "direct"
        return result, None
    error = _check_inferred(items, typ, matched)
    if error:
        return None, error
    kept = _evidence(items, typ, value)
    result["info_ids"] = [it["info_id"] for it in kept]
    result["source_scope"] = _scope(kept)
    return result, None


def verify_candidates(
    bundle: dict, candidates: list, prior: list[dict] = ()
) -> tuple[list[dict], list[dict]]:
    """후보 목록을 검증해 (통과한 후보, 거절 {index, reason})를 반환한다.

    근거 검증 외에 중복도 거른다: prior(이전에 통과한 후보)·같은 제출 안의 반복,
    본인 정보로 이미 확인된 값을 '추론'으로 다시 낸 경우.
    """
    confirmed = confirmed_values(bundle)
    seen = {(p["type"], value_key(p["type"], p["value"]), p["status"]) for p in prior}
    accepted: list[dict] = []
    rejected: list[dict] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, dict):
            valid, reason = check_candidate(bundle, candidate)
        else:
            valid, reason = None, "not_object"
        if valid is not None:
            k = (valid["type"], value_key(valid["type"], valid["value"]))
            if valid["status"] == "inferred" and k in confirmed:
                valid, reason = None, "already_confirmed"
            else:
                if k in confirmed:
                    # 본인 확인 정보와 같은 값이면 그 정보의 대표 값으로 맞춰 결과에서 하나로 합친다
                    valid["value"] = confirmed[k]
                    k = (valid["type"], value_key(valid["type"], valid["value"]))
                if (*k, valid["status"]) in seen:
                    valid, reason = None, "duplicate"
                else:
                    seen.add((*k, valid["status"]))
        if valid is not None:
            accepted.append(valid)
        else:
            rejected.append({"index": index, "reason": reason})
    return accepted, rejected
