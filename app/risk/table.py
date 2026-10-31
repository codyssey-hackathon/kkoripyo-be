"""위험도 점수표. 명세의 임시 기준이라 값이 바뀌면 이 파일만 고친다."""

from __future__ import annotations

# 점수표 항목별 기본점수
BASE_POINTS = {
    "sns_account": 5,
    "name": 8,
    "affiliation": 14,
    "activity_location": 12,
    "activity_pattern": 14,
    "relationship": 6,
    "contact": 25,
    "home_address": 30,
}

# 확인 상태 계수
STATUS_MULTIPLIERS = {"confirmed": 1.0, "inferred": 0.6}

# 정보 조합 가점 (항목, 항목, 가점). 분석 대상 전체에서 조합마다 한 번만 준다
COMBINATIONS = [
    ("name", "affiliation", 8),
    ("name", "home_address", 12),
    # '관련 장소'는 활동 맥락이 확인된 장소로 본다. 생활 패턴은 포함하지 않는다
    ("affiliation", "activity_location", 8),
    ("sns_account", "contact", 6),
]

MAX_SCORE = 100

# 교차 추론 결과의 유형 -> 점수표 항목
ITEM_OF = {
    "sns_account": "sns_account",
    "name": "name",
    "affiliation": "affiliation",
    # 생활 패턴은 교차 추론 검증에서 서로 다른 게시물 2곳 이상의 같은 장소가 이미 확인됐다
    "activity_pattern": "activity_pattern",
    "relationship": "relationship",
    "phone": "contact",
    "email": "contact",
}

# 원문 맥락이 확인돼야 점수표 항목이 되는 유형 -> 항목
CONTEXT_ITEM_OF = {
    "location": "activity_location",
    "address": "home_address",
}
