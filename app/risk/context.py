"""장소·주소가 실제 활동 장소·집 주소인지 원문 표현으로 판정한다.

장소(주소) 값이 나온 문장 안에 방문·활동(거주) 표현이 함께 있어야 인정한다. 글 전체가 아니라 문장 단위로 보는 것은
'가평 날씨 좋다. 부산 다녀옴'에서 다른 장소의 표현이 가평에 붙지 않게 하기 위해서다.
가고 싶다·못 갔다처럼 실제로 가지 않은 표현이 같은 문장에 있으면 활동 장소로 보지 않는다.
키워드를 바꾸면 tests/test_risk.py의 맥락 판정 테스트도 함께 고친다.
"""

from __future__ import annotations

import re

from app.cross_inference.schema import clean

ACTIVITY = re.compile(
    r"다녀|갔|왔|방문|도착|놀러|여행|출근|퇴근|등교|하교|산책|단골|체크인|머물|묵었|먹었|마셨|운동|데이트|매주|매일"
)
NOT_VISITED = re.compile(
    r"(못|안)\s*(가|갔|왔|와)|가\s*보?고\s*싶|갈\s*(거|예정|계획)|가야\s*(지|겠|하)"
)
RESIDENCE = re.compile(
    r"우리\s*집|사는\s*(곳|집|동네)|살고\s*있|살아요|거주|자취|이사\s*(왔|옴|했|함|감|간)|입주|본가|기숙사|집\s*주소"
)
# 문장 경계: 마침표·물음표·느낌표(숫자 사이 소수점 제외), 줄바꿈, 'ㅋㅋ'·'ㅠㅠ'·'~~' 같은 반복 문자
SENTENCE_END = re.compile(r"[.!?。]+(?=\s|$)|\n+|[ㅋㅎㅠㅜ]{2,}|~{2,}")


def check_context(kind: str, value: str, text: str) -> str | None:
    """kind: activity_location | home_address. 인정되면 None, 아니면 사유를 반환한다."""
    target = clean(value)
    sentences = [s for s in SENTENCE_END.split(text) if target and target in clean(s)]
    if not sentences:
        return "value_not_in_source"
    for s in sentences:
        if kind == "activity_location":
            if ACTIVITY.search(s) and not NOT_VISITED.search(s):
                return None
        elif RESIDENCE.search(s):
            return None
    return "no_context"
