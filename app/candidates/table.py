"""주변 계정 조사 우선순위 점수표. 명세의 임시 기준이라 값이 바뀌면 이 파일만 고친다.

최종 개인정보 위험도(app/risk)와는 별개인 탐색용 보조 신호다.
"""

from __future__ import annotations

# 관계 유형별 가점. 같은 유형은 여러 번 나와도 한 번만 준다
RELATION_POINTS = {"tagged": 5, "mentioned": 4}

# 본인과 주고받은 댓글이 COMMENT_MIN건 이상이면 주는 가점. 댓글 1건은 가산하지 않는다
COMMENT_MIN = 2
COMMENT_POINTS = 2

# follower·following은 탐색 후보일 뿐이라 가산하지 않는다
MAX_PRIORITY = 10
