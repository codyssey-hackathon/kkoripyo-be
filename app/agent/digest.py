"""수집 데이터를 에이전트가 읽기 좋게 정리한다.

- build_digest: collect_account 결과(JSON)를 사람이 읽는 형태의 짧은 텍스트로 바꾼다.
  (이미지 주소·ID 같은 분석에 필요 없는 값은 빼서 모델이 중요한 줄을 놓치지 않게 함)
- find_clues: 생일·학교·나이 등 개인정보 단서가 될 만한 문장을 규칙으로 미리 찾아 둔다.
  최종 판단은 에이전트가 하고, 여기서는 "놓치지 않게" 후보만 표시한다.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

WEEKDAYS = "월화수목금토일"

# (항목, 정규식) — 넓게 잡고 판단은 에이전트에게 맡긴다
CLUE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("생일", re.compile(r"생일|생축|생신|birthday|bday|🎂|태어난|탄신")),
    (
        "나이",
        re.compile(
            r"\d{1,2}\s?살|\d{2}\s?학번|\d{2}\s?년생|스무|서른|마흔|성인\s?됐|수능|졸업|입학"
        ),
    ),
    (
        "학교·학과",
        re.compile(
            r"대학교?|[가-힣]+대(?:\b|학)|고등학교|중학교|초등학교|캠퍼스|학교|학과|전공|과\s?동기|"
            r"동아리|교수님|강의|기말|중간고사|과제|학번|#[가-힣]+대\b"
        ),
    ),
    ("직장", re.compile(r"회사|출근|퇴근|입사|퇴사|인턴|알바|직장|팀장|사원")),
    (
        "사는 곳",
        re.compile(
            r"우리\s?동네|집\s?앞|집\s?근처|자취|이사|본가|[가-힣]{1,6}(?:동|구|시|군|역)(?=\W|$)"
        ),
    ),
    ("전화번호", re.compile(r"01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}")),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("혈액형", re.compile(r"혈액형|\b(?:AB|A|B|O)형\b|(?:AB|A|B|O)\s?형이라")),
    ("가족", re.compile(r"엄마|아빠|어머니|아버지|언니|오빠|누나|형아|동생|할머니|할아버지|가족")),
]


# 팔로워·팔로잉 계정 이름에서 찾는 단서 (학교·동아리 공식 계정, 가족 계정 등)
ACCOUNT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "학교·학과",
        re.compile(
            r"대학|대학교|학생회|총학|학과|학부|동아리|과\s?학생회|\d{2}학번|고등학교|중학교|초등학교|"
            r"univ|university|college|_hs\b|highschool|student|[가-힣]{1,4}대\s?[가-힣]{0,6}(?:과|학부)",
            re.IGNORECASE,
        ),
    ),
    ("직장", re.compile(r"회사|주식회사|official|corp|company|팀\b|사내", re.IGNORECASE)),
    (
        "가족",
        re.compile(
            r"엄마|아빠|어머니|아버지|언니|오빠|누나|동생|할머니|할아버지|mom|dad|mother|father|sister|brother|family",
            re.IGNORECASE,
        ),
    ),
    ("사는 곳", re.compile(r"[가-힣]{1,6}(?:동|구|시|군)\s?(?:주민|맘|카페|모임|소식)")),
]

# 요약에 넣는 팔로워·팔로잉 목록 최대 인원 (모델이 읽을 분량 제한)
MAX_LIST = 400

# 인구가 많은 성씨 (상위 약 20개). 같은 성씨라는 이유만으로 가족 단서로 쓰지 않는다
COMMON_SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍")


def _date(iso: str | None) -> str:
    if not iso:
        return "날짜 없음"
    try:
        d = datetime.fromisoformat(iso)
        return f"{d:%Y-%m-%d}({WEEKDAYS[d.weekday()]})"
    except ValueError:
        return iso[:10]


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def build_digest(data: dict) -> str:
    p = data.get("profile") or {}
    follow_users = (data.get("followers") or {}).get("users", []) + (
        data.get("following") or {}
    ).get("users", [])
    followers = {u["username"] for u in (data.get("followers") or {}).get("users", [])}
    following = {u["username"] for u in (data.get("following") or {}).get("users", [])}
    private = {u["username"] for u in follow_users if u.get("is_private")}
    posts = data.get("posts") or []

    lines = [
        f"# 계정 @{data.get('username')}",
        f"- 표시 이름: {p.get('display_name') or '-'}",
        f"- 소개글: {_one_line(p.get('bio')) or '-'}",
        f"- 외부 링크: {p.get('external_url') or '-'}",
        f"- 팔로워 {p.get('follower_count')}명 / 팔로잉 {p.get('following_count')}명 / "
        f"게시물 {p.get('post_count')}개 (이번에 읽은 게시물 {len(posts)}개)",
        f"- 맞팔로우 {len(followers & following)}명: {', '.join(sorted(followers & following)) or '-'}",
        "",
        "# 게시물 (최신순)",
    ]
    for post in posts:
        head = f"[{post.get('post_id')}] {_date(post.get('posted_at'))}"
        if post.get("location"):
            head += f" 위치태그: {post['location']}"
        head += f" 좋아요 {post.get('like_count')}"
        lines.append(head)
        lines.append(f"  캡션: {_one_line(post.get('text')) or '(없음)'}")
        if post.get("ocr_text"):
            lines.append(f"  사진 속 글자: {_one_line(post['ocr_text'])[:300]}")
        for c in post.get("comments") or []:
            lines.append(f"  댓글 @{c.get('username')}: {_one_line(c.get('text'))}")

    lines += ["", "# 상호작용이 많은 계정 (댓글 수 · 캡션 멘션 수)"]
    comments = Counter(
        c.get("username")
        for post in posts
        for c in (post.get("comments") or [])
        if c.get("username")
    )
    mentions = Counter(m for post in posts for m in post.get("mentions") or [])
    me = data.get("username")
    ranked = sorted(set(comments) | set(mentions), key=lambda u: -(comments[u] + mentions[u]))
    for u in [x for x in ranked if x != me][:15]:
        mutual = " (맞팔)" if u in followers and u in following else ""
        locked = " (비공개)" if u in private else ""
        lines.append(f"- @{u}: 댓글 {comments[u]}회, 멘션 {mentions[u]}회{mutual}{locked}")

    lines += _friend_lines(data)

    lines += ["", "# 팔로워·팔로잉 목록 (아이디 · 표시 이름 · 관계)"]
    seen: dict[str, dict] = {}
    for u in follow_users:
        seen.setdefault(u["username"], u)
    for name, u in list(seen.items())[:MAX_LIST]:
        rel = (
            "맞팔"
            if name in followers and name in following
            else ("팔로워" if name in followers else "팔로잉")
        )
        locked = ", 비공개" if u.get("is_private") else ""
        lines.append(f"- @{name} · {_one_line(u.get('name')) or '-'} · {rel}{locked}")
    if len(seen) > MAX_LIST:
        lines.append(f"- … 외 {len(seen) - MAX_LIST}명 생략")

    clues = find_clues(data) + find_account_clues(data)
    lines += ["", "# 규칙으로 미리 찾은 단서 후보 (오탐 가능, 직접 판단할 것)"]
    lines += clues or ["- 없음"]
    return "\n".join(lines)


def _friend_lines(data: dict) -> list[str]:
    """서버가 미리 가져온 가까운 친구들의 피드. 당신이 멘션된 게시물을 앞에 표시한다."""
    friends = data.get("friends")
    if friends is None:
        return []
    me = data.get("username")
    lines = [
        "",
        "# 가까운 친구 피드 (상호작용 상위 계정, 이미 가져옴. [당신 멘션]만 당신의 정보로 쓸 것)",
    ]
    if not friends:
        return lines + ["- 없음"]
    for f in friends:
        p = f.get("profile") or {}
        head = f"## @{f['username']} · {p.get('display_name') or '-'} · 댓글 {f.get('comments')}회, 멘션 {f.get('mentions')}회"
        if p.get("is_private"):
            lines += [head + " · 비공개라 볼 수 없음"]
            continue
        if f.get("error") and not f.get("posts"):
            lines += [head + " · 불러오지 못함"]
            continue
        lines += [head, f"  소개글: {_one_line(p.get('bio')) or '-'}"]
        for post in f.get("posts") or []:
            tag = " [당신 멘션]" if me and me in (post.get("mentions") or []) else ""
            loc = f" 위치태그: {post['location']}" if post.get("location") else ""
            caption = _one_line(post.get("text"))[:200] or "(없음)"
            lines.append(
                f"  [{post.get('post_id')}] {_date(post.get('posted_at'))}{loc}{tag} {caption}"
            )
    return lines


def find_account_clues(data: dict) -> list[str]:
    """팔로워·팔로잉 계정의 아이디·표시 이름에서 학교·학과·동아리·가족 단서를 찾는다."""
    out = []
    users: dict[str, dict] = {}
    for key in ("followers", "following"):
        for u in (data.get(key) or {}).get("users", []):
            users.setdefault(u["username"], u)

    # 같은 성씨: 본인 표시 이름이 2~4글자 한글이고, 흔한 성이 아닐 때만 단서로 본다
    # (김·이·박처럼 흔한 성은 같은 성씨 계정이 너무 많아 가족 단서가 되지 않음)
    me = _one_line((data.get("profile") or {}).get("display_name"))
    surname = me[0] if re.fullmatch(r"[가-힣]{2,4}", me) else None
    if surname in COMMON_SURNAMES:
        surname = None

    for name, u in users.items():
        display = _one_line(u.get("name"))
        text = f"{name} {display}"
        for label, pattern in ACCOUNT_PATTERNS:
            if pattern.search(text):
                out.append(f"- [{label}] 팔로워·팔로잉 계정 @{name} ({display or '-'})")
        if surname and re.fullmatch(rf"{surname}[가-힣]{{1,3}}", display):
            out.append(f"- [가족?] 같은 성씨({surname}) 계정 @{name} ({display})")
    return out


def find_clues(data: dict) -> list[str]:
    out = []
    p = data.get("profile") or {}
    sources = [("프로필 소개글", None, p.get("bio")), ("프로필 이름", None, p.get("display_name"))]
    for post in data.get("posts") or []:
        date = _date(post.get("posted_at"))
        sources.append((f"게시물 {post.get('post_id')} 캡션", date, post.get("text")))
        if post.get("ocr_text"):
            sources.append((f"게시물 {post.get('post_id')} 사진 속 글자", date, post["ocr_text"]))
        if post.get("location"):
            sources.append((f"게시물 {post.get('post_id')} 위치태그", date, str(post["location"])))
        for c in post.get("comments") or []:
            sources.append(
                (f"게시물 {post.get('post_id')} 댓글 @{c.get('username')}", date, c.get("text"))
            )
    for where, date, text in sources:
        text = _one_line(text)
        if not text:
            continue
        for label, pattern in CLUE_PATTERNS:
            # 영어 단서는 대소문자가 섞여 나온다 (사진 속 'HAPPY BIRTHDAY' 등)
            if pattern.search(text) or pattern.search(text.lower()):
                when = f" {date}" if date else ""
                out.append(f'- [{label}]{when} {where}: "{text[:120]}"')
    return out
