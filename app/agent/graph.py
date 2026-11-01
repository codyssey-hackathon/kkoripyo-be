"""LangGraph 에이전트 기본 틀.

그래프 모양:

    START → collect ──(비공개·실패)──→ END
               │
               └──→ agent ──(도구 호출 있음)──→ tools ──→ agent
                      ├──(최종 리포트임)──→ END
                      └──(계획만 쓰고 도구 호출 없음)──→ nudge ──→ agent

- state(상태): 대화 메시지 목록 + 몇 가지 값. 노드는 바뀐 부분만 돌려주고, LangGraph가 합친다.
- collect 노드: 서버가 분석 대상 계정을 직접 수집한다(에이전트 판단 전). 비공개면 여기서 끝낸다.
  수집 결과는 읽기 쉬운 요약(digest)과 단서 후보로 바꿔 에이전트에게 준다.
- agent 노드: LLM이 고객에게 보여줄 글을 쓰거나, 더 볼 계정이 있으면 도구 호출을 요청한다.
- tools 노드: LLM이 요청한 도구를 실행한다. 한 번에 여러 개를 요청하면 동시에 실행한다.
- nudge 노드: LLM이 계획만 쓰고 도구를 안 부르면 그대로 끝나버리므로 다시 요청한다. (최대 MAX_NUDGES번)

화면에는 도구 이름·인자·원본 응답을 보내지 않는다. 대신 stage 이벤트(고객용 진행 문구)를 보낸다.
나중에 단계를 더 넣으려면(예: 사진 블러 노드) add_node / add_edge로 이어 붙이면 된다.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent import blur_check
from app.agent import tools as agent_tools
from app.agent.digest import build_digest
from app.agent.prompts import FINAL_MARK, system_prompt
from app.agent.tools import TOOLS
from app.collect import instagram as ig
from app.core.config import get_settings
from app.integrations import blur, ocr
from app.integrations.everyinfra import EveryInfraError

# 그래프 전체 단계 수 상한 (무한 반복 방지). collect·agent·tools·nudge 하나가 각 1단계
# 계정 하나를 끈질기게 파고들 수 있게 넉넉히 둔다
MAX_STEPS = 120
# 조사를 마치려 할 때 빠진 항목이 있으면 다시 조사하라고 돌려보내는 최대 횟수
MAX_PUSHBACKS = 2
# 마무리 확인에서 빠졌는지 보는 핵심 항목과, 빠졌을 때 알려줄 찾는 방법
KEY_TARGETS = {
    "name": (
        "이름",
        "친구 게시물·댓글에서 당신을 부르는 이름·애칭, 친구가 당신을 태그하며 쓴 호칭, "
        "가까운 친구 프로필 소개글의 이름",
    ),
    "birthday": (
        "생일",
        "맞팔·댓글이 많은 친구들의 피드에서 당신을 멘션한 생일 축하 게시물의 날짜, "
        "당신 게시물에 같은 날짜로 몰린 축하 댓글",
    ),
    "age": (
        "나이",
        "동갑으로 보이는 친구(같은 학교·학번, 반말 댓글, 맞팔)의 소개글·게시물에 있는 'n년생'·학번·"
        "졸업·입학·수능 언급으로 같은 나이를 역산",
    ),
    "school": (
        "학교·학과",
        "가까운 친구들의 팔로잉 목록(get_following)에서 학과·대·대학·대학교 계정. "
        "두 명 이상이 같은 계정을 팔로우하면 당신도 그 소속",
    ),
    "region": (
        "사는 곳",
        "당신과 친구 게시물의 위치태그, 반복되는 동네·역 이름, 다니는 학교·직장 위치로 생활권을 좁힘",
    ),
    "close_friend": ("친한 친구", "댓글·멘션·태그가 가장 많은 계정"),
}
# 계획만 쓰고 멈췄을 때 되돌려 보내는 최대 횟수
MAX_NUDGES = 2
NUDGE_TEXT = (
    "방금 다음에 살펴볼 것만 말하고 실제로 확인하지 않았다. 더 확인할 것이 있으면 지금 바로 도구를 호출하라. "
    f"다 확인했으면 '{FINAL_MARK}'로 시작하는 최종 리포트를 써라."
)

# 고객 화면에 보여줄 진행 문구: 도구 이름 → (진행 중, 완료). {who}에는 대상 계정 이름이 들어간다
STAGE_TEXT = {
    "get_profile": ("{who}의 프로필과 소개글 살펴보는 중", "{who}의 프로필 확인"),
    "get_followers": ("{who}의 팔로워 목록 뒤지는 중", "{who}의 팔로워 목록 확인"),
    "get_following": ("{who}의 팔로잉 목록 뒤지는 중", "{who}의 팔로잉 목록 확인"),
    "get_user_posts": ("{who}의 피드에서 당신의 흔적 찾는 중", "{who}의 피드 확인"),
    "get_post_comments": ("{who}에 달린 댓글 읽는 중", "{who}의 댓글 확인"),
}


def who_text(names: list[str]) -> str:
    """['a','b','c'] → '@a, @b 외 1명'"""
    names = [f"@{n}" for n in dict.fromkeys(n for n in names if n)]
    if not names:
        return "친구 계정"
    if len(names) <= 2:
        return ", ".join(names)
    return f"{', '.join(names[:2])} 외 {len(names) - 2}명"


def stage_who(name: str, calls: list[dict]) -> str:
    if name == "get_post_comments":
        return f"게시물 {len(calls)}개"
    return who_text([str((c.get("args") or {}).get("username", "")).lstrip("@") for c in calls])


# 화면 진행 줄에 띄우지 않는 도구 (사진 선택은 리포트 뒤 blur 단계에서 따로 보여준다)
SILENT_TOOLS = {"select_photos_to_blur", "record_finding", "finish_investigation"}
REPORT_TEXT = (
    f"조사가 끝났다. 이제 최종 리포트만 써라. 첫 줄은 '{FINAL_MARK}'로 바로 시작한다. "
    "record_finding으로 기록한 내용과 대화에서 확인된 사실만 쓴다."
)


class AgentState(MessagesState):
    username: str
    nudges: int
    pushbacks: int
    stopped: str | None
    photos: dict  # post_id → {image_url, url, posted_at} (collect에서 채움, blur에서 사용)
    blurred: list  # blur 결과 [{image_id, post_id, reason, before, after, error}]
    blur_choice: list  # 리포트 전에 고른 교체할 사진 [{post_id, reason}]


def text_of(content: Any) -> str:
    """메시지 내용에서 글만 뽑는다.

    추론 모델에 도구를 붙이면 응답이 문자열이 아니라 블록 목록으로 온다
    ([{"type": "reasoning", ...}, {"type": "text", "text": "..."}]). 생각 블록은 빼고 글 블록만 잇는다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


def emit(event: dict) -> None:
    """화면용 이벤트를 보낸다. 스트리밍 실행이 아니면 아무것도 안 한다."""
    try:
        get_stream_writer()(event)
    except Exception:
        pass


def stage(stage_id: str, text: str, state: str = "running", detail: str = "") -> None:
    event = {"type": "stage", "id": stage_id, "text": text, "state": state}
    if detail:
        event["detail"] = detail
    emit(event)


def default_llm(api_key: str | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    api_key = api_key or settings.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다")
    extra = {}
    if settings.openai_reasoning_effort:
        extra["reasoning_effort"] = settings.openai_reasoning_effort
    return ChatOpenAI(model=settings.openai_model, api_key=api_key, **extra)


# ---------- collect 노드 ----------

# 진행 중일 때 함께 보여줄 "왜 이걸 하는지"
COLLECT_DETAIL = {
    "profile": "낯선 사람이 가장 먼저 보는 이름·소개글·외부 링크부터 확인합니다",
    "followers": "누가 당신을 지켜보는지, 학교·학과·가족 계정이 섞여 있는지 봅니다",
    "following": "당신이 팔로우하는 학교·학과·동아리·회사 계정에서 소속을 찾습니다",
    "posts": "올린 날짜, 위치태그, 해시태그, 캡션에서 생일·동선·소속의 흔적을 모읍니다",
    "tagged": "당신이 올리지 않았어도, 친구가 당신을 태그해 올린 사진은 누구나 볼 수 있습니다",
}

COLLECT_TEXT = {
    "profile": ("프로필 확인하는 중", lambda v: f"프로필 확인: {v.get('display_name') or '-'}"),
    "followers": ("팔로워 목록 받아오는 중", lambda v: f"팔로워 {v['count']}명 확인"),
    "following": ("팔로잉 목록 받아오는 중", lambda v: f"팔로잉 {v['count']}명 확인"),
    "posts": ("게시물 피드 읽는 중", lambda v: f"피드 {len(v)}개 확인"),
    "tagged": (
        "다른 사람이 올린, 당신이 태그된 사진 찾는 중",
        lambda v: f"당신이 태그된 사진 {len(v)}장 확인",
    ),
}


def _collect_progress(key: str, event: str, value: Any = None) -> None:
    """ig.collect_account의 진행 상황을 고객용 문구로 바꿔 보낸다."""
    if key in COLLECT_TEXT:
        running, done = COLLECT_TEXT[key]
        if event == "start":
            stage(f"collect:{key}", running, detail=COLLECT_DETAIL.get(key, ""))
        elif event == "done":
            stage(f"collect:{key}", done(value), "done")
        else:
            stage(f"collect:{key}", f"{running.replace(' 중', '')} 실패", "error")
    elif key == "friends":
        if event == "start":
            stage(
                "collect:friends",
                "가까운 친구들의 피드에서 당신의 흔적 찾는 중",
                detail="댓글·멘션이 많은 친구부터, 그들의 게시물 속에 당신이 태그·언급된 장면을 찾습니다",
            )
        elif event == "done":
            seen = [f["username"] for f in value if f.get("posts")]
            if seen:
                stage("collect:friends", f"{who_text(seen)}의 피드 확인", "done")
            else:
                stage("collect:friends", "가까운 친구들의 피드는 비공개라 볼 수 없음", "done")
    elif key == "comments":
        if event == "start":
            stage(
                "collect:comments",
                f"댓글 읽는 중 (0/{value})",
                detail="누가 당신에게 자주 말을 거는지, 어떤 말로 당신을 부르는지 읽습니다",
            )
        elif event == "progress":
            stage("collect:comments", f"댓글 읽는 중 ({value[0]}/{value[1]})")
        elif event == "done":
            stage("collect:comments", f"댓글 {value}개 확인", "done")


async def _run_ocr(posts: list[dict]) -> None:
    """게시물 사진의 글자를 읽어 각 게시물에 ocr_text를 붙인다. 실패해도 분석은 계속한다."""
    if not any(p.get("image_url") for p in posts):
        return
    stage(
        "collect:ocr",
        "사진 속 글자 읽는 중",
        detail="케이크 문구, 명찰, 학생증, 수료증, 캡처 화면처럼 사진에 적힌 글자에서 신상을 찾습니다",
    )
    try:
        summary = await ocr.ocr_posts(posts)
    except Exception:
        stage("collect:ocr", "사진 속 글자 읽기 실패", "error")
        return
    if summary["failed"] == summary["images"]:
        stage("collect:ocr", "사진 속 글자 읽기 실패", "error")
    else:
        text = f"사진 {summary['images']}장 중 {summary['with_text']}장에서 글자 발견"
        stage("collect:ocr", text, "done")


async def _run_friend_ocr(posts: list[dict]) -> None:
    """가까운 친구들 게시물 사진의 글자도 읽는다. 실패해도 분석은 계속한다."""
    if not any(p.get("image_url") for p in posts):
        return
    stage(
        "collect:friend_ocr",
        "친구 게시물 사진 속 글자 읽는 중",
        detail="친구가 올린 사진에 적힌 당신의 이름·학교·생일 문구를 찾습니다",
    )
    try:
        summary = await ocr.ocr_posts(posts)
    except Exception:
        stage("collect:friend_ocr", "친구 게시물 사진 속 글자 읽기 실패", "error")
        return
    text = f"친구 게시물 사진 {summary['images']}장 중 {summary['with_text']}장에서 글자 발견"
    stage("collect:friend_ocr", text, "done")


def _photo(p: dict, owner: str | None, linked: bool) -> dict:
    return {
        "image_url": p["image_url"],
        "url": p.get("url"),
        "posted_at": p.get("posted_at"),
        "ocr_text": p.get("ocr_text") or "",
        "owner": owner,  # None이면 본인 사진, 아니면 그 사진을 올린 계정
        "linked": linked,  # 다른 사람 사진인데 당신이 태그·멘션됨
    }


def collect_photos(data: dict) -> dict:
    """가릴 사진 후보: 본인 사진 + 당신이 태그된 남의 사진 + 가까운 친구 사진."""
    me = data.get("username")
    photos: dict[str, dict] = {}
    for p in data.get("posts") or []:
        if p.get("post_id") and p.get("image_url"):
            photos[p["post_id"]] = _photo(p, None, True)
    for p in data.get("tagged") or []:
        if p.get("post_id") and p.get("image_url") and p["post_id"] not in photos:
            owner = p.get("author") if p.get("author") != me else None
            photos[p["post_id"]] = _photo(p, owner, True)
    for f in data.get("friends") or []:
        for p in f.get("posts") or []:
            if p.get("post_id") and p.get("image_url") and p["post_id"] not in photos:
                linked = me in (p.get("mentions") or [])
                photos[p["post_id"]] = _photo(p, f["username"], linked)
    return photos


async def collect(state: AgentState) -> dict:
    username = state["username"]
    ocr_task: asyncio.Task | None = None

    prefetch: ig.FriendPrefetch | None = None

    def progress(key: str, event: str, value: Any = None) -> None:
        nonlocal ocr_task
        _collect_progress(key, event, value)
        if event == "done" and key in ("followers", "following") and value and prefetch:
            prefetch.mark_private(value.get("users") or [])
        # 게시물이 도착하면 바로 OCR과 친구 피드 받기를 시작해 댓글 수집과 동시에 돌린다
        if key == "posts" and event == "done" and value:
            if ocr_task is None:
                ocr_task = asyncio.create_task(_run_ocr(value))
            if prefetch:
                prefetch.offer([m for p in value for m in p.get("mentions") or []])
        if key == "comments" and event == "rows" and value and prefetch:
            prefetch.offer([c["username"] for c in value if c.get("username")])

    try:
        async with agent_tools.make_client() as ei:
            prefetch = ig.FriendPrefetch(ei, ig.normalize_username(username), progress=progress)
            try:
                data = await ig.collect_account(ei, username, progress=progress)
                profile = data.get("profile") or {}
                if profile and not profile.get("is_private"):
                    data["friends"] = await prefetch.finish(data)
            finally:
                prefetch.cancel()
    except (EveryInfraError, ValueError) as e:
        if ocr_task:
            ocr_task.cancel()
        return _stop("collect_failed", f"@{username} 계정 정보를 불러오지 못했습니다. ({e})")
    friend_posts = [p for f in data.get("friends") or [] for p in f.get("posts") or []]
    friend_posts += data.get("tagged") or []
    await asyncio.gather(
        ocr_task if ocr_task else asyncio.sleep(0),
        _run_friend_ocr(friend_posts),
    )

    profile = data.get("profile")
    if not profile:
        return _stop("not_found", f"@{username} 계정을 찾을 수 없습니다. 아이디를 확인해 주세요.")
    if profile.get("is_private"):
        return _stop(
            "private",
            f"@{username} 계정은 비공개 계정입니다. 팔로워가 아닌 사람은 게시물과 댓글을 볼 수 없어서, "
            "공개 정보로 알아낼 수 있는 내용이 거의 없습니다. 분석을 마칩니다.",
        )

    posts = data.get("posts") or []
    emit(
        {
            "type": "stats",
            "posts": len(posts),
            "comments": sum(len(p.get("comments") or []) for p in posts),
            "photos_with_text": sum(1 for p in posts if p.get("ocr_text")),
            "followers": (data.get("followers") or {}).get("count", 0),
            "following": (data.get("following") or {}).get("count", 0),
            "friends": sum(1 for f in data.get("friends") or [] if f.get("posts")),
        }
    )
    # 에이전트에게는 원본 JSON 대신 읽기 쉬운 요약을 준다.
    # OpenAI 형식상 tool 메시지는 같은 id의 도구 호출 다음에 와야 해서 한 쌍으로 넣는다.
    call = {"name": "collect_account", "args": {"username": username}, "id": "collect_0"}
    photos = collect_photos(data)
    return {
        "photos": photos,
        "messages": [
            AIMessage("", tool_calls=[call]),
            ToolMessage(build_digest(data), tool_call_id="collect_0", name="collect_account"),
        ],
    }


# ---------- blur 노드 ----------


def _blur_requests(state: AgentState) -> list[dict]:
    """에이전트가 select_photos_to_blur로 고른 사진을 모은다 (마지막 선택 우선, 중복·없는 게시물 제외)."""
    picked: dict[str, str] = {}
    for m in state["messages"]:
        if not isinstance(m, AIMessage):
            continue
        for c in m.tool_calls:
            if c["name"] != "select_photos_to_blur":
                continue
            for p in c["args"].get("photos") or []:
                post_id = (p or {}).get("post_id")
                if post_id in (state.get("photos") or {}):
                    picked[post_id] = (p.get("reason") or "").strip()
    return [{"post_id": k, "reason": v} for k, v in list(picked.items())[: blur.MAX_IMAGES]]


PHONE_RE = re.compile(r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# 사진 글자에 이 항목의 값이 보이면 가린다 (생일·나이 같은 값은 흔한 숫자라 제외)
BLUR_CATEGORIES = {"name", "phone", "email", "school", "workplace", "region"}


# 사진 글자에 이런 말이 있으면 개인정보가 적힌 사진일 가능성이 높다
PII_WORDS = re.compile(
    r"대학교|대학|학과|학번|고등학교|중학교|초등학교|학생증|명찰|수료증|자격증|증명서|상장|성명|이름|"
    r"생년월일|생일|주소|아파트|빌라|[가-힣]+동\s?\d+|\d+호|사원증|직원|NAME|Name|STUDENT|Student|ID"
)
DATE_RE = re.compile(r"(19|20)\d{2}[.\-/년 ]\s?\d{1,2}[.\-/월 ]\s?\d{1,2}")
# 사진 교체: 위험한 순서로 BLUR_TRY장까지 가려 보고, 완벽히 가려진 것 중 가장 앞 순위 BLUR_SHOW장만 보여준다
BLUR_TRY = 4
BLUR_SHOW = 1
PICK_CANDIDATES = 8


def _finding_values(state: AgentState) -> list[tuple[str, str]]:
    values = []
    for m in state["messages"]:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                if c["name"] != "record_finding":
                    continue
                for f in c["args"].get("findings") or []:
                    value = str((f or {}).get("value") or "").split("(")[0].strip()
                    if (f or {}).get("category") in BLUR_CATEGORIES and len(value) >= 2:
                        label = agent_tools.CATEGORY_LABEL.get(f["category"], "개인정보")
                        values.append((label, value))
    return values


def _blur_score(text: str, values: list[tuple[str, str]]) -> tuple[int, str]:
    """사진 글자가 얼마나 위험한지 점수와 화면에 보여줄 이유."""
    if PHONE_RE.search(text):
        return 100, "사진 속 글자에 전화번호가 그대로 보입니다"
    if EMAIL_RE.search(text):
        return 90, "사진 속 글자에 이메일 주소가 그대로 보입니다"
    for label, value in values:
        if value.lower() in text.lower():
            return 80, f"사진 속 글자에 {label} '{value}'가 그대로 보입니다"
    words = PII_WORDS.findall(text)
    if words:
        return 40 + len(words), f"사진 속 글자에 '{words[0]}' 같은 신상 정보가 적혀 있습니다"
    if DATE_RE.search(text):
        return 30, "사진 속 글자에 날짜가 적혀 있어 생활 패턴이 드러납니다"
    if len(text) >= 6:
        snippet = text[:18] + ("…" if len(text) > 18 else "")
        return 10, f"사진 속 글자 '{snippet}'가 그대로 읽힙니다"
    return 0, ""


def auto_blur_picks(
    state: AgentState, exclude: set[str] | None = None, limit: int = blur.MAX_IMAGES
) -> list[dict]:
    """사진 글자에 드러난 개인정보가 많은 순서로 가릴 사진을 고른다."""
    values = _finding_values(state)
    scored = []
    for post_id, photo in (state.get("photos") or {}).items():
        if exclude and post_id in exclude:
            continue
        text = " ".join((photo.get("ocr_text") or "").split())
        score, reason = _blur_score(text, values) if text else (0, "")
        owner = photo.get("owner")
        if owner and score:
            # 남의 사진은 당신이 태그·멘션됐거나, 글자에 당신의 정보가 있을 때만
            # 전화번호·학교 같은 글자도 친구 자신의 것일 수 있으니, 당신의 값이 보여야 한다
            mine = photo.get("linked") or any(v.lower() in text.lower() for _, v in values)
            if not mine:
                continue
            reason = f"@{owner}이(가) 올린 사진 — {reason}"
        if score:
            scored.append((score, post_id, reason))
    scored.sort(key=lambda x: -x[0])
    return [{"post_id": p, "reason": r[:80]} for _, p, r in scored[:limit]]


def blur_candidates(state: AgentState, limit: int = PICK_CANDIDATES) -> list[dict]:
    """교체 후보: 조사 중 AI가 고른 사진을 앞에, 그다음 사진 글자의 위험도 순."""
    chosen = _blur_requests(state)
    extra = auto_blur_picks(state, exclude={c["post_id"] for c in chosen}, limit=limit)
    return (chosen + extra)[:limit]


def choose_blur(state: AgentState) -> list[dict]:
    """순위를 매긴 사진이 있으면 그것, 없으면 후보 앞에서부터 BLUR_TRY장 (위험한 순서)."""
    if state.get("blur_choice"):
        return state["blur_choice"][:BLUR_TRY]
    return blur_candidates(state)[:BLUR_TRY]


def _picked_note(state: AgentState) -> str:
    chosen = state.get("blur_choice") or []
    if not chosen:
        return (
            " 이번 진단에서는 가린 사진을 보여주지 않는다. '가린 사진으로 교체하세요' 같은 안내는 쓰지 않고, "
            "사진은 보관·삭제·글자 가리기를 직접 하라고 쓴다."
        )
    photos = state.get("photos") or {}

    def label(c: dict) -> str:
        p = photos.get(c["post_id"]) or {}
        whose = f"@{p['owner']}이 올린 " if p.get("owner") else ""
        return f"{_date_text(p.get('posted_at'))} {whose}사진({c['reason']})"

    items = "; ".join(label(c) for c in chosen)
    return (
        f" 교체할 사진으로 다음 {len(chosen)}장을 골랐다: {items}. '지금 바로 해야 할 일'에서 이 사진들을 날짜와 내용으로 "
        "짚고 '리포트 아래의 가린 사진으로 교체하세요'라고 쓴다. 다른 사람이 올린 사진이면 "
        "'@그 계정에 가린 사진으로 바꿔 달라고 요청하세요'라고 쓴다. 다른 사진은 교체 대상으로 쓰지 않는다."
    )


def pick_prompt(state: AgentState, candidates: list[dict]) -> str:
    photos = state.get("photos") or {}
    lines = [
        "교체할 사진 고르기: 아래는 글자가 적힌 당신(의뢰인)의 사진들이다. 낯선 사람에게 가장 위험한 순서로 "
        f"지금 당장 교체해야 할 순서대로 최대 {BLUR_TRY}장을 골라 select_photos_to_blur로 답하라 "
        "(가장 시급한 것을 맨 앞에). "
        "기준: 신원을 바로 특정하게 하는 글자(실명, 전화번호, 학번·학생증, 주소, 수료증)가 가장 위험하다. "
        "다른 사람이 올린 사진도 당신의 정보가 드러나면 똑같이 후보다(친구에게 교체를 요청해야 한다). "
        "reason은 의뢰인에게 보여줄 한 줄(80자 이내)로, 사진에 무엇이 적혀 있어 왜 위험한지 구체적으로 쓴다.",
    ]
    for c in candidates:
        p = photos.get(c["post_id"]) or {}
        text = " ".join((p.get("ocr_text") or "").split())[:160]
        owner = p.get("owner")
        whose = f"@{owner}이 올린 사진(당신 태그·멘션)" if owner else "당신이 올린 사진"
        lines.append(
            f"- [{c['post_id']}] {_date_text(p.get('posted_at'))} {whose} · 사진 속 글자: {text}"
        )
    return "\n".join(lines)


def _date_text(iso: str | None) -> str:
    return (iso or "")[:10] or "날짜 모름"


def _kept_url(data_url: str | None) -> str | None:
    """블러 사진을 서버에 보관하고 화면에는 받아 갈 주소만 준다."""
    key = blur.keep_image(data_url)
    return f"/api/v1/agent/blurred/{key}" if key else None


async def blur_photos(state: AgentState) -> dict:
    """순위대로 가려 보고, 완벽히 가려진 사진 중 가장 앞 순위 BLUR_SHOW장만 비포·애프터로 보낸다.

    리포트보다 먼저 실행해서, 리포트가 실제로 보여주는 사진만 가리키게 한다. 실패해도 진단은 계속된다.
    """
    chosen = choose_blur(state)
    if not chosen:
        return {"blurred": [], "blur_choice": []}
    photos = state["photos"]
    stage(
        "blur",
        f"교체 후보 {len(chosen)}장을 가려 보는 중",
        detail="가린 사진을 다시 읽어 원래 글자가 남지 않았는지 확인하고, 완벽히 가려진 한 장만 보여줍니다",
    )
    images = [
        {"image_id": ocr.image_id(c["post_id"]), "image_url": photos[c["post_id"]]["image_url"]}
        for c in chosen
    ]
    async with blur.make_blur_client() as client:
        results = await client.blur(images)
    items = []
    for c, r in zip(chosen, results, strict=True):
        p = photos[c["post_id"]]
        items.append(
            {
                "image_id": r["image_id"],
                "post_id": c["post_id"],
                "post_url": p.get("url"),
                "posted_at": p.get("posted_at"),
                "owner": p.get("owner"),
                "reason": c["reason"],
                "before": r["image_url"],
                "after": _kept_url(r.get("blurred_image")),
                "error": r.get("error") if not r.get("blurred_image") else None,
            }
        )
    before_texts = {i["image_id"]: (photos[i["post_id"]].get("ocr_text") or "") for i in items}
    verdicts = await blur_check.assess(items, before_texts)
    shown = [i for i, v in zip(items, verdicts, strict=True) if v.ok][:BLUR_SHOW]
    for item in shown:
        # 인스타그램 사진 서버는 다른 사이트에서 사진을 띄우지 못하게 막는다(CORP).
        # 원본도 우리 서버가 받아 두고 우리 주소로 보낸다.
        original = await blur_check._download(item["before"])
        if original:
            item["before"] = f"/api/v1/agent/blurred/{blur.keep_bytes(original)}"
        emit({"type": "blur", **item})
    shown_ids = {i["post_id"] for i in shown}
    lines = []
    for rank, (item, v) in enumerate(zip(items, verdicts, strict=True), start=1):
        day = _date_text(item.get("posted_at"))
        if item["post_id"] in shown_ids:
            lines.append(f"{rank}위 {day} 사진 — 보여줌 ({v.why})")
        elif v.ok:
            lines.append(f"{rank}위 {day} 사진 — 잘 가려졌지만 더 앞 순위를 보여줌")
        else:
            lines.append(f"{rank}위 {day} 사진 — 제외 ({v.why})")
    if shown:
        stage("blur", "완벽히 가려진 사진 1장 선정", "done", detail="\n".join(lines))
    else:
        stage("blur", "완벽히 가려진 사진이 없어 교체 사진은 생략", "done", detail="\n".join(lines))
    return {
        "blurred": shown,
        "blur_choice": [{"post_id": i["post_id"], "reason": i["reason"]} for i in shown],
    }


def _stop(reason: str, text: str) -> dict:
    emit({"type": "token", "text": text})
    return {"stopped": reason, "messages": [AIMessage(text)]}


# ---------- 그래프 ----------


def _last_ai(state: AgentState) -> AIMessage | None:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return m
    return None


def found_categories(state: AgentState) -> set[str]:
    """record_finding으로 기록된 항목 종류."""
    found = set()
    for m in state["messages"]:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                if c["name"] == "record_finding":
                    found |= {(f or {}).get("category") for f in c["args"].get("findings") or []}
    return found


def pushback_text(missing: list[str], found: set[str]) -> str:
    lines = [
        "아직 조사를 끝내지 마라. 다음 항목이 드러나지 않았다. 아래 방법으로 끝까지 찾아라. "
        "찾으면 바로 record_finding으로 기록하고, 고객에게는 찾은 사실만 말한다.",
    ]
    for key in missing:
        label, how = KEY_TARGETS[key]
        lines.append(f"- {label}: {how}")
    if found:
        lines.append("이미 기록한 항목: " + ", ".join(sorted(found)))
    lines.append(
        "이 방법들을 실제로 시도했는데도 정말 더 찾을 수 없을 때만 finish_investigation을 다시 불러라."
    )
    return "\n".join(lines)


def build_graph(llm: BaseChatModel | None = None, tools: list | None = None):
    tools = tools or TOOLS
    base = llm or default_llm()
    model = base.bind_tools(tools)
    # 리포트 단계: 도구 기록이 있는 대화를 그대로 넘기되 도구는 더 부르지 않게 한다
    writer = base.bind_tools(tools, tool_choice="none")

    async def agent(state: AgentState) -> dict:
        emit({"type": "status", "text": "분석하는 중"})
        messages = [SystemMessage(system_prompt(state["username"])), *state["messages"]]
        reply = await model.ainvoke(messages)
        return {"messages": [reply]}

    picker = base.bind_tools(
        [agent_tools.select_photos_to_blur], tool_choice="select_photos_to_blur"
    )

    async def pick_photos(state: AgentState) -> dict:
        """리포트 전에, 지금 당장 교체해야 할 가장 시급한 사진 두 장을 고른다."""
        candidates = blur_candidates(state)
        if not candidates:
            return {"blur_choice": []}
        stage(
            "pick",
            "교체할 사진의 순위를 매기는 중",
            detail=f"글자가 적힌 사진 {len(candidates)}장을 다시 훑어, "
            f"지금 바로 바꿔야 할 순서대로 {min(BLUR_TRY, len(candidates))}장을 고릅니다",
        )
        chosen = candidates[:BLUR_TRY]
        if len(candidates) > 1:  # 보여줄 순서가 중요하니 2장 이상이면 AI가 순위를 매긴다
            messages = [
                SystemMessage(system_prompt(state["username"])),
                *state["messages"],
                HumanMessage(pick_prompt(state, candidates)),
            ]
            try:
                reply = await picker.ainvoke(messages)
                ids = {c["post_id"] for c in candidates}
                picked: dict[str, str] = {}
                for c in getattr(reply, "tool_calls", None) or []:
                    for p in (c.get("args") or {}).get("photos") or []:
                        pid = (p or {}).get("post_id")
                        if pid in ids and pid not in picked:
                            picked[pid] = str((p or {}).get("reason") or "").strip()[:80]
                if picked:
                    fallback = {c["post_id"]: c["reason"] for c in candidates}
                    chosen = [
                        {"post_id": pid, "reason": reason or fallback[pid]}
                        for pid, reason in list(picked.items())[:BLUR_TRY]
                    ]
                    for c in candidates:  # 하나만 골랐으면 위험도 순으로 채운다
                        if len(chosen) >= BLUR_TRY:
                            break
                        if c["post_id"] not in picked:
                            chosen.append(c)
            except Exception:
                pass  # 고르지 못하면 위험도 순 후보를 쓴다
        photos = state.get("photos") or {}
        detail = "\n".join(
            f"{n}위 · {_date_text((photos.get(c['post_id']) or {}).get('posted_at'))} 사진 — {c['reason']}"
            for n, c in enumerate(chosen, start=1)
        )
        stage("pick", f"교체 후보 {len(chosen)}장 순위 결정", "done", detail=detail)
        return {"blur_choice": chosen}

    async def report(state: AgentState) -> dict:
        stage(
            "report",
            "리포트 쓰는 중",
            detail="찾아낸 사실을 엮어, 낯선 사람이 할 수 있는 일을 정리합니다",
        )
        messages = [
            SystemMessage(system_prompt(state["username"])),
            *state["messages"],
            HumanMessage(REPORT_TEXT + _picked_note(state)),
        ]
        reply = await writer.ainvoke(messages)
        stage("report", "리포트 완성", "done")
        return {"messages": [reply]}

    def nudge(state: AgentState) -> dict:
        return {"messages": [HumanMessage(NUDGE_TEXT)], "nudges": state.get("nudges", 0) + 1}

    def gate(state: AgentState) -> dict:
        """조사를 마치려 할 때 핵심 항목이 빠졌으면, 찾는 방법을 알려주며 다시 조사하게 한다."""
        found = found_categories(state)
        missing = [k for k in KEY_TARGETS if k not in found]
        if missing and state.get("pushbacks", 0) < MAX_PUSHBACKS:
            labels = "·".join(KEY_TARGETS[k][0] for k in missing)
            stage(
                f"gate:{state.get('pushbacks', 0)}",
                f"아직 드러나지 않은 {labels}을(를) 다른 경로로 다시 찾습니다",
                "done",
                detail="여기서 멈추지 않습니다. 친구의 친구, 동갑 친구, 가족 계정까지 거슬러 올라갑니다",
            )
            return {
                "messages": [HumanMessage(pushback_text(missing, found))],
                "pushbacks": state.get("pushbacks", 0) + 1,
            }
        return {}

    def after_collect(state: AgentState) -> str:
        return END if state.get("stopped") else "agent"

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        text = text_of(last.content)
        if FINAL_MARK in text:
            return "blur"  # 조사 중에 리포트를 바로 써버린 경우
        if state.get("nudges", 0) >= MAX_NUDGES:
            return "pick_photos"
        return "nudge"

    def after_tools(state: AgentState) -> str:
        last = _last_ai(state)
        if last and any(c["name"] == "finish_investigation" for c in last.tool_calls):
            return "gate"
        return "agent"

    def after_blur(state: AgentState) -> str:
        # 조사 중에 리포트를 이미 써 버린 경우에는 다시 쓰지 않는다
        written = any(
            isinstance(m, AIMessage) and FINAL_MARK in text_of(m.content) for m in state["messages"]
        )
        return END if written else "report"

    def after_gate(state: AgentState) -> str:
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else "pick_photos"

    graph = StateGraph(AgentState)
    graph.add_node("collect", collect)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("nudge", nudge)
    graph.add_node("gate", gate)
    graph.add_node("pick_photos", pick_photos)
    graph.add_node("report", report)
    graph.add_node("blur", blur_photos)
    graph.add_edge(START, "collect")
    graph.add_conditional_edges("collect", after_collect, ["agent", END])
    graph.add_conditional_edges("agent", route, ["tools", "nudge", "blur", "pick_photos"])
    graph.add_conditional_edges("tools", after_tools, ["gate", "agent"])
    graph.add_edge("nudge", "agent")
    graph.add_conditional_edges("gate", after_gate, ["agent", "pick_photos"])
    graph.add_edge("pick_photos", "blur")
    graph.add_conditional_edges("blur", after_blur, ["report", END])
    graph.add_edge("report", END)
    return graph.compile()


def _inputs(username: str) -> dict:
    return {
        "username": username,
        "nudges": 0,
        "pushbacks": 0,
        "stopped": None,
        "photos": {},
        "blurred": [],
        "blur_choice": [],
        "messages": [HumanMessage(f"인스타그램 아이디 '{username}'를 분석해 줘.")],
    }


async def run_agent(username: str, llm: BaseChatModel | None = None) -> dict[str, Any]:
    """아이디 하나로 에이전트를 끝까지 실행하고, 최종 리포트와 도구 호출 기록을 돌려준다."""
    app = build_graph(llm)
    state = await app.ainvoke(_inputs(username), config={"recursion_limit": MAX_STEPS})
    messages = state["messages"]
    steps = []
    for m in messages:
        if isinstance(m, AIMessage):
            steps += [{"tool": c["name"], "args": c["args"]} for c in m.tool_calls]
    errors = sum(1 for m in messages if isinstance(m, ToolMessage) and '"error"' in str(m.content))
    final = messages[-1]
    return {
        "username": username,
        "stopped": state.get("stopped"),
        "report": text_of(final.content) if isinstance(final, AIMessage) else "",
        "tool_calls": steps,
        "tool_errors": errors,
        "blurred": state.get("blurred") or [],
    }


def _result_count(content: str) -> str:
    """도구 결과에서 고객용 개수 문구만 뽑는다 (원본은 화면에 보내지 않음)."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return ""
    if isinstance(data, dict) and "error" in data:
        return "error"
    if isinstance(data, list):
        return f" {len(data)}개"
    if isinstance(data, dict) and "users" in data:
        return f" {data.get('count', len(data['users']))}명"
    return ""


async def stream_agent(
    username: str, llm: BaseChatModel | None = None, api_key: str | None = None
) -> AsyncIterator[dict]:
    """에이전트를 실행하면서 고객 화면용 이벤트를 하나씩 내보낸다.

    이벤트 종류:
      token   에이전트가 고객에게 쓰는 글 조각 (발견 내용, 최종 리포트)
      stage   진행 문구 {id, text, state: running|done|error}. 같은 id는 화면에서 한 줄로 갱신
      status  상단 상태 문구
      done / error
    """
    try:
        app = build_graph(llm or default_llm(api_key))
        groups: dict[str, dict] = {}  # 도구 호출 id → 같은 응답·같은 도구 묶음
        last_msg_id = None  # 글을 쓰던 응답. 다른 응답이 이어서 쓰면 문단을 나눈다
        async for mode, chunk in app.astream(
            _inputs(username),
            config={"recursion_limit": MAX_STEPS},
            stream_mode=["updates", "messages", "custom"],
        ):
            if mode == "custom":
                if isinstance(chunk, dict):
                    yield chunk
                continue
            if mode == "messages":
                msg, meta = chunk
                node = meta.get("langgraph_node")
                text = text_of(msg.content) if node in ("agent", "report") else ""
                if text:
                    if last_msg_id is not None and msg.id != last_msg_id:
                        yield {"type": "token", "text": "\n\n"}
                    last_msg_id = msg.id
                    yield {"type": "token", "text": text}
                continue
            for node, update in chunk.items():
                for m in (update or {}).get("messages", []):
                    if node == "agent" and isinstance(m, AIMessage) and m.tool_calls:
                        # 같은 응답에서 같은 도구를 여러 번 부르면 한 줄로 묶는다
                        by_name: dict[str, list[dict]] = {}
                        for c in m.tool_calls:
                            by_name.setdefault(c["name"], []).append(c)
                        for name, calls in by_name.items():
                            if name in SILENT_TOOLS:
                                continue  # 진행 줄로 보여줄 필요 없는 내부 단계
                            running, done = STAGE_TEXT.get(
                                name, ("{who} 살펴보는 중", "{who} 확인")
                            )
                            who = stage_who(name, calls)
                            ids = [c["id"] for c in calls]
                            g = {"id": f"{ids[0]}:{name}", "left": len(ids), "total": len(ids)}
                            g["failed"], g["done"] = 0, done.format(who=who)
                            g["count"] = 0
                            for i in ids:
                                groups[i] = g
                            reasons = [
                                str((c.get("args") or {}).get("reason") or "").strip()
                                for c in calls
                            ]
                            detail = "\n".join(dict.fromkeys(r for r in reasons if r))
                            event = {
                                "type": "stage",
                                "id": g["id"],
                                "text": running.format(who=who),
                                "state": "running",
                            }
                            if detail:
                                event["detail"] = detail[:600]
                            yield event
                    elif node == "tools" and isinstance(m, ToolMessage):
                        g = groups.get(m.tool_call_id)
                        if not g:
                            continue
                        g["left"] -= 1
                        count = _result_count(str(m.content))
                        if count == "error":
                            g["failed"] += 1
                        elif count:
                            g["count"] += int("".join(ch for ch in count if ch.isdigit()) or 0)
                            g["unit"] = count.strip()[-1]
                        if g["left"] == 0:
                            state = "error" if g["failed"] == g["total"] else "done"
                            text = g["done"] if state == "done" else g["done"] + " 실패"
                            if state == "done" and g.get("unit"):
                                text += f" · {g['count']}{g['unit']}"
                            yield {"type": "stage", "id": g["id"], "text": text, "state": state}
        yield {"type": "done"}
    except Exception as e:  # 화면에 원인을 보여주고 끝낸다
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
