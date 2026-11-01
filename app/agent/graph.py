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


def stage(stage_id: str, text: str, state: str = "running") -> None:
    emit({"type": "stage", "id": stage_id, "text": text, "state": state})


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

COLLECT_TEXT = {
    "profile": ("프로필 확인하는 중", lambda v: f"프로필 확인: {v.get('display_name') or '-'}"),
    "followers": ("팔로워 목록 받아오는 중", lambda v: f"팔로워 {v['count']}명 확인"),
    "following": ("팔로잉 목록 받아오는 중", lambda v: f"팔로잉 {v['count']}명 확인"),
    "posts": ("게시물 피드 읽는 중", lambda v: f"피드 {len(v)}개 확인"),
}


def _collect_progress(key: str, event: str, value: Any = None) -> None:
    """ig.collect_account의 진행 상황을 고객용 문구로 바꿔 보낸다."""
    if key in COLLECT_TEXT:
        running, done = COLLECT_TEXT[key]
        if event == "start":
            stage(f"collect:{key}", running)
        elif event == "done":
            stage(f"collect:{key}", done(value), "done")
        else:
            stage(f"collect:{key}", f"{running.replace(' 중', '')} 실패", "error")
    elif key == "friends":
        if event == "start":
            stage("collect:friends", "가까운 친구들의 피드에서 당신의 흔적 찾는 중")
        elif event == "done":
            seen = [f["username"] for f in value if f.get("posts")]
            if seen:
                stage("collect:friends", f"{who_text(seen)}의 피드 확인", "done")
            else:
                stage("collect:friends", "가까운 친구들의 피드는 비공개라 볼 수 없음", "done")
    elif key == "comments":
        if event == "start":
            stage("collect:comments", f"댓글 읽는 중 (0/{value})")
        elif event == "progress":
            stage("collect:comments", f"댓글 읽는 중 ({value[0]}/{value[1]})")
        elif event == "done":
            stage("collect:comments", f"댓글 {value}개 확인", "done")


async def _run_ocr(posts: list[dict]) -> None:
    """게시물 사진의 글자를 읽어 각 게시물에 ocr_text를 붙인다. 실패해도 분석은 계속한다."""
    if not any(p.get("image_url") for p in posts):
        return
    stage("collect:ocr", "사진 속 글자 읽는 중")
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
    stage("collect:friend_ocr", "친구 게시물 사진 속 글자 읽는 중")
    try:
        summary = await ocr.ocr_posts(posts)
    except Exception:
        stage("collect:friend_ocr", "친구 게시물 사진 속 글자 읽기 실패", "error")
        return
    text = f"친구 게시물 사진 {summary['images']}장 중 {summary['with_text']}장에서 글자 발견"
    stage("collect:friend_ocr", text, "done")


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
    photos = {
        p["post_id"]: {
            "image_url": p["image_url"],
            "url": p.get("url"),
            "posted_at": p.get("posted_at"),
            "ocr_text": p.get("ocr_text") or "",
        }
        for p in data.get("posts") or []
        if p.get("post_id") and p.get("image_url")
    }
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


def auto_blur_picks(state: AgentState) -> list[dict]:
    """AI가 사진을 고르지 않았을 때, 사진 글자에 드러난 개인정보로 직접 고른다."""
    values: list[tuple[str, str]] = []
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
    picks = []
    for post_id, photo in (state.get("photos") or {}).items():
        text = " ".join((photo.get("ocr_text") or "").split())
        if not text:
            continue
        reason = None
        if PHONE_RE.search(text):
            reason = "사진 속 글자에 전화번호가 그대로 보입니다"
        elif EMAIL_RE.search(text):
            reason = "사진 속 글자에 이메일 주소가 그대로 보입니다"
        else:
            for label, value in values:
                if value.lower() in text.lower():
                    reason = f"사진 속 글자에 {label} '{value}'가 그대로 보입니다"
                    break
        if reason:
            picks.append({"post_id": post_id, "reason": reason[:80]})
    return picks[: blur.MAX_IMAGES]


def _kept_url(data_url: str | None) -> str | None:
    """블러 사진을 서버에 보관하고 화면에는 받아 갈 주소만 준다."""
    key = blur.keep_image(data_url)
    return f"/api/v1/agent/blurred/{key}" if key else None


async def blur_photos(state: AgentState) -> dict:
    """리포트가 끝난 뒤, 고른 사진의 글자 부분을 블러해 비포·애프터를 보낸다. 실패해도 진단은 끝난다."""
    chosen = _blur_requests(state) or auto_blur_picks(state)
    if not chosen:
        return {"blurred": []}
    photos = state["photos"]
    stage("blur", f"개인정보가 보이는 사진 {len(chosen)}장 가리는 중")
    images = [
        {"image_id": ocr.image_id(c["post_id"]), "image_url": photos[c["post_id"]]["image_url"]}
        for c in chosen
    ]
    async with blur.make_blur_client() as client:
        results = await client.blur(images)
    out = []
    for c, r in zip(chosen, results, strict=True):
        item = {
            "image_id": r["image_id"],
            "post_id": c["post_id"],
            "post_url": photos[c["post_id"]].get("url"),
            "posted_at": photos[c["post_id"]].get("posted_at"),
            "reason": c["reason"],
            "before": r["image_url"],
            "after": _kept_url(r.get("blurred_image")),
            "error": r.get("error") if not r.get("blurred_image") else None,
        }
        emit({"type": "blur", **item})
        out.append(item)
    ok = sum(1 for i in out if i["after"])
    if ok:
        stage("blur", f"사진 {ok}장 가림", "done")
    else:
        stage("blur", "사진 가리기 실패", "error")
    return {"blurred": out}


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

    async def report(state: AgentState) -> dict:
        stage("report", "리포트 쓰는 중")
        messages = [
            SystemMessage(system_prompt(state["username"])),
            *state["messages"],
            HumanMessage(REPORT_TEXT),
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
            stage(
                f"gate:{state.get('pushbacks', 0)}",
                "아직 드러나지 않은 정보를 다른 경로로 다시 찾습니다",
                "done",
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
            return "report"
        return "nudge"

    def after_tools(state: AgentState) -> str:
        last = _last_ai(state)
        if last and any(c["name"] == "finish_investigation" for c in last.tool_calls):
            return "gate"
        return "agent"

    def after_gate(state: AgentState) -> str:
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else "report"

    graph = StateGraph(AgentState)
    graph.add_node("collect", collect)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("nudge", nudge)
    graph.add_node("gate", gate)
    graph.add_node("report", report)
    graph.add_node("blur", blur_photos)
    graph.add_edge(START, "collect")
    graph.add_conditional_edges("collect", after_collect, ["agent", END])
    graph.add_conditional_edges("agent", route, ["tools", "nudge", "blur", "report"])
    graph.add_conditional_edges("tools", after_tools, ["gate", "agent"])
    graph.add_edge("nudge", "agent")
    graph.add_conditional_edges("gate", after_gate, ["agent", "report"])
    graph.add_edge("report", "blur")
    graph.add_edge("blur", END)
    return graph.compile()


def _inputs(username: str) -> dict:
    return {
        "username": username,
        "nudges": 0,
        "pushbacks": 0,
        "stopped": None,
        "photos": {},
        "blurred": [],
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
                            for i in ids:
                                groups[i] = g
                            yield {
                                "type": "stage",
                                "id": g["id"],
                                "text": running.format(who=who),
                                "state": "running",
                            }
                    elif node == "tools" and isinstance(m, ToolMessage):
                        g = groups.get(m.tool_call_id)
                        if not g:
                            continue
                        g["left"] -= 1
                        if _result_count(str(m.content)) == "error":
                            g["failed"] += 1
                        if g["left"] == 0:
                            state = "error" if g["failed"] == g["total"] else "done"
                            text = g["done"] if state == "done" else g["done"] + " 실패"
                            yield {"type": "stage", "id": g["id"], "text": text, "state": state}
        yield {"type": "done"}
    except Exception as e:  # 화면에 원인을 보여주고 끝낸다
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
