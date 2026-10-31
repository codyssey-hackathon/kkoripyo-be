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
from app.integrations import ocr
from app.integrations.everyinfra import EveryInfraError

# 그래프 전체 단계 수 상한 (무한 반복 방지). collect·agent·tools·nudge 하나가 각 1단계
MAX_STEPS = 40
# 계획만 쓰고 멈췄을 때 되돌려 보내는 최대 횟수
MAX_NUDGES = 2
NUDGE_TEXT = (
    "방금 다음에 살펴볼 것만 말하고 실제로 확인하지 않았다. 더 확인할 것이 있으면 지금 바로 도구를 호출하라. "
    f"다 확인했으면 '{FINAL_MARK}'로 시작하는 최종 리포트를 써라."
)

# 고객 화면에 보여줄 진행 문구: 도구 이름 → (진행 중, 완료)
STAGE_TEXT = {
    "get_profile": ("친구 계정 프로필 살펴보는 중", "친구 계정 프로필 확인"),
    "get_followers": ("친구 계정의 팔로워 목록 받아오는 중", "친구 계정의 팔로워 목록 확인"),
    "get_following": ("친구 계정의 팔로잉 목록 받아오는 중", "친구 계정의 팔로잉 목록 확인"),
    "get_user_posts": ("친구 피드 읽는 중", "친구 피드 확인"),
    "get_post_comments": ("댓글 읽는 중", "댓글 확인"),
}


class AgentState(MessagesState):
    username: str
    nudges: int
    stopped: str | None


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
    return ChatOpenAI(model=settings.openai_model, api_key=api_key)


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


async def collect(state: AgentState) -> dict:
    username = state["username"]
    ocr_task: asyncio.Task | None = None

    def progress(key: str, event: str, value: Any = None) -> None:
        nonlocal ocr_task
        _collect_progress(key, event, value)
        # 게시물이 도착하면 바로 OCR을 시작해 댓글 수집과 동시에 돌린다
        if key == "posts" and event == "done" and value and ocr_task is None:
            ocr_task = asyncio.create_task(_run_ocr(value))

    try:
        async with agent_tools.make_client() as ei:
            data = await ig.collect_account(ei, username, progress=progress)
    except (EveryInfraError, ValueError) as e:
        if ocr_task:
            ocr_task.cancel()
        return _stop("collect_failed", f"@{username} 계정 정보를 불러오지 못했습니다. ({e})")
    if ocr_task:
        await ocr_task

    profile = data.get("profile")
    if not profile:
        return _stop("not_found", f"@{username} 계정을 찾을 수 없습니다. 아이디를 확인해 주세요.")
    if profile.get("is_private"):
        return _stop(
            "private",
            f"@{username} 계정은 비공개 계정입니다. 팔로워가 아닌 사람은 게시물과 댓글을 볼 수 없어서, "
            "공개 정보로 알아낼 수 있는 내용이 거의 없습니다. 분석을 마칩니다.",
        )

    # 에이전트에게는 원본 JSON 대신 읽기 쉬운 요약을 준다.
    # OpenAI 형식상 tool 메시지는 같은 id의 도구 호출 다음에 와야 해서 한 쌍으로 넣는다.
    call = {"name": "collect_account", "args": {"username": username}, "id": "collect_0"}
    return {
        "messages": [
            AIMessage("", tool_calls=[call]),
            ToolMessage(build_digest(data), tool_call_id="collect_0", name="collect_account"),
        ]
    }


def _stop(reason: str, text: str) -> dict:
    emit({"type": "token", "text": text})
    return {"stopped": reason, "messages": [AIMessage(text)]}


# ---------- 그래프 ----------


def build_graph(llm: BaseChatModel | None = None, tools: list | None = None):
    tools = tools or TOOLS
    model = (llm or default_llm()).bind_tools(tools)

    async def agent(state: AgentState) -> dict:
        emit({"type": "status", "text": "분석하는 중"})
        messages = [SystemMessage(system_prompt(state["username"])), *state["messages"]]
        reply = await model.ainvoke(messages)
        return {"messages": [reply]}

    def nudge(state: AgentState) -> dict:
        return {"messages": [HumanMessage(NUDGE_TEXT)], "nudges": state.get("nudges", 0) + 1}

    def after_collect(state: AgentState) -> str:
        return END if state.get("stopped") else "agent"

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        text = last.content if isinstance(last.content, str) else ""
        if FINAL_MARK in text or state.get("nudges", 0) >= MAX_NUDGES:
            return END
        return "nudge"

    graph = StateGraph(AgentState)
    graph.add_node("collect", collect)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("nudge", nudge)
    graph.add_edge(START, "collect")
    graph.add_conditional_edges("collect", after_collect, ["agent", END])
    graph.add_conditional_edges("agent", route, ["tools", "nudge", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    return graph.compile()


def _inputs(username: str) -> dict:
    return {
        "username": username,
        "nudges": 0,
        "stopped": None,
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
        "report": final.content if isinstance(final, AIMessage) else "",
        "tool_calls": steps,
        "tool_errors": errors,
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
                if meta.get("langgraph_node") == "agent" and isinstance(msg.content, str):
                    if msg.content:
                        yield {"type": "token", "text": msg.content}
                continue
            for node, update in chunk.items():
                for m in (update or {}).get("messages", []):
                    if node == "agent" and isinstance(m, AIMessage) and m.tool_calls:
                        # 같은 응답에서 같은 도구를 여러 번 부르면 한 줄로 묶는다
                        by_name: dict[str, list[str]] = {}
                        for c in m.tool_calls:
                            by_name.setdefault(c["name"], []).append(c["id"])
                        for name, ids in by_name.items():
                            running, _ = STAGE_TEXT.get(name, ("살펴보는 중", "확인"))
                            g = {"id": f"{ids[0]}:{name}", "name": name, "left": len(ids)}
                            g["total"], g["count"], g["failed"] = len(ids), 0, 0
                            for i in ids:
                                groups[i] = g
                            suffix = f" ({len(ids)}개 계정)" if len(ids) > 1 else ""
                            yield {
                                "type": "stage",
                                "id": g["id"],
                                "text": running + suffix,
                                "state": "running",
                            }
                    elif node == "tools" and isinstance(m, ToolMessage):
                        g = groups.get(m.tool_call_id)
                        if not g:
                            continue
                        g["left"] -= 1
                        count = _result_count(str(m.content))
                        if count == "error":
                            g["failed"] += 1
                        if g["left"] == 0:
                            _, done = STAGE_TEXT.get(g["name"], ("", "확인"))
                            suffix = f" ({g['total']}개 계정)" if g["total"] > 1 else count
                            state = "error" if g["failed"] == g["total"] else "done"
                            text = done + suffix if state == "done" else done + " 실패"
                            yield {"type": "stage", "id": g["id"], "text": text, "state": state}
        yield {"type": "done"}
    except Exception as e:  # 화면에 원인을 보여주고 끝낸다
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
