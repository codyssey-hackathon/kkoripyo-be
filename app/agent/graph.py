"""LangGraph 에이전트 기본 틀.

그래프 모양 (가장 기본적인 '도구 쓰는 에이전트'):

    START → agent ──(도구 호출이 있으면)──→ tools ─┐
              ↑                                    │
              └────────────────────────────────────┘
              └──(도구 호출이 없으면)──→ END

- state(상태): 지금까지의 대화 메시지 목록. 노드는 새 메시지를 돌려주고, LangGraph가 목록에 덧붙인다.
- agent 노드: LLM에게 메시지를 보내고 답(텍스트 또는 도구 호출 요청)을 받는다.
- tools 노드: LLM이 요청한 도구를 실행한다. 한 번에 여러 개를 요청하면 동시에 실행한다.
- tools_condition: agent의 마지막 답에 도구 호출이 있으면 tools로, 없으면 END로 보낸다.

나중에 단계를 더 넣으려면(예: 사진 블러 노드) add_node / add_edge로 이어 붙이면 된다.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.core.config import get_settings

# agent ↔ tools 왕복 최대 횟수 (무한 반복 방지). 한 번 왕복 = 2 단계
MAX_STEPS = 24


def default_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다")
    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)


def build_graph(llm: BaseChatModel | None = None, tools: list | None = None):
    tools = tools or TOOLS
    model = (llm or default_llm()).bind_tools(tools)

    async def agent(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        reply = await model.ainvoke(messages)
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # → "tools" 또는 END
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_agent(username: str, llm: BaseChatModel | None = None) -> dict[str, Any]:
    """아이디 하나로 에이전트를 끝까지 실행하고, 최종 리포트와 도구 호출 기록을 돌려준다."""
    app = build_graph(llm)
    state = await app.ainvoke(
        {"messages": [HumanMessage(f"인스타그램 아이디 '{username}'를 분석해 줘.")]},
        config={"recursion_limit": MAX_STEPS},
    )
    messages = state["messages"]
    steps = []
    for m in messages:
        if isinstance(m, AIMessage):
            steps += [{"tool": c["name"], "args": c["args"]} for c in m.tool_calls]
    errors = sum(1 for m in messages if isinstance(m, ToolMessage) and '"error"' in str(m.content))
    final = messages[-1]
    return {
        "username": username,
        "report": final.content if isinstance(final, AIMessage) else "",
        "tool_calls": steps,
        "tool_errors": errors,
    }
