"""OpenAI Chat Completions로 A안 루프 실행. 조회 도구 실행(execute_tool)은 백엔드가 넘긴다."""

from __future__ import annotations

import json
from collections.abc import Callable

from .agent_contract import finalize_report, prepare_agent
from .contracts import ContractError

MAX_TURNS = 6


def run_openai(
    json2: dict,
    client,
    model: str,
    execute_tool: Callable[[str, dict], dict],
    max_turns: int = MAX_TURNS,
) -> tuple[dict, list[dict]]:
    """(JSON3, 거절된 후보 목록)을 반환.

    client: openai.OpenAI() 인스턴스
    execute_tool(name, args) -> dict: 백엔드의 조회 도구 실행 함수. 허용 계정·횟수 제한 검사 후 결과 또는 {"error": ...} 반환
    """
    spec = prepare_agent(json2)
    terminal = spec["terminal_tool"]
    messages = list(spec["messages"])
    tool_log: list[dict] = []
    rejected: list[dict] = []
    for turn in range(max_turns):
        # 매 턴 도구 호출을 강제하고, 마지막 턴에는 제출 도구만 허용해 반드시 종료시킨다
        last = turn == max_turns - 1
        tool_choice = {"type": "function", "function": {"name": terminal}} if last else "required"
        response = client.chat.completions.create(
            model=model, messages=messages, tools=spec["tools"], tool_choice=tool_choice
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append(
                {"role": "user", "content": "조회 도구 또는 submit_inferences를 호출하라."}
            )
            continue
        # 한 응답에 도구 호출이 여러 개 올 수 있다. 모든 tool_call_id에 결과를 돌려줘야 다음 요청이 거부되지 않는다
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in calls
                ],
            }
        )
        report = None
        for call in calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if not isinstance(args, dict):
                result = {"error": "invalid_json_arguments"}
            elif call.function.name == terminal and report is None:
                try:
                    report = finalize_report(
                        json2, args, tool_log=tool_log, mode="backend_agent", rejected=rejected
                    )
                    result = {"ok": True}
                except ContractError as e:
                    result = {"error": str(e)}
            else:
                result = execute_tool(call.function.name, args)
                tool_log.append(
                    {"tool": call.function.name, "args": args, "ok": "error" not in result}
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        if report is not None:
            return report, rejected
    # 마지막 턴 제출까지 형식 오류로 실패한 경우에도 빈 추론 결과를 안전하게 생성
    return finalize_report(
        json2, {"candidates": []}, tool_log=tool_log, mode="backend_agent_fallback"
    ), rejected
