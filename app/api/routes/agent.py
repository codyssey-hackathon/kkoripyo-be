import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent import run_agent, stream_agent
from app.collect import instagram as ig
from app.core.config import get_settings
from app.integrations.everyinfra import EveryInfraClient, EveryInfraError

router = APIRouter(tags=["agent"])

HANDLE = Field(pattern=r"^[A-Za-z0-9._]{1,30}$", examples=["example_user"])
DEMO_PAGE = Path(__file__).resolve().parents[2] / "static" / "agent_demo.html"


class AgentRunIn(BaseModel):
    handle: str = HANDLE


class AgentStreamIn(BaseModel):
    handle: str = HANDLE
    # 로컬 테스트용: 화면에서 입력한 OpenAI 키. 서버에 저장하지 않고 이번 실행에만 쓴다.
    openai_api_key: str | None = Field(None, max_length=300)


@router.post("/agent/run")
async def agent_run(body: AgentRunIn) -> dict:
    """에이전트를 실행해 리포트를 받는다. 도구 호출이 많으면 1~2분 걸릴 수 있다."""
    settings = get_settings()
    if not settings.openai_api_key or not settings.everyinfra_api_key:
        raise HTTPException(503, "OPENAI_API_KEY 또는 EVERYINFRA_API_KEY가 설정되지 않았습니다")
    return await run_agent(body.handle.lower())


@router.post("/agent/stream")
async def agent_stream(body: AgentStreamIn) -> StreamingResponse:
    """에이전트 진행 상황을 SSE(text/event-stream)로 실시간 전송한다.

    각 줄: `data: {"type": "token"|"tool_call"|"tool_result"|"done"|"error", ...}`
    """
    settings = get_settings()
    key = body.openai_api_key if settings.app_env == "local" else None
    if not settings.everyinfra_api_key:
        raise HTTPException(503, "EVERYINFRA_API_KEY가 설정되지 않았습니다")

    async def events() -> AsyncIterator[str]:
        async for event in stream_agent(body.handle.lower(), api_key=key):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent/demo", response_class=HTMLResponse, include_in_schema=False)
def agent_demo() -> str:
    """로컬 확인용 간이 화면 (APP_ENV=local일 때만)."""
    if get_settings().app_env != "local":
        raise HTTPException(404)
    return DEMO_PAGE.read_text(encoding="utf-8")


@router.get("/instagram/{handle}")
async def instagram_collect(handle: str, posts_limit: int = 30) -> dict:
    """LLM 없이 collect_account 도구가 돌려주는 데이터를 그대로 확인한다 (EveryInfra 요금 발생)."""
    try:
        username = ig.normalize_username(handle)
        async with EveryInfraClient() as ei:
            return await ig.collect_account(ei, username, posts_limit=posts_limit)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except EveryInfraError as e:
        raise HTTPException(502, str(e)) from e
