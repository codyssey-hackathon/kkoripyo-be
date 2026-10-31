import asyncio
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
DEMO_KEY_INPUT = (
    '<input id="key" type="password" placeholder="OpenAI API 키 (.env에 있으면 비워두기)" '
    'autocomplete="off">'
)
DEMO_LOCAL_NOTE = '<p class="sub">로컬 확인용 임시 화면입니다. 입력한 키는 이번 실행에만 쓰이고 저장되지 않습니다.</p>'


# 조용한 구간(친구 피드 수집, 긴 추론)에도 연결이 끊기지 않게 이 간격마다 SSE 주석을 보낸다.
# 브라우저·프록시·터널의 유휴 시간 제한(Cloudflare 약 100초)보다 충분히 짧게.
HEARTBEAT_SECONDS = 15.0


async def with_heartbeat(source: AsyncIterator[str], interval: float) -> AsyncIterator[str]:
    """source가 interval 동안 아무것도 내지 않으면 ': ping' 주석 줄을 끼워 보낸다."""
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(item)
        finally:
            await queue.put(done)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is done:
                break
            yield item
        await task  # pump에서 난 예외를 그대로 올린다
    finally:
        task.cancel()


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

    각 줄: `data: {"type": "token"|"stage"|"status"|"finding"|"stats"|"blur"|"done"|"error", ...}`
    조용한 구간에는 `: ping` 주석 줄이 15초마다 온다(무시하면 됨).
    """
    settings = get_settings()
    key = body.openai_api_key if settings.app_env == "local" else None
    if not settings.everyinfra_api_key:
        raise HTTPException(503, "EVERYINFRA_API_KEY가 설정되지 않았습니다")

    async def events() -> AsyncIterator[str]:
        async for event in stream_agent(body.handle.lower(), api_key=key):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        with_heartbeat(events(), HEARTBEAT_SECONDS),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent/demo", response_class=HTMLResponse, include_in_schema=False)
def agent_demo() -> str:
    """진단 데모 화면. 로컬은 항상, 운영은 DEMO_ENABLED=true일 때만 연다."""
    settings = get_settings()
    local = settings.app_env == "local"
    if not (local or settings.demo_enabled):
        raise HTTPException(404)
    page = DEMO_PAGE.read_text(encoding="utf-8")
    if not local:
        # 운영에서는 화면에서 받은 키를 쓰지 않으므로(서버 키만 사용) 입력칸과 안내 문구를 숨긴다
        page = page.replace(DEMO_KEY_INPUT, '<input id="key" type="hidden">')
        page = page.replace(DEMO_LOCAL_NOTE, "")
    return page


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
