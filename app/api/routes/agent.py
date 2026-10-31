from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.collect import instagram as ig
from app.core.config import get_settings
from app.integrations.everyinfra import EveryInfraClient, EveryInfraError

router = APIRouter(tags=["agent"])

HANDLE = Field(pattern=r"^[A-Za-z0-9._]{1,30}$", examples=["example_user"])


class AgentRunIn(BaseModel):
    handle: str = HANDLE


@router.post("/agent/run")
async def agent_run(body: AgentRunIn) -> dict:
    """에이전트를 실행해 리포트를 받는다. 도구 호출이 많으면 1~2분 걸릴 수 있다."""
    settings = get_settings()
    if not settings.openai_api_key or not settings.everyinfra_api_key:
        raise HTTPException(503, "OPENAI_API_KEY 또는 EVERYINFRA_API_KEY가 설정되지 않았습니다")
    return await run_agent(body.handle.lower())


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
