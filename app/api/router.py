from fastapi import APIRouter

from app.api.routes import agent, analyses

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(analyses.router)
api_router.include_router(agent.router)
