from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "꼬리표 API"
    app_env: str = "local"
    version: str = "0.1.0"
    database_url: str = "sqlite:///./data/kkoripyo.db"
    cors_origins: list[str] = ["http://localhost:3000"]

    # 외부 데이터 API (EveryInfra)
    everyinfra_api_key: str = ""
    everyinfra_base_url: str = "https://api.everyinfra.com"

    # 에이전트용 LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
