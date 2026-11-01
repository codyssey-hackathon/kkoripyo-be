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

    # 사진 속 글자 인식 (팀 OCR 서버). 비우면 OCR 없이 분석
    ocr_server_url: str = "https://ocr.purplecodelabs.com"

    # 사진 속 글자 블러 서버 (services/blur). 비우면 블러 없이 리포트만
    blur_server_url: str = "https://blur.purplecodelabs.com"

    # 에이전트용 LLM
    # 이 서버가 외부에서 보이는 주소. 가린 사진을 OCR 서버가 다시 읽어 검증할 때 쓴다 (비우면 그 검증 생략)
    public_base_url: str = ""

    openai_api_key: str = ""
    openai_model: str = "gpt-6.1-sol"
    # 추론 모델의 생각 강도 (low|medium|high|xhigh). 빈 값이면 넘기지 않는다
    openai_reasoning_effort: str = "high"

    # 진단 데모 화면(/api/v1/agent/demo). 로컬에서는 항상 열리고, 운영에서는 true일 때만 열린다.
    # 누구나 분석을 돌릴 수 있으므로(요금 발생) 필요할 때만 켠다.
    demo_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
