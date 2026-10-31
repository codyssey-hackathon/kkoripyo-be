"""EveryInfra 소셜 데이터 API 클라이언트 (서버 간 통신).

- 모든 기능이 POST /api/v1/social 하나로 호출된다: {"platform", "action", "params"}
- 오래 걸리는 기능(user_posts 등)은 202 + job_id를 주므로 GET /api/v1/jobs/{job_id}로 결과를 기다린다.
- 파라미터·응답 필드 명세: GET /api/v1/social/catalog?platform=instagram
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.config import get_settings

# 작업이 아직 끝나지 않은 상태값
PENDING = {"pending", "queued", "running", "finalizing"}


class EveryInfraError(Exception):
    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status


class EveryInfraClient:
    """`async with EveryInfraClient() as ei: await ei.call("profile", {...})`"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 90.0,
        poll_interval: float = 2.0,
        poll_timeout: float = 240.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        settings = get_settings()
        api_key = api_key if api_key is not None else settings.everyinfra_api_key
        if not api_key:
            raise EveryInfraError("missing_api_key", "EVERYINFRA_API_KEY가 설정되지 않았습니다")
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._http = httpx.AsyncClient(
            base_url=base_url or settings.everyinfra_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> EveryInfraClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()

    async def call(self, action: str, params: dict, platform: str = "instagram") -> Any:
        """기능 하나를 호출해 `results`만 돌려준다. 빈 결과는 [] (요금도 안 나감)."""
        body = {"platform": platform, "action": action, "params": params}
        res = await self._post_with_retry(body)
        data = self._parse(res)
        if data.get("job_id") and data.get("status") in PENDING:
            data = await self._wait_job(data["job_id"])
        results = data.get("results")
        return [] if results is None else results

    async def _post_with_retry(self, body: dict) -> httpx.Response:
        for attempt in range(3):
            res = await self._http.post("/api/v1/social", json=body)
            if res.status_code not in (429, 503) or attempt == 2:
                return res
            # 분당 한도 초과(429)나 일시 실패(503)는 잠깐 쉬고 다시 시도
            wait = float(res.headers.get("Retry-After", 2 * (attempt + 1)))
            await asyncio.sleep(min(wait, 15.0))
        return res

    async def _wait_job(self, job_id: str) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.poll_timeout
        while True:
            await asyncio.sleep(self.poll_interval)
            data = self._parse(await self._http.get(f"/api/v1/jobs/{job_id}"))
            status = data.get("status")
            if status == "succeeded":
                return data
            if status not in PENDING:
                err = data.get("error") or {}
                raise EveryInfraError(
                    err.get("code", f"job_{status}"), err.get("message", f"작업 {job_id} 실패")
                )
            if loop.time() > deadline:
                raise EveryInfraError("job_timeout", f"작업 {job_id}가 제한 시간 안에 끝나지 않음")

    @staticmethod
    def _parse(res: httpx.Response) -> dict:
        try:
            data = res.json()
        except ValueError as e:
            raise EveryInfraError("bad_response", res.text[:200], res.status_code) from e
        if res.status_code >= 400 or "error" in data and not data.get("results"):
            err = data.get("error") or {}
            if isinstance(err, str):
                err = {"message": err}
            raise EveryInfraError(
                err.get("code", f"http_{res.status_code}"),
                err.get("message", res.text[:200]),
                res.status_code,
            )
        return data
