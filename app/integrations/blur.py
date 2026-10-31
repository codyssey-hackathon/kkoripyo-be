"""블러 서버 클라이언트 (services/blur, Florence-2로 사진 속 글자 영역을 블러).

요청: POST {BLUR_SERVER_URL}/blur/urls  {"images": [{"image_id", "image_url"}]}  (최대 4장)
응답: {"images": [{"image_id", "image_url", "blurred_image": "data:image/jpeg;base64,...", "error"}]}
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings

MAX_IMAGES = 4


class BlurClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        url = base_url if base_url is not None else get_settings().blur_server_url
        self.enabled = bool(url)
        self._http = httpx.AsyncClient(
            base_url=url or "http://blur.invalid", timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> BlurClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()

    async def blur(self, images: list[dict]) -> list[dict]:
        """[{image_id, image_url}] → [{image_id, image_url, blurred_image, error}].

        서버가 꺼져 있어도 예외를 던지지 않고 이미지마다 error를 남긴다.
        """
        images = images[:MAX_IMAGES]
        if not images:
            return []
        if not self.enabled:
            return [{**i, "blurred_image": None, "error": "blur_disabled"} for i in images]
        try:
            res = await self._http.post("/blur/urls", json={"images": images})
            res.raise_for_status()
            rows = res.json().get("images") or []
        except (httpx.HTTPError, ValueError) as e:
            reason = f"blur_unavailable: {type(e).__name__}"
            return [{**i, "blurred_image": None, "error": reason} for i in images]
        by_id = {r.get("image_id"): r for r in rows}
        return [
            {
                **i,
                "blurred_image": (by_id.get(i["image_id"]) or {}).get("blurred_image"),
                "error": (by_id.get(i["image_id"]) or {}).get("error")
                if i["image_id"] in by_id
                else "no_result",
            }
            for i in images
        ]


def make_blur_client() -> BlurClient:
    """테스트에서 가짜 클라이언트로 바꿔 끼울 수 있게 함수로 둔다."""
    return BlurClient()
