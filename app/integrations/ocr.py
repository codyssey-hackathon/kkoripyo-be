"""OCR 서버 클라이언트 (팀 OCR API, GPU PaddleOCR).

요청: POST {OCR_SERVER_URL}/ocr  {"images": [{"image_id", "image_url"}]}  (한 번에 최대 20장)
응답: {"images": [{"image_id", "image_url", "isTextPresent", "data", "error"}]}
  - isTextPresent=false면 data는 ""
  - error: 이미지 다운로드 실패 등 사유 (성공·글자 없음이면 null)

image_id 규칙: {post_id}_{순번}. 지금 수집 API는 게시물당 첫 장만 주므로 항상 _0.
"""

from __future__ import annotations

import asyncio

import httpx

from app.core.config import get_settings

MAX_PER_REQUEST = 20  # OCR 서버 한도
CONCURRENCY = 2  # GPU 하나라 동시에 너무 많이 보내지 않는다


def image_id(post_id: str, index: int = 0) -> str:
    return f"{post_id}_{index}"


class OcrClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        url = base_url if base_url is not None else get_settings().ocr_server_url
        self.enabled = bool(url)
        self._http = httpx.AsyncClient(
            base_url=url or "http://ocr.invalid", timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> OcrClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()

    async def read(self, images: list[dict]) -> dict[str, dict]:
        """[{image_id, image_url}] → {image_id: {"text": str, "error": str|None}}.

        OCR 서버가 꺼져 있거나 일부 요청이 실패해도 예외를 던지지 않고, 해당 이미지에 error를 남긴다.
        """
        if not images:
            return {}
        if not self.enabled:
            return {i["image_id"]: {"text": "", "error": "ocr_disabled"} for i in images}

        sem = asyncio.Semaphore(CONCURRENCY)
        batches = [images[i : i + MAX_PER_REQUEST] for i in range(0, len(images), MAX_PER_REQUEST)]

        async def one(batch: list[dict]) -> dict[str, dict]:
            async with sem:
                try:
                    res = await self._http.post("/ocr", json={"images": batch})
                    res.raise_for_status()
                    rows = res.json().get("images") or []
                except (httpx.HTTPError, ValueError) as e:
                    reason = f"ocr_unavailable: {type(e).__name__}"
                    return {i["image_id"]: {"text": "", "error": reason} for i in batch}
            return {
                r["image_id"]: {
                    "text": (r.get("data") or "") if r.get("isTextPresent") else "",
                    "error": r.get("error"),
                }
                for r in rows
                if r.get("image_id")
            }

        out: dict[str, dict] = {}
        for part in await asyncio.gather(*(one(b) for b in batches)):
            out.update(part)
        return out


def make_ocr_client() -> OcrClient:
    """테스트에서 가짜 클라이언트로 바꿔 끼울 수 있게 함수로 둔다."""
    return OcrClient()


async def ocr_posts(posts: list[dict]) -> dict[str, int]:
    """게시물 목록의 사진을 OCR해서 각 게시물에 ocr_text를 붙인다. 반환: 개수 요약."""
    images = [
        {"image_id": image_id(p["post_id"]), "image_url": p["image_url"]}
        for p in posts
        if p.get("post_id") and p.get("image_url")
    ]
    async with make_ocr_client() as client:
        result = await client.read(images)
    with_text = 0
    for p in posts:
        r = result.get(image_id(p.get("post_id") or ""))
        p["ocr_text"] = r["text"] if r else ""
        with_text += bool(p["ocr_text"])
    failed = sum(1 for r in result.values() if r.get("error"))
    return {"images": len(images), "with_text": with_text, "failed": failed}
