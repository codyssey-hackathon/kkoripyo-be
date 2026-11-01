"""가린 사진이 '확실히' 가려졌는지 검사한다.

1) 블러가 실제로 들어갔나: 원본과 가린 사진을 비교해 바뀐 면적 비율을 잰다.
   너무 작으면(글자를 못 찾음) 탈락, 너무 크면(사진 전체가 뭉개짐) 탈락.
2) 글자가 더는 안 읽히나: 가린 사진을 OCR로 다시 읽어, 원래 글자가 얼마나 남았는지 본다.
   (서버 외부 주소가 있고 OCR 서버가 켜져 있을 때만. 없으면 1번만 본다)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.integrations import blur, ocr

MIN_CHANGED = 0.003  # 바뀐 면적이 0.3% 미만이면 블러가 거의 안 들어간 것
MAX_CHANGED = 0.9  # 90% 넘게 바뀌면 사진 전체가 뭉개진 것
MAX_TEXT_LEFT = 0.35  # 가린 뒤에도 원래 글자가 35% 넘게 읽히면 덜 가려진 것
_TEXT_RE = re.compile(r"[0-9A-Za-z가-힣]")


@dataclass
class Verdict:
    ok: bool
    why: str
    changed: float | None = None
    text_left: float | None = None


def changed_ratio(before: bytes, after: bytes, size: int = 256, threshold: int = 12) -> float:
    """두 사진을 같은 크기 흑백으로 줄여, 밝기가 눈에 띄게 달라진 픽셀 비율."""
    from PIL import Image, ImageChops

    a = Image.open(io.BytesIO(before)).convert("L")
    b = Image.open(io.BytesIO(after)).convert("L")
    w = size
    h = max(1, round(a.height * size / max(1, a.width)))
    diff = ImageChops.difference(a.resize((w, h)), b.resize((w, h)))
    hist = diff.histogram()
    return sum(hist[threshold:]) / (w * h)


def text_left(before_text: str, after_text: str) -> float | None:
    """원래 글자 수 대비 가린 뒤 읽히는 글자 수."""
    n_before = len(_TEXT_RE.findall(before_text or ""))
    if n_before < 4:
        return None  # 원래 글자가 너무 적으면 비교 의미가 없다
    return len(_TEXT_RE.findall(after_text or "")) / n_before


async def _download(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as http:
            res = await http.get(url)
            res.raise_for_status()
            return res.content
    except httpx.HTTPError:
        return None


async def _ocr_after(items: list[dict]) -> dict[str, str]:
    """가린 사진을 OCR로 다시 읽는다. {image_id: 글자}. 못 읽으면 빈 dict."""
    base = get_settings().public_base_url.rstrip("/")
    if not base:
        return {}
    images = [
        {"image_id": f"{i['image_id']}_after", "image_url": base + i["after"]}
        for i in items
        if i.get("after")
    ]
    if not images:
        return {}
    try:
        async with ocr.make_ocr_client() as client:
            result = await client.read(images)
    except Exception:
        return {}
    return {
        k.removesuffix("_after"): v.get("text") or ""
        for k, v in result.items()
        if not v.get("error")
    }


async def assess(items: list[dict], before_texts: dict[str, str]) -> list[Verdict]:
    """items: [{image_id, before(원본 주소), after(/api/v1/agent/blurred/..)}] → 사진마다 판정."""
    reread = await _ocr_after(items)
    verdicts = []
    for item in items:
        if not item.get("after"):
            verdicts.append(Verdict(False, "블러 서버가 사진을 가리지 못함"))
            continue
        after_bytes = blur.kept_image(item["after"].rsplit("/", 1)[-1])
        before_bytes = await _download(item["before"])
        changed = None
        if after_bytes and before_bytes:
            try:
                changed = changed_ratio(before_bytes, after_bytes)
            except Exception:
                changed = None
        if changed is None:
            verdicts.append(Verdict(False, "가린 결과를 확인하지 못함"))
            continue
        if changed < MIN_CHANGED:
            verdicts.append(Verdict(False, "글자를 거의 찾지 못해 가려진 곳이 없음", changed))
            continue
        if changed > MAX_CHANGED:
            verdicts.append(Verdict(False, "사진 전체가 뭉개짐", changed))
            continue
        left = None
        if item["image_id"] in reread:
            left = text_left(before_texts.get(item["image_id"], ""), reread[item["image_id"]])
        if left is not None and left > MAX_TEXT_LEFT:
            verdicts.append(
                Verdict(False, f"가린 뒤에도 글자가 {round(left * 100)}% 읽힘", changed, left)
            )
            continue
        why = "가린 뒤 글자가 읽히지 않음" if left is not None else "글자 영역이 가려짐"
        verdicts.append(Verdict(True, why, changed, left))
    return verdicts
