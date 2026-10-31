"""Text blur API server.

Takes an image, runs the Text-Blur-Florence2 workflow on a local headless
ComfyUI, and returns the blurred image. Input and output files are removed
from ComfyUI after every request so no personal data stays on disk.

Settings (environment variables):
  COMFY_URL   ComfyUI address              (default http://127.0.0.1:8188)
  COMFY_DIR   ComfyUI folder, for cleanup  (default ./ComfyUI next to this file)
  HOST, PORT  where this server listens    (default 0.0.0.0, 8190)
  API_KEY     if set, requests must send header "X-API-Key: <value>"
  TIMEOUT     seconds to wait per image    (default 300)
"""
import asyncio
import base64
import copy
import io
import json
import os
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", HERE / "ComfyUI"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8190"))
API_KEY = os.environ.get("API_KEY", "")
TIMEOUT = float(os.environ.get("TIMEOUT", "300"))
MAX_UPLOAD_MB = 50

WORKFLOW = json.loads((HERE / "workflow_api.json").read_text(encoding="utf-8"))
LOAD_NODE, GROW_NODE, SAVE_NODE = "1", "4", "8"


def remove_quietly(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


async def run_workflow(session: aiohttp.ClientSession, image_bytes: bytes, expand: int | None) -> bytes:
    name = f"textblur_{uuid.uuid4().hex}.png"
    form = aiohttp.FormData()
    form.add_field("image", image_bytes, filename=name, content_type="application/octet-stream")
    form.add_field("type", "input")
    form.add_field("overwrite", "true")
    async with session.post(f"{COMFY_URL}/upload/image", data=form) as r:
        r.raise_for_status()
        uploaded = await r.json()

    prompt = copy.deepcopy(WORKFLOW)
    prompt[LOAD_NODE]["inputs"]["image"] = uploaded["name"]
    if expand is not None:
        prompt[GROW_NODE]["inputs"]["expand"] = expand

    output_files = []
    prompt_id = None
    try:
        async with session.post(f"{COMFY_URL}/prompt", json={"prompt": prompt}) as r:
            body = await r.json()
            if r.status != 200:
                raise web.HTTPBadGateway(text=f"ComfyUI rejected the workflow: {json.dumps(body, ensure_ascii=False)}")
            prompt_id = body["prompt_id"]

        deadline = asyncio.get_running_loop().time() + TIMEOUT
        while True:
            async with session.get(f"{COMFY_URL}/history/{prompt_id}") as r:
                history = await r.json()
            if prompt_id in history:
                entry = history[prompt_id]
                break
            if asyncio.get_running_loop().time() > deadline:
                raise web.HTTPGatewayTimeout(text="Timed out waiting for ComfyUI.")
            await asyncio.sleep(0.3)

        status = entry.get("status", {})
        if status.get("status_str") != "success":
            errors = [m[1].get("exception_message", "") for m in status.get("messages", []) if m[0] == "execution_error"]
            raise web.HTTPBadGateway(text="Workflow failed: " + " | ".join(errors))

        images = entry["outputs"][SAVE_NODE]["images"]
        output_files = [COMFY_DIR / "output" / i["subfolder"] / i["filename"] for i in images]
        img = images[0]
        params = {"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]}
        async with session.get(f"{COMFY_URL}/view", params=params) as r:
            r.raise_for_status()
            return await r.read()
    finally:
        remove_quietly(COMFY_DIR / "input" / uploaded.get("subfolder", "") / uploaded["name"])
        for f in output_files:
            remove_quietly(f)
        if prompt_id:
            async with session.post(f"{COMFY_URL}/history", json={"delete": [prompt_id]}):
                pass


@web.middleware
async def check_api_key(request, handler):
    if API_KEY and request.path != "/health" and request.headers.get("X-API-Key") != API_KEY:
        raise web.HTTPUnauthorized(text="Missing or wrong X-API-Key.")
    return await handler(request)


async def blur(request: web.Request):
    if request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        image_bytes = None
        async for part in reader:
            if part.name == "image":
                image_bytes = await part.read()
        if not image_bytes:
            raise web.HTTPBadRequest(text='Send the picture in a form field named "image".')
    else:
        image_bytes = await request.read()
        if not image_bytes:
            raise web.HTTPBadRequest(text="Send the picture as the request body or as form field \"image\".")

    expand = request.query.get("expand")
    result = await run_workflow(request.app["session"], image_bytes, int(expand) if expand else None)
    return web.Response(body=result, content_type="image/png")


MAX_URL_IMAGES = 4
MAX_FETCH_MB = 10
JPEG_QUALITY = 90


def to_jpeg_data_url(png_bytes: bytes) -> str:
    """ComfyUI가 저장한 PNG를 JPEG로 바꿔 data URL로. (사진 PNG는 너무 커서 응답에 그대로 담기 어려움)"""
    from PIL import Image  # ComfyUI 설치에 포함

    with Image.open(io.BytesIO(png_bytes)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise ValueError("fetch_failed: http(s) URL만 허용")
    limit = MAX_FETCH_MB * 1024 * 1024
    chunks, size = [], 0
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
            r.raise_for_status()
            # 끝까지 다 읽는다. (content.read(n)은 지금 도착한 만큼만 돌려줘서 사진이 잘린다)
            async for chunk in r.content.iter_chunked(64 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"too_large: {MAX_FETCH_MB}MB 초과")
                chunks.append(chunk)
    except aiohttp.ClientError as e:
        raise ValueError(f"fetch_failed: 이미지 다운로드 실패 ({type(e).__name__})") from e
    data = b"".join(chunks)
    if not data:
        raise ValueError("fetch_failed: 빈 이미지")
    return data


async def blur_urls(request: web.Request):
    """이미지 링크를 최대 4개 받아 블러하고, 이미지마다 블러된 사진(JPEG data URL)을 돌려준다.

    요청: {"images": [{"image_id": "...", "image_url": "https://..."}]}
    응답: {"images": [{"image_id", "image_url", "blurred_image": "data:image/jpeg;base64,...", "error": null}]}
    한 장이 실패해도 나머지는 처리하고, 실패한 장은 blurred_image=null, error에 사유를 담는다.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(text="JSON 본문이 필요합니다.") from None
    images = body.get("images") if isinstance(body, dict) else None
    if not isinstance(images, list) or not images:
        raise web.HTTPBadRequest(text='"images" 배열이 필요합니다.')
    if len(images) > MAX_URL_IMAGES:
        raise web.HTTPBadRequest(text=f"이미지는 한 번에 최대 {MAX_URL_IMAGES}개까지입니다.")
    expand = request.query.get("expand")
    session = request.app["session"]

    async def one(item: dict) -> dict:
        image_id = str(item.get("image_id") or "")
        url = item.get("image_url") or ""
        out = {"image_id": image_id, "image_url": url, "blurred_image": None, "error": None}
        try:
            data = await fetch_image(session, url)
            png = await run_workflow(session, data, int(expand) if expand else None)
            out["blurred_image"] = to_jpeg_data_url(png)
        except ValueError as e:
            out["error"] = str(e)
        except web.HTTPException as e:
            out["error"] = f"blur_failed: {e.text}"
        except Exception as e:  # 한 장 실패가 전체를 멈추지 않게
            out["error"] = f"blur_failed: {type(e).__name__}"
        return out

    # 다운로드는 동시에, 블러는 ComfyUI가 큐에 넣어 차례로 처리한다
    results = await asyncio.gather(*(one(i if isinstance(i, dict) else {}) for i in images))
    return web.json_response({"images": results})


async def health(request: web.Request):
    try:
        async with request.app["session"].get(f"{COMFY_URL}/system_stats") as r:
            ok = r.status == 200
    except aiohttp.ClientError:
        ok = False
    return web.json_response({"server": "ok", "comfyui": "ok" if ok else "unreachable"}, status=200 if ok else 503)


async def on_startup(app):
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT + 30))


async def on_cleanup(app):
    await app["session"].close()


def make_app():
    app = web.Application(middlewares=[check_api_key], client_max_size=MAX_UPLOAD_MB * 1024 * 1024)
    app.router.add_post("/blur", blur)
    app.router.add_post("/blur/urls", blur_urls)
    app.router.add_get("/health", health)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host=HOST, port=PORT)
