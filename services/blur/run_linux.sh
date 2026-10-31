#!/usr/bin/env bash
# Starts headless ComfyUI (local only) and the blur API in front of it.
# Stop both with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"
. venv/bin/activate

COMFY_PORT=${COMFY_PORT:-8188}
export COMFY_URL="http://127.0.0.1:${COMFY_PORT}"
export COMFY_DIR="$PWD/ComfyUI"

# ComfyUI listens on 127.0.0.1 only, so just the blur API is reachable from outside.
python ComfyUI/main.py --listen 127.0.0.1 --port "$COMFY_PORT" --disable-auto-launch &
COMFY_PID=$!
trap 'kill $COMFY_PID 2>/dev/null' EXIT

echo "Waiting for ComfyUI..."
for _ in $(seq 1 180); do
  if curl -sf "$COMFY_URL/system_stats" >/dev/null; then break; fi
  if ! kill -0 $COMFY_PID 2>/dev/null; then echo "ComfyUI exited during startup."; exit 1; fi
  sleep 1
done

python blur_server.py
