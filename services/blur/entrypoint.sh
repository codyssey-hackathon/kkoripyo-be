#!/usr/bin/env bash
# 컨테이너 시작: 모델이 없으면 받고 → ComfyUI(127.0.0.1 전용) 실행 → 블러 API 실행
# run_linux.sh와 같은 순서이며, 블러 API가 끝나면 컨테이너도 끝난다.
set -euo pipefail
cd /app

FLORENCE_MODEL_REV=${FLORENCE_MODEL_REV:-21a599d414c4d928c9032694c424fb94458e3594}
MODEL_DIR=ComfyUI/models/LLM/Florence-2-large
COMFY_PORT=${COMFY_PORT:-8188}
export COMFY_URL="http://127.0.0.1:${COMFY_PORT}"
export COMFY_DIR="/app/ComfyUI"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Florence-2-large 모델 받는 중 (처음 한 번만, 약 1.5GB)..."
  python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Florence-2-large", revision="$FLORENCE_MODEL_REV", local_dir="$MODEL_DIR")
EOF
  rm -rf "$MODEL_DIR/.cache"
fi

# ComfyUI는 컨테이너 안(127.0.0.1)에서만 열린다. 밖으로는 블러 API(8190)만 나간다.
python ComfyUI/main.py --listen 127.0.0.1 --port "$COMFY_PORT" --disable-auto-launch &
COMFY_PID=$!
trap 'kill $COMFY_PID 2>/dev/null' EXIT

echo "ComfyUI 시작 대기..."
for _ in $(seq 1 180); do
  if curl -sf "$COMFY_URL/system_stats" >/dev/null; then break; fi
  if ! kill -0 $COMFY_PID 2>/dev/null; then echo "ComfyUI가 시작 중에 종료됐습니다."; exit 1; fi
  sleep 1
done

exec python blur_server.py
