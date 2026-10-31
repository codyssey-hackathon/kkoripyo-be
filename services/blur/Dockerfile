# Text Blur Server: 화면 없는 ComfyUI(내부 전용) + 블러 API(:8190)
# 버전은 setup_linux.sh와 같게 고정 (원래 PC와 동일)
FROM python:3.13-slim

ARG COMFY_COMMIT=b53f77a74e21091cbdebfdf7b85fc80326932646
ARG FLORENCE_NODE_COMMIT=9ece3de914214c5f581d725167bc9d0eeb0d1120
# 서버 GPU 드라이버에 맞는 PyTorch CUDA 빌드 (예: cu128, cu126)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.13.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ComfyUI + Florence-2 노드 (고정 커밋)
RUN git clone https://github.com/Comfy-Org/ComfyUI.git \
    && git -C ComfyUI checkout "$COMFY_COMMIT" \
    && git clone https://github.com/kijai/ComfyUI-Florence2.git ComfyUI/custom_nodes/ComfyUI-Florence2 \
    && git -C ComfyUI/custom_nodes/ComfyUI-Florence2 checkout "$FLORENCE_NODE_COMMIT" \
    && rm -rf ComfyUI/.git ComfyUI/custom_nodes/ComfyUI-Florence2/.git

# 무거운 층(PyTorch)을 먼저 깔아 두면 코드만 바꿀 때 다시 받지 않는다
RUN pip install "torch==${TORCH_VERSION}" torchvision --index-url "$TORCH_INDEX"
RUN pip install -r ComfyUI/requirements.txt \
    && pip install -r ComfyUI/custom_nodes/ComfyUI-Florence2/requirements.txt \
    && pip install transformers==5.14.1 huggingface_hub==1.25.1 timm==1.0.29

COPY workflow_api.json blur_server.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# 모델(약 1.5GB)은 이미지에 넣지 않고 이 경로를 볼륨으로 붙여 처음 한 번만 받는다
VOLUME ["/app/ComfyUI/models/LLM"]
EXPOSE 8190

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -fsS http://localhost:8190/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
