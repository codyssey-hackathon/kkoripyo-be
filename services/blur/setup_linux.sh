#!/usr/bin/env bash
# Installs headless ComfyUI + Florence-2 node + Florence-2-large model next to this script.
# Versions match the PC the workflow was built and tested on.
#
# Usage:  bash setup_linux.sh
# Change the PyTorch build with TORCH_INDEX, e.g. for CUDA 12.8:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_linux.sh
# CPU only (slow):
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash setup_linux.sh
set -euo pipefail
cd "$(dirname "$0")"

COMFY_COMMIT=b53f77a74e21091cbdebfdf7b85fc80326932646      # ComfyUI 0.41.2
FLORENCE_NODE_COMMIT=9ece3de914214c5f581d725167bc9d0eeb0d1120
FLORENCE_MODEL_REV=21a599d414c4d928c9032694c424fb94458e3594
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
PYTHON=${PYTHON:-python3}

if [ ! -d ComfyUI ]; then
  git clone https://github.com/Comfy-Org/ComfyUI.git
fi
git -C ComfyUI checkout "$COMFY_COMMIT"

if [ ! -d ComfyUI/custom_nodes/ComfyUI-Florence2 ]; then
  git clone https://github.com/kijai/ComfyUI-Florence2.git ComfyUI/custom_nodes/ComfyUI-Florence2
fi
git -C ComfyUI/custom_nodes/ComfyUI-Florence2 checkout "$FLORENCE_NODE_COMMIT"

if [ ! -d venv ]; then
  "$PYTHON" -m venv venv
fi
. venv/bin/activate
pip install --upgrade pip
pip install torch==2.13.0 torchvision --index-url "$TORCH_INDEX"
pip install -r ComfyUI/requirements.txt
pip install -r ComfyUI/custom_nodes/ComfyUI-Florence2/requirements.txt
pip install transformers==5.14.1 huggingface_hub==1.25.1 timm==1.0.29

python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Florence-2-large", revision="$FLORENCE_MODEL_REV",
                  local_dir="ComfyUI/models/LLM/Florence-2-large")
EOF
rm -rf ComfyUI/models/LLM/Florence-2-large/.cache

echo
echo "Setup done. Start the server with:  bash run_linux.sh"
