#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMFY="$ROOT/ComfyUI"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
export PATH="/opt/homebrew/bin:/Users/eunsung/.local/bin:$PATH"

if ! command -v git >/dev/null 2>&1; then
  echo "git 이 필요합니다." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12 이 필요합니다. 예: brew install python@3.12" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg ninja
  else
    echo "ffmpeg 와 ninja 가 필요합니다." >&2
    exit 1
  fi
fi

if [[ ! -d "$COMFY/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi

if [[ ! -x "$COMFY/venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$COMFY/venv"
fi

PIP=("$COMFY/venv/bin/python" -m pip)
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$COMFY/venv/bin/python" --upgrade pip
  uv pip install --python "$COMFY/venv/bin/python" \
    -r "$COMFY/requirements.txt" \
    -r "$COMFY/manager_requirements.txt" \
    huggingface_hub ninja
else
  "${PIP[@]}" install --upgrade pip
  "${PIP[@]}" install \
    -r "$COMFY/requirements.txt" \
    -r "$COMFY/manager_requirements.txt" \
    huggingface_hub ninja
fi

ASFP8="$COMFY/custom_nodes/ComfyUI-AppleSilicon-FP8"
if [[ ! -d "$ASFP8/.git" ]]; then
  git clone --depth 1 https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8.git "$ASFP8"
fi

CLIPPROJ="$COMFY/custom_nodes/ComfyUI-ClipProj"
if [[ ! -d "$CLIPPROJ/.git" ]]; then
  git clone --depth 1 https://github.com/nicolab28/ComfyUI-ClipProj.git "$CLIPPROJ"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$COMFY/venv/bin/python" -r "$ASFP8/requirements.txt"
else
  "${PIP[@]}" install -r "$ASFP8/requirements.txt"
fi

SEQ_OFFLOAD_SRC="$ROOT/custom_nodes/minimax_sequential_offload"
SEQ_OFFLOAD_DST="$COMFY/custom_nodes/minimax_sequential_offload"
if [[ -d "$SEQ_OFFLOAD_SRC" ]]; then
  mkdir -p "$COMFY/custom_nodes"
  ln -sfn "$SEQ_OFFLOAD_SRC" "$SEQ_OFFLOAD_DST"
fi

mkdir -p "$COMFY/user/default/workflows" \
  "$COMFY/models/diffusion_models" \
  "$COMFY/models/text_encoders" \
  "$COMFY/models/vae"

if [[ -d "$ROOT/workflows" ]]; then
  cp "$ROOT/workflows/"*.json "$COMFY/user/default/workflows/"
fi

"$COMFY/venv/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}")
print(f"mps available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS 를 사용할 수 없습니다. PyTorch Apple Silicon 설치를 확인하세요.")
PY

echo
echo "설치 완료."
echo "  모델:  ./download_models.sh"
echo "  실행:  ./start.sh"
