#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMFY="$ROOT/ComfyUI"

if [[ ! -x "$COMFY/venv/bin/python" ]]; then
  echo "먼저 ./setup.sh 를 실행하세요." >&2
  exit 1
fi

cd "$COMFY"
# Apple Silicon 에서 미지원 연산은 CPU 로 넘긴다.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PATH="/opt/homebrew/bin:$PATH"

# AppleSilicon-FP8 defaults HIGH_WATERMARK to 1.0 x Metal recommended_max
# (17.76 GiB here = 74% of 24GB). The env var is a multiplier of that
# recommended_max, not of total RAM.
#
# 90% (~21.6 GiB) and 98% (~23.5 GiB) both OOM: DiT INT8 is ~20 GB and the
# first sampler step is already at ~22.2 GiB when a 256 MiB scale tensor
# fails despite apparent headroom (MPS fragmentation / watermark accounting).
# 0.0 disables the PyTorch cap so that last allocation can succeed.
# Turn off AppleSilicon-FP8's prestartup cap (it would setdefault high=1.0 / low=0.8).
if [[ -z "${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-}" ]]; then
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi
export APPLESILICON_FP8_MPS_WATERMARK="${APPLESILICON_FP8_MPS_WATERMARK:-off}"
echo "MPS high watermark ratio=${PYTORCH_MPS_HIGH_WATERMARK_RATIO} (cap disabled)"

# INT8 Metal kernel is the fast W8A8 path. It OOMs under a 90/98% cap; with
# the allocator ceiling disabled it should have room for the first-step
# activations. Override with ASFP8_INT8_EXT=off if the kernel still fails.
export ASFP8_INT8_EXT="${ASFP8_INT8_EXT:-on}"
echo "ASFP8_INT8_EXT=${ASFP8_INT8_EXT}"

# INT8 DiT (~20GB) cannot reside on 24GB. Load one transformer block at a
# time from safetensors (pread + F_NOCACHE). auto = INT8 only; INT4 still
# full-loads. Set MINIMAX_DIT_BLOCK_STREAM=off to disable.
export MINIMAX_DIT_BLOCK_STREAM="${MINIMAX_DIT_BLOCK_STREAM:-auto}"
export MINIMAX_DIT_BLOCK_PREFETCH="${MINIMAX_DIT_BLOCK_PREFETCH:-1}"
echo "MINIMAX_DIT_BLOCK_STREAM=${MINIMAX_DIT_BLOCK_STREAM} prefetch=${MINIMAX_DIT_BLOCK_PREFETCH}"

# Official 32B INT8 TE (~27GB) and INT4 ConvRot TE (~15GB): one LM layer
# at a time. auto = qwen3vl_32b + (int8|int4). ClipProj 4B unchanged.
# NVFP4 still full-loads.
export MINIMAX_TE_BLOCK_STREAM="${MINIMAX_TE_BLOCK_STREAM:-auto}"
export MINIMAX_TE_BLOCK_PREFETCH="${MINIMAX_TE_BLOCK_PREFETCH:-1}"
echo "MINIMAX_TE_BLOCK_STREAM=${MINIMAX_TE_BLOCK_STREAM} prefetch=${MINIMAX_TE_BLOCK_PREFETCH}"

# CUDA SageAttention cannot install on Mac. vendor/sageattention exposes the
# sageattn() API Comfy imports for --use-sage-attention, then calls F.sdpa at
# runtime so AppleSilicon-FP8 mtlflashattn actually runs. Without this flag
# Comfy stays on sub-quadratic attention and never hits the Metal flash path.
# PYTHONPATH must be set before python starts (attention.py imports sageattn
# at module load, before custom nodes).
export PYTHONPATH="$ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
echo "attention: --use-sage-attention via vendor/sageattention (mtlflashattn SDPA)"

# 24GB unified memory: one model family at a time.
# ComfyUI --fast-disk is DynamicVRAM/aimdo (CUDA/ROCm only). On Mac the
# minimax_sequential_offload node drops modules after use and reloads from
# safetensors, which is the unified-memory equivalent.
echo "fast-disk: Mac disk-backed sequential offload (aimdo is CUDA/ROCm only)"
exec ./venv/bin/python main.py \
  --listen 127.0.0.1 \
  --port 8188 \
  --enable-manager \
  --disable-api-nodes \
  --disable-smart-memory \
  --fast-disk \
  --cache-none \
  --reserve-vram 2 \
  --use-sage-attention
