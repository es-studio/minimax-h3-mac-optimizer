#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
COMFY="$ROOT/ComfyUI"
HF="$COMFY/venv/bin/hf"
WITH_R2V=0

if [[ ! -x "$HF" ]]; then
  echo "먼저 ./setup.sh 를 실행하세요." >&2
  exit 1
fi

WITH_INT4=1
WITH_INT8_TE=1
WITH_INT4_TE=1

usage() {
  cat <<'EOF'
MiniMax H3 가중치를 Hugging Face에서 받습니다.

  ./download_models.sh              # T2V / I2V (~42GB) + INT4 DiT (~11GB) + INT8 32B TE (~27GB) + INT4 32B TE (~15GB)
  ./download_models.sh --no-int4    # 공식 pruned INT8 DiT만
  ./download_models.sh --no-int8-te # 32B INT8 TE 생략 (NVFP4 / ClipProj 4B만)
  ./download_models.sh --no-int4-te # 32B INT4 TE 생략
  ./download_models.sh --with-r2v   # 위 + R2V Ref2VA (~+21GB)
EOF
}

for arg in "$@"; do
  case "$arg" in
    --with-r2v) WITH_R2V=1 ;;
    --no-int4) WITH_INT4=0 ;;
    --no-int8-te) WITH_INT8_TE=0 ;;
    --no-int4-te) WITH_INT4_TE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "알 수 없는 옵션: $arg" >&2; usage; exit 1 ;;
  esac
done

mkdir -p \
  "$COMFY/models/diffusion_models" \
  "$COMFY/models/text_encoders" \
  "$COMFY/models/vae" \
  "$COMFY/models/clip_projections"

INCLUDES=(
  --include "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  --include "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  --include "vae/minimax_h3_video_vae_fp16.safetensors"
  --include "vae/minimax_h3_audio_vae_fp32.safetensors"
)

if [[ "$WITH_R2V" -eq 1 ]]; then
  INCLUDES+=(--include "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
fi

echo "다운로드 대상: Comfy-Org/MiniMax-H3 -> $COMFY/models"
echo "Apple Silicon 권장 양자화: pruned INT8 convrot + NVFP4 AWQ 텍스트 인코더"
"$HF" download Comfy-Org/MiniMax-H3 \
  "${INCLUDES[@]}" \
  --local-dir "$COMFY/models"

if [[ "$WITH_INT4" -eq 1 ]]; then
  echo
  echo "24GB용 pruned INT4 ConvRot DiT (Merserk/MiniMax-H3-INT4-ConvRot, ~11.3GB)"
  "$HF" download Merserk/MiniMax-H3-INT4-ConvRot \
    minimax_h3_fl2va_pruned_int4_convrot.safetensors \
    --local-dir "$COMFY/models/diffusion_models"
  INT4_NESTED="$COMFY/models/diffusion_models/minimax_h3_fl2va_pruned_int4_convrot.safetensors"
  if [[ ! -f "$INT4_NESTED" ]]; then
    found="$(find "$COMFY/models/diffusion_models" -name 'minimax_h3_fl2va_pruned_int4_convrot.safetensors' | head -1 || true)"
    if [[ -n "${found}" && "${found}" != "$INT4_NESTED" ]]; then
      mv "$found" "$INT4_NESTED"
    fi
  fi
fi

if [[ "$WITH_INT8_TE" -eq 1 ]]; then
  echo
  echo "24GB용 공식 32B INT8 ConvRot TE (레이어 스트림, ~27GB)"
  "$HF" download Comfy-Org/MiniMax-H3 \
    --include "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    --local-dir "$COMFY/models"
fi

if [[ "$WITH_INT4_TE" -eq 1 ]]; then
  echo
  echo "24GB용 32B INT4 ConvRot TE (레이어 스트림, ~15GB, Merserk)"
  "$HF" download Merserk/MiniMax-H3-INT4-ConvRot \
    qwen3vl_32b_minimax_h3_int4_convrot.safetensors \
    --local-dir "$COMFY/models/text_encoders"
  INT4_TE="$COMFY/models/text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
  if [[ ! -f "$INT4_TE" ]]; then
    found="$(find "$COMFY/models/text_encoders" -name 'qwen3vl_32b_minimax_h3_int4_convrot.safetensors' | head -1 || true)"
    if [[ -n "${found}" && "${found}" != "$INT4_TE" ]]; then
      mv "$found" "$INT4_TE"
    fi
  fi
fi

echo
echo "ClipProj 4B 텍스트 인코더 (Krea-2 fp8) + MiniMax H3 v3-mlp 투영"
"$HF" download Comfy-Org/Krea-2 \
  --include "text_encoders/qwen3vl_4b_fp8_scaled.safetensors" \
  --local-dir "$COMFY/models"
"$HF" download NicoLab28/ClipProj-MiniMax-H3 \
  mmh3-4b-ClipProj-v3-mlp.safetensors \
  --local-dir "$COMFY/models/clip_projections"

# hf may nest the projection file; flatten if needed
PROJ_NESTED="$COMFY/models/clip_projections/mmh3-4b-ClipProj-v3-mlp.safetensors"
if [[ ! -f "$PROJ_NESTED" ]]; then
  found="$(find "$COMFY/models/clip_projections" -name 'mmh3-4b-ClipProj-v3-mlp.safetensors' | head -1 || true)"
  if [[ -n "${found}" && "${found}" != "$PROJ_NESTED" ]]; then
    mv "$found" "$PROJ_NESTED"
  fi
fi

echo
echo "모델 배치 결과:"
ls -lh \
  "$COMFY/models/diffusion_models"/minimax_h3_*.safetensors 2>/dev/null \
  "$COMFY/models/text_encoders"/qwen3vl_*.safetensors \
  "$COMFY/models/vae"/minimax_h3_*.safetensors \
  "$COMFY/models/clip_projections"/mmh3-4b-ClipProj-v3-mlp.safetensors
