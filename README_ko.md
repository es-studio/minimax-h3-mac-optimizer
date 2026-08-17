# MiniMax H3 on Apple Silicon (MBA M5 24GB) 🚀

이 프로젝트는 **총 47GB에 달하는 초거대 모델(MiniMax H3 비디오 생성 20GB + 32B 텍스트 인코더 27GB)**을 물리 메모리가 **24GB**에 불과한 Mac(Apple M5)에서 완벽하게 구동하기 위해 개발된 로컬 최적화 파이프라인입니다.

## 🏆 핵심 성과 (Achievements)

NVIDIA 24GB VRAM 환경에서도 OOM(Out of Memory)이 발생하는 47GB 규모의 모델을 Apple Silicon의 **통합 메모리(Unified Memory)**와 **극한의 스트리밍 최적화**를 통해 스왑(Swap) 지옥 없이 안정적으로 구동하는 데 성공했습니다.

- **해상도 및 길이:** 864x480, 5초(124 프레임), 20 Steps
- **소요 시간:** 스텝당 약 215초 (총 1시간 12분 소요)
- **메모리 유지:** 디스크 스트리밍 기법으로 물리 램 24GB 이내에서 쾌적하게 안착

## ⚙️ 어떻게 24GB 램에서 47GB 모델을 돌렸나?

### 1. 텍스트 인코더 (TE) 레이어 스트림 
- **모델:** Qwen3-VL 32B INT8 (약 27GB)
- **방법:** 27GB를 한 번에 올리지 않고, `embed_tokens` 등 약 2.6GB만 상주(Resident)시킨 뒤, 50개의 레이어(약 465MB/개)를 인코딩할 때마다 디스크에서 순차적으로 읽고 바로 버립니다(`F_NOCACHE`). 인코딩이 끝나면 메모리에서 즉시 완전 삭제(Drop)하여 다음 프로세스를 위한 공간을 확보합니다.

### 2. DiT (Diffusion Transformer) 블록 스트림
- **모델:** MiniMax H3 DiT INT8 (약 20GB)
- **방법:** 약 1.5GB만 메모리에 상주시키고, 50개의 블록(약 369MB/개)을 연산할 때마다 순차적으로 불러옵니다. 

### 3. I/O 오버헤드 은닉 (Prefetching)
- "디스크에서 매번 읽어오면 너무 느리지 않을까?" 하는 우려를 **비동기 프리패치(Prefetch)**로 해결했습니다.
- GPU가 N번째 블록을 계산하는 동안 백그라운드 스레드(CPU)가 N+1번째 블록을 램으로 미리 가져옵니다. 그 결과, 스텝당 215초 중 **순수 디스크 대기 시간(I/O)은 단 2초 내외(~1%)**로 줄어들어 사실상 I/O 병목이 0에 수렴합니다.

### 4. AppleSilicon-FP8 네이티브 가속
- PyTorch MPS의 한계로 인해 INT8 연산이 CPU로 튕겨나가는 현상을 막기 위해 `ComfyUI-AppleSilicon-FP8` 커스텀 노드를 사용했습니다.
- M5 GPU의 Metal 전용 INT8 커널(W8A8)과 W4A16 Fast Path를 강제로 활성화하여, CPU 개입 없이 GPU 사용률을 70% 이상으로 유지하며 네이티브 하드웨어 가속을 100% 끌어냅니다.

### 5. Sage Attention → Metal Flash (mtlflashattn)
- CUDA SageAttention은 Apple Silicon에서 동작하지 않습니다. `./start.sh` 가 ComfyUI `--use-sage-attention` 을 켜고, `vendor/sageattention` 심이 런타임에 `F.scaled_dot_product_attention` 으로 넘깁니다.
- AppleSilicon-FP8 이 그 SDPA 호출을 **mtlflashattn**(온라인 소프트맥스, Lq×Lk 점수 행렬 없음)으로 라우팅합니다. 플래그가 없으면 Comfy는 sub-quadratic attention에 머물고 Metal flash가 DiT에 타지 않습니다. 864×480 5초 기준 packed 토큰은 약 1.5만, head_dim 128입니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 설치 및 모델 다운로드
```bash
cd /Users/eunsung/minimax
./setup.sh
./download_models.sh          # T2V/I2V 약 42GB 다운로드
# ./download_models.sh --with-r2v   # R2V 추가시 약 21GB 추가
```

### 2. 서버 실행
```bash
./start.sh
```
- 실행 플래그(`--disable-smart-memory --fast-disk --cache-none --reserve-vram 2`)와 MPS 워터마크 해제(`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`)가 자동으로 적용됩니다.

### 3. 모델 구동 (T2V)
24GB Mac 환경에서 가장 완벽하게 검증된 파이프라인은 아래 큐(Queue) 스크립트를 통해 백그라운드에서 실행할 수 있습니다.

```bash
# 32B INT8 TE + INT8 DiT (가장 추천하는 24GB 스트리밍 파이프라인)
python3 queue_t2v_int8_te_blockstream.py

# 4B Krea-2 TE + INT8 DiT (가벼운 인코더 대안)
python3 queue_t2v_int8_blockstream.py

# INT4 DiT 풀로드 (약 11.3GB 통째 로드 시연)
python3 queue_t2v_int4.py
```
- 실행 결과 MP4 비디오 파일은 `ComfyUI/output/video/` 디렉토리에 저장됩니다.

---

## 📂 주요 모델 파일 목록 (Hugging Face)

| 파일 | 위치 | 용도 및 크기 |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | 메인 DiT (INT8, ~20GB) |
| `minimax_h3_fl2va_pruned_int4_convrot.safetensors` | `models/diffusion_models/` | 메인 DiT (INT4, ~11.3GB) |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `models/text_encoders/` | 공식 32B TE (INT8, ~27GB) |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` | 4B 가벼운 TE 대안 |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | 비디오 디코드 공통 |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | 오디오 디코드 공통 |

## 💡 성능 최적화 팁 (향후 과제)
현재 파이프라인으로 생성 시 약 **1시간 12분 (20스텝 기준)**이 소요됩니다. 시간을 단축하고 싶다면 다음을 권장합니다:
1. **Turbo / Spectrum LoRA 적용:** 20스텝을 6스텝으로 줄여 소요 시간을 20분대로 단축.
2. **해상도/프레임 타협:** 프롬프트 테스트 단계에서는 비디오 길이를 2~3초로 줄여 반응성 확보.
3. **INT4 모델 활용:** 11GB INT4 DiT를 풀로드하여 대역폭 한계를 완화하면 스텝당 시간을 약 40초가량 앞당길 수 있습니다. (스텝당 평균 ~173초)

## 📌 참고 자료
- ComfyUI 공식 가이드: https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- 프롬프트 작성 가이드: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
