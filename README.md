# MiniMax H3 Local Optimizer for Apple Silicon (24GB Macs) 🚀

This repository provides an all-in-one, highly optimized pipeline to run the massive **MiniMax H3 Video Generation model (~47GB total)** on Apple Silicon Macs with just **24GB of Unified Memory**.

By default, loading the 27GB Text Encoder (Qwen3-VL 32B) and the 20GB DiT (Diffusion Transformer) simultaneously would cause severe Out-of-Memory (OOM) errors or extreme swap thrashing on a 24GB machine. This project solves that physical limitation using aggressive **Disk-Backed Sequential Streaming** and **Native Metal Custom Kernels**.

## 🏆 Key Achievements
We successfully run a **47GB pipeline** on a **24GB M5 Mac** with zero swap thrashing.
- **Output:** 864x480 resolution, 5 seconds (124 frames), 20 Steps
- **Performance:** ~215 seconds per step (Total time: ~1 hour 12 mins)
- **Peak Memory:** Stably capped well within 24GB physical RAM.

## ⚙️ How It Works (The Magic)

### 1. Text Encoder (TE) Layer Streaming
- **Model:** Qwen3-VL 32B INT8 (~27GB)
- **Method:** Instead of loading 27GB into VRAM, we keep only ~2.6GB of resident tensors (embeddings, vision tower) in memory. The remaining 50 transformer layers (~465MB each) are dynamically read from the NVMe SSD (`F_NOCACHE`), computed, and immediately dropped per layer. Once encoding is finished, the TE is entirely flushed from memory.

### 2. DiT (Diffusion Transformer) Block Streaming
- **Model:** MiniMax H3 DiT INT8 (~20GB)
- **Method:** Only ~1.5GB of the model remains in RAM. The 50 DiT blocks (~369MB each) are sequentially streamed from disk during the forward pass.

### 3. Hiding I/O Overhead (Asynchronous Prefetching)
- Streaming GBs of data per step from an SSD sounds slow, but we implemented **CPU Prefetching**. 
- While the GPU computes Block $N$, a background thread pre-reads Block $N+1$ into RAM. 
- **Result:** Out of the 215 seconds it takes to compute one step, the pure disk I/O waiting time is reduced to **~2 seconds (~1%)**. The SSD bottleneck is completely hidden behind the GPU compute.

### 4. AppleSilicon-FP8 Native Metal Acceleration
- PyTorch's MPS backend natively lacks support for some INT8 and FP8 operations, which normally forces a catastrophic fallback to the CPU.
- We utilize the `ComfyUI-AppleSilicon-FP8` custom node to inject **Bit-exact W8A8 Metal Kernels** and **W4A16 Fast Paths**. This forces all INT8/INT4 math to run natively on the Apple GPU, achieving >70% GPU utilization with almost 0% CPU bottleneck.

### 5. Sage Attention → Metal Flash (mtlflashattn)
- CUDA SageAttention cannot run on Apple Silicon. `./start.sh` enables ComfyUI `--use-sage-attention` via a small `vendor/sageattention` shim that calls `F.scaled_dot_product_attention` at runtime.
- AppleSilicon-FP8 then routes that SDPA call to **mtlflashattn** (online softmax, no Lq×Lk score matrix). Without the flag, Comfy stays on sub-quadratic attention and never hits the Metal flash kernels. MiniMax H3 at 864×480 / 5s is ~15k packed tokens with head_dim 128.

---

## 🚀 Quick Start Guide

### 1. Installation & Model Download
Clone this repository and run the setup scripts to automatically install dependencies, clone ComfyUI, and download the Hugging Face models.
```bash
git clone https://github.com/YOUR_USERNAME/minimax-h3-mac-optimizer.git
cd minimax-h3-mac-optimizer

./setup.sh
./download_models.sh          # Downloads ~42GB of T2V/I2V INT8 models
# ./download_models.sh --with-r2v   # Add this for R2V models (~21GB)
```

### 2. Start the Server
Start the dedicated ComfyUI instance. This script automatically applies the necessary memory flags (`--disable-smart-memory --fast-disk`) and disables the PyTorch MPS watermark (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`).
```bash
./start.sh
```

### 3. Run the Generation (Queue Scripts)
Since we are bypassing the standard ComfyUI frontend for extreme memory management, use the provided Python queue scripts to trigger generations in the background. Open a new terminal tab and run:

```bash
# 32B INT8 TE + INT8 DiT (Recommended for 24GB Streaming)
python3 queue_t2v_int8_te_blockstream.py

# 32B INT4 TE + INT4 DiT (layer-stream TE, full-load DiT ~11GB)
python3 queue_t2v_int4_te_dit.py

# 4B Krea-2 TE + INT8 DiT (Alternative lightweight encoder)
python3 queue_t2v_int8_blockstream.py

# INT4 DiT Full Load with 4B ClipProj TE
python3 queue_t2v_int4.py
```
- The final `.mp4` video will be saved in `ComfyUI/output/video/`.

---

## 📂 Model Weights Overview (Hugging Face)

| Filename | Directory | Purpose & Size |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | Main DiT (INT8, ~20GB) |
| `minimax_h3_fl2va_pruned_int4_convrot.safetensors` | `models/diffusion_models/` | Main DiT (INT4, ~11.3GB) |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | `models/text_encoders/` | 32B TE (INT4 ConvRot, Merserk, ~15GB). Layer-stream; do not full-load |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` | Lightweight 4B TE alternative |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | Video Decode |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | Audio Decode |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `models/loras/` | LightX2V 8-step Turbo LoRA (~1.8GB). Downloaded, not wired yet. |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors` | `models/loras/` | Same 8-step LoRA, pruned-compatible resize (~312MB). Use this with INT8 pruned DiT. |

---

## 🧪 Experiments (MBA M5 24GB)

All timings below are **864×480, 5s (124 frames), seed 42**, unless noted. Official MiniMax H3 wants **64GB+**; this log is what actually happened on 24GB unified memory. 24GB is the **whole machine**, not GPU VRAM — `.to("cpu")` does not free the pool.

### What worked

| | Setup | Result |
| --- | --- | --- |
| **INT4 full load** | ClipProj 4B TE + pruned INT4 ConvRot DiT (~10.8GB resident), watermark 0.0, W4A16 | **20/20 done.** ~173s/step, ~01:02 wall. No swap growth. |
| **INT8 DiT block-stream** | 4B TE + INT8 DiT, one transformer block at a time (`pread` + `F_NOCACHE` + prefetch) | DiT resident **1.5GB** instead of 20GB. ~190–225s/step, I/O ~2s. Do not drop RMSNorm (`q_norm`) or step 2 dies. |
| **INT8 32B TE layer-stream + INT8 DiT block-stream** | Official 32B INT8 TE (~27GB file), 50 LM layers streamed; then INT8 DiT block-stream | TE resident **2.6GB**, encode ~21s. DiT **1.5GB**. **20/20 + SaveVideo in 01:16:57.** This is the recommended 24GB path. |
| **Sage Attention → Metal flash** | CUDA SageAttention cannot install on Mac. `vendor/sageattention` shim + `--use-sage-attention` → `mtlflashattn` | Hook works. 2-step smoke: **193s / 212s** vs sub-quad **196s / 214s**. **No step-time win** — bottleneck is INT8 Linear / block-stream, not attention. Flag stays on for flash correctness at ~15k packed tokens. |

### What failed (do not retry)

| | Setup | Why it died |
| --- | --- | --- |
| 32B TE at 1344×768 | Full official encoder + higher canvas | Swap during load. Resolution does not shrink DiT **weights**. |
| 32B TE + Comfy `--lowvram` / `--fast-disk` | Hoped sequential offload | Darwin: `--fast-disk` is a no-op (CUDA/ROCm aimdo). `--lowvram` becomes `vram_state=SHARED` and still full-loads. |
| TE drop, then 20GB INT8 DiT **full load** | Disk-backed TE unload, DiT resident 19.996GB | First sampler step OOM (~20.3 / 20.4 GiB, needed +154 MiB). |
| ClipProj 4B + MPS watermark 90% / 98% | Shrink TE to ~5GB, raise Metal cap | DiT still 20GB. INT8 kernel OOM on first forward; fallback dies on leftover allocations. Apparent headroom is fragmentation. |
| Watermark 0.0 + 20GB INT8 DiT full load | Disable the PyTorch cap, INT8 Metal kernel on | GPU is real, but RAM 23.5/24GB + **swap 11–16GB**, disk 300–370 MB/s. Swap thrash for 30+ min at `0/20`. Kill it. |
| ClipProj `streaming` / `resident` pins | NVIDIA VRAM↔RAM trick | Unified memory: folding TE to “CPU” does not give the DiT 20GB of extra space. |
| `pip install sageattention` / Triton / `--use-ck-attention` | CUDA Sage / Kitchen INT8 attention | NVIDIA/HIP only. Bare `--use-sage-attention` without the vendor shim **exits Comfy at import**. |
| File-wide mmap of INT8 DiT | Avoid loading 20GB | One sampler step brings the working set back to ~20GB. Must stream **one block**, not the whole file. |

### In progress / next

- **INT4 32B TE + INT4 DiT:** `queue_t2v_int4_te_dit.py`. TE is community ConvRot (`Merserk`, 14952506624 bytes) with the same layer-stream hook as INT8. DiT full-loads (~10.8 GB); auto block-stream stays INT8-only. INT4 Metal kernel stays off (attempt 9). Header smoke passed (50 layers, ~233 MB/layer). Comfy 2-step encode+sample next.
- **LightX2V 8-step Turbo LoRA** is on disk (`models/loras/`). Not inserted into the queue graph yet. Expected: 20 steps → 8, ~1h12 → ~25–30 min if step time stays ~215s.
- INT4 Metal kernel (`ASFP8_INT4_EXT=1`) is a step-time experiment on the INT4 path only; it does not change the 24GB residency story.

## 💡 Future Optimizations (Speeding it up)
Currently, a 20-step generation takes about **1 hour and 12 minutes**. Sage Attention did not cut that. The remaining levers:
1. **8-step Turbo LoRA (downloaded):** LightX2V `minimax_h3_fl2v_turbo_8step_v1.0_*`. Start with the pruned-compatible 312MB file on INT8 DiT. Strength 1.0, 8 steps (4 also documented).
2. **Resolution/Frame Reduction:** For rapid prompt testing, reduce the video length to 2~3 seconds.
3. **INT4 DiT Utilization:** Fully loading the 11GB INT4 DiT into RAM (bypassing the stream) reduces step time to ~173s.

## 📌 References
- Official ComfyUI Guide: [minimax-h3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- Prompt Writing Guide: [MiniMax-H3 Docs](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
