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

# 4B Krea-2 TE + INT8 DiT (Alternative lightweight encoder)
python3 queue_t2v_int8_blockstream.py

# INT4 DiT Full Load (If testing the 11.3GB INT4 DiT directly in memory)
python3 queue_t2v_int4.py
```
- The final `.mp4` video will be saved in `ComfyUI/output/video/`.

---

## 📂 Model Weights Overview (Hugging Face)

| Filename | Directory | Purpose & Size |
| --- | --- | --- |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | Main DiT (INT8, ~20GB) |
| `minimax_h3_fl2va_pruned_int4_convrot.safetensors` | `models/diffusion_models/` | Main DiT (INT4, ~11.3GB) |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `models/text_encoders/` | Official 32B TE (INT8, ~27GB) |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` | Lightweight 4B TE alternative |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | Video Decode |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | Audio Decode |

## 💡 Future Optimizations (Speeding it up)
Currently, a 20-step generation takes about **1 hour and 12 minutes**. To reduce wall-clock time:
1. **Turbo / Spectrum LoRA:** Reduce the required steps from 20 to just 6, cutting the generation time down to ~20 minutes.
2. **Resolution/Frame Reduction:** For rapid prompt testing, reduce the video length to 2~3 seconds.
3. **INT4 DiT Utilization:** Fully loading the 11GB INT4 DiT into RAM (bypassing the stream) reduces step time to ~173s.

## 📌 References
- Official ComfyUI Guide: [minimax-h3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- Prompt Writing Guide: [MiniMax-H3 Docs](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
