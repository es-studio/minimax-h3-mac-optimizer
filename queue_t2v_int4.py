#!/usr/bin/env python3
"""Queue MiniMax H3 T2V with ClipProj 4B TE + pruned INT4 ConvRot DiT.

Same 864x480 / 5s / 20 steps as the INT8 ClipProj run. The DiT file is
~11.3 GB instead of ~20 GB.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
OUT = Path("/Users/eunsung/minimax/t2v_int4.json")
DIT = "minimax_h3_fl2va_pruned_int4_convrot.safetensors"

PROMPT_TEXT = (
    "Cinematic widescreen shot of a quiet rainy city street at dusk, "
    "neon reflections on wet asphalt, shallow depth of field, film grain. "
    "Camera slowly dollies forward. Audio: rain and distant traffic. "
    "No text, logos or watermarks."
)

prompt = {
    "6": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": DIT,
            "weight_dtype": "default",
        },
    },
    "13": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
            "type": "krea2",
            "device": "default",
        },
    },
    "13b": {
        "class_type": "ClipProjApply",
        "inputs": {
            "clip": ["13", 0],
            "projection": "mmh3-4b-ClipProj-v3-mlp.safetensors",
        },
    },
    "11": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
    },
    "24": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
    },
    "104": {
        "class_type": "MiniMaxH3ImageToVideo",
        "inputs": {
            "clip": ["13b", 0],
            "vae": ["11", 0],
            "prompt": PROMPT_TEXT,
            "width": 864,
            "height": 480,
            "length": 124,
        },
    },
    "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    "9": {
        "class_type": "BasicScheduler",
        "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0},
    },
    "16": {
        "class_type": "BasicGuider",
        "inputs": {"model": ["6", 0], "conditioning": ["104", 0]},
    },
    "14": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["15", 0],
            "guider": ["16", 0],
            "sampler": ["17", 0],
            "sigmas": ["9", 0],
            "latent_image": ["104", 1],
        },
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["14", 0], "vae": ["11", 0]},
    },
    "23": {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["14", 0], "vae": ["24", 0]},
    },
    "12": {
        "class_type": "CreateVideo",
        "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["23", 0], "bit_depth": 8},
    },
    "92": {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["12", 0],
            "filename_prefix": "video/MiniMax_H3_T2V_int4_864x480_5s",
            "format": "auto",
            "codec": "auto",
        },
    },
}


def get(path: str):
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=30) as r:
        return json.loads(r.read().decode())


def post(path: str, data: dict):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{SERVER}{path}", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> None:
    queued = post("/prompt", {"prompt": prompt})
    print(json.dumps(queued, indent=2), flush=True)
    OUT.write_text(
        json.dumps(
            {
                "prompt_id": queued["prompt_id"],
                "queued_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_epoch": time.time(),
                "settings": "T2V 864x480 5s 20steps ClipProj 4B + INT4 ConvRot",
                "dit": DIT,
                "node_errors": queued.get("node_errors") or {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
