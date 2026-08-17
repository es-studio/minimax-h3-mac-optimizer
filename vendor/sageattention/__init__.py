"""MPS stand-in for thu-ml SageAttention.

CUDA SageAttention (`pip install sageattention`) is NVIDIA-only: Triton/CUDA
kernels, no MPS backend. ComfyUI's `--use-sage-attention` does
`from sageattention import sageattn` at import time and exits if that fails.

This package is put on PYTHONPATH by start.sh so the flag works on Apple
Silicon. The call is forwarded to F.scaled_dot_product_attention at runtime
so AppleSilicon-FP8's mtlflashattn patch (Metal flash, head_dim <= 128) is
used. That is the Mac equivalent of Sage/Flash: online softmax, no Lq×Lk
score matrix.

MiniMax H3 T2V 864×480 5s is ~15k packed tokens, head_dim 128, so the
mtlflashattn correctness gate (>=4096) fires. Stock MPS fused SDPA would be
numerically wrong at that length and would try to materialize ~25GB of scores.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["sageattn"]

_logged = False


def sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    sm_scale=None,
    smooth_k=False,
    attn_mask=None,
    **_kwargs,
):
    """Comfy `attention_sage` API. Layouts: HND = (B,H,S,D), NHD = (B,S,H,D)."""
    global _logged
    if tensor_layout == "NHD":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    elif tensor_layout != "HND":
        raise ValueError(f"unsupported tensor_layout={tensor_layout!r}")

    if not _logged:
        _logged = True
        print(
            f"[sage-mps] sageattn -> F.sdpa/mtlflashattn layout={tensor_layout} "
            f"q={tuple(q.shape)} dtype={q.dtype} device={q.device}",
            flush=True,
        )

    extra = {}
    if sm_scale is not None:
        extra["scale"] = sm_scale
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, **extra
    )
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return out
