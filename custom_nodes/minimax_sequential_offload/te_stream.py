"""Stream MiniMax H3 32B INT8 TE layers from safetensors.

The official INT8 ConvRot encoder is ~27GB. 24GB unified memory cannot
reside it. Same idea as DiT block-stream: keep embeddings + vision tower
resident (~2GB) and, for each TransformerBlock.forward, pread that LM
layer, compute, then drop the large Linear weights.

T2V does not use the vision tower; it still sits in RAM (~1.1GB BF16).
After encode, sequential_offload drops the whole TE before DiT runs.
"""

from __future__ import annotations

import logging
import os

try:
    from .block_stream import (
        SafetensorsStore,
        _env_flag,
        attach_modules,
    )
except ImportError:
    from block_stream import (
        SafetensorsStore,
        _env_flag,
        attach_modules,
    )

_log = logging.getLogger("te_block_stream")
_TE_PREFIX = "model.layers."
_LOG = "te-block-stream"


def te_block_stream_mode() -> str:
    return os.environ.get("MINIMAX_TE_BLOCK_STREAM", "auto").strip().lower()


def should_te_block_stream(clip_path: str) -> bool:
    mode = te_block_stream_mode()
    if mode in ("0", "off", "false", "no"):
        return False
    name = os.path.basename(clip_path).lower()
    if "qwen3vl_4b" in name:
        return False
    if "qwen3vl_32b" not in name and "minimax_h3" not in name:
        return False
    if mode in ("1", "on", "true", "yes"):
        return True
    # auto: official 32B INT8 ConvRot. NVFP4 (~15GB) still full-loads.
    return "int8" in name


def _te_layers(clip):
    inner = clip.cond_stage_model
    name = getattr(inner, "clip_name", None) or getattr(inner, "clip", "qwen3vl_32b")
    clip_mod = getattr(inner, name, None)
    if clip_mod is None:
        raise RuntimeError(f"TE stream: no submodule {name!r} on {type(inner).__name__}")
    return clip_mod.transformer.model.layers


def load_streamed_clip(
    ckpt_paths,
    embedding_directory=None,
    clip_type=None,
    model_options=None,
    disable_dynamic=False,
):
    """CLIP loader that never materializes `model.layers.*` until each block.forward."""
    import comfy.sd as sd_mod
    import comfy.text_encoders.long_clipl
    import comfy.text_encoders.minimax
    import comfy.utils
    from comfy.sd import CLIP, TEModel, detect_te_model, llama_detect

    if clip_type is None:
        clip_type = sd_mod.CLIPType.MINIMAX
    model_options = {} if model_options is None else dict(model_options)

    path = ckpt_paths[0]
    store = SafetensorsStore(path, block_prefix=_TE_PREFIX, log_tag=_LOG)
    metadata = store.metadata
    detect_sd = store.header_state_dict()
    resident_sd = store.load_resident(device="cpu")
    detect_sd.update(resident_sd)

    if model_options.get("custom_operations", None) is None:
        detect_sd, metadata = comfy.utils.convert_old_quants(detect_sd, "", metadata=metadata)
        resident_sd, metadata = comfy.utils.convert_old_quants(resident_sd, "", metadata=metadata)

    te_model = detect_te_model(detect_sd)
    if te_model != TEModel.QWEN3VL_32B:
        store.close()
        raise RuntimeError(
            f"TE stream expected QWEN3VL_32B, got {te_model} from {os.path.basename(path)}"
        )

    class _Target:
        params = {}
        clip = comfy.text_encoders.minimax.te(**llama_detect([detect_sd]))
        tokenizer = comfy.text_encoders.minimax.MiniMaxH3Tokenizer

    parameters = comfy.utils.calculate_parameters(resident_sd)
    tokenizer_data = {}
    tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(
        resident_sd, tokenizer_data, model_options
    )

    class _SkipLayerMissing(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if "clip missing:" not in msg:
                return True
            return "model.layers." not in msg

    filt = _SkipLayerMissing()
    logging.getLogger().addFilter(filt)
    try:
        clip = CLIP(
            _Target,
            embedding_directory=embedding_directory,
            parameters=parameters,
            tokenizer_data=tokenizer_data,
            state_dict=[resident_sd],
            model_options=model_options,
            disable_dynamic=disable_dynamic,
        )
    finally:
        logging.getLogger().removeFilter(filt)

    missing_n = sum(len(store.block_keys.get(i, ())) for i in range(store.n_blocks))
    print(
        f"[{_LOG}] loaded {len(resident_sd)} resident tensors; "
        f"deferred {missing_n} layer tensors to per-layer pread",
        flush=True,
    )
    del detect_sd, resident_sd

    prefetch = _env_flag(
        "MINIMAX_TE_BLOCK_PREFETCH",
        os.environ.get("MINIMAX_DIT_BLOCK_PREFETCH", "1"),
    )
    attach_modules(
        _te_layers(clip),
        store,
        holder=clip.cond_stage_model,
        log_tag=_LOG,
        model_label="TE layers",
        prefetch=prefetch,
    )
    clip.patcher.cached_patcher_init = (
        sd_mod.load_clip_model_patcher,
        (ckpt_paths, embedding_directory, clip_type, model_options),
    )
    return clip


def install_te_load_hook():
    import comfy.sd as sd_mod

    if getattr(sd_mod.load_clip, "_minimax_te_block_stream", False):
        return
    orig = sd_mod.load_clip

    def load_clip(
        ckpt_paths,
        embedding_directory=None,
        clip_type=None,
        model_options={},
        disable_dynamic=False,
    ):
        if clip_type is None:
            clip_type = sd_mod.CLIPType.STABLE_DIFFUSION
        if len(ckpt_paths) == 1 and should_te_block_stream(ckpt_paths[0]):
            return load_streamed_clip(
                ckpt_paths,
                embedding_directory=embedding_directory,
                clip_type=clip_type,
                model_options=model_options,
                disable_dynamic=disable_dynamic,
            )
        return orig(
            ckpt_paths,
            embedding_directory=embedding_directory,
            clip_type=clip_type,
            model_options=model_options,
            disable_dynamic=disable_dynamic,
        )

    load_clip._minimax_te_block_stream = True
    sd_mod.load_clip = load_clip
    print(
        f"[{_LOG}] load_clip hooked (MINIMAX_TE_BLOCK_STREAM={te_block_stream_mode() or 'auto'})",
        flush=True,
    )


if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/eunsung/minimax/ComfyUI/models/text_encoders/"
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )
    store = SafetensorsStore(path, block_prefix=_TE_PREFIX, log_tag=_LOG)
    print(f"file={store.path}")
    print(f"layers={store.n_blocks} resident={len(store.resident_keys)}")
    print(f"layer0_mb={store.block_nbytes(0) / (1024 * 1024):.1f}")
    t0 = time.perf_counter()
    sd = store.read_block(0, device="cpu")
    dt = time.perf_counter() - t0
    q = sd["self_attn.q_proj.weight"]
    print(
        f"pread_layer0={dt:.3f}s q_proj={tuple(q.shape)} {q.dtype} {q.nbytes / 1e6:.1f} MB "
        f"keys={len(sd)}"
    )
    store.close()
