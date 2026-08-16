"""Disk-backed sequential residency for MiniMax H3 on 24GB unified memory.

ComfyUI --fast-disk is a DynamicVRAM/aimdo flag (NVIDIA/AMD, Windows/Linux).
On Apple Silicon that path never enables, and `.to("cpu")` does not free the
unified pool. This node emulates fast-disk:

  TE encode -> drop TE module (reload later from safetensors)
  DiT sample -> drop DiT module
  VAE decode -> drop VAE module

Weights come back from disk on the next load via cached_patcher_init / load_clip.

INT8 DiT (~20GB) additionally streams one transformer block at a time
(see block_stream.py). INT8 32B TE (~27GB) streams one LM layer at a time
(see te_stream.py). That is the original 24GB plan: not mmap, not INT4.
"""

from __future__ import annotations

import gc
import logging

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_log = logging.getLogger("sequential_offload")
_installed = False
_clipproj_hooked = False
_clipproj_skip_logged = False


def _model_name(loaded) -> str:
    patcher = loaded.model
    if patcher is None:
        return "?"
    inner = getattr(patcher, "model", patcher)
    return type(inner).__name__


def _empty_mps():
    gc.collect()
    if hasattr(torch, "backends") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


def _release_clip_weights(clip) -> None:
    if getattr(clip, "_minimax_disk_offloaded", False):
        return
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        return
    init = getattr(patcher, "cached_patcher_init", None)
    if init is None:
        print("[fast-disk] CLIP has no cached_patcher_init; skip", flush=True)
        return

    import comfy.model_management as mm

    try:
        mm.unload_model_and_clones(patcher)
    except Exception as exc:
        print(f"[fast-disk] CLIP unload: {exc}", flush=True)
        try:
            mm.unload_all_models()
        except Exception:
            pass

    clip._minimax_reload = init
    inner = clip.cond_stage_model
    clip.cond_stage_model = None
    try:
        patcher.model = None
    except Exception:
        pass
    clip.patcher = None
    clip._minimax_disk_offloaded = True
    del inner
    _empty_mps()
    print("[fast-disk] TE dropped; next encode reloads from disk", flush=True)


def _ensure_clip_weights(clip) -> None:
    if not getattr(clip, "_minimax_disk_offloaded", False):
        return
    from comfy.sd import load_clip

    init = clip._minimax_reload
    print("[fast-disk] reloading TE from disk...", flush=True)
    new_clip = load_clip(*init[1])
    clip.cond_stage_model = new_clip.cond_stage_model
    clip.patcher = new_clip.patcher
    clip._minimax_disk_offloaded = False


def _release_patcher_weights(patcher, label: str = "") -> None:
    if patcher is None or getattr(patcher, "_minimax_disk_offloaded", False):
        return
    inner = getattr(patcher, "model", None)
    name = type(inner).__name__ if inner is not None else (label or "model")

    import comfy.model_management as mm

    try:
        mm.unload_model_and_clones(patcher)
    except Exception as exc:
        print(f"[fast-disk] unload {name}: {exc}", flush=True)

    if inner is None:
        return

    try:
        inner.to("meta")
        patcher._minimax_disk_offloaded = True
        print(f"[fast-disk] {name} released to meta (disk-backed)", flush=True)
    except Exception as exc:
        print(f"[fast-disk] meta {name} failed: {exc}", flush=True)
        try:
            for param in inner.parameters():
                param.grad = None
                param.data = torch.empty(0)
            patcher._minimax_disk_offloaded = True
            print(f"[fast-disk] {name} parameter storage dropped", flush=True)
        except Exception as exc2:
            print(f"[fast-disk] drop {name} failed: {exc2}", flush=True)
    _empty_mps()


def _ensure_patcher_weights(patcher) -> None:
    if patcher is None or not getattr(patcher, "_minimax_disk_offloaded", False):
        return
    init = getattr(patcher, "cached_patcher_init", None)
    if init is None:
        print("[fast-disk] cannot reload: no cached_patcher_init", flush=True)
        patcher._minimax_disk_offloaded = False
        return

    print("[fast-disk] reloading weights from disk...", flush=True)
    try:
        new_patcher = init[0](*init[1], disable_dynamic=True)
    except TypeError:
        new_patcher = init[0](*init[1])
    if len(init) > 2:
        new_patcher = new_patcher[init[2]]
    patcher.model = new_patcher.model if hasattr(new_patcher, "model") else new_patcher
    patcher._minimax_disk_offloaded = False


def _install_vae_hooks() -> None:
    from comfy.sd import VAE

    orig_decode = VAE.decode
    orig_encode = VAE.encode

    def _sync_vae(self):
        _ensure_patcher_weights(self.patcher)
        inner = getattr(self.patcher, "model", None) if self.patcher is not None else None
        if inner is not None:
            self.first_stage_model = inner

    def decode(self, *args, **kwargs):
        _sync_vae(self)
        return orig_decode(self, *args, **kwargs)

    def encode(self, *args, **kwargs):
        _sync_vae(self)
        return orig_encode(self, *args, **kwargs)

    VAE.decode = decode
    VAE.encode = encode


def _install_clip_hooks() -> None:
    from comfy.sd import CLIP

    orig_enc = CLIP.encode_from_tokens
    orig_sched = CLIP.encode_from_tokens_scheduled
    orig_gen = CLIP.generate

    def encode_from_tokens(self, *args, **kwargs):
        _ensure_clip_weights(self)
        try:
            return orig_enc(self, *args, **kwargs)
        finally:
            _release_clip_weights(self)

    def encode_from_tokens_scheduled(self, *args, **kwargs):
        _ensure_clip_weights(self)
        try:
            return orig_sched(self, *args, **kwargs)
        finally:
            _release_clip_weights(self)

    def generate(self, *args, **kwargs):
        _ensure_clip_weights(self)
        try:
            return orig_gen(self, *args, **kwargs)
        finally:
            _release_clip_weights(self)

    CLIP.encode_from_tokens = encode_from_tokens
    CLIP.encode_from_tokens_scheduled = encode_from_tokens_scheduled
    CLIP.generate = generate


def _install_clipproj_hooks() -> None:
    """ProjectedCLIP encodes via _encode(), not comfy.sd.CLIP; drop TE after.

    ComfyUI-ClipProj may import after this node, so this is retried from
    load_models_gpu until ProjectedCLIP is actually in sys.modules.
    """
    global _clipproj_hooked, _clipproj_skip_logged
    if _clipproj_hooked:
        return

    import inspect
    import sys

    ProjectedCLIP = None
    for mod in list(sys.modules.values()):
        cls = getattr(mod, "ProjectedCLIP", None)
        if not inspect.isclass(cls):
            continue
        if cls.__name__ != "ProjectedCLIP":
            continue
        if "clipproj" not in getattr(cls, "__module__", ""):
            continue
        ProjectedCLIP = cls
        break
    if ProjectedCLIP is None:
        if not _clipproj_skip_logged:
            print("[fast-disk] ClipProj hook deferred until ProjectedCLIP imports", flush=True)
            _clipproj_skip_logged = True
        return

    orig_encode = ProjectedCLIP._encode

    def _encode(self, *args, **kwargs):
        try:
            return orig_encode(self, *args, **kwargs)
        finally:
            cache = self.__dict__.get("_gpu")
            if isinstance(cache, dict):
                cache.clear()
            _release_clip_weights(self._base)

    ProjectedCLIP._encode = _encode
    _clipproj_hooked = True
    print("[fast-disk] ClipProj ProjectedCLIP will drop TE after encode", flush=True)


def _install_loader_hook() -> None:
    global _installed
    if _installed:
        return

    import comfy.model_management as mm

    orig = mm.load_models_gpu

    def load_models_gpu_sequential(
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        _install_clipproj_hooks()

        wanted = []
        for model in models:
            wanted.append(model)
            if hasattr(model, "model_patches_models"):
                wanted.extend(model.model_patches_models())

        keep = []
        for loaded in list(mm.current_loaded_models):
            if loaded.model in wanted:
                keep.append(loaded)
            else:
                _release_patcher_weights(loaded.model, _model_name(loaded))

        prev = mm.DISABLE_SMART_MEMORY
        mm.DISABLE_SMART_MEMORY = True
        try:
            device = mm.get_torch_device()
            if not mm.is_device_cpu(device):
                unloaded = mm.free_memory(1e30, device, keep_loaded=keep)
                if unloaded:
                    names = ", ".join(_model_name(item) for item in unloaded)
                    _log.info("offloaded before next load: %s", names)
                    print(f"[sequential-offload] offloaded: {names}", flush=True)
        finally:
            mm.DISABLE_SMART_MEMORY = prev

        for model in wanted:
            _ensure_patcher_weights(model)

        _empty_mps()
        return orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            force_full_load=force_full_load,
        )

    mm.load_models_gpu = load_models_gpu_sequential
    try:
        from .block_stream import install_load_hook
        from .te_stream import install_te_load_hook
    except ImportError:
        from block_stream import install_load_hook
        from te_stream import install_te_load_hook

    try:
        install_load_hook()
    except Exception as exc:
        print(f"[dit-block-stream] hook failed: {exc}", flush=True)
    try:
        install_te_load_hook()
    except Exception as exc:
        print(f"[te-block-stream] hook failed: {exc}", flush=True)
    _install_clip_hooks()
    _install_vae_hooks()
    try:
        _install_clipproj_hooks()
    except Exception as exc:
        print(f"[fast-disk] ClipProj hook failed: {exc}", flush=True)
    _installed = True
    print(
        "[fast-disk] Mac disk-backed sequential offload: "
        "TE encode (INT8 32B: one layer) -> drop -> DiT (INT8: one block) -> drop -> VAE",
        flush=True,
    )


class AnyType(str):
    def __ne__(self, other):
        return False


any_typ = AnyType("*")


class UnloadAllModels:
    """Pass-through that drops every loaded model before the next node runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (any_typ,)}}

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    FUNCTION = "unload"
    CATEGORY = "minimax"
    DESCRIPTION = "Unload all models (text encoder / DiT / VAE) then pass the input through."

    def unload(self, value):
        import comfy.model_management as mm

        for loaded in list(mm.current_loaded_models):
            _release_patcher_weights(loaded.model, _model_name(loaded))
        mm.unload_all_models()
        _empty_mps()
        print("[fast-disk] UnloadAllModels: all models dropped", flush=True)
        return (value,)


try:
    _install_loader_hook()
    NODE_CLASS_MAPPINGS["UnloadAllModels"] = UnloadAllModels
    NODE_DISPLAY_NAME_MAPPINGS["UnloadAllModels"] = "Unload All Models"
except Exception as exc:
    print(f"[sequential-offload] hook failed: {exc}", flush=True)
