"""Stream MiniMax H3 DiT blocks from safetensors instead of residing the whole file.

The 20GB pruned INT8 ConvRot checkpoint cannot stay in 24GB unified memory.
mmap of the whole file does not help: every sampler step touches all 50
blocks, so the working set becomes the full file (and MPS often copies mmap
pages into MTLBuffers anyway).

This loader keeps embeddings / token_refiner / final_layer resident (~1.6GB)
and, for each `DiTBlock.forward`, preads that block (~387MB, 18 tensors),
runs the quantized load path, computes, then drops the storage. Darwin
F_NOCACHE keeps the rest of the file out of the unified buffer cache.

h3.c `--ssd-streaming` is the same idea; Comfy `--lowvram` is not (MPS forces
VRAMState.SHARED and full-loads the UNet).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import struct
import threading
import time

import torch

_log = logging.getLogger("dit_block_stream")

_ST_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
    "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8),
    "I16": torch.int16,
    "U16": getattr(torch, "uint16", torch.int16),
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "I64": torch.int64,
    "U64": torch.int64,
    "F64": torch.float64,
}

# Darwin fcntl.F_NOCACHE; fall back to the numeric value if the name is missing.
_F_NOCACHE = getattr(__import__("fcntl", fromlist=["F_NOCACHE"]), "F_NOCACHE", 48)


def block_stream_mode() -> str:
    return os.environ.get("MINIMAX_DIT_BLOCK_STREAM", "auto").strip().lower()


def should_block_stream(unet_path: str) -> bool:
    mode = block_stream_mode()
    if mode in ("0", "off", "false", "no"):
        return False
    name = os.path.basename(unet_path).lower()
    if "minimax" not in name:
        return False
    if mode in ("1", "on", "true", "yes"):
        return True
    # auto: the ~20GB INT8 DiT. INT4 (~11GB) still full-loads.
    return "int8" in name


def _block_index(key: str, block_prefix: str):
    if not key.startswith(block_prefix):
        return None
    rest = key[len(block_prefix) :]
    idx_s, _, _ = rest.partition(".")
    try:
        return int(idx_s)
    except ValueError:
        return None


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


class _HeaderTensor:
    """Shape/dtype stand-in so Comfy can detect MiniMax H3 without loading 20GB."""

    def __init__(self, shape, dtype):
        self.shape = torch.Size(shape)
        self.dtype = dtype

    def nelement(self):
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    numel = nelement


class SafetensorsStore:
    """pread tensors from a safetensors file without mmaping the payload."""

    def __init__(self, path: str, block_prefix: str = "blocks.", log_tag: str = "dit-block-stream"):
        self.path = os.path.abspath(path)
        self.block_prefix = block_prefix
        self.log_tag = log_tag
        self._fd = os.open(self.path, os.O_RDONLY)
        try:
            import fcntl

            fcntl.fcntl(self._fd, _F_NOCACHE, 1)
        except Exception as exc:
            print(f"[{log_tag}] F_NOCACHE not set: {exc}", flush=True)
        header_len = struct.unpack("<Q", os.read(self._fd, 8))[0]
        header = json.loads(os.read(self._fd, header_len))
        self.metadata = header.pop("__metadata__", None)
        self.header = header
        self.data_start = 8 + header_len
        self.block_keys = {}
        self.resident_keys = []
        for key, info in header.items():
            idx = _block_index(key, block_prefix)
            if idx is not None:
                self.block_keys.setdefault(idx, []).append(key)
            else:
                self.resident_keys.append(key)
        self.n_blocks = len(self.block_keys)
        for keys in self.block_keys.values():
            keys.sort()

    def close(self):
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def block_nbytes(self, index: int) -> int:
        total = 0
        for key in self.block_keys.get(index, ()):
            start, end = self.header[key]["data_offsets"]
            total += end - start
        return total

    def header_state_dict(self) -> dict:
        sd = {}
        for key, info in self.header.items():
            dtype = _ST_TO_TORCH.get(info["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported safetensors dtype {info['dtype']} for {key}")
            sd[key] = _HeaderTensor(info["shape"], dtype)
        return sd

    def read_tensor(self, key: str, device=None) -> torch.Tensor:
        info = self.header[key]
        dtype = _ST_TO_TORCH[info["dtype"]]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        nbytes = end - start
        raw = os.pread(self._fd, nbytes, self.data_start + start)
        if len(raw) != nbytes:
            raise IOError(f"short pread for {key}: {len(raw)}/{nbytes}")
        cpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        if nbytes == 0:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            tensor = cpu.view(dtype).reshape(shape).contiguous().clone()
        del cpu, raw
        if device is not None and torch.device(device).type != "cpu":
            tensor = tensor.to(device, copy=True)
        return tensor

    def load_resident(self, device="cpu") -> dict:
        return {key: self.read_tensor(key, device=device) for key in self.resident_keys}

    def read_block(self, index: int, device="cpu") -> dict:
        prefix = f"{self.block_prefix}{index}."
        sd = {}
        for key in self.block_keys[index]:
            sd[key[len(prefix) :]] = self.read_tensor(key, device=device)
        return sd


def _set_factory_device(module, device):
    for child in module.modules():
        factory = getattr(child, "factory_kwargs", None)
        if isinstance(factory, dict):
            factory["device"] = device


def _param_nbytes(weight) -> int:
    data = getattr(weight, "data", weight)
    qdata = getattr(data, "_qdata", None)
    if torch.is_tensor(qdata):
        return int(qdata.nbytes)
    if torch.is_tensor(data):
        return int(data.nbytes)
    return 0


def _drop_module_storage(module):
    """Drop large Linear weights only. RMSNorm must stay registered so the
    next load_state_dict can fill it; q_norm.weight is 256 bytes anyway.
    """
    min_drop = 4 * 1024 * 1024
    for child in module.modules():
        weight = getattr(child, "weight", None)
        if weight is None:
            continue
        if _param_nbytes(weight) < min_drop:
            continue
        data = getattr(weight, "data", weight)
        qdata = getattr(data, "_qdata", None)
        if qdata is not None:
            try:
                data._qdata = torch.empty(0)
            except Exception:
                pass
        params = getattr(data, "_params", None)
        if params is not None:
            for name, value in list(vars(params).items()):
                if torch.is_tensor(value):
                    try:
                        setattr(params, name, None)
                    except Exception:
                        pass
        try:
            child.weight = None
        except Exception:
            try:
                weight.data = torch.empty(0, device="cpu")
            except Exception:
                pass


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "off", "false", "no")


class BlockStreamer:
    def __init__(self, store: SafetensorsStore, log_tag: str | None = None, prefetch: bool | None = None):
        self.store = store
        self.log_tag = log_tag or store.log_tag
        self.n_blocks = store.n_blocks
        if prefetch is None:
            prefetch = _env_flag("MINIMAX_DIT_BLOCK_PREFETCH", "1")
        self.prefetch = prefetch
        self._lock = threading.Lock()
        self._thread = None
        self._cpu_next = None
        self._step = 0
        self._io = 0.0
        self._t0 = 0.0
        self._logged_first = False

    def _join_prefetch(self):
        thread = self._thread
        if thread is not None:
            thread.join()
            self._thread = None

    def _start_prefetch(self, index: int):
        if not self.prefetch or index >= self.n_blocks:
            return

        def work():
            sd = self.store.read_block(index, device="cpu")
            with self._lock:
                prev = self._cpu_next
                self._cpu_next = (index, sd)
            del prev

        self._thread = threading.Thread(target=work, daemon=True, name=f"{self.log_tag}-prefetch-{index}")
        self._thread.start()

    def _pop_prefetched(self, index: int):
        self._join_prefetch()
        with self._lock:
            cached = self._cpu_next
            if cached is not None and cached[0] == index:
                self._cpu_next = None
                return cached[1]
        return None

    def _load_into_block(self, block, sd: dict, device):
        _set_factory_device(block, device)
        moved = {}
        for key, tensor in sd.items():
            # comfy_quant is a JSON blob; ops.py calls .numpy() on it.
            if key.endswith("comfy_quant"):
                moved[key] = tensor.cpu() if tensor.device.type != "cpu" else tensor
            elif tensor.device != device:
                moved[key] = tensor.to(device, copy=True)
            else:
                moved[key] = tensor
        sd.clear()
        block.load_state_dict(moved, strict=False, assign=True)
        del moved

    def before_block(self, index: int, block, device):
        if index == 0:
            self._step += 1
            self._t0 = time.perf_counter()
            self._io = 0.0
        t0 = time.perf_counter()
        sd = self._pop_prefetched(index)
        if sd is None:
            sd = self.store.read_block(index, device="cpu")
        self._load_into_block(block, sd, torch.device(device))
        del sd
        self._io += time.perf_counter() - t0
        if not self._logged_first:
            mb = self.store.block_nbytes(index) / (1024 * 1024)
            print(
                f"[{self.log_tag}] block 0 materialized ({mb:.0f} MB); "
                f"{self.n_blocks} blocks, F_NOCACHE, prefetch="
                f"{'on' if self.prefetch else 'off'}",
                flush=True,
            )
            self._logged_first = True
        self._start_prefetch(index + 1)

    def after_block(self, index: int, block):
        _drop_module_storage(block)
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        if index + 1 == self.n_blocks:
            dt = time.perf_counter() - self._t0
            print(
                f"[{self.log_tag}] step {self._step}: {self.n_blocks} blocks in {dt:.1f}s "
                f"(io {self._io:.1f}s)",
                flush=True,
            )
            _empty_mps()


def attach_modules(modules, store: SafetensorsStore, holder=None, log_tag: str | None = None,
                   model_label: str = "blocks", prefetch: bool | None = None) -> BlockStreamer:
    """Wrap each module.forward so weights are pread, computed, then dropped.

    `x` may be positional (DiTBlock) or a kwarg (TransformerBlock).
    """
    streamer = BlockStreamer(store, log_tag=log_tag, prefetch=prefetch)
    tag = streamer.log_tag
    device_fallback = None
    try:
        import comfy.model_management as mm

        device_fallback = mm.get_torch_device()
    except Exception:
        device_fallback = torch.device("cpu")

    for index, block in enumerate(modules):
        orig = block.forward

        def _forward(*args, _orig=orig, _idx=index, _block=block, **kwargs):
            x = args[0] if args else kwargs.get("x")
            device = x.device if hasattr(x, "device") else device_fallback
            streamer.before_block(_idx, _block, device)
            try:
                return _orig(*args, **kwargs)
            finally:
                streamer.after_block(_idx, _block)

        block.forward = _forward

    if holder is not None:
        holder._minimax_block_stream = streamer
    resident_mb = sum(
        store.header[k]["data_offsets"][1] - store.header[k]["data_offsets"][0]
        for k in store.resident_keys
    ) / (1024 * 1024)
    block_mb = store.block_nbytes(0) / (1024 * 1024) if store.n_blocks else 0.0
    print(
        f"[{tag}] {os.path.basename(store.path)}: "
        f"{store.n_blocks} {model_label} streamed ({block_mb:.0f} MB each), "
        f"resident {len(store.resident_keys)} tensors ({resident_mb:.0f} MB)",
        flush=True,
    )
    if streamer.prefetch and store.n_blocks > 1:
        streamer._start_prefetch(0)
    return streamer


def attach_block_stream(dit_model, store: SafetensorsStore) -> BlockStreamer:
    return attach_modules(
        dit_model.blocks,
        store,
        holder=dit_model,
        log_tag="dit-block-stream",
        model_label="DiT blocks",
    )


def load_streamed_diffusion_model(unet_path, model_options=None, disable_dynamic=False):
    """UNET loader that never materializes `blocks.*` until each DiTBlock.forward."""
    import comfy.model_detection as model_detection
    import comfy.model_management as model_management
    import comfy.model_patcher
    import comfy.utils

    model_options = {} if model_options is None else dict(model_options)
    store = SafetensorsStore(unet_path)
    metadata = store.metadata
    detect_sd = store.header_state_dict()
    resident_sd = store.load_resident(device="cpu")
    detect_sd.update(resident_sd)

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        detect_sd, metadata = comfy.utils.convert_old_quants(detect_sd, "", metadata=metadata)
        resident_sd, metadata = comfy.utils.convert_old_quants(resident_sd, "", metadata=metadata)

    dtype = model_options.get("dtype", None)
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(detect_sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        detect_sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        detect_sd = temp_sd
        resident_sd = comfy.utils.state_dict_prefix_replace(
            resident_sd, {diffusion_model_prefix: ""}, filter_keys=True
        ) or resident_sd
        if custom_operations is None:
            detect_sd, metadata = comfy.utils.convert_old_quants(detect_sd, "", metadata=metadata)
            resident_sd, metadata = comfy.utils.convert_old_quants(resident_sd, "", metadata=metadata)

    parameters = comfy.utils.calculate_parameters(resident_sd)
    weight_dtype = comfy.utils.weight_dtype(resident_sd)

    load_device = model_options.get("load_device", model_management.get_torch_device())
    model_config = model_detection.model_config_from_unet(detect_sd, "", metadata=metadata)
    if model_config is None:
        store.close()
        raise RuntimeError(f"ERROR: Could not detect MiniMax H3 from {unet_path}")

    new_sd = detect_sd
    offload_device = model_options.get("offload_device", model_management.unet_offload_device())
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype
        )
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes
        )
    else:
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes
        )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    Patcher = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    model_patcher = Patcher(model, load_device=load_device, offload_device=offload_device)
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)

    to_load = model.model_config.process_unet_state_dict(dict(resident_sd))

    class _SkipBlockMissing(logging.Filter):
        def filter(self, record):
            return "Missing weight for layer blocks." not in record.getMessage()

    _filt = _SkipBlockMissing()
    logging.getLogger().addFilter(_filt)
    try:
        missing, unexpected = model.diffusion_model.load_state_dict(
            to_load, strict=False, assign=model_patcher.is_dynamic()
        )
    finally:
        logging.getLogger().removeFilter(_filt)
    block_missing = [key for key in missing if key.startswith("blocks.")]
    other_missing = [key for key in missing if not key.startswith("blocks.")]
    print(
        f"[dit-block-stream] loaded {len(to_load)} resident tensors; "
        f"deferred {len(block_missing)} block tensors to per-layer pread",
        flush=True,
    )
    if other_missing:
        logging.warning("unet missing (non-block): %s", other_missing)
    if unexpected:
        logging.warning("unet unexpected: %s", unexpected)
    del detect_sd, resident_sd, new_sd, to_load

    attach_block_stream(model.diffusion_model, store)
    model_patcher.cached_patcher_init = (
        load_streamed_diffusion_model,
        (unet_path, model_options),
    )
    return model_patcher


def install_load_hook():
    import comfy.sd as sd_mod

    if getattr(sd_mod.load_diffusion_model, "_minimax_block_stream", False):
        return
    orig = sd_mod.load_diffusion_model

    def load_diffusion_model(unet_path, model_options={}, disable_dynamic=False):
        if should_block_stream(unet_path):
            patcher = load_streamed_diffusion_model(
                unet_path, model_options=model_options, disable_dynamic=disable_dynamic
            )
            patcher.cached_patcher_init = (sd_mod.load_diffusion_model, (unet_path, model_options))
            return patcher
        return orig(unet_path, model_options=model_options, disable_dynamic=disable_dynamic)

    load_diffusion_model._minimax_block_stream = True
    sd_mod.load_diffusion_model = load_diffusion_model
    print(
        f"[dit-block-stream] load_diffusion_model hooked "
        f"(MINIMAX_DIT_BLOCK_STREAM={block_stream_mode() or 'auto'})",
        flush=True,
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/eunsung/minimax/ComfyUI/models/diffusion_models/"
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )
    store = SafetensorsStore(path)
    print(f"file={store.path}")
    print(f"blocks={store.n_blocks} resident={len(store.resident_keys)}")
    print(f"block0_mb={store.block_nbytes(0) / (1024 * 1024):.1f}")
    t0 = time.perf_counter()
    sd = store.read_block(0, device="cpu")
    dt = time.perf_counter() - t0
    qkv = sd["attn.qkv_proj.weight"]
    print(f"pread_block0={dt:.3f}s qkv={tuple(qkv.shape)} {qkv.dtype} {qkv.nbytes / 1e6:.1f} MB")
    store.close()
