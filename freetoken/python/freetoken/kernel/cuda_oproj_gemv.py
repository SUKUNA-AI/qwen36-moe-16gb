"""Opt-in bit-exact decode out/o-projection (tensor-wise FP8 W8A8, N=2048, K=4096, 1 <= M <= 8).

What it replaces, and nothing else: the uniform-scale ``torch._scaled_mm`` call of the decode
out-projections of Qwen3.6-35B-A3B -- GDN ``out_proj`` and attention ``o_proj`` [2048, 4096] --
issued by ``fp8_pertensor_linear._scaled_mm``, ``models/qwen3_5_moe/gdn.py`` (fused gated-norm quant)
and ``kernel/cuda_attention_gate.py`` (fused gate quant). On torch 2.11 / cuBLASLt 13 that call runs a
384-byte semaphore memset plus ``sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64`` with
split-K = 3 reduced in-kernel. libqwen_oproj_gemv reproduces that arithmetic per output (identified
bit for bit, kernels/decode-oproj/): 64-byte k-tiles dealt round robin to 3 QMMA.16832 chains from
+0, partials combined as (p0 + p2) + p1, FFMA(acc, RN(scale_a * scale_b), +0), RN to bf16 -- in one
launch without workspace, memset or semaphore. Every other shape, M or dtype keeps torch._scaled_mm.

Enable: ``QWEN36_OPROJ_GEMV=1`` and ``QWEN36_OPROJ_GEMV_LIBRARY=<absolute path of libqwen_oproj_gemv.so>``.
With ``QWEN36_OPROJ_GEMV`` unset (or anything but "1") FreeToken never imports this module.

Provenance (kernel/pinned_ctypes.py): the file must hash to ``LIBRARY_SHA256`` and is dlopen'ed from a
sealed memfd; ABI probe ``qwen_oproj_abi_version() == ABI_VERSION``, ``qwen_oproj_max_m() == MAX_M``,
``qwen_oproj_splits() == 3``, ``qwen_oproj_source_sha256() == SOURCE_SHA256``. A failure raises
``OprojGemvProvenanceError`` at model load.
Numerics gate -- decided once per process at model load (``prepare_layer`` from the layer's finalize),
never under CUDA graph capture; failures only disable (cuBLASLt stays authoritative):
  * platform pin: sm_120 and torch 2.11.x (the cuBLASLt kernel the library reproduces);
  * process self-test: every M in 1..8, uniform e4m3 codes, gaussian codes, edge codes with exact
    cancellation, negative-zero activations, scales that overflow bf16 and that give subnormal
    results, against the exact production torch._scaled_mm call, compared bitwise; an M that fails
    stays on torch._scaled_mm (e.g. if cuBLASLt picked another algorithm for it);
  * per layer: the real weight / weight_scale / input_scale against torch._scaled_mm for every
    enabled M; a mismatch disables that M for every layer.
Runtime: launches on the current stream; no host sync, no device-value read, no host branching on
tensor values; the output comes from the caching allocator (graph-pool aware during capture).
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from . import pinned_ctypes

FP8 = torch.float8_e4m3fn
ENV_ENABLE = "QWEN36_OPROJ_GEMV"
ENV_LIBRARY = "QWEN36_OPROJ_GEMV_LIBRARY"
MAX_M = 8
ABI_VERSION = 1
SPLITS = 3
SHAPE = (2048, 4096)  # (N, K) of GDN out_proj and attention o_proj
# Library variant per M (same arithmetic; CTA shape / pipeline depth from the graph micro-benchmark,
# kernels/decode-oproj/README.md).
VARIANT_BY_M = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
# kernels/decode-oproj/build/libqwen_oproj_gemv.so (ABI v1, nvcc 13.4.59, sm_120a, --cudart=static) and
# the qwen_oproj_gemv.cu it was built from. nvcc output is not byte-reproducible: a rebuild must update
# LIBRARY_SHA256 here (and requalify), otherwise the new .so is refused.
LIBRARY_SHA256 = "fa9302e31cb88a02f9d6b37afc9e19691859f21e92861baf1bef3b2cab6a78a9"
SOURCE_SHA256 = "ff47a7d57830fa593d633a0482f26adc55724f03727ec3d80725f415146a7431"
TORCH_PIN = (2, 11)
CAPABILITY_PIN = (12, 0)
_SEED = 0x0B0B


class OprojGemvProvenanceError(pinned_ctypes.PinnedLibraryError):
    """The requested library is not exactly the pinned build; it was not loaded."""


@dataclass(frozen=True)
class _Ready:
    lib: Any
    fn: Any
    memfd: int
    device_index: int
    allowed: frozenset  # M values; replaced as a whole, never mutated


_LOCK = threading.Lock()
_STATE: _Ready | None = None
_DECIDED: dict = {}
_STATUS: dict = {"requested": None, "enabled": False, "reason": "not initialized", "layers_enabled": 0,
                 "layers_declined": 0, "layer_mismatch_events": []}
_LOGGER = None


def requested() -> bool:
    return os.environ.get(ENV_ENABLE) == "1"


def status() -> dict:
    with _LOCK:
        snap = dict(_STATUS)
        snap["timings_ms"] = dict(_STATUS.get("timings_ms", {}))
        st = _STATE
    snap["requested"] = requested()
    snap["enabled"] = st is not None
    if st is not None:
        snap["allowed_m"] = sorted(st.allowed)
    return snap


def _log():
    global _LOGGER
    if _LOGGER is None:
        from freetoken.utils import init_logger

        _LOGGER = init_logger(__name__)
    return _LOGGER


def _bind(lib: ctypes.CDLL):
    abi = pinned_ctypes.constant(lib, "qwen_oproj_abi_version", ctypes.c_int)
    max_m = pinned_ctypes.constant(lib, "qwen_oproj_max_m", ctypes.c_int)
    splits = pinned_ctypes.constant(lib, "qwen_oproj_splits", ctypes.c_int)
    source = pinned_ctypes.constant(lib, "qwen_oproj_source_sha256", ctypes.c_char_p)
    if (abi, max_m, splits, source) != (ABI_VERSION, MAX_M, SPLITS, SOURCE_SHA256):
        raise OprojGemvProvenanceError(f"libqwen_oproj_gemv ABI {abi} / max_m {max_m} / splits {splits} / source "
                                       f"{source} differs from the pinned {ABI_VERSION} / {MAX_M} / {SPLITS} / "
                                       f"{SOURCE_SHA256}")
    lib.qwen_oproj_init.argtypes = []
    lib.qwen_oproj_init.restype = ctypes.c_int
    lib.qwen_oproj_error.argtypes = [ctypes.c_int]
    lib.qwen_oproj_error.restype = ctypes.c_char_p
    fn = lib.qwen_oproj_tensorwise
    # x, w, sa, sb, out, M, N, K, ldx, ldw, ldo, variant, stream
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 6 + [ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _error_text(lib, code: int) -> str:
    try:
        raw = lib.qwen_oproj_error(int(code))
        return f"{code} ({raw.decode('ascii', 'replace') if raw else '?'})"
    except Exception:  # pragma: no cover - diagnostics only
        return str(code)


def _platform_reason(device: torch.device) -> str | None:
    m = re.match(r"(\d+)\.(\d+)\.", torch.__version__)
    if not m or (int(m.group(1)), int(m.group(2))) != TORCH_PIN:
        return f"torch {torch.__version__} (the library reproduces torch 2.11's cuBLASLt kernel)"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    cap = torch.cuda.get_device_capability(device)
    if tuple(cap) != CAPABILITY_PIN:
        return f"sm_{cap[0]}{cap[1]} (the library is sm_120a only)"
    return None


def reference(qa: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """Exactly the uniform-scale branch of fp8_pertensor_linear._scaled_mm (keep them in sync)."""
    return torch._scaled_mm(qa, weight.t(), scale_a=scale_a.reshape(()), scale_b=scale_b.reshape(()),
                            out_dtype=torch.bfloat16)


def _launch(fn, qa, weight, scale_a, scale_b, out) -> int:
    M, K = qa.shape
    return fn(qa.data_ptr(), weight.data_ptr(), scale_a.data_ptr(), scale_b.data_ptr(), out.data_ptr(),
              M, weight.shape[0], K, qa.stride(0) if M > 1 else K, weight.stride(0), out.stride(0), VARIANT_BY_M[M],
              torch.cuda.current_stream(qa.device).cuda_stream)


def _codes(shape, gen, device) -> torch.Tensor:
    c = torch.randint(0, 256, shape, generator=gen, device=device, dtype=torch.uint8)
    return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)


_EDGE_CODES = (0x00, 0x80, 0x7E, 0xFE, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8)


def _edge_activation(M, K, gen, device) -> torch.Tensor:
    table = torch.tensor(_EDGE_CODES, dtype=torch.uint8, device=device)
    half = table[torch.randint(0, len(_EDGE_CODES), (M, K // 2), generator=gen, device=device)]
    return torch.cat([half, half ^ 0x80], dim=1).contiguous().view(FP8)


def _self_test(fn, device) -> tuple[frozenset, list]:
    gen = torch.Generator(device=device)
    gen.manual_seed(_SEED)
    N, K = SHAPE
    weight = _codes((N, K), gen, device)
    w8 = weight.view(torch.uint8)
    w8[:32, K // 2:] = w8[:32, :K // 2]  # exact-cancellation rows against the edge activation
    gauss_w = (torch.randn((N, K), generator=gen, device=device) * 40).clamp(-448, 448).to(FP8)
    scales = ((0.0123, 0.00071), (2.0 ** 60, 2.0 ** 70), (2.0 ** -70, 2.0 ** -60), (0.37, 2.0 ** -11))
    bad = torch.zeros(MAX_M + 1, dtype=torch.int64, device=device)
    launch_error = {}
    for M in range(1, MAX_M + 1):
        negz = torch.full((M, K), 0x80, dtype=torch.uint8, device=device)
        negz[:, ::7] = 0
        acts = (("codes", _codes((M, K), gen, device), weight), ("edge", _edge_activation(M, K, gen, device), weight),
                ("gauss", (torch.randn((M, K), generator=gen, device=device) * 3).to(FP8), gauss_w),
                ("negzero", negz.view(FP8), weight))
        for i, (_, qa, w) in enumerate(acts):
            for a, b in (scales[i], scales[(i + M) % len(scales)]):
                sa = torch.tensor(a, dtype=torch.float32, device=device)
                sb = torch.tensor(b, dtype=torch.float32, device=device)
                out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
                rc = _launch(fn, qa, w, sa, sb, out)
                if rc:
                    launch_error[M] = rc
                    break
                bad[M] += (out.view(torch.int16) != reference(qa, w, sa, sb).view(torch.int16)).sum()
            if M in launch_error:
                break
    counts = bad.tolist()
    torch.cuda.synchronize(device)
    failures = [{"M": M, "launch_error": launch_error[M]} if M in launch_error else {"M": M, "mismatched_elements": counts[M]}
                for M in range(1, MAX_M + 1) if M in launch_error or counts[M]]
    allowed = frozenset(M for M in range(1, MAX_M + 1) if M not in launch_error and counts[M] == 0)
    return allowed, failures


def _ensure_ready_locked(device: torch.device) -> _Ready | None:
    global _STATE
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _DECIDED:
        decided = _DECIDED[idx]
        if decided.startswith("error: "):
            raise OprojGemvProvenanceError(decided[len("error: "):])
        return _STATE if (_STATE is not None and _STATE.device_index == idx) else None
    if _DECIDED:
        _DECIDED[idx] = "disabled: another CUDA device already owns the o-projection GEMV"
        return None
    dev = torch.device("cuda", idx)
    _STATUS.update(requested=True, device=idx, library=os.environ.get(ENV_LIBRARY), library_sha256=LIBRARY_SHA256,
                   source_sha256=SOURCE_SHA256, abi_version=ABI_VERSION, variant_by_m=dict(VARIANT_BY_M))
    reason = _platform_reason(dev)
    if reason is not None:
        _DECIDED[idx] = "disabled: " + reason
        _STATUS.update(reason=_DECIDED[idx])
        _log().warning(f"{ENV_ENABLE}=1 ignored, torch._scaled_mm stays: {reason}")
        return None
    path = os.environ.get(ENV_LIBRARY)
    timings = _STATUS.setdefault("timings_ms", {})
    t0 = time.perf_counter()
    try:
        lib, memfd = pinned_ctypes.load(path, LIBRARY_SHA256, ENV_LIBRARY, "qwen_oproj_gemv")
        fn = _bind(lib)
    except pinned_ctypes.PinnedLibraryError as exc:
        _DECIDED[idx] = "error: " + str(exc)
        _STATUS.update(reason=_DECIDED[idx])
        raise OprojGemvProvenanceError(str(exc)) from None
    timings["load_abi"] = round((time.perf_counter() - t0) * 1e3, 1)
    from . import kernel_provenance

    kernel_provenance.note_loaded("qwen_oproj_gemv", {
        "provenance": "ctypes-pinned-sha256", "so_path": path, "so_sha256": LIBRARY_SHA256,
        "abi_version": ABI_VERSION, "source_sha256": SOURCE_SHA256, "loaded_from": "memfd"})
    with torch.cuda.device(dev):
        free0 = torch.cuda.mem_get_info(dev)[0]
        rc = lib.qwen_oproj_init()
        if rc:
            reason, allowed, failures = "qwen_oproj_init failed: " + _error_text(lib, rc), frozenset(), []
        else:
            t0 = time.perf_counter()
            try:
                allowed, failures = _self_test(fn, dev)
                reason = None
            except RuntimeError as exc:  # asynchronous CUDA error surfaced by the sync
                allowed, failures, reason = frozenset(), [], f"self-test raised {type(exc).__name__}: {exc}"
            timings["self_test"] = round((time.perf_counter() - t0) * 1e3, 1)
        torch.cuda.synchronize(dev)
        _STATUS["free_vram_mib_before_after_first_launch"] = [free0 >> 20, torch.cuda.mem_get_info(dev)[0] >> 20]
    _STATUS.update(self_test_failures=failures)
    if reason is None and not allowed:
        reason = "bitwise self-test against torch._scaled_mm failed for every M"
    if reason is not None:
        _DECIDED[idx] = "disabled: " + reason
        _STATUS.update(reason=_DECIDED[idx])
        _log().warning(f"{ENV_ENABLE} disabled, torch._scaled_mm stays: {reason}; failures={failures[:8]}")
        return None
    _STATE = _Ready(lib=lib, fn=fn, memfd=memfd, device_index=idx, allowed=allowed)
    _DECIDED[idx] = "enabled"
    _STATUS.update(reason="enabled")
    if failures:
        _log().warning(f"{ENV_ENABLE}: self-test mismatches keep these on torch._scaled_mm: {failures[:8]}")
    _log().info(f"{ENV_ENABLE} enabled: libqwen_oproj_gemv sha256 {LIBRARY_SHA256[:16]}.. ABI {ABI_VERSION}, "
                f"bit-exact self-test passed for M {sorted(allowed)}")
    return _STATE


def _disable_locked(reason: str) -> None:
    global _STATE
    _STATE = None
    for idx, decided in list(_DECIDED.items()):
        if decided == "enabled":
            _DECIDED[idx] = "disabled: " + reason
    _STATUS.update(reason="disabled: " + reason)
    _log().warning(f"{ENV_ENABLE} disabled for the rest of the process, torch._scaled_mm stays: {reason}")


def _layer_tensors_ok(weight, weight_scale, input_scale) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (weight, weight_scale, input_scale)):
        return False
    if not weight.is_cuda or weight.dtype is not FP8 or weight.dim() != 2 or tuple(weight.shape) != SHAPE:
        return False
    N, K = weight.shape
    if weight.stride() != (K, 1) or weight.data_ptr() % 16:
        return False
    if weight_scale.dtype is not torch.float32 or weight_scale.dim() != 1 or weight_scale.shape[0] != N:
        return False
    if input_scale.dtype is not torch.float32 or input_scale.numel() != 1:
        return False
    if weight_scale.device != weight.device or input_scale.device != weight.device:
        return False
    return weight_scale.data_ptr() % 4 == 0 and input_scale.data_ptr() % 4 == 0


def _check_layer_locked(st: _Ready, weight, weight_scale, input_scale) -> _Ready | None:
    """Real weights vs torch._scaled_mm for every enabled M; a failing M is disabled for every layer."""
    global _STATE
    ms = sorted(st.allowed)
    dev = weight.device
    t0 = time.perf_counter()
    gen = torch.Generator(device=dev)
    gen.manual_seed(_SEED + 1)
    bad = torch.zeros(MAX_M + 1, dtype=torch.int64, device=dev)
    sa, sb = input_scale.reshape(()), weight_scale[0].reshape(())
    with torch.cuda.device(dev):
        for M in ms:
            for qa in (_codes((M, SHAPE[1]), gen, dev),
                       (torch.randn((M, SHAPE[1]), generator=gen, device=dev) * 60).clamp(-448, 448).to(FP8)):
                out = torch.empty((M, SHAPE[0]), dtype=torch.bfloat16, device=dev)
                rc = _launch(st.fn, qa, weight, sa, sb, out)
                if rc:
                    _disable_locked(f"launch error {_error_text(st.lib, rc)} in the per-layer check")
                    return None
                bad[M] += (out.view(torch.int16) != reference(qa, weight, sa, sb).view(torch.int16)).sum()
        counts = bad.tolist()
    timings = _STATUS.setdefault("timings_ms", {})
    timings["layer_checks"] = round(timings.get("layer_checks", 0.0) + (time.perf_counter() - t0) * 1e3, 1)
    timings["layers_checked"] = timings.get("layers_checked", 0) + 1
    failed = [M for M in ms if counts[M]]
    if not failed:
        return st
    event = {"failed_m": failed, "mismatched_elements": [counts[M] for M in failed]}
    _STATUS["layer_mismatch_events"] = [*_STATUS["layer_mismatch_events"], event]
    allowed = frozenset(m for m in ms if m not in failed)
    _log().error(f"{ENV_ENABLE}: real-weight check mismatched {event}; those M stay on torch._scaled_mm")
    if not allowed:
        _disable_locked("real-weight check failed for every M")
        return None
    _STATE = _Ready(lib=st.lib, fn=st.fn, memfd=st.memfd, device_index=st.device_index, allowed=allowed)
    return _STATE


def prepare_layer(layer: Any) -> bool:
    """Decide at finalize (eager, before graph capture) whether the uniform-scale ``layer`` may use the
    GEMV: True only for the qualified shape whose real weights pass the bitwise check. Raises
    OprojGemvProvenanceError if the requested library is not the pinned build."""
    if not requested():
        return False
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale", None)
    input_scale = getattr(layer, "input_scale", None)
    declined = (getattr(layer, "bias", None) is not None or not _layer_tensors_ok(weight, weight_scale, input_scale)
                or torch.cuda.is_current_stream_capturing())
    if not declined:
        from freetoken.kernel.triton.e4m3_compat import e4m3_native

        declined = not e4m3_native()
    if declined:
        with _LOCK:
            _STATUS["layers_declined"] += 1
        return False
    with _LOCK:
        st = _ensure_ready_locked(weight.device)
        if st is not None and st.device_index == weight.device.index:
            st = _check_layer_locked(st, weight, weight_scale, input_scale)
        ok = st is not None
        _STATUS["layers_enabled" if ok else "layers_declined"] += 1
    return ok


def tensorwise(qa: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor | None:
    """The uniform-scale ``torch._scaled_mm`` result computed by the library, or None when this call is
    not covered (the caller then issues torch._scaled_mm, which gives the same bits). ``scale_a`` /
    ``scale_b`` are the 0-d fp32 device scalars the reference call receives."""
    st = _STATE
    if st is None or qa.dim() != 2 or weight.dim() != 2:
        return None
    M, K = qa.shape
    if M < 1 or M > MAX_M or M not in st.allowed or tuple(weight.shape) != SHAPE or K != SHAPE[1]:
        return None
    dev = qa.device
    if dev.type != "cuda" or dev.index != st.device_index:
        return None
    if qa.dtype is not FP8 or weight.dtype is not FP8 or qa.stride(1) != 1 or weight.stride() != (K, 1):
        return None
    if scale_a.dtype is not torch.float32 or scale_b.dtype is not torch.float32 or scale_a.numel() != 1 or scale_b.numel() != 1:
        return None
    if weight.device != dev or scale_a.device != dev or scale_b.device != dev:
        return None
    if M > 1 and qa.stride(0) < K:
        return None
    if (qa.data_ptr() | weight.data_ptr() | (qa.stride(0) if M > 1 else 0)) & 15 or (scale_a.data_ptr() | scale_b.data_ptr()) & 3:
        return None
    out = torch.empty((M, SHAPE[0]), dtype=torch.bfloat16, device=dev)
    rc = _launch(st.fn, qa, weight, scale_a, scale_b, out)
    if rc:  # validation/launch error: nothing was produced; fall back for good
        with _LOCK:
            if _STATE is st:
                _disable_locked(f"launch error {_error_text(st.lib, rc)} for M={M}")
        return None
    return out
