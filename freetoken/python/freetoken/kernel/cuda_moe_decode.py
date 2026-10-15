"""Opt-in bit-exact NVFP4 MoE decode GEMV (the two routed-expert GEMMs of a decode step).

What it replaces, and nothing else: ``moe/fused_nvfp4._decode_gemm_marlin`` (Triton
``_decode_nvfp4_marlin_kernel``, BLOCK_N 16, BLOCK_KW 16, 4 warps) for the gate_up (K = 2048) and
down (K = 512) GEMMs of ``TritonNvfp4MoEKernel`` decode, M <= MAX_M tokens. The activation, the
top-k sum, prefill and every shape or dtype this module does not recognise keep the Triton path.
libqwen_moe_decode computes every output with the Triton kernel's operations in the same order (16
FP32 partials over words w % 16, exact e2m1 x bf16 products chained from +0, fma with the e4m3
block scale, the shfl.bfly 4/2/1 reduction tree, x global [x router weight], RN to bf16;
kernels/decode-moe/) and differs only in data movement: 16-byte weight loads, a register prefetch
ring, the activation staged once per CTA.

Enable: ``QWEN36_MOE_DECODE_CUDA=1`` and ``QWEN36_MOE_DECODE_LIBRARY=<absolute path of
libqwen_moe_decode.so>``. With ``QWEN36_MOE_DECODE_CUDA`` unset FreeToken never imports this module.

Provenance (kernel/pinned_ctypes.py): the file must hash to ``LIBRARY_SHA256`` and is dlopen'ed from a
sealed memfd; ABI probe ``qwen_moe_decode_abi_version() == ABI_VERSION`` and
``qwen_moe_decode_source_sha256() == SOURCE_SHA256``. A failure raises ``MoeDecodeProvenanceError``.
Numerics gate (failures only disable; the Triton GEMV stays authoritative):
  * platform pin: sm_120, torch 2.11.x, Triton 3.6.x (the kernel the library reproduces);
  * process self-test (``prepare``, from the MoE method's finalize, eager): synthetic banks with every
    e2m1 code, every finite e4m3 scale code and globals over 2^-40..2^30, gaussian / heavy-tailed /
    edge activations (+-0, bf16 subnormals, ~3e38), both GEMMs, every M in 1..MAX_M with top_k 8 and
    slots shared between tokens, against the production Triton call, compared bitwise; an M that
    fails stays on Triton;
  * per layer, real weights: the first eager (non-capturing) decode call of each MoE layer runs both
    GEMMs on that call's real slot banks, routes and activations, for every enabled M (rows derived
    from the real hidden state, routes permuted from the real ones), against Triton; any mismatch
    keeps that layer on Triton for the rest of the process. A layer whose first decode call is under
    CUDA graph capture stays on Triton until an eager call decides it.
Runtime: launches on the current stream; no host sync, no device-value read, no host branching on
tensor values (only shapes / strides / dtypes / alignment) after the per-layer decision.
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
ENV_ENABLE = "QWEN36_MOE_DECODE_CUDA"
ENV_LIBRARY = "QWEN36_MOE_DECODE_LIBRARY"
ABI_VERSION = 1
MAX_M = 8
KS = (512, 2048)  # down, gate_up
# Library variant per (K, M) (same arithmetic; CTA shape / prefetch depth from the graph micro-benchmark,
# kernels/decode-moe/README.md).
_VARIANT_SMALL_M = {2048: 1, 512: 1}
_VARIANT_LARGE_M = {2048: 2, 512: 2}
# kernels/decode-moe/build/libqwen_moe_decode.so (ABI v1, nvcc 13.4.59, sm_120a, --cudart=static) and the
# qwen_moe_decode.cu it was built from. A rebuild must update LIBRARY_SHA256 here (and requalify).
LIBRARY_SHA256 = "83e90c43cb63ce11b284a37cdadf7eff82f147eb5873e988f358dd7df7d48dd5"
SOURCE_SHA256 = "088bb27525a60b602d9ebc84d1d6810343747fec116fa9c288b031cc5437c95a"
TORCH_PIN = (2, 11)
TRITON_PIN = (3, 6)
CAPABILITY_PIN = (12, 0)
_SEED = 0x30E0


def variant_for(K: int, M: int) -> int:
    return (_VARIANT_SMALL_M if M <= 2 else _VARIANT_LARGE_M)[K]


class MoeDecodeProvenanceError(pinned_ctypes.PinnedLibraryError):
    """The requested library is not exactly the pinned build; it was not loaded."""


@dataclass(frozen=True)
class _Ready:
    lib: Any
    fn: Any
    memfd: int
    device_index: int
    allowed: frozenset  # M values


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
    abi = pinned_ctypes.constant(lib, "qwen_moe_decode_abi_version", ctypes.c_int)
    source = pinned_ctypes.constant(lib, "qwen_moe_decode_source_sha256", ctypes.c_char_p)
    if (abi, source) != (ABI_VERSION, SOURCE_SHA256):
        raise MoeDecodeProvenanceError(f"libqwen_moe_decode ABI {abi} / source {source} differs from the pinned "
                                       f"{ABI_VERSION} / {SOURCE_SHA256}")
    lib.qwen_moe_decode_error.argtypes = [ctypes.c_int]
    lib.qwen_moe_decode_error.restype = ctypes.c_char_p
    fn = lib.qwen_moe_decode_gemv
    ll, vp = ctypes.c_longlong, ctypes.c_void_p
    fn.argtypes = [vp, ll, vp, ll, ll, vp, ll, ll, vp, ll, ll, vp, ll, ll, ll, vp, ll, ll, vp, ll, ll,
                   ll, ll, ll, ll, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp]
    fn.restype = ctypes.c_int
    return fn


def _error_text(lib, code: int) -> str:
    try:
        raw = lib.qwen_moe_decode_error(int(code))
        return f"{code} ({raw.decode('ascii', 'replace') if raw else '?'})"
    except Exception:  # pragma: no cover - diagnostics only
        return str(code)


def _platform_reason(device: torch.device) -> str | None:
    import triton

    for name, version, pin in (("torch", torch.__version__, TORCH_PIN), ("triton", triton.__version__, TRITON_PIN)):
        m = re.match(r"(\d+)\.(\d+)\.", version)
        if not m or (int(m.group(1)), int(m.group(2))) != pin:
            return f"{name} {version} (the library reproduces the Triton 3.6 / torch 2.11 decode GEMV)"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    cap = torch.cuda.get_device_capability(device)
    if tuple(cap) != CAPABILITY_PIN:
        return f"sm_{cap[0]}{cap[1]} (the library is sm_120a only)"
    return None


def _call_ok(a, packed, scale, glob, c, topk_weights, topk_ids, a_row_is_route) -> bool:
    """Static checks only (shape / stride / dtype / device / alignment)."""
    if a.dim() != 2 or packed.dim() != 3 or scale.dim() != 3 or glob.dim() != 2 or c.dim() != 3:
        return False
    if topk_ids.dim() != 2 or topk_weights.dim() != 2:
        return False
    M, top_k = topk_ids.shape
    S, N, KB = packed.shape
    K = 2 * KB
    if K not in KS or M < 1 or M > MAX_M or a.shape[1] != K or a.shape[0] != (M * top_k if a_row_is_route else M):
        return False
    if a.dtype is not torch.bfloat16 or c.dtype is not torch.bfloat16 or packed.dtype is not torch.uint8:
        return False
    if scale.dtype is not FP8 or glob.dtype is not torch.float32 or topk_weights.dtype is not torch.float32:
        return False
    if topk_ids.dtype is not torch.int32 or tuple(scale.shape) != (S, N, K // 16) or tuple(glob.shape) != (S, N):
        return False
    if tuple(c.shape) != (M, top_k, N) or tuple(topk_weights.shape) != (M, top_k):
        return False
    if a.stride(1) != 1 or packed.stride(2) != 1 or scale.stride(2) != 1 or (a.shape[0] > 1 and a.stride(0) < K):
        return False
    dev = a.device
    if any(t.device != dev for t in (packed, scale, glob, c, topk_weights, topk_ids)):
        return False
    if (a.data_ptr() | packed.data_ptr() | (a.stride(0) * 2 if M * (top_k if a_row_is_route else 1) > 1 else 0)
            | packed.stride(0) | packed.stride(1)) & 15:
        return False
    return not ((scale.data_ptr() | scale.stride(0) | scale.stride(1)) & 1 or glob.data_ptr() & 3 or c.data_ptr() & 1
                or topk_ids.data_ptr() & 3 or topk_weights.data_ptr() & 3)


def _launch(fn, a, packed, scale, glob, c, topk_weights, topk_ids, mul_routed_weight, a_row_is_route) -> int:
    M, top_k = topk_ids.shape
    S, N, KB = packed.shape
    K = 2 * KB
    a_stride = a.stride(0) if a.shape[0] > 1 else K
    return fn(a.data_ptr(), a_stride, packed.data_ptr(), packed.stride(0), packed.stride(1),
              scale.data_ptr(), scale.stride(0), scale.stride(1), glob.data_ptr(), glob.stride(0), glob.stride(1),
              c.data_ptr(), c.stride(0), c.stride(1), c.stride(2), topk_weights.data_ptr(), topk_weights.stride(0),
              topk_weights.stride(1), topk_ids.data_ptr(), topk_ids.stride(0), topk_ids.stride(1),
              M, top_k, N, K, int(bool(a_row_is_route)), int(bool(mul_routed_weight)), variant_for(K, M),
              torch.cuda.current_stream(a.device).cuda_stream)


# --------------------------------------------------------------------------------------- self-tests


def _reference_gemm():
    from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin

    return _decode_gemm_marlin


def _activation(kind, shape, gen, dev):
    if kind == "gauss":
        return torch.randn(shape, generator=gen, device=dev).to(torch.bfloat16)
    if kind == "heavy":
        x = torch.randn(shape, generator=gen, device=dev)
        return (x * torch.exp(torch.randn(shape, generator=gen, device=dev) * 3)).to(torch.bfloat16)
    table = torch.tensor([0.0, -0.0, 1e-39, -1e-39, 9.2e-41, 3.0e38, -3.0e38, 1.0, -2.5, 65504.0],
                         device=dev).to(torch.bfloat16)
    return table[torch.randint(0, len(table), shape, generator=gen, device=dev)]


def _compare(fn, ref_gemm, a, packed, scale, glob, ids, w, mul_w, a_route, bad, M) -> int:
    top_k = ids.shape[1]
    N = packed.shape[1]
    ref = torch.empty((M, top_k, N), dtype=torch.bfloat16, device=a.device)
    out = torch.empty_like(ref)
    ref_gemm(a, packed, scale, glob, ref, w, ids, mul_w, a_route)
    rc = _launch(fn, a, packed, scale, glob, out, w, ids, mul_w, a_route)
    if rc:
        return rc
    bad[M] += (out.view(torch.int16) != ref.view(torch.int16)).sum()
    return 0


def _self_test(fn, dev, H: int = 2048, I: int = 512, top_k: int = 8) -> tuple[frozenset, list]:
    gen = torch.Generator(device=dev)
    gen.manual_seed(_SEED)
    S = 16

    def scales(shape):
        c = torch.randint(0, 256, shape, generator=gen, device=dev, dtype=torch.uint8)
        return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)

    gu = torch.randint(0, 256, (S, 2 * I, H // 2), generator=gen, device=dev, dtype=torch.uint8)
    dn = torch.randint(0, 256, (S, H, I // 2), generator=gen, device=dev, dtype=torch.uint8)
    gus, dns = scales((S, 2 * I, H // 16)), scales((S, H, I // 16))
    gug = torch.exp2(torch.empty((S, 2 * I), device=dev).uniform_(-40, 30, generator=gen))
    dng = torch.exp2(torch.empty((S, H), device=dev).uniform_(-40, 30, generator=gen))
    ref_gemm = _reference_gemm()
    bad = torch.zeros(MAX_M + 1, dtype=torch.int64, device=dev)
    launch_error = {}
    for M in range(1, MAX_M + 1):
        for kind in ("gauss", "heavy", "edge"):
            ids = torch.stack([torch.randperm(S, generator=gen, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
            if M > 1:
                ids[1:, 0] = ids[0, 0]
            ids = ids.contiguous()
            w = torch.rand((M, top_k), generator=gen, device=dev)
            x = _activation(kind, (M, H), gen, dev)
            a2 = _activation(kind, (M * top_k, I), gen, dev)
            rc = (_compare(fn, ref_gemm, x, gu, gus, gug, ids, w, False, False, bad, M)
                  or _compare(fn, ref_gemm, a2, dn, dns, dng, ids, w, True, True, bad, M))
            if rc:
                launch_error[M] = rc
                break
    counts = bad.tolist()
    torch.cuda.synchronize(dev)
    failures = [{"M": M, "launch_error": launch_error[M]} if M in launch_error else {"M": M, "mismatched_elements": counts[M]}
                for M in range(1, MAX_M + 1) if M in launch_error or counts[M]]
    return frozenset(M for M in range(1, MAX_M + 1) if M not in launch_error and counts[M] == 0), failures


def _ensure_ready_locked(device: torch.device) -> _Ready | None:
    global _STATE
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _DECIDED:
        decided = _DECIDED[idx]
        if decided.startswith("error: "):
            raise MoeDecodeProvenanceError(decided[len("error: "):])
        return _STATE if (_STATE is not None and _STATE.device_index == idx) else None
    if _DECIDED:
        _DECIDED[idx] = "disabled: another CUDA device already owns the MoE decode GEMV"
        return None
    dev = torch.device("cuda", idx)
    _STATUS.update(requested=True, device=idx, library=os.environ.get(ENV_LIBRARY), library_sha256=LIBRARY_SHA256,
                   source_sha256=SOURCE_SHA256, abi_version=ABI_VERSION)
    reason = _platform_reason(dev)
    if reason is not None:
        _DECIDED[idx] = "disabled: " + reason
        _STATUS.update(reason=_DECIDED[idx])
        _log().warning(f"{ENV_ENABLE}=1 ignored, the Triton decode GEMV stays: {reason}")
        return None
    path = os.environ.get(ENV_LIBRARY)
    timings = _STATUS.setdefault("timings_ms", {})
    t0 = time.perf_counter()
    try:
        lib, memfd = pinned_ctypes.load(path, LIBRARY_SHA256, ENV_LIBRARY, "qwen_moe_decode")
        fn = _bind(lib)
    except pinned_ctypes.PinnedLibraryError as exc:
        _DECIDED[idx] = "error: " + str(exc)
        _STATUS.update(reason=_DECIDED[idx])
        raise MoeDecodeProvenanceError(str(exc)) from None
    timings["load_abi"] = round((time.perf_counter() - t0) * 1e3, 1)
    from . import kernel_provenance

    kernel_provenance.note_loaded("qwen_moe_decode", {
        "provenance": "ctypes-pinned-sha256", "so_path": path, "so_sha256": LIBRARY_SHA256,
        "abi_version": ABI_VERSION, "source_sha256": SOURCE_SHA256, "loaded_from": "memfd"})
    with torch.cuda.device(dev):
        free0 = torch.cuda.mem_get_info(dev)[0]
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
        reason = "bitwise self-test against the Triton decode GEMV failed for every M"
    if reason is not None:
        _DECIDED[idx] = "disabled: " + reason
        _STATUS.update(reason=_DECIDED[idx])
        _log().warning(f"{ENV_ENABLE} disabled, the Triton decode GEMV stays: {reason}; failures={failures[:8]}")
        return None
    _STATE = _Ready(lib=lib, fn=fn, memfd=memfd, device_index=idx, allowed=allowed)
    _DECIDED[idx] = "enabled"
    _STATUS.update(reason="enabled")
    if failures:
        _log().warning(f"{ENV_ENABLE}: self-test mismatches keep these M on Triton: {failures[:8]}")
    _log().info(f"{ENV_ENABLE} enabled: libqwen_moe_decode sha256 {LIBRARY_SHA256[:16]}.. ABI {ABI_VERSION}, "
                f"bit-exact self-test passed for M {sorted(allowed)}")
    return _STATE


def prepare(device: torch.device) -> bool:
    """Load + pin + process self-test once (eager, never under capture). Raises
    MoeDecodeProvenanceError when the requested library is not the pinned build."""
    if not requested() or torch.cuda.is_current_stream_capturing():
        return False
    with _LOCK:
        return _ensure_ready_locked(device) is not None


def _layer_check_locked(st: _Ready, x, banks, topk_weights, topk_ids, activation, alpha, limit) -> list:
    """Both GEMMs on this call's real slot banks / routes / activations for every enabled M against
    Triton; returns the list of failing M (empty = layer qualifies)."""
    from freetoken.layers import gated_act_and_mul

    gu, gus, gug, dn, dns, dng = banks
    dev = x.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(_SEED + 1)
    ref_gemm = _reference_gemm()
    bad = torch.zeros(MAX_M + 1, dtype=torch.int64, device=dev)
    real_ids = topk_ids.reshape(-1)
    top_k = topk_ids.shape[1]
    H = x.shape[1]
    rms = x.float().pow(2).mean().sqrt().clamp_min(1e-3)
    for M in sorted(st.allowed):
        rows = [x[m % x.shape[0]] for m in range(M)]
        xm = torch.stack([rows[0]] + [(r.float() + torch.randn(H, generator=gen, device=dev) * rms).to(torch.bfloat16)
                                      for r in rows[1:]]).contiguous()
        ids = torch.stack([real_ids[torch.randperm(real_ids.numel(), generator=gen, device=dev)[:top_k]]
                           for _ in range(M)]).to(torch.int32)
        ids[0] = topk_ids[0]
        ids = ids.contiguous()
        w = topk_weights[0].reshape(1, -1).expand(M, -1).contiguous()
        two_i = gu.shape[1]
        ic1 = torch.empty((M, top_k, two_i), dtype=torch.bfloat16, device=dev)
        ref_gemm(xm, gu, gus, gug, ic1, w, ids, False, False)
        ic2 = torch.empty((M * top_k, two_i // 2), dtype=torch.bfloat16, device=dev)
        gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=alpha, limit=limit)
        for a, packed, scale, glob, mul_w, a_route in ((xm, gu, gus, gug, False, False), (ic2, dn, dns, dng, True, True)):
            rc = _compare(st.fn, ref_gemm, a, packed, scale, glob, ids, w, mul_w, a_route, bad, M)
            if rc:
                return [M]
    counts = bad.tolist()
    return [M for M in sorted(st.allowed) if counts[M]]


def _disable(st: _Ready, reason: str) -> None:
    global _STATE
    with _LOCK:
        if _STATE is st:
            _STATE = None
            _STATUS.update(reason="disabled: " + reason)
            _log().warning(f"{ENV_ENABLE} disabled for the rest of the process, the Triton decode GEMV stays: {reason}")


def gemm_for_layer(layer: Any, x: torch.Tensor, banks: tuple, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                   activation: str, alpha: float, limit: float, apply_router_weight_on_input: bool):
    """The GEMV function to run this decode call with (``fused_nvfp4._decode_gemm_marlin`` calling
    convention), or None to keep the Triton GEMV. The per-layer real-weight decision happens on the
    first eager call; under capture an undecided layer stays on Triton."""
    decided = getattr(layer, "_moe_decode_cuda", None)
    if decided is False:
        return None
    if decided is None:
        if apply_router_weight_on_input or torch.cuda.is_current_stream_capturing():
            return None
        with _LOCK:
            decided = getattr(layer, "_moe_decode_cuda", None)
            if decided is None:
                st = _ensure_ready_locked(x.device)
                ok = st is not None and x.dim() == 2 and 1 <= x.shape[0] <= MAX_M and topk_ids.dtype is torch.int32
                if ok:
                    t0 = time.perf_counter()
                    failed = _layer_check_locked(st, x, banks, topk_weights, topk_ids, activation, alpha, limit)
                    timings = _STATUS.setdefault("timings_ms", {})
                    timings["layer_checks"] = round(timings.get("layer_checks", 0.0) + (time.perf_counter() - t0) * 1e3, 1)
                    if failed:
                        event = {"layer": getattr(layer, "layer_id", None), "failed_m": failed}
                        _STATUS["layer_mismatch_events"] = [*_STATUS["layer_mismatch_events"], event]
                        _log().error(f"{ENV_ENABLE}: real-weight check mismatched {event}; the layer stays on Triton")
                        ok = False
                layer._moe_decode_cuda = decided = ok
                _STATUS["layers_enabled" if ok else "layers_declined"] += 1
        if not decided:
            return None
    st = _STATE
    if st is None or x.dim() != 2 or x.shape[0] not in st.allowed:
        return None
    fn, ref = st.fn, _reference_gemm()

    def gemm(a, packed, scale, glob, c, topk_weights_, topk_ids_, mul_routed_weight, a_row_is_route):
        if _call_ok(a, packed, scale, glob, c, topk_weights_, topk_ids_, a_row_is_route):
            rc = _launch(fn, a, packed, scale, glob, c, topk_weights_, topk_ids_, mul_routed_weight, a_row_is_route)
            if rc == 0:
                return
            _disable(st, f"launch error {_error_text(st.lib, rc)}")
        ref(a, packed, scale, glob, c, topk_weights_, topk_ids_, mul_routed_weight, a_row_is_route)

    return gemm
