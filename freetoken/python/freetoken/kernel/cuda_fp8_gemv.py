"""Opt-in bit-exact FP8 W8A8 skinny GEMM (1 <= M <= 8) for the fused decode input projections.

What it replaces, and nothing else: the row-wise ``torch._scaled_mm`` call of
``fp8_pertensor_linear._scaled_mm`` (CUTLASS SM120 128x128x128 tile, 127 of 128 rows padding at
M=1) for the two fused projections of Qwen3.6-35B-A3B -- GDN ``in_proj_qkvz`` [12288, 2048] and
attention ``qkv_proj`` [9216, 2048] -- when the step has 1 <= M <= 8 tokens. Uniform-scale
projections (cuBLASLt), the per-part segment path, prefill-sized M and every input this module
does not recognise exactly keep ``torch._scaled_mm``. The library reproduces the CUTLASS kernel's
arithmetic per output (same QMMA.16832.F32.E4M3.E4M3 chain from +0 over ascending k-blocks, no
split-K; epilogue FMUL -> FFMA(+0) -> F2FP RN), so both paths give the same bits.

Enable: ``QWEN36_FP8_GEMV=1`` and ``QWEN36_FP8_GEMV_LIBRARY=<absolute path of libqwen_fp8_gemv.so>``.
With ``QWEN36_FP8_GEMV`` unset (or anything but "1") FreeToken never imports this module.

Provenance -- the ctypes private-library pattern of ``QWEN36_CUDA_NVFP4_LIBRARY``
(moe/cuda_nvfp4.py), tightened so that a stale or foreign .so can never load:
  * the file is read once (O_NOFOLLOW, regular file) and must hash to ``LIBRARY_SHA256``; the
    verified bytes are dlopen'ed from an anonymous sealed memfd, so no path is re-opened between
    the hash and the load;
  * ABI probe (constant entry points, no CUDA call): ``qwen_fp8_gemv_abi_version() == ABI_VERSION``,
    ``qwen_fp8_gemv_max_m() == MAX_M``, ``qwen_fp8_gemv_source_sha256() == SOURCE_SHA256``;
  * any provenance failure raises ``Fp8GemvProvenanceError`` at model load (like a missing
    ``QWEN36_FUSED_ATTN_GATE_LIBRARY``): an explicitly requested library that is not exactly the
    pinned build is a deployment error, never silently loaded and never silently skipped.
Numerics gate -- decided once per process at model load (``prepare_layer`` from the layer's
``finalize``), never under CUDA graph capture; failures only disable (CUTLASS stays authoritative):
  * platform pin: sm_120 and torch 2.11.x (the CUTLASS kernel the library reproduces);
  * process self-test: both production shapes x every M in 1..8, random full-range e4m3 codes
    plus edge codes (+-0, +-448, subnormals, exact cancellation), wide per-row weight scales,
    against the exact production ``torch._scaled_mm`` row-wise call, compared bitwise. An M that
    fails stays on torch._scaled_mm (e.g. if torch picked another CUTLASS config for it);
  * per layer: the real weight / weight_scale / input_scale against torch._scaled_mm for every
    enabled M; a mismatch disables that M for every layer of that shape.
Runtime: launches on the current stream; no host sync, no device-value read, no host branching
on tensor values (only on static shape/stride/dtype/alignment); the output is allocated through
the caching allocator (graph-pool aware during capture, as torch._scaled_mm's own output is).
Thread safety: the readiness decision runs under a lock; the hot path reads one immutable state.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

FP8 = torch.float8_e4m3fn
ENV_ENABLE = "QWEN36_FP8_GEMV"
ENV_LIBRARY = "QWEN36_FP8_GEMV_LIBRARY"
MAX_M = 8  # keep equal to fp8_pertensor_linear._GEMV_MAX_M
ABI_VERSION = 2
# {4 warps, 4 stages, 128-B K chunk per stage, L2::256B prefetch}: best geometric mean of the
# graph sweep over both shapes and M 1/8 (cuda-fp8-gemv results.json); every variant is the same
# arithmetic, only tiling differs.
VARIANT = 11
# (N, K) of the fused decode projections: GDN in_proj_qkvz, attention qkv_proj.
SHAPES = ((12288, 2048), (9216, 2048))
# work/wf-fp8gemv/build/libqwen_fp8_gemv.so (ABI v2, nvcc 13.4.59, sm_120a, --cudart=static) and
# the qwen_fp8_gemv.cu it was built from. nvcc output is not byte-reproducible: a rebuild must
# update LIBRARY_SHA256 here (and requalify), otherwise the new .so is refused.
LIBRARY_SHA256 = "685c78cf89e60d0205f0db3ac2040203dec27e510a1e543cb7a7d3097414093b"
SOURCE_SHA256 = "9844420833b0c33c5b5fb73280f63527597306adb49b979731ac1fe89ac708fc"
TORCH_PIN = (2, 11)
CAPABILITY_PIN = (12, 0)
_LIBRARY_MAX_BYTES = 64 << 20
_SEED = 0x5EEDF8


class Fp8GemvProvenanceError(RuntimeError):
    """The requested library is not exactly the pinned build; it was not loaded."""


@dataclass(frozen=True)
class _Ready:
    lib: Any  # ctypes.CDLL, kept alive for the process
    fn: Any  # bound qwen_fp8_gemv_rowwise
    memfd: int  # sealed memfd holding the verified bytes
    device_index: int
    allowed: dict  # (N, K) -> frozenset of M; replaced as a whole, never mutated


_LOCK = threading.Lock()
_STATE: _Ready | None = None  # None: not ready -> every call stays on torch._scaled_mm
_DECIDED: dict = {}  # device index -> "enabled" | "disabled: <reason>" | "error: <reason>"
_STATUS: dict = {"requested": None, "enabled": False, "reason": "not initialized", "layers_enabled": 0,
                 "layers_declined": 0, "layer_mismatch_events": []}
_LOGGER = None


def requested() -> bool:
    return os.environ.get(ENV_ENABLE) == "1"


def status() -> dict:
    """Snapshot of the process decision (for logs, functional checks and the release report)."""
    with _LOCK:
        snap = dict(_STATUS)
        snap["timings_ms"] = dict(_STATUS.get("timings_ms", {}))
        st = _STATE
    snap["requested"] = requested()
    snap["enabled"] = st is not None
    if st is not None:
        snap["allowed_m"] = {f"{n}x{k}": sorted(ms) for (n, k), ms in st.allowed.items()}
    return snap


def _log():
    global _LOGGER
    if _LOGGER is None:
        from freetoken.utils import init_logger

        _LOGGER = init_logger(__name__)
    return _LOGGER


# ------------------------------------------------------------------------------- provenance


def _read_pinned(path: str | None) -> bytes:
    """The library bytes, only if they hash to LIBRARY_SHA256 (read once, never through a symlink)."""
    if not path or not os.path.isabs(path):
        raise Fp8GemvProvenanceError(f"{ENV_LIBRARY} must be the absolute path of the pinned libqwen_fp8_gemv.so "
                                     f"(sha256 {LIBRARY_SHA256}), got {path!r}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise Fp8GemvProvenanceError(f"cannot open {ENV_LIBRARY}={path}: {exc.strerror}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _LIBRARY_MAX_BYTES:
            raise Fp8GemvProvenanceError(f"{ENV_LIBRARY}={path} is not a regular file of at most {_LIBRARY_MAX_BYTES} bytes")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    if digest != LIBRARY_SHA256:
        raise Fp8GemvProvenanceError(f"{ENV_LIBRARY}={path} has sha256 {digest}, but cuda_fp8_gemv.py pins "
                                     f"{LIBRARY_SHA256}: a stale or foreign build; refusing to load it")
    return data


# Linux memfd sealing ABI (some Python builds' fcntl module lacks the names; the values are stable)
_MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
_F_ADD_SEALS, _F_GET_SEALS = 1024 + 9, 1024 + 10
_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008  # F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE


def _dlopen_verified(data: bytes) -> tuple[ctypes.CDLL, int]:
    """dlopen exactly the hashed bytes, from an anonymous sealed memfd (no path is re-opened)."""
    import fcntl

    fd = os.memfd_create("qwen_fp8_gemv", os.MFD_CLOEXEC | _MFD_ALLOW_SEALING)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        try:
            fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", _F_ADD_SEALS), _SEALS)
        except OSError:
            pass  # seals are defence in depth: the bytes were hashed in this process before the write
        lib = ctypes.CDLL(f"/proc/self/fd/{fd}")
    except BaseException:
        os.close(fd)
        raise
    return lib, fd


def _bind(lib: ctypes.CDLL):
    """ABI probe (constants, no CUDA call), then the typed entry points."""

    def constant(name: str, restype):
        fn = getattr(lib, name, None)
        if fn is None:
            raise Fp8GemvProvenanceError(f"libqwen_fp8_gemv has no {name}(): a pre-v{ABI_VERSION} build; rebuild it")
        fn.argtypes = []
        fn.restype = restype
        return fn()

    abi = constant("qwen_fp8_gemv_abi_version", ctypes.c_int)
    max_m = constant("qwen_fp8_gemv_max_m", ctypes.c_int)
    raw = constant("qwen_fp8_gemv_source_sha256", ctypes.c_char_p)
    source = raw.decode("ascii", "replace") if raw else None
    if (abi, max_m, source) != (ABI_VERSION, MAX_M, SOURCE_SHA256):
        raise Fp8GemvProvenanceError(f"libqwen_fp8_gemv ABI {abi} / max_m {max_m} / source {source} differs from "
                                     f"the pinned {ABI_VERSION} / {MAX_M} / {SOURCE_SHA256}")
    lib.qwen_fp8_gemv_init.argtypes = []
    lib.qwen_fp8_gemv_init.restype = ctypes.c_int
    lib.qwen_fp8_gemv_error.argtypes = [ctypes.c_int]
    lib.qwen_fp8_gemv_error.restype = ctypes.c_char_p
    fn = lib.qwen_fp8_gemv_rowwise
    # x, w, x_scale, w_scale, out, M, N, K, ldx, ldw, ldo, xs_stride, variant, stream
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_longlong] * 7 + [ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _error_text(lib, code: int) -> str:
    try:
        raw = lib.qwen_fp8_gemv_error(int(code))
        return f"{code} ({raw.decode('ascii', 'replace') if raw else '?'})"
    except Exception:  # pragma: no cover - diagnostics only
        return str(code)


# ------------------------------------------------------------------------------- numerics


def _platform_reason(device: torch.device) -> str | None:
    m = re.match(r"(\d+)\.(\d+)\.", torch.__version__)
    if not m or (int(m.group(1)), int(m.group(2))) != TORCH_PIN:
        return f"torch {torch.__version__} (the library reproduces torch 2.11's CUTLASS kernel)"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    cap = torch.cuda.get_device_capability(device)
    if tuple(cap) != CAPABILITY_PIN:
        return f"sm_{cap[0]}{cap[1]} (the library is sm_120a only)"
    return None


def _reference(qa: torch.Tensor, weight: torch.Tensor, input_scale: torch.Tensor,
               weight_scale: torch.Tensor) -> torch.Tensor:
    """Exactly the row-wise call of fp8_pertensor_linear._scaled_mm (keep the two in sync)."""
    return torch._scaled_mm(
        qa, weight.t(),
        scale_a=input_scale.reshape(1, 1).expand(qa.shape[0], 1).contiguous(),
        scale_b=weight_scale.reshape(1, -1),
        out_dtype=torch.bfloat16,
    )


def _launch(fn, qa: torch.Tensor, weight: torch.Tensor, input_scale: torch.Tensor,
            weight_scale: torch.Tensor, out: torch.Tensor) -> int:
    M, K = qa.shape
    N = weight.shape[0]
    return fn(qa.data_ptr(), weight.data_ptr(), input_scale.data_ptr(), weight_scale.data_ptr(), out.data_ptr(),
              M, N, K, K, K, N, 0, VARIANT, torch.cuda.current_stream(qa.device).cuda_stream)


def _codes(shape, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Uniform e4m3 codes over all 254 finite values (NaN 0x7F/0xFF folded to +-448)."""
    c = torch.randint(0, 256, shape, generator=gen, device=device, dtype=torch.uint8)
    return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)


_EDGE_CODES = (0x00, 0x80, 0x7E, 0xFE, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8)  # +-0 +-448 +-sub +-min +-1


def _edge_activation(M: int, K: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Edge codes in the first half, their negation in the second (exact cancellation against
    weight rows whose halves are equal)."""
    table = torch.tensor(_EDGE_CODES, dtype=torch.uint8, device=device)
    half = table[torch.randint(0, len(_EDGE_CODES), (M, K // 2), generator=gen, device=device)]
    return torch.cat([half, half ^ 0x80], dim=1).contiguous().view(FP8)


def _self_test(fn, device: torch.device) -> tuple[dict, list]:
    """Bitwise GEMV == torch._scaled_mm for both shapes and every M; (allowed, failures)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(_SEED)
    x_scales = (3.1e-3, 0.0137, 1.0, 2.0 ** -9, 0.71, 0.0422, 5.5e-4, 0.25)
    allowed, failures = {}, []
    for N, K in SHAPES:
        weight = _codes((N, K), gen, device)
        w8 = weight.view(torch.uint8)
        w8[:16, K // 2:] = w8[:16, : K // 2]  # exact-cancellation rows: second half == first half
        # wide random per-row scales (the production vector is piecewise constant, which the
        # epilogue treats no differently) plus extreme rows: bf16 overflow to inf, bf16 subnormals
        exp = torch.empty(N, device=device).uniform_(-20.0, 4.0, generator=gen)
        weight_scale = torch.exp2(exp).float().contiguous()
        weight_scale[N // 2: N // 2 + 64] = 2.0 ** 120
        weight_scale[N // 2 + 64: N // 2 + 128] = 2.0 ** -140
        bad = torch.zeros(MAX_M + 1, dtype=torch.int64, device=device)
        launch_error = {}
        for M in range(1, MAX_M + 1):
            for ai, act in enumerate(("random", "edge", "random")):
                qa = _codes((M, K), gen, device) if act == "random" else _edge_activation(M, K, gen, device)
                xs = torch.tensor(x_scales[(3 * M + ai) % len(x_scales)], dtype=torch.float32, device=device)
                out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
                rc = _launch(fn, qa, weight, xs, weight_scale, out)
                if rc:
                    launch_error[M] = rc
                    break
                ref = _reference(qa, weight, xs, weight_scale)
                bad[M] += (out.view(torch.int16) != ref.view(torch.int16)).sum()
        counts = bad.tolist()  # the one sync of this shape
        ok = frozenset(M for M in range(1, MAX_M + 1) if M not in launch_error and counts[M] == 0)
        allowed[(N, K)] = ok
        for M in range(1, MAX_M + 1):
            if M in launch_error:
                failures.append({"shape": [N, K], "M": M, "launch_error": launch_error[M]})
            elif counts[M]:
                failures.append({"shape": [N, K], "M": M, "mismatched_elements": counts[M]})
        del weight, w8, weight_scale
    torch.cuda.synchronize(device)
    return allowed, failures


def _ensure_ready_locked(device: torch.device) -> _Ready | None:
    """Process-level decision for ``device`` (caller holds _LOCK). Raises only on provenance errors."""
    global _STATE
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _DECIDED:
        decided = _DECIDED[idx]
        if decided.startswith("error: "):
            raise Fp8GemvProvenanceError(decided[len("error: "):])
        return _STATE if (_STATE is not None and _STATE.device_index == idx) else None
    if _DECIDED:  # one device per process: the library's init state is process-global
        _DECIDED[idx] = "disabled: another CUDA device already owns the GEMV"
        return None
    dev = torch.device("cuda", idx)
    _STATUS.update(requested=True, device=idx, library=os.environ.get(ENV_LIBRARY), library_sha256=LIBRARY_SHA256,
                   source_sha256=SOURCE_SHA256, abi_version=ABI_VERSION, variant=VARIANT)
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
        data = _read_pinned(path)
        t1 = time.perf_counter()
        lib, memfd = _dlopen_verified(data)
        fn = _bind(lib)
        timings.update(read_hash=round((t1 - t0) * 1e3, 1), dlopen_abi=round((time.perf_counter() - t1) * 1e3, 1))
    except Fp8GemvProvenanceError as exc:
        _DECIDED[idx] = "error: " + str(exc)
        _STATUS.update(reason=_DECIDED[idx])
        raise
    except OSError as exc:  # dlopen of verified bytes failed: deployment error as well
        _DECIDED[idx] = f"error: cannot load the verified {ENV_LIBRARY} bytes: {exc}"
        _STATUS.update(reason=_DECIDED[idx])
        raise Fp8GemvProvenanceError(_DECIDED[idx][len("error: "):]) from None
    from . import kernel_provenance

    kernel_provenance.note_loaded("qwen_fp8_gemv", {
        "provenance": "ctypes-pinned-sha256", "so_path": path, "so_sha256": LIBRARY_SHA256,
        "abi_version": ABI_VERSION, "source_sha256": SOURCE_SHA256, "loaded_from": "memfd"})
    with torch.cuda.device(dev):
        t0 = time.perf_counter()
        rc = lib.qwen_fp8_gemv_init()
        timings["init"] = round((time.perf_counter() - t0) * 1e3, 1)
        if rc:
            reason = "qwen_fp8_gemv_init failed: " + _error_text(lib, rc)
            allowed, failures = {}, []
        else:
            t0 = time.perf_counter()
            try:
                allowed, failures = _self_test(fn, dev)
                reason = None
            except RuntimeError as exc:  # asynchronous CUDA error surfaced by the sync
                allowed, failures, reason = {}, [], f"self-test raised {type(exc).__name__}: {exc}"
            timings["self_test"] = round((time.perf_counter() - t0) * 1e3, 1)
    _STATUS.update(self_test_failures=failures)
    allowed = {shape: ms for shape, ms in allowed.items() if ms}
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
    summary = {f"{n}x{k}": sorted(ms) for (n, k), ms in allowed.items()}
    if failures:
        _log().warning(f"{ENV_ENABLE}: self-test mismatches keep these on torch._scaled_mm: {failures[:8]}")
    _log().info(f"{ENV_ENABLE} enabled: libqwen_fp8_gemv sha256 {LIBRARY_SHA256[:16]}.. ABI {ABI_VERSION} "
                f"variant {VARIANT}, bit-exact self-test passed for M per shape {summary}")
    return _STATE


def ensure_ready(device: torch.device) -> bool:
    """Load + pin + self-test once per process (eager, never under capture). Raises
    Fp8GemvProvenanceError when the requested library is not the pinned build."""
    if not requested():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False  # never decide (self-test syncs) under capture; the caller stays on CUTLASS
    with _LOCK:
        return _ensure_ready_locked(device) is not None


def _layer_tensors_ok(weight, weight_scale, input_scale) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (weight, weight_scale, input_scale)):
        return False
    if not weight.is_cuda or weight.dtype is not FP8 or weight.dim() != 2 or tuple(weight.shape) not in SHAPES:
        return False
    N, K = weight.shape
    if weight.stride() != (K, 1) or weight.data_ptr() % 16:
        return False
    if weight_scale.dtype is not torch.float32 or tuple(weight_scale.shape) != (N,) or weight_scale.stride() != (1,):
        return False
    if input_scale.dtype is not torch.float32 or input_scale.numel() != 1:
        return False
    if weight_scale.device != weight.device or input_scale.device != weight.device:
        return False
    return weight_scale.data_ptr() % 4 == 0 and input_scale.data_ptr() % 4 == 0


def _check_layer_locked(st: _Ready, weight, weight_scale, input_scale) -> _Ready | None:
    """Real weights vs torch._scaled_mm for every enabled M; a mismatching M is disabled for every
    layer of this shape (the allowed set is per shape)."""
    global _STATE
    N, K = weight.shape
    ms = sorted(st.allowed.get((N, K), ()))
    if not ms:
        return st
    dev = weight.device
    t0 = time.perf_counter()
    gen = torch.Generator(device=dev)
    gen.manual_seed(_SEED + 1)
    bad = torch.zeros(MAX_M + 1, dtype=torch.int64, device=dev)
    xs = input_scale.reshape(())
    with torch.cuda.device(dev):
        for M in ms:
            qa = _codes((M, K), gen, dev)
            out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
            rc = _launch(st.fn, qa, weight, xs, weight_scale, out)
            if rc:
                _disable_locked(f"launch error {_error_text(st.lib, rc)} in the per-layer check")
                return None
            bad[M] += (out.view(torch.int16) != _reference(qa, weight, xs, weight_scale).view(torch.int16)).sum()
        counts = bad.tolist()
    timings = _STATUS.setdefault("timings_ms", {})
    timings["layer_checks"] = round(timings.get("layer_checks", 0.0) + (time.perf_counter() - t0) * 1e3, 1)
    timings["layers_checked"] = timings.get("layers_checked", 0) + 1
    failed = [M for M in ms if counts[M]]
    if not failed:
        return st
    event = {"shape": [N, K], "failed_m": failed, "mismatched_elements": [counts[M] for M in failed]}
    _STATUS["layer_mismatch_events"] = [*_STATUS["layer_mismatch_events"], event]
    allowed = dict(st.allowed)
    allowed[(N, K)] = frozenset(m for m in ms if m not in failed)
    allowed = {shape: v for shape, v in allowed.items() if v}
    _log().error(f"{ENV_ENABLE}: real-weight check mismatched {event}; those M stay on torch._scaled_mm")
    if not allowed:
        _disable_locked("real-weight check failed for every M")
        return None
    _STATE = _Ready(lib=st.lib, fn=st.fn, memfd=st.memfd, device_index=st.device_index, allowed=allowed)
    return _STATE


def _disable_locked(reason: str) -> None:
    global _STATE
    _STATE = None
    for idx, decided in list(_DECIDED.items()):
        if decided == "enabled":
            _DECIDED[idx] = "disabled: " + reason
    _STATUS.update(reason="disabled: " + reason)
    _log().warning(f"{ENV_ENABLE} disabled for the rest of the process, torch._scaled_mm stays: {reason}")


def prepare_layer(layer: Any) -> bool:
    """Decide at finalize (eager, before graph capture) whether ``layer`` may use the GEMV.

    True only for a fused row-wise projection of exactly the qualified shapes whose real weights
    pass the bitwise check; raises Fp8GemvProvenanceError if the requested library is not the
    pinned build."""
    if not requested():
        return False
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale", None)
    input_scale = getattr(layer, "input_scale", None)
    # metadata only (no CUDA call) first; then nothing below may run under capture: the probes and
    # self-tests synchronize
    declined = not _layer_tensors_ok(weight, weight_scale, input_scale) or torch.cuda.is_current_stream_capturing()
    if not declined:
        from freetoken.kernel.triton.e4m3_compat import e4m3_native
        from freetoken.kernel.triton.fp8_pertensor_linear import rowwise_scaled_mm_ok

        # the layer must take the row-wise torch._scaled_mm branch the library reproduces
        declined = not (e4m3_native() and rowwise_scaled_mm_ok())
    if declined:
        with _LOCK:
            _STATUS["layers_declined"] += 1
        return False
    with _LOCK:
        st = _ensure_ready_locked(weight.device)
        if st is not None and st.device_index == weight.device.index:
            st = _check_layer_locked(st, weight, weight_scale, input_scale)
        ok = st is not None and bool(st.allowed.get(tuple(weight.shape)))
        _STATUS["layers_enabled" if ok else "layers_declined"] += 1
    return ok


# ------------------------------------------------------------------------------- hot path


def rowwise(qa: torch.Tensor, weight: torch.Tensor, input_scale: torch.Tensor,
            weight_scale: torch.Tensor) -> torch.Tensor | None:
    """``torch._scaled_mm`` row-wise result computed by the GEMV, or None when this call is not
    covered (the caller then issues torch._scaled_mm, which gives the same bits)."""
    st = _STATE
    if st is None or qa.dim() != 2:
        return None
    M, K = qa.shape
    if M < 1 or M > MAX_M or weight.dim() != 2:
        return None
    N = weight.shape[0]
    ms = st.allowed.get((N, K))
    if ms is None or M not in ms:
        return None
    dev = qa.device
    if dev.type != "cuda" or dev.index != st.device_index:
        return None
    if (qa.dtype is not FP8 or weight.dtype is not FP8 or weight.shape[1] != K
            or weight_scale.dtype is not torch.float32 or input_scale.dtype is not torch.float32):
        return None
    if qa.stride() != (K, 1) or weight.stride() != (K, 1):
        return None
    if weight_scale.dim() != 1 or weight_scale.shape[0] != N or weight_scale.stride() != (1,) or input_scale.numel() != 1:
        return None
    if weight.device != dev or weight_scale.device != dev or input_scale.device != dev:
        return None
    if (qa.data_ptr() | weight.data_ptr()) & 15 or (weight_scale.data_ptr() | input_scale.data_ptr()) & 3:
        return None
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    rc = _launch(st.fn, qa, weight, input_scale, weight_scale, out)
    if rc:  # validation/launch error: nothing usable was produced; fall back for good
        with _LOCK:
            if _STATE is st:
                _disable_locked(f"launch error {_error_text(st.lib, rc)} for M={M} N={N} K={K}")
        return None
    return out
