"""D5: one-kernel decode epilogue of the Qwen3.5 MoE block (``QWEN36_DECODE_SHARED_EPILOGUE=1``,
default off).

Replaces three bf16 TensorIterator kernels per decode layer -- ``torch.sigmoid(gate)``,
``shared * sig`` and ``routed + prod`` -- by one launch with the same operations and roundings:

    sig  = bf16_rn(1 / (1 + expf(-f32(gate))))       # IEEE div.rn, libdevice expf (torch: std::exp)
    prod = bf16_rn(f32(shared) * f32(sig))           # one rounded fp32 multiply
    out  = bf16_rn(f32(routed) + f32(prod))          # one rounded fp32 add

The multiply and the add are single IEEE operations, separated by the bf16 rounding (no FMA can
form across it), so they match torch by construction. The sigmoid is the only transcendental: its
input is one bf16 value, so the process self-test compares all 65536 bf16 inputs bitwise against
``torch.sigmoid`` (plus a random full-epilogue comparison) before the first fused call. Any
mismatch keeps the torch path for the life of the process; a call that arrives under graph
capture before the test ran also keeps the torch path.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENV = "QWEN36_DECODE_SHARED_EPILOGUE"


def enabled() -> bool:
    raw = os.environ.get(ENV, "0")
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV}={raw!r}: expected unset, 0 or 1")
    return raw == "1"


@triton.jit
def _sigmoid_bf16(g):
    one = tl.full(g.shape, 1.0, tl.float32)
    return tl.math.div_rn(one, one + libdevice.exp(-g.to(tl.float32))).to(tl.bfloat16)


@triton.jit
def _epilogue_kernel(routed_ptr, shared_ptr, gate_ptr, out_ptr, H, stride_r, stride_s, stride_g, stride_o,
                     BLOCK: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    hm = h < H
    sig = _sigmoid_bf16(tl.load(gate_ptr + m * stride_g + tl.zeros((BLOCK,), tl.int32))).to(tl.float32)
    s = tl.load(shared_ptr + m * stride_s + h, mask=hm, other=0.0).to(tl.float32)
    r = tl.load(routed_ptr + m * stride_r + h, mask=hm, other=0.0).to(tl.float32)
    prod = (s * sig).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + m * stride_o + h, (r + prod).to(tl.bfloat16), mask=hm)


@triton.jit
def _sigmoid_table_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    tl.store(out_ptr + i, _sigmoid_bf16(tl.load(x_ptr + i, mask=mask, other=0.0)), mask=mask)


def _launch(routed, shared, gate, out):
    M, H = routed.shape
    BLOCK = min(1024, triton.next_power_of_2(H))
    _epilogue_kernel[(M, triton.cdiv(H, BLOCK))](
        routed, shared, gate, out, H, routed.stride(0), shared.stride(0), gate.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=4,
    )


def reference(routed, shared, gate):
    return routed + shared * torch.sigmoid(gate)


def _bits(t):
    return t.contiguous().view(torch.int16)


def self_test(device) -> str | None:
    """None when the fused epilogue is bit-identical to ``reference`` on this stack, else the reason."""
    codes = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
    finite = torch.isfinite(codes)
    table = torch.empty_like(codes)
    _sigmoid_table_kernel[(triton.cdiv(codes.numel(), 1024),)](codes, table, codes.numel(), BLOCK=1024)
    want = torch.sigmoid(codes)
    if not torch.equal(_bits(table[finite]), _bits(want[finite])) or \
            not torch.equal(torch.isnan(table), torch.isnan(want)):
        bad = int((_bits(table) != _bits(want))[finite].sum())
        return f"sigmoid differs from torch on {bad} finite bf16 inputs"
    g = torch.Generator(device=device).manual_seed(0x5EED)
    for M, H in ((1, 2048), (3, 2048), (8, 2048), (1, 1000)):
        for scale in (1.0, 1e-30, 1e30):
            routed = (torch.randn((M, H), device=device, generator=g) * scale).to(torch.bfloat16)
            shared = (torch.randn((M, H), device=device, generator=g) * scale).to(torch.bfloat16)
            gate = (torch.randn((M, 1), device=device, generator=g) * 8).to(torch.bfloat16)
            shared[0, :4] = torch.tensor([0.0, -0.0, 1e-40, -1e-40], device=device)
            out = torch.empty_like(routed)
            _launch(routed, shared, gate, out)
            want = reference(routed, shared, gate)
            ok = torch.isfinite(want)
            if not torch.equal(_bits(out)[ok], _bits(want)[ok]) or not torch.equal(torch.isnan(out), torch.isnan(want)):
                return f"epilogue differs from torch at M={M} H={H} scale={scale}"
    return None


class SharedEpilogue:
    """Process-wide decision (lazy, eager-only) plus the fused call."""

    def __init__(self):
        self.decided: bool | None = None
        self.reason: str | None = None

    def usable(self, routed: torch.Tensor) -> bool:
        if self.decided is None:
            if routed.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
                return False
            self.reason = self_test(routed.device)
            self.decided = self.reason is None
            if not self.decided:
                from freetoken.utils import init_logger

                init_logger(__name__).warning(f"{ENV}=1 keeps the torch epilogue: {self.reason}")
        return self.decided

    def __call__(self, routed: torch.Tensor, shared: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``routed + shared * torch.sigmoid(gate)`` for bf16 [M, H], [M, H], [M, 1]."""
        if (routed.dtype != torch.bfloat16 or shared.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16
                or routed.dim() != 2 or routed.shape != shared.shape or tuple(gate.shape) != (routed.shape[0], 1)
                or routed.stride(-1) != 1 or shared.stride(-1) != 1 or not self.usable(routed)):
            return reference(routed, shared, gate)
        out = torch.empty_like(routed)
        _launch(routed, shared, gate, out)
        return out


EPILOGUE = SharedEpilogue()
