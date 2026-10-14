"""Opt-in bit-exact FP8 decode GEMV (QWEN36_FP8_GEMV, freetoken/kernel/cuda_fp8_gemv.py).

The integrated row-wise W8A8 path must give exactly the bits of torch._scaled_mm, eager and under
CUDA graph capture, and must decline every call it does not cover. GPU tests need sm_120,
torch 2.11 and QWEN36_FP8_GEMV_LIBRARY naming the pinned library; otherwise they skip. The
import and provenance tests run on any machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

FP8 = torch.float8_e4m3fn
LIB = os.environ.get("QWEN36_FP8_GEMV_LIBRARY", "")


def _gpu_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability() != (12, 0):
        return "the library is sm_120a only"
    if not torch.__version__.startswith("2.11."):
        return "the library reproduces torch 2.11's CUTLASS kernel"
    if not (LIB and os.path.isfile(LIB)):
        return "QWEN36_FP8_GEMV_LIBRARY does not name the pinned library"
    return None


gpu = pytest.mark.skipif(_gpu_skip_reason() is not None, reason=str(_gpu_skip_reason()))


def _fused(N: int, K: int, seed: int):
    """A fused projection: full-range e4m3 codes, one scalar per part (piecewise-constant scale)."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    codes = torch.randint(0, 256, (N, K), generator=gen, device="cuda", dtype=torch.uint8)
    weight = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes).view(FP8)
    parts = [8192, 512, 512] if N == 9216 else [8192, 4096]
    scale = torch.cat([torch.full((p,), 0.004 * (i + 1), device="cuda") for i, p in enumerate(parts)])
    return weight, scale, torch.tensor(0.0213, device="cuda")


def _enable(monkeypatch):
    monkeypatch.setenv("QWEN36_FP8_GEMV", "1")
    import freetoken.kernel.cuda_fp8_gemv as g

    assert g.ensure_ready(torch.device("cuda", torch.cuda.current_device())), g.status()
    return g


@gpu
@pytest.mark.parametrize("N", [12288, 9216])
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_gemv_matches_scaled_mm_bitwise(monkeypatch, N: int, M: int):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    g = _enable(monkeypatch)
    served = []
    real = g.rowwise

    def counted(*args):
        out = real(*args)
        served.append(out is not None)
        return out

    monkeypatch.setattr(g, "rowwise", counted)
    weight, scale, input_scale = _fused(N, 2048, seed=N + M)
    x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16) * 2
    on = fp8_pertensor_linear(x, weight, scale, None, input_scale, False, gemv=True)
    off = fp8_pertensor_linear(x, weight, scale, None, input_scale, False, gemv=False)
    assert served == [True]
    assert torch.equal(on.view(torch.int16), off.view(torch.int16))


@gpu
def test_uncovered_calls_are_declined(monkeypatch):
    g = _enable(monkeypatch)
    weight, scale, input_scale = _fused(12288, 2048, seed=1)
    qa = torch.zeros(9, 2048, dtype=FP8, device="cuda")
    wide = torch.zeros(4, 4096, dtype=FP8, device="cuda")
    assert g.rowwise(qa, weight, input_scale, scale) is None  # M > 8 (prefill)
    assert g.rowwise(qa[:0], weight, input_scale, scale) is None  # empty batch
    assert g.rowwise(qa[:4], weight[:-16], input_scale, scale[:-16]) is None  # not a qualified shape
    assert g.rowwise(wide[:, :2048], weight, input_scale, scale) is None  # row stride != K
    assert g.rowwise(qa[:4], weight, input_scale, scale.half()) is None  # weight_scale dtype
    assert g.rowwise(qa[:4], weight, input_scale.cpu(), scale) is None  # scale on the host
    assert g.rowwise(qa[:4], weight, input_scale, scale) is not None


@gpu
@pytest.mark.parametrize("M", [1, 4, 8])
def test_graph_capture_matches_eager(monkeypatch, M: int):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    _enable(monkeypatch)
    weight, scale, input_scale = _fused(9216, 2048, seed=M)
    x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16)
    eager = fp8_pertensor_linear(x, weight, scale, None, input_scale, False, gemv=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp8_pertensor_linear(x, weight, scale, None, input_scale, False, gemv=True)  # warm-up
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fp8_pertensor_linear(x, weight, scale, None, input_scale, False, gemv=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


def test_finalize_without_the_switch_never_imports_the_module():
    script = textwrap.dedent("""
        import sys, types, torch
        from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel
        scale = torch.cat([torch.full((8192,), 0.01), torch.full((4096,), 0.02)])
        layer = types.SimpleNamespace(weight=torch.zeros(12288, 2048, dtype=torch.float8_e4m3fn),
                                      weight_scale=scale, input_scale=torch.tensor(0.05), bias=None)
        TorchFp8TensorLinearKernel().finalize(layer)
        assert layer._fp8_gemv is False
        assert "freetoken.kernel.cuda_fp8_gemv" not in sys.modules
    """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("QWEN36_FP8_GEMV", None)
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=300)


def test_only_the_pinned_library_bytes_are_accepted(tmp_path):
    import freetoken.kernel.cuda_fp8_gemv as g

    if not (LIB and os.path.isfile(LIB)):
        pytest.skip("QWEN36_FP8_GEMV_LIBRARY does not name the pinned library")
    assert len(g._read_pinned(LIB)) > 0
    tampered = tmp_path / "libqwen_fp8_gemv.so"
    tampered.write_bytes(open(LIB, "rb").read() + b"\0")
    with pytest.raises(g.Fp8GemvProvenanceError):
        g._read_pinned(str(tampered))
    with pytest.raises(g.Fp8GemvProvenanceError):
        g._read_pinned("relative/libqwen_fp8_gemv.so")
