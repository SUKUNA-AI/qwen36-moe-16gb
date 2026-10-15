"""Opt-in bit-exact decode kernels: o-projection (QWEN36_OPROJ_GEMV, kernel/cuda_oproj_gemv.py) and
NVFP4 MoE decode GEMV (QWEN36_MOE_DECODE_CUDA, kernel/cuda_moe_decode.py).

The integrated paths must give exactly the bits of the paths they replace (cuBLASLt torch._scaled_mm,
Triton _decode_nvfp4_marlin_kernel), eager and under CUDA graph capture, and must decline every call
they do not cover. GPU tests need sm_120, torch 2.11 and the *_LIBRARY variables naming the pinned
libraries; otherwise they skip. The import and provenance tests run on any machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types

import pytest
import torch

FP8 = torch.float8_e4m3fn
OPROJ_LIB = os.environ.get("QWEN36_OPROJ_GEMV_LIBRARY", "")
MOE_LIB = os.environ.get("QWEN36_MOE_DECODE_LIBRARY", "")


def _gpu_skip_reason(lib: str, env: str) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability() != (12, 0):
        return "the libraries are sm_120a only"
    if not torch.__version__.startswith("2.11."):
        return "the libraries reproduce torch 2.11 / Triton 3.6 kernels"
    if not (lib and os.path.isfile(lib)):
        return f"{env} does not name the pinned library"
    return None


gpu_oproj = pytest.mark.skipif(_gpu_skip_reason(OPROJ_LIB, "QWEN36_OPROJ_GEMV_LIBRARY") is not None,
                               reason=str(_gpu_skip_reason(OPROJ_LIB, "QWEN36_OPROJ_GEMV_LIBRARY")))
gpu_moe = pytest.mark.skipif(_gpu_skip_reason(MOE_LIB, "QWEN36_MOE_DECODE_LIBRARY") is not None,
                             reason=str(_gpu_skip_reason(MOE_LIB, "QWEN36_MOE_DECODE_LIBRARY")))


# ------------------------------------------------------------------------------------------ o-proj


def _oproj_layer(seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    codes = torch.randint(0, 256, (2048, 4096), generator=gen, device="cuda", dtype=torch.uint8)
    weight = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes).view(FP8)
    return types.SimpleNamespace(weight=weight, weight_scale=torch.full((2048,), 0.0031, device="cuda"),
                                 input_scale=torch.tensor(0.0213, device="cuda"), bias=None)


def _oproj_enable(monkeypatch, seed=0):
    monkeypatch.setenv("QWEN36_OPROJ_GEMV", "1")
    import freetoken.kernel.cuda_oproj_gemv as g

    layer = _oproj_layer(seed)
    assert g.prepare_layer(layer), g.status()
    return g, layer


@gpu_oproj
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_oproj_matches_scaled_mm_bitwise(monkeypatch, M: int):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    g, layer = _oproj_enable(monkeypatch, seed=M)
    served = []
    real = g.tensorwise

    def counted(*args):
        out = real(*args)
        served.append(out is not None)
        return out

    monkeypatch.setattr(g, "tensorwise", counted)
    x = torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16) * 3
    on = fp8_pertensor_linear(x, layer.weight, layer.weight_scale, None, layer.input_scale, True, oproj=True)
    off = fp8_pertensor_linear(x, layer.weight, layer.weight_scale, None, layer.input_scale, True, oproj=False)
    assert served == [True]
    assert torch.equal(on.view(torch.int16), off.view(torch.int16))


@gpu_oproj
def test_oproj_uncovered_calls_are_declined(monkeypatch):
    g, layer = _oproj_enable(monkeypatch)
    w, sa, sb = layer.weight, layer.input_scale.reshape(()), layer.weight_scale[0].reshape(())
    qa = torch.zeros(9, 4096, dtype=FP8, device="cuda")
    wide = torch.zeros(4, 4112, dtype=FP8, device="cuda")
    assert g.tensorwise(qa, w, sa, sb) is None  # M > 8 (prefill)
    assert g.tensorwise(qa[:0], w, sa, sb) is None  # empty batch
    assert g.tensorwise(qa[:4, :2048], w[:, :2048].contiguous(), sa, sb) is None  # not the qualified shape
    assert g.tensorwise(wide[:, 8:4104], w, sa, sb) is None  # misaligned rows
    assert g.tensorwise(qa[:4], w, sa.half(), sb) is None  # scale dtype
    assert g.tensorwise(qa[:4], w, sa.cpu(), sb) is None  # scale on the host
    assert g.tensorwise(qa[:4], w, sa, sb) is not None
    assert g.tensorwise(wide[:, :4096], w, sa, sb) is not None  # padded row stride is fine


@gpu_oproj
@pytest.mark.parametrize("M", [1, 4, 8])
def test_oproj_graph_capture_matches_eager(monkeypatch, M: int):
    from freetoken.kernel.triton.fp8_pertensor_linear import uniform_scaled_mm

    _, layer = _oproj_enable(monkeypatch, seed=10 + M)
    qa = (torch.randn(M, 4096, device="cuda") * 20).to(FP8)
    eager = uniform_scaled_mm(qa, layer.weight, layer.input_scale, layer.weight_scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        uniform_scaled_mm(qa, layer.weight, layer.input_scale, layer.weight_scale, oproj=True)  # warm-up
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = uniform_scaled_mm(qa, layer.weight, layer.input_scale, layer.weight_scale, oproj=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


def test_oproj_finalize_without_the_switch_never_imports_the_module():
    script = textwrap.dedent("""
        import sys, types, torch
        from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel
        layer = types.SimpleNamespace(weight=torch.zeros(2048, 4096, dtype=torch.float8_e4m3fn),
                                      weight_scale=torch.full((2048,), 0.01), input_scale=torch.tensor(0.05),
                                      bias=None)
        TorchFp8TensorLinearKernel().finalize(layer)
        assert layer._oproj_gemv is False and layer._fp8_uniform_scale is True
        assert "freetoken.kernel.cuda_oproj_gemv" not in sys.modules
    """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("QWEN36_OPROJ_GEMV", None)
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=300)


# ------------------------------------------------------------------------------------------ MoE


def _moe_case(M: int, top_k: int = 8, S: int = 16, seed: int = 0):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    H, I = 2048, 512

    def scales(shape):
        c = torch.randint(0, 256, shape, generator=gen, device="cuda", dtype=torch.uint8)
        return torch.where((c & 0x7F) == 0x7F, c ^ 1, c).view(FP8)

    banks = (torch.randint(0, 256, (S, 2 * I, H // 2), generator=gen, device="cuda", dtype=torch.uint8),
             scales((S, 2 * I, H // 16)), torch.rand((S, 2 * I), generator=gen, device="cuda") * 1e-3,
             torch.randint(0, 256, (S, H, I // 2), generator=gen, device="cuda", dtype=torch.uint8),
             scales((S, H, I // 16)), torch.rand((S, H), generator=gen, device="cuda") * 1e-3)
    ids = torch.stack([torch.randperm(S, generator=gen, device="cuda")[:top_k] for _ in range(M)]).to(torch.int32)
    w = torch.rand((M, top_k), generator=gen, device="cuda")
    x = torch.randn((M, H), generator=gen, device="cuda").to(torch.bfloat16)
    return x, banks, ids.contiguous(), w


def _moe_enable(monkeypatch):
    monkeypatch.setenv("QWEN36_MOE_DECODE_CUDA", "1")
    import freetoken.kernel.cuda_moe_decode as m

    assert m.prepare(torch.device("cuda", torch.cuda.current_device())), m.status()
    return m


@gpu_moe
@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6, 7, 8])
def test_moe_decode_matches_triton_bitwise(monkeypatch, M: int):
    import freetoken.moe.fused_nvfp4 as fn

    m = _moe_enable(monkeypatch)
    layer = types.SimpleNamespace(layer_id=0)
    x, banks, ids, w = _moe_case(M, seed=M)
    gemm = m.gemm_for_layer(layer, x, banks, w, ids, "silu", 1.702, float("inf"), False)
    assert gemm is not None and layer._moe_decode_cuda is True, m.status()
    ref = fn.fused_experts_decode_nvfp4_marlin(x, *banks, w, ids, "silu", False)
    out = fn._fused_experts_decode_nvfp4(gemm, x, *banks, w, ids, "silu", False)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@gpu_moe
def test_moe_decode_declines_and_graph(monkeypatch):
    import freetoken.moe.fused_nvfp4 as fn

    m = _moe_enable(monkeypatch)
    x, banks, ids, w = _moe_case(9, seed=3)
    layer = types.SimpleNamespace(layer_id=1)
    assert m.gemm_for_layer(layer, x, banks, w, ids, "silu", 1.702, float("inf"), False) is None  # M > MAX_M
    layer = types.SimpleNamespace(layer_id=2)
    assert m.gemm_for_layer(layer, x[:1], banks, w[:1], ids[:1], "silu", 1.702, float("inf"), True) is None
    x, banks, ids, w = _moe_case(1, seed=4)
    layer = types.SimpleNamespace(layer_id=3)
    gemm = m.gemm_for_layer(layer, x, banks, w, ids, "silu", 1.702, float("inf"), False)
    assert gemm is not None
    eager = fn.fused_experts_decode_nvfp4_marlin(x, *banks, w, ids, "silu", False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn._fused_experts_decode_nvfp4(gemm, x, *banks, w, ids, "silu", False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fn._fused_experts_decode_nvfp4(gemm, x, *banks, w, ids, "silu", False)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


def test_moe_decode_without_the_switch_never_imports_the_module():
    script = textwrap.dedent("""
        import sys
        import freetoken.layers.quantization.moe.nvfp4 as nv
        assert nv._MOE_DECODE_CUDA is False
        assert "freetoken.kernel.cuda_moe_decode" not in sys.modules
    """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("QWEN36_MOE_DECODE_CUDA", None)
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=300)


# ------------------------------------------------------------------------------------------ provenance


@pytest.mark.parametrize("module, env", [("freetoken.kernel.cuda_oproj_gemv", "QWEN36_OPROJ_GEMV_LIBRARY"),
                                         ("freetoken.kernel.cuda_moe_decode", "QWEN36_MOE_DECODE_LIBRARY")])
def test_only_the_pinned_library_bytes_are_accepted(tmp_path, module: str, env: str):
    import importlib

    from freetoken.kernel import pinned_ctypes

    mod = importlib.import_module(module)
    lib = os.environ.get(env, "")
    if not (lib and os.path.isfile(lib)):
        pytest.skip(f"{env} does not name the pinned library")
    assert len(pinned_ctypes.read_pinned(lib, mod.LIBRARY_SHA256, env)) > 0
    tampered = tmp_path / "lib.so"
    tampered.write_bytes(open(lib, "rb").read() + b"\0")
    with pytest.raises(pinned_ctypes.PinnedLibraryError):
        pinned_ctypes.read_pinned(str(tampered), mod.LIBRARY_SHA256, env)


def test_pinned_reader_refuses_relative_missing_and_foreign_files(tmp_path):
    from freetoken.kernel import pinned_ctypes

    foreign = tmp_path / "foreign.so"
    foreign.write_bytes(b"not the pinned build")
    link = tmp_path / "link.so"
    link.symlink_to(foreign)
    sha = "0" * 64
    for path in ("relative/lib.so", str(tmp_path / "missing.so"), str(foreign), str(link), None):
        with pytest.raises(pinned_ctypes.PinnedLibraryError):
            pinned_ctypes.read_pinned(path, sha, "QWEN36_TEST_LIBRARY")
    import hashlib

    good = hashlib.sha256(foreign.read_bytes()).hexdigest()
    assert pinned_ctypes.read_pinned(str(foreign), good, "QWEN36_TEST_LIBRARY") == foreign.read_bytes()
    with pytest.raises(pinned_ctypes.PinnedLibraryError):  # never through a symlink
        pinned_ctypes.read_pinned(str(link), good, "QWEN36_TEST_LIBRARY")
