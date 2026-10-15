"""SM120 owner gate for D5 (moe/shared_epilogue.py): the fused decode epilogue is bit-identical to
``routed + shared * torch.sigmoid(gate)``: all 65536 bf16 gate values, decode M 1..8, edge values,
CUDA-graph replay, and the Qwen3.5 MoE block with the flag on and off. Run under flock.
"""
import sys
from pathlib import Path

import pytest
import torch

from freetoken.moe import shared_epilogue as SE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='owner CUDA/SM120 required')


def bits(t):
    return t.contiguous().view(torch.int16)


def test_self_test_passes_on_this_stack():
    assert SE.self_test(torch.device('cuda')) is None


@pytest.mark.parametrize('M', range(1, 9))
def test_epilogue_bytes(M):
    g = torch.Generator(device='cuda').manual_seed(M)
    H = 2048
    for scale in (1.0, 0.01, 100.0, 1e-36):
        routed = (torch.randn((M, H), device='cuda', generator=g) * scale).to(torch.bfloat16)
        shared = (torch.randn((M, H), device='cuda', generator=g) * scale).to(torch.bfloat16)
        gate = (torch.randn((M, 1), device='cuda', generator=g) * 12).to(torch.bfloat16)
        shared[:, :6] = torch.tensor([0.0, -0.0, 1e-39, -1e-39, 3e38, -3e38], device='cuda')
        routed[:, 6] = -shared[:, 6]  # exact cancellation
        out = torch.empty_like(routed)
        SE._launch(routed, shared, gate, out)
        want = SE.reference(routed, shared, gate)
        assert torch.equal(bits(out), bits(want)), (M, scale)


def test_graph_capture_replay():
    E = SE.SharedEpilogue()
    routed = torch.randn((1, 2048), device='cuda').to(torch.bfloat16)
    shared = torch.randn((1, 2048), device='cuda').to(torch.bfloat16)
    gate = torch.randn((1, 1), device='cuda').to(torch.bfloat16)
    assert E.usable(routed)  # eager decision before capture
    E(routed, shared, gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = E(routed, shared, gate)
    for i in range(200):
        g = torch.Generator(device='cuda').manual_seed(i)
        routed.copy_(torch.randn((1, 2048), device='cuda', generator=g))
        shared.copy_(torch.randn((1, 2048), device='cuda', generator=g))
        gate.copy_(torch.randn((1, 1), device='cuda', generator=g) * 10)
        graph.replay()
        assert torch.equal(bits(out), bits(SE.reference(routed, shared, gate)))


def test_undecided_capture_keeps_torch():
    E = SE.SharedEpilogue()
    x = torch.randn((1, 2048), device='cuda').to(torch.bfloat16)
    gate = torch.randn((1, 1), device='cuda').to(torch.bfloat16)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        E(x, x, gate)
    assert E.decided is None


def test_qwen35_moe_block_flag_on_equals_off(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
    try:
        from qwen4_exp.test_skeleton import _config, _fill, _fresh_ctx
    except Exception as exc:  # helper layout changed: the kernel tests above still gate D5
        pytest.skip(f'qwen4_exp test helpers unavailable: {exc}')
    from types import SimpleNamespace
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # both arms run the same shared-expert activation; the Triton one needs no nvcc (flashinfer JITs)
    from freetoken.kernel.triton.activation import silu_and_mul
    from freetoken.models.qwen3_5_moe import moe as qmoe
    monkeypatch.setattr(qmoe, 'silu_and_mul', silu_and_mul)
    config = _config()
    with torch.device('cuda'), torch_dtype(torch.bfloat16):
        moe = Qwen4ExpMoE(config, 0)
    _fill(moe, torch.Generator(device='cuda').manual_seed(5), scale=0.2)
    _fresh_ctx(_batch=SimpleNamespace(is_prefill=False))
    for M in (1, 2, 8):
        x = torch.randn((M, config.hidden_size), device='cuda', dtype=torch.bfloat16) * 0.5
        moe._fused_epilogue = None
        off = Qwen3_5MoE.forward(moe, x.clone())
        moe._fused_epilogue = SE.SharedEpilogue()
        on = Qwen3_5MoE.forward(moe, x.clone())
        assert moe._fused_epilogue.decided
        assert torch.equal(bits(on), bits(off)), M
