"""Owner-only regression for the batch-memcpy self-probe; never run by CPU-checks.sh.

The probe zero-fills its destination on the current stream and copies on a side stream. With the
current stream busy (a prefill's GEMMs, here a matmul chain), an unordered copy lands before the
zero-fill and the probe fails, which disables hit-D2D (T1) for the whole process.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires owner CUDA/SM120 stack")
def test_probe_passes_behind_a_busy_current_stream():
    from freetoken.kernel.batch_memcpy import _jit_batch_memcpy_module, _probe, load_batch_memcpy

    fn = _jit_batch_memcpy_module().batch_memcpy
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(50):
        for _ in range(4):
            a = (a @ a).clamp_(-1, 1)
        _probe(fn)
    for _ in range(4):
        a = (a @ a).clamp_(-1, 1)
    assert callable(load_batch_memcpy())
    torch.cuda.synchronize()
