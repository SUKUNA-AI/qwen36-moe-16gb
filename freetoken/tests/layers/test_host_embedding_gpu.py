"""SM120 owner gate for QWEN36_HOST_EMBEDDINGS: host-table lookups equal the GPU path byte for byte.

The reference is the unchanged GPU-resident VocabParallelEmbedding (JIT ``indexing``) over the
same table, at the released Qwen3.6-35B-A3B shape. Run without a profiler under the GPU lock.
"""
import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers import VocabParallelEmbedding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="owner CUDA/SM120 required")

VOCAB, HIDDEN = 248320, 2048


def _table(device):
    # random bf16 bit patterns, plus rows of the special values a float path would canonicalize
    g = torch.Generator(device="cpu").manual_seed(1234)
    bits = torch.randint(-(2**15), 2**15, (VOCAB, HIDDEN), dtype=torch.int16, generator=g)
    specials = torch.tensor([0x7FC1, 0x7F81, -0x7F, 0x7F80, -0x80, 0x0001, -0x8000, 0x0000], dtype=torch.int16)
    # NaN payloads of both signs, +-inf, a subnormal, -0 and +0
    bits[:8] = specials.repeat_interleave(HIDDEN // 8)
    bits[-1] = specials.repeat(HIDDEN // 8)
    return bits.to(device).view(torch.bfloat16)


def _layer(weight):
    layer = VocabParallelEmbedding(VOCAB, HIDDEN)
    layer.weight = weight
    return layer


@pytest.fixture(scope="module")
def layers():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda")
    table = _table(device)
    reference = _layer(table)
    host = _layer(table.clone())
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    nbytes = host.move_to_host()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    freed = before - torch.cuda.memory_allocated()
    return reference, host, nbytes, freed


def _same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_move_releases_vram_and_keeps_a_pinned_table(layers):
    reference, host, nbytes, freed = layers
    assert nbytes == VOCAB * HIDDEN * 2
    assert freed == nbytes
    assert host.weight.device.type == "cpu" and host.weight.is_pinned()
    assert host.compute_device == reference.weight.device
    assert torch.equal(host.weight.view(torch.int16), reference.weight.cpu().view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [1, 2, 7, 64, 2048, 8192, 32768])
def test_lookup_bytes_match_gpu_path(layers, dtype, rows):
    reference, host, _, _ = layers
    g = torch.Generator(device="cpu").manual_seed(rows)
    ids = torch.randint(0, VOCAB, (rows,), generator=g)
    ids[: min(rows, 4)] = torch.tensor([0, VOCAB - 1, 7, 0])[: min(rows, 4)]
    ids = ids.to(device="cuda", dtype=dtype)
    expected = reference.forward(ids)
    actual = host.forward(ids)
    torch.cuda.synchronize()
    assert actual.device == expected.device
    assert _same_bytes(actual, expected)


def test_every_row_once(layers):
    reference, host, _, _ = layers
    for start in range(0, VOCAB, 65536):
        ids = torch.arange(start, min(start + 65536, VOCAB), device="cuda", dtype=torch.int32)
        assert _same_bytes(host.forward(ids), reference.forward(ids))


def test_out_of_range_ids_store_zero_rows(layers):
    _, host, _, _ = layers
    ids = torch.tensor([VOCAB, -1, 3, 2**31 - 1], device="cuda", dtype=torch.int64)
    out = host.forward(ids)
    assert torch.equal(out[[0, 1, 3]].view(torch.int16), torch.zeros((3, HIDDEN), dtype=torch.int16, device="cuda"))
    assert torch.equal(out[2].view(torch.int16), host.weight[3].cuda().view(torch.int16))


@pytest.mark.parametrize("batch", [1, 4])
def test_cuda_graph_capture_replay(layers, batch):
    reference, host, _, _ = layers
    ids = torch.zeros(batch, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # warm the Triton compile outside the capture
            host.forward(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = host.forward(ids)
    g = torch.Generator(device="cpu").manual_seed(batch)
    for step in range(200):
        ids.copy_(torch.randint(0, VOCAB, (batch,), generator=g, dtype=torch.int32))
        graph.replay()
        assert _same_bytes(out, reference.forward(ids)), step
