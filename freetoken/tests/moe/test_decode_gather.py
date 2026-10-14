"""The opt-in decode gather (QWEN36_DECODE_GATHER=tuned) must move exactly the same bytes as the
default fused multi-bank copy, eager and inside a captured CUDA graph, for every miss count
(including zero), and leave every other slot untouched. The env switch is fail-closed.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.decode_gather import TUNED_DECODE_GATHER, decode_gather_config
from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# 6 banks with mixed 16B-aligned per-row sizes, including rows shorter than one 16 KiB chunk and
# rows that end in a partial chunk.
FEATS = [40976, 512, 16, 16400, 4096, 272]


def test_env_switch(monkeypatch):
    monkeypatch.delenv("QWEN36_DECODE_GATHER", raising=False)
    assert decode_gather_config() is None
    for value in ("", "0", "off", "default"):
        monkeypatch.setenv("QWEN36_DECODE_GATHER", value)
        assert decode_gather_config() is None
    monkeypatch.setenv("QWEN36_DECODE_GATHER", " Tuned ")
    assert decode_gather_config() == TUNED_DECODE_GATHER
    for value in ("1", "true", "fast"):
        monkeypatch.setenv("QWEN36_DECODE_GATHER", value)
        with pytest.raises(ValueError):
            decode_gather_config()


def _build_cache(num_layers, num_experts, cache_size):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format="mxfp4_triton",
    )
    schema = _BANK_SCHEMAS["mxfp4_triton"]
    sources = {
        name: list(torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev)
                   .split(num_experts))
        for name, feat in zip(schema, FEATS)
    }
    cache.set_bank_sources(sources)
    return cache


def _stage(cache, layer_id, n, gen):
    dev = cache.evict_slots.device
    cache._pending_src_layer, cache._pending_whole_layer = layer_id, False
    cache.num_indices.fill_(n)
    if n:
        cache.evict_slots[:n] = torch.randperm(cache.cache_size, generator=gen)[:n].to(dev, torch.int32)
        cache.src_indices[:n] = torch.randperm(cache.num_experts, generator=gen)[:n].to(dev, torch.int32)


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("num_indices", [0, 1, 3, 8, 13])
def test_tuned_gather_matches_default(num_indices):
    num_layers, num_experts, cache_size = 4, 16, 48
    cache = _build_cache(num_layers, num_experts, cache_size)
    assert cache._copy_fused_ok
    gen = torch.Generator().manual_seed(1234 + num_indices)
    for layer_id in (0, 3):
        _stage(cache, layer_id, num_indices, gen)
        results = []
        for config in (None, TUNED_DECODE_GATHER):
            for _, c in cache.banks:
                c.fill_(0xA5)
            cache._decode_gather = config
            cache.copy_missing()
            torch.cuda.synchronize()
            results.append([c.clone() for _, c in cache.banks])
        for b, (ref, got) in enumerate(zip(*results)):
            assert torch.equal(ref, got), f"bank {b} (feat={FEATS[b]}) tuned != default at n={num_indices}"


@CUDA
@pytest.mark.slow
def test_tuned_gather_graph_replay():
    num_layers, num_experts, cache_size = 4, 16, 48
    cache = _build_cache(num_layers, num_experts, cache_size)
    gen = torch.Generator().manual_seed(7)
    graphs = {}
    for config in (None, TUNED_DECODE_GATHER):
        cache._decode_gather = config
        _stage(cache, 2, 1, gen)
        cache.copy_missing()  # eager warm-up (module load) before capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cache.copy_missing()
        graphs[config] = g
    for n in (0, 2, 8, 1):
        _stage(cache, 2, n, gen)
        results = []
        for config in (None, TUNED_DECODE_GATHER):
            for _, c in cache.banks:
                c.fill_(0x5A)
            graphs[config].replay()
            torch.cuda.synchronize()
            results.append([c.clone() for _, c in cache.banks])
        for b, (ref, got) in enumerate(zip(*results)):
            assert torch.equal(ref, got), f"bank {b} graph replay tuned != default at n={n}"
