"""Owner-only D6 gate (decode warm-up from the prefill double buffer); never run by CPU-checks.sh.

The decode region starts FULL with experts of every layer, so each install must evict. With the
snapshot-based hit-D2D split, evicting an expert of a later layer of the same chunk would leave its
buffer row stale; the consumer bytes of every layer therefore must still equal the host banks. After
the chunk, the warmed experts must be resident with their host bytes, the maps mutually inverse, and
a decode ensure of them must not miss.
"""
import pytest
import torch
from freetoken.moe.offload_cache import (
    PREFILL_EPOCH_INVALIDATE_ENV, PREFILL_SELECTED_ENV, PREFILL_TMA_GATHER_ENV, PREFILL_WARM_DECODE_ENV,
    OffloadMoeCache, _BANK_SCHEMAS,
)

E, L, N = 32, 6, 6
ROW_BYTES = {"gate_up_packed": 1 << 16, "gate_up_scale": 4096, "gate_up_global": 512,
             "down_packed": 1 << 15, "down_scale": 2048, "down_global": 1024}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires owner CUDA/SM120 stack")


def ordered_probe(fn):
    src = torch.arange(16, dtype=torch.uint8).pin_memory()
    dst = torch.zeros(16, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    fn(torch.tensor([dst.data_ptr()]), torch.tensor([src.data_ptr()]), torch.tensor([16]), stream.cuda_stream)
    stream.synchronize()
    if not torch.equal(dst.cpu(), src):
        raise RuntimeError("cudaMemcpyBatchAsync probe copied wrong bytes")


def make(monkeypatch, env):
    from freetoken.kernel import batch_memcpy

    monkeypatch.setattr(batch_memcpy, "_probe", ordered_probe)
    for name in (PREFILL_EPOCH_INVALIDATE_ENV, PREFILL_SELECTED_ENV, PREFILL_TMA_GATHER_ENV, PREFILL_WARM_DECODE_ENV):
        monkeypatch.setenv(name, str(env.get(name, 0)))
    monkeypatch.setenv("QWEN36_PREFILL_SYNCFREE_INVALIDATE", "1")
    g = torch.Generator().manual_seed(3)
    sources = {k: [torch.randint(0, 256, (E, ROW_BYTES[k]), dtype=torch.uint8, generator=g).pin_memory()
                   for _ in range(L)] for k in _BANK_SCHEMAS["nvfp4"]}
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=2 * E + 3 * E, device=torch.device("cuda"),
                            prefill_overlap=True, prefill_hit_d2d=True, quant_format="nvfp4")
    cache.set_bank_sources(sources)
    assert cache._hit_d2d_usable()
    return cache


def fill_decode_region(cache, epoch):
    """Every decode slot holds an expert, spread over all layers, with scrambled usage."""
    cache.slot_for_id.fill_(-1)
    cache.id_of_slot.fill_(-1)
    g = torch.Generator().manual_seed(100 + epoch)
    flat = torch.randperm(L * E, generator=g)[: cache.cache_size - 2 * E]
    usage = torch.randperm(cache.cache_size, generator=g) + 1
    cache.usage.copy_(usage)
    cache.step.fill_(int(usage.max()) + 1)
    for i, f in enumerate(flat.tolist()):
        slot = 2 * E + i
        layer, e = divmod(f, E)
        cache.slot_for_id[layer, e] = slot
        cache.id_of_slot[slot] = f
        for host, bank in cache.banks:
            bank[slot].copy_(host[layer][e])


def routing(m, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(L, m, E, generator=g) + torch.linspace(3.0, 0.0, E)
    return [s.topk(8, dim=1).indices.to(torch.int32).cuda() for s in scores]


@cuda
@pytest.mark.parametrize("env", ({}, {PREFILL_EPOCH_INVALIDATE_ENV: 1}, {PREFILL_SELECTED_ENV: 512},
                                 {PREFILL_TMA_GATHER_ENV: 1}), ids=("hitd2d", "epoch", "c1", "c2"))
def test_d6_warm_install_keeps_bytes_and_maps(monkeypatch, env):
    cache = make(monkeypatch, {**env, PREFILL_WARM_DECODE_ENV: N})
    for epoch in range(8):
        fill_decode_region(cache, epoch)
        m = (3, 17, 60)[epoch % 3]
        topk = routing(m, epoch)
        cache.select_prefill_route(m)
        assert cache.prefill_warm_active
        before = int(cache.prefill_warm_rows)
        for layer in range(L):
            if cache.prefill_selected_active:
                if layer == 0:
                    cache.begin_prefill()
                views = cache.selected_prefill_layer(layer, topk[layer].clone())
                rows = torch.unique(topk[layer]).cpu()
            else:
                if layer == 0:
                    cache.begin_prefill()
                cache.prefetch_prefill_layer(layer)
                cache.prefetch_prefill_layer(layer + 1)
                views = cache.wait_prefill_layer(layer)
                rows = torch.arange(E)
            for (host, _), view in zip(cache.banks, views):
                assert torch.equal(view.cpu()[rows], host[layer][rows]), f"epoch {epoch} layer {layer}: buffer bytes"
            cache.warm_decode_layer(layer, topk[layer])
            cache.release_prefill_layer(layer)
        torch.cuda.synchronize()
        assert int(cache.prefill_warm_rows) > before, "nothing was installed: the gate did not exercise D6"
        sf, ids = cache.slot_for_id.cpu(), cache.id_of_slot.cpu()
        mapped = sf[sf >= 0]
        assert not (mapped < 2 * E).any(), "a map points into the double buffers"
        assert mapped.unique().numel() == mapped.numel()
        for layer in range(L):
            for e in range(E):
                s = int(sf[layer, e])
                if s >= 0:
                    assert int(ids[s]) == layer * E + e
            counts = torch.bincount(topk[layer].reshape(-1).cpu().long(), minlength=E)
            top = (counts * E + (E - 1 - torch.arange(E))).topk(N).indices
            for e in top[counts[top] > 0].tolist():
                s = int(sf[layer, e])
                assert s >= 2 * E, f"epoch {epoch} layer {layer}: warmed expert {e} not resident"
                for host, bank in cache.banks:
                    assert torch.equal(bank[s].cpu(), host[layer][e]), "warmed slot holds wrong bytes"
            want = top[counts[top] > 0].to(torch.int32).cuda().reshape(1, -1)
            cache.ensure_experts(layer, want.clone())
            assert int(cache.num_indices) == 0, f"layer {layer}: decode missed a warmed expert"
