"""Owner-only C1/C2 transport gate (selected-expert prefill, TMA-gather prefill H2D); never run by CPU-checks.sh.

C2: every byte the consumer reads and the slot maps after each epoch equal the flags-off cache.
C1: per layer, only the routed experts are moved into the double buffer; the Triton NVFP4 prefill
GEMM over that buffer must return the same output BYTES as over the full-layer buffer, with the
same slot maps and no LRU usage/stats written. Real Qwen3.6 NVFP4 row shapes, fewer experts.
"""
import os

import pytest
import torch
from freetoken.moe.offload_cache import (
    PREFILL_COPY_PRIORITY_ENV, PREFILL_EPOCH_INVALIDATE_ENV, PREFILL_SELECTED_ENV, PREFILL_SELECTED_SYNC_ENV,
    PREFILL_TMA_GATHER_ENV, OffloadMoeCache,
)

H, I, E, L, TOPK = 2048, 512, 64, 4, 8
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires owner CUDA/SM120 stack")
ENVS = (PREFILL_COPY_PRIORITY_ENV, PREFILL_EPOCH_INVALIDATE_ENV, PREFILL_SELECTED_ENV, PREFILL_SELECTED_SYNC_ENV,
        PREFILL_TMA_GATHER_ENV)



@pytest.fixture(scope="module")
def sources():
    g = torch.Generator().manual_seed(77)

    def u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    def glob(*shape):
        return torch.rand(*shape, generator=g) * 0.5 + 0.5

    total = L * E
    flat = {"gate_up_packed": u8(total, 2 * I, H // 2), "gate_up_scale": scale(total, 2 * I, H // 16),
            "gate_up_global": glob(total, 2 * I), "down_packed": u8(total, H, I // 2),
            "down_scale": scale(total, H, I // 16), "down_global": glob(total, H)}
    return {name: list(t.pin_memory().split(E)) for name, t in flat.items()}


def make_cache(monkeypatch, sources, hit_d2d=True, **env):
    for name in ENVS:
        monkeypatch.setenv(name, str(env.get(name, "0")))
    monkeypatch.setenv("QWEN36_PREFILL_SYNCFREE_INVALIDATE", "1")
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=4 * E, device=torch.device("cuda"),
                            prefill_overlap=True, prefill_hit_d2d=hit_d2d, quant_format="nvfp4")
    cache.set_bank_sources(sources)
    return cache


def seed_maps(cache, epoch):
    """Deterministic resident maps (slots >= 2E) plus stale decode maps into both buffers."""
    dev = cache.slot_for_id.device
    cache.slot_for_id.fill_(-1)
    cache.id_of_slot.fill_(-1)
    cache.usage.copy_(torch.arange(cache.cache_size, device=dev) + 7)
    nxt = 2 * E
    for layer in range(L):
        for e in range(E):
            if (e + layer + epoch) % 4 == 0 and nxt < cache.cache_size:
                slot, nxt = nxt, nxt + 1
            elif (e * 7 + layer + epoch) % 5 == 0:
                slot = (e * 3 + layer * 11 + epoch) % (2 * E)
                if int(cache.id_of_slot[slot]) >= 0:
                    continue
            else:
                continue
            cache.slot_for_id[layer, e] = slot
            cache.id_of_slot[slot] = layer * E + e
            if slot >= 2 * E:
                for host, bank in cache.banks:
                    bank[slot].copy_(host[layer][e], non_blocking=True)


def maps(cache):
    torch.cuda.synchronize()
    return [t.cpu() for t in (cache.slot_for_id, cache.id_of_slot, cache.usage, cache.lru_stats)]


def busy(a):
    for _ in range(2):
        a = (a @ a).clamp_(-1, 1)
    return a


def full_epoch(cache, layer_major=False, consume=None):
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    if layer_major:
        cache.begin_layer_major_prefill()
    else:
        cache.begin_prefill()
    out = []
    for layer in range(L):
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        views = cache.wait_prefill_layer(layer)
        a = busy(a)
        out.append(consume(layer, views) if consume else [v.clone() for v in views])
        cache.release_prefill_layer(layer)
    if layer_major:
        for event in cache.prefill_ready_events:
            torch.cuda.current_stream().wait_event(event)
        cache.end_layer_major_prefill()
    return out


@cuda
@pytest.mark.parametrize("hit_d2d", (False, True))
@pytest.mark.parametrize("c0", (False, True))
def test_c2_bytes_and_slot_maps_match_default(monkeypatch, sources, hit_d2d, c0):
    ref = make_cache(monkeypatch, sources, hit_d2d)
    flags = {PREFILL_TMA_GATHER_ENV: 1}
    if c0:
        flags.update({PREFILL_COPY_PRIORITY_ENV: 1, PREFILL_EPOCH_INVALIDATE_ENV: 1})
    cand = make_cache(monkeypatch, sources, hit_d2d, **flags)
    for epoch in range(9):
        layer_major = epoch % 3 == 2
        for cache in (ref, cand):
            seed_maps(cache, epoch)
        before = (cand.prefill_h2d_bytes, cand.prefill_d2d_bytes)
        got_ref = full_epoch(ref, layer_major)
        got = full_epoch(cand, layer_major)
        assert cand._prefill_tma_active == (not layer_major)
        torch.cuda.synchronize()
        for layer in range(L):
            for (host, _), r, c in zip(cand.banks, got_ref[layer], got[layer]):
                assert torch.equal(c.cpu(), host[layer]), f"epoch {epoch} layer {layer}: bytes differ"
                assert torch.equal(r.cpu(), host[layer])
        for name, r, c in zip(("slot_for_id", "id_of_slot", "usage", "lru_stats"), maps(ref), maps(cand)):
            assert torch.equal(r, c), f"epoch {epoch}: {name} differs from the default cache"
        h2d, d2d = cand.prefill_h2d_bytes - before[0], cand.prefill_d2d_bytes - before[1]
        assert h2d > 0
        if hit_d2d and not layer_major:
            assert d2d > 0


def routing(m, seed):
    g = torch.Generator().manual_seed(seed)
    skew = torch.linspace(2.0, 0.0, E)  # low ids are popular, like real routers
    scores = torch.randn(L, m, E, generator=g) + skew
    return [s.topk(TOPK, dim=1).indices.to(torch.int32).cuda() for s in scores]


@cuda
@pytest.mark.parametrize("sync", (False, True))
@pytest.mark.parametrize("c0", (False, True))
def test_c1_selected_gemm_bytes_match_full_layer(monkeypatch, sources, sync, c0):
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    ref = make_cache(monkeypatch, sources)
    flags = {PREFILL_SELECTED_ENV: 512, PREFILL_SELECTED_SYNC_ENV: int(sync)}
    if c0:
        flags.update({PREFILL_COPY_PRIORITY_ENV: 1, PREFILL_EPOCH_INVALIDATE_ENV: 1})
    cand = make_cache(monkeypatch, sources, **flags)
    for chunk, m in enumerate((1, 4, 16, 33, 100, 511)):
        topk = routing(m, 900 + chunk)
        weights = torch.rand(m, TOPK, generator=torch.Generator().manual_seed(chunk)).cuda()
        hidden = (torch.randn(m, H, generator=torch.Generator().manual_seed(50 + chunk)) / 4).to(torch.bfloat16).cuda()
        for cache in (ref, cand):
            seed_maps(cache, chunk)

        def gemm(layer, views):
            return fused_experts_nvfp4(hidden.clone(), *views, weights, topk[layer], E, "silu", False)

        want = full_epoch(ref, consume=gemm)
        ref.select_prefill_route(m)
        assert not ref.prefill_selected_active
        cand.select_prefill_route(m)
        assert cand.prefill_selected_active, "C1 refused a short chunk: a failed gate, not a fallback PASS"
        d2d_before = cand.prefill_d2d_bytes
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        got = []
        for layer in range(L):
            if layer == 0:
                cand.begin_prefill()
            views = cand.selected_prefill_layer(layer, topk[layer].clone())
            a = busy(a)
            got.append(gemm(layer, views))
            routed = torch.unique(topk[layer]).cpu()
            for (host, _), view in zip(cand.banks, views):
                assert torch.equal(view.cpu()[routed], host[layer][routed]), f"chunk {chunk} layer {layer}"
            cand.release_prefill_layer(layer)
        torch.cuda.synchronize()
        for layer in range(L):
            assert torch.equal(got[layer].view(torch.int16), want[layer].view(torch.int16)), \
                f"chunk {chunk} (M={m}) layer {layer}: GEMM output bytes differ"
        for name, r, c in zip(("slot_for_id", "id_of_slot", "usage", "lru_stats"), maps(ref), maps(cand)):
            assert torch.equal(r, c), f"chunk {chunk}: {name} differs (C1 must not touch LRU state)"
        assert not bool(cand.lru_stats.any())
        assert cand.prefill_d2d_bytes > d2d_before, "no resident expert was served D2D"
        assert not cand._prefill_epoch_open


@cuda
def test_c1_route_guards(monkeypatch, sources):
    off = make_cache(monkeypatch, sources)
    off.select_prefill_route(4)
    assert not off.prefill_selected_active
    cache = make_cache(monkeypatch, sources, **{PREFILL_SELECTED_ENV: 512})
    for tokens, want in ((0, False), (1, True), (511, True), (512, False), (4096, False)):
        cache.select_prefill_route(tokens)
        assert cache.prefill_selected_active == want, tokens
    cache.refill = object()
    cache.select_prefill_route(4)
    assert not cache.prefill_selected_active
    cache.refill = None
    cache.layer_major_active = True
    cache.select_prefill_route(4)
    assert not cache.prefill_selected_active
    cache.layer_major_active = False
    cache.select_prefill_route(4)
    cache.begin_layer_major_prefill()
    assert not cache.prefill_selected_active
    cache.end_layer_major_prefill()
    monkeypatch.setenv(PREFILL_SELECTED_ENV, "lots")
    with pytest.raises(ValueError):
        OffloadMoeCache(num_layers=L, num_experts=E, cache_size=4 * E, device=torch.device("cuda"))
    assert os.environ[PREFILL_SELECTED_ENV] == "lots"
