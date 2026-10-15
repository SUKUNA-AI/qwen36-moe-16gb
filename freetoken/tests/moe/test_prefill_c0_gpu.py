"""Owner-only C0 transport gate (prefill copy-stream priority, epoch invalidation); never run by CPU-checks.sh.

Each epoch maps some experts into the resident region and leaves stale decode maps inside the
double buffers, then streams every layer while a large matmul occupies the SMs between the
ready wait and the consumer's read. The consumer's bytes must equal the host banks, and the
slot maps after every epoch must equal the flags-off cache driven through the same sequence.
"""
import pytest
import torch
from freetoken.moe.offload_cache import (
    PREFILL_COPY_PRIORITY_ENV, PREFILL_EPOCH_INVALIDATE_ENV, OffloadMoeCache, _BANK_SCHEMAS,
)

# Real Qwen3.6-35B-A3B NVFP4 row bytes per expert (hidden 2048, moe_intermediate 512).
ROW_BYTES = {"gate_up_packed": 1 << 20, "gate_up_scale": 1 << 17, "gate_up_global": 4096,
             "down_packed": 1 << 19, "down_scale": 1 << 16, "down_global": 8192}
E, L, EPOCHS = 32, 4, 12


def ordered_probe(fn):
    """batch_memcpy._probe with its dst zero-fill ordered before the copy (the original races)."""
    src = torch.arange(16, dtype=torch.uint8).pin_memory()
    dst = torch.zeros(16, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    fn(torch.tensor([dst.data_ptr()]), torch.tensor([src.data_ptr()]), torch.tensor([16]), stream.cuda_stream)
    stream.synchronize()
    if not torch.equal(dst.cpu(), src):
        raise RuntimeError("cudaMemcpyBatchAsync probe copied wrong bytes")


def make_cache(monkeypatch, sources, hit_d2d, priority, epoch):
    from freetoken.kernel import batch_memcpy

    monkeypatch.setattr(batch_memcpy, "_probe", ordered_probe)
    monkeypatch.setenv(PREFILL_COPY_PRIORITY_ENV, "1" if priority else "0")
    monkeypatch.setenv(PREFILL_EPOCH_INVALIDATE_ENV, "1" if epoch else "0")
    monkeypatch.setenv("QWEN36_PREFILL_SYNCFREE_INVALIDATE", "1")
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=4 * E, device=torch.device("cuda"),
                            prefill_overlap=True, prefill_hit_d2d=hit_d2d, quant_format="nvfp4")
    cache.set_bank_sources(sources)
    if hit_d2d:
        assert cache._hit_d2d_usable(), "hit-D2D unavailable: a failed gate, not a fallback PASS"
    assert (cache.prefill_copy_stream.priority < 0) == priority
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
            flat = layer * E + e
            if (e + layer + epoch) % 4 == 0 and nxt < cache.cache_size:
                slot = nxt
                nxt += 1
            elif (e * 7 + layer + epoch) % 5 == 0:
                slot = (e * 3 + layer * 11 + epoch) % (2 * E)  # stale map into a double buffer
                if int(cache.id_of_slot[slot]) >= 0:
                    continue
            else:
                continue
            cache.slot_for_id[layer, e] = slot
            cache.id_of_slot[slot] = flat
            if slot >= 2 * E:
                for host, bank in cache.banks:
                    bank[slot].copy_(host[layer][e], non_blocking=True)


def run_epoch(cache, layer_major):
    stream = torch.cuda.current_stream()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    if layer_major:
        cache.begin_layer_major_prefill()
    got = []
    for layer in range(L):
        if layer == 0 and not layer_major:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(layer)
        cache.prefetch_prefill_layer(layer + 1)
        views = cache.wait_prefill_layer(layer)
        for _ in range(3):
            a = (a @ a).clamp_(-1, 1)  # keep the SMs busy while the next copy is in flight
        got.append([v.clone() for v in views])
        cache.release_prefill_layer(layer)
    if layer_major:
        for event in cache.prefill_ready_events:
            stream.wait_event(event)
        cache.end_layer_major_prefill()
    return got


def state(cache):
    torch.cuda.synchronize()
    return cache.slot_for_id.cpu(), cache.id_of_slot.cpu(), cache.usage.cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires owner CUDA/SM120 stack")
@pytest.mark.parametrize("hit_d2d", (False, True))
@pytest.mark.parametrize("priority,epoch", ((True, False), (False, True), (True, True)))
def test_c0_bytes_and_slot_maps_match_flags_off(monkeypatch, hit_d2d, priority, epoch):
    gen = torch.Generator().manual_seed(1234)
    sources = {name: [torch.randint(0, 256, (E, ROW_BYTES[name]), dtype=torch.uint8, generator=gen).pin_memory()
                      for _ in range(L)] for name in _BANK_SCHEMAS["nvfp4"]}
    ref = make_cache(monkeypatch, sources, hit_d2d, False, False)
    cand = make_cache(monkeypatch, sources, hit_d2d, priority, epoch)
    for epoch_id in range(EPOCHS):
        layer_major = epoch_id % 3 == 2
        for cache in (ref, cand):
            seed_maps(cache, epoch_id)
        d2d_before = cand.prefill_d2d_bytes
        got_ref = run_epoch(ref, layer_major)
        got = run_epoch(cand, layer_major)
        torch.cuda.synchronize()
        for layer in range(L):
            for (host, _), r, c in zip(cand.banks, got_ref[layer], got[layer]):
                assert torch.equal(c.cpu(), host[layer]), f"epoch {epoch_id} layer {layer}: bytes differ"
                assert torch.equal(r.cpu(), host[layer])
        for name, r, c in zip(("slot_for_id", "id_of_slot", "usage"), state(ref), state(cand)):
            assert torch.equal(r, c), f"epoch {epoch_id}: {name} differs from the flags-off cache"
        assert not (cand.slot_for_id.cpu().lt(2 * E) & cand.slot_for_id.cpu().ge(0)).any()
        if hit_d2d and not layer_major:
            assert cand.prefill_d2d_bytes > d2d_before
        assert not cand._prefill_epoch_open
