"""Owner-only transport gate for short-route; never run by CPU-checks.sh.

Large aligned rows exercise hit-D2D, then a layer-major epoch overwrites borrowed
slots. Compare every destination byte and require real D2D on the short path.
"""
import pytest
import torch
from freetoken.moe.offload_cache import OffloadMoeCache, _BANK_SCHEMAS

@pytest.mark.parametrize('quant_format', ('bf16', 'nvfp4_marlin', 'nvfp4'))
@pytest.mark.parametrize('hit_d2d', (False, True))
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires owner CUDA/SM120 stack')
def test_short_hits_and_borrowed_slot_epochs_byte_exact(hit_d2d, quant_format):
    E, L, ROW = 4, 2, 262144
    dev = torch.device('cuda')
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=4*E,
                           device=dev, prefill_overlap=True, prefill_hit_d2d=hit_d2d, quant_format=quant_format)
    sizes = {name:(16 if 'global' in name else 16384 if 'scale' in name else ROW)
             for name in _BANK_SCHEMAS[quant_format]}
    sources = {name:[torch.randint(0,256,(E,size),dtype=torch.uint8).pin_memory() for _ in range(L)]
               for name,size in sizes.items()}
    cache.set_bank_sources(sources)
    if hit_d2d:
        assert cache._hit_d2d_usable(), 'hit-D2D unavailable: this is a failed component gate, not a fallback PASS'
    def resident():
        for layer in range(L):
            slots=torch.arange(2*E+layer*E,3*E+layer*E,device=dev,dtype=torch.int32)
            cache.slot_for_id[layer].copy_(slots)
            cache.id_of_slot[2*E+layer*E:3*E+layer*E]=torch.arange(layer*E,(layer+1)*E,device=dev)
            for host, bank in cache.banks:
                bank[2*E+layer*E:3*E+layer*E].copy_(host[layer],non_blocking=True)
    def byte_check():
        for layer in range(L):
            views=cache.wait_prefill_layer(layer)
            for (host,_),got in zip(cache.banks,views):
                assert torch.equal(got.cpu(),host[layer]), f'bank bytes differ at layer {layer}'
            cache.release_prefill_layer(layer)
    for _ in range(50):
        resident();before=cache.prefill_d2d_bytes
        cache.begin_prefill();byte_check()
        assert (cache.prefill_d2d_bytes > before) if hit_d2d else (cache.prefill_d2d_bytes == before)
        resident();before=cache.prefill_d2d_bytes
        # HiddenStore borrows/unmaps on compute before begin_prefill records the
        # producer fence. Preserve that order so copy-stream invalidation follows it.
        borrowed=cache.borrow_tail_slots(E)
        for view in borrowed:view.fill_(165)
        cache.begin_layer_major_prefill()
        assert not cache._prefill_hit_d2d_active
        byte_check()
        assert cache.prefill_d2d_bytes == before
        # Same close ordering as LayerMajorExecutor.close; enqueue waits before reuse.
        for event in cache.prefill_ready_events:torch.cuda.current_stream(dev).wait_event(event)
        cache.end_layer_major_prefill()
        assert not cache.layer_major_active and cache.prefill_hit_d2d == hit_d2d
    torch.cuda.synchronize()
