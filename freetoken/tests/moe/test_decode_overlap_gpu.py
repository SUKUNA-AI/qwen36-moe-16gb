"""SM120 owner gate: real cache/LRU/gather + fork/join, eager and graph replay.

The routed consumer here checks bytes (not a numerical MoE reference). The full
model teacher gate is additionally required. Run without a profiler under flock.
"""
import pytest
import torch
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache, _BANK_SCHEMAS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='owner CUDA/SM120 required')


@pytest.mark.parametrize('gather', ['', 'tuned'])
@pytest.mark.parametrize('layers', [1, 3])
def test_shared_overlap_bytes_and_capture_replay(monkeypatch, gather, layers):
    monkeypatch.setenv('QWEN36_DECODE_SHARED_OVERLAP', '1')
    monkeypatch.setenv('QWEN36_DECODE_GATHER', gather)
    device=torch.device('cuda'); E=32
    # Released native NVFP4 layout, including both full expert matrices and tiny globals.
    sizes=(1048576,131072,4096,524288,65536,8192)
    sources={name:[torch.randint(0,256,(E,size),dtype=torch.uint8).pin_memory() for _ in range(layers)]
             for name,size in zip(_BANK_SCHEMAS['nvfp4'],sizes)}
    caches=[]; models=[]
    for _ in range(2):
        cache=OffloadMoeCache(layers,E,E,device,quant_format='nvfp4')
        cache.set_bank_sources(sources);assert cache._copy_fused_ok
        caches.append(cache);parts=[]
        for lid in range(layers):
            model=object.__new__(OffloadMoELayer)
            model.offload_cache=cache;model.layer_id=lid;model.top_k=8;model.renormalize=True;model.tp_size=1
            def consume(cache,hidden,weights,ids,*,views,**kw):
                # Byte consumers must run after gather, before a subsequent layer overwrites metadata.
                return tuple(bank.index_select(0,ids.flatten().long()).clone() for bank in views)
            model._expert_gemm=consume;parts.append(model)
        models.append(parts)
    logits=torch.zeros((1,E),device=device)
    hidden=torch.randn((1,64),device=device);weight=torch.randn((64,64),device=device)
    def shared(x):return torch.sigmoid(x @ weight) * (x @ weight)
    def execute(arm):
        result=[]
        for model in models[arm]:
            if arm:
                routed,s=model.forward_with_shared(hidden,logits,shared)
            else:
                s=shared(hidden);routed=model.decode_forward(hidden,logits)
            result.extend(routed);result.append(s)
        return tuple(result)
    def choose(n):
        ids=list(range(8-n))+list(range(8,8+n))
        logits.fill_(-100)
        logits[0,ids]=torch.arange(1,9,device=device,dtype=logits.dtype)
    def compare(left,right):
        for a,b in zip(left,right):assert torch.equal(a,b)
        for ca,cb in zip(caches[0].banks,caches[1].banks):assert torch.equal(ca[1],cb[1])
        for attr in ('slot_for_id','id_of_slot','usage'):
            assert torch.equal(getattr(caches[0],attr),getattr(caches[1],attr))
    choose(0)
    for cache in caches:
        for _,bank in cache.banks:bank.fill_(165)
    compare(execute(0),execute(1))  # real eager cold gather; compile/warm outside capture
    torch.cuda.synchronize()
    for generation in range(2):
        graphs=[];outputs=[]
        for arm in (0,1):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=execute(arm)
            graphs.append(graph);outputs.append(out)
        # Reset invalidates mappings but retains storage/graph addresses. Repeated
        # graph replays change the device miss count; nothing is staged on the host.
        for cache in caches:
            cache.reset()
            for _,bank in cache.banks:bank.fill_(165)
        for i in range(180):
            choose(i % 9)
            for graph in graphs:graph.replay()
            compare(*outputs)
        del graphs,outputs
    # Exception/cancel boundary: enqueued side copies must rejoin before slot reuse.
    cache=caches[1];choose(8)
    def cancel(_):raise RuntimeError('synthetic cancellation')
    with pytest.raises(RuntimeError,match='cancellation'):
        models[1][0].forward_with_shared(hidden,logits,cancel)
    for _,bank in cache.banks:bank.fill_(91)
    torch.cuda.synchronize()
    for _,bank in cache.banks:assert torch.all(bank == 91)
