"""Byte-preservation/ownership/graph tests using real mapped host memory."""
import argparse,gc,json,os,pathlib,types
import torch
from freetoken.moe.complement import ComplementCache
from freetoken.layers.quantization.moe.base import BankSpec

ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')

class Method:
    def __init__(self,actual=False):self.actual=actual
    def layout(self):
        sizes={'gate_up':1048576,'gate_up_scale':131072,'gate_up_global':2048,'down':524288,'down_scale':65536,'down_global':4096} if self.actual else {'gate_up':2048,'down':1024}
        return {k:BankSpec((n,),torch.uint8) for k,n in sizes.items()}
    def pack(self,piece,out):
        for k in out:out[k].copy_(piece[k])
        return {}


def check(cache,reference):
    torch.cuda.synchronize()
    gs=cache.slot_for_id.cpu().view(-1).tolist();hs=cache.host_slot.cpu().tolist()
    assert sum(s>=0 for s in gs)==cache.owner_slots
    assert sorted(s for s in hs if s>=0)==list(range(len(hs)-cache.owner_slots))
    for gid in range(len(gs)):
        assert (gs[gid]>=0) != (hs[gid]>=0)
        for name in reference:
            actual=cache.bank_caches[name][gs[gid]+cache.offset].cpu() if gs[gid]>=0 else cache._host_banks[name].tensor[hs[gid]]
            assert torch.equal(actual,reference[name][gid]),(gid,name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--actual',action='store_true');p.add_argument('--name',default='complement-kernel');args=p.parse_args()
    torch.manual_seed(407)
    reports=[]
    for mode,policy in [('fixed','lru'),('complement','lru'),('complement','decayed_frequency')]:
        os.environ['QWEN36_RESIDENCY']=mode
        os.environ['QWEN36_CACHE_POLICY']=policy
        method=Method(args.actual)
        layers,experts,slots=(2,256,700) if args.actual else (4,16,52)
        cache=ComplementCache(layers,experts,slots,torch.device('cuda'),layout=method.layout(),prefill_overlap=True)
        cache.allocate(method)
        ref={name:torch.randint(0,256,(layers*experts,*s.shape),dtype=s.dtype) for name,s in method.layout().items()}
        cache.load(method,((l,0,experts,{k:t[l*experts:(l+1)*experts] for k,t in ref.items()}) for l in range(layers)))
        check(cache,ref)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for step in range(128):
                layer=step%layers; original=torch.randperm(experts,device='cuda')[:8].to(torch.int32)
                ids=original.clone();cache.ensure_experts(layer,ids);cache.copy_missing()
                torch.cuda.synchronize()
                for name in ref:
                    assert torch.equal(cache.bank_caches[name][ids.long()].cpu(),ref[name][layer*experts+original.cpu().long()])
                if step%16==0:
                    cache.begin_prefill()
                    for l in range(layers):
                        cache.prefetch_prefill_layer(l)
                        views=cache.wait_prefill_layer(l)
                        for name,t in zip(cache.bank_schema,views):assert torch.equal(t.cpu(),ref[name][l*experts:(l+1)*experts])
                        cache.release_prefill_layer(l)
        torch.cuda.current_stream().wait_stream(stream);check(cache,ref)
        # Same graph, changing source ids, repeated exchanges and current maps.
        ids=torch.arange(8,device='cuda',dtype=torch.int32);original=ids.clone()
        with torch.cuda.stream(stream):
            cache.ensure_experts(0,ids);cache.copy_missing()
        stream.synchronize()
        ids.copy_(original)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            cache.ensure_experts(0,ids);cache.copy_missing()
        for _ in range(32):
            raw=torch.randperm(experts,device='cuda')[:8].to(torch.int32)
            ids.copy_(raw);graph.replay();torch.cuda.synchronize()
            for name in ref:assert torch.equal(cache.bank_caches[name][ids.long()].cpu(),ref[name][raw.cpu().long()])
        check(cache,ref)
        cache.reset();check(cache,ref)
        reports.append({'mode':mode,'policy':policy,'actual_geometry':args.actual,'transport':os.environ.get('QWEN36_COMPLEMENT_TRANSPORT','triton'),
                        'byte_preservation':True,'ownership':True,'graph_replays':32,'eager_steps':128,'prefill_gathers':32})
        del graph,cache,ref;gc.collect();torch.cuda.empty_cache()
    (ROOT/'reports'/f'{args.name}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'PASS':reports}))

if __name__=='__main__':main()
