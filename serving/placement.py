"""Read-only logging of actual host bank tensors in each spawned model worker."""
import json,os,pathlib
def install():
    from freetoken.engine import engine
    original=engine.load_expert_banks
    def measured(*args,**kw):
        banks=original(*args,**kw)
        tensors=[t for per_layer in banks.sources.values() for t in per_layer]
        doc={'pid':os.getpid(),'format':banks.quant_format,'kernel':banks.kernel,'bank_tensor_count':len(tensors),'host_bank_bytes':sum(t.numel()*t.element_size() for t in tensors),'devices':sorted({str(t.device) for t in tensors}),'pinned_tensor_count':sum(t.is_pinned() for t in tensors),'pageable_tensor_count':sum(t.device.type=='cpu' and not t.is_pinned() for t in tensors),'layer_residency':banks.layer_residency}
        root=pathlib.Path(os.environ.get('QWEN36_DATA_ROOT','/home/USER/qwen36-api'))/'reports';root.mkdir(mode=0o700,exist_ok=True)
        (root/f'placement-{os.getpid()}.json').write_text(json.dumps(doc,indent=2))
        print('QWEN36_ACTUAL_PLACEMENT '+json.dumps(doc),flush=True)
        return banks
    engine.load_expert_banks=measured
    from freetoken.moe.offload_cache import OffloadMoeCache
    original_set=OffloadMoeCache.set_bank_sources
    def measured_set(self,*args,**kw):
        result=original_set(self,*args,**kw)
        doc={'pid':os.getpid(),'decode_target':self.decode_target,'slots':self.cache_size,'total_experts':self.num_layers*self.num_experts,'gpu_bank_bytes':sum(t.numel()*t.element_size() for t in self.bank_caches.values()),'gpu_bank_devices':sorted({str(t.device) for t in self.bank_caches.values()}),'layer_residency':self.layer_residency}
        root=pathlib.Path(os.environ.get('QWEN36_DATA_ROOT','/home/USER/qwen36-api'))/'reports';root.mkdir(mode=0o700,exist_ok=True)
        (root/f'gpu-cache-{os.getpid()}.json').write_text(json.dumps(doc,indent=2))
        print('QWEN36_ACTUAL_GPU_CACHE '+json.dumps(doc),flush=True)
        return result
    OffloadMoeCache.set_bank_sources=measured_set
