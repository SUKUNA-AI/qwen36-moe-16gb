"""Aggregate GPU spans after the scheduler's existing completion barrier."""
import atexit
import os
import pathlib
import time
from collections import OrderedDict

ENABLED = os.environ.get('QWEN36_ENGINE_METRICS','1') == '1'
rows = OrderedDict()
finished = OrderedDict()
writer = None


def get_writer():
    global writer
    if writer is None:
        from request_metrics import BufferedNumericLog
        writer = BufferedNumericLog(pathlib.Path(os.environ['QWEN36_DATA_ROOT'])/'reports')
        atexit.register(writer.close)
    return writer


def batch_started(batch,engine):
    if not ENABLED:return
    for req in batch.reqs:
        if req.uid in finished or req.uid in rows:continue
        if not rows:
            import torch
            torch.cuda.reset_peak_memory_stats()
        row=dict(uid=req.uid,pid=os.getpid(),run_id=os.environ.get('QWEN36_RUN_ID'),prefill_gpu_ms=0.0,decode_gpu_ms=0.0,
                 prefill_forward_tokens=0,decode_forwards=0,started=time.monotonic())
        cache=engine.moe_offload_cache
        if cache is not None and cache.collect_stats:
            row['_initial_expert_stats']=cache.lru_stats.sum(0).clone()
            row['_initial_prefill_bytes']=(cache.prefill_h2d_bytes,cache.prefill_d2d_bytes)
            if cache.decode_target=='hybrid':
                row['_initial_hybrid_stats']=tuple(value.clone() for value in
                    (cache.stat_active,cache.stat_missing,cache.stat_fetched))
        rows[req.uid]=row
    while len(rows)>64:rows.popitem(last=False)

def batch_completed(batch,engine):
    if not ENABLED or not hasattr(batch,'qwen36_timing'): return
    begin,end = batch.qwen36_timing
    duration = begin.elapsed_time(end)
    for req in batch.reqs:
        if req.uid in finished: continue
        row = rows.get(req.uid)
        if row is None:continue
        if batch.is_prefill:
            row['prefill_gpu_ms'] += duration / len(batch.reqs)
            row['prefill_forward_tokens'] += batch.log_new_tokens // len(batch.reqs)
        else:
            row['decode_gpu_ms'] += duration / len(batch.reqs)
            row['decode_forwards'] += 1
    while len(rows)>64: rows.popitem(last=False)

def complete(req,engine,cache_manager=None):
    global writer
    # A speculative step records verify/replay/publication cost after its
    # terminal drain. Its owner retries completion in a finally block.
    if getattr(req,'_qwen36_spec_metrics_deferred',False):return
    from .qualification import complete as teacher_complete
    teacher_complete(req.uid)
    if not ENABLED: return
    if req.uid in finished: return
    row = rows.pop(req.uid,None)
    finished[req.uid] = True
    while len(finished)>1024: finished.popitem(last=False)
    if row is None:return
    initial_stats=row.pop('_initial_expert_stats',None)
    initial_hybrid_stats=row.pop('_initial_hybrid_stats',None)
    initial_bytes=row.pop('_initial_prefill_bytes',(0,0))
    import torch
    row['engine_span_s'] = time.monotonic()-row.pop('started')
    row['cancelled'] = req.aborted
    if hasattr(req,'spec_cost'):
        row['speculation']=req.spec_cost.report()
    row['torch_allocated_bytes'] = torch.cuda.memory_allocated()
    row['torch_reserved_bytes'] = torch.cuda.memory_reserved()
    row['request_torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    row['request_torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    row['gpu_peak_scope'] = 'PyTorch allocator peaks for one active request; excludes desktop, driver and external allocators'
    if not hasattr(engine,'_qwen_model_storage') and hasattr(engine.model,'state_dict'):
        storages={};by_dtype={}
        cache=engine.moe_offload_cache
        excluded={tensor.untyped_storage().data_ptr() for tensor in cache.bank_caches.values()} if cache is not None else set()
        for tensor in engine.model.state_dict().values():
            if tensor.device.type!='cuda' or not tensor.numel():continue
            storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
            if storage.data_ptr() in excluded:continue
            if key in storages:continue
            size=storage.nbytes();storages[key]=size
            by_dtype[str(tensor.dtype)]=by_dtype.get(str(tensor.dtype),0)+size
        engine._qwen_model_storage={'bytes':sum(storages.values()),'by_dtype':by_dtype,
                                   'scope':'unique CUDA storages exposed by model.state_dict; predominantly dense weights/scales, not offload cache or KV'}
    if hasattr(engine,'_qwen_model_storage'):row['model_cuda_state_storage']=engine._qwen_model_storage
    layers=getattr(getattr(getattr(engine.model,'model',None),'layers',None),'op_list',[])
    row['fused_attention_gate_initialized_layers']=sum(hasattr(getattr(layer,'self_attn',None),'_fused_attention_gate') for layer in layers)
    row['fp32_gdn_snapshot_enabled']=os.environ.get('QWEN36_FP32_GDN_SNAPSHOT')=='1'
    row['retain_chunk_snapshot_enabled']=bool(getattr(cache_manager,'retain_chunk_snapshot',False))
    if getattr(engine,'layer_major_stats',None):
        row['layer_major_prefill']=dict(engine.layer_major_stats)
    refill_stats=getattr(engine.moe_offload_cache,'layer_major_refill_stats',None)
    if refill_stats:
        row['layer_major_refill']=dict(refill_stats,slots_published=int(engine.moe_offload_cache.layer_major_refill_published))
    if cache_manager is not None and hasattr(cache_manager,'snapshot_stats'):
        row['prefix_snapshot_counters_total']={key:int(value) for key,value in cache_manager.snapshot_stats.items()}
    checks=[getattr(getattr(layer,'self_attn',None),'_fused_qk_errors',None) for layer in layers]
    checks=[value for value in checks if value is not None]
    if checks:
        row['fused_qk_errors_total']=torch.stack(checks).sum(0).cpu().tolist()
    if engine.native_mtp is not None:
        row['mtp_weight_bytes']=engine.native_mtp.weight_bytes
        row['mtp_kv_bytes']=2*engine.native_mtp.k.numel()*engine.native_mtp.k.element_size()
    kv=engine.kv_cache
    if hasattr(kv,'_kv_buffer'):
        row['kv_bytes'] = kv._kv_buffer.numel()*kv._kv_buffer.element_size()
        row['kv_dtype'] = str(kv.dtype)
        row['kv_clipped_values_total'] = int(kv.overflow.item())
        kv.save_calibration()
    if engine.linear_state_pool is not None:
        pool=engine.linear_state_pool
        row['gdn_bytes'] = sum(t.numel()*t.element_size() for t in (pool.conv_states,pool.recurrent_states))
    row['host_embedding_bytes'] = int(getattr(engine,'_host_embed_bytes',0))
    cache=engine.moe_offload_cache
    if cache is not None:
        row['gpu_expert_bytes'] = sum(t.numel()*t.element_size() for t in cache.bank_caches.values())
        row['host_expert_bytes'] = sum(t.numel()*t.element_size() for layers in cache.bank_sources.values() for t in layers)
        row['host_pinned_expert_bytes'] = sum(t.numel()*t.element_size() for layers in cache.bank_sources.values() for t in layers if t.is_pinned())
        row['expert_slots'] = cache.cache_size
        row['valid_gpu_expert_slots'] = int((cache.id_of_slot>=0).sum().item())
        row['prefill_h2d_bytes_total'] = cache.prefill_h2d_bytes
        row['prefill_d2d_bytes_total'] = cache.prefill_d2d_bytes
        from freetoken.moe.complement import ComplementCache
        if isinstance(cache,ComplementCache):
            # Custom mapped exchange has separate profiler accounting. Zero in
            # the inherited full-bank counters does not mean zero PCIe traffic.
            row['prefill_h2d_bytes_total']=None
            row['prefill_d2d_bytes_total']=None
            row['expert_transfer_accounting']='unavailable_for_experimental_complement'
        if cache.decode_target=='cpu':
            row['expert_transfer_accounting']='unavailable_for_cpu_decode'
        if cache.collect_stats and cache.decode_target in ('gpu','hybrid'):
            row['expert_access_counters_total'] = cache.lru_stats.sum(0).cpu().tolist()
            if cache.decode_target=='hybrid':
                row['expert_access_counters_total']=[int(value.item()) for value in
                    (cache.stat_active,cache.stat_missing,cache.stat_calls)]
            stats=cache.decode_miss_stats()
            if cache.decode_target=='gpu':
                # Upstream fetched/cpu fields are hybrid-only counters and would
                # otherwise label every pure-GPU miss as CPU computation.
                stats.update(fetched_per_layer=stats['missing_per_layer'],cpu_per_layer=0.0,
                             fetch_rate=1.0 if stats['missing_per_layer'] else 0.0)
            row['expert_decode_stats']=stats
            per_expert=sum(t[0].numel()*t.element_size() for t in cache.bank_caches.values())
            row['expert_bytes_per_slot']=per_expert
            if not isinstance(cache,ComplementCache):
                totals=dict(prefill_h2d_bytes=cache.prefill_h2d_bytes,prefill_d2d_bytes=cache.prefill_d2d_bytes,
                            decode_active=row['expert_access_counters_total'][0],
                            decode_missing=row['expert_access_counters_total'][1])
                initial_values=initial_stats.cpu().tolist() if initial_stats is not None else [0,0,0]
                previous=dict(prefill_h2d_bytes=initial_bytes[0],prefill_d2d_bytes=initial_bytes[1],
                              decode_active=initial_values[0],decode_missing=initial_values[1])
                if cache.decode_target=='hybrid':
                    initial=[int(value.item()) for value in initial_hybrid_stats] if initial_hybrid_stats is not None else [0,0,0]
                    previous.update(decode_active=initial[0],decode_missing=initial[1],decode_fetched=initial[2])
                    totals['decode_fetched']=int(cache.stat_fetched.item())
                delta={k:max(0,v-previous.get(k,0)) for k,v in totals.items()}
                delta['decode_h2d_payload_bytes']=delta.get('decode_fetched',delta['decode_missing'])*per_expert
                delta['decode_cache_hits']=delta['decode_active']-delta['decode_missing']
                row['expert_counter_deltas']=delta
                row['expert_transfer_accounting']='logical_weight_payload_not_PCIe_packet_bytes'
    accounted=sum(row.get(name,0) for name in ('kv_bytes','gdn_bytes','gpu_expert_bytes'))
    accounted+=row.get('model_cuda_state_storage',{}).get('bytes',0)
    row['allocator_bytes_outside_named_components']=row['torch_allocated_bytes']-accounted
    row['request_allocator_peak_above_end_bytes']=max(0,row['request_torch_peak_allocated_bytes']-row['torch_allocated_bytes'])
    get_writer().write('engine.jsonl',row)
    from freetoken.engine import memory_history
    memory_history.dump({k:row.get(k) for k in ('run_id','uid','prompt_tokens','output_tokens')})
