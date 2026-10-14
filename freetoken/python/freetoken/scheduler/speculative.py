"""Experimental single-request prompt lookup with transactional KV/GDN state.

Disabled by default. Greedy first; sampled requests use the ordinary sampler.
The verifier currently uses the existing short-prefill path, and the cost
controller disables it when that path costs more than ordinary graph decode.
"""
import os,time
import torch
from freetoken.core import Batch
from freetoken.engine.speculative import CostController,PromptLookup,final_probs,pick_gpu,verify_gpu,uniform,sampled_enabled


def state(req):
    if not hasattr(req,'spec_cost'):
        import secrets
        req.spec_cost=CostController()
        req.spec_lookup=PromptLookup(req.input_ids[:-1].tolist())
        req.spec_seed=req.sampling_params.seed if req.sampling_params.seed is not None else secrets.randbits(63)
    return req.spec_cost


def observe_baseline(batch,elapsed):
    if batch.is_decode:
        for req in batch.reqs:
            if sampled_enabled(req):state(req).baseline(elapsed)


class _Cancelled(Exception):
    pass


class _VerificationTransaction:
    """Own the provisional page-size-one suffix until any reply is published."""
    def __init__(self,s,req,draft_len):
        self.s,self.req=s,req
        self.old_device,self.old_cached=req.device_len,req.cached_len
        self.end=self.old_device+draft_len
        self.committed=False
        pool=s.engine.linear_state_pool
        self.slot=req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        self.tensors=[] if pool is None else [pool.conv_states,pool.recurrent_states,*pool.slot_states.values()]
        self.saved=[t[:,self.slot].clone() for t in self.tensors]
        self.page_view=s.engine.page_table[req.table_idx,self.old_cached:self.end]
        self.old_pages=self.page_view.clone()
        self.token_view=s.token_pool[req.table_idx,self.old_device:self.end]
        self.old_tokens=self.token_view.clone()
        self.mtp=s.engine.native_mtp
        self.mtp_length=None if self.mtp is None else self.mtp.length
        self.mtp_previous=getattr(s.engine,'_mtp_previous',None)
        # The slice starts after the committed KV prefix. A sentinel distinguishes
        # pages written by prepare from stale entries belonging to an earlier request.
        self.page_view.fill_(-1)

    def restore_live(self):
        for tensor,snapshot in zip(self.tensors,self.saved):
            tensor[:,self.slot].copy_(snapshot)

    def rollback(self):
        if self.committed:return
        self.restore_live()
        allocated=self.page_view[self.page_view>=0].clone()
        self.s.cache_manager._free(allocated)
        self.page_view.copy_(self.old_pages)
        self.token_view.copy_(self.old_tokens)
        self.req.device_len,self.req.cached_len=self.old_device,self.old_cached
        if self.mtp is not None:
            self.mtp.truncate(self.mtp_length)
            self.s.engine._mtp_previous=self.mtp_previous

    def commit_pages(self,consumed):
        tail=self.page_view[consumed:]
        self.s.cache_manager._free(tail[tail>=0].clone())
        tail.fill_(-1)
        self.req.device_len,self.req.cached_len=self.old_device,self.old_cached
        self.committed=True


def try_step(s):
    if s.prefill_manager.runnable or len(s.decode_manager.running_reqs)!=1:return False
    req=next(iter(s.decode_manager.running_reqs))
    if not sampled_enabled(req) or req.aborted or req.remain_len<2:return False
    if s.config.page_size!=1 or s.cache_manager.swa_pool is not None:return False
    cost=state(req)
    if cost.disabled or len(cost.base)<8:return False
    tick=time.perf_counter()
    window=min(int(os.environ.get('QWEN36_SPEC_WINDOW','3')),3,req.remain_len-1)
    mtp=s.engine.native_mtp
    draft_probs=[]
    if mtp is not None:
        if not s.engine._mtp_eligible:return False
        mtp_length=mtp.length
        token=s.token_pool[req.table_idx,req.device_len-1:req.device_len]
        hidden=s.engine._mtp_previous
        draft=[]
        try:
            for i in range(window):
                position=torch.tensor([req.device_len-1+i],device=s.device,dtype=torch.int32)
                logits,hidden=mtp.forward(token,hidden,position)
                if req.sampling_params.is_greedy:
                    token=logits.argmax(-1).to(torch.int32)
                else:
                    q=final_probs(logits,req.sampling_params)[0]
                    draft_probs.append(q)
                    token=torch.tensor([pick_gpu(q,uniform(req.spec_seed,req.device_len+i,1))],device=s.device,dtype=torch.int32)
                draft.append(int(token.item()))
        finally:
            mtp.truncate(mtp_length)
    else:
        draft=req.spec_lookup.propose(req.input_ids.tolist(),window)
    draft_s=time.perf_counter()-tick
    if not draft:return False
    s.cache_manager.snapshot_toolcall_anchor([req])
    transaction=_VerificationTransaction(s,req,len(draft))
    try:
        return _verify_and_commit(s,req,draft,draft_probs,cost,tick,draft_s,transaction)
    except _Cancelled:
        transaction.rollback()
        s.decode_manager.remove_req(req)
        s._free_req_resources(req)
        s.finished_reqs.add(req)
        return True
    except BaseException:
        transaction.rollback()
        raise


def _verify_and_commit(s,req,draft,draft_probs,cost,tick,draft_s,transaction):
    engine=s.engine;mtp=engine.native_mtp
    old_device,old_cached=transaction.old_device,transaction.old_cached
    assert old_device-old_cached==1 and len(req.input_ids)==old_device
    # Track metadata requires an extend of at least 65 tokens; this path uses <=4.
    req.device_len=old_device+len(draft)
    s.token_pool[req.table_idx,old_device:req.device_len]=torch.tensor(draft,device=s.device,dtype=torch.int32)
    batch=Batch([req],'prefill',verify_all_logits=True)
    forward=s._prepare_batch(batch)
    batch.input_ids=s.token_pool[forward.input_tuple]
    started=time.perf_counter()
    with engine.ctx.forward_batch(batch),engine.model.forward_host_ctx(batch,False):
        logits=engine.model.forward()
    if req.sampling_params.is_greedy:
        choices=logits.argmax(-1).cpu().tolist()
    else:
        probabilities=final_probs(logits,req.sampling_params)
        choices=[]
    verify_s=time.perf_counter()-started
    if req.aborted:raise _Cancelled()
    accepted=0
    if req.sampling_params.is_greedy:
        for proposed,chosen in zip(draft,choices):
            if proposed!=chosen:break
            accepted+=1
        emitted=draft[:accepted]+[choices[accepted]]
    else:
        emitted=[]
        for i,proposed in enumerate(draft):
            token,ok=verify_gpu(probabilities[i],proposed,draft_probs[i] if draft_probs else None,
                uniform(req.spec_seed,old_device+i,2),uniform(req.spec_seed,old_device+i,3))
            emitted.append(token)
            if not ok:break
            accepted+=1
        if accepted==len(draft):
            emitted.append(pick_gpu(probabilities[-1],uniform(req.spec_seed,old_device+accepted,0)))
    # Stop/EOS and tool anchors must commit exactly their consumed prefix.
    prompt_len=req.max_device_len-req.output_len
    for i,token in enumerate(emitted):
        tail=req.input_ids[prompt_len:].tolist()+emitted[:i+1]
        stop=(not req.sampling_params.ignore_eos and token in s.eos_token_ids)
        if req.sampling_params.stop_strs:
            text=s.tokenizer.decode(tail)
            stop=stop or any(v in text for v in req.sampling_params.stop_strs)
        if stop or token==s.toolcall_anchor_id:
            emitted=emitted[:i+1];break
    consumed=len(emitted)
    replay_s=0.
    if consumed < len(draft)+1:
        started=time.perf_counter()
        transaction.restore_live()
        req.device_len=old_cached+consumed
        replay=Batch([req],'prefill',verify_all_logits=True)
        prepared=s._prepare_batch(replay,allocate=False)
        replay.input_ids=s.token_pool[prepared.input_tuple]
        with engine.ctx.forward_batch(replay),engine.model.forward_host_ctx(replay,False):
            engine.model.forward()
        torch.cuda.current_stream().synchronize()
        replay_s=time.perf_counter()-started
        batch=replay
    if mtp is not None:
        started=time.perf_counter()
        from freetoken.engine.mtp import observe_target
        observe_target(engine,batch)
        torch.cuda.current_stream().synchronize()
        replay_s+=time.perf_counter()-started
    if req.aborted:raise _Cancelled()
    transaction.commit_pages(consumed)
    # Reuse the ordinary stop/detokenize/resource-free path for every committed
    # output. Intermediate replies never own or expose provisional KV pages.
    from freetoken.engine import ForwardOutput
    # Terminal _process_last_data frees the request and normally emits metrics
    # immediately. Defer that one observation until this step's cost, including
    # reply publication, is known. The matching guard lives in telemetry.complete.
    req._qwen36_spec_metrics_deferred=True
    published=0
    publication_completed=False
    publication_started=time.perf_counter()
    try:
        for token in emitted:
            req.complete_one()
            cpu=torch.tensor([token],dtype=torch.int32)
            gpu=cpu.to(s.device)
            s.token_pool[req.table_idx,req.device_len-1]=gpu[0]
            committed=Batch([req],'decode');committed.padded_reqs=[req]
            done=torch.cuda.Event();done.record()
            published+=1
            s._process_last_data((forward._replace(batch=committed),ForwardOutput(gpu,cpu,done)))
            if req.table_idx==-1:break
        s.decode_manager.filter_reqs([])
        publication_completed=True
    finally:
        try:
            cost.publish_s=getattr(cost,'publish_s',0.)+time.perf_counter()-publication_started
            cost.wall_timing_includes_publication=True
            # verify/replay call model.forward directly, so their wall spans
            # must never masquerade as engine forward CUDA-event measurements.
            cost.verify_replay_in_engine_gpu_timers=False
            cost.publication_failed=not publication_completed
            if published:
                cost.observe(len(draft),min(accepted,published),published,draft_s,verify_s,replay_s,
                             total_s=time.perf_counter()-tick)
        finally:
            req._qwen36_spec_metrics_deferred=False
            if req.table_idx==-1:
                from freetoken.engine.telemetry import complete
                complete(req,engine,cache_manager=s.cache_manager)
    return True
