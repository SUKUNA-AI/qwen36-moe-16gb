"""CPU fault injection for the opt-in speculation transaction; no CUDA launches."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler, ForwardInput
from freetoken.scheduler.speculative import state, try_step


class Fixture:
    def __init__(self, choices=(5,6,7,8), *, fault=None, eos=(), stops=(), abort=False):
        self.req=Req(torch.tensor([1,2],dtype=torch.int32),0,0,8,91,
                     SamplingParams(max_tokens=8,stop_strs=list(stops),seed=193),None)
        for token in (3,4):
            self.req.complete_one();self.req.append_host(torch.tensor([token],dtype=torch.int32))
        self.req.linear_slot_idx=1
        cost=state(self.req)
        for _ in range(8):cost.baseline(1.)
        self.req.spec_lookup=SimpleNamespace(propose=lambda *args:[5,6,7])
        self.fault,self.choices,self.abort=fault,choices,abort
        self.calls=0;self.sent=[];self.freed=[];self.returned=[]
        self.pages=torch.full((1,16),-1,dtype=torch.int32)
        self.pages[0,:3]=torch.arange(10,13)
        self.tokens=torch.full((1,16),-91,dtype=torch.int32)
        self.tokens[0,:4]=self.req.input_ids
        pool=SimpleNamespace(conv_states=torch.full((1,2,2),100.),
                             recurrent_states=torch.full((1,2,2),200.),
                             slot_states={'extra':torch.full((1,2,2),300.)})
        self.states=[pool.conv_states,pool.recurrent_states,*pool.slot_states.values()]
        self.original=[t.clone() for t in self.states]
        self.original_pages=self.pages.clone();self.original_tokens=self.tokens.clone()
        engine=SimpleNamespace(linear_state_pool=pool,native_mtp=None,page_table=self.pages,
                               ctx=SimpleNamespace(forward_batch=self.context))
        engine.model=SimpleNamespace(forward=self.forward,forward_host_ctx=lambda *_:nullcontext())
        cm=SimpleNamespace(swa_pool=None,snapshot_toolcall_anchor=lambda _:None,
                           _free=lambda ids:self.returned.extend(ids.tolist()),
                           lazy_free_region=nullcontext,cache_req=lambda *_,**__:None)
        dm=DecodeManager(1);dm.running_reqs.add(self.req)
        self.s=SimpleNamespace(
            device=torch.device('cpu'),engine=engine,token_pool=self.tokens,
            prefill_manager=SimpleNamespace(runnable=False,pending_list=[]),decode_manager=dm,
            config=SimpleNamespace(page_size=1),cache_manager=cm,_prepare_batch=self.prepare,
            eos_token_ids=set(eos),toolcall_anchor_id=None,finished_reqs=set(),
            tokenizer=SimpleNamespace(decode=lambda ids:''.join({5:'E',6:'N',7:'D'}.get(t,'x') for t in ids)),
            status_reporter=SimpleNamespace(report_batch=lambda *_,**__:None),
            _kv_usage_pages=lambda:(0,16),_mamba_slot_usage=lambda:None,
            _swa_token_usage=lambda:None,_gpu_mem_bytes=lambda:0,send_result=self.sent.extend,
        )
        self.s._match_stop_str=lambda req:Scheduler._match_stop_str(self.s,req)
        self.s._process_last_data=lambda data:Scheduler._process_last_data(self.s,data)
        self.s._free_req_resources=self.free

    def free(self,req):
        assert req not in self.freed
        self.freed.append(req);req.table_idx=-1

    @contextmanager
    def context(self,batch):
        self.batch=batch
        yield

    def prepare(self,batch,allocate=True):
        if self.fault=='prepare_before':raise RuntimeError(self.fault)
        start,end=self.req.cached_len,self.req.device_len
        if allocate:self.pages[0,start:end]=torch.arange(100,100+end-start)
        if self.fault=='prepare_after':raise RuntimeError(self.fault)
        positions=torch.arange(start,end)
        return ForwardInput(batch,None,(torch.zeros_like(positions),positions),None)

    def forward(self):
        self.calls+=1
        for tensor in self.states:tensor[:,1].add_(self.batch.input_ids.sum())
        if self.fault=='forward' or (self.fault=='replay' and self.calls==2):
            raise RuntimeError(self.fault)
        if self.abort:self.req.aborted=True
        logits=torch.full((len(self.batch.input_ids),16),-100.)
        for row,choice in enumerate(self.choices[:len(logits)]):logits[row,choice]=100.
        return logits

    def run(self):
        event=SimpleNamespace(record=lambda:None,synchronize=lambda:None)
        with patch('torch.cuda.Event',return_value=event), \
             patch('torch.cuda.current_stream',return_value=event), \
             patch.dict('os.environ',{'QWEN36_SPEC_WINDOW':'3','QWEN36_ENGINE_METRICS':'0'}):
            return try_step(self.s)

    def assert_rolled_back(self):
        assert (self.req.device_len,self.req.cached_len)==(4,3)
        assert self.req.input_ids.tolist()==[1,2,3,4]
        assert torch.equal(self.pages,self.original_pages)
        assert torch.equal(self.tokens,self.original_tokens)
        assert all(torch.equal(t,ref) for t,ref in zip(self.states,self.original))
        assert self.sent==[]


def test_faults_restore_unpublished_state_and_propagate():
    for fault in ('prepare_before','prepare_after','forward','replay'):
        f=Fixture(choices=(5,9,7,8),fault=fault)
        try:f.run()
        except RuntimeError as error:assert str(error)==fault
        else:raise AssertionError('fault was swallowed')
        f.assert_rolled_back()
        assert f.returned==([] if fault=='prepare_before' else [100,101,102,103])
        assert len(f.returned)==len(set(f.returned))


def test_acceptance_prefix_commits_and_frees_only_rejected_suffix():
    for accepted in range(4):
        choices=[5,6,7,8]
        if accepted<3:choices[accepted]=9
        emitted=[5,6,7][:accepted]+[choices[accepted]]
        f=Fixture(choices=choices)
        assert f.run()
        assert [m.next_token for m in f.sent]==emitted
        assert f.req.input_ids.tolist()==[1,2,3,4]+emitted
        assert (f.req.device_len,f.req.cached_len)==(4+len(emitted),3+len(emitted))
        # The last emitted token is not a processed KV input yet.
        delta=sum(([4]+[5,6,7])[:len(emitted)])
        for tensor,ref in zip(f.states,f.original):
            expected=ref.clone();expected[:,1]+=delta
            assert torch.equal(tensor,expected)
        assert f.returned==list(range(100+len(emitted),104))
        assert f.calls==(1 if accepted==3 else 2)


def test_eos_stop_and_tool_anchor_cut_the_verified_prefix():
    for mode in ('eos','stop','anchor'):
        f=Fixture(eos={6} if mode=='eos' else (),stops=['END'] if mode=='stop' else ())
        if mode=='anchor':f.s.toolcall_anchor_id=6
        assert f.run()
        expected=[5,6,7] if mode=='stop' else [5,6]
        assert [m.next_token for m in f.sent]==expected
        assert f.req.cached_len==3+len(expected)
        if mode=='anchor':
            assert f.req.toolcall_anchor_len==6 and not f.sent[-1].finished
        else:
            assert f.sent[-1].finish_reason=='stop' and len(f.freed)==1


def test_abort_during_verification_restores_then_frees_without_output():
    f=Fixture(abort=True)
    assert f.run()
    f.assert_rolled_back()
    assert f.req.aborted and len(f.freed)==1
    assert not f.s.decode_manager.running_reqs
    assert f.req in f.s.finished_reqs


def test_mtp_draft_exception_always_truncates_local_cache():
    f=Fixture()
    class Draft:
        length=3
        def forward(self,*args):
            self.length+=1
            raise RuntimeError('draft failure')
        def truncate(self,length):self.length=length
    draft=Draft();f.s.engine.native_mtp=draft
    f.s.engine._mtp_eligible=True;f.s.engine._mtp_previous=torch.zeros((1,2))
    try:f.run()
    except RuntimeError as error:assert str(error)=='draft failure'
    else:raise AssertionError('draft exception was swallowed')
    assert draft.length==3
    f.assert_rolled_back()


def test_request_state_keeps_explicit_seed_and_cost_object():
    f=Fixture();first=state(f.req)
    assert f.req.spec_seed==193
    assert state(f.req) is first and f.req.spec_seed==193


def test_sampling_and_short_budget_remain_guarded():
    f=Fixture();f.req.sampling_params.temperature=.7
    with patch.dict('os.environ',{'QWEN36_SPEC_SAMPLING':'0'}):
        assert not f.run()
    f.assert_rolled_back()
    f=Fixture();f.req.device_len=f.req.max_device_len-1
    assert not f.run()
    assert f.calls==0 and f.returned==[]


def test_published_output_is_not_rolled_back_if_transport_raises():
    f=Fixture()
    ordinary=f.s._process_last_data
    def publish_then_fail(data):
        ordinary(data)
        raise RuntimeError('reply transport failed')
    f.s._process_last_data=publish_then_fail
    try:f.run()
    except RuntimeError as error:assert str(error)=='reply transport failed'
    else:raise AssertionError('transport failure was swallowed')
    # Once output may be externally visible, the worker must fail closed rather
    # than rewind its request and permit a retry to duplicate published tokens.
    assert len(f.sent)==1 and f.sent[0].next_token==5
    assert f.req.input_ids.tolist()==[1,2,3,4,5]
    assert f.returned==[]


def test_terminal_metrics_include_the_last_attempt_before_teacher_completion():
    from freetoken.engine import telemetry
    for terminal in (False,True):
        f=Fixture(eos={6} if terminal else ())
        original_free=f.free
        observed=[]
        def free_with_metrics(req):
            telemetry.complete(req,f.s.engine,cache_manager=f.s.cache_manager)
            original_free(req)
        f.s._free_req_resources=free_with_metrics
        def teacher_complete(uid):
            observed.append((uid,dict(f.req.spec_cost.report())))
        with patch.object(telemetry,'ENABLED',False), \
             patch('freetoken.engine.qualification.complete',side_effect=teacher_complete):
            assert f.run()
        cost=f.req.spec_cost.report()
        assert cost['attempts']==1 and cost['proposed']==3
        assert cost['accepted']==(2 if terminal else 3)
        assert cost['emitted']==(2 if terminal else 4)
        assert cost['publish_s']>=0 and cost['total_s']>=cost['publish_s']
        assert cost['wall_timing_includes_publication'] is True
        assert cost['verify_replay_in_engine_gpu_timers'] is False
        assert cost['publication_failed'] is False
        assert not f.req._qwen36_spec_metrics_deferred
        assert len(observed)==(1 if terminal else 0)
        if terminal:
            assert observed[0][1]==cost


def test_terminal_transport_exception_clears_defer_and_observes_once():
    from freetoken.engine import telemetry
    f=Fixture(eos={6})
    original_free=f.free
    observed=[]
    def free_with_metrics(req):
        telemetry.complete(req,f.s.engine,cache_manager=f.s.cache_manager)
        original_free(req)
    f.s._free_req_resources=free_with_metrics
    ordinary=f.s._process_last_data
    def terminal_then_fail(data):
        ordinary(data)
        if f.req.table_idx==-1:raise RuntimeError('terminal transport failed')
    f.s._process_last_data=terminal_then_fail
    with patch.object(telemetry,'ENABLED',False), \
         patch('freetoken.engine.qualification.complete',side_effect=lambda uid:observed.append(dict(f.req.spec_cost.report()))):
        try:f.run()
        except RuntimeError as error:assert str(error)=='terminal transport failed'
        else:raise AssertionError('terminal transport failure swallowed')
    assert len(observed)==1
    assert observed[0]['attempts']==1 and observed[0]['emitted']==2
    assert observed[0]['publication_failed'] is True
    assert not f.req._qwen36_spec_metrics_deferred
    assert [reply.next_token for reply in f.sent]==[5,6]
    assert len(f.freed)==1
