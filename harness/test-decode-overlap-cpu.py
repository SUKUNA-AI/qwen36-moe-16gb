"""Execute production control flow with fake streams; no CUDA race claim."""
import ast
import contextlib
import importlib.util
import json
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from cpu_control_flow import methods
from repo_paths import FREETOKEN, SERVING, HARNESS
from freetoken.decode_overlap import DecodeSharedOverlap, enabled, ENV

MOE=FREETOKEN/'python/freetoken/layers/moe.py'
MODEL=FREETOKEN/'python/freetoken/models/qwen3_5_moe/moe.py'
HELPER=FREETOKEN/'python/freetoken/decode_overlap.py'

class FakeCuda:
    def __init__(self):
        self.trace=[];self.event_count=0;self.main=self.Stream(name='main');self.current=self.main
    def Stream(self,device=None,name='side'):
        trace=self.trace
        class Stream:
            def wait_event(self,event):trace.append(f'{name}<-{event.name}')
        stream=Stream();stream.name=name;return stream
    def Event(self):
        name=('ready','done')[self.event_count];self.event_count+=1
        return S(name=name,record=lambda stream:self.trace.append(f'{name}@{stream.name}'))
    def current_stream(self,device):return self.current
    @contextlib.contextmanager
    def stream(self,stream):
        old=self.current;self.current=stream
        try:yield
        finally:self.current=old


def flow(mutate=None, failure=None):
    cuda=FakeCuda();overlap=DecodeSharedOverlap(cuda,'cuda');cuda.trace.clear()
    if mutate:
        overlap.run=methods(HELPER,'DecodeSharedOverlap',{'run'},{},mutate)['run'].__get__(overlap)
    def mark(name,result=None):
        def call(*a,**kw):
            cuda.trace.append(name)
            if failure==name:raise RuntimeError(name)
            return result
        return call
    cache=S(decode_shared_overlap=overlap,ensure_experts=mark('ensure'),copy_missing=mark('copy@side'),
            bank_views=lambda:(),alphas_for_slots=lambda _:None)
    obj=S(offload_cache=cache,top_k=8,renormalize=True,layer_id=0,_expert_gemm=mark('routed',17))
    call=methods(MOE,'OffloadMoELayer',{'forward_with_shared'},{'fused_topk':mark('topk',(1,2))})['forward_with_shared']
    error=None
    try:result=call(obj,'hidden','router',mark('shared@main',23))
    except RuntimeError as e:error=str(e);result=None
    return cuda.trace,result,error

EXPECTED=['topk','ensure','ready@main','side<-ready','copy@side','done@side','shared@main','main<-done','routed']

class OverlapTests(unittest.TestCase):
    def test_actual_routed_control_flow(self):
        trace,result,error=flow();self.assertEqual(trace,EXPECTED);self.assertEqual(result,(17,23));self.assertIsNone(error)
    def test_shared_exception_joins_before_propagating(self):
        trace,_,error=flow(failure='shared@main');self.assertEqual(trace,EXPECTED[:-1]);self.assertEqual(error,'shared@main')
    def test_copy_exception_records_completion_and_joins(self):
        trace,_,error=flow(failure='copy@side');self.assertEqual(trace,EXPECTED[:6]+['main<-done']);self.assertEqual(error,'copy@side')
    def test_producer_and_consumer_wait_mutants_are_detected(self):
        for receiver in ('self.stream','producer'):
            def mutate(tree):
                class Remove(ast.NodeTransformer):
                    def visit_Expr(self,node):
                        call=node.value
                        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='wait_event' and ast.unparse(call.func.value)==receiver:
                            return ast.Pass()
                        return self.generic_visit(node)
                Remove().visit(tree)
            with self.subTest(receiver=receiver):self.assertNotEqual(flow(mutate)[0],EXPECTED)
    def test_repeated_calls_rejoin_before_next_metadata(self):
        cuda=FakeCuda();overlap=DecodeSharedOverlap(cuda,'cuda');cuda.trace.clear()
        for _ in range(3):overlap.run(lambda:cuda.trace.append('copy'),lambda:cuda.trace.append('shared'))
        self.assertEqual(cuda.trace,3*['ready@main','side<-ready','copy','done@side','shared','main<-done'])
    def test_handles_are_materialized_before_capture(self):
        cuda=FakeCuda();DecodeSharedOverlap(cuda,'cuda');self.assertEqual(cuda.trace,['ready@main','side<-ready','done@side','main<-done'])
    def test_env_fail_closed(self):
        with patch.dict(os.environ,{},clear=True):self.assertFalse(enabled())
        for value,want in [('',False),('0',False),('1',True)]:
            with patch.dict(os.environ,{ENV:value}):self.assertEqual(enabled(),want)
        for value in ('true','on','2','-1',' 1 '):
            with patch.dict(os.environ,{ENV:value}),self.assertRaises(ValueError):enabled()
    def test_unsupported_modes_use_serial(self):
        cache=S(decode_shared_overlap=object(),decode_target='gpu',is_cpu_layer=lambda _:False,
                is_unpinned_layer=lambda _:False,_copy_fused_ok=True,layer_major_active=False,refill=None)
        obj=S(offload_cache=cache,tp_size=1,layer_id=0);batch=S(is_prefill=False)
        call=methods(MOE,'OffloadMoELayer',{'can_overlap_shared'},{'get_global_ctx':lambda:S(batch=batch)})['can_overlap_shared']
        hidden=S(shape=(1,32));self.assertTrue(call(obj,hidden))
        for target,key,value in [(cache,'decode_shared_overlap',None),(cache,'decode_target','cpu'),(cache,'decode_target','hybrid'),
                                 (cache,'_copy_fused_ok',False),(cache,'layer_major_active',True),(cache,'refill',object()),
                                 (cache,'is_cpu_layer',lambda _:True),(cache,'is_unpinned_layer',lambda _:True),
                                 (obj,'tp_size',2),(batch,'is_prefill',True),(hidden,'shape',(2,32))]:
            old=getattr(target,key);setattr(target,key,value)
            with self.subTest(key=key):self.assertFalse(call(obj,hidden))
            setattr(target,key,old)
    def test_model_default_and_enabled_merge_keep_input_and_order(self):
        for overlap in (False,True):
            trace=[];hidden=torch.tensor([[2.,3.]])
            def shared(x):trace.append('shared');return x*3
            def routed(hidden_states,router_logits):trace.append('routed');hidden_states.fill_(5);return hidden_states
            experts=S(can_overlap_shared=lambda _:overlap,forward=routed)
            def combined(x,r,cb):
                shared_result=cb(x);return routed(x,r),shared_result
            experts.forward_with_shared=combined
            obj=S(experts=experts,gate=S(forward=lambda x:x.clone()),_shared_forward=shared)
            call=methods(MODEL,'Qwen3_5MoE',{'forward'},{})['forward']
            self.assertTrue(torch.equal(call(obj,hidden),torch.tensor([[11.,14.]])))
            self.assertEqual(trace,['shared','routed'])
    def test_profile_is_optional_hash_bound_and_unqualified(self):
        spec=importlib.util.spec_from_file_location('overlap_profile',SERVING/'profile-runtime.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
        # the main128 release draft enables overlap; the baseline here is the same profile without it
        p=dict(json.loads((SERVING/'profiles.json').read_text())['profiles']['main128'],decode_shared_overlap=False);r.validate_profile_schema('main128',p)
        candidate=dict(p,qualified=False,decode_shared_overlap=True);r.validate_profile_schema('main128',candidate)
        self.assertNotEqual(r.canonical_profile_sha256(candidate),r.canonical_profile_sha256(dict(candidate,decode_shared_overlap=False)))
        for value in ('1',1,None):
            with self.assertRaises(ValueError):r.validate_profile_schema('main128',dict(candidate,decode_shared_overlap=value))
        # T3 is qualifiable only with receipt evidence (QUALIFIABLE_EXPERIMENTAL + EXPERIMENTAL_EVIDENCE); a closed guard refuses it.
        r.validate_profile_schema('main128',dict(candidate,qualified=True))
        record={'evidence':[{'path':'e.json','sha256':'0'*64}]}
        with self.assertRaisesRegex(ValueError,'needs component, teacher and A/B'):r._experimental_evidence_valid('main128',dict(candidate,qualified=True),record)
        r.QUALIFIABLE_EXPERIMENTAL=frozenset()
        with self.assertRaisesRegex(ValueError,'experimental'):r.validate_profile_schema('main128',dict(candidate,qualified=True))
        with self.assertRaisesRegex(ValueError,'refill'):r.validate_profile_schema('main128',dict(candidate,layer_major_refill=True))
        env={ENV:'1'};r.configure_environment('main128',p,env);self.assertNotIn(ENV,env)
        r.configure_environment('main128',candidate,env);self.assertEqual(env[ENV],'1')
        for name in ('main-release-overlap','main-release-shortroute-fla-wu-overlap'):
            cfg=json.loads((HARNESS/(name+'.json')).read_text());self.assertIs(cfg['qualified'],False);self.assertEqual(cfg['env'][ENV],'1')
            spec=importlib.util.spec_from_file_location('overlap_receipt',HARNESS/'make-qualification.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
            mapped=q.run_equivalent_profile(cfg,r.FEATURE_SWITCHES,r.KERNEL_LOADER_ENV,r.VALUE_FIELDS)
            self.assertIs(mapped['decode_shared_overlap'],True)
            cfg['env'][ENV]='unknown'
            with self.assertRaises(ValueError):q.run_equivalent_profile(cfg,r.FEATURE_SWITCHES,r.KERNEL_LOADER_ENV,r.VALUE_FIELDS)

if __name__=='__main__':unittest.main(verbosity=2)
