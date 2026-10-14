import os
import contextlib
import types
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as S
from cpu_control_flow import methods
from cpu_imports import enable
from repo_paths import FREETOKEN
enable()
import torch
from freetoken.engine.layer_major import min_tokens
from freetoken.scheduler.layer_major import LayerMajorJob
CACHE=FREETOKEN/'python/freetoken/moe/offload_cache.py'

class ShortRoute(unittest.TestCase):
    def test_parser_patch_compatibility(self):
        for raw,want in ((None,0),('',0),('0',0),('-1',0),('bad',0),('2049',2049)):
            with self.subTest(raw=raw),patch.dict(os.environ,{},clear=True):
                if raw is not None:os.environ['QWEN36_LAYER_MAJOR_MIN_TOKENS']=raw
                self.assertEqual(min_tokens(),want)
    def route(self, full, cached, budget, threshold):
        from freetoken.scheduler import layer_major as lm
        req=S(cached_len=cached,extend_len=min(budget,full-cached),device_len=cached+min(budget,full-cached))
        sched=S(prefill_manager=S(pending_list=[S(input_len=full)]),prefill_budget=budget,
                cache_manager=S(prefill_chunk_align=1),engine=S(moe_offload_cache=None,
                model=S(model=S(embed_tokens=S(weight=torch.zeros(1,8))))))
        fi=S(batch=S(reqs=[req]))
        moe=types.ModuleType('freetoken.moe.layer_major');moe.bank_row_bytes=lambda c:[]
        moe.lendable_slots=lambda c:0;plans=[]
        def plan(*a,**k):plans.append(a);return None
        moe.plan_store=plan
        with patch.dict(os.environ,{'QWEN36_LAYER_MAJOR_MIN_TOKENS':str(threshold)}),patch.dict(sys.modules,{'freetoken.moe.layer_major':moe}),\
             patch.object(lm,'scheduler_reason',return_value=None),patch('freetoken.engine.layer_major.store_banks',return_value=(None,False)):
            LayerMajorJob.maybe_start(sched,fi)
        return bool(plans)
    def test_boundaries_prefix_suffix_and_chunk_progression(self):
        for chunk in (128,1024,2048):
            for n in (2048,2049,2050):self.assertEqual(self.route(n,0,chunk,2049),n>=2049)
            self.assertFalse(self.route(16000,15900,chunk,2049))
            self.assertFalse(self.route(16000,16000-2048,chunk,2049))
            self.assertTrue(self.route(16000,16000-2049,chunk,2049))
            self.assertFalse(self.route(10000,10000-100,chunk,0))
        for n in (599,600,601):self.assertEqual(self.route(n,0,128,600),n>=600)
        self.assertTrue(self.route(700,0,128,0))
    def cache(self):
        ns=methods(CACHE,'OffloadMoeCache',{'begin_layer_major_prefill','end_layer_major_prefill','begin_prefill',
                          'prefetch_prefill_layer','_invalidate_prefill_buffer'}, {'torch':torch})
        cls=type('OffloadMoeCache',(),{k:v for k,v in ns.items() if callable(v) and k!='torch'});ns['OffloadMoeCache']=cls
        c=cls();c.device=torch.device('cpu');c.layer_major_active=False;c.prefill_hit_d2d=True;c._prefill_hit_d2d_active=True
        c.prefill_overlap=True;c.prefill_copy_stream=None;c._hit_d2d_usable=lambda:True
        c.num_layers=2;c.num_experts=2;c.syncfree_invalidate=True
        c.slot_for_id=torch.tensor([[4,5],[6,7]],dtype=torch.int32)
        c.id_of_slot=torch.full((8,),-1,dtype=torch.int32);c.usage=torch.zeros(8,dtype=torch.int32)
        c.banks=[([torch.ones(2,2),torch.ones(2,2)],torch.zeros(8,2))]
        c.prefill_bank_buffers=[torch.zeros(2,2,2)];c.prefill_h2d_bytes=0
        c._prefetch_split=lambda *a: self.fail('D2D read of borrowed slots')
        return c
    def test_epoch_lifetime_and_restoration(self):
        c=self.cache();c.begin_layer_major_prefill()
        self.assertTrue(c.prefill_hit_d2d);self.assertFalse(c._prefill_hit_d2d_active)
        c.begin_prefill();c._prefill_hit_d2d_active=True
        c.prefetch_prefill_layer(0);self.assertGreater(c.prefill_h2d_bytes,0)
        c.end_layer_major_prefill();self.assertFalse(c.layer_major_active)
        c.prefill_copy_stream=S(wait_event=lambda e:None,synchronize=lambda:None)
        c.prefill_begin_event=S(record=lambda s:None)
        c._prefill_slot_snapshot=torch.empty_like(c.slot_for_id)
        with patch('torch.cuda.stream',return_value=contextlib.nullcontext()),patch('torch.cuda.current_stream',return_value=None):
            c.begin_prefill()
        self.assertTrue(c._prefill_hit_d2d_active)
    def test_engine_coexistence_gate(self):
        from freetoken.engine.layer_major import engine_reason
        c=self.cache()
        class Model:
            def forward(self):pass
            layer_major_plain_forward=forward
        model=Model();model.model=S(layer_major_supported=True)
        engine=S(model=model,attn_backend=S(layer_major_safe=True),moe_offload_cache=c)
        module=types.ModuleType('freetoken.moe.offload_cache');module.OffloadMoeCache=type(c)
        with patch.dict(sys.modules,{'freetoken.moe.offload_cache':module}):
            with patch.dict(os.environ,{'QWEN36_LAYER_MAJOR_MIN_TOKENS':'0'}):self.assertIn('hit-D2D',engine_reason(engine))
            with patch.dict(os.environ,{'QWEN36_LAYER_MAJOR_MIN_TOKENS':'2049'}):self.assertIsNone(engine_reason(engine))

    def test_begin_exception_and_cancel(self):
        for error in (RuntimeError('synthetic'), KeyboardInterrupt()):
            c=self.cache();c.begin_prefill=lambda:(_ for _ in ()).throw(error)
            with self.assertRaises(type(error)):c.begin_layer_major_prefill()
            self.assertFalse(c.layer_major_active);self.assertFalse(c._prefill_hit_d2d_active);self.assertTrue(c.prefill_hit_d2d)
        c=self.cache();c.begin_layer_major_prefill();c.end_layer_major_prefill();c.end_layer_major_prefill()
        self.assertFalse(c.layer_major_active)
if __name__=='__main__':unittest.main(verbosity=2)
