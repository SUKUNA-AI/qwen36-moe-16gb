"""D3b speculative pre-gate prefetch (moe/pregate.py): fake-stream control flow; no CUDA race claim.

The Triton kernels run under TRITON_INTERPRET=1 against Python references. The GPU byte gate is
freetoken/tests/moe/test_decode_pregate_gpu.py; the model teacher gate is the coordinator's.
"""
import os

os.environ['TRITON_INTERPRET'] = '1'  # before triton.jit decorates the kernels below

import ast
import contextlib
import random
import sys
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch

import torch
from cpu_control_flow import methods
from repo_paths import FREETOKEN

from freetoken.moe import pregate as P

CACHE = FREETOKEN / 'python/freetoken/moe/offload_cache.py'
MOE = FREETOKEN / 'python/freetoken/layers/moe.py'
MODEL = FREETOKEN / 'python/freetoken/models/qwen3_5_moe/model.py'
HELPER = FREETOKEN / 'python/freetoken/moe/pregate.py'


class FakeCuda:
    def __init__(self):
        self.trace = []
        self.main = self.Stream(name='main')
        self.current = self.main

    def Stream(self, device=None, name='side', **kw):
        trace = self.trace
        self.made = getattr(self, 'made', []) + [kw]

        class Stream:
            def wait_event(self, event):
                trace.append(f'{name}<-{event.name}')

        stream = Stream()
        stream.name = name
        return stream

    def Event(self):
        names = ('fork', 'read', 'done')
        name = names[sum(1 for _ in getattr(self, '_events', ()))]
        self._events = getattr(self, '_events', []) + [name]
        return S(name=name, record=lambda stream: self.trace.append(f'{name}@{stream.name}'))

    def current_stream(self, device):
        return self.current

    @contextlib.contextmanager
    def stream(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old


def make_pregate(linear=1, attn=2, fail=None, mutate=None):
    cuda = FakeCuda()
    cache = S(num_experts=16, num_layers=4)
    pg = P.DecodePregate(cache, cuda, 'cpu', linear, attn)
    cuda.trace.clear()
    if mutate is not None:
        ns = methods(HELPER, 'DecodePregate', {'launch', 'release_input', 'join'}, {'torch': torch}, mutate)
        for name in ('launch', 'release_input', 'join'):
            setattr(pg, name, ns[name].__get__(pg))

    def mark(name):
        def call(*a, **kw):
            cuda.trace.append(f'{name}@{cuda.current.name}')
            if fail == name:
                raise RuntimeError(name)
        return call

    pg._predict = mark('predict')
    pg._admit_and_copy = mark('admit')
    return pg, cuda


FORK = ['fork@main', 'side<-fork', 'predict@side', 'read@side', 'admit@side', 'done@side']


class Flags(unittest.TestCase):
    def test_caps(self):
        cases = (({}, (0, 0)), ({P.ENV: ''}, (0, 0)), ({P.ENV: '1'}, (1, 1)), ({P.ENV: '1', P.ATTN_ENV: '2'}, (1, 2)),
                 ({P.ATTN_ENV: '3'}, (0, 3)), ({P.ENV: ' 2 '}, (2, 2)), ({P.ENV: '8', P.ATTN_ENV: '0'}, (8, 0)))
        for env, want in cases:
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(P.caps(), want, env)
        for bad in ('9', '-1', 'on', '1.5', 'true'):
            for name in (P.ENV, P.ATTN_ENV):
                with patch.dict(os.environ, {name: bad}, clear=True), self.assertRaises(ValueError):
                    P.caps()

    def test_priority_flag_and_stream_choice(self):
        for raw, want in ((None, False), ('', False), ('0', False), ('1', True)):
            with patch.dict(os.environ, {} if raw is None else {P.PRIORITY_ENV: raw}, clear=True):
                self.assertEqual(P.priority(), want)
        with patch.dict(os.environ, {P.PRIORITY_ENV: 'yes'}, clear=True), self.assertRaises(ValueError):
            P.priority()
        cache = S(num_experts=16, num_layers=4)
        t3 = S(name='t3', wait_event=lambda e: None)
        cuda = FakeCuda()
        self.assertIs(P.DecodePregate(cache, cuda, 'cpu', 1, 2, stream=t3).stream, t3, 'shares T3 by default')
        cuda = FakeCuda()
        own = P.DecodePregate(cache, cuda, 'cpu', 1, 2, stream=t3, high_priority=True)
        self.assertIsNot(own.stream, t3)
        self.assertEqual(cuda.made[-1], {'priority': -1})

    def test_cache_wiring_is_default_off_and_excludes_refill_and_trace(self):
        text = CACHE.read_text(encoding='utf-8')
        body = text[text.index('pregate_caps = caps()'):text.index('def begin_layer_major_prefill')]
        for cond in ('any(pregate_caps)', 'not refill_enabled()', 'self.route_trace is None',
                     'self.device.type == "cuda"', 'self.decode_target == "gpu"',
                     'self.decode_shared_overlap.stream if self.decode_shared_overlap else None',
                     'high_priority=priority()'):
            self.assertIn(cond, body)


class Protocol(unittest.TestCase):
    def test_fork_read_done_and_joins(self):
        pg, cuda = make_pregate()
        pg.launch(3, None, None, None, 1, 8)
        self.assertEqual(cuda.trace, FORK)
        pg.release_input()
        pg.release_input()
        pg.join()
        pg.join()
        self.assertEqual(cuda.trace, FORK + ['main<-read', 'main<-done'])

    def test_join_without_release_waits_done_only(self):
        pg, cuda = make_pregate()
        pg.launch(1, None, None, None, 1, 8)
        pg.join()
        pg.release_input()
        self.assertEqual(cuda.trace, FORK + ['main<-done'])

    def test_a_second_launch_joins_the_first(self):
        pg, cuda = make_pregate()
        pg.launch(1, None, None, None, 1, 8)
        pg.launch(2, None, None, None, 1, 8)
        self.assertEqual(cuda.trace, FORK + ['main<-done'] + FORK)

    def test_failures_record_both_events_and_join(self):
        for fail, side in (('predict', ['predict@side', 'read@side', 'done@side']),
                           ('admit', ['predict@side', 'read@side', 'admit@side', 'done@side'])):
            pg, cuda = make_pregate(fail=fail)
            with self.assertRaises(RuntimeError):
                pg.launch(1, None, None, None, 1, 8)
            self.assertEqual(cuda.trace, ['fork@main', 'side<-fork'] + side + ['main<-done'], fail)
            self.assertIs(cuda.current, cuda.main)
            pg.join()
            pg.release_input()
            self.assertEqual(cuda.trace[-1], 'main<-done')

    def test_wait_mutants_are_detected(self):
        for receiver in ('self.stream', 'self.cuda.current_stream(self.device)'):
            def mutate(tree, receiver=receiver):
                class Remove(ast.NodeTransformer):
                    def visit_Expr(self, node):
                        call = node.value
                        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                                and call.func.attr == 'wait_event' and ast.unparse(call.func.value) == receiver):
                            return ast.Pass()
                        return self.generic_visit(node)
                Remove().visit(tree)
            pg, cuda = make_pregate(mutate=mutate)
            pg.launch(1, None, None, None, 1, 8)
            pg.release_input()
            pg.join()
            self.assertNotEqual(cuda.trace, FORK + ['main<-read', 'main<-done'], receiver)

    def test_caps_per_mixer(self):
        pg, _ = make_pregate(linear=1, attn=3)
        self.assertEqual((pg.cap(False), pg.cap(True), pg.width), (1, 3, 3))
        pg, _ = make_pregate(linear=1, attn=1)
        self.assertEqual(pg.width, 2, 'lru_ensure always sees a query of at least two ids')

    def test_predictor_shape_checks(self):
        res = torch.zeros(1, 32, dtype=torch.bfloat16)
        norm = S(weight=torch.ones(32, dtype=torch.bfloat16), eps=1e-6)
        gate = S(weight=torch.zeros(16, 32, dtype=torch.bfloat16))
        self.assertTrue(P.DecodePregate.predictor_ok(res, norm, gate, 16))
        self.assertFalse(P.DecodePregate.predictor_ok(res, norm, S(weight=torch.zeros(16, 32).t()), 16))
        self.assertFalse(P.DecodePregate.predictor_ok(res, norm, gate, 24))
        self.assertFalse(P.DecodePregate.predictor_ok(res, S(weight=None, eps=1e-6), gate, 16))
        self.assertFalse(P.DecodePregate.predictor_ok(torch.zeros(1, 64)[:, ::2], norm, gate, 16))


def decoder_layer(trace, *, can=True, cap=1, linear=True, mutate=None):
    ns = methods(MODEL, 'Qwen3_5DecoderLayer', {'forward', '_launch_pregate'},
                 {'nvtx_annotate': lambda *a, **k: (lambda f: f)}, mutate)
    pregate = S(cap=lambda attention: cap, predictor_ok=lambda *a: True,
                launch=lambda lid, res, norm, gate, c, k: trace.append(f'launch{lid}:{c}/{k}'),
                release_input=lambda: trace.append('release'))
    cache = S(decode_pregate=pregate)
    experts = S(offload_cache=cache, layer_id=5, num_experts=16, top_k=8, can_pregate=lambda r: can)

    def mlp_forward(h):
        trace.append('mlp')
        return h

    def norm(name):
        return S(forward_add_residual=lambda h, r: (trace.append(name), (h, r))[1], forward=lambda h: h)

    layer = S(_layer_id=5, _is_linear=linear, input_layernorm=norm('input_norm'),
              post_attention_layernorm=norm('post_norm'),
              linear_attn=S(forward=lambda h: (trace.append('gdn'), h)[1]),
              self_attn=S(forward=lambda h: (trace.append('attn'), h)[1]),
              mlp=S(experts=experts, gate=S(), forward=mlp_forward))
    layer._launch_pregate = ns['_launch_pregate'].__get__(layer)
    return ns['forward'], layer


class DecoderLayer(unittest.TestCase):
    def test_fork_after_input_norm_release_before_post_norm(self):
        trace = []
        forward, layer = decoder_layer(trace, cap=3)
        forward(layer, 'h', 'r')
        self.assertEqual(trace, ['input_norm', 'launch5:3/8', 'gdn', 'release', 'post_norm', 'mlp'])

    def test_cap_is_clamped_to_top_k_and_per_mixer(self):
        trace = []
        forward, layer = decoder_layer(trace, cap=8, linear=False)
        layer.mlp.experts.top_k = 4
        forward(layer, 'h', 'r')
        self.assertEqual(trace, ['input_norm', 'launch5:4/4', 'attn', 'release', 'post_norm', 'mlp'])

    def test_disabled_paths_never_fork(self):
        for kw in ({'can': False}, {'cap': 0}):
            trace = []
            forward, layer = decoder_layer(trace, **kw)
            forward(layer, 'h', 'r')
            self.assertEqual(trace, ['input_norm', 'gdn', 'post_norm', 'mlp'], kw)
        trace = []
        forward, layer = decoder_layer(trace)
        layer.mlp = S(forward=lambda h: (trace.append('mlp'), h)[1])  # dense MLP / resident experts
        forward(layer, 'h', 'r')
        self.assertEqual(trace, ['input_norm', 'gdn', 'post_norm', 'mlp'])

    def test_release_mutant_is_detected(self):
        def mutate(tree):
            class Remove(ast.NodeTransformer):
                def visit_If(self, node):
                    if 'release_input' in ast.unparse(node):
                        return ast.Pass()
                    return self.generic_visit(node)
            Remove().visit(tree)
        trace = []
        forward, layer = decoder_layer(trace, mutate=mutate)
        forward(layer, 'h', 'r')
        self.assertNotIn('release', trace)


class Gating(unittest.TestCase):
    def layer(self, prefill=False):
        ns = methods(MOE, 'OffloadMoELayer', {'can_pregate'},
                     {'get_global_ctx': lambda: S(batch=S(is_prefill=prefill))})
        cache = S(decode_pregate=object(), decode_target='gpu', is_cpu_layer=lambda l: False,
                  is_unpinned_layer=lambda l: False, _copy_fused_ok=True, layer_major_active=False, refill=None)
        layer = S(offload_cache=cache, layer_id=3, tp_size=1)
        return (lambda r: ns['can_pregate'](layer, r)), layer, cache

    def test_conditions(self):
        res = torch.zeros(1, 8)
        can, layer, cache = self.layer()
        self.assertTrue(can(res))
        self.assertFalse(can(None))
        self.assertFalse(can(torch.zeros(2, 8)), 'batch > 1')
        for owner, attr, value in ((layer, 'layer_id', 0), (layer, 'tp_size', 2), (layer, 'offload_cache', None),
                                   (cache, 'decode_pregate', None), (cache, 'decode_target', 'hybrid'),
                                   (cache, 'is_cpu_layer', lambda l: True), (cache, 'is_unpinned_layer', lambda l: True),
                                   (cache, '_copy_fused_ok', False), (cache, 'layer_major_active', True),
                                   (cache, 'refill', object())):
            old = getattr(owner, attr)
            setattr(owner, attr, value)
            self.assertFalse(can(res), attr)
            setattr(owner, attr, old)
        self.assertFalse(self.layer(prefill=True)[0](res), 'prefill')


class CacheJoins(unittest.TestCase):
    JOINING = ('ensure_experts', 'ensure_experts_hybrid', 'materialize_layer', 'begin_prefill', 'reset', 'rebuild',
               'borrow_tail_slots')

    def test_every_metadata_writer_joins_first(self):
        tree = ast.parse(CACHE.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OffloadMoeCache')
        bodies = {n.name: ast.unparse(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name in self.JOINING:
            body = bodies[name]
            self.assertIn('self.decode_pregate.join()', body, name)
        for name, first_write in (('ensure_experts', 'ensure_experts(self, layer_id, expert_ids)'),
                                  ('materialize_layer', 'materialize_layer(self, layer_id)'),
                                  ('reset', 'reset_cache(self)'), ('begin_prefill', 'self._prefill_buffer_layer ='),
                                  ('borrow_tail_slots', 'sf.masked_fill_'), ('rebuild', 'self.refill = None')):
            body = bodies[name]
            self.assertLess(body.index('self.decode_pregate.join()'), body.index(first_write), name)

    def test_ensure_experts_joins_before_the_lru_kernel(self):
        log = []
        kernels = S(ensure_experts=lambda cache, lid, ids: log.append(f'lru{lid}'))
        ns = methods(CACHE, 'OffloadMoeCache', {'ensure_experts'}, {'torch': torch})
        cache = S(collect_decode_freq=False, route_trace=None,
                  decode_pregate=S(join=lambda: log.append('join')))
        with patch.dict(sys.modules, {'freetoken.moe.offload_kernels': kernels}):
            ns['ensure_experts'](cache, 4, torch.zeros(8, dtype=torch.int32))
            cache.decode_pregate = None
            ns['ensure_experts'](cache, 5, torch.zeros(8, dtype=torch.int32))
        self.assertEqual(log, ['join', 'lru4', 'lru5'])
        self.assertEqual(cache._pending_src_layer, 5)

    def test_copy_dispatch_is_unchanged(self):
        log = []
        gather = S(decode_gather_jit=lambda dst, src, feat, ev, si, ni, config: log.append(('tma', src)))
        fast = S(fast_index_copy_multi_jit=lambda dst, src, feat, ev, si, ni: log.append(('fast', src)))
        ns = methods(CACHE, 'OffloadMoeCache', {'copy_missing', 'copy_planned'}, {'torch': torch})
        cache = S(banks=[1], _unpinned_layers=frozenset(), _copy_fused_ok=True, _copy_dst_ptrs='d',
                  _copy_src_ptrs=['s0', 's1', 's2'], _copy_feat_bytes='f', evict_slots='e', src_indices='i',
                  num_indices='n', _decode_gather='cfg')
        cache.copy_planned = ns['copy_planned'].__get__(cache)
        with patch.dict(sys.modules, {'freetoken.kernel.decode_gather': gather,
                                      'freetoken.kernel.fast_index_copy': fast}):
            for layer, whole, cfg in ((1, False, 'cfg'), (2, True, 'cfg'), (0, False, None)):
                cache._pending_src_layer, cache._pending_whole_layer, cache._decode_gather = layer, whole, cfg
                ns['copy_missing'](cache)
            cache._decode_gather = 'cfg'
            cache.copy_planned(2)
        self.assertEqual(log, [('tma', 's1'), ('fast', 's2'), ('fast', 's0'), ('tma', 's2')])


def select_reference(logits, slot_row, cap, top_k, width):
    order = sorted(range(len(logits)), key=lambda e: (-logits[e], e))[:top_k]
    miss = [e for e in order if slot_row[e] < 0][:cap]
    pad = miss[0] if miss else order[0]
    return (miss + [pad] * width)[:width]


class Kernels(unittest.TestCase):
    def test_select_matches_reference(self):
        rng = random.Random(7)
        for trial in range(60):
            E = rng.choice((16, 24, 32))
            cap = rng.randint(1, 4)
            width = max(2, cap + rng.randint(0, 2))
            values = [float(rng.randint(-6, 6)) for _ in range(E)]  # many ties: lower id must win
            slots = [rng.choice((-1, -1, 0, 3, 7)) for _ in range(E)]
            logits = torch.tensor(values, dtype=torch.float32)
            row = torch.tensor(slots, dtype=torch.int32)
            query = torch.full((width,), -7, dtype=torch.int32)
            P._pregate_select_kernel[(1,)](logits, row, query, cap, E, TOP_K=8, Q=width,
                                           BLOCK=1 << (E - 1).bit_length())
            self.assertEqual(query.tolist(), select_reference(values, slots, cap, 8, width), (trial, values, slots))

    def test_select_keeps_ids_in_range_for_nan(self):
        logits = torch.full((16,), float('nan'))
        query = torch.full((2,), -7, dtype=torch.int32)
        P._pregate_select_kernel[(1,)](logits, torch.full((16,), -1, dtype=torch.int32), query, 2, 16,
                                       TOP_K=8, Q=2, BLOCK=16)
        self.assertTrue(all(0 <= q < 16 for q in query.tolist()))

    def test_logits_follow_rmsnorm_then_router(self):
        torch.manual_seed(0)
        H, E = 64, 16
        res = torch.randn(1, H).to(torch.bfloat16)
        w = (torch.rand(H) + 0.5).to(torch.bfloat16)
        gate = torch.randn(E, H).to(torch.bfloat16)
        out = torch.zeros(E)
        P._pregate_logits_kernel[(E // 8,)](res, w, gate, out, 1e-6, gate.stride(0), H=H, BLOCK_H=H, BLOCK_E=8)
        x = res.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        ref = (y @ gate.float().t())[0]
        # speculative only: bf16 rounding of the normed input is allowed to move the logits a little
        self.assertLessEqual(float((out - ref).abs().max()), 0.02 * float(ref.abs().max()))
        self.assertGreater(float((out - (x @ gate.float().t())[0]).abs().max()), 0.5, 'the norm is applied')


if __name__ == '__main__':
    unittest.main()
