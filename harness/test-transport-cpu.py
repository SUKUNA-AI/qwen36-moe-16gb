"""S4 transport switches (C0/C1/C2, T3 priority, D3a trace) with fake streams; no CUDA race claim.

The GPU byte gates are freetoken/tests/moe/test_prefill_c0_gpu.py and test_prefill_c12_gpu.py.
"""
import ast
import contextlib
import importlib.util
import os
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch

import numpy as np
import torch
from cpu_control_flow import methods
from repo_paths import CHECKOUT, FREETOKEN

CACHE = FREETOKEN / 'python/freetoken/moe/offload_cache.py'
OVERLAP = FREETOKEN / 'python/freetoken/decode_overlap.py'


def module_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Trace:
    def __init__(self):
        self.log = []

    def event(self, name):
        return S(record=lambda stream: self.log.append(f'{name}@{stream.name}'))

    def stream(self, name):
        log = self.log
        return S(name=name, wait_event=lambda e: log.append(f'{name}.wait'), synchronize=lambda: None)


def fake_torch(trace, compute):
    cur = [compute]

    @contextlib.contextmanager
    def stream(s):
        old, cur[0] = cur[0], s
        try:
            yield
        finally:
            cur[0] = old

    return S(cuda=S(current_stream=lambda device=None: cur[0], stream=stream))


def cache_class(names, trace):
    ns = methods(CACHE, 'OffloadMoeCache', set(names), {'torch': fake_torch(trace, trace.stream('compute'))})
    cls = type('OffloadMoeCache', (), {k: v for k, v in ns.items() if callable(v) and k in names})
    ns['OffloadMoeCache'] = cls
    return cls


class Flags(unittest.TestCase):
    def test_strict_and_int_flags(self):
        ns = module_functions(CACHE, {'strict_flag', 'int_flag'}, {'os': os})
        for raw, want in ((None, False), ('', False), ('0', False), ('1', True)):
            env = {} if raw is None else {'X': raw}
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(ns['strict_flag']('X'), want)
        for bad in ('2', 'on', 'true', ' 1'):
            with patch.dict(os.environ, {'X': bad}, clear=True), self.assertRaises(ValueError):
                ns['strict_flag']('X')
        for raw, want in ((None, 0), ('', 0), ('0', 0), ('512', 512), (' 64 ', 64)):
            env = {} if raw is None else {'X': raw}
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(ns['int_flag']('X'), want)
        for bad in ('-1', 'lots', '1.5'):
            with patch.dict(os.environ, {'X': bad}, clear=True), self.assertRaises(ValueError):
                ns['int_flag']('X')

    def test_defaults_are_off_in_source(self):
        text = CACHE.read_text(encoding='utf-8')
        for env in ('QWEN36_PREFILL_COPY_PRIORITY', 'QWEN36_PREFILL_EPOCH_INVALIDATE', 'QWEN36_PREFILL_SELECTED',
                    'QWEN36_PREFILL_SELECTED_SYNC', 'QWEN36_PREFILL_TMA_GATHER'):
            self.assertIn(f'"{env}"', text)


class T3Priority(unittest.TestCase):
    def build(self, env):
        from freetoken.decode_overlap import DecodeSharedOverlap
        made = []

        class Cuda:
            def Stream(self, device=None, **kw):
                made.append(kw)
                return S(wait_event=lambda e: None)

            def Event(self):
                return S(record=lambda s: None)

            def current_stream(self, device):
                return S(wait_event=lambda e: None)

        with patch.dict(os.environ, env, clear=True):
            DecodeSharedOverlap(Cuda(), 'cuda')
        return made

    def test_priority_only_when_asked(self):
        self.assertEqual(self.build({}), [{}])
        self.assertEqual(self.build({'QWEN36_DECODE_OVERLAP_PRIORITY': '0'}), [{}])
        self.assertEqual(self.build({'QWEN36_DECODE_OVERLAP_PRIORITY': '1'}), [{'priority': -1}])
        with self.assertRaises(ValueError):
            self.build({'QWEN36_DECODE_OVERLAP_PRIORITY': 'yes'})


class EpochInvalidation(unittest.TestCase):
    NAMES = ('begin_prefill', 'prefetch_prefill_layer', 'release_prefill_layer', '_invalidate_prefill_buffer')

    def make(self, epoch):
        trace = Trace()
        cls = cache_class(self.NAMES, trace)
        c = cls()
        c._invalidate_prefill_buffer = lambda b: trace.log.append(f'invalidate{b}')
        c.prefill_overlap = True
        c.prefill_epoch_invalidate = epoch
        c._prefill_epoch_open = False
        c.prefill_copy_stream = trace.stream('copy')
        c.prefill_begin_event = trace.event('begin')
        c.prefill_ready_events = [trace.event('ready0'), trace.event('ready1')]
        c.prefill_release_events = [trace.event('release0'), trace.event('release1')]
        c._prefill_buffer_has_release_event = [False, False]
        c.prefill_hit_d2d = False
        c.layer_major_active = False
        c._hit_d2d_usable = lambda: False
        c.prefill_tma_gather = False
        c.num_layers = 3
        c.device = 'cuda'
        host = [torch.ones(2, 4) * i for i in range(3)]
        c.banks = [(host, None)]
        c.prefill_bank_buffers = [torch.zeros(2, 2, 4)]
        c.prefill_h2d_bytes = 0
        return c, trace

    def run_epoch(self, c):
        c.begin_prefill()
        for layer in range(c.num_layers):
            c.prefetch_prefill_layer(layer)
            c.prefetch_prefill_layer(layer + 1)
            c.release_prefill_layer(layer)

    def test_default_invalidates_on_the_copy_stream_per_layer(self):
        c, trace = self.make(False)
        self.run_epoch(c)
        self.assertEqual(trace.log.count('invalidate0') + trace.log.count('invalidate1'), 3)
        self.assertNotIn('invalidate0', trace.log[:trace.log.index('begin@compute')])

    def test_epoch_invalidates_both_buffers_before_the_fence(self):
        c, trace = self.make(True)
        self.run_epoch(c)
        fence = trace.log.index('begin@compute')
        self.assertEqual(trace.log[:fence], ['invalidate0', 'invalidate1'])
        self.assertEqual(trace.log.count('invalidate0') + trace.log.count('invalidate1'), 2)
        self.assertFalse(c._prefill_epoch_open, 'the last layer release must close the epoch')
        # a prefetch outside an epoch falls back to per-layer invalidation
        c._prefill_buffer_layer = [None, None]
        c._prefill_buffer_released = [True, True]
        c.prefetch_prefill_layer(0)
        self.assertEqual(trace.log[-3:], ['copy.wait', 'invalidate0', 'ready0@copy'])

    def test_epoch_needs_a_copy_stream(self):
        c, trace = self.make(True)
        c.prefill_copy_stream = None
        c.begin_prefill()
        self.assertNotIn('invalidate0', trace.log)
        self.assertFalse(c._prefill_epoch_open)


class SelectedRoute(unittest.TestCase):
    def make(self):
        trace = Trace()
        cls = cache_class(('select_prefill_route',), trace)
        c = cls()
        c.prefill_selected_max = 512
        c.prefill_selected_sync = False
        c.layer_major_active = False
        c.refill = None
        c.prefill_selected_chunks = 0
        c._xfer_usable = lambda sync: True
        return c, cls

    def test_guards(self):
        c, cls = self.make()
        for tokens, want in ((0, False), (1, True), (511, True), (512, False), (2049, False)):
            c.select_prefill_route(tokens)
            self.assertEqual(c.prefill_selected_active, want, tokens)
        self.assertEqual(c.prefill_selected_chunks, 2)
        c.refill = object()
        c.select_prefill_route(8)
        self.assertFalse(c.prefill_selected_active)
        c.refill, c.layer_major_active = None, True
        c.select_prefill_route(8)
        self.assertFalse(c.prefill_selected_active)
        c.layer_major_active = False
        c.prefill_selected_max = 0
        c.select_prefill_route(8)
        self.assertFalse(c.prefill_selected_active)
        c.prefill_selected_max = 512
        c._xfer_usable = lambda sync: False
        c.select_prefill_route(8)
        self.assertFalse(c.prefill_selected_active)
        sub = type('Complement', (cls,), {})()
        sub.__dict__.update({k: v for k, v in c.__dict__.items()})
        sub._xfer_usable = lambda sync: True
        sub.select_prefill_route(8)
        self.assertFalse(sub.prefill_selected_active, 'subclasses keep their own prefetch')

    def test_router_decides_at_layer_zero_before_begin(self):
        text = (FREETOKEN / 'python/freetoken/layers/moe.py').read_text(encoding='utf-8')
        body = text[text.index('def _prefill_routed'):text.index('def _wait_prefill_overlap')]
        self.assertLess(body.index('select_prefill_route'), body.index('begin_prefill'))
        self.assertIn('not getattr(cache, "layer_major_active", False)', body)


class Counters(unittest.TestCase):
    def test_device_rows_fold_into_bytes(self):
        ns = methods(CACHE, 'OffloadMoeCache', {'_fold_xfer_rows'}, {'torch': torch})
        c = S(_xfer_rows_dirty=True, _xfer_rows=torch.tensor([3, 2]), _copy_feat_bytes_host=[100, 10],
              _prefill_h2d_bytes=5, _prefill_d2d_bytes=7)
        ns['_fold_xfer_rows'](c)
        self.assertEqual((c._prefill_h2d_bytes, c._prefill_d2d_bytes), (5 + 330, 7 + 220))
        self.assertEqual(c._xfer_rows.tolist(), [0, 0])
        self.assertFalse(c._xfer_rows_dirty)
        ns['_fold_xfer_rows'](c)
        self.assertEqual(c._prefill_h2d_bytes, 335)


class RouteTrace(unittest.TestCase):
    def test_flag(self):
        from freetoken.moe import route_trace
        for raw, want in ((None, False), ('0', False), ('1', True)):
            with patch.dict(os.environ, {} if raw is None else {route_trace.ENV: raw}, clear=True):
                self.assertEqual(route_trace.enabled(), want)
        with patch.dict(os.environ, {route_trace.ENV: 'on'}, clear=True), self.assertRaises(ValueError):
            route_trace.enabled()

    def test_replay_lru_matches_flashlib_rules(self):
        spec = importlib.util.spec_from_file_location('d3a_replay', CHECKOUT / 'next-update/transport/scripts/d3a_replay.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 2 layers x 4 experts, 3 slots; slot 0 holds id 1 (usage 5), slot 2 holds id 6 (usage 1)
        lru = mod.Lru(np.array([[-1, 0, -1, -1], [-1, -1, 2, -1]]), np.array([1, -1, 6]),
                      np.array([5, 0, 1]), 9)
        # hit 1 (bumped, not evictable); misses 3 then 0 take victims by (usage, slot): slot 1, then slot 2
        self.assertEqual(lru.ensure(np.array([3, 1, 0, 3]), 0), {0, 3})
        self.assertEqual(lru.slot_of_id[[0, 1, 3, 6]].tolist(), [1, 0, 2, -1])
        self.assertEqual(lru.usage.tolist(), [10, 10, 10])
        self.assertEqual(lru.ensure(np.array([2]), 4), {2})
        self.assertEqual(int(lru.slot_of_id[6]), 0)


if __name__ == '__main__':
    unittest.main()
