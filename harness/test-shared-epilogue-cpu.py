"""D5 fused decode epilogue (moe/shared_epilogue.py): flag, model control flow, fallback decision.

The kernel runs under TRITON_INTERPRET=1 for indexing only; its bit-identity with torch is the
GPU self-test (all 65536 bf16 sigmoid inputs) and freetoken/tests/moe/test_shared_epilogue_gpu.py.
"""
import os

os.environ['TRITON_INTERPRET'] = '1'

import unittest
from types import SimpleNamespace as S
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from cpu_control_flow import methods
from repo_paths import FREETOKEN

from freetoken.moe import shared_epilogue as SE

MODEL = FREETOKEN / 'python/freetoken/models/qwen3_5_moe/moe.py'


class Flag(unittest.TestCase):
    def test_flag(self):
        for raw, want in ((None, False), ('', False), ('0', False), ('1', True)):
            with patch.dict(os.environ, {} if raw is None else {SE.ENV: raw}, clear=True):
                self.assertEqual(SE.enabled(), want)
        for bad in ('2', 'on', ' 1'):
            with patch.dict(os.environ, {SE.ENV: bad}, clear=True), self.assertRaises(ValueError):
                SE.enabled()


def block(trace, *, epilogue, prefill=False, overlap=False):
    ns = methods(MODEL, 'Qwen3_5MoE', {'forward', '_shared_forward', '_shared_parts'},
                 {'torch': S(sigmoid=lambda g: (trace.append('sigmoid'), g)[1]),
                  'get_global_ctx': lambda: S(batch=S(is_prefill=prefill))})

    class Hidden:
        shape = (1, 4)

        def view(self, *a):
            return self

    class Out:
        def __add__(self, other):
            trace.append('add')
            return self

        def __mul__(self, other):
            trace.append('mul')
            return self

        def view(self, *a):
            return self

    out = Out()

    def experts_forward(hidden_states, router_logits):
        trace.append('routed')
        return out

    def with_shared(hidden, logits, shared_forward):
        trace.append('ensure')
        shared = shared_forward(hidden)
        trace.append('routed')
        return out, shared

    m = S(gate=S(forward=lambda h: (trace.append('router'), 'logits')[1]),
          shared_expert=S(forward=lambda h: (trace.append('shared'), out)[1]),
          shared_expert_gate=S(forward=lambda h: (trace.append('gate'), out)[1]),
          experts=S(forward=experts_forward, can_overlap_shared=lambda h: overlap, forward_with_shared=with_shared))
    if epilogue:
        m._fused_epilogue = lambda routed, shared, gate: (trace.append('epilogue'), routed)[1]
    for name in ('_shared_forward', '_shared_parts'):
        setattr(m, name, ns[name].__get__(m))
    return ns['forward'], m, Hidden()


class ModelFlow(unittest.TestCase):
    def run_block(self, **kw):
        trace = []
        forward, m, h = block(trace, **kw)
        forward(m, h)
        return trace

    def test_unfused_order_is_unchanged(self):
        self.assertEqual(self.run_block(epilogue=False), ['router', 'shared', 'gate', 'sigmoid', 'mul', 'routed', 'add'])
        self.assertEqual(self.run_block(epilogue=True, prefill=True),
                         ['router', 'shared', 'gate', 'sigmoid', 'mul', 'routed', 'add'])

    def test_fused_computes_both_projections_before_routed(self):
        self.assertEqual(self.run_block(epilogue=True), ['router', 'shared', 'gate', 'routed', 'epilogue'])
        self.assertEqual(self.run_block(epilogue=True, overlap=True),
                         ['router', 'ensure', 'shared', 'gate', 'routed', 'epilogue'])

    def test_missing_attribute_means_off(self):
        trace = []
        forward, m, h = block(trace, epilogue=False)
        self.assertFalse(hasattr(m, '_fused_epilogue'))
        forward(m, h)
        self.assertIn('add', trace)


class Decision(unittest.TestCase):
    def test_fallbacks(self):
        r = torch.zeros(1, 8, dtype=torch.bfloat16)
        e = SE.SharedEpilogue()
        self.assertFalse(e.usable(r), 'cpu tensors keep torch')
        self.assertIsNone(e.decided)
        cuda_like = S(device=S(type='cuda'))
        with patch.object(SE.torch.cuda, 'is_current_stream_capturing', lambda: True):
            self.assertFalse(e.usable(cuda_like))
            self.assertIsNone(e.decided, 'a capturing call must not decide')
        with patch.object(SE.torch.cuda, 'is_current_stream_capturing', lambda: False), \
                patch.object(SE, 'self_test', lambda device: 'sigmoid differs'):
            self.assertFalse(e.usable(cuda_like))
            self.assertFalse(e.decided)
            self.assertEqual(e.reason, 'sigmoid differs')
        e2 = SE.SharedEpilogue()
        with patch.object(SE.torch.cuda, 'is_current_stream_capturing', lambda: False), \
                patch.object(SE, 'self_test', lambda device: None):
            self.assertTrue(e2.usable(cuda_like))
        self.assertTrue(e2.usable(cuda_like), 'decided once')

    def test_unsupported_shapes_take_the_reference(self):
        e = SE.SharedEpilogue()
        e.decided = True
        routed = torch.randn(2, 8).to(torch.bfloat16)
        shared = torch.randn(2, 8).to(torch.bfloat16)
        gate = torch.randn(2, 1).to(torch.bfloat16)
        strided = torch.randn(8, 2).to(torch.bfloat16).t()
        with patch.object(SE, '_launch', lambda *a: self.fail('kernel must not run')):
            for args in ((routed.float(), shared, gate), (routed, shared, gate.float()), (routed, shared[:1], gate),
                         (strided, shared, gate), (routed, strided, gate), (routed[0], shared[0], gate[0])):
                torch.testing.assert_close(e(*args), SE.reference(*args))


def _trunc(x):
    return (x.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


@triton.jit
def _interp_exp(x):
    return tl.exp(x)


class Kernel(unittest.TestCase):
    def test_interpreted_kernel_indexing(self):
        torch.manual_seed(0)
        # the interpreter has no libdevice; the GPU self-test owns the transcendental
        with patch.object(SE, 'libdevice', S(exp=_interp_exp)):
            self._run()

    def _run(self):
        for M, H in ((1, 64), (3, 100), (2, 1500)):
            routed = torch.randn(M, H).to(torch.bfloat16)
            shared = torch.randn(M, H).to(torch.bfloat16)
            gate = (torch.randn(M, 1) * 4).to(torch.bfloat16)
            out = torch.empty_like(routed)
            SE._launch(routed, shared, gate, out)
            sig = torch.empty_like(gate)
            SE._sigmoid_table_kernel[(1,)](gate, sig, M, BLOCK=4)
            # the same chain in torch fp32 ops around the interpreter's own sigmoid; the interpreter
            # truncates fp32 -> bf16 where the GPU rounds to nearest (cvt.rn, the GPU test's subject)
            want = _trunc(routed.float() + _trunc(shared.float() * sig.float()).float())
            self.assertTrue(torch.equal(out.view(torch.int16), want.view(torch.int16)), (M, H))
            torch.testing.assert_close(sig.float(), torch.sigmoid(gate.float()), rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
