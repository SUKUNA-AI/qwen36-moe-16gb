"""CPU regression for host output accounting with one forward of overlap.

Drive the real scheduler loops and DecodeManager; only GPU execution, streams and
cache storage are replaced. complete_one() runs at launch, append_host() at drain.
"""
from __future__ import annotations

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.message import AbortBackendMsg, DetokenizeMsg, ErrorReplyMsg
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler


class Harness:
    def __init__(self, budget, *, prompt_len=5, cached_len=0, tokens=None,
                 eos=(), ignore_eos=False, stop_strs=()):
        self.prompt_len = prompt_len
        self.req = Req(
            input_ids=torch.arange(prompt_len, dtype=torch.int32), table_idx=0,
            cached_len=cached_len, output_len=budget, uid=19, cache_handle=None,
            sampling_params=SamplingParams(max_tokens=budget, ignore_eos=ignore_eos,
                                           stop_strs=list(stop_strs)),
        )
        self.tokens = tokens if tokens is not None else list(range(100, 100 + budget))
        self.sent = []
        self.freed = []
        self.launched = 0
        self.last_data = None
        self.pending_messages = []
        self.dm = DecodeManager(page_size=1)
        self.pm = SimpleNamespace(runnable=True, pending_list=[], abort_req=lambda _: None)
        stream = SimpleNamespace(wait_stream=lambda _: None)
        self.stub = SimpleNamespace(
            engine=SimpleNamespace(stream=stream), stream=stream,
            engine_stream_ctx=nullcontext(), _pending_rebuild=None,
            cache_manager=SimpleNamespace(lazy_free_region=nullcontext,
                                          cache_req=lambda *_, **__: None),
            decode_manager=self.dm, prefill_manager=self.pm, finished_reqs=set(),
            eos_token_ids=set(eos), toolcall_anchor_id=None,
            config=SimpleNamespace(page_size=1), _pending_abort_acks=set(),
            _last_data=None, _kv_usage_pages=lambda: (0, 256),
            _mamba_slot_usage=lambda: None, _swa_token_usage=lambda: None,
            _gpu_mem_bytes=lambda: 0, send_result=self.sent.extend,
            status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
            tokenizer=SimpleNamespace(decode=lambda ids: ''.join(
                {201: 'EN', 202: 'D'}.get(t, 'x') for t in ids)),
            _restore_linear_states=lambda _: None,
        )
        self.stub.receive_msg = self.receive
        self.stub._schedule_next_batch = self.schedule
        self.stub._forward = self.forward
        self.stub._free_req_resources = self.free
        self.stub._process_last_data = lambda data: Scheduler._process_last_data(self.stub, data)
        self.stub._process_one_msg = lambda msg: Scheduler._process_one_msg(self.stub, msg)
        self.stub._flush_abort_acks = lambda: Scheduler._flush_abort_acks(self.stub)
        self.stub._match_stop_str = lambda req: Scheduler._match_stop_str(self.stub, req)

    def receive(self, *, blocking):
        messages, self.pending_messages = self.pending_messages, []
        return messages

    def schedule(self):
        if self.pm.runnable:
            self.pm.runnable = False
            batch = Batch(reqs=[self.req], phase='prefill')
        else:
            batch = self.dm.schedule_next_batch()
        return SimpleNamespace(batch=batch) if batch is not None else None

    def forward(self, forward_input):
        assert self.req.can_decode, 'scheduler launched beyond the device token budget'
        token = self.tokens[self.launched]
        self.launched += 1
        self.req.complete_one()
        self.dm.filter_reqs(forward_input.batch.reqs)
        return (None, torch.tensor([token], dtype=torch.int32),
                SimpleNamespace(synchronize=lambda: None))

    def free(self, req):
        assert req not in self.freed, 'request resources freed twice'
        self.freed.append(req)
        req.table_idx = -1

    def step(self, overlap=True):
        if overlap:
            self.last_data = Scheduler.overlap_loop(self.stub, self.last_data)
        else:
            Scheduler.normal_loop(self.stub)

    def run(self, overlap=True):
        with patch.dict(os.environ, {'QWEN36_SPEC': ''}):
            for _ in range(self.req.output_len + 3):
                if not (self.pm.runnable or self.dm.runnable or self.last_data is not None):
                    break
                self.step(overlap)
            else:
                raise AssertionError('scheduler did not drain within the bounded budget')
        assert self.last_data is None
        assert not self.dm.runnable
        assert len(self.freed) == 1
        assert self.req.device_len <= self.req.max_device_len
        return [msg for msg in self.sent if isinstance(msg, DetokenizeMsg)]


class TestOverlapOutputBudget(unittest.TestCase):
    def assert_tokens(self, harness, messages, expected, reason):
        self.assertEqual([m.next_token for m in messages], expected)
        self.assertEqual(harness.req.input_ids[harness.prompt_len:].tolist(), expected)
        self.assertEqual([m.finished for m in messages], [False] * (len(expected) - 1) + [True])
        self.assertEqual(messages[-1].finish_reason, reason)
        self.assertTrue(all(m.finish_reason is None for m in messages[:-1]))

    def test_exact_budget_overlap_and_normal(self):
        for overlap in (False, True):
            for budget in (1, 2, 128):
                for prompt_len, cached_len in ((1, 0), (17, 0), (17, 16)):
                    with self.subTest(overlap=overlap, budget=budget,
                                      prompt_len=prompt_len, cached_len=cached_len):
                        harness = Harness(budget, prompt_len=prompt_len, cached_len=cached_len)
                        messages = harness.run(overlap)
                        self.assert_tokens(harness, messages, harness.tokens, 'length')
                        self.assertEqual(harness.launched, budget)
                        self.assertEqual(harness.req.input_ids.numel(), harness.req.max_device_len)

    def test_inflight_final_token_does_not_finish_previous_token(self):
        harness = Harness(2)
        harness.step()
        harness.step()
        self.assertEqual(harness.req.device_len, harness.req.max_device_len)
        self.assertEqual(harness.req.input_ids.numel(), harness.req.max_device_len - 1)
        self.assertFalse(harness.req.can_decode)
        self.assertEqual(len(harness.sent), 1)
        self.assertFalse(harness.sent[0].finished)
        self.assertEqual(harness.freed, [])
        self.assert_tokens(harness, harness.run(), harness.tokens, 'length')

    def test_eos_at_budget_wins_over_length(self):
        for overlap in (False, True):
            for budget in (1, 2, 128):
                with self.subTest(overlap=overlap, budget=budget):
                    harness = Harness(budget, eos={100 + budget - 1})
                    self.assert_tokens(harness, harness.run(overlap), harness.tokens, 'stop')

    def test_stop_spanning_tokens_at_budget_wins_over_length(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                harness = Harness(2, tokens=[201, 202], stop_strs=['END'])
                messages = harness.run(overlap)
                self.assert_tokens(harness, messages, [201, 202], 'stop')
                self.assertEqual(messages[-1].matched_stop, 'END')

    def test_eos_or_stop_discards_already_launched_overlap_result(self):
        for use_eos in (False, True):
            with self.subTest(use_eos=use_eos):
                harness = Harness(128, tokens=[201, 202] + [100] * 126,
                                  eos={202} if use_eos else (),
                                  stop_strs=[] if use_eos else ['END'])
                messages = harness.run()
                self.assert_tokens(harness, messages, [201, 202], 'stop')
                self.assertEqual(harness.launched, 3)
                self.assertEqual(len(harness.freed), 1)

    def test_ignored_eos_still_obeys_full_budget(self):
        for budget in (1, 2, 128):
            with self.subTest(budget=budget):
                harness = Harness(budget, eos={100}, ignore_eos=True)
                self.assert_tokens(harness, harness.run(), harness.tokens, 'length')

    def test_abort_inflight_emits_no_token_and_frees_once(self):
        harness = Harness(128)
        harness.step()
        harness.pending_messages = [AbortBackendMsg(uid=harness.req.uid)]
        harness.run()
        self.assertTrue(harness.req.aborted)
        self.assertEqual(harness.req.input_ids.numel(), harness.prompt_len)
        self.assertEqual(harness.launched, 1)
        self.assertEqual(len(harness.sent), 1)
        self.assertIsInstance(harness.sent[0], ErrorReplyMsg)
        self.assertEqual(harness.sent[0].error, 'request aborted')


if __name__ == '__main__':
    unittest.main()
