"""CPU-only telemetry regression tests; stdlib mocks, no torch/CUDA/service import.

Run: python test-engine-metrics.py --report engine-metrics-tests.json
The optional bench probes reproduce known unsafe joins; they do not contact HTTP.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
SOURCE = WORKSPACE / 'work/FreeToken-opt/python/freetoken/engine/telemetry.py'
BENCH = pathlib.Path(__file__).with_name('bench.py')


def flatten(value):
    if isinstance(value, list):
        return [item for group in value for item in flatten(group)]
    return [value]


class Tensor:
    """Deliberately NOT JSON serializable; clones own independent Python values."""
    def __init__(self, value, size=8, pinned=False):
        self.value, self.size, self.pinned = copy.deepcopy(value), size, pinned

    def sum(self, dim=None):
        if dim == 0 and isinstance(self.value, list) and self.value and isinstance(self.value[0], list):
            return Tensor([sum(col) for col in zip(*self.value)])
        return Tensor(sum(flatten(self.value)))

    def clone(self): return Tensor(self.value, self.size, self.pinned)
    def cpu(self): return self
    def tolist(self): return copy.deepcopy(self.value)
    def item(self):
        values = flatten(self.value)
        assert len(values) == 1
        return values[0]
    def numel(self): return len(flatten(self.value))
    def element_size(self): return self.size
    def is_pinned(self): return self.pinned
    def __getitem__(self, idx): return Tensor(self.value[idx], self.size, self.pinned)
    def __ge__(self, other): return Tensor([int(v >= other) for v in self.value])


class Storage:
    def __init__(self, count, size=1, pinned=False, rows=1):
        self.count, self.size, self.pinned, self.rows = count, size, pinned, rows
    def numel(self): return self.count
    def element_size(self): return self.size
    def is_pinned(self): return self.pinned
    def __getitem__(self, idx):
        assert 0 <= idx < self.rows
        return Storage(self.count // self.rows, self.size, self.pinned)


class Writer:
    def __init__(self): self.records = []
    def write(self, path, row):
        # Reject tensors and nonfinite values rather than silently dropping them.
        self.records.append((path, json.loads(json.dumps(row, allow_nan=False))))
    def close(self): pass


class Event:
    def __init__(self, milliseconds): self.milliseconds = milliseconds
    def elapsed_time(self, end): return end.milliseconds - self.milliseconds


class ComplementCache: pass


class Cache:
    collect_stats = True
    decode_target = 'gpu'
    cache_size = 4
    def __init__(self):
        self.lru_stats = Tensor([[40, 8, 5], [60, 12, 5]])
        self.prefill_h2d_bytes, self.prefill_d2d_bytes = 1000, 2000
        self.bank_caches = {'weight': Storage(64, rows=4), 'scale': Storage(16, rows=4)}
        self.bank_sources = {'weight': [Storage(64, pinned=True), Storage(64)],
                             'scale': [Storage(16, pinned=True), Storage(16)]}
        self.id_of_slot = Tensor([0, 1, 2, -1])
    def decode_miss_stats(self):
        active, missing, calls = self.lru_stats.sum(0).tolist()
        return dict(layer_calls=calls, active_per_layer=active/calls if calls else 0.,
                    missing_per_layer=missing/calls if calls else 0.,
                    fetched_per_layer=0., cpu_per_layer=missing/calls if calls else 0.,
                    miss_rate=missing/active if active else 0., fetch_rate=0.)


class HybridCache(Cache):
    decode_target = 'hybrid'
    def __init__(self):
        super().__init__()
        self.lru_stats = Tensor([[0, 0, 0], [0, 0, 0]])
        self.stat_active, self.stat_missing, self.stat_fetched = Tensor(24), Tensor(8), Tensor(3)
        self.stat_calls = Tensor(3)
    def decode_miss_stats(self):
        return dict(layer_calls=3, active_per_layer=8., missing_per_layer=8/3,
                    fetched_per_layer=1., cpu_per_layer=5/3, miss_rate=1/3, fetch_rate=3/8)


class Complement(ComplementCache, Cache): pass


def engine(cache=None):
    return types.SimpleNamespace(moe_offload_cache=cache, model=types.SimpleNamespace(),
                                 native_mtp=None, kv_cache=object(), linear_state_pool=None)


def req(uid=0, aborted=False): return types.SimpleNamespace(uid=uid, aborted=aborted)


def batch(request, prefill=True, tokens=2048, milliseconds=10.):
    requests = request if isinstance(request, list) else [request]
    return types.SimpleNamespace(reqs=requests, is_prefill=prefill, log_new_tokens=tokens,
                                 qwen36_timing=(Event(0.), Event(milliseconds)))


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.
        self.teacher_calls = []
        self.writer = Writer()
        self.peak_resets=[]
        fake_torch = types.ModuleType('torch')
        fake_torch.cuda = types.SimpleNamespace(memory_allocated=lambda: 4096, memory_reserved=lambda: 8192,
            reset_peak_memory_stats=lambda:self.peak_resets.append(True),max_memory_allocated=lambda:6144,max_memory_reserved=lambda:12288)
        fake_torch.stack = lambda values: Tensor([value.tolist() for value in values])
        qualification = types.ModuleType('freetoken.engine.qualification')
        qualification.complete = self.teacher_calls.append
        complement = types.ModuleType('freetoken.moe.complement')
        complement.ComplementCache = ComplementCache
        packages = {name: types.ModuleType(name) for name in ('freetoken', 'freetoken.engine', 'freetoken.moe')}
        for package in packages.values(): package.__path__ = []
        self.modules = patch.dict(sys.modules, {**packages, 'torch': fake_torch,
                                'freetoken.engine.qualification': qualification,
                                'freetoken.moe.complement': complement})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location('freetoken.engine.telemetry', SOURCE)
        self.telemetry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.telemetry)
        self.telemetry.ENABLED = True
        self.telemetry.writer = self.writer
        self.telemetry.time = types.SimpleNamespace(monotonic=lambda: self.clock)

    def finish(self, request, model):
        self.telemetry.complete(request, model)
        return self.writer.records[-1][1]

    def test_span_includes_first_prefill_and_chunks_accumulate(self):
        r, e = req(), engine()
        first = batch(r, tokens=2048, milliseconds=20.)
        self.telemetry.batch_started(first, e)
        self.clock = 103.
        self.telemetry.batch_completed(first, e)
        second = batch(r, tokens=1024, milliseconds=12.)
        self.telemetry.batch_started(second, e)
        self.telemetry.batch_completed(second, e)
        for _ in range(3):
            decode = batch(r, prefill=False, tokens=1, milliseconds=2.)
            self.telemetry.batch_started(decode, e)
            self.telemetry.batch_completed(decode, e)
        self.clock = 104.5
        row = self.finish(r, e)
        self.assertEqual(row['engine_span_s'], 4.5)
        self.assertEqual(row['prefill_gpu_ms'], 32.)
        self.assertEqual(row['prefill_forward_tokens'], 3072)
        self.assertEqual(row['decode_gpu_ms'], 6.)
        self.assertEqual(row['decode_forwards'], 3)
        self.assertEqual(len(self.peak_resets),1)
        self.assertEqual(row['request_torch_peak_allocated_bytes'],6144)
        self.assertEqual(row['request_torch_peak_reserved_bytes'],12288)

    def test_counter_baseline_is_snapshot_before_first_forward(self):
        cache, r = Cache(), req()
        e = engine(cache)
        self.telemetry.batch_started(batch(r), e)
        cache.lru_stats.value[:] = [[52, 11, 6], [72, 14, 7]]
        cache.prefill_h2d_bytes += 111
        cache.prefill_d2d_bytes += 222
        row = self.finish(r, e)
        self.assertEqual(row['expert_counter_deltas'], dict(prefill_h2d_bytes=111,
                         prefill_d2d_bytes=222, decode_active=24, decode_missing=5,
                         decode_h2d_payload_bytes=100, decode_cache_hits=19))
        self.assertEqual(row['expert_access_counters_total'], [124, 25, 13])

    def test_repeated_begin_does_not_reset_baseline(self):
        c, r = Cache(), req(); e = engine(c)
        self.telemetry.batch_started(batch(r), e)
        c.prefill_h2d_bytes += 200
        self.clock += 3.
        self.telemetry.batch_started(batch(r), e)
        row = self.finish(r, e)
        self.assertEqual(row['engine_span_s'], 3.)
        self.assertEqual(row['expert_counter_deltas']['prefill_h2d_bytes'], 200)

    def test_gpu_misses_are_fetched_not_cpu_compute(self):
        c, r = Cache(), req(); e = engine(c)
        self.telemetry.batch_started(batch(r), e)
        stats = self.finish(r, e)['expert_decode_stats']
        self.assertEqual(stats['fetched_per_layer'], 2.)
        self.assertEqual(stats['cpu_per_layer'], 0.)
        self.assertEqual(stats['fetch_rate'], 1.)

    def test_gpu_zero_misses_no_fetch(self):
        c, r = Cache(), req(); c.lru_stats = Tensor([[8, 0, 1]])
        e = engine(c); self.telemetry.batch_started(batch(r), e)
        stats = self.finish(r, e)['expert_decode_stats']
        self.assertEqual(stats['cpu_per_layer'], 0.)
        self.assertEqual(stats['fetched_per_layer'], 0.)
        self.assertEqual(stats['fetch_rate'], 0.)

    def test_hybrid_summary_is_not_overwritten_by_gpu_fix(self):
        c, r = HybridCache(), req(); e = engine(c)
        expected = c.decode_miss_stats()
        self.telemetry.batch_started(batch(r), e)
        self.assertEqual(self.finish(r, e)['expert_decode_stats'], expected)

    def test_hybrid_delta_uses_separate_counters_and_only_fetched_payload(self):
        c, r = HybridCache(), req(); e = engine(c)
        c.lru_stats = Tensor([[900, 800, 700]])  # Deliberately plausible but unrelated.
        self.telemetry.batch_started(batch(r), e)
        c.stat_active.value, c.stat_missing.value, c.stat_fetched.value = 48, 16, 6
        c.lru_stats.value = [[1900, 1600, 1700]]
        row = self.finish(r, e)
        delta = row['expert_counter_deltas']
        self.assertEqual(row['expert_access_counters_total'], [48, 16, 3])
        self.assertEqual(delta['decode_active'], 24)
        self.assertEqual(delta['decode_missing'], 8)
        self.assertEqual(delta['decode_fetched'], 3)
        self.assertEqual(delta['decode_cache_hits'], 16)
        self.assertEqual(delta['decode_h2d_payload_bytes'], 60)

    def test_cpu_decode_is_explicitly_unavailable(self):
        c, r = Cache(), req(); c.decode_target = 'cpu'; e = engine(c)
        self.telemetry.batch_started(batch(r), e)
        row = self.finish(r, e)
        self.assertNotIn('expert_counter_deltas', row)
        self.assertNotIn('expert_decode_stats', row)
        self.assertEqual(row['expert_transfer_accounting'], 'unavailable_for_cpu_decode')

    def test_engine_row_uses_inherited_run_id(self):
        r, e = req(), engine()
        with patch.dict('os.environ', {'QWEN36_RUN_ID':'synthetic-run-id'}):
            self.telemetry.batch_started(batch(r), e)
        self.assertEqual(self.finish(r, e)['run_id'], 'synthetic-run-id')

    def test_memory_counts_slots_and_pinning(self):
        c, r = Cache(), req(); e = engine(c)
        self.telemetry.batch_started(batch(r), e)
        row = self.finish(r, e)
        self.assertEqual(row['gpu_expert_bytes'], 80)
        self.assertEqual(row['host_expert_bytes'], 160)
        self.assertEqual(row['host_pinned_expert_bytes'], 80)
        self.assertEqual(row['expert_bytes_per_slot'], 20)
        self.assertEqual(row['expert_slots'], 4)
        self.assertEqual(row['valid_gpu_expert_slots'], 3)

    def test_complement_does_not_claim_zero_pcie_bytes(self):
        c, r = Complement(), req(); e = engine(c)
        self.telemetry.batch_started(batch(r), e)
        row = self.finish(r, e)
        self.assertIsNone(row['prefill_h2d_bytes_total'])
        self.assertIsNone(row['prefill_d2d_bytes_total'])
        self.assertNotIn('expert_counter_deltas', row)
        self.assertEqual(row['expert_transfer_accounting'], 'unavailable_for_experimental_complement')

    def test_optional_tensor_metrics_convert_to_json_scalars(self):
        e, r = engine(Cache()), req()
        saved = []
        e.kv_cache = types.SimpleNamespace(_kv_buffer=Storage(100, 1), dtype='torch.float8_e4m3fn',
                          overflow=Tensor(7), save_calibration=lambda: saved.append(True))
        e.native_mtp = types.SimpleNamespace(weight_bytes=500, k=Storage(8, 2))
        e.linear_state_pool = types.SimpleNamespace(conv_states=Storage(3, 2), recurrent_states=Storage(4, 4))
        e.model.model = types.SimpleNamespace(layers=types.SimpleNamespace(op_list=[
            types.SimpleNamespace(self_attn=types.SimpleNamespace(_fused_qk_errors=Tensor([1, 2]))),
            types.SimpleNamespace(self_attn=types.SimpleNamespace(_fused_qk_errors=Tensor([3, 4])))]))
        r.spec_cost = types.SimpleNamespace(report=lambda: dict(draft_tokens=3, accepted_tokens=2))
        self.telemetry.batch_started(batch(r), e)
        row = self.finish(r, e)
        self.assertEqual(row['fused_qk_errors_total'], [4, 6])
        self.assertEqual(row['kv_clipped_values_total'], 7)
        self.assertEqual(row['mtp_kv_bytes'], 32)
        self.assertEqual(row['gdn_bytes'], 22)
        self.assertEqual(row['kv_bytes'], 100)
        self.assertEqual(saved, [True])
        self.assertFalse(any(key.startswith('_') for key in row))
        self.assertEqual(row['speculation'], dict(draft_tokens=3, accepted_tokens=2))

    def test_cancel_complete_is_idempotent_for_engine_log(self):
        e, r = engine(), req(aborted=True)
        self.telemetry.batch_started(batch(r), e)
        self.telemetry.complete(r, e)
        self.telemetry.complete(r, e)
        self.telemetry.batch_completed(batch(r), e)
        self.assertEqual(len(self.writer.records), 1)
        self.assertTrue(self.writer.records[0][1]['cancelled'])
        self.assertNotIn(r.uid, self.telemetry.rows)

    def test_cancel_before_forward_has_no_fabricated_metrics(self):
        e, r = engine(), req(aborted=True)
        self.telemetry.complete(r, e)
        self.telemetry.complete(r, e)
        self.assertEqual(self.writer.records, [])
        self.assertIn(r.uid, self.telemetry.finished)

    def test_completed_uid_cannot_restart_while_retained(self):
        e, r = engine(), req()
        self.telemetry.batch_started(batch(r), e)
        self.telemetry.complete(r, e)
        self.telemetry.batch_started(batch(r), e)
        self.assertNotIn(r.uid, self.telemetry.rows)

    def test_active_state_bounded_64_evicts_oldest(self):
        e = engine()
        for uid in range(65): self.telemetry.batch_started(batch(req(uid)), e)
        self.assertEqual(len(self.telemetry.rows), 64)
        self.assertNotIn(0, self.telemetry.rows)
        self.telemetry.complete(req(0), e)
        self.assertEqual(self.writer.records, [])

    def test_finished_state_bounded_1024_and_old_uid_can_be_reused(self):
        e = engine()
        for uid in range(1025): self.telemetry.complete(req(uid), e)
        self.assertEqual(len(self.telemetry.finished), 1024)
        self.assertNotIn(0, self.telemetry.finished)
        self.telemetry.batch_started(batch(req(0)), e)
        self.assertIn(0, self.telemetry.rows)

    def test_absent_completion_event_or_begin_does_not_fabricate_row(self):
        e, r = engine(), req()
        self.telemetry.batch_completed(batch(r), e)
        self.assertEqual(len(self.telemetry.rows), 0)
        item = batch(r); del item.qwen36_timing
        self.telemetry.batch_started(item, e)
        self.telemetry.batch_completed(item, e)
        self.assertEqual(self.telemetry.rows[r.uid]['prefill_gpu_ms'], 0.)

    def test_disabled_metrics_still_invokes_qualification_completion(self):
        self.telemetry.ENABLED = False
        e, r = engine(), req()
        self.telemetry.batch_started(batch(r), e)
        self.telemetry.batch_completed(batch(r), e)
        self.telemetry.complete(r, e)
        self.assertEqual(self.teacher_calls, [0])
        self.assertEqual(self.writer.records, [])

    def test_missing_stats_does_not_fabricate_request_transfer_delta(self):
        c, r = Cache(), req(); c.collect_stats = False
        e = engine(c); self.telemetry.batch_started(batch(r), e)
        row = self.finish(r, e)
        self.assertNotIn('expert_counter_deltas', row)
        self.assertNotIn('expert_access_counters_total', row)

    def test_source_integration_records_begin_before_model_forward(self):
        tree = ast.parse(SOURCE.with_name('engine.py').read_text(encoding='utf-8'))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'forward_batch')
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        started = [node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'batch_started']
        forwards = [node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr in ('forward', 'forward_batch')]
        self.assertEqual(len(started), 1)
        self.assertTrue(forwards)
        self.assertLess(started[0], min(forwards))


def extract_attach_metrics(root, sleeper):
    """Extract only the pure join function: no requests/tokenizer/key imports."""
    tree = ast.parse(BENCH.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'attach_metrics')
    namespace = dict(ROOT=root, json=json, time=types.SimpleNamespace(sleep=sleeper))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BENCH), 'exec'), namespace)
    return namespace['attach_metrics']


class BenchJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen36-join-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        (self.root/'logs').mkdir(); (self.root/'reports').mkdir()
        self.http = self.root/'logs/requests.jsonl'
        self.engine = self.root/'reports/engine.jsonl'
        self.http.write_text(json.dumps(dict(request_id='new-request', engine_uid=0, run_id='new-run'))+'\n')
        self.result = dict(request_id='new-request')
        self.waits = []

    def row(self, run='new-run', marker='new-process'):
        return json.dumps(dict(uid=0, pid=222, run_id=run, marker=marker))+'\n'

    def test_never_joins_old_uid_before_delayed_new_row(self):
        self.engine.write_text(self.row('old-run', 'old-process'))
        def arrive(seconds):
            self.waits.append(seconds)
            with self.engine.open('a') as out: out.write(self.row())
        extract_attach_metrics(self.root, arrive)(self.result)
        self.assertEqual(self.result['engine']['marker'], 'new-process')
        self.assertEqual(self.result['engine']['run_id'], 'new-run')
        self.assertEqual(len(self.waits), 1)
        self.assertNotIn('pid', self.result['engine'])

    def test_waits_for_delayed_engine_writer(self):
        def arrive(seconds):
            self.waits.append(seconds); self.engine.write_text(self.row())
        extract_attach_metrics(self.root, arrive)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'joined')
        self.assertEqual(len(self.waits), 1)

    def test_waits_for_delayed_http_writer(self):
        content = self.http.read_text(); self.http.unlink()
        self.engine.write_text(self.row())
        def arrive(seconds):
            self.waits.append(seconds); self.http.write_text(content)
        extract_attach_metrics(self.root, arrive)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'joined')
        self.assertEqual(len(self.waits), 1)

    def test_torn_line_and_nonobject_are_skipped(self):
        self.engine.write_text(self.row()+'null\n'+ '[1,2]\n'+ '{"uid":')
        extract_attach_metrics(self.root, self.waits.append)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'joined')
        self.assertEqual(self.waits, [])

    def test_rotation_is_searched(self):
        self.http.rename(str(self.http)+'.1')
        pathlib.Path(str(self.engine)+'.2').write_text(self.row())
        extract_attach_metrics(self.root, self.waits.append)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'joined')

    def test_missing_run_id_is_fail_closed(self):
        self.http.write_text(json.dumps(dict(request_id='new-request', engine_uid=0))+'\n')
        self.engine.write_text(self.row())
        extract_attach_metrics(self.root, self.waits.append)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'missing_run_id')
        self.assertNotIn('engine', self.result)
        self.assertEqual(self.waits, [])

    def test_old_run_only_times_out_with_bounded_retries(self):
        self.engine.write_text(self.row('old-run'))
        extract_attach_metrics(self.root, self.waits.append)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'engine_row_timeout')
        self.assertNotIn('engine', self.result)
        self.assertEqual(len(self.waits), 60)

    def test_missing_http_record_is_reported(self):
        self.http.write_text('')
        extract_attach_metrics(self.root, self.waits.append)(self.result)
        self.assertEqual(self.result['engine_metrics_status'], 'request_row_timeout')

    def test_run_id_initialization_precedes_server_import(self):
        path = BENCH.parents[1]/'serve_guarded.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        assignments = [n for n in main.body if isinstance(n, ast.Assign) and 'QWEN36_RUN_ID' in ast.unparse(n)]
        imports = [n for n in main.body if isinstance(n, ast.ImportFrom) and n.module == 'freetoken.server']
        self.assertEqual(len(assignments), 1)
        self.assertLess(assignments[0].lineno, imports[0].lineno)

    def test_request_record_inherits_run_id(self):
        path = BENCH.parents[1]/'request_metrics.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RequestRecord')
        env = dict(os=types.SimpleNamespace(environ={'QWEN36_RUN_ID':'new-run'}),
                   time=types.SimpleNamespace(monotonic=lambda:0.), utc_now=lambda:'synthetic-time',
                   uuid=types.SimpleNamespace(uuid4=lambda:types.SimpleNamespace(hex='request-id')),
                   KINDS={('POST','/v1/chat/completions'):'generation'})
        exec(compile(ast.Module(body=[cls],type_ignores=[]), str(path), 'exec'), env)
        record = env['RequestRecord'](dict(method='POST',path='/v1/chat/completions'), 'session', object())
        self.assertEqual(record.row['run_id'], 'new-run')


def bench_probes():
    """Return evidence of defects, not passing product acceptance tests."""
    findings = []
    with tempfile.TemporaryDirectory(prefix='qwen36-telemetry-cpu-') as temp:
        root = pathlib.Path(temp); (root/'logs').mkdir(); (root/'reports').mkdir()
        http_log, engine_log = root/'logs/requests.jsonl', root/'reports/engine.jsonl'
        http_log.write_text(json.dumps(dict(request_id='new-request', engine_uid=0, run_id='new-run'))+'\n')
        engine_log.write_text(json.dumps(dict(uid=0, pid=111, run_id='old-run', marker='old-process'))+'\n')
        waits = []
        join = extract_attach_metrics(root, waits.append)
        result = dict(request_id='new-request'); join(result)
        findings.append(dict(id='stale_uid_join', reproduced=result.get('engine', {}).get('marker') == 'old-process',
                             evidence='New request joined prior process uid=0 before new row was logged.'))
        engine_log.write_text(''); waits.clear()
        def arrive_after_wait(seconds):
            waits.append(seconds)
            engine_log.write_text(json.dumps(dict(uid=0, pid=222, run_id='new-run', marker='new-process'))+'\n')
        result = dict(request_id='new-request'); extract_attach_metrics(root, arrive_after_wait)(result)
        findings.append(dict(id='does_not_wait_engine_writer', reproduced='engine' not in result and not waits,
                             evidence='HTTP record exists, engine writer is delayed; join returned without retry.'))
        engine_log.write_text(json.dumps(dict(uid=0, pid=222, run_id='new-run'))+'\n'+'{"uid":')
        try:
            extract_attach_metrics(root, waits.append)(dict(request_id='new-request'))
            reproduced = False
        except json.JSONDecodeError:
            reproduced = True
        findings.append(dict(id='torn_jsonl_crashes_join', reproduced=reproduced,
                             evidence='Partial trailing engine JSONL line raised JSONDecodeError.'))
    return findings


def hybrid_probe():
    test = TelemetryTests('test_hybrid_summary_is_not_overwritten_by_gpu_fix')
    test.setUp()
    try:
        c, r = HybridCache(), req(); e = engine(c)
        test.telemetry.batch_started(batch(r), e)
        c.stat_active, c.stat_missing, c.stat_fetched = Tensor(48), Tensor(16), Tensor(6)
        row = test.finish(r, e)
        delta = row.get('expert_counter_deltas', {})
        return dict(id='hybrid_delta_uses_gpu_lru_counters',
                    reproduced=delta.get('decode_active') == 0 and delta.get('decode_h2d_payload_bytes') == 0,
                    evidence='Separate hybrid counters increased by 24 active / 8 missing / 3 fetched, '
                             'but request delta reports zero activity/transfer from unchanged GPU lru_stats.')
    finally:
        test.doCleanups()


def main():
    global SOURCE, BENCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=pathlib.Path, default=SOURCE)
    parser.add_argument('--bench', type=pathlib.Path, default=BENCH)
    parser.add_argument('--report', type=pathlib.Path)
    args = parser.parse_args(); SOURCE, BENCH = args.source.resolve(), args.bench.resolve()
    log = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (TelemetryTests, BenchJoinTests))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    probes = bench_probes() + [hybrid_probe()]
    receipt = dict(schema=1, scope='CPU stdlib mocks only; no GPU, services, secrets or actual torch imported',
                   python=sys.version, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   success=result.wasSuccessful(), known_defect_probes=probes,
                   source_sha256={str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path):
                       hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, SOURCE.with_name('engine.py'), BENCH,
                           BENCH.parents[1]/'serve_guarded.py', BENCH.parents[1]/'request_metrics.py', pathlib.Path(__file__).resolve())},
                   test_output=log.getvalue())
    if args.report:
        args.report.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(log.getvalue(), end='')
    print(json.dumps(dict(known_defect_probes=probes), indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
