"""CPU-only tests for the soak v2 gate: synthetic receipts/JSONL, fake clock and fake HTTP.

Run (WSL): CUDA_VISIBLE_DEVICES= python test-soak-cpu.py --report <json>
No network, tokenizer, model, GPU, systemd or live ROOT is touched: bench.py is
read by AST (validate_receipt only), request_metrics.NumericLog writes into a
temporary directory, and every HTTP response comes from FakeClient.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import inspect
import io
import json
import os
import pathlib
import re
import runpy
import subprocess
import sys
import tempfile
import time
import types
import unittest
import uuid
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2] if len(HERE.parents) > 2 else HERE
BEFORE = WORKSPACE / 'work/agents/soak-gate/before'
UTC = datetime.timezone.utc


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soak = load('soak_under_test', HERE / 'soak.py')
analysis = load('soak_analysis_under_test', HERE / 'soak-analysis.py')
request_metrics = load('request_metrics_under_test', next(
    path for path in (HERE / 'request_metrics.py', __import__('repo_paths').SERVING / 'request_metrics.py') if path.is_file()))
SOAK_SHA256 = soak.file_sha256(HERE / 'soak.py')


def extract_bench():
    """validate_receipt and its constants from the current bench.py, without its imports.

    generate_payload(...) is bench.generate's own payload expression (no HTTP), so the
    fake service records exactly what bench.generate would send.
    """
    tree = ast.parse((HERE / 'bench.py').read_text(encoding='utf-8'))
    functions = {'valid_sha', 'validate_speed_config', 'validate_receipt', 'canonical_hash'}
    body = [node for node in tree.body
            if (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name in functions)]
    generate = copy.deepcopy(next(node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name == 'generate'))
    payload = next(node for node in generate.body if isinstance(node, ast.Assign)
                   and [getattr(target, 'id', None) for target in node.targets] == ['payload'])
    generate.name = 'generate_payload'
    generate.body = [payload, ast.Return(value=ast.Name(id='payload', ctx=ast.Load()))]
    body.append(generate)
    namespace = {'os':os, 'hashlib': hashlib, 'json': json, 're': re, 'pathlib': pathlib}
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(HERE / 'bench.py'), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


BENCH = extract_bench()
MIB = 1024 * 1024


def validate_run(run):
    return BENCH.validate_receipt(run, run.get('profile') if isinstance(run, dict) else None)


def recorded_sampling(payload):
    """The sampling dict the real request_metrics.RequestRecord.configure stores for payload."""
    record = object.__new__(request_metrics.RequestRecord)
    record.row = {}
    request_metrics.RequestRecord.configure(record, payload)
    return record.row['sampling']


def make_run(run_id, profile='main-retain', source_marker='a'):
    files = {'engine/engine.py': source_marker * 64, 'server/api_server.py': 'b' * 64}
    return {'started_utc': '2026-10-13T11:58:00+00:00', 'profile': profile, 'run_id': run_id,
            'native_libraries': {}, 'configuration': {'context': 131072, 'chunk': 4096, 'flags': [], 'env': {}},
            'source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'source_files': files, 'checkpoint': BENCH.CHECKPOINT, 'checkpoint_revision': BENCH.CHECKPOINT_REVISION,
            'packages': {name: '1.0' for name in BENCH.PACKAGES},
            'service_source_files': {name: 'c' * 64 for name in BENCH.SERVICE_FILES}}


def candidate_for(profile='main-retain', source_marker='a'):
    """Frozen-candidate binding as soak.resolve_candidate returns it for a sha256 argument."""
    identity = BENCH.validate_receipt(make_run('0' * 32, profile, source_marker), profile)
    return {'profile': profile, 'identity_sha256': soak.canonical_sha256(identity), 'source': 'sha256'}


CANDIDATE = candidate_for()


@contextlib.contextmanager
def foreign_lock(path):
    """Another process (not a descendant holding our descriptor) holds the flock on path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = ('import fcntl,os,sys\n'
            'fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600)\n'
            'fcntl.flock(fd,fcntl.LOCK_EX)\n'
            'print("locked",flush=True)\n'
            'sys.stdin.read()\n')
    proc = subprocess.Popen([sys.executable, '-I', '-c', code, str(path)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    try:
        if proc.stdout.readline().strip() != 'locked':
            raise RuntimeError('lock holder did not start')
        yield proc
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
        proc.stdout.close()


class FakeWorld:
    """Deterministic service double: clock, request/engine/state JSONL writers, resources."""
    STATE_PERIOD = 10.0

    def __init__(self, root, *, prefill_tps=2500.0, decode_tps=70.0, request_log_max=48 * 1024,
                 engine_rotate_bytes=64 * 1024, state_rotate_bytes=96 * 1024):
        self.root = pathlib.Path(root)
        (self.root / 'logs').mkdir(parents=True, exist_ok=True)
        (self.root / 'reports').mkdir(parents=True, exist_ok=True)
        self.t = 0.0
        self.wall0 = datetime.datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
        self.run_id = uuid.uuid4().hex
        self.serving_run_id = self.run_id
        self.session_id = uuid.uuid4().hex
        self.instance_id = str(uuid.uuid4())
        self.active_run = make_run(self.run_id)
        self.disk_run = copy.deepcopy(self.active_run)
        self.unit = {'active_state': 'active', 'sub_state': 'running', 'invocation_id': 'invocation-1',
                     'main_pid': 4242, 'n_restarts': 0}
        self.request_log = request_metrics.NumericLog(self.root / 'logs', max_bytes=request_log_max)
        self.engine_rotate_bytes, self.state_rotate_bytes = engine_rotate_bytes, state_rotate_bytes
        self.prefill_tps, self.decode_tps = prefill_tps, decode_tps
        self.uid = 0
        self.slot_busy_until = -1.0
        self.next_state = self.STATE_PERIOD
        self.memory = {'anon': 2_730_000_000, 'shmem': 17_040_000_000, 'file_cache': 19_000_000,
                       'gpu_used': 13_000_000_000, 'torch_allocated': 9_000_000_000, 'torch_reserved': 9_500_000_000}
        self.leak_anon_per_request = 0
        self.leak_shmem_per_request = 0
        self.leak_torch_allocated_per_request = 0
        self.leak_torch_reserved_per_request = 0
        self.counters = {'metrics_dropped': 0, 'metrics_failed_writes': 0, 'resource_sample_failures': 0}
        self.tool_mode, self.cancel_mode, self.fail_generate = 'ok', 'ok', False
        # Model answers (correct by default) and wrong-answer switches.
        self.arithmetic_answer, self.tool_final_answer, self.followup_answer = '56', 'Результат: 42', '7'
        self.tool_call_type, self.tool_finish, self.tool_final_calls = 'function', 'tool_calls', False
        self.prompt_scale = 1.0
        self.protocol_errors = []  # requests whose messages are not the original workload
        self.engine_delay = 0.0  # engine.jsonl rows written this many seconds after the response
        self.pending_engine = []
        self.candidate = dict(CANDIDATE)
        self.lock_path = self.root / 'run' / 'gpu-exclusive.lock'
        self.hooks = []
        self.started = False

    def monotonic(self):
        return self.t

    def wall(self):
        return self.wall0 + datetime.timedelta(seconds=self.t)

    def sleep(self, seconds):
        self.advance(seconds, busy=False)

    def advance(self, seconds, busy):
        end = self.t + seconds
        while self.next_state <= end:
            self.t = self.next_state
            self._flush_engine()
            self.emit_state(busy)
            self.next_state += self.STATE_PERIOD
        self.t = end
        self._flush_engine()

    def _flush_engine(self):
        due = [row for when, row in self.pending_engine if when <= self.t]
        if due:
            self.pending_engine = [(when, row) for when, row in self.pending_engine if when > self.t]
            for row in due:
                self._append('reports/engine.jsonl', row, self.engine_rotate_bytes)

    def _append(self, relative, row, rotate_at):
        path = self.root / relative
        line = json.dumps(row) + '\n'
        if rotate_at and path.exists() and path.stat().st_size + len(line) > rotate_at:
            for index in range(5, 0, -1):
                old = path if index == 1 else path.with_name(f'{path.name}.{index - 1}')
                if old.exists():
                    os.replace(old, path.with_name(f'{path.name}.{index}'))
        with open(path, 'a', encoding='utf-8') as out:
            out.write(line)

    def resources(self):
        return {'service_cgroup_memory_bytes': self.memory['anon'] + self.memory['shmem'] + self.memory['file_cache'],
                'service_memory_anon_bytes': self.memory['anon'], 'service_memory_shmem_bytes': self.memory['shmem'],
                'service_memory_file_bytes': self.memory['shmem'] + self.memory['file_cache'],
                'service_memory_file_cache_estimate_bytes': self.memory['file_cache'],
                'gpu_total_used_bytes': self.memory['gpu_used'], 'gpu_free_bytes': 16_000_000_000 - self.memory['gpu_used'],
                'gpu_total_bytes': 16_000_000_000, 'wsl_mem_available_bytes': 20_000_000_000,
                'wsl_swap_free_bytes': 8_000_000_000}

    def emit_state(self, busy):
        row = {'ts': self.wall().isoformat(), 'schema': 2, 'session_id': self.session_id, 'model': 'qwen36-35b',
               'ingress': {}, 'metrics_queue_depth': 0, 'kv_dtype': 'bf16', 'profile': self.active_run['profile'],
               'context_limit': 131072, 'moe_strategy': 'offload', 'memory_ratio': .73, 'prefill_chunk': 4096,
               'active_inference': 1 if busy else 0, 'completed_requests': self.uid, 'model_ready': True,
               'instance_id': self.instance_id, **self.counters, **self.resources()}
        self._append('logs/state.jsonl', row, self.state_rotate_bytes)
        for hook in list(self.hooks):
            hook()

    def server_request(self, *, status=200, outcome='completed', prompt=0, completion=0, cached=0, duration=.1,
                       engine=True, cancelled=False, finish='length', sampling=None):
        self.advance(duration, busy=status == 200)
        rid, uid = uuid.uuid4().hex, None
        if engine:
            uid, self.uid = self.uid, self.uid + 1
            self.memory['anon'] += self.leak_anon_per_request
            self.memory['shmem'] += self.leak_shmem_per_request
            self.memory['torch_allocated'] += self.leak_torch_allocated_per_request
            self.memory['torch_reserved'] += self.leak_torch_reserved_per_request
        peaks = {key: value for key, value in self.resources().items()
                 if key in ('service_cgroup_memory_bytes', 'service_memory_anon_bytes', 'service_memory_shmem_bytes',
                            'service_memory_file_bytes', 'service_memory_file_cache_estimate_bytes',
                            'gpu_total_used_bytes')}
        peaks['min_gpu_free_bytes'] = 16_000_000_000 - self.memory['gpu_used']
        row = {'schema': 2, 'ts': self.wall().isoformat(), 'request_id': rid, 'run_id': self.serving_run_id,
               'model': 'qwen36-35b', 'context_limit': 131072, 'session_id': self.session_id, 'kind': 'generation',
               'stream': True, 'status': status, 'outcome': outcome, 'prompt_tokens': prompt,
               'completion_tokens': completion, 'cached_prompt_tokens': cached, 'counts_final': outcome == 'completed',
               'ttft_generated_s': .1, 'resource_samples': max(1, int(duration)), 'peaks': peaks, 'duration_s': duration,
               'counter_deltas': {}, 'engine_uid': uid, 'finish_reason': finish}
        if sampling is not None:  # absent like a 429 refused at admission, before the body is parsed
            row['sampling'] = sampling
        self.request_log.write('requests.jsonl', row)
        if engine:
            engine_row = {
                'uid': uid, 'pid': 999, 'run_id': self.serving_run_id, 'cancelled': cancelled,
                'engine_span_s': duration, 'prefill_gpu_ms': 1.0, 'decode_gpu_ms': 1.0,
                'torch_allocated_bytes': self.memory['torch_allocated'],
                'torch_reserved_bytes': self.memory['torch_reserved'], 'kv_bytes': 2_500_000_000}
            if self.engine_delay:
                self.pending_engine.append((self.t + self.engine_delay, engine_row))
            else:
                self._append('reports/engine.jsonl', engine_row, self.engine_rotate_bytes)
        return rid


class FakeClient:
    def __init__(self, world):
        self.world = world

    @staticmethod
    def tokens(text):
        match = re.match(r'<doc:(\d+):(\d+)>', text or '')
        rest = len(text or '') - (len(match.group(0)) if match else 0)
        return (int(match.group(1)) if match else 0) + rest // 3 + 1

    ANSWER = 'Синтетический ответ о проверке записей.'

    def _protocol_error(self, where):
        """The request is not the original workload: the fake API rejects it with 400."""
        self.world.protocol_errors.append(where)
        return self.world.server_request(status=400, outcome='rejected', engine=False, duration=.005, finish=None)

    def _generate_ok(self, messages):
        first_ok = (messages and messages[0].get('role') == 'user'
                    and (messages[0].get('content') or '').endswith(soak.FRESH_SUFFIX))
        if len(messages) == 1:
            return bool(first_ok)
        return bool(first_ok and len(messages) == 3 and messages[1] == {'role': 'assistant', 'content': self.ANSWER}
                    and messages[2] == {'role': 'user', 'content': soak.PREFIX_FOLLOWUP})

    def generate(self, messages, maximum, seed, temperature):
        world = self.world
        if world.fail_generate:
            world.advance(.05, busy=False)
            return {'status': 500, 'error_code': 'http_request_failed'}, ''
        if not self._generate_ok(messages):
            rid = self._protocol_error('generate')
            return {'status': 400, 'error_code': 'http_request_failed', 'request_id': rid}, ''
        payload = BENCH.generate_payload(messages, maximum, seed, temperature)
        prompt = sum(self.tokens(message['content']) for message in messages) + 10 * len(messages)
        prompt = int(prompt * world.prompt_scale)
        cached = prompt - 40 if len(messages) > 1 else 0
        duration = prompt / world.prefill_tps + maximum / world.decode_tps
        rid = world.server_request(prompt=prompt, completion=maximum, cached=cached, duration=duration,
                                   sampling=recorded_sampling(payload))
        answer = self.ANSWER
        return ({'status': 200, 'usage': {'prompt_tokens': prompt, 'completion_tokens': maximum},
                 'finish_reason': 'length', 'content_sha256': hashlib.sha256(answer.encode()).hexdigest(),
                 'request_id': rid, 'ttft_content_s': prompt / world.prefill_tps, 'elapsed_s': duration,
                 'decode_client_tps': world.decode_tps, 'serving_run_id': world.serving_run_id,
                 'engine_metrics_status': 'joined', 'cached_prompt_tokens': cached, 'cache_hit_observed': cached > 0},
                answer)

    def _meta_kind(self, messages):
        user = lambda text: [{'role': 'user', 'content': text}]
        for kind, text in (('arithmetic', soak.ARITHMETIC_PROMPT), ('followup', soak.CANCEL_FOLLOWUP),
                           ('tool_call', soak.TOOL_PROMPT)):
            if messages == user(text):
                return kind
        if (len(messages) == 3 and messages[0] == user(soak.TOOL_PROMPT)[0] and messages[1].get('role') == 'assistant'
                and messages[2] == {'role': 'tool', 'tool_call_id': 'call_1', 'content': soak.TOOL_RESULT}):
            return 'tool_final'
        return None

    def request_meta(self, messages, **kwargs):
        world = self.world
        if world.t < world.slot_busy_until:  # guard admission check: refused before the body is parsed
            rid = world.server_request(status=429, outcome='rejected', engine=False, duration=.005, finish=None)
            raise soak.HTTPStatusError({'status': 429, 'request_id': rid, 'elapsed_s': .005})
        kind = self._meta_kind(messages)
        if kind is None:
            rid = self._protocol_error('request_meta')
            raise soak.HTTPStatusError({'status': 400, 'request_id': rid, 'elapsed_s': .005})
        payload = soak.meta_payload(messages, **kwargs)
        finish = 'stop'
        if kind == 'tool_call' and kwargs.get('tools'):
            if world.tool_mode == 'ok':
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'call_1', 'type': world.tool_call_type,
                    'function': {'name': 'fixture_add', 'arguments': json.dumps({'a': 17, 'b': 25})}}]}
                finish = world.tool_finish
            else:
                message = {'role': 'assistant', 'content': 'ok'}
        elif kind == 'tool_final':
            message = {'role': 'assistant', 'content': world.tool_final_answer}
            if world.tool_final_calls:
                message['tool_calls'] = [{'id': 'call_2', 'type': 'function',
                                          'function': {'name': 'fixture_add', 'arguments': '{"a":1,"b":2}'}}]
        elif kind == 'arithmetic':
            message = {'role': 'assistant', 'content': world.arithmetic_answer}
        elif kind == 'followup':
            message = {'role': 'assistant', 'content': world.followup_answer}
        else:
            message = {'role': 'assistant', 'content': ''}
        rid = world.server_request(prompt=40, completion=5, duration=.3, finish=finish,
                                   sampling=recorded_sampling(payload))
        return ({'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]},
                {'status': 200, 'request_id': rid, 'elapsed_s': .3})

    def cancel_stream(self, payload):
        world = self.world
        if not (payload.get('messages') == [{'role': 'user', 'content': soak.CANCEL_PROMPT}]
                and payload.get('ignore_eos') is True and payload.get('stream') is True):
            rid = self._protocol_error('cancel_stream')  # ignore_eos is not recorded server-side
            raise soak.HTTPStatusError({'status': 400, 'request_id': rid})
        completion = 8192 if world.cancel_mode == 'ran_to_completion' else 4
        rid = world.server_request(outcome='cancelled', prompt=20, completion=completion, duration=.8,
                                   cancelled=True, finish=None, sampling=recorded_sampling(payload))
        world.slot_busy_until = world.t + .15
        return {'status': 200, 'request_id': rid, 'content_seen': world.cancel_mode != 'no_content', 'data_events': 3}


def make_runner(world, *, name='soak-test', seconds=7200, partial_interval=600.0, background=True):
    monitor = soak.TelemetryMonitor(world.root, monotonic=world.monotonic, poll_seconds=.01)
    identity = soak.IdentityGuard(
        candidate=world.candidate,
        read_receipt=lambda: copy.deepcopy(world.active_run), validate=validate_run,
        current_identity=lambda profile: BENCH.validate_receipt(world.disk_run, profile),
        unit_probe=lambda: dict(world.unit), monotonic=world.monotonic)
    events = []
    runner = soak.SoakRunner(name=name, seconds=seconds, reports_dir=world.root / 'reports', client=FakeClient(world),
                             document=lambda length, variant: f'<doc:{length}:{variant}>', identity=identity,
                             monitor=monitor, gpu_lock_path=world.lock_path, monotonic=world.monotonic,
                             wall=world.wall, sleep=world.sleep, harness=soak.harness_hashes(HERE), emit=events.append,
                             partial_interval=partial_interval)
    runner.events = events
    if not background:
        monitor.start_background = lambda tick=None: None
    # Fake time outruns the real background thread, so every simulated state period runs one
    # cycle of that thread's own body (poll + the tick passed to start_background) while the
    # thread is running. The runner's own end-of-iteration _tick is not substituted here.
    world.hooks.append(lambda: monitor.background_running() and monitor.background_step())
    return runner


def finish_run(world, runner, name='soak-test'):
    code = runner.run()
    reports = world.root / 'reports'
    path, copy_path = reports / f'{name}.json', reports / f'{name}-run.json'
    receipt = json.loads(path.read_text()) if path.exists() else None
    runner.run_copy_data = json.loads(copy_path.read_text()) if copy_path.exists() else None
    return runner, code, receipt


def run_soak(world, **kwargs):
    return finish_run(world, make_runner(world, **kwargs), kwargs.get('name', 'soak-test'))


def evaluate(receipt, **kwargs):
    """The CLI's bindings: expected candidate main-retain/'a', validate against that profile; the run
    copy defaults to a matching copy of receipt['run'] (run-copy tests pass their own)."""
    profile = kwargs.pop('expected_profile', CANDIDATE['profile'])
    params = dict(soak=soak, soak_sha256=SOAK_SHA256, expected_name='soak-test', expected_profile=profile,
                  expected_identity_sha256=CANDIDATE['identity_sha256'],
                  run_copy=copy.deepcopy(receipt.get('run')) if isinstance(receipt, dict) else None,
                  validate_run=lambda run: BENCH.validate_receipt(run, profile))
    params.update(kwargs)
    return analysis.evaluate(receipt, **params)


_PASS = {}


def pass_receipt():
    if 'receipt' not in _PASS:
        with tempfile.TemporaryDirectory(prefix='qwen36-soak-pass-') as temp:
            world = FakeWorld(temp)
            runner, code, receipt = run_soak(world)
            _PASS.update(receipt=receipt, code=code, events=runner.events, run_copy=runner.run_copy_data,
                         protocol_errors=list(world.protocol_errors), active_run=copy.deepcopy(world.active_run),
                         lock_path=str(world.lock_path))
    return copy.deepcopy(_PASS['receipt'])


def fresh_server(receipt, length=None, index=0):
    rows = [row for row in receipt['rows'] if row['case'] == 'fresh' and (length is None or row['length'] == length)]
    return receipt['telemetry']['requests'][rows[index]['requests'][0]['request_id']]


def server_of(receipt, case, entry=0, iteration=None):
    rows = [row for row in receipt['rows'] if row['case'] == case and (iteration is None or row['iteration'] == iteration)]
    return receipt['telemetry']['requests'][rows[0]['requests'][entry]['request_id']]


def engine_of(receipt, server):
    return receipt['telemetry']['engine'][f"{receipt['run']['run_id']}:{server['engine_uid']}"]


def first(rows, case):
    return next(row for row in rows if row['case'] == case)


# ---------------------------------------------------------------------------
class PlanAndContractTests(unittest.TestCase):
    def test_plan_runs_without_network_tokenizer_or_bench(self):
        blocked = {name: None for name in ('requests', 'transformers', 'bench')}
        out = io.StringIO()
        with mock.patch.dict(sys.modules, blocked), contextlib.redirect_stdout(out):
            self.assertEqual(soak.main(['--plan']), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['mode'], 'plan-only-no-network')
        self.assertEqual(plan['coverage_floors'], {'fresh@64': 12, 'fresh@4096': 12, 'fresh@16384': 12,
                                                   'fresh@65536': 12, 'fresh@120000': 12, 'prefix': 12,
                                                   'arithmetic': 9, 'tool': 4, 'cancel': 4})
        self.assertEqual(plan['planned_case_counts'], plan['coverage_floors'])
        self.assertEqual(len(plan['schedule']), soak.MIN_ITERATIONS)
        self.assertEqual(plan['workload_sha256'], soak.workload_sha256())
        for text in (soak.FRESH_SUFFIX, soak.TOOL_PROMPT, soak.CANCEL_PROMPT):  # prompt texts only as hashes
            self.assertNotIn(json.dumps(text)[1:-1], out.getvalue())
            self.assertNotIn(text, out.getvalue())

    def test_module_has_no_heavy_top_level_imports(self):
        tree = ast.parse((HERE / 'soak.py').read_text(encoding='utf-8'))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module)
        self.assertFalse(names & {'requests', 'transformers', 'bench', 'quality', 'torch'})

    def test_original_workload_is_preserved(self):
        original = BEFORE / 'soak.py'
        if not original.is_file():
            self.skipTest('pre-change copy unavailable outside the workspace')
        source = original.read_text(encoding='utf-8')
        literals = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        for text in (soak.FRESH_SUFFIX, soak.PREFIX_FOLLOWUP, soak.ARITHMETIC_PROMPT, soak.TOOL_PROMPT,
                     soak.CANCEL_PROMPT, soak.CANCEL_FOLLOWUP, soak.TOOLS[0]['function']['description']):
            self.assertIn(text, literals)
        compact = source.replace(' ', '')
        for fragment in ('[64,4096,16384,65536,120000][iteration%5]', 'generate(messages,256,iteration+341,temperature=.6)',
                         'generate(follow,128,iteration+341,temperature=.6)', 'document(t,length,1000000+iteration)',
                         'iteration%5==1', 'iteration%7==0', 'iteration%17==0', 'iteration%19==0', 'time.sleep(5)',
                         'errors>=3', "'max_tokens':8192", "'ignore_eos':True", 'max_tokens=32', 'range(50)',
                         'time.sleep(.1)', 'timeout=60', "default=7200", "'{\"result\":42}'"):
            self.assertIn(fragment, compact)
        self.assertEqual(soak.LENGTHS, (64, 4096, 16384, 65536, 120000))
        self.assertEqual((soak.FRESH_MAX_TOKENS, soak.PREFIX_MAX_TOKENS, soak.SEED_OFFSET, soak.DOCUMENT_VARIANT_OFFSET),
                         (256, 128, 341, 1000000))
        self.assertEqual((soak.PREFIX_EVERY, soak.PREFIX_PHASE, soak.ARITHMETIC_EVERY, soak.TOOL_EVERY,
                          soak.CANCEL_EVERY, soak.PAUSE_SECONDS, soak.MAX_CONSECUTIVE_ERRORS, soak.REQUIRED_SECONDS),
                         (5, 1, 7, 17, 19, 5, 3, 7200))
        self.assertEqual((soak.CANCEL_MAX_TOKENS, soak.CANCEL_ATTEMPTS, soak.CANCEL_RETRY_SECONDS,
                          soak.CANCEL_STREAM_TIMEOUT, soak.SMOKE_MAX_TOKENS), (8192, 50, .1, 60, 32))

    def test_workload_hash_is_pinned_in_the_gate(self):
        self.assertEqual(soak.workload_sha256(), analysis.PINNED_WORKLOAD_SHA256)
        self.assertEqual(analysis.PINNED_WORKLOAD_SHA256,
                         '70d533a63d298ef6a2b9b070bd205d7aa2fe647ed474b9e521c07aeb9b47e89b')

    def test_request_meta_payload_matches_quality_request(self):
        captured = []

        class Response:
            status_code = 200
            headers = {'x-qwen36-request-id': 'f' * 32}

            def raise_for_status(self):
                return None

            def json(self):
                return {'choices': []}

        def post(url, **kwargs):
            captured.append((url, kwargs))
            return Response()
        fake_requests = types.SimpleNamespace(post=post)
        fake_bench = types.SimpleNamespace(headers=lambda: {'Authorization': 'Bearer synthetic'})
        tree = ast.parse((HERE / 'quality.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'request')
        joined = []
        # quality.request also joins the request log for its serving run id (H1); the payload is what is compared.
        namespace = {'requests': fake_requests, 'headers': fake_bench.headers,
                     'attach_metrics': lambda result: joined.append(result['request_id'])}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'quality.py', 'exec'), namespace)
        messages = [{'role': 'user', 'content': 'x'}]
        namespace['request'](messages, max_tokens=32, tools=soak.TOOLS)
        self.assertEqual(joined, ['f' * 32])
        with mock.patch.dict(sys.modules, {'requests': fake_requests, 'bench': fake_bench}):
            body, meta = soak.request_meta(messages, max_tokens=32, tools=soak.TOOLS)
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(meta['request_id'], 'f' * 32)
        self.assertEqual(meta['status'], 200)

    def test_request_meta_raises_with_metadata_on_429(self):
        class Response:
            status_code = 429
            headers = {'x-qwen36-request-id': 'e' * 32}
        fake_requests = types.SimpleNamespace(post=lambda url, **kwargs: Response())
        fake_bench = types.SimpleNamespace(headers=lambda: {})
        with mock.patch.dict(sys.modules, {'requests': fake_requests, 'bench': fake_bench}):
            with self.assertRaises(soak.HTTPStatusError) as caught:
                soak.request_meta([{'role': 'user', 'content': 'x'}])
        self.assertEqual(caught.exception.meta['status'], 429)
        self.assertEqual(caught.exception.meta['request_id'], 'e' * 32)


class CancelParserTests(unittest.TestCase):
    ROLE = b'data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":null}]}'
    NULL = b'data: {"choices":[{"delta":{"content":null},"index":0}]}'
    REASONING = b'data: {"choices":[{"delta":{"reasoning_content":"think"},"index":0}]}'
    REAL = 'data: {"choices":[{"delta":{"content":"Двоичный"},"index":0}]}'.encode()

    def test_original_predicate_stopped_on_empty_role_chunk(self):
        original = lambda line: line.startswith(b'data: ') and b'content' in line
        self.assertTrue(original(self.ROLE))  # the defect: cancelled before any token
        self.assertTrue(original(self.REASONING))

    def test_waits_for_nonempty_content(self):
        consumed = []

        def lines():
            for line in (b': keep-alive', b'', self.ROLE, self.NULL, self.REASONING, self.REAL, b'data: never-read'):
                consumed.append(line)
                yield line
        self.assertEqual(soak.first_nonempty_content(lines()), (True, 4))
        self.assertEqual(consumed[-1], self.REAL)

    def test_stream_without_content_is_not_a_cancellation(self):
        self.assertEqual(soak.first_nonempty_content([self.ROLE, b'data: [DONE]', self.REAL]), (False, 1))

    def test_malformed_event_fails(self):
        with self.assertRaises(soak.CaseFailure):
            soak.first_nonempty_content([b'data: {"choices":'])
        with self.assertRaises(soak.CaseFailure):
            soak.first_nonempty_content([b'data: NaN'])


class NameAndAtomicityTests(unittest.TestCase):
    def test_name_validation(self):
        for good in ('soak128', 'soak-final-v5', 'soak'):
            self.assertEqual(soak.validate_name(good), good)
        for bad in ('', '../soak', 'Soak128', 'soak_128', 'final', 'soak/x', 'soak-', 'soak--x', 'soak-analysis',
                    'soak-x-analysis-y', 'soak' + 'x' * 61, None, 'soak-run', 'soak-x-run'):
            with self.assertRaises(ValueError):
                soak.validate_name(bad)
        self.assertEqual(soak.validate_name('soak-run1'), 'soak-run1')

    def test_existing_run_copy_blocks_the_name(self):
        with tempfile.TemporaryDirectory() as temp:
            reports = pathlib.Path(temp)
            (reports / 'soak-c-run.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                soak.ensure_unused(reports, 'soak-c')
            with self.assertRaises(FileExistsError):
                soak.claim_report(reports, 'soak-c', {})
            self.assertFalse((reports / 'soak-c.json').exists())

    def test_claim_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            reports = pathlib.Path(temp)
            path = soak.claim_report(reports, 'soak-a', {'status': 'claimed'})
            self.assertEqual(json.loads(path.read_text())['status'], 'claimed')
            with self.assertRaises(FileExistsError):
                soak.claim_report(reports, 'soak-a', {'status': 'claimed'})
            (reports / 'soak-b-analysis.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                soak.claim_report(reports, 'soak-b', {})
            self.assertEqual(oct(path.stat().st_mode & 0o777), '0o600')

    def test_atomic_write_keeps_previous_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / 'soak-x.json'
            soak.atomic_write_json(path, {'status': 'running', 'n': 1})
            with mock.patch.object(soak.os, 'replace', side_effect=OSError('simulated')):
                with self.assertRaises(OSError):
                    soak.atomic_write_json(path, {'status': 'running', 'n': 2})
            self.assertEqual(json.loads(path.read_text())['n'], 1)
            self.assertEqual(sorted(p.name for p in pathlib.Path(temp).iterdir()), ['soak-x.json'])
            with self.assertRaises(ValueError):
                soak.atomic_write_json(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text())['n'], 1)

    def test_partial_receipt_is_complete_json_during_run(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            seen = []

            def capture():
                path = world.root / 'reports' / 'soak-test.json'
                if world.t > 200 and path.exists() and len(seen) < 3:
                    seen.append(json.loads(path.read_text()))
            world.hooks.append(capture)
            runner, code, receipt = run_soak(world, seconds=400, partial_interval=30.0)
            self.assertTrue(seen)
            self.assertTrue(all(item['status'] == 'running' and item['complete'] is False for item in seen))
            self.assertFalse([p for p in (world.root / 'reports').iterdir() if p.name.startswith('.')])
            self.assertEqual(receipt['status'], 'finished')


class JsonlTailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = pathlib.Path(self.temp.name) / 'state.jsonl'
        self.counter = 0

    def write(self, count=1, rotate=False):
        for _ in range(count):
            if rotate:
                self.rotate()
            with open(self.path, 'a') as out:
                out.write(json.dumps({'n': self.counter}) + '\n')
            self.counter += 1

    def rotate(self, backups=5):
        for index in range(backups, 0, -1):
            old = self.path if index == 1 else pathlib.Path(f'{self.path}.{index - 1}')
            if old.exists():
                os.replace(old, f'{self.path}.{index}')

    def numbers(self, rows):
        return [row['n'] for row in rows]

    def test_starts_at_end_and_reads_partial_lines(self):
        self.write(3)
        tail = soak.JsonlTail(self.path)
        tail.start_at_end()
        with open(self.path, 'a') as out:
            out.write('{"n": 3}\n{"n"')
        self.assertEqual(self.numbers(tail.poll()), [3])
        with open(self.path, 'a') as out:
            out.write(': 4}\n')
        self.assertEqual(self.numbers(tail.poll()), [4])
        self.assertEqual(tail.stats['malformed_lines'], 0)

    def test_single_and_double_rotation_between_polls_are_recovered(self):
        self.write(2)
        tail = soak.JsonlTail(self.path)
        tail.start_at_end()
        self.write(2)
        self.write(1, rotate=True)
        self.assertEqual(self.numbers(tail.poll()), [2, 3, 4])
        self.write(2)
        self.write(2, rotate=True)  # two generations rotated before the next poll
        self.write(1)
        self.assertEqual(self.numbers(tail.poll()), [5, 6, 7, 8, 9])
        self.assertEqual(tail.stats['loss_events'], 0)
        self.assertGreaterEqual(tail.stats['recovered_files'], 1)
        self.assertEqual(tail.stats['rotations'], 3)

    def test_rotation_beyond_backup_chain_is_loss(self):
        self.write(1)
        tail = soak.JsonlTail(self.path)
        tail.start_at_end()
        self.write(7, rotate=True)  # generation we held fell off .5
        tail.poll()
        self.assertEqual(tail.stats['loss_events'], 1)

    def test_truncation_and_malformed_lines_are_counted(self):
        self.write(3)
        tail = soak.JsonlTail(self.path)
        tail.start_at_end()
        with open(self.path, 'w') as out:  # rewritten in place and longer than before
            out.write('{"n": 100}\nnot-json\n[1]\n{"x": NaN}\n')
        rows = tail.poll()
        self.assertEqual(tail.stats['truncations'], 1)
        self.assertEqual(self.numbers(rows), [100])
        self.assertEqual(tail.stats['malformed_lines'], 3)
        with open(self.path, 'w') as out:  # shorter than what was read
            out.write('{"n": 7}\n')
        self.assertEqual(self.numbers(tail.poll()), [7])
        self.assertEqual(tail.stats['truncations'], 2)

    def test_append_between_stat_and_drain_is_not_truncation(self):
        self.write(2)
        tail = soak.JsonlTail(self.path)
        tail.start_at_end()
        real_stat = os.stat

        def stat_then_append(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if str(path) == str(self.path) and self.counter < 12:
                self.write(1)  # the writer races the reader
            return result
        with mock.patch.object(soak.os, 'stat', side_effect=stat_then_append):
            rows = tail.poll() + tail.poll() + tail.poll()
        rows += tail.poll()
        self.assertEqual(self.numbers(rows), list(range(2, self.counter)))
        self.assertEqual((tail.stats['truncations'], tail.stats['malformed_lines']), (0, 0))

    def test_missing_file_is_read_from_start_when_created(self):
        tail = soak.JsonlTail(self.path)
        tail.start_at_end()
        self.assertTrue(tail.stats['missing_at_start'])
        self.assertEqual(tail.poll(), [])
        self.write(2)
        self.assertEqual(self.numbers(tail.poll()), [0, 1])

    def test_real_numericlog_rotation_is_lossless(self):
        root = pathlib.Path(self.temp.name) / 'logs'
        writer = request_metrics.NumericLog(root, max_bytes=600)
        tail = soak.JsonlTail(root / 'requests.jsonl')
        tail.start_at_end()
        seen = []
        for index in range(400):
            writer.write('requests.jsonl', {'n': index, 'pad': 'x' * 40})
            if index % 9 == 0:
                seen += self.numbers(tail.poll())
        seen += self.numbers(tail.poll())
        self.assertEqual(seen, list(range(400)))
        self.assertGreater(tail.stats['rotations'], 20)
        self.assertEqual(tail.stats['loss_events'], 0)


class RunnerScenarioTests(unittest.TestCase):
    def test_full_synthetic_run_passes(self):
        receipt = pass_receipt()
        self.assertEqual(_PASS['code'], 0)
        self.assertEqual(_PASS['protocol_errors'], [])  # every request was the original workload
        self.assertEqual(_PASS['run_copy'], _PASS['active_run'])  # NAME-run.json = bound active-run receipt
        self.assertEqual(receipt['run_copy'], 'soak-test-run.json')
        self.assertEqual(receipt['gpu_lock'], {'path': _PASS['lock_path'], 'mode': 'acquired'})
        self.assertEqual(receipt['candidate'], CANDIDATE)
        report = evaluate(receipt, run_copy=_PASS['run_copy'])
        self.assertEqual(report['verdict'], 'PASS', report['gate_reasons'])
        self.assertTrue(report['two_hour_execution_gate_pass'])
        self.assertEqual(report['gate_reasons'], [])
        self.assertTrue(report['summary']['candidate']['match'])
        self.assertEqual(report['summary']['candidate']['run_identity_sha256'], CANDIDATE['identity_sha256'])
        self.assertGreaterEqual(receipt['elapsed_s'], 7200)
        readers = receipt['telemetry']['readers']
        self.assertTrue(all(readers[name]['rotations'] > 0 for name in ('requests', 'engine', 'state')))
        self.assertEqual(report['memory']['leak_verdict'], 'not_determined_automatically')
        self.assertEqual(report['memory']['status'], 'NO_THRESHOLD_EXCEEDED_REVIEW_REQUIRED')
        self.assertNotIn('no leak', json.dumps(report['memory']).lower())
        coverage = report['summary']['coverage']
        self.assertTrue(all(coverage['passed'][key] >= need for key, need in coverage['floors'].items()))
        text = json.dumps(receipt, ensure_ascii=False)
        for secret_or_prompt in ('Bearer', soak.TOOL_PROMPT, soak.CANCEL_PROMPT, 'Синтетический ответ'):
            self.assertNotIn(secret_or_prompt, text)
        _PASS['report'] = report

    def test_stale_run_is_refused_before_any_request(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.disk_run = make_run(world.run_id, source_marker='d')
            runner, code, receipt = run_soak(world)
            self.assertEqual(code, 2)
            self.assertIsNone(receipt)
            self.assertEqual(runner.events[-1], {'result': 'REFUSED', 'reason': 'stale_run_source_mismatch'})
            self.assertEqual(world.uid, 0)

    def test_invalid_receipt_and_inactive_unit_are_refused(self):
        for mutate, reason in ((lambda w: w.active_run.pop('run_id'), 'run_receipt_invalid'),
                               (lambda w: w.unit.update(active_state='inactive'), 'service_not_active')):
            with tempfile.TemporaryDirectory() as temp:
                world = FakeWorld(temp)
                mutate(world)
                runner, code, receipt = run_soak(world)
                self.assertEqual((code, receipt), (2, None))
                self.assertEqual(runner.events[-1]['reason'], reason)

    def test_existing_name_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            (world.root / 'reports' / 'soak-test.json').write_text('{"keep": true}')
            runner, code, receipt = run_soak(world)
            self.assertEqual(code, 2)
            self.assertEqual(receipt, {'keep': True})

    def test_run_id_change_mid_run_aborts_and_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)

            def restart():
                if world.t >= 1500 and world.serving_run_id == world.run_id:
                    world.serving_run_id = uuid.uuid4().hex
                    world.active_run = make_run(world.serving_run_id)
                    world.uid = 0
            world.hooks.append(restart)
            runner, code, receipt = run_soak(world)
            self.assertEqual(receipt['termination_reason'], 'fatal_evidence_failure')
            self.assertIn('active_run_changed', receipt['abort_codes'])
            report = evaluate(receipt)
            self.assertEqual(report['verdict'], 'FAIL')
            for code in ('identity_check_failed', 'soak_terminated_early', 'elapsed_below_required'):
                self.assertIn(code, report['failure_reasons'])
            if any(row.get('run_id') != world.run_id for row in receipt['telemetry']['requests'].values()):
                self.assertIn('serving_run_id_mismatch', report['failure_reasons'])

    def test_service_restart_detected_by_unit_invocation(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.hooks.append(lambda: world.t >= 900 and world.unit.update(invocation_id='invocation-2', main_pid=5151))
            runner, code, receipt = run_soak(world)
            self.assertIn('service_restarted', receipt['abort_codes'])
            self.assertIn('identity_check_failed', evaluate(receipt)['failure_reasons'])

    def test_telemetry_drop_counter_aborts_and_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.hooks.append(lambda: world.t >= 2000 and world.counters.update(metrics_dropped=2))
            runner, code, receipt = run_soak(world)
            self.assertIn('telemetry_metrics_dropped', receipt['abort_codes'])
            self.assertIn('telemetry_metrics_dropped', evaluate(receipt)['failure_reasons'])

    def test_short_run_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, code, receipt = run_soak(FakeWorld(temp), seconds=900)
            self.assertEqual(receipt['termination_reason'], 'duration_reached')
            report = evaluate(receipt)
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertIn('required_seconds_below_7200', report['failure_reasons'])
            self.assertIn('elapsed_below_required', report['failure_reasons'])
            self.assertIn('insufficient_coverage', report['inconclusive_reasons'])

    def test_short_pilot_covers_every_case_but_never_passes_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, code, receipt = run_soak(FakeWorld(temp), name='soak-pilot', seconds=30)
            report = evaluate(receipt, expected_name='soak-pilot')
            self.assertFalse(report['two_hour_execution_gate_pass'])
            self.assertTrue(report['pilot_view']['all_case_types_passed'], report['pilot_view'])
            self.assertEqual(report['pilot_view']['failures_excluding_duration'], [])
            self.assertTrue({'required_seconds_below_7200', 'elapsed_below_required'} <= set(report['failure_reasons']))
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.tool_mode = 'no_call'
            runner, code, receipt = run_soak(world, name='soak-pilot', seconds=30)
            report = evaluate(receipt, expected_name='soak-pilot')
            self.assertIn('case_failed', report['pilot_view']['failures_excluding_duration'])

    def test_slow_service_is_inconclusive_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp, prefill_tps=200.0)
            runner, code, receipt = run_soak(world)
            report = evaluate(receipt)
            self.assertEqual(report['verdict'], 'INCONCLUSIVE', report['gate_reasons'])
            self.assertFalse(report['two_hour_execution_gate_pass'])
            self.assertEqual(report['failure_reasons'], [])
            self.assertIn('insufficient_coverage', report['inconclusive_reasons'])
            self.assertIn('memory_insufficient_samples', report['inconclusive_reasons'])
            self.assertEqual(report['memory']['status'], 'INCONCLUSIVE')
            # ~600 s requests: only the background reader polls and checks identity meanwhile.
            self.assertLessEqual(receipt['telemetry']['monitor']['max_poll_interval_s'], 2 * FakeWorld.STATE_PERIOD)
            self.assertLessEqual(report['summary']['identity']['max_check_gap_s'], analysis.IDENTITY_GAP_LIMIT_S)
            self.assertTrue(any(row['t_end_s'] - row['t_start_s'] > 500 for row in receipt['rows']))

    def test_memory_growth_at_equal_length_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.leak_anon_per_request = 3 * 1024 * 1024
            runner, code, receipt = run_soak(world)
            report = evaluate(receipt)
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertEqual(report['failure_reasons'], ['memory_growth_above_threshold'])
            self.assertEqual(report['memory']['status'], 'THRESHOLD_EXCEEDED')
            flagged = {flag.get('length', 'idle') for flag in report['memory']['flags']}
            self.assertTrue({64, 120000, 'idle'} <= flagged)

    def test_consecutive_errors_stop_and_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.fail_generate = True
            runner, code, receipt = run_soak(world)
            self.assertEqual(receipt['termination_reason'], 'consecutive_errors')
            self.assertEqual(len(receipt['rows']), 3)
            report = evaluate(receipt)
            for code in ('case_failed', 'soak_terminated_early', 'scheduled_case_missing'):
                self.assertIn(code, report['failure_reasons'])

    def test_unexecuted_tool_and_cancel_fail(self):
        for attribute, value, failure in (('tool_mode', 'no_call', 'tool_call_invalid'),
                                          ('cancel_mode', 'no_content', 'cancel_no_content_before_disconnect')):
            with tempfile.TemporaryDirectory() as temp:
                world = FakeWorld(temp)
                setattr(world, attribute, value)
                runner, code, receipt = run_soak(world, seconds=600)
                failed = [row for row in receipt['rows'] if not row['pass']]
                self.assertTrue(failed and all(row['failure'] == failure for row in failed))
                self.assertIn('case_failed', evaluate(receipt)['failure_reasons'])

    def test_cancel_that_ran_to_completion_fails_server_check(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.cancel_mode = 'ran_to_completion'
            runner, code, receipt = run_soak(world, seconds=600)
            self.assertIn('cancel_not_interrupted', evaluate(receipt)['failure_reasons'])

    def test_wrong_model_answers_fail_their_case(self):
        # The gate cannot see answers; the runner's facts are the only functional check.
        for attribute, value, case, fact, failure in (
                ('arithmetic_answer', '55', 'arithmetic', 'contains_expected', 'arithmetic_check_failed'),
                ('tool_final_answer', 'Результат: 41', 'tool', 'final_contains_expected', 'tool_check_failed'),
                ('tool_final_calls', True, 'tool', 'final_no_tool_calls', 'tool_check_failed'),
                ('tool_call_type', 'custom', 'tool', 'call_type_function', 'tool_call_invalid'),
                ('tool_finish', 'stop', 'tool', 'finish_reason_tool_calls', 'tool_call_invalid'),
                ('followup_answer', '8', 'cancel', 'followup_contains_expected', 'cancel_followup_content'),
                ('prompt_scale', .5, 'fresh', 'prompt_tokens_cover_length', 'fresh_check_failed')):
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as temp:
                world = FakeWorld(temp)
                setattr(world, attribute, value)
                runner, code, receipt = run_soak(world, seconds=1)  # iteration 0: fresh, arithmetic, tool, cancel
                row = first(receipt['rows'], case)
                self.assertFalse(row['pass'])
                self.assertIs(row['facts'][fact], False)
                self.assertEqual(row['failure'], failure)
                self.assertIn('case_failed', evaluate(receipt)['failure_reasons'])

    def test_changed_request_parameters_fail_the_gate(self):
        def fresh_budget(client):
            original = client.generate
            client.generate = lambda m, maximum, seed, t: original(m, soak.PREFIX_MAX_TOKENS
                                                                   if maximum == soak.FRESH_MAX_TOKENS else maximum, seed, t)

        def fresh_seed(client):
            original = client.generate
            client.generate = lambda m, maximum, seed, t: original(m, maximum, seed + 1, 1.0)

        def arithmetic_budget(client):
            original = client.request_meta
            client.request_meta = lambda m, **kw: original(
                m, **(dict(kw, max_tokens=512) if m[-1].get('content') == soak.ARITHMETIC_PROMPT else kw))

        def cancel_budget(client):
            original = client.cancel_stream
            client.cancel_stream = lambda payload: original(dict(payload, max_tokens=64))

        for label, change in (('control', None), ('fresh_budget', fresh_budget), ('fresh_seed', fresh_seed),
                              ('arithmetic_budget', arithmetic_budget), ('cancel_budget', cancel_budget)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                world = FakeWorld(temp)
                runner = make_runner(world, seconds=30)
                if change is not None:
                    change(runner.client)
                runner, code, receipt = finish_run(world, runner)
                self.assertEqual(world.protocol_errors, [])
                reasons = evaluate(receipt)['failure_reasons']
                if change is None:
                    self.assertNotIn('workload_request_mismatch', reasons)
                else:
                    self.assertIn('workload_request_mismatch', reasons)

    def test_changed_workload_constant_fails_the_pinned_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            runner, code, receipt = run_soak(FakeWorld(temp), seconds=30)
            self.assertNotIn('workload_definition_mismatch', evaluate(receipt)['failure_reasons'])
        with mock.patch.object(soak, 'PREFIX_MAX_TOKENS', 64), tempfile.TemporaryDirectory() as temp:
            runner, code, receipt = run_soak(FakeWorld(temp), seconds=30)
            report = evaluate(receipt)  # receipt, runner and soak.py agree; only the pin differs
            self.assertEqual(receipt['workload_sha256'], soak.workload_sha256())
            self.assertNotIn('workload_request_mismatch', report['failure_reasons'])
            self.assertIn('workload_definition_mismatch', report['failure_reasons'])
            self.assertEqual(report['reason_details']['workload_definition_mismatch']['examples'],
                             [{'field': 'pinned_soak128_workload'}])

    def test_shmem_and_allocator_growth_at_equal_length_fail(self):
        for leaks, metrics, idle in (
                ({'leak_shmem_per_request': 2 * MIB}, ('request.service_memory_shmem_bytes',), True),
                ({'leak_torch_allocated_per_request': 2 * MIB, 'leak_torch_reserved_per_request': 4 * MIB},
                 ('engine.torch_allocated_bytes', 'engine.torch_reserved_bytes'), False)):
            with self.subTest(leaks=sorted(leaks)), tempfile.TemporaryDirectory() as temp:
                world = FakeWorld(temp)
                for attribute, value in leaks.items():
                    setattr(world, attribute, value)
                runner, code, receipt = run_soak(world)
                report = evaluate(receipt)
                self.assertEqual(report['failure_reasons'], ['memory_growth_above_threshold'])
                flags = report['memory']['flags']
                pairs = {(flag['length'], flag['metric']) for flag in flags if 'length' in flag}
                self.assertEqual(pairs, {(length, metric) for length in soak.LENGTHS for metric in metrics})
                idle_flags = [flag['metric'] for flag in flags if flag.get('idle')]
                self.assertEqual(idle_flags, ['service_memory_shmem_bytes'] if idle else [])

    def test_final_drain_waits_for_late_engine_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.engine_delay = 8.0  # engine.jsonl lags the HTTP response; the last rows arrive after the loop
            runner, code, receipt = run_soak(world, seconds=300)
            self.assertEqual(world.pending_engine, [])
            reasons = evaluate(receipt)['failure_reasons']
            self.assertNotIn('engine_row_missing', reasons)
            self.assertNotIn('unjoined_engine_rows', reasons)
            last = next(entry for row in reversed(receipt['rows']) for entry in reversed(row['requests'])
                        if entry['role'] != 'cancel_followup_rejected')
            server = receipt['telemetry']['requests'][last['request_id']]
            self.assertIn(f"{world.run_id}:{server['engine_uid']}", receipt['telemetry']['engine'])


class CandidateBindingTests(unittest.TestCase):
    def test_wrong_profile_or_build_is_refused_before_any_request(self):
        for label, run in (('profile', lambda w: make_run(w.run_id, profile='bf128')),
                           ('build', lambda w: make_run(w.run_id, source_marker='e'))):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                world = FakeWorld(temp)
                world.active_run = run(world)
                world.disk_run = copy.deepcopy(world.active_run)  # running service matches its own disk
                runner, code, receipt = run_soak(world)
                self.assertEqual((code, receipt), (2, None))
                self.assertEqual(runner.events[-1], {'result': 'REFUSED', 'reason': 'candidate_mismatch'})
                self.assertEqual(world.uid, 0)
                self.assertFalse((world.root / 'reports' / 'soak-test-run.json').exists())

    def test_other_candidate_receipt_fails_analysis(self):
        # The probe from the review: a complete run of another profile used to PASS.
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.active_run = make_run(world.run_id, profile='baseline64', source_marker='f')
            world.disk_run = copy.deepcopy(world.active_run)
            world.candidate = candidate_for('baseline64', 'f')
            runner, code, receipt = run_soak(world, seconds=30)
            self.assertEqual(receipt['status'], 'finished')
            wrong = evaluate(receipt)
            self.assertIn('candidate_identity_mismatch', wrong['failure_reasons'])
            self.assertEqual(wrong['reason_details']['candidate_identity_mismatch']['examples'],
                             [{'fields': ['profile', 'identity_sha256', 'runner_candidate']}])
            right = evaluate(receipt, expected_profile='baseline64',
                             expected_identity_sha256=world.candidate['identity_sha256'])
            self.assertNotIn('candidate_identity_mismatch', right['failure_reasons'])
            self.assertNotIn('run_receipt_invalid', right['failure_reasons'])

    def test_candidate_mismatch_on_a_complete_receipt(self):
        receipt = pass_receipt()
        for label, kwargs, field in (
                ('profile', {'expected_profile': 'bf128'}, 'profile'),
                ('build', {'expected_identity_sha256': candidate_for(source_marker='e')['identity_sha256']},
                 'identity_sha256')):
            with self.subTest(label=label):
                report = evaluate(copy.deepcopy(receipt), **kwargs)
                self.assertEqual(report['verdict'], 'FAIL')
                self.assertIn('candidate_identity_mismatch', report['failure_reasons'])
                self.assertIn(field, report['reason_details']['candidate_identity_mismatch']['examples'][0]['fields'])
        tampered = copy.deepcopy(receipt)
        tampered['candidate']['identity_sha256'] = '0' * 64
        report = evaluate(tampered)
        self.assertEqual(report['failure_reasons'], ['candidate_identity_mismatch'])

    def test_run_copy_must_match_the_bound_receipt(self):
        receipt = pass_receipt()
        for label, run_copy in (('missing', None), ('other_run', make_run(uuid.uuid4().hex)),
                                ('not_json_object', [])):
            with self.subTest(label=label):
                report = evaluate(copy.deepcopy(receipt), run_copy=run_copy)
                self.assertEqual(report['failure_reasons'], ['run_copy_mismatch'])

    def test_evaluate_bindings_are_mandatory(self):
        parameters = inspect.signature(analysis.evaluate).parameters
        for name in ('soak_sha256', 'expected_name', 'expected_profile', 'expected_identity_sha256', 'run_copy',
                     'validate_run'):
            self.assertIs(parameters[name].default, inspect.Parameter.empty, name)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY, name)
        report = evaluate(pass_receipt(), soak_sha256=None)
        self.assertIn('harness_source_mismatch', report['failure_reasons'])

    def test_resolve_candidate_forms(self):
        identity = BENCH.validate_receipt(make_run('1' * 32), 'main-retain')
        sha = soak.canonical_sha256(identity)
        self.assertEqual(sha, BENCH.canonical_hash(identity))  # same digest as bench's manifests
        self.assertEqual(soak.resolve_candidate('main-retain', sha),
                         {'profile': 'main-retain', 'identity_sha256': sha, 'source': 'sha256'})
        with tempfile.TemporaryDirectory() as temp:
            manifest = pathlib.Path(temp) / 'final-v5-manifest.json'
            manifest.write_text(json.dumps({'arms': {'main-retain': identity, 'bf128': {'x': 1}}}))
            resolved = soak.resolve_candidate('main-retain', str(manifest))
            self.assertEqual((resolved['identity_sha256'], resolved['source'], resolved['manifest_name']),
                             (sha, 'manifest', 'final-v5-manifest.json'))
            self.assertEqual(resolved['manifest_sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())
            for profile, expected in (('bf128', str(manifest)), ('baseline64', str(manifest)),
                                      ('Bad_Profile', sha), ('main-retain', ''), ('main-retain', 'A' * 64)):
                with self.subTest(profile=profile, expected=expected[-20:]):
                    with self.assertRaises((ValueError, OSError)):
                        soak.resolve_candidate(profile, expected)

    def test_soak_cli_requires_the_frozen_candidate(self):
        with mock.patch.object(soak, 'build_live_runner', side_effect=AssertionError('must not start')):
            for argv in (['--name', 'soak-x'], ['--name', 'soak-x', '--profile', 'main-retain'],
                         ['--name', 'soak-x', '--expected-identity', 'a' * 64],
                         ['--name', 'soak-x', '--profile', 'Bad_Profile', '--expected-identity', 'a' * 64],
                         ['--name', 'soak-x', '--profile', 'main-retain', '--expected-identity', '/nonexistent/m.json']):
                with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    soak.main(argv)


class GpuLockTests(unittest.TestCase):
    def test_busy_lock_refuses_before_any_request(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            with foreign_lock(world.lock_path):
                runner, code, receipt = run_soak(world)
            self.assertEqual((code, receipt), (2, None))
            self.assertEqual(runner.events[-1]['reason'], 'gpu_lock_busy')
            self.assertEqual(world.uid, 0)
            self.assertEqual(sorted(p.name for p in (world.root / 'reports').iterdir()), [])

    def test_lock_is_held_through_drain_and_released(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            runner = make_runner(world, seconds=300)
            probes = {'loop': [], 'drain': []}

            def probe():
                phase = 'drain' if runner.termination is not None else 'loop'
                if runner.started is None or len(probes[phase]) >= 3:
                    return
                fd = os.open(world.lock_path, os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    probes[phase].append('free')
                except BlockingIOError:
                    probes[phase].append('busy')
                finally:
                    os.close(fd)
            world.engine_delay = 8.0  # makes the final drain wait for at least one state period
            world.hooks.insert(0, probe)
            runner, code, receipt = finish_run(world, runner)
            self.assertEqual(probes['loop'], ['busy'] * 3)
            self.assertTrue(probes['drain'] and set(probes['drain']) == {'busy'}, probes)
            self.assertEqual(receipt['gpu_lock'], {'path': str(world.lock_path), 'mode': 'acquired'})
            fd, mode = soak.acquire_gpu_lock(world.lock_path)  # released after the final receipt
            os.close(fd)
            self.assertEqual(mode, 'acquired')

    def test_inherited_wrapper_descriptor_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            world = FakeWorld(temp)
            world.lock_path.parent.mkdir()
            wrapper = os.open(world.lock_path, os.O_RDWR | os.O_CREAT, 0o600)  # what flock(1) passes on
            try:
                fcntl.flock(wrapper, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(soak.acquire_gpu_lock(world.lock_path), (wrapper, 'inherited'))
                runner, code, receipt = run_soak(world, seconds=30)
                self.assertEqual(receipt['gpu_lock']['mode'], 'inherited')
                os.fstat(wrapper)  # the wrapper's descriptor is not closed by the runner
            finally:
                os.close(wrapper)
            with foreign_lock(world.lock_path), self.assertRaises(soak.GpuLockBusy):
                soak.acquire_gpu_lock(world.lock_path)


class ReceiptMutationTests(unittest.TestCase):
    def assertFails(self, receipt, *codes, verdict='FAIL'):
        report = evaluate(receipt)
        self.assertEqual(report['verdict'], verdict, report['gate_reasons'])
        self.assertFalse(report['two_hour_execution_gate_pass'])
        for code in codes:
            self.assertIn(code, report['gate_reasons'])
        return report

    def test_empty_workload(self):
        receipt = pass_receipt()
        receipt['rows'] = []
        receipt['iterations'] = 0
        self.assertFails(receipt, 'no_workload_rows', 'foreign_generation_traffic', 'insufficient_coverage',
                         'memory_insufficient_samples')

    def test_original_format_receipt_cannot_pass(self):
        old = {'run': {}, 'required_seconds': 7200, 'elapsed_s': 7300, 'rows': [], 'complete': True, 'error_count': 0}
        self.assertFails(old, 'receipt_schema_invalid')
        self.assertFails(None, 'receipt_schema_invalid')

    def test_missing_resources(self):
        receipt = pass_receipt()
        rid = first(receipt['rows'], 'fresh')['requests'][0]['request_id']
        del receipt['telemetry']['requests'][rid]['peaks']['service_memory_shmem_bytes']
        self.assertFails(receipt, 'request_resources_missing')
        receipt = pass_receipt()
        receipt['telemetry']['state'][5].pop('service_memory_anon_bytes')
        self.assertFails(receipt, 'state_resources_missing')
        receipt = pass_receipt()
        receipt['telemetry']['state'][7]['gpu_total_used_bytes'] = None
        self.assertFails(receipt, 'state_resources_missing')
        receipt = pass_receipt()
        key = next(iter(receipt['telemetry']['engine']))
        receipt['telemetry']['engine'][key].pop('torch_allocated_bytes')
        self.assertFails(receipt, 'engine_resources_missing')

    def test_stale_serving_run(self):
        receipt = pass_receipt()
        for row in receipt['telemetry']['requests'].values():
            row['run_id'] = 'f' * 32
        self.assertFails(receipt, 'serving_run_id_mismatch')
        receipt = pass_receipt()
        for key in list(receipt['telemetry']['engine']):
            row = receipt['telemetry']['engine'].pop(key)
            receipt['telemetry']['engine'][f"{'f' * 32}:{row['uid']}"] = dict(row, run_id='f' * 32)
        self.assertFails(receipt, 'engine_row_missing', 'engine_run_id_mismatch')

    def test_run_receipt_must_validate_and_match_binding(self):
        receipt = pass_receipt()
        receipt['run']['source_sha256'] = '0' * 64
        self.assertFails(receipt, 'run_receipt_invalid', 'identity_hash_mismatch')
        receipt = pass_receipt()
        receipt['identity']['identity_sha256'] = '0' * 64
        self.assertFails(receipt, 'identity_hash_mismatch')

    def test_identity_check_failure_and_gap(self):
        receipt = pass_receipt()
        receipt['identity']['checks'][3].update(ok=False, codes=['source_changed_on_disk'])
        self.assertFails(receipt, 'identity_check_failed')
        receipt = pass_receipt()
        receipt['identity']['checks'] = [c for c in receipt['identity']['checks'] if not 2000 < c['t_s'] < 2300]
        self.assertFails(receipt, 'identity_check_gap')
        receipt = pass_receipt()
        receipt['identity']['checks'] = [c for c in receipt['identity']['checks'] if c['kind'] != 'source' or c['t_s'] < 7000]
        self.assertFails(receipt, 'identity_source_check_missing')

    def test_lost_joins(self):
        receipt = pass_receipt()
        rid = first(receipt['rows'], 'prefix')['requests'][0]['request_id']
        uid = receipt['telemetry']['requests'][rid]['engine_uid']
        del receipt['telemetry']['engine'][f"{receipt['run']['run_id']}:{uid}"]
        self.assertFails(receipt, 'engine_row_missing')
        receipt = pass_receipt()
        rid = first(receipt['rows'], 'tool')['requests'][1]['request_id']
        del receipt['telemetry']['requests'][rid]
        self.assertFails(receipt, 'request_row_missing', 'unjoined_engine_rows')
        receipt = pass_receipt()
        rid = first(receipt['rows'], 'arithmetic')['requests'][0]['request_id']
        receipt['telemetry']['requests'][rid]['engine_uid'] = None
        self.assertFails(receipt, 'engine_uid_missing')
        receipt = pass_receipt()
        first(receipt['rows'], 'fresh')['requests'][0]['request_id'] = None
        self.assertFails(receipt, 'request_id_missing')

    def test_unexecuted_or_unchecked_cases(self):
        for case in ('tool', 'cancel', 'prefix', 'arithmetic'):
            receipt = pass_receipt()
            receipt['rows'] = [row for row in receipt['rows'] if row['case'] != case]
            self.assertFails(receipt, 'scheduled_case_missing')
        receipt = pass_receipt()
        first(receipt['rows'], 'tool')['facts']['call_arguments_match'] = False
        self.assertFails(receipt, 'row_facts_inconsistent')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'cancel')
        row['facts'].pop('content_seen_before_disconnect')
        self.assertFails(receipt, 'row_facts_inconsistent')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'prefix')
        row.update(**{'pass': False, 'failure': 'prefix_check_failed'})
        self.assertFails(receipt, 'case_failed')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'cancel')
        receipt['telemetry']['requests'][row['requests'][0]['request_id']]['outcome'] = 'completed'
        self.assertFails(receipt, 'cancel_outcome_invalid')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'cancel')
        uid = receipt['telemetry']['requests'][row['requests'][0]['request_id']]['engine_uid']
        receipt['telemetry']['engine'][f"{receipt['run']['run_id']}:{uid}"]['cancelled'] = False
        self.assertFails(receipt, 'cancel_engine_not_cancelled')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'prefix')
        fresh_rid = next(r for r in receipt['rows'] if r['case'] == 'fresh' and r['iteration'] == row['iteration'])['requests'][0]['request_id']
        receipt['telemetry']['requests'][row['requests'][0]['request_id']]['prompt_tokens'] = \
            receipt['telemetry']['requests'][fresh_rid]['prompt_tokens']
        self.assertFails(receipt, 'prefix_server_tokens_invalid')
        receipt = pass_receipt()
        row = next(r for r in receipt['rows'] if r['case'] == 'fresh' and r['length'] == 120000)
        receipt['telemetry']['requests'][row['requests'][0]['request_id']]['prompt_tokens'] = 60000
        self.assertFails(receipt, 'fresh_server_tokens_invalid')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'fresh')
        row['length'] = 4096
        self.assertFails(receipt, 'row_schedule_mismatch')
        receipt = pass_receipt()
        row = first(receipt['rows'], 'tool')
        row['requests'] = row['requests'][:1]
        self.assertFails(receipt, 'request_roles_invalid')

    def test_drops_and_writer_failures(self):
        for key, code in (('metrics_dropped', 'telemetry_metrics_dropped'),
                          ('metrics_failed_writes', 'telemetry_write_failures'),
                          ('resource_sample_failures', 'telemetry_resource_sample_failures')):
            receipt = pass_receipt()
            receipt['telemetry']['state'][100][key] = 1
            self.assertFails(receipt, code)
            receipt = pass_receipt()
            receipt['telemetry']['state'][100].pop(key)
            self.assertFails(receipt, 'state_counters_missing')

    def test_rotation_loss_and_reader_failures(self):
        for field, code in (('loss_events', 'telemetry_rotation_loss'), ('truncations', 'telemetry_truncated'),
                            ('malformed_lines', 'telemetry_malformed_lines'), ('read_errors', 'telemetry_read_errors')):
            receipt = pass_receipt()
            receipt['telemetry']['readers']['engine'][field] = 1
            self.assertFails(receipt, code)
        receipt = pass_receipt()
        del receipt['telemetry']['readers']['state']
        self.assertFails(receipt, 'telemetry_reader_missing')
        receipt = pass_receipt()
        receipt['telemetry']['monitor']['failures'] = ['OSError']
        self.assertFails(receipt, 'telemetry_monitor_failed')
        receipt = pass_receipt()
        receipt['telemetry']['monitor']['background']['alive_before_stop'] = False
        self.assertFails(receipt, 'telemetry_monitor_not_running')
        receipt = pass_receipt()
        receipt['telemetry']['duplicates']['engine'] = 1
        self.assertFails(receipt, 'telemetry_duplicate_rows')

    def test_state_gaps_and_coverage(self):
        receipt = pass_receipt()
        started = soak.parse_utc(receipt['started_utc'])
        receipt['telemetry']['state'] = [row for row in receipt['telemetry']['state']
                                         if not 3000 < (soak.parse_utc(row['ts']) - started).total_seconds() < 3100]
        self.assertFails(receipt, 'state_gap')
        receipt = pass_receipt()
        receipt['telemetry']['state'] = receipt['telemetry']['state'][:-10]
        self.assertFails(receipt, 'state_coverage_end')
        receipt = pass_receipt()
        receipt['telemetry']['state'] = receipt['telemetry']['state'][5:]
        self.assertFails(receipt, 'state_coverage_start')
        receipt = pass_receipt()
        receipt['telemetry']['state'] = []
        self.assertFails(receipt, 'state_rows_missing')

    def test_session_instance_stall_and_profile(self):
        receipt = pass_receipt()
        receipt['telemetry']['state'][50]['session_id'] = 'other-session'
        self.assertFails(receipt, 'service_session_changed')
        receipt = pass_receipt()
        receipt['telemetry']['state'][50]['instance_id'] = 'other-instance'
        self.assertFails(receipt, 'service_instance_changed')
        receipt = pass_receipt()
        receipt['telemetry']['events'] = [{'ts': receipt['finished_utc'], 'event': 'engine_stall_shutdown'}]
        self.assertFails(receipt, 'service_stall_event')
        receipt = pass_receipt()
        receipt['telemetry']['state'][50]['profile'] = 'bf128'
        self.assertFails(receipt, 'state_profile_mismatch')
        receipt = pass_receipt()
        receipt['telemetry']['state'][50]['model_ready'] = False
        self.assertFails(receipt, 'model_not_ready')

    def test_foreign_traffic(self):
        receipt = pass_receipt()
        receipt['telemetry']['requests']['a' * 32] = dict(next(iter(receipt['telemetry']['requests'].values())),
                                                          request_id='a' * 32)
        self.assertFails(receipt, 'foreign_generation_traffic')

    def test_duration_and_status(self):
        receipt = pass_receipt()
        receipt.update(elapsed_s=7199.0, complete=False)
        self.assertFails(receipt, 'elapsed_below_required', 'receipt_incomplete')
        receipt = pass_receipt()
        receipt['required_seconds'] = 3600
        self.assertFails(receipt, 'required_seconds_below_7200')
        receipt = pass_receipt()
        receipt['status'] = 'running'
        self.assertFails(receipt, 'receipt_not_final')
        receipt = pass_receipt()
        receipt['termination_reason'] = 'interrupted'
        self.assertFails(receipt, 'soak_terminated_early')
        receipt = pass_receipt()
        receipt['finished_utc'] = receipt['started_utc']
        self.assertFails(receipt, 'clock_inconsistent')

    def test_workload_harness_and_name_binding(self):
        receipt = pass_receipt()
        receipt['workload']['lengths'] = [64, 4096]
        self.assertFails(receipt, 'workload_definition_mismatch')
        receipt = pass_receipt()
        receipt['harness']['soak.py'] = '0' * 64
        self.assertFails(receipt, 'harness_source_mismatch')
        report = evaluate(pass_receipt(), expected_name='soak-other')
        self.assertIn('receipt_name_mismatch', report['failure_reasons'])

    def test_type_corruption_never_crashes_or_passes(self):
        paths = [('status',), ('rows',), ('telemetry',), ('identity',), ('run',), ('workload',), ('harness',),
                 ('started_utc',), ('finished_utc',), ('elapsed_s',), ('required_seconds',), ('iterations',),
                 ('telemetry', 'requests'), ('telemetry', 'engine'), ('telemetry', 'state'),
                 ('telemetry', 'readers'), ('telemetry', 'monitor'), ('telemetry', 'events'),
                 ('telemetry', 'duplicates'), ('identity', 'checks'), ('identity', 'unit_initial'),
                 ('run', 'configuration'), ('rows', 0), ('rows', 0, 'requests'), ('rows', 0, 'facts'),
                 ('rows', 0, 'requests', 0), ('telemetry', 'state', 0), ('identity', 'checks', 0),
                 ('candidate',), ('candidate', 'identity_sha256'), ('gpu_lock',), ('gpu_lock', 'mode')]
        base = pass_receipt()
        for path in paths:
            for value in (None, [], {}, 'x', 5, float('inf')):
                receipt = copy.deepcopy(base)
                target = receipt
                for key in path[:-1]:
                    target = target[key]
                original = target[path[-1]]
                if type(original) is type(value) and original == value:
                    continue  # identical to the passing receipt (e.g. no events)
                target[path[-1]] = value
                report = evaluate(receipt)
                self.assertFalse(report['two_hour_execution_gate_pass'], (path, value))
                self.assertTrue(report['gate_reasons'], (path, value))

    def test_vram_floor_is_gated_and_minutes_below_are_counted(self):
        self.assertEqual(analysis.VRAM_FLOOR_BYTES, BENCH.VRAM_FLOOR_BYTES)
        self.assertEqual(analysis.VRAM_FLOOR_BYTES, int(0.8 * (1 << 30)))
        vram = evaluate(pass_receipt())['summary']['vram']
        self.assertEqual((vram['samples_below_floor'], vram['minutes_below_floor']), (0, 0))
        self.assertGreaterEqual(vram['p1_gpu_free_bytes'], vram['min_gpu_free_bytes'])
        receipt = pass_receipt()
        rows = sorted(receipt['telemetry']['state'], key=lambda row: row['ts'])
        # Six consecutive samples just below the floor (f1 passed the old 0.5 GiB floor at these values).
        for row in rows[3:9]:
            row['gpu_free_bytes'] = analysis.VRAM_FLOOR_BYTES - 1
        report = self.assertFails(receipt, 'vram_floor_breached')
        self.assertEqual(report['failure_reasons'], ['vram_floor_breached'])
        seconds = sum(min((analysis._parse_time(soak, b['ts']) - analysis._parse_time(soak, a['ts'])).total_seconds(),
                          analysis.STATE_GAP_LIMIT_S) for a, b in zip(rows[3:9], rows[4:10]))
        self.assertEqual(report['summary']['vram']['samples_below_floor'], 6)
        self.assertEqual(report['summary']['vram']['minutes_below_floor'], round(seconds / 60, 2))
        self.assertGreater(seconds, 0)
        receipt = pass_receipt()
        for row in receipt['telemetry']['state']:
            row['gpu_free_bytes'] = analysis.VRAM_FLOOR_BYTES
        self.assertEqual(evaluate(receipt)['verdict'], 'PASS')
        for row in receipt['telemetry']['state']:
            row.pop('gpu_free_bytes')
        self.assertFails(receipt, 'vram_floor_unverified', 'state_resources_missing')

    def test_memory_needs_idle_samples(self):
        receipt = pass_receipt()
        for row in receipt['telemetry']['state']:
            row['active_inference'] = 1
        report = self.assertFails(receipt, 'memory_insufficient_samples', verdict='INCONCLUSIVE')
        self.assertEqual(report['memory']['status'], 'INCONCLUSIVE')

    def assertMutationFails(self, mutate, code):
        receipt = pass_receipt()
        mutate(receipt)
        return self.assertFails(receipt, code)

    def test_each_join_reason(self):
        def rejected(receipt):
            row = first(receipt['rows'], 'cancel')
            entry = next(e for e in row['requests'] if e['role'] == 'cancel_followup_rejected')
            return receipt['telemetry']['requests'][entry['request_id']]

        def ambiguous(receipt):
            server_of(receipt, 'arithmetic', iteration=0)['engine_uid'] = \
                server_of(receipt, 'fresh', iteration=0)['engine_uid']

        def duplicate(receipt):
            first(receipt['rows'], 'arithmetic')['requests'][0]['request_id'] = \
                first(receipt['rows'], 'fresh')['requests'][0]['request_id']
        for code, mutate in (
                ('request_status_mismatch', lambda r: server_of(r, 'arithmetic').update(status=201)),
                ('request_row_invalid', lambda r: server_of(r, 'arithmetic').update(kind='models')),
                ('session_id_missing', lambda r: server_of(r, 'fresh').update(session_id=None)),
                ('request_outcome_invalid', lambda r: server_of(r, 'arithmetic').update(counts_final=False)),
                ('request_outcome_invalid', lambda r: server_of(r, 'tool', 1).update(outcome='cancelled')),
                ('request_outcome_invalid', lambda r: rejected(r).update(outcome='completed')),
                ('request_outcome_invalid', lambda r: rejected(r).update(status=400)),
                ('engine_join_ambiguous', ambiguous),
                ('engine_cancelled_unexpectedly', lambda r: engine_of(r, server_of(r, 'fresh')).update(cancelled=True)),
                ('engine_cancelled_unexpectedly', lambda r: engine_of(r, server_of(r, 'tool', 1)).pop('cancelled')),
                ('tool_server_finish_invalid', lambda r: server_of(r, 'tool', 0).update(finish_reason='stop')),
                ('request_id_duplicate', duplicate),
                ('telemetry_invalid_rows', lambda r: r['telemetry']['invalid_rows'].update(state=1)),
                ('telemetry_invalid_rows', lambda r: r['telemetry'].pop('invalid_rows')),
                ('engine_run_id_mismatch', lambda r: engine_of(r, server_of(r, 'fresh')).update(uid=-1)),
                ('unjoined_engine_rows', lambda r: r['telemetry']['engine'].update(
                    {f"{r['run']['run_id']}:999999": dict(next(iter(r['telemetry']['engine'].values())),
                                                         uid=999999)}))):
            with self.subTest(code=code):
                self.assertMutationFails(mutate, code)

    def test_each_state_and_receipt_reason(self):
        def swap_ts(receipt):
            state = receipt['telemetry']['state']
            state[100]['ts'], state[101]['ts'] = state[101]['ts'], state[100]['ts']

        def unexpected(receipt):
            rows = receipt['rows']
            extra = copy.deepcopy(first(rows, 'tool'))
            extra['iteration'] = 1
            rows.insert(max(i for i, row in enumerate(rows) if row['iteration'] == 1) + 1, extra)
        for code, mutate in (
                ('state_time_regression', swap_ts),
                ('state_context_mismatch', lambda r: r['telemetry']['state'][10].update(context_limit=65536)),
                ('instance_id_missing', lambda r: r['telemetry']['state'][10].update(instance_id=None)),
                ('session_id_missing', lambda r: r['telemetry']['state'][10].update(session_id='')),
                ('telemetry_resource_sample_stale', lambda r: r['telemetry']['state'][10].update(
                    resource_sample_stale=True)),
                ('state_timestamp_invalid', lambda r: r['telemetry']['state'][10].update(ts='bad')),
                ('state_timestamp_invalid', lambda r: r['telemetry']['state'][10].update(ts='2026-10-13T12:00:00')),
                ('state_row_invalid', lambda r: r['telemetry']['state'].__setitem__(10, 'x')),
                ('identity_checks_missing', lambda r: r['identity']['checks'].append('x')),
                ('identity_checks_missing', lambda r: r['identity']['checks'][3].update(t_s=None)),
                ('identity_checks_missing', lambda r: r['identity'].update(checks=[])),
                ('unit_identity_missing', lambda r: r['identity'].update(unit_initial=None)),
                ('unexpected_case', unexpected),
                ('row_schema_invalid', lambda r: r['rows'][5].update({'pass': 'yes'})),
                ('row_schema_invalid', lambda r: r['rows'][5].update(case='other')),
                ('telemetry_missing', lambda r: r.update(telemetry=None)),
                ('receipt_time_invalid', lambda r: r.update(started_utc='bad')),
                ('iteration_sequence_invalid', lambda r: r.update(iterations=r['iterations'] + 1)),
                ('iteration_sequence_invalid', lambda r: r['rows'].reverse())):
            with self.subTest(code=code):
                self.assertMutationFails(mutate, code)

    def test_server_token_and_cancel_bounds(self):
        def fresh(value):
            return lambda r: fresh_server(r).update(completion_tokens=value)

        def prefix(value):
            return lambda r: server_of(r, 'prefix').update(completion_tokens=value)
        for code, mutate in (('fresh_server_tokens_invalid', fresh(soak.FRESH_MAX_TOKENS + 1)),
                             ('fresh_server_tokens_invalid', fresh(0)),
                             ('prefix_server_tokens_invalid', prefix(soak.PREFIX_MAX_TOKENS + 1)),
                             ('prefix_server_tokens_invalid', prefix(0))):
            with self.subTest(code=code, mutation=mutate):
                self.assertMutationFails(mutate, code)
        for value in (soak.FRESH_MAX_TOKENS, 1):
            receipt = pass_receipt()
            fresh_server(receipt)['completion_tokens'] = value
            self.assertEqual(evaluate(receipt)['verdict'], 'PASS')

        def too_many_attempts(receipt):  # 51 refused follow-ups + 1 answered: above CANCEL_ATTEMPTS
            row = first(receipt['rows'], 'cancel')
            rejected = [{'role': 'cancel_followup_rejected', 'request_id': uuid.uuid4().hex, 'status': 429}
                        for _ in range(soak.CANCEL_ATTEMPTS + 1)]
            row['requests'] = row['requests'][:1] + rejected + row['requests'][-1:]
            row['measurements']['followup_attempts'] = len(row['requests']) - 1
        for mutate in (too_many_attempts,
                       lambda r: first(r['rows'], 'cancel')['measurements'].update(followup_attempts=99),
                       lambda r: first(r['rows'], 'cancel')['measurements'].update(recovery_s=None),
                       lambda r: first(r['rows'], 'cancel').pop('measurements')):
            with self.subTest(mutation=mutate):
                self.assertMutationFails(mutate, 'request_roles_invalid')

    def test_server_sampling_must_match_the_workload(self):
        def set_sampling(case, entry=0, **changes):
            return lambda r: server_of(r, case, entry)['sampling'].update(changes)

        def rejected_with(sampling):
            def mutate(receipt):
                row = first(receipt['rows'], 'cancel')
                entry = next(e for e in row['requests'] if e['role'] == 'cancel_followup_rejected')
                receipt['telemetry']['requests'][entry['request_id']]['sampling'] = sampling
            return mutate
        quality32 = dict(analysis.QUALITY_SAMPLING, max_tokens=soak.SMOKE_MAX_TOKENS)
        for label, mutate in (
                ('fresh_max_tokens', set_sampling('fresh', max_tokens=soak.PREFIX_MAX_TOKENS)),
                ('fresh_seed', lambda r: fresh_server(r)['sampling'].update(seed=fresh_server(r)['sampling']['seed'] + 1)),
                ('fresh_temperature', set_sampling('fresh', temperature=1.0)),
                ('fresh_thinking', set_sampling('fresh', enable_thinking=True)),
                ('fresh_thinking_number', set_sampling('fresh', enable_thinking=0)),
                ('fresh_extra_key', set_sampling('fresh', top_k=20)),
                ('fresh_missing', lambda r: fresh_server(r).pop('sampling')),
                ('prefix_temperature', set_sampling('prefix', temperature=1.0)),
                ('arithmetic_max_tokens', set_sampling('arithmetic', max_tokens=512)),
                ('tool_call_seed', set_sampling('tool', 0, seed=418)),
                ('tool_final_temperature', set_sampling('tool', 1, temperature=.6)),
                ('cancel_stream_max_tokens', set_sampling('cancel', 0, max_tokens=64)),
                ('cancel_followup_max_tokens', lambda r: server_of(r, 'cancel', -1)['sampling'].update(max_tokens=512)),
                ('rejected_recorded_wrong', rejected_with(dict(quality32, max_tokens=512)))):
            with self.subTest(label=label):
                self.assertMutationFails(mutate, 'workload_request_mismatch')
        receipt = pass_receipt()
        rejected_with(quality32)(receipt)  # a refusal after body parsing records the planned sampling
        self.assertEqual(evaluate(receipt)['verdict'], 'PASS')
        self.assertEqual(analysis.expected_sampling(soak, 'fresh', 7), {
            'max_tokens': 256, 'seed': 348, 'temperature': .6, 'top_p': .95, 'enable_thinking': False})

    def test_gpu_lock_evidence_required(self):
        for label, mutate in (('missing', lambda r: r.pop('gpu_lock')), ('empty', lambda r: r.update(gpu_lock={})),
                              ('mode', lambda r: r['gpu_lock'].update(mode='none')),
                              ('path', lambda r: r['gpu_lock'].update(path='/tmp/other.lock'))):
            with self.subTest(label=label):
                self.assertMutationFails(mutate, 'gpu_lock_not_held')

    @staticmethod
    def _set_fresh_series(receipt, length, metric, delta):
        """Early/warm-up points keep their value; the last MEMORY_WINDOW_POINTS get +delta."""
        rows = [row for row in receipt['rows'] if row['case'] == 'fresh' and row['length'] == length]
        kind, name = metric.split('.', 1)
        targets = []
        for row in rows:
            server = receipt['telemetry']['requests'][row['requests'][0]['request_id']]
            targets.append(server['peaks'] if kind == 'request' else engine_of(receipt, server))
        base = targets[0][name]
        for index, target in enumerate(targets):
            target[name] = base + (delta if index >= len(targets) - soak.MEMORY_WINDOW_POINTS else 0)

    def test_equal_length_memory_thresholds_are_exact(self):
        gated = {**{'request.' + k: v for k, v in analysis.REQUEST_MEMORY_FLAGS.items()},
                 **{'engine.' + k: v for k, v in analysis.ENGINE_MEMORY_FLAGS.items()}}
        self.assertEqual(gated, {'request.service_memory_anon_bytes': 256 * MIB,
                                 'request.service_memory_shmem_bytes': 64 * MIB,
                                 'engine.torch_allocated_bytes': 64 * MIB, 'engine.torch_reserved_bytes': 256 * MIB})
        for metric, threshold in gated.items():
            with self.subTest(metric=metric):
                receipt = pass_receipt()
                self._set_fresh_series(receipt, 64, metric, threshold)  # at the threshold: not flagged
                report = evaluate(receipt)
                self.assertEqual(report['verdict'], 'PASS', report['gate_reasons'])
                self.assertEqual(report['memory']['per_length']['64'][metric]['late_minus_early'], threshold)
                receipt = pass_receipt()
                self._set_fresh_series(receipt, 64, metric, threshold + 1)
                report = self.assertFails(receipt, 'memory_growth_above_threshold')
                self.assertEqual(report['failure_reasons'], ['memory_growth_above_threshold'])
                self.assertEqual([(f['length'], f['metric']) for f in report['memory']['flags']], [(64, metric)])
        for metric in analysis.MEMORY_REPORT_ONLY:  # reported, never gated
            receipt = pass_receipt()
            self._set_fresh_series(receipt, 64, 'request.' + metric, 10 * 1024 * MIB)
            self.assertEqual(evaluate(receipt)['verdict'], 'PASS')

    def test_idle_memory_threshold_uses_the_late_window(self):
        self.assertEqual(analysis.IDLE_MEMORY_FLAGS, {'service_memory_anon_bytes': 256 * MIB,
                                                      'service_memory_shmem_bytes': 64 * MIB})
        for metric, threshold in analysis.IDLE_MEMORY_FLAGS.items():
            for delta, flagged in ((threshold, False), (threshold + 1, True)):
                with self.subTest(metric=metric, delta=delta):
                    receipt = pass_receipt()
                    started = soak.parse_utc(receipt['started_utc'])
                    for row in receipt['telemetry']['state']:
                        offset = (soak.parse_utc(row['ts']) - started).total_seconds()
                        if receipt['elapsed_s'] - analysis.IDLE_WINDOW_S <= offset <= receipt['elapsed_s']:
                            row[metric] += delta  # growth only inside the last 1200 s
                    report = evaluate(receipt)
                    idle = [(f['metric'], f['late_minus_early']) for f in report['memory']['flags'] if f.get('idle')]
                    self.assertEqual(idle, [(metric, delta)] if flagged else [])
                    self.assertEqual(report['verdict'], 'FAIL' if flagged else 'PASS', report['gate_reasons'])


class AnalysisCliTests(unittest.TestCase):
    @staticmethod
    def _bench(reports):
        fake = types.ModuleType('bench')
        fake.validate_receipt = BENCH.validate_receipt
        fake.ROOT = reports.parent
        return fake

    def test_cli_writes_once_and_exit_codes(self):
        receipt = pass_receipt()
        sha = CANDIDATE['identity_sha256']
        with tempfile.TemporaryDirectory() as temp:
            reports = pathlib.Path(temp)
            (reports / 'soak-test.json').write_text(json.dumps(receipt))
            (reports / 'soak-test-run.json').write_text(json.dumps(_PASS['run_copy']))
            bind = ['--profile', 'main-retain', '--expected-identity', sha, '--reports-dir', temp]
            out = io.StringIO()
            with mock.patch.dict(sys.modules, {'bench': self._bench(reports)}), contextlib.redirect_stdout(out):
                self.assertEqual(analysis.main(['--name', 'soak-test', *bind]), 0)
                written = json.loads((reports / 'soak-test-analysis.json').read_text())
                self.assertTrue(written['two_hour_execution_gate_pass'])
                self.assertEqual(written['source_sha256'],
                                 hashlib.sha256((reports / 'soak-test.json').read_bytes()).hexdigest())
                self.assertEqual(written['run_copy_sha256'],
                                 hashlib.sha256((reports / 'soak-test-run.json').read_bytes()).hexdigest())
                self.assertEqual(written['candidate'], {'profile': 'main-retain', 'identity_sha256': sha,
                                                        'source': 'sha256'})
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    analysis.main(['--name', 'soak-test', *bind])
                broken = dict(receipt, status='running')
                (reports / 'soak-bad.json').write_text(json.dumps(broken))
                self.assertEqual(analysis.main(['--name', 'soak-bad', *bind]), 1)
                self.assertEqual(analysis.main(['--name', 'soak-test', *bind, '--tag', 'r2']), 0)
            printed = json.loads(out.getvalue().splitlines()[0])
            self.assertEqual((printed['profile'], printed['run_id'], printed['identity_sha256'],
                              printed['expected_identity_sha256']),
                             ('main-retain', receipt['run']['run_id'], sha, sha))
            self.assertFalse([p for p in reports.iterdir() if p.name.startswith('.')])

    def test_cli_requires_and_checks_the_candidate(self):
        receipt = pass_receipt()
        with tempfile.TemporaryDirectory() as temp:
            reports = pathlib.Path(temp)
            (reports / 'soak-test.json').write_text(json.dumps(receipt))
            (reports / 'soak-test-run.json').write_text(json.dumps(_PASS['run_copy']))
            identity = BENCH.validate_receipt(_PASS['run_copy'], 'main-retain')
            manifest = reports / 'frozen-manifest.json'
            manifest.write_text(json.dumps({'arms': {'main-retain': identity, 'bf128': identity}}))
            other = candidate_for(source_marker='e')['identity_sha256']
            base = ['--name', 'soak-test', '--reports-dir', temp]
            stderr = io.StringIO()
            with mock.patch.dict(sys.modules, {'bench': self._bench(reports)}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                for argv in ([*base], [*base, '--profile', 'main-retain'], [*base, '--expected-identity', other]):
                    with self.assertRaises(SystemExit):
                        analysis.main(argv)
                self.assertEqual(analysis.main([*base, '--profile', 'main-retain', '--expected-identity', str(manifest),
                                                '--tag', 'm1']), 0)
                for tag, argv in (('p', ['--profile', 'bf128', '--expected-identity', str(manifest)]),
                                  ('b', ['--profile', 'main-retain', '--expected-identity', other])):
                    self.assertEqual(analysis.main([*base, *argv, '--tag', tag]), 1)
                    report = json.loads((reports / f'soak-test-analysis-{tag}.json').read_text())
                    self.assertIn('candidate_identity_mismatch', report['failure_reasons'])
                    # validate_receipt runs against the expected profile, not the receipt's own
                    self.assertEqual('run_receipt_invalid' in report['failure_reasons'], tag == 'p')
                (reports / 'soak-test-run.json').unlink()
                self.assertEqual(analysis.main([*base, '--profile', 'main-retain', '--expected-identity',
                                                CANDIDATE['identity_sha256'], '--tag', 'nocopy']), 1)
                report = json.loads((reports / 'soak-test-analysis-nocopy.json').read_text())
                self.assertEqual(report['failure_reasons'], ['run_copy_mismatch'])


class GateReasonCoverageTests(unittest.TestCase):
    def test_every_gate_reason_is_named_in_a_test(self):
        """Static floor: each fail/inconclusive code of soak-analysis.py appears in some test method."""
        tree = ast.parse((HERE / 'soak-analysis.py').read_text(encoding='utf-8'))
        codes = {node.args[0].value for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in ('fail', 'inconclusive') and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == 'findings' and node.args and isinstance(node.args[0], ast.Constant)}
        codes |= {code for _, code in analysis.STATE_COUNTERS + analysis.READER_COUNTERS}
        named = set()
        for cls in ast.parse((HERE / 'test-soak-cpu.py').read_text(encoding='utf-8')).body:
            for function in getattr(cls, 'body', []) if isinstance(cls, ast.ClassDef) else []:
                if isinstance(function, ast.FunctionDef) and function.name.startswith('test_') \
                        and function.name != 'test_every_gate_reason_is_named_in_a_test':
                    named |= {node.value for node in ast.walk(function)
                              if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertGreaterEqual(len(codes), 85)
        self.assertEqual(sorted(codes - named), [])


class OriginalDefectProbes(unittest.TestCase):
    """Evidence that the pre-change gate produced a false PASS (expected to reproduce)."""

    def test_original_analysis_passed_an_empty_run(self):
        original = BEFORE / 'soak-analysis.py'
        if not original.is_file():
            self.skipTest('pre-change copy unavailable outside the workspace')
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / 'reports').mkdir()
            (root / 'reports' / 'soak128.json').write_text(json.dumps(
                {'run': {'source_receipt': 'not_recorded_in_earlier_run'}, 'required_seconds': 7200,
                 'elapsed_s': 7200.5, 'rows': [], 'complete': True, 'error_count': 0}))
            fake = types.ModuleType('bench')
            fake.ROOT = root
            with mock.patch.dict(sys.modules, {'bench': fake}), mock.patch.object(sys, 'argv', ['x']), \
                    contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(original), run_name='__main__')
            old = json.loads((root / 'reports' / 'soak128-analysis.json').read_text())
        self.assertTrue(old['two_hour_execution_gate_pass'])  # the false PASS
        self.assertEqual(old['requests'], 0)
        new = evaluate({'run': {}, 'required_seconds': 7200, 'elapsed_s': 7200.5, 'rows': [], 'complete': True})
        self.assertFalse(new['two_hour_execution_gate_pass'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report')
    args, rest = parser.parse_known_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    outcome = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'passed': result.testsRun - len(result.failures) - len(result.errors)
               - len(result.skipped), 'seconds': time.monotonic() - started,
               'failed_tests': [test.id() for test, _ in result.failures + result.errors],
               'skipped_tests': [test.id() for test, _ in result.skipped],
               'pass_report_summary': {key: _PASS.get('report', {}).get(key)
                                       for key in ('verdict', 'gate_reasons')},
               'pass_receipt_counts': None}
    if 'receipt' in _PASS:
        receipt = _PASS['receipt']
        outcome['pass_receipt_counts'] = {
            'elapsed_s': receipt['elapsed_s'], 'iterations': receipt['iterations'], 'rows': len(receipt['rows']),
            'server_request_rows': len(receipt['telemetry']['requests']),
            'engine_rows': len(receipt['telemetry']['engine']), 'state_rows': len(receipt['telemetry']['state']),
            'identity_checks': len(receipt['identity']['checks']),
            'rotations': {name: stats['rotations'] for name, stats in receipt['telemetry']['readers'].items()},
            'coverage': _PASS.get('report', {}).get('summary', {}).get('coverage', {}).get('passed')}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(outcome, indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
