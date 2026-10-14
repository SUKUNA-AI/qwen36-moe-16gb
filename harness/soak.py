"""Two-hour bounded synthetic mixed workload with fail-closed evidence (soak v2).

The workload is the audited soak128 schedule, unchanged: prompt lengths
64/4096/16384/65536/120000 in rotation, fresh 256-token answers, a real-answer
prefix follow-up at 4096 (iteration % 5 == 1), an arithmetic smoke
(iteration % 7 == 0), an auto tool cycle (iteration % 17 == 0), a streamed
cancellation with slot recovery (iteration % 19 == 0), a 5 s pause, >=7200 s.

Added evidence only: a validated run name that never overwrites a receipt,
atomic partial receipts, binding to the expected candidate (--profile and
--expected-identity: sha256 of the canonical identity or a frozen manifest with
arms[profile]), to active-run/source hashes, the systemd invocation and the
serving run_id, the GPU-exclusive flock held from preflight to the final
receipt, background tailing of request/engine/state JSONL with rotation and
loss accounting, and per-request metadata for request<->engine joins. A copy of
the bound active-run receipt is written as NAME-run.json (make-qualification
run_receipts). This script never declares PASS: soak-analysis.py evaluates the
receipt fail-closed. Reports contain numbers and hashes only; no prompts,
answers, tool arguments, keys or headers.

Plan without network or tokenizer:  python soak.py --plan
Run (under the lock that also covered start-experiment):
  flock -n /home/USER/qwen36-api/run/gpu-exclusive.lock \\
    python soak.py --name soak-<tag> --profile <profile> --expected-identity <sha256|manifest.json>
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import threading
import time

SCHEMA = 'qwen36-soak'
SCHEMA_VERSION = 2
REQUIRED_SECONDS = 7200
NAME_PATTERN = re.compile(r'soak[a-z0-9]*(?:-[a-z0-9]+)*')
NAME_MAX = 64
API_URL = 'http://127.0.0.1:1919/v1/chat/completions'
MODEL_PATH = '/path/to/LLM/qwen36-api/model'
DEFAULT_UNIT = 'qwen36-experiment.service'
# bench.ROOT, config/experiment.env and active-run.json belong to the isolated
# experiment stand only; the production unit has a different data root.
UNIT_PATTERN = re.compile(r'qwen36-experiment\.service')
HARNESS_FILES = ('soak.py', 'soak-analysis.py', 'bench.py', 'quality.py', 'request_metrics.py')
PROFILE_PATTERN = re.compile(r'[a-z0-9][a-z0-9-]{0,70}')  # bench.current_run_spec profile names
SHA256_PATTERN = re.compile(r'[0-9a-f]{64}')
# Keys of bench.validate_receipt's identity (= matrix manifest arms[profile]).
IDENTITY_KEYS = frozenset(('source_sha256', 'service_source_files', 'checkpoint', 'checkpoint_revision',
                           'packages', 'configuration', 'native_libraries'))
GPU_LOCK_NAME = 'gpu-exclusive.lock'  # bench.GPU_LOCK basename
GPU_LOCK_MODES = ('acquired', 'inherited')

# ---- Original soak128 workload (values unchanged) ---------------------------
LENGTHS = (64, 4096, 16384, 65536, 120000)
PAUSE_SECONDS = 5
MAX_CONSECUTIVE_ERRORS = 3
FRESH_MAX_TOKENS = 256
PREFIX_MAX_TOKENS = 128
SAMPLING_TEMPERATURE = .6
SEED_OFFSET = 341
DOCUMENT_VARIANT_OFFSET = 1000000
PREFIX_EVERY, PREFIX_PHASE = 5, 1
ARITHMETIC_EVERY = 7
TOOL_EVERY = 17
CANCEL_EVERY = 19
FRESH_SUFFIX = '\nКратко объясни, как проверить корректность этих записей.'
PREFIX_FOLLOWUP = 'Добавь проверку типов.'
ARITHMETIC_PROMPT = 'Сколько будет 7 * 8? Ответь одним числом.'
ARITHMETIC_EXPECTED = '56'
SMOKE_MAX_TOKENS = 32
TOOL_PROMPT = 'Используй fixture_add для сложения 17 и 25, затем сообщи результат.'
TOOLS = [{'type': 'function', 'function': {
    'name': 'fixture_add', 'description': 'Add two integers in a safe synthetic fixture.',
    'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                   'required': ['a', 'b'], 'additionalProperties': False}}}]
TOOL_NAME = 'fixture_add'
TOOL_ARGUMENTS = {'a': 17, 'b': 25}
TOOL_RESULT = '{"result":42}'
TOOL_EXPECTED = '42'
CANCEL_PROMPT = 'Подробно объясни принципы двоичного поиска.'
CANCEL_MAX_TOKENS = 8192
CANCEL_STREAM_TIMEOUT = 60
CANCEL_FOLLOWUP = 'Сколько будет 3+4?'
CANCEL_EXPECTED = '7'
CANCEL_ATTEMPTS = 50
CANCEL_RETRY_SECONDS = .1

# ---- Evidence requirements derived from that workload -----------------------
CASES = ('fresh', 'prefix', 'arithmetic', 'tool', 'cancel')
# Memory comparison per prompt length: 2 warm-up points, then disjoint early
# and late windows of 5 points. 12 fresh requests per length => 60 iterations.
MEMORY_WARMUP_POINTS = 2
MEMORY_WINDOW_POINTS = 5
MIN_FRESH_PER_LENGTH = MEMORY_WARMUP_POINTS + 2 * MEMORY_WINDOW_POINTS
MIN_ITERATIONS = MIN_FRESH_PER_LENGTH * len(LENGTHS)
PROMPT_COVERAGE_RATIO = .95
REQUIRED_FACTS = {
    'fresh': ('status_200', 'content_nonempty', 'completion_tokens_positive', 'prompt_tokens_cover_length'),
    'prefix': ('status_200', 'content_nonempty', 'completion_tokens_positive', 'prompt_tokens_exceed_fresh'),
    'arithmetic': ('status_200', 'contains_expected'),
    'tool': ('call_status_200', 'one_tool_call', 'call_type_function', 'call_id_present', 'call_name_matches',
             'call_arguments_match', 'finish_reason_tool_calls', 'final_status_200', 'final_contains_expected',
             'final_no_tool_calls'),
    'cancel': ('stream_status_200', 'content_seen_before_disconnect', 'followup_status_200',
               'followup_contains_expected', 'followup_within_attempts'),
}
ENGINE_ROLES = ('fresh', 'prefix', 'arithmetic', 'tool_call', 'tool_final', 'cancel_followup')

# ---- Telemetry ---------------------------------------------------------------
LOG_BACKUPS = 5  # request_metrics.NumericLog keeps name.1 .. name.5
IDENTITY_PERIOD_SECONDS = 30.0
SOURCE_PERIOD_SECONDS = 300.0
PARTIAL_INTERVAL_SECONDS = 30.0
FINAL_DRAIN_SECONDS = 30.0
REQUEST_FIELDS = (
    'schema', 'ts', 'request_id', 'run_id', 'session_id', 'kind', 'stream', 'status', 'outcome', 'engine_uid',
    'prompt_tokens', 'completion_tokens', 'cached_prompt_tokens', 'counts_final', 'counts_note', 'finish_reason',
    'ttft_generated_s', 'ttft_content_s', 'ttft_reasoning_s', 'ttft_tool_s', 'decode_ack_tps', 'prefill_tps',
    'duration_s', 'resource_samples', 'resource_sampling_period_s', 'peaks', 'counter_deltas', 'sampling',
    'context_limit', 'model')
ENGINE_FIELDS = (
    'run_id', 'uid', 'cancelled', 'engine_span_s', 'prefill_gpu_ms', 'decode_gpu_ms', 'prefill_forward_tokens',
    'decode_forwards', 'torch_allocated_bytes', 'torch_reserved_bytes', 'request_torch_peak_allocated_bytes',
    'request_torch_peak_reserved_bytes', 'kv_bytes', 'gdn_bytes', 'gpu_expert_bytes', 'host_expert_bytes',
    'host_pinned_expert_bytes', 'expert_slots', 'valid_gpu_expert_slots', 'allocator_bytes_outside_named_components',
    'request_allocator_peak_above_end_bytes', 'retain_chunk_snapshot_enabled', 'fp32_gdn_snapshot_enabled')
STATE_FIELDS = (
    'ts', 'session_id', 'instance_id', 'model_ready', 'profile', 'context_limit', 'active_inference',
    'completed_requests', 'metrics_dropped', 'metrics_failed_writes', 'resource_sample_failures',
    'metrics_queue_depth', 'resource_sample_stale', 'kv_used_pages', 'kv_total_pages', 'mamba_used_slots',
    'mamba_total_slots', 'service_cgroup_memory_bytes', 'service_memory_anon_bytes', 'service_memory_shmem_bytes',
    'service_memory_file_bytes', 'service_memory_file_cache_estimate_bytes', 'service_memory_kernel_bytes',
    'service_memory_peak_bytes', 'gpu_total_used_bytes', 'gpu_free_bytes', 'gpu_total_bytes',
    'wsl_mem_available_bytes', 'wsl_swap_free_bytes')


# ---- Small helpers -----------------------------------------------------------
def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(moment):
    return moment.astimezone(datetime.timezone.utc).isoformat()


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError('timestamp is not a string')
    moment = datetime.datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError('naive timestamp')
    return moment


def _reject_constant(_value):
    raise ValueError('non-finite JSON constant')


def strict_loads(data):
    return json.loads(data, parse_constant=_reject_constant)


class CaseFailure(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class HTTPStatusError(RuntimeError):
    def __init__(self, meta):
        super().__init__(f"HTTP {meta.get('status')}")
        self.meta = meta
        self.code = 'http_status'


class IdentityError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class GpuLockBusy(RuntimeError):
    """Another process holds the GPU-exclusive lock; nothing was started."""


# ---- Workload definition -----------------------------------------------------
def schedule(iteration):
    """Exactly the original per-iteration plan; analysis recomputes it."""
    if type(iteration) is not int or iteration < 0:
        raise ValueError('iteration must be a non-negative int')
    cases = ['fresh']
    if iteration % PREFIX_EVERY == PREFIX_PHASE:
        cases.append('prefix')
    if iteration % ARITHMETIC_EVERY == 0:
        cases.append('arithmetic')
    if iteration % TOOL_EVERY == 0:
        cases.append('tool')
    if iteration % CANCEL_EVERY == 0:
        cases.append('cancel')
    return {'iteration': iteration, 'length': LENGTHS[iteration % len(LENGTHS)],
            'seed': iteration + SEED_OFFSET, 'document_variant': DOCUMENT_VARIANT_OFFSET + iteration,
            'cases': cases}


def workload_spec():
    """Hashable description of the workload; prompt texts appear only as hashes."""
    return {
        'version': 'soak128-original-workload', 'lengths': list(LENGTHS), 'pause_s': PAUSE_SECONDS,
        'max_consecutive_errors': MAX_CONSECUTIVE_ERRORS, 'required_seconds': REQUIRED_SECONDS,
        'fresh': {'max_tokens': FRESH_MAX_TOKENS, 'temperature': SAMPLING_TEMPERATURE, 'seed': f'iteration+{SEED_OFFSET}',
                  'document': f'bench.document(tokenizer,length,{DOCUMENT_VARIANT_OFFSET}+iteration)',
                  'suffix_sha256': text_sha256(FRESH_SUFFIX), 'client': 'bench.generate(stream)'},
        'prefix': {'every': PREFIX_EVERY, 'phase': PREFIX_PHASE, 'max_tokens': PREFIX_MAX_TOKENS,
                   'followup_sha256': text_sha256(PREFIX_FOLLOWUP), 'history': 'fresh messages + real fresh answer'},
        'arithmetic': {'every': ARITHMETIC_EVERY, 'prompt_sha256': text_sha256(ARITHMETIC_PROMPT),
                       'expected': ARITHMETIC_EXPECTED, 'max_tokens': SMOKE_MAX_TOKENS,
                       'client': 'quality.request payload (temperature 0, seed 417), nonstream'},
        'tool': {'every': TOOL_EVERY, 'prompt_sha256': text_sha256(TOOL_PROMPT), 'tools_sha256': canonical_sha256(TOOLS),
                 'arguments': TOOL_ARGUMENTS, 'result_sha256': text_sha256(TOOL_RESULT), 'expected': TOOL_EXPECTED,
                 'tool_choice': 'auto (omitted)'},
        'cancel': {'every': CANCEL_EVERY, 'prompt_sha256': text_sha256(CANCEL_PROMPT), 'max_tokens': CANCEL_MAX_TOKENS,
                   'ignore_eos': True, 'temperature': SAMPLING_TEMPERATURE, 'stream_timeout_s': CANCEL_STREAM_TIMEOUT,
                   'followup_sha256': text_sha256(CANCEL_FOLLOWUP), 'expected': CANCEL_EXPECTED,
                   'followup_attempts': CANCEL_ATTEMPTS, 'retry_s': CANCEL_RETRY_SECONDS,
                   'disconnect_after': 'first SSE delta with non-empty content'},
    }


def workload_sha256():
    return canonical_sha256(workload_spec())


def coverage_floors():
    """Minimum passed cases: the schedule's counts over MIN_ITERATIONS iterations."""
    floors = {f'fresh@{length}': 0 for length in LENGTHS}
    floors.update({case: 0 for case in CASES if case != 'fresh'})
    for iteration in range(MIN_ITERATIONS):
        plan = schedule(iteration)
        for case in plan['cases']:
            floors[f"fresh@{plan['length']}" if case == 'fresh' else case] += 1
    return floors


def plan_document(iterations):
    counts = {}
    rows = []
    for iteration in range(iterations):
        plan = schedule(iteration)
        rows.append({'iteration': iteration, 'length': plan['length'], 'cases': plan['cases']})
        for case in plan['cases']:
            key = f"fresh@{plan['length']}" if case == 'fresh' else case
            counts[key] = counts.get(key, 0) + 1
    return {'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'mode': 'plan-only-no-network',
            'required_seconds': REQUIRED_SECONDS, 'workload': workload_spec(), 'workload_sha256': workload_sha256(),
            'coverage_floors': coverage_floors(), 'min_iterations_for_floors': MIN_ITERATIONS,
            'required_facts': {case: list(facts) for case, facts in REQUIRED_FACTS.items()},
            'planned_iterations_shown': iterations, 'planned_case_counts': counts, 'schedule': rows,
            'note': 'Iterations actually executed depend on server speed; the run ends at the first iteration '
                    'boundary at or after required_seconds. Gate evaluation: soak-analysis.py.'}


# ---- Name, receipts ----------------------------------------------------------
def validate_name(name):
    # '-run' is reserved for NAME-run.json (copy of the bound active-run receipt).
    if (not isinstance(name, str) or len(name) > NAME_MAX or not NAME_PATTERN.fullmatch(name)
            or name.endswith('-analysis') or '-analysis-' in name or name.endswith('-run')):
        raise ValueError('Invalid soak name: use soak[a-z0-9-], at most 64 characters, no -analysis, '
                         'not ending in -run')
    return name


def report_paths(reports_dir, name):
    validate_name(name)
    reports_dir = pathlib.Path(reports_dir)
    return reports_dir / f'{name}.json', reports_dir / f'{name}-analysis.json'


def run_copy_path(reports_dir, name):
    validate_name(name)
    return pathlib.Path(reports_dir) / f'{name}-run.json'


def ensure_unused(reports_dir, name):
    for path in (*report_paths(reports_dir, name), run_copy_path(reports_dir, name)):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'{path.name} already exists; soak receipts are never overwritten')


def _fsync_directory(directory):
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path, data):
    """Write a complete JSON document beside path, fsync it, then rename over path."""
    path = pathlib.Path(path)
    payload = json.dumps(data, indent=1, ensure_ascii=True, allow_nan=False) + '\n'
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{threading.get_ident()}')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


def create_exclusive_json(path, data):
    """Create path with O_EXCL (never overwrites), write complete JSON and fsync."""
    path = pathlib.Path(path)
    payload = json.dumps(data, indent=1, ensure_ascii=True, allow_nan=False) + '\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    _fsync_directory(path.parent)
    return path


def claim_report(reports_dir, name, claim):
    """Reserve name.json exclusively (O_EXCL); later writes replace it atomically."""
    ensure_unused(reports_dir, name)
    path, _ = report_paths(reports_dir, name)
    return create_exclusive_json(path, claim)


def file_sha256(path):
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def harness_hashes(directory):
    directory = pathlib.Path(directory)
    result = {}
    for name in HARNESS_FILES:
        candidates = [directory / name, directory.parent / name]
        result[name] = next((file_sha256(path) for path in candidates if path.is_file()), None)
    return result


# ---- Candidate binding and GPU-exclusive lock --------------------------------
def resolve_candidate(profile, expected):
    """The frozen candidate the soak must run on (shared by soak.py and soak-analysis.py).

    expected is either the sha256 of the canonical identity (canonical_sha256 of
    bench.validate_receipt's identity == bench.canonical_hash) or the path of a
    frozen manifest whose arms[profile] is that identity (matrix NAME-manifest.json).
    """
    if not isinstance(profile, str) or not PROFILE_PATTERN.fullmatch(profile):
        raise ValueError('Invalid profile name')
    if not isinstance(expected, str) or not expected:
        raise ValueError('Missing expected identity')
    if SHA256_PATTERN.fullmatch(expected):
        return {'profile': profile, 'identity_sha256': expected, 'source': 'sha256'}
    path = pathlib.Path(expected)
    data = path.read_bytes()
    manifest = strict_loads(data)
    arms = manifest.get('arms') if isinstance(manifest, dict) else None
    arm = arms.get(profile) if isinstance(arms, dict) else None
    if not isinstance(arm, dict) or set(arm) != IDENTITY_KEYS:
        raise ValueError('Manifest has no frozen identity for this profile')
    return {'profile': profile, 'identity_sha256': canonical_sha256(arm), 'source': 'manifest',
            'manifest_name': path.name, 'manifest_sha256': hashlib.sha256(data).hexdigest()}


def acquire_gpu_lock(path):
    """Non-blocking exclusive flock on the GPU-exclusive lock (bench.GPU_LOCK).

    Returns (fd, mode). 'inherited': a descriptor of the same file inherited from
    `flock -n <lock> ...` is the wrapper's own open file description, so re-flocking
    it succeeds (same contract as lookup-qualification.acquire_gpu_lock). A lock held
    by anybody else raises GpuLockBusy. The lock file content is never modified.
    """
    import fcntl
    path = pathlib.Path(path)
    target = os.path.realpath(path)
    fd_dir = pathlib.Path('/proc/self/fd')
    if fd_dir.is_dir():
        for entry in list(fd_dir.iterdir()):
            try:
                fd = int(entry.name)
                if os.path.realpath(os.readlink(entry)) != target:
                    continue
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd, 'inherited'
            except (ValueError, OSError):
                continue
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise GpuLockBusy('gpu_lock_busy') from None
    except OSError:
        os.close(fd)
        raise
    return fd, 'acquired'


# ---- SSE cancellation parsing ------------------------------------------------
def first_nonempty_content(lines):
    """Consume SSE lines until a delta carries non-empty content.

    The original check stopped at the first line containing b'content', which is
    FreeToken's opening role chunk {"delta":{"role":"assistant","content":""}};
    the stream was therefore closed before any token was generated.
    Returns (content_seen, data_events).
    """
    events = 0
    for raw in lines:
        if isinstance(raw, str):
            raw = raw.encode()
        if not raw.startswith(b'data: '):
            continue
        data = raw[6:].strip()
        if data == b'[DONE]':
            break
        try:
            event = strict_loads(data)
        except ValueError:
            raise CaseFailure('cancel_stream_malformed_event') from None
        if not isinstance(event, dict):
            raise CaseFailure('cancel_stream_malformed_event')
        events += 1
        for choice in event.get('choices') or []:
            delta = choice.get('delta') if isinstance(choice, dict) else None
            content = delta.get('content') if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                return True, events
    return False, events


# ---- Live HTTP client (network; not exercised by CPU tests) ------------------
def meta_payload(messages, **kwargs):
    """The quality.request payload, unchanged (temperature 0, seed 417, 512 unless overridden)."""
    return {'model': 'qwen36-35b', 'messages': messages, 'max_tokens': 512, 'temperature': 0,
            'seed': 417, 'chat_template_kwargs': {'enable_thinking': False}, **kwargs}


def request_meta(messages, **kwargs):
    """quality.request with the identical payload/URL/timeout plus status and request id."""
    import requests
    from bench import headers
    payload = meta_payload(messages, **kwargs)
    started = time.perf_counter()
    response = requests.post(API_URL, headers=headers(), json=payload, timeout=900)
    meta = {'status': response.status_code, 'request_id': response.headers.get('x-qwen36-request-id'),
            'elapsed_s': time.perf_counter() - started}
    if response.status_code != 200:
        raise HTTPStatusError(meta)
    return response.json(), meta


def cancel_stream(payload):
    """Open the original cancellation stream and close it after real content."""
    import requests
    from bench import headers
    with requests.post(API_URL, headers=headers(), json=payload, stream=True,
                       timeout=CANCEL_STREAM_TIMEOUT) as response:
        meta = {'status': response.status_code, 'request_id': response.headers.get('x-qwen36-request-id')}
        if response.status_code != 200:
            raise HTTPStatusError(meta)
        seen, events = first_nonempty_content(response.iter_lines(chunk_size=1))
    meta.update(content_seen=seen, data_events=events)
    return meta


class HttpClient:
    def __init__(self, bench):
        self.bench = bench

    def generate(self, messages, maximum, seed, temperature):
        return self.bench.generate(messages, maximum, seed, temperature=temperature)

    def request_meta(self, messages, **kwargs):
        return request_meta(messages, **kwargs)

    def cancel_stream(self, payload):
        return cancel_stream(payload)


def systemd_unit_probe(unit):
    import subprocess
    if not UNIT_PATTERN.fullmatch(unit):
        raise ValueError('Unexpected unit name')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=ActiveState,SubState,InvocationID,MainPID,NRestarts'],
                            capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0:
        raise RuntimeError('systemctl show failed')
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition('=')
        if separator:
            if key in values:
                raise RuntimeError('duplicate systemctl field')
            values[key] = value
    if set(values) != {'ActiveState', 'SubState', 'InvocationID', 'MainPID', 'NRestarts'}:
        raise RuntimeError('unexpected systemctl fields')
    return {'active_state': values['ActiveState'], 'sub_state': values['SubState'],
            'invocation_id': values['InvocationID'], 'main_pid': int(values['MainPID']),
            'n_restarts': int(values['NRestarts'])}


def unit_active(unit):
    return (isinstance(unit, dict) and unit.get('active_state') == 'active' and unit.get('sub_state') == 'running'
            and isinstance(unit.get('invocation_id'), str) and bool(unit['invocation_id'])
            and type(unit.get('main_pid')) is int and unit['main_pid'] > 0 and type(unit.get('n_restarts')) is int)


# ---- Identity binding --------------------------------------------------------
class IdentityGuard:
    """Binds the soak to the expected candidate, one active-run receipt, its on-disk sources and one
    unit invocation."""

    def __init__(self, *, candidate, read_receipt, validate, current_identity, unit_probe, monotonic=time.monotonic,
                 period=IDENTITY_PERIOD_SECONDS, source_period=SOURCE_PERIOD_SECONDS):
        if not (isinstance(candidate, dict) and isinstance(candidate.get('profile'), str)
                and isinstance(candidate.get('identity_sha256'), str)
                and SHA256_PATTERN.fullmatch(candidate['identity_sha256'])):
            raise ValueError('candidate binding requires profile and identity_sha256')
        self.candidate = dict(candidate)
        self.read_receipt, self.validate = read_receipt, validate
        self.current_identity, self.unit_probe = current_identity, unit_probe
        self.monotonic, self.period, self.source_period = monotonic, period, source_period
        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.checks = []
        self.fatal = set()
        self.t0 = None
        self.last_check = self.last_source = None
        self.receipt = self.identity = self.unit = None

    def preflight(self):
        try:
            receipt = self.read_receipt()
        except (OSError, ValueError):
            raise IdentityError('active_run_unreadable') from None
        profile = receipt.get('profile') if isinstance(receipt, dict) else None
        if isinstance(profile, str) and profile != self.candidate['profile']:
            raise IdentityError('candidate_mismatch')
        try:
            identity = self.validate(receipt)
        except Exception:
            raise IdentityError('run_receipt_invalid') from None
        if canonical_sha256(identity) != self.candidate['identity_sha256']:
            raise IdentityError('candidate_mismatch')
        try:
            current = self.current_identity(receipt['profile'])
        except Exception:
            raise IdentityError('source_inventory_failed') from None
        if canonical_sha256(current) != canonical_sha256(identity):
            raise IdentityError('stale_run_source_mismatch')
        try:
            unit = self.unit_probe()
        except Exception:
            raise IdentityError('unit_probe_failed') from None
        if not unit_active(unit):
            raise IdentityError('service_not_active')
        self.receipt, self.identity, self.unit = copy.deepcopy(receipt), identity, unit

    @property
    def run_id(self):
        return self.receipt['run_id']

    @property
    def profile(self):
        return self.receipt['profile']

    def mark_start(self, t0):
        self.t0 = self.last_check = self.last_source = t0
        with self.lock:
            self.checks.append({'t_s': 0.0, 'kind': 'source', 'ok': True, 'codes': [], 'stage': 'preflight'})

    def check(self, kind):
        codes = set()
        try:
            if canonical_sha256(self.read_receipt()) != canonical_sha256(self.receipt):
                codes.add('active_run_changed')
        except Exception:
            codes.add('active_run_unreadable')
        try:
            unit = self.unit_probe()
            if not unit_active(unit):
                codes.add('service_not_active')
            if any(unit.get(key) != self.unit.get(key) for key in ('invocation_id', 'main_pid', 'n_restarts')):
                codes.add('service_restarted')
        except Exception:
            codes.add('unit_probe_failed')
        if kind == 'source':
            try:
                if canonical_sha256(self.current_identity(self.profile)) != canonical_sha256(self.identity):
                    codes.add('source_changed_on_disk')
            except Exception:
                codes.add('source_inventory_failed')
        now = self.monotonic()
        entry = {'t_s': now - self.t0, 'kind': kind, 'ok': not codes, 'codes': sorted(codes)}
        with self.lock:
            self.checks.append(entry)
            self.fatal.update(codes)
            self.last_check = now
            if kind == 'source':
                self.last_source = now
        return entry

    def maybe_check(self):
        if self.t0 is None or not self.busy.acquire(blocking=False):
            return None
        try:
            now = self.monotonic()
            if now - self.last_source >= self.source_period:
                return self.check('source')
            if now - self.last_check >= self.period:
                return self.check('periodic')
            return None
        finally:
            self.busy.release()

    def final_checks(self):
        with self.busy:
            self.check('periodic')
            self.check('source')

    def fatal_codes(self):
        with self.lock:
            return set(self.fatal)

    def summary(self):
        with self.lock:
            checks = list(self.checks)
        return {'profile': self.profile, 'run_id': self.run_id,
                'receipt_sha256': canonical_sha256(self.receipt), 'identity_sha256': canonical_sha256(self.identity),
                'unit_initial': dict(self.unit), 'period_s': self.period, 'source_period_s': self.source_period,
                'checks': checks}


# ---- Telemetry tailing -------------------------------------------------------
class JsonlTail:
    """Follow one rotating JSONL file (NumericLog: name -> name.1 ... name.N by os.replace).

    Rotated generations missed between polls are recovered from the backup chain;
    an unreachable generation, an in-place truncation or a malformed line is
    counted and never silently skipped.
    """

    HEAD_BYTES = 256

    def __init__(self, path, backups=LOG_BACKUPS):
        self.path = pathlib.Path(path)
        self.backups = backups
        self.handle = None
        self.identity = None
        self.offset = 0
        self.head = b''
        self.fragment = b''
        self.stats = {'rotations': 0, 'recovered_files': 0, 'loss_events': 0, 'truncations': 0,
                      'malformed_lines': 0, 'read_errors': 0, 'lines': 0, 'started_offset': None,
                      'missing_at_start': False}

    @staticmethod
    def _id(stat):
        return stat.st_dev, stat.st_ino

    def _open(self, path, at_end=False):
        handle = open(path, 'rb')
        stat = os.fstat(handle.fileno())
        if self.handle is not None:
            self.handle.close()
        self.handle, self.identity = handle, self._id(stat)
        self.offset = stat.st_size if at_end else 0
        self.head = os.pread(handle.fileno(), self.HEAD_BYTES, 0)
        handle.seek(self.offset)

    def _rewritten(self):
        """Same inode, but shorter than what was read or a different first bytes."""
        size = os.fstat(self.handle.fileno()).st_size
        if size < self.offset:
            return True
        head = os.pread(self.handle.fileno(), self.HEAD_BYTES, 0)
        if len(self.head) < self.HEAD_BYTES and len(head) > len(self.head) and head.startswith(self.head):
            self.head = head
        return not head.startswith(self.head)

    def start_at_end(self):
        try:
            self._open(self.path, at_end=True)
            self.stats['started_offset'] = self.offset
        except FileNotFoundError:
            self.stats.update(started_offset=0, missing_at_start=True)

    def _parse(self, data, out, final):
        lines = data.split(b'\n')
        self.fragment = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = strict_loads(line)
            except ValueError:
                row = None
            if not isinstance(row, dict):
                self.stats['malformed_lines'] += 1
                continue
            self.stats['lines'] += 1
            out.append(row)
        if final and self.fragment:
            self.stats['malformed_lines'] += 1
            self.fragment = b''

    def _drain(self, out):
        data = self.handle.read()
        self.offset += len(data)
        self._parse(self.fragment + data, out, final=False)

    def _read_whole(self, path, out):
        with open(path, 'rb') as handle:
            data = handle.read()
        saved, self.fragment = self.fragment, b''
        self._parse(data, out, final=True)
        self.fragment = saved

    def poll(self):
        out = []
        try:
            if self.handle is None:
                try:
                    self._open(self.path)  # created after the soak started: read from byte 0
                except FileNotFoundError:
                    return out
            for _ in range(self.backups + 2):
                try:
                    current = os.stat(self.path)
                except FileNotFoundError:
                    current = None
                if current is not None and self._id(current) == self.identity and self._rewritten():
                    # Checked before draining: bytes past a truncation point belong to new content.
                    self.stats['truncations'] += 1
                    self.fragment = b''
                    self._open(self.path)
                    continue
                self._drain(out)
                if current is not None and self._id(current) == self.identity:
                    return out
                if current is None:
                    return out
                # Rotated: the old generation is closed for writing and fully drained.
                if self.fragment:
                    self.stats['malformed_lines'] += 1
                    self.fragment = b''
                old = self.identity
                self._open(self.path)
                newest = self.identity
                position, chain = None, {}
                for index in range(1, self.backups + 1):
                    try:
                        chain[index] = self._id(os.stat(f'{self.path}.{index}'))
                    except FileNotFoundError:
                        continue
                    if chain[index] == old:
                        position = index
                        break
                if position is None:
                    self.stats['loss_events'] += 1
                else:
                    for index in range(position - 1, 0, -1):
                        if chain.get(index) not in (None, newest):
                            self._read_whole(f'{self.path}.{index}', out)
                            self.stats['recovered_files'] += 1
                self.stats['rotations'] += position or 1  # generations that moved past us
            return out
        except OSError:
            self.stats['read_errors'] += 1
            return out

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class TelemetryMonitor:
    """Background reader of requests/engine/state JSONL for the soak window."""

    def __init__(self, data_root, *, monotonic=time.monotonic, poll_seconds=1.0, backups=LOG_BACKUPS):
        data_root = pathlib.Path(data_root)
        self.tails = {'requests': JsonlTail(data_root / 'logs' / 'requests.jsonl', backups),
                      'engine': JsonlTail(data_root / 'reports' / 'engine.jsonl', backups),
                      'state': JsonlTail(data_root / 'logs' / 'state.jsonl', backups)}
        self.monotonic, self.poll_seconds = monotonic, poll_seconds
        self.lock = threading.Lock()
        self.requests, self.engine, self.state, self.events = {}, {}, [], []
        self.other_requests, self.invalid_rows = {}, {'requests': 0, 'engine': 0, 'state': 0}
        self.duplicates = {'requests': 0, 'engine': 0}
        self.sessions, self.instances, self.request_run_ids, self.engine_run_ids = set(), set(), set(), set()
        self.counter_flags = set()
        self.failures = []
        self.polls = 0
        self.last_poll = None
        self.max_poll_interval = 0.0
        self.thread = None
        self.stop_event = threading.Event()
        self.background = {'started': False, 'alive_before_stop': None, 'joined': None}
        self.tick = None

    def start_positions(self):
        with self.lock:
            for tail in self.tails.values():
                tail.start_at_end()

    def _ingest(self, source, row):
        if source == 'requests':
            if row.get('kind') != 'generation':
                kind = row.get('kind') if isinstance(row.get('kind'), str) and len(row.get('kind')) <= 32 else 'other'
                self.other_requests[kind] = self.other_requests.get(kind, 0) + 1
                return
            rid = row.get('request_id')
            if not isinstance(rid, str) or not rid:
                self.invalid_rows['requests'] += 1
                return
            if rid in self.requests:
                self.duplicates['requests'] += 1
                return
            self.requests[rid] = {key: copy.deepcopy(row[key]) for key in REQUEST_FIELDS if key in row}
            self.request_run_ids.add(row.get('run_id'))
            if isinstance(row.get('session_id'), str):
                self.sessions.add(row['session_id'])
        elif source == 'engine':
            key = f"{row.get('run_id')}:{row.get('uid')}"
            if key in self.engine:
                self.duplicates['engine'] += 1
                return
            self.engine[key] = {name: copy.deepcopy(row[name]) for name in ENGINE_FIELDS if name in row}
            self.engine_run_ids.add(row.get('run_id'))
        else:
            if 'event' in row:
                event = row.get('event')
                self.events.append({'ts': row.get('ts') if isinstance(row.get('ts'), str) else None,
                                    'event': event[:64] if isinstance(event, str) else 'invalid'})
                return
            self.state.append({key: copy.deepcopy(row[key]) for key in STATE_FIELDS if key in row})
            if isinstance(row.get('session_id'), str):
                self.sessions.add(row['session_id'])
            self.instances.add(row.get('instance_id'))
            for key, code in (('metrics_dropped', 'telemetry_metrics_dropped'),
                              ('metrics_failed_writes', 'telemetry_write_failures'),
                              ('resource_sample_failures', 'telemetry_resource_sample_failures')):
                if not (type(row.get(key)) is int and row[key] == 0):
                    self.counter_flags.add(code)
            if row.get('resource_sample_stale'):
                self.counter_flags.add('telemetry_resource_sample_stale')

    def poll(self):
        with self.lock:
            now = self.monotonic()
            if self.last_poll is not None:
                self.max_poll_interval = max(self.max_poll_interval, now - self.last_poll)
            self.last_poll = now
            for name, tail in self.tails.items():
                for row in tail.poll():
                    self._ingest(name, row)
            self.polls += 1

    def background_step(self):
        """One cycle of the background reader: poll the JSONL tails, then the identity tick."""
        try:
            self.poll()
        except Exception as error:  # recorded and fatal; never silently ignored
            with self.lock:
                self.failures.append(type(error).__name__)
        if self.tick is not None:
            try:
                self.tick()
            except Exception as error:
                with self.lock:
                    self.failures.append('tick:' + type(error).__name__)

    def background_running(self):
        return self.thread is not None and not self.stop_event.is_set()

    def _run(self):
        while not self.stop_event.wait(self.poll_seconds):
            self.background_step()

    def start_background(self, tick=None):
        self.tick = tick
        self.thread = threading.Thread(target=self._run, name='qwen36-soak-telemetry', daemon=True)
        self.thread.start()
        self.background['started'] = True

    def stop_background(self):
        if self.thread is None:
            return
        self.background['alive_before_stop'] = self.thread.is_alive()
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.background['joined'] = not self.thread.is_alive()

    def latest_state_ts(self):
        with self.lock:
            stamps = []
            for row in self.state:
                try:
                    stamps.append(parse_utc(row.get('ts')))
                except (TypeError, ValueError):
                    continue
            return max(stamps, default=None)

    def has_rows(self, request_id, run_id):
        with self.lock:
            row = self.requests.get(request_id)
            if row is None:
                return False
            uid = row.get('engine_uid')
            return uid is None or f'{run_id}:{uid}' in self.engine

    def fatal_codes(self, run_id):
        with self.lock:
            codes = set(self.counter_flags)
            for tail in self.tails.values():
                for key, code in (('loss_events', 'telemetry_rotation_loss'), ('truncations', 'telemetry_truncated'),
                                  ('malformed_lines', 'telemetry_malformed_lines'),
                                  ('read_errors', 'telemetry_read_errors')):
                    if tail.stats[key]:
                        codes.add(code)
            if self.failures:
                codes.add('telemetry_monitor_failed')
            if self.events:
                codes.add('service_stall_event')
            if any(self.duplicates.values()):
                codes.add('telemetry_duplicate_rows')
            if len(self.sessions) > 1:
                codes.add('service_session_changed')
            if len(self.instances) > 1:
                codes.add('service_instance_changed')
            if self.request_run_ids - {run_id}:
                codes.add('serving_run_id_mismatch')
            if self.engine_run_ids - {run_id}:
                codes.add('engine_run_id_mismatch')
            return codes

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({
                'requests': self.requests, 'engine': self.engine, 'state': self.state, 'events': self.events,
                'other_requests': self.other_requests, 'invalid_rows': self.invalid_rows,
                'duplicates': self.duplicates,
                'readers': {name: dict(tail.stats) for name, tail in self.tails.items()},
                'monitor': {'polls': self.polls, 'max_poll_interval_s': self.max_poll_interval,
                            'poll_seconds': self.poll_seconds, 'failures': list(self.failures),
                            'background': dict(self.background)}})

    def close(self):
        with self.lock:
            for tail in self.tails.values():
                tail.close()


# ---- Runner ------------------------------------------------------------------
class SoakRunner:
    def __init__(self, *, name, seconds, reports_dir, client, document, identity, monitor, gpu_lock_path,
                 monotonic=time.monotonic, wall=utc_now, sleep=time.sleep, harness=None, emit=None,
                 partial_interval=PARTIAL_INTERVAL_SECONDS, drain_seconds=FINAL_DRAIN_SECONDS):
        self.name, self.seconds = validate_name(name), seconds
        self.reports_dir = pathlib.Path(reports_dir)
        self.gpu_lock_path = pathlib.Path(gpu_lock_path)
        self.gpu_lock = None
        self.run_copy = None
        self.client, self.document, self.identity, self.monitor = client, document, identity, monitor
        self.monotonic, self.wall, self.sleep = monotonic, wall, sleep
        self.harness = harness or {}
        self.emit = emit or (lambda value: print(json.dumps(value), flush=True))
        self.partial_interval, self.drain_seconds = partial_interval, drain_seconds
        self.rows = []
        self.iterations = 0
        self.termination = None
        self.abort_codes = set()
        self.path = None
        self.started = self.started_utc = None
        self.last_partial = None

    # -- evidence helpers --
    def _t(self):
        return self.monotonic() - self.started

    def _tick(self):
        self.monitor.poll()
        self.identity.maybe_check()

    def _fatal(self):
        return self.identity.fatal_codes() | self.monitor.fatal_codes(self.identity.run_id)

    def _receipt(self, status, elapsed=None, finished_utc=None):
        complete = (status == 'finished' and self.termination == 'duration_reached'
                    and elapsed is not None and elapsed >= self.seconds)
        return {
            'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'name': self.name, 'status': status,
            'termination_reason': self.termination, 'abort_codes': sorted(self.abort_codes),
            'required_seconds': self.seconds, 'elapsed_s': elapsed if elapsed is not None else self._t(),
            'complete': complete, 'started_utc': iso(self.started_utc),
            'finished_utc': iso(finished_utc) if finished_utc is not None else None,
            'workload': workload_spec(), 'workload_sha256': workload_sha256(),
            'coverage_floors': coverage_floors(), 'harness': dict(self.harness), 'run': self.identity.receipt,
            'candidate': dict(self.identity.candidate), 'gpu_lock': dict(self.gpu_lock or {}),
            'run_copy': self.run_copy,
            'identity': self.identity.summary(), 'iterations': self.iterations, 'rows': self.rows,
            'telemetry': self.monitor.snapshot(),
            'note': 'Evidence only. Gate verdict: soak-analysis.py. No prompts, answers or keys are stored.'}

    def _persist_partial(self, force=False):
        now = self.monotonic()
        if force or self.last_partial is None or now - self.last_partial >= self.partial_interval:
            atomic_write_json(self.path, self._receipt('running'))
            self.last_partial = now

    def _record(self, row, role, meta, **extra):
        entry = {'role': role, 'request_id': meta.get('request_id'), 'status': meta.get('status')}
        if is_number(meta.get('elapsed_s')):
            entry['elapsed_s'] = meta['elapsed_s']
        entry.update(extra)
        row['requests'].append(entry)
        return entry

    def _record_generate(self, row, role, result, answer):
        usage = result.get('usage') if isinstance(result.get('usage'), dict) else {}
        return self._record(row, role, result, prompt_tokens=usage.get('prompt_tokens'),
                            completion_tokens=usage.get('completion_tokens'),
                            finish_reason=result.get('finish_reason'), content_sha256=result.get('content_sha256'),
                            content_nonempty=bool(answer.strip()), ttft_content_s=result.get('ttft_content_s'),
                            decode_client_tps=result.get('decode_client_tps'),
                            inline_engine_metrics_status=result.get('engine_metrics_status'),
                            inline_serving_run_id=result.get('serving_run_id'),
                            cached_prompt_tokens=result.get('cached_prompt_tokens'),
                            cache_hit_observed=result.get('cache_hit_observed'))

    def _call_meta(self, row, role, messages, **kwargs):
        try:
            body, meta = self.client.request_meta(messages, **kwargs)
        except HTTPStatusError as error:
            self._record(row, role, error.meta)
            raise
        self._record(row, role, meta)
        return body, meta

    # -- cases (workload exactly as in the original soak) --
    def _case_fresh(self, plan, row, context):
        messages = [{'role': 'user', 'content': self.document(plan['length'], plan['document_variant']) + FRESH_SUFFIX}]
        result, answer = self.client.generate(messages, FRESH_MAX_TOKENS, plan['seed'], SAMPLING_TEMPERATURE)
        entry = self._record_generate(row, 'fresh', result, answer)
        facts = row['facts']
        facts['status_200'] = result.get('status') == 200
        if not facts['status_200']:
            raise CaseFailure('fresh_http_status')
        prompt, completion = entry['prompt_tokens'], entry['completion_tokens']
        facts['content_nonempty'] = entry['content_nonempty']
        facts['completion_tokens_positive'] = type(completion) is int and completion > 0
        facts['prompt_tokens_cover_length'] = (type(prompt) is int
                                               and prompt >= math.ceil(PROMPT_COVERAGE_RATIO * plan['length']))
        context.update(messages=messages, answer=answer, prompt_tokens=prompt)

    def _case_prefix(self, plan, row, context):
        follow = context['messages'] + [{'role': 'assistant', 'content': context['answer']},
                                        {'role': 'user', 'content': PREFIX_FOLLOWUP}]
        result, answer = self.client.generate(follow, PREFIX_MAX_TOKENS, plan['seed'], SAMPLING_TEMPERATURE)
        entry = self._record_generate(row, 'prefix', result, answer)
        facts = row['facts']
        facts['status_200'] = result.get('status') == 200
        if not facts['status_200']:
            raise CaseFailure('prefix_http_status')
        prompt, completion, fresh = entry['prompt_tokens'], entry['completion_tokens'], context.get('prompt_tokens')
        facts['content_nonempty'] = entry['content_nonempty']
        facts['completion_tokens_positive'] = type(completion) is int and completion > 0
        facts['prompt_tokens_exceed_fresh'] = type(prompt) is int and type(fresh) is int and prompt > fresh

    def _case_arithmetic(self, plan, row, context):
        facts = row['facts']
        facts['status_200'] = False
        body, _ = self._call_meta(row, 'arithmetic', [{'role': 'user', 'content': ARITHMETIC_PROMPT}],
                                  max_tokens=SMOKE_MAX_TOKENS)
        facts['status_200'] = True
        content = body['choices'][0]['message'].get('content') or ''
        facts['contains_expected'] = ARITHMETIC_EXPECTED in content

    def _case_tool(self, plan, row, context):
        facts = row['facts']
        facts['call_status_200'] = False
        messages = [{'role': 'user', 'content': TOOL_PROMPT}]
        first, _ = self._call_meta(row, 'tool_call', messages, tools=TOOLS)
        facts['call_status_200'] = True
        choice = first['choices'][0]
        message = choice['message']
        calls = message.get('tool_calls') or []
        facts['one_tool_call'] = isinstance(calls, list) and len(calls) == 1
        facts['finish_reason_tool_calls'] = choice.get('finish_reason') == 'tool_calls'
        call = calls[0] if facts['one_tool_call'] and isinstance(calls[0], dict) else {}
        function = call.get('function') if isinstance(call.get('function'), dict) else {}
        facts['call_type_function'] = call.get('type') == 'function'
        facts['call_id_present'] = isinstance(call.get('id'), str) and bool(call['id'])
        facts['call_name_matches'] = function.get('name') == TOOL_NAME
        try:
            arguments = strict_loads(function.get('arguments'))
        except (TypeError, ValueError):
            arguments = None
        facts['call_arguments_match'] = arguments == TOOL_ARGUMENTS
        if not all(facts[key] for key in ('one_tool_call', 'call_type_function', 'call_id_present',
                                          'call_name_matches', 'call_arguments_match', 'finish_reason_tool_calls')):
            raise CaseFailure('tool_call_invalid')
        facts['final_status_200'] = False
        last, _ = self._call_meta(row, 'tool_final', messages + [
            message, {'role': 'tool', 'tool_call_id': call['id'], 'content': TOOL_RESULT}], tools=TOOLS)
        facts['final_status_200'] = True
        final = last['choices'][0]['message']
        facts['final_contains_expected'] = TOOL_EXPECTED in (final.get('content') or '')
        facts['final_no_tool_calls'] = not final.get('tool_calls')

    def _case_cancel(self, plan, row, context):
        facts = row['facts']
        payload = {'model': 'qwen36-35b', 'messages': [{'role': 'user', 'content': CANCEL_PROMPT}],
                   'max_tokens': CANCEL_MAX_TOKENS, 'ignore_eos': True, 'stream': True,
                   'temperature': SAMPLING_TEMPERATURE, 'chat_template_kwargs': {'enable_thinking': False}}
        started = self.monotonic()
        facts['stream_status_200'] = False
        try:
            stream = self.client.cancel_stream(payload)
        except HTTPStatusError as error:
            self._record(row, 'cancel_stream', error.meta)
            raise
        self._record(row, 'cancel_stream', stream, data_events=stream.get('data_events'))
        facts['stream_status_200'] = stream.get('status') == 200
        facts['content_seen_before_disconnect'] = stream.get('content_seen') is True
        if not facts['content_seen_before_disconnect']:
            raise CaseFailure('cancel_no_content_before_disconnect')
        disconnected = self.monotonic()
        facts['followup_status_200'] = False
        facts['followup_within_attempts'] = False
        for attempt in range(CANCEL_ATTEMPTS):
            self.sleep(CANCEL_RETRY_SECONDS)
            try:
                body, meta = self.client.request_meta([{'role': 'user', 'content': CANCEL_FOLLOWUP}],
                                                      max_tokens=SMOKE_MAX_TOKENS)
            except HTTPStatusError as error:
                if error.meta.get('status') == 429:
                    self._record(row, 'cancel_followup_rejected', error.meta)
                    continue
                self._record(row, 'cancel_followup', error.meta)
                raise
            self._record(row, 'cancel_followup', meta)
            now = self.monotonic()
            facts['followup_status_200'] = True
            facts['followup_within_attempts'] = True
            facts['followup_contains_expected'] = CANCEL_EXPECTED in (body['choices'][0]['message'].get('content') or '')
            row['measurements'] = {'followup_attempts': attempt + 1, 'recovery_s': now - started,
                                   'recovery_after_disconnect_s': now - disconnected}
            if not facts['followup_contains_expected']:
                raise CaseFailure('cancel_followup_content')
            return
        raise CaseFailure('cancel_slot_not_released')

    def _run_iteration(self, plan):
        context = {}
        for case in plan['cases']:
            row = {'iteration': plan['iteration'], 'case': case, 'requests': [], 'facts': {},
                   't_start_s': self._t(), 'pass': False, 'failure': None, 'error_type': None}
            if case in ('fresh', 'prefix'):
                row.update(length=plan['length'], seed=plan['seed'], document_variant=plan['document_variant'])
            try:
                getattr(self, '_case_' + case)(plan, row, context)
                if not all(row['facts'].get(key) is True for key in REQUIRED_FACTS[case]):
                    raise CaseFailure(case + '_check_failed')
                row['pass'] = True
            except Exception as error:
                row['failure'] = getattr(error, 'code', None) or 'exception'
                row['error_type'] = type(error).__name__
            finally:
                row['t_end_s'] = self._t()
                self.rows.append(row)
            if not row['pass']:
                return True
        return False

    def _loop(self):
        iteration = 0
        consecutive = 0
        while self.monotonic() - self.started < self.seconds:
            fatal = self._fatal()
            if fatal:
                self.abort_codes.update(fatal)
                return 'fatal_evidence_failure'
            failed = self._run_iteration(schedule(iteration))
            self.iterations = iteration + 1
            consecutive = consecutive + 1 if failed else 0
            self._tick()
            self._persist_partial()
            self.emit({'iteration': iteration, 'elapsed_s': self._t(), 'errors': consecutive})
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                return 'consecutive_errors'
            iteration += 1
            self.sleep(PAUSE_SECONDS)
        return 'duration_reached'

    def _final_drain(self, finished_utc):
        deadline = self.monotonic() + self.drain_seconds
        issued = [(entry.get('request_id')) for row in self.rows for entry in row['requests']
                  if isinstance(entry.get('request_id'), str)]
        while True:
            self.monitor.poll()
            latest = self.monitor.latest_state_ts()
            joined = all(self.monitor.has_rows(rid, self.identity.run_id) for rid in issued)
            if joined and latest is not None and latest >= finished_utc:
                return
            if self.monotonic() >= deadline:
                return
            self.sleep(.5)

    def run(self):
        """Hold the GPU-exclusive lock from before preflight until the final receipt is written."""
        try:
            ensure_unused(self.reports_dir, self.name)
        except (FileExistsError, ValueError) as error:
            self.emit({'result': 'REFUSED', 'reason': 'name_unavailable', 'detail': str(error)})
            return 2
        try:
            fd, mode = acquire_gpu_lock(self.gpu_lock_path)
        except GpuLockBusy:
            self.emit({'result': 'REFUSED', 'reason': 'gpu_lock_busy', 'lock': str(self.gpu_lock_path)})
            return 2
        except OSError as error:
            self.emit({'result': 'REFUSED', 'reason': 'gpu_lock_unavailable', 'detail': type(error).__name__})
            return 2
        self.gpu_lock = {'path': str(self.gpu_lock_path), 'mode': mode}
        try:
            return self._run_locked()
        finally:
            if mode == 'acquired':
                os.close(fd)  # an inherited descriptor belongs to the flock(1) wrapper

    def _run_locked(self):
        try:
            ensure_unused(self.reports_dir, self.name)
            self.identity.preflight()
        except IdentityError as error:
            self.emit({'result': 'REFUSED', 'reason': error.code})
            return 2
        except (FileExistsError, ValueError) as error:
            self.emit({'result': 'REFUSED', 'reason': 'name_unavailable', 'detail': str(error)})
            return 2
        self.started, self.started_utc = self.monotonic(), self.wall()
        try:
            self.path = claim_report(self.reports_dir, self.name, {
                'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'name': self.name, 'status': 'claimed',
                'started_utc': iso(self.started_utc)})
            # The bound active-run receipt, top-level, for make-qualification run_receipts.
            self.run_copy = create_exclusive_json(run_copy_path(self.reports_dir, self.name),
                                                  self.identity.receipt).name
        except (FileExistsError, ValueError) as error:
            self.emit({'result': 'REFUSED', 'reason': 'name_unavailable', 'detail': str(error)})
            return 2
        self.monitor.start_positions()
        self.identity.mark_start(self.started)
        self.monitor.start_background(tick=self.identity.maybe_check)
        status, pending = 'aborted', None
        elapsed = finished_utc = None
        try:
            self._persist_partial(force=True)
            self.termination = self._loop()
            status = 'finished'
        except KeyboardInterrupt:
            self.termination = 'interrupted'
        except BaseException as error:
            self.termination = 'runner_exception:' + type(error).__name__
            pending = error
        finally:
            elapsed, finished_utc = self._t(), self.wall()
            try:
                self._final_drain(finished_utc)
                self.identity.final_checks()
            finally:
                self.monitor.stop_background()
                self.monitor.poll()
                atomic_write_json(self.path, self._receipt(status, elapsed, finished_utc))
                self.monitor.close()
        self.emit({'result': 'RECEIPT_WRITTEN', 'status': status, 'termination_reason': self.termination,
                   'elapsed_s': elapsed, 'rows': len(self.rows), 'receipt': str(self.path), 'run_copy': self.run_copy,
                   'next': f"python soak-analysis.py --name {self.name} --profile {self.identity.candidate['profile']} "
                           f"--expected-identity {self.identity.candidate['identity_sha256']}"})
        if pending is not None:
            raise pending  # after the aborted receipt is safely on disk
        return 0 if status == 'finished' else 1


# ---- CLI ---------------------------------------------------------------------
def build_live_runner(args, candidate):
    import bench
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    identity = IdentityGuard(
        candidate=candidate,
        read_receipt=bench.run_receipt,
        validate=lambda receipt: bench.validate_receipt(receipt, candidate['profile']),
        current_identity=lambda profile: bench.current_run_spec(
            bench.ROOT, [profile], bench.DEFAULT_LENGTHS, FRESH_MAX_TOKENS, 1)['arms'][profile],
        unit_probe=lambda: systemd_unit_probe(args.unit))
    return SoakRunner(name=args.name, seconds=args.seconds, reports_dir=bench.ROOT / 'reports',
                      client=HttpClient(bench), document=lambda length, variant: bench.document(tokenizer, length, variant),
                      identity=identity, monitor=TelemetryMonitor(bench.ROOT), gpu_lock_path=bench.GPU_LOCK,
                      harness=harness_hashes(pathlib.Path(__file__).resolve().parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--name', default='soak128', help='new receipt name, soak[a-z0-9-]; never overwritten')
    parser.add_argument('--profile', help='frozen candidate profile (required for a run)')
    parser.add_argument('--expected-identity',
                        help='sha256 of the candidate canonical identity, or a frozen manifest with arms[profile]')
    parser.add_argument('--seconds', type=int, default=REQUIRED_SECONDS,
                        help='workload duration; the gate requires >= 7200')
    parser.add_argument('--unit', default=DEFAULT_UNIT, help='systemd user unit that must stay one invocation')
    parser.add_argument('--plan', action='store_true', help='print the schedule and gate floors; no network')
    parser.add_argument('--plan-iterations', type=int, default=MIN_ITERATIONS)
    args = parser.parse_args(argv)
    if args.plan:
        if not 1 <= args.plan_iterations <= 10000:
            parser.error('--plan-iterations must be 1..10000')
        print(json.dumps(plan_document(args.plan_iterations), ensure_ascii=True, indent=1))
        return 0
    try:
        validate_name(args.name)
    except ValueError as error:
        parser.error(str(error))
    if args.profile is None or args.expected_identity is None:
        parser.error('--profile and --expected-identity are required: the soak runs on the frozen candidate only')
    try:
        candidate = resolve_candidate(args.profile, args.expected_identity)
    except (OSError, ValueError) as error:
        parser.error(f'invalid --profile/--expected-identity: {error}')
    if args.seconds < 1 or not UNIT_PATTERN.fullmatch(args.unit):
        parser.error('invalid --seconds/--unit')
    if args.seconds < REQUIRED_SECONDS:
        print(json.dumps({'warning': 'seconds below 7200: the receipt cannot pass the two-hour gate'}), flush=True)
    import signal

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    return build_live_runner(args, candidate).run()


if __name__ == '__main__':
    sys.exit(main())
