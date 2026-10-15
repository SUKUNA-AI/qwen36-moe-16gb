"""Fail-closed gate for a soak v2 receipt written by soak.py.

two_hour_execution_gate_pass is true only when every condition holds: a final
receipt of the original workload (pinned hash) for >=7200 s; the run is the
expected frozen candidate (--profile, --expected-identity) and the runner held
the GPU-exclusive lock; a valid active-run receipt, its NAME-run.json copy,
bound to one systemd invocation, one HTTP session, one engine instance and to
every request/engine row; mandatory request<->engine joins; server-recorded
sampling of every request equals the workload plan; numeric cgroup/anon/shmem/
filecache/GPU resources; every scheduled case executed and passed, with coverage
floors met; complete state telemetry (no gaps, drops, writer failures, rotation
loss); memory comparisons at equal prompt length and over idle samples; NVML free
VRAM at or above the owner's 0.8 GiB floor in every state sample (the minutes below
the floor are counted in summary.vram for operations).

Missing evidence is FAIL. A valid run with too few points is INCONCLUSIVE.
There is no automatic "no leak" verdict: growth above a threshold is FAIL, and
otherwise memory numbers are reported for review, never as proof of no leak.
Output contains numbers, hashes and reason codes only.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ANALYSIS_SCHEMA = 'qwen36-soak-analysis'
ANALYSIS_SCHEMA_VERSION = 2
MIB = 1024 * 1024
STATE_GAP_LIMIT_S = 30.0       # state.jsonl is written every ~10 s
STATE_EDGE_LIMIT_S = 30.0
IDENTITY_GAP_LIMIT_S = 120.0   # soak.py checks every 30 s, sources every 300 s
CLOCK_SKEW_LIMIT_S = 60.0
REQUEST_RESOURCE_KEYS = ('service_cgroup_memory_bytes', 'service_memory_anon_bytes', 'service_memory_shmem_bytes',
                         'service_memory_file_cache_estimate_bytes', 'gpu_total_used_bytes')
STATE_RESOURCE_KEYS = REQUEST_RESOURCE_KEYS + ('gpu_free_bytes',)
ENGINE_NUMERIC_KEYS = ('torch_allocated_bytes', 'torch_reserved_bytes', 'engine_span_s')
STATE_COUNTERS = (('metrics_dropped', 'telemetry_metrics_dropped'),
                  ('metrics_failed_writes', 'telemetry_write_failures'),
                  ('resource_sample_failures', 'telemetry_resource_sample_failures'))
READER_COUNTERS = (('loss_events', 'telemetry_rotation_loss'), ('truncations', 'telemetry_truncated'),
                   ('malformed_lines', 'telemetry_malformed_lines'), ('read_errors', 'telemetry_read_errors'))
# Growth (late median - early median) above these is a gate FAIL that needs
# investigation; it is not a proof of a leak. Process-anonymous memory, the
# shared expert banks and the PyTorch allocator are gated. cgroup total and the
# file-cache estimate include reclaimable page cache; NVML total GPU use includes
# other WDDM processes. Those are reported, not gated.
REQUEST_MEMORY_FLAGS = {'service_memory_anon_bytes': 256 * MIB, 'service_memory_shmem_bytes': 64 * MIB}
ENGINE_MEMORY_FLAGS = {'torch_allocated_bytes': 64 * MIB, 'torch_reserved_bytes': 256 * MIB}
MEMORY_REPORT_ONLY = ('service_cgroup_memory_bytes', 'service_memory_file_cache_estimate_bytes',
                      'gpu_total_used_bytes')
IDLE_MEMORY_FLAGS = dict(REQUEST_MEMORY_FLAGS)
# Owner decision 2026-10-15 (VRAM test): below ~0.8 GiB free the Windows desktop starts paging and the model
# slows by ~20 %. Same value as bench.VRAM_FLOOR_BYTES; a CPU test pins the equality.
VRAM_FLOOR_BYTES = 4 * 1024 * MIB // 5
IDLE_WARMUP_S = 600.0
IDLE_WINDOW_S = 1200.0
IDLE_MIN_SAMPLES = 10
REQUEST_ID = re.compile(r'[0-9a-f]{32}')
TAG_PATTERN = re.compile(r'[a-z0-9]{1,16}')
EXPECTED_ROLES = {'fresh': ['fresh'], 'prefix': ['prefix'], 'arithmetic': ['arithmetic'],
                  'tool': ['tool_call', 'tool_final']}
EXIT_CODES = {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 3}
PILOT_DURATION_CODES = {'required_seconds_below_7200', 'elapsed_below_required', 'receipt_incomplete'}
# soak128 original workload; a soak.py whose workload_spec() differs cannot pass.
PINNED_WORKLOAD_SHA256 = '70d533a63d298ef6a2b9b070bd205d7aa2fe647ed474b9e521c07aeb9b47e89b'
# Request sampling as request_metrics.RequestRecord.configure records it. bench.generate sends
# top_p .95 when temperature != 0; quality.request sends temperature 0, seed 417, 512 tokens.
GENERATE_TOP_P = .95
QUALITY_SAMPLING = {'max_tokens': 512, 'temperature': 0, 'seed': 417, 'enable_thinking': False}


def expected_sampling(soak, role, iteration):
    """Server-side sampling dict for one request of the original workload."""
    plan = soak.schedule(iteration)
    if role in ('fresh', 'prefix'):
        return {'max_tokens': soak.FRESH_MAX_TOKENS if role == 'fresh' else soak.PREFIX_MAX_TOKENS,
                'seed': plan['seed'], 'temperature': soak.SAMPLING_TEMPERATURE, 'top_p': GENERATE_TOP_P,
                'enable_thinking': False}
    if role in ('arithmetic', 'cancel_followup', 'cancel_followup_rejected'):
        return dict(QUALITY_SAMPLING, max_tokens=soak.SMOKE_MAX_TOKENS)
    if role in ('tool_call', 'tool_final'):
        return dict(QUALITY_SAMPLING)
    if role == 'cancel_stream':
        return {'max_tokens': soak.CANCEL_MAX_TOKENS, 'temperature': soak.SAMPLING_TEMPERATURE,
                'enable_thinking': False}
    return None


def sampling_matches(actual, expected):
    """Same keys and values; a bool never equals a number (False != 0)."""
    return (isinstance(actual, dict) and isinstance(expected, dict) and set(actual) == set(expected)
            and all((type(actual[key]) is bool) == (type(expected[key]) is bool) and actual[key] == expected[key]
                    for key in expected))


def load_soak(path=HERE / 'soak.py'):
    spec = importlib.util.spec_from_file_location('qwen36_soak_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonnegative_int(value):
    return type(value) is int and value >= 0


class Findings:
    def __init__(self):
        self.buckets = {'fail': {}, 'inconclusive': {}}

    def add(self, kind, code, **detail):
        entry = self.buckets[kind].setdefault(code, {'count': 0, 'examples': []})
        entry['count'] += 1
        if detail and len(entry['examples']) < 5:
            entry['examples'].append(detail)

    def fail(self, code, **detail):
        self.add('fail', code, **detail)

    def inconclusive(self, code, **detail):
        self.add('inconclusive', code, **detail)


def compare_series(values, warmup, window):
    count = len(values)
    if count < warmup + 2 * window:
        return {'status': 'insufficient', 'points': count, 'required_points': warmup + 2 * window}
    early, late = values[warmup:warmup + window], values[-window:]
    middle_start = max(warmup + window, (count - window) // 2)
    middle = values[middle_start:middle_start + window] or late
    early_median, late_median = statistics.median(early), statistics.median(late)
    return {'status': 'ok', 'points': count, 'early_median': early_median,
            'middle_median': statistics.median(middle), 'late_median': late_median,
            'late_minus_early': late_median - early_median, 'min': min(values), 'max': max(values)}


def _parse_time(soak, value):
    try:
        return soak.parse_utc(value)
    except (TypeError, ValueError):
        return None


def _check_receipt(receipt, soak, findings, summary, soak_sha256, expected_name):
    if receipt.get('name') != expected_name:
        findings.fail('receipt_name_mismatch')
    if receipt.get('status') != 'finished':
        findings.fail('receipt_not_final', status=receipt.get('status'))
    if receipt.get('termination_reason') != 'duration_reached':
        findings.fail('soak_terminated_early', termination_reason=receipt.get('termination_reason'),
                      abort_codes=receipt.get('abort_codes'))
    required, elapsed = receipt.get('required_seconds'), receipt.get('elapsed_s')
    if type(required) is not int or required < soak.REQUIRED_SECONDS:
        findings.fail('required_seconds_below_7200', required_seconds=required)
    floor = max(soak.REQUIRED_SECONDS, required if type(required) is int else 0)
    if not is_number(elapsed) or elapsed < floor:
        findings.fail('elapsed_below_required', elapsed_s=elapsed, required_s=floor)
    if receipt.get('complete') is not True:
        findings.fail('receipt_incomplete')
    workload = receipt.get('workload')
    try:
        workload_hash = soak.canonical_sha256(workload)
    except (TypeError, ValueError):
        workload_hash = None
    if receipt.get('workload_sha256') != soak.workload_sha256() or workload_hash != soak.workload_sha256():
        findings.fail('workload_definition_mismatch')
    if soak.workload_sha256() != PINNED_WORKLOAD_SHA256:
        findings.fail('workload_definition_mismatch', field='pinned_soak128_workload')
    if receipt.get('coverage_floors') != soak.coverage_floors():
        findings.fail('workload_definition_mismatch', field='coverage_floors')
    harness = receipt.get('harness') if isinstance(receipt.get('harness'), dict) else {}
    if not isinstance(soak_sha256, str) or harness.get('soak.py') != soak_sha256:
        findings.fail('harness_source_mismatch')
    lock = receipt.get('gpu_lock') if isinstance(receipt.get('gpu_lock'), dict) else {}
    if lock.get('mode') not in soak.GPU_LOCK_MODES or not isinstance(lock.get('path'), str) \
            or pathlib.PurePosixPath(lock['path']).name != soak.GPU_LOCK_NAME:
        findings.fail('gpu_lock_not_held', mode=lock.get('mode') if isinstance(lock.get('mode'), str) else None)
    started, finished = _parse_time(soak, receipt.get('started_utc')), _parse_time(soak, receipt.get('finished_utc'))
    if started is None or finished is None:
        findings.fail('receipt_time_invalid')
    elif is_number(elapsed) and abs((finished - started).total_seconds() - elapsed) > CLOCK_SKEW_LIMIT_S:
        findings.fail('clock_inconsistent', wall_s=(finished - started).total_seconds(), monotonic_s=elapsed)
    summary.update(elapsed_s=elapsed, required_seconds=required, complete=receipt.get('complete'),
                   termination_reason=receipt.get('termination_reason'), status=receipt.get('status'),
                   iterations=receipt.get('iterations'), harness=harness)
    return started, finished, elapsed if is_number(elapsed) else 0.0


def _check_candidate(receipt, soak, identity, profile, run, run_copy, expected_profile, expected_identity_sha256,
                     findings, summary):
    """The run must be the frozen candidate the caller names, not just internally consistent."""
    try:
        identity_sha256 = soak.canonical_sha256(identity) if identity is not None else None
    except (TypeError, ValueError):
        identity_sha256 = None
    bound = receipt.get('candidate') if isinstance(receipt.get('candidate'), dict) else {}
    mismatched = [field for field, ok in (
        ('profile', profile == expected_profile),
        ('identity_sha256', identity_sha256 is not None and identity_sha256 == expected_identity_sha256),
        ('runner_candidate', bound.get('profile') == expected_profile
         and bound.get('identity_sha256') == expected_identity_sha256)) if not ok]
    if mismatched:
        findings.fail('candidate_identity_mismatch', fields=mismatched)
    try:
        copy_matches = isinstance(run_copy, dict) and soak.canonical_sha256(run_copy) == soak.canonical_sha256(run)
    except (TypeError, ValueError):
        copy_matches = False
    if not copy_matches:
        findings.fail('run_copy_mismatch')
    summary['candidate'] = {'expected_profile': expected_profile, 'expected_identity_sha256': expected_identity_sha256,
                            'run_profile': profile, 'run_identity_sha256': identity_sha256,
                            'runner_candidate': bound or None, 'match': not mismatched,
                            'run_copy_matches': copy_matches}


def _check_identity(receipt, soak, validate_run, findings, summary, elapsed, run_copy, expected_profile,
                    expected_identity_sha256):
    run = receipt.get('run')
    identity = None
    try:
        identity = validate_run(run)
    except Exception:
        findings.fail('run_receipt_invalid')
    run = run if isinstance(run, dict) else {}
    run_id, profile = run.get('run_id'), run.get('profile')
    _check_candidate(receipt, soak, identity, profile, run, run_copy, expected_profile, expected_identity_sha256,
                     findings, summary)
    configuration = run.get('configuration') if isinstance(run.get('configuration'), dict) else {}
    bound = receipt.get('identity') if isinstance(receipt.get('identity'), dict) else {}
    try:
        receipt_hash = soak.canonical_sha256(run)
    except (TypeError, ValueError):
        receipt_hash = None
    if (identity is not None and bound.get('identity_sha256') != soak.canonical_sha256(identity)) \
            or bound.get('receipt_sha256') != receipt_hash or bound.get('run_id') != run_id \
            or bound.get('profile') != profile:
        findings.fail('identity_hash_mismatch')
    if not soak.unit_active(bound.get('unit_initial')):
        findings.fail('unit_identity_missing')
    raw_checks = bound.get('checks') if isinstance(bound.get('checks'), list) else []
    checks = [check for check in raw_checks if isinstance(check, dict)]
    if not checks or len(checks) != len(raw_checks):
        findings.fail('identity_checks_missing')
    checks.sort(key=lambda check: check.get('t_s') if is_number(check.get('t_s')) else math.inf)
    for check in checks:
        if not is_number(check.get('t_s')):
            findings.fail('identity_checks_missing', reason='invalid_time')
        if check.get('ok') is not True or check.get('codes'):
            findings.fail('identity_check_failed', t_s=check.get('t_s'), codes=check.get('codes'))
    times = [check['t_s'] for check in checks if is_number(check.get('t_s'))]
    sources = [check['t_s'] for check in checks if check.get('kind') == 'source' and is_number(check.get('t_s'))]
    if not sources or sources[0] > 0 or sources[-1] < elapsed:
        findings.fail('identity_source_check_missing', first=sources[0] if sources else None,
                      last=sources[-1] if sources else None)
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    if times and (max(gaps, default=0.0) > IDENTITY_GAP_LIMIT_S or times[-1] < elapsed):
        findings.fail('identity_check_gap', max_gap_s=max(gaps, default=0.0), last_t_s=times[-1])
    summary['identity'] = {'profile': profile, 'run_id': run_id, 'identity_sha256': bound.get('identity_sha256'),
                           'checks': len(checks), 'source_checks': len(sources),
                           'max_check_gap_s': max(gaps, default=None)}
    return run_id, profile, configuration.get('context')


def _check_rows(rows, iterations, soak, findings, summary):
    passed = []
    if not rows:
        findings.fail('no_workload_rows')
        return passed
    if not nonnegative_int(iterations) or iterations != len({row['iteration'] for row in rows if isinstance(row, dict)
                                                              and nonnegative_int(row.get('iteration'))}):
        findings.fail('iteration_sequence_invalid', reason='iteration_count')
    groups, previous = {}, -1
    for row in rows:
        valid = (isinstance(row, dict) and nonnegative_int(row.get('iteration')) and row.get('case') in soak.CASES
                 and isinstance(row.get('requests'), list) and isinstance(row.get('facts'), dict)
                 and type(row.get('pass')) is bool)
        if not valid:
            findings.fail('row_schema_invalid')
            continue
        if row['iteration'] < previous:
            findings.fail('iteration_sequence_invalid', iteration=row['iteration'])
        previous = row['iteration']
        groups.setdefault(row['iteration'], []).append(row)
    if sorted(groups) != list(range(len(groups))):
        findings.fail('iteration_sequence_invalid', reason='non_contiguous')
    for iteration, group in sorted(groups.items()):
        plan = soak.schedule(iteration)
        actual, expected = [row['case'] for row in group], plan['cases']
        if actual != expected:
            remaining = iter(expected)
            if all(case in remaining for case in actual):  # ordered subsequence: something was not executed
                findings.fail('scheduled_case_missing', iteration=iteration,
                              missing=[case for case in expected if case not in actual])
            else:
                findings.fail('unexpected_case', iteration=iteration)
        for row in group:
            case = row['case']
            if case in ('fresh', 'prefix') and (row.get('length') != plan['length'] or row.get('seed') != plan['seed']
                                                or row.get('document_variant') != plan['document_variant']):
                findings.fail('row_schedule_mismatch', iteration=iteration, case=case)
                continue
            facts_ok = all(row['facts'].get(key) is True for key in soak.REQUIRED_FACTS[case])
            if row['pass'] and not facts_ok:
                findings.fail('row_facts_inconsistent', iteration=iteration, case=case)
                continue
            if not row['pass']:
                findings.fail('case_failed', iteration=iteration, case=case, failure=row.get('failure'))
                continue
            roles = [entry.get('role') if isinstance(entry, dict) else None for entry in row['requests']]
            if case == 'cancel':
                valid = (len(roles) >= 2 and roles[0] == 'cancel_stream' and roles[-1] == 'cancel_followup'
                         and all(role == 'cancel_followup_rejected' for role in roles[1:-1])
                         and len(roles) - 1 <= soak.CANCEL_ATTEMPTS)
                measurements = row.get('measurements') if isinstance(row.get('measurements'), dict) else {}
                valid = valid and is_number(measurements.get('recovery_s')) and \
                    measurements.get('followup_attempts') == len(roles) - 1
            else:
                valid = roles == EXPECTED_ROLES[case]
            if not valid:
                findings.fail('request_roles_invalid', iteration=iteration, case=case)
                continue
            passed.append(row)
    summary['rows'] = {'total': len(rows), 'passed': len(passed), 'iterations': len(groups)}
    return passed


def _check_joins(rows, passed, telemetry, run_id, soak, findings, summary):
    requests = telemetry.get('requests') if isinstance(telemetry.get('requests'), dict) else {}
    engine = telemetry.get('engine') if isinstance(telemetry.get('engine'), dict) else {}
    passed_ids = {id(row) for row in passed}
    issued, sessions, joined_keys = set(), set(), set()
    fresh_prompt = {}
    counts = {'client_requests': 0, 'server_rows': 0, 'engine_joins': 0}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('requests'), list):
            continue
        for entry in row['requests']:
            if not isinstance(entry, dict):
                findings.fail('request_id_missing', iteration=row.get('iteration'))
                continue
            role, rid = entry.get('role'), entry.get('request_id')
            where = {'iteration': row.get('iteration'), 'role': role}
            counts['client_requests'] += 1
            if not isinstance(rid, str) or not REQUEST_ID.fullmatch(rid):
                findings.fail('request_id_missing', **where)
                continue
            if rid in issued:
                findings.fail('request_id_duplicate', **where)
                continue
            issued.add(rid)
            if id(row) not in passed_ids:
                continue  # already a failure; the id still counts as our own traffic
            server = requests.get(rid)
            if not isinstance(server, dict):
                findings.fail('request_row_missing', **where)
                continue
            counts['server_rows'] += 1
            if server.get('kind') != 'generation':
                findings.fail('request_row_invalid', **where)
            if server.get('run_id') != run_id or entry.get('inline_serving_run_id') not in (None, run_id):
                findings.fail('serving_run_id_mismatch', **where)
            if server.get('status') != entry.get('status'):
                findings.fail('request_status_mismatch', **where)
            if isinstance(server.get('session_id'), str) and server['session_id']:
                sessions.add(server['session_id'])
            else:
                findings.fail('session_id_missing', **where)
            # The request actually executed (server-recorded sampling) must be the planned one. A 429 refused
            # at admission, before the body is parsed, has no sampling; if recorded it must match too.
            if not (role == 'cancel_followup_rejected' and 'sampling' not in server) and not sampling_matches(
                    server.get('sampling'), expected_sampling(soak, role, row.get('iteration'))):
                findings.fail('workload_request_mismatch', **where)
            if role == 'cancel_followup_rejected':
                if server.get('status') != 429 or server.get('outcome') != 'rejected':
                    findings.fail('request_outcome_invalid', **where)
                continue
            if role == 'cancel_stream':
                completion = server.get('completion_tokens')
                if server.get('status') != 200 or server.get('outcome') != 'cancelled':
                    findings.fail('cancel_outcome_invalid', **where)
                if not (nonnegative_int(completion) and completion < soak.CANCEL_MAX_TOKENS):
                    findings.fail('cancel_not_interrupted', **where)
            elif server.get('status') != 200 or server.get('outcome') != 'completed' or server.get('counts_final') is not True:
                findings.fail('request_outcome_invalid', **where)
            peaks = server.get('peaks') if isinstance(server.get('peaks'), dict) else {}
            missing = [key for key in REQUEST_RESOURCE_KEYS if not is_number(peaks.get(key))]
            if missing or not (type(server.get('resource_samples')) is int and server['resource_samples'] >= 1):
                findings.fail('request_resources_missing', missing=missing, **where)
            uid = server.get('engine_uid')
            if not nonnegative_int(uid):
                findings.fail('engine_uid_missing', **where)
                continue
            key = f'{run_id}:{uid}'
            if key in joined_keys:
                findings.fail('engine_join_ambiguous', **where)
                continue
            joined_keys.add(key)
            row_engine = engine.get(key)
            if not isinstance(row_engine, dict):
                findings.fail('engine_row_missing', **where)
                continue
            counts['engine_joins'] += 1
            if row_engine.get('run_id') != run_id or row_engine.get('uid') != uid:
                findings.fail('engine_run_id_mismatch', **where)
            if role == 'cancel_stream' and row_engine.get('cancelled') is not True:
                findings.fail('cancel_engine_not_cancelled', **where)
            if role != 'cancel_stream' and row_engine.get('cancelled') is not False:
                findings.fail('engine_cancelled_unexpectedly', **where)
            missing = [name for name in ENGINE_NUMERIC_KEYS if not is_number(row_engine.get(name))]
            if missing:
                findings.fail('engine_resources_missing', missing=missing, **where)
            prompt, completion = server.get('prompt_tokens'), server.get('completion_tokens')
            if role == 'fresh':
                needed = math.ceil(soak.PROMPT_COVERAGE_RATIO * row.get('length', 0))
                if not (nonnegative_int(prompt) and prompt >= needed and nonnegative_int(completion)
                        and 0 < completion <= soak.FRESH_MAX_TOKENS):
                    findings.fail('fresh_server_tokens_invalid', **where)
                fresh_prompt[row.get('iteration')] = prompt
            elif role == 'prefix':
                base = fresh_prompt.get(row.get('iteration'))
                if not (nonnegative_int(prompt) and nonnegative_int(base) and prompt > base and nonnegative_int(completion)
                        and 0 < completion <= soak.PREFIX_MAX_TOKENS):
                    findings.fail('prefix_server_tokens_invalid', **where)
            elif role == 'tool_call' and server.get('finish_reason') != 'tool_calls':
                findings.fail('tool_server_finish_invalid', **where)
    foreign = [rid for rid in requests if rid not in issued]
    if foreign:
        findings.fail('foreign_generation_traffic', count=len(foreign))
    wrong_run = [key for key, row in engine.items() if not isinstance(row, dict) or row.get('run_id') != run_id]
    if wrong_run:
        findings.fail('engine_run_id_mismatch', count=len(wrong_run))
    unjoined = [key for key in engine if key not in joined_keys]
    if unjoined and not findings.buckets['fail'].get('case_failed'):
        findings.fail('unjoined_engine_rows', count=len(unjoined))
    duplicates = telemetry.get('duplicates') if isinstance(telemetry.get('duplicates'), dict) else None
    if duplicates is None or set(duplicates) != {'requests', 'engine'} \
            or any(not nonnegative_int(value) or value for value in duplicates.values()):
        findings.fail('telemetry_duplicate_rows', duplicates=duplicates)
    invalid = telemetry.get('invalid_rows') if isinstance(telemetry.get('invalid_rows'), dict) else None
    if invalid is None or set(invalid) != {'requests', 'engine', 'state'} \
            or any(not nonnegative_int(value) or value for value in invalid.values()):
        findings.fail('telemetry_invalid_rows', invalid=invalid)
    counts.update(foreign_generation_rows=len(foreign), unjoined_engine_rows=len(unjoined))
    summary['joins'] = counts
    return sessions


def _check_coverage(passed, soak, findings, summary):
    floors = soak.coverage_floors()
    counts = {key: 0 for key in floors}
    for row in passed:
        key = f"fresh@{row.get('length')}" if row['case'] == 'fresh' else row['case']
        counts[key] = counts.get(key, 0) + 1
    for key, need in floors.items():
        if counts.get(key, 0) < need:
            findings.inconclusive('insufficient_coverage', case=key, passed=counts.get(key, 0), required=need)
    summary['coverage'] = {'passed': counts, 'floors': floors}


def _check_state(telemetry, soak, findings, summary, started, finished, profile, context, sessions):
    state = telemetry.get('state') if isinstance(telemetry.get('state'), list) else []
    if not state:
        findings.fail('state_rows_missing')
    instances, stamps, previous = set(), [], None
    maxima = {key: 0 for key, _ in STATE_COUNTERS}
    for index, row in enumerate(state):
        if not isinstance(row, dict):
            findings.fail('state_row_invalid', index=index)
            continue
        missing = [key for key in STATE_RESOURCE_KEYS if not is_number(row.get(key))]
        if missing:
            findings.fail('state_resources_missing', index=index, missing=missing)
        for key, code in STATE_COUNTERS:
            value = row.get(key)
            if not nonnegative_int(value):
                findings.fail('state_counters_missing', index=index, counter=key)
            elif value > 0:
                findings.fail(code, index=index, value=value)
                maxima[key] = max(maxima[key], value)
        if row.get('resource_sample_stale'):
            findings.fail('telemetry_resource_sample_stale', index=index)
        if row.get('model_ready') is not True:
            findings.fail('model_not_ready', index=index)
        if row.get('profile') != profile:
            findings.fail('state_profile_mismatch', index=index)
        if row.get('context_limit') != context:
            findings.fail('state_context_mismatch', index=index)
        instance = row.get('instance_id')
        if isinstance(instance, str) and instance:
            instances.add(instance)
        else:
            findings.fail('instance_id_missing', index=index)
        if isinstance(row.get('session_id'), str) and row['session_id']:
            sessions.add(row['session_id'])
        else:
            findings.fail('session_id_missing', index=index)
        moment = _parse_time(soak, row.get('ts'))
        if moment is None:
            findings.fail('state_timestamp_invalid', index=index)
            continue
        if previous is not None and moment < previous:
            findings.fail('state_time_regression', index=index)
        previous = moment
        stamps.append(moment)
    if len(instances) > 1:
        findings.fail('service_instance_changed', instances=len(instances))
    if len(sessions) > 1:
        findings.fail('service_session_changed', sessions=len(sessions))
    elif not sessions:
        findings.fail('session_id_missing', scope='run')
    events = telemetry.get('events')
    if not isinstance(events, list) or events:
        findings.fail('service_stall_event', events=len(events) if isinstance(events, list) else None)
    stamps.sort()
    max_gap = max(((later - earlier).total_seconds() for earlier, later in zip(stamps, stamps[1:])), default=None)
    start_edge = end_edge = None
    if stamps and started is not None and finished is not None:
        start_edge = (stamps[0] - started).total_seconds()
        end_edge = (finished - stamps[-1]).total_seconds()
        if start_edge > STATE_EDGE_LIMIT_S:
            findings.fail('state_coverage_start', seconds=start_edge)
        if end_edge > STATE_EDGE_LIMIT_S:
            findings.fail('state_coverage_end', seconds=end_edge)
    if max_gap is not None and max_gap > STATE_GAP_LIMIT_S:
        findings.fail('state_gap', max_gap_s=max_gap)
    free = [row['gpu_free_bytes'] for row in state if isinstance(row, dict) and is_number(row.get('gpu_free_bytes'))]
    depth = [row['metrics_queue_depth'] for row in state
             if isinstance(row, dict) and is_number(row.get('metrics_queue_depth'))]
    summary['state'] = {'rows': len(state), 'max_gap_s': max_gap, 'start_edge_s': start_edge, 'end_edge_s': end_edge,
                        'counter_maxima': maxima, 'min_gpu_free_bytes': min(free, default=None),
                        'max_metrics_queue_depth': max(depth, default=None), 'stall_events': len(events)
                        if isinstance(events, list) else None}
    summary['vram'] = _check_vram_floor(state, soak, findings)
    return state


def _check_vram_floor(state, soak, findings):
    """Free VRAM must stay >= VRAM_FLOOR_BYTES at every state sample. Time below the floor is the sum of the
    intervals that start at a below-floor sample (each capped at the state gap limit)."""
    samples = sorted((moment, row['gpu_free_bytes']) for row in state
                     if isinstance(row, dict) and is_number(row.get('gpu_free_bytes'))
                     for moment in [_parse_time(soak, row.get('ts'))] if moment is not None)
    out = {'floor_bytes': VRAM_FLOOR_BYTES, 'samples': len(samples)}
    if not samples:
        findings.fail('vram_floor_unverified')
        return out
    free = sorted(value for _, value in samples)
    below = [value < VRAM_FLOOR_BYTES for _, value in samples]
    seconds = sum(min((later - earlier).total_seconds(), STATE_GAP_LIMIT_S)
                  for (earlier, _), (later, _), low in zip(samples, samples[1:], below) if low)
    rank = lambda q: free[min(len(free) - 1, int(q * (len(free) - 1)))]
    out.update(min_gpu_free_bytes=free[0], p1_gpu_free_bytes=rank(.01), p5_gpu_free_bytes=rank(.05),
               median_gpu_free_bytes=statistics.median(free), samples_below_floor=sum(below),
               minutes_below_floor=round(seconds / 60, 2))
    if free[0] < VRAM_FLOOR_BYTES:
        findings.fail('vram_floor_breached', min_gpu_free_bytes=free[0], floor_bytes=VRAM_FLOOR_BYTES,
                      minutes_below_floor=out['minutes_below_floor'])
    return out


def _check_readers(telemetry, findings, summary):
    readers = telemetry.get('readers') if isinstance(telemetry.get('readers'), dict) else {}
    out = {}
    for name in ('requests', 'engine', 'state'):
        stats = readers.get(name)
        if not isinstance(stats, dict):
            findings.fail('telemetry_reader_missing', reader=name)
            continue
        for key, code in READER_COUNTERS:
            value = stats.get(key)
            if not nonnegative_int(value) or value:
                findings.fail(code, reader=name, value=value)
        out[name] = {key: stats.get(key) for key in ('rotations', 'recovered_files', 'loss_events', 'truncations',
                                                    'malformed_lines', 'read_errors', 'lines')}
    monitor = telemetry.get('monitor') if isinstance(telemetry.get('monitor'), dict) else {}
    if not isinstance(monitor.get('failures'), list) or monitor['failures']:
        findings.fail('telemetry_monitor_failed', failures=monitor.get('failures'))
    background = monitor.get('background') if isinstance(monitor.get('background'), dict) else {}
    if not (background.get('started') is True and background.get('alive_before_stop') is True
            and background.get('joined') is True):
        findings.fail('telemetry_monitor_not_running')
    summary['readers'] = out
    summary['monitor'] = {'polls': monitor.get('polls'), 'max_poll_interval_s': monitor.get('max_poll_interval_s')}


def _check_memory(passed, telemetry, state, soak, findings, started, elapsed):
    requests = telemetry.get('requests') if isinstance(telemetry.get('requests'), dict) else {}
    engine = telemetry.get('engine') if isinstance(telemetry.get('engine'), dict) else {}
    flags, insufficient, per_length = [], [], {}
    for length in soak.LENGTHS:
        series = {}
        for row in passed:
            if row['case'] != 'fresh' or row.get('length') != length:
                continue
            server = requests.get(row['requests'][0].get('request_id'))
            if not isinstance(server, dict):
                continue
            peaks = server.get('peaks') if isinstance(server.get('peaks'), dict) else {}
            for metric in tuple(REQUEST_MEMORY_FLAGS) + MEMORY_REPORT_ONLY:
                if is_number(peaks.get(metric)):
                    series.setdefault('request.' + metric, []).append(peaks[metric])
            engine_row = engine.get(f"{server.get('run_id')}:{server.get('engine_uid')}")
            if isinstance(engine_row, dict):
                for metric in ENGINE_MEMORY_FLAGS:
                    if is_number(engine_row.get(metric)):
                        series.setdefault('engine.' + metric, []).append(engine_row[metric])
        results = {}
        gated = {**{'request.' + k: v for k, v in REQUEST_MEMORY_FLAGS.items()},
                 **{'engine.' + k: v for k, v in ENGINE_MEMORY_FLAGS.items()}}
        for metric in list(gated) + ['request.' + k for k in MEMORY_REPORT_ONLY]:
            result = compare_series(series.get(metric, []), soak.MEMORY_WARMUP_POINTS, soak.MEMORY_WINDOW_POINTS)
            result['gated_threshold_bytes'] = gated.get(metric)
            if result['status'] != 'ok':
                if metric in gated:
                    insufficient.append({'length': length, 'metric': metric, 'points': result['points']})
            elif metric in gated and result['late_minus_early'] > gated[metric]:
                result['threshold_exceeded'] = True
                flags.append({'length': length, 'metric': metric, 'late_minus_early': result['late_minus_early'],
                              'threshold': gated[metric]})
            results[metric] = result
        per_length[str(length)] = results
    idle = {'status': 'insufficient', 'warmup_s': IDLE_WARMUP_S, 'window_s': IDLE_WINDOW_S,
            'min_samples': IDLE_MIN_SAMPLES}
    early, late = [], []
    if started is not None:
        for row in state:
            if not isinstance(row, dict) or not nonnegative_int(row.get('active_inference')) or row['active_inference']:
                continue
            moment = _parse_time(soak, row.get('ts'))
            if moment is None:
                continue
            offset = (moment - started).total_seconds()
            if IDLE_WARMUP_S <= offset < IDLE_WARMUP_S + IDLE_WINDOW_S:
                early.append(row)
            if elapsed - IDLE_WINDOW_S <= offset <= elapsed:
                late.append(row)
    idle.update(early_samples=len(early), late_samples=len(late))
    if elapsed >= IDLE_WARMUP_S + 2 * IDLE_WINDOW_S and len(early) >= IDLE_MIN_SAMPLES and len(late) >= IDLE_MIN_SAMPLES:
        idle['status'] = 'ok'
        idle['metrics'] = {}
        for metric in tuple(IDLE_MEMORY_FLAGS) + MEMORY_REPORT_ONLY:
            before = [row[metric] for row in early if is_number(row.get(metric))]
            after = [row[metric] for row in late if is_number(row.get(metric))]
            if len(before) < IDLE_MIN_SAMPLES or len(after) < IDLE_MIN_SAMPLES:
                idle['metrics'][metric] = {'status': 'insufficient'}
                if metric in IDLE_MEMORY_FLAGS:
                    insufficient.append({'idle_metric': metric})
                continue
            delta = statistics.median(after) - statistics.median(before)
            item = {'status': 'ok', 'early_median': statistics.median(before), 'late_median': statistics.median(after),
                    'late_minus_early': delta, 'gated_threshold_bytes': IDLE_MEMORY_FLAGS.get(metric)}
            if metric in IDLE_MEMORY_FLAGS and delta > IDLE_MEMORY_FLAGS[metric]:
                item['threshold_exceeded'] = True
                flags.append({'idle': True, 'metric': metric, 'late_minus_early': delta,
                              'threshold': IDLE_MEMORY_FLAGS[metric]})
            idle['metrics'][metric] = item
    else:
        insufficient.append({'idle': 'too_few_idle_state_samples'})
    for flag in flags:
        findings.fail('memory_growth_above_threshold', **flag)
    for item in insufficient:
        findings.inconclusive('memory_insufficient_samples', **item)
    status = ('THRESHOLD_EXCEEDED' if flags else 'INCONCLUSIVE' if insufficient
              else 'NO_THRESHOLD_EXCEEDED_REVIEW_REQUIRED')
    return {'status': status, 'leak_verdict': 'not_determined_automatically',
            'method': 'per prompt length: median of request/engine values for fresh requests 3..7 vs last 5; '
                      'idle state samples 600..1800 s vs last 1200 s',
            'per_length': per_length, 'idle_state': idle, 'flags': flags, 'insufficient': insufficient}


def _case_summary(passed, telemetry):
    requests = telemetry.get('requests') if isinstance(telemetry.get('requests'), dict) else {}
    recovery, after_disconnect, attempts, ttft = [], [], [], {}
    hits = total_prefix = 0
    for row in passed:
        if row['case'] == 'cancel':
            measured = row['measurements']
            recovery.append(measured['recovery_s'])
            if is_number(measured.get('recovery_after_disconnect_s')):
                after_disconnect.append(measured['recovery_after_disconnect_s'])
            attempts.append(measured['followup_attempts'])
        elif row['case'] == 'prefix':
            total_prefix += 1
            server = requests.get(row['requests'][0].get('request_id')) or {}
            hits += int(nonnegative_int(server.get('cached_prompt_tokens')) and server['cached_prompt_tokens'] > 0)
        elif row['case'] == 'fresh' and is_number(row['requests'][0].get('ttft_content_s')):
            ttft.setdefault(str(row['length']), []).append(row['requests'][0]['ttft_content_s'])

    def stats(values):
        return {'n': len(values), 'median': statistics.median(values), 'max': max(values)} if values else {'n': 0}
    return {'cancel_recovery_s': stats(recovery), 'cancel_recovery_after_disconnect_s': stats(after_disconnect),
            'cancel_followup_attempts_max': max(attempts, default=None),
            'prefix_cache_hits': {'hits': hits, 'prefix_requests': total_prefix,
                                  'note': 'measured, not required (real-answer continuation)'},
            'fresh_ttft_content_s': {length: stats(values) for length, values in sorted(ttft.items(), key=lambda x: int(x[0]))}}


def evaluate(receipt, *, soak, validate_run, soak_sha256, expected_name, expected_profile, expected_identity_sha256,
             run_copy):
    """All bindings are mandatory: the soak.py hash, the receipt name, the frozen candidate
    (profile + canonical identity sha256) and the NAME-run.json copy of the active-run receipt."""
    findings, summary = Findings(), {}
    memory = None
    if not isinstance(receipt, dict) or receipt.get('schema') != soak.SCHEMA \
            or receipt.get('schema_version') != soak.SCHEMA_VERSION:
        findings.fail('receipt_schema_invalid')
    else:
        started, finished, elapsed = _check_receipt(receipt, soak, findings, summary, soak_sha256, expected_name)
        run_id, profile, context = _check_identity(receipt, soak, validate_run, findings, summary, elapsed, run_copy,
                                                   expected_profile, expected_identity_sha256)
        rows = receipt.get('rows') if isinstance(receipt.get('rows'), list) else []
        telemetry = receipt.get('telemetry') if isinstance(receipt.get('telemetry'), dict) else {}
        if not isinstance(receipt.get('telemetry'), dict):
            findings.fail('telemetry_missing')
        passed = _check_rows(rows, receipt.get('iterations'), soak, findings, summary)
        sessions = _check_joins(rows, passed, telemetry, run_id, soak, findings, summary)
        _check_coverage(passed, soak, findings, summary)
        state = _check_state(telemetry, soak, findings, summary, started, finished, profile, context, sessions)
        _check_readers(telemetry, findings, summary)
        memory = _check_memory(passed, telemetry, state, soak, findings, started, elapsed)
        summary['cases'] = _case_summary(passed, telemetry)
    fail, inconclusive = findings.buckets['fail'], findings.buckets['inconclusive']
    verdict = 'FAIL' if fail else 'INCONCLUSIVE' if inconclusive else 'PASS'
    # A short pilot (soak.py --seconds 30, new name) cannot pass the gate; this view
    # only shows whether anything other than its length failed.
    executed = {}
    for row in (receipt.get('rows') if isinstance(receipt, dict) and isinstance(receipt.get('rows'), list) else []):
        if isinstance(row, dict) and row.get('pass') is True and row.get('case') in soak.CASES:
            executed[row['case']] = executed.get(row['case'], 0) + 1
    pilot_view = {'passed_cases': executed, 'all_case_types_passed': set(executed) == set(soak.CASES),
                  'failures_excluding_duration': sorted(set(fail) - PILOT_DURATION_CODES)}
    return {'schema': ANALYSIS_SCHEMA, 'schema_version': ANALYSIS_SCHEMA_VERSION, 'verdict': verdict,
            'two_hour_execution_gate_pass': verdict == 'PASS',
            'candidate': {'profile': expected_profile, 'identity_sha256': expected_identity_sha256},
            'gate_reasons': sorted(set(fail) | set(inconclusive)), 'failure_reasons': sorted(fail),
            'inconclusive_reasons': sorted(inconclusive), 'reason_details': {**fail, **inconclusive},
            'summary': summary, 'memory': memory, 'pilot_view': pilot_view,
            'thresholds': {'state_gap_s': STATE_GAP_LIMIT_S, 'state_edge_s': STATE_EDGE_LIMIT_S,
                           'identity_gap_s': IDENTITY_GAP_LIMIT_S, 'clock_skew_s': CLOCK_SKEW_LIMIT_S,
                           'request_memory_flags_bytes': REQUEST_MEMORY_FLAGS,
                           'engine_memory_flags_bytes': ENGINE_MEMORY_FLAGS, 'idle_memory_flags_bytes': IDLE_MEMORY_FLAGS,
                           'vram_floor_bytes': VRAM_FLOOR_BYTES,
                           'required_seconds': soak.REQUIRED_SECONDS,
                           'pinned_workload_sha256': PINNED_WORKLOAD_SHA256},
            'interpretation': 'PASS means the two-hour execution evidence is complete, the run was the named '
                              'frozen candidate and every gated check held. It is not a proof that no leak exists; '
                              'memory figures still need review.'}


def write_exclusive(path, data):
    """Atomic, never-overwriting JSON write: complete temp file, then link()."""
    path = pathlib.Path(path)
    payload = json.dumps(data, indent=1, ensure_ascii=True, allow_nan=False) + '\n'
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fail-closed soak v2 gate')
    parser.add_argument('--name', default='soak128')
    parser.add_argument('--profile', required=True, help='frozen candidate profile')
    parser.add_argument('--expected-identity', required=True,
                        help='sha256 of the candidate canonical identity, or a frozen manifest with arms[profile]')
    parser.add_argument('--reports-dir', default=None, help='defaults to bench.ROOT/reports')
    parser.add_argument('--tag', default=None, help='write NAME-analysis-TAG.json instead of NAME-analysis.json')
    args = parser.parse_args(argv)
    soak = load_soak()
    try:
        soak.validate_name(args.name)
    except ValueError as error:
        parser.error(str(error))
    if args.tag is not None and not TAG_PATTERN.fullmatch(args.tag):
        parser.error('--tag must be [a-z0-9]{1,16}')
    try:
        candidate = soak.resolve_candidate(args.profile, args.expected_identity)
    except (OSError, ValueError) as error:
        parser.error(f'invalid --profile/--expected-identity: {error}')
    import bench
    reports = pathlib.Path(args.reports_dir) if args.reports_dir else bench.ROOT / 'reports'
    source = reports / f'{args.name}.json'
    run_copy_file = soak.run_copy_path(reports, args.name)
    target = reports / (f'{args.name}-analysis.json' if args.tag is None else f'{args.name}-analysis-{args.tag}.json')
    if target.exists():
        parser.error(f'{target.name} exists; analyses are never overwritten (use --tag)')
    data = source.read_bytes()
    try:
        receipt = soak.strict_loads(data)
    except ValueError:
        receipt = None
    try:
        run_copy_data = run_copy_file.read_bytes()
        run_copy = soak.strict_loads(run_copy_data)
    except (OSError, ValueError):
        run_copy_data = run_copy = None
    soak_sha256 = soak.file_sha256(HERE / 'soak.py')
    report = evaluate(receipt, soak=soak, soak_sha256=soak_sha256, expected_name=args.name,
                      expected_profile=candidate['profile'], expected_identity_sha256=candidate['identity_sha256'],
                      run_copy=run_copy, validate_run=lambda run: bench.validate_receipt(run, candidate['profile']))
    report.update(source=source.name, source_sha256=hashlib.sha256(data).hexdigest(),
                  run_copy=run_copy_file.name,
                  run_copy_sha256=hashlib.sha256(run_copy_data).hexdigest() if run_copy_data is not None else None,
                  analysis_sha256=soak.file_sha256(pathlib.Path(__file__).resolve()), soak_sha256=soak_sha256,
                  candidate=candidate)
    write_exclusive(target, report)
    identity = report['summary'].get('identity') or {}
    print(json.dumps({'verdict': report['verdict'], 'two_hour_execution_gate_pass': report['two_hour_execution_gate_pass'],
                      'gate_reasons': report['gate_reasons'], 'profile': identity.get('profile'),
                      'run_id': identity.get('run_id'),
                      'identity_sha256': (report['summary'].get('candidate') or {}).get('run_identity_sha256'),
                      'expected_profile': candidate['profile'],
                      'expected_identity_sha256': candidate['identity_sha256'],
                      'analysis': str(target)}), flush=True)
    return EXIT_CODES[report['verdict']]


if __name__ == '__main__':
    sys.exit(main())
