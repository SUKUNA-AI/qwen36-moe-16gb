#!/usr/bin/env python3
"""Prompt-lookup qualification against the normal overlap path; CPU-only plan by default.

Three isolated arms share the exact frozen base profile, inputs, sampling, seed and
thinking setting; each arm is a fresh qwen36-experiment process:

  overlap    reference = normal production path (overlap scheduler ON, speculation OFF)
  nooverlap  diagnostic (overlap scheduler OFF, speculation OFF): attributes the cost
             of leaving the overlap scheduler, which the lookup path requires
  lookup     overlap scheduler OFF + QWEN36_SPEC=lookup, window 3, sampled speculation OFF

Acceptance is judged ONLY lookup-vs-overlap. Greedy rows must be byte-identical to the
overlap reference; the default-sampling row must be byte-identical too and must show
that speculation never ran on it (greedy-only applicability). --pairs 1 is a screen;
a multi-pair --run requires --screen-report of a SCREEN_PASS with the identical harness,
configs, fixtures and runtime build. Only five complete pairs can reach
ACCEPTANCE_GATES_PASS, and qualified stays false (release qualification, general
quality gates and soak are separate). Any failure, missing pair, error or incomplete
receipt yields decision OFF.

--run is explicit, must run inside WSL in the owner's GPU slot, takes the non-blocking
GPU-exclusive flock (or shares an inherited one from `flock -n <lock> ...`), refuses
active inference/experiments, and controls only its own qwen36-experiment invocation.
No runtime source, base profile or release default is edited. This file deliberately
does not import bench.py (paired-harness V5 changes in parallel). Readiness polls the
authenticated /health at 1 Hz and HTTP 429 admissions are retried a bounded number of
times, because APIGuard rate-limits every authenticated route (60-token bucket, +1/s).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
import uuid

ROOT = Path('/home/USER/qwen36-api/optimization')
WORKSPACE = Path('/path/to/qwen36-workspace')
WORKSPACE_OPT = WORKSPACE/'outputs/qwen36-api/optimization'
FORK_SOURCE = WORKSPACE/'work/FreeToken-opt/python/freetoken'
GPU_LOCK = Path('/home/USER/qwen36-api/run/gpu-exclusive.lock')
UNIT = 'qwen36-experiment.service'
PRODUCTION_UNIT = 'qwen36-inference.service'
URL = 'http://127.0.0.1:1919'
SEED = 627319
SCHEMA_VERSION = 2
EXIT_GPU_LOCK_BUSY = 75

ARMS = ('overlap', 'nooverlap', 'lookup')
REFERENCE_ARM = 'overlap'
CANDIDATE_ARM = 'lookup'
DIAGNOSTIC_COMPARISONS = (('lookup', 'nooverlap'), ('nooverlap', 'overlap'))
# Each pair is a full permutation; lookup precedes overlap in pairs 1 and 3 only.
ORDERS = (('overlap', 'nooverlap', 'lookup'),
          ('lookup', 'nooverlap', 'overlap'),
          ('nooverlap', 'overlap', 'lookup'),
          ('lookup', 'overlap', 'nooverlap'),
          ('overlap', 'lookup', 'nooverlap'))

# Preregistered criteria (unchanged thresholds; reference is now the overlap path).
REQUIRED_PAIRS = 5
SPEED_WIN_CASES = ('copy-code', 'quote-ru')
EXERCISE_CASES = ('copy-code', 'quote-ru', 'copy-code-long')
SPEED_WIN_LOWER = 1.05
NO_REGRESSION_LOWER = 1/1.03
SCREEN_SIGNAL = 1.05
CANCEL_MIN_CHUNKS = 32
CANCEL_RECOVERY_MAX_S = 10.0
# APIGuard (guard.py) admits every authenticated route, GET /health included, from one
# 60-token bucket refilled at 1 token/s and answers 429 'Request rate exceeded' with
# Retry-After: 2 when it is empty. uvicorn binds before the ~90-200 s model load, so
# polling /health faster than 1 Hz drains the bucket and the first request after ready
# is rejected. Poll at 1 Hz like check-experiment.py, and retry a 429 admission (guard
# rejects before inference, nothing runs) a bounded number of times.
HEALTH_POLL_INTERVAL_S = 1.0
ADMISSION_RETRY_ATTEMPTS = 8
ADMISSION_RETRY_DEFAULT_S = 2.0
ADMISSION_RETRY_MAX_S = 5.0
TCRIT95 = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305, 5: 2.776445105}

SPEC_FIELDS = ('attempts', 'disabled', 'draft_s', 'verify_s', 'replay_s',
               'total_s', 'proposed', 'accepted', 'emitted', 'publish_s',
               'wall_timing_includes_publication', 'verify_replay_in_engine_gpu_timers',
               'publication_failed')
LOOKUP_SOURCE_FILES = ('engine/speculative.py', 'scheduler/speculative.py', 'scheduler/scheduler.py',
                       'engine/telemetry.py', 'engine/sample.py', 'env.py')
FORBIDDEN_BASE_ENV = ('QWEN36_TEACHER_FIXTURE', 'QWEN36_LOGIT_REFERENCE', 'QWEN36_NSYS_FORWARDS',
                      'QWEN36_KV_CALIBRATE', 'QWEN36_SPEC', 'QWEN36_SPEC_SAMPLING')
QUALITY_KEYS = ('input_sha256', 'content_sha256', 'reasoning_sha256', 'finish_reason')
USAGE_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


class GpuLockBusy(RuntimeError):
    """Another process holds the GPU-exclusive lock; nothing was started."""


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def env_true(value):
    """Mirror of FreeToken env._TO_BOOL for FREETOKEN_* switches."""
    return value is not None and str(value).lower() in ('1', 'true', 'yes')


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def retry_after_s(headers):
    """Bounded numeric Retry-After of a 429 (APIGuard sends '2'); default when absent/invalid."""
    value = None
    try:
        value = next((item for key, item in (headers or {}).items() if str(key).lower() == 'retry-after'), None)
        value = float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        value = None
    if value is None or not math.isfinite(value):
        value = ADMISSION_RETRY_DEFAULT_S
    return min(max(value, 1.0), ADMISSION_RETRY_MAX_S)


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def fixtures():
    code = '\n\n'.join(f'def fixture_{i}(value):\n    return value * {i+1} + {i*i}' for i in range(12))
    quote = '\n'.join(['Очередь хранит задания в порядке поступления. Каждое задание имеет номер и состояние.'] * 5)
    stopped = '\n'.join(['Сначала проверяем входные данные, затем записываем результат.'] * 5)
    journal = '\n'.join(f'Запись {i:04d}: контрольное значение {(i*7919) % 10007}, состояние '
                        f'{"готово" if i % 3 else "ожидание"}.' for i in range(600))
    common = 'Верни только запрошенный текст. Не добавляй пояснений, кавычек или разметки Markdown.'
    copy_prompt = 'Повтори весь текст между BEGIN и END дословно.\nBEGIN\n'
    result = []
    for name, body, expected, maximum, stop, sampling, allow_length in (
        ('copy-code', copy_prompt+code+'\nEND', code, 1024, None, 'greedy', False),
        ('quote-ru', copy_prompt+quote+'\nEND', quote, 512, None, 'greedy', False),
        ('stop-string', copy_prompt+stopped+'\n<END_LOOKUP>\nХвост после маркера.\nEND', stopped, 512,
         ['\n<END_LOOKUP>'], 'greedy', False),
        ('eos-short', 'Сколько будет 17 + 25? Ответь только числом.', '42', 32, None, 'greedy', False),
        # Free greedy generation: no expected text, byte identity with the reference is the gate.
        ('free-code', 'Напиши на Python функцию merge_sort(items), выполняющую стабильную сортировку '
         'слиянием. Только код.', None, 384, None, 'greedy', True),
        # Long prompt index (13422 prompt tokens with the checkpoint chat template) + the same copy task.
        ('copy-code-long', 'Ниже журнал, его не нужно пересказывать.\n'+journal+'\n\n'+copy_prompt+code+'\nEND',
         code, 1024, None, 'greedy', False),
        # Checkpoint generation_config defaults (temperature 1.0, top_k 20, top_p 0.95) + fixed seed:
        # the API default path; lookup must stay inactive and leave the output unchanged.
        ('sampled-default', 'Опиши в трёх предложениях, зачем процессору нужен кэш.', None, 160, None,
         'default', True),
    ):
        messages = [{'role': 'system', 'content': common}, {'role': 'user', 'content': body}]
        result.append({'case': name, 'messages': messages, 'expected': expected, 'max_tokens': maximum,
                       'stop': stop, 'sampling': sampling, 'allow_length': allow_length})
    return result


def fixture(name):
    return next(case for case in fixtures() if case['case'] == name)


# Per-arm warmup right after readiness; not a measured fixture (not in fixture_sha256).
WARMUP = {'case': 'warmup', 'messages': [{'role': 'user', 'content': 'Ответь только числом: 3 + 5.'}],
          'expected': '8', 'max_tokens': 32, 'stop': None, 'sampling': 'greedy', 'allow_length': False}


def payload(case):
    doc = {'model': 'qwen36-35b', 'messages': case['messages'], 'max_tokens': case['max_tokens'],
           'seed': SEED, 'ignore_eos': False, 'chat_template_kwargs': {'enable_thinking': False},
           'stream': True, 'stream_options': {'include_usage': True}}
    if case.get('sampling', 'greedy') == 'greedy':
        # Engine greedy needs temperature<=0 AND top_p==1; omitting top_p would inherit 0.95.
        doc.update(temperature=0, top_p=1)
    if case['stop']:
        doc['stop'] = case['stop']
    return doc


def fixture_manifest():
    return [{'case': case['case'], 'sampling': case['sampling'], 'max_tokens': case['max_tokens'],
             'allow_length': case['allow_length'], 'input_sha256': digest(payload(case)),
             'expected_sha256': None if case['expected'] is None else digest(case['expected'].encode())}
            for case in fixtures()]


def arm_configs(base):
    if not isinstance(base, dict) or not all(key in base for key in ('context', 'chunk', 'memory_ratio', 'env', 'flags')):
        raise ValueError('Base must be an explicit frozen experiment profile')
    if not isinstance(base['env'], dict) or not isinstance(base['flags'], list):
        raise ValueError('Base env/flags have an invalid shape')
    if any(base['env'].get(key) not in (None, '', '0') for key in FORBIDDEN_BASE_ENV):
        raise ValueError('Base profile contains another experiment')
    if env_true(base['env'].get('FREETOKEN_DISABLE_OVERLAP_SCHEDULING')):
        raise ValueError('Base profile must be the normal overlap path')
    flags = [str(flag) for flag in base['flags']]
    for index, flag in enumerate(flags):
        value = flag.split('=', 1)[1] if flag.startswith('--page-size=') else (
            flags[index+1] if flag == '--page-size' and index+1 < len(flags) else None)
        if value is not None and value != '1':
            raise ValueError('Prompt lookup supports only page_size=1')
    arms = {}
    for arm in ARMS:
        config = copy.deepcopy(base)
        config['qualified'] = False
        config['env'].update(FREETOKEN_DISABLE_OVERLAP_SCHEDULING='0' if arm == 'overlap' else '1',
                             QWEN36_SPEC='lookup' if arm == 'lookup' else '',
                             QWEN36_SPEC_WINDOW='3', QWEN36_SPEC_SAMPLING='0',
                             QWEN36_ENGINE_METRICS='1')
        arms[arm] = config
    return arms


def schedule(pairs):
    if not 1 <= pairs <= len(ORDERS):
        raise ValueError('pairs must be in 1..5')
    return [(pair, arm) for pair in range(pairs) for arm in ORDERS[pair]]


def safe_numeric(value):
    """Only numbers/bools and fixed-schema nested dictionaries leave live metrics."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    if isinstance(value, dict):
        return {key: safe_numeric(item) for key, item in value.items()
                if isinstance(key, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,80}', key)
                and (isinstance(item, (int, float, bool, dict)) or item is None)}
    return None


def evaluate(case, response):
    text = response.pop('_content', '')
    reasoning = response.pop('_reasoning', '')
    usage = response.get('usage') or {}
    completion = usage.get('completion_tokens')
    finish = response.get('finish_reason')
    expected = case['expected']
    counted = is_int(completion) and 0 < completion <= case['max_tokens']
    natural = finish == 'stop' and counted and completion < case['max_tokens']
    length = bool(case.get('allow_length')) and finish == 'length' and counted and completion == case['max_tokens']
    response.update(case=case['case'], sampling=case.get('sampling', 'greedy'),
                    input_sha256=digest(payload(case)),
                    expected_sha256=None if expected is None else digest(expected.encode()),
                    content_sha256=digest(text.encode()), reasoning_sha256=digest(reasoning.encode()),
                    content_chars=len(text), reasoning_chars=len(reasoning),
                    expected_match=None if expected is None else text.strip() == expected,
                    unexpected_reasoning=bool(reasoning), budget_not_exhausted=natural)
    content_ok = bool(text.strip()) if expected is None else response['expected_match'] is True
    response['functional_pass'] = bool(response.get('status') == 200 and content_ok and not reasoning
                                       and (natural or length))
    # The API uses finish_reason=stop for natural EOS and user stop alike. In
    # the no-stop fixtures it proves natural stopping, not a specific token ID.
    response['natural_stop_observed'] = natural and not case['stop']
    return response


def compare_rows(candidate, reference):
    """Pair check. Speed ratio is reference/candidate elapsed: >1 means the candidate is faster."""
    same_bytes = all(candidate.get(key) is not None and candidate.get(key) == reference.get(key)
                     for key in QUALITY_KEYS)
    cu, ru = candidate.get('usage') or {}, reference.get('usage') or {}
    same_counts = all(is_int(cu.get(key)) and cu.get(key) == ru.get(key) for key in USAGE_KEYS)
    same_prefix = is_int(candidate.get('cached_prompt_tokens')) and \
        candidate.get('cached_prompt_tokens') == reference.get('cached_prompt_tokens')
    same_source = valid_sha(candidate.get('artifact_sha256')) and \
        candidate.get('artifact_sha256') == reference.get('artifact_sha256') and \
        valid_sha(candidate.get('source_sha256')) and candidate.get('source_sha256') == reference.get('source_sha256')
    functional = bool(candidate.get('functional_pass') and reference.get('functional_pass'))
    joined = bool(candidate.get('engine_metrics_joined') and reference.get('engine_metrics_joined'))
    ratio = None
    if finite_positive(candidate.get('elapsed_s')) and finite_positive(reference.get('elapsed_s')):
        ratio = reference['elapsed_s']/candidate['elapsed_s']
    decode_ratio = None
    c_ttft, r_ttft = candidate.get('ttft_content_s'), reference.get('ttft_content_s')
    if ratio is not None and finite_positive(c_ttft) and finite_positive(r_ttft):
        c_decode, r_decode = candidate['elapsed_s']-c_ttft, reference['elapsed_s']-r_ttft
        if c_decode > 0 and r_decode > 0:
            decode_ratio = r_decode/c_decode
    return {'quality_pass': bool(same_bytes and same_counts and same_prefix and same_source and functional and joined),
            'same_bytes': same_bytes, 'same_token_counts': same_counts, 'same_cached_prompt_tokens': same_prefix,
            'same_source': same_source, 'both_functional': functional, 'both_engine_metrics_joined': joined,
            'elapsed_speedup_ratio': ratio, 'decode_phase_speedup_ratio': decode_ratio}


def interval(log_ratios):
    n = len(log_ratios)
    if not n:
        return {'valid_pairs': 0, 'geometric_mean_speedup': None, 'paired_speedup_ci95': None}
    center = statistics.mean(log_ratios)
    tcrit = TCRIT95.get(n)
    ci = None
    if tcrit:
        spread = tcrit*statistics.stdev(log_ratios)/math.sqrt(n)
        ci = [math.exp(center-spread), math.exp(center+spread)]
    return {'valid_pairs': n, 'geometric_mean_speedup': math.exp(center), 'paired_speedup_ci95': ci}


def speculation_attempts(row):
    spec = row.get('speculation')
    if not spec:
        return 0
    attempts = spec.get('attempts', 0) if isinstance(spec, dict) else None
    return attempts if is_int(attempts) else -1


def paired_summary(rows, runs, required_pairs, expected_config_sha256=None):
    case_names = [case['case'] for case in fixtures()]
    case_map = {case['case']: case for case in fixtures()}
    reasons = []
    by_key, duplicates, unexpected = {}, [], 0
    for row in rows:
        key = (row.get('repeat'), row.get('arm'), row.get('case'))
        if key[1] not in ARMS or key[2] not in case_map or not is_int(key[0]) or not 0 <= key[0] < required_pairs:
            unexpected += 1
            continue
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row
    run_by_key, run_duplicates = {}, 0
    for run in runs:
        key = (run.get('repeat'), run.get('arm'))
        if key in run_by_key:
            run_duplicates += 1
        run_by_key[key] = run
    integrity = not duplicates and not unexpected and not run_duplicates
    if not integrity:
        reasons.append('duplicate_or_unexpected_rows')

    # Source/config receipts: one source and one runtime artifact across every arm and pair.
    receipts_ok = True
    sources, artifacts, lookup_sources = set(), set(), set()
    for repeat in range(required_pairs):
        for arm in ARMS:
            run = run_by_key.get((repeat, arm))
            if run is None:
                receipts_ok = False
                continue
            if not all(valid_sha(run.get(key)) for key in ('source_sha256', 'artifact_sha256', 'config_sha256')):
                receipts_ok = False
            if expected_config_sha256 is not None and run.get('config_sha256') != expected_config_sha256.get(arm):
                receipts_ok = False
            files = run.get('lookup_source_files')
            if not isinstance(files, dict) or set(files) != set(LOOKUP_SOURCE_FILES) or not all(map(valid_sha, files.values())):
                receipts_ok = False
            else:
                lookup_sources.add(digest(files))
            sources.add(run.get('source_sha256'))
            artifacts.add(run.get('artifact_sha256'))
    if len(sources) != 1 or len(artifacts) != 1 or len(lookup_sources) != 1:
        receipts_ok = False
    for (repeat, arm, _), row in by_key.items():
        run = run_by_key.get((repeat, arm)) or {}
        if row.get('source_sha256') != run.get('source_sha256') or row.get('artifact_sha256') != run.get('artifact_sha256'):
            receipts_ok = False
    if not receipts_ok:
        reasons.append('source_or_config_receipt_mismatch')

    complete = all((repeat, arm, name) in by_key for repeat in range(required_pairs)
                   for arm in ARMS for name in case_names)
    if not complete:
        reasons.append('missing_rows_or_pairs')

    pairs, primary, diagnostic = [], {}, {}
    quality = complete
    for name in case_names:
        primary_logs = []
        diag_logs = {f'{c}_vs_{r}': [] for c, r in DIAGNOSTIC_COMPARISONS}
        case_quality = True
        for repeat in range(required_pairs):
            rows_here = {arm: by_key.get((repeat, arm, name)) for arm in ARMS}
            for arm, row in rows_here.items():
                if row is not None and not row.get('functional_pass'):
                    case_quality = False
                    who = 'reference' if arm == REFERENCE_ARM else arm
                    reasons.append(f'{who}_functional_fail:{name}')
            cand, ref = rows_here[CANDIDATE_ARM], rows_here[REFERENCE_ARM]
            if cand is None or ref is None:
                case_quality = False
                continue
            check = compare_rows(cand, ref)
            if not check['quality_pass']:
                case_quality = False
                if not check['same_bytes']:
                    reasons.append(f'lookup_output_differs_from_overlap:{name}')
                elif not check['same_token_counts'] or not check['same_cached_prompt_tokens']:
                    reasons.append(f'lookup_counts_differ_from_overlap:{name}')
                elif not check['both_engine_metrics_joined']:
                    reasons.append(f'engine_metrics_not_joined:{name}')
                elif not check['same_source']:
                    reasons.append(f'source_differs_between_arms:{name}')
            if check['quality_pass'] and check['elapsed_speedup_ratio']:
                primary_logs.append(math.log(check['elapsed_speedup_ratio']))
            pair = {'repeat': repeat, 'case': name, 'comparison': 'lookup_vs_overlap', **check}
            for c, r in DIAGNOSTIC_COMPARISONS:
                a, b = rows_here.get(c), rows_here.get(r)
                if a is None or b is None:
                    continue
                diag = compare_rows(a, b)
                pair[f'{c}_vs_{r}_same_bytes'] = diag['same_bytes']
                pair[f'{c}_vs_{r}_elapsed_speedup_ratio'] = diag['elapsed_speedup_ratio']
                if diag['same_bytes'] and diag['both_functional'] and diag['elapsed_speedup_ratio']:
                    diag_logs[f'{c}_vs_{r}'].append(math.log(diag['elapsed_speedup_ratio']))
            pairs.append(pair)
        quality &= case_quality
        primary[name] = dict(interval(primary_logs), quality_pass=case_quality)
        diagnostic[name] = {key: interval(values) for key, values in diag_logs.items()}

    # Speculation scope: reference/diagnostic arms never speculate; sampled requests never
    # speculate (QWEN36_SPEC_SAMPLING=0); repetition cases speculate in every lookup arm.
    scope_ok = True
    for (repeat, arm, name), row in by_key.items():
        attempts = speculation_attempts(row)
        if arm != CANDIDATE_ARM and attempts != 0:
            scope_ok = False
            reasons.append(f'speculation_in_{arm}_arm:{name}')
        if arm == CANDIDATE_ARM and case_map[name]['sampling'] != 'greedy' and attempts != 0:
            scope_ok = False
            reasons.append(f'speculation_on_sampled_request:{name}')
    exercised = complete and all(speculation_attempts(by_key[(repeat, CANDIDATE_ARM, name)]) > 0
                                 for repeat in range(required_pairs) for name in EXERCISE_CASES)
    if not exercised:
        reasons.append('lookup_not_exercised_in_every_repetition_row')
    lookup_rows = [row for (_, arm, _), row in by_key.items() if arm == CANDIDATE_ARM and speculation_attempts(row) > 0]
    timing_scope = exercised and all(
        row['speculation'].get('wall_timing_includes_publication') is True
        and row['speculation'].get('verify_replay_in_engine_gpu_timers') is False
        and row['speculation'].get('publication_failed') is False for row in lookup_rows)
    if not timing_scope:
        reasons.append('speculation_timing_scope_unconfirmed')

    speed_win = any(primary[name]['valid_pairs'] >= REQUIRED_PAIRS and primary[name]['paired_speedup_ci95'][0] >= SPEED_WIN_LOWER
                    for name in SPEED_WIN_CASES)
    no_regression = all(value['valid_pairs'] >= REQUIRED_PAIRS and value['paired_speedup_ci95'][0] >= NO_REGRESSION_LOWER
                        for value in primary.values())
    gates = bool(complete and quality and integrity and receipts_ok and scope_ok and exercised and timing_scope)
    screen_signal = any(primary[name]['geometric_mean_speedup'] is not None
                        and primary[name]['geometric_mean_speedup'] >= SCREEN_SIGNAL for name in SPEED_WIN_CASES)
    return {'complete': complete, 'integrity_pass': integrity, 'source_receipts_pass': receipts_ok,
            'quality_pass': bool(complete and quality), 'speculation_scope_pass': scope_ok,
            'lookup_exercised_on_repetition_cases': exercised,
            'speculation_timing_scope_confirmed': timing_scope,
            'reference_arm': REFERENCE_ARM, 'paired_cases': pairs,
            'lookup_vs_overlap_by_case': primary, 'diagnostic_by_case': diagnostic,
            'speed_win_vs_overlap': speed_win, 'no_regression_vs_overlap': no_regression,
            'speed_gate_pass': bool(gates and speed_win and no_regression),
            'screen_speed_signal_vs_overlap': screen_signal,
            'quality_failure_reasons': sorted(set(reasons)), 'gates_pass': gates,
            'qualified': False, 'greedy_only_applicability': True}


def decide(summary, state, requested_pairs, cancellation):
    """OFF unless every gate passed; five pairs are the only route to acceptance."""
    reasons = list(summary.get('quality_failure_reasons', []))
    if state != 'complete':
        reasons.append('run_state_'+str(state))
    probes = [row for row in cancellation or [] if row.get('arm') == CANDIDATE_ARM]
    if not probes or not all(row.get('pass') is True for row in probes):
        reasons.append('lookup_cancellation_probe_failed_or_missing')
    if reasons or not summary.get('gates_pass'):
        return 'OFF', sorted(set(reasons or ['gates_failed']))
    if requested_pairs < REQUIRED_PAIRS:
        if summary.get('screen_speed_signal_vs_overlap'):
            return 'SCREEN_PASS', []
        return 'OFF', ['screen_no_speed_signal_vs_overlap']
    if summary.get('speed_gate_pass'):
        return 'ACCEPTANCE_GATES_PASS', []
    return 'OFF', ['speed_gate_vs_overlap_failed']


def screen_report_problem(screen, report):
    """A multi-pair run must follow a passing 1-pair screen of the identical setup."""
    if not isinstance(screen, dict):
        return 'screen_report_invalid'
    if screen.get('schema_version') != SCHEMA_VERSION or screen.get('state') != 'complete' \
            or screen.get('requested_pairs') != 1 or screen.get('decision') != 'SCREEN_PASS':
        return 'screen_report_not_a_passing_screen'
    for key in ('base_config_sha256', 'arm_config_sha256', 'fixture_sha256'):
        if screen.get(key) != report.get(key):
            return 'screen_report_setup_differs:'+key
    if (screen.get('harness') or {}).get('lookup_qualification_sha256') != report['harness']['lookup_qualification_sha256']:
        return 'screen_report_harness_differs'
    artifacts = {run.get('artifact_sha256') for run in screen.get('runs') or []}
    if len(artifacts) != 1 or not valid_sha(next(iter(artifacts))):
        return 'screen_report_runtime_artifact_invalid'
    return None


def output_path_problem(path, workspace_opt=None, root=None):
    workspace_opt = Path(os.path.abspath(workspace_opt or WORKSPACE_OPT))
    config = Path(os.path.abspath((root or ROOT)/'config'))
    resolved = Path(os.path.abspath(path))
    if resolved.suffix != '.json':
        return 'output_must_be_json'
    if resolved.parent in (workspace_opt, Path(os.path.abspath(Path(__file__).parent))):
        # start-experiment.sh copies "$workspace"/*.json into native config at every arm start.
        return 'output_in_directory_copied_to_native_config_use_subdirectory'
    if resolved.parent == config or config in resolved.parents:
        return 'output_inside_private_config'
    if resolved.exists():
        return 'output_exists_use_new_name'
    return None


class ServiceOwner:
    def __init__(self, root=ROOT):
        self.root, self.profile, self.invocation = root, None, None

    @staticmethod
    def command(args, timeout=60):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError('subprocess_failed_' + str(result.returncode))
        return result.stdout.strip()

    @staticmethod
    def state(unit, prop):
        """None means unknown (no user bus/systemctl); callers fail closed on it."""
        try:
            result = subprocess.run(['systemctl', '--user', 'show', unit, '--property='+prop, '--value'],
                                    capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def require_idle(self):
        for unit in (PRODUCTION_UNIT, UNIT):
            state = self.state(unit, 'ActiveState')
            if state is None:
                raise RuntimeError('service_state_unknown')
            if state not in ('inactive', 'failed'):
                raise RuntimeError('a_model_service_is_busy')
        if not port_free():
            raise RuntimeError('backend_port_is_busy')

    def start(self, profile):
        self.require_idle()
        self.profile = profile
        self.command(['bash', str(self.root/'start-experiment.sh'), profile])
        self.invocation = self.state(UNIT, 'InvocationID')
        if not self.invocation:
            raise RuntimeError('experiment_invocation_missing')

    def stop(self):
        if self.profile is None:
            return
        active = self.state(UNIT, 'ActiveState')
        if active in ('', 'inactive', 'failed'):
            self.profile = self.invocation = None
            return
        if active is None:
            raise RuntimeError('experiment_state_unknown_not_stopped')
        # Only our unique generated profile identifies ownership after a start
        # timeout. Never stop an unrelated experiment that reused this unit.
        command = self.state(UNIT, 'ExecStart') or ''
        current = self.state(UNIT, 'InvocationID')
        if self.profile not in command or (self.invocation and current != self.invocation):
            raise RuntimeError('experiment_ownership_changed_not_stopped')
        self.command(['systemctl', '--user', 'stop', UNIT])
        self.profile = self.invocation = None


def port_free(port=1919):
    with socket.socket() as probe:
        probe.settimeout(.3)
        return probe.connect_ex(('127.0.0.1', port)) != 0


def acquire_gpu_lock(path=None):
    """Non-blocking exclusive flock held for the whole run.

    Also correct under `flock -n <lock> python lookup-qualification.py ...`: an inherited
    descriptor of the same file is the wrapper's open file description, which this
    process shares, so re-flocking it succeeds; a lock held by anybody else fails.
    """
    import fcntl
    path = Path(GPU_LOCK if path is None else path)
    target = os.path.realpath(path)
    fd_dir = Path('/proc/self/fd')
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


def preflight(base_name, base, owner=None, root=None, workspace_opt=None, fork_source=None, check_port=True):
    """Read-only readiness checks; never starts or stops anything."""
    root = Path(root or ROOT)
    workspace_opt = Path(workspace_opt or WORKSPACE_OPT)
    fork_source = Path(fork_source or FORK_SOURCE)
    owner = owner or ServiceOwner(root)
    units = {unit: owner.state(unit, 'ActiveState') for unit in (PRODUCTION_UNIT, UNIT)}
    checks = {'model_units_idle': all(state in ('inactive', 'failed') for state in units.values()),
              'backend_port_free': port_free() if check_port else True}
    native_start, workspace_start = root/'start-experiment.sh', workspace_opt/'start-experiment.sh'
    native_sha = sha_file(native_start)
    checks['start_experiment_synced'] = native_sha is not None and native_sha == sha_file(workspace_start)
    try:
        line = re.search(r"^workspace='([^']+)'", native_start.read_text(), re.M)
    except OSError:
        line = None
    checks['start_experiment_workspace_is_current'] = bool(line) and Path(line.group(1)).is_dir() and \
        os.path.realpath(line.group(1)) == os.path.realpath(workspace_opt)
    workspace_base = workspace_opt/f'{base_name}.json'
    try:
        checks['workspace_base_matches_native'] = not workspace_base.exists() or \
            digest(json.loads(workspace_base.read_text())) == digest(base)
    except (OSError, ValueError):
        checks['workspace_base_matches_native'] = False
    checks['native_run_experiment_present'] = (root/'run-experiment.sh').is_file() and (root/'experiment.py').is_file()
    info = {'units': units,
            'native_start_experiment_sha256': native_sha,
            'native_experiment_py_sha256': sha_file(root/'experiment.py'),
            'workspace_experiment_py_sha256': sha_file(workspace_opt/'experiment.py'),
            'lookup_sources': {rel: {'native': sha_file(root/'python/freetoken'/rel),
                                     'workspace_fork': sha_file(fork_source/rel)} for rel in LOOKUP_SOURCE_FILES}}
    info['lookup_sources_native_equals_fork'] = all(
        value['native'] is not None and value['native'] == value['workspace_fork'] for value in info['lookup_sources'].values())
    return {'ok': all(checks.values()), 'checks': checks, 'info': info}


def read_key(root):
    line = (root/'config/experiment.env').read_text().strip()
    if not line.startswith('QWEN36_API_KEY='):
        raise RuntimeError('experiment_secret_format_invalid')
    return {'Authorization': 'Bearer '+line.split('=', 1)[1]}


def wait_ready(owner, config, profile, timeout=300):
    """Authenticated /health at most once per HEALTH_POLL_INTERVAL_S (APIGuard rate bucket).

    Consecutive polls start >= 1 s apart, so the bucket (+1 token/s) never drains during a
    long load and the warmup right after ready is admitted; a 429 also waits Retry-After.
    """
    import requests
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if owner.state(UNIT, 'ActiveState') != 'active':
            raise RuntimeError('experiment_not_active_during_load')
        polled = time.monotonic()
        pause = HEALTH_POLL_INTERVAL_S
        try:
            auth = read_key(owner.root)
            response = requests.get(URL+'/health', headers=auth, timeout=3)
            if response.status_code == 429:
                pause = max(pause, retry_after_s(response.headers))
            elif response.status_code == 200 and response.json().get('status') in ('ok', 'ready'):
                receipt = json.loads((owner.root/'active-run.json').read_text())
                if receipt.get('profile') != profile or digest(receipt.get('configuration')) != digest(config):
                    raise RuntimeError('ready_profile_receipt_mismatch')
                if not re.fullmatch('[0-9a-f]{32}', str(receipt.get('run_id', ''))):
                    raise RuntimeError('ready_run_id_invalid')
                return receipt
        except (OSError, requests.RequestException, ValueError):
            pass
        time.sleep(max(0.0, polled+pause-time.monotonic()))
    raise RuntimeError('model_readiness_timeout')


def run_receipt_digest(receipt):
    """Runtime artifact identity: source, service files, packages, checkpoint and native libraries."""
    return digest({'source': receipt.get('source_sha256'), 'packages': receipt.get('packages'),
                   'checkpoint_revision': receipt.get('checkpoint_revision'),
                   'service_source_files': receipt.get('service_source_files'),
                   'libraries': {key: value.get('sha256') for key, value in (receipt.get('native_libraries') or {}).items()}})


def _recent_rows(root, directory, name):
    # A rotation may occur between HTTP completion and the engine flush.
    for suffix in ('', '.1', '.2', '.3', '.4', '.5'):
        path = root/directory/(name+suffix)
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except (OSError, UnicodeError):
            continue
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                yield row


def join_engine_metrics(result, expected_run_id, root=None, attempts=60, delay=.05):
    """Bounded join by request ID, then engine UID + this arm's serving run ID.

    Same semantics as bench.attach_metrics (V4/V5) but local, so bench changes cannot
    alter this harness, and the join must belong to the current arm's process.
    """
    root = Path(root or ROOT)
    rid = result.pop('request_id', None)
    if not rid:
        result['engine_metrics_status'] = 'no_request_id'
        return {}
    record = None
    for _ in range(attempts):
        if record is None:
            record = next((row for row in _recent_rows(root, 'logs', 'requests.jsonl')
                           if row.get('request_id') == rid), None)
        if record is not None:
            cached = record.get('cached_prompt_tokens')
            if is_int(cached) and cached >= 0:
                result['cached_prompt_tokens'] = cached
            for key in ('ttft_generated_s', 'decode_ack_tps'):
                if isinstance(record.get(key), (int, float)) and not isinstance(record.get(key), bool) \
                        and math.isfinite(record[key]):
                    result[key] = record[key]
            for key in ('peaks', 'counter_deltas'):
                if isinstance(record.get(key), dict):
                    result[key] = safe_numeric(record[key])
            uid, run_id = record.get('engine_uid'), record.get('run_id')
            if uid is None:
                result['engine_metrics_status'] = 'no_engine_uid'
                return {}
            if not isinstance(run_id, str) or not run_id or run_id != expected_run_id:
                result['engine_metrics_status'] = 'run_id_mismatch'
                return {}
            row = next((row for row in _recent_rows(root, 'reports', 'engine.jsonl')
                        if row.get('uid') == uid and row.get('run_id') == run_id), None)
            if row is not None:
                result['engine_metrics_status'] = 'joined'
                return {key: value for key, value in row.items() if key not in ('uid', 'pid', 'run_id')}
        time.sleep(delay)
    result['engine_metrics_status'] = 'engine_row_timeout' if record is not None else 'request_row_timeout'
    return {}


def speculation_summary(engine):
    spec = engine.get('speculation') if isinstance(engine.get('speculation'), dict) else {}
    result = {key: safe_numeric(spec[key]) for key in SPEC_FIELDS if key in spec}
    if is_int(spec.get('proposed')) and spec['proposed'] > 0:
        result['acceptance_fraction'] = spec.get('accepted', 0)/spec['proposed']
    if is_int(spec.get('attempts')) and spec['attempts'] > 0:
        result['emitted_tokens_per_verification'] = spec.get('emitted', 0)/spec['attempts']
    return result


def call_model(case, expected_run_id):
    import requests
    start = time.perf_counter()
    first = first_reasoning = last = None
    content, reasoning, usage, finish = [], [], {}, None
    with requests.post(URL+'/v1/chat/completions', headers=read_key(ROOT), json=payload(case),
                       stream=True, timeout=(5, 120)) as response:
        result = {'status': response.status_code}
        if response.status_code == 429:
            result['retry_after_s'] = retry_after_s(response.headers)
        if response.status_code != 200:
            return evaluate(case, dict(result, elapsed_s=time.perf_counter()-start, engine_metrics_joined=False))
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if event.get('usage'):
                usage = {key: event['usage'][key] for key in USAGE_KEYS if is_int(event['usage'].get(key))}
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                if delta.get('content'):
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    content.append(delta['content'])
                if delta.get('reasoning_content'):
                    first_reasoning = time.perf_counter() if first_reasoning is None else first_reasoning
                    reasoning.append(delta['reasoning_content'])
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
        result.update(elapsed_s=time.perf_counter()-start, usage=usage, finish_reason=finish,
                      ttft_content_s=None if first is None else first-start,
                      ttft_reasoning_s=None if first_reasoning is None else first_reasoning-start,
                      decode_client_tps=(usage.get('completion_tokens', 0)-1)/(last-first) if first and last > first else None,
                      request_id=response.headers.get('x-qwen36-request-id'))
    # Joined after the client timer stopped; the raw request/run IDs never enter the report.
    engine = join_engine_metrics(result, expected_run_id)
    result['engine_metrics_joined'] = result.get('engine_metrics_status') == 'joined'
    result['engine_timing'] = {key: safe_numeric(value) for key, value in engine.items()
                               if key.endswith(('_ms', '_s', '_tokens', '_bytes')) and isinstance(value, (int, float))}
    result['speculation'] = speculation_summary(engine)
    return evaluate(case, dict(result, _content=''.join(content), _reasoning=''.join(reasoning)))


def call_model_admitted(case, expected_run_id, attempts=ADMISSION_RETRY_ATTEMPTS):
    """call_model that retries only HTTP 429 admissions, at most `attempts` times in total.

    APIGuard returns 429 (rate bucket or busy slot) before the app, so a rejected attempt
    ran no inference and cannot change the prompt cache. Each attempt is timed on its
    own; the returned row is the first non-429 attempt, or the last 429, which fails
    functionally and therefore yields OFF. The retry count stays on the row.
    """
    for attempt in range(attempts):
        row = call_model(case, expected_run_id)
        if row.get('status') != 429 or attempt == attempts-1:
            row['admission_retries'] = attempt
            return row
        time.sleep(row.get('retry_after_s') or ADMISSION_RETRY_DEFAULT_S)
    raise ValueError('attempts must be positive')


def cancellation_probe(arm, expected_run_id, min_chunks=CANCEL_MIN_CHUNKS):
    """Cancel a greedy copy stream after it is decoding, then require a working follow-up.

    On the lookup arm the engine row of the cancelled request shows whether speculation
    had already run. This is a live API/slot-release smoke on the lookup-configured
    process; it does not prove an abort inside one particular verification step.
    """
    import requests
    seen, status, rid, stream_retries = 0, None, None, 0
    for attempt in range(ADMISSION_RETRY_ATTEMPTS):
        pause = None
        with requests.post(URL+'/v1/chat/completions', headers=read_key(ROOT), json=payload(fixture('copy-code')),
                           stream=True, timeout=(5, 120)) as response:
            status = response.status_code
            rid = response.headers.get('x-qwen36-request-id')
            if status == 429 and attempt < ADMISSION_RETRY_ATTEMPTS-1:
                pause = retry_after_s(response.headers)  # rejected before inference
            elif status == 200:
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b'data: ') or line == b'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    if any((choice.get('delta') or {}).get('content') for choice in event.get('choices', [])):
                        seen += 1
                        if seen >= min_chunks:
                            break
        if pause is None:
            break
        stream_retries += 1
        time.sleep(pause)
    cancelled = time.perf_counter()
    follow, recovery = None, None
    for _ in range(40):
        attempt_started = time.perf_counter()
        follow = call_model(fixture('eos-short'), expected_run_id)
        if follow.get('status') != 429:
            # Up to the follow-up's HTTP completion; the metrics join wait is excluded.
            if finite_positive(follow.get('elapsed_s')):
                recovery = attempt_started-cancelled+follow['elapsed_s']
            break
        time.sleep(.25)
    probe = {'request_id': rid}
    engine = join_engine_metrics(probe, expected_run_id)
    attempts = speculation_summary(engine).get('attempts')
    result = {'arm': arm, 'stream_status': status, 'stream_admission_retries': stream_retries,
              'content_chunks_before_cancel': seen,
              'followup_status': None if follow is None else follow.get('status'),
              'followup_functional_pass': bool(follow and follow.get('functional_pass')),
              'recovery_s': recovery, 'cancelled_engine_metrics_joined': probe.get('engine_metrics_status') == 'joined',
              'engine_marked_cancelled': engine.get('cancelled') if isinstance(engine.get('cancelled'), bool) else None,
              'speculation_attempts_before_cancel': attempts if is_int(attempts) else None}
    result['pass'] = bool(status == 200 and seen >= min_chunks and result['followup_functional_pass']
                          and finite_positive(recovery) and recovery <= CANCEL_RECOVERY_MAX_S)
    result['speculation_active_at_cancel'] = None if arm != CANDIDATE_ARM else bool(is_int(attempts) and attempts > 0)
    return result


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def new_report(args, base, arms):
    return {'schema_version': SCHEMA_VERSION, 'qualified': False, 'decision': 'OFF', 'decision_reasons': ['not_run'],
            'base_profile': args.base_profile, 'base_config_sha256': digest(base),
            'arm_config_sha256': {key: digest(value) for key, value in arms.items()},
            'arms': list(ARMS), 'reference_arm': REFERENCE_ARM, 'candidate_arm': CANDIDATE_ARM,
            'order': [list(item) for item in schedule(args.pairs)],
            'harness': {'lookup_qualification_sha256': sha_file(Path(__file__)), 'imports_bench_py': False},
            'fixture_sha256': digest(fixtures()), 'fixtures': fixture_manifest(),
            'requested_pairs': args.pairs, 'required_pairs_for_acceptance': REQUIRED_PAIRS,
            'criteria': {'speed_metric': 'client elapsed_s, ratio overlap/lookup', 'speed_win_cases': list(SPEED_WIN_CASES),
                         'speed_win_ci95_lower': SPEED_WIN_LOWER, 'no_regression_ci95_lower': NO_REGRESSION_LOWER,
                         'screen_point_speedup': SCREEN_SIGNAL, 'exercise_cases': list(EXERCISE_CASES),
                         'cancel_min_chunks': CANCEL_MIN_CHUNKS, 'cancel_recovery_max_s': CANCEL_RECOVERY_MAX_S},
            'greedy_only': True, 'thinking': False, 'seed': SEED,
            'state': 'prepared_only', 'rows': [], 'runs': [], 'cancellation': []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-profile', required=True, help='Frozen profile name under private optimization/config')
    parser.add_argument('--pairs', type=int, choices=range(1, 6), required=True,
                        help='1 = screen; 5 = the only acceptance route')
    parser.add_argument('--output', type=Path, required=True,
                        help='New .json path, not directly in the workspace optimization directory')
    parser.add_argument('--run', action='store_true', help='Start the model arms (GPU); default only writes the plan')
    parser.add_argument('--screen-report', type=Path,
                        help='Required with --run when --pairs > 1: the SCREEN_PASS report of the same setup')
    args = parser.parse_args(argv)
    if not re.fullmatch('[a-z0-9][a-z0-9-]{0,70}', args.base_profile):
        parser.error('base profile must be a simple profile name')
    problem = output_path_problem(args.output)
    if problem:
        parser.error(problem)
    base_path = ROOT/'config'/f'{args.base_profile}.json'
    try:
        base = json.loads(base_path.read_text())
    except (OSError, ValueError):
        parser.error('base_profile_missing_or_invalid')
    try:
        arms = arm_configs(base)
    except ValueError as error:
        parser.error(str(error))
    report = new_report(args, base, arms)
    screen_artifact = None
    if args.run and args.pairs > 1:
        try:
            screen = json.loads(args.screen_report.read_text(encoding='utf-8')) if args.screen_report else None
        except (OSError, ValueError):
            screen = None
        problem = screen_report_problem(screen, report)
        if problem:
            parser.error(problem)
        screen_artifact = screen['runs'][0]['artifact_sha256']
        report['screen_report_sha256'] = sha_file(args.screen_report)
    report['preflight'] = preflight(args.base_profile, base)
    if not args.run:
        save(args.output, report)
        print(json.dumps({'prepared': True, 'preflight_ok': report['preflight']['ok'],
                          'decision': 'OFF', 'qualified': False, 'model_started': False}))
        return 0
    if os.name != 'posix':
        raise RuntimeError('Run inside the existing WSL environment')
    if not report['preflight']['ok']:
        report.update(state='preflight_failed', decision_reasons=['preflight_failed'])
        save(args.output, report)
        print(json.dumps({'state': 'preflight_failed', 'checks': report['preflight']['checks'], 'model_started': False}))
        return 1
    try:
        lock_fd, lock_mode = acquire_gpu_lock()
    except GpuLockBusy:
        report.update(state='gpu_lock_busy', decision_reasons=['gpu_lock_busy'])
        save(args.output, report)
        print(json.dumps({'state': 'gpu_lock_busy', 'model_started': False}))
        return EXIT_GPU_LOCK_BUSY
    report['gpu_lock'] = lock_mode
    owner = ServiceOwner()
    session = uuid.uuid4().hex[:12]
    source_sha = artifact_sha = lookup_sources = None
    report['state'] = 'running'
    save(args.output, report)
    try:
        owner.require_idle()
        for repeat, arm in schedule(args.pairs):
            if digest(json.loads(base_path.read_text())) != report['base_config_sha256']:
                raise RuntimeError('base_profile_changed_during_run')
            profile = f'lookupq-{session}-{repeat}-{arm}'
            config_path = ROOT/'config'/f'{profile}.json'
            fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as config_file:
                json.dump(arms[arm], config_file)
            started = time.monotonic()
            try:
                owner.start(profile)
                receipt = wait_ready(owner, arms[arm], profile)
                run_id = receipt['run_id']
                current_source = receipt.get('source_sha256')
                current_artifact = run_receipt_digest(receipt)
                files = receipt.get('source_files') or {}
                current_lookup = {rel: files.get(rel) for rel in LOOKUP_SOURCE_FILES}
                if not valid_sha(current_source) or (source_sha is not None and current_source != source_sha):
                    raise RuntimeError('source_changed_between_arms')
                if artifact_sha is not None and current_artifact != artifact_sha:
                    raise RuntimeError('runtime_artifact_changed_between_arms')
                if screen_artifact is not None and current_artifact != screen_artifact:
                    raise RuntimeError('runtime_artifact_differs_from_screen')
                if lookup_sources is not None and current_lookup != lookup_sources:
                    raise RuntimeError('lookup_source_changed_between_arms')
                source_sha, artifact_sha, lookup_sources = current_source, current_artifact, current_lookup
                report['runs'].append({'repeat': repeat, 'arm': arm, 'source_sha256': current_source,
                                       'artifact_sha256': current_artifact, 'lookup_source_files': current_lookup,
                                       'service_source_files': receipt.get('service_source_files'),
                                       'packages': receipt.get('packages'),
                                       'checkpoint_revision': receipt.get('checkpoint_revision'),
                                       'config_sha256': digest(receipt['configuration']),
                                       'load_ready_s': time.monotonic()-started,
                                       'run_id_sha256': digest(run_id.encode())})
                # The first request after a long load is the one APIGuard's rate bucket would
                # reject; admissions are retried (bounded) and the retry count is recorded.
                warm_row = call_model_admitted(copy.deepcopy(WARMUP), run_id)
                report['runs'][-1]['warmup_admission_retries'] = warm_row.get('admission_retries')
                if not warm_row['functional_pass']:
                    raise RuntimeError('warmup_rate_limited' if warm_row.get('status') == 429
                                       else 'warmup_expected_response_failed')
                for case in fixtures():
                    row = call_model_admitted(case, run_id)
                    row.update(repeat=repeat, arm=arm, source_sha256=current_source, artifact_sha256=current_artifact)
                    report['rows'].append(row)
                    save(args.output, report)
                    print(json.dumps({'repeat': repeat, 'arm': arm, 'case': case['case'],
                                      'functional_pass': row['functional_pass'], 'elapsed_s': row['elapsed_s'],
                                      'admission_retries': row.get('admission_retries'),
                                      'speculation': row.get('speculation', {})}), flush=True)
                if repeat == 0 and arm in (CANDIDATE_ARM, REFERENCE_ARM):
                    report['cancellation'].append(cancellation_probe(arm, run_id))
                    save(args.output, report)
            finally:
                owner.stop()
            if len([run for run in report['runs'] if run['repeat'] == repeat]) == len(ARMS):
                screening = paired_summary(report['rows'], report['runs'], repeat+1, report['arm_config_sha256'])
                if not screening['gates_pass']:
                    report['state'] = 'screen_failed'
                    break
        if report['state'] == 'running':
            report['state'] = 'complete'
    except KeyboardInterrupt:
        report['state'] = 'interrupted'
    except Exception as error:
        report['state'] = 'error'
        # Never persist HTTP error bodies, raw subprocess stderr, auth or prompts.
        report['error_type'] = type(error).__name__
        if isinstance(error, RuntimeError) and re.fullmatch('[a-z0-9_]{1,100}', str(error)):
            report['error_code'] = str(error)
    finally:
        try:
            owner.stop()
        finally:
            report['summary'] = paired_summary(report['rows'], report['runs'], args.pairs, report['arm_config_sha256'])
            report['decision'], report['decision_reasons'] = decide(report['summary'], report['state'],
                                                                   args.pairs, report['cancellation'])
            save(args.output, report)
            os.close(lock_fd)
    print(json.dumps({'state': report['state'], 'decision': report['decision'],
                      'decision_reasons': report['decision_reasons'][:12], 'qualified': False}))
    return 0 if report['decision'] in ('SCREEN_PASS', 'ACCEPTANCE_GATES_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
