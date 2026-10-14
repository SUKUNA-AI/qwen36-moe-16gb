"""eval-v1: pre-registered 60-case quality suite (code/ru/tool/agent/long) for basis decisions.

Modes
  verify                              CPU: regenerate all cases, compare with eval-v1/freeze.json
  freeze --out PATH [--tokenizer]     CPU: write a NEW freeze (O_EXCL; never overwrites)
  run --name N --expect-profile P     suite against the ALREADY loaded profile (like tool-task-bench
                                      --pilot): never starts/stops a unit
  managed --name N --profile P        take the GPU-exclusive flock (non-blocking), start P through
                                      start-experiment.sh, readiness, run the suite, stop ONLY the unit
                                      invocation this process started
Receipts: reports/N-eval-v1.json, written once with O_EXCL (never overwritten). Rows carry
pass/fail, static failure codes, token counts and sha256 of the replies, never their text.
A run is VALID only if all 60 cases executed without infrastructure errors, the serving receipt
matches the frozen profile configuration and runtime files, and the serving generation did not change.
Exit: 0 VALID; 1 INVALID or failed; 2 refused/usage; 65 fixture/freeze mismatch; 73 unit active;
75 GPU lock busy.
"""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'eval-v1'))
import cases_v1 as cases  # noqa: E402
import sandbox_v1 as sandbox  # noqa: E402

SCHEMA_VERSION = 1
KIND = 'eval-v1'
NAME_PATTERN = re.compile(r'[a-z0-9][a-z0-9.-]{0,90}')
FREEZE = HERE / 'eval-v1' / 'freeze.json'
SOURCE_FILES = ('eval-v1.py', 'eval-v1/cases_v1.py', 'eval-v1/sandbox_v1.py', 'eval-v1-compare.py')
ENDPOINT = 'http://127.0.0.1:1919/v1/chat/completions'
NATIVE_ROOT = pathlib.Path('/home/USER/qwen36-api/optimization')
WORKSPACE_OPT = pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
RUNTIME_FILES = ('models/nvfp4_banks.py', 'models/qwen3_5_moe/weight.py', 'layers/quantization/moe/base.py')
PROFILES = {'bf128': {'context': 131072, 'chunk': 2048, 'flags': [], 'env': {}},
            'bf128-fp16compat': {'context': 131072, 'chunk': 2048, 'flags': [],
                                 'env': {'QWEN36_NVFP4_GLOBAL_FP16_COMPAT': '1'}}}
INFRA_CODES = frozenset({'transport_error', 'http_5xx', 'http_4xx_fixed_input', 'bad_response_schema'})
REPORT_CODES = ('fixture_mismatch', 'source_mismatch', 'receipt_invalid', 'profile_config_mismatch',
                'runtime_mismatch', 'serving_generation_changed', 'infra_error', 'incomplete',
                'launcher_problems', 'managed_failed', 'interrupted', 'stop_failed')
EXIT_REFUSED, EXIT_FIXTURE, EXIT_ACTIVE, EXIT_LOCK = 2, 65, 73, 75
REQUEST_TIMEOUT = (5, 900)


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha256_file(HERE / name) for name in SOURCE_FILES}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_compare_module():
    spec = importlib.util.spec_from_file_location('eval_v1_compare', HERE / 'eval-v1-compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- fixture/freeze
def verify_fixture(freeze_path=None):
    """Regenerate every case and compare with the freeze. Returns (cases, freeze, problems)."""
    problems = []
    try:
        freeze = json.loads(pathlib.Path(freeze_path or FREEZE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None, ['freeze_unreadable']
    suite = cases.materialize()
    frozen = [(c.get('id'), c.get('sha256')) for c in freeze.get('cases', [])]
    actual = [(c['id'], cases.case_sha256(c)) for c in suite]
    if frozen != actual:
        problems.append('fixture_mismatch')
    if freeze.get('suite_sha256') != cases.suite_sha256(suite):
        problems.append('fixture_mismatch')
    if (freeze.get('sources') != source_hashes()
            or freeze.get('harness_imports', {}).get('bench.py') != sha256_file(HERE / 'bench.py')):
        problems.append('source_mismatch')
    if freeze.get('profiles') != PROFILES or freeze.get('kind') != 'eval-v1-freeze':
        problems.append('fixture_mismatch')
    return suite, freeze, sorted(set(problems))


def build_freeze(tokenizer_path=None):
    suite = cases.materialize()
    compare = load_compare_module()
    tokens = {}
    if tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        for case in suite:
            if case['category'] == 'long':
                tokens[case['id']] = len(tok.encode(case['requests'][0]['messages'][0]['content'],
                                                    add_special_tokens=False))
    counts = collections.Counter(c['category'] for c in suite)
    rows = []
    for case in suite:
        row = {'id': case['id'], 'category': case['category'], 'sha256': cases.case_sha256(case),
               'requests': [r['label'] for r in case['requests']]}
        if case['id'] in tokens:
            row['user_content_tokens'] = tokens[case['id']]
        rows.append(row)
    return {
        'schema': 1, 'kind': 'eval-v1-freeze', 'suite_id': cases.SUITE_ID, 'suite_version': cases.SUITE_VERSION,
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'sampling': cases.SAMPLING, 'categories': dict(counts), 'case_count': len(suite),
        'suite_sha256': cases.suite_sha256(suite), 'cases': rows,
        'sources': source_hashes(),
        'harness_imports': {'bench.py': sha256_file(HERE / 'bench.py')},
        'profiles': PROFILES,
        'runtime_expectations': {'root': 'python/freetoken (receipt source_files)',
                                 'source_files': {rel: sha256_file(runtime_file(rel)) for rel in RUNTIME_FILES}},
        'decision_rule': compare.RULE,
        'decision_rule_sha256': canonical_hash(compare.RULE),
        'failure_codes': list(cases.failure_codes_registry()) + list(REPORT_CODES),
        'estimate': {
            'requests_expected': sum(1 + len(r.get('calls', [])) if r['kind'] == 'tool' else 1
                                     for c in suite for r in c['requests']),
            'long_prefill_s': '4x16K~6s + 4x64K~22s + 4x120K~58s ~= 344 s (bf128 Triton TTFT, cold prefix)',
            'suite_minutes': '9-12', 'per_profile_minutes_including_load_and_stop': '13-18', 'budget_minutes': 45},
        'sandbox': {'wall_timeout_s': sandbox.WALL_TIMEOUT_S, 'cpu_limit_s': sandbox.CPU_LIMIT_S,
                    'memory_limit_bytes': sandbox.MEMORY_LIMIT_BYTES, 'safe_builtins': list(sandbox.SAFE_BUILTINS)},
    }


def runtime_file(rel):
    return HERE.parent.parent.parent / 'work' / 'FreeToken-opt' / 'python' / 'freetoken' / rel


# ----------------------------------------------------------------------------- HTTP
def http_post(payload):
    """POST one non-stream completion. Returns (status|None, json|None, request_id|None)."""
    import requests
    import bench
    try:
        response = requests.post(ENDPOINT, headers=bench.headers(), json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None, None, None
    if response.status_code != 200:
        return response.status_code, None, response.headers.get('x-qwen36-request-id')  # body never read
    try:
        data = response.json()
    except ValueError:
        data = None
    return 200, data, response.headers.get('x-qwen36-request-id')


def numeric_usage(value):
    return {k: v for k, v in value.items() if type(v) is int} if isinstance(value, dict) else {}


def request_step(http, messages, request, after_model_output):
    """One request. Returns (step, message|None); step['error_code'] marks a failed request."""
    payload = {'model': 'qwen36-35b', 'messages': messages, 'max_tokens': request['max_tokens'],
               'temperature': cases.SAMPLING['temperature'], 'top_p': cases.SAMPLING['top_p'],
               'seed': cases.SAMPLING['seed'], 'stream': False,
               'chat_template_kwargs': {'enable_thinking': cases.SAMPLING['thinking']}}
    if request.get('tools'):
        payload['tools'] = request['tools']
        payload['tool_choice'] = 'auto'
    step = {'request_sha256': canonical_hash({'messages': messages, 'tools': request.get('tools')})}
    started = time.perf_counter()
    status, data, request_id = http(payload)
    step.update(status=status, elapsed_s=round(time.perf_counter() - started, 3), request_id=request_id)
    if status is None:
        step['error_code'] = 'transport_error'
        return step, None
    if status >= 500:
        step['error_code'] = 'http_5xx'
        return step, None
    if status != 200:
        step['error_code'] = 'http_4xx_after_model_output' if after_model_output else 'http_4xx_fixed_input'
        return step, None
    choices = data.get('choices') if isinstance(data, dict) else None
    if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get('message'), dict)):
        step['error_code'] = 'bad_response_schema'
        return step, None
    message = choices[0]['message']
    content = message.get('content')
    reasoning = message.get('reasoning_content')
    step.update(usage=numeric_usage(data.get('usage')), finish_reason=choices[0].get('finish_reason'),
                reasoning_chars=len(reasoning) if isinstance(reasoning, str) else 0,
                content_sha256=hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None,
                tool_calls=len(message.get('tool_calls') or []),
                message_sha256=canonical_hash(message))
    return step, message


def run_text(http, request):
    step, message = request_step(http, copy.deepcopy(request['messages']), request, False)
    if step.get('error_code'):
        return [step['error_code']], [step], {}
    codes = cases.check_text(request['check'], message.get('content'))
    if message.get('tool_calls'):
        codes.append('unexpected_call')
    return codes, [step], {}


def run_code(http, request):
    step, message = request_step(http, copy.deepcopy(request['messages']), request, False)
    if step.get('error_code'):
        return [step['error_code']], [step], {}
    code = sandbox.extract_code(message.get('content'))
    result = sandbox.run_tests(code, request['function'], request['tests'])
    extra = {k: result[k] for k in ('tests_total', 'tests_passed', 'stripped_toplevel', 'network_isolated')}
    extra['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
    return ([result['failure_code']] if result['failure_code'] else []), [step], extra


def run_tool(http, request):
    messages = copy.deepcopy(request['messages'])
    remaining = copy.deepcopy(request['calls'])
    used_ids, steps, codes = set(), [], []
    executed = 0
    for index in range(len(request['calls']) + 2):
        step, message = request_step(http, messages, request, after_model_output=index > 0)
        steps.append(step)
        if step.get('error_code'):
            codes.append(step['error_code'])
            break
        if message.get('role') != 'assistant':
            codes.append('bad_response_schema')
            break
        calls = message.get('tool_calls')
        if calls is not None and not isinstance(calls, list):
            codes.append('bad_response_schema')
            break
        if calls:
            if not remaining:
                codes.append('unexpected_call')
                break
            call_codes, matches = cases.validate_tool_calls(message, step.get('finish_reason'),
                                                            step.get('reasoning_chars'), request, remaining, used_ids)
            codes += call_codes
            if cases.blocking(call_codes):
                break
            # Preserve the returned assistant message exactly; send each genuine local result with its id.
            messages.append(message)
            for call, expected in matches:
                used_ids.add(call['id'])
                remaining.remove(expected)
                executed += 1
                messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                 'content': json.dumps(expected['result'], ensure_ascii=False, separators=(',', ':'))})
            continue
        if remaining:
            codes.append('no_call' if executed == 0 else 'missing_call')
            break
        if step.get('reasoning_chars'):
            codes.append('final_reasoning_present')
        if step.get('finish_reason') != 'stop':
            codes.append('final_bad_finish')
        codes += cases.check_text(request['final'], message.get('content'))
        break
    else:
        codes.append('too_many_steps')
    return codes, steps, {'calls_expected': len(request['calls']), 'calls_executed': executed}


RUNNERS = {'text': run_text, 'code': run_code, 'tool': run_tool}


def ordered_codes(codes):
    registry = list(cases.failure_codes_registry())
    unknown = set(codes) - set(registry)
    if unknown:
        raise RuntimeError(f'unregistered failure code(s): {sorted(unknown)}')
    return sorted(set(codes), key=registry.index)


def run_case(http, case):
    row = {'id': case['id'], 'category': case['category'], 'sha256': cases.case_sha256(case), 'requests': []}
    all_codes = []
    for request in case['requests']:
        codes, steps, extra = RUNNERS[request['kind']](http, request)
        codes = ordered_codes(codes)
        all_codes += codes
        row['requests'].append({'label': request['label'], 'kind': request['kind'], 'pass': not codes,
                                'failure_codes': codes, 'steps': steps, **extra})
    row['failure_codes'] = ordered_codes(all_codes)
    row['pass'] = not row['failure_codes']
    row['infra_error'] = bool(INFRA_CODES & set(row['failure_codes']))
    row['completion_tokens'] = sum(s.get('usage', {}).get('completion_tokens', 0)
                                   for r in row['requests'] for s in r['steps'])
    row['prompt_tokens'] = sum(s.get('usage', {}).get('prompt_tokens', 0) for r in row['requests'] for s in r['steps'])
    return row


def summarize(rows):
    per = {}
    for category in cases.CATEGORIES:
        sel = [r for r in rows if r['category'] == category]
        per[category] = {'cases': len(sel), 'pass': sum(r['pass'] for r in sel)}
    counts = collections.Counter(code for r in rows for code in r['failure_codes'])
    return {'cases': len(rows), 'pass': sum(r['pass'] for r in rows), 'by_category': per,
            'failure_code_counts': dict(sorted(counts.items())),
            'infra_error_cases': [r['id'] for r in rows if r['infra_error']]}


def execute_suite(http, suite, progress=True):
    rows = []
    for case in suite:
        started = time.perf_counter()
        row = run_case(http, case)
        row['elapsed_s'] = round(time.perf_counter() - started, 3)
        rows.append(row)
        if progress:
            print(json.dumps({'case': row['id'], 'pass': row['pass'], 'failure_codes': row['failure_codes'],
                              'elapsed_s': row['elapsed_s']}), flush=True)
    return rows


# ----------------------------------------------------------------------------- receipts
def check_receipt(receipt, profile, freeze):
    """Return list of report codes for the serving receipt (empty = OK)."""
    import bench
    codes = []
    try:
        bench.validate_receipt(receipt, profile)
    except (ValueError, KeyError, TypeError, AttributeError):
        return ['receipt_invalid']
    if receipt.get('configuration') != freeze['profiles'].get(profile):
        codes.append('profile_config_mismatch')
    files = receipt.get('source_files') or {}
    expected = freeze['runtime_expectations']['source_files']
    if any(files.get(rel) != sha for rel, sha in expected.items()):
        codes.append('runtime_mismatch')
    return codes


def write_new(path, document):
    """Receipts are never overwritten (O_EXCL)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as target:
        target.write(json.dumps(document, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def new_report(name, mode, profile, freeze, suite):
    import bench
    return {'kind': KIND, 'schema_version': SCHEMA_VERSION, 'name': name, 'mode': mode, 'profile': profile,
            'freeze_sha256': sha256_file(FREEZE), 'suite_sha256': cases.suite_sha256(suite),
            'decision_rule_sha256': freeze.get('decision_rule_sha256'),
            'sources': source_hashes(), 'bench_sha256': sha256_file(bench.__file__),
            'sampling': cases.SAMPLING, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status': 'INVALID', 'invalid_codes': [], 'rows': []}


def read_receipt(root):
    try:
        value = json.loads((root / 'active-run.json').read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def run_loaded(root, name, profile, report, freeze, suite, http=http_post, progress=True):
    """Fill `report` by running the suite on the loaded profile (no unit control)."""
    receipt = read_receipt(root)
    report['run'] = receipt
    codes = check_receipt(receipt, profile, freeze)
    report['receipt_checks'] = codes
    if codes:
        report['invalid_codes'] = sorted(set(report['invalid_codes'] + codes))
        return report
    report['network_isolated_sandbox'] = sandbox.network_isolation_available()
    started = time.perf_counter()
    rows = execute_suite(http, suite, progress)
    report['suite_elapsed_s'] = round(time.perf_counter() - started, 1)
    report['rows'] = rows
    report['summary'] = summarize(rows)
    after = read_receipt(root)
    invalid = list(report['invalid_codes'])
    if after.get('run_id') != receipt.get('run_id'):
        invalid.append('serving_generation_changed')
    if report['summary']['infra_error_cases']:
        invalid.append('infra_error')
    if len(rows) != len(suite) or len(rows) != freeze['case_count']:
        invalid.append('incomplete')
    report['invalid_codes'] = sorted(set(invalid))
    report['status'] = 'VALID' if not report['invalid_codes'] else 'INVALID'
    return report


def report_path(root, name):
    return root / 'reports' / f'{name}-eval-v1.json'


def cmd_run(args, root=NATIVE_ROOT, http=http_post):
    path = report_path(root, args.name)
    if path.exists():
        print(json.dumps({'result': 'refused', 'reason': 'report_exists'}), file=sys.stderr)
        return EXIT_REFUSED
    suite, freeze, problems = verify_fixture()
    if problems:
        print(json.dumps({'result': 'refused', 'reason': problems}), file=sys.stderr)
        return EXIT_FIXTURE
    report = new_report(args.name, 'loaded', args.expect_profile, freeze, suite)
    try:
        run_loaded(root, args.name, args.expect_profile, report, freeze, suite, http)
    except (KeyboardInterrupt, SystemExit):
        report['invalid_codes'] = sorted(set(report['invalid_codes'] + ['interrupted']))
        report['status'] = 'INVALID'
        write_new(path, report)
        raise
    write_new(path, report)
    print(json.dumps({'status': report['status'], 'invalid_codes': report['invalid_codes'],
                      'summary': report.get('summary'), 'report': path.name}, ensure_ascii=False))
    return 0 if report['status'] == 'VALID' else 1


# ----------------------------------------------------------------------------- managed
def systemctl_state(prop, unit='qwen36-experiment'):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '--property=' + prop, '--value'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(f'systemctl show {unit} {prop}: exit {result.returncode}')
    return result.stdout.strip()


def units_inactive(state=systemctl_state):
    return all(state('ActiveState', unit) in ('', 'inactive', 'failed') for unit in ('qwen36-inference', 'qwen36-experiment'))


def run_command(args, timeout):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if args[:3] == ['systemctl', '--user', 'stop'] and result.returncode == 5:
        return ''
    if result.returncode:
        raise RuntimeError(f'{pathlib.Path(args[0]).name}: exit {result.returncode}')
    return result.stdout


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def cmd_managed(args, root=NATIVE_ROOT, workspace=WORKSPACE_OPT, lock_path=None, http=http_post,
                state=systemctl_state, command=run_command, launcher=None):
    import bench
    path = report_path(root, args.name)
    if path.exists():
        print(json.dumps({'result': 'refused', 'reason': 'report_exists'}), file=sys.stderr)
        return EXIT_REFUSED
    if args.profile not in PROFILES:
        print(json.dumps({'result': 'refused', 'reason': 'profile_not_frozen'}), file=sys.stderr)
        return EXIT_REFUSED
    suite, freeze, problems = verify_fixture()
    if problems:
        print(json.dumps({'result': 'refused', 'reason': problems}), file=sys.stderr)
        return EXIT_FIXTURE
    try:
        workspace_config = json.loads((workspace / f'{args.profile}.json').read_text())
    except (OSError, ValueError):
        workspace_config = None
    native_runtime = {rel: (sha256_file(root / 'python' / 'freetoken' / rel)
                            if (root / 'python' / 'freetoken' / rel).is_file() else None) for rel in RUNTIME_FILES}
    if workspace_config != PROFILES[args.profile] or native_runtime != freeze['runtime_expectations']['source_files']:
        print(json.dumps({'result': 'refused', 'reason': 'profile_or_runtime_not_synced',
                          'workspace_config_ok': workspace_config == PROFILES[args.profile],
                          'native_runtime_ok': native_runtime == freeze['runtime_expectations']['source_files']}),
              file=sys.stderr)
        return EXIT_FIXTURE
    try:
        lock = bench.acquire_gpu_lock(lock_path)
    except bench.GpuLockBusy as error:
        print(json.dumps({'result': 'refused', 'reason': 'gpu_lock_busy', 'message': str(error)}), file=sys.stderr)
        return EXIT_LOCK
    for name in ('SIGTERM', 'SIGHUP'):
        signal.signal(getattr(signal, name), _terminate)
    try:
        active = not units_inactive(state)
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        active = True
    if active:
        os.close(lock)
        print(json.dumps({'result': 'refused', 'reason': 'unit_active_or_unknown'}), file=sys.stderr)
        return EXIT_ACTIVE
    report = new_report(args.name, 'managed', args.profile, freeze, suite)
    managed = {'invocation_id': None, 'step': 'preflight'}
    report['managed'] = managed
    interrupted = None
    try:
        started = time.monotonic()
        managed['step'] = 'start'
        command(['bash', str(root / 'start-experiment.sh'), args.profile], 60)
        managed['invocation_id'] = state('InvocationID') or None
        if not managed['invocation_id']:
            raise RuntimeError('started experiment has no invocation id')
        managed['step'] = 'readiness'
        command([sys.executable, str(root / 'check-experiment.py'), str(args.ready_timeout)], args.ready_timeout + 20)
        managed['load_ready_s'] = round(time.monotonic() - started, 1)
        managed['step'] = 'launcher'
        info = (launcher or bench.launcher_receipt)(args.profile, root)
        managed['launcher'] = {k: info[k] for k in ('files', 'environment_sha256', 'problems', 'main_pid', 'invocation_id')}
        if info['invocation_id'] != managed['invocation_id'] or info['problems']:
            report['invalid_codes'].append('launcher_problems')
        managed['step'] = 'suite'
        run_loaded(root, args.name, args.profile, report, freeze, suite, http)
        if 'launcher_problems' in report['invalid_codes']:
            report['status'] = 'INVALID'
        managed['step'] = 'done'
    except (RuntimeError, subprocess.TimeoutExpired, ValueError, OSError, KeyError, TypeError) as error:
        managed['error_type'] = type(error).__name__
        report['invalid_codes'].append('managed_failed')
        report['status'] = 'INVALID'
    except (KeyboardInterrupt, SystemExit) as error:
        managed['error_type'] = type(error).__name__
        report['invalid_codes'].append('interrupted')
        report['status'] = 'INVALID'
        interrupted = error
    finally:
        try:
            invocation = managed['invocation_id']
            if invocation is None and managed['step'] == 'start':
                # start-experiment.sh may have started the unit and failed afterwards: under the GPU lock
                # with both units verified inactive just before, that unit is ours.
                orphan = state('InvocationID')
                if orphan:
                    invocation = orphan
                    managed['stopped_after_start_failure'] = True
            if invocation and state('InvocationID') == invocation:
                command(['systemctl', '--user', 'stop', 'qwen36-experiment'], 60)
                managed['stopped'] = True
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            managed['stop_error'] = type(error).__name__
            report['invalid_codes'].append('stop_failed')
            report['status'] = 'INVALID'
        report['invalid_codes'] = sorted(set(report['invalid_codes']))
        if report['invalid_codes']:
            report['status'] = 'INVALID'
        write_new(path, report)
        try:
            os.close(lock)
        except OSError:
            pass
    if interrupted is not None:
        raise interrupted
    print(json.dumps({'status': report['status'], 'invalid_codes': report['invalid_codes'],
                      'summary': report.get('summary'), 'report': path.name}, ensure_ascii=False))
    return 0 if report['status'] == 'VALID' else 1


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('verify')
    fr = sub.add_parser('freeze')
    fr.add_argument('--out', required=True)
    fr.add_argument('--tokenizer')
    rn = sub.add_parser('run')
    rn.add_argument('--name', required=True)
    rn.add_argument('--expect-profile', required=True)
    mg = sub.add_parser('managed')
    mg.add_argument('--name', required=True)
    mg.add_argument('--profile', required=True)
    mg.add_argument('--ready-timeout', type=int, default=600)
    args = parser.parse_args(argv)
    if args.mode == 'verify':
        suite, freeze, problems = verify_fixture()
        print(json.dumps({'ok': not problems, 'problems': problems, 'cases': len(suite or []),
                          'suite_sha256': cases.suite_sha256(suite) if suite else None,
                          'freeze_sha256': sha256_file(FREEZE) if FREEZE.exists() else None}))
        return 0 if not problems else EXIT_FIXTURE
    if args.mode == 'freeze':
        document = build_freeze(args.tokenizer)
        write_new(args.out, document)
        print(json.dumps({'written': args.out, 'suite_sha256': document['suite_sha256'], 'cases': document['case_count']}))
        return 0
    if not NAME_PATTERN.fullmatch(args.name):
        parser.error('invalid --name')
    if args.mode == 'run':
        if args.expect_profile not in PROFILES:
            parser.error('--expect-profile must be a frozen eval-v1 profile')
        return cmd_run(args)
    if not 60 <= args.ready_timeout <= 1800:
        parser.error('--ready-timeout 60..1800')
    return cmd_managed(args)


if __name__ == '__main__':
    raise SystemExit(main())
