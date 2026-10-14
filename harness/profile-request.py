"""Profiling workload for an already running experiment unit; never starts or stops units.

Legacy (unchanged):  profile-request.py --length 120000 --output 128
Plan (CPU only):     profile-request.py plan --variant main-retain [--write]
Compute config:      profile-request.py ncu-config --variant main-retain --targets PRIVATE.json --rank 1
                     [--scope fresh_120k] [--reference N] [--nvtx-phase] [--write]
Run (needs model):   profile-request.py run --profile profile-final-main-retain --wait-ready 900 --wait-exit 1800

The final scenario is fixed: two warmup requests (short prompt; one full prefill chunk
plus remainder), a warmed short decode, fresh 16K and fresh 120K documents, then a
filler request that the capture end (cudaProfilerStop -> stop-shutdown) interrupts.
QWEN36_NSYS_FORWARDS is an upper bound of forwards through the last measured request;
the filler absorbs the slack. Prompts are synthetic; receipts keep only hashes/numbers.
"""
import argparse, datetime, hashlib, json, math, os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))         # bench.py lives next to this script (also under python -I)
ROOT = pathlib.Path('/home/USER/qwen36-api/optimization')
MODEL = '/path/to/LLM/qwen36-api/model'
QUESTION = '\nОбъясни алгоритм проверки данных.'
OUTPUT_MARGIN = 512          # template/question tokens plus safety inside the context
EXTRA_FORWARDS = 3           # overlap-scheduled step, prefix-margin tail chunk, alignment split
PLAN_SCHEMA = 1
VARIANT_RE = r'^[a-z0-9][a-z0-9-]{0,40}$'
SECTIONS = ['SpeedOfLight', 'MemoryWorkloadAnalysis', 'MemoryWorkloadAnalysis_Chart',
            'MemoryWorkloadAnalysis_Tables', 'ComputeWorkloadAnalysis', 'Occupancy',
            'WarpStateStats', 'SourceCounters', 'SchedulerStats', 'LaunchStats']


def final_requests(chunk, context):
    """(label, document tokens, output tokens, document seed) in execution order."""
    long_tokens = min(120000, (context - 64 - OUTPUT_MARGIN) // 1000 * 1000)
    rows = [('warmup_short', 256, 32, 46001),
            ('warmup_chunk', chunk + 1536, 16, 46002),
            ('warm_decode', 256, 128, 46003)]
    if context >= 16384 + 64 + OUTPUT_MARGIN:
        rows.append(('fresh_16k', 16384, 64, 46016))
    rows.append(('fresh_%dk' % (long_tokens // 1000), long_tokens, 64, 46120))
    return rows


FILLER = ('filler', 256, 512, 46999)


def messages_for(tokenizer, length, seed):
    from bench import document
    return [{'role': 'user', 'content': document(tokenizer, length, seed) + QUESTION}]


def prompt_tokens(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                        enable_thinking=False)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def tokenizer_load():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_plan(tokenizer, variant, config, requests):
    chunk = int(config.get('chunk', 2048))
    context = int(config.get('context', 65536))
    rows = []
    bound = 0
    for label, length, output, seed in requests:
        messages = messages_for(tokenizer, length, seed)
        tokens = prompt_tokens(tokenizer, messages)
        if tokens + output > context:
            raise SystemExit(f'{label}: {tokens}+{output} exceeds context {context}')
        upper = math.ceil(tokens / chunk) + output + EXTRA_FORWARDS
        lower = max(1, math.ceil(tokens / chunk) - 1) + output - 1
        bound += upper
        rows.append({'label': label, 'document_tokens': length, 'document_seed': seed,
                     'output_tokens': output, 'prompt_tokens_planned': tokens,
                     'forwards_lower': lower, 'forwards_upper': upper,
                     'messages_sha256': canonical(messages)})
    label, length, output, seed = FILLER
    filler_messages = messages_for(tokenizer, length, seed)
    filler_tokens = prompt_tokens(tokenizer, filler_messages)
    filler_min = max(1, math.ceil(filler_tokens / chunk) - 1) + output - 1
    slack = bound - sum(row['forwards_lower'] for row in rows)
    if filler_min <= slack:
        raise SystemExit('filler cannot absorb the forward-count slack')
    return {'schema': PLAN_SCHEMA, 'source_variant': variant, 'chunk': chunk, 'context': context,
            'requests': rows,
            'filler': {'label': label, 'document_tokens': length, 'document_seed': seed,
                       'output_tokens': output, 'prompt_tokens_planned': filler_tokens,
                       'forwards_min': filler_min, 'messages_sha256': canonical(filler_messages)},
            'nsys_forwards': bound, 'slack_max_forwards': slack,
            'sampling': {'seed': 17, 'temperature': .6, 'top_p': .95, 'thinking': False, 'ignore_eos': True},
            'counting_rule': 'forwards = ceil(prompt/chunk) prefill chunks + (output-1) decode steps; '
                             'upper adds 1 for the first token bookkeeping and EXTRA_FORWARDS=3'}


def variant_config(variant, source_dir):
    import re
    if not re.match(VARIANT_RE, variant) or variant.startswith('profile'):
        raise SystemExit('variant must be a plain config name such as main-retain')
    for folder in (source_dir, HERE, ROOT / 'config'):
        path = pathlib.Path(folder) / f'{variant}.json'
        if path.is_file():
            return path, json.loads(path.read_text(encoding='utf-8'))
    raise SystemExit(f'{variant}.json not found')


def write_json(path, value, private=False, overwrite_same_only=True):
    text = json.dumps(value, ensure_ascii=False, indent=1) + '\n'
    path = pathlib.Path(path)
    if path.exists():
        if overwrite_same_only and path.read_text(encoding='utf-8') == text:
            return 'unchanged'
        raise SystemExit(f'{path} exists with different content; refusing to overwrite')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    return 'written'


def profile_config(variant, source_path, config, plan, nsys_block):
    result = json.loads(json.dumps(config))          # identical variant parameters
    env = dict(result.get('env', {}))
    env['QWEN36_NSYS_FORWARDS'] = str(plan['nsys_forwards'])
    result['env'] = env
    result['nsys'] = nsys_block
    plan = dict(plan, source_path=source_path.name,
                source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    result['profile_plan'] = plan
    return result


def command_plan(args):
    source_path, config = variant_config(args.variant, args.source_dir)
    plan = build_plan(tokenizer_load(), args.variant, config,
                      final_requests(int(config.get('chunk', 2048)), int(config.get('context', 65536))))
    block = {'tool': 'nsys', 'graph_nvtx': args.graph_nvtx, 'cuda_event_trace': 'false',
             'output_prefix': f'model-final-{args.variant}'}
    result = profile_config(args.variant, source_path, config, plan, block)
    name = f'profile-final-{args.variant}'
    if args.graph_nvtx == 'live':
        name += '-live'
        block['output_prefix'] += '-live'
    if args.write:
        print(json.dumps({'config': name, 'status': write_json(HERE / f'{name}.json', result),
                          'nsys_forwards': plan['nsys_forwards']}))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=1))


TARGETS_MIN_SCHEMA = 3       # schema 2 chose instances without shape class/request (wrong M possible)
# Clock/cache independent counters for checking that the profiled launch has the intended M
# (names verified offline with `ncu --query-metrics --chips gb203`).
FINGERPRINT_METRICS = ['gpu__time_duration.sum', 'smsp__inst_executed.sum',
                       'smsp__inst_executed_pipe_tensor.sum', 'dram__bytes_op_read.sum',
                       'dram__bytes_op_write.sum']


def build_ncu_config(args, tokenizer=None):
    """(config name, config) for one ranked target from analyze-model-nsys.py --private-targets."""
    import re
    source_path, config = variant_config(args.variant, args.source_dir)
    targets = json.loads(pathlib.Path(args.targets).read_text(encoding='utf-8'))
    if not isinstance(targets.get('schema_version'), int) or targets['schema_version'] < TARGETS_MIN_SCHEMA:
        raise SystemExit('private targets predate schema 3 (instance not matched by shape/request); '
                         'rerun analyze-model-nsys.py')
    if args.scope != 'all_measured' and not re.match(r'^[a-z0-9_]{1,32}$', args.scope):
        raise SystemExit('scope must be all_measured or a request label')
    scope = 'all_measured' if args.scope == 'all_measured' else 'request:' + args.scope
    rows = [row for row in targets.get('candidates', []) if row.get('scope') == scope]
    if not 1 <= args.rank <= len(rows):
        raise SystemExit(f'rank outside the private target list of scope {scope}')
    target = rows[args.rank - 1]
    source = target
    phase_of_source = target['phase']
    if args.reference:
        # A second instance of the same launch config (another request or shape) used as the
        # reference for the M check: its instruction counts scale with M, durations with context.
        groups = target.get('launch_config_groups') or []
        if not 1 <= args.reference <= len(groups):
            raise SystemExit('reference outside launch_config_groups of this target')
        source = groups[args.reference - 1]
        if source.get('selected'):
            raise SystemExit('reference group is the target group itself')
        shape = source.get('shape_class')
        phase_of_source = 'decode' if shape == 'decode_M1' else (
            'prefill' if shape in ('prefill_full_chunk', 'prefill_last_chunk') else None)
        request_index, label = source.get('request_index'), source.get('request_label')
    else:
        request_index, label = target.get('instance_request_index'), target.get('instance_request_label')
    if not isinstance(request_index, int) or not isinstance(label, str):
        raise SystemExit('target has no labelled request instance (analyze with --run-receipt)')
    chunk, context = int(config.get('chunk', 2048)), int(config.get('context', 65536))
    planned = final_requests(chunk, context)
    if request_index >= len(planned) or planned[request_index][0] != label:
        raise SystemExit('request order of the analysed trace does not match the final plan '
                         f'(index {request_index} is {label}); ordinals would point at another launch')
    requests = planned[:request_index + 1]
    plan = build_plan(tokenizer or tokenizer_load(), args.variant, config, requests)
    phase = phase_of_source if args.nvtx_phase else 'any'
    if phase is None:
        raise SystemExit('reference group lies outside forward ranges; use it without --nvtx-phase')
    if args.filter_mode == 'per-launch-config':
        skip = source.get('launch_skip_same_launch_config')
    elif phase == 'any':
        skip = source.get('launch_skip_same_name_in_window')
    else:
        skip = source.get('launch_skip_same_name_same_phase')
    if not isinstance(skip, int):
        raise SystemExit('target has no launch ordinal')
    tag = f'ncu{args.rank}'
    if scope != 'all_measured':
        tag += '-' + args.scope.replace('_', '-')
    if args.reference:
        tag += f'-ref{args.reference}'
    name = f'profile-final-{args.variant}-{tag}'
    if not re.match(r'^profile-final-[a-z0-9][a-z0-9-]{0,63}$', name):
        raise SystemExit('generated config name is not a valid profile-final-* name')
    block = {'tool': 'ncu', 'output_prefix': f'ncu-final-{args.variant}-{tag}'}
    result = profile_config(args.variant, source_path, config, plan, block)
    result['env']['QWEN36_STALL_SECONDS'] = '1800'
    result['ncu'] = {'kernel_regex': target['ncu_kernel_regex'], 'kernel_name_base': 'demangled',
                     'phase': phase, 'launch_skip': skip, 'launch_count': args.count,
                     'filter_mode': args.filter_mode, 'sections': SECTIONS,
                     'metrics': [] if args.no_metrics else FINGERPRINT_METRICS,
                     'cache_control': 'all', 'clock_control': args.clock_control}
    result['profile_plan']['ncu_target'] = {key: target.get(key) for key in (
        'scope', 'rank', 'kernel_id', 'phase', 'shape_class', 'category', 'grid', 'block',
        'shared_memory_bytes', 'critical_ms', 'share_of_phase_critical_percent',
        'share_of_request_critical_percent', 'instance_request_label', 'instance_shape_class',
        'instance_chunk_or_step_position', 'instance_layer', 'instance_duration_us',
        'chain_duration_p50_us_in_request', 'launch_config_identifies_instance_group',
        'other_shape_classes_same_launch_config', 'launch_config_groups')}
    result['profile_plan']['ncu_instance'] = {
        'role': 'reference' if args.reference else 'target', 'request_index': request_index,
        'request_label': label, 'launch_skip': skip, 'filter_mode': args.filter_mode, 'phase': phase,
        'reference_group': source if args.reference else None}
    return name, result


def command_ncu_config(args):
    """Compute config for one ranked target from analyze-model-nsys.py --private-targets."""
    name, result = build_ncu_config(args)
    if args.write:
        # Kernel symbols stay private: written only to the native config folder (0600).
        print(json.dumps({'config': name, 'status': write_json(ROOT / 'config' / f'{name}.json',
                          result, private=True)}))
    else:
        print(json.dumps({key: value for key, value in result.items() if key != 'ncu'}, indent=1))


def http_ready():
    import requests
    from bench import headers
    try:
        response = requests.get('http://127.0.0.1:1919/v1/models', headers=headers(), timeout=5)
        return response.status_code == 200
    except (requests.RequestException, OSError, IndexError):
        return False


def unit_active(unit):
    # Read-only state query; this script never starts or stops units.
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0


EXPERIMENT_UNIT = 'qwen36-experiment.service'
CLOCK_SLACK_S = 2.0          # profile.sh writes started_utc truncated to whole seconds
OUTPUT_BASE_RE = r'^(model|ncu)-final-[a-z0-9][a-z0-9-]{0,80}-\d{8}T\d{6}Z(-\d{1,4})?$'


def unit_show(unit):
    """Read-only systemd properties of the experiment unit (never starts/stops it)."""
    result = subprocess.run(['systemctl', '--user', 'show', '-p',
                             'ActiveState,InvocationID,ActiveEnterTimestampMonotonic,ExecStart', unit],
                            capture_output=True, text=True, timeout=15)
    props = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition('=')
        if sep:
            props[key] = value
    return props


def unit_started_epoch(props):
    """Wall-clock start of the unit from its CLOCK_MONOTONIC enter timestamp (microseconds)."""
    try:
        monotonic_us = int(props.get('ActiveEnterTimestampMonotonic') or 0)
    except ValueError:
        return None
    if monotonic_us <= 0:
        return None
    return time.time() - (time.clock_gettime(time.CLOCK_MONOTONIC) - monotonic_us / 1e6)


def exec_argv(value):
    import re
    match = re.search(r'argv\[\]=(.*?) ;', value or '')
    return match.group(1).split() if match else []


def parse_utc(text):
    value = datetime.datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('timestamp without timezone')
    return value.timestamp()


def run_guard_errors(profile, config, active, pointer, props, started_epoch, receipt_exists):
    """Reasons to refuse sending the workload; empty list means this unit is this profile's capture.

    active-run.json is written by experiment.py whoever launched it, so it cannot tell a
    profiled start from a plain run-experiment.sh start; the unit's ExecStart and the
    pointer's invocation id can, and the timestamps reject a pointer left by an older run.
    """
    import re
    errors = []
    if props.get('ActiveState') != 'active':
        errors.append('experiment unit is not active')
    argv = exec_argv(props.get('ExecStart'))
    if len(argv) < 2 or not argv[-2].endswith('/profile.sh') or argv[-1] != profile:
        errors.append('experiment unit was not started as profile.sh ' + profile +
                      ' (start-experiment.sh does not route profile-final-* ?)')
    if active.get('profile') != profile or active.get('configuration') != config:
        errors.append('active-run.json does not describe this profile config')
    if pointer.get('profile') != profile:
        errors.append('profile.sh pointer names another profile')
    invocation = props.get('InvocationID') or ''
    if not invocation or pointer.get('invocation_id') != invocation:
        errors.append('profile.sh pointer was not written by this unit invocation (stale pointer)')
    tool = (config.get('nsys') or {}).get('tool', 'nsys')
    if pointer.get('tool') != tool:
        errors.append('pointer tool differs from the profile config')
    if not re.match(OUTPUT_BASE_RE, str(pointer.get('output_base', ''))):
        errors.append('pointer output_base is malformed')
    try:
        pointer_started = parse_utc(pointer.get('started_utc'))
        active_started = parse_utc(active.get('started_utc'))
    except (TypeError, ValueError):
        errors.append('pointer/active-run start time missing or malformed')
    else:
        if started_epoch is None:
            errors.append('experiment unit start time unknown')
        else:
            if pointer_started < started_epoch - CLOCK_SLACK_S:
                errors.append('profile.sh pointer is older than the unit start (stale pointer)')
            if active_started < started_epoch - CLOCK_SLACK_S:
                errors.append('active-run.json is older than the unit start')
        if pointer_started > active_started + CLOCK_SLACK_S:
            errors.append('profile.sh pointer is newer than active-run.json (not this run)')
    if receipt_exists:
        errors.append('a run receipt for this report already exists')
    return errors


def command_run(args):
    import re
    if not re.match(r'^profile-final-[a-z0-9][a-z0-9-]{0,63}$', args.profile):
        raise SystemExit('profile must be a profile-final-* config name')
    config = json.loads((ROOT / 'config' / f'{args.profile}.json').read_text(encoding='utf-8'))
    plan = config.get('profile_plan')
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise SystemExit('config has no profile_plan')
    if unit_active('qwen36-inference.service'):
        raise SystemExit('production unit is active; profiling refuses to share the GPU')
    deadline = time.monotonic() + args.wait_ready
    while not http_ready():
        if time.monotonic() > deadline:
            raise SystemExit('experiment endpoint not ready')
        time.sleep(5)
    from bench import generate, run_receipt
    active = run_receipt()
    try:
        pointer = json.loads((ROOT / 'reports' / 'profile-final-current.json').read_text())
    except (OSError, ValueError):
        raise SystemExit('profile.sh pointer is missing; the unit was not started through profile.sh')
    props = unit_show(EXPERIMENT_UNIT)
    base = str(pointer.get('output_base', ''))
    receipt_exists = bool(re.match(OUTPUT_BASE_RE, base)) and (ROOT / 'reports' / f'{base}.run-receipt.json').exists()
    errors = run_guard_errors(args.profile, config, active, pointer, props, unit_started_epoch(props), receipt_exists)
    if errors:
        raise SystemExit('refusing to send requests: ' + '; '.join(errors))
    tokenizer = tokenizer_load()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = []
    for planned in plan['requests'] + [dict(plan['filler'], filler=True)]:
        messages = messages_for(tokenizer, planned['document_tokens'], planned['document_seed'])
        if canonical(messages) != planned['messages_sha256']:
            raise SystemExit(f"{planned['label']}: synthetic document changed since the plan")
        row = {'label': planned['label'], 'planned_prompt_tokens': planned['prompt_tokens_planned'],
               'output_tokens': planned['output_tokens']}
        try:
            result, _ = generate(messages, planned['output_tokens'], 17, ignore_eos=True)
        except Exception as error:                 # filler is expected to be cut by capture end
            row.update(status='interrupted', error_type=type(error).__name__,
                       unexpected=not planned.get('filler'))
            rows.append(row)
            break
        usage = result.get('usage') or {}
        details = usage.get('prompt_tokens_details') or {}
        row.update(status=result.get('status'), prompt_tokens=usage.get('prompt_tokens'),
                   completion_tokens=usage.get('completion_tokens'),
                   cached_tokens=details.get('cached_tokens'), finish_reason=result.get('finish_reason'),
                   ttft_content_s=result.get('ttft_content_s'), elapsed_s=result.get('elapsed_s'),
                   content_sha256=result.get('content_sha256'), request_id=result.get('request_id'))
        if planned['label'].startswith('fresh') and isinstance(row['cached_tokens'], int):
            row['fresh_cache_ok'] = row['cached_tokens'] <= 64
        rows.append(row)
        if result.get('status') != 200 and not planned.get('filler'):
            row['unexpected'] = True
            break
    measured = [r for r in rows if r['label'] != 'filler']
    measured_completed = len(measured) == len(plan['requests']) and all(
        r.get('status') == 200 and r.get('completion_tokens') == r['output_tokens'] for r in measured)
    deadline = time.monotonic() + args.wait_exit
    while unit_active('qwen36-experiment.service') and time.monotonic() < deadline:
        time.sleep(5)
    receipt = {'schema': 1, 'profile': args.profile, 'run_id': active.get('run_id'),
               'unit_invocation_id': props.get('InvocationID'),
               'output_base': pointer.get('output_base'), 'started_utc': started,
               'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'nsys_forwards': plan['nsys_forwards'], 'requests': rows,
               'measured_requests_completed': measured_completed,
               'experiment_unit_inactive_after_run': not unit_active('qwen36-experiment.service'),
               'timings_under_profiler': True,
               'note': 'Timings were taken under a profiler and are not speed evidence.'}
    target = ROOT / 'reports' / f"{pointer.get('output_base')}.run-receipt.json"
    write_json(target, receipt, private=True, overwrite_same_only=False)
    print(json.dumps({'receipt': target.name, 'measured_requests_completed': measured_completed,
                      'unit_inactive': receipt['experiment_unit_inactive_after_run']}))


def legacy(argv):
    from bench import generate
    p = argparse.ArgumentParser()
    p.add_argument('--length', type=int, default=4096)
    p.add_argument('--output', type=int, default=64)
    a = p.parse_args(argv)
    t = tokenizer_load()
    r, _ = generate(messages_for(t, a.length, 46217), a.output, 17, ignore_eos=True)
    print(json.dumps(r))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith('-'):
        return legacy(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--variant', required=True)
    plan.add_argument('--source-dir', default=str(HERE))
    plan.add_argument('--graph-nvtx', choices=('precapture', 'live'), default='precapture')
    plan.add_argument('--write', action='store_true')
    ncu = sub.add_parser('ncu-config')
    ncu.add_argument('--variant', required=True)
    ncu.add_argument('--source-dir', default=str(HERE))
    ncu.add_argument('--targets', required=True)
    ncu.add_argument('--rank', type=int, default=1)
    ncu.add_argument('--scope', default='all_measured',
                     help='all_measured or a measured request label such as fresh_120k')
    ncu.add_argument('--reference', type=int, default=0,
                     help='profile launch_config_groups[N] of the target instead (M check reference)')
    ncu.add_argument('--no-metrics', action='store_true', help='omit the fingerprint --metrics list')
    ncu.add_argument('--count', type=int, default=1)
    ncu.add_argument('--filter-mode', choices=('global', 'per-launch-config'), default='global')
    ncu.add_argument('--nvtx-phase', action='store_true', help="add --nvtx-include 'Qwen36\\/<phase>/'")
    ncu.add_argument('--clock-control', choices=('base', 'none'), default='base')
    ncu.add_argument('--write', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('--profile', required=True)
    run.add_argument('--wait-ready', type=int, default=900)
    run.add_argument('--wait-exit', type=int, default=1800)
    args = parser.parse_args(argv)
    {'plan': command_plan, 'ncu-config': command_ncu_config, 'run': command_run}[args.command](args)


if __name__ == '__main__':
    main()
