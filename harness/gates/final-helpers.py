"""CPU-only helpers for final-gates.sh / final-install.sh / final-prep.sh (stdlib only, no torch, no GPU, no secrets).

Subcommands (exit 0 ok, 3 check failed, 2 usage):
  configs  --gather in|out --opt DIR (--print | --check | --write)
           derive main-release.json / long256-release.json from the pinned r2-attn.json (+ tuned gather if in,
           + FREETOKEN_DISABLE_JIT=1); --write creates missing files O_EXCL, never overwrites a different file.
  mapping  --gather in|out --validator PROFILE_RUNTIME --mq MAKE_QUALIFICATION --draft PROFILES_DRAFT
           --main CFG --long CFG --b64 CFG
           the run configs (and their qualify.py teacher variants) must map exactly onto the profiles that
           final-install.sh will write (staged make-qualification run_equivalent_profile == production_semantics).
  profiles --gather in|out --draft PROFILES_DRAFT --qualified a,b,c --out PATH [--validator PROFILE_RUNTIME] [--check]
  qcheck   REPORT                        strict teacher/quality gate (make-qualification's bit-exact rule + PPL gate)
  extract-run --report R --profile P --config CFG [--teacher] --out OUT
           standalone run receipt (make-qualification run_receipts) from a quality.py report's 'run' object.
  provenance FILE [--require-module NAME]  kernel_provenance.py check output: all modules prebuilt, extension
           _pinned_tensor prebuilt, required module present and prebuilt.
  summary  --steps FILE --run RUN --gather in|out --reports DIR --out OUT --decision-out DRAFT [--release-prep-draft F]
  decision-check --decision D --summary S --reports DIR   (prints the qualified profile list)
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

OPT_LIB = '/home/USER/qwen36-api/optimization'
PINNED_R2ATTN = {
    'context': 131072, 'chunk': 2048, 'memory_ratio': 0.75, 'qualified': False, 'flags': [],
    'env': {
        'QWEN36_PREFIX_MARGIN': '8',
        'QWEN36_FUSED_ATTN_GATE': '1',
        'QWEN36_FUSED_ATTN_GATE_LIBRARY': OPT_LIB + '/private-libraries/libqwen_attention_gate.so',
        'QWEN36_FP32_GDN_SNAPSHOT': '1',
        'QWEN36_RETAIN_CHUNK_SNAPSHOT': '1',
        'QWEN36_NVFP4_PREFILL': 'cuda',
        'QWEN36_CUDA_NVFP4_LIBRARY': OPT_LIB + '/private-libraries/libqwen_nvfp4_bitexact.so',
        'QWEN36_LAYER_MAJOR_PREFILL': '1',
        'QWEN36_CUDA_GDN': '1',
        'QWEN36_CUDA_GDN_LIBRARY': OPT_LIB + '/private-cuda-gdn/pinned/libqwen36_gdn-a8a806eda3a90a65.so',
        'QWEN36_CUDA_GDN_DECODE': 'cluster',
        'QWEN36_FP8_GEMV': '1',
        'QWEN36_FP8_GEMV_LIBRARY': OPT_LIB + '/private-fp8gemv/libqwen_fp8_gemv-685c78cf89e60d02.so',
        'QWEN36_LAYER_MAJOR_ATTN_BATCH': '2'}}
GATHER_ENV = {'QWEN36_DECODE_GATHER': 'tuned', 'TVM_FFI_CUDA_ARCH_LIST': '12.0'}
LOADER_ENV = {'FREETOKEN_DISABLE_JIT': '1'}
LONG256 = {'context': 262144, 'memory_ratio': 0.73}
ATTN_BATCH = 2
GATHER_MODULE = 'freetoken__decode_gather_tma_8_3_16384'
TEACHER_ENV = {'QWEN36_TEACHER_FIXTURE': OPT_LIB + '/config/teacher-fixture.json',
               'QWEN36_LOGIT_REFERENCE': OPT_LIB + '/reports/logit-reference'}
PROFILE_OF = {'main-release': 'main128', 'long256-release': 'long256', 'baseline64-final': 'baseline64'}


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_excl(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)


# ------------------------------------------------------------------------------------------------ configs
def derive_configs(opt, gather):
    src = load_json(pathlib.Path(opt) / 'r2-attn.json')
    need(src == PINNED_R2ATTN, 'r2-attn.json differs from the pinned final base (refusing to derive release configs)')
    main = copy.deepcopy(src)
    if gather == 'in':
        main['env'].update(GATHER_ENV)
        r3 = pathlib.Path(opt) / 'main-candidate-r3.json'
        if r3.exists():
            expect = copy.deepcopy(main)
            need(load_json(r3) == expect, 'main-candidate-r3.json differs from r2-attn + tuned gather (inconsistent candidate)')
    main['env'].update(LOADER_ENV)
    long = copy.deepcopy(main)
    long.update(LONG256)
    return {'main-release': main, 'long256-release': long}


def config_text(cfg):
    return json.dumps(cfg, separators=(',', ':'))  # same compact form as r2/r3 drivers, no newline


def cmd_configs(a):
    cfgs = derive_configs(a.opt, a.gather)
    out = {}
    for name, cfg in cfgs.items():
        text = config_text(cfg)
        path = pathlib.Path(a.opt) / (name + '.json')
        state = 'absent'
        if path.exists():
            need(load_json(path) == cfg, f'{path.name} exists with different content (gather mode or base changed); '
                                         'move it away deliberately, the freeze/manifests would change')
            state = 'same'
        elif a.write:
            write_excl(path, text)
            state = 'written'
        elif a.check:
            raise Fail(f'{path.name} missing')
        out[name] = {'state': state, 'sha256_of_derived_text': sha256_bytes(text.encode()),
                     'env_keys': sorted(cfg['env'])}
        if a.print:
            out[name]['text'] = text
    print(json.dumps(out, indent=1 if a.print else None))


# ------------------------------------------------------------------------------------------------ profiles
def build_profiles(draft_path, gather, qualified):
    doc = load_json(draft_path)
    need(set(doc['profiles']) == {'main128', 'long256', 'baseline64'} and doc['default'] == 'main128', 'unexpected draft')
    for name in ('main128', 'long256'):
        old = doc['profiles'][name]
        need(old.get('decode_gather') is True and old.get('layer_major_prefill') is True, 'draft ' + name + ' switches')
        new = {}
        for key, value in old.items():
            if key == 'decode_gather' and gather == 'out':
                continue  # absent = off (draft rationale: optional switches stay absent until evidence exists)
            if key == 'qualified':
                new['layer_major_attn_batch'] = ATTN_BATCH
            new[key] = value
        doc['profiles'][name] = new
    unknown = set(qualified) - set(doc['profiles'])
    need(not unknown, 'unknown profiles: ' + ','.join(sorted(unknown)))
    for name, profile in doc['profiles'].items():
        profile['qualified'] = name in qualified
    return doc


def profiles_text(doc):
    return json.dumps(doc, indent=2) + '\n'


def cmd_profiles(a):
    qualified = [x for x in a.qualified.split(',') if x]
    doc = build_profiles(a.draft, a.gather, qualified)
    if a.validator:
        validator = load_module('qwen36_profile_runtime_final', a.validator)
        validator.validate_profiles_document(doc)
        canon = {n: validator.canonical_profile_sha256(p) for n, p in doc['profiles'].items()}
        modules = validator.required_kernel_modules(doc) if qualified else []
    else:
        canon, modules = None, None
    text = profiles_text(doc)
    out = pathlib.Path(a.out)
    state = 'same' if out.exists() and out.read_bytes() == text.encode() else ('would-write' if a.check else 'written')
    if state == 'written':
        tmp = out.with_name('.' + out.name + '.final-install.tmp')
        tmp.write_text(text, encoding='utf-8', newline='')
        os.replace(tmp, out)
    print(json.dumps({'state': state, 'sha256': sha256_bytes(text.encode()), 'qualified': qualified,
                      'canonical_sha256': canon, 'required_kernel_modules': modules}))


# ------------------------------------------------------------------------------------------------ mapping
def cmd_mapping(a):
    validator = load_module('qwen36_profile_runtime_final', a.validator)
    mq = load_module('qwen36_make_qualification_final', a.mq)
    doc = build_profiles(a.draft, a.gather, ['main128', 'long256', 'baseline64'])
    validator.validate_profiles_document(doc)
    result, ok = {}, True
    for name, cfg_path in (('main128', a.main), ('long256', a.long), ('baseline64', a.b64)):
        cfg = load_json(cfg_path)
        want = mq.production_semantics(doc['profiles'][name], validator)
        variants = {'run': cfg}
        if name != 'baseline64':
            teacher = copy.deepcopy(cfg)
            teacher['env'].update(TEACHER_ENV)
            variants['teacher'] = teacher
        rows = {}
        for label, variant in variants.items():
            try:
                got = mq.run_equivalent_profile(variant, validator.FEATURE_SWITCHES, validator.KERNEL_LOADER_ENV,
                                                validator.VALUE_FIELDS)
                diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
                rows[label] = {'equal': not diff, 'diff': {k: [got.get(k), want.get(k)] for k in diff}}
            except Exception as error:  # QualificationError and friends
                rows[label] = {'equal': False, 'error': type(error).__name__ + ': ' + str(error)[:300]}
            ok = ok and rows[label]['equal']
        result[name] = rows
    modules = validator.required_kernel_modules(doc)
    expect_modules = [GATHER_MODULE] if a.gather == 'in' else []
    result['required_kernel_modules'] = {'got': modules, 'expected': expect_modules}
    ok = ok and modules == expect_modules
    result['required_libraries'] = sorted(validator.required_libraries(doc))
    result['canonical_sha256'] = {n: validator.canonical_profile_sha256(p) for n, p in doc['profiles'].items()}
    result['pass'] = ok
    print(json.dumps(result, indent=1, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o)))
    need(ok, 'run configs do not map onto the release profiles')


# ------------------------------------------------------------------------------------------------ qcheck
def cmd_qcheck(a):
    here = pathlib.Path(__file__).resolve().parent
    mq_path = here.parent/'make-qualification.py'
    if not mq_path.is_file():mq_path = here/'make-qualification.py'
    mq = load_module('quality_gate_contract', mq_path)
    try:
        d = mq.quality_evidence_module().strict_json(pathlib.Path(a.report).read_text())
        if type(d) is not dict:raise ValueError('quality report must be an object')
        mq.check_quality_bitexact_report(d)
        need(d.get('ppl_gate_pass') is True, 'PPL gate failed')
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({'pass':False,'validation_error':str(error)}))
        need(False, 'teacher/quality gate failed')
    lc = d['logit_comparisons']
    out = {'candidate': d['candidate'], 'baseline': d['baseline'], 'expected': lc['expected'],
           'observed': lc['observed'], 'finite_and_complete': lc['finite_and_complete'],
           'max_abs': lc['max_abs'], 'max_relative_l2': lc['max_relative_l2'],
           'forced_targets_complete': d['forced_targets_complete'], 'ppl_gate_pass': d['ppl_gate_pass'],
           'quality_gate_pass': d['quality_gate_pass'], 'new_functional_failures': d['new_functional_failures'],
           'relative_perplexity_growth': d.get('relative_perplexity_growth'),
           'evidence_valid':d['evidence_valid'], 'bit_exact_pass':d['bit_exact_pass'],
           'all_byte_equal':lc['all_byte_equal'], 'pass':True}
    print(json.dumps(out))


# ------------------------------------------------------------------------------------------------ extract-run
def cmd_extract_run(a):
    report = load_json(a.report)
    run = report.get('run')
    cfg = load_json(a.config)
    if a.teacher:
        cfg = copy.deepcopy(cfg)
        cfg['env'].update(TEACHER_ENV)
    need(isinstance(run, dict), 'report has no run receipt')
    for key in ('run_id', 'profile', 'configuration', 'source_sha256', 'source_files', 'service_source_files',
                'native_libraries', 'checkpoint', 'checkpoint_revision', 'packages'):
        need(key in run, 'run receipt lacks ' + key)
    need(run['profile'] == a.profile, f"run receipt profile {run['profile']} != {a.profile}")
    need(run['configuration'] == cfg, 'run receipt configuration differs from ' + a.config)
    for env, row in run['native_libraries'].items():
        need(isinstance(row, dict) and row.get('path') == cfg['env'].get(env), 'native library path mismatch ' + env)
    text = json.dumps(run, indent=2) + '\n'
    out = pathlib.Path(a.out)
    if out.exists():
        need(out.read_bytes() == text.encode(), out.name + ' exists with other content')
        state = 'same'
    else:
        write_excl(out, text)
        state = 'written'
    print(json.dumps({'state': state, 'out': str(out), 'run_id': run['run_id'], 'source_sha256': run['source_sha256'],
                      'native_libraries': {k: v.get('sha256', '')[:16] for k, v in run['native_libraries'].items()}}))


# ------------------------------------------------------------------------------------------------ provenance
def cmd_provenance(a):
    raw = pathlib.Path(a.file).read_text(encoding='utf-8', errors='replace')
    start = raw.find('{')
    need(start >= 0, 'no JSON in provenance output')
    d = json.loads(raw[start:])
    need(d.get('ok', True) is not False, 'kernel provenance check failed: ' + str(d.get('error'))[:300])
    need(d.get('manifest'), 'no kernel provenance manifest')
    modules = {m['name']: m['status'] for m in d.get('modules', [])}
    extensions = {e['name']: e['status'] for e in d.get('extensions', [])}
    bad = sorted(n for n, s in modules.items() if s != 'prebuilt')
    out = {'modules': len(modules), 'not_prebuilt': bad, '_pinned_tensor': extensions.get('_pinned_tensor')}
    if a.require_module:
        out[a.require_module] = modules.get(a.require_module, 'ABSENT')
    print(json.dumps(out))
    need(not bad, 'modules not prebuilt (release requires prebuilt): ' + ','.join(bad[:5]))
    if not a.modules_only:  # the workspace tree has no native-only extension .so (final-prep checks modules only)
        need(extensions.get('_pinned_tensor') == 'prebuilt', '_pinned_tensor not prebuilt')
    if a.require_module:
        need(modules.get(a.require_module) == 'prebuilt', a.require_module + ' is not a manifest-bound prebuilt')


# ------------------------------------------------------------------------------------------------ summary
GATES = {
    'main128': [('A', {'PASS'}), ('B.main-release', {'PASS'}), ('D.pilot', {'PASS'}), ('D.matrix', {'PASS'}),
                ('D.compare', {'PASS', 'INCONCLUSIVE'}), ('C.c0', {'PASS', 'SKIPPED_BY_PARENT'}),
                ('E.pilot', {'PASS'}), ('E.soak', {'PASS'})],
    'long256': [('A', {'PASS'}), ('B.long256-release', {'PASS', 'PASS_EQUIV_R1B256'}), ('D.manifest', {'PASS'}), ('C.l4', {'PASS'})],
    'baseline64': [('A', {'PASS'}), ('D.manifest', {'PASS'}), ('C.f1', {'PASS', 'PASS_WITH_KNOWN_LIMITATION'})],
}


def read_steps(path):
    steps = {}
    if not pathlib.Path(path).exists():
        return steps
    for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split(' ', 2)
        if len(parts) >= 2:
            steps[parts[0]] = {'status': parts[1], 'detail': parts[2] if len(parts) > 2 else ''}
    return steps


def detail_value(steps, step, key):
    for token in steps.get(step, {}).get('detail', '').split():
        if token.startswith(key + '='):
            return token[len(key) + 1:]
    return None


def evidence_lists(run, steps):
    """Evidence per profile; attempt-suffixed names come from the steps file (tag=/name= tokens)."""
    T = 'final-' + run
    R = 'reports/'
    pilot = detail_value(steps, 'D.pilot', 'name') or f'{T}-tool-a1'
    c0 = detail_value(steps, 'C.c0', 'tag')
    l4 = detail_value(steps, 'C.l4', 'tag') or f'{T}-l256-a1'
    f1 = detail_value(steps, 'C.f1', 'tag') or f'{T}-b64-a1'
    soak_pilot = detail_value(steps, 'E.pilot', 'name') or f'soak-pilot-{T}-a1'
    soak = detail_value(steps, 'E.soak', 'name') or f'soak-{T}-a1'
    common_ref = [R + n for n in ('teacher-bf128-logits.json', 'functional-bf128.json',
                                  'eval-v1-basis-decision-legacyfp16.json', 'logit-reference-r1-vs-pre-r1.json')]
    manifest = [R + f'{T}-m-manifest.json']
    gate_files = lambda tag: [R + f'{tag}-{k}.json' for k in ('identity', 'preflight', 'run', 'functional', 'boundary',
                                                              'backend', 'prefill-cancel', 'evaluation')]
    qual = lambda c: [R + f'teacher-{c}.json', R + f'functional-{c}.json', R + f'quality-{c}.json',
                      R + f'{T}-qrun-teacher-{c}.json', R + f'{T}-qrun-functional-{c}.json']
    cuda = lambda c: {'run_receipts': [R + f'{T}-qrun-teacher-{c}.json'], 'bitexact_reports': [R + f'quality-{c}.json']}
    main_ev = (qual('main-release') + common_ref + manifest
               + [R + f'{T}-m-matrix.json', R + f'{T}-m-main-release-vs-bf128-comparison.json',
                  R + f'{pilot}-tool-pilot.json', R + f'{pilot}-tool-pilot-runner.json']
               + (gate_files(c0) if c0 else [])
               + [R + f'{soak_pilot}-analysis.json', R + f'{soak_pilot}-tunables-pre.json']
               + [R + f'{soak}.json', R + f'{soak}-run.json', R + f'{soak}-analysis.json',
                  R + f'{soak}-tunables-pre.json', R + f'{soak}-tunables-post.json', R + f'kernel-provenance-{T}.json'])
    # 262144-context teacher deviates from the 128K reference at long-120000 exactly like the R1b 256K profile
    # (parent rule PASS_EQUIV_R1B256): carry that profile's report and the main128 bit-exact report of the same libraries.
    long_ev = (qual('long256-release') + common_ref + manifest + gate_files(l4) + [R + f'kernel-provenance-{T}.json']
               + [R + 'quality-long256-final-cuda2048.json', R + 'teacher-long256-final-cuda2048.json', R + 'quality-main-release.json'])
    b64_ev = manifest + gate_files(f1) + [R + 'functional-baseline64-original-60k.json',
                                          R + 'eval-v1-basis-decision-legacyfp16.json']
    return {
        'main128': {'evidence': main_ev,
                    'run_receipts': [R + f'{T}-qrun-functional-main-release.json']
                                    + ([R + f'{c0}-run.json'] if c0 else []) + [R + f'{soak}-run.json'],
                    'cuda': cuda('main-release')},
        'long256': {'evidence': long_ev,
                    'run_receipts': [R + f'{T}-qrun-functional-long256-release.json', R + f'{l4}-run.json'],
                    'cuda': {'run_receipts': [R + f'{T}-qrun-teacher-long256-release.json'],
                             'bitexact_reports': [R + 'quality-main-release.json']}},
        'baseline64': {'evidence': b64_ev, 'run_receipts': [R + f'{f1}-run.json'], 'cuda': None},
    }


def cmd_summary(a):
    steps = read_steps(a.steps)
    profiles = {}
    for name, gates in GATES.items():
        rows, ok = [], True
        for step, allowed in gates:
            status = steps.get(step, {}).get('status', 'NOT_RUN')
            good = status in allowed
            ok = ok and good
            rows.append({'step': step, 'status': status, 'ok': good})
        profiles[name] = {'pass': ok, 'gates': rows}
    summary = {'tool': 'final-gates', 'run_id': a.run, 'gather': a.gather,
               'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'steps': steps, 'profiles': profiles, 'releasable': profiles['main128']['pass'],
               'note': 'main128 is the default profile: nothing can be released unless it passes'}
    tmp = pathlib.Path(a.out).with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, a.out)
    # Decision draft (owner/parent must review; final-install refuses any TODO).
    reports = pathlib.Path(a.reports)
    lists = evidence_lists(a.run, steps)
    carried = []
    if a.release_prep_draft and pathlib.Path(a.release_prep_draft).exists():
        old = load_json(a.release_prep_draft)['profiles']
        carried = [x for x in old['main128']['known_limitations'] if x.startswith(('Numeric basis', 'Inherited baseline', 'API contract'))]
        b64_60k = [x for x in old['baseline64']['known_limitations'] if x.startswith('Functional long-facts-60000')]
    else:
        b64_60k = []
    decision = {'schema': 1, 'profiles': {}}
    missing = {}
    for name in ('main128', 'long256', 'baseline64'):
        if not profiles[name]['pass']:
            continue
        lst = lists[name]
        present = [p for p in lst['evidence'] if (reports.parent / p).is_file()]
        missing[name] = [p for p in lst['evidence'] if p not in present]
        limits = ['TODO(parent): review every known limitation below, add speed/VRAM/soak facts from the summary, '
                  'then delete this line'] + list(carried)
        dc = steps.get('D.compare', {})
        if name == 'main128':
            limits.append(f"AUTO (review): speed final-{a.run}-m main-release vs bf128 (5 pairs, V6): {dc.get('status')} "
                          f"{dc.get('detail', '')}".strip())
            limits.append(f"AUTO (review): soak {steps.get('E.soak', {}).get('detail', '')}".strip())
            limits.append(f"AUTO (review): main128 context gates {steps.get('C.c0', {}).get('status')} "
                          f"{steps.get('C.c0', {}).get('detail', '')}".strip())
        if name == 'long256':
            limits.append(f"AUTO (review): long256 L4 {steps.get('C.l4', {}).get('detail', '')}; no dedicated 2 h soak "
                          f"at 256K (soak evidence = main128, same build)".strip())
            limits.append(f"AUTO (review): long256 teacher {steps.get('B.long256-release', {}).get('status')}: vs the 128K "
                          "teacher-bf128 reference every case is bit-identical except long-120000 (max_abs 1.232421875, "
                          "mean NLL +0.0177), identical to the last digit to the R1b-qualified long256-final-cuda2048 profile "
                          "(262144-context property, not this build); CUDA bit-exact evidence = quality-main-release.json "
                          "(same libraries); bf256 (Triton 256K) teacher comparison NOT_RUN")
        if name == 'baseline64':
            limits.append(f"AUTO (review): baseline64 F1 {steps.get('C.f1', {}).get('detail', '')}; no dedicated soak".strip())
            if steps.get('C.f1', {}).get('status') == 'PASS_WITH_KNOWN_LIMITATION':
                limits += b64_60k
        entry = {'acceptance_status': 'accepted_with_known_limits', 'known_limitations': limits, 'evidence': present,
                 'run_receipts': [p for p in lst['run_receipts'] if p in present]}
        if lst['cuda']:
            cuda = {'run_receipts': [p for p in lst['cuda']['run_receipts'] if p in present],
                    'bitexact_reports': [p for p in lst['cuda']['bitexact_reports'] if p in present]}
            for key in ('cuda_nvfp4_evidence', 'cuda_gdn_evidence', 'cuda_fp8_gemv_evidence'):
                entry[key] = copy.deepcopy(cuda)
        decision['profiles'][name] = entry
    state = 'not-written (main128 did not pass)'
    if decision['profiles'].get('main128'):
        draft = pathlib.Path(a.decision_out)
        if draft.exists():
            state = 'exists (kept)'
        else:
            write_excl(draft, json.dumps(decision, indent=2) + '\n')
            state = 'written'
    print(json.dumps({'releasable': summary['releasable'],
                      'profiles': {n: p['pass'] for n, p in profiles.items()},
                      'decision_draft': state, 'missing_evidence': missing}))


def cmd_decision_check(a):
    text = pathlib.Path(a.decision).read_text(encoding='utf-8')
    need('TODO' not in text, 'decision still contains TODO markers (parent review pending)')
    decision = json.loads(text)
    summary = load_json(a.summary)
    names = sorted(decision.get('profiles', {}))
    need('main128' in names, 'decision does not accept main128 (default profile)')
    for name in names:
        need(summary['profiles'].get(name, {}).get('pass') is True,
             f'{name}: final-gates did not pass all its gates (see summary); refusing to qualify it')
        entry = decision['profiles'][name]
        for rel in entry.get('evidence', []):
            need((pathlib.Path(a.reports).parent / rel).is_file(), f'{name}: evidence missing: {rel}')
    print(','.join(names))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('configs'); s.add_argument('--gather', choices=('in', 'out'), required=True)
    s.add_argument('--opt', required=True); g = s.add_mutually_exclusive_group(required=True)
    g.add_argument('--print', action='store_true'); g.add_argument('--check', action='store_true')
    g.add_argument('--write', action='store_true')
    s = sub.add_parser('mapping'); s.add_argument('--gather', choices=('in', 'out'), required=True)
    for k in ('--validator', '--mq', '--draft', '--main', '--long', '--b64'):
        s.add_argument(k, required=True)
    s = sub.add_parser('profiles'); s.add_argument('--gather', choices=('in', 'out'), required=True)
    s.add_argument('--draft', required=True); s.add_argument('--qualified', required=True)
    s.add_argument('--out', required=True); s.add_argument('--validator'); s.add_argument('--check', action='store_true')
    s = sub.add_parser('qcheck'); s.add_argument('report')
    s = sub.add_parser('extract-run'); s.add_argument('--report', required=True); s.add_argument('--profile', required=True)
    s.add_argument('--config', required=True); s.add_argument('--teacher', action='store_true'); s.add_argument('--out', required=True)
    s = sub.add_parser('provenance'); s.add_argument('file'); s.add_argument('--require-module')
    s.add_argument('--modules-only', action='store_true')
    s = sub.add_parser('summary'); s.add_argument('--steps', required=True); s.add_argument('--run', required=True)
    s.add_argument('--gather', choices=('in', 'out'), required=True); s.add_argument('--reports', required=True)
    s.add_argument('--out', required=True); s.add_argument('--decision-out', required=True)
    s.add_argument('--release-prep-draft')
    s = sub.add_parser('decision-check'); s.add_argument('--decision', required=True)
    s.add_argument('--summary', required=True); s.add_argument('--reports', required=True)
    a = p.parse_args(argv)
    fn = {'configs': cmd_configs, 'mapping': cmd_mapping, 'profiles': cmd_profiles, 'qcheck': cmd_qcheck,
          'extract-run': cmd_extract_run, 'provenance': cmd_provenance, 'summary': cmd_summary,
          'decision-check': cmd_decision_check}[a.cmd]
    try:
        fn(a)
    except Fail as error:
        print(json.dumps({'result': 'FAIL', 'cmd': a.cmd, 'reason': str(error)}), file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'result': 'ERROR', 'cmd': a.cmd, 'error': type(error).__name__ + ': ' + str(error)[:300]}),
              file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
