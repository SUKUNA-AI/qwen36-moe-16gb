"""CPU-only helpers for release-gates.sh (stdlib only, no torch, no GPU, no secrets).

Subcommands (exit 0 ok, 3 check failed, 2 usage):
  env FILE                      site paths from config/local.env as shell assignments (QWEN36_ROOT, QWEN36_NATIVE)
  evidence --reports DIR --reference REF --functional-baseline FB
                                REF is run-bound teacher-bf128 evidence and FB is its bound functional arm (H1)
  extract-run --report R --profile P --config CFG [--teacher] --out OUT
                                standalone run receipt; a teacher run may differ from CFG only by RUN_DIAGNOSTIC_ENV
  qcheck REPORT                 the f1 strict teacher/quality gate (final-helpers.py qcheck, unchanged)
  context-check --reports DIR --name NAME --profile P --teacher-report T --functional-report F
                                run-context-gates.sh NAME: every report passed, the functional run is the teacher
                                arm's serving run, no functional case fails that passed in F
  screen --reports DIR --matrix M --baseline B --candidate C --out OUT
                                median elapsed/TTFT/decode per length from a matrix (any pair count); not a gate
  reference-soak --analysis A  A is a PASS soak analysis whose receipt sits next to it (soak-analysis load_reference)
  soak-check --analysis A [--release | --pilot]
                                soak-analysis verdict: --release needs the two-hour PASS; default (screen soak): every
                                case type passed and nothing but the duration failed (the reference comparison
                                included); --pilot (30 s): as default but without the reference comparison
  summary --steps FILE --run RUN --out OUT [--release]
"""
import argparse
import importlib.util
import json
import os
import pathlib
import re
import shlex
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ENV_KEYS = ('QWEN36_ROOT', 'QWEN36_NATIVE')
# Placeholders of the sanitized tracked files; a real site config never contains them.
PLACEHOLDERS = ('/home/USER', '/path/to/')
# Stage -> statuses that let main128 ship. Screen mode never ships (3 pairs, 30 min soak).
RELEASE_GATES = (('A', {'PASS'}), ('B', {'PASS'}), ('D.pilot', {'PASS'}), ('D.matrix', {'PASS'}),
                 ('D.compare', {'PASS', 'INCONCLUSIVE'}), ('C', {'PASS'}), ('E.pilot', {'PASS'}), ('E.soak', {'PASS'}))


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_qualification():
    path = HERE.parent / 'make-qualification.py'
    if not path.is_file():
        path = HERE / 'make-qualification.py'
    return load_module('release_make_qualification', path)


def evidence_module():
    return make_qualification().quality_evidence_module()


def load_json(path):
    return evidence_module().strict_json(pathlib.Path(path).read_text(encoding='utf-8'))


def write_excl(path, text):
    path = pathlib.Path(path)
    if path.exists():
        need(path.read_bytes() == text.encode(), path.name + ' exists with other content')
        return 'same'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)
    return 'written'


def cmd_env(a):
    values = {}
    for line in pathlib.Path(a.file).read_text(encoding='utf-8').splitlines():
        m = re.match(r"\s*(QWEN36_[A-Z_]+)\s*=\s*('([^']*)'|\"([^\"]*)\"|([^\s#]*))", line)
        if m:
            values[m.group(1)] = next(g for g in m.group(3, 4, 5) if g is not None)
    for key in ENV_KEYS:
        value = values.get(key, '')
        need(value.startswith('/') and not any(p in value for p in PLACEHOLDERS),
             f'{a.file}: {key} must be an absolute site path, not a placeholder')
        print(f'{key}={shlex.quote(value.rstrip("/"))}')


def cmd_evidence(a):
    qe = evidence_module()
    reports = pathlib.Path(a.reports)
    try:
        reference = qe.strict_json((reports / f'{a.reference}.json').read_text(encoding='utf-8'))
        qe.bind_teacher(reference, qe.evidence_records(reports / 'teacher.jsonl', [reference]), 'teacher-bf128')
        functional = qe.strict_json((reports / f'{a.functional_baseline}.json').read_text(encoding='utf-8'))
        qe.functional_rows(functional, qe.serving_profile('teacher-bf128'), reference['run'])
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise Fail(f'reference/functional baseline is not run-bound schema-2 evidence: {error}') from None
    print(json.dumps({'reference_run': reference['run']['run_id'], 'functional_run': functional['run']['run_id']}))


def cmd_extract_run(a):
    diagnostic = make_qualification().RUN_DIAGNOSTIC_ENV
    report, cfg = load_json(a.report), load_json(a.config)
    run = report.get('run')
    need(isinstance(run, dict), 'report has no run receipt')
    for key in ('run_id', 'profile', 'configuration', 'source_sha256', 'source_files', 'service_source_files',
                'native_libraries', 'checkpoint', 'checkpoint_revision', 'packages'):
        need(key in run, 'run receipt lacks ' + key)
    need(run['profile'] == a.profile, f"run receipt profile {run['profile']} != {a.profile}")
    served = json.loads(json.dumps(run['configuration']))
    need(isinstance(served, dict) and isinstance(served.get('env'), dict), 'run receipt configuration is invalid')
    knobs = set(served['env']) & set(diagnostic)
    if a.teacher:
        # qualify.py adds exactly the teacher knobs to the candidate config; nothing else may differ.
        need(knobs == set(diagnostic), f'teacher run lacks teacher knobs: {sorted(set(diagnostic) - knobs)}')
    else:
        need(not knobs, f'functional run carries teacher knobs: {sorted(knobs)}')
    for key in knobs:
        served['env'].pop(key)
    for doc in (served, cfg):
        doc.setdefault('flags', [])
    need(served == cfg, 'run receipt configuration differs from ' + a.config)
    for env, row in run['native_libraries'].items():
        need(isinstance(row, dict) and row.get('path') == cfg['env'].get(env), 'native library path mismatch ' + env)
    state = write_excl(a.out, json.dumps(run, indent=2) + '\n')
    print(json.dumps({'state': state, 'out': str(a.out), 'run_id': run['run_id'], 'teacher_knobs': sorted(knobs)}))


def cmd_qcheck(a):
    final = load_module('final_helpers_qcheck', HERE / 'final-helpers.py')
    try:
        final.cmd_qcheck(a)
    except final.Fail as error:
        raise Fail(str(error)) from None


def cmd_context_check(a):
    qe = evidence_module()
    reports = pathlib.Path(a.reports)
    read = lambda name: qe.strict_json((reports / f'{name}.json').read_text(encoding='utf-8'))
    try:
        teacher, baseline = read(a.teacher_report), read(a.functional_report)
        functional = read(a.name + '-functional')
        rows = qe.functional_rows(functional, a.profile, teacher['run'])
        base = qe.functional_rows(baseline, a.profile, teacher['run'])
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise Fail(f'context functional evidence is not bound to the qualified run: {error}') from None
    # The long case may use another length than the qualify run; it is compared with that run's long case.
    base_long = base[f'long-facts-{baseline["functional_matrix"]["long"]}']
    regressions = sorted(case for case, row in rows.items()
                         if not row['pass'] and base.get(case, base_long)['pass'])
    checks = {'boundary': read(a.name + '-boundary').get('pass') is True,
              'backend': read(a.name + '-backend').get('all_pass') is True}
    cancel = read(a.name + '-cancel')
    checks['cancel'] = cancel.get('pass') is True and (cancel.get('run') or {}).get('run_id') == functional['run']['run_id']
    out = {'name': a.name, 'functional_run': functional['run']['run_id'], 'functional_failures':
           sorted(c for c, r in rows.items() if not r['pass']), 'new_functional_failures': regressions, **checks}
    print(json.dumps(out))
    need(not regressions and all(checks.values()), 'context gate failed: ' + json.dumps(out))


def cmd_screen(a):
    reports = pathlib.Path(a.reports)
    arms = json.loads((reports / f'{a.matrix}-matrix.json').read_text(encoding='utf-8'))
    values = {}
    for arm in arms:
        if arm.get('status') != 'complete' or arm.get('profile') not in (a.baseline, a.candidate):
            continue
        bench = json.loads((reports / arm['report']).read_text(encoding='utf-8'))
        for row in bench.get('rows', []):
            if row.get('case') != 'fresh' or row.get('status') != 200:
                continue
            for metric in ('elapsed_s', 'ttft_content_s', 'decode_client_tps'):
                if type(row.get(metric)) in (int, float):
                    values.setdefault((arm['profile'], row['length'], metric), []).append(row[metric])
    lengths = sorted({key[1] for key in values})
    rows = []
    for length in lengths:
        row = {'length': length}
        for metric in ('elapsed_s', 'ttft_content_s', 'decode_client_tps'):
            b, c = values.get((a.baseline, length, metric)), values.get((a.candidate, length, metric))
            row[metric] = {'baseline_median': statistics.median(b) if b else None,
                           'candidate_median': statistics.median(c) if c else None, 'pairs': min(len(b or ()), len(c or ()))}
        e = row['elapsed_s']
        row['elapsed_speedup'] = (e['baseline_median'] / e['candidate_median']
                                  if e['baseline_median'] and e['candidate_median'] else None)
        rows.append(row)
    out = {'kind': 'screen', 'gate': False, 'matrix': a.matrix, 'baseline': a.baseline, 'candidate': a.candidate,
           'rows': rows, 'note': 'medians only; the release decision is compare-ab over five pairs (--release)'}
    write_excl(a.out, json.dumps(out, indent=2) + '\n')
    print(json.dumps({'out': str(a.out), 'lengths': lengths}))
    need(bool(rows), 'matrix has no complete arms to screen')


def cmd_reference_soak(a):
    analysis = load_module('release_soak_analysis', HERE.parent / 'soak-analysis.py')
    try:
        profile = analysis.load_reference(a.analysis, analysis.load_soak())
    except (OSError, ValueError) as error:
        raise Fail(f'reference soak unusable: {error}') from None
    need(profile['vram'].get('p1_gpu_free_bytes') is not None and profile['fresh'], 'reference soak has no VRAM/TTFT data')
    print(json.dumps({'source': profile['source'], 'p1_gpu_free_mib': profile['vram']['p1_gpu_free_bytes'] // (1 << 20),
                      'lengths': sorted(profile['fresh'], key=int)}))


def cmd_soak_check(a):
    d = json.loads(pathlib.Path(a.analysis).read_text(encoding='utf-8'))
    pilot = d.get('pilot_view') or {}
    vram = (d.get('summary') or {}).get('vram') or {}
    key = 'failures_excluding_duration_and_reference' if a.pilot else 'failures_excluding_duration'
    if a.release:
        ok = d.get('verdict') == 'PASS' and d.get('two_hour_execution_gate_pass') is True
    else:
        ok = pilot.get('all_case_types_passed') is True and pilot.get(key) == []
    print(json.dumps({'mode': 'release' if a.release else 'pilot' if a.pilot else 'screen', 'verdict': d.get('verdict'),
                      'pass': ok, key: pilot.get(key), 'vram_report_only': vram}))
    need(ok, 'soak gate failed')


def steps_status(path):
    status, detail = {}, {}
    if pathlib.Path(path).exists():
        for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
            parts = line.split(' ', 2)
            if len(parts) >= 2 and parts[0] not in ('CONFIG',):
                status[parts[0]] = parts[1]
                detail[parts[0]] = parts[2] if len(parts) > 2 else ''
    return status, detail


def cmd_summary(a):
    status, detail = steps_status(a.steps)
    gates = []
    for step, allowed in RELEASE_GATES:
        # A screen run has no compare-ab verdict; its median screen is recorded instead.
        allowed = allowed if a.release or step != 'D.compare' else {'SCREEN'}
        gates.append({'step': step, 'status': status.get(step, 'NOT_RUN'), 'ok': status.get(step) in allowed,
                      'detail': detail.get(step, '')})
    blocked = [g['step'] for g in gates if not g['ok']]
    reasons = blocked + ([] if a.release else ['screen_mode_never_releasable'])
    out = {'run': a.run, 'mode': 'release' if a.release else 'screen', 'gates': gates, 'gates_pass': not blocked,
           'releasable': not reasons, 'blocking': reasons, 'stopped': status.get('STOP')}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'gates_pass': out['gates_pass'], 'releasable': out['releasable'], 'blocking': reasons}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('env'); s.add_argument('file')
    s = sub.add_parser('evidence'); s.add_argument('--reports', required=True); s.add_argument('--reference', required=True)
    s.add_argument('--functional-baseline', required=True)
    s = sub.add_parser('extract-run'); s.add_argument('--report', required=True); s.add_argument('--profile', required=True)
    s.add_argument('--config', required=True); s.add_argument('--teacher', action='store_true'); s.add_argument('--out', required=True)
    s = sub.add_parser('qcheck'); s.add_argument('report')
    s = sub.add_parser('context-check')
    for key in ('--reports', '--name', '--profile', '--teacher-report', '--functional-report'):
        s.add_argument(key, required=True)
    s = sub.add_parser('screen')
    for key in ('--reports', '--matrix', '--baseline', '--candidate', '--out'):
        s.add_argument(key, required=True)
    s = sub.add_parser('reference-soak'); s.add_argument('--analysis', required=True)
    s = sub.add_parser('soak-check'); s.add_argument('--analysis', required=True)
    g = s.add_mutually_exclusive_group(); g.add_argument('--release', action='store_true'); g.add_argument('--pilot', action='store_true')
    s = sub.add_parser('summary'); s.add_argument('--steps', required=True); s.add_argument('--run', required=True)
    s.add_argument('--out', required=True); s.add_argument('--release', action='store_true')
    a = p.parse_args(argv)
    fn = {'env': cmd_env, 'evidence': cmd_evidence, 'extract-run': cmd_extract_run, 'qcheck': cmd_qcheck,
          'context-check': cmd_context_check, 'screen': cmd_screen, 'soak-check': cmd_soak_check,
          'reference-soak': cmd_reference_soak,
          'summary': cmd_summary}[a.cmd]
    try:
        fn(a)
    except Fail as error:
        print(json.dumps({'result': 'FAIL', 'cmd': a.cmd, 'reason': str(error)[:600]}), file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'result': 'ERROR', 'cmd': a.cmd, 'error': type(error).__name__ + ': ' + str(error)[:300]}),
              file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
