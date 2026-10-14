"""GPU validation of the harness-v7 teacher clean-cache step (patch/quality.py), phase 1.

Runs the FULL 20-case teacher sequence for one already-allowlisted release teacher profile with the patched
quality.teacher() (19 cases on one server, then a restart of the same profile, then long-120000 as the fresh
server's first request), against the EXISTING teacher-bf128 reference (reports/logit-reference, run 345788b2).

Read-only use of the native harness: bench/fixtures/matrix/compare-quality/make-qualification are imported or
executed from $N unchanged; nothing in $N is written except what the engine itself writes for any run
(reports/teacher.jsonl, engine/request logs, active-run.json). fixtures.prepare() is NOT called (it would write
config/teacher-fixture.json); the fixture file is instead checked equal to the cases sent.

Assembly check: the UNCHANGED compare-quality.py runs in a sandbox root (out_dir/cq-<cand>/reports) holding a
copy of teacher.jsonl taken after the run, this run's teacher report, and copies of the functional reports.
Then make-qualification.check_quality_bitexact_report (the release bit-exact rule) is applied to its output.

The GPU lock is taken by the caller (gpu-session.sh: flock -w 3600 -E 75), so this script does not take it.
"""
import argparse, hashlib, importlib.util, json, math, pathlib, shutil, subprocess, sys, time

sys.dont_write_bytecode = True
NATIVE = pathlib.Path('/home/USER/qwen36-api/optimization')
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE))
from fixtures import teacher_cases  # noqa: E402  (native, read-only)
from matrix import run, PY          # noqa: E402  (same start semantics as qualify.start)

REF_RUN = '345788b251dc4ec1a6788148abbd7596'  # teacher-bf128 reference run (logit-reference written 21:40-21:42Z)
ALLOWED = {'teacher-main-release': 'main-release', 'teacher-long256-release': 'long256-release'}
NATIVE_SHA = {'compare-quality.py': 'fae2690d122c39cc4d9df9f9b7d2d297b317eb4c08426335d6d6155135a6a947',
              'make-qualification.py': '6398c923d3a7a4a93c22a4cb18b467a130acae51bc1cb32ddc22feff437268dc',
              'fixtures.py': '555a3cd62e95ec4ece2ea01650bfb654d40efe7e1a196bfceb36ae6f0b8a947e',
              'bench.py': '204d28710ef0fb762b4a57bb6f4fe579450d862996e90ddafa5285ef28e3b28a',
              'matrix.py': '687b22f2d02c2c0be2d6f90ae966c20a73546de1016609e0945767823c9dc823',
              'start-experiment.sh': '90b8b08c054e992dcd6cace8cc048427f60d6d13bb4ba71f5072d8f0e297bdb7',
              'check-experiment.py': '9bb324ad9be0694f06e9af1755576327186bb14424e48b1c824e48f4968c05cc',
              'quality.py': '6e035f052857b98f366681a2c29233d9433be831655f6fd7baff7752bff5e34d'}
CQ_ROOT_LINE = "ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher_jsonl():
    return [json.loads(l) for l in (NATIVE / 'reports/teacher.jsonl').read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', required=True, choices=sorted(ALLOWED))
    p.add_argument('--out-dir', required=True)
    p.add_argument('--assemble-only', help='CPU: skip the GPU run, re-assemble from this teacher report JSON')
    a = p.parse_args()
    cand = ALLOWED[a.profile]
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    hashes = {name: sha(NATIVE / name) for name in NATIVE_SHA}
    if hashes != NATIVE_SHA:
        raise SystemExit(f'native harness differs from the frozen hashes: {hashes}')
    patched = HERE / 'patch/quality.py'
    quality = load('quality_v7', patched)  # imports bench/fixtures/matrix from $N
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model', local_files_only=True)
    cases = teacher_cases(tok)
    fixture = json.loads((NATIVE / 'config/teacher-fixture.json').read_text())
    expected = {str(c['seed']): {'case': c['case'], 'ids': c['ids']} for c in cases}
    if fixture.get('targets') != expected:
        raise SystemExit('config/teacher-fixture.json differs from the fixture cases')
    timeline = []
    if a.assemble_only:
        report = json.loads(pathlib.Path(a.assemble_only).read_text())
        rows, first, clean = report['rows'], report['run'], report['clean_cache']['runs']
        before_lines = 0
    else:
        before_lines = len(teacher_jsonl())
        t0 = time.monotonic()
        run(['systemctl', '--user', 'stop', 'qwen36-experiment'], 60)
        try:
            run(['bash', str(NATIVE / 'start-experiment.sh'), a.profile], 60)
            run([PY, str(NATIVE / 'check-experiment.py'), '300'], 310)
            timeline.append(('first_ready_s', round(time.monotonic() - t0, 1)))
            rows, first, clean = quality.teacher(tok, cases)
            timeline.append(('teacher_done_s', round(time.monotonic() - t0, 1)))
        finally:
            subprocess.run(['systemctl', '--user', 'stop', 'qwen36-experiment'], timeout=60)
        time.sleep(2)
        report = {'fixture_version': 1, 'run': first, 'rows': rows,
                  'clean_cache': {'harness': 'v7', 'mode': 'restart_same_profile', 'cases': list(quality.CLEAN_CACHE_CASES),
                                  'runs': clean}}
        (out / f'{a.profile}.json').write_text(json.dumps(report, indent=2))

    # --- assembly with the unchanged compare-quality.py (sandbox root) ---
    box = out / f'cq-{cand}'
    if box.exists(): shutil.rmtree(box)
    (box / 'reports').mkdir(parents=True)
    shutil.copyfile(NATIVE / 'reports/teacher.jsonl', box / 'reports/teacher.jsonl')
    for name in ('functional-bf128.json', f'functional-{cand}.json'):
        shutil.copyfile(NATIVE / 'reports' / name, box / 'reports' / name)
    (box / 'reports' / f'{a.profile}.json').write_text(json.dumps(report, indent=2))
    src = (NATIVE / 'compare-quality.py').read_text()
    assert src.count(CQ_ROOT_LINE) == 1
    src = src.replace(CQ_ROOT_LINE, f'ROOT=pathlib.Path({str(box)!r})')
    argv = sys.argv
    sys.argv = ['compare-quality.py', '--candidate', a.profile, '--functional-candidate', f'functional-{cand}',
                '--name', f'quality-{cand}-v7']
    try:
        exec(compile(src, 'compare-quality.py', 'exec'), {'__name__': '__main__'})
    finally:
        sys.argv = argv
    cq = json.loads((box / 'reports' / f'quality-{cand}-v7.json').read_text())
    mq = load('make_qualification_ro', NATIVE / 'make-qualification.py')
    try:
        mq.check_quality_bitexact_report(cq); bitexact_rule = 'PASS'
    except Exception as e:  # noqa: BLE001
        bitexact_rule = f'FAIL: {e}'

    # --- independent recomputation vs the reference run ---
    teacher = teacher_jsonl()
    ref = {r['case']: r for r in teacher if r.get('run_id') == REF_RUN}
    run_ids = {first['run_id'], *(c['run_id'] for c in clean)}
    new = [r for r in teacher[before_lines:] if r.get('profile') == a.profile]
    mine = {r['case']: r for r in new if r.get('run_id') in run_ids}
    per_case, total, tokens, delta = [], 0, 0, 0.0
    for c in cases:
        r, b = mine[c['case']], ref[c['case']]
        comps = r.get('logit_comparisons', [])
        total += len(comps); tokens += len(b['nll']); delta += sum(r['nll']) - sum(b['nll'])
        per_case.append({'case': c['case'], 'run_id': r['run_id'], 'nll_identical': r['nll'] == b['nll'],
                         'comparisons': len(comps), 'max_abs': max((x['max_abs'] for x in comps), default=None),
                         'max_relative_l2': max((x['relative_l2'] for x in comps), default=None)})
    summary = {
        'profile': a.profile, 'patched_quality_sha256': sha(patched), 'native_harness_sha256': hashes,
        'reference_run_id': REF_RUN, 'first_run_id': first['run_id'], 'clean_run_ids': [c['run_id'] for c in clean],
        'clean_cases': list(quality.CLEAN_CACHE_CASES), 'context': first['configuration'].get('context'),
        'source_sha256': first['source_sha256'], 'timeline_s': dict(timeline),
        'new_teacher_rows_this_profile': [(r['case'], r['run_id'][:8]) for r in new],
        'comparisons_total': total,
        'comparisons_all_zero': all(x['max_abs'] == 0.0 and x['max_relative_l2'] == 0.0 for x in per_case),
        'nll_identical_cases': sum(x['nll_identical'] for x in per_case),
        'ppl_delta_recomputed': math.exp(delta / tokens) - 1,
        'compare_quality_unchanged': {k: cq[k] for k in ('relative_perplexity_growth', 'relative_growth_ci95',
                                                         'ppl_gate_pass', 'logit_comparisons', 'forced_targets_complete',
                                                         'new_functional_failures', 'quality_gate_pass')},
        'make_qualification_bitexact_rule': bitexact_rule,
        'rows_forced_ok': all(r['status'] == 200 and r['forced_text_matches'] and r['usage']['completion_tokens'] == r['expected_tokens']
                              for r in rows),
        'per_case': per_case}
    (out / f'validate-{cand}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_case', 'native_harness_sha256',
                                                                     'new_teacher_rows_this_profile')}, indent=1))
    ok = (total == 125 and summary['comparisons_all_zero'] and summary['nll_identical_cases'] == 20
          and summary['ppl_delta_recomputed'] == 0.0 and bitexact_rule == 'PASS' and cq['quality_gate_pass'])
    print('VERDICT', 'PASS' if ok else 'FAIL', a.profile)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
