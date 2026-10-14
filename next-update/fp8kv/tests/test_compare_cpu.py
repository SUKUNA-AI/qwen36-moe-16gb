"""CPU self-test of eval-fp8kv-compare.py on synthetic reports (no GPU, temp root)."""
import importlib.util, json, pathlib, tempfile, copy, sys
F = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('cmp', F / 'eval-fp8kv-compare.py'); cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)
freeze = json.loads((F / 'eval-fp8kv-freeze.json').read_text())
fsha = cmp.sha256_file(F / 'eval-fp8kv-freeze.json')
base_cfg = json.loads((F / 'configs/main-release.json').read_text()); fp8_cfg = json.loads((F / 'configs/main-release-fp8kv.json').read_text())
rows = [{'id': c['id'], 'sha256': c['sha256'], 'category': c['category'], 'pass': True} for c in freeze['cases']]
def ev(profile, cfg, rid, name, t, fails=()):
    r = copy.deepcopy(rows)
    for x in r:
        if x['id'] in fails: x['pass'] = False
    run = {'run_id': rid, 'configuration': cfg, 'source_sha256': 's', 'packages': {}, 'checkpoint_revision': 'c', 'native_libraries': {}, 'service_source_files': {}}
    return {'kind': 'eval-v1', 'status': 'VALID', 'profile': profile, 'name': name, 'started_utc': t, 'freeze_sha256': fsha,
            'suite_sha256': freeze['suite_sha256'], 'sources': {}, 'sampling': {}, 'decision_rule_sha256': freeze['decision_rule_sha256'],
            'bench_sha256': 'b', 'run': run, 'rows': r}
def setup(root, q_growth=0.0003, q_hi=0.008, fp8_fails=(), clip=0, func_fail=False):
    (root / 'reports').mkdir(parents=True)
    w = lambda n, d: (root / 'reports' / n).write_text(json.dumps(d))
    w('a-eval-v1.json', ev('main-release', base_cfg, 'r' * 32, 'a', '2026-10-15T08:00:00Z'))
    w('b-eval-v1.json', ev('main-release-fp8kv', fp8_cfg, 'f' * 32, 'b', '2026-10-15T08:10:00Z', fp8_fails))
    w('q.json', {'baseline': 'teacher-bf128', 'candidate': 'teacher-main-release-fp8kv', 'relative_perplexity_growth': q_growth,
                 'relative_growth_ci95': [-0.007, q_hi], 'forced_targets_complete': True,
                 'logit_comparisons': {'finite_and_complete': True, 'max_abs': 1.2}, 'rows': []})
    fr = [{'case': c, 'pass': c not in ('executable-code', 'long-facts-16384')} for c in ('russian', 'executable-code', 'tool-cycle', 'long-facts-16384', 'long-facts-120000')]
    w('fb.json', {'rows': fr}); fc = copy.deepcopy(fr)
    if func_fail: fc[2]['pass'] = False
    w('fc.json', {'rows': fc})
    eng = [{'run_id': rid, 'kv_dtype': 'torch.float8_e4m3fn', 'kv_clipped_values_total': clip, 'expert_slots': 3917, 'kv_bytes': 1343498240} for rid in ('t' * 32, 'u' * 32, 'f' * 32)]
    (root / 'reports' / 'engine.jsonl').write_text('\n'.join(json.dumps(e) for e in eng) + '\n')
def run(**kw):
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d); setup(root, **kw)
        out = root / 'decision.json'
        rc = cmp.main(['--quality', 'q', '--baseline', 'a', '--candidate', 'b', '--functional-baseline', 'fb', '--functional-candidate', 'fc',
                       '--out', str(out), '--fp8-run-ids', ','.join(x * 32 for x in 'tuf')], root=root)
        return rc, json.loads(out.read_text())['decision']
results = {
 'pass': run(), 'ppl_point_fail': run(q_growth=0.003), 'ppl_ci_fail': run(q_hi=0.011),
 'tool_loss_fail': run(fp8_fails=('tool-01',) if any(c['id']=='tool-01' for c in freeze['cases']) else (next(c['id'] for c in freeze['cases'] if c['category']=='tool'),)),
 'one_long_loss_pass': run(fp8_fails=(next(c['id'] for c in freeze['cases'] if c['category']=='long'),)),
 'two_long_loss_fail': run(fp8_fails=tuple(c['id'] for c in freeze['cases'] if c['category']=='long')[:2]),
 'clip_fail': run(clip=3), 'functional_fail': run(func_fail=True),
}
expect = {'pass': 'QUALITY_PASS', 'ppl_point_fail': 'QUALITY_FAIL', 'ppl_ci_fail': 'QUALITY_FAIL', 'tool_loss_fail': 'QUALITY_FAIL',
          'one_long_loss_pass': 'QUALITY_PASS', 'two_long_loss_fail': 'QUALITY_FAIL', 'clip_fail': 'QUALITY_FAIL', 'functional_fail': 'QUALITY_FAIL'}
ok = all(results[k][1] == v for k, v in expect.items())
print(json.dumps({'passed': sum(results[k][1] == v for k, v in expect.items()), 'total': len(expect), 'ok': ok, 'results': results}))
sys.exit(0 if ok else 1)
