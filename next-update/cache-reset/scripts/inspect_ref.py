"""CPU-only, read-only: inventory of the teacher-bf128 reference and the release teacher runs in the native reports."""
import json, os, pathlib, datetime, collections, sys
N = pathlib.Path('/home/USER/qwen36-api/optimization')
R = N / 'reports'
rows = [json.loads(l) for l in (R / 'teacher.jsonl').read_text().splitlines() if l.strip()]
order = collections.OrderedDict()
for i, r in enumerate(rows):
    k = (r.get('profile'), r.get('run_id'))
    e = order.setdefault(k, {'first_line': i, 'cases': []})
    e['cases'].append(r['case']); e['last_line'] = i
print('teacher.jsonl rows', len(rows))
for (p, rid), e in order.items():
    if p in ('teacher-bf128', 'teacher-main-release', 'teacher-long256-release'):
        print(p, rid, 'lines', e['first_line'], '-', e['last_line'], 'n', len(e['cases']), 'last', e['cases'][-1])
# last teacher-bf128 rows per case (compare-quality grouping)
last = {}
for r in rows:
    if r.get('profile') == 'teacher-bf128':
        last[r['case']] = r.get('run_id')
print('compare-quality bf128 rows by run_id:', collections.Counter(last.values()))
def mt(p):
    return datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat()
for name in ('teacher-bf128-logits.json', 'teacher-main-release.json', 'teacher-long256-release.json',
             'quality-main-release.json', 'quality-long256-release.json', 'functional-bf128.json',
             'functional-main-release.json', 'functional-long256-release.json'):
    p = R / name
    if not p.exists():
        print(name, 'MISSING'); continue
    d = json.loads(p.read_text())
    run = d.get('run', {})
    print(name, 'mtime', mt(p), 'run_id', run.get('run_id'), 'started', run.get('started_utc'),
          'profile', run.get('profile'), 'ctx', (run.get('configuration') or {}).get('context'),
          'ratio', (run.get('configuration') or {}).get('memory_ratio'))
    if name.startswith('quality-'):
        print('   ', {k: d.get(k) for k in ('baseline', 'candidate', 'relative_perplexity_growth', 'logit_comparisons', 'quality_gate_pass')})
ref = R / 'logit-reference'
files = sorted(ref.glob('*.npy'))
mts = sorted(f.stat().st_mtime for f in files)
print('logit-reference files', len(files), 'mtime range',
      datetime.datetime.fromtimestamp(mts[0], datetime.timezone.utc).isoformat(),
      datetime.datetime.fromtimestamp(mts[-1], datetime.timezone.utc).isoformat())
seeds = collections.Counter(f.name.split('-')[0] for f in files)
print('seeds', len(seeds), 'files for long-120000 seed 900019:', seeds.get('900019'))
for name in ('teacher-main-release.json', 'teacher-long256-release.json'):
    d = json.loads((R / name).read_text())
    print(name, [(r['case'], r['usage']['prompt_tokens'], r['expected_tokens']) for r in d['rows']][-4:])
for c in ('teacher-main-release', 'teacher-long256-release', 'teacher-bf128', 'main-release', 'long256-release', 'bf128'):
    p = N / 'config' / (c + '.json')
    print('config', c, p.exists() and json.loads(p.read_text()).get('context'), p.exists() and json.loads(p.read_text()).get('memory_ratio'))
