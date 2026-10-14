"""Write FP8-KV experiment configs as copies of the released run configs (only KV env / ratio added)."""
import json, pathlib, sys, hashlib
O = pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
F = pathlib.Path(sys.argv[1]) / 'configs'
SCALES = '/home/USER/qwen36-api/optimization/config/kv-scales-bound.json'
FP8 = {'QWEN36_KV_DTYPE': 'fp8', 'QWEN36_KV_SCALES': SCALES}
TEACHER = {'QWEN36_TEACHER_FIXTURE': '/home/USER/qwen36-api/optimization/config/teacher-fixture.json',
           'QWEN36_LOGIT_REFERENCE': '/home/USER/qwen36-api/optimization/reports/logit-reference'}
def load(name): return json.loads((O / f'{name}.json').read_text())
def dump(name, cfg, compact=True):
    text = json.dumps(cfg, separators=(',', ':')) if compact else json.dumps(cfg)
    (F / f'{name}.json').write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
out = {}
main = load('main-release'); long = load('long256-release')
out['main-release'] = dump('main-release', main)
m = json.loads(json.dumps(main)); m['env'].update(FP8); out['main-release-fp8kv'] = dump('main-release-fp8kv', m)
t = json.loads(json.dumps(m)); t['env'].update(TEACHER); out['teacher-main-release-fp8kv'] = dump('teacher-main-release-fp8kv', t, compact=False)
h = json.loads(json.dumps(m)); h['memory_ratio'] = 0.665; out['main-release-fp8kv-hr'] = dump('main-release-fp8kv-hr', h)
l = json.loads(json.dumps(long)); l['env'].update(FP8); out['long256-release-fp8kv'] = dump('long256-release-fp8kv', l)
tl = json.loads(json.dumps(l)); tl['env'].update(TEACHER); out['teacher-long256-release-fp8kv'] = dump('teacher-long256-release-fp8kv', tl, compact=False)
assert (F / 'main-release.json').read_bytes() == (O / 'main-release.json').read_bytes(), 'main-release copy differs'
print(json.dumps(out, indent=1))
