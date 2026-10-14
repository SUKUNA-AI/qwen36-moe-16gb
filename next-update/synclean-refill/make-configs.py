"""Write the candidate run configs = native main-release.json + the feature env delta (same compact format)."""
import hashlib, json, pathlib, sys
N = pathlib.Path('/home/USER/qwen36-api/optimization')
W = pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
D = pathlib.Path(sys.argv[1])
raw = (N / 'config/main-release.json').read_bytes()
assert raw == (W / 'main-release.json').read_bytes(), 'workspace main-release.json differs from native'
base = json.loads(raw)
assert json.dumps(base, separators=(',', ':')).encode() == raw.rstrip(b'\n'), 'main-release.json is not compact JSON'
for name, delta in (('main-release-synclean', {'QWEN36_SYNC_LEAN': '1'}),
                    ('main-release-refill-restore', {'QWEN36_LAYER_MAJOR_REFILL': '1',
                                                     'QWEN36_LAYER_MAJOR_REFILL_POLICY': 'restore'})):
    cfg = json.loads(raw)
    assert not set(delta) & set(cfg['env'])
    cfg['env'].update(delta)
    out = json.dumps(cfg, separators=(',', ':')).encode() + (b'\n' if raw.endswith(b'\n') else b'')
    (D / 'configs' / f'{name}.json').write_bytes(out)
    print(name, hashlib.sha256(out).hexdigest()[:12], sorted(delta))
print('main-release', hashlib.sha256(raw).hexdigest()[:12])
