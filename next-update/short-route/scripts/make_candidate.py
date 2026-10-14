"""Candidate run config for the teacher gate / matrix AFTER the patch is synced natively (no shim):
main-release + --moe-prefill-hit-d2d + QWEN36_LAYER_MAJOR_MIN_TOKENS=<T>. Usage: make_candidate.py [T]"""
import json, pathlib, sys
N = pathlib.Path('/home/USER/qwen36-api/optimization')
HERE = pathlib.Path('/path/to/qwen36-workspace/work/next-update/short-route')
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2049
cfg = json.loads((N / 'config/main-release.json').read_text())
cfg['qualified'] = False
cfg['flags'] = ['--moe-prefill-hit-d2d']
cfg['env']['QWEN36_LAYER_MAJOR_MIN_TOKENS'] = str(T)
(HERE / 'configs' / 'main-release-shortroute.json').write_text(json.dumps(cfg))
print(json.dumps({k: v for k, v in cfg.items() if k != 'env'}), cfg['env']['QWEN36_LAYER_MAJOR_MIN_TOKENS'])
