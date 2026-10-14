"""Build the short-route arm configs from the release config (read-only). Arms differ only in:
  r : release as installed (shim loaded but inert: QWEN36_LAYER_MAJOR_MIN_TOKENS unset)
  d : release + --moe-prefill-hit-d2d, MIN_TOKENS unset -> engine refuses layer-major: every prefill is
      the ordinary chunked path with the hit/miss split (decode-resident experts D2D, misses H2D)
  p : release + --moe-prefill-hit-d2d + QWEN36_LAYER_MAJOR_MIN_TOKENS=<T> (the patch, via the shim):
      uncached tokens < T -> chunked + hit-D2D, >= T -> layer-major (full-layer copies)
Usage: make_configs.py [T]   (default 2049)"""
import json
import pathlib
import sys

N = pathlib.Path('/home/USER/qwen36-api/optimization')
HERE = pathlib.Path('/path/to/qwen36-workspace/work/next-update/short-route')
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2049
base = json.loads((N / 'config/main-release.json').read_text())
assert base['flags'] == [] and base['env'].get('QWEN36_LAYER_MAJOR_PREFILL') == '1', base
for arm in ('r', 'd', 'p'):
    cfg = json.loads(json.dumps(base))
    cfg['qualified'] = False
    cfg['env'].update({
        'PYTHONPATH': f'{HERE}/shim:{N}/python',
        'QWEN36_SHORTROUTE_SHIM': '1',
        'QWEN36_SHORTROUTE_OUT': f'{HERE}/results/shim/{arm}',
    })
    if arm in ('d', 'p'):
        cfg['flags'] = ['--moe-prefill-hit-d2d']
    if arm == 'p':
        cfg['env']['QWEN36_LAYER_MAJOR_MIN_TOKENS'] = str(T)
    (HERE / 'configs' / f'shortroute-{arm}.json').write_text(json.dumps(cfg))
    print(arm, cfg['flags'], {k: v for k, v in cfg['env'].items() if 'SHORTROUTE' in k or 'MIN_TOKENS' in k or k == 'PYTHONPATH'})
