"""Build the FLA on/off Nsight configs (new names only) from the native release config.

profile-final-fla-off.json = main-release.json + QWEN36_NSYS_FORWARDS + nsys block + plan
profile-final-fla-on.json  = the same + QWEN36_CUDA_GDN_FLA=wu
fla-release-wu.json        = main-release.json + QWEN36_CUDA_GDN_FLA=wu (speed A/B arm, only if needed)
Written to $N/config (O_EXCL, 0600) and copied to the workspace dir for evidence.
"""
import hashlib, importlib.util, json, os, pathlib, sys

N = pathlib.Path('/home/USER/qwen36-api/optimization')
OUT_WS = pathlib.Path('/path/to/qwen36-workspace/work/next-update/fla/configs')
spec = importlib.util.spec_from_file_location('profile_request', N / 'profile-request.py')
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

src_path = N / 'config' / 'main-release.json'
base = json.loads(src_path.read_text())
assert base['chunk'] == 2048 and base['env'].get('QWEN36_LAYER_MAJOR_PREFILL') == '1'
assert 'QWEN36_CUDA_GDN_FLA' not in base['env']
REQUESTS = [('warmup_short', 256, 16, 46001),
            ('warmup_chunk', 2048 + 1536, 4, 46002),
            ('warmup_16k', 16384, 4, 46017),
            ('fresh_16k', 16384, 4, 46016),
            ('fresh_120k', 120000, 4, 46120)]
tok = pr.tokenizer_load()
written = {}


def put(name, cfg):
    text = json.dumps(cfg, ensure_ascii=False, indent=1) + '\n'
    path = N / 'config' / f'{name}.json'
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f'{path} exists with different content')
        state = 'unchanged'
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        state = 'written'
    (OUT_WS / f'{name}.json').write_text(text)
    written[name] = {'state': state, 'sha256': hashlib.sha256(text.encode()).hexdigest()}


for arm, extra in (('off', {}), ('on', {'QWEN36_CUDA_GDN_FLA': 'wu'})):
    cfg = json.loads(json.dumps(base))
    cfg['env'].update(extra)
    plan = pr.build_plan(tok, f'fla-{arm}', cfg, REQUESTS)
    block = {'tool': 'nsys', 'graph_nvtx': 'live', 'cuda_event_trace': 'false',
             'output_prefix': f'model-final-fla-{arm}'}
    prof = pr.profile_config(f'fla-{arm}', src_path, cfg, plan, block)
    prof['profile_plan']['note'] = ('FLA root-cause capture (work/next-update/fla); release env '
                                    + ('+ QWEN36_CUDA_GDN_FLA=wu' if extra else 'unchanged'))
    put(f'profile-final-fla-{arm}', prof)
ab = json.loads(json.dumps(base))
ab['env']['QWEN36_CUDA_GDN_FLA'] = 'wu'
put('fla-release-wu', ab)
print(json.dumps({'source': str(src_path), 'source_sha256': hashlib.sha256(src_path.read_bytes()).hexdigest(),
                  'written': written}, indent=1))
