"""Self-test of evaluate.py on synthetic screen reports (no GPU): a +3% feature -> KEEP, a +1% one -> REJECT,
one round in the wrong direction -> REJECT, content mismatch -> REJECT, missing session -> INCONCLUSIVE."""
import json, os, pathlib, random, subprocess, sys, tempfile
D = pathlib.Path(__file__).resolve().parent
PY = sys.executable
pre = json.loads((D / 'prereg.json').read_text())

def write(tmp, run, feat, gains, *, content_ok=True, drop=None, teacher_ok=True):
    spec = pre['features'][feat]; cand = spec['candidate_profile']
    rnd = random.Random(1)
    for r in (1, 2, 3):
        for out in spec['outputs']:
            name = f'srf-{run}-{feat}-r{r}-o{out}'
            if drop == (r, out):
                continue
            for prof in ('main-release', cand):
                rows = []
                for L in spec['lengths']:
                    g = gains(r, L, out)
                    ttft, dec = 1.0 + L / 6000, 100.0
                    if prof == cand:
                        if spec['primary_metric'] == 'ttft_content_s': ttft = ttft / (1 + g)
                        else: dec = dec * (1 + g)
                    rows.append({'case': 'fresh', 'length': L, 'ttft_content_s': ttft, 'elapsed_s': ttft + out / dec,
                                 'decode_client_tps': dec, 'min_gpu_free_bytes': 1200 << 20,
                                 'content_sha256': 'x' if (content_ok or prof == 'main-release') else 'y',
                                 'engine': {'expert_counter_deltas': {'decode_missing': 3800}}})
                (tmp / f'{name}-{prof}-chunk-screen.json').write_text(json.dumps(
                    {'status': 'measured', 'vram_gate': 'PASS', 'min_gpu_free_bytes': 1200 << 20, 'rows': rows}))
    (tmp / f'quality-{cand}.json').write_text(json.dumps({'teacher': {'max_abs': 0.0 if teacher_ok else 0.5,
        'forced_targets_complete': True, 'new_functional_failures': [], 'ppl_gate_pass': True}}))

def run(feat, run_id, **kw):
    tmp = pathlib.Path(tempfile.mkdtemp())
    write(tmp, run_id, feat, **kw)
    env = dict(os.environ, SRF_REPORTS=str(tmp))
    out = subprocess.run([PY, '-I', str(D / 'evaluate.py'), '--run', run_id, '--feature', feat], env=env,
                         capture_output=True, text=True).stdout
    res = json.loads((D / 'results' / f'{feat}-{run_id}.json').read_text())
    (D / 'results' / f'{feat}-{run_id}.json').unlink()
    return res['decision'], out.splitlines()[0]

cases = [
    ('refill', 'st1', dict(gains=lambda r, L, o: 0.03), 'KEEP'),
    ('refill', 'st2', dict(gains=lambda r, L, o: 0.01), 'REJECT'),
    ('refill', 'st3', dict(gains=lambda r, L, o: -0.01 if r == 2 else 0.04), 'REJECT'),
    ('refill', 'st4', dict(gains=lambda r, L, o: 0.03, content_ok=False), 'REJECT'),
    ('refill', 'st5', dict(gains=lambda r, L, o: 0.03, drop=(2, 512)), 'INCONCLUSIVE->REJECT'),
    ('synclean', 'st6', dict(gains=lambda r, L, o: 0.03), 'KEEP'),
    ('synclean', 'st7', dict(gains=lambda r, L, o: 0.03 if L == 120000 else -0.05), 'REJECT'),
    ('synclean', 'st8', dict(gains=lambda r, L, o: 0.03, teacher_ok=False), 'REJECT'),
    ('synclean', 'st9', dict(gains=lambda r, L, o: 0.03 if (L == 120000 and o == 512) else 0.0), 'REJECT'),
]
bad = 0
for feat, rid, kw, want in cases:
    got, line = run(feat, rid, **kw)
    ok = got == want; bad += not ok
    print('ok ' if ok else 'BAD', feat, rid, want, got, '|', line[:150])
print('selftest', 'PASS' if not bad else f'FAIL {bad}')
sys.exit(bad)
