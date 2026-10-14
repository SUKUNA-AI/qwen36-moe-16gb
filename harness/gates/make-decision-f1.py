"""Parent: build the REVIEWED decision for run f1 from reports/final-gates-f1-decision-draft.json.

Removes the TODO line, adds the reviewed speed/VRAM/soak/256K facts, and (when the driver-24 C/D results exist)
copies them into reports/ as evidence for long256. Writes work/agents/final-gates/decision-f1.json (O_EXCL unless --force).
Usage: make-decision-f1.py [--force]
"""
import hashlib, json, os, pathlib, shutil, sys

R = pathlib.Path('/home/USER/qwen36-api/optimization/reports')
G = pathlib.Path('/path/to/qwen36-workspace/work/l256-deviation/gpu-results')
OUT = pathlib.Path('/path/to/qwen36-workspace/work/agents/final-gates/decision-f1.json')
draft = json.loads((R / 'final-gates-f1-decision-draft.json').read_text())
WANT_D = {8: 1.0390625, 16: 0.9375, 24: 0.6875, 32: 1.232421875}


def case(path, name):
    d = json.loads(path.read_text())
    s = next(x for x in d['vs_teacher_bf128'] if x['case'] == name)
    lc = {c['position']: c['max_abs'] for c in (s['logit_comparisons'] or [])}
    return d, s, lc


conf = {}
for tag, fname in (('D', 'D-main-release-dedup.json'), ('C', 'C-long256-release-fresh.json')):
    p = G / fname
    if not p.is_file():
        conf[tag] = None
        continue
    d, s, lc = case(p, 'long-120000')
    if tag == 'D':
        ok = lc.get(0) == 0.0 and all(lc.get(k) == v for k, v in WANT_D.items()) and abs(s['mean_nll_delta'] - 0.01774165504857106) < 1e-12
    else:
        ok = bool(lc) and all(v == 0.0 for v in lc.values()) and s['first_nll_diff_position'] is None
    dst = R / f'final-f1-l256-confirm-{tag}.json'
    if not dst.exists():
        shutil.copyfile(p, dst)
    conf[tag] = {'ok': ok, 'evidence': str(dst), 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(), 'lc': lc,
                 'run_id': d['run_id']}

common = [
    'Reviewed speed (final-f1-m, 5 pairs, V6, main-release vs bf128 = original FreeToken Triton baseline): verdict PASS '
    '(noninferior on all gating cases, gain >= 5% Bonferroni). Speedup geomean bf128/main-release: TTFT 4K 2.95x, 16K 4.23x, '
    '64K 3.09x, 120K 2.52x, history 4K 2.88x, last-message edit 16K 9.08x, tool-result edit 16K 9.47x; TTFT at 64 tokens 0.94x '
    '(CI 0.84-1.04, non-gating); decode 1.19-1.32x; full request 1.13x (64) to 4.49x (edits); tool cycle task 1.08x (CI 0.97-1.20). '
    'Outputs differ in 15/40 vs bf128 (observation only; prefix-hit cases with different snapshot flags; quality gated by teacher 0-diff).',
    'Reproducibility property (all profiles, not a defect): the hybrid radix cache re-points a request\'s shared chat-header KV to '
    'pages written by earlier requests (same tokens/positions, rounding-level different values); greedy outputs on near-ties can '
    'depend on prior cache contents. Root cause analysis work/l256-deviation/results.json + numerics/.',
    'VRAM headroom depends on desktop load on the same GPU (4K@159Hz display, browser video): gates measured min free 1.20 GiB '
    '(main128 context), 1.41 GiB (long256 L4), matrix arm minimum about 1.16 GiB; the 2 h soak saw median 1.63 GiB free, '
    '268/721 samples < 1 GiB and ONE sample at 0.43 GiB (below the 0.5 GiB gate floor; no stall, no slowdown; soak does not gate VRAM).',
]
facts = {
    'main128': common + [
        'Reviewed soak soak-final-f1-a1: PASS, 7212 s, 509 iterations, 741/741 rows, 0 errors, 0 stalls; anon RSS +45..48 MB over 2 h '
        '(threshold 268 MB); torch allocated/reserved +0; GLIBC_TUNABLES PASS pre/post; fresh TTFT median 64/4K/16K/64K/120K = '
        '0.60/0.61/1.36/8.14/20.97 s; cancel recovery median 1.45 s.',
        'Teacher forcing vs teacher-bf128: 125/125 logits bit-identical, PPL growth 0.0, no new functional failures (quality-main-release.json).',
    ],
    'long256': common + [
        'Context gates L4 final-f1-l256-a1 PASS: long-facts-250000 PASS with TTFT 80.4 s (R1b profile 115.7 s, pre-optimization 133.7 s); '
        'min free 1.41 GiB (TARGET). No dedicated 2 h soak at 256K (soak evidence = main128, same build).',
    ],
    'baseline64': ['VRAM: F1 min free 0.79 GiB (FLOOR band: above the 0.5 GiB floor, below the 1.0 GiB target); unchanged fallback profile.']
           + common[2:],
}
c, dd = conf.get('C'), conf.get('D')
if c and c['ok']:
    facts['long256'].append(
        'Teacher: the 20-case run is bit-identical to teacher-bf128 except long-120000 (prefix-cache header dedup, see above). '
        f'long-120000 re-run on a fresh server (clean cache, run {c["run_id"][:12]}) is bit-identical: all logit comparisons 0.0 '
        f'(evidence {pathlib.Path(c["evidence"]).name}, sha256 {c["sha256"][:16]}).'
        + (f' Mechanism confirmed by run D (main128 with a small cache reproduces the 256K deviation to the last digit, '
           f'{pathlib.Path(dd["evidence"]).name}).' if dd and dd['ok'] else ' Mechanism run D: ' + ('NOT_RUN' if not dd else 'prediction NOT met')))
else:
    facts['long256'].append('Teacher: PASS_EQUIV_R1B256 only (clean-cache 120K re-run ' + ('NOT_RUN' if not c else 'did NOT show 0.0') + ').')

dec = json.loads(json.dumps(draft))
for name, entry in dec['profiles'].items():
    lim = [x for x in entry['known_limitations'] if not x.startswith('TODO(parent)')]
    lim = [x.replace('AUTO (review): ', 'Reviewed: ') for x in lim]
    entry['known_limitations'] = lim + facts.get(name, [])
    if name == 'long256':
        for tag in ('C', 'D'):
            info = conf.get(tag)
            if info and info['ok']:
                rel = os.path.relpath(info['evidence'], R)
                prefix = os.path.commonprefix([e for e in entry['evidence'] if '/' in e]) if any('/' in e for e in entry['evidence']) else ''
                ev = entry['evidence'][0]
                path_style = str(R / rel) if ev.startswith('/') else (str(pathlib.Path(ev).parent / rel) if '/' in ev else rel)
                if path_style not in entry['evidence']:
                    entry['evidence'].append(path_style)
if OUT.exists() and '--force' not in sys.argv:
    raise SystemExit(f'{OUT} exists (use --force)')
OUT.write_text(json.dumps(dec, indent=1, ensure_ascii=False) + '\n')
print(json.dumps({'out': str(OUT), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()[:16],
                  'confirm': {k: (v and {'ok': v['ok'], 'lc': v['lc']}) for k, v in conf.items()},
                  'profiles': {n: len(e['known_limitations']) for n, e in dec['profiles'].items()}}, ensure_ascii=False))
