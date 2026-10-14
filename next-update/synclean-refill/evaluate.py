"""Read-only evaluation of the synclean-refill A/B against the pre-registered rule (prereg.json).

  evaluate.py --run a1 --feature refill            decision -> results/<feature>-<run>.json (+ compact lines)
  evaluate.py --run a1 --feature refill --peek N   one session's fresh rows (progress)
Inputs: $N/reports/srf-<run>-<feature>-r<r>-o<out>-<profile>-chunk-screen.json and quality-<candidate>.json.
"""
import argparse
import json
import os
import pathlib
import statistics as st
import sys

N = pathlib.Path('/home/USER/qwen36-api/optimization')
R = pathlib.Path(os.environ.get('SRF_REPORTS', str(N / 'reports')))  # override for the self-test only
D = pathlib.Path(__file__).resolve().parent
BASE = 'main-release'
FLOOR = 1 << 29  # bench.VRAM_FLOOR_BYTES = GiB // 2
METRICS = {'ttft_content_s': 'lower', 'elapsed_s': 'lower', 'decode_client_tps': 'higher'}


def load(name, profile):
    p = R / f'{name}-{profile}-chunk-screen.json'
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def fresh(doc):
    return {x['length']: x for x in (doc or {}).get('rows', []) if x.get('case') == 'fresh'}


def gain(metric, base, cand):
    if not base or not cand:
        return None
    return base / cand - 1 if METRICS[metric] == 'lower' else cand / base - 1


def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                return v
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None:
                return r
    return None


def teacher(cand):
    try:
        d = json.loads((R / f'quality-{cand}.json').read_text())
    except (OSError, ValueError):
        return {'status': 'MISSING'}
    out = {k: find(d, k) for k in ('max_abs', 'forced_targets_complete', 'new_functional_failures', 'ppl_gate_pass')}
    ok = out['max_abs'] == 0 and not out['new_functional_failures'] and out['forced_targets_complete'] is not False
    out['status'] = 'PASS' if ok else 'FAIL'
    return out


def counter(row, key):
    d = (row.get('engine') or {}).get('expert_counter_deltas')
    return d.get(key) if isinstance(d, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--feature', required=True, choices=('refill', 'synclean'))
    ap.add_argument('--peek')
    a = ap.parse_args()
    pre = json.loads((D / 'prereg.json').read_text())
    spec = pre['features'][a.feature]
    cand = spec['candidate_profile']
    if a.peek:
        for prof in (BASE, cand):
            doc = load(a.peek, prof)
            if doc is None:
                print(f'peek {a.peek} {prof}: missing')
                continue
            print(f"peek {a.peek} {prof}: {doc.get('status')} vram={doc.get('vram_gate')} " + ' '.join(
                f"{L}:ttft={x.get('ttft_content_s') or 0:.2f}s,dec={x.get('decode_client_tps') or 0:.1f},"
                f"miss={counter(x, 'decode_missing')},free={(x.get('min_gpu_free_bytes') or 0) >> 20}MiB"
                for L, x in sorted(fresh(doc).items())))
        return 0
    lengths, outputs = spec['lengths'], spec['outputs']
    cells, problems, content_pairs, content_same = {}, [], 0, 0
    vram = {BASE: [], cand: []}
    misses = {}
    for r in (1, 2, 3):
        for out in outputs:
            name = f'srf-{a.run}-{a.feature}-r{r}-o{out}'
            docs = {p: load(name, p) for p in (BASE, cand)}
            for p, doc in docs.items():
                if doc is None or doc.get('status') != 'measured':
                    problems.append(f'{name} {p}: {"missing" if doc is None else doc.get("status")}')
                    continue
                vram[p].append((doc.get('min_gpu_free_bytes'), doc.get('vram_gate')))
            fb, fc = fresh(docs[BASE]), fresh(docs[cand])
            for L in lengths:
                b, c = fb.get(L), fc.get(L)
                if not b or not c:
                    continue
                cell = cells.setdefault(f'{L}/{out}', {m: [] for m in METRICS})
                for m in METRICS:
                    g = gain(m, b.get(m), c.get(m))
                    cell[m].append(None if g is None else round(g, 5))
                cell.setdefault('base_abs', []).append({m: b.get(m) for m in METRICS})
                cell.setdefault('cand_abs', []).append({m: c.get(m) for m in METRICS})
                misses.setdefault(f'{L}/{out}', []).append((counter(b, 'decode_missing'), counter(c, 'decode_missing')))
                content_pairs += 1
                content_same += b.get('content_sha256') is not None and b.get('content_sha256') == c.get('content_sha256')
    rule = pre['decision_rule']
    summary = {}
    for key, cell in cells.items():
        summary[key] = {}
        for m in METRICS:
            gs = [g for g in cell[m] if g is not None]
            if len(gs) < 3:
                summary[key][m] = {'rounds': cell[m], 'verdict': 'INCOMPLETE'}
                continue
            med = st.median(gs)
            verdict = ('PASS' if med >= 0.02 and min(gs) > 0 else 'HARM' if med <= -0.02 and max(gs) < 0 else 'NEUTRAL')
            summary[key][m] = {'rounds': [round(g, 4) for g in gs], 'median': round(med, 4), 'verdict': verdict}
    t = teacher(cand)
    primary = spec['primary_metric']
    prim = {f'{L}/{o}': summary.get(f'{L}/{o}', {}).get(primary, {}).get('verdict', 'INCOMPLETE')
            for L, o in spec['primary_cells']}
    harms = [(k, m) for k, v in summary.items() for m, s in v.items() if s.get('verdict') == 'HARM']
    vram_ok = bool(vram[cand]) and all(g == 'PASS' and (f or 0) >= FLOOR for f, g in vram[cand])
    complete = not problems and all(s.get('verdict') != 'INCOMPLETE' for v in summary.values() for s in v.values()) \
        and len(summary) == len(lengths) * len(outputs)
    checks = {
        'teacher_pass': t.get('status') == 'PASS',
        'content_identical': content_pairs > 0 and content_same == content_pairs,
        'vram_gate_pass': vram_ok,
        'primary_cells_pass': all(v == 'PASS' for v in prim.values()),
        'no_harm_cell': not harms,
        'complete': complete,
    }
    decision = 'KEEP' if all(checks.values()) else ('INCONCLUSIVE->REJECT' if not complete else 'REJECT')
    result = {'kind': 'synclean-refill-decision', 'run': a.run, 'feature': a.feature, 'base': BASE, 'candidate': cand,
              'prereg_rule': rule, 'teacher': t, 'checks': checks, 'decision': decision,
              'primary_metric': primary, 'primary_cells': prim, 'harm_cells': harms, 'cells': summary,
              'decode_missing_base_cand': misses,
              'vram_min_free_mib': {p: min(((f or 0) >> 20 for f, _ in v), default=None) for p, v in vram.items()},
              'vram_gates': {p: [g for _, g in v] for p, v in vram.items()},
              'content_identical_pairs': f'{content_same}/{content_pairs}', 'problems': problems}
    out = D / 'results' / f'{a.feature}-{a.run}.json'
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=1) + '\n')
    os.replace(tmp, out)
    print(f'DECISION {a.feature} {decision} checks={json.dumps(checks)}')
    print(f"teacher {t.get('status')} max_abs={t.get('max_abs')}; content {content_same}/{content_pairs}; "
          f"VRAM min free MiB {result['vram_min_free_mib']}")
    for key in sorted(summary, key=lambda k: tuple(int(x) for x in k.split('/'))):
        print(key, ' '.join(f"{m.split('_')[0]}:{s.get('median')}{s.get('rounds')}{s.get('verdict')[0]}"
                            for m, s in summary[key].items()), 'miss(b,c)=', misses.get(key))
    if problems:
        print('PROBLEMS', problems)
    return 0


if __name__ == '__main__':
    sys.exit(main())
