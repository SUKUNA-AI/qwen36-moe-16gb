"""CPU-only: paired rotating matrix -> per-step medians and paired deltas (numbers only, chart-friendly).

python3 -I ladder_ab.py --reports <native>/reports --name AB --steps steps.json --out ab.json
Per arm it reads AB-<repeat>-<profile>.json (bench rows with the engine join) and AB-matrix.json (VRAM per arm).
Deltas are medians over pairs of the per-pair ratio (same repeat = same session slot), in percent;
for TTFT/elapsed negative is faster, for decode tok/s positive is faster.
"""
import argparse, json, pathlib, statistics

METRICS = ('ttft_s', 'decode_tps', 'elapsed_s', 'misses_per_token', 'prefill_d2d_gb', 'prefill_h2d_gb')
DELTA_METRICS = ('ttft_s', 'decode_tps', 'elapsed_s')


def med(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def case_key(row):
    return f"{row.get('case')}_{row.get('length')}"


def case_order(key):
    name, _, length = key.rpartition('_')
    return (name != 'fresh', int(length) if length.isdigit() else 0, name)


def arm_cases(doc):
    out = {}
    for row in doc.get('rows', []):
        if row.get('case') in (None, 'warmup', 'warmup_sustained') or row.get('status') != 200:
            continue
        e = row.get('engine') or {}
        d = e.get('expert_counter_deltas') or {}
        fw = e.get('decode_forwards') or 0
        miss = d.get('decode_missing')
        out[case_key(row)] = {
            'ttft_s': row.get('ttft_content_s'), 'decode_tps': row.get('decode_client_tps'), 'elapsed_s': row.get('elapsed_s'),
            'misses_per_token': miss / fw if fw and isinstance(miss, int) else None,
            'prefill_d2d_gb': (d.get('prefill_d2d_bytes') or 0) / 1e9 if d else None,
            'prefill_h2d_gb': (d.get('prefill_h2d_bytes') or 0) / 1e9 if d else None,
            'cached_prompt_tokens': row.get('cached_prompt_tokens'),
            'expert_slots': e.get('expert_slots'),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', type=pathlib.Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--steps', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    steps = json.loads(a.steps.read_text())['steps']
    matrix = json.loads((a.reports / f'{a.name}-matrix.json').read_text())
    manifest = json.loads((a.reports / f'{a.name}-manifest.json').read_text())
    arms = {}          # (profile, repeat) -> arm summary
    for row in matrix:
        if row.get('status') != 'complete':
            continue
        doc = json.loads((a.reports / row['report']).read_text())
        tool = None
        if row.get('tool_task_report'):
            try:
                t = json.loads((a.reports / row['tool_task_report']).read_text()).get('row') or {}
                tool = {'elapsed_s': t.get('elapsed_s'), 'correct': t.get('correct')}
            except (OSError, ValueError):
                pass
        v = row.get('vram') or {}
        arms[(row['profile'], row['repeat'])] = {
            'cases': arm_cases(doc), 'tool': tool, 'arm_verdict': row.get('arm_verdict'),
            'vram': {k: v.get(k) for k in ('process_peak_bytes', 'budget_bytes', 'peak_gpu_used_bytes',
                                           'min_gpu_free_bytes', 'gate', 'foreign_gpu_used_bytes_pre_start')}}
    repeats = sorted({r for _, r in arms})
    cases = sorted({c for arm in arms.values() for c in arm['cases']}, key=case_order)
    out = {'schema': 'ladder-ab-1', 'matrix': a.name, 'pairs': len(repeats), 'lengths': manifest.get('lengths'),
           'output_tokens': manifest.get('output'), 'warmup_s': 60, 'profiler': 'none',
           'order': [[r['repeat'], r['profile']] for r in manifest.get('schedule', [])],
           'cases': cases, 'steps': []}
    for s in steps:
        prof = s['config']
        mine = {r: arms.get((prof, r)) for r in repeats}
        st = {'id': s['id'], 'config': prof, 'lever_short': s['lever_short'],
              'arms_complete': sum(1 for v in mine.values() if v), 'cases': {}}
        for c in cases:
            st['cases'][c] = {}
            for m in METRICS:
                vals = [((mine[r] or {}).get('cases', {}).get(c) or {}).get(m) for r in repeats]
                st['cases'][c][m] = {'pairs': vals, 'median': med(vals)}
        tt = [((mine[r] or {}).get('tool') or {}).get('elapsed_s') for r in repeats]
        st['tool_task_elapsed_s'] = {'pairs': tt, 'median': med(tt)}
        vr = [(mine[r] or {}).get('vram') or {} for r in repeats]
        peaks = [x.get('process_peak_bytes') for x in vr]
        free = [x.get('min_gpu_free_bytes') for x in vr]
        st['vram'] = {'process_peak_bytes_max': max([x for x in peaks if x], default=None),
                      'process_peak_bytes_median': med(peaks),
                      'budget_bytes': next((x.get('budget_bytes') for x in vr if x.get('budget_bytes')), None),
                      'min_gpu_free_bytes_min': min([x for x in free if x is not None], default=None),
                      'gates': [x.get('gate') for x in vr]}
        st['expert_slots'] = sorted({v['expert_slots'] for r in repeats if mine[r]
                                     for v in mine[r]['cases'].values() if v.get('expert_slots')})
        out['steps'].append(st)

    def ratios(base, cand):
        res = {}
        for c in cases:
            res[c] = {}
            for m in DELTA_METRICS:
                pr = []
                for r in repeats:
                    b = (((arms.get((base, r)) or {}).get('cases', {})).get(c) or {}).get(m)
                    x = (((arms.get((cand, r)) or {}).get('cases', {})).get(c) or {}).get(m)
                    ok = isinstance(b, (int, float)) and isinstance(x, (int, float)) and b
                    pr.append(100 * (x / b - 1) if ok else None)
                res[c][m + '_pct'] = {'pairs': pr, 'median': med(pr)}
        return res

    ids = {s['id']: s['config'] for s in steps}
    out['deltas_vs_prev'] = {s['id']: ratios(ids[s['prev']], s['config']) for s in steps if s['prev']}
    out['deltas_vs_l0'] = {s['id']: ratios(ids['l0'], s['config']) for s in steps if s['id'] != 'l0'}
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'pairs': len(repeats), 'arms': len(arms), 'cases': cases}))


if __name__ == '__main__':
    main()
