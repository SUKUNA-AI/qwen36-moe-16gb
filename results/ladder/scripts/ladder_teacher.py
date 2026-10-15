"""CPU-only: teacher-gate outcome + run-bound T1 telemetry per ladder step -> public JSON (numbers only).

python3 -I ladder_teacher.py --reports <native>/reports --ts TS --ref REF --fb FB --steps steps.json --logs DIR --out teacher.json
Reads quality-<name>.json, <name>-teacher.json, <name>-functional.json, engine.jsonl* (rows of the
step's own teacher/functional run ids), the qcheck output and the unit journal counts written by step3.
"""
import argparse, json, pathlib, re

SHORT_MAX = 2048          # T1 short route: uncached prompt <= 2048 tokens goes to the slot route
CKPT_STEP = 8


def load(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None


def engine_rows(reports, run_ids):
    rows = []
    for suffix in ('.5', '.4', '.3', '.2', '.1', ''):
        try:
            lines = (reports / ('engine.jsonl' + suffix)).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('run_id') in run_ids:
                rows.append(row)
    seen, out = set(), []
    for row in rows:
        key = (row.get('run_id'), row.get('uid'))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def numeric(d):
    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def telemetry(rows):
    short = [r for r in rows if 0 < (r.get('prefill_forward_tokens') or 0) <= SHORT_MAX]
    long = [r for r in rows if (r.get('prefill_forward_tokens') or 0) > SHORT_MAX]
    d2d = lambda r: (r.get('expert_counter_deltas') or {}).get('prefill_d2d_bytes') or 0
    lm = lambda r: numeric(r.get('layer_major_prefill'))
    slots = sorted({r.get('expert_slots') for r in rows if isinstance(r.get('expert_slots'), int)})
    dec_tok = sum(r.get('decode_forwards') or 0 for r in rows)
    dec_miss = sum((r.get('expert_counter_deltas') or {}).get('decode_missing') or 0 for r in rows)
    return {
        'engine_rows': len(rows),
        'short_rows': len(short), 'short_rows_with_d2d': sum(1 for r in short if d2d(r) > 0),
        'short_d2d_bytes': sum(d2d(r) for r in short),
        'long_rows': len(long),
        'long_rows_with_layer_major_stats': sum(1 for r in long if lm(r)),
        'layer_major_prefill_last': lm(long[-1]) if long else {},
        'expert_slots': slots,
        'host_embedding_bytes': sorted({r.get('host_embedding_bytes') for r in rows if 'host_embedding_bytes' in r}),
        'decode_missing_per_forward': dec_miss / dec_tok if dec_tok else None,
        'peak_torch_reserved_bytes': max((r.get('torch_reserved_bytes') or 0 for r in rows), default=None),
    }


def journal(logs, label):
    try:
        text = (logs / f'q-{label}.journal').read_text(errors='replace')
    except OSError:
        return None
    return {'hit_d2d_unavailable': len(re.findall(r'hit-D2D requested but unavailable', text)),
            'c1_c2_transport_unavailable': len(re.findall(r'C1/C2 transport unavailable', text)),
            'tracebacks': text.count('Traceback')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', type=pathlib.Path, required=True)
    p.add_argument('--logs', type=pathlib.Path, required=True)
    p.add_argument('--ts', default='20261015t1555'); p.add_argument('--ref', required=True); p.add_argument('--fb', required=True)
    p.add_argument('--steps', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    steps = json.loads(a.steps.read_text())['steps']
    out = {'schema': 'ladder-teacher-1', 'reference_report': a.ref, 'functional_baseline': a.fb,
           'rule': '125/125 checkpoints, max_abs 0, relative_l2 0, all_byte_equal, PPL gate, no new functional failure '
                   'vs the bf128 base; T1 steps also need d2d on a short case, layer-major on long cases, no hit-D2D fallback',
           'steps': []}
    for step in steps:
        label, cand = step['id'], step['config']
        name = f"s5-q-{label}-{step.get('teacher_ts', a.ts)}"
        q = load(a.reports / f'quality-{name}.json')
        func = load(a.reports / f'{name}-functional.json')
        row = {'id': label, 'config': cand, 'quality_report': f'quality-{name}.json'}
        if not q:
            row['status'] = 'NOT_RUN_OR_FAILED'
            out['steps'].append(row); continue
        lc = q.get('logit_comparisons') or {}
        row['teacher'] = {k: q.get(k) for k in ('evidence_valid', 'bit_exact_pass', 'quality_gate_pass', 'ppl_gate_pass',
                                                 'relative_perplexity_growth', 'relative_growth_ci95',
                                                 'new_functional_failures', 'forced_targets_complete', 'tokens')}
        row['teacher'].update({k: lc.get(k) for k in ('expected', 'observed', 'all_byte_equal', 'max_abs', 'max_relative_l2')})
        row['functional'] = [{'case': r.get('case'), 'pass': r.get('pass')} for r in (func or {}).get('rows', [])]
        runs = set((q.get('case_runs') or {}).get('candidate', {}).values())
        frun = (q.get('functional_runs') or {}).get('candidate')
        rows = engine_rows(a.reports, runs | {frun})
        row['telemetry'] = telemetry(rows)
        row['journal'] = journal(a.logs, label)
        t = row['telemetry']
        if step.get('t1'):
            row['t1_checks'] = {'short_case_with_d2d': t['short_rows_with_d2d'] > 0,
                                'long_cases_layer_major': t['long_rows'] > 0 and t['long_rows_with_layer_major_stats'] == t['long_rows'],
                                'no_hit_d2d_fallback': (row['journal'] or {}).get('hit_d2d_unavailable', 1) == 0}
        ok = (q.get('bit_exact_pass') is True and lc.get('observed') == lc.get('expected') == 125 and lc.get('max_abs') == 0
              and lc.get('all_byte_equal') is True and not q.get('new_functional_failures')
              and all((row.get('t1_checks') or {}).values()))
        row['status'] = 'PASS' if ok else 'FAIL'
        if step.get('teacher_ts_previous_tree'):
            prev = load(a.reports / f"quality-s5-q-{label}-{step['teacher_ts_previous_tree']}.json") or {}
            plc = prev.get('logit_comparisons') or {}
            row['previous_tree'] = {'quality_report': f"quality-s5-q-{label}-{step['teacher_ts_previous_tree']}.json",
                                    'bit_exact_pass': prev.get('bit_exact_pass'), 'observed': plc.get('observed'),
                                    'max_abs': plc.get('max_abs'), 'new_functional_failures': prev.get('new_functional_failures')}
        out['steps'].append(row)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps([(s['id'], s['status']) for s in out['steps']]))


if __name__ == '__main__':
    main()
