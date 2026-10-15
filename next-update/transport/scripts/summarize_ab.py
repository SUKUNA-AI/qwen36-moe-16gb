"""Summarize S4 A/B rounds (driver.py output): per-arm medians and paired ratios against the base arm
started in the same lock session (identical prompts, same round).

Usage: summarize_ab.py RESULTS_DIR BASE_ARM OUT_JSON
"""
import json
import pathlib
import statistics
import sys
from collections import defaultdict

ROW_BYTES = 1781760  # one Qwen3.6 NVFP4 expert over all six banks
LAYERS = 40
FIELDS = ('ttft_s', 'prefill_gpu_ms', 'itl_early_ms', 'itl_late_ms')


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load(d):
    runs = []
    for meta_path in sorted(d.glob('*.meta.json')):
        meta = json.loads(meta_path.read_text())
        rows_path = meta_path.with_name(meta_path.name.replace('.meta.json', '.jsonl'))
        rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
        if meta.get('error') or not rows:
            print('skip', meta_path.name, meta.get('error'))
            continue
        runs.append({'arm': meta['arm'], 'round': meta['round'], 'start': meta['start'], 'rows': rows})
    return runs


def main():
    d, base, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
    runs = load(d)
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r['arm']].extend(r['rows'])
    arms = {}
    for arm, rows in by_arm.items():
        cases = defaultdict(list)
        for row in rows:
            cases[row['case']].append(row)
        arms[arm] = {}
        for case, rs in sorted(cases.items()):
            h2d = med([r.get('prefill_h2d_bytes') for r in rs])
            d2d = med([r.get('prefill_d2d_bytes') for r in rs])
            arms[arm][case] = {
                'n': len(rs), **{f: med([r.get(f) for r in rs]) for f in FIELDS},
                'uncached_tokens': med([(r['prompt_tokens'] or 0) - (r['cached'] or 0) for r in rs]),
                'h2d_gb': h2d / 1e9 if h2d is not None else None, 'd2d_gb': d2d / 1e9 if d2d is not None else None,
                'moved_experts_per_layer': ((h2d or 0) + (d2d or 0)) / ROW_BYTES / LAYERS if h2d is not None else None,
                'decode_missing_per_token': med([r['decode_missing'] / max(1, r['decode_forwards'] or 1)
                                                 for r in rs if r.get('decode_missing') is not None]),
                'content_sha256': sorted({r['content_sha256'] for r in rs}),
            }
    pairs = defaultdict(lambda: defaultdict(list))
    bases = [r for r in runs if r['arm'] == base]
    for r in runs:
        if r['arm'] == base:
            continue
        same = [b for b in bases if b['round'] == r['round']]
        if not same:
            continue
        b = min(same, key=lambda x: abs(x['start'] - r['start']))
        ref = {(x['case'], x['rep']): x for x in b['rows']}
        per_case = defaultdict(list)
        for row in r['rows']:
            x = ref.get((row['case'], row['rep']))
            if x and x.get('ttft_s') and row.get('ttft_s'):
                per_case[row['case']].append(row['ttft_s'] / x['ttft_s'])
                if row['content_sha256'] != x['content_sha256']:
                    pairs[r['arm']]['_output_differs'].append(row['case'])
        for case, ratios in per_case.items():
            pairs[r['arm']][case].append(med(ratios))
    paired = {arm: {case: {'pairs': len(v), 'ttft_ratio_median': med(v), 'ttft_ratio_each': v}
                    for case, v in cases.items() if not case.startswith('_')}
              for arm, cases in pairs.items()}
    for arm, cases in pairs.items():
        paired[arm]['output_differs_cases'] = sorted(set(cases.get('_output_differs', [])))
    result = {'schema': 1, 'base': base, 'arms': arms, 'paired_vs_base': paired,
              'note': 'TTFT ratio = candidate / base on identical prompts in the same round; < 1 is faster'}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    cases = sorted({c for a in arms.values() for c in a})
    print('arm'.ljust(7), ' '.join(c[:10].rjust(10) for c in cases))
    for arm in sorted(arms):
        print(arm.ljust(7), ' '.join(f"{arms[arm].get(c, {}).get('ttft_s') or 0:10.3f}" for c in cases))
    for arm in sorted(paired):
        print(('ratio ' + arm).ljust(12), ' '.join(
            f"{(paired[arm].get(c) or {}).get('ttft_ratio_median') or 0:8.3f}" for c in cases),
            'pairs', max((v['pairs'] for k, v in paired[arm].items() if isinstance(v, dict)), default=0),
            'diff', paired[arm]['output_differs_cases'])


if __name__ == '__main__':
    main()
