"""Summarize short-route sessions: per arm x case medians (+ every round) and paired ratios vs arm r.
Usage: summarize.py SESSION [SESSION ...]  -> results/summary.json + printed table"""
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path('/path/to/qwen36-workspace/work/next-update/short-route')
CASES = ['fresh-64', 'fresh-512', 'fresh-1024', 'fresh-2000', 'fresh-2200', 'fresh-4096', 'prefixhit-4096',
         'fresh-8192', 'edit16k-primer', 'edit16k']


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def itl_ms(times, lo, hi):
    t = times or []
    if len(t) <= hi:
        return None
    return 1000 * (t[hi] - t[lo]) / (hi - lo)


def load(sessions):
    rows = []
    for s in sessions:
        for f in sorted((HERE / 'results' / s).glob('*.jsonl')):
            prev_lm = None
            for line in f.read_text().splitlines():
                r = json.loads(line)
                r['session'] = s
                lm = r.get('lm_prefills_total') or 0
                r["lm_used"] = (lm > prev_lm) if prev_lm is not None else False  # first row: fresh-64, 1 chunk
                prev_lm = lm
                r['itl_early_ms'] = itl_ms(r.get('delta_times_s'), 0, 8)    # tokens 2..9
                r['itl_late_ms'] = itl_ms(r.get('delta_times_s'), 16, 62)   # tokens 18..63
                r['h2d_gb'] = (r.get('prefill_h2d_bytes') or 0) / 1e9
                r['d2d_gb'] = (r.get('prefill_d2d_bytes') or 0) / 1e9
                df = r.get('decode_forwards') or 0
                r['decode_step_ms'] = (r['decode_gpu_ms'] / df) if df and r.get('decode_gpu_ms') is not None else None
                r['miss_per_tok'] = (r['decode_missing'] / df) if df and r.get('decode_missing') is not None else None
                rows.append(r)
    return rows


def main():
    sessions = sys.argv[1:]
    rows = load(sessions)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['arm'], r['case'])].append(r)
    arms = sorted({r['arm'] for r in rows}, key='rdp'.index)
    summary = {'sessions': sessions, 'arms': {}, 'paired': {}, 'output_identity': {}}
    keys = ('prompt_tokens', 'cached', 'ttft_s', 'prefill_gpu_ms', 'h2d_gb', 'd2d_gb', 'decode_client_tps',
            'itl_early_ms', 'itl_late_ms', 'decode_step_ms', 'miss_per_tok')
    print(f"{'arm':3} {'case':15} {'n':>2} {'tok':>6} {'cached':>6} {'TTFT':>6} {'pf_ms':>7} {'H2D':>6} {'D2D':>5} "
          f"{'LM':>3} {'dec/s':>6} {'itl8':>5} {'itlL':>5} {'step':>5} {'miss':>5}")
    for a in arms:
        summary['arms'][a] = {}
        for c in CASES:
            rs = by.get((a, c), [])
            if not rs:
                continue
            m = {k: med([r.get(k) for r in rs]) for k in keys}
            m['n'] = len(rs)
            m['lm_used'] = sum(r['lm_used'] for r in rs)
            m['rounds_ttft_s'] = [round(r['ttft_s'], 4) if r.get('ttft_s') else None for r in rs]
            summary['arms'][a][c] = m
            f = lambda v, p=3: '-' if v is None else f'{v:.{p}f}'  # noqa: E731
            print(f"{a:3} {c:15} {m['n']:>2} {f(m['prompt_tokens'],0):>6} {f(m['cached'],0):>6} {f(m['ttft_s']):>6} "
                  f"{f(m['prefill_gpu_ms'],1):>7} {f(m['h2d_gb'],2):>6} {f(m['d2d_gb'],2):>5} {m['lm_used']:>3} "
                  f"{f(m['decode_client_tps'],1):>6} {f(m['itl_early_ms'],2):>5} {f(m['itl_late_ms'],2):>5} "
                  f"{f(m['decode_step_ms'],2):>5} {f(m['miss_per_tok'],1):>5}")
    # paired: same session, rep, case
    idx = {(r['session'], r['rep'], r['case'], r['arm']): r for r in rows}
    print('\npaired TTFT ratio (arm / r), median [min, max] over (session, rep):')
    for a in arms:
        if a == 'r':
            continue
        summary['paired'][a] = {}
        for c in CASES:
            ratios, deltas = [], []
            for (s, rep, case, arm), r in idx.items():
                if arm != a or case != c:
                    continue
                b = idx.get((s, rep, case, 'r'))
                if b and r.get('ttft_s') and b.get('ttft_s'):
                    ratios.append(r['ttft_s'] / b['ttft_s'])
                    deltas.append(r['ttft_s'] - b['ttft_s'])
            if ratios:
                summary['paired'][a][c] = {'n': len(ratios), 'ratio_median': med(ratios), 'ratio_min': min(ratios),
                                           'ratio_max': max(ratios), 'delta_s_median': med(deltas)}
                print(f"  {a}/r {c:15} n={len(ratios):2} ratio {med(ratios):.3f} [{min(ratios):.3f}, {max(ratios):.3f}]"
                      f"  dTTFT {med(deltas):+.3f} s")
    # greedy output identity across arms (observation, not the teacher gate)
    same = diff = 0
    for (s, rep, case, arm), r in idx.items():
        if arm == 'r':
            continue
        b = idx.get((s, rep, case, 'r'))
        if b and r.get('messages_sha256') == b.get('messages_sha256'):
            if r.get('content_sha256') == b.get('content_sha256'):
                same += 1
            else:
                diff += 1
    summary['output_identity'] = {'same': same, 'different': diff}
    print(f'\ngreedy outputs identical to arm r: {same}, different: {diff}')
    (HERE / 'results' / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
