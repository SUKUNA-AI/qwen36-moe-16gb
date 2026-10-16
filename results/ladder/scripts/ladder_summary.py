"""CPU-only: teacher.json + ab.json + nsys.json (+ ncu, release A/B) -> summary.json (qwen36-ladder-summary v1) and README.md.

python3 -I ladder_summary.py --dir results/ladder --configs harness --win-csv win-gpu-shared-matrix.csv \
    [--release-ab rab.json --release-compare rcmp.json --ncu-dir DIR]
Only numbers and generic names; kernel symbols never appear.
"""
import argparse, csv, json, pathlib, statistics

LENGTHS = ['64', '2048', '4096', '16384', '120000']
BUCKET_MAP = {'moe': 'moe_experts', 'fp8_dense_oproj': 'fp8_gemv_gemm', 'gather': 'decode_gather', 'attention': 'full_attention',
              'gdn': 'gdn', 'h2d_exposed': 'expert_h2d', 'idle': 'cpu_gaps_gpu_idle',
              'copies_other': 'norms_misc', 'norm_sampling_misc': 'norms_misc'}
MIB = 1 << 20


def jload(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return None


def sid(step_id):
    return 'T3only' if step_id == 't3' else step_id.upper()


def per_length(step, key):
    out = {}
    for length in LENGTHS:
        cell = (step.get('cases') or {}).get(f'fresh_{length}', {}).get(key)
        if cell:
            out[length] = cell['pairs']
    return out


def med(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def percentile(sorted_values, q):
    """Linear interpolation between order statistics (numpy default); with few values p95/p99 approach the max."""
    k = (len(sorted_values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def stats(values):
    v = [x for x in values or [] if isinstance(x, (int, float))]
    if not v:
        return None
    return {'median': statistics.median(v), 'mean': statistics.fmean(v), 'n': len(v), 'min': min(v), 'max': max(v),
            'p95': percentile(sorted(v), .95), 'p99': percentile(sorted(v), .99)}


def buckets(raw):
    out = {}
    for key, value in (raw or {}).items():
        out[BUCKET_MAP.get(key, key)] = out.get(BUCKET_MAP.get(key, key), 0) + value
    return out


def nsys_block(entry):
    if not entry or entry.get('status') != 'OK':
        return None
    req = entry['requests']
    out = {}
    d = req.get('fresh_16k')
    if d and d.get('decode_buckets_ms_per_step'):
        out['decode_16k'] = {'ms_per_step': d['decode_step_critical_ms'], 'buckets_ms': buckets(d['decode_buckets_ms_per_step']),
                             'capture_complete': d.get('capture_complete')}
    for name, label in (('prefill_short', 'warm_decode'), ('prefill_16k', 'fresh_16k')):
        r = req.get(label)
        if r:
            h2d = ((r.get('transport') or {}).get('prefill') or {}).get('h2d') or {}
            out[name] = {'window_ms': r['prefill_critical_ms'], 'buckets_ms': buckets(r['prefill_buckets_ms']),
                         'h2d_busy_ms': h2d.get('union_ms'), 'h2d_overlap_pct': h2d.get('overlap_percent')}
    return out or None


def windows_shared(path, lo=None, hi=None):
    try:
        rows = [r for r in csv.DictReader(open(path)) if (lo is None or r['utc'] >= lo) and (hi is None or r['utc'] <= hi)]
    except OSError:
        return None
    if not rows:
        return None
    shared = [float(r['shared_mib']) for r in rows]
    n = max(1, len(shared) // 10)
    return {'samples': len(shared), 'first_decile_median_mib': statistics.median(shared[:n]),
            'last_decile_median_mib': statistics.median(shared[-n:]), 'min_mib': min(shared), 'max_mib': max(shared),
            'drift_mib': statistics.median(shared[-n:]) - statistics.median(shared[:n])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', type=pathlib.Path, required=True)
    p.add_argument('--configs', type=pathlib.Path, required=True)
    p.add_argument('--win-csv', type=pathlib.Path)
    p.add_argument('--win-from'); p.add_argument('--win-to'); p.add_argument('--ncu-note')
    p.add_argument('--release-ab', type=pathlib.Path)
    p.add_argument('--release-compare', type=pathlib.Path)
    p.add_argument('--ncu-dir', type=pathlib.Path)
    p.add_argument('--generated-utc', required=True)
    a = p.parse_args()
    steps = jload(a.dir / 'steps.json')['steps']
    teacher = {s['id']: s for s in (jload(a.dir / 'teacher.json') or {}).get('steps', [])}
    ab = jload(a.dir / 'ab.json') or {'steps': [], 'deltas_vs_prev': {}, 'deltas_vs_l0': {}}
    abs_ = {s['id']: s for s in ab['steps']}
    nsys = {s['id']: s for s in (jload(a.dir / 'nsys.json') or {}).get('steps', [])}
    out_steps = []
    for s in steps:
        i = s['id']
        cfg = jload(a.configs / f"{s['config']}.json") or {}
        t = teacher.get(i, {})
        tt = t.get('teacher') or {}
        row = {'id': sid(i), 'label': s['lever_short'], 'lever': s['lever'], 'config': f"harness/{s['config']}.json",
               'parent': sid(s['prev']) if s['prev'] else None, 'cumulative': i != 't3'}
        row['teacher'] = {'status': {'PASS': 'pass', 'FAIL': 'fail'}.get(t.get('status'), 'not_run'), 'passed': tt.get('observed'),
                          'total': tt.get('expected'), 'max_abs': tt.get('max_abs'), 'report': t.get('quality_report'),
                          'all_byte_equal': tt.get('all_byte_equal'), 'new_functional_failures': tt.get('new_functional_failures')}
        cur = abs_.get(i)
        if cur:
            par = abs_.get(s['prev']) if s['prev'] else None
            row['ab'] = {'reference': sid(s['prev']) if s['prev'] else None, 'pairs': ab.get('pairs'),
                         'ttft_s': per_length(cur, 'ttft_s'), 'ref_ttft_s': per_length(par, 'ttft_s') if par else None,
                         'decode_tps': per_length(cur, 'decode_tps'), 'ref_decode_tps': per_length(par, 'decode_tps') if par else None,
                         'elapsed_s': per_length(cur, 'elapsed_s'), 'report': ab.get('matrix') + '-screen.json'}
            row['ab']['stats'] = {m: {l: stats(v) for l, v in row['ab'][m].items()} for m in ('ttft_s', 'decode_tps', 'elapsed_s')}
            vram = cur.get('vram') or {}
            row['memory'] = {'memory_ratio': cfg.get('memory_ratio'), 'expert_slots': (cur.get('expert_slots') or [None])[-1],
                             'process_peak_mib': (vram.get('process_peak_bytes_max') or 0) / MIB or None,
                             'process_budget_mib': (vram.get('budget_bytes') or 0) / MIB or None,
                             'free_vram_min_mib': (vram.get('min_gpu_free_bytes_min') or 0) / MIB if vram.get('min_gpu_free_bytes_min') is not None else None,
                             'misses_per_token': {l: med((cur['cases'].get(f'fresh_{l}', {}).get('misses_per_token') or {}).get('pairs', []))
                                                  for l in LENGTHS}}
        row['nsys'] = nsys_block(nsys.get(i))
        out_steps.append(row)
    summary = {'schema': 'qwen36-ladder-summary', 'schema_version': 1, 'synthetic': False, 'generated_utc': a.generated_utc,
               'profile': 'main128', 'lengths': [int(x) for x in LENGTHS], 'output_tokens': 128, 'steps': out_steps,
               'windows_shared_gpu': dict(windows_shared(a.win_csv, a.win_from, a.win_to) or {}, window='steady window of the 2 h soak; shared usage includes the pinned expert memory of a loaded unit (≈18.5 GiB) and moves with every unit start/stop, so only a single-load window shows paging') if a.win_csv else None,
               'notes': ['TTFT and decode: raw per-pair values of the A/B screen (3 pairs, rotating order, 60 s sustained warm-up, no profiler); '
                         'keys are document token counts of the fresh cases.',
                         'nsys timings are diagnostic (under the profiler); buckets are critical-chain ms: decode per step, prefill per window.']}
    if a.release_ab:
        r = jload(a.release_ab)
        cmp_ = jload(a.release_compare) if a.release_compare else None
        if r and len(r['steps']) == 2:
            ref, cand = r['steps']
            summary['release_ab'] = {'candidate': cand['config'], 'reference': ref['config'], 'pairs': r['pairs'],
                                     'ttft_s': per_length(cand, 'ttft_s'), 'ref_ttft_s': per_length(ref, 'ttft_s'),
                                     'stats': {'candidate': {m: {l: stats(v) for l, v in per_length(cand, m).items()} for m in ('ttft_s', 'decode_tps', 'elapsed_s')},
                                               'reference': {m: {l: stats(v) for l, v in per_length(ref, m).items()} for m in ('ttft_s', 'decode_tps', 'elapsed_s')}},
                                     'decode_tps': per_length(cand, 'decode_tps'), 'ref_decode_tps': per_length(ref, 'decode_tps'),
                                     'verdict': (cmp_ or {}).get('verdict'), 'contract_pass': (cmp_ or {}).get('contract_pass'),
                                     'report': r['matrix']}
    if a.ncu_dir and a.ncu_dir.is_dir():
        summary['ncu'] = {p.stem: [json.loads(l) for l in p.read_text().splitlines() if l.strip()] for p in sorted(a.ncu_dir.glob('*.jsonl'))}
    if 'ncu' not in summary:
        summary['ncu'] = {'status': 'NOT_RUN', 'reason': a.ncu_note}
    (a.dir / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    write_readme(a.dir, summary, ab)
    print(json.dumps({'steps': len(out_steps), 'release_ab': 'release_ab' in summary, 'ncu': 'ncu' in summary}))


def fmt(x, nd=2):
    return '—' if x is None else f'{x:.{nd}f}'


def pct(x):
    return '—' if x is None else f'{x:+.1f}%'


def write_readme(d, s, ab):
    lines = ['# Лесенка main128: вклад каждого рычага', '',
             f"Сгенерировано {s['generated_utc']} скриптами `scripts/ladder_*.py` из отчётов GPU-прогона. Бит-точность: teacher 125/125, max_abs 0.",
             'A/B без профайлера: ротирующаяся матрица, 3 пары, прогрев 60 с, вывод 128 токенов. В ячейках: медиана (среднее; min–max; p95, p99; n) по парам; при n=3–5 перцентили — грубая оценка, она стремится к максимуму. Шаги кумулятивные: ступень = предыдущая + один рычаг.', '',
             *( (d / 'notes.md').read_text(encoding='utf-8').splitlines() + [''] if (d / 'notes.md').exists() else []),
             '## Teacher и память', '', '| Ступень | Рычаг | teacher | max_abs | ratio | слоты | пик VRAM процесса, МиБ | промахов/ток (16K) |', '|---|---|---|---|---|---|---|---|']
    for r in s['steps']:
        t, m = r['teacher'], r.get('memory') or {}
        lines.append(f"| {r['id']} | {r['label']} | {t['status']} {t.get('passed') or '—'}/{t.get('total') or '—'} | {t.get('max_abs')} | "
                     f"{m.get('memory_ratio', '—')} | {m.get('expert_slots', '—')} | {fmt(m.get('process_peak_mib'), 0)} | {fmt((m.get('misses_per_token') or {}).get('16384'), 1)} |")
    for kind, title, key in (('decode_tps', 'Декод, ток/с', 'decode_tps'), ('ttft_s', 'TTFT, с', 'ttft_s'),
                             ('elapsed_s', 'Полная латентность запроса (отправка → последний токен, `elapsed_s`), с', 'elapsed_s')):
        lines += ['', f'## {title}', '', '| Ступень | ' + ' | '.join(LENGTHS) + ' | vs пред., 16K | vs L0, 16K |', '|---|' + '---|' * (len(LENGTHS) + 2)]
        for r in s['steps']:
            ab_ = r.get('ab') or {}
            st_ = [stats((ab_.get(key) or {}).get(l, [])) for l in LENGTHS]
            nd_ = 1 if key == 'decode_tps' else 3
            meds = [None if x is None else f"{x['median']:.{nd_}f} (ср. {x['mean']:.{nd_}f}; {x['min']:.{nd_}f}–{x['max']:.{nd_}f}; p95 {x['p95']:.{nd_}f}, p99 {x['p99']:.{nd_}f}; n={x['n']})" for x in st_]
            dp = ab['deltas_vs_prev'].get(r['id'].lower() if r['id'] != 'T3only' else 't3', {}).get('fresh_16384', {}).get(key + '_pct', {}).get('median')
            d0 = ab['deltas_vs_l0'].get(r['id'].lower() if r['id'] != 'T3only' else 't3', {}).get('fresh_16384', {}).get(key + '_pct', {}).get('median')
            lines.append(f"| {r['id']} | " + ' | '.join('—' if v is None else v for v in meds) + f' | {pct(dp)} | {pct(d0)} |')
    lines += ['', '## Nsight Systems: критический путь', '',
              'Корзины в мс: декод на шаг (16K), prefill на окно (короткий 278 токенов и 16K). Время под профайлером, только диагностика.', '']
    for name, title in (('decode_16k', 'Декод 16K, мс/шаг'), ('prefill_short', 'Prefill короткого запроса, мс'), ('prefill_16k', 'Prefill 16K, мс')):
        keys = ['moe_experts', 'fp8_gemv_gemm', 'decode_gather', 'full_attention', 'gdn', 'expert_h2d', 'norms_misc', 'cpu_gaps_gpu_idle']
        lines += [f'### {title}', '', '| Ступень | всего | ' + ' | '.join(keys) + ' | H2D занято / перекрытие |', '|---|---|' + '---|' * (len(keys) + 1)]
        for r in s['steps']:
            n = (r.get('nsys') or {}).get(name)
            if not n:
                lines.append(f"| {r['id']} | н/д |" + ' |' * (len(keys) + 1))
                continue
            total = n.get('ms_per_step') if name == 'decode_16k' else n.get('window_ms')
            h2d = '' if name == 'decode_16k' else f"{fmt(n.get('h2d_busy_ms'), 0)} мс / {fmt(n.get('h2d_overlap_pct'), 0)}%"
            lines.append(f"| {r['id']} | {fmt(total, 2 if name == 'decode_16k' else 0)} | " + ' | '.join(fmt(n['buckets_ms'].get(k), 2 if name == 'decode_16k' else 0) for k in keys) + f' | {h2d} |')
        lines.append('')
    ra = s.get('release_ab')
    if ra:
        lines += ['## Релизный A/B (5 пар) и compare-ab', '', f"Кандидат `{ra['candidate']}` против `{ra['reference']}`: вердикт **{ra.get('verdict')}** (contract_pass={ra.get('contract_pass')}).", '']
    w = s.get('windows_shared_gpu')
    if w:
        lines += ['## Подкачка Windows', '', f"Shared GPU memory за окно soak 2 ч ({w['samples']} замеров): первая/последняя декады {w['first_decile_median_mib']:.0f}/{w['last_decile_median_mib']:.0f} МиБ, дрейф {w['drift_mib']:+.0f} МиБ, максимум {w['max_mib']:.0f}.", '']
    if s['ncu'].get('status') == 'NOT_RUN':
        lines += ['## Nsight Compute', '', 'NOT_RUN: ' + str(s['ncu'].get('reason')), '']
    (d / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
