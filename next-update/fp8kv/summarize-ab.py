"""Summarize the FP8-KV speed/VRAM A/B (screen-chunk.py receipts) against a PRE-REGISTERED report rule.

Reads $N/reports/<name>-<profile>-chunk-screen.json for every --round name; pairs the baseline and the FP8 arm
measured in the same round (same GPU-lock session, same sustained warm-up), reports every round and the medians.
Writes --out once (O_EXCL). Not a qualification; speed is an owner decision input.

SPEED RULE (fp8kv-speed-rule-1, report-only):
  SPEED_CONFIRMED iff, over all rounds (>= 3 for main128):
    decode_client_tps ratio fp8/base: median >= 1.05 at 65536 AND >= 1.08 at 120000, and > 1.00 at 120000 in every round;
    TTFT ratio fp8/base: median <= 1.05 at every length;
  VRAM_OK iff every FP8 request min_gpu_free >= 0.5 GiB (owner floor) and the fp8 arm's median min free is not more
    than 0.25 GiB below the base arm's median (same memory_ratio => same budget);
  SLOTS_AS_PREDICTED iff expert_slots within +-3 of the CPU prediction (main128 fp8 3917, long256 fp8 2987, hr ~3167).
"""
import argparse
import json
import os
import pathlib
import statistics as st

ROOT = pathlib.Path('/home/USER/qwen36-api/optimization')
GIB = 1 << 30
PREDICTED_SLOTS = {'main-release': 3164, 'main-release-fp8kv': 3917, 'main-release-fp8kv-hr': 3167,
                   'long256-release': 1481, 'long256-release-fp8kv': 2987}


def fresh(path):
    d = json.loads(path.read_text())
    rows = {x['length']: x for x in d.get('rows', []) if x.get('case') == 'fresh'}
    return d, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', required=True, help='comma list of screen-chunk --name values')
    p.add_argument('--base', required=True)
    p.add_argument('--fp8', required=True)
    p.add_argument('--extra', default='', help='optional third profile measured in some rounds (e.g. the hr arm)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit('refused: report exists')
    rounds, per_len = [], {}
    arms = [a.base, a.fp8] + ([a.extra] if a.extra else [])
    for name in a.rounds.split(','):
        entry = {'round': name, 'arms': {}}
        for prof in arms:
            path = ROOT / 'reports' / f'{name}-{prof}-chunk-screen.json'
            if not path.exists():
                continue
            d, rows = fresh(path)
            entry['arms'][prof] = {
                'status': d.get('status'), 'min_gpu_free_gib': (d.get('min_gpu_free_bytes') or 0) / GIB, 'vram_gate': d.get('vram_gate'),
                'load_ready_s': d.get('load_ready_s'),
                'expert_slots': sorted({(x.get('engine') or {}).get('expert_slots') for x in rows.values()}),
                'by_length': {L: {k: x.get(k) for k in ('ttft_content_s', 'elapsed_s', 'decode_client_tps', 'min_gpu_free_bytes')}
                              | {'decode_missing': ((x.get('engine') or {}).get('expert_counter_deltas') or {}).get('decode_missing'),
                                 'decode_gpu_ms_per_forward': ((x.get('engine') or {}).get('decode_gpu_ms') or 0)
                                 / max(1, (x.get('engine') or {}).get('decode_forwards') or 1)}
                              for L, x in rows.items()}}
        b, f = entry['arms'].get(a.base), entry['arms'].get(a.fp8)
        if b and f:
            entry['ratios'] = {}
            for L in sorted(set(b['by_length']) & set(f['by_length'])):
                xb, xf = b['by_length'][L], f['by_length'][L]
                r = {'decode_tps': xf['decode_client_tps'] / xb['decode_client_tps'],
                     'ttft': xf['ttft_content_s'] / xb['ttft_content_s'], 'elapsed': xf['elapsed_s'] / xb['elapsed_s']}
                entry['ratios'][L] = r
                per_len.setdefault(L, []).append(r)
        rounds.append(entry)
    med = {L: {k: st.median(r[k] for r in v) for k in ('decode_tps', 'ttft', 'elapsed')} | {'rounds': len(v)}
           for L, v in sorted(per_len.items())}
    speed = None
    if 65536 in med and 120000 in med:
        speed = (med[65536]['decode_tps'] >= 1.05 and med[120000]['decode_tps'] >= 1.08
                 and all(r['decode_tps'] > 1.0 for r in per_len[120000])
                 and all(m['ttft'] <= 1.05 for m in med.values()))
    fp8_free = [x['min_gpu_free_bytes'] for e in rounds for x in e['arms'].get(a.fp8, {}).get('by_length', {}).values()
                if isinstance(x.get('min_gpu_free_bytes'), int)]
    base_free = [x['min_gpu_free_bytes'] for e in rounds for x in e['arms'].get(a.base, {}).get('by_length', {}).values()
                 if isinstance(x.get('min_gpu_free_bytes'), int)]
    vram_ok = bool(fp8_free) and min(fp8_free) >= GIB // 2 and (not base_free or st.median(fp8_free) >= st.median(base_free) - GIB // 4)
    slots = {prof: sorted({s for e in rounds for s in e['arms'].get(prof, {}).get('expert_slots', [])}) for prof in arms}
    slots_ok = all(all(abs(s - PREDICTED_SLOTS[prof]) <= 3 for s in v if isinstance(s, int)) for prof, v in slots.items()
                   if prof in PREDICTED_SLOTS)
    report = {'kind': 'fp8kv-ab-summary', 'rule': 'fp8kv-speed-rule-1 (see module docstring)', 'base': a.base, 'fp8': a.fp8,
              'extra': a.extra or None, 'rounds': rounds, 'median_ratios': med, 'SPEED_CONFIRMED': speed,
              'VRAM_OK': vram_ok, 'min_free_gib': {'fp8_min': min(fp8_free) / GIB if fp8_free else None,
                                                   'fp8_median': st.median(fp8_free) / GIB if fp8_free else None,
                                                   'base_median': st.median(base_free) / GIB if base_free else None},
              'expert_slots': slots, 'SLOTS_AS_PREDICTED': slots_ok}
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as fh:
        fh.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'median_ratios': med, 'SPEED_CONFIRMED': speed, 'VRAM_OK': vram_ok, 'slots': slots}))


if __name__ == '__main__':
    main()
