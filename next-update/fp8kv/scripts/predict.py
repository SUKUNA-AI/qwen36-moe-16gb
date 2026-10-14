"""CPU-only predictions for FP8 KV on the released build (VRAM plan via the engine's own budget code, decode model).
Inputs are measured release facts (sizing logs, engine.jsonl); outputs go to results/predictions.json."""
import json, math, sys, pathlib
sys.path.insert(0, '/home/USER/qwen36-api/optimization/python')
from freetoken.engine.cache_budget import plan_cache_budget

GIB = 1 << 30
SLOT = 1_781_760                       # expert_bytes_per_slot (engine.jsonl: 5,637,488,640 B / 3164 slots)
KV_BF16, KV_FP8 = 20_480, 10_240       # bytes/token: 10 layers x (K,V) x 2 heads x 256 x 2|1 B
E, TOTAL = 256, 40 * 256


def budget_from(slots, pages, per_page, reserve_tokens):
    """Smallest net budget B consistent with an observed (slots, pages) plan (MoE-first split)."""
    lo1 = reserve_tokens * per_page + slots * SLOT
    lo2 = slots * SLOT + pages * per_page
    return max(lo1, lo2)


def plan(budget, per_page, reserve):
    return plan_cache_budget(budget, SLOT, per_page, E, TOTAL, True, reserve, TOTAL)


out = {'slot_bytes': SLOT, 'kv_bytes_per_token': {'bf16': KV_BF16, 'fp8': KV_FP8}, 'profiles': {}}
# observed release plans (final-f1 sizing logs): main128 3164/131111 (.75), long256 1481/262175 (.73); baseline64 3917/65600 (.75)
obs = {'main128': (3164, 131111, 131072, 0.75), 'long256': (1481, 262175, 262144, 0.73)}
baseline_free = 14.65 * GIB
for name, (slots, pages, reserve, ratio) in obs.items():
    B = budget_from(slots, pages, KV_BF16, reserve)
    assert plan(B, KV_BF16, reserve)[:2] == (slots, pages), (name, plan(B, KV_BF16, reserve))
    s8, p8, _ = plan(B, KV_FP8, reserve)
    out['profiles'][name] = {'memory_ratio': ratio, 'bf16': {'slots': slots, 'kv_tokens': pages, 'kv_gib': pages * KV_BF16 / GIB},
                             'fp8': {'slots': s8, 'kv_tokens': p8, 'kv_gib': p8 * KV_FP8 / GIB},
                             'kv_saved_gib': (pages * KV_BF16 - p8 * KV_FP8) / GIB, 'slots_gained': s8 - slots,
                             'slot_gain_pct': 100 * (s8 - slots) / slots}
    if name == 'main128':
        # headroom variant: keep ~the release slot count, lower memory_ratio
        d = reserve * (KV_BF16 - KV_FP8)
        r_hr = ratio - d / baseline_free
        B_hr = B - int((ratio - 0.665) * baseline_free)
        out['profiles']['main128-hr'] = {'memory_ratio_exact': r_hr, 'memory_ratio_used': 0.665,
                                         'fp8': dict(zip(('slots', 'kv_tokens'), plan(B_hr, KV_FP8, reserve)[:2])),
                                         'extra_free_gib': (B - B_hr) / GIB}
        # 256K context at the main128 budget (.75): one profile for both
        s, p, _ = plan(B, KV_FP8, 262144)
        out['profiles']['main256-fp8-at-.75'] = {'fp8': {'slots': s, 'kv_tokens': p, 'kv_gib': p * KV_FP8 / GIB}}

# decode model (engine ms/forward) fitted on release telemetry: ms = c + m*misses + a*ctx_k
A_ATT = 0.0258           # ms per 1K context per forward (release medians 7.51/8.78/10.18 ms at 16.4K/65.6K/120K)
M_MISS = 0.0447          # ms per (miss/token) (within-length regression on 57 release requests, r=0.69)
MISS_3164 = 32.4         # misses/token at 3164 slots (release, steady)
MISS_1481 = 88.7         # long256 release, same 120K request
b = math.log(MISS_1481 / MISS_3164) / math.log(3164 / 1481)       # power law miss ~ slots^-b
miss = lambda s: MISS_3164 * (s / 3164) ** (-b)
C = 7.51 - M_MISS * MISS_3164 - A_ATT * 16.413
pred = {}
for label, slots_bf16, slots_fp8 in (('main128', 3164, out['profiles']['main128']['fp8']['slots']),
                                     ('main128-hr', 3164, out['profiles']['main128-hr']['fp8']['slots']),
                                     ('long256', 1481, out['profiles']['long256']['fp8']['slots'])):
    rows = {}
    for ctx in (16.4, 65.6, 120.0, 250.0):
        if label.startswith('main') and ctx > 130: continue
        base = C + M_MISS * miss(slots_bf16) + A_ATT * ctx
        rng = []
        for eff in (1.0, 0.7):          # FP8 decode attention bandwidth efficiency relative to BF16 (1.0 = halves time)
            t = C + M_MISS * miss(slots_fp8) + A_ATT * ctx * 0.5 / eff + 0.025   # +25 us: 10 x `out *= v_scale`
            rng.append(base / t - 1)
        rows[f'{ctx:g}K'] = {'bf16_ms_per_forward': round(base, 2), 'misses_bf16': round(miss(slots_bf16), 1),
                             'misses_fp8': round(miss(slots_fp8), 1), 'decode_gain_pct_range': [round(100 * min(rng), 1), round(100 * max(rng), 1)]}
    pred[label] = rows
out['decode_model'] = {'c_ms': C, 'ms_per_miss': M_MISS, 'ms_per_kctx_attention': A_ATT, 'miss_power_b': b, 'predictions': pred}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
