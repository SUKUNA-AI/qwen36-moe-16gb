"""Write kv-scales-bound.json: static per-layer e4m3 K/V scales from analytic magnitude bounds (no clipping possible).
K: Gemma RMSNorm (head_dim 256) => |k| <= 16 * max|1+w_k| (pairwise max over the partial-RoPE pairs).
V: |v_j| <= ||W_v[j,:] * (1+w_ln)||_2 * sqrt(2048) * (1 + 2^-4)  (Cauchy-Schwarz; e4m3 input rounding).
Scale = bound * 1.02 / 448 (1.02 covers bf16 rounding of the projection/RoPE output). Same schema as kv-scales.json."""
import json, sys, pathlib
F = pathlib.Path(sys.argv[1])
kb = json.loads((F/'results-kbound.json').read_text())['layers']
vb = json.loads((F/'results-vbound.json').read_text())
old = json.loads(pathlib.Path(sys.argv[2]).read_text())
MARGIN = 1.02
amax = [[0.0, 0.0] for _ in range(40)]
scales = {}
for layer in sorted(int(k) for k in vb):
    k = kb[str(layer)]['k_abs_hard_bound'] * MARGIN
    v = vb[str(layer)]['v_bound'] * MARGIN
    amax[layer] = [k, v]
    scales[str(layer)] = [k / 448, v / 448]
doc = {'checkpoint_revision': old['checkpoint_revision'], 'format': 'e4m3fn', 'method': 'analytic-bound-v1',
       'margin': MARGIN, 'amax': amax, 'scales': scales,
       'note': 'amax = analytic magnitude bound x margin (not observed values); derivation: work/next-update/fp8kv/scripts/kbound.py, vbound.py, gen_scales.py',
       'previous_calibrated_scales': old['scales']}
out = F/'configs'/'kv-scales-bound.json'
out.write_text(json.dumps(doc, indent=2) + '\n')
print(json.dumps({l: [round(a, 5), round(b, 5)] for l, (a, b) in scales.items()}))
