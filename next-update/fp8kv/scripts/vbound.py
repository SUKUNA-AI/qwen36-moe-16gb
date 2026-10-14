"""CPU-only: hard upper bound of |V| for each full-attention layer. V = v_proj(RMSNorm(h) * (1+w_ln)) with FP8 static
weights: |v_j| <= ||W_j * g||_2 * sqrt(d_model) (Cauchy-Schwarz, RMSNorm gives ||xhat||_2 = sqrt(d)), times (1+2^-4)
for the e4m3 input-quantization rounding (clipping only shrinks). Also the K bound recomputed. Compared with kv-scales."""
import json, math, sys, pathlib, torch
from safetensors import safe_open
model = pathlib.Path('/path/to/LLM/qwen36-api/model')
idx = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
scales = json.loads(pathlib.Path(sys.argv[1]).read_text())
def get(name):
    with safe_open(str(model/idx[name]), 'pt') as f: return f.get_tensor(name)
out = {}
for layer in sorted(int(k) for k in scales['scales']):
    p = f'model.language_model.layers.{layer}.'
    g = (1.0 + get(p+'input_layernorm.weight').float()).to(torch.bfloat16).float()
    w = get(p+'self_attn.v_proj.weight'); ws = get(p+'self_attn.v_proj.weight_scale').float(); isc = get(p+'self_attn.v_proj.input_scale').float()
    wd = w.float() * (ws if ws.numel() == 1 else ws.view(-1, 1))
    d = wd.shape[1]
    row = (wd * g).norm(dim=1)                     # ||W_j * g||_2
    bound = float(row.max()) * math.sqrt(d) * (1 + 2**-4)
    # alternative bound through the saturating FP8 input: |x_i| <= 448*input_scale
    bound_sat = float(wd.abs().sum(1).max()) * 448 * float(isc.max())
    amax_v = scales['amax'][layer][1]; s_v = scales['scales'][str(layer)][1]
    vb = min(bound, bound_sat)
    out[layer] = {'v_bound_cauchy': bound, 'v_bound_fp8_input_saturation': bound_sat, 'v_bound': vb,
                  'calibrated_amax_v': amax_v, 'fp8_v_clip_threshold': 448*s_v, 'bound_over_clip': vb/(448*s_v),
                  'v_scale_noclip': vb/448, 'weight_shape': list(w.shape), 'weight_dtype': str(w.dtype),
                  'weight_scale_numel': ws.numel(), 'input_scale': float(isc.max())}
print(json.dumps(out, indent=1))
