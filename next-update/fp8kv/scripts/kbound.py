"""CPU-only: hard upper bound of |K| after Gemma (1+w) RMSNorm + partial NeoX RoPE, per full-attention layer,
compared with the static FP8 K clip threshold (448*scale) in kv-scales.json. Reads checkpoint tensors only."""
import json, sys, pathlib, math
import torch
from safetensors import safe_open
model = pathlib.Path('/path/to/LLM/qwen36-api/model')
cfg = json.loads((model/'config.json').read_text())
tc = cfg.get('text_config', cfg)
idx = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
names = {k: v for k, v in idx.items() if 'k_norm' in k or 'q_norm' in k}
scales = json.loads(pathlib.Path(sys.argv[1]).read_text())
head_dim = tc['head_dim']; rot = int(head_dim * tc.get('partial_rotary_factor', tc.get('rope_parameters', {}).get('partial_rotary_factor', 1.0)))
out = {'head_dim': head_dim, 'rotary_dim': rot, 'rms_norm_eps': tc.get('rms_norm_eps'), 'layers': {}}
for name, shard in sorted(names.items()):
    if 'k_norm' not in name: continue
    layer = int(name.split('layers.')[1].split('.')[0])
    with safe_open(str(model/shard), 'pt') as f:
        w = f.get_tensor(name).float()
    g = 1.0 + w                      # Gemma-style (1+w)
    g_bf16 = (1.0 + w).to(torch.bfloat16).float()  # FreeToken folds (1+w) to bf16 at load
    a = g_bf16.abs()
    # RMSNorm: sum(xhat^2) = d (up to eps) -> |xhat_i| <= sqrt(d). Partial NeoX RoPE rotates pairs (i, i+rot/2), i<rot/2.
    half = rot // 2
    pair = torch.maximum(a[:half], a[half:rot])
    bound_rot = math.sqrt(head_dim) * float(pair.max())
    bound_pass = math.sqrt(head_dim) * float(a[rot:].max()) if rot < head_dim else 0.0
    bound = max(bound_rot, bound_pass)
    s = scales['scales'].get(str(layer))
    clip = 448 * s[0] if s else None
    out['layers'][layer] = {'tensor': name, 'max_abs_gain': float(a.max()), 'k_abs_hard_bound': bound,
                            'calibrated_amax_k': scales['amax'][layer][0] if s else None,
                            'fp8_k_clip_threshold': clip, 'bound_over_clip': (bound / clip) if clip else None,
                            'gain_p99': float(a.quantile(.99)), 'gain_median': float(a.median())}
print(json.dumps(out, indent=2))
