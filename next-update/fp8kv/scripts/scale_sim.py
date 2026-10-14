"""CPU-only synthetic check: e4m3 per-tensor quantization error of K for the calibrated scale vs the analytic
no-clip scale (K = Gemma-RMSNorm output with the real checkpoint k_norm gains). Not a model-quality result."""
import json, sys, torch
torch.manual_seed(0)
kb = json.load(open(sys.argv[1]))
res = {}
from safetensors import safe_open
import pathlib
model = pathlib.Path('/path/to/LLM/qwen36-api/model')
idx = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
for layer, row in sorted(((l, r) for l, r in kb['layers'].items() if r['fp8_k_clip_threshold']), key=lambda x: int(x[0])):
    with safe_open(str(model/idx[row['tensor']]), 'pt') as f:
        g = (1.0 + f.get_tensor(row['tensor']).float()).to(torch.bfloat16).float()
    n = 20000
    x = torch.randn(n, 256)
    # heavy-tailed variant: a few dominant channels per vector (outlier-like)
    x2 = torch.randn(n, 256) * torch.where(torch.rand(n, 256) < 0.02, 8.0, 1.0)
    out = {}
    for name, base in (('gauss', x), ('spiky', x2)):
        xh = base / base.pow(2).mean(-1, keepdim=True).sqrt()
        k = (xh * g).to(torch.bfloat16).float()
        q = torch.randn(n, 256)
        q = q / q.norm(dim=-1, keepdim=True) * 16 * 1.5  # |q| typical after RMSNorm with gain ~1.5
        for sname, s in (('calibrated', row['fp8_k_clip_threshold'] / 448), ('nocip_bound', row['k_abs_hard_bound'] / 448)):
            kq = (k / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
            clipped = int(((k / s).abs() > 448).sum())
            err = (kq - k)
            rel = float(err.norm() / k.norm())
            dlogit = ((q * err).sum(-1) / 16.0)  # sm_scale 1/sqrt(256)
            out[f'{name}/{sname}'] = {'rel_l2': rel, 'clipped': clipped, 'max_abs_k': float(k.abs().max()),
                                      'logit_err_std': float(dlogit.std()), 'logit_err_p999': float(dlogit.abs().quantile(.999))}
    res[layer] = out
print(json.dumps(res, indent=1))
