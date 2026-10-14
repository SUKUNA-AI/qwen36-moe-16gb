"""CPU-only synthetic: e4m3 per-tensor V quantization error, calibrated scale vs analytic no-clip scale.
V modeled as Gaussian with sigma = calibrated_amax/5 and a heavy-tailed (Student-t, df=3) variant scaled the same."""
import json, sys, torch
torch.manual_seed(1)
vb = json.load(open(sys.argv[1]))
res = {}
for layer, r in vb.items():
    sig = r['calibrated_amax_v'] / 5
    out = {}
    for name, v in (('gauss', torch.randn(4_000_000) * sig),
                    ('student_t3', torch.distributions.StudentT(3.0).sample((4_000_000,)) * sig / 1.732)):
        for sname, s in (('calibrated', r['fp8_v_clip_threshold'] / 448), ('noclip_bound', r['v_bound'] * 1.02 / 448)):
            q = (v / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
            e = q - v
            out[f'{name}/{sname}'] = {'rel_l2': round(float(e.norm() / v.norm()), 5), 'clipped': int(((v / s).abs() > 448).sum()),
                                      'subnormal_frac': round(float(((v / s).abs() < 2**-6).float().mean()), 6)}
    res[layer] = out
print(json.dumps(res, indent=1))
