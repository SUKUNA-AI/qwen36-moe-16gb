<!-- DRAFT ONLY, NOT POSTED. Local file; the owner decides the repository and whether to post. Suggested target: FlashML-org/FreeToken (issue). -->

# Qwen3.5/3.6: Gemma-style `(1 + w)` RMSNorm weights are folded into bf16 at load time; transformers applies `(1 + w)` in fp32

## Summary

Qwen3.5/3.6 store zero-centred RMSNorm weights (`scale - 1`), and the reference implementation multiplies by
`(1 + w)` in fp32. This applies to `input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`,
`self_attn.k_norm` and the final `model.norm`. The transformers `Qwen3_5MoeRMSNorm` computes
`output * (1.0 + self.weight.float())`.

FreeToken's Qwen3.5 loader instead yields `tensor + 1.0` on the bf16 checkpoint tensor. That stores
**bf16(1 + w)**, and `GemmaRMSNorm` then scales by this rounded weight. In `nvidia/Qwen3.6-35B-A3B-NVFP4`,
**77.4% of the 171,008 norm multipliers change**. The per-channel relative error is on average 1.4e-3 and at most
3.9e-3 (half a bf16 ulp). The error is systematic per channel and the same for every token.

FreeToken already documents this pitfall elsewhere. The `GemmaPlusOneRMSNorm` docstring (`layers/norm.py:80-82`)
says the `+1` must be added in fp32 at runtime and must not be folded into the bf16 weight. The `qwen4_exp`
loader comment (`models/qwen4_exp/weight.py:63-66`) explains that folding would round away small |w|. As far as
we can see, only the `qwen3_5_moe` path folds.

## Environment

Same as the NVFP4-global report:
- FreeToken 0.1.3 `cac247a860e316e06580d05aeb05f2e647bde214`
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5`
- RTX 5070 Ti (sm_120), torch 2.11.0+cu130, sgl-kernel 0.4.5, flashinfer-python 0.6.18.post1, transformers 5.16.1

The fold is a load-time property and is independent of GPU and flags. It affects every Qwen3.5/3.6 checkpoint
served through `models/qwen3_5_moe`, including the dense variants, which share `model.py`.

## Minimal reproduction (CPU only)

```python
import json, pathlib, re, torch
from safetensors import safe_open

MODEL = pathlib.Path("<local snapshot of nvidia/Qwen3.6-35B-A3B-NVFP4>")
idx = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
pat = re.compile(r"^model\.language_model\.(norm|layers\.\d+\.(input_layernorm|post_attention_layernorm|"
                 r"self_attn\.q_norm|self_attn\.k_norm))\.weight$")
total = inexact = 0
rels = []
for n in sorted(n for n in idx if pat.match(n)):
    with safe_open(str(MODEL / idx[n]), framework="pt") as f:
        w = f.get_tensor(n)                       # bf16 in the checkpoint
    folded = (w + 1.0).double()                   # what FreeToken stores: bf16(1 + w)
    exact = w.double() + 1.0                      # what transformers multiplies by: (1 + w) in fp32 (exact here)
    nz = exact != 0
    rels.append((folded - exact).abs()[nz] / exact.abs()[nz])
    inexact += int((folded != exact).sum())
    total += w.numel()
rel = torch.cat(rels)
print("elements:", total, "inexact share: %.4f" % (inexact / total))
print("relative error mean/max: %.3g %.3g" % (float(rel.mean()), float(rel.max())))
```

Output on this checkpoint:

```
elements: 171008 inexact share: 0.7741
relative error mean/max: 0.00138 0.00389
```

Per norm type, the mean relative error over all multipliers is 1.34e-3 (input), 1.43e-3 (post-attention),
1.28e-3 (q_norm) and 1.34e-3 (final norm). The maximum, 3.89e-3, is 2^-8, half a bf16 ulp.

Op-level check on GPU, on real layer inputs from layers 0/3/18/19/38/39 and the final norm, against an fp64
reference with exact `(1 + w)`:

| norm | FreeToken today: `sgl_kernel.rmsnorm(x, bf16(1+w))` | proposed: `sgl_kernel.gemma_rmsnorm(x, w)` |
|---|---|---|
| input_layernorm | bf16 match 70.0%, rel L2 3.0e-3 | 99.999%, rel L2 1.8e-3 (= output rounding) |
| post_attention_layernorm | 70.2%, 2.5e-3 | 99.999%, 1.7e-3 |
| q_norm / k_norm | 74.3% / 73.3%, 2.4e-3 / 2.6e-3 | q_norm 99.999%, 1.7e-3 (k_norm not run) |
| final norm (before lm_head) | 74.7%, 2.6e-3 | 100%, 1.6e-3 |

"bf16 match" is the share of bf16 outputs equal to the correctly rounded fp64 result. We audited the other ops
the same way: FP8 W8A8 GEMMs, GDN conv/gating/chunk/decode, gated norm, RoPE, FlashInfer prefill and decode
attention, router top-k, NVFP4 experts and `lm_head`. Apart from the NVFP4-global issue reported separately, they
were all at output-rounding level (relative L2 1.6e-3 to 6e-3) or explained by bf16 intermediates that the
kernels use by design. Router top-k sets matched the reference on 100% of tokens.

## Root cause (file:line at `cac247a8`)

- `python/freetoken/models/qwen3_5_moe/weight.py:279-280`:

  ```python
  elif _is_gemma_norm(name):
      yield name, tensor + 1.0  # (1 + weight) baked into the stored norm weight
  ```

  The suffix list is at `weight.py:39-44`, and `_is_gemma_norm` at `weight.py:66-67`.
- `python/freetoken/layers/norm.py:29-76`: `GemmaRMSNorm` multiplies by the stored weight with plain
  `rmsnorm`/`fused_add_rmsnorm` (lines 63 and 75).
- Used at `models/qwen3_5_moe/model.py:59-60` (input/post-attention), `model.py:89` (final norm) and
  `models/qwen3_5_moe/attention.py:44-47` (q/k norm; the comment at 44-45 documents the fold).

## Measured impact

- Weight level: 77.4% of 171,008 multipliers are inexact, mean relative error 1.4e-3, max 3.9e-3. This is about
  3x the median NVFP4-global fp16 error from the separate report.
- Op level: see the table. 70-75% bf16 agreement today, 99.999-100% with the raw weight and a `(1 + w)`-in-fp32
  kernel.
- Model level: in an independent high-precision CPU reference of the quantised model, we switched only between
  exact `(1 + w)` and `bf16(1 + w)`. This moved the decisive top-2 logit margin at the first generated token by
  about **0.4 logit** on both prompts we examined: 0.48 to 0.89 on one, and from a 0.09 lead for one answer to a
  0.32 lead for the other on the second. The shift had opposite signs on the two prompts, so it acts as a weight
  perturbation, not a consistent bias. For comparison, the fp16-global error moved the same margins by about 0.1.
- **Not measured:** whole-model perplexity and eval impact of the fix. We did not apply it to our serving build,
  because it changes the numerics, and our release gates compare against a fixed teacher reference.

## Proposed fix

Stop folding. Keep the raw `w`, and use the `(1 + w)`-in-fp32 kernels that the existing dependencies already ship
(`sgl_kernel.gemma_rmsnorm` / `gemma_fused_add_rmsnorm`, with the Triton fallbacks in
`freetoken/kernel/triton/norm.py`). This is the smallest change that keeps the `forward_add_residual` call sites
unchanged. Full patch: `patches/gemma-norm-plus-one-fp32.upstream.diff` (5 files, applies to `cac247a8`).

```diff
--- a/python/freetoken/models/qwen3_5_moe/weight.py
+++ b/python/freetoken/models/qwen3_5_moe/weight.py
                 elif _is_gemma_norm(name):
-                    yield name, tensor + 1.0  # (1 + weight) baked into the stored norm weight
+                    yield name, tensor  # raw w: GemmaRMSNorm(plus_one=True) applies (1 + w) in fp32 in-kernel (HF semantics)
--- a/python/freetoken/layers/norm.py
+++ b/python/freetoken/layers/norm.py
-    def __init__(self, size: int, eps: float, with_scale: bool = True) -> None:
+    def __init__(self, size: int, eps: float, with_scale: bool = True, plus_one: bool = False) -> None:
         ...
             from sgl_kernel import fused_add_rmsnorm, rmsnorm
+            from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm
         else:
             from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm
+            from freetoken.kernel.triton.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
+        if plus_one:
+            assert with_scale
+            rmsnorm, fused_add_rmsnorm = gemma_rmsnorm, gemma_fused_add_rmsnorm
--- a/python/freetoken/models/qwen3_5_moe/model.py   (input/post-attention/final norm)
--- a/python/freetoken/models/qwen3_5_moe/attention.py   (q_norm/k_norm)
-        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
+        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps, plus_one=True)
--- a/tests/models/test_qwen3_5_moe_weight.py   (loader now yields the raw norm weights)
-    assert torch.equal(loaded["model.norm.weight"], raw[f"{LM}.norm.weight"] + 1.0)
+    assert torch.equal(loaded["model.norm.weight"], raw[f"{LM}.norm.weight"])
```

An equivalent alternative is to switch these modules to the existing `GemmaPlusOneRMSNorm` /
`GemmaPlusOneRMSNormFused` classes (flashinfer-backed, as `qwen4_exp` does). That needs small call-site changes,
because `GemmaPlusOneRMSNormFused.forward(x, residual)` replaces `forward_add_residual`. The linear-attention gated
norm (`linear_attn.norm`) is a plain `w * x` norm and is correctly left out of the suffix list.

## Verification we did

1. The weight-level and op-level numbers above (CPU audit of all 101 norm tensors, plus a GPU op audit with
   `sgl_kernel` on real activations from 6 layers and the final norm).
2. The high-precision CPU reference that produced the margin numbers was validated against the transformers
   Qwen3.5-MoE modules: RMSNorm, attention layer and GDN layer all at max |diff| 0.0.
3. The patch passes `git apply --check` on `cac247a8`. **Not run**: an end-to-end GPU run of the patched model, and
   PPL or eval with the fix.
