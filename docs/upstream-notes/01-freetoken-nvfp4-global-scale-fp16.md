<!-- DRAFT ONLY, NOT POSTED. Local file; the owner decides the repository and whether to post. Suggested target: FlashML-org/FreeToken (issue). -->

# NVFP4: per-tensor global scale (`weight_scale_2`) is stored as fp16, so ~94% of ModelOpt globals become fp16 subnormals

## Summary

ModelOpt NVFP4 checkpoints store the per-tensor global scale `weight_scale_2` as **fp32**. FreeToken casts it to
**fp16** at load time, in the dense reader and in the routed-expert bank builder. In
`nvidia/Qwen3.6-35B-A3B-NVFP4` the globals lie between 1.6e-5 and 3.1e-4. The smallest normal fp16 value is
6.1e-5, so **28,964 of 30,841 globals (93.9%) become fp16 subnormals** and lose mantissa bits. Only 1,434 survive
the cast unchanged.

The effective weight `W = e2m1 * e4m3_block_scale * global` is therefore scaled by a constant `(1 + d)` per tensor.
The relative error `d` has a median of 4.8e-4 and a maximum of 1.66e-3. This error is small next to the NVFP4
quantisation noise, but it is systematic: every NVFP4 projection output (routed experts, shared expert, `lm_head`)
carries the same per-tensor bias. All kernels already apply the global to an fp32 accumulator, so only the
**storage dtype** loses the precision.

Keeping the globals in fp32 improved teacher-forced perplexity by **1.13%** (95% CI [-2.27%, -0.08%]). It cost
about 63 MB of pinned host memory for a 256-expert x 40-layer model.

## Environment

- FreeToken 0.1.3, commit `cac247a860e316e06580d05aeb05f2e647bde214`
- Checkpoint `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5` (ModelOpt mixed precision:
  NVFP4 W4A16 for the routed experts, the shared expert and `lm_head`; FP8 for the attention and GDN projections)
- RTX 5070 Ti 16 GB (sm_120), Linux (WSL2), torch 2.11.0+cu130, triton 3.6.0, flashinfer-python 0.6.18.post1,
  sgl-kernel 0.4.5, transformers 5.16.1, Python 3.12
- `ft serve ... --moe-strategy offload --moe-cache-auto --cache-type radix --max-running-requests 1 --text-model-only`
  (the cast is independent of these flags: it happens in the loaders)

## Minimal reproduction (CPU only)

1. Statistics of the cast on the checkpoint (needs only `safetensors` and `torch`):

```python
import json, pathlib, torch
from safetensors import safe_open

MODEL = pathlib.Path("<local snapshot of nvidia/Qwen3.6-35B-A3B-NVFP4>")
idx = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
names = sorted(n for n in idx if n.endswith(".weight_scale_2") and not n.startswith("mtp."))
files = {f: safe_open(str(MODEL / f), framework="pt") for f in set(idx[n] for n in names)}
g = torch.stack([files[idx[n]].get_tensor(n).reshape(()) for n in names]).double()   # stored as fp32
g16 = g.float().half().double()                                                       # what FreeToken keeps
rel = (g16 - g).abs() / g
tiny = torch.finfo(torch.float16).tiny
print(len(names), float(g.min()), float(g.max()))
print("fp16 subnormal:", int((g < tiny).sum()), "exact:", int((rel == 0).sum()))
print("rel err median/p99/max: %.3g %.3g %.3g" % (float(rel.median()), float(rel.quantile(0.99)), float(rel.max())))
```

Output on this checkpoint:

```
30841 1.6439529645140283e-05 0.0003138950851280242
fp16 subnormal: 28964 exact: 1434
rel err median/p99/max: 0.000481 0.00146 0.00166
```

By group: the 30,720 routed-expert globals have median error 4.8e-4 (max 1.66e-3) and 28,954 of them are
subnormal. The 120 shared-expert globals have median 1.9e-4 (max 4.2e-4), with 10 subnormal. The single `lm_head`
global has error 1.8e-4.

2. FreeToken keeps the fp16 value. The routed-expert path is `nvfp4_banks._ingest_global` and then
   `moe/base.global_rows`, which store into banks declared fp16 by `TritonNvfp4MoEKernel.layout()`. The dense path
   (shared expert, `lm_head`) is `_DenseReader._check`, which writes into `Nvfp4LinearMethod.weight_global` (fp16).
   Loading any expert through `nvfp4_expert_spec -> iter_nvfp4_expert_pieces -> TritonNvfp4MoEKernel.pack` shows
   `gate_up_global.dtype == torch.float16`, and the values equal `g.half()`, not `g`.

## Root cause (file:line at `cac247a8`)

| Location | Code |
|---|---|
| `python/freetoken/models/qwen3_5_moe/weight.py:230` (`_DenseReader._check`) | `out["weight_global"] = g.to(torch.float16).expand(rows).contiguous()` |
| `python/freetoken/layers/quantization/linear/nvfp4.py:122` (`Nvfp4LinearMethod.create_weights`) | `layer.weight_global = torch.empty(g.out_features, dtype=torch.float16)` |
| `python/freetoken/models/nvfp4_banks.py:36-39` (`_ingest_global`) | `return tensor.to(torch.float16)` |
| `python/freetoken/layers/quantization/moe/base.py:94-96` (`global_rows`) | `flat = piece.reshape(piece.shape[0], -1).to(torch.float16)` |
| `python/freetoken/layers/quantization/moe/nvfp4.py:55,58` (bank layout) | `"gate_up_global": BankSpec((2 * i,), torch.float16)`, `"down_global": BankSpec((h,), torch.float16)` |
| same pattern, other loaders | `models/gemma4/weight.py:72`, `models/glm4_moe/weight.py:86`, `models/loader.py:215` (compressed-tensors `1 / weight_global_scale`) |

The consumers already work in fp32 after the load. Examples are
`kernel/triton/nvfp4_fused_moe.py:122,219,314`, `kernel/triton/nvfp4_linear.py:220,305,472,519`,
`kernel/triton/nvfp4_dequant.py:52` (`tl.load(...).to(tl.float32)`) and
`kernel/csrc/cpu_moe/cpu_moe_ext.cpp:1500-1524` (`fp16_to_f32(gu_global_l[r])` / `fp16_to_f32(dn_global_l[r])`).
The comment at `nvfp4_dequant.py:50-51` says the global is kept separate so that it does not underflow. The
fp16 storage reintroduces that underflow for real checkpoints, because their globals are around 1e-4.

## Measured impact

We changed only the global storage dtype and kept everything else bit-identical. The fp16 arm reproduces the old
numerics exactly: its banks are bitwise equal to `fp32(fp16(g))`.

- **Kernel level** (GPU): we compared the Triton fused-MoE prefill and decode kernels and the Triton NVFP4 linear
  (shared expert) with an FP64 reference that uses the checkpoint's fp32 globals. The test used real weights from
  layers 0/13/26/39 with M in {1, 17, 256, 2048, 4096}. With fp32 globals, 99.97-99.99% of the bf16 outputs equal
  the correctly rounded reference; with fp16 globals, 88-94%. The MoE layer output moves by relative L2
  2.2e-3 to 4.6e-3 between the two.
- **Teacher forcing** (20 fixed prompts, 943 target tokens, greedy targets): the relative perplexity change from
  fp16 to fp32 globals is **-1.13%**, with a paired cluster bootstrap 95% CI of **[-2.27%, -0.08%]**. Logits on
  the same set show max |diff| 2.68, mean relative L2 0.076 and top-1 agreement 0.976. For comparison, other
  rounding-level changes we tested on the same set gave between -0.37% and +0.07%: CUDA vs Triton MoE GEMM,
  prefill chunk sizes, and a 128K vs 256K pool.
- **Pre-registered functional eval** (60 private cases: code with sandboxed tests, Russian, tool calling, agent
  history, long documents of 16K/64K/120K; greedy): **59/60 with fp32 globals vs 57/60 with fp16 globals**. Both
  discordant cases favour fp32, and fp32 lost no case. With only two discordant pairs this part is not significant
  on its own (exact McNemar p = 0.5).
- **Caveat: single greedy answers can flip either way.** On 5 separate fixed smoke prompts, fp16 passed 5/5 and
  fp32 passed 3/5. We investigated both regressions. An independent high-precision CPU reference of the quantised
  model (fp32 activations, exactly dequantised weights, the checkpoint's FP8 W8A8 / NVFP4 W4A16 semantics,
  validated against the transformers Qwen3.5-MoE modules) reproduces the **fp32-global outputs byte for byte** on
  both prompts. Both are decided at the first generated token by near-ties, with top-2 logit margins of 0.48 and
  0.09 in the exact model. The fp16 "passes" were near-tie flips. Under T=0.6 sampling, fp16 is not better
  (code 9/40 vs 14/40 for fp32, retrieval 13/40 vs 15/40).
- **Cost**: each expert gains 6 KB (3,072 rows x 2 extra bytes), which is about 63 MB of pinned host memory for
  40 x 256 experts. With `--moe-cache-auto`, that is about 0.35% fewer GPU expert slots.

## Proposed fix

Keep the global in fp32 from the reader to the kernel. The Triton kernels need no code change, because they
already `tl.load(...).to(tl.float32)` the global. Only the pointer dtype changes, and Triton specialises on it
automatically. The C++ CPU-MoE path must read `const float*`.

Excerpt (full patch: `patches/nvfp4-global-fp32.upstream.diff`, 14 files, applies to `cac247a8`):

```diff
--- a/python/freetoken/models/nvfp4_banks.py
+++ b/python/freetoken/models/nvfp4_banks.py
@@ def _ingest_global(spec: "Nvfp4ExpertSourceSpec", tensor: torch.Tensor) -> torch.Tensor:
     if spec.global_reciprocal:
         tensor = 1.0 / tensor.float()
-    return tensor.to(torch.float16)
+    # fp32, exactly as stored: in fp16 most real globals (~1e-5..3e-4) are subnormal
+    return tensor.to(torch.float32)
--- a/python/freetoken/layers/quantization/moe/base.py
+++ b/python/freetoken/layers/quantization/moe/base.py
-    flat = piece.reshape(piece.shape[0], -1).to(torch.float16)
+    flat = piece.reshape(piece.shape[0], -1).to(torch.float32)
--- a/python/freetoken/layers/quantization/moe/nvfp4.py
+++ b/python/freetoken/layers/quantization/moe/nvfp4.py
-            "gate_up_global": BankSpec((2 * i,), torch.float16),
+            "gate_up_global": BankSpec((2 * i,), torch.float32),
-            "down_global": BankSpec((h,), torch.float16),
+            "down_global": BankSpec((h,), torch.float32),
--- a/python/freetoken/models/qwen3_5_moe/weight.py
+++ b/python/freetoken/models/qwen3_5_moe/weight.py
-            out["weight_global"] = g.to(torch.float16).expand(rows).contiguous()
+            out["weight_global"] = g.expand(rows).contiguous()
--- a/python/freetoken/layers/quantization/linear/nvfp4.py
+++ b/python/freetoken/layers/quantization/linear/nvfp4.py
-        layer.weight_global = torch.empty(g.out_features, dtype=torch.float16)
+        layer.weight_global = torch.empty(g.out_features, dtype=torch.float32)
--- a/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
+++ b/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
-                         const uint8_t* gu_scale_l, const uint16_t* gu_global_l, int e, int row,
+                         const uint8_t* gu_scale_l, const float* gu_global_l, int e, int row,
-                     fp16_to_f32(gu_global_l[r]), xi8, H, e4m3_lut, xas);
+                     gu_global_l[r], xi8, H, e4m3_lut, xas);
   (same for down / dn_global_l, plus an `nvfp4_global_bits() == 32` ABI marker that
    CpuMoeExecutor probes, mirroring the existing `max_generic_act_id` probe)
```

The same one-line change applies to the gemma4, glm4_moe and compressed-tensors (`models/loader.py`) NVFP4
loaders. Tests that hard-code fp16 globals need updating: `tests/models/test_qwen3_5_moe_weight.py:447,515,519,562`,
`tests/moe/test_nvfp4_backends.py:64,66,481,484` and `tests/moe/test_offload.py:571,574`. `moe/benchbw.py:347-353`
also builds fp16 globals.

If fp16 storage has to stay, an alternative is to store `g * 2**16` in fp16 and multiply by `2**-16` in the
epilogue. That keeps the values out of the subnormal range (error at most ~4.9e-4), but it is still not exact.

Not covered: the Marlin NVFP4 paths (`linear/nvfp4.py:64`, `moe/nvfp4.py:164-172`) hand the global to vLLM's
Marlin helpers as bf16. That is a separate constraint of those helpers, and we did not evaluate it.

## Verification we did (on a downstream build of `cac247a8` with the equivalent change)

1. **CPU, through FreeToken's own loader functions** (`iter_weights`/`_DenseReader`,
   `nvfp4_expert_spec -> iter_nvfp4_expert_pieces -> TritonNvfp4MoEKernel.pack`): with the fix, the globals are
   bitwise equal to the checkpoint's fp32 values for all 10,240 experts x {gate, up, down} and all 80 shared-expert
   modules. The dequantised weights are bitwise equal to an FP64 reference for 280 experts and the 80 shared
   modules.
2. **GPU kernels**: the bf16 match rates against FP64 listed above. The CPU-MoE extension was rebuilt and the probe
   was checked.
3. **Whole model**: teacher-forced logits and perplexity, the 60-case functional eval, a seed study (10 seeds greedy
   plus 40 seeds T=0.6 per prompt) and the high-precision reference, with the numbers above.
4. The attached upstream patch is a port of that change to the clean base. It passes `git apply --check` on
   `cac247a8`. The port itself has **not** been re-run on an unmodified checkout.
