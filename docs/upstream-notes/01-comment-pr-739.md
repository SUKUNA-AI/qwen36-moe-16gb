<!-- READY, NOT POSTED. Target: comment on FlashML-org/FreeToken#739 (merged 2026-10-19, fixes the issue drafted in 01-freetoken-nvfp4-global-scale-fp16.md, which is therefore NOT to be posted as a new issue). Paste everything below this line. -->

Thanks for this fix. We had found the same fp16 cast downstream and carried an fp32 patch, so here are independent
measurements of its effect. They might be useful for the release notes or for anyone deciding whether the
breaking change is worth it. Setup: `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`, RTX 5070 Ti 16 GB (sm_120),
FreeToken 0.1.3 plus the equivalent change, offload MoE backend, batch 1.

- **Checkpoint:** 28,964 of the 30,841 `weight_scale_2` values (93.9%) are fp16 subnormals; only 1,434 survive the
  cast unchanged. Relative error of the cast: median 4.8e-4, p99 1.46e-3, max 1.66e-3.
- **Kernels** (Triton fused-MoE prefill/decode and the NVFP4 linear, real weights from 4 layers, M in
  {1, 17, 256, 2048, 4096}, against an fp64 reference): with fp32 globals, 99.97-99.99% of the bf16 outputs are
  correctly rounded; with fp16 globals, 88-94%.
- **Teacher forcing** (20 fixed prompts, 943 target tokens): perplexity **-1.13%** with fp32 globals, paired
  cluster-bootstrap 95% CI [-2.27%, -0.08%]. Other rounding-level changes we tested on the same set moved it by
  -0.37% to +0.07%.
- **Functional eval** (60 private cases, greedy): 59/60 with fp32 vs 57/60 with fp16. Both discordant cases favour
  fp32, but with two discordant pairs this alone is not significant.
- **Caveat for anyone comparing outputs before and after:** single greedy answers can flip in either direction. On
  5 smoke prompts, fp16 passed 5/5 and fp32 passed 3/5. Both "regressions" were first-token near-ties (top-2
  margins of 0.48 and 0.09 logit). An independent high-precision reference of the quantised model reproduces the
  **fp32** outputs byte for byte on both prompts.
- **Cost:** +2 bytes per row, about 63 MB of pinned host memory for 40 x 256 experts (about 0.35% fewer
  auto-sized expert slots on a 16 GB card).
