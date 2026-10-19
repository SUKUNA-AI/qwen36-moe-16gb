<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Bit-exact FP8 W8A8 skinny GEMV for decode (M <= 8) on sm_120: same QMMA chain as CUTLASS, 1.44-1.48x on the decode projection sequence

## Summary

At decode (M = 1..8), the fused row-wise FP8 input projections of Qwen3.6-35B-A3B go through `torch._scaled_mm`.
On sm_120 that is a CUTLASS kernel with a 128x128x128 tile, so 127 of 128 rows are padding at M = 1 and the
tensor cores sit mostly idle. Our fork adds an opt-in skinny GEMV that issues **the same
`QMMA.16832.F32.E4M3.E4M3` chain, with the same K order and the same epilogue**, so the output bits are
identical. On our setup:

- the decode projection sequence under CUDA graph runs 1.44-1.48x faster, about 1 ms per decode step;
- one GEMV runs 34.4 us at 92.9% DRAM vs 52.7 us at 59.5% for CUTLASS (ncu);
- 0 mismatches over 6,400 cases.

## Setup

- RTX 5070 Ti 16 GB (sm_120), WSL2; CUDA 13.4, GCC 15.3, built for `sm_120a`
- torch 2.11.0+cu130: the reference CUTLASS kernel is the one this torch picks
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`. The checkpoint stores attention and GDN projections in FP8 W8A8
  with static input scales. The fused projections have piecewise-constant per-row weight scales.
- FreeToken 0.1.3 `cac247a8` + our fork, CUDA-graph decode, batch 1

## Problem in upstream today

All paths are under `python/freetoken/`.

- `layers/quantization/linear/fp8_tensor.py:18` (`TorchFp8TensorLinearKernel`) calls `fp8_pertensor_linear`
  (`kernel/triton/fp8_pertensor_linear.py:314`). With `input_scale` on sm_89+, every M runs W8A8 (`:339`).
- A fused projection with a piecewise-constant `weight_scale` takes the row-wise call
  `torch._scaled_mm(qa, wt, scale_a=..., scale_b=weight_scale.reshape(1, -1))` (`:306-311`). This happens for
  any M, including decode M = 1.
- The Triton split-K GEMV at `:344` is only used for W8A16 (no input scale, or no native e4m3), so it has
  different numerics.
- The docstring at `:321-330` explains why the quantization scheme must not depend on M: so that bs = 1 replies
  reproduce. Our GEMV keeps the W8A8 scheme and its exact bits, so it respects that rule.
- These two files are unchanged between our base `cac247a8` and main `4d2e2e2`. We found no issue or PR
  proposing this. (#585 is a BF16 single-row GEMV, which is a different path.)

## What we changed

Switches: `QWEN36_FP8_GEMV=1` + `QWEN36_FP8_GEMV_LIBRARY=<absolute .so path>`, default off. With the flag unset,
the module is never imported.

Fork modules:

- `python/freetoken/kernel/cuda_fp8_gemv.py`: loader, self-test, dispatch;
- hooks in `python/freetoken/layers/quantization/linear/fp8_tensor.py` (`finalize`) and
  `python/freetoken/kernel/triton/fp8_pertensor_linear.py`;
- test `tests/kernels/test_cuda_fp8_gemv.py`.

Library sources are in our `kernels/fp8-gemv/` (`libqwen_fp8_gemv.so`, ABI 2).

- **Scope.** The GEMV replaces only the row-wise `torch._scaled_mm` call, only for GDN `in_proj_qkvz`
  [12288, 2048] and attention `qkv_proj` [9216, 2048], and only for 1 <= M <= 8. Everything else keeps
  `torch._scaled_mm`: uniform-scale projections, the per-part segment path, prefill-sized M, and any input the
  module does not recognise exactly.
- **Arithmetic.** For each output it uses the same QMMA chain, starting from +0 over ascending k-blocks, with no
  split-K. The epilogue is FMUL -> FFMA(+0) -> F2FP RN, as in the CUTLASS kernel.
- **Numerics gate**, decided once at model load in `finalize` and never under CUDA-graph capture:
  - a platform pin: sm_120 and torch 2.11.x;
  - a process self-test over both shapes x M 1..8, with random full-range e4m3 codes plus edge codes (+-0,
    +-448, subnormals, exact cancellation) and wide row scales, compared bitwise with the production
    `torch._scaled_mm` call;
  - a per-layer check with the real weights.

  Any mismatch disables only that M, and CUTLASS stays authoritative.
- **Provenance.** The `.so` must match a pinned sha256 and is loaded from a sealed memfd. An ABI probe checks
  the version, max M and the compiled-in source hash. A provenance failure is a load error; it never falls
  back silently.
- **Runtime.** The GEMV launches on the current stream, with no host sync and no branching on tensor values.
  Its output comes from the caching allocator, so it is graph-capturable.

## Measured impact

Microbenchmarks (`kernels/README.md`, `docs/ARCHITECTURE.md` §3.4):

- The decode projection sequence under CUDA graph runs 1.44-1.48x faster, about 1 ms per decode step
  (`kernels/fp8-gemv/scripts/step4_speed.py`, 5 alternating pairs, no profiler). The architecture summary
  gives 1.45-1.47x for the 80-projection decode sequence. The median/mean split is not in the published
  summary.
- The 39 CUTLASS kernels per step are replaced 1:1.
- One GEMV moves 708 GB/s vs 409 GB/s for CUTLASS. With ncu: GEMV 34.4 us at 92.9% DRAM vs CUTLASS 52.7 us at
  59.5%.
- Component ncu runs (`results/nsight/levers.json`): 12288x2048 takes 35.264 us at 90.36% DRAM (808 GB/s), and
  2048x4096 takes 13.568 us at 70.76% (632 GB/s).

Model level: we have **no isolated model-level A/B** for this kernel alone. Together with our decode expert
gather change, decode was +13 to +17% at 16K / 64K / 120K (`kernels/README.md`).

## Correctness evidence

- **Our runs**: 0 mismatches over 6,400 cases (`scripts/step3_bitwise.py`).
  - That is 117.2M elements eager and 78.2M under CUDA graph, with real weights, all 80 dense FP8
    projections, M 1..8, and normal, heavy-tailed and edge activations.
  - All 11 fallback scenarios pass: env unset, platform pin, tampered, missing or stale library, self-test
    launch error and mismatch, runtime launch error, real-weight mismatch, capture guard, threads.
  - The in-tree test passes 22/22.
- **Independent review** (separate harness): 0 mismatches over 195.4M elements x 3 comparisons eager and 97.7M
  under graph capture. Also 0 mismatches for extreme synthetic scales, non-blocking side streams, two
  concurrent streams and non-contiguous inputs.
- **Model level**: teacher forcing 125/125, max_abs 0 (release f1 gate B, `results/metrics/metrics.json`).

## Limitations / not measured

- sm_120a only. Two projection shapes of one model, M <= 8.
- Bit-exactness is tied to the CUTLASS kernel that torch 2.11.x selects. A different torch or CUTLASS version
  can change that kernel; the self-test then disables the GEMV, and porting needs a new identification of the
  kernel. We have not checked which torch main currently targets.
- The model-level decode gain of this kernel alone is not measured; only the combined figure exists.
- The decode out/o-projection (cuBLASLt split-K) is a separate kernel in our fork. It is not part of this issue.
- The library is a prebuilt `.so` pinned by sha256. nvcc output is not byte-reproducible.

## Proposal

We could contribute the kernel and loader behind a default-off flag, with the bitwise self-test kept as a hard
gate.

Questions for maintainers:

1. Would you take a platform-pinned replica of a torch/CUTLASS kernel at all, given that it must be
   re-identified on torch upgrades?
2. Would you prefer a generic skinny W8A8 GEMV with its own (non-bit-identical) numerics, selected for the
   whole deployment rather than per M, as your docstring requires?
3. Prebuilt library pinned by hash, or JIT under `kernel/csrc/jit`?
