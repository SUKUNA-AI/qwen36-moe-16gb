<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

> **Upstream status: partial overlap.** Main gained a faster sm_120 NVFP4 MoE prefill GEMM on 2026-10-17 (#733,
> Gluon, needs Triton 3.8). PR #733 reports at most 1 bf16 ulp difference from the previous kernel on sm_120.
> Our kernel is bit-identical to the pre-#733 Triton kernel. We have **not** measured it against #733.

# [Perf] Bit-exact CUDA NVFP4 MoE prefill kernel for sm_120 (7.25x per layer vs the Triton W4A16 prefill at M=2048, on our setup)

## Summary

The FreeToken NVFP4 W4A16 MoE prefill (gate/up GEMM, SiLU-and-mul, down GEMM with router weights, sum reduce)
was the main compute cost of long prefills on our sm_120 card. We wrote a hand-written sm_120a CUDA kernel for
this path. It uses inline-PTX `mma.sync.m16n8k16` (BF16), `ldmatrix` and `cp.async`, with exact E2M1/E4M3
dequant. It reproduces the Triton 3.6 per-MMA K order and FlashInfer's SiLU instruction sequence, so its output
is **bit-identical** to the Triton path. Measured on our setup:

- one MoE layer at M = 2048 drops from 12.82 ms to 1.77 ms (7.25x);
- model prefill time is 1.7-1.9x faster at 16K-120K;
- teacher forcing gives 125/125 with max_abs 0.

The current version was written before #733. This issue asks whether a bit-exact path of this kind is still
of interest.

## Setup

- RTX 5070 Ti 16 GB (sm_120), WSL2; CUDA 13.4 (nvcc V13.4.59), GCC 15.3, no `--use_fast_math`
- torch 2.11.0+cu130, triton 3.6.0, flashinfer-python 0.6.18.post1
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`: H 2048, I 512, 256 experts, top-8
- FreeToken 0.1.3 `cac247a8` + our fork, `--moe-strategy offload`, prefill chunk 2048

## Problem in upstream today

- The prefill dispatch is at `python/freetoken/layers/quantization/moe/nvfp4.py:76-77`, which goes to
  `fused_experts_nvfp4` (`moe/fused_nvfp4.py:111`). That function launches 4 kernels:
  - gate/up `nvfp4_moe.prefill_gemm` (`fused_nvfp4.py:142`);
  - `gated_act_and_mul` (`:148`);
  - down `prefill_gemm` (`:150`);
  - `moe_sum_reduce_triton` (`:156`).
- On main, `kernel/nvfp4_moe.py:337-338` routes `prefill_gemm` to the Gluon kernel
  `kernel/gluon/nvfp4/moe/sm120.py`. This happens when the GPU is sm_120, Gluon is available (Triton 3.8,
  `FREETOKEN_DISABLE_GLUON` unset; `kernel/gluon/__init__.py:22-33`) and `BLOCK_SIZE_M >= 64`. Otherwise
  the Triton kernel (`kernel/triton/nvfp4_fused_moe.py:356`) runs.
- PR #733 reports the following for Qwen3.6-35B-A3B at T = 8192 on an RTX PRO 6000:
  - 1.41x faster than the previous main;
  - output on sm_120 within 1 bf16 ulp of the previous main ("norm rel 2.5e-4"), so not bitwise equal.
- An opt-in flashinfer b12x W4A16 backend (`--quant-backend moe.nvfp4=b12x`) existed already at our base. We
  did not evaluate it.

## What we changed

Switches: `QWEN36_NVFP4_PREFILL=cuda` + `QWEN36_CUDA_NVFP4_LIBRARY=<.so>`, default off. The fork module is
`python/freetoken/moe/cuda_nvfp4.py`, dispatched from `layers/quantization/moe/nvfp4.py` for prefill chunks
with M >= 512. Smaller M keeps the Triton path. Library sources, build script and tests are in our
`kernels/nvfp4-moe/` (`libqwen_nvfp4_bitexact.so` v4).

- **Scope.** One library replaces gate/up GEMM + SiLU, then down GEMM + router-weighted sum:
  - E2M1/E4M3 dequant done exactly as in the Triton path;
  - FlashInfer `silu_and_mul` SASS replicated;
  - the same sum order as `moe_sum_reduce`.
- **Bit-exactness needs the Triton accumulation order.** Triton 3.6 lowers these variants with
  `dot_op kWidth=4`. That packs K residues mod 8 into each MMA as {0,2}/{4,6}/{1,3}/{5,7} and chains
  even/kRep0 -> even/kRep1 -> odd/kRep0 -> odd/kRep1 per K64. The CUDA kernel reproduces this order.
- **Guards.**
  - The loader requires `qwen_nvfp4_global_bits() == 32`: the kernel reads fp32 NVFP4 global scales (see
    our separate issue on fp16 `weight_scale_2`).
  - The ABI probe runs in a subprocess with `CUDA_VISIBLE_DEVICES=''`.
  - The prototype only accepts the Qwen H2048 / I512 / top-8 geometry and raises on anything else.
- **Why bit-exact.** Earlier WMMA and "matched" variants were about 99.9% bit-equal and failed our 16K
  functional gate (`kernels/README.md`, "Experimental, not shipped").

## Measured impact

- **Layer microbenchmark** at M = 2048: 12.82 ms -> 1.77 ms (7.25x) vs the Triton prefill of our base.
  - Bench: `kernels/nvfp4-moe/py/bench.py`, 5 paired layer timings, no profiler. The median/mean split is
    not in the published summary.
  - Sources: `docs/ARCHITECTURE.md` §3.2, `kernels/README.md`.
- **Model level**: prefill time is 1.7-1.9x faster at 16K-120K, and host-to-device expert streaming is
  unaffected (`kernels/README.md`). The number of pairs for this A/B is not documented in our published files.
- **Release level**: the kernel is one of several changes in our release f1, so the f1-vs-original TTFT
  numbers are not attributable to it alone (see our layer-major issue).
- **Diagnostics** (Nsight Compute, component run; `results/nsight/levers.json`):
  - gate_up runs 1075.2 us at 82.93% SM throughput with HMMA at 41.15% of peak; it is compute-bound, and the
    top stall is `math_pipe_throttle`;
  - the prefill MoE GEMM is 36% of a 16K and 17% of a 120K prefill (Nsight Systems).

## Correctness evidence

All checks are raw-bit comparisons with Triton (`kernels/README.md`):

- **v4 (released, fp32 globals)**: 0 mismatches.
  - synthetic: 657 comparisons, 6.89e9 elements;
  - real checkpoint weights (layers 0/13/26/39, real router, M in {1, 17, 256, 2048, 4096}): 860
    comparisons, 9.78e9 elements.
- **v3 (same arithmetic, fp16 globals)**: 0 mismatches.
  - real weights: 1,376 + 608 comparisons, about 1.4e10 elements;
  - synthetic: 2,218 comparisons, 1.5e10 elements;
  - dequant and SiLU checked exhaustively.
- **Negative control**: the library's oracle with the old K layouts reproduces mismatches at about 1e-4.
- **Model level**: teacher forcing 125/125, max_abs 0 against the fp32-global reference.

## Limitations / not measured

- sm_120a only. Only Qwen3.6-35B-A3B geometry (H 2048, I 512, top-8) and only M >= 512.
- Bit-exact against the **pre-#733** Triton 3.6 kernel. Against main's Gluon path the output may differ by up
  to the 1 ulp that #733 reports; we have not checked this.
- Speed vs #733 Gluon, vs b12x and vs main's updated Triton kernel: **not measured**. Our environment uses
  Triton 3.6, where main's Gluon route is unavailable.
- The kernel is compute-bound on the BF16 HMMA pipe because the bit-exact dequant + HMMA order is fixed. An
  FP4/FP8 MMA path would be faster but would need a new quality gate.
- nvcc output is not byte-reproducible (module id). Our loader pins the `.so` by sha256, so a rebuild needs a
  new pin and a rerun of the bit-exact suites.

## Proposal

We can contribute the kernel source (`.cu`, build script, bit-exact and negative-control tests) behind a
default-off flag, or just share the K-order findings.

Questions for maintainers:

1. With #733 merged, is a CUDA prefill kernel still wanted, or would you rather fold the findings into the
   Gluon kernel?
2. Do you want prefill to stay bit-identical across kernel backends, or is about 1 ulp acceptable?
3. If you want it, should it ship as a prebuilt library pinned by hash (as in our fork) or as a
   JIT/tvm-ffi module like the existing `kernel/csrc/jit` kernels?
4. We would first rerun our suites against main's Triton 3.8 kernels; is that a reasonable precondition?
