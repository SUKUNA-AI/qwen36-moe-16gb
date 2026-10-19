<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] sm_120 decode: bit-exact CUDA GEMVs for the FP8 o-projection (replaces cuBLASLt split-K) and the NVFP4 MoE decode GEMV, with a load-time bit-exact self-test

## Summary

On an RTX 5070 Ti we replaced two decode GEMMs (1 <= M <= 8) with CUDA kernels that reproduce the existing
kernels' arithmetic **bit for bit**:

- the tensor-wise FP8 W8A8 out/o-projection (`torch._scaled_mm`, which cuBLASLt runs as a 384-byte semaphore
  memset plus a split-K = 3 xmma kernel);
- the Triton NVFP4 MoE decode GEMV (`_decode_nvfp4_marlin_kernel`, gate_up and down).

Each library runs a bitwise self-test against the production call at model load. Any M (or layer) that does not
match keeps the old kernel, so a mismatch can only cost speed. Together the two raised decode at a 16K prompt from
159.5 to 170.3 tok/s (median of 3 pairs), with teacher-forced logits byte-identical to the reference.

## Setup

- FreeToken 0.1.3 `cac247a860e316e06580d05aeb05f2e647bde214` plus our fork; `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`
- RTX 5070 Ti 16 GB (sm_120), WSL2, torch 2.11.0+cu130, Triton 3.6.0, nvcc 13.4.59
- Offloaded MoE (host-memory expert bank, GPU slot cache), `--max-running-requests 1`

## Problem in upstream today

- **o-projection.** `python/freetoken/kernel/triton/fp8_pertensor_linear.py:333` selects W8A8 whenever the checkpoint
  has `input_scale` on sm_89+. A uniform-scale weight then goes to `torch._scaled_mm(qa, wt, scale_a=sa, scale_b=...)`
  at `:294-297`. For Qwen3.6 that covers 40 projections per decode step (30 GDN `out_proj`, 10 attention `o_proj`;
  N 2048, K 4096). On torch 2.11 / cuBLASLt 13 every one of them is a 384-byte memset plus
  `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warpsize2x2x1_tensor16x8x32`, grid (1, 32, 3),
  split-K 3 with an in-kernel semaphore reduction. Our release trace shows 0.67 us of memset plus 13.8-15.5 us of xmma
  per projection. Neither file has changed since 0.1.3.
- **MoE decode.** `python/freetoken/layers/quantization/moe/nvfp4.py:78` still sends every decode step to
  `fused_experts_decode_nvfp4_marlin`, then to `kernel/nvfp4_moe.py:96` (`decode_gemm_marlin`) and the Triton kernel at
  `kernel/triton/nvfp4_fused_moe.py:162`. **That kernel was rewritten after 0.1.3** by #631 (7644984: fp16-trick e2m1
  decode with 2^14 folded into the scale, a transposed partial tile, an `UNROLL` loop, int64 activation loads). Our
  baseline and our bit-exact target are the 0.1.3 kernel. We have not measured #631's speed, and our library is
  **not** bit-exact against it.
- main now pins torch 2.14 / Triton 3.8 (`pyproject.toml`). Our libraries pin torch 2.11 and Triton 3.6 and disable
  themselves on anything else. Whether cuBLASLt on torch 2.14 picks the same split-K kernel is not checked.

## What we changed

- **o-projection:** `QWEN36_OPROJ_GEMV=1` + `QWEN36_OPROJ_GEMV_LIBRARY=<.so>`; module
  `freetoken/python/freetoken/kernel/cuda_oproj_gemv.py`, sources `kernels/decode-oproj/`.
  - We identified the cuBLASLt arithmetic with probe GEMMs and SASS. K is cut into 64-byte k-tiles, dealt round
    robin to 3 splits (tile t goes to split t mod 3). Each split is one QMMA.16832 chain from +0. The parts are
    combined as `(p0 + p2) + p1`. The epilogue is `FFMA(acc, RN(scale_a*scale_b), +0)` and then RN to bf16.
  - One launch replaces memset and xmma, with no workspace and no semaphore. All call sites go through one helper
    (`uniform_scaled_mm`); the per-layer flag is set at finalize in `layers/quantization/linear/fp8_tensor.py`.
- **MoE decode:** `QWEN36_MOE_DECODE_CUDA=1` + `QWEN36_MOE_DECODE_LIBRARY=<.so>`; module
  `freetoken/python/freetoken/kernel/cuda_moe_decode.py`, wired only into the decode branch of
  `layers/quantization/moe/nvfp4.py`; sources `kernels/decode-moe/`.
  - The 0.1.3 Triton kernel's order comes from its TTGIR/PTX/SASS: 16 per-word partial sums, an fma with the e4m3
    block scale per word, a fixed butterfly reduction, then the global (and the router weight for down) before bf16
    rounding. The CUDA kernel keeps this order. Only data movement changes: 16-byte weight loads, a register prefetch
    ring, and 4 threads per weight row.
- **Load gate.** Both libraries are loaded through a sealed memfd only if their sha256 matches the pinned value, with
  an ABI probe. Each library also checks the platform pin (sm_120, torch 2.11, plus Triton 3.6 for MoE).
  - Self-test inputs cover every M in 1..8. o-proj: uniform, gaussian and edge codes with exact cancellation, -0
    activations, scales that overflow bf16 or give subnormal results. MoE: every e2m1 code, every finite e4m3 scale
    code, globals 2^-40..2^30, and +-0, subnormal and +-3e38 activations. Results are compared bitwise with the
    production call.
  - Real-weight checks: o-proj on each layer's weights at finalize; MoE on each layer's first eager decode call,
    before graph capture.
  - A mismatch keeps that M or layer on the old kernel and is logged. Nothing runs under capture. Neither kernel
    syncs with the host.

## Measured impact

Microbenchmark (CUDA graph, 40 projections or layers over distinct weights, median of 5 alternating pairs;
`kernels/decode-oproj/README.md`, `kernels/decode-moe/README.md`):

| | old | new |
|---|---:|---:|
| o-proj, us/projection, M = 1 / 2 / 4 / 8 | 17.4 / 17.4 / 17.5 / 17.5 | 12.1-12.5 / 12.2-12.3 / 12.5-12.6 / 12.7-12.9 |
| MoE, us/layer, M = 1: gate_up / down / both | 18.5 / 9.4 / 28.4 | 12.5-12.9 / 7.4-7.8 / 20.5 |
| MoE, us/layer, M = 4: both | 86.1 | 41.4 |

That is about 0.2 ms (o-proj) plus 0.32 ms (MoE) per decode step at M = 1. Nsight Compute, one eager launch
(`results/ladder/ncu-kernels.json`): o-proj 16.4 -> 14.8 us at 58.1 -> 65.5% of peak DRAM; MoE 23.0 -> 17.8 us at
47.6 -> 64.5% of peak DRAM.

Model A/B (3 rounds, alternating order, fresh server per arm, greedy with `ignore_eos`, 256 output tokens; medians):

| Arm | decode tok/s, 64-token prompt | decode tok/s, 16K prompt |
|---|---:|---:|
| base | 158.4 | 148.1 |
| + o-proj | 161.8 (+2.1%) | 150.7 (+1.7%) |
| + MoE | 166.2 (+4.9%) | 154.9 (+4.6%) |
| + both | 172.1 (+8.6%) | 159.4 (+7.6%) |

Release ladder step (`results/ladder/summary.json`, step L6; 3 pairs, 128 output tokens): decode at 16K
159.5 (mean 158.9) -> 170.3 (mean 169.2) tok/s; the per-pair values are 156.6 / 160.7 / 159.5 -> 170.3 / 165.0 / 172.3.
At 120K: 112.1 (111.7) -> 118.4 (117.3). With n = 3, p95/p99 are not reported. Nsight Systems decode step at 16K:
6.42 -> 5.87 ms (MoE 1.45 -> 1.26 ms; dense layers incl. o-proj 2.59 -> 2.46 ms).

## Correctness evidence

- o-proj: 832 cases x 6 library variants, 0 mismatches (96 synthetic cases, plus all 40 real layers x M 1..8 x
  two activation distributions), in eager mode and inside a CUDA graph.
- MoE: 4284 GEMM comparisons, 0 mismatches (synthetic banks and real experts of 6 layers; M 1..8; top_k 8, plus top_k
  4/2/1); 21 full MoE-decode cases byte-equal.
- Model: teacher-forced gate 125/125 logit checkpoints at max_abs 0 (see our teacher-forcing proposal); generated
  text byte-identical across all 12 A/B server runs; no self-test mismatch was logged in any run.

## Limitations / not measured

- Shape- and platform-specific: o-proj handles only N 2048 x K 4096, tensor-wise scale, sm_120a, torch 2.11 /
  cuBLASLt 13. MoE matches only the 0.1.3 Triton kernel on Triton 3.6. Other GPUs, other shapes, M > 8 and the
  #631 kernel are not measured.
- The self-tests leave about 14 MiB of scratch in the caching allocator. Our budget counts it as weights, so 4611
  expert slots became 4603.
- Bit-exactness is against a specific cuBLASLt algorithm choice. If cuBLASLt picks another algorithm, the self-test
  disables that M and the win is lost.

## Proposal

We can contribute the two kernels as source (no prebuilt `.so`), together with the self-test/fallback harness, which
works for any "drop-in kernel must match the incumbent bit for bit" case. Open questions:

1. Should FreeToken want bit-exact replacements at all, or would a faster kernel with its own tolerance-tested
   numerics be preferred (e.g. retargeting the MoE GEMV to #631's arithmetic)?
2. For the o-projection, would a non-split-K GEMV under the existing W8A8 contract be acceptable instead of
   reproducing cuBLASLt's order?
3. Should the libraries be built through the prebuilt kernel cache instead of ctypes `.so` files?

**Our implementation, for reference** (fork of FreeToken 0.1.3, Apache-2.0, pinned commit; feel free to reuse):

- [`kernel/cuda_oproj_gemv.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/kernel/cuda_oproj_gemv.py)
- [`layers/quantization/linear/fp8_tensor.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/layers/quantization/linear/fp8_tensor.py)
- [`kernel/cuda_moe_decode.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/kernel/cuda_moe_decode.py)
- [`layers/quantization/moe/nvfp4.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/layers/quantization/moe/nvfp4.py)
- [`kernels/decode-oproj/README.md`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/kernels/decode-oproj/README.md)
- [`kernels/decode-moe/README.md`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/kernels/decode-moe/README.md)
- [`kernels/decode-oproj`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/tree/745b28a34fc5b68439a34cc7efbefa55209c660f/kernels/decode-oproj)
- [`kernels/decode-moe`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/tree/745b28a34fc5b68439a34cc7efbefa55209c660f/kernels/decode-moe)

Evidence and write-ups (`docs/RESULTS.md` is in Russian):

- [`results/ladder/ncu-kernels.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/results/ladder/ncu-kernels.json)
- [`results/ladder/summary.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/results/ladder/summary.json)
