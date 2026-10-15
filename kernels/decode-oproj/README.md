# decode-oproj/ — `libqwen_oproj_gemv.so` (decode out/o-projection, ABI 1)

Candidate for the next update (D4), not in the released build. Switch: `QWEN36_OPROJ_GEMV=1` +
`QWEN36_OPROJ_GEMV_LIBRARY=<.so>` (profile key `cuda_oproj_gemv`), default off.

## What it replaces

The tensor-wise FP8 W8A8 GEMM of the 40 decode out-projections (30 GDN `out_proj` + 10 attention
`o_proj`, N = 2048, K = 4096, one scalar weight scale) for 1 ≤ M ≤ 8:
`torch._scaled_mm(qa, W.t(), scale_a=0-d, scale_b=0-d, out_dtype=bf16)`. On torch 2.11 / cuBLASLt 13
that is, for every M in 1..8, a 384-byte memset (split-K semaphores) plus
`sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warpsize2x2x1_tensor16x8x32`,
grid (1, 32, 3), split-K = 3 reduced **inside** the kernel (spin on `LDG.STRONG`, `ATOMG.ADD`
semaphore, serial `FADD`). The release trace (`model-final-main-release` nsys) shows exactly this pair
per projection: memset 0.67 µs + xmma 13.8–15.5 µs.

Correction to `results/nsight/levers.json`: the `cublasLt::splitKreduce_kernel` with grid (1,1,1)
(5.5 µs, 0.23 ms per step) is **not** the o-projection. It follows the bf16 router GEMV
(split-K 228) in the trace. The o-projection lever is ≈ 0.62 ms per step (xmma) + 40 memset nodes.

## Identified arithmetic (bit for bit)

`scripts/probe_splitk.py` (absorption probes: ±448·448 and 2⁻¹⁸ terms show which k-tiles share a
chain and the reduction order), `scripts/identify_splitk.py` (100 % match required over 72 cases:
uniform e4m3 codes, gaussian codes and real `out_proj`/`o_proj` weights of layers 0/3/13/26/39, every
M 1..8), `scripts/splitk_determinism.py` (400 eager + graph runs: one distinct output), and the SASS
of the xmma kernel:

- K is cut into 64-byte k-tiles, and the tiles are dealt **round-robin** to the splits: split z owns tiles
  t ≡ z (mod 3), so split 0 has 22 tiles and splits 1 and 2 have 21. The splits are not contiguous ranges.
  Every contiguous-partition hypothesis fails on 1–7 elements per case.
- Each split is one `QMMA.16832.F32.E4M3.E4M3` chain from +0 over its tiles in ascending order, two k32
  MMAs per tile (bytes 0..31, then 32..63). These are the same per-instruction K sets as the CUTLASS
  kernel reproduced by `libqwen_fp8_gemv`. Within one QMMA, a large product swallows a tiny
  accumulator even when ±P cancel exactly (alignment to the largest exponent). That is why the K sets and
  the C operand must match, not just the sum.
- Partials are combined as `FADD(FADD(p0, p2), p1)` (the serial semaphore order).
- Epilogue: `alpha = FMUL(scale_a, scale_b)`; `FFMA(acc, alpha, +0)`; clamp (no-op); `F2FP.BF16` RN.

The kernel (`src/qwen_oproj_gemv.cu`) computes exactly this per output. One CTA handles a strip of 16
weight rows. Each of its 3 warps runs one split's chain from its own cp.async ring, and the partials meet
in shared memory. One launch replaces memset + xmma: no workspace, no semaphore.

## FreeToken side

`python/freetoken/kernel/cuda_oproj_gemv.py`:
- **Library loading.** The library is dlopen'ed from a sealed memfd only if its sha256 matches
  `LIBRARY_SHA256` (`kernel/pinned_ctypes.py`). The ABI probe checks `abi_version 1`, `max_m 8`,
  `splits 3` and the source sha256.
- **Process self-test** (at `finalize`, eager). Covers every M 1..8 with codes, gaussian codes, edge
  codes with exact cancellation, −0 activations, bf16 overflow and subnormal scales. Compared bitwise
  with the production `torch._scaled_mm` call.
- **Per-layer check.** Real weights are checked for every enabled M.
- **On mismatch.** That M (or the whole feature) stays on cuBLASLt, and the event is logged.

All call sites use `fp8_pertensor_linear.uniform_scaled_mm`: the generic uniform branch,
`models/qwen3_5_moe/gdn.py` (CUDA gated-norm quant) and `kernel/cuda_attention_gate.py` (fused gate
quant). The per-layer flag is set in `layers/quantization/linear/fp8_tensor.py` finalize.

## Build and verification

```bash
CUDA_HOME=<cuda-13.4> CUDA_CCBIN=g++-15 CUDA_BIN_TOOLS=<dir with cuobjdump> ./build.sh   # CPU only
python scripts/test_bitexact.py --lib build/libqwen_oproj_gemv.so --model "$QWEN36_MODEL_DIR" --layers all --bench --out r.json
python scripts/vram_first_launch.py --oproj build/libqwen_oproj_gemv.so --moe ../decode-moe/build/libqwen_moe_decode.so --out v.json
pytest freetoken/tests/kernels/test_cuda_decode_kernels.py   # integrated path, eager + graph
```

`build.sh` refuses a CRLF source, a `CALL.ABS` / `__cuda_syscall` in the SASS, `.shared::cluster` in the
PTX and register spills. Build of the pinned library: 64 registers, 0 spills.

| | |
|---|---|
| Library sha256 (pinned) | `fa9302e31cb88a02f9d6b37afc9e19691859f21e92861baf1bef3b2cab6a78a9` |
| Source sha256 | `ff47a7d57830fa593d633a0482f26adc55724f03727ec3d80725f415146a7431` |
| Toolchain | nvcc 13.4.59, `-gencode arch=compute_120a,code=sm_120a`, `--cudart=static`, g++-15 |

## Results (RTX 5070 Ti, torch 2.11.0+cu130, owner GPU, 2026-10-15)

- **Bit-exact:** 832 cases × 6 variants, 0 mismatches. The cases are 96 synthetic and 736 real (all 40
  layers × M 1..8 × gaussian/heavy-tailed activations), run eager and inside CUDA graphs.
- **Graph micro-benchmark** (40 projections over distinct weights, median of 5 alternating pairs, µs per
  projection):

| M | cuBLASLt (memset + xmma) | library (variant used) |
|---|---|---|
| 1 | 17.4 | 12.1–12.5 |
| 2 | 17.4 | 12.2–12.3 |
| 4 | 17.5 | 12.5–12.6 |
| 8 | 17.5 | 12.7–12.9 |

  That is ≈ 5 µs × 40 ≈ 0.2 ms per decode step at M = 1. The kernel moves the 8 MiB of weights at ≈ 690 GB/s.
