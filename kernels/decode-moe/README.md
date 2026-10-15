# decode-moe/ — `libqwen_moe_decode.so` (NVFP4 MoE decode GEMV, ABI 1)

Candidate for the next update (D4), not in the released build. Switch: `QWEN36_MOE_DECODE_CUDA=1` +
`QWEN36_MOE_DECODE_LIBRARY=<.so>` (profile key `cuda_moe_decode`), default off.

## What it replaces

The two routed-expert GEMMs of every decode step (`TritonNvfp4MoEKernel.apply`, decode branch):
`moe/fused_nvfp4._decode_gemm_marlin` → Triton `_decode_nvfp4_marlin_kernel` (BLOCK_N 16, BLOCK_KW 16,
4 warps), for gate_up (N 1024, K 2048) and down (N 2048, K 512, router weight in the epilogue).
In the release trace these are the "moe_decode_nvfp4_marlin" family: 20.2 + 9.8 µs per layer,
1.24 ms per step at 16K. The name is historical: it is a Triton FP32 CUDA-core GEMV, not vLLM Marlin.
The gated activation and the top-k sum keep their kernels.

## Identified arithmetic

From the TTGIR / PTX / SASS of the pinned Triton 3.6 build (`scripts/dump_triton.py`). Layout
`blocked<[2,1],[8,4],[1,4]>`: 8 lanes × 2 rows hold the 16 kw partials of one output.

- word w (8 e2m1 codes, k = 8w + j, nibble j) belongs to partial `P[w % 16]`; each partial walks its
  words in ascending order: `acc = +0; acc += a[8w+j] · lut[code_j]` (j = 0..7; e2m1 × bf16 products are
  exact in FP32, so the FFMA chain the SASS uses equals FMUL + FADD bit for bit);
  `P = fma(acc, scale[w >> 1], P)` with the e4m3 scale converted e4m3 → f16 → f32;
- reduction: `s_i = P[2i] + P[2i+1]`, then `shfl.bfly` 4, 2, 1:
  `v = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7))`;
- `out = bf16_rn(v · global[slot, n])` (gate_up), `bf16_rn((v · global) · topk_w)` (down).

The kernel (`src/qwen_moe_decode.cu`) does the same operations in the same order. Only the data movement
differs:
- 16-byte weight loads with a register prefetch ring;
- four threads per weight row: thread r owns partials 4r..4r+3, i.e. 16-byte chunks c ≡ r (mod 4),
  in ascending order;
- the activation is staged once per CTA in FP32;
- the shuffle tree is xor 2, xor 1, then an in-thread add.

## FreeToken side

`python/freetoken/kernel/cuda_moe_decode.py`, hooked in `layers/quantization/moe/nvfp4.py` (decode
branch only; `layers/moe.py` and the offload cache are untouched):
- **Library loading.** The library is dlopen'ed from a sealed memfd only if its sha256 matches
  `LIBRARY_SHA256`. The ABI probe checks `abi_version 1` and the source sha256.
- **Platform pin.** sm_120, torch 2.11, Triton 3.6.
- **Process self-test.** Synthetic banks: every e2m1 code, every finite e4m3 scale code including
  negative ones, globals 2⁻⁴⁰..2³⁰. Gaussian, heavy-tailed and edge activations (±0, bf16 subnormals,
  ±3e38). Both GEMMs, every M 1..8, top_k 8, slots shared between tokens. Compared bitwise with the
  production Triton call.
- **Per-layer check on real weights.** It runs on the first eager (non-capturing) decode call of each MoE
  layer, during warm-up before graph capture. Both GEMMs run on that call's real slot banks, routes and
  activations for every enabled M. A mismatch keeps the layer on Triton, and the event is logged. A layer
  whose first decode call is captured stays on Triton until an eager call decides it.

## Build and verification

```bash
CUDA_HOME=<cuda-13.4> CUDA_CCBIN=g++-15 CUDA_BIN_TOOLS=<dir with cuobjdump> ./build.sh   # CPU only
python scripts/test_bitexact.py --lib build/libqwen_moe_decode.so --model "$QWEN36_MODEL_DIR" \
    --layers 0,7,13,26,33,39 --bench --out r.json
pytest freetoken/tests/kernels/test_cuda_decode_kernels.py
```

`build.sh` applies the same checks as `decode-oproj/build.sh`: LF source, no `CALL.ABS` /
`__cuda_syscall`, no `.shared::cluster`, no spills. The pinned build has 0 spills.

| | |
|---|---|
| Library sha256 (pinned) | `83e90c43cb63ce11b284a37cdadf7eff82f147eb5873e988f358dd7df7d48dd5` |
| Source sha256 | `088bb27525a60b602d9ebc84d1d6810343747fec116fa9c288b031cc5437c95a` |
| Toolchain | nvcc 13.4.59, `-gencode arch=compute_120a,code=sm_120a`, `--cudart=static`, g++-15 |

## Results (RTX 5070 Ti, torch 2.11.0+cu130, Triton 3.6.0, owner GPU, 2026-10-15)

- **Bit-exact:** 4284 GEMM comparisons, 0 mismatches. The cases are synthetic banks plus real experts of
  layers 0/7/13/26/33/39; M 1..8 × top_k 8, plus top_k 4/2/1 at M 1/3/8; gaussian/heavy/edge activations;
  6 library variants; eager and CUDA graph. The whole decode MoE (gate_up → act → down → sum) with the GEMV
  swapped is also byte-identical: 21 cases, 0 mismatches.
- **Graph micro-benchmark** (40 layers over distinct slot banks larger than L2, median of 5 alternating
  pairs, µs per layer):

| | Triton | library (variant used) |
|---|---|---|
| M=1 gate_up | 18.5 | 12.5–12.9 |
| M=1 down | 9.4 | 7.4–7.8 |
| M=1 both | 28.4 | 20.5 |
| M=4 both | 86.1 | 41.4 |

  At M = 1 that is ≈ 8 µs × 40 ≈ 0.32 ms per decode step. gate_up runs at ≈ 720 GB/s, down at ≈ 600 GB/s.

## Model A/B (owner GPU, 2026-10-15, main-release profile + switches, S3 branch tree)

Three rounds, alternating arm order. Each arm is a fresh server start: the `main-release` configuration,
optionally with `QWEN36_OPROJ_GEMV=1` and/or `QWEN36_MOE_DECODE_CUDA=1` and the pinned libraries.
Workload per arm: greedy, `ignore_eos`, 2 × (64-token prompt, 256 output) and 1 × (16K prompt, 256
output). The prompts come from a fixed seed, so every arm receives byte-identical requests. No profiler.
Medians are in `ab-model-2026-10-15.json`.

| arm | decode tok/s, 64-token prompt | vs base | decode tok/s, 16K prompt | vs base |
|---|---|---|---|---|
| base (main-release) | 158.4 | | 148.1 | |
| + o-projection | 161.8 | +2.1 % | 150.7 | +1.7 % |
| + MoE GEMV | 166.2 | +4.9 % | 154.9 | +4.6 % |
| + both | 172.1 | +8.6 % | 159.4 | +7.6 % |

- **Generated text:** byte-identical in all 12 server starts.
- **Self-tests:** both libraries loaded their pinned sha256 and passed the self-test for M 1..8 in every
  run. No mismatch event was logged.
- **Round 1:** slower for every arm (base 148 tok/s against 159 in rounds 2–3). The paired comparisons
  of rounds 2–3 alone give o-projection +1.8…+3.0 %, MoE +4.7…+5.0 % and both +8.3…+8.4 %.
- **TTFT:** unchanged with both switches (0.38 s at 64 tokens, 1.28 s at 16K). The o-projection-only
  arm showed a higher 64-token TTFT median (0.45 s) that the "both" arm, which also runs the
  o-projection kernel, does not reproduce. This is not explained; re-check it in the paired release A/B.
- **VRAM:** the first launch of every kernel variant of both libraries in a fresh process took 0 MiB
  (`scripts/vram_first_launch.py`).
- **Not yet done:** the teacher gate (125/125, max_abs 0) is for the coordinator.
