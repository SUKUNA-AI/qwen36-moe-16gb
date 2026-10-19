<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Bit-exact CUDA GatedDeltaNet kernels for Qwen3.5/3.6 on sm_120 (glue fusion + fused FLA chunk_h/chunk_o)

## Summary

In the 30 GatedDeltaNet (GDN) layers of Qwen3.6-35B-A3B, the work around FLA's chunked kernels takes many small
launches and copies: conv input transpose, causal conv + SiLU, L2 norm, gate cumsum, gated RMSNorm, and the FP8
quant for `out_proj`. Our fork adds an opt-in CUDA library (`libqwen36_gdn`, ABI 2) that fuses this glue and
runs decode as one launch per layer. Its outputs are bit-identical to the existing path. Measured on our setup:

- one GDN layer at T = 2048 drops from 1.230 ms to 0.509 ms (2.41x);
- model TTFT is -8.6% at 64K and -6.2% at 120K, with decode neutral.

The same library has a fused CUDA version of FLA `recompute_w_u` + `chunk_h` + `chunk_o`. It is bit-exact and
cuts DRAM traffic by about 3x, but its model-level gain is small: TTFT 16K 1.343 -> 1.316 s on one step of our
lever ladder, and below noise in an earlier A/B.

## Setup

- RTX 5070 Ti 16 GB (sm_120), WSL2; CUDA 13.4, GCC 15.3. The library is built with
  `--ftz=false --prec-div=true --prec-sqrt=true` and no `--use_fast_math`.
- torch 2.11.0+cu130, triton 3.6.0. The reference is FreeToken's FLA Triton kernels, compiled by ptxas 13.4 and
  also by ptxas 12.9.
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`, GDN geometry Hg16 / H32 / K128 / V128, FP32 state pool
- FreeToken 0.1.3 `cac247a8` + our fork, prefill chunk 2048, single request, layer-major prefill on (one
  2048-token chunk per GDN call)

## Problem in upstream today

Paths at main `4d2e2e2`, all in `python/freetoken/`. Prefill:

- `models/qwen3_5_moe/gdn.py:99` copies the conv input with `conv_in.transpose(0, 1).contiguous()` before
  `causal_conv1d_varlen` (`:100`, sgl_kernel wrapper in `kernel/causal_conv1d.py`). The conv result then
  goes through `torch.split` / `reshape(...).to(dtype)`.
- `_gate_params` (`gdn.py:86-89`) computes `beta` and `g` with separate torch ops. `chunk_local_cumsum`
  (`kernel/fla/chunk.py:43`) follows.
- `kernel/fla/chunk.py:48-75` runs the following Triton kernels:
  - the kkt+solve_tril kernel (`kernel/fla/chunk_fwd.py:391`);
  - `recompute_w_u_fwd` (`chunk_fwd.py:407`);
  - `chunk_gated_delta_rule_fwd_h` (`chunk.py:57`);
  - `chunk_fwd_o` (`chunk.py:67`).
- `gdn.py:198` runs the gated norm. `out_proj` (`gdn.py:203`) then quantizes the activation to FP8 in a
  separate kernel (`kernel/triton/fp8_pertensor_linear.py:291`, `_static_quant`).

Decode: `_conv_decode` (`gdn.py:104-109`) runs, then `gdn_decode_fla` (`gdn.py:163`), which calls
`fused_sigmoid_gating_delta_rule_update` (`models/qwen3_5_moe/gdn_kernels.py:61-63`).

There is no CUDA GDN path on main. Since our base, only #701 (GGUF) has touched `gdn.py`. We found no issue or PR
proposing this; #617 only concerns pre-sm70 launch configs.

## What we changed

Switches, default off:

- `QWEN36_CUDA_GDN=1` + `QWEN36_CUDA_GDN_LIBRARY=<.so>` enable the library;
- `QWEN36_CUDA_GDN_DECODE=cta|cluster` selects the decode kernel (we use cluster);
- `QWEN36_CUDA_GDN_FLA=wu|ho` enables the fused FLA kernels (our current profile uses `wu`).

Fork modules: `python/freetoken/kernel/cuda_gdn.py` (ctypes loader, requires ABI exactly 2) and
`python/freetoken/models/qwen3_5_moe/gdn.py`. The library sources are in our `kernels/gdn/`.

| Kernel | Replaces |
|---|---|
| `qwen_gdn_conv_qkv` | causal conv + SiLU read straight from the fused qkvz rows, with L2 norm in the epilogue. No transpose copy; -7 launches per layer. |
| `qwen_gdn_gate_cumsum` | gate + chunk cumsum; -9 launches |
| `qwen_gdn_gated_norm` | gated RMSNorm with the FP8 activation quant for `out_proj` fused in |
| `qwen_gdn_decode_fused` / `qwen_gdn_decode_cluster` | decode, one launch per layer, per CTA or as an 8-CTA cluster |
| `qwen_gdn_fla_wu_chunk_ho` / `qwen_gdn_fla_chunk_ho` (ABI 2) | FLA `chunk_h` + `chunk_o` fused, with `recompute_w_u` folded in for `wu`; kkt+solve stays in Triton |

Design constraints:

- No host sync. Inside CUDA-graph capture there is no allocation except through torch's caching allocator.
- Every function validates layouts and dtypes. Unsupported inputs fall back to the original path before any
  state is touched:
  - T <= 32 for FLA;
  - unusual shapes;
  - a BF16 snapshot without FP32;
  - batches with an empty sequence.
- The fused FLA kernels write the FP32 GDN prefix snapshots. Those snapshots are covered in our separate
  prefix-cache issue and are not part of this proposal.

## Measured impact

Component level (`kernels/README.md`, `docs/ARCHITECTURE.md` §3.3; no profiler unless noted):

- One GDN layer at T = 2048 drops from 1.230 to 0.509 ms (2.41x). At T = 4096 it goes from 26 kernels / 3.04 ms
  to 8 kernels / 1.02 ms.
- The fused FLA kernels alone at T = 2048 take 0.325 -> 0.185 ms. DRAM traffic for kkt+wu+h+o at T = 2048 drops
  from 261 to 88 MB (ncu).
- In the model, under Nsight Systems (structure only; `results/experiments/fla-results.json`):
  - per-call median at T = 2048 is 325.5 -> 215.1 us (1.51x);
  - GDN core time is 89.5 -> 57.0 ms for a fresh 16K and 617 -> 407 ms for a fresh 120K;
  - the whole prefill window changes by -3.0% (16K) and -0.9% (120K).

Model level, A/B without profiler:

- Glue kernels: TTFT -8.6% at 64K (3 pairs) and -6.2% (median) at 120K; decode neutral.
- Fused FLA, earlier A/B (release f1 build): below noise (paired SD 9.5% / 3.2%). 16K was +2.6%, and 64K/120K
  were no better than without it (`docs/RESULTS.md` §8, `kernels/README.md`). It stayed off in f1.
- Fused FLA, lever-ladder step "CUDA GDN kernels for prefill" (`docs/RESULTS.md` §9, `results/ladder/ab.json`,
  `summary.json`): 3 pairs, 60 s warm-up, 128 output tokens, each step = previous + one lever. Values are
  median (mean):

| Metric | Before step | With fused FLA |
|---|---:|---:|
| TTFT 16K, s | 1.343 (1.343) | 1.316 (1.315) |
| TTFT 64-token prompt, s | 0.315 (0.312) | 0.317 (0.319) |
| Decode 16K, tok/s | 143.9 (142.9) | 143.6 (142.0) |
| Nsight 16K prefill window, ms (diagnostic) | 1284 | 1271 |

The two fused-FLA A/Bs disagree in sign at 16K. Both effects are about the size of the run-to-run noise
documented in `docs/RESULTS.md` §10.

## Correctness evidence

- **GDN op suite**: 124 cases on raw bits, where -0.0 vs +0.0 and NaN payloads count and the entire FP32 state
  pool is compared, including untouched slots. The inputs are synthetic, stress distributions and real
  activations of layers 0, 1, 2, 13, 26 and 38. Results: 124/124 for the base library, 124/124 with `--fla`
  `wu` (60 fused calls, T 64..8192) and 124/124 with `ho`. With `wu` and the reference built by ptxas 12.9 it
  is 94/94.
- **FLA kernel suite vs Triton** `chunk_h` / `chunk_o` / `recompute_w_u`: 123/123 with ptxas 13.4 (841M `o`
  elements + 676M state elements per variant) and 85/85 with ptxas 12.9.
- **Negative controls**: 3 builds, each with one recipe point broken, are all detected (10, 10 and 4 failing
  cases).
- **HMMA probe** (491,520 outputs per question): orientation and in-instruction permutations give 0
  mismatches, while a different K-set grouping gives 238,372. Only the K-set and chain order matter.
- **SASS check**: every `MUFU.EX2` has the non-FTZ expansion; there is no local memory and no FTZ instruction.
- **Teacher forcing** at model level: 125/125, max_abs 0 (f1, and the ladder step above).

## Limitations / not measured

- sm_120a only, Qwen3.5/3.6 GDN geometry only, B = 1 varlen (single prompt). Verified against Triton 3.6 FLA
  kernels; main's kernels may need a re-check under the Triton version you ship.
- The FLA gain is small by Amdahl: GDN core is 6.3% of a 16K window and 2.8% of a 120K window.
- Decode with the cluster kernel is "neutral" at model level. No separate decode-only microbench is published.
- Our measurements use layer-major prefill (2048-token GDN calls). Plain chunked prefill and larger chunks
  were not measured.
- The library is a prebuilt `.so` pinned by sha256. nvcc output is not byte-reproducible, so a rebuild needs a
  new pin.

## Proposal

We could split this into two PRs, both default-off:

1. Glue kernels (conv_qkv, gate_cumsum, gated_norm + FP8 quant, decode), which carry the model-level gain.
2. Fused FLA `wu_chunk_ho`, optional given its small effect.

Questions:

- Should the kernels ship as a JIT/tvm-ffi module next to `kernel/csrc/jit`, or as a pinned prebuilt like
  our fork?
- Is bitwise parity with the Triton path a requirement you want enforced in CI for such kernels?
- Do you want the fused FLA part at all, given its small effect?
