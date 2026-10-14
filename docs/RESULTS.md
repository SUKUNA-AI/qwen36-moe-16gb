# Results (release f1, 2026-10-14)

This file lists every key number with its **definition** and **provenance**. Provenance is a file under
`results/`, plus the original report name where the two differ. Files were copied from the private
harness with paths, host names and user names replaced by placeholders such as `<native-harness>`,
`<workspace>` and `api.example.invalid`. Numbers and hashes are unchanged.

Conventions:
- Sizes: GiB = 2^30 B, MiB = 2^20 B, GB = 1e9 B.
- TTFT is client time to the first content token, measured over SSE. Decode tok/s is client decode
  throughput unless marked as engine.
- Percentiles are linear interpolation (numpy). `n` is the sample count.
- `NOT_RUN` means nothing was measured. A value is estimated only where it is labelled DERIVED or
  estimate.

Build under test: `main-release`, served as profile `main128`. It combines:
- layer-major prefill;
- bit-exact CUDA NVFP4 prefill at chunk 2048;
- CUDA GDN (ABI 2, cluster decode);
- FP8 GEMV;
- attention batching k=2;
- the TMA decode gather;
- FP32 GDN snapshots, the retained chunk snapshot and the fused attention gate;
- `FREETOKEN_DISABLE_JIT=1` and BF16 KV.

Source identity: `04dbae8b33be6c66`. Release receipt: `5968b76fe607e281`.

## 0. Gate summary (final gates run f1)

| Gate | Verdict |
|---|---|
| A: CPU suites, sync, kernel provenance, tunables | PASS |
| B: teacher forcing, main-release | PASS: 125/125, max_abs 0, PPL +0.0% |
| B: teacher forcing, long256-release | PASS_EQUIV_R1B256, then strictly bit-exact 125/125 with the clean-cache harness v7 (§2) |
| D: tool pilot, 5-pair matrix vs bf128, compare | PASS |
| C0: main128 at 131,072 | PASS (min free 1.197 GiB, TARGET) |
| L4: long256 at 262,144 | PASS (min free 1.406 GiB, TARGET) |
| F1: baseline64 at 65,536 | PASS_WITH_KNOWN_LIMITATION (min free 0.789 GiB, FLOOR band) |
| E: soak pilot + 7200 s soak | PASS |
| Install, rollback check, live start/stop/profile switch | PASS. Network acceptance 59/60 (one known near-tie) |

Sources: `results/metrics/metrics.json` (`release.gate_verdicts`, from `final-gates-f1-summary.json`)
and the dashboard `results/dashboard/qwen36-release-f1.html` (Russian).

## 1. Speed matrix (release vs bf128)

- **Definition.**
  - 2 arms: `bf128` (upstream mechanics + FP32 globals, chunk 2048, 131072 context, no f1 switches) and
    `main-release`.
  - 5 pairs in rotating order, exactly 128 output tokens, in one session with no profiler.
  - Fresh prompts at 64 / 4K / 16K / 64K / 120K tokens, plus a history continuation, a last-message edit
    at 16K, a tool-result edit at 16K, and a natural-EOS tool-call cycle task.
  - Gating metric: full-request elapsed time. TTFT and decode are reported, not gated.
  - Rule: see [METHODOLOGY.md](METHODOLOGY.md) §2.
- **Provenance.** `results/matrix/final-f1-m-main-release-vs-bf128-comparison.json` (per-case log-ratio
  t-intervals and every pair's ratios) and `results/matrix/final-f1-m-matrix.json` (manifest, arm
  identities). Manifest sha prefix `742a52b0`.

Medians, bf128 -> release:

| Case | TTFT | Decode tok/s | Full-request speedup (geomean of 5 pairs) |
|---|---|---|---|
| fresh 64 tok | 0.75 -> 0.79 s (0.94x, CI 0.84-1.04, non-gating) | 96.4 -> 122.8 | 1.13 |
| fresh 4K | 2.24 -> 0.77 s (2.95x) | - | 1.95 |
| fresh 16K | 6.78 -> 1.59 s (4.23x) | 94.2 -> 119.1 | 3.03 |
| fresh 64K | 28.46 -> 9.14 s (3.09x) | 84.4 -> 99.3 | 2.86 |
| fresh 120K | 54.77 -> 22.07 s (2.52x) | 80.7 -> 94.6 | 2.45 |
| history continuation 4K | 2.88x | | |
| last-message edit 16K | 6.85 -> 0.78 s (9.08x) | | 4.44 |
| tool-result edit 16K | 7.11 -> 0.76 s (9.47x) | | 4.49 |
| tool-cycle task | | | 1.08 (CI 0.97-1.20) |

- Decode ratios across cases are 1.19-1.32x.
- 15/40 outputs differ from bf128. This is an observation only: they are prefix-hit cases with
  different snapshot flags. Quality is gated by teacher forcing.
- Verdict: **PASS**, with `noninferior_all_gating_cases` and `gain_at_least_5pct_bonferroni`.

### 1.1 Like-for-like at 64K (original FreeToken vs release)

- **Definition.**
  - Context 65536, output 192 tokens, 60 s sustained warm-up.
  - 3 arms in one session: `ab64-fp16` (unmodified FreeToken numerics), `ab64-fp32` (+ FP32 global fix)
    and `ab64-rel` (release switches at 65536).
  - 2 valid rounds in opposite order; the table shows their mean. Round 1 was lost to a configuration
    error and replaced by round 3.
- **Provenance.** `results/ab64/ab64-2-chunk-screen.json` and `results/ab64/ab64-3-chunk-screen.json`.

| Arm | 4K TTFT / decode | 16K | 60K |
|---|---|---|---|
| original FreeToken (FP16 globals) | 1.93 s / 116.0 tok/s | 6.04 / 111.3 | 22.86 / 101.3 |
| + FP32 globals | 1.85 / 118.1 | 5.86 / 115.7 | 22.43 / 107.6 |
| release f1 | **0.58 / 142.4** | **1.49 / 138.0** | **7.50 / 122.2** |

Release vs original: decode +23% / +24% / +21%, TTFT 3.3x / 4.1x / 3.0x. The two release rounds agree
within 0.3%. FP32 globals cost no decode speed (+2..6%, order-sensitive).

### 1.2 256K-class

| Measurement | Value | Definition / provenance |
|---|---|---|
| f1 250K fresh TTFT | 80.4 s (3112 tok/s) | `long-facts-250000`, 250,069 prompt tokens, single run in gate L4. `results/context-gates/final-f1-l256-a1-evaluation.json`, `metrics.json` (`throughput_capacity`) |
| f1 250K fresh, 30-min test | TTFT p50 77.38 s (max 77.88), decode p50 60.8 tok/s, n=11 | `results/long256-30min/metrics.json` |
| f1 at the limit | 261,468 + 676 tokens accepted, +1 rejected (400); TTFT 91.3 s, decode 52.5 tok/s | gate L4 boundary check, `metrics.json` |
| f1 250K decode step | 19.1 ms (engine GPU time) | `metrics.json` |
| Original mechanics at 261K (`bf256-final-r1b`) | TTFT median 139.5 s (138.6-140.1), decode ~52.5 tok/s, 5 runs | arm of the earlier 256K paired matrix, `results/matrix/l256r1b-r1-m1-...-comparison.json` (that matrix compared bf256 to an intermediate build: TTFT 1.26x, elapsed 1.256x, decode 1.009x) |
| Original mechanics at 250K | TTFT 126.5 s | first long256 live gate, 2026-10-13: Triton, chunk 4096, FP16 globals, single run (historical, not re-measured) |
| Pre-layer-major chunk-major CUDA build at 250K | 115.7 s | R1b gate, intermediate build |

The rows above come from different sessions. Only §1 and §1.1 are paired comparisons.

## 2. Quality

| Metric | Value | Provenance |
|---|---|---|
| Teacher forcing, main-release vs bf128 | 20/20 cases, 943/943 forced tokens, **125/125** logit checkpoints, max \|Δlogit\| 0, relative PPL growth 0.0 (CI [0, 0]), no new functional failures | `results/quality/quality-main-release.json` |
| Teacher forcing, long256-release, standard order | 19/20 bit-identical. long-120000: max_abs 1.232, NLL +0.0177, PPL +0.072% | `results/quality/quality-long256-release.json` |
| Clean-cache confirmation (C) | long256, long-120000 alone on a fresh server: all comparisons 0.0 | `results/quality/final-f1-l256-confirm-C.json` |
| Same-history reproduction (D) | main128 with ru-0 -> long-4096 -> long-120000 reproduces the 256K deviation to the last digit | `results/quality/final-f1-l256-confirm-D.json`, `results/experiments/l256-deviation-results.json` |
| Harness v7 (clean cache before long-120000) | main-release 125/125 and long256-release **125/125 at 0.0**, 20/20 NLL lists identical, PPL 0.0 | `results/quality/quality-long256-release-v7-clean-cache.json`; `results/quality/cache-reset-result.json`. The v7 main-release report is byte-identical to `quality-main-release.json`, sha `92569dea...` |
| FP32 vs FP16 globals (basis decision) | NLL relative PPL -1.13% (CI [-2.27%, -0.08%]); eval-v1 59/60 vs 57/60; no tool losses | pre-registered rule; summary in `metrics.json` (`quality`). The eval cases are private and not published |
| Functional known limitations | `executable-code` and `long-facts-16384` (all profiles), `long-facts-60000` (baseline64). Identical on the reference; first-token near-ties of the checkpoint | gate evaluations in `results/context-gates/` |

## 3. Latency and the 2-hour soak

- **Definition.**
  - `main-release`, sequential requests, 256 output tokens, 5 s pauses.
  - Mixed schedule: fresh 64 / 4K / 16K / 64K / 120K, prefix follow-ups, arithmetic, tool calls and
    client cancels.
  - Run identity is bound to the matrix manifest.
- **Provenance.** `results/soak/soak-final-f1-a1-analysis.json` and `results/metrics/metrics.json`
  (`latency.soak`). Summary text: `results/metrics/METRICS.txt`.
- **Run.** 7212 s, 509 iterations, 771 requests, **741/741 rows passed**, 0 errors, 0 stall events,
  0 dropped metrics.

| Prompt | TTFT p50 / p90 / p99 / max (s) | Decode p50 / p10 (tok/s) | E2E for 256 tokens, p50 / p99 (s) |
|---:|---|---|---|
| 93 | 0.601 / 0.636 / 0.671 / 1.391 | 133.7 / 126.6 | 2.51 / 2.84 |
| 4,125 | 0.606 / 0.634 / 0.649 / 0.665 | 135.0 / 127.7 | 2.49 / 2.78 |
| 16,413 | 1.360 / 1.401 / 1.610 / 1.649 | 129.8 / 124.1 | 3.32 / 3.82 |
| 65,565 | 8.140 / 8.523 / 9.288 / 9.598 | 112.2 / 107.1 | 10.41 / 11.82 |
| 120,029 | 20.97 / 21.63 / 22.92 / 23.29 | 97.5 / 93.7 | 23.59 / 25.81 |

- Prefix hit: 4400-token prompt with 4032 cached, TTFT p50 0.594 s, 138.2 tok/s, 102/102 hits.
- Cancel recovery: median 1.45 s (max 1.61). A cancel during prefill recovers in 0.80-1.03 s.
- Engine:
  - decode step GPU time p50 is 7.3 / 7.2 / 7.5 / 8.7 / 10.0 ms (64 tokens .. 120K);
  - the scheduler gap is ~0.22 ms per step;
  - ingress overhead outside the engine is 8-182 ms and grows with prompt length.
- Mean ITL is 1000 / decode tok/s. A per-token ITL distribution is NOT_MEASURED.

### 3.1 256K stability test (30 min, long256-release)

Provenance: `results/long256-30min/metrics.json` and `l256short-r1-analysis.json`. This is not a soak
and not the 7200 s gate.

- Run: 1854 s, 42 iterations, 109/109 cases, 0 errors, stalls or restarts.
- Fresh TTFT p50: 16K 1.47 s, 64K 8.56 s, 120K 21.94 s, 250K 77.38 s.
- Decode p50: 96.9 / 85.7 / 77.1 / 60.8 tok/s. This is lower than main128 because there are 1481 slots
  instead of 3164.
- 250K continuations: 11/11 cache hits with TTFT ~1.06 s.
- 6 of 6 cancels landed mid-prefill on a 250K prompt; recovery p50 1.64 s.
- VRAM min free 2.56 GiB, with 0 samples below 1 GiB. KV pool peak 95.6% of 262,175 pages.
- Host RAM grew +6..65 MB in 30 min. This is indicative only.

## 4. Nsight profile of the release build

- **Definition.** One Nsight Systems capture of `main-release` (same source as the gate runs) over 5
  scenarios: warm-up short, warm-up chunk, warm decode, fresh 16K and fresh 120K.
  - Wall-time decomposition uses critical-path attribution. Overlapping CUDA durations are not summed.
  - No speed was measured under the profiler.
- **Provenance.** `results/nsight/dashboard.json`, `results/nsight/levers.json` and
  `results/nsight/decode-step-stats.json`. The raw trace is private.

| Scenario | Breakdown |
|---|---|
| prefill 16K, 1.53 s | MoE 40%, attention 19%, FP8 dense 15%, GDN 8%, exposed H2D 5%, idle 4.5% |
| prefill 120K, 21.9 s | attention 65%, MoE 18%, FP8 7%, GDN 4%, idle 1.3% |
| short prompt, 278 tokens, 0.60 s | exposed H2D 75.5% |
| decode 16K, 8.10 ms/step | FP8 28%, MoE 20%, gather 18%, attention 7%, idle 8.6% |
| decode 120K, 11.39 ms/step | attention 31%, FP8 20%, MoE 15%, gather 14% |

- A decode step is ~998 kernels with ~0.48 ms of inter-kernel gaps.
- PCIe: 18.25 GB per fresh prefill at 34-41 GB/s. Copy/compute overlap is ~85% at 16K and 120K, and
  12% for short prompts.
- Compared with the 2026-10-13 baseline trace (64K profile, not fully comparable): decode step p50
  10.0 -> 8.0-8.1 ms, gather 4.58 -> 1.49 ms/step, FP8 dense 3.04 -> 2.24 ms/step.

### 4.1 Component Nsight Compute

ncu inside the model process is NOT_RUN. Injection drove host RAM to ~38.9 GiB, and the 120K prefill
thrashed at the cap. It was replaced by 21 component-level ncu rows of the same kernels with their
provenance: `results/nsight/ncu-component-kernels.json`.

| Kernel | Bound |
|---|---|
| Prefill attention (FlashInfer FA2, SM120) | tensor pipe, SM ~82%, top stall math_pipe_throttle |
| Decode attention at 131072 | DRAM, 94% (841 GB/s) |
| CUDA NVFP4 MoE prefill (`fast_gemm`) | tensor pipe, SM ~83% |
| GDN chunk kernels | DRAM, 79-86% |
| FP8 GEMV (12288x2048) | DRAM, 88-90% (808 GB/s) |
| Decode gather (TMA) | PCIe; ~30-33 GB/s under ncu, ~42 GB/s in the model; host-link ceiling 41-45 GB/s |

The three ranked bit-exact levers are in `results/nsight/levers.json`:
1. gather bandwidth + prefetch, estimated +7.5..13% decode;
2. skipping resident experts on short prefills, estimated -27% short TTFT;
3. MoE decode / o-proj kernels, estimated ~+11% at 16K.

## 5. Memory, VRAM, KV

Provenance: `results/metrics/metrics.json` (`memory`, `kv_cache`).

| Item | Value |
|---|---|
| Host: service anon (soak median) | 19.72 GiB, including the pinned expert bank of 18.245 GB = 16.99 GiB |
| Host: cgroup / shmem / file cache | 19.90 / 0.105 / 0.013 GiB |
| Host: growth over 2 h (late minus early) | anon +45..48 MB at every prompt length (flag 268 MB); shmem 0. The first soak without tunables grew +394 MB |
| Host: unit peak incl. model load | 27.8 GiB. WSL MemAvailable median 25.0 GiB |
| VRAM: torch allocated / reserved | 11.15 / 11.39 GiB = dense 2.52 + KV 2.50 + GDN 0.54 + expert slots 5.25 (3164 x 1.78 MB) + other 0.34 |
| VRAM free in the soak (NVML, includes desktop use) | median 1667 MiB, p5 697 MiB, min 436 MiB (state sampler). The 1 Hz request sampler saw a min of 367 MiB; 4/825 requests were below 512 MiB and 310 below 1 GiB |
| Gate minimum free | main128 1.20, long256 1.41, baseline64 0.79 GiB |
| KV | 20,480 B/token. Pools: 131,111 pages (2.50 GiB) / 262,175 (5.00 GiB) / 65,600 (1.25 GiB). Soak peak 120,284 pages (91.7%) |
| Expert slots | main128 3164, long256 1481, baseline64 3917. Decode miss rate 0.08-0.12 (main128), 0.38 (long256) |
| Startup | cold start to edge-ready 132 s, restart 125 s, profile switch 103-110 s |

## 6. Tool-use cycle (end to end, production main128)

- **Definition.**
  - 16 synthetic tasks x 10 conditions: T = 0 / 0.6 / 0.7, thinking on/off, stream/non-stream.
  - 160 runs and 357 rounds through the public HTTPS route, with TLS verified, on one instance.
  - Strict success requires the right calls with correct names and args in dependency order, a correct
    final answer, and no protocol error. Intervals are Wilson 95%.
- **Provenance.** `results/tool-use/summary.md` and `results/tool-use/results-redacted.json` (final
  answers and call arguments removed), plus `results/tool-use/selftest-results.json`.

| Metric | Value |
|---|---|
| API contract | **PASS**: 14/14 unsupported options -> 400 JSON before SSE; 0 protocol/transport errors in 357 rounds; 228/228 tool-call arguments valid JSON matching the schema; `tool_choice=none` respected 20/20 |
| Strict success, tool tasks | 106/120 = 88.3% [81.4, 92.9]; final answer correct 117/120 = 97.5% |
| By group | chains of 2-4 calls 30/30; tool choice 10/10; parallel 19/20 (all 20 emitted 2-3 calls in one turn); 16K long-context EN 10/10, RU 5/10 |
| Thinking on vs off | 92.5% vs 85.0%. All 11 "no call when required" failures were thinking-off |
| Latency per round | thinking off: TTFT 0.57 s, total 0.75 s; thinking on: total 1.56 s (p90 3.7 s); 16K first round: TTFT 1.58 s uncached / 0.62 s cached |

Cosmetic: with thinking on, streamed final content starts with two newlines (47/48). Non-stream content
never does. Recommendation: enable thinking for tool workflows that need real data.

## 7. FP8 KV cache: measured, rejected for now

FP8 KV halves KV bytes (2.69 -> 1.34 GB at 128K), which frees room for more expert slots
(main128 3164 -> 3917, long256 1481 -> ~2987).

Speed screens: speed only, no quality claim. 2 alternating rounds, output 512, 60 s warm-up. No
clipping was observed (`kv_clipped_values_total` 0 with weight-bound scales). Provenance:
`results/fp8kv/fp8q-{1,2}-chunk-screen.json` (main128), `results/fp8kv/fp8l-{1,2}-chunk-screen.json`
(long256), and `results/fp8kv/predictions.json` (model predictions made before the run).

| Profile, prompt | TTFT bf16 -> fp8 | Decode bf16 -> fp8 | Full request, 512 out |
|---|---|---|---|
| main128 16K | 1.48 -> 1.58 s (+6%) | 131.8 -> 144.2 (+9.5%) | |
| main128 120K | 21.8 -> 27.5 s (+26%) | 95.2 -> 119.7 (+26%) | |
| long256 4K | 0.52 -> 0.58 (+11%) | 104.9 -> 131.8 (+26%) | -17% (faster) |
| long256 16K | 1.45 -> 1.59 (+10%) | 101.3 -> 131.2 (+29%) | -16% |
| long256 120K | 21.81 -> 27.59 (+26%) | 80.2 -> 114.2 (+42%) | +14% (slower) |
| long256 250K | 77.18 -> 102.35 (+33%) | 62.0 -> 90.5 (+46%) | +26% |

Why it was rejected:
- FP8 KV is **not bit-exact**, and its quality is unverified (teacher + eval would take ~1.5 h of GPU).
- It loses 10-33% TTFT on every fresh prompt. The FP8 prefill-attention path in FlashInfer FA2 on
  sm_120 is the cost.
- Full-request break-even needs ~2500 output tokens at 120K and ~5000 at 250K.
- The decode gain matters only for long multi-turn or agent sessions.
- Bit-exact levers of similar value come first.

It may come back as an optional `long256-fp8` profile. See
[upstream-notes/04c](upstream-notes/04c-fp8-kv-cache.md).

## 8. Measured and excluded from f1

| Item | Result | Why excluded | Provenance |
|---|---|---|---|
| Fused FLA in CUDA (`QWEN36_CUDA_GDN_FLA=wu`) | Bit-exact 123/123; per call 325.5 -> 215.1 us (1.51x); GDN core time 16K 89.5 -> 57.0 ms, 120K 617 -> 407 ms. Window only -3.0% (16K) / -0.9% (120K) | Below the A/B noise (paired SD 9.5% / 3.2%); prefill is GPU-bound elsewhere. Candidate for the next update | `results/experiments/fla-results.json`, `fla-nsys-compare.json` |
| Layer-major slot refill | Bit-exact (teacher pass). Decode ratio medians: 16K 0.984 (last_chunk) / 1.010 (restore); 120K 0.980 / 1.060 | No gain within noise; last_chunk costs more misses at 120K; +11% borrowed slots | not published (pre-release A/B) |
| Sync-lean (fewer host syncs, FlashInfer plan reuse) | Bit-exact (272/272 standalone); est. -56..-89 ms GPU per 120K (0.24-0.38%) | Below noise, and it conflicts with attention batching | not published |
| MTP / speculative decoding | Numeric guard rel L2 0.0521 > 0.03 on mixed chunk shapes; the BF16 MTP head (1.69 GB) would cost ~950 expert slots | Owner decision: off without new evidence | [upstream-notes/04a](upstream-notes/04a-mtp-speculative-decoding.md) |
| Vision input | Vision tower 0.89 GB | Owner decision: API stays text-only | - |
| `(1+w)` RMSNorm FP32 fix | Upstream folds `(1+w)` into BF16 at load: 77% of 171,008 elements change, mean rel err 1.4e-3. Shifts near-tie margins by ~0.4 logit | Breaks bit-exactness vs the reference; expected effect within noise; does not fix the two known failures. Not scheduled | [upstream-notes/02](upstream-notes/02-freetoken-qwen35-gemma-norm-bf16-fold.md) |
| RAM complement residency | 120K TTFT ~117.6 s, decode 33-41 tok/s vs 81.5 | 2-3x slower | [upstream-notes/04b](upstream-notes/04b-ram-complement-residency.md) |
| Larger prefill chunks (c4096 / c8192) | Faster TTFT before layer-major, at the cost of expert slots | Superseded by layer-major (same TTFT class, keeps c2048 numerics and slots) | STATE history |
| Bit-exact FlashInfer prefill replica | 420/420 bitwise, but no variant beats FlashInfer on long KV (0.59-1.03x) | FlashInfer already runs at 72-89 of a measured 91-104 TFLOPS peak | not published |

## 9. Experiments for the next update (not in f1)

| Experiment | Result | Provenance |
|---|---|---|
| Short-prompt route (`--moe-prefill-hit-d2d`, T = 2049) | TTFT medians over 10 rounds and 2 sessions: 64 tokens 0.565 -> 0.446 s; prefix hit 0.585 -> 0.468; 16K edit 0.596 -> 0.481. 2K-16K fresh unchanged (0.998-1.007). Greedy outputs 200/200 identical. Teacher gate NOT_RUN | `results/experiments/short-route-summary.json` |
| Commit/decode race | SAFE. Poison arm bit-identical (16/16 texts, 640/640 logit checksums); negative control fails at the race site (13/16) | `results/experiments/commit-race-results.json`, `commit-race-compare.json` |
| 256K header-dedup deviation | Root cause confirmed (ARCHITECTURE §5.1) | `results/experiments/l256-deviation-results.json` |

## 10. Environment findings

- **Measurement noise.** Engine GPU-event times vary as much as client wall time:
  - CV 0.4-1.7% at 64K in a quiet session;
  - 3-6% under CPU load;
  - up to 30% between sessions.

  Identical kernels slow down periodically, by +0.19-0.42 ms every ~2.3 ms. The hypothesis is WDDM
  time-slicing with the 4K@159 Hz desktop. Handling is described in [METHODOLOGY.md](METHODOLOGY.md) §6.
- **WSL interop.** WSL 2.7.3 registers binfmt `WSLInterop` once per VM boot. systemd 262's
  `systemd-shutdown` unregisters all binfmt entries whenever the distro instance stops, and on Arch
  `systemd-binfmt.service` is skipped because `/etc/binfmt.d` is empty. After any instance restart,
  Windows `.exe` calls from WSL then fail with "Exec format error".
  - Fix: `/etc/binfmt.d/WSLInterop.conf` containing `:WSLInterop:M::MZ::/init:P`, so WSL's drop-in
    re-registers it on every instance boot.
  - Tooling should check interop before any step that calls Windows binaries.
- **Single copy engine under WSL2** (`asyncEngineCount=1`). No hardware decompression is available
  (mask 0), and NVFP4 E2M1 data does not compress (3.97 bit/code).

## 11. NOT_RUN / NOT_MEASURED

- Nsight Compute inside the model process (2 attempts; host RAM). Replaced by component ncu (§4.1).
- 2-hour soak of long256; the 30-minute test replaced it. No soak or matrix of baseline64 on f1.
- Teacher forcing against a 256K Triton reference (`bf256`).
- External end-to-end test from outside the LAN. The tool-use cycle did use the public HTTPS route.
- Multi-client concurrency, per-token ITL distribution, separate queue-wait time, CUDA-graph memory of
  the release, and the split of foreign vs CUDA-context VRAM.
- FP8 KV quality (teacher, eval); PPL/eval of the RMSNorm fix; teacher gate of the short-prompt route.
- GPU A/B of sync-lean and refill on the final release build.
- The upstream patches in `docs/upstream-notes/patches/` were never run on an unmodified upstream
  checkout.

## 12. File map of `results/`

| Path | Contents |
|---|---|
| `dashboard/qwen36-release-f1.html` | Release dashboard (Russian), self-contained |
| `metrics/metrics.json`, `metrics/METRICS.txt` | All release metrics with source + definition per value; text summary |
| `nsight/` | `dashboard.json` (scenarios, kernels, PCIe), `levers.json`, `decode-step-stats.json`, `ncu-component-kernels.json` |
| `matrix/` | f1 5-pair comparison + matrix manifest; the earlier 256K matrix comparison |
| `soak/` | 2 h soak analysis |
| `context-gates/` | C0 / L4 / F1 gate evaluations |
| `quality/` | Teacher-forcing quality reports, clean-cache confirmations C/D, harness-v7 result |
| `ab64/` | Like-for-like 64K A/B summaries |
| `fp8kv/` | FP8 KV speed screens and predictions |
| `long256-30min/` | 256K stability test metrics and analysis |
| `tool-use/` | Tool-use cycle summary and redacted results |
| `experiments/` | FLA, short route, commit race, 256K deviation |
