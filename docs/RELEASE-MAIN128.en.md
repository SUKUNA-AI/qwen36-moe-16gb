# Release main128: the cumulative ladder (draft)

[Русский](RELEASE-MAIN128.md)

> **Draft.** There are no ladder numbers yet: the GPU runs are in progress. The tables below are
> placeholders. Charts will appear in `results/ladder/charts/` after the runs. The draft charts in
> `results/ladder/charts-draft/` are drawn from a synthetic fixture and are not results.

## What is in it

Profile main128 (context 131072 tokens). Each step adds one lever to the previous one. Every lever is
off by default and is switched on by configuration flags.

| Step | Lever | What it does | Config / flags |
|---|---|---|---|
| L0 | main-release (f1) | the f1 release, the reference point | `harness/main-release.json` |
| L1 | +T1 | hit-D2D: experts already in VRAM are copied on the GPU; layer-major prefill from 2049 tokens | `harness/main-release-shortroute.json` |
| L2 | +T2 | fused CUDA GDN kernel (FLA `wu`) in prefill | `harness/main-release-shortroute-fla-wu.json` |
| L3 | +T3 | decode: the expert gather overlaps the shared expert of the same layer | `harness/main-release-shortroute-fla-wu-overlap.json` |
| L4 | +host embeddings | the BF16 embedding table (0.95 GiB) moves to pinned host memory; the freed VRAM goes to expert slots | `QWEN36_HOST_EMBEDDINGS=1`, `harness/main-release-next-he.json` |
| L5 | +memory_ratio 0.85 | more expert cache slots | `harness/main-release-next-he-r085.json` |
| L6 | +S3 kernels | CUDA decode kernels: o-proj GEMV and MoE GEMV replace cuBLAS/Triton, bit for bit | `QWEN36_OPROJ_GEMV=1`, `QWEN36_MOE_DECODE_CUDA=1` |
| L7 | +C0 | copy-stream priority and buffer invalidation once per prefill epoch | `QWEN36_PREFILL_COPY_PRIORITY=1`, `QWEN36_PREFILL_EPOCH_INVALIDATE=1` |
| L8 | +C1 = candidate | a short prefill copies only the experts the router selected | `QWEN36_PREFILL_SELECTED=160` |
| T3 alone | f1 + T3 | T3 without the other levers, a separate arm | `harness/main-release-overlap.json` |

The configs of L6–L8 are recorded in `results/ladder/summary.json` as run.

## How it was measured

- **Teacher gate at every step.** 20 cases, 943 teacher-forced tokens, 125 logit checkpoints. The
  comparison with the reference is bitwise: 125/125 and `max_abs` 0 are required. A step without a
  PASS does not go into the candidate.
- **A/B at every step: 3 pairs.** The step against the previous one in the same session, with the arm
  order alternating between rounds. Raw per-pair values are kept for every prompt length
  (64 / 2K / 4K / 16K / 120K). A step's contribution is the median of the "step / previous" ratios
  within a pair. That keeps bus drift between server starts (S4 saw H2D vary from 29 to 40 GB/s) out of
  the estimate.
- **Release A/B: 5 pairs.** Candidate L8 against f1 with `harness/compare-ab.py`: the geometric mean of
  the 5 paired ratios with one-sided 97.5 % Student-t bounds. Elapsed is gating; TTFT and decode are
  reported as observations.
- **Soak.** The gate is relative to the f1 soak. It fails on Windows shared GPU memory growing by more
  than 256 MiB (paging) and on TTFT or decode medians getting more than 3 % worse. Free VRAM is report
  only.
- **Nsight Systems at every step.** Critical path of the decode step at 16K, of the short-prompt prefill
  and of the 16K prefill. The buckets are those of `results/nsight/dashboard.json`: MoE experts, FP8
  dense and o-proj, gather, attention, GDN, norms/lm_head/router, exposed H2D, GPU idle. Absolute times
  are inflated under the profiler, so nsys is read for structure only.

### Owner's VRAM rule

1 GiB of VRAM stays with Windows; everything else goes to the model. The process peak (peak used memory
minus memory used before start) must fit into "total VRAM − 1 GiB". Gates fail on paging (shared GPU
memory growth) and on slowdowns, not on a free-memory floor. Free VRAM is shown for information.

## Steps

Numbers are A/B medians. In brackets: the step's contribution against the previous step (median of
paired ratios). `tools/ladder-charts.py` writes the tables to `results/ladder/charts/ladder-table.en.md`;
they are copied here after the runs.

### TTFT, s

| Step | 64 | 2K | 4K | 16K | 120K |
|---|---|---|---|---|---|
| L0 f1 | … | … | … | … | … |
| L1 +T1 | … | … | … | … | … |
| L2 +T2 | … | … | … | … | … |
| L3 +T3 | … | … | … | … | … |
| L4 +host embeddings | … | … | … | … | … |
| L5 +memory_ratio 0.85 | … | … | … | … | … |
| L6 +S3 kernels | … | … | … | … | … |
| L7 +C0 | … | … | … | … | … |
| L8 +C1, candidate | … | … | … | … | … |
| T3 alone | … | … | … | … | … |

### Decode, tok/s

| Step | 64 | 2K | 4K | 16K | 120K |
|---|---|---|---|---|---|
| L0 f1 | … | … | … | … | … |
| L1 +T1 | … | … | … | … | … |
| L2 +T2 | … | … | … | … | … |
| L3 +T3 | … | … | … | … | … |
| L4 +host embeddings | … | … | … | … | … |
| L5 +memory_ratio 0.85 | … | … | … | … | … |
| L6 +S3 kernels | … | … | … | … | … |
| L7 +C0 | … | … | … | … | … |
| L8 +C1, candidate | … | … | … | … | … |
| T3 alone | … | … | … | … | … |

### Teacher and memory

| Step | Teacher | memory_ratio | Expert slots | Process peak, MiB | Misses per token (16K) |
|---|---|---|---|---|---|
| L0 f1 | … | 0.75 | … | … | … |
| L1 +T1 | … | 0.75 | … | … | … |
| L2 +T2 | … | 0.75 | … | … | … |
| L3 +T3 | … | 0.75 | … | … | … |
| L4 +host embeddings | … | 0.75 | … | … | … |
| L5 +memory_ratio 0.85 | … | 0.85 | … | … | … |
| L6 +S3 kernels | … | 0.85 | … | … | … |
| L7 +C0 | … | 0.85 | … | … | … |
| L8 +C1, candidate | … | 0.85 | … | … | … |
| T3 alone | … | 0.75 | … | … | … |

### Release A/B: candidate against f1, 5 pairs

| | 64 | 2K | 4K | 16K | 120K |
|---|---|---|---|---|---|
| TTFT, candidate / f1 | … | … | … | … | … |
| Decode, candidate / f1 | … | … | … | … | … |
| compare-ab verdict | … | | | | |

## Charts

The chart labels are in Russian, like the README charts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/teacher-steps-dark.svg">
  <img alt="Teacher gate per step" src="../results/ladder/charts/teacher-steps.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/ttft-steps-dark.svg">
  <img alt="TTFT per step for 64, 2K, 4K, 16K and 120K token prompts" src="../results/ladder/charts/ttft-steps.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/decode-steps-dark.svg">
  <img alt="Decode speed per step" src="../results/ladder/charts/decode-steps.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/waterfall-ttft-short-dark.svg">
  <img alt="Contribution of each lever to short-prompt TTFT" src="../results/ladder/charts/waterfall-ttft-short.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/waterfall-decode16k-dark.svg">
  <img alt="Contribution of each lever to decode speed at 16K" src="../results/ladder/charts/waterfall-decode16k.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/nsys-decode16k-dark.svg">
  <img alt="Decode step breakdown at 16K per step, Nsight Systems" src="../results/ladder/charts/nsys-decode16k.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/nsys-prefill-short-dark.svg">
  <img alt="Short-prompt prefill breakdown per step, Nsight Systems" src="../results/ladder/charts/nsys-prefill-short.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/nsys-prefill16k-dark.svg">
  <img alt="16K prefill breakdown per step, Nsight Systems" src="../results/ladder/charts/nsys-prefill16k.svg" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../results/ladder/charts/memory-steps-dark.svg">
  <img alt="VRAM, expert slots and misses per token per step" src="../results/ladder/charts/memory-steps.svg" width="100%">
</picture>

## Regenerating

```bash
python3 tools/ladder-charts.py
```

The script reads `results/ladder/summary.json` (schema `qwen36-ladder-summary` v1, described at the top
of `tools/ladder-charts.py`) and writes 18 SVG files (light and dark) plus the tables `ladder-table.md` /
`ladder-table.en.md` to `results/ladder/charts/`. No third-party packages are needed. Empty fields are
drawn as "н/д" (n/a), so the file can be filled as runs finish. Synthetic input (`"synthetic": true`) is
never written to `results/ladder/charts/`, only to `charts-draft/`.
