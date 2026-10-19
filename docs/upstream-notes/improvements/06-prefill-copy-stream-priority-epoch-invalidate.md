<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Offloaded prefill: keep buffer invalidation off the prefill copy stream and give that stream top priority (short-prompt H2D 21 -> 42 GB/s under Nsight)

## Summary

**Partly in upstream already:** `--moe-prefill-hit-d2d` (cached experts gathered device-to-device, misses as one
`cudaMemcpyBatchAsync`) is on `main`, default off. The race in its `cudaMemcpyBatchAsync` probe that silently
disables it is fixed by open PR #414, not merged on `main` `4d2e2e2`. This issue is about a separate,
smaller change, which we call C0. On upstream, each layer's prefill copy is queued behind a few small bookkeeping
kernels on the same stream. Those kernels wait for SM space behind the previous layer's MoE GEMM, so the copy
engine idles between layers. We make two changes, both opt-in. The prefill copy stream gets the highest
priority, and both double buffers are invalidated once per prefill on the compute stream, so the copy stream
carries only DMA work. On our setup, a short prompt's copy runs at 42.2 instead of 21.4 GB/s under Nsight, and
short-prompt TTFT drops to 0.76-0.94x. Outputs are bit-identical.

## Setup

- RTX 5070 Ti 16 GB (sm_120), PCIe 5.0 x16, WSL2 (one copy engine); the Windows desktop shares the GPU
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`: 40 MoE layers x 256 experts, 18.25 GB per full-bank chunk
- FreeToken 0.1.3 `cac247a8` + our fork; `--moe-strategy offload` with prefill overlap and `--moe-prefill-hit-d2d`,
  3164-4603 expert slots, batch 1, CUDA-graph decode

## Problem in upstream today

All paths are under `python/freetoken/`, at `main` `4d2e2e2`.

- The prefill copy stream is created at default priority: `torch.cuda.Stream(device=self.device)`
  (`moe/offload_cache.py:824`).
- `prefetch_prefill_layer` runs `_invalidate_prefill_buffer` inside `with torch.cuda.stream(self.prefill_copy_stream)`,
  right before each layer's DMA (`:894-897`, `:904-908`; the hit-D2D split path does the same at `:1000-1003`).
- `_invalidate_prefill_buffer` (`:841-853`) is several small device ops per layer: index reads, a masked scatter
  into `slot_for_id`, fills, `usage` zeroing, and zeroing of the stage tails. The masked index
  `old_ids[old_ids >= 0]` (`:845`) also needs a device-to-host sync.
- At equal stream priority, these kernels' CTAs are scheduled behind the previous layer's expert-GEMM blocks.
  The DMA queued after them on the copy stream waits too.

Our measurement was on our `cac247a8`-based fork, where the same function issues 6 small kernels per layer. In a
narrow Nsight capture of a fresh short prompt, the copy stream ran 240 such kernels per prefill. Together they
waited 55-60 ms in the queue, and the copy bursts had ~1.4 ms gaps per layer (`docs/RESULTS.md` §9). We have not
re-profiled `main`, whose invalidation code has since gained stage-tail handling.

## What we changed

- `QWEN36_PREFILL_COPY_PRIORITY=1`: the copy stream is created with `priority=-1` (torch maps it to the device's
  highest priority).
- `QWEN36_PREFILL_EPOCH_INVALIDATE=1`: `begin_prefill` invalidates **both** double buffers once, on the compute
  stream. This is ordered after any decode that mapped buffer slots and before the existing begin fence, so every
  copy in this prefill lands in unmapped slots. Per-layer prefetches then skip the invalidation. This is valid
  because no slot-cache writer runs inside a prefill epoch: the buffers stay unmapped until the last layer's
  release. An `_prefill_epoch_open` flag tracks the epoch. With the switch off, per-layer invalidation is
  unchanged.
- Module: `moe/offload_cache.py` in our fork. No kernels and no numerics change; GEMM inputs are byte-identical.
- In our fork the hit-D2D route only covers chunks below 2049 uncached tokens, because longer prompts use our
  layer-major prefill. Upstream has no such split.

## Measured impact

Nsight Systems, fresh 64-token prompt (`next-update/transport/results/nsys-c0-*.h2d.json`, case `f64b`; times
under the profiler are diagnostics only):

| Arm | Prefill window, ms | H2D GB/s while copying | Kernels on copy stream | Their queue time, ms |
|---|---:|---:|---:|---:|
| hit-D2D only (base) | 907.4 | 21.4 | 240 | 61.3 |
| + priority | 518.5 | 35.6 | 240 | 9.9 |
| + epoch invalidation | 409.3 | 44.2 | 0 | 0 |
| + both (C0) | 429.2 | 42.2 | 0 | 0 |

TTFT screen without the profiler, ratio vs the base in the same round (`next-update/transport/results/ab-ttft.json`;
C0 3 pairs, single arms 1 pair). Outputs were identical in all cases.

| Case (uncached tokens) | C0: median (each pair) | priority only | epoch only |
|---|---|---:|---:|
| fresh 64-token doc (125) | 0.844 (0.844 / 0.767 / 0.883) | 0.843 | 0.844 |
| fresh 512 (574) | 0.931 (0.931 / 0.820 / 0.941) | 0.912 | 0.914 |
| new turn (115) | 0.850 (0.850 / 0.763 / 0.882) | 0.844 | 0.839 |
| prefix + 26 new tokens | 0.927 (0.927 / 0.814 / 0.944) | 0.919 | 0.915 |
| fresh 1024 (1087) | 0.913 (0.913 / 0.801 / 0.932) | 0.892 | 0.895 |

- 120K, which takes our layer-major path: ratio 0.998 (0.999 / 0.997, 2 pairs; `ab-long120k.json`), so long
  prompts saw no regression.
- Model-level ladder step L7 (`results/ladder/summary.json`; 3 pairs, 60 s warm-up, 128 output tokens, no
  profiler), median (mean), n = 3:
  - TTFT, 64-token document: 0.270 (0.287) -> 0.252 (0.254) s;
  - TTFT 16K: 1.321 (1.347) -> 1.315 (1.316) s;
  - decode 16K: 170.3 (169.2) -> 171.7 (172.3) tok/s.

  The reference step had one slow pair (0.334 s), so its mean overstates the gain.
- For context, the existing hit-D2D route with the probe fixed (ladder step L1): TTFT for the 64-token document
  went from 0.390 (0.399) to 0.315 (0.312) s, with 16K/120K unchanged. Per short chunk, H2D fell from 18.25 GB to
  14.07 GB plus 4.17 GB D2D, at ~23% of each layer resident. Microbench (10 rounds, 2 sessions,
  `results/experiments/short-route-summary.json`): 0.565 -> 0.446 s on a fresh 123-token prompt; 200/200 greedy
  outputs identical.

## Correctness evidence

- Teacher forcing at ladder step L7: 125/125 cases, max_abs 0 on logits, all outputs byte-equal. Screen outputs
  identical in every case. GPU tests: `tests/moe/test_prefill_c0_gpu.py` in our fork.
- Probe race (#414): we hit it independently. It disabled hit-D2D for the whole process. Our test keeps the
  current stream busy and runs 50 probes; it fails deterministically on the old probe and passes with
  `stream.wait_stream(torch.cuda.current_stream())` (`docs/RESULTS.md` §10). +1 for merging #414.

## Limitations / not measured

- sm_120, one copy engine (WSL2), batch 1, one model. Multi-request batches and cancellation in the middle of an
  epoch go through the existing fences; we did not stress them separately.
- Not measured on `main`'s per-format slot pools, hybrid or CPU decode targets, or with other high-priority
  streams in the process.
- The main gain comes from the invalidation move. Priority alone helps less under Nsight (518 vs 409 ms) but
  about as much in the TTFT screen.

## Proposal

We can send a PR with both switches default off (names up to you), ported to the pool-aware
`_invalidate_prefill_buffer`. Open questions: should epoch invalidation become the default whenever no slot
writer can run mid-prefill? And do you want the masked-index host sync at `:845` replaced by a sync-free
`masked_fill_`? We have a flag-off variant of that, but did not measure it separately as part of C0.

**Our implementation, for reference** (fork of FreeToken 0.1.3, Apache-2.0, pinned commit; feel free to reuse):

- [`moe/offload_cache.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/moe/offload_cache.py)

Evidence and write-ups (`docs/RESULTS.md` is in Russian):

- [`docs/RESULTS.md`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/docs/RESULTS.md)
- [`next-update/transport/results/ab-ttft.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/next-update/transport/results/ab-ttft.json)
- [`results/ladder/summary.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/results/ladder/summary.json)
- [`results/experiments/short-route-summary.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/results/experiments/short-route-summary.json)
