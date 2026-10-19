<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Offloaded decode: overlap the expert-miss gather with the same layer's shared expert, event-ordered inside the CUDA graph (bit-exact implementation)

## Summary

Related: #678, which proposes the same idea; this issue describes our independent, bit-exact, CUDA-graph-captured
implementation and offers it as a PR. In offloaded GPU decode, each layer runs ensure, miss copy, shared expert
and routed GEMM back to back on one stream, but the shared expert does not depend on the miss copy. We move the
**miss gather** to a side stream, keep the **shared expert** on the compute stream, and fork and join with two
events that are captured into the decode graph. On our setup, decode at 16K context went from 143.6 to 148.6 tok/s
(+3.5%, ladder median of 3 pairs). Teacher forcing stays bit-exact (125/125 cases, max_abs 0). A negative control
without the join gives 14,248 checksum mismatches.

## Setup

- RTX 5070 Ti 16 GB (sm_120), PCIe 5.0 x16, WSL2; the Windows desktop shares the GPU
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`: 40 MoE layers x 256 experts, top-8, plus a gated shared expert
- FreeToken 0.1.3 `cac247a8` + our fork; `--moe-strategy offload`, LRU slot cache (3164 slots, ~28 misses per
  token at 16K), our TMA decode gather, CUDA-graph decode, batch 1

## Problem in upstream today

All paths are under `python/freetoken/`, at `main` `4d2e2e2`.

- `Qwen3_5MoE.forward` computes the router, then the shared expert and its sigmoid gate, then the routed experts,
  all on the current stream (`models/qwen3_5_moe/moe.py:78-82`).
- The routed path runs `ensure_experts`, then `copy_missing` (the PCIe gather of missed experts), then the GEMM,
  serially (`layers/moe.py:284-295`).
- `main` has no side stream in the MoE layers or the offload cache, apart from the prefill copy stream
  (`moe/offload_cache.py:824`). #678 is open, and its implementation lives on a fork.

## What we changed

Switch `QWEN36_DECODE_SHARED_OVERLAP=1` (unset/0 = serial; other values raise). Modules in our fork:
`decode_overlap.py` (new, about 60 lines), `layers/moe.py` (`can_overlap_shared`, `forward_with_shared`),
`models/qwen3_5_moe/moe.py`, `moe/offload_cache.py`.

Order inside one layer:

```
compute: router -> fused_topk -> ensure_experts -> record(ready) -> shared expert + gate -> wait(done) -> routed GEMM -> routed + shared
side:                                              wait(ready)   -> copy_missing (gather) -> record(done)
```

- **Who writes what.** `ensure_experts`, which reserves LRU slots, rewrites ids and writes the miss plan, stays on
  the compute stream, so the slot map has a single writer. The side stream only reads the plan and writes slot
  bytes after `ready`. The routed GEMM, and the next layer's `ensure_experts` that may reuse those slots or plan
  buffers, run after `done`.
- **Graph capture.** The side stream and both events are created and materialised once at cache construction,
  before any capture. Each `wait(ready)` forks the capture and `wait(done)` rejoins it inside the same captured
  forward, so no side work outlives a layer. Replay uses fixed addresses and device-side miss counts, and there
  is no host sync. The join runs in a `finally`, so a Python error in the shared callback cannot leave the side
  stream unjoined.
- **Bit-exactness.** The kernels, the dtypes and the final `(routed + shared)` order are unchanged; only where
  the gather runs differs. The shared expert keeps reading the original hidden state before the routed GEMM,
  which may overwrite it in place.
- **Fallback to serial:** prefill, batch != 1, TP != 1, CPU/hybrid decode targets, unpinned layers, a non-fused
  copy layout, an active layer-major epoch, and our slot-refill feature (configured refill disables the overlap
  before capture).
- **Difference from #678:** there the miss copy stays on the compute stream and the shared expert moves to the
  side stream. Our split keeps every slot-map write and every consumer on the compute stream, with only one
  producer-consumer edge pair per layer. #678 notes that a pooled `torch.cuda.Stream()` can alias the capture
  stream; we use a pooled stream and did not test for aliasing. In our ordering an alias would serialize the
  layer (lose the overlap), not reorder it.

## Measured impact

Model-level ladder (`results/ladder/summary.json`; 3 pairs, rotating order, 60 s sustained warm-up, 128 output
tokens, no profiler). Each step adds one change to the previous one. Values are median (mean), n = 3.

| Step | Decode 16K, tok/s | Decode 120K, tok/s | Decode, 64-token doc | Decode step 16K (Nsight, ms) |
|---|---|---|---|---:|
| L2 (before) | 143.6 (142.0), 137.9-144.6 | 103.4 (103.7), 102.4-105.3 | 151.5 (150.4) | 7.02 |
| L3 = L2 + overlap | 148.6 (144.9), 135.5-150.5 | 106.1 (104.9), 99.8-108.9 | 158.8 (156.5) | 6.82 |
| f1 base (L0) | 143.4 (142.2), 137.7-145.6 | 103.1 (103.8), 102.4-105.9 | 151.8 (148.8) | 7.07 |
| overlap alone on f1 | 142.1 (143.3), 140.8-147.1 | 105.2 (106.0), 104.6-108.1 | 149.8 (152.1) | 6.85 |

- In the L3 step, the first pair was slower than its reference (135.5 vs 137.9 at 16K); the other two were
  faster. The overlap alone on f1 was faster in 2 of 3 pairs at 16K and in 3 of 3 at 120K (+2.0-2.1%). Treat
  the 16K result for this arm as inconclusive.
- Our estimate before implementing was +3-4% decode (`docs/RESULTS.md` §9). The ladder shows +5.0 tok/s (+3.5%)
  at 16K.
- Giving the side stream the highest priority (`QWEN36_DECODE_OVERLAP_PRIORITY=1`) made the decode step
  0.5-2% worse (2 pairs; `docs/RESULTS.md` §8). It stays off.
- Unexplained: the "overlap alone on f1" arm also showed a slower 64-token TTFT (0.428 vs 0.390 s). The prefill
  path never takes the overlap, and we did not investigate this.

## Correctness evidence

- Teacher forcing (logits vs the reference, 125 cases) at L3 and in the "overlap alone" arm: 125/125, max_abs 0,
  all outputs byte-equal.
- GPU component test `tests/moe/test_decode_overlap_gpu.py`, 4/4 without skips. It uses the real
  `ensure_experts`/LRU/gather with the real six-bank NVFP4 row sizes in pinned memory, and byte-checking
  consumers. It covers the default and TMA gathers, 1 and 3 layers, eager mode, and two captured graphs replayed
  over 180 steps with changing routes and a cancel.
- Negative control: an archived run without the join gave **14,248** checksum mismatches
  (`next-update/gather-prefetch/OVERLAP-DESIGN.md`).
- A CPU test with fake streams (`harness/test-decode-overlap-cpu.py`) covers the control flow, the disabled and unsupported modes, and callback errors.
  It also detects the removal of either the producer wait or the consumer wait.

## Limitations / not measured

- sm_120, PCIe 5.0, one model, batch 1, LRU eviction. Not combined with scored eviction (#174) as in #678.
- Nsight Compute serialises kernels, so two-stream capture hung under ncu: profile with
  `QWEN36_DECODE_SHARED_OVERLAP=0`.
- 3 pairs per step; ladder steps share one session. Profiler timings are diagnostics only.

## Proposal

We can send a PR: the `DecodeSharedOverlap` helper, a `forward_with_shared` path in `OffloadMoELayer`, and a
default-off server flag, with the GPU byte test and its negative control. If you prefer #678's split (shared
expert on the side stream), we are happy to run that variant through the same teacher gate and A/B on our setup.
