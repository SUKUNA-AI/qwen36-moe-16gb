<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Offloaded decode: gather missed experts with a few TMA bulk-copy CTAs (`cp.async.bulk`, `.shared::cta`) instead of 48 x 1024 LDG threads

## Summary

In offloaded GPU decode, each layer's missed experts are copied from pinned host banks into slot-cache slots by
`fast_index_copy_multi`: 8 CTAs of 1024 threads per bank, each thread with one 16-byte load in flight. Our fork
adds an opt-in, byte-exact drop-in that spreads the call's total missed bytes over 8 CTAs. Each CTA streams 16 KiB
chunks through a 3-stage shared-memory ring with `cp.async.bulk` (TMA). On our setup the gather is 1.12-1.30x
faster per call, and decode is +5.6% at 16K and +4.8% at 120K context (median of 3 paired rounds). In the model
the gather then runs at ~42 GB/s, which is at the 41-45 GB/s host-link ceiling we measured on this host. One
toolchain pitfall matters for anyone who writes such a kernel for sm_120: the `.shared::cluster` destination form
silently costs 1.37 GiB of VRAM. The `.shared::cta` form avoids it.

## Setup

- RTX 5070 Ti 16 GB (sm_120), PCIe 5.0 x16, WSL2 (one copy engine, `asyncEngineCount=1`); the Windows desktop
  shares the GPU
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`: 40 MoE layers x 256 experts, top-8, 1,781,760 B per expert in
  6 banks (gate_up packed/scale/global, down packed/scale/global)
- FreeToken 0.1.3 `cac247a8` + our fork; `--moe-strategy offload`, pinned host banks backed by transparent huge
  pages, CUDA-graph decode, batch 1, 3164 expert slots (26-39 misses per token at a miss rate of 0.08-0.12)

## Problem in upstream today

All paths are under `python/freetoken/`, at `main` `4d2e2e2`.

- Decode calls `cache.ensure_experts(...)` then `cache.copy_missing()` before the routed GEMM
  (`layers/moe.py:284-285`). `copy_missing` launches `fast_index_copy_multi_jit` over all banks with
  `blocks_per_bank=self._copy_blocks_per_bank` (`moe/offload_cache.py:1239-1255`).
- `default_blocks_per_bank` returns 8 on PCIe hosts and 32 on GH200/GB200/GB300, or the value of
  `FREETOKEN_H2D_BLOCKS_PER_BANK` (`kernel/fast_index_copy.py:152-168`; set at `moe/offload_cache.py:650`).
  The prebuilt variants are `(1024, 8)`, `(1024, 32)` and `(1024, 64)` (`kernel/aot.py:164-167`).
- Each thread copies with `ld.global.L1::no_allocate.v4.b32` / `st.global.wt.v4.b32`
  (`kernel/csrc/jit/fast_index_copy.cuh:70`, `:104`). With 6 NVFP4 banks that is 48 CTAs x 1024 threads, about
  768 KiB of 16-byte reads in flight. Every bank gets the same number of CTAs: the ~1 MiB gate_up row gets 1/6 of
  the threads while the threads of the 4 KiB global rows idle.
- `git grep` finds no `cp.async.bulk` and no TMA bulk copy in upstream's own CUDA sources.

On our host, the mapped-host LDG path is sensitive to how many small reads are in flight. Our byte-checked
ceiling sweep (below) measured at most 26.1 GB/s for plain LDG gathers at 70-280 CTAs (1 expert). The sweep did
not include upstream's exact 48 x 1024 configuration. Upstream's own docstring at `fast_index_copy.py:153-155`
describes the same bandwidth-delay trade-off for NVLink-C2C.

## What we changed

- Switch: `QWEN36_DECODE_GATHER=tuned` (default unset = upstream kernel; any other value raises, so a typo cannot
  silently select the default). Modules: `kernel/decode_gather.py`, `kernel/csrc/jit/decode_gather.cuh`,
  wired into the offload cache's decode miss path only. Whole-layer `materialize_layer` copies and the prefill
  hit-D2D gather are untouched.
- Kernel `DecodeGatherTmaKernel<8, 3, 16384>`: the call's rows x banks are cut into chunks of up to 16 KiB, each
  inside one bank row. The chunks are handed out round-robin to 8 CTAs. One thread per CTA runs a 3-stage ring:
  `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` from mapped host memory into shared memory,
  then a bulk store into the slot. That is 384 KiB in flight and 48 KiB of shared memory per CTA (no opt-in).
- Same arguments and same bytes as `fast_index_copy_multi_jit`. The miss count is read on the device, so the
  launch is fixed-shape and CUDA-graph capturable. Limits: at most 8 banks; row sizes must be multiples of 16 B
  and bases 16-byte aligned (the fused copy plan already guarantees both). Pre-sm_90 builds compile a warp-LDG
  fallback of the same chunks, so a multi-arch kernel-cache build still works.
- **sm_120 pitfall.** Our first build used the common `cp.async.bulk.shared::cluster...` form. On sm_120 ptxas
  lowers it to `CALL.ABS.NOINC __cuda_syscall_cp_async_bulk_unicast`. On the first launch the driver raises the
  context's per-thread stack limit from 1024 to 14416 B, which is (14416 - 1024) x 70 SMs x 1536 threads =
  1373 MiB, and never returns it. Free VRAM around graph capture fell from 1.82 to 0.78 GiB with that build. The
  `.shared::cta` destination (PTX ISA 8.6+) lowers to native `UBLKCP`: 0 MiB, identical bytes, unchanged
  speed, and 41 instead of 46 registers. We are reporting the driver behaviour to NVIDIA separately.

## Measured impact

Model-level A/B (our f1 release candidate with the tuned gather vs the same build without it): 3 alternating
rounds, 512 output tokens, identical output content. Source: internal run log, summarised in
`docs/ARCHITECTURE.md` §3.6 (the raw rows are not in `results/`).

| Decode, tok/s | without (3 rounds) | with (3 rounds) | per-round ratio: median (mean), min-max |
|---|---|---|---|
| 16K context | 120.3 / 118.6 / 114.1 | 122.9 / 125.2 / 124.0 | +5.6% (+5.5%), +2.2% to +8.7% |
| 120K context | 86.0 / 85.9 / 84.2 | 89.5 / 90.0 / 90.0 | +4.8% (+5.2%), +4.1% to +6.9% |

The means and ranges of the ratios are derived from the listed rounds. TTFT ratio was 0.96-1.03 (neutral). n = 3
is too small for p95/p99. Together with our FP8 decode GEMV, decode was +13-17% (`docs/ARCHITECTURE.md` §3.6).

Microbenchmarks and diagnostics:

- Per call, on THP banks, reviewer run: 1.12-1.30x for 1-8 missed experts, a mean of 1.22x on a mixed miss
  pattern, and 1.08-1.25x for a captured 16-layer graph. A retuned grid for the old kernel reached at most 0.83 of
  the new kernel's speed. Source: internal review notes; the 1.12-1.30x is quoted in `docs/ARCHITECTURE.md` §3.6.
- In-model rate ~42 GB/s (38.9-42.8 GB/s for 1-8 experts). Under Nsight Compute it shows 30-33 GB/s, with clocks
  fixed at base (`docs/RESULTS.md` §4.1).
- H2D ceiling sweep, one session, 12 reps per cell, every copy verified word-exact
  (`next-update/gather-prefetch/results/ceiling-verdict.json`). The best TMA result was 43.37 GB/s, with 4 CTAs x
  4 stages x 16 KiB = 256 KiB in flight; 8 CTAs gave 40.8-42.1 GB/s on a flat 64 MiB copy. 35-140 CTAs collapsed
  to 12-20 GB/s, as did anything above ~1 MiB in flight. The copy engine reached 40.75 GB/s in that session.
- Nsight Systems, f1 vs an earlier baseline trace (64K profile, not fully comparable, and other changes landed in
  between): the gather time fell from 4.58 to 1.49 ms per decode step (`docs/RESULTS.md` §4).

## Correctness evidence

- Byte-exact on 44,280 verified calls. Adversarial review cases: 0..600 misses, boundary slots and experts, the
  last layer, eager and graph replay. Unit tests: `tests/moe/test_decode_gather.py`.
- Whole model: the f1 release gate B (teacher forcing, 125/125 cases, max_abs 0, PPL +0.0%) ran with the
  gather on (`docs/RESULTS.md` §0). The main128 ladder steps, all with the gather on, also pass 125/125 with
  max_abs 0.

## Limitations / not measured

- Tuned and measured only on sm_120, PCIe 5.0 x16, WSL2 with one copy engine, and THP-backed pinned banks. The
  8 x 3 x 16 KiB config is this host's optimum; the ceiling sweep shows the optimum depends on bytes in flight.
  We did not run it on Hopper or NVLink-C2C (where upstream uses 32 blocks/bank), on native Linux, or on
  4 KiB-page banks in the model.
- Batch 1, one model. Profiler numbers are diagnostics only. The 120K decode gain is from 3 rounds.

## Related

Our short-prefill "router-selected experts only" route reuses this kernel with device-side lists. That overlaps
with #579 / PR #676, so we posted that evidence as a comment there instead of in this issue.

## Proposal

We can send a PR with `decode_gather.cuh` as a JIT module plus its AOT entry, behind a default-off option (for
example `FREETOKEN_DECODE_GATHER=tma`), selected only on sm_90+ with the fused copy plan. Open questions:

1. Should the CTA count and stage depth come from a per-link setting like `FREETOKEN_H2D_BLOCKS_PER_BANK`, or from
   a small startup probe? Our sweep suggests targeting bytes in flight (~256-384 KiB on PCIe 5.0), not threads.
2. Would you take a CI or compile-time check that rejects `.shared::cluster` bulk copies in sm_120 builds?

**Our implementation, for reference** (fork of FreeToken 0.1.3, Apache-2.0, pinned commit; feel free to reuse):

- [`kernel/decode_gather.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/kernel/decode_gather.py)
- [`kernel/csrc/jit/decode_gather.cuh`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/kernel/csrc/jit/decode_gather.cuh)

Evidence and write-ups (`docs/RESULTS.md` is in Russian):

- [`docs/ARCHITECTURE.md`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/docs/ARCHITECTURE.md)
- [`docs/RESULTS.md`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/docs/RESULTS.md)
- [`next-update/gather-prefetch/results/ceiling-verdict.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/next-update/gather-prefetch/results/ceiling-verdict.json)
