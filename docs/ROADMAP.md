# Roadmap

Every item must stay **bit-exact** against the teacher reference, unless it is explicitly offered as a
separate, separately qualified profile. Every item goes through the same gates as f1
([METHODOLOGY.md](METHODOLOGY.md)).

## Next update (owner decision, 2026-10-14)

### 1. Short-prompt route: hit-d2d (patch ready)

- **Problem.** Prompts with 2048 or fewer uncached tokens take the chunked path. That covers short
  prompts, prefix hits and edits. The chunked path copies the full 18.25 GB expert bank H2D for every
  chunk, which sets a TTFT floor of ~0.57 s. A 64-token prompt already touches most of the 256 experts
  per layer.
- **Lever.** `--moe-prefill-hit-d2d` copies the ~23% of experts that are already resident in decode
  slots on the GPU (device to device) instead of over PCIe. Layer-major stays on for prompts of 2049
  tokens and more.
- **Measured** with an import-time shim: 10 rounds, 2 sessions, rotated order.

  | Case | Release | hit-d2d |
  |---|---|---|
  | 64-token prompt | 0.565 s | 0.446 s |
  | prefix hit | 0.585 s | 0.468 s |
  | 16K edit | 0.596 s | 0.481 s |
  | 2K-16K fresh | unchanged (ratios 0.998-1.007) | |

  Greedy outputs 200/200 identical; decode unchanged.
- **Expected matrix effect.**
  - 64-token TTFT 0.79 -> ~0.67 s, ahead of bf128's 0.75 s.
  - Edits 0.78 -> ~0.66 s.
- **Remaining before release:**
  - the teacher gate (the hit-d2d path itself is not yet qualified);
  - relaxing the profile-runtime rule that refuses layer-major together with hit-d2d;
  - a native sync.
- **Patch.** Touches `engine/layer_major.py`, `moe/offload_cache.py` and `scheduler/layer_major.py`.
  The env switch is `QWEN36_LAYER_MAJOR_MIN_TOKENS`; unset keeps the current behaviour. 7 new CPU tests
  pass plus 59 existing ones.

### 2. Hide the decode expert gather (overlap), not a faster copy

- **Measured ceiling (2026-10-15).** A byte-checked sweep (728 measurements, 0 mismatches) shows every host-to-GPU path on this WSL2 host tops out at 41-45 GB/s:
  - copy engine: 40.8 GB/s;
  - TMA `.shared::cta`: 43.4 GB/s, peak 45.0;
  - mapped loads: 41-44 GB/s at 4-8 CTAs.

  More CTAs, streams or bytes in flight *lower* throughput, which points to a host-side limit. Native Linux is untested.
- **Today.** The gather already runs at ~42 GB/s inside the model (38.9 / 41.7 / 42.8 GB/s for 1 / 4 / 8 experts). The 30-33 GB/s in the Nsight Compute table is the kernel measured under the profiler. The gather sits serially on the decode critical path: 18% of a 16K decode step, 14% at 120K.
- **Next.**
  - Overlap the routed-expert gather with the same layer's shared-expert MLP: side stream plus an explicit event join, captured in the decode graph. Byte-identical by construction. Estimate +3-4% decode tok/s, about 5 h.
  - Then predicting layer L+1's experts with its router one layer early. This needs a route trace first: previous-token prefetch catches nothing, because those experts are already resident.
  - Any cross-stream scheme needs the join. The negative control without it produced 14,248 checksum mismatches.

### 3. Enable fused FLA (`QWEN36_CUDA_GDN_FLA=wu`)

- The kernel is already shipped flag-off and is bit-exact 123/123.
- The nsys structure check confirms the model-level effect:
  - GDN core time -36% at 16K, i.e. -2..3% TTFT;
  - -0.9% of the window at 120K.
- It needs a model-level paired A/B at the production chunk size and an owner decision.

## Deferred (measured, not scheduled)

| Item | Status / reason |
|---|---|
| FP8 KV cache | Not now: non-bit-exact, TTFT +10-33%; worth it only for long agent sessions. Revisit as an optional `long256-fp8` profile after the FP8 prefill-attention TTFT cost is fixed ([RESULTS.md](RESULTS.md) §7) |
| Sync-lean (FLA chunk-table cache, FlashInfer plan reuse) | Bit-exact; 0.1-0.4%, below noise; needs a rebase on attention batching |
| Layer-major slot refill (`restore` policy) | Bit-exact; no gain within noise at 128-token outputs; re-test with decode-heavy long outputs |
| MoE decode (marlin) / o-proj GEMV kernels | ~+11% decode at 16K estimated (o-proj alone ~+4%); owner: not wanted now |
| Bit-exact decode-attention replica | Est. -10..15% decode attention (~3-5% decode at 120K-256K) |
| `(1+w)` RMSNorm FP32 fix | Needs a new reference + PPL / functional / eval re-runs; not scheduled |
| 2-CTA dedup in fused FLA | <= 0.4%; not worth it |
| 256K: 2 h soak, `bf256` teacher | NOT_RUN in f1 |
| MTP / speculative decoding, vision | Off by owner decision; MTP only with new evidence |

Harness and ops findings to fix:
- `compare-quality` should bind teacher rows to a run id.
- `teacher.jsonl` rotation at 2 MiB will eventually move the reference rows into `.1`.
- `profile-request` should wait for `/health`, not `/v1/models`.
- Add a regression unit test that asserts the scheduler stream waits from the commit/decode race
  analysis.
- Run a WSL interop check before release steps that call Windows binaries.

## Upstream issue drafts (never posted)

The drafts live in [upstream-notes/](upstream-notes/00-INDEX.md). Nothing has been opened, posted or
submitted; the owner decides. File and line references point to FreeToken `cac247a8`. Paths mentioned
inside the drafts, such as `work/...` and `outputs/...`, refer to the private project workspace.

| # | Draft | Kind | Ready? |
|---|---|---|---|
| 1 | NVFP4 `weight_scale_2` stored as FP16, 94% subnormal | bug / precision | yes; patch included |
| 2 | Qwen3.5 `(1+w)` RMSNorm folded into BF16 at load | bug / precision | yes, stating that whole-model PPL of the fix was not run |
| 3 | Greedy outputs depend on cache history (header dedup re-point, BF16 GDN snapshot); ask for a flush endpoint / deterministic mode | feature / discussion | yes |
| 4a | MTP / speculative decoding prototype notes | discussion | no |
| 4b | RAM complement residency (negative result) | negative result | no |
| 4c | Optional FP8 KV cache | feature request | optional |

The sm_120 `.shared::cluster` syscall finding ([ARCHITECTURE.md](ARCHITECTURE.md) §5.3) is not an
upstream FreeToken issue, because upstream has no bulk copies. It is documented here only.
