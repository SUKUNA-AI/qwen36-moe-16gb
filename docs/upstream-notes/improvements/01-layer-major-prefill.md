<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Layer-major prefill for offloaded MoE: stream each expert layer once per prompt, not once per chunk (qwen3_5_moe)

## Summary

With `--moe-strategy offload`, chunked prefill runs every chunk through all 40 decoder layers, and each layer
visit copies that layer's 256-expert bank into the prefill double buffer. With 2048-token chunks, a fresh 120K
prompt on Qwen3.6-35B-A3B-NVFP4 moves 59 chunks x 18.2 GB = 1076 GB over PCIe, so long prefills become
PCIe-bound once the MoE GEMM is fast. Our fork adds an opt-in **layer-major** prefill. Each layer's bank is
copied once, and every chunk of the prompt goes through that layer before the next layer starts. The chunk
shapes and kernels stay the same, so the output is **bitwise equal** to chunked prefill (teacher forcing
125/125, max_abs 0). On our setup, PCIe traffic per fresh prompt drops to 18.2 GB at any length up to 120K.
An add-on batches 2 adjacent chunks into one FlashInfer prefill call when that is provably bit-identical
(TTFT ratio 0.924 / 0.966 / 0.950 at 16K / 64K / 120K vs one call per chunk).

## Setup

- RTX 5070 Ti 16 GB (sm_120), Linux under WSL2. One copy engine (`asyncEngineCount=1`), about 47 GB/s H2D
  (48.4-49.4 GB/s with huge pages)
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`: 40 layers (10 full attention, 30 GatedDeltaNet), 256 experts,
  top-8. The pinned host expert bank is 18.245 GB (1,781,760 B per expert).
- FreeToken 0.1.3 `cac247a8` + our fork. `--moe-strategy offload`, prefill chunk 2048, single request, context
  131,072, BF16 KV, 3164 GPU expert slots (release f1)

## Problem in upstream today

- `python/freetoken/layers/moe.py:345` (`_prefill_routed`) streams whole layers. Its docstring says:
  "stream whole layers -- double-buffered behind the previous layer's GEMMs". `_wait_prefill_overlap`
  (`layers/moe.py:386-396`) calls `begin_prefill()` at layer 0 and `prefetch_prefill_layer(L)` /
  `(L + 1)` on every layer of every forward pass, which means once per chunk.
- `python/freetoken/moe/offload_cache.py:878` (`prefetch_prefill_layer`) copies the full layer bank into the
  buffer: `copy()` at lines 894-897, `view.copy_(per_layer[layer_id], non_blocking=True)`.
- `--moe-prefill-hit-d2d` (`offload_cache.py:128-133`, `server/args.py:849`) gathers experts that are already
  resident in decode slots device-to-device. The remaining experts still cross PCIe on every chunk.
- The chunk size is `min(self.token_budget, remain_len)` (`scheduler/prefill.py:148`). The default is
  `max_extend_tokens = 8192` (`scheduler/config.py:16`); our profiles use 2048.
- `git grep -i layer.major main` finds nothing on main. Open PR #686 adds `--prefill-layer-major` for
  Qwen3.8 / Qwen4Exp only, capped at 32,768 new tokens; it is not merged and does not cover `qwen3_5_moe`.
  Related open issues: #579 and #501. This proposal does not duplicate an existing issue (we searched
  issues and PRs).

## What we changed

Switches, all default off:

- `QWEN36_LAYER_MAJOR_PREFILL=1` enables the path.
- `QWEN36_LAYER_MAJOR_MIN_TOKENS=<T>` sets the threshold.
  - Unset: layer-major starts for prompts longer than one chunk.
  - Our next profile uses 2049, so short prefills stay chunked and keep using hit-D2D.
- `QWEN36_LAYER_MAJOR_ATTN_BATCH=2` turns on attention batching. The code accepts 1-4; we only use 2.

Modules (fork): `python/freetoken/engine/layer_major.py`, `python/freetoken/moe/layer_major.py`,
`python/freetoken/scheduler/layer_major.py`, `python/freetoken/attention/fi.py` and
`python/freetoken/moe/offload_cache.py`.

How it works:

- **Loop order.** For a prompt with more than 2048 uncached tokens:
  1. Copy layer L's 256-expert bank once.
  2. Run every chunk through layer L, with the same 2048-token chunk shapes as chunked prefill.
  3. Move to layer L+1.
- **Activations.** The hidden and residual tensors of all chunks are parked in borrowed expert slots at the
  tail of the slot cache, with no host sync. At 120K this borrows 568 slots, or 624 with attention batching.
  If the activations do not fit, the prompt is split into windows; 250K uses 2 windows of at most 62 chunks.
- **Bookkeeping.** FlashInfer is re-planned per chunk, and the LM head runs only for the last chunk. RNG
  consumption is replayed, and cancellation waits at most 2 steps.
- **Eligibility.** `engine/layer_major.py` (`engine_reason`) admits the path only when all of these hold:
  - the model has layer-major hooks and a plain forward;
  - no native MTP and TP = 1;
  - the FlashInfer backend;
  - `OffloadMoeCache` with prefill-overlap double buffers;
  - hit-D2D only together with `MIN_TOKENS`.

  Anything else stays chunked.
- **Attention batching.** QKV, RoPE and the KV write still run per chunk; one FlashInfer prefill call covers
  2 chunks. At chunk 2048 a single call launches 512 CTAs, which is 3.66 waves on 70 SMs; the last wave is
  mostly empty. Batching is used only when all of these hold:
  - each chunk's own plan and the group plan share `CTA_TILE_Q`;
  - no split-KV;
  - starts aligned to the q tile;
  - single request, page size 1, causal.

  Otherwise it falls back to k = 1. This work also fixed a latent bug in our plan cache, which was keyed by
  rows although split-KV also depends on KV length. We have not checked whether upstream has a similar cache.

## Measured impact

- **Attention batching, model level**: median TTFT ratio k=2 vs k=1, both on layer-major, is 0.924 / 0.966 /
  0.950 at 16K / 64K / 120K (`docs/ARCHITECTURE.md` §3.5). The per-pair data is not in our published
  `results/`.
- **Layer-major alone**: we have **no published paired A/B** against chunked prefill on the same build.
  Mechanism evidence:
  - H2D per fresh prompt drops from 59 x 18.2 = 1076 GB to 18.2 GB at 120K (derived), and to 36.5 GB for
    2 windows at 250K.
  - Under Nsight, the 120K prefill is GPU-bound: the compute stream is busy 98%, and copy/compute overlap is
    about 85% (`docs/ARCHITECTURE.md` §3.1). The current release shows 18.3 GB H2D at 120K with 86% hidden
    under compute (`docs/RESULTS.md` §4, `results/ladder/nsys.json`). Nsight timings are diagnostics only.
  - Cross-build, not paired: a fresh 250K prompt took 115.7 s on an interim CUDA build with chunk-by-chunk
    order and 77.4 s on release f1 (median of 11), in different sessions (`docs/RESULTS.md` §1.2).
- **Release level (not attributable to this change alone)**: the table below compares original FreeToken
  mechanics (`bf128`, chunk 2048) with our release f1.
  - Protocol: 5 interleaved pairs, 128 output tokens.
  - Source: `results/matrix/final-f1-m-main-release-vs-bf128-comparison.json`, summarised in
    `docs/RESULTS.md` §1.
  - f1 also contains the CUDA NVFP4 prefill, CUDA GDN, FP8 GEMV, decode gather, FP32 GDN snapshots and the
    fused attention gate.

| Fresh prompt | TTFT original, median | TTFT f1, median | Whole request, geomean of paired ratios |
|---|---:|---:|---:|
| 4K | 2.24 s | 0.77 s | x1.95 |
| 16K | 6.78 s | 1.59 s | x3.03 |
| 64K | 28.46 s | 9.14 s | x2.86 |
| 120K | 54.77 s | 22.07 s | x2.45 |

## Correctness evidence

- The result is bitwise equal by construction: every chunk sees exactly the same kernels and shapes as in
  chunked prefill.
- CPU tests compare logits, tokens, KV, GDN live state, snapshots, the radix tree and the free lists.
  Ordering tests live in our harness (for example `harness/test-stream-ordering-cpu.py`), not yet under
  `freetoken/tests/`.
- Model-level teacher forcing gives 125/125 with max_abs 0 (release f1 gate B, and again for the current
  release; `results/ladder/gates.json`).
- Attention-batching counters in the f1 131,072-token context gate (`results/metrics/metrics.json`,
  `latency.context_gates.profiles.main128`):
  - fresh long prompt (59 chunks): 330 groups covering 660 chunks, 0 fallback groups;
  - boundary prompt (64 chunks): 640 groups, 0 fallback groups.

## Limitations / not measured

- Tested only on sm_120, one model (`qwen3_5_moe`), TP = 1, one request at a time, and the FlashInfer
  backend. Behaviour with concurrent requests has not been measured.
- Decode re-fetches the borrowed slots after prefill, which costs a few percent of decode right after a long
  prompt; we have no published per-step number for this. A slot-refill variant was bit-exact but showed no
  gain within noise, so it stays off (`docs/RESULTS.md` §8).
- Prompts with 2048 uncached tokens or fewer (short prompts, prefix hits, edits) stay on the chunked path.
- Attention batching with k > 2 has not been measured.
- The fork code is based on `cac247a8`. It has not been rebased onto or run on main `4d2e2e2`, which has
  since split the offload cache into slot pools (#689, #700).

## Proposal

We could contribute a PR behind a default-off flag, scoped to `qwen3_5_moe` + FlashInfer + `offload`, with the
CPU equivalence tests ported to `tests/`. Attention batching would follow as a separate PR.

Open questions:

1. Should this share a scheduler hook with #686 (Qwen4Exp) instead of adding a second model-specific path?
   Which flag name do you prefer?
2. Is it acceptable to park activations in borrowed expert slots, or do you want a dedicated pool?
3. Should hit-D2D (short prefills) and layer-major (long prefills) coexist through a token threshold, as in
   our fork?
