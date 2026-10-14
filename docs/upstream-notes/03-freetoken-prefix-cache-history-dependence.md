<!-- DRAFT ONLY, NOT POSTED. Local file; the owner decides the repository and whether to post. Suggested target: FlashML-org/FreeToken (issue labelled question/feature, or a discussion). Informational: no correctness bug claimed. -->

# Hybrid radix cache: greedy outputs depend on request history even when `cached_tokens` is 0 (request: cache flush endpoint / deterministic mode)

## Summary

With `--cache-type radix` on a hybrid GDN model (Qwen3.5/3.6), the same request with `temperature=0` can produce
different logits and occasionally a different greedy answer depending on the server's cache history. Part of this
is the usual prefix-cache batch-shape non-invariance: a prefix hit resumes from KV and a GDN state that were
computed with different shapes. Two parts are less obvious, and we think they are worth documenting or making
optional:

1. **KV dedup re-point without a reported hit.** When a prefill is committed, `insert()` finds token spans that
   are already in the tree as KV-only nodes, typically the chat-template header that every conversation shares.
   The request's own pages for that span are freed, and its page-table row is re-pointed at the tree's pages,
   which were written by an earlier request. From the next decode step on, the request attends to that foreign
   KV. `match_prefix` reports only nodes with a GDN snapshot, so the request reports **0 cached tokens**.
2. **The GDN snapshot is stored through a bf16 buffer.** Prefix hits restore the recurrent state from a snapshot
   that went through bf16 (`h = k.new_empty(...)` in the FLA chunk kernel). A fresh run continues from the fp32
   state.

Both are valid numerics: the values differ only by rounding, and a stream-ordering review found no race. They do,
however, make A/B evaluation and bit-exact regression testing depend on the order of requests and on the size of
the KV pool. We would like an opt-in **deterministic mode** and/or a cheap **cache flush endpoint**.

## Environment

- FreeToken 0.1.3 `cac247a860e316e06580d05aeb05f2e647bde214`, `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`
- RTX 5070 Ti (sm_120), torch 2.11.0+cu130, flashinfer-python 0.6.18.post1, Linux (WSL2)
- `ft serve ... --cache-type radix --max-running-requests 1 --max-prefill-length 2048 --moe-strategy offload`.
  The scheduler is the default overlap scheduler, `page_size` is 1, and `--max-seq-len-override` was 131072 or
  262144.
- The observations come from a downstream build with local performance patches. The cache code involved, the
  insert, free and re-point steps below, is functionally the same as at `cac247a8`.

## Evidence

**A. Same prompt, different request history (teacher-forced logits).**
A 120,076-token prompt with a forced 38-token continuation. The reference is our standard teacher run: 20 prompts in
a fixed order on a 131K KV pool. Admitting the 120K prompt evicts the tree that the earlier prompts built, so in the
reference this prompt runs entirely on its own KV.

| Run (freshly started server) | Logits vs reference (positions 0/8/16/24/32) | Mean NLL delta |
|---|---|---|
| the 120K prompt alone (262K pool) | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 (bit-identical) | 0.0 |
| a 31-token prompt, a 4,171-token prompt, then the 120K prompt (131K pool, nothing evicted) | max \|diff\| 0.0 / 1.04 / 0.94 / 0.69 / 1.23 (rel L2 up to 0.095) | +0.0177 nats/token (PPL +0.07%) |
| the full 20-prompt run on a 262K pool (nothing evicted) | identical to the previous row, to the last digit | +0.0177 |

The three prompts share a 24-token chat header. The canonical KV for header positions 0-2 was written by the
31-token prompt and for positions 3-23 by the 4K prompt. The 120K request reports 0 cached tokens. The first
prefill logit is bit-identical. Decode step 1 is also identical, because the overlap scheduler launches it before
the prefill commit runs. The first NLL difference appears at position 2 (+2e-5), grows to +0.075 by position 4,
and then stays in a band of max |logit diff| about 1.

So whether the dedup happens depends on the request history **and** on the KV pool size: a smaller pool evicts the
header nodes before the long prompt commits. The same request sequence gives bit-different results under two
`--max-seq-len-override` values. The deviation was identical to the last digit across three runs on two different
builds.

**B. Greedy answer flips with cache state** (T=0, thinking off, identical request bytes):
- A tool-calling request sent twice to a freshly started server: the first response is a direct text answer and
  the second is the tool call. This is deterministic across restarts. It is a near-tie between "answer" and "call
  tool" in this quantised model, and the cache state decides it.
- Two prompts (76 and 16,452 tokens), each sent as 10 identical requests in a row on a build with the stock fp16
  NVFP4 globals: the first request (fresh prefill) produced output A, and the nine prefix-cache-path requests all
  produced output B. The first generated token is a near-tie, with top-2 margins of 0.48 and 0.09 logit in a
  high-precision reference.

**C. GDN snapshot precision.** On the GDN prefill wrapper (32 value heads, 16 key heads, K = V = 128, snapshot
boundaries at 64/128 tokens), the stored snapshot differs from the fp32 state at the same boundary by relative L2
1.66e-3 (max abs 9.8e-4). A variant that copies the fp32 accumulator directly is bit-equal to it. Outputs and live
state are unaffected; only prefix-hit continuations start from the rounded state.

## Where it happens (file:line at `cac247a8`)

- `python/freetoken/scheduler/cache.py:394-397`: the hybrid prefill-chunk commit calls `prefix_cache.insert(...)`
  and then `self._free(page_indices[old_handle.cached_len : prefix_len])` (frees the request's own pages for the
  deduped span).
- `python/freetoken/scheduler/cache.py:401-406`: `match_prefix`, then
  `self.page_table[req.table_idx, old_handle.cached_len : prefix_len].copy_(m.kv_indices[...])` (re-point to the
  canonical pages). The generic radix path does the same at `cache.py:300-308` and `320-323`.
- `python/freetoken/kvcache/hybrid_radix_cache.py:90-112`: `insert` returns `prefix_len` over existing nodes,
  including KV-only nodes without a snapshot. At `:76-88`, `match_prefix` truncates to the deepest node with a
  live snapshot, which is why the hit is reported as 0.
- `python/freetoken/kernel/fla/chunk_delta_h.py:324`: `h = k.new_empty(B, NT, H, V, K)`, so `h` is bf16 for bf16
  `k`. `python/freetoken/models/qwen3_5_moe/gdn.py:117-118` copies `h[0, track_h_row].to(rec.dtype)` into the fp32
  state pool.
- No cheap way to drop the tree today:
  - `kvcache/radix_cache.py:205-206`: `reset()` raises `NotImplementedError`.
  - `scheduler/cache.py:551-563`: `rebuild()` builds a new prefix cache, but it is reached only through a pool
    resize in `scheduler.py:183-188` (`POST /v1/cache/rebuild`), which also re-captures CUDA graphs.
  - `--cache-type naive` (`server/args.py:434-441`) disables reuse entirely.

## Measured impact

- Quality: neutral on average. For the 120K case above, the mean NLL delta of +0.0177 nats/token sits in the
  middle of the deltas we see from other legitimate numeric changes: |delta| p50 0.017 and p90 0.047 over 48
  cases, covering MoE kernel choice, prefill chunk size and the NVFP4-global dtype. Distributions and perplexity do not change, but individual
  greedy near-ties do.
- Testing cost: bit-exact regression gates and A/B evals have to control the request order and the pool size. We
  had to add a "clean cache" re-run (fresh server, single request) to tell real regressions from cache-history
  effects.

## Proposed options (opt-in; defaults unchanged)

1. **`POST /v1/cache/flush`** (idle-only, like `/v1/cache/rebuild`): drop the radix tree and reclaim the
   donated GDN snapshot slots, with no pool reallocation and no CUDA-graph re-capture. `CacheManager.rebuild()`
   already does the tree part. Implementing `RadixPrefixCache.reset()` would expose it. Eval harnesses can call it
   between cases.
2. **Deterministic mode, e.g. `--prefix-dedup-repoint=off`**: at prefill commit, insert the new nodes for future
   requests, but keep the live request on its own pages for spans that were not a reported hit. Free the duplicates
   when the request finishes instead of at commit. The cost is a short-lived duplicate of the shared span (24 tokens
   here, about 0.5 MB at 20 KB/token). The benefit is that a request with `cached_tokens == 0` is numerically
   independent of the tree. This needs care in the `_cache_req_hybrid` bookkeeping (handle, lock and free ranges),
   so we have not attached a patch.
3. **fp32 GDN snapshots**: write the snapshot from the fp32 recurrent state (or allocate `h` as fp32 when
   `return_h` is used for tracking), so prefix hits resume from the same state a fresh run would have.
4. **Visibility**: report the deduped or re-pointed token count in the cache report or the usage details. Today
   `cached_tokens: 0` suggests that no foreign KV was used.
5. If none of these fit, a short note in the docs: with `--cache-type radix`, greedy outputs are a function of the
   request **and** the cache history, and `--cache-type naive` or a restart is the way to get history-independent
   results.

## Verification we did

- Mechanism confirmed on GPU with two fresh-server runs (table A): exact reproduction of the deviation with the
  3-request history on the small pool, and bit-exactness with the single request on the large pool. These were
  pre-registered predictions of the CPU analysis.
- CPU code review of the free and re-point against the overlapped decode step: safe. All page-table and KV writes
  are ordered on the scheduler and engine streams, decode step 1 reads the request's own pages, and later steps
  read the canonical pages. Not a race.
- Option 3 was implemented downstream as an opt-in: bit-equal to the same-boundary fp32 reference and correct under
  CUDA graphs. Options 1, 2 and 4 are proposals only (**not implemented, not run**).
