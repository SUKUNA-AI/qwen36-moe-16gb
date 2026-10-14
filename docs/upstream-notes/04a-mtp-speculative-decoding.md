<!-- DRAFT ONLY, NOT POSTED. Local file; the owner decides whether to post anything. Recommendation: do NOT open an issue (no upstream defect; evidence is component-level only). If upstream starts MTP work, this text can go into a discussion or feature-request comment. -->

# Discussion: MTP speculative decoding for Qwen3.5/3.6 MoE with expert offload. Notes from a downstream prototype (not adopted)

## Status for upstream

- FreeToken 0.1.3 (`cac247a8`) has no speculative decoding for Qwen3.5/3.6. The loader drops the MTP head:
  `python/freetoken/models/qwen3_5_moe/weight.py:58` skips `mtp.*` (also `qwen4_exp/weight.py:89`). The only
  traces are pool-sizing hooks (`kvcache/qsa_pool.py:49-51`) and a comment in
  `kernel/fla/fused_sigmoid_gating_recurrent.py:319`.
- Everything below was tried in **our own downstream code**: `engine/mtp.py`, `engine/speculative.py` and
  `scheduler/speculative.py` do not exist upstream. **There is no upstream bug to report.** At most this is input
  for a feature discussion.

## What was tried

- A native BF16 MTP head for `nvidia/Qwen3.6-35B-A3B-NVFP4`, ported from vLLM's `qwen3_5_mtp.py`: 19 tensors,
  1.69 GB in BF16, including its own 256-expert MoE layer.
- Draft-and-verify in the scheduler with a provisional/commit/rollback transaction. On failure it restores the GDN
  live state, positions, the provisional token suffix, the page table and the MTP hidden state. Verification
  extends up to 4 tokens; one request; `page_size=1`; the overlap scheduler is off.
- A GPU rejection-sampling verifier. It passes 15,000 samples over 3 known distributions plus near-one/zero-support
  boundary cases, and 17 CPU tests (distribution, fault injection, EOS/stop/tool-call boundaries, cancellation).
- A prompt-lookup drafter on the same transaction path, with CPU tests only.

## Why it was not adopted (evidence)

1. **Numerical guard failed, at component level with synthetic activations and real MTP weights.** We compared
   the MTP draft logits from a prefix built in mixed chunks with those from the full prefix: relative L2
   **0.052**, against a pre-set guard of 0.03. Full vs incremental gave 0.0061. Localisation: exact rollback, a
   poisoned discarded suffix and provisional commit are all **bit-identical**, so the rollback is not defective.
   The difference starts as BF16 rounding (o_proj rel L2 7e-4, post-attention norm 1.1e-3). One router row then
   selects different experts, and the final norm ends at rel L2 0.046.
   Conclusion: MTP drafts are sensitive to the chunking of the BF16 prefix. This lowers acceptance, but it is not
   a bug.
2. **Memory.** On a 16 GB card with offloaded experts, the BF16 MTP layer (1.69 GB) costs about 950 GPU
   expert-cache slots at about 1.78 MB per NVFP4 expert. Decode on this setup is bound by expert misses: each miss
   costs about 85 us on the critical path, and the miss gather is about 4.6 of 11.2 ms per token at 64K (measured
   before our later gather optimisations). Fewer slots means more misses.
3. **Expert union.** Verifying k drafts touches the union of the experts that k+1 tokens route to. With offload
   this multiplies PCIe misses per step. We did **not** measure this; it is the expected main cost.
4. **No model-level run.** We have no acceptance rate, no end-to-end speed and no quality run. One MTP decode step
   took about 1.4-1.7 ms eager, which is a component timing only.

## Side note for any implementer (bug found in our own sampler, not upstream code)

Drawing a token by inverse CDF in float32 with a Python uniform close to 1.0 can select a **zero-probability**
token. `float32(u)` rounds to 1.0, `searchsorted` runs past the positive support, and a `clamp` picks the last
vocabulary entry. Reproduced with `u = nextafter(1, 0)` and `p = [0.25, 0.75, 0, 0]`, which gave index 3 with
probability 0. The fix: build the CDF in float64 from the represented weights, scale `u` by the actual total, cap
it below the total, and search the right edge of each interval.

## Readiness

Evidence is insufficient for an upstream issue: component-level only, synthetic activations, and no model run.
Keep it as background in case upstream adds MTP.
