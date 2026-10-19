<!-- READY, NOT POSTED. Target: comment on FlashML-org/FreeToken#693 (seed silently ignored; per-request seeded sampling). Seed part only. Checked against upstream main 4d2e2e2 (2026-10-19). Paste everything below this line. -->

We implemented the `seed` part of this proposal in a fork of FreeToken 0.1.3 (single RTX 5070 Ti, FlashInfer
sampler), in case the details help a PR.

**Still present on main 4d2e2e2:** `ChatCompletionRequest` has `extra="allow"` and no `seed` (`server/api_models.py:63-64`).
`engine/sample.py:41-52` calls the sampling functions without `seed`/`offset`.

**What we did** (a small diff against 0.1.3; `server/api_models.py`, `server/generation.py`, `server/openai_api.py`, `core.py`, `engine/sample.py`):

- `seed: int | None` on `ChatCompletionRequest` and `SamplingParams`, passed through `resolve_sampling` like the
  other fields.
- `Sampler.prepare` derives one 63-bit key per seeded row with a SplitMix64 mix of `(seed, req.device_len)`. The key
  depends on the **absolute token position**, not on a step counter, so a replay draws the same number at the same
  position wherever the step boundaries fall. That includes resuming from a cached prefix.
- `Sampler.sample`: if any row has a seed, every row is sampled on its own with `seed=key, offset=0` passed to the
  FlashInfer `*_sampling_from_probs` functions. A batch without seeded rows takes exactly the old path. A seed on a
  build without FlashInfer raises, instead of silently falling back to an unseeded draw.

**Limitations of ours:**

- Sampling row by row costs one sampler launch per row. That is fine for our `--max-running-requests 1`, but we did
  not measure it for batches. A per-row Philox key inside the Triton kernel, as you suggest, would avoid it.
- No Triton-path implementation.
- We have no dedicated reproducibility test. Our teacher-forcing harness depends on the seed reaching the engine (it
  keys its fixtures by request seed), but that does not test the sampled draw.
- main's `Sampler` has gained penalties (#527) and a greedy mask (#471) since 0.1.3, so our patch would need a rebase.
- As you note, a seed makes the draw reproducible, not the logits.

We are happy to turn this into a PR on top of current main, or to review one.
