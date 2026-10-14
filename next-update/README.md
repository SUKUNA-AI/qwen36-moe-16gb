# next-update/ — pending work for the next release (not released)

Each item is CPU-verified at most and still needs the teacher gate (0-diff vs the bf128
reference) and a paired A/B on GPU before it can enter a release. Raw run outputs, original
file copies and full-tree snapshots stayed in the archive (see `ARCHIVE.md`).

| Item | Status | Content |
|---|---|---|
| `short-route/` | Patch ready (`shortroute.patch`), CPU tests pass, GPU screening done via shim | Route short prompts / prefix hits to the cached-expert path (`--moe-prefill-hit-d2d`, `QWEN36_LAYER_MAJOR_MIN_TOKENS`): 64-token TTFT 0.565 -> 0.446 s, outputs identical. Needs teacher gate + profile-runtime rule change. |
| `fla/` | Explained, flag-off in the tree | Fused FLA chunk kernels (`QWEN36_CUDA_GDN_FLA=wu`) are bit-exact but only -2..3 % TTFT at 16K; nsys comparison scripts and configs. `taskc/` = `.shared::cluster` -> `.shared::cta` patches for experimental (not shipped) kernels. |
| `cache-reset/` | Applied to `harness/quality.py` (v7) | Clean prefix cache before the `long-120000` teacher case; validation scripts and result. |
| `fp8kv/` | Quick speed screens done, quality NOT run, owner decision pending | FP8 KV cache: weight-bound no-clip scales, configs, GPU plan (`gpu/`), pre-registered quality rule. Non-bit-exact; costs 10-33 % TTFT, gains 25-46 % decode. |
| `synclean-refill/` | Rebased patch + pre-registered rule, GPU A/B not run | Fewer host syncs / FlashInfer plans per chunk under layer-major; refill `restore` policy re-evaluation. Expected gains below noise. |
| `gather-prefetch/` | IN PROGRESS at copy time (re-sync from the archive) | Decode gather bandwidth + prefetch (nsight lever 1, est. +7.5..13 % decode). |

`AGENT-RULES.md` holds the working rules used for the post-release batch (GPU lock, timeouts,
CPU priority).
