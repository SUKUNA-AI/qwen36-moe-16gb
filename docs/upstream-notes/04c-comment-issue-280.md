<!-- READY, NOT POSTED. Target: comment on FlashML-org/FreeToken#280 (FP8 KV cache; PR #354 is open, not merged on main 4d2e2e2 as of 2026-10-19). Replaces a new issue from 04c-fp8-kv-cache.md, which would duplicate #280. Paste everything below this line. -->

One more data point for this request and #354, from a different setup: a single RTX 5070 Ti 16 GB (sm_120),
`nvidia/Qwen3.6-35B-A3B-NVFP4`, offload MoE backend, FreeToken 0.1.3. We prototyped FP8 KV on the **FlashInfer**
path (not the Triton attention used in the fork above), so it may help with the design of #354.

**What we built (downstream, not upstream code):**
- per-layer static e4m3 K/V scales, calibrated on synthetic prompts up to 60K tokens with a 1.2 margin;
- a quantise-on-store kernel for the paged K/V scatter, after q/k-norm and RoPE; it counts e4m3 overflows,
  is byte-exact against a reference scatter, and works under CUDA graphs;
- FlashInfer paged prefill and decode with BF16 Q and FP8 KV, with the K/V scales passed in.

One thing to watch: `attention/fi.py:169-170` and `:185-186` plan both passes with
`q_data_type=metadata.dtype, kv_data_type=metadata.dtype`. Changing only the pool dtype therefore also changes
what FlashInfer expects of Q; the two have to be split.

**Results on that model:**
- attention output against BF16 KV (1 and 17 query tokens): relative L2 0.037-0.038;
- teacher forcing, 20 fixed prompts, 943 target tokens: perplexity change **+0.027%**, 95% CI [-0.76%, +0.83%];
- 4 of 5 fixed smoke prompts passed. The one miss is a first-token near-tie (0.48-logit top-2 margin), and its
  output is byte-identical to that of an exact high-precision reference of the same quantised model. In other
  words, FP8 KV moved it *to* the exact answer.

**Memory:** 10 of 40 layers have KV, so BF16 costs 20,480 B/token. FP8 saves 0.625 / 1.25 / 2.5 GiB at
64K / 128K / 256K. With offloaded experts that is about 375 / 750 / 1,500 more NVFP4 expert slots (1.78 MB each).

**Not measured:** decode speed, long-context retrieval at 128K/256K with FP8 KV, a broad functional eval, and a
prefix-cache stress test. We did not ship it because we met our context targets without it. If #354 lands, we
can run our teacher-forcing and long-context checks against it on sm_120.
