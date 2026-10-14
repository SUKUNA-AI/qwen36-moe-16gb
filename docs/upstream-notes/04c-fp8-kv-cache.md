<!-- DRAFT ONLY, NOT POSTED. Local file; the owner decides whether to post. Recommendation: optional feature request; the evidence is limited (component + teacher-forcing on one model, no broad eval, no speed numbers). Not a bug report. -->

# Feature request: optional FP8 (e4m3) KV cache on the FlashInfer path. Q and KV dtypes are currently tied

## Summary

FreeToken 0.1.3 has no KV-cache dtype option. The FlashInfer adapter plans prefill and decode with
`q_data_type=metadata.dtype, kv_data_type=metadata.dtype`, so swapping the KV pool dtype alone would also change
what is expected of Q. For hybrid models like Qwen3.5/3.6, only the full-attention layers have KV (10 of 40 in
Qwen3.6-35B-A3B), so the absolute saving is moderate. On small GPUs with offloaded experts, however, every freed
GiB becomes about 600 more NVFP4 expert-cache slots. A downstream prototype showed no measurable quality loss on
the checks we ran. We did not adopt it because we reached our context targets without it and did not measure a
speed gain.

## Environment

FreeToken 0.1.3 `cac247a8`, `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`, RTX 5070 Ti 16 GB (sm_120), flashinfer-python
0.6.18.post1, torch 2.11.0+cu130.

## Where (file:line at `cac247a8`)

- `python/freetoken/attention/fi.py:168-170` (prefill plan) and `:185-186` (decode plan): Q and KV dtypes both come
  from `metadata.dtype`.
- `python/freetoken/kvcache/mha_pool.py:32,53`: one `dtype` for the K/V buffers. There is no CLI option for the
  KV dtype in `server/args.py`.

## Expected benefit (Qwen3.6-35B-A3B)

The BF16 KV costs 20,480 B/token: 10 attention layers x (K + V) x 2 KV heads x 256 x 2 B. FP8 halves that, which
saves **0.625 GiB at 64K, 1.25 GiB at 128K and 2.5 GiB at 256K** of context, or about 375 / 750 / 1,500 NVFP4
expert slots at 1.78 MB per expert. We did not measure the decode speed effect of the extra slots.

## What we tried downstream (our code, not upstream)

- Per-layer static e4m3 K/V scales, calibrated on synthetic long prompts up to 60K tokens with a 1.2 margin.
- A quantise-and-store kernel for the paged K/V scatter. It works on post-norm, post-RoPE K/V, counts values that
  overflow e4m3 (|x| > 448), is byte-exact against a reference scatter (including strided and flattened inputs),
  and works under CUDA graphs.
- FlashInfer with BF16 Q and FP8 KV for paged prefill and decode, with the K/V scales passed in.

## Evidence

- Attention output vs BF16 KV, fixture with 1 and 17 query tokens: relative L2 0.037-0.038.
- Teacher forcing (20 fixed prompts, 943 target tokens): all forced targets were completed. Relative perplexity
  change **+0.027%**, 95% CI [-0.76%, +0.83%].
- 5 fixed smoke prompts: 4/5 passed. The one failure (a short coding prompt) is not evidence of FP8 harm. Its
  output is **byte-identical (same sha256)** to the output of an independent high-precision reference of the same
  quantised model, and to the output of a later build that fixed an unrelated precision issue. That prompt is a
  first-token near-tie, with a 0.48-logit top-2 margin in the exact model, and FP8 KV flipped it to the exact
  model's answer.
- Not run: a broad functional eval, long-context retrieval at 128K/256K with FP8 KV, decode speed, or a
  prefix-cache stress test.

## Proposal

An opt-in `--kv-cache-dtype {auto,fp8_e4m3}` for the FlashInfer backend:
1. separate `q_data_type` (model dtype) and `kv_data_type` (fp8) in both plans, and pass the K/V scales to prefill
   and decode;
2. quantise on store, after q/k-norm and RoPE, with per-layer (or per-head) scales: static from the checkpoint or a
   calibration file, or dynamic;
3. recompute the page budget from the KV element size, so the freed memory goes to pages or the expert cache;
4. keep the GDN recurrent state and snapshots out of scope (fp32/bf16 as today).

## Readiness

Can be posted as a feature request if wanted. The evidence is supportive but narrow: one model, component checks,
and teacher-forced PPL on 20 prompts. It is not a bug report.
