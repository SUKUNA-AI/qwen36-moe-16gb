<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, comment on issue #677. Paste everything below this line. Checked against upstream main 4d2e2e2 (2026-10-19): no lookahead prefetch on main. -->

A negative data point from a downstream fork, in case it helps scope this. We tried a one-layer-ahead
speculative expert prefetch and did not ship it: it was bit-exact, but slower on the model on our setup.

**Setup:** RTX 5070 Ti 16 GB (sm_120), PCIe 5.0 x16 under WSL2, `nvidia/Qwen3.6-35B-A3B-NVFP4` (40 MoE layers:
30 linear-attention (GDN) layers and 10 full-attention layers, 256 experts each), LRU slot cache with 4603 slots
(~12 misses per token at 16K). The baseline already includes the shared-expert/miss-gather overlap (#678-style),
a TMA miss gather, and bit-exact CUDA decode GEMVs. CUDA-graph decode, batch 1.

**Our variant** (`QWEN36_DECODE_PREGATE=<n>`, our fork's `moe/pregate.py`):

- At the start of layer L+1, before its attention/GDN mixer, we apply layer L+1's router to its
  post-attention-norm of the residual that leaves layer L. Your lookahead runs one layer earlier.
- Up to n predicted misses go through the normal `lru_ensure` and gather on a side stream while the mixer runs.
  The real `ensure_experts` still runs on the real routing after a join, so every routed expert is resident with
  exact bytes before the GEMM.
- Caps were 1 per GDN layer and 2 per full-attention layer.
- Predictor quality, measured under exact LRU replay (1,530 decode steps, uncapped): coverage of real misses
  0.749, routing recall 0.844, precision of the speculative fetches 0.652
  (`next-update/transport/results/d3a-route-trace.json`).

**Model result** (3 pairs, 60 s warm-up, 128 output tokens, no profiler; median (mean)):

| | Decode 16K, tok/s | Decode 120K, tok/s | Misses/token left on the critical path, 16K | Decode step 16K, Nsight |
|---|---|---|---|---|
| baseline | 176.4 (175.4) | 120.5 (120.3) | 12.4 | 5.78 ms |
| + prefetch | 169.2 (169.0) | 118.2 (116.5) | 6.5 | 5.88 ms |

Teacher forcing: 125/125 cases, max_abs 0. The prefetch halved the misses left on the critical path, but decode
lost 4% at 16K. Peak process memory grew by 46 MiB.

**Why, as far as we could tell** (microbenchmark of one layer: mixer kernel + speculative chain):

- On a stream of equal priority, the speculative chain hid about 0% of the copy (hidden fraction -0.06 to 0.04)
  behind a mixer that is a single large kernel. CTAs of equal-priority grids are dispatched in launch order, so
  the chain waits behind the mixer's last wave. With a mixer of six kernels, even equal priority hid
  47-85%.
- With its own highest-priority stream (`QWEN36_DECODE_PREGATE_PRIORITY=1`) it hid 44-92% of the copy, and the
  loop step was 15-41% shorter. That is an upper bound from a synthetic loop, and we never ran the
  priority variant on the model (`next-update/next-candidate/results/microbench-w2.json`, `microbench-w3.json`).

This matches your choice of a dedicated high-priority stream. Your +22% is measured on a link-bound baseline
(the copies are about half of the step), while our gather was already a small part of the step (0.64 of 5.78 ms in
Nsight). That may explain the different outcome.

**Our implementation, for reference** (fork of FreeToken 0.1.3, Apache-2.0, pinned commit; feel free to reuse):

- [`moe/pregate.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/moe/pregate.py)

Evidence and write-ups (`docs/RESULTS.md` is in Russian):

- [`next-update/transport/results/d3a-route-trace.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/next-update/transport/results/d3a-route-trace.json)
- [`next-update/next-candidate/results/microbench-w2.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/next-update/next-candidate/results/microbench-w2.json)
