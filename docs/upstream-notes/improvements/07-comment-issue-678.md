<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, comment on issue #678. Paste everything below this line. Checked against upstream main 4d2e2e2 (2026-10-19): no shared-expert/miss-copy overlap on main. Full write-up: our new issue 07-shared-expert-overlap.md (post it first and replace "<our issue>" below with its number). -->

Independent confirmation from a downstream fork, with a different split of the streams. Details and a PR offer
are in <our issue>.

**Setup:** RTX 5070 Ti 16 GB (sm_120), PCIe 5.0 x16 under WSL2, `nvidia/Qwen3.6-35B-A3B-NVFP4` (40 MoE layers x 256
experts + gated shared expert), FreeToken 0.1.3 `cac247a8` + our changes, LRU slot cache (3164 slots, ~28 misses
per token at 16K), TMA-based miss gather, CUDA-graph decode, batch 1.

**What we do differently:** the **miss gather** goes to the side stream and the **shared expert** stays on the
compute stream (`QWEN36_DECODE_SHARED_OVERLAP=1`):

- `ensure_experts` stays on the compute stream, so the slot map and the miss plan have a single writer.
- We record `ready` on the compute stream; the side stream waits on it and runs `copy_missing`, then records
  `done`. The compute stream waits on `done` before the routed GEMM.
- The stream and both events are created at cache construction, before capture. The fork and join are captured
  inside each layer of the decode graph. The join runs in a `finally`.
- We use a pooled `torch.cuda.Stream()` and did not test the aliasing you describe. In our ordering an alias would
  serialize the layer rather than reorder it, but your `ExternalStream` fix is the safer choice either way.

**Numbers on our setup** (3 pairs, rotating order, 60 s warm-up, 128 output tokens, no profiler; median (mean)):

| | Decode 16K, tok/s | Decode 120K, tok/s | Decode step 16K, Nsight |
|---|---|---|---|
| before | 143.6 (142.0) | 103.4 (103.7) | 7.02 ms |
| + overlap | 148.6 (144.9) | 106.1 (104.9) | 6.82 ms |

- That is +5.0 tok/s (+3.5%) at 16K; one of the 3 pairs was slower than its reference.
- Applied alone to an older build of ours, the 16K result was inconclusive (faster in 2 of 3 pairs) and 120K was
  +2.0-2.1% in all 3 pairs.
- Our gather already runs near the host-link ceiling (~42 GB/s), which may be why we gain less than your +7.8%.
  We also have no scored eviction.
- Giving the side stream the highest priority made the decode step 0.5-2% worse (2 pairs), so we keep it off.

**Correctness:**

- Teacher forcing against our reference: 125/125 cases with max_abs 0 on the logits, and byte-equal outputs.
- A GPU test with the real ensure/LRU/gather and byte-checking consumers passes 4/4. It covers eager mode and two
  captured graphs replayed over 180 steps with changing routes.
- A negative control without the join gave 14,248 checksum mismatches, so the join is load-bearing.

One practical note: Nsight Compute hung while capturing the two-stream graph (ncu serialises kernels). We profile
with the overlap switched off.

**Our implementation, for reference** (fork of FreeToken 0.1.3, Apache-2.0, pinned commit; feel free to reuse):

- [`decode_overlap.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/decode_overlap.py)
- [`layers/moe.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/layers/moe.py)
- [`models/qwen3_5_moe/moe.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/models/qwen3_5_moe/moe.py)
- [`moe/offload_cache.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/moe/offload_cache.py)
- [`freetoken/tests/moe/test_decode_overlap_gpu.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/tests/moe/test_decode_overlap_gpu.py)
- [`harness/test-decode-overlap-cpu.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/harness/test-decode-overlap-cpu.py)
