<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, comment on issue #579 (PR #676 implements it). Paste everything below this line. Checked against upstream main 4d2e2e2 (2026-10-19): neither #676 nor an equivalent is on main. -->

An independent data point for this issue, from a downstream fork. Our setup: RTX 5070 Ti 16 GB (sm_120), PCIe 5.0
x16 under WSL2 (one copy engine), `nvidia/Qwen3.6-35B-A3B-NVFP4` (40 MoE layers x 256 experts, top-8, ~455 MB
per layer, 18.25 GB per streamed chunk), FreeToken 0.1.3 `cac247a8` + our changes, batch 1. We see the same
problem and a similar crossover, using a slightly different mechanism.

**What we do (opt-in `QWEN36_PREFILL_SELECTED=160`).** Before a chunk starts, we check its uncached token
count. A chunk under 160 tokens, outside our layer-major long-prompt path, moves only the experts its router
selected. We do not admit them into the slot cache. Instead, we fill the existing prefill double buffer, keeping
row == expert id so the GEMM sees the same views as on the full-layer path:

- experts already resident in slots are gathered device-to-device;
- the rest cross PCIe through our TMA decode gather, driven by device-side lists (no host sync per layer);
- the slot maps are only read: no LRU usage, admissions or decode statistics are written.

Like yours, layer L's copy has to wait for layer L's router, so it cannot hide behind the previous layer. That is
why the route needs a threshold. Code: `moe/offload_cache.py` (`select_prefill_route`,
`selected_prefill_layer`), `moe/offload_kernels.py`, `kernel/decode_gather.py`.

**How many experts a short chunk touches** (real router, per layer): 26 tokens -> 66 experts (26% of the
bank), 41 -> 88, 74 -> 114, 125 -> 140 (55%). Below 160 tokens the gain was at least 11% everywhere; from about
180-200 tokens the copy that cannot overlap ate the gain (`docs/RESULTS.md` §9). That is well below your ~700-token
crossover. Our guess is that a faster link and a smaller layer make full streaming cheaper here; we have not
checked this.

**Measurements on our setup:**

- Threshold screen, 2 pairs, 8 requests per arm and case, TTFT ratio vs the same build without the route
  (`next-update/transport/results/ab-c1-threshold.json`). The experts moved per layer fell from 256 to 123-147,
  and H2D per chunk from 14.07 GB to 5.7-7.1 GB.

  | Uncached tokens | Ratio, median of 2 pairs (each) |
  |---|---|
  | 90 | 0.726 (0.676 / 0.776) |
  | 114 | 0.835 (0.775 / 0.894) |
  | 130 | 0.825 (0.767 / 0.882) |
  | 170 | 0.906 (0.839 / 0.973) |
  | 574 (route not taken) | 0.981 (0.895 / 1.066) |

- Model-level ladder step, 3 pairs, 60 s warm-up, 128 output tokens, no profiler (`results/ladder/summary.json`,
  step L8). A fresh 64-token document, about 125 uncached tokens after the chat template, went from 0.252 to
  0.207 s TTFT: median (mean) 0.252 (0.254) -> 0.207 (0.205), range 0.197-0.210. 16K and 120K TTFT were
  unchanged (1.315 -> 1.323 s, 20.43 -> 20.38 s).
- Together with our copy-stream change (C0), relative to cached-expert D2D alone: a ~25-token continuation took
  0.28-0.34x the TTFT, and a fresh ~125-token prompt 0.61-0.73x (3 pairs;
  `next-update/transport/results/ab-ttft.json`).
- Correctness: teacher forcing 125/125 cases, max_abs 0 on the logits; screen outputs identical in every case.

**Two things that did not help us** (`docs/RESULTS.md` §8-9):

- A variant that reads the miss list back to the host and issues one `cudaMemcpyBatchAsync` per layer was slower
  than the device-list version: TTFT ratio 0.50-0.84 vs 0.29-0.62 on short cases, one pair.
- Gathering a whole layer by TMA instead of the copy engine made TTFT 3-36% worse in 3 pairs (on par at 4K and
  16K). The copy engine does ~49 GB/s here vs ~35 GB/s for TMA on full layers.

We also tried installing the chunk's most-routed experts into decode slots after prefill, which is close to what
admitting through the slot cache gives you for free. It was bit-exact but showed no model-level gain: decode at
16K 173.7 vs 176.4 tok/s, 64-token TTFT 0.205 vs 0.201 s.

Limits: sm_120 only, one model, batch 1. The screens use 2-3 pairs. Not run on your Qwen3.8 / 512-expert setup.

**Our implementation, for reference** (fork of FreeToken 0.1.3, Apache-2.0, pinned commit; feel free to reuse):

- [`moe/offload_cache.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/moe/offload_cache.py)
- [`moe/offload_kernels.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/moe/offload_kernels.py)
- [`kernel/decode_gather.py`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/freetoken/python/freetoken/kernel/decode_gather.py)

Evidence and write-ups (`docs/RESULTS.md` is in Russian):

- [`docs/RESULTS.md`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/docs/RESULTS.md)
- [`next-update/transport/results/ab-c1-threshold.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/next-update/transport/results/ab-c1-threshold.json)
- [`results/ladder/summary.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/results/ladder/summary.json)
- [`next-update/transport/results/ab-ttft.json`](https://github.com/SUKUNA-AI/qwen36-moe-16gb/blob/745b28a34fc5b68439a34cc7efbefa55209c660f/next-update/transport/results/ab-ttft.json)
