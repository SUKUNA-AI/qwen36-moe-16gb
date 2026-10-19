<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Feature] Optional host-resident token-embedding table (pinned, mapped, UVA lookups): ~1 GB of VRAM back for offloaded expert slots

## Summary

A decode step reads one embedding row per token, 4 KiB for a 2048-wide bf16 table, but the whole table stays in
VRAM. For Qwen3.6-35B-A3B that is 248,320 x 2048 bf16 = 1,017,118,720 B (0.95 GiB). On a 16 GB card that offloads
experts, the same bytes hold about 565 more expert slots. Our fork adds an opt-in switch that moves the input
embedding table into pinned, device-mapped host memory before the weights are measured. Lookups then copy
rows over UVA with a small byte-copy kernel. The rows are bit-identical, and the lookup is CUDA-graph
capturable. On our setup, `--moe-cache-auto` at the same `memory_ratio` 0.75 then sized 3729 instead of 3164 expert
slots. Decode misses per token at 16K fell from 27.7 to 19.9, and decode rose from 148.6 to 153.8 tok/s.

## Setup

- RTX 5070 Ti 16 GB (sm_120), PCIe 5.0 x16, WSL2; the Windows desktop shares the GPU
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a`: untied `embed_tokens` / `lm_head`, bf16 table 248,320 x 2048;
  1,781,760 B per routed expert
- FreeToken 0.1.3 `cac247a8` + our fork; `--moe-strategy offload --moe-cache-auto`, `memory_ratio` 0.75, TP 1,
  batch 1, CUDA-graph decode

## Problem in upstream today

All paths are under `python/freetoken/`, at `main` `4d2e2e2`.

- The unquantized embedding method allocates the table as an ordinary device weight
  (`layers/quantization/embedding/unquantized.py:32-33`). Lookups index it on the GPU
  (`IndexingEmbeddingKernel.embed`, `:19-22`, called from `VocabParallelEmbedding.forward`,
  `layers/embedding.py:52-53`).
- The engine measures `weights_bytes` as free VRAM before minus after loading (`engine/engine.py:410-411`). The
  expert/KV pool budget is `memory_ratio * baseline_free - weights_bytes - fixed_cache_size`
  (`engine/cache_budget.py:38-45`). Every byte of the table therefore comes out of the expert-slot budget.
- `main` has host-resident options for vision towers and other banks, but none for the token embedding.

## What we changed

- Switch `QWEN36_HOST_EMBEDDINGS=1` (default off; other values raise). Modules in our fork:
  `engine/host_embeddings.py`, `layers/embedding.py` (`VocabParallelEmbedding.move_to_host`),
  `kernel/triton/host_rows.py`, plus a call in `engine/engine.py` **before** `weights_bytes` is measured.
  The existing budget arithmetic then turns the freed bytes into expert slots by itself.
- `move_to_host` copies the table into exact-size pinned, device-mapped host memory and checks that the bytes
  equal the GPU table. It then releases the VRAM (`torch.cuda.synchronize()` + `empty_cache()`, so the measurement
  sees it) and keeps the device-visible address.
- Lookup: a Triton kernel, one program per requested row, dereferences the mapped table over UVA. It moves whole
  integer words, never through a float type, so every in-range row is bit-identical to the GPU path. The table
  address is fixed and there is no host sync, so it captures in CUDA graphs. Out-of-range ids store zero rows.
- Refusals: a tied `lm_head` (the head multiplies by the same table every step, so it must stay in VRAM),
  TP > 1, or a table that is not on CUDA.

## Measured impact

Model-level ladder step L4 (`results/ladder/summary.json`; 3 pairs, rotating order, 60 s warm-up, 128 output
tokens, no profiler; the step before already had our decode gather/shared-expert overlap). Values are median
(mean), min-max, n = 3:

| | before (L3) | host embeddings (L4) |
|---|---|---|
| Expert slots (`memory_ratio` 0.75) | 3164 | 3729 |
| Misses per token, 16K | 27.7 | 19.9 |
| Decode 16K, tok/s | 148.6 (144.9), 135.5-150.5 | 153.8 (151.7), 146.7-154.6 |
| Decode 120K, tok/s | 106.1 (104.9), 99.8-108.9 | 108.9 (108.4), 106.7-109.5 |
| Decode, 64-token doc, tok/s | 158.8 (156.5) | 161.4 (160.6) |
| TTFT 64-token doc, s | 0.304 (0.310) | 0.314 (0.305) |
| TTFT 16K / 120K, s | 1.330 / 20.44 | 1.319 / 20.47 |
| Process peak VRAM, MiB | 12,287 | 12,315 |
| Decode step 16K (Nsight), miss-gather part | 6.82 ms, 1.10 ms | 6.60 ms, 0.87 ms |

- The 565 extra slots x 1,781,760 B come to 1.007e9 B, close to the table's 1,017,118,720 B (derived).
- Host side: the table appears as +0.95 GiB of shmem for the service (1.056 vs 0.105 GiB). The GPU accounting of
  our release shows dense weights at 1.57 instead of 2.52 GiB (`docs/RESULTS.md` §5).
- TTFT stayed within the spread of 3 pairs, including 120K, where every prompt token's row crosses PCIe
  (~0.49 GB at 120K, derived). We did not measure the lookup's own time.

## Correctness evidence

- Teacher forcing at L4: 125/125 cases, max_abs 0 on logits, outputs byte-equal.
- GPU tests `tests/layers/test_host_embedding_gpu.py`: VRAM is released and the table is pinned; lookup bytes
  match the GPU path for several dtypes and row counts; every row checked once; out-of-range ids give zero rows;
  CUDA-graph capture and replay. CPU tests `tests/engine/test_host_embeddings.py` cover flag parsing and the
  tied/TP/CPU refusals. We do not quote pass counts here; they are not in `results/`.

## Limitations / not measured

- Only untied tables, TP 1, one model, batch 1, sm_120 under WSL2. Not tested with quantized or GGUF embedding
  methods (on `main` the lookup sits behind `quant_method.embed`, so a port would add a kernel to the unquantized
  method), multimodal embedding merges, or Gemma's `embed_scale` (which would still apply on the GPU after the
  lookup).
- It costs ~1 GB of pinned host memory. Contention with expert H2D traffic during long prefills was not
  measured separately.
- The decode gain depends on the expert miss rate. With enough VRAM it is zero.

## Proposal

We can send a PR: a `HostIndexingEmbeddingKernel` (or a `--embed-on-host` flag) for the unquantized method, the
Triton row-copy kernel, the call before `weights_bytes` is measured, and the tests above, default off. Open
questions: should it be offered only when `--moe-strategy offload` is active, and should it refuse
automatically for tied heads (as we do) or warn?
