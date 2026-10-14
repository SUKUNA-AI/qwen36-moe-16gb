# Architecture

This document covers how a 35B-parameter MoE with 256 experts per layer runs from a 16 GB GPU, and
what release f1 changed in FreeToken to make it fast without changing a single logit. Numbers are
quoted from [RESULTS.md](RESULTS.md), which also gives their sources.

## 1. Model and memory layout

Qwen3.6-35B-A3B has 40 decoder layers. 10 are full-attention layers (16 query heads and 2 KV heads,
head dim 256) and 30 are GatedDeltaNet (GDN) linear-attention layers. Every layer has a 256-expert MoE
with top-8 routing plus a shared expert. The NVIDIA checkpoint stores routed experts in NVFP4 (E2M1
values, E4M3 block scales, an FP32 per-tensor global scale), dense projections in FP8, and a few
tensors in BF16.

| Where | What | Size |
|---|---|---|
| Host RAM, pinned | **Expert bank**: all 40 x 256 routed experts | 18.245 GB (16.99 GiB); 1,781,760 B per expert, ~455 MB per layer |
| GPU (torch) | Dense weights (attention, GDN, shared experts, embeddings, LM head) | 2.52 GiB |
| GPU | **Expert slot cache** (main128: 3164 slots) | 5.25 GiB |
| GPU | KV cache, BF16, 20,480 B/token (10 layers x 2 KV heads x 256 x K/V x 2 B) | 2.50 GiB at 131,072 tokens |
| GPU | GDN state pool (recurrent + conv state, FP32 snapshots) | 0.54 GiB |
| GPU | Other (workspaces, activations) | 0.34 GiB |

Torch holds 11.15 GiB allocated (11.39 GiB reserved). The Windows desktop uses about 2.5-3 GiB of the
same GPU. The profiles size the slot cache with `memory_ratio` so that at least 0.5 GiB stays free
(1 GiB is the target).

### 1.1 Offload design: pinned bank + GPU slot cache

- The expert bank lives in host memory that is page-locked and mapped for DMA. Prefill reads experts
  from it with `cudaMemcpyAsync` on the copy engine. Under WSL2 the GPU exposes a single copy engine
  (`asyncEngineCount=1`), measured at ~47 GB/s, or 48.4-49.4 GB/s with huge pages (section 4.1).
- **Decode** uses the slot cache. After routing, each layer's 8 selected experts are looked up in a
  per-layer LRU map. Hits run from VRAM. Misses are gathered from the pinned bank straight into free
  slots by an SM-driven gather kernel (section 3.6), then the NVFP4 MoE decode kernel runs on slot
  indices. On main128 the miss rate is 0.08-0.12 (26-39 misses per token, 46-70 MB of H2D traffic per
  token). On long256, with 1481 slots, it is 0.38.
- **Prefill** of a chunk touches almost every expert of every layer: a 64-token prompt already routes
  to most of the 256. So prefill streams whole layers. Upstream FreeToken copies the full layer bank
  (256 experts) for every 2048-token chunk. At 120K that is 59 chunks x 18.2 GB = 1076 GB over PCIe,
  which made long prefills PCIe-bound once the MoE GEMM got fast. Layer-major prefill (section 3.1)
  removes this.

## 2. What upstream FreeToken already provides (and keeps)

Hybrid radix prefix cache with GDN state snapshots, FlashInfer paged attention for the 10 full-attention
layers, FLA Triton kernels for GDN chunked prefill, Triton NVFP4 MoE kernels, a marlin-style NVFP4 MoE
for decode, CUTLASS/cuBLAS FP8 dense GEMMs, CUDA-graph decode and an overlap scheduler.

## 3. f1 engine changes and kernels

All switches are opt-in environment variables. With none of them set, the engine behaves like upstream
apart from bug fixes. `freetoken/MODIFICATIONS.md` maps each switch to its modules.

### 3.1 Layer-major prefill (`QWEN36_LAYER_MAJOR_PREFILL=1`)

For a prompt with more than 2048 uncached tokens, the engine runs **layer by layer across all chunks**
instead of chunk by chunk across all layers:

1. Copy layer L's 256-expert bank to the GPU once.
2. Run every chunk of the prompt through layer L, using the same 2048-token chunk shapes as chunked
   prefill.
3. Move to layer L+1.

- The hidden and residual tensors of all chunks are parked in **borrowed expert slots** at the tail of
  the slot cache, without a host sync. At 120K this borrows 568 slots, or 624 with attention batching.
- If the activations do not fit, the prompt is split into **windows**: 250K uses 2 windows of at most
  62 chunks.
- FlashInfer is re-planned per chunk. The LM head runs only for the last chunk. RNG consumption is
  replayed, and cancellation waits at most 2 steps.
- Because each chunk sees exactly the same kernels and shapes as in chunked prefill, the result is
  **bitwise equal by construction**. CPU tests compare logits, tokens, KV, GDN live state, snapshots,
  radix tree and free lists. The model-level teacher run gives 125/125, max_abs 0.
- PCIe traffic per fresh prompt drops to 18.2 GB at any length up to 120K (36.5 GB for 2 windows at
  250K). The 120K prefill becomes GPU-bound: the compute stream is busy 98%, and copy/compute overlap
  is ~85%.
- Decode re-fetches the borrowed slots after prefill, which costs a few percent of decode right after a
  long prompt.

Short prompts, prefix hits and edits (2048 uncached tokens or fewer) still take the chunked path. The
next update addresses that path ([ROADMAP.md](ROADMAP.md)).

### 3.2 Bit-exact CUDA NVFP4 MoE prefill (`QWEN36_NVFP4_PREFILL=cuda`, `kernels/nvfp4-moe`)

- A hand-written sm_120a kernel built from inline-PTX `mma.sync.m16n8k16` (BF16), `ldmatrix` and
  `cp.async`. It has fused gate/up + SiLU, then down + router-weighted sum, with E2M1/E4M3 dequant done
  exactly like the Triton path.
- **Bit-exactness requires the Triton accumulation order.** Triton 3.6 lowers these variants with
  `dot_op kWidth=4`, which packs K residues mod 8 into each MMA as {0,2}/{4,6}/{1,3}/{5,7} and chains
  even/kRep0 -> even/kRep1 -> odd/kRep0 -> odd/kRep1 per K64. The CUDA kernel reproduces this order,
  plus the SiLU instruction sequence that FlashInfer's SASS uses.
- Results:
  - 0 mismatches over ~1.4e10 BF16 elements on real checkpoint weights (layers 0/13/26/39, M = 1..4096),
    and 1.5e10 synthetic.
  - The negative control (old layouts) reproduces the ~1e-4 mismatch rate.
  - Layer speed at M = 2048: 12.82 -> 1.77 ms (7.25x).
- Used for prefill chunks with M >= 512. Smaller batches keep the Triton path.

### 3.3 CUDA GatedDeltaNet (`QWEN36_CUDA_GDN=1`, `kernels/gdn`) and fused FLA (`QWEN36_CUDA_GDN_FLA`, off in f1)

- `libqwen36_gdn` (ABI 2) replaces the GDN glue around FLA's chunked kernels:
  - `conv_qkv` runs causal conv + SiLU straight from the fused qkvz rows, with no transpose copy and an
    L2-norm epilogue;
  - `gate_cumsum`;
  - `gated_norm` with the FP8 quantization for `out_proj` fused in;
  - a one-launch-per-layer decode kernel that runs as an 8-CTA cluster.
- Component results (bit-exact 124/124):
  - one GDN layer at T = 2048 drops from 1.230 to 0.509 ms (2.41x);
  - one layer at T = 4096 goes from 26 kernels / 3.04 ms to 8 kernels / 1.02 ms.
- Model-level effect (paired A/B): TTFT -8.6% at 64K and -6.2% at 120K. Decode is neutral.
- The same library also carries a CUDA rewrite of FLA's `chunk_h` / `chunk_o` / `recompute_w_u`
  (`QWEN36_CUDA_GDN_FLA=wu|ho`). It is bit-exact on 123/123 cases and cuts DRAM traffic by about 3x.
  It is **disabled in f1** because the model-level gain is below noise ([RESULTS.md](RESULTS.md) §8).

### 3.4 FP8 GEMV via QMMA (`QWEN36_FP8_GEMV=1`, `kernels/fp8-gemv`)

- In decode (M <= 8), the row-wise FP8 dense projections ran on a CUTLASS SM120 128x128x128 tile: 39
  kernels per step, mostly idle tensor cores.
- The replacement is a skinny GEMV that issues **the same `QMMA.16832.F32.E4M3.E4M3` instruction with
  the same K order and epilogue**, so the output bits are identical.
- Results:
  - 0 mismatches over 6400 cases, including extreme scales and graph M = 1..8;
  - the 39 kernels are replaced 1:1;
  - a single GEMV moves 708 GB/s vs 409 GB/s for CUTLASS;
  - the 80-projection decode sequence is 1.45-1.47x faster.
- The library is loaded only by sha256-pinned path, after an ABI probe and a bit-exact self-test at
  start-up.

### 3.5 Attention batching, k=2 (`QWEN36_LAYER_MAJOR_ATTN_BATCH=2`)

- At chunk 2048, a FlashInfer prefill call launches 512 CTAs = 3.66 waves on the 70 SMs. The last wave
  is mostly empty (SM utilization 89%).
- Under layer-major, adjacent chunks of the same layer are available together. So QKV, RoPE and the KV
  write run per chunk, and **one FlashInfer call covers 2 chunks**.
- This is used only when it is provably bit-identical:
  - every chunk's own plan and the group plan share `CTA_TILE_Q`;
  - no split-KV;
  - q-tile-aligned starts;
  - single request, page size 1, causal.
  Otherwise the engine falls back to k = 1.
- TTFT ratio vs k = 1: 0.924 / 0.966 / 0.950 at 16K / 64K / 120K. The change also fixed a latent
  plan-cache-by-rows bug: split-KV also depends on KV length.

### 3.6 Decode gather with TMA bulk copies (`QWEN36_DECODE_GATHER=tuned`)

- Upstream gathers missed experts with `fast_index_copy_multi<int32,1024,8>`: 48 CTAs x 1024 threads,
  each with one 16-byte load in flight.
- The f1 kernel `DecodeGatherTmaKernel<8,3,16384>` uses 8 CTAs, each with a 3-stage ring of 16 KiB
  `cp.async.bulk` transfers. It is byte-exact (44,280 verified calls).
- Results:
  - per call it is 1.12-1.30x faster;
  - model decode is +5.6% at 16K and +4.8% at 120K (paired A/B);
  - combined with FP8 GEMV, decode is +13-17%.
- In the model it moves ~42 GB/s (38.9-42.8 GB/s for 1-8 experts). A byte-checked ceiling sweep shows every
  host-to-GPU path on this WSL2 host tops out at 41-45 GB/s, so the gather already runs at the link ceiling.
  The remaining lever is hiding it (overlap with the shared expert), not copying faster.

### 3.7 Smaller f1 items

- **Fused attention output gate.** Gate + static FP8 quantization run in one kernel
  (`kernels/attention-gate`).
- **FP32 GDN snapshots.** Prefix-cache GDN snapshots are taken in FP32, instead of upstream's rounding
  of the FP32 accumulator through a BF16 intermediate.
- **Retained chunk snapshot and prefix margin of 8 tokens.** Edited conversations now reuse 16K-60K
  cached tokens instead of 0. Edits at 16K are 8.8-9.4x faster in TTFT.
- **Kernel provenance.**
  - Upstream loads prebuilt kernels by name only. An edited `.cu` file could silently run an old `.so`.
  - f1 binds every module to a source-sha256 manifest and serves with `FREETOKEN_DISABLE_JIT=1`, so it
    refuses anything unbound and never compiles at serve time.
  - ctypes libraries are bound by sha256 and checked with an ABI probe.
- **API guard.**
  - Requests are admitted only if their tokenized length fits the context; context + 1 returns 400
    `context_length_exceeded`.
  - Unsupported tool options return 400 before any SSE output.
  - Seeds are request-local.
  - An overlap-scheduler output-budget bug was fixed: requests had stopped at 127 of 128 tokens.

## 4. Host-side runtime

### 4.1 Expert bank as THP-backed `MAP_PRIVATE` memory

- A proposed huge-page patch called `madvise(MADV_HUGEPAGE)` on Python's `mmap.mmap(-1, n)`. That
  mapping is **`MAP_SHARED`** (shmem), and WSL2 runs `shmem_enabled=never`, so the call was a silent
  no-op. A 256 MiB test showed 0 kB `ShmemPmdMapped`.
- f1 maps the banks **`MAP_PRIVATE` anonymous** + `MADV_HUGEPAGE`. The test then showed 262144/262144 kB
  `AnonHugePages`. Bank loads are threaded, and child processes are spawned rather than forked.
  `FREETOKEN_BANK_THP=0` restores the old mapping.
- No system setting was changed. In the release soak the bank shows up as anonymous memory, with shmem
  at 0.1 GiB.
- Microbenchmark: the copy engine goes from 47.0 to 48.4-49.4 GB/s.

### 4.2 glibc malloc tunables

- The first 2-hour soak failed its memory gate: host anon memory grew +394 MB, in steps that lined up
  with fresh 64K/120K requests. glibc arena retention was the hypothesis, given 28 cores and up to
  224 arenas.
- The launchers now export
  `GLIBC_TUNABLES=glibc.malloc.arena_max=2:glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=4194304`.
- Results:
  - the 1-hour test grew +13.7 MB, and the baseline anon dropped by 192 MB;
  - the f1 soak grew +45-48 MB in 2 hours, against a flag threshold of 268 MB;
  - the tunables are verified in the service process environment before and after the soak.

## 5. Correctness properties checked along the way

### 5.1 Prefix-cache header dedup makes logits depend on cache history

- The 256K profile first showed one deviating teacher case, long-120000: max |Δlogit| 1.23, NLL
  +0.0177, PPL +0.072%. Every other case was bit-identical.
- Root cause:
  - When a prefill is committed, the hybrid radix cache `insert()` finds the shared 24-token chat
    header already in the tree.
  - It frees the request's own KV pages and re-points `page_table[req, 0:24]` at pages written by
    **other requests**. Those requests computed the header with different GEMM shapes, so the values
    differ at rounding level.
  - `#cached-token` stays 0, because match_prefix counts only nodes with a GDN snapshot.
- Why it showed only at 256K:
  - At 128K, admitting 120K tokens evicts the header nodes, so there is nothing to dedup.
  - At 256K nothing is evicted, so the header is deduped.
- Confirmed on GPU in both directions:
  - the 128K build reproduces the 256K numbers exactly when the same cache history is replayed;
  - the 256K build is fully bit-exact when the case runs on a clean cache.
- This is standard prefix-cache behaviour (SGLang-style re-point) and not a correctness bug. Both
  results are valid numerics. The teacher harness (v7) now restarts the server before that case. The
  property is written up as an upstream feature request ([upstream-notes/03](upstream-notes/03-freetoken-prefix-cache-history-dependence.md)).

### 5.2 Commit/decode race: verified safe

- **The concern.** On commit, the request's own header pages are freed while decode step 1 may already
  be in flight. Step 1 is launched before the overlap scheduler drains.
- **Code analysis.**
  - The engine and scheduler streams wait on each other at every hand-off.
  - Step 1's read/write indices are copied before the drain.
  - Freed pages go to the back of the free list.
  - The page-table re-point is a device-to-device copy issued after the wait, and GDN slot reuse is
    ordered on the engine stream.
- **GPU confirmation.** The workload was 16 greedy requests with shared prefixes, giving 13 dedup
  commits, all while step 1 was in flight.
  - Arm P NaN-poisoned every freed KV/GDN slot (23 fills, 23,925 tokens). It is bit-identical to the
    unpoisoned arm: 16/16 texts and 640/640 per-step logit checksums.
  - The negative control removes the stream wait at the hand-off. It fails exactly at the race site:
    13/16 texts differ, and the first divergence is decode step 1 of the first dedup request.
- Not covered: eviction churn and multi-request batches (the server runs batch 1).

### 5.3 sm_120: `cp.async.bulk.shared::cluster` turns into a driver syscall

- The first TMA gather build used the common `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`
  form. That build cost **1.37 GiB of VRAM**, which pushed the GPU into WDDM paging.
- What happens on sm_120: that form lowers to `CALL.ABS.NOINC __cuda_syscall_cp_async_bulk_unicast`,
  visible as `EIATTR_EXTERNS` / `EIATTR_SYSCALL_OFFSETS`. `ptxas` reports 0 bytes of stack. But on the
  first launch the driver raises the context's per-thread stack limit from 1024 to 14416 B:
  (14416 - 1024) x 70 SMs x 1536 threads = 1373 MiB, never returned.
- The fix is the `.shared::cta` destination form (PTX 8.6+). It lowers to native `UBLKCP`, costs 0 MiB,
  produces identical bytes at unchanged speed, and drops registers from 46 to 41.
- Rule for this repository: **no `.shared::cluster` bulk copies in sm_120 kernels.** The same pattern
  was cleaned out of the experimental, unshipped kernels; a compile check found 0 syscall references.
- The issue is in NVIDIA's toolchain/driver rather than FreeToken (upstream has no bulk copies), so it
  is noted here instead of in an issue draft.

## 6. Where the time goes (release f1, Nsight Systems)

| Phase | Breakdown |
|---|---|
| Prefill 16K (1.53 s) | MoE 40%, attention 19%, FP8 dense 15%, GDN 8%, exposed H2D 5%, idle 4.5% |
| Prefill 120K (21.9 s) | attention 65%, MoE 18%, FP8 7%, GDN 4%, idle 1.3% |
| Short prompt, 278 tokens (0.60 s) | exposed H2D 75.5% (the 18.25 GB bank, only 12% overlapped) |
| Decode at 16K (8.10 ms/step) | FP8 28%, MoE 20%, expert gather 18%, attention 7%, idle 8.6% |
| Decode at 120K (11.39 ms/step) | attention 31%, FP8 20%, MoE 15%, gather 14% |

A decode step is about 998 kernels with about 0.48 ms of inter-kernel gaps.

Component ncu bounds:

- prefill attention and the MoE GEMM are tensor-pipe-bound (SM ~82-83%);
- decode attention runs at 94% of DRAM peak;
- the GDN chunk kernels and FP8 GEMV run at 79-90% of DRAM;
- the decode gather shows ~30-33 GB/s under the profiler; in the model it runs at ~42 GB/s, at the measured 41-45 GB/s host-link ceiling.

The remaining bit-exact levers are therefore the PCIe side, short-prompt streaming and the gather;
see [ROADMAP.md](ROADMAP.md).
