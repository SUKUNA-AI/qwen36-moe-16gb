# qwen36-moe-16gb

[Русская версия](README.md)

Qwen3.6-35B-A3B (NVFP4, all 256 experts, top-8 routing) served from a single 16 GB consumer GPU,
an RTX 5070 Ti (SM120), through a modified [FreeToken](https://github.com/FlashML-org/FreeToken) engine.
The full expert set (18.25 GB) lives in pinned host RAM. The GPU holds the dense weights, the KV cache
and a cache of expert slots.

The engine changes and CUDA kernels in this repository make the server 2.5-9x faster on prompts of
4K tokens and up and about 20-30% faster at decode than the original FreeToken setup on the same
machine. Teacher forcing shows **bit-identical logits** against the reference build. Release **f1**
passed every gate on 2026-10-14.

> Status: a single-user home deployment (batch 1). The numbers below are from one machine. Every
> number has a definition and a source file; see [docs/RESULTS.md](docs/RESULTS.md).

## Headline results

### 1. Paired matrix, release f1 vs the FreeToken baseline at 128K

The baseline `bf128` is upstream FreeToken mechanics: Triton NVFP4 MoE, chunk 2048, BF16 KV,
131072 context, plus the FP32 global-scale fix (see "Quality" below). The protocol was 5 paired rounds
in rotating order with exactly 128 output tokens per request, run in one session. The table shows
medians.

| Case | TTFT bf128 | TTFT f1 | TTFT speedup | Decode bf128 | Decode f1 | Full request, geomean speedup |
|---|---:|---:|---:|---:|---:|---:|
| 64-token prompt | 0.75 s | 0.79 s | 0.94x (non-gating) | 96.4 tok/s | 122.8 tok/s | 1.13x |
| 4K | 2.24 s | 0.77 s | 2.9x | - | - | 1.95x |
| 16K | 6.78 s | 1.59 s | 4.3x | 94.2 tok/s | 119.1 tok/s | 3.03x |
| 64K | 28.46 s | 9.14 s | 3.1x | 84.4 tok/s | 99.3 tok/s | 2.86x |
| 120K | 54.77 s | 22.07 s | 2.5x | 80.7 tok/s | 94.6 tok/s | 2.45x |
| Last-message edit at 16K | 6.85 s | 0.78 s | 8.8x | | | 4.44x |
| Tool-result edit at 16K | 7.11 s | 0.76 s | 9.4x | | | 4.49x |
| Tool-call cycle (natural EOS) | | | | | | 1.08x (CI 0.97-1.20) |

Verdict: **PASS**. f1 is non-inferior on all 9 gating cases, and the gain of at least 5% holds
under Bonferroni correction. Decode across the matrix is 1.19-1.32x.

### 2. Like-for-like at 64K context (original FreeToken vs release)

The three arms ran in one session at context 65536 with 192 output tokens, a 60 s warm-up and 2 valid
rounds in opposite order. Each cell shows TTFT / decode, as the mean of the 2 rounds.

| Prompt | Original FreeToken (FP16 globals) | + FP32 global fix | Release f1 at 64K | f1 vs original |
|---|---:|---:|---:|---|
| 4K | 1.93 s / 116.0 tok/s | 1.85 s / 118.1 | **0.58 s / 142.4** | TTFT 3.3x, decode +23% |
| 16K | 6.04 s / 111.3 | 5.86 s / 115.7 | **1.49 s / 138.0** | TTFT 4.1x, decode +24% |
| 60K | 22.86 s / 101.3 | 22.43 s / 107.6 | **7.50 s / 122.2** | TTFT 3.0x, decode +21% |

### 3. 250K-class context (`long256` profile, 262144 tokens)

| Measurement | Original FreeToken mechanics | Release f1 |
|---|---:|---:|
| Fresh 250K prompt, TTFT | 126.5 s (single run, 2026-10-13, Triton, chunk 4096) | **77.4 s** p50 of 11 (30-min test); 80.4 s in the release gate |
| Fresh 250K prompt, decode | - | 60.8 tok/s p50 |
| 261,468-token prompt at the context limit, TTFT | 139.5 s (median of 5, `bf256`, chunk 2048) | **91.3 s** (single run) |
| Same, decode | ~52.5 tok/s | 52.5 tok/s |
| 250K continuation (prefix-cache hit) | - | 1.06 s TTFT, 11/11 hits |

These are cross-session comparisons, not paired rounds. The 256K paired matrix ran on an earlier
build; see [docs/RESULTS.md](docs/RESULTS.md).

### 4. Latency under a 2-hour soak (main128, 256 output tokens, sequential)

| Prompt tokens | TTFT p50 / p99 | Decode p50 |
|---:|---:|---:|
| 93 | 0.60 / 0.67 s | 133.7 tok/s |
| 4K | 0.61 / 0.65 s | 135.0 tok/s |
| 16K | 1.36 / 1.61 s | 129.8 tok/s |
| 64K | 8.14 / 9.29 s | 112.2 tok/s |
| 120K | 20.97 / 22.92 s | 97.5 tok/s |

The soak ran for 7212 s: 741/741 checks passed with 0 errors and 0 stalls. Host memory grew by only
45-48 MB over the 2 hours.

## Quality guarantee

- **Bit-exact teacher forcing.** The release build and the reference `bf128` are fed the same 20
  fixed cases (943 forced target tokens). At all **125/125** logit checkpoints the logits are
  identical: max |Δlogit| = 0. Relative perplexity growth is **+0.0%** (CI [0, 0]), and there are no
  new functional failures. The 256K profile is also 125/125 bit-exact when the 120K case runs on a
  clean prefix cache (harness v7). See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
- Every speed change in f1 is bit-exact by construction and was gated individually: layer-major
  prefill, the CUDA MoE / GDN / FP8 GEMV kernels, attention batching and the TMA gather.
  Non-bit-exact options were measured and left out, such as FP8 KV, larger chunks and fused FLA in f1.
- The reference keeps the NVFP4 global scales (`weight_scale_2`) in **FP32**. Upstream FreeToken casts
  them to FP16, and 94% of them become subnormal. That fix changes numerics on purpose. Relative to the
  FP16 basis, perplexity improves by 1.13% (CI [-2.27%, -0.08%]), and a pre-registered 60-case eval
  scores 59/60 vs 57/60.
- Scope: bit-exactness is proven on fixed fixtures. It is not a proof for arbitrary inputs. Prefix-cache
  state can change rounding in both builds in the same way. See the header-dedup note in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5070 Ti 16 GB (SM120, Blackwell), PCIe 5.0 x16. It also drives the Windows desktop (4K at 159 Hz), which uses about 2.5-3 GiB of VRAM |
| CPU / RAM | Intel Core i7-14700KF. The WSL2 VM sees about 47 GiB, and the service is capped at MemoryHigh 32 GiB |
| OS | Windows 11 + WSL2 (WSL 2.7.3, Arch Linux, kernel 6.6, systemd 262); NVIDIA Windows driver 617.14 |
| CUDA | CUDA 13.4 (toolkit 13.4.1, nvcc 13.4.59), GCC 15.3 |
| Python stack | Python 3.12.14, torch 2.11.0 (cu130), Triton 3.6.0, FlashInfer 0.6.18.post1, sglang-kernel 0.4.5, transformers 5.16.1 |
| Engine | FreeToken 0.1.3 (`cac247a860e316e06580d05aeb05f2e647bde214`) + the changes in `freetoken/` |
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a052410cfd62085d94b58866fd0f2c3c5`: NVFP4 experts, FP8 dense, selected BF16 |

## main128 next (release candidate)

> **Status: installed on 2026-10-16** (label `main128-s5r1`; f1 is kept for rollback). All release gates passed: teacher 125/125, release A/B against f1 with 5 pairs (PASS), 131072 context, a 2 h soak, and long256 on the new tree.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/model-blocks-dark.svg">
  <img alt="Block diagram of the model: a token's path through 40 layers, where weights and experts live, where the release levers act" src="docs/img/model-blocks.svg" width="100%">
</picture>

The diagram labels are in Russian; edit it in draw.io: [`docs/model-blocks.drawio`](docs/model-blocks.drawio), generator [`tools/model-diagram.py`](tools/model-diagram.py).

The release is f1 plus the levers below. Each lever has its own switch and is off by default. Each is bit-exact: teacher 125/125, `max_abs` 0. Run configuration: [`harness/main-release-next-d5.json`](harness/main-release-next-d5.json). Profile: `main128` in [`serving/profiles.json`](serving/profiles.json).

| Lever | Profile → environment | What it does |
|---|---|---|
| hit-D2D | flag `--moe-prefill-hit-d2d`, `layer_major_min_tokens: 2049` → `QWEN36_LAYER_MAJOR_MIN_TOKENS=2049` | short prefill copies experts already in VRAM on the GPU; layer-major starts at 2049 uncached tokens |
| FLA on CUDA | `cuda_gdn_fla: "wu"` → `QWEN36_CUDA_GDN_FLA=wu` | fused bit-exact CUDA kernels for the GDN FLA chunk |
| T3 | `decode_shared_overlap` → `QWEN36_DECODE_SHARED_OVERLAP=1` | the decode miss gather overlaps the same layer's shared expert |
| host embeddings | `host_embeddings` → `QWEN36_HOST_EMBEDDINGS=1` | the embedding table (~0.95 GiB) moves to pinned host memory; the VRAM goes to expert slots |
| memory step | `memory_ratio: 0.85` | more expert slots: 4611 instead of 3164 |
| decode o-proj | `cuda_oproj_gemv` → `QWEN36_OPROJ_GEMV=1` + `libqwen_oproj_gemv.so` | bitwise copy of the cuBLASLt split-K for the 40 o-projections, M 1..8 |
| decode MoE | `cuda_moe_decode` → `QWEN36_MOE_DECODE_CUDA=1` + `libqwen_moe_decode.so` | CUDA GEMV instead of Triton for decode gate_up/down |
| C0 | `prefill_copy_priority` → `QWEN36_PREFILL_COPY_PRIORITY=1`, `QWEN36_PREFILL_EPOCH_INVALIDATE=1` | top-priority prefill copy stream, one buffer invalidation per epoch |
| C1 | `prefill_selected: 160` → `QWEN36_PREFILL_SELECTED=160` (needs `decode_gather`) | a chunk under 160 tokens moves only the experts its router selected |
| D5 | `decode_shared_epilogue` → `QWEN36_DECODE_SHARED_EPILOGUE=1` | the MoE block epilogue (σ(gate)·shared + routed) in one kernel, with torch's roundings |

Tested and left out: D3b (one-layer-ahead prefetch, decode −4%), D6 (warming slots from prefill, no gain), C2 (TMA instead of the copy engine, slower). D3b also has a variant with its own high-priority stream (`QWEN36_DECODE_PREGATE_PRIORITY=1`, off by default): in the microbenchmark it hides 44–92% of the copy, on the model it is not measured yet — a next-cycle candidate ([report](next-update/next-candidate/REPORT.md)). Nsight Compute on the model is not captured: under ncu the server does not process requests ([diagnosis](results/ladder/ncu.json)); kernel profiles come next from the component benchmarks.

Modules and switches: [`freetoken/MODIFICATIONS.md`](freetoken/MODIFICATIONS.md).

**Memory.** Owner rule: exactly 1 GiB of VRAM stays with Windows and the model gets the rest. The validator caps `memory_ratio < 0.95`. The 0.85 step won the soaks; at 0.90, decode and TTFT were worse from memory pressure.

**Release A/B, f1 vs main128** (5 pairs in one session, 128 output tokens, medians; `compare-ab` PASS):

| Case | TTFT f1 → main128 | Decode f1 → main128, tok/s | Whole request |
|---|---:|---:|---:|
| 64-token prompt | 0.389 → **0.199 s** (−49%) | 153 → 182 (+19%) | ×1.37 |
| 4K | 0.407 → 0.388 s | 150 → 189 (+25%) | ×1.18 |
| 16K | 1.343 → 1.315 s | 143 → **176** (+23%) | ×1.10 |
| 120K | 20.52 → 20.37 s | 106 → 121 (+14%) | ×1.015 |
| History continuation, 4K | 0.391 → 0.255 s (−35%) | 154 → 191 (+24%) | ×1.33 |
| Last-message edit, 16K | 0.400 → 0.298 s (−26%) | 137 → 169 (+24%) | ×1.29 |
| Tool-result edit, 16K | 0.400 → **0.152 s** (−62%) | 138 → 175 (+27%) | ×1.51 |

**Ladder** (cumulative, one lever per step; 3 pairs, medians; every step teacher 125/125):

| Step | TTFT 64 | TTFT 16K | Decode 16K | Decode 120K | Slots |
|---|---:|---:|---:|---:|---:|
| f1 | 0.390 | 1.344 | 143.4 | 103.1 | 3164 |
| + hit-D2D (T1) | 0.315 | 1.343 | 143.9 | 103.5 | 3164 |
| + CUDA FLA (T2) | 0.317 | 1.316 | 143.6 | 103.4 | 3164 |
| + T3 | 0.304 | 1.330 | 148.6 | 106.1 | 3164 |
| + host embeddings | 0.314 | 1.319 | 153.8 | 108.9 | 3729 |
| + `memory_ratio` 0.85 | 0.259 | 1.315 | 159.5 | 112.1 | 4611 |
| + CUDA o-proj and MoE decode | 0.270 | 1.321 | 170.3 | 118.4 | 4603 |
| + C0 | 0.252 | 1.315 | 171.7 | 118.4 | 4603 |
| + C1 | 0.207 | 1.323 | 173.8 | 119.5 | 4603 |
| **+ D5 = main128** | **0.201** | 1.318 | **176.4** | **120.5** | 4603 |

Decode step at 16K (Nsight Systems): 7.07 → 5.78 ms. long256 on the new tree, 250K: TTFT 72.7 s, decode 67.3 tok/s (f1: 77.4 s / 60.8); with the main128 levers at 262,144: 76.0 tok/s.

Charts: [`results/ladder/charts/ttft-steps.svg`](results/ladder/charts/ttft-steps.svg), [`decode-steps.svg`](results/ladder/charts/decode-steps.svg), [`memory-steps.svg`](results/ladder/charts/memory-steps.svg), [`teacher-steps.svg`](results/ladder/charts/teacher-steps.svg). Table: [`results/ladder/charts/ladder-table.en.md`](results/ladder/charts/ladder-table.en.md). Write-up: [`docs/RELEASE-MAIN128.en.md`](docs/RELEASE-MAIN128.en.md).

**Release and rollback.** One procedure, [`harness/release-main128.sh`](harness/release-main128.sh): `--dry-run`, then `--check`, then `--apply` (owner approval only). `--rollback` goes back to f1. The steps and the log lines to watch are in [`harness/OPERATIONS.md`](harness/OPERATIONS.md#main128-next-релиз-и-откат) (Russian).

## Profiles

| Profile | Context | Expert slots on GPU | Use |
|---|---:|---:|---|
| `main128` (default) | 131,072 | 4603 | everyday use: f1 plus the main128 next levers |
| `long256` | 262,144 | 1481 | long documents; same switches, more KV and fewer slots (slower decode) |
| `baseline64` | 65,536 | 3917 | unchanged fallback (original mechanics, FP32 globals) |

## Repository layout

| Path | Contents |
|---|---|
| `freetoken/` | The modified FreeToken engine (Apache-2.0). `freetoken/MODIFICATIONS.md` lists every switch and change |
| `kernels/` | CUDA sources for the bit-exact kernels: `nvfp4-moe`, `gdn`, `fp8-gemv`, `attention-gate`. The TMA decode gather lives in `freetoken/` as a JIT module |
| `serving/` | API ingress/guard, profile runtime validator, service and launcher scripts |
| `harness/` | Qualification and benchmark harness: teacher forcing, paired matrix, soak, context gates, run configs |
| `config/` | Profile definitions |
| `docs/` | [ARCHITECTURE](docs/ARCHITECTURE.md), [RESULTS](docs/RESULTS.md), [METHODOLOGY](docs/METHODOLOGY.md), [ROADMAP](docs/ROADMAP.md), [upstream-notes](docs/upstream-notes/00-INDEX.md) |
| `results/` | Evidence: dashboard, metrics, Nsight summaries, matrix/soak/gate JSONs, experiment results. See [docs/RESULTS.md](docs/RESULTS.md) for the file map |
| `local/` | Machine-specific material. It is **gitignored** and not part of the public tree |

## Quick start (pointers)

The project targets one specific setup: Linux (WSL2) with the checkpoint on local disk and a 16 GB SM120
GPU. It is not a one-command install.

1. Install the Python stack above in a venv and install `freetoken/` from source. Follow
   `freetoken/README.md` for upstream's instructions, then read `freetoken/MODIFICATIONS.md`.
2. Build the kernels in `kernels/` for `sm_120a`. The engine loads them by sha256-pinned path. With
   `FREETOKEN_DISABLE_JIT=1` it serves only manifest-bound prebuilt kernels and never compiles at serve
   time.
3. Pick a profile from `config/` and start it through the launcher in `serving/`. The profile runtime
   refuses unqualified or inconsistent switch combinations.
4. To check a modified build, run the harness in `harness/`: teacher forcing against your own
   reference, then the paired matrix. The procedure is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

The API is OpenAI-compatible chat completions with streaming and auto tool calling.

## Limitations

- **One request at a time** (batch 1, no queue; a busy server returns 429). Concurrency was never measured.
- **Text only.** Vision and MTP/speculative decoding are not loaded.
- **No constrained decoding.** `tool_choice` set to required or named, `strict: true`,
  `parallel_tool_calls=false` with tools, and legacy `functions` are all rejected with 400.
- **Short-prompt TTFT floor of about 0.57-0.6 s.** Every prefill streams the full 18.25 GB expert bank
  over PCIe. A 64-token prompt is slightly slower than the baseline (0.79 vs 0.75 s). The fix is the
  next update ([ROADMAP](docs/ROADMAP.md)).
- **Known near-tie failures of the checkpoint.** Two fixed functional cases fail under greedy decoding,
  identically in the reference and the release; a third fails on `baseline64`. On one tool fixture,
  thinking-off sometimes answers directly instead of calling the tool. Enabling thinking is recommended
  for tool workflows.
- **VRAM headroom depends on the desktop.** The soak saw 4 of 825 requests dip below 512 MiB free
  (lowest 367 MiB), with no stall.
- **WSL2 specifics.** There is a single copy engine. WDDM time-slices the GPU with the desktop
  (measurement noise). A WSL interop issue is described in [docs/RESULTS.md](docs/RESULTS.md).
- **256K was not fully soaked.** It passed a 30-minute stability test only, not a 2-hour soak.

## Credits and license

- **FreeToken** (FlashML-org/FreeToken, Apache-2.0) is the engine this work modifies. The upstream
  copyright and license are kept in `freetoken/LICENSE`, and the changes are listed in
  `freetoken/MODIFICATIONS.md`.
- **NVIDIA** published the `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint, quantized from Qwen3.6-35B-A3B
  by the **Qwen team**. The weights are not included here and keep their own license.
- The work also builds on FlashInfer, Triton, flash-linear-attention, sglang-kernel and PyTorch.
