# Modifications to FreeToken

This directory is a modified copy of **FreeToken** (https://github.com/FlashML-org/FreeToken),
licensed under the Apache License, Version 2.0 (see `LICENSE` in this directory).

- Upstream base: commit `cac247a860e316e06580d05aeb05f2e647bde214` ("chore(release): 0.1.3 (#489)").
- Modified by the qwen36-moe-16gb project, 2026-10. The modifications are also licensed under
  Apache-2.0. Upstream copyright and license notices are retained unchanged.
- Upstream git history is not included; to reproduce the diff, check out the base commit and
  compare the files listed below.

Every change targets Qwen3.6-35B-A3B-NVFP4 on a single 16 GB SM120 GPU (RTX 5070 Ti) with
host-memory expert offload. All performance paths are opt-in environment switches; with none of
them set the engine behaves like upstream except for the bug fixes and telemetry listed below.
The released profiles enable them through `serving/profile-runtime.py`.

## Switches used by the released profiles (`main128`, `long256`)

| Switch | Module(s) | What it does |
|---|---|---|
| `QWEN36_LAYER_MAJOR_PREFILL=1` | `engine/layer_major.py`, `moe/layer_major.py`, `scheduler/layer_major.py` | Layer-major prefill: each MoE layer's experts are streamed once per prompt instead of once per chunk; bitwise equal to chunked prefill. |
| `QWEN36_LAYER_MAJOR_ATTN_BATCH=2` | same + `attention/fi.py` | Batches 2 adjacent chunks into one FlashInfer prefill call when that is provably bit-identical. |
| `QWEN36_NVFP4_PREFILL=cuda` + `QWEN36_CUDA_NVFP4_LIBRARY` | `moe/cuda_nvfp4.py` | Bit-exact CUDA NVFP4 MoE prefill (library sources in `kernels/nvfp4-moe`). |
| `QWEN36_CUDA_GDN=1` + `QWEN36_CUDA_GDN_LIBRARY` | `kernel/cuda_gdn.py`, `models/qwen3_5_moe/gdn.py` | Bit-exact fused CUDA GatedDeltaNet kernels (ABI 2; `kernels/gdn`). |
| `QWEN36_FP8_GEMV=1` + `QWEN36_FP8_GEMV_LIBRARY` | `kernel/cuda_fp8_gemv.py` | Bit-exact FP8 row-wise GEMV for decode (M<=8); library pinned by sha256 (`kernels/fp8-gemv`). |
| `QWEN36_DECODE_GATHER=tuned` | `kernel/decode_gather.py`, `kernel/csrc/jit/decode_gather.cuh` | TMA bulk decode-miss expert gather (`.shared::cta` destination), byte-exact. |
| `QWEN36_FUSED_ATTN_GATE=1` + `QWEN36_FUSED_ATTN_GATE_LIBRARY` | `kernel/cuda_attention_gate.py`, `kernel/csrc/qwen36_attention_gate.cu` | Fused attention output gate (`kernels/attention-gate`). |
| `QWEN36_FP32_GDN_SNAPSHOT=1`, `QWEN36_RETAIN_CHUNK_SNAPSHOT=1`, `QWEN36_PREFIX_MARGIN=8` | `scheduler/*`, `models/qwen3_5_moe/*` | FP32 GDN prefix snapshots and snapshot placement that keeps prefix-cache hits exact. |
| `FREETOKEN_DISABLE_JIT=1` | `kernel/kernel_provenance.py`, `kernel/kernel_cache_manifest.json` | Serve only manifest-bound prebuilt kernels; never JIT at serve time. |
| `FREETOKEN_BANK_THP` (default on) | `moe/host_banks.py` | Private anonymous host expert banks with `MADV_HUGEPAGE`. |

## Other changes

- NVFP4 global scales kept in FP32 (upstream casts `weight_scale_2` to FP16, which makes most of
  them subnormal); `QWEN36_NVFP4_GLOBAL_FP16_COMPAT=1` restores the upstream behaviour.
- Teacher-forcing qualification support (`engine/qualification.py`, `QWEN36_TEACHER_FIXTURE`,
  `QWEN36_LOGIT_REFERENCE`) used by the bit-exact quality gate.
- Engine telemetry (`engine/telemetry.py`): GPU event spans, bounded asynchronous metrics.
- Strict tokenized context admission, request-local seeds, overlap-scheduler output budget fix,
  cancellation fixes (see the per-file list below).

## Present but OFF (not part of any released profile)

Kept flag-off and bit-exact for future work; documented in `docs/` and `next-update/`:
`QWEN36_CUDA_GDN_FLA` (fused FLA chunk kernels), `QWEN36_LAYER_MAJOR_REFILL` (`moe/slot_refill.py`),
`QWEN36_PREFILL_SYNCFREE_INVALIDATE`. Rejected experiments that remain importable but disabled:
native MTP (`engine/mtp.py`), speculative/prompt-lookup decoding (`engine/speculative.py`,
`scheduler/speculative.py`), RAM complement residency (`moe/complement*.py`,
`moe/cuda_transport.py`, `kernel/csrc/qwen36_complement.cu`, `qwen36_transport.cu`), fused QK-norm/RoPE/KV
(`kernel/cuda_kv*.py`, `kernel/csrc/qwen36_kv.cu`) and FP8 KV (`kernel/triton/fp8_kv.py`).

## Files modified relative to the upstream base

```
python/freetoken/attention/fi.py
python/freetoken/attention/linear.py
python/freetoken/core.py
python/freetoken/engine/engine.py
python/freetoken/engine/sample.py
python/freetoken/kernel/aot.py
python/freetoken/kernel/aot_models.py
python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp
python/freetoken/kernel/fla/chunk.py
python/freetoken/kernel/fla/chunk_delta_h.py
python/freetoken/kernel/fla/utils.py
python/freetoken/kernel/triton/fp8_pertensor_linear.py
python/freetoken/kernel/utils.py
python/freetoken/kvcache/mha_pool.py
python/freetoken/layers/embedding.py
python/freetoken/layers/moe.py
python/freetoken/layers/quantization/linear/fp8_tensor.py
python/freetoken/layers/quantization/linear/nvfp4.py
python/freetoken/layers/quantization/moe/base.py
python/freetoken/layers/quantization/moe/nvfp4.py
python/freetoken/message/tokenizer.py
python/freetoken/models/gemma4/weight.py
python/freetoken/models/glm4_moe/weight.py
python/freetoken/models/loader.py
python/freetoken/models/nvfp4_banks.py
python/freetoken/models/qwen3_5_moe/attention.py
python/freetoken/models/qwen3_5_moe/gdn.py
python/freetoken/models/qwen3_5_moe/gdn_kernels.py
python/freetoken/models/qwen3_5_moe/model.py
python/freetoken/models/qwen3_5_moe/weight.py
python/freetoken/moe/benchbw.py
python/freetoken/moe/cpu_executor.py
python/freetoken/moe/fused_nvfp4.py
python/freetoken/moe/host_banks.py
python/freetoken/moe/offload_cache.py
python/freetoken/scheduler/cache.py
python/freetoken/scheduler/prefill.py
python/freetoken/scheduler/scheduler.py
python/freetoken/server/api_models.py
python/freetoken/server/generation.py
python/freetoken/server/openai_api.py
python/freetoken/tokenizer/tokenize.py
```

## Files added (not in upstream)

```
python/freetoken/engine/layer_major.py
python/freetoken/engine/mtp.py
python/freetoken/engine/qualification.py
python/freetoken/engine/speculative.py
python/freetoken/engine/telemetry.py
python/freetoken/kernel/csrc/jit/decode_gather.cuh
python/freetoken/kernel/csrc/qwen36_attention_gate.cu
python/freetoken/kernel/csrc/qwen36_complement.cu
python/freetoken/kernel/csrc/qwen36_kv.cu
python/freetoken/kernel/csrc/qwen36_nvfp4_matched.cu
python/freetoken/kernel/csrc/qwen36_nvfp4_prefill.cu
python/freetoken/kernel/csrc/qwen36_transport.cu
python/freetoken/kernel/cuda_attention_gate.py
python/freetoken/kernel/cuda_fp8_gemv.py
python/freetoken/kernel/cuda_gdn.py
python/freetoken/kernel/cuda_kv.py
python/freetoken/kernel/cuda_kv_integration.py
python/freetoken/kernel/decode_gather.py
python/freetoken/kernel/kernel_cache_manifest.json
python/freetoken/kernel/kernel_provenance.py
python/freetoken/kernel/triton/fp8_kv.py
python/freetoken/moe/complement.py
python/freetoken/moe/complement_cuda.py
python/freetoken/moe/complement_kernels.py
python/freetoken/moe/cuda_nvfp4.py
python/freetoken/moe/cuda_transport.py
python/freetoken/moe/layer_major.py
python/freetoken/moe/slot_refill.py
python/freetoken/scheduler/layer_major.py
python/freetoken/scheduler/speculative.py
tests/engine/test_speculative.py
tests/kernels/test_cuda_fp8_gemv.py
tests/kernels/test_gdn_fp32_snapshot.py
tests/moe/test_decode_gather.py
tests/scheduler/test_overlap_output_budget.py
tests/scheduler/test_prefix_snapshot_identity.py
tests/scheduler/test_retain_chunk_snapshot.py
tests/scheduler/test_speculative_state.py
```

Two default library paths were additionally rewritten from an absolute user path to `~/qwen36-api/...`
(`moe/cuda_nvfp4.py`, `moe/cuda_transport.py`) for publication; behaviour is unchanged when the runtime root is `~/qwen36-api`.
