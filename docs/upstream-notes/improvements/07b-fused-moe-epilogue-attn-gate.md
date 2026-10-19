<!-- READY, NOT POSTED. Target: FlashML-org/FreeToken, new issue (feature proposal). Paste from the "# " title line down. Checked against upstream main 4d2e2e2 (2026-10-19). -->

# [Perf] Qwen3.5/3.6: bit-exact fused elementwise epilogues for the MoE block (sigmoid gate, multiply, add) and the attention output gate (+ FP8 quant)

## Summary

Two small, bit-exact launch-count cuts for Qwen3.5-family decode. (1) The MoE block ends with three bf16 PyTorch
kernels per layer, `sigmoid(gate)`, `shared * sig` and `routed + prod`. A single Triton kernel does the same
operations with the same roundings, 80 fewer launches per decode step. It must be compiled with
`enable_fp_fusion=False`, otherwise Triton contracts the multiply-add into one `fma.rn.bf16` and the bits
change. (2) Before `o_proj`, each full-attention layer runs sigmoid, a bf16 multiply and the static FP8 quant as
three kernels. One CUDA kernel does all three and keeps both intermediate bf16 roundings. On our setup, (1) took
the 40-layer epilogue from 131.2 to 61.3 us per step and decode at 16K from 173.8 to 176.4 tok/s (ladder
median). (2) runs the elementwise chain 2.05-3.28x faster in isolation; we have no separate model-level A/B
for it.

## Setup

- RTX 5070 Ti 16 GB (sm_120), WSL2; torch 2.11.0+cu130, triton 3.6.0; CUDA 13.4 for the gate library (`sm_120`)
- `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a` (hidden 2048; 10 full-attention layers with a 4096-wide gated
  output; FP8 `o_proj` with a static input scale)
- FreeToken 0.1.3 `cac247a8` + our fork, CUDA-graph decode, batch 1

## Problem in upstream today

All paths are under `python/freetoken/`, at `main` `4d2e2e2`.

- MoE block: `shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))`, then
  `return (routed + shared)` (`models/qwen3_5_moe/moe.py:80-82`). That is three elementwise launches per layer
  per step, 120 over 40 layers.
- Attention output: `gated = attn_out.reshape(...) * torch.sigmoid(gate)`, then `self.o_proj.forward(gated)`
  (`models/qwen3_5_moe/attention.py:86-88`). The FP8 `o_proj` then quantises the bf16 product with
  `_static_quant` (`kernel/triton/fp8_pertensor_linear.py:249-268`, called at `:291`). That is three launches,
  and the bf16 intermediates make about 13 bytes of logical traffic per element.
- `enable_fp_fusion` appears nowhere on `main`, so any Triton rewrite of these epilogues would be contracted to FMA
  by default.

## What we changed

**D5, fused MoE-block epilogue** (`QWEN36_DECODE_SHARED_EPILOGUE=1`, default off; our fork's
`moe/shared_epilogue.py`, wired in `models/qwen3_5_moe/moe.py`; decode only):

```
sig  = bf16_rn(1 / (1 + expf(-f32(gate))))      # IEEE div.rn, libdevice expf, as torch.sigmoid
prod = bf16_rn(f32(shared) * f32(sig))
out  = bf16_rn(f32(routed) + f32(prod))
```

- The multiply and the add are single IEEE operations separated by a bf16 rounding, so they match torch
  provided they are not contracted. Hence `enable_fp_fusion=False` on the launch.
- The sigmoid is the only transcendental, and its input is a single bf16 value. So a process self-test compares
  all 65,536 bf16 inputs bitwise against `torch.sigmoid`, plus random full-epilogue cases including 1e-30/1e30
  scales and signed zeros and subnormals. It runs before the first fused call. Any mismatch, or a first call
  under graph capture, keeps the torch path for the life of the process.

**Fused attention output gate** (`QWEN36_FUSED_ATTN_GATE=1` + `QWEN36_FUSED_ATTN_GATE_LIBRARY`, default off;
`kernel/cuda_attention_gate.py`, `kernel/csrc/qwen36_attention_gate.cu`; source and test harness in
`kernels/attention-gate/` of our repository):

- One kernel computes sigmoid (bf16 round), multiply (bf16 round) and the quant exactly like the pinned Triton
  `_static_quant`: `div.full.f32`, FP32 multiply, clamp to ±448, E4M3 satfinite RN.
- Its FP8 output goes straight into the existing uniform-scale `torch._scaled_mm` branch, with no second quant.
  Logical traffic drops from 13 to 5 bytes per element.
- A guard admits only `TorchFp8TensorLinearKernel`, bf16 activations, a static scalar FP32 input scale, a uniform
  FP32 weight scale, an FP8 weight without bias, width 4096, aligned strides, and sm_120. Anything else returns
  `None` and the original `_combine` runs. The scale is validated once, before graph capture.

## Measured impact

- **D5, microbenchmark** (`next-update/next-candidate/results/microbench-w2.json`): 131.2 -> 61.3 us per step for
  40 layers, saving 70 us per step (about 1.1%). A second run (`microbench-w3.json`) gave 132.2 -> 53.5 us.
- **D5, model ladder step L9** (`results/ladder/summary.json`; 3 pairs, rotating order, 60 s warm-up, 128 output
  tokens, no profiler), median (mean), min-max, n = 3:

  | | before | after |
  |---|---|---|
  | Decode 16K, tok/s | 173.8 (172.1), 167.5-174.9 | 176.4 (175.4), 172.5-177.4 |
  | Decode 120K, tok/s | 119.5 (118.5), 116.6-119.6 | 120.5 (120.3), 119.4-120.9 |
  | Decode step 16K (Nsight) | 5.88 ms | 5.78 ms |

  The Nsight "norms, misc" bucket went from 0.31 to 0.24 ms per step. Nsight timings are diagnostics only.
- **Attention gate, microbenchmark** (`kernels/README.md`; 5 alternating pairs, CUDA graph), speed of the
  elementwise chain: 2.05x at M=1, 3.28x at M=2048, 2.24x at M=4096, 2.62x at M=8192. **Model level: not
  measured separately.** The gate has been in every released profile of ours since f1, so it was never an
  isolated ladder step.

## Correctness evidence

- D5: the self-test above runs at startup. A PTX check runs without a GPU and includes a negative control
  (`docs/RESULTS.md` §9). Teacher forcing at L9: 125/125 cases, max_abs 0 on logits, outputs byte-equal.
- Attention gate: 9 fixtures were byte-identical in all three stages (sigmoid, product, FP8). They cover every
  finite bf16 gate code plus ±Inf at four input scales (NaN codes replaced by 0), and random [M, 4096] inputs with
  M = 1/33/2048/4096/8192, including row stride 4112 and amplitude 100. The following FP8 GEMM gives
  byte-identical bf16 output at M=1 and M=33. Our release teacher gates (125/125, max_abs 0) all ran with it on.

## Limitations / not measured

- sm_120 only. The gate library is built for `sm_120` and loaded by ctypes from a sha256-pinned path; upstream
  would want it as a JIT module. Its bit-exactness was checked against the `_static_quant` of our `cac247a8`
  base; `main`'s `:249-256` has the same sequence, but we did not re-run against `main`.
- D5 covers decode only (prefill keeps torch). Measured on one model at batch 1. The attention-gate gain at model
  level is not measured.

## Proposal

We can send two independent PRs, both default off: D5 as a Triton kernel with its self-test, and the attention
gate as a JIT CUDA module, or a Triton port if you prefer, with the same fixtures. Open question: would you
take a lint or test that flags Triton elementwise kernels claiming torch parity but not compiled with
`enable_fp_fusion=False`?
