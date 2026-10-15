# Custom CUDA kernels

The released build loads four native libraries on top of FreeToken. All four are bit-exact
replacements: on the same inputs they produce the same bytes as the FreeToken path they replace
(Triton, CUTLASS or PyTorch), so model output does not change. Each directory holds the source
of the shipped build, its build script and the test harness that checked bit-exactness.

| Library (released name) | Directory | sha256 of the released `.so` | Source sha256 |
|---|---|---|---|
| `libqwen36_gdn.so` (ABI 2) | [`gdn/`](gdn/) | `a8a806eda3a90a65c45b3ab02d8c7beaacb507aa9c1f380a0f255fa99ba067a4` | `qwen36_gdn.cu` `f4100781d652a6885383e6fd025282b3696aec61a70f18e5226272fcaf8920be`, `fla/qwen36_gdn_fla.cuh` `6d23b7b04750e0ace48858fe2c50e219f3bc3ca78a176c751c4f26b7a454a2cf` |
| `libqwen_fp8_gemv.so` (ABI 2) | [`fp8-gemv/`](fp8-gemv/) | `685c78cf89e60d0205f0db3ac2040203dec27e510a1e543cb7a7d3097414093b` | `src/qwen_fp8_gemv.cu` `9844420833b0c33c5b5fb73280f63527597306adb49b979731ac1fe89ac708fc` |
| `libqwen_nvfp4_bitexact.so` (v4, fp32 globals) | [`nvfp4-moe/`](nvfp4-moe/) | `b7fea22750256c2a34d45ce512bae62027165fa211bb2581003af96055225e4b` | `src/qwen36_nvfp4_bitexact.cu` `fb4496e2cfaafde9b1608bbeb3d9904dd50c435438eaf2ff79acbe74c71ed291` |
| `libqwen_attention_gate.so` | [`attention-gate/`](attention-gate/) | `f654f87a0a3f621fa51324fb00482ef51084d26432b1ac12b7710c9a5877f55e` | `attention_gate_fp8.cu` `329524efd5416f0b5dc35e326bbbc016686f522675a0d1129da47c76bfed89e9` |

The `.so` sha256 values are the ones pinned in the release qualification receipt (`native_libraries`)
and in the FreeToken ctypes loaders. The source files in this directory are byte-identical to the
sources of those builds (checked against the build checksums).

**nvcc output is not byte-reproducible.** Each compilation embeds its own module id, so a rebuild
from the same source and flags gives a different `.so` sha256. The release tooling and the FreeToken
loaders accept only the pinned hashes. After a rebuild you must update the pin and rerun the
bit-exact suites below. Binaries, cubins, PTX/SASS dumps and profiler reports are not committed
(see [`.gitignore`](.gitignore)).

## Toolchain

- CUDA 13.4 (nvcc V13.4.59), host compiler GCC 15.3 (`g++-15`), Linux x86-64 (built under WSL2).
- Target RTX 5070 Ti 16 GB (SM 12.0): `sm_120a` for GDN, FP8 GEMV and NVFP4; `sm_120` for the
  attention gate.
- No `--use_fast_math` anywhere. GDN also builds with `--ftz=false --prec-div=true --prec-sqrt=true`.
- Every build script is CPU-only and reads `CUDA_HOME` (default `/usr/local/cuda`) and
  `CUDA_CCBIN` (default `g++-15`).
- The test harnesses need the GPU, the FreeToken tree of this repository on `PYTHONPATH`
  (`FREETOKEN_PYTHON`, e.g. `freetoken/python`), the pinned Torch/Triton from the runtime lock,
  and for real-weight cases the NVFP4 checkpoint (`QWEN36_MODEL_DIR`). Run one GPU job at a time
  with the inference service stopped. Timings are only valid on an otherwise idle GPU.

```bash
export CUDA_HOME=/path/to/cuda-13.4 CUDA_CCBIN=g++-15
export FREETOKEN_PYTHON=$PWD/freetoken/python PYTHONPATH=$FREETOKEN_PYTHON
export QWEN36_MODEL_DIR=/path/to/model          # NVFP4 checkpoint directory
```

---

## gdn/ — `libqwen36_gdn.so` (Gated DeltaNet, ABI 2)

**What it replaces in FreeToken.** It covers the 30 linear-attention (GDN) layers:

- `qwen_gdn_conv_qkv`: causal conv1d + SiLU read directly from the `qkvz` projection rows, with no
  transpose/contiguous copy and the L2-norm in the epilogue. Replaces the sgl_kernel `causal_conv1d`
  and its copies (-7 launches per layer).
- `qwen_gdn_gate_cumsum`: gate + chunk cumsum (-9 launches).
- `qwen_gdn_gated_norm`: gated RMSNorm, plus the FP8 activation quant for `out_proj`.
- `qwen_gdn_decode_fused` / `qwen_gdn_decode_cluster`: decode in one launch per layer, either per
  CTA or as an 8-CTA cluster.
- ABI 2 adds `qwen_gdn_fla_chunk_ho` and `qwen_gdn_fla_wu_chunk_ho`. These are the FLA
  `chunk_h` + `chunk_o` kernels fused into one, with `recompute_w_u` folded in for the `wu`
  variant. They are bit-exact to Triton 3.6. They ship inside the released library, but
  `QWEN36_CUDA_GDN_FLA` is **off** in the released profiles: the model-level A/B showed no gain
  (16K +2.6 %, 64K/120K no better than without them).
- `qwen_gdn_abi_version() == 2`. The `qwen_gdn_silu_*` and `qwen_gdn_gate_eval` entry points are
  test hooks for exhaustive SiLU and gate checks.

FreeToken side (in `freetoken/`): `python/freetoken/kernel/cuda_gdn.py` (ctypes loader; requires
ABI exactly 2 even with FLA off) and `python/freetoken/models/qwen3_5_moe/gdn.py`. Switches:
`QWEN36_CUDA_GDN=1`, `QWEN36_CUDA_GDN_LIBRARY=<.so>`, `QWEN36_CUDA_GDN_DECODE=cta|cluster`
(released: cluster), `QWEN36_CUDA_GDN_FLA=wu|ho` (released: unset). Every unsupported case falls
back to the original path before any state is touched: T ≤ 32 for FLA, unusual shapes, a BF16
snapshot without FP32, and batches with an empty sequence.

**Build.** `./build.sh` writes `build/libqwen36_gdn.so`, the cubin/PTX/SASS, `resource-usage.txt`
and `SHA256SUMS` (0 spills). For the negative controls:
`BUILD_OUT=build-neg1 EXTRA_NVCC=-DQWEN_FLA_NEGCTRL=1 ./build.sh` (same for 2 and 3).

**Verification.**

```bash
python test_gdn_bitexact.py --lib build/libqwen36_gdn.so --out gdn.json                       # 124 cases
python test_gdn_bitexact.py --lib build/libqwen36_gdn.so --out gdn-wu.json --fla              # fused FLA, wu
python test_gdn_bitexact.py --lib build/libqwen36_gdn.so --out gdn-ho.json --fla --fla-variant ho
python fla/test_fla_fused.py --lib build/libqwen36_gdn.so --out fla.json                      # kernel suite vs Triton
TRITON_PTXAS_BLACKWELL_PATH=<triton>/backends/nvidia/bin/ptxas-blackwell \
  python fla/test_fla_fused.py --lib build/libqwen36_gdn.so --out fla-ptxas129.json --quick   # reference built by ptxas 12.9
python fla/test_fla_fused.py --lib build-neg1/libqwen36_gdn.so --out neg1.json --quick         # must FAIL (rc 1)
nvcc -O3 -gencode arch=compute_120a,code=sm_120a fla/hmma_probe.cu -o hmma_probe && ./hmma_probe 30
python fla/sass_check.py build/qwen36_gdn.sass fla-sass.json                                  # CPU only
python tools/check_silu_sass.py <sgl_kernel causal_conv1d sm_120 SASS...> build/qwen36_gdn.sass silu.json
python bench_gdn.py --out bench.json ; python fla/bench_fla.py --out bench-fla.json             # speed, no profiler
```

Comparisons are on raw bits: −0.0 vs +0.0 and NaN payloads count, and the entire FP32 state pool
is compared, including untouched slots. Inputs are synthetic, stress distributions, and
real-checkpoint activations of layers 0, 1, 2, 13, 26 and 38.

Results for the released build (a8a806…):

- Existing 124-case suite with the ABI-2 library: 124/124. With `--fla` wu: 124/124 (60 fused
  calls, T 64..8192). With ho: 124/124. With wu and reference Triton built by ptxas 12.9: 94/94.
  The ABI-1 kernels had passed 124/124 in two independent runs before ABI 2 was added.
- FLA kernel suite against Triton `chunk_h`/`chunk_o`/`recompute_w_u`, both variants: 123/123
  with ptxas 13.4 (841M `o` + 676M state elements per variant) and 85/85 with ptxas 12.9.
- Negative controls (one recipe point broken per build): all 3 detected (10, 10 and 4 failing
  cases).
- HMMA probe (491,520 outputs per question): a transposed orientation, extra zero steps and
  in-instruction permutations give 0 mismatches. A different K-set grouping (the control) gives
  238,372 mismatches. So only the K-set and chain order matter.
- SASS check: every `MUFU.EX2` has the non-FTZ expansion; no local memory; no FTZ instructions.
- Model level: teacher-forcing qualification 125/125 with max_abs 0 against the reference.
  Prefill TTFT −8.6 % at 64K (3 pairs) and −6.2 % median at 120K. Decode neutral.
- FLA kernels alone at T=2048: 0.325 → 0.185 ms. DRAM traffic for kkt+wu+h+o at T=2048:
  261 → 88 MB (ncu). This had no measurable model-level effect, so FLA stays flag-off.

## fp8-gemv/ — `libqwen_fp8_gemv.so` (decode FP8 GEMV, QMMA, ABI 2)

**What it replaces in FreeToken.** At decode (M = 1..8), the 39 row-wise fused input projections
per step (shapes 12288×2048 and 9216×2048) go through the CUTLASS SM120 `torch._scaled_mm` kernel
with a 128³ tile. This library replaces that with a skinny GEMV. The GEMV uses the same
`QMMA.16832.F32.E4M3.E4M3` instruction, the same K order and the same epilogue, so the output is
bit-identical. Any other shape or M, including prefill, keeps the original path.

C ABI: `qwen_fp8_gemv_abi_version() == 2`, `qwen_fp8_gemv_source_sha256()` (the source hash is
compiled in), `qwen_fp8_gemv_init`, `qwen_fp8_gemv_rowwise`, `qwen_fp8_gemv_max_m() == 8`,
`qwen_fp8_gemv_error`, `qwen_fp8_gemv_smem_bytes`, `qwen_fp8_gemv_variant_count/info`. The
analysis-only entry points (`splitk`, `gemm_partial_f32`, `read_bw`) are compiled out by
`-DQWEN_FP8_GEMV_PRODUCTION`.

FreeToken side: `python/freetoken/kernel/cuda_fp8_gemv.py` loads the library from a sealed memfd
after checking the pinned sha256 (`LIBRARY_SHA256`), then runs the ABI probe and a bit-exact
self-test when the layers are finalized (never under CUDA graph capture). Any failure only
disables the GEMV, and CUTLASS stays authoritative. It is called from `layers/quantization/linear/fp8_tensor.py` and
`kernel/triton/fp8_pertensor_linear.py`; the in-tree test is `tests/kernels/test_cuda_fp8_gemv.py`.
Switches: `QWEN36_FP8_GEMV=1`, `QWEN36_FP8_GEMV_LIBRARY=<.so>`.

**Build.** `./build.sh` writes `build/libqwen_fp8_gemv.so`, the SASS, `build-info.txt`,
`exports.txt` and `sha256.txt`. It refuses to overwrite an existing pinned build without
`FORCE_REBUILD=1`, and refuses CRLF sources. Set `PRIOR_SASS=<sass of an earlier build>` to
compare the row-wise kernels instruction by instruction. The release build matched the analysis
build exactly.

**Verification.** The harness in `scripts/` compares the integrated module path, not just the
kernel. It needs `FREETOKEN_PYTHON` (tree with the integration) and `FREETOKEN_PRISTINE_PYTHON`
(the same tree without it, which supplies the baseline `fp8_pertensor_linear.py`).

```bash
python -I -S scripts/abi_probe.py build/libqwen_fp8_gemv.so            # CPU only, CUDA_VISIBLE_DEVICES=''
python scripts/step3_bitwise.py --output bitwise.json                     # 80 real projections x M 1..8, eager + graph
for c in env_unset platform_pin provenance_tampered provenance_missing_path provenance_stale_v1 \
         selftest_launch_error selftest_mismatch_m3_m5 runtime_launch_error real_weight_mismatch_m2 \
         capture_guard threads_first_decision; do python scripts/fallback_case.py $c; done
python scripts/step4_speed.py --output speed.json                         # 5 alternating pairs, no profiler
```

Results for the released build (685c78cf…):

- Author: 0 mismatches over 6,400 cases. That is 117.2M elements eager and 78.2M under CUDA graph,
  real weights, all 80 dense FP8 projections, M 1..8, including normal, heavy-tailed and edge
  activations. All 11 fallback scenarios pass. In-tree test 22/22.
- Independent review (separate harness): 195.4M elements × 3 comparisons eager and 97.7M in graph
  capture, 0 mismatches. Also 0 mismatches for extreme synthetic scales, non-blocking side
  streams, two concurrent streams, and non-contiguous inputs. Prefill-sized M is never served.
- Speed: the 80-projection sequence under graph runs 1.44–1.48× faster, about 1 ms per decode
  step. ncu: GEMV 34.4 µs at 92.9 % DRAM vs CUTLASS 52.7 µs at 59.5 %. Model level, together with
  the decode gather: decode +13 to +17 % at 16K/64K/120K, and teacher-forcing 125/125 with max_abs 0.

## nvfp4-moe/ — `libqwen_nvfp4_bitexact.so` (NVFP4 MoE prefill, v4)

**What it replaces in FreeToken.** For prefill chunks with M ≥ 512 it replaces the whole Triton
NVFP4 fused-MoE prefill: gate/up GEMM, FlashInfer `silu_and_mul`, down GEMM with router weights,
and `moe_sum_reduce`. The kernel uses inline-PTX `mma.sync m16n8k16` + `ldmatrix` + `cp.async`
with E2M1/E4M3 dequant. It reproduces the per-HMMA K placement of Triton 3.6: K residues
{0,2}/{4,6}/{1,3}/{5,7} packed into each MMA and chained even/kRep0 → even/kRep1 → odd/kRep0 →
odd/kRep1 per K64. It also replicates the FlashInfer SiLU SASS and the sum order. v4 reads the
NVFP4 global scales as fp32; the fp16 v3 is not shipped.

C ABI: `qwen_nvfp4_prefill`, `qwen_nvfp4_cuda_error`, `qwen_nvfp4_global_bits() == 32` (v4) and
`qwen_nvfp4_build_id`. Test hooks: `qwen_nvfp4_oracle`, `_dequant_fixture`, `_silu_and_mul`,
`_moe_sum`, `_prefill_variant`, `_variant_count/bm`, `_mma_peak`.

FreeToken side: `python/freetoken/moe/cuda_nvfp4.py` (ABI probe in a subprocess with
`CUDA_VISIBLE_DEVICES=''`; requires fp32 banks). Switches: `QWEN36_NVFP4_PREFILL=cuda`,
`QWEN36_CUDA_NVFP4_LIBRARY=<.so>`.

**Build.** `./build.sh [OUT_DIR]` writes `libqwen_nvfp4_bitexact.so`, the cubin/PTX/SASS,
`regs.txt` and `build-sha256.txt`.

**Verification** (`py/`; `py/bitexact_lib.py` is the ctypes wrapper used by the tests):

```bash
cd py
python test_bitexact.py --output synthetic.json     # suites: dequant/silu exhaustive, sum, triton_self, synthetic, negative, high_slot
python test_real.py --output real.json              # real checkpoint, layers 0/13/26/39, real router
python bench.py --output bench.json                 # 5 paired layer timings, no profiler
```

The negative controls use the library's own oracle modes with the old K layouts, and these must
mismatch. Setting `QWEN36_NVFP4_PREVIOUS_LIBRARY` also compares against the earlier "matched"
variant. `test_real.py` takes a GPU lock file (`QWEN36_GPU_LOCK`) and refuses to run while a
`qwen36-*` service is active.

Results:

- v4 (released): synthetic 657 comparisons / 6.89e9 elements; real weights (layers 0/13/26/39,
  M ∈ {1,17,256,2048,4096}) 860 comparisons / 9.78e9 elements. **0 mismatches.**
- v3 kernel (same arithmetic, fp16 globals): real weights 1,376 + 608 comparisons, about 1.4e10
  elements; synthetic 2,218 comparisons / 1.5e10; dequant and SiLU exhaustive. 0 mismatches. The
  negative control with the old layouts reproduced mismatches at about 1e-4.
- Model level: teacher-forcing 125/125 bit-identical (max_abs 0) against the fp32-global reference.
- Layer speed at M=2048: 12.82 → 1.77 ms (7.25×). Model prefill elapsed is 1.7–1.9× faster at
  16K–120K; host-to-device expert streaming is unaffected.

## attention-gate/ — `libqwen_attention_gate.so` (fused full-attention output gate)

**What it replaces in FreeToken.** In the 10 full-attention layers, three launches run before
`o_proj`: sigmoid, BF16 multiply, and the Triton static FP8 activation quant (`_static_quant`).
This library does all three in one kernel. It keeps both intermediate BF16 roundings (after the
sigmoid and after the multiply). The quant matches the pinned Triton exactly: `div.full.f32`, FP32
multiply, clamp ±448, E4M3 satfinite RN. The FP8 tensor goes straight into the existing
uniform-scale `torch._scaled_mm` branch; no second quant runs. Logical traffic drops from 13 to
5 bytes per element.

C ABI: `int qwen_attention_gate_fp8(attention, gate, scale, out_fp8, dbg_sigmoid, dbg_product,
M, N, stride_attention, stride_gate, stream)` and `const char* qwen_gate_error(int)`.

FreeToken side: `python/freetoken/kernel/cuda_attention_gate.py` and
`python/freetoken/kernel/csrc/qwen36_attention_gate.cu` (a byte-identical copy of
`attention_gate_fp8.cu`), called from `Qwen3_5Attention._combine`. The guard admits only
`TorchFp8TensorLinearKernel`, BF16 activations, a static scalar FP32 input scale, uniform FP32
weight scale, FP8 weight without bias, and the correct shapes, strides and alignment on SM120.
Switches: `QWEN36_FUSED_ATTN_GATE=1`, `QWEN36_FUSED_ATTN_GATE_LIBRARY=<.so>` (profile key
`fused_attention_gate`).

**Build.** `./build.sh` writes `build/libqwen_attention_gate.so` (sm_120; 23/28 registers, 0
spills, 0 barriers). Keep the source file name: it is part of the anonymous-namespace symbols.

**Verification.** Run `python test_bench.py --output gate.json` (`--numeric-only` skips the
timing). `gate_fp8.py` is the standalone ctypes wrapper. The reference is FreeToken's pinned
`_static_quant`.

- 9 fixtures were byte-identical on all three stages (sigmoid, product, FP8). They cover every
  finite BF16 gate code plus ±Inf at four input scales (NaN codes replaced by 0), and random
  [M,4096] with M = 1/33/2048/4096/8192, including row stride 4112 and amplitude 100. The
  following FP8 GEMM gives byte-identical BF16 output at M=1 and M=33.
- Elementwise-chain speed (5 alternating pairs, graph): 2.05× at M=1, 3.28× at M=2048, 2.24× at
  M=4096, 2.62× at M=8192. Shipped in every released profile except `baseline64`.

---

## Decode gather kernel (shipped, lives in the FreeToken tree)

The decode expert gather (`decode_gather_tma`) is a JIT/tvm-ffi module inside FreeToken, not a
separate library: `freetoken/python/freetoken/kernel/csrc/jit/decode_gather.cuh` and
`kernel/decode_gather.py`. Production loads only the manifest-bound prebuilt
(`FREETOKEN_DISABLE_JIT=1`). Its bulk copies use the `.shared::cta` destination. The
`.shared::cluster` form lowers on sm_120 to a syscall that raises the context stack limit and
permanently costs about 1373 MiB of VRAM on first launch. Served sm_120 kernels must not use
`.shared::cluster` bulk copies.

## Next-update candidates (in this tree, not in the released build)

Two bit-exact decode kernels (plan item D4). Each sits behind its own switch, off by default. Each
library is pinned by sha256 and ABI, and is self-tested against the path it replaces before use. Each
directory has its own README with the identification method, build, tests and results.

| Library | Directory | Replaces | Switch |
|---|---|---|---|
| `libqwen_oproj_gemv.so` (ABI 1) | [`decode-oproj/`](decode-oproj/) | decode out/o-projection `torch._scaled_mm` (cuBLASLt memset + xmma split-K 3) | `QWEN36_OPROJ_GEMV` |
| `libqwen_moe_decode.so` (ABI 1) | [`decode-moe/`](decode-moe/) | Triton NVFP4 MoE decode GEMV (gate_up and down) | `QWEN36_MOE_DECODE_CUDA` |

## Experimental, not shipped

These are kept in the private work archive only; none of them is loaded by the released build.
Paths are relative to the archive root.

| Work | Archive path | Why not shipped |
|---|---|---|
| `qfa_prefill` FlashInfer-attention prefill replica (v1..v7si) and its probes | `work/cuda-attention/src/qfa_prefill*.cu`, `work/cuda-attention/qfa-prefill-integration.patch` | Research replica. Also uses `.shared::cluster` bulk copies (the VRAM trap above). |
| Decode expert prefetch prototype | `work/cuda-decode-prefetch/src/` (`decode_prefetch_bench.cu`, `dp_common.cuh`, `dp_kernels.cuh`), `work/cuda-decode-prefetch/patch-src/qwen36_decode_prefetch.cu` | Prototype, `.shared::cluster` bulk copies. Redesign planned for the next update. |
| PCIe transfer benchmark and rANS GPU decoder | `work/pcie-transfer/pcie_bench.cu`, `work/pcie-transfer/rans_gpu.cu`, `rans_lib.cpp` | Measurement tools only. |
| FP8 GEMV v1 analysis library and `mma_forms.cu` | `work/cuda-fp8-gemv/` | Superseded by `fp8-gemv/` (v2). The v1 build is used only as the "stale ABI" fallback case. |
| NVFP4 MoE WMMA and "matched" variants | `work/cuda-nvfp4/` | About 99.9 % bit-equal; failed the 16K functional gate. Superseded by `nvfp4-moe/`. |
| GDN conv layout copy, KV, transport, MoE-decode, prefill-transport drafts | `work/cuda-gdn-layout/`, `work/cuda-kv/`, `work/cuda-transport/`, `work/cuda-moe-decode/`, `work/cuda-prefill-transport/` | Stopped or unverified drafts. |
| Next-update work (gather prefetch, FLA follow-ups) | `work/next-update/` | Not part of this release. |
