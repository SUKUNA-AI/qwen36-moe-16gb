// qwen36_gdn.cu -- bit-exact CUDA C++/PTX replacements for the data-movement and elementwise
// parts of the Qwen3.6-35B-A3B Gated DeltaNet (GDN) layer on SM120 (RTX 5070 Ti).
//
// Every kernel here reproduces, instruction for instruction, the floating-point dataflow of the
// path it replaces (the sgl_kernel 0.4.5 causal-conv1d SASS, the Triton 3.6 FLA kernels' SASS
// compiled by ptxas 12.9, and the PyTorch 2.11+cu130 elementwise kernels). Where the original
// compiler contracted, rounded or reassociated, this file does the same thing explicitly with
// inline PTX (explicit .rn on every add/mul/fma, so ptxas cannot contract differently).
// The reference SASS lines are quoted next to each recipe; see REPORT.md for the evidence.
//
// Prefill (one launch each, replacing 13+ launches of the original per GDN layer):
//   qwen_gdn_conv_qkv      conv_in strided rows (qkvz GEMM output) -> depthwise causal conv(4)
//                          + SiLU (sgl_kernel exact) -> BF16 -> Triton-exact L2 norm for q,k ->
//                          contiguous q[T,16,128], k[T,16,128], v[T,32,128]; conv state update.
//                          Replaces: transpose().contiguous(), cu_seqlens int32 cast,
//                          causal_conv1d_fwd, 3x input_guard .contiguous(), 2x l2norm_fwd.
//   qwen_gdn_gate_cumsum   a,b (ba GEMM output) -> g = -exp(A_log)*softplus(a+dt_bias) (ATen
//                          exact) -> chunk-local cumsum (Triton tl.cumsum exact) and
//                          beta = float(bf16(sigmoid(b))) (ATen exact).
//                          Replaces: sigmoid, exp, neg, float copy, add, softplus, mul, beta
//                          float copy, chunk_local_cumsum (9 launches).
//   qwen_gdn_gated_norm    o[T,32,128] and z read in place from the qkvz buffer (row stride)
//                          -> GatedRMSNorm (Triton _layer_norm_fwd_1pass exact, both reduction
//                          layouts) -> BF16 -> optional FP8-E4M3 static quantization
//                          (FreeToken _static_quant_kernel exact) for out_proj.
//                          Replaces: z reshape .contiguous(), rms_norm_gated, _static_quant.
// Decode (one launch per GDN layer, replacing conv update + recurrent + gated norm + quant):
//   qwen_gdn_decode_fused  see the kernel comment.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

// ------------------------------------------------------------------------------------------
// Pinned PTX arithmetic (explicit rounding => no contraction / reassociation by ptxas).
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float d; asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float d; asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float sub_rn(float a, float b) {
  float d; asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float d; asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float div_rn(float a, float b) {
  float d; asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float sqrt_approx_ftz(float a) {
  float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a)); return d;
}
__device__ __forceinline__ float ex2_approx(float a) {  // non-ftz (Triton tl.exp / sigmoid)
  float d; asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a)); return d;
}
__device__ __forceinline__ float max_f(float a, float b) {
  float d; asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float min_f(float a, float b) {
  float d; asm("min.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bf_at(const uint4& v, int i) {
  const uint32_t w = (i >> 1) == 0 ? v.x : (i >> 1) == 1 ? v.y : (i >> 1) == 2 ? v.z : v.w;
  return (i & 1) ? bf_hi(w) : bf_lo(w);
}
__device__ __forceinline__ float bf16_to_f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint32_t pack_bf16_rn(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ uint16_t bf16_rn(float x) {
  uint16_t r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return r;
}
__device__ __forceinline__ uint16_t e4m3x2_rn_satfinite(float lo, float hi) {
  uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ uint16_t half_at(const uint32_t* w, int e) {
  return (e & 1) ? static_cast<uint16_t>(w[e >> 1] >> 16) : static_cast<uint16_t>(w[e >> 1] & 0xffffu);
}
__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint4 ld16_coherent(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st16(void* p, const uint4& v) { *reinterpret_cast<uint4*>(p) = v; }

// ------------------------------------------------------------------------------------------
// SiLU exactly as compiled in sgl_kernel causal_conv1d_{fwd,update} (sm_120 cubin):
//   FFMA.SAT t, x, -0x3bbb989d, 0.5 ; FFMA.RM j, t, 252, 12582913 ; FADD f, j, -12583039 ;
//   SHL s, j, 23 ; FFMA r, x, -1.4426950216, -f ; FFMA r, x, -1.925963e-08, r ; MUFU.EX2 e, r ;
//   FFMA d, s, e, 1 ; IEEE division x / d (MUFU.RCP + Newton + FCHK + slow path).
// i.e. x / (1 + expf(-x)) with precise expf and div.rn, no FTZ.
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ float silu_sgl(float x) {
  float t, j, f, r, e, d, o;
  uint32_t sbits;
  asm("fma.rn.sat.f32 %0, %1, 0fBBBB989D, 0f3F000000;" : "=f"(t) : "f"(x));
  asm("fma.rm.f32 %0, %1, 0f437C0000, 0f4B400001;" : "=f"(j) : "f"(t));
  asm("add.rn.f32 %0, %1, 0fCB40007F;" : "=f"(f) : "f"(j));
  asm("shl.b32 %0, %1, 23;" : "=r"(sbits) : "r"(__float_as_uint(j)));
  asm("{\n\t.reg .f32 nf;\n\tneg.f32 nf, %2;\n\tfma.rn.f32 %0, %1, 0fBFB8AA3B, nf;\n\t}"
      : "=f"(r) : "f"(x), "f"(f));
  asm("fma.rn.f32 %0, %1, 0fB2A57060, %2;" : "=f"(e) : "f"(x), "f"(r));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(e));
  asm("fma.rn.f32 %0, %1, %2, 0f3F800000;" : "=f"(d) : "f"(__uint_as_float(sbits)), "f"(r));
  asm("div.rn.f32 %0, %1, %2;" : "=f"(o) : "f"(x), "f"(d));
  return o;
}

// Depthwise causal conv tap order of sgl_kernel: out = 0.f; out += w[k]*x[t-3+k], k=0..3 (FFMA).
__device__ __forceinline__ float conv4(float x0, float x1, float x2, float x3,
                                       float w0, float w1, float w2, float w3) {
  float acc = fma_rn(x0, w0, 0.0f);
  acc = fma_rn(x1, w1, acc);
  acc = fma_rn(x2, w2, acc);
  return fma_rn(x3, w3, acc);
}

// ------------------------------------------------------------------------------------------
// Triton l2norm_fwd_kernel (D=128, BT=16, BD=128, num_warps=8; layout sizePerThread [1,8],
// threadsPerWarp [2,16]) per row, as compiled (SASS):
//   s = FMUL x1,x1 ; s = FFMA x0,x0,s ; s = FFMA x2..x7 (in order) ;
//   SHFL.BFLY 8,4,2,1 with FADD own+other ; v = FADD s, eps ; r = MUFU.SQRT v (sqrt.approx.ftz)
//   y_i = div.full(x_i, r) ; F2FP.BF16 (RN).
// Called by the 16 lanes (a half-warp) that hold one 128-element row, 8 contiguous each.
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ void l2norm_row16(float (&x)[8], float eps) {
  float s = mul_rn(x[1], x[1]);
  s = fma_rn(x[0], x[0], s);
#pragma unroll
  for (int i = 2; i < 8; ++i) s = fma_rn(x[i], x[i], s);
#pragma unroll
  for (int off = 8; off >= 1; off >>= 1) s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  const float r = sqrt_approx_ftz(add_rn(s, eps));
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = div_full(x[i], r);
}

// ==========================================================================================
// Kernel 1: prefill conv + SiLU + (q,k) L2 norm, strided input, contiguous outputs.
// ==========================================================================================
constexpr int kConvThreads = 128;
constexpr int kVec = 8;
constexpr int kGroup = kConvThreads * kVec;  // 1024 channels per block

struct ConvParams {
  const __nv_bfloat16* x;           // [T, ldx], conv channels in columns [0, C)
  const __nv_bfloat16* weight;      // [C, 4] contiguous
  __nv_bfloat16* conv_states;       // [S, C, 3] (c stride 3, l stride 1)
  const int64_t* cu_seqlens;        // [B + 1]
  const int32_t* cache_indices;     // [B]
  const bool* has_initial_state;    // [B]
  __nv_bfloat16* out_q;             // [T, key_dim]   (L2-normalised)
  __nv_bfloat16* out_k;             // [T, key_dim]   (L2-normalised)
  __nv_bfloat16* out_v;             // [T, value_dim]
  int64_t ldx, state_batch_stride;
  int32_t B, C, key_dim, value_dim, pad_slot_id, norm_qk;
  float l2_eps;
};

struct TileInfo { int64_t bos, tb, te, seqlen; int32_t seq; };

// Map blockIdx.y to (sequence, token range); tiles never straddle sequences. seq=-1 for
// surplus blocks (grid.y is a host-side upper bound computed without a device sync).
template <int TT>
__device__ __forceinline__ TileInfo find_tile(const int64_t* cu, int B) {
  __shared__ TileInfo s_info;
  if (threadIdx.x == 0) {
    TileInfo info; info.seq = -1; info.bos = info.tb = info.te = info.seqlen = 0;
    int64_t acc = 0;
    const int64_t tile = blockIdx.y;
    int64_t b = cu[0];
    for (int i = 0; i < B; ++i) {
      const int64_t e = cu[i + 1];
      const int64_t nt = (e - b + TT - 1) / TT;
      if (tile < acc + nt) {
        info.seq = i; info.bos = b; info.seqlen = e - b;
        info.tb = b + (tile - acc) * TT;
        info.te = min(e, info.tb + TT);
        break;
      }
      acc += nt; b = e;
    }
    s_info = info;
  }
  __syncthreads();
  return s_info;
}

template <int TT, int BR>
__global__ __launch_bounds__(kConvThreads) void gdn_conv_qkv_kernel(const ConvParams p) {
  static_assert(BR >= 3, "window carry needs at least 3 rows per batch");
  const TileInfo ti = find_tile<TT>(p.cu_seqlens, p.B);
  if (ti.seq < 0) return;
  const int64_t tb = ti.tb, te = ti.te;
  const int c0 = (int)blockIdx.x * kGroup + threadIdx.x * kVec;  // first channel of this thread
  // c0 < C always (C == 8 * 1024 checked on the host); the whole block takes the same branch.
  __nv_bfloat16* out;
  int64_t ldo;
  bool l2 = false;
  if (c0 < p.key_dim) { out = p.out_q + c0; ldo = p.key_dim; l2 = p.norm_qk; }
  else if (c0 < 2 * p.key_dim) { out = p.out_k + (c0 - p.key_dim); ldo = p.key_dim; l2 = p.norm_qk; }
  else { out = p.out_v + (c0 - 2 * p.key_dim); ldo = p.value_dim; }
  const __nv_bfloat16* xc = p.x + c0;

  const int32_t slot = p.cache_indices[ti.seq];
  if (slot == p.pad_slot_id) {
    // Original: the whole conv block returns, its in-place buffer keeps the raw input, which
    // the original path then L2-normalises (q, k) / uses as is (v). Conv state untouched.
    for (int64_t t = tb; t < te; ++t) {
      uint4 r = ldg16(xc + t * p.ldx);
      if (l2) {
        float y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = bf_at(r, i);
        l2norm_row16(y, p.l2_eps);
        r.x = pack_bf16_rn(y[0], y[1]); r.y = pack_bf16_rn(y[2], y[3]);
        r.z = pack_bf16_rn(y[4], y[5]); r.w = pack_bf16_rn(y[6], y[7]);
      }
      st16(out + t * ldo, r);
    }
    return;
  }
  const bool has_init = p.has_initial_state[ti.seq];
  const bool first_tile = (tb == ti.bos);

  float w[4][kVec];
  {
    const __nv_bfloat16* wp = p.weight + (int64_t)c0 * 4;
    uint32_t raw[16];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 v = ldg16(wp + q * 8);
      raw[q * 4 + 0] = v.x; raw[q * 4 + 1] = v.y; raw[q * 4 + 2] = v.z; raw[q * 4 + 3] = v.w;
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int e = i * 4 + k;
        w[k][i] = (e & 1) ? bf_hi(raw[e >> 1]) : bf_lo(raw[e >> 1]);
      }
  }

  // Old conv state (first tile only): 8 channels x 3 = 24 BF16, layout [channel][pos].
  uint32_t st[12];
#pragma unroll
  for (int q = 0; q < 12; ++q) st[q] = 0u;
  __nv_bfloat16* sp = p.conv_states + (int64_t)slot * p.state_batch_stride + (int64_t)c0 * 3;
  if (first_tile && has_init) {
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      const uint4 v = ld16_coherent(sp + q * 8);
      st[q * 4 + 0] = v.x; st[q * 4 + 1] = v.y; st[q * 4 + 2] = v.z; st[q * 4 + 3] = v.w;
    }
  }
  // Window of the three previous rows (packed BF16, oldest first).
  uint4 prev[3];
  if (first_tile) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      uint32_t r[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        r[q] = (uint32_t)half_at(st, (2 * q) * 3 + j) | ((uint32_t)half_at(st, (2 * q + 1) * 3 + j) << 16);
      prev[j] = make_uint4(r[0], r[1], r[2], r[3]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 3; ++j) prev[j] = ldg16(xc + (tb - 3 + j) * p.ldx);
  }

  for (int64_t t0 = tb; t0 < te; t0 += BR) {
    uint4 rows[BR];
#pragma unroll
    for (int u = 0; u < BR; ++u)
      rows[u] = (t0 + u < te) ? ldg16(xc + (t0 + u) * p.ldx) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < BR; ++u) {
      if (t0 + u < te) {  // uniform across the block (same token for all threads)
        const uint4 r0 = u >= 3 ? rows[u >= 3 ? u - 3 : 0] : prev[u < 3 ? u : 0];
        const uint4 r1 = u >= 2 ? rows[u >= 2 ? u - 2 : 0] : prev[u < 2 ? u + 1 : 0];
        const uint4 r2 = u >= 1 ? rows[u >= 1 ? u - 1 : 0] : prev[2];
        const uint4 r3 = rows[u];
        float o[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          o[i] = silu_sgl(conv4(bf_at(r0, i), bf_at(r1, i), bf_at(r2, i), bf_at(r3, i),
                                w[0][i], w[1][i], w[2][i], w[3][i]));
        uint4 ov;
        ov.x = pack_bf16_rn(o[0], o[1]); ov.y = pack_bf16_rn(o[2], o[3]);
        ov.z = pack_bf16_rn(o[4], o[5]); ov.w = pack_bf16_rn(o[6], o[7]);
        if (l2) {
          // l2norm_fwd reads the BF16 conv output (q/k .contiguous() copies) -> round first.
          float y[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) y[i] = bf_at(ov, i);
          l2norm_row16(y, p.l2_eps);
          ov.x = pack_bf16_rn(y[0], y[1]); ov.y = pack_bf16_rn(y[2], y[3]);
          ov.z = pack_bf16_rn(y[4], y[5]); ov.w = pack_bf16_rn(y[6], y[7]);
        }
        st16(out + (t0 + u) * ldo, ov);
      }
    }
    prev[0] = rows[BR - 3]; prev[1] = rows[BR - 2]; prev[2] = rows[BR - 1];
  }

  // Conv state update by the block that read the old state of this sequence/channel group
  // (same thread, after the read): last 3 entries of [old_state | x] (zeros if no state).
  if (first_tile) {
    const int64_t L = ti.seqlen;
    uint32_t ns[12];
    uint16_t h[3][kVec];
#pragma unroll
    for (int wpos = 0; wpos < 3; ++wpos) {
      const int64_t rel = L - 3 + wpos;
      if (rel >= 0) {
        const uint4 v = ldg16(xc + (ti.bos + rel) * p.ldx);
        const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < kVec; ++i) h[wpos][i] = half_at(vw, i);
      } else {
        const int sl = (int)(rel + 3);  // L < 3 only: old-state position (zero without state)
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          h[wpos][i] = sl == 0 ? half_at(st, i * 3 + 0) : sl == 1 ? half_at(st, i * 3 + 1)
                                                                 : half_at(st, i * 3 + 2);
      }
    }
#pragma unroll
    for (int q = 0; q < 12; ++q) {
      const int e0 = 2 * q, e1 = 2 * q + 1;
      ns[q] = (uint32_t)h[e0 % 3][e0 / 3] | ((uint32_t)h[e1 % 3][e1 / 3] << 16);
    }
#pragma unroll
    for (int q = 0; q < 3; ++q)
      st16(sp + q * 8, make_uint4(ns[q * 4 + 0], ns[q * 4 + 1], ns[q * 4 + 2], ns[q * 4 + 3]));
  }
}

// ==========================================================================================
// Kernel 2: gate (ATen-exact) + chunk-local cumsum (Triton-exact) + beta.
//   g_raw = (-expf(A_log[h])) * softplus(float(a) + dt_bias[h])   (ATen fp32 kernels)
//   softplus(x) = x > 20 ? x : log1pf(expf(x))                    (beta=1, threshold=20)
//   g = inclusive cumsum of g_raw over each 64-token chunk of each sequence:
//       Triton chunk_local_cumsum_scalar_kernel (BT=64, 8 warps, 1 elem/thread): Hillis-Steele
//       shfl.up 1,2,4,8,16 (own+up) inside each 32-lane warp, then elements 32..63 get
//       FADD(total(0..31), own).
//   beta = float(bf16_rn(1 / (1 + expf(-float(b)))))              (ATen sigmoid, div.rn)
// One block = 4 heads x 64 tokens (256 threads).
// ==========================================================================================
struct GateParams {
  const __nv_bfloat16* ba;   // [T, ldba]: b in columns [0, H), a in columns [H, 2H)
  const float* A_log;        // [H]
  const float* dt_bias;      // [H]
  const int64_t* cu_seqlens; // [B + 1]
  float* g_out;              // [T, H] fp32 (chunk-local cumsum)
  float* beta_out;           // [T, H] fp32
  int64_t ldba;
  int32_t B, H;
};

__device__ __forceinline__ float softplus_aten(float x) {
  // ATen softplus_kernel: (aop * beta) > threshold ? aop : ::log1p(std::exp(aop * beta)) / beta
  // with runtime beta=1.f, threshold=20.f. The multiply/divide by beta are kept as real FMUL /
  // IEEE division (inline PTX, not folded) so even NaN inputs follow the original instruction
  // path (canonical NaN); for every finite or infinite input they are exact identities.
  const float xb = mul_rn(x, 1.0f);
  return (xb > 20.0f) ? x : div_rn(log1pf(expf(xb)), 1.0f);
}
__device__ __forceinline__ float sigmoid_aten(float x) {
  return div_rn(1.0f, add_rn(1.0f, expf(-x)));
}

__global__ __launch_bounds__(256) void gdn_gate_cumsum_kernel(const GateParams p) {
  // Map blockIdx.x -> (sequence, chunk) like prepare_chunk_indices (chunk size 64).
  __shared__ int64_t s_bos, s_eos, s_c;
  __shared__ float s_total[4];
  if (threadIdx.x == 0) {
    int64_t acc = 0, b = p.cu_seqlens[0];
    s_bos = -1;
    for (int i = 0; i < p.B; ++i) {
      const int64_t e = p.cu_seqlens[i + 1];
      const int64_t nt = (e - b + 63) / 64;
      if ((int64_t)blockIdx.x < acc + nt) { s_bos = b; s_eos = e; s_c = blockIdx.x - acc; break; }
      acc += nt; b = e;
    }
  }
  __syncthreads();
  if (s_bos < 0) return;
  const int hl = threadIdx.x >> 6;            // head within block
  const int h = blockIdx.y * 4 + hl;
  const int i = threadIdx.x & 63;             // token within chunk
  const int lane = threadIdx.x & 31;
  const int64_t t = s_bos + s_c * 64 + i;
  const bool valid = (t < s_eos) && (h < p.H);
  float graw = 0.0f;  // masked lanes load 0 (Triton: mov 0 then predicated load)
  if (valid) {
    const uint16_t* row = reinterpret_cast<const uint16_t*>(p.ba) + t * p.ldba;
    const float a = bf16_to_f(row[p.H + h]);
    const float b = bf16_to_f(row[h]);
    const float x = add_rn(a, __ldg(p.dt_bias + h));
    const float sp = softplus_aten(x);
    const float ne = -expf(__ldg(p.A_log + h));
    graw = mul_rn(ne, sp);
    const float beta = bf16_to_f(bf16_rn(sigmoid_aten(b)));
    p.beta_out[t * p.H + h] = beta;
  }
  // Hillis-Steele inclusive scan inside the warp (lane 0 keeps its value).
  float v = graw;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const float up = __shfl_up_sync(0xffffffffu, v, d);
    const float s = add_rn(v, up);
    v = (lane >= d) ? s : v;
  }
  const int half = i >> 5;
  if (lane == 31 && half == 0) s_total[hl] = v;
  __syncthreads();
  if (half == 1) v = add_rn(s_total[hl], v);
  if (valid) p.g_out[t * p.H + h] = v;
}

// ==========================================================================================
// Kernel 3: GatedRMSNorm (Triton _layer_norm_fwd_1pass_kernel, is_rms_norm, norm_before_gate,
// silu gate, N=128, 1 warp) + optional FP8 static quant (FreeToken _static_quant_kernel).
// Two reduction layouts, selected exactly as the Python wrapper selects ROWS_PER_BLOCK:
//   RPB = next_pow2(cdiv(M, 2*SMs)) clipped to 4; RPB==1 -> 32 lanes x 4 elements,
//   RPB>=2 -> 16 lanes x 8 elements (sizePerThread [1,8], threadsPerWarp [2,16]).
// Per element (SASS of the Triton cubin, ptxas 12.9):
//   ss = FMUL x1,x1 ; FFMA x0.. ; FFMA x2..x(n-1) ; SHFL.BFLY (16,)8,4,2,1 FADD
//   ve = FFMA ss, 0.0078125, eps          (div.full by 128 folded into the eps add)
//   rstd = MUFU.RSQ ve                    (rsqrt.approx.ftz)
//   y = (rstd*x)*w ; nz = 0 - z ; e = ex2.approx(nz*log2e) ; sg = div.full(1, e+1)
//   out = y * (sg*z) -> BF16 RN
// FP8: inv = div.full(1, scale); q = min(max(inv*x_bf16, -448), 448) -> e4m3 RN satfinite.
// ==========================================================================================
struct NormParams {
  const __nv_bfloat16* o;   // [M = T*heads, 128] contiguous
  const __nv_bfloat16* z;   // z[t, h, :] at z + t*ldz + h*128
  const __nv_bfloat16* w;   // [128]
  const float* in_scale;    // FP8 static input scale (device scalar) or nullptr
  __nv_bfloat16* y;         // [M, 128] BF16 output or nullptr
  uint8_t* q;               // [M, 128] FP8 output or nullptr
  int64_t ldz;
  int32_t M, heads;
  float eps;
};

template <int LANES>  // 16 (RPB>=2) or 32 (RPB==1)
__global__ __launch_bounds__(256) void gdn_gated_norm_kernel(const NormParams p) {
  constexpr int E = 128 / LANES;           // elements per lane: 8 or 4
  constexpr int RPW = 32 / LANES;          // rows per warp
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sub = lane / LANES, l = lane % LANES;
  const int row = (blockIdx.x * (blockDim.x >> 5) + warp) * RPW + sub;   // M < 2^31 (host check)
  const bool valid = row < p.M;
  const int rr = valid ? row : 0;
  const int t = rr / p.heads, h = rr - t * p.heads;
  float x[E], z[E], wv[E];
  {
    const uint16_t* op = reinterpret_cast<const uint16_t*>(p.o) + (int64_t)rr * 128 + l * E;
    const uint16_t* zp = reinterpret_cast<const uint16_t*>(p.z) + (int64_t)t * p.ldz + h * 128 + l * E;
    const uint16_t* wp = reinterpret_cast<const uint16_t*>(p.w) + l * E;
    if constexpr (E == 8) {
      const uint4 a = ldg16(op), b = ldg16(zp), c = ldg16(wp);
#pragma unroll
      for (int i = 0; i < 8; ++i) { x[i] = bf_at(a, i); z[i] = bf_at(b, i); wv[i] = bf_at(c, i); }
    } else {
      const uint2 a = __ldg(reinterpret_cast<const uint2*>(op));
      const uint2 b = __ldg(reinterpret_cast<const uint2*>(zp));
      const uint2 c = __ldg(reinterpret_cast<const uint2*>(wp));
      x[0] = bf_lo(a.x); x[1] = bf_hi(a.x); x[2] = bf_lo(a.y); x[3] = bf_hi(a.y);
      z[0] = bf_lo(b.x); z[1] = bf_hi(b.x); z[2] = bf_lo(b.y); z[3] = bf_hi(b.y);
      wv[0] = bf_lo(c.x); wv[1] = bf_hi(c.x); wv[2] = bf_lo(c.y); wv[3] = bf_hi(c.y);
    }
  }
  float ss = mul_rn(x[1], x[1]);
  ss = fma_rn(x[0], x[0], ss);
#pragma unroll
  for (int i = 2; i < E; ++i) ss = fma_rn(x[i], x[i], ss);
#pragma unroll
  for (int off = LANES / 2; off >= 1; off >>= 1) ss = add_rn(ss, __shfl_xor_sync(0xffffffffu, ss, off));
  const float rstd = rsqrt_approx_ftz(fma_rn(ss, 0.0078125f, p.eps));
  float out[E];
#pragma unroll
  for (int i = 0; i < E; ++i) {
    const float yv = mul_rn(mul_rn(rstd, x[i]), wv[i]);
    const float nz = sub_rn(0.0f, z[i]);
    const float e = ex2_approx(mul_rn(nz, 1.4426950216293334961f));
    const float sg = div_full(1.0f, add_rn(e, 1.0f));
    out[i] = mul_rn(yv, mul_rn(sg, z[i]));
  }
  if (!valid) return;
  uint32_t packed[E / 2];
#pragma unroll
  for (int i = 0; i < E / 2; ++i) packed[i] = pack_bf16_rn(out[2 * i], out[2 * i + 1]);
  if (p.y != nullptr) {
    uint16_t* yp = reinterpret_cast<uint16_t*>(p.y) + (int64_t)rr * 128 + l * E;
    if constexpr (E == 8) st16(yp, make_uint4(packed[0], packed[1], packed[2], packed[3]));
    else *reinterpret_cast<uint2*>(yp) = make_uint2(packed[0], packed[1]);
  }
  if (p.q != nullptr) {
    const float inv = div_full(1.0f, __ldg(p.in_scale));
    uint16_t q8[E / 2];
#pragma unroll
    for (int i = 0; i < E / 2; ++i) {
      const float lo = min_f(max_f(mul_rn(inv, bf_lo(packed[i])), -448.0f), 448.0f);
      const float hi = min_f(max_f(mul_rn(inv, bf_hi(packed[i])), -448.0f), 448.0f);
      q8[i] = e4m3x2_rn_satfinite(lo, hi);
    }
    uint8_t* qp = p.q + (int64_t)rr * 128 + l * E;
    if constexpr (E == 8) {
      *reinterpret_cast<uint2*>(qp) = make_uint2((uint32_t)q8[0] | ((uint32_t)q8[1] << 16),
                                                 (uint32_t)q8[2] | ((uint32_t)q8[3] << 16));
    } else {
      *reinterpret_cast<uint32_t*>(qp) = (uint32_t)q8[0] | ((uint32_t)q8[1] << 16);
    }
  }
}

// ==========================================================================================
// Kernel 4: decode (one token per sequence), fully fused per GDN layer:
//   sgl_kernel causal_conv1d_update (+SiLU, in-place state shift)            [conv]
//   Triton fused_sigmoid_gating_delta_rule_update_kernel (BK=128, BV=32, 1 warp) [recurrence]
//   Triton _layer_norm_fwd_1pass_kernel (ROWS_PER_BLOCK rule)                 [gated norm]
//   FreeToken _static_quant_kernel                                            [FP8 for out_proj]
// One CTA per (sequence, key head kh) = 8 warps; warp w plays the Triton program
// (i_v = w % 4, i_hv = 2*kh + w / 4) with the identical lane layout (lane l owns K rows
// 4l..4l+3 for all 32 V columns), so every reduction tree and FMA chain is the original one.
// The q/k conv channels of kh are shared by both value heads; computing and updating them in
// one CTA removes the cross-program conv-state race the separate kernels never had to face.
// Recurrence recipe (Triton PTX/SASS):
//   h = state + 0.0 (add.f32x2 with the zero accumulator: -0 -> +0)
//   x = a + dt_bias ; bx = softplus_beta*x ; e = ex2.approx(bx*log2e) ; y = e + 1
//   lg = libdevice __nv_logf (FTZ build) of y ; sp = (bx <= thr) ? div.full(1,beta)*lg : x
//   g = (0 - ex2.approx(A_log*log2e)) * sp ; beta = div.full(1, ex2.approx((0-b)*log2e) + 1)
//   q,k: s = x1*x1 ; fma x0,x2,x3 ; bfly 16..1 ; r = sqrt.approx.ftz(s + 1e-6) ; div.full(x, r)
//   q *= scale ; h *= ex2.approx(g*log2e)
//   per v: d = k1*h1 ; fma k0,h0 ; fma k2,h2 ; fma k3,h3 ; bfly ; v = (v - d) * beta
//   h = fma(k, v, h) ; o = q1*h1 ; fma q0,q2,q3 ; bfly ; o -> bf16 ; state = h
// ==========================================================================================
constexpr int kDecWarps = 8;

struct DecodeParams {
  const __nv_bfloat16* qkvz;     // [B, ldx]: conv_in columns [0, 8192), z columns [8192, 12288)
  const __nv_bfloat16* ba;       // [B, ldba]: b columns [0, 32), a columns [32, 64)
  const __nv_bfloat16* conv_w;   // [8192, 4]
  __nv_bfloat16* conv_states;    // [S, 8192, 3]
  const float* A_log;            // [32]
  const float* dt_bias;          // [32]
  float* rec;                    // [S, 32, 128(V), 128(K)] fp32
  const int32_t* idx;            // [B]
  const __nv_bfloat16* norm_w;   // [128]
  const float* in_scale;         // FP8 static scale (device scalar)
  uint8_t* q8;                   // [B, 4096] FP8 out (out_proj input) or nullptr
  __nv_bfloat16* y;              // [B, 4096] BF16 norm output or nullptr (tests)
  __nv_bfloat16* o_core;         // [B, 32, 128] recurrence output or nullptr (tests)
  __nv_bfloat16* mixed;          // [B, 8192] conv output or nullptr (tests)
  int64_t ldx, ldba, state_stride;  // conv_states slot stride (elements)
  int32_t pad_slot_id, norm_lanes;
  float scale, norm_eps;
  float softplus_beta, softplus_threshold;
};

// libdevice __nv_logf as inlined by Triton (enable_reflect_ftz=True), transcribed from the
// kernel's PTX; returns the value before the y==0 -> -inf select (done by the caller as in the
// original, on the pre-scaled operand).
__device__ __forceinline__ float nv_logf_ftz_core(float y, float& scaled) {
  float res, sc;
  asm("{\n\t"
      ".reg .pred p6, p7;\n\t"
      ".reg .b32 b51, t269, t270, t271;\n\t"
      ".reg .f32 f267, f51, f268, f272, f274, f275, ff, pp, f293, f294, c0, c1, k23, kln2, kinf;\n\t"
      "setp.lt.f32 p6, %2, 0f00800000;\n\t"
      "mul.rn.f32 f267, %2, 0f4B000000;\n\t"
      "selp.f32 f51, f267, %2, p6;\n\t"
      "selp.f32 f268, 0fC1B80000, 0f00000000, p6;\n\t"
      "mov.b32 b51, f51;\n\t"
      "add.s32 t269, b51, -1059760811;\n\t"
      "and.b32 t270, t269, -8388608;\n\t"
      "sub.s32 t271, b51, t270;\n\t"
      "cvt.rn.f32.s32 f272, t270;\n\t"
      "mov.b32 k23, 0f34000000;\n\t"
      "fma.rn.ftz.f32 f274, f272, k23, f268;\n\t"
      "mov.b32 ff, t271;\n\t"
      "add.rn.f32 f275, ff, 0fBF800000;\n\t"
      "mov.b32 c0, 0f3E1039F6;\n\t"
      "mov.b32 c1, 0fBE055027;\n\t"
      "fma.rn.ftz.f32 pp, c1, f275, c0;\n\t"
      "mov.b32 c0, 0fBDF8CDCC;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mov.b32 c0, 0f3E0F2955;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mov.b32 c0, 0fBE2AD8B9;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mov.b32 c0, 0f3E4CED0B;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mov.b32 c0, 0fBE7FFF22;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mov.b32 c0, 0f3EAAAA78;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mov.b32 c0, 0fBF000000;\n\t"
      "fma.rn.ftz.f32 pp, pp, f275, c0;\n\t"
      "mul.rn.f32 f293, f275, pp;\n\t"
      "fma.rn.ftz.f32 f294, f293, f275, f275;\n\t"
      "mov.b32 kln2, 0f3F317218;\n\t"
      "fma.rn.ftz.f32 %0, f274, kln2, f294;\n\t"
      "setp.lt.u32 p7, b51, 2139095040;\n\t"
      "mov.b32 kinf, 0f7F800000;\n\t"
      "@!p7 fma.rn.ftz.f32 %0, f51, kinf, kinf;\n\t"
      "mov.f32 %1, f51;\n\t"
      "}"
      : "=f"(res), "=f"(sc) : "f"(y));
  scaled = sc;
  return res;
}

__device__ __forceinline__ float l2norm4_butterfly(float (&x)[4]) {
  float s = mul_rn(x[1], x[1]);
  s = fma_rn(x[0], x[0], s);
  s = fma_rn(x[2], x[2], s);
  s = fma_rn(x[3], x[3], s);
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  return sqrt_approx_ftz(add_rn(s, 0.000001f));   // 0f358637BD
}

// ---- decode building blocks (shared by the 16-CTA and the cluster kernels) ---------------
// sgl_kernel causal_conv1d_update for 8 channels c..c+7 of sequence n: returns the conv output
// (raw input for a PAD slot) and the new state words [s1, s2, x] per channel.
__device__ __forceinline__ uint4 conv_update8(const DecodeParams& p, int n, int c, int32_t slot,
                                              bool pad, uint4 (&ns_words)[3]) {
  const uint4 xr = ldg16(p.qkvz + (int64_t)n * p.ldx + c);
  if (pad) return xr;  // the original returns early: output stays the raw input, state untouched
  const __nv_bfloat16* sp = p.conv_states + (int64_t)slot * p.state_stride + (int64_t)c * 3;
  uint32_t st[12];
#pragma unroll
  for (int q = 0; q < 3; ++q) {
    const uint4 v = ld16_coherent(sp + q * 8);
    st[q * 4 + 0] = v.x; st[q * 4 + 1] = v.y; st[q * 4 + 2] = v.z; st[q * 4 + 3] = v.w;
  }
  uint32_t raw[16];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 v = ldg16(p.conv_w + (int64_t)c * 4 + q * 8);
    raw[q * 4 + 0] = v.x; raw[q * 4 + 1] = v.y; raw[q * 4 + 2] = v.z; raw[q * 4 + 3] = v.w;
  }
  const uint32_t xw[4] = {xr.x, xr.y, xr.z, xr.w};
  float o[8];
  uint16_t ns[24];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float w0 = (((i * 4 + 0) & 1) ? bf_hi(raw[(i * 4 + 0) >> 1]) : bf_lo(raw[(i * 4 + 0) >> 1]));
    const float w1 = (((i * 4 + 1) & 1) ? bf_hi(raw[(i * 4 + 1) >> 1]) : bf_lo(raw[(i * 4 + 1) >> 1]));
    const float w2 = (((i * 4 + 2) & 1) ? bf_hi(raw[(i * 4 + 2) >> 1]) : bf_lo(raw[(i * 4 + 2) >> 1]));
    const float w3 = (((i * 4 + 3) & 1) ? bf_hi(raw[(i * 4 + 3) >> 1]) : bf_lo(raw[(i * 4 + 3) >> 1]));
    const uint16_t s0 = half_at(st, i * 3 + 0), s1 = half_at(st, i * 3 + 1), s2 = half_at(st, i * 3 + 2);
    const uint16_t xn = half_at(xw, i);
    o[i] = silu_sgl(conv4(bf16_to_f(s0), bf16_to_f(s1), bf16_to_f(s2), bf16_to_f(xn), w0, w1, w2, w3));
    ns[i * 3 + 0] = s1; ns[i * 3 + 1] = s2; ns[i * 3 + 2] = xn;
  }
#pragma unroll
  for (int q = 0; q < 3; ++q) {
    ns_words[q].x = (uint32_t)ns[q * 8 + 0] | ((uint32_t)ns[q * 8 + 1] << 16);
    ns_words[q].y = (uint32_t)ns[q * 8 + 2] | ((uint32_t)ns[q * 8 + 3] << 16);
    ns_words[q].z = (uint32_t)ns[q * 8 + 4] | ((uint32_t)ns[q * 8 + 5] << 16);
    ns_words[q].w = (uint32_t)ns[q * 8 + 6] | ((uint32_t)ns[q * 8 + 7] << 16);
  }
  uint4 outv;
  outv.x = pack_bf16_rn(o[0], o[1]); outv.y = pack_bf16_rn(o[2], o[3]);
  outv.z = pack_bf16_rn(o[4], o[5]); outv.w = pack_bf16_rn(o[6], o[7]);
  return outv;
}

__device__ __forceinline__ void store_conv_state8(const DecodeParams& p, int c, int32_t slot,
                                                  const uint4 (&ns_words)[3]) {
  __nv_bfloat16* sp = p.conv_states + (int64_t)slot * p.state_stride + (int64_t)c * 3;
#pragma unroll
  for (int q = 0; q < 3; ++q) st16(sp + q * 8, ns_words[q]);
}

__device__ __forceinline__ void load_state_warp(const DecodeParams& p, int32_t slot, bool has_state,
                                                int hv, int iv, int lane, float (&h)[4][32]) {
  const float* hp = p.rec + (int64_t)(has_state ? slot : 0) * 32 * 128 * 128 +
                    (int64_t)hv * 128 * 128 + (int64_t)(iv * 32) * 128 + lane * 4;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (has_state) v4 = __ldcs(reinterpret_cast<const float4*>(hp + j * 128));
    // Triton: b_h = zeros; b_h += load  ->  load + 0.0 (turns -0.0 into +0.0)
    h[0][j] = add_rn(v4.x, 0.0f); h[1][j] = add_rn(v4.y, 0.0f);
    h[2][j] = add_rn(v4.z, 0.0f); h[3][j] = add_rn(v4.w, 0.0f);
  }
}

__device__ __forceinline__ void store_state_warp(const DecodeParams& p, int32_t slot, int hv, int iv,
                                                 int lane, const float (&h)[4][32]) {
  float* hp = p.rec + (int64_t)slot * 32 * 128 * 128 + (int64_t)hv * 128 * 128 +
              (int64_t)(iv * 32) * 128 + lane * 4;
#pragma unroll
  for (int j = 0; j < 32; ++j)
    __stcs(reinterpret_cast<float4*>(hp + j * 128), make_float4(h[0][j], h[1][j], h[2][j], h[3][j]));
}

// One warp == one Triton program (i_v, i_hv): gating + L2 + delta rule. q, k: this lane's
// 4 raw BF16 conv outputs (elements 4l..4l+3) as float; v: the program's 32 values.
// Returns o_j (identical in all lanes) for the 32 V columns.
__device__ __forceinline__ void decode_warp_core(const DecodeParams& p, int n, int hv,
                                                 float (&h)[4][32], float (&q)[4], float (&k)[4],
                                                 const float (&v)[32], float (&o)[32]) {
  const uint16_t* bar_ = reinterpret_cast<const uint16_t*>(p.ba) + (int64_t)n * p.ldba;
  const float b_b = bf16_to_f(bar_[hv]);
  const float b_a = bf16_to_f(bar_[32 + hv]);
  const float A_log = __ldg(p.A_log + hv);
  const float dtb = __ldg(p.dt_bias + hv);
  const float x = add_rn(b_a, dtb);
  const float bx = mul_rn(p.softplus_beta, x);
  const float y1 = add_rn(ex2_approx(mul_rn(bx, 1.4426950216293334961f)), 1.0f);
  float scaled;
  const float lg0 = nv_logf_ftz_core(y1, scaled);
  const float lg = (scaled == 0.0f) ? __uint_as_float(0xff800000u) : lg0;
  const float inv_beta = div_full(1.0f, p.softplus_beta);
  const float sp = (bx <= p.softplus_threshold) ? mul_rn(inv_beta, lg) : x;
  const float g = mul_rn(sub_rn(0.0f, ex2_approx(mul_rn(A_log, 1.4426950216293334961f))), sp);
  const float beta = div_full(1.0f, add_rn(ex2_approx(mul_rn(sub_rn(0.0f, b_b), 1.4426950216293334961f)), 1.0f));

  const float rq = l2norm4_butterfly(q);
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = div_full(q[i], rq);
  const float rk = l2norm4_butterfly(k);
#pragma unroll
  for (int i = 0; i < 4; ++i) k[i] = div_full(k[i], rk);
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = mul_rn(p.scale, q[i]);

  const float eg = ex2_approx(mul_rn(g, 1.4426950216293334961f));
#pragma unroll
  for (int j = 0; j < 32; ++j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i][j] = mul_rn(h[i][j], eg);
  }
  float d[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    float s = mul_rn(k[1], h[1][j]);
    s = fma_rn(k[0], h[0][j], s);
    s = fma_rn(k[2], h[2][j], s);
    d[j] = fma_rn(k[3], h[3][j], s);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
#pragma unroll
    for (int j = 0; j < 32; ++j) d[j] = add_rn(d[j], __shfl_xor_sync(0xffffffffu, d[j], off));
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const float vn = mul_rn(beta, sub_rn(v[j], d[j]));
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i][j] = fma_rn(k[i], vn, h[i][j]);
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    float s = mul_rn(q[1], h[1][j]);
    s = fma_rn(q[0], h[0][j], s);
    s = fma_rn(q[2], h[2][j], s);
    o[j] = fma_rn(q[3], h[3][j], s);
  }
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j] = add_rn(o[j], __shfl_xor_sync(0xffffffffu, o[j], off));
  }
}

__device__ __forceinline__ uint16_t pick_lane(const float (&o)[32], int lane) {
  float mine = o[0];
#pragma unroll
  for (int j = 1; j < 32; ++j) mine = (lane == j) ? o[j] : mine;
  return bf16_rn(mine);
}

// GatedRMSNorm (+FP8) of one value-head row. get(i) returns the BF16 bits of o[i] (i in 0..127).
template <typename Get>
__device__ __forceinline__ void decode_norm_row(const DecodeParams& p, int n, int hv, int lane, Get get) {
  const int L = p.norm_lanes;          // 32 (ROWS_PER_BLOCK 1) or 16 (>= 2)
  const int E = 128 / L;               // 4 or 8 elements per lane
  if (lane >= L) return;               // 16-lane layout: the row lives in lanes 0..15
  float xv[8], zv[8], wv[8];
  const uint16_t* zr = reinterpret_cast<const uint16_t*>(p.qkvz) + (int64_t)n * p.ldx + 8192 + hv * 128 + lane * E;
  const uint16_t* wr = reinterpret_cast<const uint16_t*>(p.norm_w) + lane * E;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < E) {
      xv[i] = bf16_to_f(get(lane * E + i));
      zv[i] = bf16_to_f(zr[i]);
      wv[i] = bf16_to_f(wr[i]);
    }
  }
  float ss = mul_rn(xv[1], xv[1]);
  ss = fma_rn(xv[0], xv[0], ss);
#pragma unroll
  for (int i = 2; i < 8; ++i) if (i < E) ss = fma_rn(xv[i], xv[i], ss);
  const unsigned mask = (L == 32) ? 0xffffffffu : 0x0000ffffu;
  for (int off = L / 2; off >= 1; off >>= 1) ss = add_rn(ss, __shfl_xor_sync(mask, ss, off));
  const float rstd = rsqrt_approx_ftz(fma_rn(ss, 0.0078125f, p.norm_eps));
  uint16_t yb[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < E) {
      const float yv = mul_rn(mul_rn(rstd, xv[i]), wv[i]);
      const float e = ex2_approx(mul_rn(sub_rn(0.0f, zv[i]), 1.4426950216293334961f));
      const float sg = div_full(1.0f, add_rn(e, 1.0f));
      yb[i] = bf16_rn(mul_rn(yv, mul_rn(sg, zv[i])));
    }
  }
  const int64_t ob = (int64_t)n * 4096 + hv * 128 + lane * E;
  if (p.y != nullptr) {
#pragma unroll
    for (int i = 0; i < 8; ++i) if (i < E) reinterpret_cast<uint16_t*>(p.y)[ob + i] = yb[i];
  }
  if (p.q8 != nullptr) {
    const float inv = div_full(1.0f, __ldg(p.in_scale));
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      if (i < E) {
        const float lo = min_f(max_f(mul_rn(inv, bf16_to_f(yb[i])), -448.0f), 448.0f);
        const float hi = min_f(max_f(mul_rn(inv, bf16_to_f(yb[i + 1])), -448.0f), 448.0f);
        *reinterpret_cast<uint16_t*>(p.q8 + ob + i) = e4m3x2_rn_satfinite(lo, hi);
      }
    }
  }
}

// ---- variant A: one CTA per (sequence, key head), 8 warps ----------------------------------
__global__ __launch_bounds__(kDecWarps * 32) void gdn_decode_fused_kernel(const DecodeParams p) {
  const int kh = blockIdx.x;           // key head 0..15
  const int n = blockIdx.y;            // sequence (one token each)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int hvl = warp >> 2, iv = warp & 3, hv = 2 * kh + hvl;
  __shared__ __align__(16) uint16_t s_q[128], s_k[128], s_v[256];
  __shared__ __align__(16) uint16_t s_o[2][128];

  const int32_t slot = p.idx[n];
  const bool pad = (slot == p.pad_slot_id);
  const bool has_state = slot >= 0;   // Triton: load/store the recurrent state only if idx >= 0
  float h[4][32];
  load_state_warp(p, slot, has_state, hv, iv, lane, h);   // issued first, overlaps the conv

  if (tid < 64) {
    int c;  // first of 8 channels
    if (tid < 16) c = kh * 128 + tid * 8;                       // q
    else if (tid < 32) c = 2048 + kh * 128 + (tid - 16) * 8;    // k
    else c = 4096 + (2 * kh) * 128 + (tid - 32) * 8;            // v of both value heads
    uint4 ns[3];
    const uint4 outv = conv_update8(p, n, c, slot, pad, ns);
    if (!pad) store_conv_state8(p, c, slot, ns);   // sole reader+writer of these channels
    uint16_t* dst = tid < 16 ? s_q + tid * 8 : tid < 32 ? s_k + (tid - 16) * 8 : s_v + (tid - 32) * 8;
    *reinterpret_cast<uint4*>(dst) = outv;
    if (p.mixed != nullptr) st16(p.mixed + (int64_t)n * 8192 + c, outv);
  }
  __syncthreads();

  float q[4], k[4], v[32], o[32];
  {
    const uint2 qq = *reinterpret_cast<const uint2*>(s_q + lane * 4);
    const uint2 kk = *reinterpret_cast<const uint2*>(s_k + lane * 4);
    q[0] = bf_lo(qq.x); q[1] = bf_hi(qq.x); q[2] = bf_lo(qq.y); q[3] = bf_hi(qq.y);
    k[0] = bf_lo(kk.x); k[1] = bf_hi(kk.x); k[2] = bf_lo(kk.y); k[3] = bf_hi(kk.y);
#pragma unroll
    for (int j = 0; j < 32; ++j) v[j] = bf16_to_f(s_v[hvl * 128 + iv * 32 + j]);
  }
  decode_warp_core(p, n, hv, h, q, k, v, o);
  const uint16_t ob = pick_lane(o, lane);
  s_o[hvl][iv * 32 + lane] = ob;
  if (p.o_core != nullptr)
    reinterpret_cast<uint16_t*>(p.o_core)[((int64_t)n * 32 + hv) * 128 + iv * 32 + lane] = ob;
  if (has_state) store_state_warp(p, slot, hv, iv, lane, h);
  __syncthreads();
  if (iv != 0) return;
  decode_norm_row(p, n, hv, lane, [&](int i) { return s_o[hvl][i]; });
}

// ---- variant B: Triton's decomposition (128 one-warp programs per sequence) in clusters of 8
// CTAs = one key head (2 value heads x 4 V blocks). The q/k conv of the key head is computed by
// every CTA (identical values); its state is rewritten by rank 0 only after a cluster barrier
// guarantees that all 8 CTAs have read the old state. The two norm CTAs gather their head's
// 128 outputs from the 4 V-block CTAs through distributed shared memory.
__global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(32)
gdn_decode_cluster_kernel(const DecodeParams p) {
  cg::cluster_group cluster = cg::this_cluster();
  const int r = (int)cluster.block_rank();          // 0..7
  const int kh = blockIdx.x / 8, n = blockIdx.y, lane = threadIdx.x;
  const int hvl = r >> 2, iv = r & 3, hv = 2 * kh + hvl;
  __shared__ __align__(16) uint16_t s_q[128], s_k[128], s_v[32], s_o[32];

  const int32_t slot = p.idx[n];
  const bool pad = (slot == p.pad_slot_id);
  const bool has_state = slot >= 0;
  float h[4][32];
  load_state_warp(p, slot, has_state, hv, iv, lane, h);

  // q (lanes 0..15) and k (lanes 16..31) channels of the key head: 8 channels per lane
  const int cqk = lane < 16 ? kh * 128 + lane * 8 : 2048 + kh * 128 + (lane - 16) * 8;
  uint4 ns_qk[3];
  const uint4 out_qk = conv_update8(p, n, cqk, slot, pad, ns_qk);
  *reinterpret_cast<uint4*>((lane < 16 ? s_q + lane * 8 : s_k + (lane - 16) * 8)) = out_qk;
  // v channels of this program (32 channels): lanes 0..3, 8 channels each, owned by this CTA
  if (lane < 4) {
    const int cv = 4096 + hv * 128 + iv * 32 + lane * 8;
    uint4 ns_v[3];
    const uint4 out_v = conv_update8(p, n, cv, slot, pad, ns_v);
    if (!pad) store_conv_state8(p, cv, slot, ns_v);
    *reinterpret_cast<uint4*>(s_v + lane * 8) = out_v;
    if (p.mixed != nullptr) st16(p.mixed + (int64_t)n * 8192 + cv, out_v);
  }
  cluster.sync();                                   // every CTA has read the old q/k state
  if (r == 0 && !pad) store_conv_state8(p, cqk, slot, ns_qk);
  if (r == 0 && p.mixed != nullptr) st16(p.mixed + (int64_t)n * 8192 + cqk, out_qk);
  __syncwarp();

  float q[4], k[4], v[32], o[32];
  {
    const uint2 qq = *reinterpret_cast<const uint2*>(s_q + lane * 4);
    const uint2 kk = *reinterpret_cast<const uint2*>(s_k + lane * 4);
    q[0] = bf_lo(qq.x); q[1] = bf_hi(qq.x); q[2] = bf_lo(qq.y); q[3] = bf_hi(qq.y);
    k[0] = bf_lo(kk.x); k[1] = bf_hi(kk.x); k[2] = bf_lo(kk.y); k[3] = bf_hi(kk.y);
#pragma unroll
    for (int j = 0; j < 32; ++j) v[j] = bf16_to_f(s_v[j]);
  }
  decode_warp_core(p, n, hv, h, q, k, v, o);
  const uint16_t ob = pick_lane(o, lane);
  s_o[lane] = ob;
  if (p.o_core != nullptr)
    reinterpret_cast<uint16_t*>(p.o_core)[((int64_t)n * 32 + hv) * 128 + iv * 32 + lane] = ob;
  if (has_state) store_state_warp(p, slot, hv, iv, lane, h);
  cluster.sync();                                   // the head's 4 output slices are published
  if (iv == 0) {
    const uint16_t* remote[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) remote[b] = cluster.map_shared_rank(s_o, hvl * 4 + b);
    decode_norm_row(p, n, hv, lane, [&](int i) { return remote[i >> 5][i & 31]; });
  }
  cluster.sync();                                   // keep s_o alive until the readers are done
}

// ==========================================================================================
// Exhaustive-check helpers (GPU tests): evaluate single recipes over input ranges.
// ==========================================================================================
__global__ void silu_eval_kernel(uint64_t begin, uint64_t count, uint32_t* out) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
    out[i] = __float_as_uint(silu_sgl(__uint_as_float((uint32_t)(begin + i))));
}
__global__ void silu_points_kernel(const float* in, float* out, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = silu_sgl(in[i]);
}
// Pinned PTX SiLU vs the plain source expression of sgl_kernel (x / (1 + expf(-x)), compiled
// here without fast-math) over a range of bit patterns; counts bit mismatches.
__global__ void silu_compare_kernel(uint64_t begin, uint64_t count, unsigned long long* mismatches) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  unsigned long long local = 0;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
    const float x = __uint_as_float((uint32_t)(begin + i));
    const float a = silu_sgl(x);
    const float b = x / (1 + expf(-x));
    local += (__float_as_uint(a) != __float_as_uint(b));
  }
  if (local) atomicAdd(mismatches, local);
}
// Per (head, bf16 a pattern): g_raw recipe; per bf16 b pattern: beta recipe.
__global__ void gate_eval_kernel(const float* A_log, const float* dt_bias, int H,
                                 float* graw_out /*[H,65536]*/, float* beta_out /*[65536]*/) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * 65536) return;
  const int h = idx >> 16;
  const uint16_t bits = (uint16_t)(idx & 0xffff);
  const float a = bf16_to_f(bits);
  const float x = add_rn(a, dt_bias[h]);
  graw_out[idx] = mul_rn(-expf(A_log[h]), softplus_aten(x));
  if (h == 0) beta_out[bits] = bf16_to_f(bf16_rn(sigmoid_aten(a)));
}

// FLA chunk kernels (ABI 2): fused chunk_h + chunk_o.
#include "qwen36_gdn_fla.cuh"

}  // namespace

// ==========================================================================================
// C ABI
// ==========================================================================================
extern "C" {

// 2: + qwen_gdn_fla_chunk_ho (fused FLA chunk_h + chunk_o)
//    + qwen_gdn_fla_wu_chunk_ho (the same with recompute_w_u folded in).
int qwen_gdn_abi_version() { return 2; }

// Returns 0 on success, positive error code otherwise.
int qwen_gdn_conv_qkv(const void* x, int64_t ldx, const void* weight, void* conv_states,
                      int64_t state_batch_stride, const void* cu_seqlens,
                      const void* cache_indices, const void* has_initial_state, void* out_q,
                      void* out_k, void* out_v, int32_t T, int32_t B, int32_t C,
                      int32_t key_dim, int32_t value_dim, int32_t pad_slot_id, int32_t norm_qk,
                      float l2_eps, int32_t tile_tokens, int32_t batch_rows, void* stream) {
  if (T <= 0 || B <= 0) return 1;
  if (C != 2 * key_dim + value_dim || key_dim % kGroup || value_dim % kGroup) return 2;
  if (ldx < C || ldx % kVec) return 3;
  if (state_batch_stride < (int64_t)C * 3 || state_batch_stride % kVec) return 5;
  const uintptr_t align = reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(weight) |
                          reinterpret_cast<uintptr_t>(conv_states) |
                          reinterpret_cast<uintptr_t>(out_q) | reinterpret_cast<uintptr_t>(out_k) |
                          reinterpret_cast<uintptr_t>(out_v);
  if (align % 16) return 6;
  ConvParams p;
  p.x = static_cast<const __nv_bfloat16*>(x);
  p.weight = static_cast<const __nv_bfloat16*>(weight);
  p.conv_states = static_cast<__nv_bfloat16*>(conv_states);
  p.cu_seqlens = static_cast<const int64_t*>(cu_seqlens);
  p.cache_indices = static_cast<const int32_t*>(cache_indices);
  p.has_initial_state = static_cast<const bool*>(has_initial_state);
  p.out_q = static_cast<__nv_bfloat16*>(out_q);
  p.out_k = static_cast<__nv_bfloat16*>(out_k);
  p.out_v = static_cast<__nv_bfloat16*>(out_v);
  p.ldx = ldx; p.state_batch_stride = state_batch_stride;
  p.B = B; p.C = C; p.key_dim = key_dim; p.value_dim = value_dim;
  p.pad_slot_id = pad_slot_id; p.norm_qk = norm_qk; p.l2_eps = l2_eps;
  const int TT = tile_tokens;
  if (TT != 16 && TT != 32 && TT != 64 && TT != 128) return 7;
  if (batch_rows != 4 && batch_rows != 8) return 7;
  const int64_t tiles = ((int64_t)T + TT - 1) / TT + B;
  if (tiles > 65535) return 8;
  const dim3 grid(C / kGroup, (unsigned)tiles);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
#define QWEN_GDN_LAUNCH(TTV, BRV) gdn_conv_qkv_kernel<TTV, BRV><<<grid, kConvThreads, 0, s>>>(p)
  if (batch_rows == 8) {
    switch (TT) { case 16: QWEN_GDN_LAUNCH(16, 8); break; case 32: QWEN_GDN_LAUNCH(32, 8); break;
                  case 64: QWEN_GDN_LAUNCH(64, 8); break; default: QWEN_GDN_LAUNCH(128, 8); break; }
  } else {
    switch (TT) { case 16: QWEN_GDN_LAUNCH(16, 4); break; case 32: QWEN_GDN_LAUNCH(32, 4); break;
                  case 64: QWEN_GDN_LAUNCH(64, 4); break; default: QWEN_GDN_LAUNCH(128, 4); break; }
  }
#undef QWEN_GDN_LAUNCH
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

// num_chunks: host upper bound of sum_i cdiv(len_i, 64) (e.g. cdiv(T,64)+B); surplus exits.
int qwen_gdn_gate_cumsum(const void* ba, int64_t ldba, const void* A_log, const void* dt_bias,
                         const void* cu_seqlens, void* g_out, void* beta_out, int32_t T,
                         int32_t B, int32_t H, int32_t num_chunks, void* stream) {
  if (T <= 0 || B <= 0 || H <= 0 || num_chunks <= 0) return 1;
  if (ldba < 2 * H) return 3;
  GateParams p;
  p.ba = static_cast<const __nv_bfloat16*>(ba);
  p.A_log = static_cast<const float*>(A_log);
  p.dt_bias = static_cast<const float*>(dt_bias);
  p.cu_seqlens = static_cast<const int64_t*>(cu_seqlens);
  p.g_out = static_cast<float*>(g_out);
  p.beta_out = static_cast<float*>(beta_out);
  p.ldba = ldba; p.B = B; p.H = H;
  const dim3 grid((unsigned)num_chunks, (unsigned)((H + 3) / 4));
  gdn_gate_cumsum_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

// lanes: 16 (Triton ROWS_PER_BLOCK >= 2) or 32 (ROWS_PER_BLOCK == 1).
int qwen_gdn_gated_norm(const void* o, const void* z, int64_t ldz, const void* w,
                        const void* in_scale, void* y, void* q, int32_t M, int32_t heads,
                        float eps, int32_t lanes, void* stream) {
  if (M <= 0 || heads <= 0 || M > (1 << 30)) return 1;
  if (y == nullptr && q == nullptr) return 2;
  if (q != nullptr && in_scale == nullptr) return 2;
  if (ldz % 8 || (reinterpret_cast<uintptr_t>(o) | reinterpret_cast<uintptr_t>(z) |
                  reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(y)) % 16) return 6;
  if (q != nullptr && reinterpret_cast<uintptr_t>(q) % 8) return 6;
  NormParams p;
  p.o = static_cast<const __nv_bfloat16*>(o);
  p.z = static_cast<const __nv_bfloat16*>(z);
  p.w = static_cast<const __nv_bfloat16*>(w);
  p.in_scale = static_cast<const float*>(in_scale);
  p.y = static_cast<__nv_bfloat16*>(y);
  p.q = static_cast<uint8_t*>(q);
  p.ldz = ldz; p.M = M; p.heads = heads; p.eps = eps;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (lanes == 16) {
    const int rows_per_block = 8 * 2;
    gdn_gated_norm_kernel<16><<<(M + rows_per_block - 1) / rows_per_block, 256, 0, s>>>(p);
  } else if (lanes == 32) {
    const int rows_per_block = 8;
    gdn_gated_norm_kernel<32><<<(M + rows_per_block - 1) / rows_per_block, 256, 0, s>>>(p);
  } else {
    return 7;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

// Decode, one launch per GDN layer. B sequences (one token each). Returns 0 on success.
int qwen_gdn_decode_fused(const void* qkvz, const void* ba, const void* conv_w, void* conv_states,
                          const void* A_log, const void* dt_bias, void* rec, const void* idx,
                          const void* norm_w, const void* in_scale, void* q8, void* y,
                          void* o_core, void* mixed, int64_t ldx, int64_t ldba, int64_t state_stride,
                          int32_t B, int32_t norm_lanes, float scale, float norm_eps, void* stream) {
  if (B <= 0 || B > 65535) return 1;
  if (ldx < 12288 || ldx % 8 || ldba < 64) return 3;
  if (state_stride < 8192 * 3 || state_stride % 8) return 5;
  if (norm_lanes != 16 && norm_lanes != 32) return 7;
  if (q8 == nullptr && y == nullptr) return 2;
  if ((reinterpret_cast<uintptr_t>(qkvz) | reinterpret_cast<uintptr_t>(conv_w) |
       reinterpret_cast<uintptr_t>(conv_states) | reinterpret_cast<uintptr_t>(rec)) % 16) return 6;
  DecodeParams p;
  p.qkvz = static_cast<const __nv_bfloat16*>(qkvz);
  p.ba = static_cast<const __nv_bfloat16*>(ba);
  p.conv_w = static_cast<const __nv_bfloat16*>(conv_w);
  p.conv_states = static_cast<__nv_bfloat16*>(conv_states);
  p.A_log = static_cast<const float*>(A_log);
  p.dt_bias = static_cast<const float*>(dt_bias);
  p.rec = static_cast<float*>(rec);
  p.idx = static_cast<const int32_t*>(idx);
  p.norm_w = static_cast<const __nv_bfloat16*>(norm_w);
  p.in_scale = static_cast<const float*>(in_scale);
  p.q8 = static_cast<uint8_t*>(q8);
  p.y = static_cast<__nv_bfloat16*>(y);
  p.o_core = static_cast<__nv_bfloat16*>(o_core);
  p.mixed = static_cast<__nv_bfloat16*>(mixed);
  p.ldx = ldx; p.ldba = ldba; p.state_stride = state_stride;
  p.pad_slot_id = -1; p.norm_lanes = norm_lanes;
  p.scale = scale; p.norm_eps = norm_eps;
  p.softplus_beta = 1.0f; p.softplus_threshold = 20.0f;
  gdn_decode_fused_kernel<<<dim3(16, B), kDecWarps * 32, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

// Same contract as qwen_gdn_decode_fused; Triton's 128-program decomposition in clusters of 8.
int qwen_gdn_decode_cluster(const void* qkvz, const void* ba, const void* conv_w, void* conv_states,
                            const void* A_log, const void* dt_bias, void* rec, const void* idx,
                            const void* norm_w, const void* in_scale, void* q8, void* y,
                            void* o_core, void* mixed, int64_t ldx, int64_t ldba, int64_t state_stride,
                            int32_t B, int32_t norm_lanes, float scale, float norm_eps, void* stream) {
  if (B <= 0 || B > 65535) return 1;
  if (ldx < 12288 || ldx % 8 || ldba < 64) return 3;
  if (state_stride < 8192 * 3 || state_stride % 8) return 5;
  if (norm_lanes != 16 && norm_lanes != 32) return 7;
  if (q8 == nullptr && y == nullptr) return 2;
  if ((reinterpret_cast<uintptr_t>(qkvz) | reinterpret_cast<uintptr_t>(conv_w) |
       reinterpret_cast<uintptr_t>(conv_states) | reinterpret_cast<uintptr_t>(rec)) % 16) return 6;
  DecodeParams p;
  p.qkvz = static_cast<const __nv_bfloat16*>(qkvz);
  p.ba = static_cast<const __nv_bfloat16*>(ba);
  p.conv_w = static_cast<const __nv_bfloat16*>(conv_w);
  p.conv_states = static_cast<__nv_bfloat16*>(conv_states);
  p.A_log = static_cast<const float*>(A_log);
  p.dt_bias = static_cast<const float*>(dt_bias);
  p.rec = static_cast<float*>(rec);
  p.idx = static_cast<const int32_t*>(idx);
  p.norm_w = static_cast<const __nv_bfloat16*>(norm_w);
  p.in_scale = static_cast<const float*>(in_scale);
  p.q8 = static_cast<uint8_t*>(q8);
  p.y = static_cast<__nv_bfloat16*>(y);
  p.o_core = static_cast<__nv_bfloat16*>(o_core);
  p.mixed = static_cast<__nv_bfloat16*>(mixed);
  p.ldx = ldx; p.ldba = ldba; p.state_stride = state_stride;
  p.pad_slot_id = -1; p.norm_lanes = norm_lanes;
  p.scale = scale; p.norm_eps = norm_eps;
  p.softplus_beta = 1.0f; p.softplus_threshold = 20.0f;
  gdn_decode_cluster_kernel<<<dim3(128, B), 32, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

int qwen_gdn_silu_eval(uint64_t begin, uint64_t count, void* out, void* stream) {
  silu_eval_kernel<<<2048, 256, 0, static_cast<cudaStream_t>(stream)>>>(begin, count, static_cast<uint32_t*>(out));
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

int qwen_gdn_silu_points(const void* in, void* out, int64_t n, void* stream) {
  if (n <= 0) return 1;
  silu_points_kernel<<<(unsigned)((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(in), static_cast<float*>(out), n);
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

int qwen_gdn_silu_compare(uint64_t begin, uint64_t count, void* mismatches, void* stream) {
  silu_compare_kernel<<<4096, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      begin, count, static_cast<unsigned long long*>(mismatches));
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

int qwen_gdn_gate_eval(const void* A_log, const void* dt_bias, int32_t H, void* graw_out,
                       void* beta_out, void* stream) {
  gate_eval_kernel<<<(H * 65536 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(A_log), static_cast<const float*>(dt_bias), H,
      static_cast<float*>(graw_out), static_cast<float*>(beta_out));
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

// Fused FLA chunk_h + chunk_o for the Qwen3.6 GDN prefill (H=32, Hg=16, K=V=128, BT=64, varlen).
// q,k [T,16,128] bf16; w [T,32,128] bf16; u [T,32,128] bf16; g [T,32] fp32 (chunk-local cumsum);
// state [S,32,128,128] fp32 (read at state_idx[n], final state written back in place, snapshot
// at the start of chunk snap_rows[s]-chunk_offsets[n] into slot snap_dsts[s]); o [T,32,128] bf16.
// cu_seqlens [N+1] int64, chunk_offsets [N] int64 (only read when num_snap > 0).
// The caller guarantees the shapes; T_total must be > 32 (chunk_o BT == 64).
int qwen_gdn_fla_chunk_ho(const void* q, const void* k, const void* w, const void* u, const void* g,
                          void* state, const void* state_idx, const void* snap_rows,
                          const void* snap_dsts, int32_t num_snap, const void* cu_seqlens,
                          const void* chunk_offsets, int32_t N, float scale, void* o, void* stream) {
  if (N <= 0 || N > 65535 || num_snap < 0) return 1;
  if (num_snap > 0 && (!snap_rows || !snap_dsts || !chunk_offsets)) return 1;
  const uintptr_t align = reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                          reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(u) |
                          reinterpret_cast<uintptr_t>(o) | reinterpret_cast<uintptr_t>(state);
  if (align % 16) return 6;
  static bool smem_set = false;
  if (!smem_set) {
    if (cudaFuncSetAttribute(fla::chunk_ho_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             fla::kSmem) != cudaSuccess) return 9;
    smem_set = true;
  }
  const dim3 grid(fla::kVD / fla::kBV, fla::kNH, (unsigned)N);
  fla::chunk_ho_fused_kernel<<<grid, fla::kThreads, fla::kSmem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(w), static_cast<const __nv_bfloat16*>(u),
      static_cast<const float*>(g), static_cast<float*>(state), static_cast<const int32_t*>(state_idx),
      static_cast<const int64_t*>(snap_rows), static_cast<const int64_t*>(snap_dsts), num_snap,
      static_cast<const int64_t*>(cu_seqlens), static_cast<const int64_t*>(chunk_offsets), scale,
      static_cast<__nv_bfloat16*>(o));
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

// Same, with FLA recompute_w_u folded in: v [T,32,128] bf16 (raw values), A_wy [T,32,64] bf16
// (kkt_solve output), beta [T,32] fp32 replace w and u.
int qwen_gdn_fla_wu_chunk_ho(const void* q, const void* k, const void* v, const void* a_wy,
                             const void* beta, const void* g, void* state, const void* state_idx,
                             const void* snap_rows, const void* snap_dsts, int32_t num_snap,
                             const void* cu_seqlens, const void* chunk_offsets, int32_t N, float scale,
                             void* o, void* stream) {
  if (N <= 0 || N > 65535 || num_snap < 0) return 1;
  if (num_snap > 0 && (!snap_rows || !snap_dsts || !chunk_offsets)) return 1;
  const uintptr_t align = reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                          reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(a_wy) |
                          reinterpret_cast<uintptr_t>(o) | reinterpret_cast<uintptr_t>(state);
  if (align % 16) return 6;
  static bool smem_set = false;
  if (!smem_set) {
    if (cudaFuncSetAttribute(fla::chunk_wu_ho_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             fla::kW2Smem) != cudaSuccess) return 9;
    smem_set = true;
  }
  const dim3 grid(fla::kVD / fla::kBV, fla::kNH, (unsigned)N);
  fla::chunk_wu_ho_fused_kernel<<<grid, fla::kThreads, fla::kW2Smem, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<const __nv_bfloat16*>(a_wy),
      static_cast<const float*>(beta), static_cast<const float*>(g), static_cast<float*>(state),
      static_cast<const int32_t*>(state_idx), static_cast<const int64_t*>(snap_rows),
      static_cast<const int64_t*>(snap_dsts), num_snap, static_cast<const int64_t*>(cu_seqlens),
      static_cast<const int64_t*>(chunk_offsets), scale, static_cast<__nv_bfloat16*>(o));
  return cudaGetLastError() == cudaSuccess ? 0 : 9;
}

}  // extern "C"
