// qwen_moe_decode.cu -- bit-exact replacement of FreeToken's NVFP4 MoE decode GEMV
// (moe/fused_nvfp4._decode_gemm_marlin -> Triton _decode_nvfp4_marlin_kernel, BLOCK_N 16, BLOCK_KW 16,
// 4 warps) on SM120, for both decode GEMMs of Qwen3.6-35B-A3B (gate_up: N 1024, K 2048;
// down: N 2048, K 512, router weight in the epilogue).
//
// Arithmetic of the Triton kernel (TTGIR blocked<[2,1],[8,4],[1,4]>, PTX and SASS of the pinned
// Triton 3.6 build, kernels/decode-moe/scripts/dump_triton.py), per output (route, n):
//   * 16 FP32 partials P[l], l = 0..15; word w (8 e2m1 codes, k = 8w + j, nibble j) belongs to
//     P[w % 16]; each partial walks its words in ascending order:
//       acc = +0; for j in 0..7: acc = acc + a[8w+j] * lut[code_j]   (the products are exact, so
//                                                                     FFMA == FMUL+FADD bit for bit)
//       P[l] = fma(acc, scale[w >> 1], P[l])                         (fma.rn.f32, e4m3 -> f16 -> f32)
//   * reduction (in-thread, then shfl.bfly 4, 2, 1 over the 8 lanes that hold kw):
//       s_i = P[2i] + P[2i+1];  v = ((s0+s4) + (s2+s6)) + ((s1+s5) + (s3+s7))
//   * out = bf16_rn(v * global[slot, n])            (gate_up)
//     out = bf16_rn((v * global[slot, n]) * topk_w)  (down: MUL_ROUTED_WEIGHT)
// lut[] is FreeToken's _E2M1_VALUES (code 8 = -0.0), i.e. exactly cvt.rn.f16x2.e2m1x2 -> f32.
// This kernel does the same operations in the same order per output; only data movement differs:
// 16-byte weight loads with a register prefetch ring, 4 threads per weight row (thread r owns
// partials 4r..4r+3, i.e. 16-byte chunks c = 4i + r), the activation staged once per CTA in FP32.
//
// C ABI v1 (FreeToken python/freetoken/kernel/cuda_moe_decode.py): qwen_moe_decode_abi_version()
// == 1, qwen_moe_decode_source_sha256() (compiled in), qwen_moe_decode_gemv(),
// qwen_moe_decode_error(), qwen_moe_decode_variant_count/info().
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define QWEN_MOE_DECODE_ABI_VERSION 1
#ifndef QWEN_MOE_DECODE_SOURCE_SHA256
#error "build with -DQWEN_MOE_DECODE_SOURCE_SHA256=\"<sha256 of this file>\" (see build.sh)"
#endif

namespace qwen_moe {

struct Params {
  const __nv_bfloat16* a;   // activations; row = token (gate_up) or route (down)
  long long a_stride;       // elements
  const uint8_t* packed;    // [S, N, K/2] e2m1 pairs (k = 2*byte + {0: low nibble, 1: high})
  long long s_pe, s_pn;     // bytes
  const uint8_t* scale;     // [S, N, K/16] e4m3
  long long s_se, s_sn;     // bytes
  const float* glob;        // [S, N] fp32
  long long s_ge, s_gn;     // elements
  __nv_bfloat16* c;         // [M, TOP_K, N]
  long long s_cm, s_ck, s_cn;
  const float* topk_w;      // [M, TOP_K]
  long long s_wm, s_wk;
  const int* topk_ids;      // [M, TOP_K] -> slot
  long long s_im, s_ik;
  int top_k, N, K;
  int a_is_route, mul_w;
};

__device__ __forceinline__ uint4 ldg_v4_stream(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldg_u16(const void* p) {
  unsigned short v;
  asm("ld.global.nc.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// two e2m1 codes (one byte; low nibble -> .x) -> exact f32 (the Triton LUT values incl. -0.0)
__device__ __forceinline__ float2 e2m1x2_to_f32(uint32_t byte) {
  uint32_t h2;
  asm("{\n .reg .b8 t;\n cvt.u8.u32 t, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}\n" : "=r"(h2) : "r"(byte));
  float lo, hi;
  asm("{\n .reg .f16 l, h;\n mov.b32 {l, h}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, h;\n}\n"
      : "=f"(lo), "=f"(hi)
      : "r"(h2));
  return make_float2(lo, hi);
}
// two e4m3 scale bytes (low byte -> .x) -> exact f32 via f16, as the Triton kernel converts them
__device__ __forceinline__ float2 e4m3x2_to_f32(uint32_t two) {
  uint32_t h2;
  asm("{\n .reg .b16 t;\n cvt.u16.u32 t, %1;\n cvt.rn.f16x2.e4m3x2 %0, t;\n}\n" : "=r"(h2) : "r"(two));
  float lo, hi;
  asm("{\n .reg .f16 l, h;\n mov.b32 {l, h}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, h;\n}\n"
      : "=f"(lo), "=f"(hi)
      : "r"(h2));
  return make_float2(lo, hi);
}

// acc = +0; acc += a[8w + j] * code_j (j = 0..7) -- products exact; FFMA chain from +0 as Triton's SASS
__device__ __forceinline__ float word_dot(uint32_t word, const float4 a0, const float4 a1) {
  const float2 c01 = e2m1x2_to_f32(word & 0xFF);
  const float2 c23 = e2m1x2_to_f32((word >> 8) & 0xFF);
  const float2 c45 = e2m1x2_to_f32((word >> 16) & 0xFF);
  const float2 c67 = e2m1x2_to_f32(word >> 24);
  float acc = fma_rn(a0.x, c01.x, 0.0f);
  acc = fma_rn(a0.y, c01.y, acc);
  acc = fma_rn(a0.z, c23.x, acc);
  acc = fma_rn(a0.w, c23.y, acc);
  acc = fma_rn(a1.x, c45.x, acc);
  acc = fma_rn(a1.y, c45.y, acc);
  acc = fma_rn(a1.z, c67.x, acc);
  acc = fma_rn(a1.w, c67.y, acc);
  return acc;
}

// One CTA: one route x ROWS weight rows (ROWS = TPB / 4 * RPT). Thread (team, r): rows
// n0 + team + i * (TPB / 4), i < RPT; 16-byte chunks c = 4 * it + r (words 4c..4c+3 -> partials
// 4r..4r+3), it = 0..ITERS-1 with ITERS = K / 128.
template <int ITERS, int RPT, int TPB, int PF>
__global__ void __launch_bounds__(TPB, 1)
moe_gemv_kernel(const Params p) {
  extern __shared__ __align__(16) float a_s[];  // [K] fp32
  constexpr int TEAMS = TPB / 4;
  constexpr int ROWS = TEAMS * RPT;
  const int K = p.K;
  const int route = blockIdx.x;
  const int token = route / p.top_k, rk = route - token * p.top_k;
  const long long slot = p.topk_ids[token * p.s_im + rk * p.s_ik];
  const int team = threadIdx.x >> 2, r = threadIdx.x & 3;
  const int n0 = blockIdx.y * ROWS + team;

  const uint8_t* wrow[RPT];
  const uint8_t* srow[RPT];
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    const long long n = n0 + i * TEAMS;
    wrow[i] = p.packed + slot * p.s_pe + n * p.s_pn + r * 16;
    srow[i] = p.scale + slot * p.s_se + n * p.s_sn + r * 2;
  }
  // weights first (independent of the activation), then stage the activation
  static_assert(ITERS % PF == 0, "the prefetch ring must divide the K iterations");
  uint4 wv[PF][RPT];
  uint32_t sv[PF][RPT];
#pragma unroll
  for (int it = 0; it < PF; ++it)
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
      wv[it][i] = ldg_v4_stream(wrow[i] + it * 64);
      sv[it][i] = ldg_u16(srow[i] + it * 8);
    }
  {
    const __nv_bfloat16* arow = p.a + (long long)(p.a_is_route ? route : token) * p.a_stride;
    for (int k = threadIdx.x * 8; k < K; k += TPB * 8) {
      const uint4 v = *reinterpret_cast<const uint4*>(arow + k);
      const uint32_t u[4] = {v.x, v.y, v.z, v.w};
      float4 f0, f1;
      f0.x = __uint_as_float(u[0] << 16); f0.y = __uint_as_float(u[0] & 0xFFFF0000u);
      f0.z = __uint_as_float(u[1] << 16); f0.w = __uint_as_float(u[1] & 0xFFFF0000u);
      f1.x = __uint_as_float(u[2] << 16); f1.y = __uint_as_float(u[2] & 0xFFFF0000u);
      f1.z = __uint_as_float(u[3] << 16); f1.w = __uint_as_float(u[3] & 0xFFFF0000u);
      *reinterpret_cast<float4*>(a_s + k) = f0;
      *reinterpret_cast<float4*>(a_s + k + 4) = f1;
    }
  }
  __syncthreads();

  float P[RPT][4];
#pragma unroll
  for (int i = 0; i < RPT; ++i)
#pragma unroll
    for (int q = 0; q < 4; ++q) P[i][q] = 0.0f;

  // ring of PF chunks in flight; ITERS % PF == 0, so every slot index is a compile-time constant
  for (int base = 0; base < ITERS; base += PF) {
#pragma unroll
    for (int s = 0; s < PF; ++s) {
      const int it = base + s;
      uint4 w[RPT];
      float2 scl[RPT];
#pragma unroll
      for (int i = 0; i < RPT; ++i) {
        w[i] = wv[s][i];
        scl[i] = e4m3x2_to_f32(sv[s][i]);  // blocks 2c (words 4c, 4c+1), 2c+1 (words 4c+2, 4c+3)
      }
      if (it + PF < ITERS) {
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
          wv[s][i] = ldg_v4_stream(wrow[i] + (it + PF) * 64);
          sv[s][i] = ldg_u16(srow[i] + (it + PF) * 8);
        }
      }
      const int c = 4 * it + r;  // chunk: words 4c..4c+3, k = 32c..32c+31
      const float4* ap = reinterpret_cast<const float4*>(a_s + 32 * c);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 a0 = ap[2 * q], a1 = ap[2 * q + 1];
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
          const uint32_t word = q == 0 ? w[i].x : q == 1 ? w[i].y : q == 2 ? w[i].z : w[i].w;
          P[i][q] = fma_rn(word_dot(word, a0, a1), q < 2 ? scl[i].x : scl[i].y, P[i][q]);
        }
      }
    }
  }

  float wgt = 1.0f;
  if (p.mul_w) wgt = p.topk_w[token * p.s_wm + rk * p.s_wk];
#pragma unroll
  for (int i = 0; i < RPT; ++i) {
    // Triton's reduction over kw: s_i = P[2i] + P[2i+1]; ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))
    const float s0 = add_rn(P[i][0], P[i][1]);   // s_{2r}
    const float s1 = add_rn(P[i][2], P[i][3]);   // s_{2r+1}
    const float t0 = add_rn(s0, __shfl_xor_sync(0xffffffffu, s0, 2));
    const float t1 = add_rn(s1, __shfl_xor_sync(0xffffffffu, s1, 2));
    const float u0 = add_rn(t0, __shfl_xor_sync(0xffffffffu, t0, 1));
    const float u1 = add_rn(t1, __shfl_xor_sync(0xffffffffu, t1, 1));
    const float v = add_rn(u0, u1);
    if (r == 0) {
      const long long n = n0 + i * TEAMS;
      float o = mul_rn(v, p.glob[slot * p.s_ge + n * p.s_gn]);
      if (p.mul_w) o = mul_rn(o, wgt);
      p.c[token * p.s_cm + rk * p.s_ck + n * p.s_cn] = __float2bfloat16_rn(o);
    }
  }
}

struct Variant {
  int rpt, tpb, pf;
};
// Same arithmetic for every variant; CTA shape and prefetch depth differ.
static constexpr Variant kVariants[] = {
    {2, 128, 4},   // 0
    {2, 128, 2},   // 1
    {4, 128, 2},   // 2
    {2, 256, 4},   // 3
    {4, 64, 2},    // 4
    {1, 128, 4},   // 5
};
static constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

template <int ITERS, int RPT, int TPB, int PF>
static int launch_t(const Params& p, cudaStream_t s, int routes) {
  constexpr int ROWS = TPB / 4 * RPT;
  if (p.N % ROWS) return int(cudaErrorInvalidValue);
  const dim3 grid(routes, p.N / ROWS);
  const int smem = p.K * 4;
  moe_gemv_kernel<ITERS, RPT, TPB, (PF < ITERS ? PF : ITERS)><<<grid, TPB, smem, s>>>(p);
  return int(cudaGetLastError());
}

template <int ITERS>
static int launch_iters(int v, const Params& p, cudaStream_t s, int routes) {
  switch (v) {
    case 0: return launch_t<ITERS, 2, 128, 4>(p, s, routes);
    case 1: return launch_t<ITERS, 2, 128, 2>(p, s, routes);
    case 2: return launch_t<ITERS, 4, 128, 2>(p, s, routes);
    case 3: return launch_t<ITERS, 2, 256, 4>(p, s, routes);
    case 4: return launch_t<ITERS, 4, 64, 2>(p, s, routes);
    case 5: return launch_t<ITERS, 1, 128, 4>(p, s, routes);
    default: return int(cudaErrorInvalidValue);
  }
}

}  // namespace qwen_moe

using namespace qwen_moe;

extern "C" int qwen_moe_decode_abi_version() { return QWEN_MOE_DECODE_ABI_VERSION; }
extern "C" const char* qwen_moe_decode_source_sha256() { return QWEN_MOE_DECODE_SOURCE_SHA256; }
extern "C" int qwen_moe_decode_variant_count() { return kNumVariants; }
extern "C" int qwen_moe_decode_variant_info(int v, int* rpt, int* tpb, int* pf) {
  if (v < 0 || v >= kNumVariants || !rpt || !tpb || !pf) return int(cudaErrorInvalidValue);
  *rpt = kVariants[v].rpt;
  *tpb = kVariants[v].tpb;
  *pf = kVariants[v].pf;
  return 0;
}
extern "C" const char* qwen_moe_decode_error(int code) { return cudaGetErrorString(cudaError_t(code)); }

// One decode GEMM of _fused_experts_decode_nvfp4 (same operands and strides as the Triton launch;
// the packed bank is addressed in bytes). K must be 512 or 2048 (ITERS 4 / 16). Returns 0 or a
// cudaError_t code; on non-zero nothing was launched.
extern "C" int qwen_moe_decode_gemv(const void* a, long long a_stride, const void* packed, long long s_pe,
                                    long long s_pn, const void* scale, long long s_se, long long s_sn,
                                    const void* glob, long long s_ge, long long s_gn, void* c, long long s_cm,
                                    long long s_ck, long long s_cn, const void* topk_w, long long s_wm,
                                    long long s_wk, const void* topk_ids, long long s_im, long long s_ik,
                                    long long M, long long top_k, long long N, long long K, int a_is_route,
                                    int mul_w, int variant, void* stream) {
  if (!a || !packed || !scale || !glob || !c || !topk_ids || (mul_w && !topk_w)) return int(cudaErrorInvalidValue);
  if (M < 1 || M > 64 || top_k < 1 || top_k > 64 || M * top_k > 65535 || N < 1 || N > (1 << 20))
    return int(cudaErrorInvalidValue);
  if (K != 512 && K != 2048) return int(cudaErrorInvalidValue);
  if (a_stride < K || a_stride % 8 || (reinterpret_cast<uintptr_t>(a) % 16)) return int(cudaErrorInvalidValue);
  if (s_pn < K / 2 || s_pn % 16 || s_pe % 16 || reinterpret_cast<uintptr_t>(packed) % 16) return int(cudaErrorInvalidValue);
  if (s_sn < K / 16 || s_sn % 2 || s_se % 2 || reinterpret_cast<uintptr_t>(scale) % 2) return int(cudaErrorInvalidValue);
  if (reinterpret_cast<uintptr_t>(glob) % 4 || reinterpret_cast<uintptr_t>(c) % 2 || reinterpret_cast<uintptr_t>(topk_ids) % 4)
    return int(cudaErrorInvalidValue);
  if (variant < 0 || variant >= kNumVariants) return int(cudaErrorInvalidValue);
  Params p{};
  p.a = static_cast<const __nv_bfloat16*>(a);
  p.a_stride = a_stride;
  p.packed = static_cast<const uint8_t*>(packed);
  p.s_pe = s_pe;
  p.s_pn = s_pn;
  p.scale = static_cast<const uint8_t*>(scale);
  p.s_se = s_se;
  p.s_sn = s_sn;
  p.glob = static_cast<const float*>(glob);
  p.s_ge = s_ge;
  p.s_gn = s_gn;
  p.c = static_cast<__nv_bfloat16*>(c);
  p.s_cm = s_cm;
  p.s_ck = s_ck;
  p.s_cn = s_cn;
  p.topk_w = static_cast<const float*>(topk_w);
  p.s_wm = s_wm;
  p.s_wk = s_wk;
  p.topk_ids = static_cast<const int*>(topk_ids);
  p.s_im = s_im;
  p.s_ik = s_ik;
  p.top_k = int(top_k);
  p.N = int(N);
  p.K = int(K);
  p.a_is_route = a_is_route ? 1 : 0;
  p.mul_w = mul_w ? 1 : 0;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int routes = int(M * top_k);
  return K == 2048 ? launch_iters<16>(variant, p, s, routes) : launch_iters<4>(variant, p, s, routes);
}
