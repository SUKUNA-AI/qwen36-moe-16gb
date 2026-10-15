// qwen_oproj_gemv.cu -- bit-exact replacement of the tensor-wise FP8 (E4M3) GEMM of the decode
// out/o-projection (GDN out_proj, attention o_proj: N = 2048, K = 4096, 1 <= M <= 8) on SM120.
//
// Target (torch 2.11 cu130, cuBLASLt 13): torch._scaled_mm(qa [M,K] e4m3, W.t(), scale_a = 0-d fp32,
// scale_b = 0-d fp32, out bf16) runs, for every M in 1..8, one memset of 384 B (split-K semaphores)
// and the cuBLASLt kernel
//   sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warpsize2x2x1_tensor16x8x32
// with grid (1, 32, 3): split-K = 3 reduced in-kernel, serially, under a semaphore. Identified bit
// for bit (kernels/decode-oproj/scripts: probe_splitk.py, identify_splitk.py; SASS of the xmma
// kernel) on random e4m3 codes, gaussian codes and real out_proj/o_proj weights, M 1..8:
//   * K is cut in 64-byte k-tiles; split z owns tiles t with t % 3 == z (round robin, NOT
//     contiguous ranges): split 0 gets 22 tiles, splits 1 and 2 get 21;
//   * each split is one chain of QMMA.16832.F32.E4M3.E4M3 from +0 over its tiles in ascending order,
//     two k32 MMAs per tile (bytes 0..31, then 32..63) -- the same per-instruction K sets and chain
//     discipline as the CUTLASS row-wise kernel that libqwen_fp8_gemv reproduces;
//   * the partial sums are combined as FADD(FADD(p0, p2), p1);
//   * epilogue: alpha = FMUL(scale_a, scale_b); out = F2FP.BF16.RN(FFMA(acc, alpha, +0)).
// This kernel computes exactly that per output and differs only in data movement: one launch, no
// workspace, no memset, no semaphore; each of the 3 warps of a strip streams its own split's tiles
// through a cp.async ring and the strip's partials meet in shared memory.
//
// C ABI v1 (FreeToken python/freetoken/kernel/cuda_oproj_gemv.py): qwen_oproj_abi_version() == 1,
// qwen_oproj_max_m() == 8, qwen_oproj_source_sha256() (compiled in), qwen_oproj_init(),
// qwen_oproj_tensorwise(), qwen_oproj_error(), qwen_oproj_variant_count/info().
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define QWEN_OPROJ_ABI_VERSION 1
#define QWEN_OPROJ_MAX_M 8
#define QWEN_OPROJ_SPLITS 3
#ifndef QWEN_OPROJ_SOURCE_SHA256
#error "build with -DQWEN_OPROJ_SOURCE_SHA256=\"<sha256 of this file>\" (see build.sh)"
#endif

namespace qwen_oproj {

constexpr int kSplits = QWEN_OPROJ_SPLITS;
constexpr int kTileBytes = 64;                    // cuBLAS k-tile (tilesize32x64x64)
constexpr int kStageBytes = 16 * kTileBytes;      // 16 weight rows x one k-tile per warp

struct Params {
  const uint8_t* x;      // [M, K] e4m3, row stride ldx bytes
  const uint8_t* w;      // [N, K] e4m3, row stride ldw bytes
  const float* sa;       // scale_a (device scalar)
  const float* sb;       // scale_b (device scalar)
  __nv_bfloat16* out;    // [M, N], row stride ldo elements
  int M, N, K;
  long long ldx, ldw, ldo;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
template <int L2PF>
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  if constexpr (L2PF == 256) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
  } else if constexpr (L2PF == 128) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
  }
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void lds128(uint32_t addr, uint32_t (&v)[4]) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ uint4 ldg_nc_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// D = A(16x32, rows = activation rows m) * B(32x8, cols = weight rows n) + C, FP32 accumulate.
// Rows 8..15 of A are zero (M <= 8): a1 = a3 = 0.
__device__ __forceinline__ void qmma(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
// Spelled out in PTX: no contraction / reassociation.
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
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// Activation smem layout ("fragment-permuted", as libqwen_fp8_gemv): per row m, per 64-byte k-tile
// p, lane-slot t (16 B) = { a0(kb=2p), a2(kb=2p), a0(kb=2p+1), a2(kb=2p+1) }, a0(kb) =
// X[m][32kb+4t..+3], a2(kb) = X[m][32kb+16+4t..+3]. Row stride K+64 bytes (conflict-free LDS.128).
__host__ __device__ constexpr int act_row_stride(int K) { return K + 64; }
__host__ __device__ constexpr int act_bytes(int M, int K) { return ((M * act_row_stride(K)) + 127) & ~127; }
// 16-B chunk swizzle inside a 64-B weight row of the ring (conflict-free ldmatrix phases).
__device__ __forceinline__ int swz(int r, int c) { return c ^ ((r >> 1) & 3); }

template <int STRIPS, int STAGES>
struct Cfg {
  static constexpr int kWarps = STRIPS * kSplits;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kRingBytes = STAGES * kStageBytes;  // per warp
  static constexpr int kPartBytes = STRIPS * (kSplits - 1) * 8 * 16 * 4;  // partials of splits 1, 2
  static int smem_bytes(int M, int K) { return act_bytes(M, K) + kPartBytes + kWarps * kRingBytes; }
};

// One CTA = STRIPS strips of 16 weight rows; warp (strip, z) runs the QMMA chain of split z.
template <int STRIPS, int STAGES, int L2PF>
__global__ void __launch_bounds__(STRIPS * kSplits * 32)
oproj_kernel(const Params p) {
  using C = Cfg<STRIPS, STAGES>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int M = p.M, K = p.K;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int strip = warp / kSplits, z = warp - strip * kSplits;
  const int n0 = (blockIdx.x * STRIPS + strip) * 16;
  const bool active = n0 < p.N;
  const int tiles = K / kTileBytes;
  const int ntile = (tiles - z + kSplits - 1) / kSplits;  // tiles z, z+3, ... < tiles
  const int astride = act_row_stride(K);
  float* part = reinterpret_cast<float*>(smem + act_bytes(M, K));  // [STRIPS][2][8][16]
  uint8_t* ring = smem + act_bytes(M, K) + C::kPartBytes + warp * C::kRingBytes;
  const uint32_t ring_u32 = smem_u32(ring);
  const uint8_t* wrows = p.w + (long long)n0 * p.ldw;

  // cp.async of the 16 rows x 64 B of this split's i-th tile into ring slot `slot`.
  auto issue = [&](int i, int slot) {
    const uint8_t* src0 = wrows + (long long)(z + kSplits * i) * kTileBytes;
    const uint32_t dst0 = ring_u32 + slot * kStageBytes;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int idx = j * 32 + lane;
      const int r = idx >> 2, c = idx & 3;
      cp_async16<L2PF>(dst0 + r * kTileBytes + (swz(r, c) << 4), src0 + (long long)r * p.ldw + c * 16);
    }
  };

  if (active) {
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ntile) issue(s, s);
      cp_async_commit();
    }
  }
  {  // stage the activation (whole CTA)
    const int cpr = K >> 4;
    for (int c = threadIdx.x; c < cpr; c += C::kThreads) {
      const int o = c << 4;
      const int kb = o >> 5, h = (o >> 4) & 1;
      const int doff = (kb >> 1) * 64 + (kb & 1) * 8 + h * 4;
      uint4 v[8];
#pragma unroll
      for (int m = 0; m < 8; ++m)
        if (m < M) v[m] = ldg_nc_v4(p.x + (long long)m * p.ldx + o);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        if (m < M) {
          uint8_t* dst = smem + m * astride + doff;
          *reinterpret_cast<uint32_t*>(dst + 0) = v[m].x;
          *reinterpret_cast<uint32_t*>(dst + 16) = v[m].y;
          *reinterpret_cast<uint32_t*>(dst + 32) = v[m].z;
          *reinterpret_cast<uint32_t*>(dst + 48) = v[m].w;
        }
      }
    }
  }
  __syncthreads();

  const int g = lane >> 2, t = lane & 3;
  const bool arow = g < M;
  const uint32_t act_lane = smem_u32(smem) + g * astride + t * 16;
  const int mi = lane >> 3;
  const int lrow = ((mi >> 1) << 3) | (lane & 7);
  const int lhalf = mi & 1;
  const uint32_t lbase = ring_u32 + lrow * kTileBytes;
  float acc0[4] = {0.f, 0.f, 0.f, 0.f};  // weight rows n0 + 0..7
  float acc1[4] = {0.f, 0.f, 0.f, 0.f};  // weight rows n0 + 8..15
  if (active) {
    int slot = 0, wslot = STAGES - 1;
    for (int i = 0; i < ntile; ++i) {
      cp_async_wait<STAGES - 2>();
      __syncwarp();
      if (i + STAGES - 1 < ntile) issue(i + STAGES - 1, wslot);
      cp_async_commit();
      wslot = (wslot + 1 == STAGES) ? 0 : wslot + 1;
      const uint32_t sbuf = lbase + slot * kStageBytes;
      uint32_t a[4] = {0u, 0u, 0u, 0u};
      if (arow) lds128(act_lane + (uint32_t)((z + kSplits * i) * kTileBytes), a);
#pragma unroll
      for (int q = 0; q < 2; ++q) {  // k32 halves of the tile, ascending; D == C
        uint32_t b0, b1, b2, b3;
        ldsm_x4(sbuf + (swz(lrow, (q << 1) | lhalf) << 4), b0, b1, b2, b3);
        qmma(acc0, a[2 * q], a[2 * q + 1], b0, b1);
        qmma(acc1, a[2 * q], a[2 * q + 1], b2, b3);
      }
      slot = (slot + 1 == STAGES) ? 0 : slot + 1;
    }
    cp_async_wait<0>();
  }
  if (active && arow && z != 0) {
    float* pw = part + ((strip * 2 + (z - 1)) * 8 + g) * 16 + 2 * t;
    pw[0] = acc0[0];
    pw[1] = acc0[1];
    pw[8] = acc1[0];
    pw[9] = acc1[1];
  }
  __syncthreads();
  if (!active || z != 0 || !arow) return;
  const float* p1 = part + ((strip * 2 + 0) * 8 + g) * 16 + 2 * t;
  const float* p2 = part + ((strip * 2 + 1) * 8 + g) * 16 + 2 * t;
  const float alpha = mul_rn(*p.sa, *p.sb);
  // cuBLAS serial reduction: FADD(FADD(p0, p2), p1); epilogue FFMA(acc, alpha, +0) -> bf16 RN
  const float r0 = fma_rn(add_rn(add_rn(acc0[0], p2[0]), p1[0]), alpha, 0.0f);
  const float r1 = fma_rn(add_rn(add_rn(acc0[1], p2[1]), p1[1]), alpha, 0.0f);
  const float r8 = fma_rn(add_rn(add_rn(acc1[0], p2[8]), p1[8]), alpha, 0.0f);
  const float r9 = fma_rn(add_rn(add_rn(acc1[1], p2[9]), p1[9]), alpha, 0.0f);
  uint32_t* orow = reinterpret_cast<uint32_t*>(p.out + (long long)g * p.ldo + n0 + 2 * t);
  orow[0] = pack_bf16x2_rn(r0, r1);  // columns n, n+1
  orow[4] = pack_bf16x2_rn(r8, r9);  // columns n+8, n+9
}

struct Variant {
  int strips, stages, l2pf;
};
// All variants are the same arithmetic; only the CTA shape and pipeline depth differ.
static constexpr Variant kVariants[] = {
    {1, 8, 128},   // 0: 128 CTAs x 96 threads
    {1, 12, 128},  // 1
    {2, 8, 128},   // 2: 64 CTAs x 192 threads
    {1, 8, 256},   // 3
    {1, 16, 256},  // 4
    {2, 10, 256},  // 5
};
static constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

template <int STRIPS, int STAGES, int L2PF>
static int configure_t() {
  cudaError_t e = cudaFuncSetAttribute(oproj_kernel<STRIPS, STAGES, L2PF>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
  if (e != cudaSuccess) return int(e);
  return int(cudaFuncSetAttribute(oproj_kernel<STRIPS, STAGES, L2PF>,
                                  cudaFuncAttributePreferredSharedMemoryCarveout, int(cudaSharedmemCarveoutMaxShared)));
}
static bool g_configured = false;

template <int STRIPS, int STAGES, int L2PF>
static int launch_t(const Params& p, cudaStream_t stream) {
  using C = Cfg<STRIPS, STAGES>;
  const int smem = C::smem_bytes(p.M, p.K);
  if (smem > 99 * 1024) return int(cudaErrorInvalidValue);
  if (smem > 48 * 1024 && !g_configured) return int(cudaErrorNotReady);  // qwen_oproj_init() first
  const int strips = p.N / 16;
  oproj_kernel<STRIPS, STAGES, L2PF><<<(strips + STRIPS - 1) / STRIPS, C::kThreads, smem, stream>>>(p);
  return int(cudaGetLastError());
}

static int launch_variant(int v, const Params& p, cudaStream_t s) {
  switch (v) {
    case 0: return launch_t<1, 8, 128>(p, s);
    case 1: return launch_t<1, 12, 128>(p, s);
    case 2: return launch_t<2, 8, 128>(p, s);
    case 3: return launch_t<1, 8, 256>(p, s);
    case 4: return launch_t<1, 16, 256>(p, s);
    case 5: return launch_t<2, 10, 256>(p, s);
    default: return int(cudaErrorInvalidValue);
  }
}

static int configure_all() {
  int e = 0;
  if ((e = configure_t<1, 8, 128>())) return e;
  if ((e = configure_t<1, 12, 128>())) return e;
  if ((e = configure_t<2, 8, 128>())) return e;
  if ((e = configure_t<1, 8, 256>())) return e;
  if ((e = configure_t<1, 16, 256>())) return e;
  if ((e = configure_t<2, 10, 256>())) return e;
  g_configured = true;
  return 0;
}

}  // namespace qwen_oproj

using namespace qwen_oproj;

extern "C" int qwen_oproj_abi_version() { return QWEN_OPROJ_ABI_VERSION; }
extern "C" int qwen_oproj_max_m() { return QWEN_OPROJ_MAX_M; }
extern "C" int qwen_oproj_splits() { return QWEN_OPROJ_SPLITS; }
extern "C" const char* qwen_oproj_source_sha256() { return QWEN_OPROJ_SOURCE_SHA256; }
extern "C" int qwen_oproj_variant_count() { return kNumVariants; }
extern "C" int qwen_oproj_variant_info(int v, int* strips, int* stages, int* l2pf) {
  if (v < 0 || v >= kNumVariants || !strips || !stages || !l2pf) return int(cudaErrorInvalidValue);
  *strips = kVariants[v].strips;
  *stages = kVariants[v].stages;
  *l2pf = kVariants[v].l2pf;
  return 0;
}
// Call once per process/device before use (and before any CUDA graph capture).
extern "C" int qwen_oproj_init() { return configure_all(); }
extern "C" const char* qwen_oproj_error(int code) { return cudaGetErrorString(cudaError_t(code)); }

// out[m, n] = bf16_rn(fma(acc[m, n], RN(*sa * *sb), +0)), acc = cuBLASLt's split-K=3 round-robin
// QMMA chains reduced as (p0 + p2) + p1. Returns 0 or a cudaError_t code; on non-zero nothing ran.
extern "C" int qwen_oproj_tensorwise(const void* x, const void* w, const void* sa, const void* sb, void* out,
                                     long long M, long long N, long long K, long long ldx, long long ldw,
                                     long long ldo, int variant, void* stream) {
  if (!x || !w || !sa || !sb || !out) return int(cudaErrorInvalidValue);
  if (M < 1 || M > QWEN_OPROJ_MAX_M || N < 16 || N % 16 || N > (1LL << 30)) return int(cudaErrorInvalidValue);
  if (K < kSplits * kTileBytes || K % kTileBytes || K > (1 << 16)) return int(cudaErrorInvalidValue);
  if (ldx < K || ldw < K || ldo < N || ldx % 16 || ldw % 16 || ldo % 2) return int(cudaErrorInvalidValue);
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16) return int(cudaErrorInvalidValue);
  if (reinterpret_cast<uintptr_t>(out) % 4 || reinterpret_cast<uintptr_t>(sa) % 4 ||
      reinterpret_cast<uintptr_t>(sb) % 4)
    return int(cudaErrorInvalidValue);
  if (variant < 0 || variant >= kNumVariants) return int(cudaErrorInvalidValue);
  Params p{};
  p.x = static_cast<const uint8_t*>(x);
  p.w = static_cast<const uint8_t*>(w);
  p.sa = static_cast<const float*>(sa);
  p.sb = static_cast<const float*>(sb);
  p.out = static_cast<__nv_bfloat16*>(out);
  p.M = int(M);
  p.N = int(N);
  p.K = int(K);
  p.ldx = ldx;
  p.ldw = ldw;
  p.ldo = ldo;
  return launch_variant(variant, p, reinterpret_cast<cudaStream_t>(stream));
}
