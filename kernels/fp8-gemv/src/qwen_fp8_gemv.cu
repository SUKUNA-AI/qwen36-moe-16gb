// qwen_fp8_gemv.cu -- bit-exact skinny FP8 (E4M3) W8A8 GEMM for M <= 8 on SM120 (RTX 50xx).
//
// Target (the current path, see cutlass-mapping.md): torch 2.11 torch._scaled_mm with
//   scale_a fp32 [M,1], scale_b fp32 [1,N], out bf16, no bias  ("row-wise" scaling)
// dispatches, on sm_120a, the CUTLASS 3.x kernel
//   GemmUniversal< CollectiveMma< MainloopSm120TmaWarpSpecialized<2,2,1x1x1,
//                  KernelTmaWarpSpecializedCooperativeSm120<2>>, tile 128x128x128,
//                  A=e4m3 K-major (activation), B=e4m3 K-major (weight),
//                  TiledMMA<SM120_16x8x32_TN<e4m3,e4m3,f32>, warps 4x2x1> >,
//                  epilogue EVT  bf16_rn( plus(bias[n], mul(w_scale[n], mul(x_scale[m], acc))) ) >
// whose SASS is:  QMMA.16832.F32.E4M3.E4M3 chains (accumulators zeroed by CS2R, D==C, k-blocks
// strictly ascending, no split-K / no FADD promotion) and the epilogue
//   FMUL t, acc, x_scale ; FFMA r, w_scale, t, bias(+0 when null) ; F2FP.BF16.F32.PACK_AB.
//
// This kernel reproduces exactly that arithmetic per output element:
//   * the same MMA instruction (mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ->
//     QMMA.16832.F32.E4M3.E4M3), with the same operand roles (activation rows in A, weight
//     rows in B) and the same element positions inside each MMA (ldmatrix of K-major tiles);
//   * one accumulator chain per output, starting from +0.0, k-blocks of 32 in ascending order;
//   * the same epilogue: mul.rn.f32(acc, xs) -> fma.rn.f32(ws, t, +0.0) -> cvt.rn.bf16x2.f32.
// Only the tiling / data movement differ: weights are streamed once from DRAM through a per-warp
// cp.async ring (16 weight rows per warp, many CTAs), with no padding of M to 128 rows.
// Split-K is deliberately NOT used: it would change the FP32 summation order.
//
// ABI v2 (work/wf-fp8gemv, FreeToken freetoken/kernel/cuda_fp8_gemv.py): adds constant-returning
// probe entry points (qwen_fp8_gemv_abi_version / _max_m / _source_sha256; no CUDA call, safe in a
// CPU-only probe process) so a loader refuses a stale or foreign build before any launch. The
// arithmetic, the kernels and qwen_fp8_gemv_rowwise/_init/_error are unchanged from v1.
// QWEN_FP8_GEMV_PRODUCTION compiles out the analysis-only entry points (tensor-wise split-K
// reproduction, raw FP32 partials, bandwidth probe), which the production loader never binds.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define QWEN_FP8_GEMV_ABI_VERSION 2
#define QWEN_FP8_GEMV_MAX_M 8
#ifndef QWEN_FP8_GEMV_SOURCE_SHA256
#error "build with -DQWEN_FP8_GEMV_SOURCE_SHA256=\"<sha256 of this file>\" (see build.sh)"
#endif

namespace qwen_fp8 {

struct Params {
  const uint8_t* x;       // [M, K] e4m3, row stride ldx bytes
  const uint8_t* w;       // [N, K] e4m3, row stride ldw bytes
  const float* x_scale;   // x_scale[m * xs_stride]
  const float* w_scale;   // [N]
  __nv_bfloat16* out;     // [M, N], row stride ldo elements
  float* out_f32;         // debug/analysis only (EPI == 1): raw FP32 accumulators [M, N]
  int M, N, K;
  long long ldx, ldw, ldo;
  int xs_stride;
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
// Rows 8..15 of A are always zero here (M <= 8): a1 = a3 = 0.
__device__ __forceinline__ void qmma(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// Epilogue arithmetic spelled out in PTX so that no contraction / reassociation can happen.
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
// CUTLASS: FMUL t = acc * x_scale[m];  FFMA r = w_scale[n] * t + bias[n]  (bias == +0 when null)
__device__ __forceinline__ float epilogue_value(float acc, float xs, float ws) {
  return fma_rn(ws, mul_rn(acc, xs), 0.0f);
}

// Activation smem layout ("fragment-permuted"): per row m, per pair p of k-blocks (64 K bytes),
// lane-slot t (16 B) = { a0(kb=2p), a2(kb=2p), a0(kb=2p+1), a2(kb=2p+1) } where
// a0(kb) = X[m][32kb+4t .. +3] and a2(kb) = X[m][32kb+16+4t .. +3]  (m16n8k32 A-fragment words).
// Row stride K+64 bytes keeps the LDS.128 of two rows in one quarter-warp conflict-free.
__host__ __device__ constexpr int act_row_stride(int K) { return K + 64; }
__host__ __device__ constexpr int act_bytes(int M, int K) { return ((M * act_row_stride(K)) + 127) & ~127; }

// 16-B chunk swizzle inside a weight row of the per-warp ring: conflict-free ldmatrix phases
// (8 rows x one logical chunk) and cp.async writes.  128-B and 256-B rows: chunk ^ (row & 7);
// 64-B rows (analysis-only KCB == 2): chunk ^ ((row >> 1) & 3).
template <int KCB>
__device__ __forceinline__ int swz(int r, int c) {
  if constexpr (KCB == 2) return c ^ ((r >> 1) & 3);
  else return c ^ (r & 7);
}

template <int WARPS, int STAGES, int KCB, int L2PF>
struct Cfg {
  static constexpr int kThreads = WARPS * 32;
  static constexpr int kRowBytes = KCB * 32;           // K bytes per weight row per stage
  static constexpr int kStageBytes = 16 * kRowBytes;   // 16 weight rows per warp
  static constexpr int kWarpBytes = STAGES * kStageBytes;
  static constexpr int kChunksPerRow = kRowBytes / 16; // 16-B chunks per row per stage
  static_assert(KCB == 2 || KCB == 4 || KCB == 8, "KCB must be 2, 4 or 8 k-blocks (64/128/256 B)");
  static_assert(STAGES >= 2, "need at least double buffering");
  static int smem_bytes(int M, int K) { return act_bytes(M, K) + WARPS * kWarpBytes; }
};

// EPI == 0: CUTLASS row-wise epilogue -> bf16.  EPI == 1: raw FP32 accumulators (analysis only).
template <int WARPS, int STAGES, int KCB, int L2PF, int EPI = 0>
__global__ void __launch_bounds__(WARPS * 32)
gemv_rowwise_kernel(const Params p) {
  using C = Cfg<WARPS, STAGES, KCB, L2PF>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int M = p.M, K = p.K;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * WARPS + warp) * 16;
  const bool active = n0 < p.N;
  const int nchunks = K / C::kRowBytes;
  const int astride = act_row_stride(K);
  uint8_t* wsm = smem + act_bytes(M, K) + warp * C::kWarpBytes;
  const uint32_t wsm_u32 = smem_u32(wsm);
  const uint8_t* wrows = p.w + (long long)n0 * p.ldw;

  // cp.async of one 16-row x kRowBytes weight chunk into ring slot `slot` (swizzled 16-B chunks).
  auto issue = [&](int chunk, int slot) {
    const uint8_t* src0 = wrows + (long long)chunk * C::kRowBytes;
    const uint32_t dst0 = wsm_u32 + slot * C::kStageBytes;
#pragma unroll
    for (int j = 0; j < KCB; ++j) {
      const int idx = j * 32 + lane;
      const int r = idx / C::kChunksPerRow;
      const int c = idx % C::kChunksPerRow;
      cp_async16<L2PF>(dst0 + r * C::kRowBytes + (swz<KCB>(r, c) << 4), src0 + (long long)r * p.ldw + c * 16);
    }
  };

  // 1) Start streaming weights before anything else: they do not depend on the activation.
  if (active) {
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < nchunks) issue(s, s);
      cp_async_commit();
    }
  }

  // 2) Stage the (tiny) activation into the fragment-permuted smem layout, whole CTA.
  {
    const int cpr = K >> 4;  // 16-B chunks per activation row
    for (int c = threadIdx.x; c < cpr; c += C::kThreads) {
      const int o = c << 4;
      const int kb = o >> 5, h = (o >> 4) & 1;
      const int doff = (kb >> 1) * 64 + (kb & 1) * 8 + h * 4;
      uint4 v[8];
#pragma unroll
      for (int m = 0; m < 8; ++m)  // all loads first (up to 8 in flight), then the stores
        if (m < M) v[m] = ldg_nc_v4(p.x + (long long)m * p.ldx + o);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        if (m < M) {
          uint8_t* dst = smem + m * astride + doff;
          *reinterpret_cast<uint32_t*>(dst + 0) = v[m].x;  // word t -> lane-slot t
          *reinterpret_cast<uint32_t*>(dst + 16) = v[m].y;
          *reinterpret_cast<uint32_t*>(dst + 32) = v[m].z;
          *reinterpret_cast<uint32_t*>(dst + 48) = v[m].w;
        }
      }
    }
  }
  __syncthreads();
  if (!active) return;

  const int g = lane >> 2;  // MMA groupID: activation row m (A row), weight col n inside n8 tile
  const int t = lane & 3;   // thread in group
  const bool arow = g < M;
  const uint32_t act_lane = smem_u32(smem) + g * astride + t * 16;
  // ldmatrix.x4 lane address: matrix mi = lane>>3 -> (tile = mi>>1, k-half = mi&1), row lane&7.
  const int mi = lane >> 3;
  const int lrow = ((mi >> 1) << 3) | (lane & 7);
  const int lhalf = mi & 1;
  const uint32_t lbase = wsm_u32 + lrow * C::kRowBytes;

  float acc0[4] = {0.f, 0.f, 0.f, 0.f};  // weight rows n0 + 0..7
  float acc1[4] = {0.f, 0.f, 0.f, 0.f};  // weight rows n0 + 8..15
  int slot = 0, wslot = STAGES - 1;
  for (int ch = 0; ch < nchunks; ++ch) {
    cp_async_wait<STAGES - 2>();
    __syncwarp();
    if (ch + STAGES - 1 < nchunks) issue(ch + STAGES - 1, wslot);
    cp_async_commit();
    wslot = (wslot + 1 == STAGES) ? 0 : wslot + 1;
    const uint32_t sb = lbase + slot * C::kStageBytes;
#pragma unroll
    for (int kp = 0; kp < KCB / 2; ++kp) {
      uint32_t a[4] = {0u, 0u, 0u, 0u};
      if (arow) lds128(act_lane + (uint32_t)((ch * (KCB / 2) + kp) * 64), a);
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int kb = kp * 2 + q;
        uint32_t b0, b1, b2, b3;
        ldsm_x4(sb + (swz<KCB>(lrow, (kb << 1) | lhalf) << 4), b0, b1, b2, b3);
        qmma(acc0, a[2 * q], a[2 * q + 1], b0, b1);  // k-block ascending, D == C
        qmma(acc1, a[2 * q], a[2 * q + 1], b2, b3);
      }
    }
    slot = (slot + 1 == STAGES) ? 0 : slot + 1;
  }
  cp_async_wait<0>();

  if constexpr (EPI == 1) {
    if (arow) {
      const int n = n0 + 2 * t;
      float* orow = p.out_f32 + (long long)g * p.ldo + n;
      orow[0] = acc0[0];
      orow[1] = acc0[1];
      orow[8] = acc1[0];
      orow[9] = acc1[1];
    }
    return;
  }
  if (arow) {
    const float xs = p.x_scale[(long long)g * p.xs_stride];
    const int n = n0 + 2 * t;
    const float w00 = p.w_scale[n], w01 = p.w_scale[n + 1];
    const float w10 = p.w_scale[n + 8], w11 = p.w_scale[n + 9];
    const uint32_t o0 = pack_bf16x2_rn(epilogue_value(acc0[0], xs, w00), epilogue_value(acc0[1], xs, w01));
    const uint32_t o1 = pack_bf16x2_rn(epilogue_value(acc1[0], xs, w10), epilogue_value(acc1[1], xs, w11));
    uint32_t* orow = reinterpret_cast<uint32_t*>(p.out + (long long)g * p.ldo + n);
    orow[0] = o0;  // columns n, n+1
    orow[4] = o1;  // columns n+8, n+9
  }
}

#ifndef QWEN_FP8_GEMV_PRODUCTION
// ---------------------------------------------------------------------------------------------
// Tensor-wise (scalar scales) split-K variant: reproduces a cuBLASLt split-K GEMM + separate
// reduction *if* (and only if) its K partition and reduction order are known exactly (see
// analyze_tensorwise.py).  Each slice is an independent QMMA chain (computed by its own warp, in
// parallel), partial FP32 sums are combined in the identified order, then scaled and rounded.
struct SplitParams {
  const uint8_t* x;
  const uint8_t* w;
  const float* sa;     // activation scale (scalar)
  const float* sb;     // weight scale (scalar)
  __nv_bfloat16* out;
  int M, N, K;
  long long ldx, ldw, ldo;
  int splits;
  int bound[9];        // byte offsets of the K slices; bound[0] = 0, bound[splits] = K, multiples of 64
  int combine;         // 0: ((+0 + p0) + p1) + ...   1: (p0 + p1) + ...   2: +0 + p0*a + p1*a + ...
  int scale;           // 0: s * RN(sa*sb)   1: RN(RN(s*sa)*sb)   2: RN(RN(s*sb)*sa)
};

template <int SPLITS, int STAGES, int L2PF>
__global__ void __launch_bounds__(SPLITS * 32)
gemv_splitk_kernel(const SplitParams p) {
  constexpr int KCB = 2, kRowBytes = 64, kStageBytes = 16 * kRowBytes, kChunksPerRow = 4;
  extern __shared__ __align__(128) uint8_t smem[];
  const int M = p.M, K = p.K;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * 16;
  const int astride = act_row_stride(K);
  float* part = reinterpret_cast<float*>(smem + act_bytes(M, K));            // [SPLITS][8][16]
  uint8_t* wsm = smem + act_bytes(M, K) + SPLITS * 8 * 16 * 4 + warp * STAGES * kStageBytes;
  const uint32_t wsm_u32 = smem_u32(wsm);
  const bool active = warp < p.splits;
  int kbeg = 0, kend = 0;  // select without dynamic indexing of the param array (no local copy)
#pragma unroll
  for (int i = 0; i < SPLITS; ++i) {
    if (i == warp) {
      kbeg = p.bound[i];
      kend = p.bound[i + 1];
    }
  }
  const int nchunks = active ? (kend - kbeg) / kRowBytes : 0;
  const uint8_t* wrows = p.w + (long long)n0 * p.ldw + kbeg;
  auto issue = [&](int chunk, int slot) {
    const uint8_t* src0 = wrows + (long long)chunk * kRowBytes;
    const uint32_t dst0 = wsm_u32 + slot * kStageBytes;
#pragma unroll
    for (int j = 0; j < KCB; ++j) {
      const int idx = j * 32 + lane;
      const int r = idx / kChunksPerRow, c = idx % kChunksPerRow;
      cp_async16<L2PF>(dst0 + r * kRowBytes + (swz<KCB>(r, c) << 4), src0 + (long long)r * p.ldw + c * 16);
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nchunks) issue(s, s);
    cp_async_commit();
  }
  {
    const int cpr = K >> 4;
    for (int c = threadIdx.x; c < cpr; c += SPLITS * 32) {
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
  const uint32_t act_lane = smem_u32(smem) + g * astride + t * 16 + kbeg;  // kbeg % 64 == 0
  const int mi = lane >> 3;
  const int lrow = ((mi >> 1) << 3) | (lane & 7);
  const int lhalf = mi & 1;
  const uint32_t lbase = wsm_u32 + lrow * kRowBytes;
  float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
  int slot = 0, wslot = STAGES - 1;
  for (int ch = 0; ch < nchunks; ++ch) {
    cp_async_wait<STAGES - 2>();
    __syncwarp();
    if (ch + STAGES - 1 < nchunks) issue(ch + STAGES - 1, wslot);
    cp_async_commit();
    wslot = (wslot + 1 == STAGES) ? 0 : wslot + 1;
    const uint32_t sb = lbase + slot * kStageBytes;
    uint32_t a[4] = {0u, 0u, 0u, 0u};
    if (arow) lds128(act_lane + (uint32_t)(ch * 64), a);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4(sb + (swz<KCB>(lrow, (q << 1) | lhalf) << 4), b0, b1, b2, b3);
      qmma(acc0, a[2 * q], a[2 * q + 1], b0, b1);
      qmma(acc1, a[2 * q], a[2 * q + 1], b2, b3);
    }
    slot = (slot + 1 == STAGES) ? 0 : slot + 1;
  }
  cp_async_wait<0>();
  if (active && arow) {
    float* pw = part + (warp * 8 + g) * 16 + 2 * t;
    pw[0] = acc0[0];
    pw[1] = acc0[1];
    pw[8] = acc1[0];
    pw[9] = acc1[1];
  }
  __syncthreads();
  if (warp != 0 || !arow) return;
  const float sa = *p.sa, sb = *p.sb;
  const float ab = mul_rn(sa, sb);
  float r[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int col = (e < 2 ? 0 : 8) + 2 * t + (e & 1);
    float s;
    if (p.combine == 1) {
      s = part[(0 * 8 + g) * 16 + col];
      for (int z = 1; z < p.splits; ++z) s = __fadd_rn(s, part[(z * 8 + g) * 16 + col]);
    } else if (p.combine == 2) {
      s = 0.0f;
      for (int z = 0; z < p.splits; ++z) s = __fadd_rn(s, mul_rn(part[(z * 8 + g) * 16 + col], ab));
    } else {
      s = 0.0f;
      for (int z = 0; z < p.splits; ++z) s = __fadd_rn(s, part[(z * 8 + g) * 16 + col]);
    }
    if (p.combine == 2) r[e] = s;
    else if (p.scale == 1) r[e] = mul_rn(mul_rn(s, sa), sb);
    else if (p.scale == 2) r[e] = mul_rn(mul_rn(s, sb), sa);
    else r[e] = mul_rn(s, ab);
  }
  uint32_t* orow = reinterpret_cast<uint32_t*>(p.out + (long long)g * p.ldo + n0 + 2 * t);
  orow[0] = pack_bf16x2_rn(r[0], r[1]);
  orow[4] = pack_bf16x2_rn(r[2], r[3]);
}
#endif  // QWEN_FP8_GEMV_PRODUCTION

struct Variant {
  int warps, stages, kcb, l2pf;
};
// Tuning table (all variants are the same arithmetic; only tiling/pipelining differ). The FreeToken
// loader pins variant 11 (fastest geometric mean of the graph sweep, cuda-fp8-gemv results.json).
static constexpr Variant kVariants[] = {
    {4, 4, 4, 128},  // 0
    {4, 6, 4, 128},  // 1
    {4, 8, 4, 128},  // 2
    {2, 8, 4, 128},  // 3
    {4, 4, 8, 256},  // 4
    {2, 4, 8, 256},  // 5
    {8, 4, 4, 128},  // 6
    {1, 8, 8, 256},  // 7
    {2, 6, 8, 256},  // 8
    {4, 3, 8, 256},  // 9
    {2, 4, 4, 0},    // 10
    {4, 4, 4, 256},  // 11
};
static constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

template <int WARPS, int STAGES, int KCB, int L2PF>
static int configure_t() {
  // Allow the largest opt-in dynamic smem once (outside graph capture) and prefer max carveout.
  cudaError_t e = cudaFuncSetAttribute(gemv_rowwise_kernel<WARPS, STAGES, KCB, L2PF>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
  if (e != cudaSuccess) return int(e);
  e = cudaFuncSetAttribute(gemv_rowwise_kernel<WARPS, STAGES, KCB, L2PF>,
                           cudaFuncAttributePreferredSharedMemoryCarveout, int(cudaSharedmemCarveoutMaxShared));
  return int(e);
}

template <int WARPS, int STAGES, int KCB, int L2PF>
static int launch_t(const Params& p, cudaStream_t stream, bool configured) {
  using C = Cfg<WARPS, STAGES, KCB, L2PF>;
  if (p.K % C::kRowBytes) return int(cudaErrorInvalidValue);
  const int smem = C::smem_bytes(p.M, p.K);
  if (smem > 48 * 1024 && !configured) return int(cudaErrorNotReady);  // call qwen_fp8_gemv_init() first
  const int warps_total = p.N / 16;
  const dim3 grid((warps_total + WARPS - 1) / WARPS);
  gemv_rowwise_kernel<WARPS, STAGES, KCB, L2PF><<<grid, C::kThreads, smem, stream>>>(p);
  return int(cudaGetLastError());
}

static bool g_configured = false;

static int launch_variant(int v, const Params& p, cudaStream_t s) {
  switch (v) {
    case 0: return launch_t<4, 4, 4, 128>(p, s, g_configured);
    case 1: return launch_t<4, 6, 4, 128>(p, s, g_configured);
    case 2: return launch_t<4, 8, 4, 128>(p, s, g_configured);
    case 3: return launch_t<2, 8, 4, 128>(p, s, g_configured);
    case 4: return launch_t<4, 4, 8, 256>(p, s, g_configured);
    case 5: return launch_t<2, 4, 8, 256>(p, s, g_configured);
    case 6: return launch_t<8, 4, 4, 128>(p, s, g_configured);
    case 7: return launch_t<1, 8, 8, 256>(p, s, g_configured);
    case 8: return launch_t<2, 6, 8, 256>(p, s, g_configured);
    case 9: return launch_t<4, 3, 8, 256>(p, s, g_configured);
    case 10: return launch_t<2, 4, 4, 0>(p, s, g_configured);
    case 11: return launch_t<4, 4, 4, 256>(p, s, g_configured);
    default: return int(cudaErrorInvalidValue);
  }
}

static int configure_all() {
  int e = 0;
  if ((e = configure_t<4, 4, 4, 128>())) return e;
  if ((e = configure_t<4, 6, 4, 128>())) return e;
  if ((e = configure_t<4, 8, 4, 128>())) return e;
  if ((e = configure_t<2, 8, 4, 128>())) return e;
  if ((e = configure_t<4, 4, 8, 256>())) return e;
  if ((e = configure_t<2, 4, 8, 256>())) return e;
  if ((e = configure_t<8, 4, 4, 128>())) return e;
  if ((e = configure_t<1, 8, 8, 256>())) return e;
  if ((e = configure_t<2, 6, 8, 256>())) return e;
  if ((e = configure_t<4, 3, 8, 256>())) return e;
  if ((e = configure_t<2, 4, 4, 0>())) return e;
  if ((e = configure_t<4, 4, 4, 256>())) return e;
  g_configured = true;
  return 0;
}

static int smem_for_variant(int v, int M, int K) {
  if (v < 0 || v >= kNumVariants) return -1;
  const Variant& c = kVariants[v];
  return act_bytes(M, K) + c.warps * c.stages * 16 * c.kcb * 32;
}

}  // namespace qwen_fp8

using namespace qwen_fp8;

// ABI probe (v2): constants only -- no CUDA runtime call, so a CPU-only process can dlopen the
// library and check it without initializing the driver.
extern "C" int qwen_fp8_gemv_abi_version() { return QWEN_FP8_GEMV_ABI_VERSION; }
extern "C" int qwen_fp8_gemv_max_m() { return QWEN_FP8_GEMV_MAX_M; }
extern "C" const char* qwen_fp8_gemv_source_sha256() { return QWEN_FP8_GEMV_SOURCE_SHA256; }

extern "C" int qwen_fp8_gemv_variant_count() { return kNumVariants; }

// Call once per process/device before use (and before any CUDA graph capture).
extern "C" int qwen_fp8_gemv_init() { return configure_all(); }

extern "C" int qwen_fp8_gemv_variant_info(int v, int* warps, int* stages, int* kcb, int* l2pf) {
  if (v < 0 || v >= kNumVariants) return int(cudaErrorInvalidValue);
  *warps = kVariants[v].warps;
  *stages = kVariants[v].stages;
  *kcb = kVariants[v].kcb;
  *l2pf = kVariants[v].l2pf;
  return 0;
}

extern "C" int qwen_fp8_gemv_smem_bytes(int v, long long M, long long K) {
  if (M < 1 || M > 8 || K < 1 || K > (1 << 20)) return -1;
  return smem_for_variant(v, int(M), int(K));
}

// out[m, n] = bf16_rn( fma(w_scale[n], x_scale[m*xs_stride] * acc[m, n], +0) ),
// acc[m, n] = sum_k x[m, k] * w[n, k] in the CUTLASS SM120 QMMA chain order.
// Returns 0 or a cudaError_t code; on any non-zero return nothing was launched.
extern "C" int qwen_fp8_gemv_rowwise(const void* x, const void* w, const void* x_scale,
                                     const void* w_scale, void* out, long long M, long long N,
                                     long long K, long long ldx, long long ldw, long long ldo,
                                     long long xs_stride, int variant, void* stream) {
  if (!x || !w || !x_scale || !w_scale || !out) return int(cudaErrorInvalidValue);
  if (M < 1 || M > QWEN_FP8_GEMV_MAX_M || N < 16 || N % 16 || K < 128 || K % 128 || K > (1 << 20))
    return int(cudaErrorInvalidValue);
  if (N > (1LL << 30) || ldx < K || ldw < K || ldo < N || ldx % 16 || ldw % 16 || ldo % 2) return int(cudaErrorInvalidValue);
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16) return int(cudaErrorInvalidValue);
  if (reinterpret_cast<uintptr_t>(out) % 4 || reinterpret_cast<uintptr_t>(x_scale) % 4 ||
      reinterpret_cast<uintptr_t>(w_scale) % 4)
    return int(cudaErrorInvalidValue);
  if (xs_stride < 0 || xs_stride > (1 << 20)) return int(cudaErrorInvalidValue);
  if (variant < 0 || variant >= kNumVariants) return int(cudaErrorInvalidValue);
  const int smem = smem_for_variant(variant, int(M), int(K));
  if (smem < 0 || smem > 99 * 1024) return int(cudaErrorInvalidValue);
  Params p{};
  p.x = static_cast<const uint8_t*>(x);
  p.w = static_cast<const uint8_t*>(w);
  p.x_scale = static_cast<const float*>(x_scale);
  p.w_scale = static_cast<const float*>(w_scale);
  p.out = static_cast<__nv_bfloat16*>(out);
  p.M = int(M);
  p.N = int(N);
  p.K = int(K);
  p.ldx = ldx;
  p.ldw = ldw;
  p.ldo = ldo;
  p.xs_stride = int(xs_stride);
  return launch_variant(variant, p, reinterpret_cast<cudaStream_t>(stream));
}

extern "C" const char* qwen_fp8_gemv_error(int code) { return cudaGetErrorString(cudaError_t(code)); }

#ifndef QWEN_FP8_GEMV_PRODUCTION
namespace qwen_fp8 {
template <int SPLITS>
static int splitk_launch(const SplitParams& p, cudaStream_t s) {
  constexpr int STAGES = 8;
  const int smem = act_bytes(p.M, p.K) + SPLITS * 8 * 16 * 4 + SPLITS * STAGES * 16 * 64;
  static bool configured = false;
  if (!configured) {  // first call happens in the eager warm-up, never under capture (see wrapper)
    cudaError_t e = cudaFuncSetAttribute(gemv_splitk_kernel<SPLITS, STAGES, 128>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    if (e != cudaSuccess) return int(e);
    configured = true;
  }
  if (smem > 99 * 1024) return int(cudaErrorInvalidValue);
  gemv_splitk_kernel<SPLITS, STAGES, 128><<<p.N / 16, SPLITS * 32, smem, s>>>(p);
  return int(cudaGetLastError());
}
}  // namespace qwen_fp8

// Tensor-wise split-K reproduction (see SplitParams). bounds: splits+1 byte offsets.
extern "C" int qwen_fp8_gemv_splitk(const void* x, const void* w, const void* sa, const void* sb, void* out,
                                    long long M, long long N, long long K, long long ldx, long long ldw,
                                    long long ldo, int splits, const int* bounds, int combine, int scale,
                                    void* stream) {
  if (!x || !w || !sa || !sb || !out || !bounds || M < 1 || M > 8 || N < 16 || N % 16 || K < 64 || K % 64 ||
      K > (1 << 20) || ldx < K || ldw < K || ldo < N || ldx % 16 || ldw % 16 || ldo % 2 || splits < 1 ||
      splits > 4 || combine < 0 || combine > 2 || scale < 0 || scale > 2 ||
      ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16) || reinterpret_cast<uintptr_t>(out) % 4)
    return int(cudaErrorInvalidValue);
  SplitParams p{};
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
  p.splits = splits;
  p.combine = combine;
  p.scale = scale;
  if (bounds[0] != 0 || bounds[splits] != K) return int(cudaErrorInvalidValue);
  for (int i = 0; i <= splits; ++i) {
    if (bounds[i] % 64 || (i && bounds[i] <= bounds[i - 1])) return int(cudaErrorInvalidValue);
    p.bound[i] = bounds[i];
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  switch (splits) {
    case 1: return splitk_launch<1>(p, s);
    case 2: return splitk_launch<2>(p, s);
    case 3: return splitk_launch<3>(p, s);
    default: return splitk_launch<4>(p, s);
  }
}

// Analysis only: raw FP32 accumulators of the same QMMA chain over x[:, 0:K] . w[:, 0:K]
// (caller offsets x and w to a K sub-range); K % 64 == 0. Used to identify how other kernels
// (cuBLASLt split-K) partition K -- never on the production path.
extern "C" int qwen_fp8_gemm_partial_f32(const void* x, const void* w, void* out_f32, long long M,
                                         long long N, long long K, long long ldx, long long ldw,
                                         long long ldo, void* stream) {
  if (!x || !w || !out_f32 || M < 1 || M > 8 || N < 16 || N % 16 || K < 64 || K % 64 || ldx < K ||
      ldw < K || ldo < N || ldx % 16 || ldw % 16 || ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16))
    return int(cudaErrorInvalidValue);
  Params p{};
  p.x = static_cast<const uint8_t*>(x);
  p.w = static_cast<const uint8_t*>(w);
  p.out_f32 = static_cast<float*>(out_f32);
  p.M = int(M);
  p.N = int(N);
  p.K = int(K);
  p.ldx = ldx;
  p.ldw = ldw;
  p.ldo = ldo;
  using C = Cfg<4, 4, 2, 128>;
  const int smem = C::smem_bytes(p.M, p.K);
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(gemv_rowwise_kernel<4, 4, 2, 128, 1>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    if (e != cudaSuccess) return int(e);
  }
  const dim3 grid((p.N / 16 + 3) / 4);
  gemv_rowwise_kernel<4, 4, 2, 128, 1><<<grid, 128, smem, reinterpret_cast<cudaStream_t>(stream)>>>(p);
  return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Roofline helper (benchmark only): stream-read `bytes` (multiple of 16) once with 16-B
// non-coherent loads, XOR-fold, write one word per CTA so the loads cannot be eliminated.
namespace qwen_fp8 {
__global__ void __launch_bounds__(256) read_bw_kernel(const uint4* __restrict__ src, long long n16,
                                                      uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 4
  for (; i < n16; i += stride) {
    const uint4 v = ldg_nc_v4(src + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  acc = __reduce_xor_sync(0xffffffffu, acc);
  if ((threadIdx.x & 31) == 0) atomicXor(sink + blockIdx.x, acc);
}
}  // namespace qwen_fp8

extern "C" int qwen_fp8_read_bw(const void* src, long long bytes, void* sink, int blocks, void* stream) {
  if (!src || !sink || bytes <= 0 || bytes % 16 || blocks < 1 || reinterpret_cast<uintptr_t>(src) % 16)
    return int(cudaErrorInvalidValue);
  qwen_fp8::read_bw_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint4*>(src), bytes / 16, static_cast<uint32_t*>(sink));
  return int(cudaGetLastError());
}
#endif  // QWEN_FP8_GEMV_PRODUCTION
