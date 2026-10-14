// Qwen3.6-35B-A3B NVFP4 MoE prefill: CUDA kernels that are BIT-EXACT with the pinned
// FreeToken Triton path (Triton 3.6.0, sm_120, _prefill_nvfp4_moe_kernel with
// BLOCK_SIZE_KB=32, dot-operand kWidth=4) followed by FlashInfer 0.6.18 silu_and_mul
// and FreeToken moe_sum_reduce_kernel.
//
// Bit-exactness contract (see triton-mapping.md):
//  * every output element is produced by the SAME chain of
//    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 instructions as in Triton:
//    for each K64 block, in order: (even K, kRep0), (even K, kRep1), (odd K, kRep0),
//    (odd K, kRep1); K64 blocks in ascending order; accumulator starts at +0.0;
//  * inside each MMA the logical K element held at fragment position p is exactly the
//    one Triton places there (kWidth=4 "largeK" register split):
//        p in [0,8):  K = k0 + 8*(p/2)      + 4*s + 2*(p%2) + parity
//        p in [8,16): K = k0 + 32 + 8*((p-8)/2) + 4*s + 2*(p%2) + parity
//    (s = kRep, parity 0 for the even-K dot, 1 for the odd-K dot);
//  * B values are bf16(E2M1 * E4M3) which is exact in BF16, so any exact decode gives
//    identical bits; epilogue is fp32 (acc * fp32 global) [* router] -> RN bf16, no FMA;
//  * SiLU replicates the FlashInfer SASS (FMUL.FTZ, MUFU.EX2, FADD.FTZ, MUFU.RCP,
//    FMUL.FTZ, FMUL.FTZ, F2FP.BF16 RN);  top-k sum replicates Triton's fp32 serial sum.
// No split-K, no atomics, no fast-math: the build must not use --use_fast_math.
//
// Target: sm_120a only (uses cvt.rn.bf16x2.e2m1x2 / .e4m3x2 -> F2FP.BF16.E2M1/E4M3).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <climits>

struct QwenNvfp4Args {
  const __nv_bfloat16* a;
  const uint8_t* packed;
  const uint8_t* scales;
  const float* globals;  // fp32 NVFP4 weight_scale_2 per output row (ABI: qwen_nvfp4_global_bits() == 32)
  __nv_bfloat16* out;
  const float* route_weights;
  const int32_t* sorted_routes;
  const int32_t* expert_slots;
  const int32_t* padded_count;
  int64_t a_row_stride, packed_slot_stride, scale_slot_stride, global_slot_stride;
  int32_t n, k, sorted_capacity, valid_routes, input_topk, multiply_weight;
};

namespace bx {

// ------------------------------------------------------------------------------------
// Primitive wrappers (inline PTX)
// ------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                          uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}

// Two E2M1 codes in byte 0 (or byte 2) of x -> bf16x2 (low nibble -> low half). Exact.
__device__ __forceinline__ uint32_t e2m1x2_byte0(uint32_t x) {
  uint32_t r;
  asm("{\n .reg .b8 q0, q1, q2, q3;\n mov.b32 {q0, q1, q2, q3}, %1;\n"
      " cvt.rn.bf16x2.e2m1x2 %0, q0;\n}\n"
      : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t e2m1x2_byte2(uint32_t x) {
  uint32_t r;
  asm("{\n .reg .b8 q0, q1, q2, q3;\n mov.b32 {q0, q1, q2, q3}, %1;\n"
      " cvt.rn.bf16x2.e2m1x2 %0, q2;\n}\n"
      : "=r"(r) : "r"(x));
  return r;
}
// Two E4M3 codes (low 16 bits) -> bf16x2. Exact (E4M3 subnormals are BF16 normals).
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint32_t x) {
  uint32_t r;
  const uint16_t h = static_cast<uint16_t>(x);
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(h));
  return r;
}
// bf16x2 multiply, round-to-nearest.  For E2M1 x E4M3 the product has <= 6 significant
// bits and lies in [2^-10, 2688], so it is exact: identical to Triton's fp32 product
// rounded to bf16 (including signed zeros).
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// ------------------------------------------------------------------------------------
// Fragment builders (the Triton kWidth=4 mapping)
// ------------------------------------------------------------------------------------
// One packed word holds K = base..base+7 (nibble j = K base+j, low nibble = even K).
// Output (bf16x2, low half = lower MMA position):
//   q[0] = (K+0, K+2)  -> even-K dot, kRep0
//   q[1] = (K+4, K+6)  -> even-K dot, kRep1
//   q[2] = (K+1, K+3)  -> odd-K dot,  kRep0
//   q[3] = (K+5, K+7)  -> odd-K dot,  kRep1
// (m & a) | (~m & b) in one LOP3: immLut = f(0xF0,0xCC,0xAA) = (0xAA&0xF0)|(~0xAA&0xCC) = 0xE4
__device__ __forceinline__ uint32_t bitsel(uint32_t m, uint32_t a, uint32_t b) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xE4;\n" : "=r"(r) : "r"(a), "r"(b), "r"(m));
  return r;
}

__device__ __forceinline__ void dequant_word(uint32_t w, uint32_t s2, uint32_t (&q)[4]) {
  constexpr uint32_t M = 0x000F000Fu;
  const uint32_t w4 = w >> 4, w8 = w >> 8;
  const uint32_t xe = bitsel(M, w, w4);    // byte0 = n0|n2<<4, byte2 = n4|n6<<4
  const uint32_t xo = bitsel(M, w4, w8);   // byte0 = n1|n3<<4, byte2 = n5|n7<<4
  q[0] = bmul2(e2m1x2_byte0(xe), s2);
  q[1] = bmul2(e2m1x2_byte2(xe), s2);
  q[2] = bmul2(e2m1x2_byte0(xo), s2);
  q[3] = bmul2(e2m1x2_byte2(xo), s2);
}

// Eight consecutive bf16 (K = base..base+7) -> the same four pairings as dequant_word.
__device__ __forceinline__ void split_a(const uint4& v, uint32_t (&q)[4]) {
  q[0] = prmt(v.x, v.y, 0x5410u);
  q[1] = prmt(v.z, v.w, 0x5410u);
  q[2] = prmt(v.x, v.y, 0x7632u);
  q[3] = prmt(v.z, v.w, 0x7632u);
}

// ------------------------------------------------------------------------------------
// FlashInfer silu_and_mul (bf16, -use_fast_math, sm_120f build) replica.
// SASS of act_and_mul_kernel<__nv_bfloat16, silu>:
//   FMUL.FTZ t, x, -1.4426950216293334961 ; MUFU.EX2 e, t ; FADD.FTZ d, e, 1 ;
//   MUFU.RCP r, d ; FMUL.FTZ s, x, r ; FMUL.FTZ o, y, s ; F2FP.BF16.F32.PACK_AB (RN)
// ------------------------------------------------------------------------------------
__device__ __forceinline__ float silu_mul_flashinfer(float x, float y) {
  float t, e, d, r, s, o;
  asm("mul.rn.ftz.f32 %0, %1, 0fBFB8AA3B;\n" : "=f"(t) : "f"(x));
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(e) : "f"(t));
  asm("add.rn.ftz.f32 %0, %1, 0f3F800000;\n" : "=f"(d) : "f"(e));
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(d));
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(s) : "f"(x), "f"(r));
  asm("mul.rn.ftz.f32 %0, %1, %2;\n" : "=f"(o) : "f"(y), "f"(s));
  return o;
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t h16) { return __uint_as_float(h16 << 16); }

__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float round_bf16(float x) {
  return bf16_bits_to_float(pack_bf16x2_rn(x, 0.0f) & 0xFFFFu);
}

// ------------------------------------------------------------------------------------
// Oracle: one warp = one 16x8 output tile, fragments assembled element by element from
// the mapping formula.  Deliberately independent of the fast kernel's PRMT/F2FP path.
// ------------------------------------------------------------------------------------
__constant__ float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// mode 0: Triton kWidth=4 mapping (the claim under test).
// mode 1: negative control = previous "matched" CUDA variant: even32 then odd32 per K64,
//         natural order inside each K16 step.
// mode 2: negative control = natural K order (original WMMA prototype).
// modes 3-5 keep Triton's SET of K elements per MMA but move them to other positions:
//   3: swap the two 8-position halves (p ^ 8); 4: reverse thread pairs inside each half;
//   5: swap the two elements of every pair (p ^ 1).
__device__ __forceinline__ int oracle_k(int pos, int s, int parity, int mode) {
  if (mode == 1) return 2 * (16 * s + pos) + parity;
  if (mode == 2) return 16 * (2 * parity + s) + pos;
  if (mode == 3) pos ^= 8;
  if (mode == 4) pos = (pos & 8) | (6 - (pos & 6)) | (pos & 1);
  if (mode == 5) pos ^= 1;
  const int j = pos < 8 ? (4 * (pos >> 1) + 2 * s + (pos & 1))
                        : (16 + 4 * ((pos - 8) >> 1) + 2 * s + (pos & 1));
  return 2 * j + parity;
}

__device__ __forceinline__ uint16_t oracle_b(const uint8_t* wrow, const uint8_t* srow, int k) {
  const uint8_t byte = wrow[k >> 1];
  const int code = (k & 1) ? (byte >> 4) : (byte & 15);
  __nv_fp8_e4m3 sc;
  sc.__x = srow[k >> 4];
  const float v = kE2M1[code] * static_cast<float>(sc);
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__global__ void __launch_bounds__(32) oracle_gemm(QwenNvfp4Args p, int align, int mode) {
  const int n0 = blockIdx.x * 8;
  const int row0 = blockIdx.y * 16;
  if (row0 >= *p.padded_count) return;
  const int slot = p.expert_slots[row0 / align];
  if (slot < 0) return;
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int r_lo = p.sorted_routes[row0 + g], r_hi = p.sorted_routes[row0 + g + 8];
  const bool v_lo = r_lo >= 0 && r_lo < p.valid_routes, v_hi = r_hi >= 0 && r_hi < p.valid_routes;
  const __nv_bfloat16* a_lo = p.a + (v_lo ? int64_t(r_lo / p.input_topk) * p.a_row_stride : 0);
  const __nv_bfloat16* a_hi = p.a + (v_hi ? int64_t(r_hi / p.input_topk) * p.a_row_stride : 0);
  const int nrow = n0 + g;
  const uint8_t* wrow = p.packed + int64_t(slot) * p.packed_slot_stride + int64_t(nrow) * (p.k / 2);
  const uint8_t* srow = p.scales + int64_t(slot) * p.scale_slot_stride + int64_t(nrow) * (p.k / 16);
  auto av = [&](const __nv_bfloat16* row, bool valid, int k) -> uint32_t {
    return valid ? uint32_t(__bfloat16_as_ushort(row[k])) : 0u;
  };
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k0 = 0; k0 < p.k; k0 += 64) {
    for (int ks = 0; ks < 4; ++ks) {
      const int parity = ks >> 1, s = ks & 1;
      const int kA = k0 + oracle_k(2 * t, s, parity, mode), kB = k0 + oracle_k(2 * t + 1, s, parity, mode);
      const int kC = k0 + oracle_k(2 * t + 8, s, parity, mode), kD = k0 + oracle_k(2 * t + 9, s, parity, mode);
      const uint32_t a0 = av(a_lo, v_lo, kA) | (av(a_lo, v_lo, kB) << 16);
      const uint32_t a1 = av(a_hi, v_hi, kA) | (av(a_hi, v_hi, kB) << 16);
      const uint32_t a2 = av(a_lo, v_lo, kC) | (av(a_lo, v_lo, kD) << 16);
      const uint32_t a3 = av(a_hi, v_hi, kC) | (av(a_hi, v_hi, kD) << 16);
      const uint32_t b0 = uint32_t(oracle_b(wrow, srow, kA)) | (uint32_t(oracle_b(wrow, srow, kB)) << 16);
      const uint32_t b1 = uint32_t(oracle_b(wrow, srow, kC)) | (uint32_t(oracle_b(wrow, srow, kD)) << 16);
      mma_16816(acc, a0, a1, a2, a3, b0, b1);
    }
  }
  const float* gl = p.globals + int64_t(slot) * p.global_slot_stride;
  const int col = n0 + 2 * t;
  const float g0 = gl[col], g1 = gl[col + 1];
  if (v_lo) {
    float x0 = acc[0] * g0, x1 = acc[1] * g1;
    if (p.multiply_weight) { const float w = p.route_weights[r_lo]; x0 = x0 * w; x1 = x1 * w; }
    *reinterpret_cast<uint32_t*>(p.out + int64_t(r_lo) * p.n + col) = pack_bf16x2_rn(x0, x1);
  }
  if (v_hi) {
    float x0 = acc[2] * g0, x1 = acc[3] * g1;
    if (p.multiply_weight) { const float w = p.route_weights[r_hi]; x0 = x0 * w; x1 = x1 * w; }
    *reinterpret_cast<uint32_t*>(p.out + int64_t(r_hi) * p.n + col) = pack_bf16x2_rn(x0, x1);
  }
}

// ------------------------------------------------------------------------------------
// Fast kernel.
//   CTA = BM routes x (NWARP*NT*8) B rows; warps split only along N, so every warp owns
//   the whole BM and each B fragment is dequantized exactly once per CTA.
//   FUSED: B rows [0,BNB/2) are gate rows, [BNB/2,BNB) the matching up rows; the
//   epilogue rounds gate/up to bf16, applies the FlashInfer SiLU and writes I columns.
//   One cp.async stage = KS consecutive K64 blocks:
//     A [BM][KS*128B]: 16B chunk ch of K64 block kk at kk*128 + ((ch ^ 4*(row&1))*16)
//       -> the 8 lanes of each LDS.128 phase (rows g,g+1 x t) hit 8 distinct 16B groups;
//     B [BNB][KS*32B]: 16B chunk c=(2kk+h) at (c ^ swzB(row))*16 -> conflict-free ldmatrix;
//     S [BNB][KS*4B] block scales.
//   The MMA issue order of independent accumulators is free; for every accumulator the
//   chain is exactly Triton's (see file header).
// ------------------------------------------------------------------------------------
template <int BM, int NT, int NWARP, int STAGES, int KS, bool FUSED>
struct Cfg {
  static constexpr int THREADS = NWARP * 32;
  static constexpr int MT = BM / 16;
  static constexpr int BNB = NWARP * NT * 8;
  static constexpr int A_ROW = 128 * KS;
  static constexpr int B_ROW = 32 * KS;
  static constexpr int S_ROW = 4 * KS;
  static constexpr int A_BYTES = BM * A_ROW;
  static constexpr int B_BYTES = BNB * B_ROW;
  static constexpr int S_BYTES = BNB * S_ROW;
  static constexpr int STAGE_BYTES = A_BYTES + B_BYTES + S_BYTES;
  static constexpr int SMEM = STAGES * STAGE_BYTES;
  static_assert(BM % 16 == 0 && BM <= 128, "BM");
  static_assert(NT % 2 == 0, "ldmatrix.x4 covers two n8 tiles; fused needs gate/up pairs");
  static_assert(KS == 1 || KS == 2 || KS == 4, "KS");
  static_assert(STAGES >= 2, "STAGES");
  static_assert(BNB <= THREADS, "one scale row per thread");
};

template <int KS>
__device__ __forceinline__ int swz_b(int row) {
  if constexpr (KS == 1) return (row >> 2) & 1;
  else if constexpr (KS == 2) return (row >> 1) & 3;
  else return row & 7;
}

template <int BYTES>
__device__ __forceinline__ void cp_async_small(uint32_t dst, const void* src) {
  if constexpr (BYTES == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
  else asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(BYTES));
}

// Main loop + epilogue for a CTA whose valid rows all lie in the first NA m16 tiles.
// Padded rows inside those tiles are zero-filled and never stored, so processing them is
// harmless; NA only removes whole all-padding tiles (warp-uniform, no per-tile branches).
template <int BM, int NT, int NWARP, int STAGES, int KS, bool FUSED, int NA>
__device__ __forceinline__ void fast_body(const QwenNvfp4Args& p, int slot, const int* s_route, uint8_t* smem) {
  using C = Cfg<BM, NT, NWARP, STAGES, KS, FUSED>;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int K = p.k;
  const int NSTEP = K / (64 * KS);
  const int out_n = FUSED ? p.n / 2 : p.n;
  constexpr int HALF = C::BNB / 2;
  auto b_global_row = [&](int r) -> int {
    if constexpr (FUSED) {
      return r < HALF ? int(blockIdx.x) * HALF + r : out_n + int(blockIdx.x) * HALF + (r - HALF);
    } else {
      return int(blockIdx.x) * C::BNB + r;
    }
  };

  const uint8_t* wbase = p.packed + int64_t(slot) * p.packed_slot_stride;
  const uint8_t* sbase = p.scales + int64_t(slot) * p.scale_slot_stride;
  const uint32_t sm0 = smem_u32(smem);

  // ---- per-thread copy assignments (fixed for the whole K loop) ----
  constexpr int A_CPR = 8 * KS;                      // 16B chunks per A row per stage
  constexpr int A_CHUNKS = BM * A_CPR;
  constexpr int A_PER_T = (A_CHUNKS + C::THREADS - 1) / C::THREADS;
  const __nv_bfloat16* a_src[A_PER_T];
  uint32_t a_dst[A_PER_T];
  bool a_ok[A_PER_T], a_on[A_PER_T];
#pragma unroll
  for (int i = 0; i < A_PER_T; ++i) {
    const int c = tid + i * C::THREADS;
    a_on[i] = (A_CHUNKS % C::THREADS == 0) || c < A_CHUNKS;
    const int row = c / A_CPR, cc = c % A_CPR, kk = cc >> 3, ch = cc & 7;
    const int r = a_on[i] ? s_route[row] : -1;
    a_ok[i] = r >= 0;
    a_src[i] = p.a + (a_ok[i] ? int64_t(r / p.input_topk) * p.a_row_stride : 0) + kk * 64 + ch * 8;
    a_dst[i] = uint32_t(row * C::A_ROW + kk * 128 + ((ch ^ ((row & 1) << 2)) << 4));
  }
  constexpr int B_CPR = 2 * KS;
  constexpr int B_CHUNKS = C::BNB * B_CPR;
  constexpr int B_PER_T = (B_CHUNKS + C::THREADS - 1) / C::THREADS;
  const uint8_t* b_src[B_PER_T];
  uint32_t b_dst[B_PER_T];
  bool b_on[B_PER_T];
#pragma unroll
  for (int i = 0; i < B_PER_T; ++i) {
    const int c = tid + i * C::THREADS;
    b_on[i] = (B_CHUNKS % C::THREADS == 0) || c < B_CHUNKS;
    const int row = c / B_CPR, cc = c % B_CPR;
    const int grow = b_on[i] ? b_global_row(row) : 0;
    b_src[i] = wbase + int64_t(grow) * (K / 2) + cc * 16;
    b_dst[i] = uint32_t(C::A_BYTES + row * C::B_ROW + ((cc ^ swz_b<KS>(row)) << 4));
  }
  const bool s_on = tid < C::BNB;
  const uint8_t* s_src = sbase + int64_t(s_on ? b_global_row(tid) : 0) * (K / 16);
  const uint32_t s_dst = uint32_t(C::A_BYTES + C::B_BYTES + tid * C::S_ROW);

  auto load_stage = [&](int stage, int step) {
    const uint32_t base = sm0 + stage * C::STAGE_BYTES;
#pragma unroll
    for (int i = 0; i < A_PER_T; ++i)
      if (a_on[i]) cp_async16(base + a_dst[i], a_src[i] + step * (64 * KS), a_ok[i]);
#pragma unroll
    for (int i = 0; i < B_PER_T; ++i)
      if (b_on[i]) cp_async16(base + b_dst[i], b_src[i] + step * (32 * KS), true);
    if (s_on) cp_async_small<C::S_ROW>(base + s_dst, s_src + step * (4 * KS));
  };

  // ---- per-lane fragment addressing ----
  auto tile_row = [&](int j) -> int {
    if constexpr (FUSED) {
      return j < NT / 2 ? warp * (NT / 2) * 8 + j * 8 : HALF + warp * (NT / 2) * 8 + (j - NT / 2) * 8;
    } else {
      return warp * NT * 8 + j * 8;
    }
  };
  uint32_t ldsm_off[NT / 2][KS];
#pragma unroll
  for (int jp = 0; jp < NT / 2; ++jp) {
    const int mi = lane >> 3, r = lane & 7;
    const int j = 2 * jp + (mi >> 1), h = mi & 1;
    const int row = tile_row(j) + r;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk)
      ldsm_off[jp][kk] = uint32_t(C::A_BYTES + row * C::B_ROW + (((2 * kk + h) ^ swz_b<KS>(row)) << 4));
  }
  uint32_t scale_off[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) scale_off[j] = uint32_t(C::A_BYTES + C::B_BYTES + (tile_row(j) + g) * C::S_ROW);
  const uint32_t sel_a = uint32_t(t >> 1) * 0x11u, sel_b = uint32_t((t >> 1) + 2) * 0x11u;
  const uint32_t swz = uint32_t((g & 1) << 2);
  const uint32_t a_off_h0 = uint32_t(g * C::A_ROW) + ((uint32_t(t) ^ swz) << 4);
  const uint32_t a_off_h1 = uint32_t(g * C::A_ROW) + ((uint32_t(4 + t) ^ swz) << 4);

  float acc[NA][NT][4];
#pragma unroll
  for (int i = 0; i < NA; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  // ---- prologue ----
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < NSTEP) load_stage(s, s);
    cp_async_commit();
  }

  int stage = 0;
  for (int step = 0; step < NSTEP; ++step) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = step + STAGES - 1;
      int ns = stage + STAGES - 1;
      if (ns >= STAGES) ns -= STAGES;
      if (nk < NSTEP) load_stage(ns, nk);
      cp_async_commit();
    }
    const uint32_t base = sm0 + stage * C::STAGE_BYTES;
    if (++stage == STAGES) stage = 0;

#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      // B fragments: bq[j][ks][0] from word0 (K 8t..8t+7), bq[j][ks][1] from word1 (K 32+8t..)
      uint32_t bq[NT][4][2];
#pragma unroll
      for (int jp = 0; jp < NT / 2; ++jp) {
        uint32_t w0a, w1a, w0b, w1b;
        ldsm_x4(base + ldsm_off[jp][kk], w0a, w1a, w0b, w1b);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int j = 2 * jp + u;
          const uint32_t sw = lds32(base + scale_off[j] + 4 * kk);
          const uint32_t sa = e4m3x2_bf16x2(__byte_perm(sw, 0u, sel_a));
          const uint32_t sb = e4m3x2_bf16x2(__byte_perm(sw, 0u, sel_b));
          uint32_t q0[4], q1[4];
          dequant_word(u == 0 ? w0a : w0b, sa, q0);
          dequant_word(u == 0 ? w1a : w1b, sb, q1);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            bq[j][ks][0] = q0[ks];
            bq[j][ks][1] = q1[ks];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < NA; ++i) {
        const uint32_t rb = base + uint32_t(i * 16 * C::A_ROW + kk * 128);
        uint32_t p00[4], p01[4], p10[4], p11[4];
        split_a(lds128(rb + a_off_h0), p00);                  // row g,   K 8t..
        split_a(lds128(rb + a_off_h1), p01);                  // row g,   K 32+8t..
        split_a(lds128(rb + 8 * C::A_ROW + a_off_h0), p10);   // row g+8, K 8t..
        split_a(lds128(rb + 8 * C::A_ROW + a_off_h1), p11);   // row g+8, K 32+8t..
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
          for (int j = 0; j < NT; ++j)
            mma_16816(acc[i][j], p00[ks], p10[ks], p01[ks], p11[ks], bq[j][ks][0], bq[j][ks][1]);
      }
    }
  }
  cp_async_wait<0>();

  // ---- epilogue ----
  const float* gl = p.globals + int64_t(slot) * p.global_slot_stride;
  if constexpr (FUSED) {
#pragma unroll
    for (int jj = 0; jj < NT / 2; ++jj) {
      const int col = int(blockIdx.x) * HALF + warp * (NT / 2) * 8 + jj * 8 + 2 * t;
      const float gg0 = gl[col], gg1 = gl[col + 1];
      const float gu0 = gl[out_n + col], gu1 = gl[out_n + col + 1];
#pragma unroll
      for (int i = 0; i < NA; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = s_route[i * 16 + g + 8 * hh];
          if (r < 0) continue;
          const float* ag = &acc[i][jj][2 * hh];
          const float* au = &acc[i][jj + NT / 2][2 * hh];
          float x0 = ag[0] * gg0, x1 = ag[1] * gg1;
          float y0 = au[0] * gu0, y1 = au[1] * gu1;
          if (p.multiply_weight) {
            const float w = p.route_weights[r];
            x0 = x0 * w; x1 = x1 * w; y0 = y0 * w; y1 = y1 * w;
          }
          const uint32_t gb = pack_bf16x2_rn(x0, x1), ub = pack_bf16x2_rn(y0, y1);
          const float o0 = silu_mul_flashinfer(bf16_bits_to_float(gb & 0xFFFFu), bf16_bits_to_float(ub & 0xFFFFu));
          const float o1 = silu_mul_flashinfer(bf16_bits_to_float(gb >> 16), bf16_bits_to_float(ub >> 16));
          *reinterpret_cast<uint32_t*>(p.out + int64_t(r) * out_n + col) = pack_bf16x2_rn(o0, o1);
        }
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = int(blockIdx.x) * C::BNB + warp * NT * 8 + j * 8 + 2 * t;
      const float g0 = gl[col], g1 = gl[col + 1];
#pragma unroll
      for (int i = 0; i < NA; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r = s_route[i * 16 + g + 8 * hh];
          if (r < 0) continue;
          float x0 = acc[i][j][2 * hh] * g0, x1 = acc[i][j][2 * hh + 1] * g1;
          if (p.multiply_weight) {
            const float w = p.route_weights[r];
            x0 = x0 * w; x1 = x1 * w;
          }
          *reinterpret_cast<uint32_t*>(p.out + int64_t(r) * p.n + col) = pack_bf16x2_rn(x0, x1);
        }
      }
    }
  }
}

template <int BM, int NT, int NWARP, int STAGES, int KS, bool FUSED>
__global__ void __launch_bounds__(NWARP * 32)
fast_gemm(QwenNvfp4Args p, int align) {
  using C = Cfg<BM, NT, NWARP, STAGES, KS, FUSED>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_route[BM];
  __shared__ unsigned s_mask;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row_start = blockIdx.y * BM;
  if (row_start >= *p.padded_count) return;
  const int slot = p.expert_slots[row_start / align];
  if (slot < 0) return;

  if (tid == 0) s_mask = 0u;
  __syncthreads();
  if (tid < BM) {
    const int r = p.sorted_routes[row_start + tid];
    const bool valid = r >= 0 && r < p.valid_routes;
    s_route[tid] = valid ? r : -1;
    if (valid) atomicOr(&s_mask, 1u << (tid >> 4));
  }
  __syncthreads();
  const unsigned mmask = s_mask;
  if (mmask == 0u) return;

  const int na = 32 - __clz(mmask);   // tiles >= na contain only padding
  switch (na) {
    case 1: fast_body<BM, NT, NWARP, STAGES, KS, FUSED, 1>(p, slot, s_route, smem); break;
    case 2: if constexpr (C::MT >= 2) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 2 ? 2 : 1)>(p, slot, s_route, smem); break;
    case 3: if constexpr (C::MT >= 3) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 3 ? 3 : 1)>(p, slot, s_route, smem); break;
    case 4: if constexpr (C::MT >= 4) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 4 ? 4 : 1)>(p, slot, s_route, smem); break;
    case 5: if constexpr (C::MT >= 5) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 5 ? 5 : 1)>(p, slot, s_route, smem); break;
    case 6: if constexpr (C::MT >= 6) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 6 ? 6 : 1)>(p, slot, s_route, smem); break;
    case 7: if constexpr (C::MT >= 7) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 7 ? 7 : 1)>(p, slot, s_route, smem); break;
    default: if constexpr (C::MT >= 8) fast_body<BM, NT, NWARP, STAGES, KS, FUSED, (C::MT >= 8 ? 8 : 1)>(p, slot, s_route, smem); break;
  }
}

// ------------------------------------------------------------------------------------
// Stand-alone replicas used by the tests and by the optional all-CUDA path.
// ------------------------------------------------------------------------------------
__global__ void silu_and_mul_kernel(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                    int64_t rows, int d) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * d) return;
  const int64_t row = idx / d;
  const int col = int(idx - row * d);
  const float x = __bfloat162float(in[row * 2 * d + col]);
  const float y = __bfloat162float(in[row * 2 * d + d + col]);
  out[idx] = __ushort_as_bfloat16(uint16_t(pack_bf16x2_rn(silu_mul_flashinfer(x, y), 0.0f) & 0xFFFFu));
}

// Triton moe_sum_reduce_kernel: fp32 accumulator = +0.0, then acc += x_i for i = 0..topk-1
// in order, finally RN to bf16.
__global__ void moe_sum_kernel(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
                               int tokens, int topk, int hidden) {
  const int64_t idx2 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;   // pair index
  const int64_t pairs = int64_t(tokens) * (hidden / 2);
  if (idx2 >= pairs) return;
  const int64_t tok = idx2 / (hidden / 2);
  const int c = int(idx2 - tok * (hidden / 2)) * 2;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(in + tok * int64_t(topk) * hidden + c);
  float a0 = 0.0f, a1 = 0.0f;
  for (int i = 0; i < topk; ++i) {
    const uint32_t v = __ldg(src + int64_t(i) * (hidden / 2));
    a0 = __fadd_rn(a0, bf16_bits_to_float(v & 0xFFFFu));
    a1 = __fadd_rn(a1, bf16_bits_to_float(v >> 16));
  }
  *reinterpret_cast<uint32_t*>(out + tok * hidden + c) = pack_bf16x2_rn(a0, a1);
}

// Roofline probe: independent HMMA.16816.F32.BF16 chains, operands in registers.
__global__ void __launch_bounds__(256) mma_peak_kernel(float* out, int iters, uint32_t seed) {
  float acc[8][4];
#pragma unroll
  for (int c = 0; c < 8; ++c)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[c][e] = 0.0f;
  const uint32_t a = 0x3F803F80u ^ (seed & 0x00010001u) ^ threadIdx.x;
  const uint32_t b = 0x3F803F80u ^ (seed >> 16 & 0x00010001u);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < 8; ++c) mma_16816(acc[c], a, a ^ c, a, a, b, b ^ c);
  }
  float s = 0.0f;
#pragma unroll
  for (int c = 0; c < 8; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  if (s == 1.2345f) out[threadIdx.x] = s;
}

// Dequant fixture: word i with scale byte s[i] -> 8 bf16 in natural K order, produced by
// exactly the fast-kernel decode path (dequant_word + e4m3x2_bf16x2).
__global__ void dequant_fixture_kernel(const uint32_t* words, const uint8_t* scales, uint16_t* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t sb = scales[i];
  const uint32_t s2 = e4m3x2_bf16x2(sb | (sb << 8));
  uint32_t q[4];
  dequant_word(words[i], s2, q);
  uint16_t* o = out + int64_t(i) * 8;
  o[0] = uint16_t(q[0]); o[2] = uint16_t(q[0] >> 16);
  o[4] = uint16_t(q[1]); o[6] = uint16_t(q[1] >> 16);
  o[1] = uint16_t(q[2]); o[3] = uint16_t(q[2] >> 16);
  o[5] = uint16_t(q[3]); o[7] = uint16_t(q[3] >> 16);
}

// ------------------------------------------------------------------------------------
// Host-side launch table
// ------------------------------------------------------------------------------------
template <int BM, int NT, int NWARP, int STAGES, int KS, bool FUSED>
cudaError_t launch_fast(const QwenNvfp4Args& p, int align, cudaStream_t stream) {
  using C = Cfg<BM, NT, NWARP, STAGES, KS, FUSED>;
  auto kern = fast_gemm<BM, NT, NWARP, STAGES, KS, FUSED>;
  static bool attr_done = false;   // per instantiation; idempotent
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    if (e != cudaSuccess) return e;
    attr_done = true;
  }
  if (align % BM || p.k % (64 * KS)) return cudaErrorInvalidValue;
  const int out_cols = FUSED ? p.n / 2 : p.n;
  const int cols_per_cta = FUSED ? C::BNB / 2 : C::BNB;
  if (out_cols % cols_per_cta) return cudaErrorInvalidValue;
  const dim3 grid(out_cols / cols_per_cta, (p.sorted_capacity + BM - 1) / BM);
  kern<<<grid, C::THREADS, C::SMEM, stream>>>(p, align);
  return cudaGetLastError();
}

using LaunchFn = cudaError_t (*)(const QwenNvfp4Args&, int, cudaStream_t);
struct Variant {
  int bm;
  LaunchFn fused;
  LaunchFn plain;
};
// id -> config.  BM, NT, NWARP, STAGES, KS
#define BX_VARIANT(BM, NT, NW, ST, KS)   {BM, launch_fast<BM, NT, NW, ST, KS, true>, launch_fast<BM, NT, NW, ST, KS, false>}
static const Variant kVariants[] = {
    BX_VARIANT(64, 2, 8, 3, 1),   // 0: large M
    BX_VARIANT(32, 2, 8, 4, 1),   // 1
    BX_VARIANT(16, 2, 8, 4, 1),   // 2: small M
    BX_VARIANT(64, 4, 8, 2, 1),   // 3
    BX_VARIANT(64, 2, 4, 4, 1),   // 4
    BX_VARIANT(128, 2, 8, 2, 1),  // 5
    BX_VARIANT(32, 4, 8, 3, 1),   // 6
    BX_VARIANT(64, 2, 8, 2, 2),   // 7: K128 stages
    BX_VARIANT(32, 2, 8, 3, 2),   // 8
    BX_VARIANT(16, 2, 8, 3, 2),   // 9
    BX_VARIANT(16, 2, 4, 4, 1),   // 10: small M, narrow N (more CTAs)
    BX_VARIANT(64, 4, 4, 3, 1),   // 11
};
#undef BX_VARIANT
constexpr int kNumVariants = int(sizeof(kVariants) / sizeof(kVariants[0]));

// Tuned on RTX 5070 Ti (results/bench-v2.json, variant sweep, uniform routing).  Every
// variant produces identical bits, so this only changes speed.
//   gate/up+SiLU (K=2048): small batches are bandwidth bound -> K128 stages keep more
//   bytes in flight: v8 (BM32,KS2) up to 2048 routes, v9 (BM16,KS2) up to 8192, else v0.
//   down (K=512): v1 (BM32) up to 2048 routes, else v0.
int choose_variant(const QwenNvfp4Args& p, int align, bool fused) {
  const int routes = p.valid_routes;
  int v;
  if (fused) v = routes <= 2048 ? 8 : (routes <= 8192 ? 9 : 0);
  else v = routes <= 2048 ? 1 : 0;
  if (align % kVariants[v].bm == 0 && p.k % (v >= 7 && v <= 9 ? 128 : 64) == 0) return v;
  for (int c = 0; c < 3; ++c) if (align % kVariants[c].bm == 0) return c;
  return 2;
}

bool args_ok(const QwenNvfp4Args* p, int align, bool fused) {
  if (!p || !p->a || !p->packed || !p->scales || !p->globals || !p->out || !p->sorted_routes ||
      !p->expert_slots || !p->padded_count)
    return false;
  if (p->k <= 0 || p->k % 64 || p->n <= 0 || p->n % 16 || align <= 0 || align % 16) return false;
  if (p->a_row_stride % 8 || (reinterpret_cast<uintptr_t>(p->a) & 15)) return false;
  if ((reinterpret_cast<uintptr_t>(p->packed) & 15) || p->packed_slot_stride % 16) return false;
  if ((reinterpret_cast<uintptr_t>(p->scales) & 3) || p->scale_slot_stride % 4) return false;
  if ((reinterpret_cast<uintptr_t>(p->out) & 3)) return false;
  if ((reinterpret_cast<uintptr_t>(p->globals) & 3)) return false;
  if (p->input_topk < 1 || p->valid_routes < 0 || p->sorted_capacity < 0) return false;
  if (p->multiply_weight && !p->route_weights) return false;
  if (fused && p->n % 2) return false;
  return true;
}

}  // namespace bx

// ======================================================================================
// C ABI
// ======================================================================================
// Compatible with the previous opt-in wrapper (freetoken/moe/cuda_nvfp4.py):
//   bm = moe_align_block_size block (rows per expert_slots entry); bn/bk are ignored.
extern "C" int qwen_nvfp4_prefill(const QwenNvfp4Args* p, int bm, int bn, int bk, int fused_silu,
                                  void* stream) {
  (void)bn;
  (void)bk;
  if (!bx::args_ok(p, bm, fused_silu != 0)) return int(cudaErrorInvalidValue);
  const int v = bx::choose_variant(*p, bm, fused_silu != 0);
  const auto& var = bx::kVariants[v];
  auto s = reinterpret_cast<cudaStream_t>(stream);
  return int(fused_silu ? var.fused(*p, bm, s) : var.plain(*p, bm, s));
}

// Explicit variant (benchmarks / tuning).  variant < 0 -> automatic.
extern "C" int qwen_nvfp4_prefill_variant(const QwenNvfp4Args* p, int align, int variant, int fused_silu,
                                          void* stream) {
  if (!bx::args_ok(p, align, fused_silu != 0)) return int(cudaErrorInvalidValue);
  if (variant >= bx::kNumVariants) return int(cudaErrorInvalidValue);
  const int v = variant < 0 ? bx::choose_variant(*p, align, fused_silu != 0) : variant;
  if (align % bx::kVariants[v].bm) return int(cudaErrorInvalidValue);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  return int(fused_silu ? bx::kVariants[v].fused(*p, align, s) : bx::kVariants[v].plain(*p, align, s));
}

extern "C" int qwen_nvfp4_variant_count() { return bx::kNumVariants; }
extern "C" int qwen_nvfp4_variant_bm(int v) { return (v >= 0 && v < bx::kNumVariants) ? bx::kVariants[v].bm : -1; }

// Reference-mapping oracle (slow; non-fused only).  mode 0 = Triton mapping,
// 1/2 = negative controls (see oracle_k).
extern "C" int qwen_nvfp4_oracle(const QwenNvfp4Args* p, int align, int mode, void* stream) {
  if (!bx::args_ok(p, align, false) || p->n % 8 || mode < 0 || mode > 5) return int(cudaErrorInvalidValue);
  const dim3 grid(p->n / 8, (p->sorted_capacity + 15) / 16);
  bx::oracle_gemm<<<grid, 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(*p, align, mode);
  return int(cudaGetLastError());
}

extern "C" int qwen_nvfp4_silu_and_mul(const void* in, void* out, int64_t rows, int d, void* stream) {
  if (!in || !out || rows < 0 || d <= 0) return int(cudaErrorInvalidValue);
  const int64_t n = rows * d;
  if (n == 0) return 0;
  bx::silu_and_mul_kernel<<<unsigned((n + 255) / 256), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(in), static_cast<__nv_bfloat16*>(out), rows, d);
  return int(cudaGetLastError());
}

extern "C" int qwen_nvfp4_moe_sum(const void* in, void* out, int tokens, int topk, int hidden, void* stream) {
  if (!in || !out || tokens < 0 || topk < 1 || hidden % 2) return int(cudaErrorInvalidValue);
  if ((reinterpret_cast<uintptr_t>(in) & 3) || (reinterpret_cast<uintptr_t>(out) & 3)) return int(cudaErrorInvalidValue);
  const int64_t pairs = int64_t(tokens) * (hidden / 2);
  if (pairs == 0) return 0;
  bx::moe_sum_kernel<<<unsigned((pairs + 255) / 256), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(in), static_cast<__nv_bfloat16*>(out), tokens, topk, hidden);
  return int(cudaGetLastError());
}

extern "C" int qwen_nvfp4_dequant_fixture(const void* words, const void* scales, void* out, int count,
                                          void* stream) {
  if (count < 1) return int(cudaErrorInvalidValue);
  bx::dequant_fixture_kernel<<<(count + 127) / 128, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t*>(words), static_cast<const uint8_t*>(scales), static_cast<uint16_t*>(out),
      count);
  return int(cudaGetLastError());
}

// HMMA roofline probe: blocks x 256 threads, each warp issues iters*8 m16n8k16 MMAs.
extern "C" int qwen_nvfp4_mma_peak(void* scratch, int blocks, int iters, void* stream) {
  if (!scratch || blocks < 1 || iters < 1) return int(cudaErrorInvalidValue);
  bx::mma_peak_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<float*>(scratch), iters,
                                                                                  0x12345u);
  return int(cudaGetLastError());
}

extern "C" const char* qwen_nvfp4_cuda_error(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}

// Width of the per-row NVFP4 global the kernels read (wrappers refuse anything else).
extern "C" int qwen_nvfp4_global_bits() { return 32; }

extern "C" const char* qwen_nvfp4_build_id() {
  return "qwen36-nvfp4-bitexact v4 (fp32 globals, triton-3.6 kWidth4 mapping, flashinfer-0.6.18 silu, sm_120a, tuned selector)";
}
