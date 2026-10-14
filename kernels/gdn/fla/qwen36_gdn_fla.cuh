// qwen36_gdn_fla.cuh -- fused FLA chunk_h + chunk_o (GDN prefill) for SM120, bit-exact to
//   chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (Triton 3.6, BV32 nw4 ns2, IS_VARLEN, USE_G,
//   USE_INITIAL_STATE, INPLACE_UPDATE, FP32 snapshot) followed by
//   chunk_fwd_kernel_o (BT64 BK128 BV64 nw4 ns2, IS_VARLEN, USE_G).
// Included inside the anonymous namespace of qwen36_gdn.cu (uses mul_rn/sub_rn/add_rn/fma_rn/
// ex2_approx from there).
//
// Floating-point recipe (production PTX 9.4 / ptxas 13.4 SASS of both kernels, quoted):
//   H0      = 0.0 + state                         FADD Rx, RZ, Rload
//   Hb      = bf16_rn(H)                          cvt.rn.bf16x2.f32 (same bytes as the bf16 h tensor)
//   v_dot   = mma chain over K=128 (8 x k16, natural K order, zero init)   w . Hb^T
//   vn      = f32(u) - v_dot                      sub.f32x2 / FADD R, -vd, u
//   v_new   = bf16_rn(vn)
//   e_t     = ex2.approx(sel(g_last - g_t))       sub; setp.le x,0; mul x*0f3FB8AA3B; selp(.,-inf)
//   vs      = bf16_rn(vn * e_t)                   mul.f32; cvt.rn.bf16x2
//   eg      = ex2.approx(g_last * log2e)
//   D       = mma chain over K=64 (4 x k16 over t, zero init)   k^T . vs
//   H       = fma(H, eg, D)                       fma.rn.f32x2 H, eg, D
//   qh      = mma chain over K=128 (zero init)    q . Hb^T          (chunk_o b_o)
//   A       = mma chain over K=128 (zero init)    q . k^T           (chunk_o b_A)
//   Ag      = i>=j ? bf16_rn(A * ex2.approx(sel(g_i - g_j))) : +0
//   d2      = mma chain over K=64 (zero init)     Ag . v_new
//   o       = bf16_rn(fma(scale, qh * ex2.approx(g_t*log2e), scale * d2))
// ex2.approx.f32 is the non-ftz form (ptxas expands the -126 denormal fix identically for both).
// Every HMMA output element sees exactly the K-set and chain order of the Triton kernels; the
// operand orientation (which tensor is the A operand) differs where that saves data movement,
// which the probe in work/cuda-gdn/fla/hmma_probe.cu checks is value-neutral on SM120.
//
// Tiling: one CTA per (sequence, value head, 64-wide V block) walks all chunks of its sequence.
// 4 warps; warp w owns V rows [16w,16w+16) of the block and keeps H^T[v][k] (fp32, 64 regs) as
// mma accumulators, so bf16(H) is directly the A operand of v_dot^T / qh^T, and v_new / vs are
// directly the A operands of D^T / d2^T (FlashAttention-style register reuse, no H round trip).
// k/u/g are double-buffered with cp.async, w/q single-buffered and refilled right after use.

namespace fla {

constexpr int kBT = 64, kKD = 128, kVD = 128, kNH = 32, kNHg = 16, kBV = 64, kThreads = 128;
constexpr float kLog2e = 1.4426950216293334961f;          // 0f3FB8AA3B
// shared memory (bytes)
constexpr int kSW = 0;          // w      [64][128] bf16, 256-byte rows, 16-byte XOR swizzle
constexpr int kSQ = 16384;      // q      [64][128]
constexpr int kSK = 32768;      // k   [2][64][128]
constexpr int kSU = 65536;      // u   [2][64][64]  bf16, 128-byte rows
constexpr int kSA = 81920;      // Ag     [64][64]
constexpr int kSO = 90112;      // o      [64][64]  (staging for coalesced stores)
constexpr int kSG = 98304;      // g   [2][64] f32
constexpr int kSmem = 98816;

__device__ __forceinline__ uint32_t swz256(int r, int c) {
  return (uint32_t)(r * 256 + ((((c >> 3) ^ (r & 7))) << 4) + ((c & 7) << 1));
}
__device__ __forceinline__ uint32_t swz128(int r, int c) {
  return (uint32_t)(r * 128 + ((((c >> 3) ^ (r & 7))) << 4) + ((c & 7) << 1));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
  const int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp4(uint32_t dst, const void* src, bool valid) {
  const int n = valid ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void stsm4t(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n"
               :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void sts128(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ float2 lds64f(uint32_t addr) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ float lds32f(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(addr) : "memory");
  return v;
}

// D += A(16x16 bf16) * B(16x8 bf16), fp32 accumulate (in place: the chain).
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// bf16x2 {lo, hi}, round-to-nearest-even (cvt.rn.bf16x2.f32 d, hi, lo)
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bflo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bfhi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Triton op.safe_exp: exp(where(x <= 0, x, -inf)) == ex2.approx(selp(x*log2e, -inf, x <= 0))
__device__ __forceinline__ float safe_exp_tr(float x) {
  float t = mul_rn(x, kLog2e);
  t = (x <= 0.f) ? t : __int_as_float(0xff800000u);
  return ex2_approx(t);
}
__device__ __forceinline__ float exp_tr(float x) { return ex2_approx(mul_rn(x, kLog2e)); }

// Recipe points (QWEN_FLA_NEGCTRL=1/2/3 builds deliberately break one of them; test-sensitivity
// controls only, never shipped).
#ifndef QWEN_FLA_NEGCTRL
#define QWEN_FLA_NEGCTRL 0
#endif
__device__ __forceinline__ float h_init(float x) {          // b_h = zeros; b_h += load
#if QWEN_FLA_NEGCTRL == 3
  return x;
#else
  return add_rn(0.f, x);
#endif
}
__device__ __forceinline__ float h_update(float h, float eg, float d) {   // fma.rn H, eg, D
#if QWEN_FLA_NEGCTRL == 1
  return add_rn(mul_rn(h, eg), d);
#else
  return fma_rn(h, eg, d);
#endif
}
__device__ __forceinline__ float o_final(float scale, float o1, float d2) {  // fma.rn scale, o1, scale*d2
#if QWEN_FLA_NEGCTRL == 2
  return add_rn(mul_rn(scale, o1), mul_rn(scale, d2));
#else
  return fma_rn(scale, o1, mul_rn(scale, d2));
#endif
}

// ---- async loads of one chunk ------------------------------------------------------------
__device__ __forceinline__ void load_wq(uint32_t sm, const __nv_bfloat16* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ q, int bos, int Tn, int c,
                                        int h, int hk, int tid) {
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    const int id = tid + kThreads * x, row = id >> 4, cc = id & 15;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kSW + swz256(row, cc * 8), w + (tt * kNH + h) * kKD + cc * 8, ok);
    cp16(sm + kSQ + swz256(row, cc * 8), q + (tt * kNHg + hk) * kKD + cc * 8, ok);
  }
}
__device__ __forceinline__ void load_kug(uint32_t sm, int buf, const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ u, const float* __restrict__ g,
                                         int bos, int Tn, int c, int h, int hk, int v0, int tid) {
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    const int id = tid + kThreads * x, row = id >> 4, cc = id & 15;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kSK + buf * 16384 + swz256(row, cc * 8), k + (tt * kNHg + hk) * kKD + cc * 8, ok);
  }
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    const int id = tid + kThreads * x, row = id >> 3, cc = id & 7;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kSU + buf * 8192 + swz128(row, cc * 8), u + (tt * kNH + h) * kVD + v0 + cc * 8, ok);
  }
  if (tid < kBT) {
    const int t = c * kBT + tid;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp4(sm + kSG + buf * 256 + tid * 4, g + tt * kNH + h, ok);
  }
}

// ---- A = q k^T tiles (16 x 8 each, K = 128), the 20 tiles of the 64x64 block that are not
//      entirely above the diagonal (those are where(i>=j, ., 0) == +0 and stay zero in smem).
//      Warp tile lists: w0 (3,0..4) | w1 (3,5..7)+(0,0..1) | w2 (2,0..4) | w3 (2,5)+(1,0..3)
template <int MT0, int NB0, int NC0, int MT1, int NB1, int NC1>
__device__ __forceinline__ void a_tiles(float (&acc)[5][4], uint32_t sQ, uint32_t sK, int lane) {
#pragma unroll
  for (int x = 0; x < 5; ++x) acc[x][0] = acc[x][1] = acc[x][2] = acc[x][3] = 0.f;
  const int ar = 8 * ((lane >> 3) & 1) + (lane & 7), ac = 8 * (lane >> 4);   // A-operand pattern
  const int br = 8 * (lane >> 4) + (lane & 7), bc = 8 * ((lane >> 3) & 1);   // B-operand pattern
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    uint32_t a0[4], a1[4];
    ldsm4(a0, sQ + swz256(16 * MT0 + ar, 16 * s + ac));
    if (NC1 > 0) ldsm4(a1, sQ + swz256(16 * MT1 + ar, 16 * s + ac));
#pragma unroll
    for (int p = 0; p < (NC0 + 1) / 2; ++p) {
      const int nt = NB0 + 2 * p;
      if (2 * p + 1 < NC0) {
        uint32_t b[4];
        ldsm4(b, sK + swz256(8 * nt + br, 16 * s + bc));
        mma(acc[2 * p], a0, b[0], b[1]);
        mma(acc[2 * p + 1], a0, b[2], b[3]);
      } else {
        uint32_t b[2];
        ldsm2(b, sK + swz256(8 * nt + (lane & 7), 16 * s + bc));
        mma(acc[2 * p], a0, b[0], b[1]);
      }
    }
    if (NC1 > 0) {
#pragma unroll
      for (int p = 0; p < (NC1 + 1) / 2; ++p) {
        const int nt = NB1 + 2 * p;
        if (2 * p + 1 < NC1) {
          uint32_t b[4];
          ldsm4(b, sK + swz256(8 * nt + br, 16 * s + bc));
          mma(acc[NC0 + 2 * p], a1, b[0], b[1]);
          mma(acc[NC0 + 2 * p + 1], a1, b[2], b[3]);
        } else {
          uint32_t b[2];
          ldsm2(b, sK + swz256(8 * nt + (lane & 7), 16 * s + bc));
          mma(acc[NC0 + 2 * p], a1, b[0], b[1]);
        }
      }
    }
  }
}

// Ag tile -> smem (bf16, [i][j]); g from smem (zero-filled past the sequence end like Triton).
__device__ __forceinline__ void a_gate_tile(const float (&acc)[4], int mt, int nt, uint32_t sA, uint32_t sG,
                                            int gq, int tq) {
  const int j0 = 8 * nt + 2 * tq;
  const float2 gj = lds64f(sG + 4 * j0);
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int i = 16 * mt + gq + 8 * hh;
    const float gi = lds32f(sG + 4 * i);
    // branch-free: masked lanes compute a discarded value, then select bf16 +0 (selp.b16 0x0000)
    const float v0 = mul_rn(acc[2 * hh], safe_exp_tr(sub_rn(gi, gj.x)));
    const float v1 = mul_rn(acc[2 * hh + 1], safe_exp_tr(sub_rn(gi, gj.y)));
    uint32_t pr = pk(v0, v1);
    pr &= (i >= j0 ? 0x0000ffffu : 0u) | (i >= j0 + 1 ? 0xffff0000u : 0u);
    sts32(sA + swz128(i, j0), pr);
  }
}

template <int MT0, int NB0, int NC0, int MT1, int NB1, int NC1>
__device__ __forceinline__ void a_gate(const float (&acc)[5][4], uint32_t sA, uint32_t sG, int gq, int tq) {
#pragma unroll
  for (int x = 0; x < NC0; ++x) a_gate_tile(acc[x], MT0, NB0 + x, sA, sG, gq, tq);
#pragma unroll
  for (int x = 0; x < NC1; ++x) a_gate_tile(acc[NC0 + x], MT1, NB1 + x, sA, sG, gq, tq);
}

__global__ void __launch_bounds__(kThreads, 1)
chunk_ho_fused_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                      const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ u,
                      const float* __restrict__ g, float* __restrict__ state,
                      const int32_t* __restrict__ state_idx, const int64_t* __restrict__ snap_rows,
                      const int64_t* __restrict__ snap_dsts, int num_snap,
                      const int64_t* __restrict__ cu_seqlens, const int64_t* __restrict__ chunk_offsets,
                      float scale, __nv_bfloat16* __restrict__ o) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sm = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const int vt = blockIdx.x, h = blockIdx.y, n = blockIdx.z, hk = h >> 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
  const int v0 = vt * kBV;
  const int bos = (int)cu_seqlens[n], eos = (int)cu_seqlens[n + 1];
  const int Tn = eos - bos;
  const int NT = (Tn + kBT - 1) / kBT;
  const int slot = state_idx[n];

  long long snap_chunk = 0, snap_dst = -1;   // Triton: sums over the owning snapshot rows
  if (num_snap > 0) {
    const long long boh = (long long)(int)chunk_offsets[n];
    long long sc = 0, sd = 0;
    for (int s = 0; s < num_snap; ++s) {
      const long long r = snap_rows[s], d = snap_dsts[s];
      if (r >= boh && r < boh + NT && d >= 0) { sc += r - boh; sd += d + 1; }
    }
    snap_chunk = sc;
    snap_dst = sd - 1;
  }

  // H^T accumulators: tile j covers k in [8j, 8j+8); [0],[1]: v = 16w+gq, k = 8j+2tq(+1);
  // [2],[3]: v = 16w+gq+8.
  float H[16][4];
  float* const hrow = state + (((long long)slot * kNH + h) * kVD + v0 + 16 * warp + gq) * kKD + 2 * tq;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float2 a = *reinterpret_cast<const float2*>(hrow + 8 * j);
    const float2 b = *reinterpret_cast<const float2*>(hrow + 8 * kKD + 8 * j);
    H[j][0] = h_init(a.x); H[j][1] = h_init(a.y);
    H[j][2] = h_init(b.x); H[j][3] = h_init(b.y);
  }

  if (NT > 0) {
    // prologue: chunk 0 (w,q | k,u,g), chunk 1 (k,u,g)
    load_wq(sm, w, q, bos, Tn, 0, h, hk, tid);
    cp_commit();
    load_kug(sm, 0, k, u, g, bos, Tn, 0, h, hk, v0, tid);
    cp_commit();
    if (NT > 1) load_kug(sm, 1, k, u, g, bos, Tn, 1, h, hk, v0, tid);
    cp_commit();
    // tiles of Ag entirely above the diagonal are +0 for every chunk
    for (int id = tid; id < 192; id += kThreads) {
      const int tile = id >> 4, r = id & 15;
      const int mt = tile < 6 ? 0 : (tile < 10 ? 1 : 2);
      const int nt = tile < 6 ? 2 + tile : (tile < 10 ? tile - 2 : tile - 4);
      sts128(sm + kSA + swz128(16 * mt + r, 8 * nt), make_uint4(0u, 0u, 0u, 0u));
    }
  }

  const int br = 8 * (lane >> 4) + (lane & 7), bc = 8 * ((lane >> 3) & 1);   // B-operand ldmatrix
  const int ar = 8 * ((lane >> 3) & 1) + (lane & 7), ac = 8 * (lane >> 4);   // A-operand ldmatrix
  const int vr = 16 * warp;

  for (int i = 0; i < NT; ++i) {
    const int cur = i & 1;
    const uint32_t sK = sm + kSK + cur * 16384, sU = sm + kSU + cur * 8192, sG = sm + kSG + cur * 256;
    cp_wait1();
    __syncthreads();                                                           // S1

    if (snap_dst >= 0 && (long long)i == snap_chunk) {                          // state at chunk start
      float* sp = state + ((snap_dst * kNH + h) * kVD + v0 + vr + gq) * kKD + 2 * tq;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        *reinterpret_cast<float2*>(sp + 8 * j) = make_float2(H[j][0], H[j][1]);
        *reinterpret_cast<float2*>(sp + 8 * kKD + 8 * j) = make_float2(H[j][2], H[j][3]);
      }
    }

    // ---- phase 1a: A = q k^T tiles -> gate/mask -> Ag (bf16) in smem ------------------------
    {
      float aacc[5][4];
      switch (warp) {
        case 0: a_tiles<3, 0, 5, 0, 0, 0>(aacc, sm + kSQ, sK, lane);
                a_gate<3, 0, 5, 0, 0, 0>(aacc, sm + kSA, sG, gq, tq); break;
        case 1: a_tiles<3, 5, 3, 0, 0, 2>(aacc, sm + kSQ, sK, lane);
                a_gate<3, 5, 3, 0, 0, 2>(aacc, sm + kSA, sG, gq, tq); break;
        case 2: a_tiles<2, 0, 5, 0, 0, 0>(aacc, sm + kSQ, sK, lane);
                a_gate<2, 0, 5, 0, 0, 0>(aacc, sm + kSA, sG, gq, tq); break;
        default: a_tiles<2, 5, 1, 1, 0, 4>(aacc, sm + kSQ, sK, lane);
                 a_gate<2, 5, 1, 1, 0, 4>(aacc, sm + kSA, sG, gq, tq); break;
      }
    }

    // ---- phase 1b: v_dot^T = Hb.w^T, qh^T = Hb.q^T (K = 128) --------------------------------
    float vd[8][4], qh[8][4];
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      vd[x][0] = vd[x][1] = vd[x][2] = vd[x][3] = 0.f;
      qh[x][0] = qh[x][1] = qh[x][2] = qh[x][3] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const uint32_t a[4] = {pk(H[2 * s][0], H[2 * s][1]), pk(H[2 * s][2], H[2 * s][3]),
                             pk(H[2 * s + 1][0], H[2 * s + 1][1]), pk(H[2 * s + 1][2], H[2 * s + 1][3])};
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bw[4], bq[4];
        ldsm4(bw, sm + kSW + swz256(16 * np + br, 16 * s + bc));
        ldsm4(bq, sm + kSQ + swz256(16 * np + br, 16 * s + bc));
        mma(vd[2 * np], a, bw[0], bw[1]);
        mma(vd[2 * np + 1], a, bw[2], bw[3]);
        mma(qh[2 * np], a, bq[0], bq[1]);
        mma(qh[2 * np + 1], a, bq[2], bq[3]);
      }
    }
    __syncthreads();                                                           // S2: w, q free; Ag visible
    if (i + 1 < NT) load_wq(sm, w, q, bos, Tn, i + 1, h, hk, tid);
    cp_commit();

    // ---- phase 2: v_new, vs ------------------------------------------------------------------
    const int L = min(kBT, Tn - kBT * i) - 1;
    const float glast = lds32f(sG + 4 * L);
    uint32_t vsf[4][4], vnf[4][4];   // A operands (M = v, K = t) of D^T and d2^T
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      uint32_t uu[4];
      ldsm4t(uu, sU + swz128(16 * np + br, vr + bc));
      float vn[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int nt = 2 * np + hh;
        vn[hh][0] = sub_rn(bflo(uu[2 * hh]), vd[nt][0]);
        vn[hh][1] = sub_rn(bfhi(uu[2 * hh]), vd[nt][1]);
        vn[hh][2] = sub_rn(bflo(uu[2 * hh + 1]), vd[nt][2]);
        vn[hh][3] = sub_rn(bfhi(uu[2 * hh + 1]), vd[nt][3]);
      }
      const float2 ga = lds64f(sG + 4 * (16 * np + 2 * tq));
      const float2 gb = lds64f(sG + 4 * (16 * np + 8 + 2 * tq));
      const float e00 = safe_exp_tr(sub_rn(glast, ga.x));
      const float e01 = safe_exp_tr(sub_rn(glast, ga.y));
      const float e10 = safe_exp_tr(sub_rn(glast, gb.x));
      const float e11 = safe_exp_tr(sub_rn(glast, gb.y));
      vnf[np][0] = pk(vn[0][0], vn[0][1]);
      vnf[np][1] = pk(vn[0][2], vn[0][3]);
      vnf[np][2] = pk(vn[1][0], vn[1][1]);
      vnf[np][3] = pk(vn[1][2], vn[1][3]);
      vsf[np][0] = pk(mul_rn(vn[0][0], e00), mul_rn(vn[0][1], e01));
      vsf[np][1] = pk(mul_rn(vn[0][2], e00), mul_rn(vn[0][3], e01));
      vsf[np][2] = pk(mul_rn(vn[1][0], e10), mul_rn(vn[1][1], e11));
      vsf[np][3] = pk(mul_rn(vn[1][2], e10), mul_rn(vn[1][3], e11));
    }
    // ---- phase 3: d2^T = v_new^T . Ag^T (K = 64), o; D^T = vs^T . k (K = 64), H update -------
    {
      float d2[8][4];
#pragma unroll
      for (int x = 0; x < 8; ++x) d2[x][0] = d2[x][1] = d2[x][2] = d2[x][3] = 0.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t b[4];
          ldsm4(b, sm + kSA + swz128(16 * np + br, 16 * s + bc));
          mma(d2[2 * np], vnf[s], b[0], b[1]);
          mma(d2[2 * np + 1], vnf[s], b[2], b[3]);
        }
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t r[4];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int nt = 2 * np + hh;
          const float2 gv = lds64f(sG + 4 * (8 * nt + 2 * tq));
          const float eg0 = exp_tr(gv.x), eg1 = exp_tr(gv.y);
          const float o0 = o_final(scale, mul_rn(qh[nt][0], eg0), d2[nt][0]);
          const float o1 = o_final(scale, mul_rn(qh[nt][1], eg1), d2[nt][1]);
          const float o2 = o_final(scale, mul_rn(qh[nt][2], eg0), d2[nt][2]);
          const float o3 = o_final(scale, mul_rn(qh[nt][3], eg1), d2[nt][3]);
          r[2 * hh] = pk(o0, o1);
          r[2 * hh + 1] = pk(o2, o3);
        }
        stsm4t(sm + kSO + swz128(16 * np + br, vr + bc), r[0], r[1], r[2], r[3]);
      }
    }
    {
      const float eg = exp_tr(glast);
#pragma unroll
      for (int jg = 0; jg < 4; ++jg) {
        float D[4][4];
#pragma unroll
        for (int x = 0; x < 4; ++x) D[x][0] = D[x][1] = D[x][2] = D[x][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
          for (int jp = 0; jp < 2; ++jp) {
            uint32_t b[4];
            ldsm4t(b, sK + swz256(16 * s + ar, 32 * jg + 16 * jp + ac));
            mma(D[2 * jp], vsf[s], b[0], b[1]);
            mma(D[2 * jp + 1], vsf[s], b[2], b[3]);
          }
        }
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          H[4 * jg + x][0] = h_update(H[4 * jg + x][0], eg, D[x][0]);
          H[4 * jg + x][1] = h_update(H[4 * jg + x][1], eg, D[x][1]);
          H[4 * jg + x][2] = h_update(H[4 * jg + x][2], eg, D[x][2]);
          H[4 * jg + x][3] = h_update(H[4 * jg + x][3], eg, D[x][3]);
        }
      }
    }
    __syncthreads();                                                           // S4: o staged; k/u/g[cur] free
    {
      const int rv = min(kBT, Tn - kBT * i);
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        const int id = tid + kThreads * x, row = id >> 3, cc = id & 7;
        if (row < rv) {
          const uint4 val = lds128(sm + kSO + swz128(row, cc * 8));
          *reinterpret_cast<uint4*>(o + ((long)(bos + kBT * i + row) * kNH + h) * kVD + v0 + cc * 8) = val;
        }
      }
    }
    if (i + 2 < NT) load_kug(sm, cur, k, u, g, bos, Tn, i + 2, h, hk, v0, tid);
    cp_commit();
  }

#pragma unroll
  for (int j = 0; j < 16; ++j) {
    *reinterpret_cast<float2*>(hrow + 8 * j) = make_float2(H[j][0], H[j][1]);
    *reinterpret_cast<float2*>(hrow + 8 * kKD + 8 * j) = make_float2(H[j][2], H[j][3]);
  }
}

// ==============================================================================================
// v2 (WU): recompute_w_u folded in front of chunk_h + chunk_o. Bit-exact to
//   recompute_w_u_fwd_kernel (Triton 3.6, BT64 BK=BV=64 nw4 ns3, IS_VARLEN; PTX 9.4):
//     eg_j = ex2.approx(g_j * log2e)                         (tl.exp of the loaded g)
//     kb   = bf16_rn((k * beta_j) * eg_j)                    mul.f32x2 beta, k; mul.f32x2 eg, .
//     vb   = bf16_rn(v * beta_j)                             mul.f32x2 beta, v
//     w    = bf16_rn(mma chain over K=64 (j), zero init)     A_wy . kb      (M=t, N=k)
//     u    = bf16_rn(mma chain over K=64 (j), zero init)     A_wy . vb      (here M=v, N=t)
//   Rows past the sequence end: recompute_w_u does not store them and chunk_h zero-fills its w/u
//   loads there, so w = u = +0 for those rows. Everything after that is the v1 recipe.
// Inputs: q, k [T,16,128], v [T,32,128] (raw values), A_wy [T,32,64] bf16 (kkt_solve output),
// beta, g [T,32] fp32. w and u never touch global memory.
// smem: w (computed; doubles as the o staging buffer), q, k[2], A_wy[2], v, Ag, g[2], beta[2].
// ==============================================================================================
constexpr int kW2SW = 0;         // w      [64][128] bf16 (256-byte rows) / o staging [64][64]
constexpr int kW2SQ = 16384;     // q      [64][128]
constexpr int kW2SK = 32768;     // k   [2][64][128]
constexpr int kW2SAW = 65536;    // A_wy[2][64][64]  bf16 (128-byte rows)
constexpr int kW2SV = 81920;     // v      [64][64]  (single buffer, this CTA's 64 V columns)
constexpr int kW2SA = 90112;     // Ag     [64][64]
constexpr int kW2SG = 98304;     // g   [2][64] f32
constexpr int kW2SB = 98816;     // beta[2][64] f32
constexpr int kW2Smem = 99328;

__device__ __forceinline__ void w2_load_q(uint32_t sm, const __nv_bfloat16* __restrict__ q, int bos, int Tn,
                                          int c, int hk, int tid) {
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    const int id = tid + kThreads * x, row = id >> 4, cc = id & 15;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kW2SQ + swz256(row, cc * 8), q + (tt * kNHg + hk) * kKD + cc * 8, ok);
  }
}
__device__ __forceinline__ void w2_load_v(uint32_t sm, const __nv_bfloat16* __restrict__ v, int bos, int Tn,
                                          int c, int h, int v0, int tid) {
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    const int id = tid + kThreads * x, row = id >> 3, cc = id & 7;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kW2SV + swz128(row, cc * 8), v + (tt * kNH + h) * kVD + v0 + cc * 8, ok);
  }
}
__device__ __forceinline__ void w2_load_kagb(uint32_t sm, int buf, const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ aw, const float* __restrict__ g,
                                             const float* __restrict__ beta, int bos, int Tn, int c, int h,
                                             int hk, int tid) {
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    const int id = tid + kThreads * x, row = id >> 4, cc = id & 15;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kW2SK + buf * 16384 + swz256(row, cc * 8), k + (tt * kNHg + hk) * kKD + cc * 8, ok);
  }
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    const int id = tid + kThreads * x, row = id >> 3, cc = id & 7;
    const int t = c * kBT + row;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    cp16(sm + kW2SAW + buf * 8192 + swz128(row, cc * 8), aw + (tt * kNH + h) * kBT + cc * 8, ok);
  }
  {
    const int r = tid & 63;
    const int t = c * kBT + r;
    const bool ok = t < Tn;
    const long tt = ok ? (long)(bos + t) : (long)bos;
    if (tid < 64) cp4(sm + kW2SG + buf * 256 + r * 4, g + tt * kNH + h, ok);
    else          cp4(sm + kW2SB + buf * 256 + r * 4, beta + tt * kNH + h, ok);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
chunk_wu_ho_fused_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                         const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ aw,
                         const float* __restrict__ beta, const float* __restrict__ g,
                         float* __restrict__ state, const int32_t* __restrict__ state_idx,
                         const int64_t* __restrict__ snap_rows, const int64_t* __restrict__ snap_dsts,
                         int num_snap, const int64_t* __restrict__ cu_seqlens,
                         const int64_t* __restrict__ chunk_offsets, float scale, __nv_bfloat16* __restrict__ o) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sm = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const int vt = blockIdx.x, h = blockIdx.y, n = blockIdx.z, hk = h >> 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
  const int v0 = vt * kBV;
  const int bos = (int)cu_seqlens[n], eos = (int)cu_seqlens[n + 1];
  const int Tn = eos - bos;
  const int NT = (Tn + kBT - 1) / kBT;
  const int slot = state_idx[n];

  long long snap_chunk = 0, snap_dst = -1;
  if (num_snap > 0) {
    const long long boh = (long long)(int)chunk_offsets[n];
    long long sc = 0, sd = 0;
    for (int s = 0; s < num_snap; ++s) {
      const long long r = snap_rows[s], d = snap_dsts[s];
      if (r >= boh && r < boh + NT && d >= 0) { sc += r - boh; sd += d + 1; }
    }
    snap_chunk = sc;
    snap_dst = sd - 1;
  }

  float H[16][4];
  float* const hrow = state + (((long long)slot * kNH + h) * kVD + v0 + 16 * warp + gq) * kKD + 2 * tq;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float2 a = *reinterpret_cast<const float2*>(hrow + 8 * j);
    const float2 b = *reinterpret_cast<const float2*>(hrow + 8 * kKD + 8 * j);
    H[j][0] = h_init(a.x); H[j][1] = h_init(a.y);
    H[j][2] = h_init(b.x); H[j][3] = h_init(b.y);
  }

  if (NT > 0) {
    // prologue, commit order [q0][k,A,g,b 0][v0][k,A,g,b 1]; per chunk i then: [q i+1] [v i+1] [kAgb i+2]
    w2_load_q(sm, q, bos, Tn, 0, hk, tid);
    cp_commit();
    w2_load_kagb(sm, 0, k, aw, g, beta, bos, Tn, 0, h, hk, tid);
    cp_commit();
    w2_load_v(sm, v, bos, Tn, 0, h, v0, tid);
    cp_commit();
    if (NT > 1) w2_load_kagb(sm, 1, k, aw, g, beta, bos, Tn, 1, h, hk, tid);
    cp_commit();
    for (int id = tid; id < 192; id += kThreads) {
      const int tile = id >> 4, r = id & 15;
      const int mt = tile < 6 ? 0 : (tile < 10 ? 1 : 2);
      const int nt = tile < 6 ? 2 + tile : (tile < 10 ? tile - 2 : tile - 4);
      sts128(sm + kW2SA + swz128(16 * mt + r, 8 * nt), make_uint4(0u, 0u, 0u, 0u));
    }
  }

  const int br = 8 * (lane >> 4) + (lane & 7), bc = 8 * ((lane >> 3) & 1);   // B-operand ldmatrix
  const int ar = 8 * ((lane >> 3) & 1) + (lane & 7), ac = 8 * (lane >> 4);   // A-operand ldmatrix
  const int vr = 16 * warp;

  for (int i = 0; i < NT; ++i) {
    const int cur = i & 1;
    const uint32_t sK = sm + kW2SK + cur * 16384, sAw = sm + kW2SAW + cur * 8192;
    const uint32_t sG = sm + kW2SG + cur * 256, sB = sm + kW2SB + cur * 256;
    const int rv = min(kBT, Tn - kBT * i);
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    __syncthreads();                                                           // S1

    if (snap_dst >= 0 && (long long)i == snap_chunk) {
      float* sp = state + ((snap_dst * kNH + h) * kVD + v0 + vr + gq) * kKD + 2 * tq;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        *reinterpret_cast<float2*>(sp + 8 * j) = make_float2(H[j][0], H[j][1]);
        *reinterpret_cast<float2*>(sp + 8 * kKD + 8 * j) = make_float2(H[j][2], H[j][3]);
      }
    }

    // ---- phase 0: w[:, 32*warp .. +32) = A_wy . kb  (M = t, N = k, K = j) -> bf16 -> smem -----
    //      two passes of 16 columns (np), each chained over the 4 k-steps (j) in order
#pragma unroll 1
    for (int np = 0; np < 2; ++np) {
      float wa[4][2][4];
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) wa[a][b][0] = wa[a][b][1] = wa[a][b][2] = wa[a][b][3] = 0.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int jA = 16 * s + 2 * tq, jB = jA + 8;
        const float2 bA = lds64f(sB + 4 * jA), bB = lds64f(sB + 4 * jB);
        const float2 gA = lds64f(sG + 4 * jA), gB = lds64f(sG + 4 * jB);
        const float eA0 = exp_tr(gA.x), eA1 = exp_tr(gA.y), eB0 = exp_tr(gB.x), eB1 = exp_tr(gB.y);
        uint32_t kr[4];
        ldsm4t(kr, sK + swz256(16 * s + ar, 32 * warp + 16 * np + ac));
        // kr[0],kr[2]: j = jA, jA+1 ; kr[1],kr[3]: j = jB, jB+1 (lo, hi halves)
        const uint32_t sb0 = pk(mul_rn(mul_rn(bflo(kr[0]), bA.x), eA0), mul_rn(mul_rn(bfhi(kr[0]), bA.y), eA1));
        const uint32_t sb1 = pk(mul_rn(mul_rn(bflo(kr[1]), bB.x), eB0), mul_rn(mul_rn(bfhi(kr[1]), bB.y), eB1));
        const uint32_t sb2 = pk(mul_rn(mul_rn(bflo(kr[2]), bA.x), eA0), mul_rn(mul_rn(bfhi(kr[2]), bA.y), eA1));
        const uint32_t sb3 = pk(mul_rn(mul_rn(bflo(kr[3]), bB.x), eB0), mul_rn(mul_rn(bfhi(kr[3]), bB.y), eB1));
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          uint32_t af[4];
          ldsm4(af, sAw + swz128(16 * mt + ar, 16 * s + ac));
          mma(wa[mt][0], af, sb0, sb1);
          mma(wa[mt][1], af, sb2, sb3);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const int t0 = 16 * mt + gq, t1 = t0 + 8;
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int kk = 32 * warp + 16 * np + 8 * x + 2 * tq;
          sts32(sm + kW2SW + swz256(t0, kk), t0 < rv ? pk(wa[mt][x][0], wa[mt][x][1]) : 0u);
          sts32(sm + kW2SW + swz256(t1, kk), t1 < rv ? pk(wa[mt][x][2], wa[mt][x][3]) : 0u);
        }
      }
    }

    // ---- phase 1a: A = q k^T tiles -> gate/mask -> Ag ----------------------------------------
    {
      float aacc[5][4];
      switch (warp) {
        case 0: a_tiles<3, 0, 5, 0, 0, 0>(aacc, sm + kW2SQ, sK, lane);
                a_gate<3, 0, 5, 0, 0, 0>(aacc, sm + kW2SA, sG, gq, tq); break;
        case 1: a_tiles<3, 5, 3, 0, 0, 2>(aacc, sm + kW2SQ, sK, lane);
                a_gate<3, 5, 3, 0, 0, 2>(aacc, sm + kW2SA, sG, gq, tq); break;
        case 2: a_tiles<2, 0, 5, 0, 0, 0>(aacc, sm + kW2SQ, sK, lane);
                a_gate<2, 0, 5, 0, 0, 0>(aacc, sm + kW2SA, sG, gq, tq); break;
        default: a_tiles<2, 5, 1, 1, 0, 4>(aacc, sm + kW2SQ, sK, lane);
                 a_gate<2, 5, 1, 1, 0, 4>(aacc, sm + kW2SA, sG, gq, tq); break;
      }
    }
    __syncthreads();                                                           // S1': w, Ag complete

    // ---- phase 1b: v_dot^T = Hb.w^T, qh^T = Hb.q^T (K = 128) --------------------------------
    float vd[8][4], qh[8][4];
#pragma unroll
    for (int x = 0; x < 8; ++x) {
      vd[x][0] = vd[x][1] = vd[x][2] = vd[x][3] = 0.f;
      qh[x][0] = qh[x][1] = qh[x][2] = qh[x][3] = 0.f;
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const uint32_t a[4] = {pk(H[2 * s][0], H[2 * s][1]), pk(H[2 * s][2], H[2 * s][3]),
                             pk(H[2 * s + 1][0], H[2 * s + 1][1]), pk(H[2 * s + 1][2], H[2 * s + 1][3])};
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bw[4], bq[4];
        ldsm4(bw, sm + kW2SW + swz256(16 * np + br, 16 * s + bc));
        ldsm4(bq, sm + kW2SQ + swz256(16 * np + br, 16 * s + bc));
        mma(vd[2 * np], a, bw[0], bw[1]);
        mma(vd[2 * np + 1], a, bw[2], bw[3]);
        mma(qh[2 * np], a, bq[0], bq[1]);
        mma(qh[2 * np + 1], a, bq[2], bq[3]);
      }
    }
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");                     // v(i) landed
    __syncthreads();                                                           // S2: q, w free; v visible
    if (i + 1 < NT) w2_load_q(sm, q, bos, Tn, i + 1, hk, tid);
    cp_commit();

    // ---- phase 2: u^T = vb^T . A_wy^T (M = v, N = t, K = j); v_new, vs ------------------------
    float ua[8][4];
#pragma unroll
    for (int x = 0; x < 8; ++x) ua[x][0] = ua[x][1] = ua[x][2] = ua[x][3] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t vv[4];
      ldsm4t(vv, sm + kW2SV + swz128(16 * s + br, vr + bc));
      const float2 bA = lds64f(sB + 4 * (16 * s + 2 * tq)), bB = lds64f(sB + 4 * (16 * s + 8 + 2 * tq));
      const uint32_t af[4] = {pk(mul_rn(bflo(vv[0]), bA.x), mul_rn(bfhi(vv[0]), bA.y)),
                              pk(mul_rn(bflo(vv[1]), bA.x), mul_rn(bfhi(vv[1]), bA.y)),
                              pk(mul_rn(bflo(vv[2]), bB.x), mul_rn(bfhi(vv[2]), bB.y)),
                              pk(mul_rn(bflo(vv[3]), bB.x), mul_rn(bfhi(vv[3]), bB.y))};
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        ldsm4(b, sAw + swz128(16 * np + br, 16 * s + bc));
        mma(ua[2 * np], af, b[0], b[1]);
        mma(ua[2 * np + 1], af, b[2], b[3]);
      }
    }
    const int L = rv - 1;
    const float glast = lds32f(sG + 4 * L);
    uint32_t vsf[4][4], vnf[4][4];
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      float vn[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int nt = 2 * np + hh;
        const int t0 = 8 * nt + 2 * tq;
        const uint32_t u01 = pk(ua[nt][0], ua[nt][1]), u23 = pk(ua[nt][2], ua[nt][3]);
        const float uv0 = t0 < rv ? bflo(u01) : 0.f, uv1 = t0 + 1 < rv ? bfhi(u01) : 0.f;
        const float uv2 = t0 < rv ? bflo(u23) : 0.f, uv3 = t0 + 1 < rv ? bfhi(u23) : 0.f;
        vn[hh][0] = sub_rn(uv0, vd[nt][0]);
        vn[hh][1] = sub_rn(uv1, vd[nt][1]);
        vn[hh][2] = sub_rn(uv2, vd[nt][2]);
        vn[hh][3] = sub_rn(uv3, vd[nt][3]);
      }
      const float2 ga = lds64f(sG + 4 * (16 * np + 2 * tq));
      const float2 gb = lds64f(sG + 4 * (16 * np + 8 + 2 * tq));
      const float e00 = safe_exp_tr(sub_rn(glast, ga.x));
      const float e01 = safe_exp_tr(sub_rn(glast, ga.y));
      const float e10 = safe_exp_tr(sub_rn(glast, gb.x));
      const float e11 = safe_exp_tr(sub_rn(glast, gb.y));
      vnf[np][0] = pk(vn[0][0], vn[0][1]);
      vnf[np][1] = pk(vn[0][2], vn[0][3]);
      vnf[np][2] = pk(vn[1][0], vn[1][1]);
      vnf[np][3] = pk(vn[1][2], vn[1][3]);
      vsf[np][0] = pk(mul_rn(vn[0][0], e00), mul_rn(vn[0][1], e01));
      vsf[np][1] = pk(mul_rn(vn[0][2], e00), mul_rn(vn[0][3], e01));
      vsf[np][2] = pk(mul_rn(vn[1][0], e10), mul_rn(vn[1][1], e11));
      vsf[np][3] = pk(mul_rn(vn[1][2], e10), mul_rn(vn[1][3], e11));
    }

    // ---- phase 3: d2^T, o (staged in the w buffer); D^T, H update ------------------------------
    {
      float d2[8][4];
#pragma unroll
      for (int x = 0; x < 8; ++x) d2[x][0] = d2[x][1] = d2[x][2] = d2[x][3] = 0.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t b[4];
          ldsm4(b, sm + kW2SA + swz128(16 * np + br, 16 * s + bc));
          mma(d2[2 * np], vnf[s], b[0], b[1]);
          mma(d2[2 * np + 1], vnf[s], b[2], b[3]);
        }
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t r[4];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int nt = 2 * np + hh;
          const float2 gv = lds64f(sG + 4 * (8 * nt + 2 * tq));
          const float eg0 = exp_tr(gv.x), eg1 = exp_tr(gv.y);
          const float o0 = o_final(scale, mul_rn(qh[nt][0], eg0), d2[nt][0]);
          const float o1 = o_final(scale, mul_rn(qh[nt][1], eg1), d2[nt][1]);
          const float o2 = o_final(scale, mul_rn(qh[nt][2], eg0), d2[nt][2]);
          const float o3 = o_final(scale, mul_rn(qh[nt][3], eg1), d2[nt][3]);
          r[2 * hh] = pk(o0, o1);
          r[2 * hh + 1] = pk(o2, o3);
        }
        stsm4t(sm + kW2SW + swz128(16 * np + br, vr + bc), r[0], r[1], r[2], r[3]);
      }
    }
    {
      const float eg = exp_tr(glast);
#pragma unroll
      for (int jg = 0; jg < 4; ++jg) {
        float D[4][4];
#pragma unroll
        for (int x = 0; x < 4; ++x) D[x][0] = D[x][1] = D[x][2] = D[x][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
          for (int jp = 0; jp < 2; ++jp) {
            uint32_t b[4];
            ldsm4t(b, sK + swz256(16 * s + ar, 32 * jg + 16 * jp + ac));
            mma(D[2 * jp], vsf[s], b[0], b[1]);
            mma(D[2 * jp + 1], vsf[s], b[2], b[3]);
          }
        }
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          H[4 * jg + x][0] = h_update(H[4 * jg + x][0], eg, D[x][0]);
          H[4 * jg + x][1] = h_update(H[4 * jg + x][1], eg, D[x][1]);
          H[4 * jg + x][2] = h_update(H[4 * jg + x][2], eg, D[x][2]);
          H[4 * jg + x][3] = h_update(H[4 * jg + x][3], eg, D[x][3]);
        }
      }
    }
    __syncthreads();                                                           // S4
    {
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        const int id = tid + kThreads * x, row = id >> 3, cc = id & 7;
        if (row < rv) {
          const uint4 val = lds128(sm + kW2SW + swz128(row, cc * 8));
          *reinterpret_cast<uint4*>(o + ((long)(bos + kBT * i + row) * kNH + h) * kVD + v0 + cc * 8) = val;
        }
      }
    }
    if (i + 1 < NT) w2_load_v(sm, v, bos, Tn, i + 1, h, v0, tid);
    cp_commit();
    if (i + 2 < NT) w2_load_kagb(sm, cur, k, aw, g, beta, bos, Tn, i + 2, h, hk, tid);
    cp_commit();
  }

#pragma unroll
  for (int j = 0; j < 16; ++j) {
    *reinterpret_cast<float2*>(hrow + 8 * j) = make_float2(H[j][0], H[j][1]);
    *reinterpret_cast<float2*>(hrow + 8 * kKD + 8 * j) = make_float2(H[j][2], H[j][3]);
  }
}

}  // namespace fla
