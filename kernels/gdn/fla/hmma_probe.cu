// HMMA bf16 -> f32 (mma.sync m16n8k16) numerical probe on SM120.
// Questions (all answered bit-wise on random data with wide exponent spread):
//  Q1 orientation : is d[m][n] of chain(A=X, B=Y^T) identical to d^T[n][m] of chain(A=Y, B=X^T)?
//  Q2 zero step   : does an extra mma step whose products are all zero leave the accumulator unchanged?
//  Q3 slot perm   : does permuting the k->slot assignment inside one instruction (same perm for A and B)
//                   change the result (only the K-set per instruction matters)?
//  Q4 K grouping  : sanity: a different K-set grouping (k-steps of 16 non-consecutive k) DOES change it.
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a hmma_probe.cu -o hmma_probe
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>

__device__ __forceinline__ void mma16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }

// One warp per 16x8 output tile. X: [M][K] bf16 bits, Y: [N][K] bf16 bits, C: [M][N] f32 (or null => 0).
// kmap: [K] actual k index used for "slot" k (slot k of step s = kmap[16 s + k]); identity = natural.
// zero_extra: append one extra mma step with all-zero A operand at the end of the chain.
__global__ void chain_kernel(const uint16_t* X, const uint16_t* Y, const float* C, float* D,
                             int M, int N, int K, const int* kmap, int zero_extra) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  int tilesN = N / 8;
  int mt = warp / tilesN, nt = warp % tilesN;
  if (mt * 16 >= M) return;
  int g = lane >> 2, t = lane & 3;
  int r0 = mt * 16 + g, r1 = r0 + 8, cn = nt * 8 + g;
  float acc[4];
  int c0 = nt * 8 + 2 * t;
  if (C) {
    acc[0] = C[r0 * N + c0]; acc[1] = C[r0 * N + c0 + 1];
    acc[2] = C[r1 * N + c0]; acc[3] = C[r1 * N + c0 + 1];
  } else { acc[0] = acc[1] = acc[2] = acc[3] = 0.f; }
  for (int s = 0; s < K / 16; ++s) {
    const int* km = kmap + 16 * s;
    uint32_t a[4], b[2];
    a[0] = pack2(X[r0 * K + km[2 * t]],     X[r0 * K + km[2 * t + 1]]);
    a[1] = pack2(X[r1 * K + km[2 * t]],     X[r1 * K + km[2 * t + 1]]);
    a[2] = pack2(X[r0 * K + km[2 * t + 8]], X[r0 * K + km[2 * t + 9]]);
    a[3] = pack2(X[r1 * K + km[2 * t + 8]], X[r1 * K + km[2 * t + 9]]);
    b[0] = pack2(Y[cn * K + km[2 * t]],     Y[cn * K + km[2 * t + 1]]);
    b[1] = pack2(Y[cn * K + km[2 * t + 8]], Y[cn * K + km[2 * t + 9]]);
    float o[4];
    mma16816(o, a, b, acc);
    acc[0] = o[0]; acc[1] = o[1]; acc[2] = o[2]; acc[3] = o[3];
  }
  if (zero_extra) {
    uint32_t a[4] = {0u, 0u, 0u, 0u}, b[2];
    b[0] = pack2(Y[cn * K + 0], Y[cn * K + 1]);
    b[1] = pack2(Y[cn * K + 2], Y[cn * K + 3]);
    float o[4];
    mma16816(o, a, b, acc);
    acc[0] = o[0]; acc[1] = o[1]; acc[2] = o[2]; acc[3] = o[3];
  }
  D[r0 * N + c0] = acc[0]; D[r0 * N + c0 + 1] = acc[1];
  D[r1 * N + c0] = acc[2]; D[r1 * N + c0 + 1] = acc[3];
}

// Signed-zero probe: one mma on a single 16x8 tile with given C and A, B.
//  case 0: C = -0, A = 0 (all +0), B random      -> does -0 survive an all-zero-product step?
//  case 1: C = +0, products all -0 (A = +0, B < 0) -> +0 or -0?
//  case 2: C = -0, products all -0                 -> -0 expected (IEEE)
//  case 3: C = +0, A = -0, B > 0 (products -0)
__global__ void zero_sign_kernel(uint32_t* out) {
  const int lane = threadIdx.x & 31;
  const uint16_t neg_one = 0xBF80, pos_one = 0x3F80, nz = 0x8000, pz = 0x0000;
  for (int cs = 0; cs < 4; ++cs) {
    float c0 = (cs == 0 || cs == 2) ? -0.f : 0.f;
    uint16_t av = (cs == 3) ? nz : pz;
    uint16_t bv = (cs == 3) ? pos_one : neg_one;
    uint32_t a[4] = {pack2(av, av), pack2(av, av), pack2(av, av), pack2(av, av)};
    uint32_t b[2] = {pack2(bv, bv), pack2(bv, bv)};
    float c[4] = {c0, c0, c0, c0}, d[4];
    mma16816(d, a, b, c);
    if (lane == 0) out[cs] = __float_as_uint(d[0]);
  }
}

static uint16_t f2bf(float f) {  // RN-even
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t r = u + 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(r >> 16);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

struct Dev {
  uint16_t *X, *Y; float *C, *D1, *D2; int* km;
};

static void run(const Dev& d, const uint16_t* X, const uint16_t* Y, const float* C, int M, int N, int K,
                const std::vector<int>& kmap, int zero_extra, std::vector<float>& out) {
  CK(cudaMemcpy(d.X, X, sizeof(uint16_t) * M * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.Y, Y, sizeof(uint16_t) * N * K, cudaMemcpyHostToDevice));
  if (C) CK(cudaMemcpy(d.C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.km, kmap.data(), sizeof(int) * K, cudaMemcpyHostToDevice));
  int warps = (M / 16) * (N / 8);
  chain_kernel<<<(warps * 32 + 127) / 128, 128>>>(d.X, d.Y, C ? d.C : nullptr, d.D1, M, N, K, d.km, zero_extra);
  CK(cudaGetLastError());
  out.resize((size_t)M * N);
  CK(cudaMemcpy(out.data(), d.D1, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
}

static int count_diff(const std::vector<float>& a, const std::vector<float>& b, bool transposed, int M, int N) {
  int bad = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float x = a[(size_t)i * N + j];
      float y = transposed ? b[(size_t)j * M + i] : b[(size_t)i * N + j];
      uint32_t ux, uy; memcpy(&ux, &x, 4); memcpy(&uy, &y, 4);
      if (ux != uy) ++bad;
    }
  return bad;
}

int main(int argc, char** argv) {
  const int M = 128, N = 128, K = 128;
  int trials = argc > 1 ? atoi(argv[1]) : 50;
  std::mt19937_64 rng(1234);
  Dev d;
  CK(cudaMalloc(&d.X, sizeof(uint16_t) * M * K)); CK(cudaMalloc(&d.Y, sizeof(uint16_t) * N * K));
  CK(cudaMalloc(&d.C, sizeof(float) * M * N)); CK(cudaMalloc(&d.D1, sizeof(float) * M * N));
  CK(cudaMalloc(&d.D2, sizeof(float) * M * N)); CK(cudaMalloc(&d.km, sizeof(int) * K));
  std::vector<uint16_t> X(M * K), Y(N * K);
  std::vector<float> C(M * N), Ct(M * N);
  std::vector<int> ident(K), perm_half(K), perm_pairs(K), perm_rand(K), group_alt(K);
  for (int k = 0; k < K; ++k) ident[k] = k;
  for (int s = 0; s < K / 16; ++s)
    for (int k = 0; k < 16; ++k) {
      perm_half[16 * s + k] = 16 * s + ((k + 8) & 15);                 // swap the two 8-halves
      perm_pairs[16 * s + k] = 16 * s + (k ^ 2);                        // swap neighbouring pairs
      group_alt[16 * s + k] = (k < 8) ? (8 * s + k) : (64 + 8 * s + k - 8);  // different K-set per step
    }
  long tot[8] = {0};
  long cmp[8] = {0};
  for (int tr = 0; tr < trials; ++tr) {
    // random permutation within each step (same for A and B)
    for (int s = 0; s < K / 16; ++s) {
      int p[16]; for (int k = 0; k < 16; ++k) p[k] = k;
      for (int k = 15; k > 0; --k) { int j = rng() % (k + 1); int tmp = p[k]; p[k] = p[j]; p[j] = tmp; }
      for (int k = 0; k < 16; ++k) perm_rand[16 * s + k] = 16 * s + p[k];
    }
    int mode = tr % 3;  // 0: wide exponents, 1: narrow (cancellation-heavy), 2: mixed with zeros
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::uniform_int_distribution<int> E(-20, 20);
    auto gen = [&](void) -> float {
      float v = U(rng);
      if (mode == 0) v = ldexpf(v, E(rng));
      if (mode == 2 && (rng() % 4 == 0)) v = 0.f;
      return v;
    };
    for (auto& x : X) x = f2bf(gen());
    for (auto& y : Y) y = f2bf(gen());
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
      float c = (tr & 4) ? 0.f : gen() * 4.f;
      C[i * N + j] = c; Ct[j * M + i] = c;
    }
    std::vector<float> d1, d2, d3, d4, d5, d6, d7;
    run(d, X.data(), Y.data(), C.data(), M, N, K, ident, 0, d1);
    run(d, Y.data(), X.data(), Ct.data(), N, M, K, ident, 0, d2);   // transposed problem
    tot[0] += count_diff(d1, d2, true, M, N); cmp[0] += M * N;
    run(d, X.data(), Y.data(), C.data(), M, N, K, ident, 1, d3);    // + zero step
    tot[1] += count_diff(d1, d3, false, M, N); cmp[1] += M * N;
    run(d, X.data(), Y.data(), C.data(), M, N, K, perm_half, 0, d4);
    tot[2] += count_diff(d1, d4, false, M, N); cmp[2] += M * N;
    run(d, X.data(), Y.data(), C.data(), M, N, K, perm_pairs, 0, d5);
    tot[3] += count_diff(d1, d5, false, M, N); cmp[3] += M * N;
    run(d, X.data(), Y.data(), C.data(), M, N, K, perm_rand, 0, d6);
    tot[4] += count_diff(d1, d6, false, M, N); cmp[4] += M * N;
    run(d, X.data(), Y.data(), C.data(), M, N, K, group_alt, 0, d7);
    tot[5] += count_diff(d1, d7, false, M, N); cmp[5] += M * N;
  }
  const char* names[6] = {"Q1 orientation (transposed problem)", "Q2 extra all-zero step",
                          "Q3a swap 8-halves in-instruction", "Q3b swap pairs in-instruction",
                          "Q3c random in-instruction perm", "Q4 different K-set grouping (control)"};
  for (int i = 0; i < 6; ++i) printf("%-40s mismatches %ld / %ld\n", names[i], tot[i], cmp[i]);
  printf("{\"orientation\":%ld,\"zero_step\":%ld,\"swap_halves\":%ld,\"swap_pairs\":%ld,\"rand_perm\":%ld,\"kset_control\":%ld,\"compared\":%ld}\n",
         tot[0], tot[1], tot[2], tot[3], tot[4], tot[5], cmp[0]);
  uint32_t* zs; CK(cudaMalloc(&zs, 16));
  zero_sign_kernel<<<1, 32>>>(zs);
  uint32_t hz[4]; CK(cudaMemcpy(hz, zs, 16, cudaMemcpyDeviceToHost));
  printf("zero-sign: C=-0,A=0 -> %08x | C=+0,prod=-0 -> %08x | C=-0,prod=-0 -> %08x | C=+0,A=-0,B>0 -> %08x\n",
         hz[0], hz[1], hz[2], hz[3]);
  return 0;
}
