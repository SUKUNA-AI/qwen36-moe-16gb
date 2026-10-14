#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

// ABI v1: pointers refer to CUDA tensors. Strides count tensor elements.
struct Params {
  const __nv_bfloat16 *q, *k, *v, *gate, *qw, *kw;
  const float *rope;
  const void *positions, *locations;
  __nv_bfloat16 *qo, *go;
  void *kc, *vc;
  unsigned long long *errors;
  int64_t tokens, qs, qh, ks, kh, vs, vh, gs, gh;
  int64_t capacity, rope_rows;
  float epsilon, kscale, vscale;
  int32_t fp8, pos64, loc64, reserved;
};

__device__ __forceinline__ int64_t index_at(const void *p, int64_t i, int wide) {
  return wide ? static_cast<const int64_t *>(p)[i] : static_cast<const int32_t *>(p)[i];
}

template<bool FP8, int WARPS>
__global__ void qwen36_qknorm_rope_kv(Params p) {
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5);
  if (row >= p.tokens * 18) return;
  const int64_t token = row / 18;
  const int head = row % 18;
  const bool is_q = head < 16;
  const int h = is_q ? head : head - 16;
  const int64_t pos = index_at(p.positions, token, p.pos64);
  const int64_t loc = index_at(p.locations, token, p.loc64);
  if (pos < 0 || pos >= p.rope_rows || loc < 0 || loc >= p.capacity) {
    if (lane == 0 && head == 0 && p.errors) atomicAdd(p.errors, 1ULL);
    return;
  }
  const __nv_bfloat16 *input = is_q ? p.q + token * p.qs + h * p.qh : p.k + token * p.ks + h * p.kh;
  const __nv_bfloat16 *weight = is_q ? p.qw : p.kw;
  float x[8];
  float sum = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    x[i] = __bfloat162float(input[lane + i * 32]);
    sum += x[i] * x[i];
  }
  #pragma unroll
  for (int d = 16; d > 0; d >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, d);
  const float inv = rsqrtf(sum * (1.f / 256.f) + p.epsilon);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Preserve the unfused RMSNorm's BF16 boundary before RoPE.
    x[i] = __bfloat162float(__float2bfloat16_rn((x[i] * inv) * __bfloat162float(weight[lane + i * 32])));
  }
  const float c = p.rope[pos * 64 + lane];
  const float s = p.rope[pos * 64 + 32 + lane];
  const float lo = x[0], hi = x[1];
  x[0] = __bfloat162float(__float2bfloat16_rn(lo * c - hi * s));
  x[1] = __bfloat162float(__float2bfloat16_rn(hi * c + lo * s));
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int dim = lane + 32 * i;
    if (is_q) {
      const int64_t out = (token * 16 + h) * 256 + dim;
      p.qo[out] = __float2bfloat16_rn(x[i]);
      if (p.go && p.gate) p.go[out] = p.gate[token * p.gs + h * p.gh + dim];
    } else {
      const int64_t out = (loc * 2 + h) * 256 + dim;
      const __nv_bfloat16 val = p.v[token * p.vs + h * p.vh + dim];
      if constexpr (FP8) {
        const float fk = x[i] * p.kscale;
        const float fv = __bfloat162float(val) * p.vscale;
        // Finite saturation matches the existing experimental store_fp8 contract.
        static_cast<__nv_fp8_storage_t *>(p.kc)[out] = __nv_cvt_float_to_fp8(fminf(448.f, fmaxf(-448.f, fk)), __NV_SATFINITE, __NV_E4M3);
        static_cast<__nv_fp8_storage_t *>(p.vc)[out] = __nv_cvt_float_to_fp8(fminf(448.f, fmaxf(-448.f, fv)), __NV_SATFINITE, __NV_E4M3);
        if (p.errors && (fabsf(fk) > 448.f || fabsf(fv) > 448.f)) atomicAdd(p.errors + 1, 1ULL);
      } else {
        static_cast<__nv_bfloat16 *>(p.kc)[out] = __float2bfloat16_rn(x[i]);
        static_cast<__nv_bfloat16 *>(p.vc)[out] = val;
      }
    }
  }
}

extern "C" int qwen36_kv_abi_size() { return sizeof(Params); }
extern "C" int qwen36_kv_launch(const Params *p, void *stream, int warps) {
  if (!p || !p->q || !p->k || !p->v || !p->qw || !p->kw || !p->rope || !p->positions || !p->locations || !p->qo || !p->kc || !p->vc) return cudaErrorInvalidValue;
  if (p->tokens < 0 || p->tokens > 1048576 || p->capacity <= 0 || p->rope_rows <= 0 || p->epsilon <= 0 || !std::isfinite(p->epsilon)) return cudaErrorInvalidValue;
  if (p->fp8 && (!(p->kscale > 0) || !(p->vscale > 0) || !std::isfinite(p->kscale) || !std::isfinite(p->vscale))) return cudaErrorInvalidValue;
  if (p->tokens == 0) return cudaSuccess;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  #define LAUNCH(W) do { \
    const unsigned blocks = static_cast<unsigned>((p->tokens * 18 + W - 1) / W); \
    if (p->fp8) qwen36_qknorm_rope_kv<true,W><<<blocks, W*32, 0, s>>>(*p); \
    else qwen36_qknorm_rope_kv<false,W><<<blocks, W*32, 0, s>>>(*p); \
  } while (0)
  if (warps == 1) LAUNCH(1);
  else if (warps == 4) LAUNCH(4);
  else if (warps == 8) LAUNCH(8);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

extern "C" int64_t qwen36_kv_offset(int64_t loc, int head, int dim) {
  return (loc * int64_t{2} + head) * int64_t{256} + dim;
}
