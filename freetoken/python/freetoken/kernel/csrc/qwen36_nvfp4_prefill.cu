#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <mma.h>
#include <cstdint>

// Standalone C ABI: no PyTorch ABI or runtime allocation in the launch path.
// Build only for the owner's SM120 GPU. All pointer arithmetic is 64 bit.
struct QwenNvfp4Args {
  const __nv_bfloat16* a;
  const uint8_t* packed;
  const uint8_t* scales;
  const __half* globals;
  __nv_bfloat16* out;
  const float* route_weights;
  const int32_t* sorted_routes;
  const int32_t* expert_slots;
  const int32_t* padded_count;
  int64_t a_row_stride, packed_slot_stride, scale_slot_stride, global_slot_stride;
  int32_t n, k, sorted_capacity, valid_routes, input_topk, multiply_weight;
};

namespace {
using namespace nvcuda;

__device__ __forceinline__ float scale_float(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

// FP4 has two significand bits; E4M3 has four. Their product is exactly
// representable in BF16, including E4M3 subnormals. Native BF16x2 conversion and
// multiplication therefore match Triton's float product -> BF16 without a LUT.
__device__ __forceinline__ void unpack_word(uint32_t word, float scale,
                                           __nv_bfloat16* dest) {
  const __nv_bfloat162 s = __floats2bfloat162_rn(scale, scale);
  #pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const __nv_bfloat162 b = static_cast<__nv_bfloat162>(__nv_cvt_fp4x2_to_bf162raw(
        static_cast<uint8_t>(word >> (8 * pair)), __NV_E2M1));
    reinterpret_cast<__nv_bfloat162*>(dest)[pair] = __hmul2(b, s);
  }
}

__global__ void dequant_reference_fixture(const uint32_t* words, const uint8_t* scales,
                                          __nv_bfloat16* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) unpack_word(words[i], scale_float(scales[i]), out + int64_t(i) * 8);
}

template<int BM, int BN, int BK, bool FusedSilu>
__global__ __launch_bounds__(256) void qwen_prefill_wmma(QwenNvfp4Args p) {
  const int row_start = blockIdx.x * BM;
  if (row_start >= *p.padded_count) return;
  const int slot = p.expert_slots[blockIdx.x];
  if (slot < 0) return;
  const int n_start = blockIdx.y * BN;
  const int output_n = FusedSilu ? p.n / 2 : p.n;
  constexpr int WM = BM / 32, WN = BN / 64;
  constexpr int BCOUNT = FusedSilu ? 2 : 1;
  extern __shared__ __align__(32) unsigned char storage[];
  auto* sa = reinterpret_cast<__nv_bfloat16*>(storage);
  auto* sb = sa + BM * BK;
  auto* sb_up = sb + BN * BK;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int warp_m = (warp % 2) * 16;
  const int warp_n = (warp / 2) * 16;
  using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  Acc acc[BCOUNT][WM][WN];
  #pragma unroll
  for (int b = 0; b < BCOUNT; ++b)
    #pragma unroll
    for (int m = 0; m < WM; ++m)
      #pragma unroll
      for (int n = 0; n < WN; ++n) wmma::fill_fragment(acc[b][m][n], 0.0f);

  const auto* wp = p.packed + int64_t(slot) * p.packed_slot_stride;
  const auto* sp = p.scales + int64_t(slot) * p.scale_slot_stride;
  for (int ks = 0; ks < p.k; ks += BK) {
    // Eight BF16 activations per 128-bit transaction. Hidden size is divisible
    // by 64, and each fragment is padded with zeros for an incomplete route tile.
    for (int v = tid; v < BM * BK / 8; v += 256) {
      const int m = v / (BK / 8), k = (v % (BK / 8)) * 8;
      const int route = p.sorted_routes[row_start + m];
      uint4 value = make_uint4(0, 0, 0, 0);
      if (route < p.valid_routes && route >= 0)
        value = *reinterpret_cast<const uint4*>(
            p.a + int64_t(route / p.input_topk) * p.a_row_stride + ks + k);
      *reinterpret_cast<uint4*>(sa + m * BK + k) = value;
    }
    for (int v = tid; v < BN * BK / 8; v += 256) {
      const int n = v / (BK / 8), k = (v % (BK / 8)) * 8;
      #pragma unroll
      for (int b = 0; b < BCOUNT; ++b) {
        const int actual_n = n_start + n + b * output_n;
        uint32_t word = 0;
        float scale = 0;
        if (n_start + n < output_n) {
          word = *reinterpret_cast<const uint32_t*>(
              wp + int64_t(actual_n) * (p.k / 2) + (ks + k) / 2);
          scale = scale_float(sp[int64_t(actual_n) * (p.k / 16) + (ks + k) / 16]);
        }
        unpack_word(word, scale, (b == 0 ? sb : sb_up) + n * BK + k);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[WM];
      #pragma unroll
      for (int m = 0; m < WM; ++m)
        wmma::load_matrix_sync(af[m], sa + (warp_m + m * 32) * BK + kk, BK);
      #pragma unroll
      for (int b = 0; b < BCOUNT; ++b) {
        #pragma unroll
        for (int n = 0; n < WN; ++n) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
          wmma::load_matrix_sync(bf, (b == 0 ? sb : sb_up) + (warp_n + n * 64) * BK + kk, BK);
          #pragma unroll
          for (int m = 0; m < WM; ++m) wmma::mma_sync(acc[b][m][n], af[m], bf, acc[b][m][n]);
        }
      }
    }
    __syncthreads();
  }

  // Reuse the staging shared memory after all readers finish. An ordinary
  // float epilogue avoids relying on implementation-defined WMMA lane layouts.
  auto* c = reinterpret_cast<float*>(storage);
  #pragma unroll
  for (int b = 0; b < BCOUNT; ++b)
    #pragma unroll
    for (int m = 0; m < WM; ++m)
      #pragma unroll
      for (int n = 0; n < WN; ++n)
        wmma::store_matrix_sync(c + b * BM * BN + (warp_m + m * 32) * BN + warp_n + n * 64,
                                acc[b][m][n], BN, wmma::mem_row_major);
  __syncthreads();
  for (int v = tid; v < BM * BN; v += 256) {
    const int m = v / BN, n = n_start + v % BN;
    const int route = p.sorted_routes[row_start + m];
    if (route >= p.valid_routes || route < 0 || n >= output_n) continue;
    const float multiplier = p.multiply_weight ? p.route_weights[route] : 1.0f;
    const float gate = c[v] * __half2float(p.globals[int64_t(slot) * p.global_slot_stride + n]) * multiplier;
    if constexpr (FusedSilu) {
      const float up = c[BM * BN + v] *
          __half2float(p.globals[int64_t(slot) * p.global_slot_stride + output_n + n]) * multiplier;
      // Preserve the intermediate BF16 roundings of gemm1 -> activation.
      const float g = __bfloat162float(__float2bfloat16_rn(gate));
      const float u = __bfloat162float(__float2bfloat16_rn(up));
      p.out[int64_t(route) * output_n + n] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
    } else {
      p.out[int64_t(route) * output_n + n] = __float2bfloat16_rn(gate);
    }
  }
}

template<int BM, int BN, int BK, bool Fused>
cudaError_t launch(const QwenNvfp4Args& p, cudaStream_t stream) {
  constexpr int stage_bytes = (BM + BN * (Fused ? 2 : 1)) * BK * sizeof(__nv_bfloat16);
  constexpr int output_bytes = BM * BN * (Fused ? 2 : 1) * sizeof(float);
  constexpr int shared_bytes = stage_bytes > output_bytes ? stage_bytes : output_bytes;
  if constexpr (shared_bytes > 48 * 1024) {
    const auto e = cudaFuncSetAttribute(qwen_prefill_wmma<BM, BN, BK, Fused>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    if (e != cudaSuccess) return e;
  }
  const int n = Fused ? p.n / 2 : p.n;
  const dim3 grid((p.sorted_capacity + BM - 1) / BM, (n + BN - 1) / BN);
  qwen_prefill_wmma<BM, BN, BK, Fused><<<grid, 256, shared_bytes, stream>>>(p);
  return cudaGetLastError();
}
}

extern "C" int qwen_nvfp4_prefill(const QwenNvfp4Args* p, int bm, int bn, int bk,
                                  int fused_silu, void* stream) {
  if (!p || p->k % 128 || p->n % 64 || p->a_row_stride % 8 || p->input_topk < 1)
    return int(cudaErrorInvalidValue);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  #define LAUNCH(M,N,K) if (bm==M && bn==N && bk==K) { \
    return int(fused_silu ? launch<M,N,K,true>(*p,s) : launch<M,N,K,false>(*p,s)); }
  LAUNCH(32,64,64)
  LAUNCH(64,64,64)
  LAUNCH(32,128,64)
  LAUNCH(64,128,64)
  LAUNCH(32,64,128)
  LAUNCH(64,64,128)
  #undef LAUNCH
  return int(cudaErrorInvalidValue);
}

extern "C" const char* qwen_nvfp4_cuda_error(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int qwen_nvfp4_dequant_fixture(const void* words, const void* scales,
                                          void* out, int count, void* stream) {
  if (count < 1) return int(cudaErrorInvalidValue);
  dequant_reference_fixture<<<(count + 127) / 128, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t*>(words), static_cast<const uint8_t*>(scales),
      static_cast<__nv_bfloat16*>(out), count);
  return int(cudaGetLastError());
}
