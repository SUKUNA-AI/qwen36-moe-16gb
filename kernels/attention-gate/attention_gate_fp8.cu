#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// Qwen full-attention output gate followed by the checkpoint's static FP8 cast.
// Keep both original BF16 roundings; fusion only removes intermediate traffic.
namespace {
template<bool Debug>
__global__ void attention_gate_fp8(const __nv_bfloat16* attention,
                                  const __nv_bfloat16* gate,
                                  const float* scale, uint8_t* output,
                                  __nv_bfloat16* sigmoid_debug,
                                  __nv_bfloat16* product_debug,
                                  int64_t count, int64_t width,
                                  int64_t attention_stride, int64_t gate_stride) {
  const int64_t base = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (base >= count) return;
  const int64_t row = base / width, col = base % width;
  const uint4 av = *reinterpret_cast<const uint4*>(attention + row * attention_stride + col);
  const uint4 gv = *reinterpret_cast<const uint4*>(gate + row * gate_stride + col);
  const auto* aa = reinterpret_cast<const __nv_bfloat16*>(&av);
  const auto* gg = reinterpret_cast<const __nv_bfloat16*>(&gv);
  // Pinned Triton emits div.full.f32, not rcp.approx or div.rn.
  float inverse;
  asm("div.full.f32 %0, 0f3f800000, %1;" : "=f"(inverse) : "f"(*scale));
  uint2 result = make_uint2(0, 0);
  auto* packed = reinterpret_cast<uint8_t*>(&result);
  #pragma unroll
  for (int i=0; i<8; ++i) {
    const float value = __bfloat162float(gg[i]);
    const __nv_bfloat16 sigmoid = __float2bfloat16_rn(1.0f / (1.0f + expf(-value)));
    const __nv_bfloat16 product = __float2bfloat16_rn(
        __bfloat162float(aa[i]) * __bfloat162float(sigmoid));
    const float scaled = __bfloat162float(product) * inverse;
    const float clipped = fminf(fmaxf(scaled, -448.0f), 448.0f);
    packed[i] = __nv_cvt_float_to_fp8(clipped, __NV_SATFINITE, __NV_E4M3);
    if constexpr (Debug) {
      sigmoid_debug[base+i] = sigmoid;
      product_debug[base+i] = product;
    }
  }
  *reinterpret_cast<uint2*>(output + base) = result;
}
}

extern "C" int qwen_attention_gate_fp8(const void* attention, const void* gate,
    const void* scale, void* output, void* sigmoid_debug, void* product_debug,
    int64_t rows, int64_t width, int64_t attention_stride, int64_t gate_stride,
    void* stream) {
  if (!attention || !gate || !scale || !output || rows < 1 || width != 4096 ||
      attention_stride < width || gate_stride < width ||
      attention_stride % 8 || gate_stride % 8 || rows > INT32_MAX / width ||
      (uintptr_t(attention) | uintptr_t(gate)) % 16 || uintptr_t(output) % 8 ||
      bool(sigmoid_debug) != bool(product_debug)) return int(cudaErrorInvalidValue);
  const int64_t count = rows * width;
  const auto grid = static_cast<unsigned>((count + 1023) / 1024);
  const auto current = reinterpret_cast<cudaStream_t>(stream);
  if (sigmoid_debug) attention_gate_fp8<true><<<grid,128,0,current>>>(
      static_cast<const __nv_bfloat16*>(attention), static_cast<const __nv_bfloat16*>(gate),
      static_cast<const float*>(scale), static_cast<uint8_t*>(output),
      static_cast<__nv_bfloat16*>(sigmoid_debug), static_cast<__nv_bfloat16*>(product_debug),
      count, width, attention_stride, gate_stride);
  else attention_gate_fp8<false><<<grid,128,0,current>>>(
      static_cast<const __nv_bfloat16*>(attention), static_cast<const __nv_bfloat16*>(gate),
      static_cast<const float*>(scale), static_cast<uint8_t*>(output), nullptr, nullptr,
      count, width, attention_stride, gate_stride);
  return int(cudaGetLastError());
}

extern "C" const char* qwen_gate_error(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }
