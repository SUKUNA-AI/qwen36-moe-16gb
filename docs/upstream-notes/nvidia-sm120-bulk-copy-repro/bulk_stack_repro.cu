// Minimal repro: on sm_120, cp.async.bulk with a .shared::cluster destination is compiled to a
// driver syscall (__cuda_syscall_cp_async_bulk_unicast), and the first launch raises the context's
// per-thread stack limit, reserving VRAM that is not returned. The .shared::cta form (PTX 8.6+)
// compiles to native UBLKCP and does not.
//
// Build both variants (see run.sh):  nvcc -arch=sm_120 -DDST_CLUSTER=1|0 bulk_stack_repro.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef DST_CLUSTER
#define DST_CLUSTER 1
#endif

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

constexpr int kBytes = 4096;
constexpr int kThreads = 256;

__global__ void bulk_copy_kernel(const uint4* __restrict__ src, unsigned long long* __restrict__ out) {
  __shared__ alignas(128) uint4 buf[kBytes / 16];
  __shared__ alignas(8) unsigned long long mbar;
  const uint32_t buf_s = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
  const uint32_t mbar_s = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_s));
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar_s), "r"(kBytes) : "memory");
#if DST_CLUSTER
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(buf_s), "l"(src + (size_t)blockIdx.x * (kBytes / 16)), "r"(kBytes), "r"(mbar_s) : "memory");
#else
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(buf_s), "l"(src + (size_t)blockIdx.x * (kBytes / 16)), "r"(kBytes), "r"(mbar_s) : "memory");
#endif
  }
  uint32_t done = 0;
  while (!done) {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(mbar_s) : "memory");
  }
  unsigned long long acc = 0;
  for (int i = threadIdx.x; i < kBytes / 16; i += blockDim.x) acc += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
  atomicAdd(out, acc);
}

int main() {
  int dev = 0, sms = 0, per_sm = 0;
  CHECK(cudaSetDevice(dev));
  CHECK(cudaFree(nullptr));  // create the context
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, dev));
  sms = prop.multiProcessorCount;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, bulk_copy_kernel, kThreads, 0));
  const int blocks = sms * per_sm;

  uint4* src = nullptr;
  unsigned long long* out = nullptr;
  CHECK(cudaMalloc(&src, (size_t)blocks * kBytes));
  CHECK(cudaMemset(src, 1, (size_t)blocks * kBytes));
  CHECK(cudaMalloc(&out, sizeof(*out)));
  CHECK(cudaMemset(out, 0, sizeof(*out)));
  CHECK(cudaDeviceSynchronize());

  size_t stack0 = 0, stack1 = 0, free0 = 0, free1 = 0, total = 0;
  CHECK(cudaDeviceGetLimit(&stack0, cudaLimitStackSize));
  CHECK(cudaMemGetInfo(&free0, &total));
  bulk_copy_kernel<<<blocks, kThreads>>>(src, out);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaDeviceGetLimit(&stack1, cudaLimitStackSize));
  CHECK(cudaMemGetInfo(&free1, &total));
  // A second launch to show the reservation is kept, not transient.
  bulk_copy_kernel<<<blocks, kThreads>>>(src, out);
  CHECK(cudaDeviceSynchronize());
  size_t free2 = 0;
  CHECK(cudaMemGetInfo(&free2, &total));
  // Workaround check: lower the limit back by hand, then launch again.
  size_t stack3 = 0, stack4 = 0, free3 = 0, free4 = 0;
  CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack0));
  CHECK(cudaDeviceGetLimit(&stack3, cudaLimitStackSize));
  CHECK(cudaMemGetInfo(&free3, &total));
  bulk_copy_kernel<<<blocks, kThreads>>>(src, out);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaDeviceGetLimit(&stack4, cudaLimitStackSize));
  CHECK(cudaMemGetInfo(&free4, &total));

  unsigned long long h = 0;
  CHECK(cudaMemcpy(&h, out, sizeof(h), cudaMemcpyDeviceToHost));
  const unsigned long long expect = 3ull * blocks * (kBytes / 4) * 0x01010101ull;
  int drv = 0, rt = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rt);
  std::printf("variant=%s gpu=\"%s\" sm_%d%d sms=%d maxThreads/SM=%d driver=%d runtime=%d blocks=%d\n",
              DST_CLUSTER ? "shared::cluster" : "shared::cta", prop.name, prop.major, prop.minor, sms,
              prop.maxThreadsPerMultiProcessor, drv, rt, blocks);
  std::printf("stack_limit_bytes before=%zu after=%zu\n", stack0, stack1);
  std::printf("free_vram_MiB before=%.1f after_1st=%.1f after_2nd=%.1f delta=%.1f\n", free0 / 1048576.0,
              free1 / 1048576.0, free2 / 1048576.0, (double)((long long)free0 - (long long)free2) / 1048576.0);
  std::printf("after cudaDeviceSetLimit(%zu): stack_limit=%zu free_vram_MiB=%.1f; after 3rd launch: stack_limit=%zu free_vram_MiB=%.1f\n",
              stack0, stack3, free3 / 1048576.0, stack4, free4 / 1048576.0);
  std::printf("checksum %s\n", h == expect ? "OK" : "MISMATCH");
  return h == expect ? 0 : 2;
}
