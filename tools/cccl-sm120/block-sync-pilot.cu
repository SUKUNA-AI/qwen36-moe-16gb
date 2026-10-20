#include <cooperative_groups.h>
namespace cg = cooperative_groups;
template <bool UseCG>
__device__ __forceinline__ void bsync(cg::thread_block const& b) { if constexpr (UseCG) b.sync(); else __syncthreads(); }
__device__ __noinline__ void init_agg(cg::thread_block const& block, int kind, int* agg, int cnt) {
  for (int i = block.thread_rank(); i < cnt; i += block.num_threads()) {
    switch (kind & 3) { case 0: agg[i] = 0; break; case 1: agg[i] = 1 << 30; break; case 2: agg[i] = -(1 << 30); break; default: agg[i] = kind; }
  }
}
__device__ __noinline__ void pre_agg(int kind, const int* keys, int n, int* agg, int cols) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int* p = &agg[(keys[i] & 63) + 64 * (i % cols)];
    switch (kind & 3) { case 0: atomicAdd(p, 1); break; case 1: atomicMin(p, keys[i]); break; case 2: atomicMax(p, keys[i]); break; default: atomicAdd(p, keys[i]); }
  }
}
template <bool UseCG>
__global__ void k(const int* keys, const int* kinds, int n, int ncols, int* out) {
  auto const block = cg::this_thread_block();
  __shared__ int col_start, col_end;
  extern __shared__ int agg[];
  if (block.thread_rank() == 0) { col_start = 0; col_end = 0; }
  bsync<UseCG>(block);
  while (col_end < ncols) {
    bsync<UseCG>(block);
    if (block.thread_rank() == 0) { col_start = col_end; col_end = col_end + 1; }
    bsync<UseCG>(block);
    init_agg(block, kinds[col_start], agg, 64);
    bsync<UseCG>(block);
    pre_agg(kinds[col_start], keys, n, agg, 1);
    bsync<UseCG>(block);
    for (int i = block.thread_rank(); i < 64; i += block.num_threads()) out[i + 64 * col_start] = agg[i];
  }
}
template __global__ void k<true>(const int*, const int*, int, int, int*);
template __global__ void k<false>(const int*, const int*, int, int, int*);
