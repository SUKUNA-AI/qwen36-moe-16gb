#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;
__global__ void __cluster_dims__(2, 1, 1) k(int* out) {
  __shared__ int buf[4];
  cg::cluster_group cl = cg::this_cluster();
  unsigned me = cl.block_rank(), peer = me ^ 1u;
  int* local = buf;
  int* remote = cl.map_shared_rank(buf, peer);
  cl.sync();
  if (threadIdx.x == 0) {
    int* o = out + me * 4;
    o[0] = __isShared(local); o[1] = __isClusterShared(local);
    o[2] = __isShared(remote); o[3] = __isClusterShared(remote);
  }
  cl.sync();
}
int main() {
  int* d; cudaMalloc(&d, 8 * sizeof(int)); cudaMemset(d, 0xff, 8 * sizeof(int));
  k<<<2, 32>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  int h[8]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  printf("status=%s\n", cudaGetErrorString(e));
  for (int r = 0; r < 2; ++r)
    printf("block %d: isShared(local)=%d isClusterShared(local)=%d isShared(peer)=%d isClusterShared(peer)=%d\n", r, h[r*4], h[r*4+1], h[r*4+2], h[r*4+3]);
}
