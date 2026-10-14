// Lossless NVFP4 byte transport. No arithmetic on weights or their scales.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads=256;
constexpr int kTiles=8;

__global__ void exchange_blocks(const uint64_t* host,const uint64_t* gpu,const int64_t* bytes,
    const int32_t* host_slot,const int32_t* src,const int32_t* dst,const int32_t* count,
    int base,int offset,bool exchange) {
    const int item=blockIdx.y,bank=blockIdx.z;
    if(item>=*count)return;
    const int64_t vectors=bytes[bank]/sizeof(int4);
    const int hs=host_slot[base+src[item]],gs=dst[item]+offset;
    int4* h=reinterpret_cast<int4*>(host[bank])+int64_t(hs)*vectors;
    int4* g=reinterpret_cast<int4*>(gpu[bank])+int64_t(gs)*vectors;
    for(int64_t i=blockIdx.x*kThreads+threadIdx.x;i<vectors;i+=kTiles*kThreads) {
        const int4 promoted=h[i];
        if(exchange){const int4 victim=g[i];h[i]=victim;}
        g[i]=promoted;
    }
    // The compact host slot may become another expert's source next launch.
    if(exchange)__threadfence_system();
}

__global__ void gather_blocks(const uint64_t* host,const uint64_t* gpu,const int64_t* bytes,
    const int32_t* host_slot,const int32_t* gpu_slot,int base,int offset,int destination) {
    const int e=blockIdx.y,bank=blockIdx.z;
    const int64_t vectors=bytes[bank]/sizeof(int4);
    const int gs=gpu_slot[base+e];
    const int4* source=gs>=0 ? reinterpret_cast<const int4*>(gpu[bank])+int64_t(gs+offset)*vectors
                            : reinterpret_cast<const int4*>(host[bank])+int64_t(host_slot[base+e])*vectors;
    int4* target=reinterpret_cast<int4*>(gpu[bank])+int64_t(destination+e)*vectors;
    for(int64_t i=blockIdx.x*kThreads+threadIdx.x;i<vectors;i+=kTiles*kThreads)target[i]=source[i];
}
}

extern "C" int qwen36_exchange(const uint64_t* host,const uint64_t* gpu,const int64_t* bytes,
    const int32_t* host_slot,const int32_t* src,const int32_t* dst,const int32_t* count,
    int base,int offset,int banks,int capacity,int exchange,uintptr_t stream) {
    exchange_blocks<<<dim3(kTiles,capacity,banks),kThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        host,gpu,bytes,host_slot,src,dst,count,base,offset,exchange!=0);
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int qwen36_gather(const uint64_t* host,const uint64_t* gpu,const int64_t* bytes,
    const int32_t* host_slot,const int32_t* gpu_slot,int base,int offset,int destination,
    int banks,int experts,uintptr_t stream) {
    gather_blocks<<<dim3(kTiles,experts,banks),kThreads,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        host,gpu,bytes,host_slot,gpu_slot,base,offset,destination);
    return static_cast<int>(cudaPeekAtLastError());
}
