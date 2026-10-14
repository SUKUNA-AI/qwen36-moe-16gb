// Byte-preserving mapped-host transport for SM120. Ownership is committed later
// on this stream: a normal multi-CTA launch has no grid-wide completion barrier.
#include <cuda_runtime.h>
#include <cstdint>

struct Task { int bank, tile, tiles, unused; };
namespace {

template<int Threads, int Unroll, bool Exchange>
__global__ void transport_exchange(const uint64_t* host, const uint64_t* gpu,
    const int64_t* bytes, const int32_t* host_slot, const int32_t* src,
    const int32_t* dst, const int32_t* count, const Task* tasks, int base, int offset) {
    const int item = blockIdx.y;
    if (item >= *count) return;
    const Task task = tasks[blockIdx.x];
    const int64_t n = bytes[task.bank] / sizeof(uint4);
    const int64_t hs = host_slot[base + src[item]], gs = int64_t(dst[item]) + offset;
    uint4* h = reinterpret_cast<uint4*>(host[task.bank]) + hs * n;
    uint4* g = reinterpret_cast<uint4*>(gpu[task.bank]) + gs * n;
    const int64_t stride = int64_t(task.tiles) * Threads;
    bool wrote_host = false;
    for (int64_t start = int64_t(task.tile) * Threads + threadIdx.x;
         start < n; start += stride * Unroll) {
        uint4 promoted[Unroll], victim[Unroll];
        // Independent reads increase the number of outstanding PCIe requests.
        #pragma unroll
        for (int j = 0; j < Unroll; ++j) if (start + j * stride < n) {
            promoted[j] = h[start + j * stride];
            if constexpr (Exchange) victim[j] = g[start + j * stride];
        }
        #pragma unroll
        for (int j = 0; j < Unroll; ++j) if (start + j * stride < n) {
            if constexpr (Exchange) { h[start + j * stride] = victim[j]; wrote_host = true; }
            g[start + j * stride] = promoted[j];
        }
    }
    if constexpr (Exchange) if (wrote_host) __threadfence_system();
}

template<int Threads, int Unroll>
__global__ void transport_gather(const uint64_t* host, const uint64_t* gpu,
    const int64_t* bytes, const int32_t* host_slot, const int32_t* gpu_slot,
    const Task* tasks, int base, int offset, int destination) {
    const Task task = tasks[blockIdx.x];
    const int expert = blockIdx.y, gs = gpu_slot[base + expert];
    const int64_t n = bytes[task.bank] / sizeof(uint4);
    const uint4* source = gs >= 0
        ? reinterpret_cast<const uint4*>(gpu[task.bank]) + (int64_t(gs) + offset) * n
        : reinterpret_cast<const uint4*>(host[task.bank]) + int64_t(host_slot[base + expert]) * n;
    uint4* target = reinterpret_cast<uint4*>(gpu[task.bank]) + (int64_t(destination) + expert) * n;
    const int64_t stride = int64_t(task.tiles) * Threads;
    for (int64_t start = int64_t(task.tile) * Threads + threadIdx.x;
         start < n; start += stride * Unroll) {
        uint4 value[Unroll];
        #pragma unroll
        for (int j = 0; j < Unroll; ++j) if (start + j * stride < n) value[j] = source[start + j * stride];
        #pragma unroll
        for (int j = 0; j < Unroll; ++j) if (start + j * stride < n) target[start + j * stride] = value[j];
    }
}

template<int Threads, int Unroll, bool Exchange>
__global__ void transport_flat_exchange(const uint64_t* host, const uint64_t* gpu,
    const int64_t* bytes, const int32_t* host_slot, const int32_t* src,
    const int32_t* dst, const int32_t* count, const Task* tasks, int base, int offset) {
    const Task task=tasks[blockIdx.x];
    const int64_t n=bytes[task.bank]/sizeof(uint4), total=int64_t(*count)*n;
    const int64_t stride=int64_t(task.tiles)*Threads;
    uint4* h=reinterpret_cast<uint4*>(host[task.bank]);
    uint4* g=reinterpret_cast<uint4*>(gpu[task.bank]);
    bool wrote_host=false;
    for(int64_t start=int64_t(task.tile)*Threads+threadIdx.x;start<total;start+=stride*Unroll) {
        uint4 promoted[Unroll],victim[Unroll];int64_t hi[Unroll],gi[Unroll];
        #pragma unroll
        for(int j=0;j<Unroll;++j) if(start+j*stride<total) {
            const int64_t u=start+j*stride;
            const int row=task.unused>=0 ? int(u>>task.unused) : int(u/n);
            const int64_t col=u-int64_t(row)*n;
            hi[j]=int64_t(host_slot[base+src[row]])*n+col;
            gi[j]=(int64_t(dst[row])+offset)*n+col;
            promoted[j]=h[hi[j]];
            if constexpr(Exchange)victim[j]=g[gi[j]];
        }
        #pragma unroll
        for(int j=0;j<Unroll;++j) if(start+j*stride<total) {
            if constexpr(Exchange){h[hi[j]]=victim[j];wrote_host=true;}
            g[gi[j]]=promoted[j];
        }
    }
    if constexpr(Exchange)if(wrote_host)__threadfence_system();
}

template<int Threads,int Unroll>
__global__ void transport_flat_gather(const uint64_t* host,const uint64_t* gpu,
    const int64_t* bytes,const int32_t* host_slot,const int32_t* gpu_slot,
    const Task* tasks,int base,int offset,int destination,int experts) {
    const Task task=tasks[blockIdx.x];
    const int64_t n=bytes[task.bank]/sizeof(uint4),total=int64_t(experts)*n;
    const int64_t stride=int64_t(task.tiles)*Threads;
    const uint4* h=reinterpret_cast<const uint4*>(host[task.bank]);
    uint4* g=reinterpret_cast<uint4*>(gpu[task.bank]);
    for(int64_t start=int64_t(task.tile)*Threads+threadIdx.x;start<total;start+=stride*Unroll) {
        uint4 values[Unroll];
        #pragma unroll
        for(int j=0;j<Unroll;++j)if(start+j*stride<total) {
            const int64_t u=start+j*stride;
            const int row=task.unused>=0 ? int(u>>task.unused) : int(u/n);
            const int64_t col=u-int64_t(row)*n;
            const int slot=gpu_slot[base+row];
            values[j]=slot>=0 ? g[(int64_t(slot)+offset)*n+col] : h[int64_t(host_slot[base+row])*n+col];
        }
        #pragma unroll
        for(int j=0;j<Unroll;++j)if(start+j*stride<total)g[int64_t(destination)*n+start+j*stride]=values[j];
    }
}

template<int T, int U>
int launch_exchange(const uint64_t* h, const uint64_t* g, const int64_t* b,
    const int32_t* hs, const int32_t* s, const int32_t* d, const int32_t* c,
    const Task* tasks, int nt, int capacity, int base, int offset, bool exchange, cudaStream_t stream) {
    if (exchange) transport_exchange<T,U,true><<<dim3(nt,capacity),T,0,stream>>>(h,g,b,hs,s,d,c,tasks,base,offset);
    else transport_exchange<T,U,false><<<dim3(nt,capacity),T,0,stream>>>(h,g,b,hs,s,d,c,tasks,base,offset);
    return int(cudaPeekAtLastError());
}

}
extern "C" int qwen_transport_flat_exchange(const uint64_t* h,const uint64_t* g,const int64_t* b,
    const int32_t* hs,const int32_t* s,const int32_t* d,const int32_t* c,
    const Task* tasks,int nt,int capacity,int base,int offset,int exchange,int threads,int unroll,uintptr_t raw_stream) {
    if(nt<=0 || capacity<0 || capacity>65535)return int(cudaErrorInvalidValue);
    auto stream=reinterpret_cast<cudaStream_t>(raw_stream);
    #define CASE(T,U) if(threads==T && unroll==U) { \
        if(exchange)transport_flat_exchange<T,U,true><<<nt,T,0,stream>>>(h,g,b,hs,s,d,c,tasks,base,offset); \
        else transport_flat_exchange<T,U,false><<<nt,T,0,stream>>>(h,g,b,hs,s,d,c,tasks,base,offset); \
        return int(cudaPeekAtLastError()); }
    CASE(128,1);CASE(128,4);CASE(256,1);CASE(256,4);CASE(512,1);CASE(512,4);CASE(1024,1);CASE(1024,4);
    #undef CASE
    return int(cudaErrorInvalidValue);
}

extern "C" int qwen_transport_flat_gather(const uint64_t* h,const uint64_t* g,const int64_t* b,
    const int32_t* hs,const int32_t* gs,const Task* tasks,int nt,int experts,
    int base,int offset,int destination,int threads,int unroll,uintptr_t raw_stream) {
    if(nt<=0 || experts<=0 || experts>65535)return int(cudaErrorInvalidValue);
    auto stream=reinterpret_cast<cudaStream_t>(raw_stream);
    #define CASE(T,U) if(threads==T && unroll==U) { \
        transport_flat_gather<T,U><<<nt,T,0,stream>>>(h,g,b,hs,gs,tasks,base,offset,destination,experts); \
        return int(cudaPeekAtLastError()); }
    CASE(128,1);CASE(128,4);CASE(256,1);CASE(256,4);CASE(512,1);CASE(512,4);CASE(1024,1);CASE(1024,4);
    #undef CASE
    return int(cudaErrorInvalidValue);
}
namespace {
template<int T, int U>
int launch_gather(const uint64_t* h, const uint64_t* g, const int64_t* b,
    const int32_t* hs, const int32_t* gs, const Task* tasks, int nt, int experts,
    int base, int offset, int destination, cudaStream_t stream) {
    transport_gather<T,U><<<dim3(nt,experts),T,0,stream>>>(h,g,b,hs,gs,tasks,base,offset,destination);
    return int(cudaPeekAtLastError());
}
}

extern "C" int qwen_transport_exchange(const uint64_t* h, const uint64_t* g, const int64_t* b,
    const int32_t* hs, const int32_t* s, const int32_t* d, const int32_t* c,
    const Task* tasks, int nt, int capacity, int base, int offset, int exchange,
    int threads, int unroll, uintptr_t raw_stream) {
    if (nt <= 0 || capacity < 0 || capacity > 65535) return int(cudaErrorInvalidValue);
    if (!capacity) return int(cudaSuccess);
    auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    #define CASE(T,U) if(threads==T && unroll==U) return launch_exchange<T,U>(h,g,b,hs,s,d,c,tasks,nt,capacity,base,offset,exchange!=0,stream)
    CASE(128,1); CASE(128,4); CASE(256,1); CASE(256,4); CASE(512,1); CASE(512,4);
    #undef CASE
    return int(cudaErrorInvalidValue);
}

extern "C" int qwen_transport_gather(const uint64_t* h, const uint64_t* g, const int64_t* b,
    const int32_t* hs, const int32_t* gs, const Task* tasks, int nt, int experts,
    int base, int offset, int destination, int threads, int unroll, uintptr_t raw_stream) {
    if (nt <= 0 || experts <= 0 || experts > 65535) return int(cudaErrorInvalidValue);
    auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    #define CASE(T,U) if(threads==T && unroll==U) return launch_gather<T,U>(h,g,b,hs,gs,tasks,nt,experts,base,offset,destination,stream)
    CASE(128,1); CASE(128,4); CASE(256,1); CASE(256,4); CASE(512,1); CASE(512,4);
    #undef CASE
    return int(cudaErrorInvalidValue);
}
