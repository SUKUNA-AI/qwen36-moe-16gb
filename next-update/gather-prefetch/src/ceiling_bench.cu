// ceiling_bench.cu -- H2D ceiling of the RTX 5070 Ti / PCIe 5.0 x16 / WSL2 link (work/next-update/gather-prefetch).
//
// Mechanisms:
//   ce      copy engine: cudaMemcpyAsync H2D from pinned host memory (one big copy; n x 1.78 MB expert-sized
//           copies back to back; the 6-row production layout = 6 copies per expert; and the same as graph
//           memcpy nodes)
//   tma     SM bulk copies: cp.async.bulk.shared::cta.global (+ mbarrier) -> cp.async.bulk.global.shared::cta,
//           C CTAs x W issuing warps (lane 0) x S ring stages x X-byte chunks; "sink" variant skips the store
//   ldg     SM plain loads: warp per chunk, K 16-B loads in flight per thread (ld.global.v4 plain or
//           ld.global.nc.L1::no_allocate), then 16-B stores (or xor-sink)
// Host memory kinds: hostalloc (cudaHostAlloc Portable|Mapped), thp (mmap MAP_PRIVATE + MADV_HUGEPAGE,
// cudaHostRegister Portable|Mapped, as FreeToken host_banks), 4k (same with MADV_NOHUGEPAGE).
// NEVER .shared::cluster (sm_120: driver syscall + 1.37 GiB stack reservation).
//
// Every SM/CE run is verified: the destination must equal the host pattern for exactly the copied ranges.
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <tuple>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)

static constexpr uint64_t EXPERT_BYTES = 1781760;  // 1 MiB + 128 KiB + 4 KiB + 512 KiB + 64 KiB + 8 KiB
static constexpr uint64_t ROWS[6] = {1048576, 131072, 4096, 524288, 65536, 8192};
static constexpr uint64_t SLOT = 2ull << 20;       // expert-sized blocks live in 2 MiB slots

__host__ __device__ inline uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__global__ void spin_kernel(uint64_t ns) {
    const uint64_t t0 = gtimer();
    while (gtimer() - t0 < ns) {
    }
}

struct Entry {
    uint64_t src;   // device-visible source address
    uint64_t dst;   // device destination address
    uint64_t soff;  // source offset in the host region (pattern key)
    uint32_t bytes;
    uint32_t pad;
};

// ---------------------------------------------------------------- verify
// dst word w of entry e must equal mix(soff/8 + w)
__global__ void verify_kernel(const Entry* __restrict__ ent, int n, unsigned long long* bad) {
    unsigned long long local = 0;
    for (int e = blockIdx.y; e < n; e += gridDim.y) {
        const uint64_t* d = reinterpret_cast<const uint64_t*>(ent[e].dst);
        const uint64_t base = ent[e].soff / 8;
        const uint64_t nw = ent[e].bytes / 8;
        for (uint64_t w = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; w < nw; w += uint64_t(gridDim.x) * blockDim.x)
            local += (d[w] != mix(base + w));
    }
    for (int o = 16; o > 0; o >>= 1) local += __shfl_xor_sync(0xFFFFFFFFu, local, o);
    if ((threadIdx.x & 31) == 0 && local) atomicAdd(bad, local);
}
__global__ void clear_kernel(const Entry* __restrict__ ent, int n) {
    for (int e = blockIdx.y; e < n; e += gridDim.y) {
        uint64_t* d = reinterpret_cast<uint64_t*>(ent[e].dst);
        const uint64_t nw = ent[e].bytes / 8;
        for (uint64_t w = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; w < nw; w += uint64_t(gridDim.x) * blockDim.x)
            d[w] = 0xDEADBEEFDEADBEEFull;
    }
}

// ---------------------------------------------------------------- TMA
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint64_t to_global(uint64_t p) {
    uint64_t g;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(g) : "l"(p));
    return g;
}
static constexpr int MAXW = 8, MAXS = 16;

// Entries are pre-split into chunks of <= X bytes. Worker w = blockIdx.x * W + warp takes chunks w, w + workers, ...
template <int S, int X, bool STORE>
__global__ __launch_bounds__(256) void tma_kernel(const Entry* __restrict__ ent, int n) {
    extern __shared__ __align__(128) uint8_t ring[];
    __shared__ __align__(8) uint64_t bars[MAXW][S];
    __shared__ uint64_t sdst[MAXW][S];
    __shared__ uint32_t sbytes[MAXW][S];
    const int warp = threadIdx.x >> 5;
    const int W = blockDim.x >> 5;
    if ((threadIdx.x & 31) != 0) return;
    uint8_t* my = ring + size_t(warp) * S * X;
    for (int s = 0; s < S; ++s)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bars[warp][s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const int workers = gridDim.x * W;
    int next = blockIdx.x * W + warp;
    auto issue = [&](int s) {
        const Entry e = ent[next];
        next += workers;
        sdst[warp][s] = e.dst;
        sbytes[warp][s] = e.bytes;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bars[warp][s])), "r"(e.bytes)
                     : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                         smem_u32(my + size_t(s) * X)),
                     "l"(to_global(e.src)), "r"(e.bytes), "r"(smem_u32(&bars[warp][s]))
                     : "memory");
    };
    int inflight = 0;
    for (int s = 0; s < S; ++s)
        if (next < n) {
            issue(s);
            ++inflight;
        }
    uint32_t parity = 0;
    int head = 0;
    while (inflight > 0) {
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                : "=r"(done)
                : "r"(smem_u32(&bars[warp][head])), "r"((parity >> head) & 1u)
                : "memory");
        }
        parity ^= 1u << head;
        if (STORE) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(to_global(sdst[warp][head])),
                         "r"(smem_u32(my + size_t(head) * X)), "r"(sbytes[warp][head])
                         : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
        --inflight;
        if (next < n) {
            if (STORE) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            issue(head);
            ++inflight;
        }
        head = head + 1 == S ? 0 : head + 1;
    }
    if (STORE) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// ---------------------------------------------------------------- LDG
template <int LD>
__device__ __forceinline__ uint4 ld16(const uint4* p) {
    uint4 v;
    if (LD == 0)
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    return v;
}
template <int K, int LD, bool STORE>
__global__ __launch_bounds__(256) void ldg_kernel(const Entry* __restrict__ ent, int n, unsigned long long* sink) {
    const int lane = threadIdx.x & 31;
    const int gw = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int nw = int((gridDim.x * blockDim.x) >> 5);
    uint32_t acc = 0;
    for (int c = gw; c < n; c += nw) {
        const uint4* s = reinterpret_cast<const uint4*>(ent[c].src);
        uint4* d = reinterpret_cast<uint4*>(ent[c].dst);
        const uint32_t nv = ent[c].bytes / 16;
        for (uint32_t base = 0; base < nv; base += 32 * K) {
            uint4 v[K];
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const uint32_t i = base + k * 32 + lane;
                if (i < nv) v[k] = ld16<LD>(s + i);
            }
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const uint32_t i = base + k * 32 + lane;
                if (i < nv) {
                    if (STORE)
                        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(d + i), "r"(v[k].x), "r"(v[k].y),
                                     "r"(v[k].z), "r"(v[k].w)
                                     : "memory");
                    else
                        acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
                }
            }
        }
    }
    if (!STORE && acc == 0x9E3779B9u) atomicAdd(sink, 1ull);
}

// ---------------------------------------------------------------- host side
struct Region {
    std::string kind;
    uint8_t* host = nullptr;
    uint64_t dev = 0;  // device alias
    uint64_t bytes = 0;
};
static Region make_region(const std::string& kind, uint64_t bytes, int threads) {
    Region r;
    r.kind = kind;
    r.bytes = bytes;
    if (kind == "hostalloc") {
        void* p = nullptr;
        CK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
        r.host = static_cast<uint8_t*>(p);
    } else {
        void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            perror("mmap");
            exit(2);
        }
        madvise(m, bytes, kind == "thp" ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        r.host = static_cast<uint8_t*>(m);
    }
    // fill pattern
    std::vector<std::thread> th;
    const uint64_t nw = bytes / 8;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([=] {
            uint64_t* d = reinterpret_cast<uint64_t*>(r.host);
            const uint64_t a = nw * t / threads, b = nw * (t + 1) / threads;
            for (uint64_t w = a; w < b; ++w) d[w] = mix(w);
        });
    for (auto& x : th) x.join();
    if (kind != "hostalloc") CK(cudaHostRegister(r.host, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
    void* dp = nullptr;
    CK(cudaHostGetDevicePointer(&dp, r.host, 0));
    r.dev = reinterpret_cast<uint64_t>(dp);
    return r;
}

struct Stat {
    double med, p10, p90;
};
static Stat stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, size_t(f * (v.size() - 1) + 0.5))]; };
    return {q(0.5), q(0.1), q(0.9)};
}

struct Job {
    std::vector<Entry> whole;   // un-split ranges (for CE and verify)
    uint64_t bytes = 0;
};

// n expert-sized blocks (EXPERT_BYTES each, contiguous, at random 2 MiB host slots) -> random 2 MiB device slots
// layout6: each expert is 6 separate rows from 6 separate bank areas (production layout)
static Job expert_job(const Region& R, uint8_t* dbase, uint64_t dslots, int n, bool layout6, std::mt19937_64& rng) {
    Job j;
    const uint64_t hslots = R.bytes / SLOT;
    std::vector<uint64_t> hs(hslots), ds(dslots);
    std::iota(hs.begin(), hs.end(), 0);
    std::iota(ds.begin(), ds.end(), 0);
    std::shuffle(hs.begin(), hs.end(), rng);
    std::shuffle(ds.begin(), ds.end(), rng);
    for (int i = 0; i < n; ++i) {
        if (!layout6) {
            const uint64_t so = hs[i] * SLOT, dof = ds[i] * SLOT;
            j.whole.push_back({R.dev + so, uint64_t(dbase) + dof, so, uint32_t(EXPERT_BYTES), 0});
        } else {
            // 6 rows from 6 different host slots (a different bank area each), packed into one device slot
            uint64_t dof = ds[i] * SLOT;
            for (int b = 0; b < 6; ++b) {
                const uint64_t so = hs[(size_t(i) * 6 + b) % hslots] * SLOT + (b * 4096) % (SLOT - ROWS[b] + 1);
                j.whole.push_back({R.dev + so, uint64_t(dbase) + dof, so, uint32_t(ROWS[b]), 0});
                dof += ROWS[b];
            }
        }
        j.bytes += EXPERT_BYTES;
    }
    return j;
}
static Job flat_job(const Region& R, uint8_t* dbase, uint64_t bytes) {
    Job j;
    j.whole.push_back({R.dev, uint64_t(dbase), 0, 0, 0});
    // split into <= 1 GiB pieces is unnecessary here; bytes fit in uint32 only below 4 GiB
    j.whole[0].bytes = uint32_t(bytes);
    j.bytes = bytes;
    return j;
}
static std::vector<Entry> split(const std::vector<Entry>& w, uint32_t X) {
    std::vector<Entry> out;
    for (const Entry& e : w)
        for (uint64_t off = 0; off < e.bytes; off += X)
            out.push_back({e.src + off, e.dst + off, e.soff + off, uint32_t(std::min<uint64_t>(X, e.bytes - off)), 0});
    return out;
}

struct Ctx {
    cudaStream_t st;
    cudaEvent_t e0, e1;
    Entry* d_ent;       // capacity
    size_t ent_cap;
    Entry* d_whole;
    size_t whole_cap;
    unsigned long long* d_bad;
    unsigned long long* d_sink;
    cudaStream_t xs[4];
    cudaEvent_t xe[4];
};

static unsigned long long verify(Ctx& C, const Job& j) {
    CK(cudaMemcpyAsync(C.d_whole, j.whole.data(), sizeof(Entry) * j.whole.size(), cudaMemcpyHostToDevice, C.st));
    CK(cudaMemsetAsync(C.d_bad, 0, 8, C.st));
    verify_kernel<<<dim3(64, std::min<int>(int(j.whole.size()), 1024)), 256, 0, C.st>>>(C.d_whole, int(j.whole.size()), C.d_bad);
    clear_kernel<<<dim3(64, std::min<int>(int(j.whole.size()), 1024)), 256, 0, C.st>>>(C.d_whole, int(j.whole.size()));
    unsigned long long bad = 0;
    CK(cudaMemcpyAsync(&bad, C.d_bad, 8, cudaMemcpyDeviceToHost, C.st));
    CK(cudaStreamSynchronize(C.st));
    return bad;
}

using Launch = void (*)(cudaStream_t, const Entry*, int, int ctas, int warps, unsigned long long* sink);
template <int S, int X, bool STORE>
static void launch_tma(cudaStream_t st, const Entry* e, int n, int ctas, int warps, unsigned long long*) {
    tma_kernel<S, X, STORE><<<ctas, 32 * warps, size_t(warps) * S * X, st>>>(e, n);
}
template <int K, int LD, bool STORE>
static void launch_ldg(cudaStream_t st, const Entry* e, int n, int ctas, int, unsigned long long* sink) {
    ldg_kernel<K, LD, STORE><<<ctas, 256, 0, st>>>(e, n, sink);
}
static int g_optin = 0;
static std::map<std::tuple<int, int, int>, int> tma_maxdyn;
template <int S, int X, bool STORE>
static void attr_tma() {
    cudaFuncAttributes fa{};
    CK(cudaFuncGetAttributes(&fa, tma_kernel<S, X, STORE>));
    const int maxdyn = g_optin - int(fa.sharedSizeBytes);
    CK(cudaFuncSetAttribute(tma_kernel<S, X, STORE>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxdyn));
    tma_maxdyn[{S, X, STORE ? 1 : 0}] = maxdyn;
}

struct SmVariant {
    std::string name;
    std::string family;
    int ctas, warps, S, X, K, ld;
    bool store;
    uint64_t inflight;
    Launch fn;
};

#define TMA_SX(S, X)                                                                                   \
    {                                                                                                  \
        attr_tma<S, X, true>();                                                                        \
        attr_tma<S, X, false>();                                                                       \
        tma_fns[{S, X, 1}] = launch_tma<S, X, true>;                                                   \
        tma_fns[{S, X, 0}] = launch_tma<S, X, false>;                                                  \
    }
static std::map<std::tuple<int, int, int>, Launch> tma_fns;
static std::map<std::tuple<int, int, int>, Launch> ldg_fns;
static void init_fns() {
    TMA_SX(1, 16384) TMA_SX(2, 16384) TMA_SX(3, 16384) TMA_SX(4, 16384) TMA_SX(6, 16384)
    TMA_SX(2, 8192) TMA_SX(3, 8192) TMA_SX(4, 8192) TMA_SX(6, 8192) TMA_SX(8, 8192) TMA_SX(12, 8192)
    TMA_SX(2, 4096) TMA_SX(4, 4096) TMA_SX(8, 4096) TMA_SX(16, 4096)
    TMA_SX(1, 32768) TMA_SX(2, 32768) TMA_SX(3, 32768)
    TMA_SX(1, 65536)
    ldg_fns[{1, 0, 1}] = launch_ldg<1, 0, true>;
    ldg_fns[{2, 0, 1}] = launch_ldg<2, 0, true>;
    ldg_fns[{4, 0, 1}] = launch_ldg<4, 0, true>;
    ldg_fns[{8, 0, 1}] = launch_ldg<8, 0, true>;
    ldg_fns[{1, 1, 1}] = launch_ldg<1, 1, true>;
    ldg_fns[{2, 1, 1}] = launch_ldg<2, 1, true>;
    ldg_fns[{4, 1, 1}] = launch_ldg<4, 1, true>;
    ldg_fns[{8, 1, 1}] = launch_ldg<8, 1, true>;
    ldg_fns[{4, 1, 0}] = launch_ldg<4, 1, false>;
    ldg_fns[{8, 1, 0}] = launch_ldg<8, 1, false>;
}

static std::vector<std::string> strs(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(t);
    return v;
}
static std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    for (auto& t : strs(s)) v.push_back(std::stoi(t));
    return v;
}

// spec grammar: tma:C:W:S:X[:sink] | ldg:C:K:ld[:sink]
static SmVariant parse_variant(const std::string& spec) {
    std::vector<std::string> p;
    {
        std::stringstream ss(spec);
        std::string t;
        while (std::getline(ss, t, ':')) p.push_back(t);
    }
    SmVariant v{};
    v.name = spec;
    v.family = p[0];
    if (p[0] == "tma") {
        v.ctas = std::stoi(p[1]);
        v.warps = std::stoi(p[2]);
        v.S = std::stoi(p[3]);
        v.X = std::stoi(p[4]);
        v.store = !(p.size() > 5 && p[5] == "sink");
        auto it = tma_fns.find({v.S, v.X, v.store ? 1 : 0});
        if (it == tma_fns.end() || long(v.warps) * v.S * v.X > tma_maxdyn[{v.S, v.X, v.store ? 1 : 0}] || v.warps > MAXW) {
            fprintf(stderr, "unsupported tma variant %s\n", spec.c_str());
            exit(1);
        }
        v.fn = it->second;
        v.inflight = uint64_t(v.ctas) * v.warps * v.S * v.X;
    } else if (p[0] == "ldg") {
        v.ctas = std::stoi(p[1]);
        v.K = std::stoi(p[2]);
        v.ld = p[3] == "nc" ? 1 : 0;
        v.store = !(p.size() > 4 && p[4] == "sink");
        v.X = 16384;
        auto it = ldg_fns.find({v.K, v.ld, v.store ? 1 : 0});
        if (it == ldg_fns.end()) {
            fprintf(stderr, "unsupported ldg variant %s\n", spec.c_str());
            exit(1);
        }
        v.fn = it->second;
        v.warps = 8;
        v.inflight = uint64_t(v.ctas) * 256 * v.K * 16;
    } else {
        fprintf(stderr, "bad variant %s\n", spec.c_str());
        exit(1);
    }
    return v;
}

int main(int argc, char** argv) {
    std::string kinds = "thp", sm_specs, out;
    std::vector<int> ns = {1, 2, 4, 8, 16};
    int reps = 15, threads = 16;
    uint64_t footprint_mb = 2048, flat_mb = 256;
    bool do_ce = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--kinds") kinds = nx();
        else if (a == "--sm") sm_specs = nx();
        else if (a == "--ns") ns = ints(nx());
        else if (a == "--reps") reps = std::stoi(nx());
        else if (a == "--footprint-mb") footprint_mb = std::stoull(nx());
        else if (a == "--flat-mb") flat_mb = std::stoull(nx());
        else if (a == "--no-ce") do_ce = false;
        else if (a == "--out") out = nx();
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 1;
        }
    }
    CK(cudaSetDevice(0));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, 0));
    CK(cudaDeviceGetAttribute(&g_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    init_fns();
    std::vector<SmVariant> sm;
    if (!sm_specs.empty())
        for (auto& s : strs(sm_specs)) sm.push_back(parse_variant(s));

    Ctx C{};
    CK(cudaStreamCreateWithFlags(&C.st, cudaStreamNonBlocking));
    CK(cudaEventCreate(&C.e0));
    CK(cudaEventCreate(&C.e1));
    for (int s = 0; s < 4; ++s) {
        CK(cudaStreamCreateWithFlags(&C.xs[s], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&C.xe[s], cudaEventDisableTiming));
    }
    const uint64_t dbytes = std::max<uint64_t>(flat_mb << 20, 64 * SLOT);
    uint8_t* dbase = nullptr;
    CK(cudaMalloc(&dbase, dbytes));
    C.ent_cap = (dbytes / 4096) + 4096;
    CK(cudaMalloc(&C.d_ent, sizeof(Entry) * C.ent_cap));
    C.whole_cap = 4096;
    CK(cudaMalloc(&C.d_whole, sizeof(Entry) * C.whole_cap));
    CK(cudaMalloc(&C.d_bad, 8));
    CK(cudaMalloc(&C.d_sink, 8));
    std::mt19937_64 rng(777);
    const uint64_t SPIN = 20000;

    std::ostringstream js;
    js << "{\"schema\":1,\"device\":\"" << pr.name << "\",\"sms\":" << pr.multiProcessorCount
       << ",\"async_engines\":" << pr.asyncEngineCount << ",\"pcie_gen_hint\":\"5.0 x16 (63.0 GB/s raw/dir)\""
       << ",\"expert_bytes\":" << EXPERT_BYTES << ",\"footprint_mb\":" << footprint_mb << ",\"flat_mb\":" << flat_mb
       << ",\"reps\":" << reps << ",\"rows\":[\n";
    bool first = true;
    auto row = [&](const std::string& kind, const std::string& mech, const std::string& pattern, int n, uint64_t bytes,
                   const Stat& s, unsigned long long bad, uint64_t inflight) {
        const double gbps = bytes / (s.med * 1e3);
        js << (first ? "" : ",\n") << "{\"kind\":\"" << kind << "\",\"mech\":\"" << mech << "\",\"pattern\":\"" << pattern
           << "\",\"n\":" << n << ",\"bytes\":" << bytes << ",\"median_us\":" << s.med << ",\"p10_us\":" << s.p10
           << ",\"p90_us\":" << s.p90 << ",\"GBps_median\":" << gbps << ",\"GBps_best\":" << bytes / (s.p10 * 1e3)
           << ",\"inflight\":" << inflight << ",\"mismatches\":" << bad << "}";
        first = false;
        fprintf(stderr, "%-9s %-26s %-8s n=%3d %10.1f us (p10 %9.1f p90 %9.1f) %6.2f GB/s (best %6.2f) bad=%llu\n",
                kind.c_str(), mech.c_str(), pattern.c_str(), n, s.med, s.p10, s.p90, gbps, bytes / (s.p10 * 1e3), bad);
    };

    for (const auto& kind : strs(kinds)) {
        const auto t0 = std::chrono::steady_clock::now();
        Region R = make_region(kind, footprint_mb << 20, threads);
        fprintf(stderr, "region %s %llu MiB ready in %.1f s\n", kind.c_str(), (unsigned long long)footprint_mb,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        // ---- jobs: flat + experts (contiguous) + experts (6-row layout)
        struct Pat {
            std::string name;
            int n;
            bool flat, layout6;
        };
        std::vector<Pat> pats;
        pats.push_back({"flat", 0, true, false});
        for (int n : ns) pats.push_back({"exp", n, false, false});
        for (int n : ns) pats.push_back({"exp6", n, false, true});

        // ---- copy engine
        if (do_ce) {
            // 1 stream, and 4 streams (copies round-robin; the flat job is cut into 4 equal pieces)
            for (int nst : {1, 4}) {
                for (const auto& p : pats) {
                    std::vector<double> t;
                    unsigned long long bad = 0;
                    uint64_t bytes = 0;
                    for (int r = 0; r < reps + 1; ++r) {
                        Job j = p.flat ? flat_job(R, dbase, flat_mb << 20) : expert_job(R, dbase, 64, p.n, p.layout6, rng);
                        bytes = j.bytes;
                        std::vector<Entry> list = j.whole;
                        if (p.flat && nst > 1) list = split(j.whole, uint32_t(j.bytes / nst));
                        spin_kernel<<<1, 32, 0, C.st>>>(SPIN);
                        CK(cudaEventRecord(C.e0, C.st));
                        for (int s = 0; s < nst; ++s) CK(cudaStreamWaitEvent(C.xs[s], C.e0, 0));
                        for (size_t i = 0; i < list.size(); ++i) {
                            const Entry& e = list[i];
                            cudaStream_t s = nst == 1 ? C.st : C.xs[i % nst];
                            CK(cudaMemcpyAsync(reinterpret_cast<void*>(e.dst), R.host + e.soff, e.bytes, cudaMemcpyHostToDevice, s));
                        }
                        if (nst > 1)
                            for (int s = 0; s < nst; ++s) {
                                CK(cudaEventRecord(C.xe[s], C.xs[s]));
                                CK(cudaStreamWaitEvent(C.st, C.xe[s], 0));
                            }
                        CK(cudaEventRecord(C.e1, C.st));
                        CK(cudaStreamSynchronize(C.st));
                        float ms = 0;
                        CK(cudaEventElapsedTime(&ms, C.e0, C.e1));
                        if (r) t.push_back(ms * 1e3);
                        bad += verify(C, j);
                    }
                    row(kind, nst == 1 ? "ce_memcpy_1s" : "ce_memcpy_4s", p.name, p.n, bytes, stats(t), bad, 0);
                }
            }
        }
        // ---- SM variants (rotate variant order per rep)
        for (const auto& p : pats) {
            const int V = int(sm.size());
            if (!V) break;
            std::vector<std::vector<double>> t(V);
            std::vector<unsigned long long> bad(V, 0);
            uint64_t bytes = 0;
            for (int r = 0; r < reps + 1; ++r) {
                for (int k = 0; k < V; ++k) {
                    const int vi = (k + r) % V;
                    const SmVariant& v = sm[vi];
                    Job j = p.flat ? flat_job(R, dbase, flat_mb << 20) : expert_job(R, dbase, 64, p.n, p.layout6, rng);
                    bytes = j.bytes;
                    auto ent = split(j.whole, uint32_t(v.X));
                    if (ent.size() > C.ent_cap) {
                        fprintf(stderr, "entry capacity\n");
                        exit(2);
                    }
                    CK(cudaMemcpyAsync(C.d_ent, ent.data(), sizeof(Entry) * ent.size(), cudaMemcpyHostToDevice, C.st));
                    spin_kernel<<<1, 32, 0, C.st>>>(SPIN);
                    CK(cudaEventRecord(C.e0, C.st));
                    v.fn(C.st, C.d_ent, int(ent.size()), v.ctas, v.warps, C.d_sink);
                    CK(cudaEventRecord(C.e1, C.st));
                    CK(cudaStreamSynchronize(C.st));
                    CK(cudaGetLastError());
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, C.e0, C.e1));
                    if (r) t[vi].push_back(ms * 1e3);
                    if (v.store) bad[vi] += verify(C, j);
                }
            }
            for (int vi = 0; vi < V; ++vi)
                row(kind, sm[vi].name, p.name, p.n, bytes, stats(t[vi]), sm[vi].store ? bad[vi] : 0, sm[vi].inflight);
        }
        if (kind != "hostalloc") {
            CK(cudaHostUnregister(R.host));
            munmap(R.host, R.bytes);
        } else {
            CK(cudaFreeHost(R.host));
        }
    }
    js << "\n]}\n";
    if (!out.empty()) {
        FILE* f = fopen(out.c_str(), "w");
        fputs(js.str().c_str(), f);
        fclose(f);
    } else {
        fputs(js.str().c_str(), stdout);
    }
    fprintf(stderr, "DONE\n");
    return 0;
}
