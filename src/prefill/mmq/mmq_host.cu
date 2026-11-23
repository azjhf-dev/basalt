// The host side of the MMQ kernels: error reporting, the properties of each device and the cache of scratch buffers
// that the stream-k fixup of a launch takes from, and the explicit load of a kernel.
#include "mmq_common.cuh"

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include "basalt/log.hpp"

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    basalt::log::error("basalt", "mmq: %s: %s\n  in %s at %s:%d", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_func_load(const void * symbol) {
    using Load = CUresult (CUDAAPI *)(CUfunction);
    static const Load load = [] {
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuFuncLoad", &p, 12040, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess) {
            (void) cudaGetLastError();
            p = nullptr;
        }
        return (Load) p;
    }();
    cudaFunction_t f = nullptr;
    CUDA_CHECK(cudaGetFuncBySymbol(&f, symbol));
    if (load == nullptr || load((CUfunction) f) != CUDA_SUCCESS)
        GGML_ABORT("cannot load a kernel (cuFuncLoad)");
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        for (int id = 0; id < n; ++id) {
            // cudaDeviceGetAttribute is ABI-stable across runtime versions, a cudaDeviceProp is not.
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            auto & d = in.devices[id];
            d.smpbo = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
            d.nsm = attr(cudaDevAttrMultiProcessorCount);
            d.warp_size = attr(cudaDevAttrWarpSize);
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).  The cache
// is per device and shared by every context, and mmq::load warms it at start: a first cudaMalloc mid-run can meet a
// card packed to its reserve (decode at 262K context and --peer-reserve-mib 128 died here with "out of memory").
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        const cudaError_t err = cudaMalloc(&p, size);
        if (err != cudaSuccess) {
            size_t free_b = 0, total_b = 0;
            (void) cudaMemGetInfo(&free_b, &total_b);
            ggml_cuda_error("cudaMalloc(&p, size)", __func__, __FILE__, __LINE__,
                            (std::string(cudaGetErrorString(err)) + ": " + std::to_string(size >> 20) +
                             " MiB of MMQ stream-k scratch with " + std::to_string(free_b >> 20) + " of " +
                             std::to_string(total_b >> 20) + " MiB free on the card - raise --vram-reserve-mib / "
                             "--peer-reserve-mib").c_str());
        }
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) cudaFree(b.p);
    }
};

CachingPool & shared_pool(int device) {
    static CachingPool * pools = new CachingPool[GGML_CUDA_MAX_DEVICES]();   // leaked: a process-lifetime cache
    return pools[device >= 0 && device < GGML_CUDA_MAX_DEVICES ? device : 0];
}

// Every context's pool forwards to its device's shared cache, so one warm covers every prompt path.
struct SharedPool final : ggml_cuda_pool {
    int dev;
    explicit SharedPool(int d) : dev(d) {}
    void * alloc(size_t size, size_t * actual_size) override { return shared_pool(dev).alloc(size, actual_size); }
    void free(void * ptr, size_t s) override { shared_pool(dev).free(ptr, s); }
};
}  // namespace

void ggml_cuda_mmq_scratch_reserve(int device, size_t bytes, int count) {
    CachingPool & p = shared_pool(device);
    std::vector<void *> held;
    for (int i = 0; i < count; ++i) {
        size_t a = 0;
        held.push_back(p.alloc(bytes, &a));
    }
    for (void * q : held) p.free(q, bytes);
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
    return std::unique_ptr<ggml_cuda_pool>(new SharedPool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
