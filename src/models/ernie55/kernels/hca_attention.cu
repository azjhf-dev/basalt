// src/models/ernie55/kernels/hca_attention.cu - see hca_attention.hpp. The head math lives in one
// BASALT_HD function so nvcc and the host compiler compile the identical code: the CUDA kernel and the CPU
// fallback differ only in how (token, head) pairs are iterated. Without a toolkit this file is plain C++
// (compile with `-x c++`); the CMake build compiles it with nvcc and takes the GPU path.
#include "models/ernie55/kernels/hca_attention.hpp"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define BASALT_HD __host__ __device__
#else
#define BASALT_HD
#endif

#include <cmath>
#include <cstddef>

namespace basalt::models::ernie55::kernels {

namespace {

BASALT_HD float dot_f32(const float* a, const float* b, int n) {
    double acc = 0.0;  // double accumulation, like the CPU reference, keeps the parity bound at fp32 rounding
    for (int i = 0; i < n; ++i) acc += (double) a[i] * b[i];
    return (float) acc;
}

// One (token, head): scores against kv[j0..t] then comp[0..ncomp), softmax whose max and denominator both
// include the sink (as in ref::softmax_with_sink), weighted sum, normalized at the end. qh/out point at the
// (t*nh + h)*hd slice. Two passes over the keys, dots recomputed - simple beats fast here.
BASALT_HD void hca_attention_head(const float* qh, const float* kv, const float* comp, float sink, int t,
                                  int hd, int window, int ratio, float scale, float* out) {
    const int j0 = (t - window + 1) > 0 ? (t - window + 1) : 0;
    const int ncomp = ratio > 0 ? t / ratio : 0;
    float mx = sink;
    for (int j = j0; j <= t; ++j) {
        const float s = scale * dot_f32(qh, kv + (size_t) j * hd, hd);
        mx = mx > s ? mx : s;
    }
    for (int b = 0; b < ncomp; ++b) {
        const float s = scale * dot_f32(qh, comp + (size_t) b * hd, hd);
        mx = mx > s ? mx : s;
    }
    double den = exp((double) sink - mx);
    for (int d = 0; d < hd; ++d) out[d] = 0.f;
    for (int j = j0; j <= t; ++j) {
        const float w = (float) exp((double) (scale * dot_f32(qh, kv + (size_t) j * hd, hd)) - mx);
        den += w;
        const float* v = kv + (size_t) j * hd;
        for (int d = 0; d < hd; ++d) out[d] += w * v[d];
    }
    for (int b = 0; b < ncomp; ++b) {
        const float w = (float) exp((double) (scale * dot_f32(qh, comp + (size_t) b * hd, hd)) - mx);
        den += w;
        const float* v = comp + (size_t) b * hd;
        for (int d = 0; d < hd; ++d) out[d] += w * v[d];
    }
    const float inv = (float) (1.0 / den);
    for (int d = 0; d < hd; ++d) out[d] *= inv;
}

#if defined(__CUDACC__)
__global__ void hca_attention_kernel(const float* q, const float* kv, const float* comp,
                                     const float* sinks, int T, int nh, int hd, int window, int ratio,
                                     float scale, float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * nh) return;
    const int t = idx / nh;
    hca_attention_head(q + (size_t) idx * hd, kv, comp, sinks[idx % nh], t, hd, window, ratio, scale,
                       out + (size_t) idx * hd);
}
#endif

}  // namespace

bool hca_attention_forward_is_cuda() {
#if defined(__CUDACC__)
    return true;
#else
    return false;
#endif
}

bool hca_attention_forward(const float* q, const float* kv, const float* comp, const float* sinks, int T,
                           int nh, int hd, int window, int ratio, float* out) {
    if (T <= 0 || nh <= 0 || hd <= 0) return false;
    const float scale = 1.f / std::sqrt((float) hd);
#if defined(__CUDACC__)
    const size_t nq = (size_t) T * nh * hd, nkv = (size_t) T * hd,
                 ncomp = (size_t) (ratio > 0 ? T / ratio : 0) * hd;
    float *dq = nullptr, *dkv = nullptr, *dc = nullptr, *ds = nullptr, *dout = nullptr;
    bool ok = cudaMalloc(&dq, nq * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dkv, nkv * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dc, (ncomp ? ncomp : 1) * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&ds, nh * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&dout, nq * sizeof(float)) == cudaSuccess;
    if (ok)
        ok = cudaMemcpy(dq, q, nq * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(dkv, kv, nkv * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(ds, sinks, nh * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok && ncomp)
        ok = cudaMemcpy(dc, comp, ncomp * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        const int threads = 256, blocks = (T * nh + threads - 1) / threads;
        hca_attention_kernel<<<blocks, threads>>>(dq, dkv, dc, ds, T, nh, hd, window, ratio, scale, dout);
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    if (ok) ok = cudaMemcpy(out, dout, nq * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dq);
    cudaFree(dkv);
    cudaFree(dc);
    cudaFree(ds);
    cudaFree(dout);
    return ok;
#else
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < nh; ++h)
            hca_attention_head(q + ((size_t) t * nh + h) * hd, kv, comp, sinks[h], t, hd, window, ratio,
                               scale, out + ((size_t) t * nh + h) * hd);
    return true;
#endif
}

}  // namespace basalt::models::ernie55::kernels
