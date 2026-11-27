// src/kernels/cuda/lora_ops.cu - see include/basalt/kernels/dense_ops.hpp (the LoRA products at the end).
//
// The two products of a runtime LoRA adapter (docs/lora.md): delta = scale * (x . A) . B, A and B the adapter's fp16
// factors as the GGUF stores them (A [R][K] rows, B [N][R] rows), x and the output fp32 rows.  Plain kernels, no
// tensor cores: rank is small (96 here) and v1 is about matching the reference, not speed.
#include "basalt/kernels/dense_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// tmp[t][k] = sum_i x[t][i] * a[k][i]: one warp per (t, k), the warp striding K
__global__ void lora_product_a_kernel(const float* __restrict__ x, const uint16_t* __restrict__ a,
                                      float* __restrict__ tmp, int64_t R, int64_t K) {
    const int64_t t = blockIdx.x;
    const int k = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (k >= R) return;
    const int lane = threadIdx.x & 31;
    const float* xr = x + t * K;
    const __half* ar = (const __half*) a + (int64_t) k * K;
    float s = 0.0f;
    for (int64_t i = lane; i < K; i += 32) s = fmaf(xr[i], __half2float(ar[i]), s);
    s = warp_sum(s);
    if (lane == 0) tmp[t * R + k] = s;
}

// y[t][n] (+)= scale * sum_k tmp[t][k] * b[n][k]: one thread per (t, n), the rank loop serial
__global__ void lora_product_b_kernel(const float* __restrict__ tmp, const uint16_t* __restrict__ b,
                                      float* __restrict__ y, float scale, int64_t N, int64_t R, bool add) {
    const int64_t t = blockIdx.y;
    const int64_t n = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* tr = tmp + t * R;
    const __half* br = (const __half*) b + n * R;
    float s = 0.0f;
    for (int64_t k = 0; k < R; ++k) s = fmaf(tr[k], __half2float(br[k]), s);
    const int64_t i = t * N + n;
    if (add) y[i] += scale * s;
    else y[i] = scale * s;
}

}  // namespace

void lora_product_a(const float* x, const uint16_t* a, float* tmp, int64_t T, int64_t R, int64_t K, void* stream) {
    if (T <= 0) return;
    dim3 grid((unsigned) T, (unsigned) ((R + 3) / 4));
    lora_product_a_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(x, a, tmp, R, K);
    check("lora_product_a");
}

void lora_product_b(const float* tmp, const uint16_t* b, float* y, float scale, int64_t T, int64_t N, int64_t R,
                    bool add, void* stream) {
    if (T <= 0) return;
    dim3 grid((unsigned) ((N + 255) / 256), (unsigned) T);
    lora_product_b_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(tmp, b, y, scale, N, R, add);
    check("lora_product_b");
}

}  // namespace basalt::kernels
