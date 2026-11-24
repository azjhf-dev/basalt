// src/models/deepseek41/kernels/directions.cu - the runtime-directions state and the ablation kernel.
//
// One block per token row of the layer's wo_b output: dot(r, y) over the full width, then the update.  The round
// ops are the strict __fadd_rn/__fmul_rn/__fsub_rn forms of the TensorFold reference kernel, so two runs of the
// same request are bitwise identical.
#include "basalt/models/deepseek41/directions.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace basalt::models::deepseek41::directions {

namespace {

constexpr int THREADS = 256;

Bundle g_bundle;
bool g_loaded = false;
float g_strength = 0.0f;   // the state the current request runs with
uint64_t g_mask = 0;

// the vectors on every device a layer's wo_b output is computed on (lazy; the bundle is under a MiB)
constexpr int kDevices = 64;
float* g_dev[kDevices] = {};

int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}

float* upload_here() {
    float*& t = g_dev[cur_device()];
    if (t != nullptr) return t;
    const size_t bytes = g_bundle.vectors.size() * sizeof(float);
    if (cudaMalloc(&t, bytes) != cudaSuccess ||
        cudaMemcpy(t, g_bundle.vectors.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(t);
        t = nullptr;
        throw std::runtime_error("directions: device allocation failed");
    }
    return t;
}

__global__ void directions_kernel(float* __restrict__ y, const float* __restrict__ r, float alpha, int width,
                                  unsigned int rows) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float* yrow = y + (size_t) row * width;
    __shared__ float partial[THREADS];
    const unsigned int t = threadIdx.x;
    float dot = 0.0f;
    for (int j = t; j < width; j += THREADS) dot = __fadd_rn(dot, __fmul_rn(yrow[j], r[j]));
    partial[t] = dot;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride >>= 1) {
        if (t < stride) partial[t] = __fadd_rn(partial[t], partial[t + stride]);
        __syncthreads();
    }
    const float scale = __fmul_rn(alpha, partial[0]);
    for (int j = t; j < width; j += THREADS) yrow[j] = __fsub_rn(yrow[j], __fmul_rn(scale, r[j]));
}

}  // namespace

bool open(const std::string& path, int64_t n_layers, int64_t n_embd, std::string& err) {
    Bundle b;
    if (!read(path, n_layers, n_embd, b, err)) return false;
    int prev = 0;   // a new bundle replaces the old one on every device
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d] == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d]);
        g_dev[d] = nullptr;
    }
    cudaSetDevice(prev);
    g_bundle = std::move(b);
    g_loaded = true;
    g_strength = 0.0f;
    g_mask = 0;
    return true;
}

bool loaded() { return g_loaded; }

std::string summary() {
    if (!g_loaded) return {};
    int n = 0;
    for (int64_t l = 0; l < g_bundle.n_layers; ++l) n += (int) ((g_bundle.mask >> l) & 1);
    return std::to_string(n) + " directions over " + std::to_string((long long) g_bundle.n_layers) + " layers";
}

bool set(float strength, uint64_t mask, std::string& err) {
    if (!std::isfinite(strength) || strength < 0.0f || strength > 1.0f) {
        err = "strength must be in [0,1]";
        return false;
    }
    if (!g_loaded) {
        if (strength == 0.0f) return true;
        err = "the engine was started without --directions";
        return false;
    }
    if (mask == ~0ULL) mask = g_bundle.mask;   // an absent ablmask: every layer the bundle carries
    if (strength > 0.0f && (mask & ~g_bundle.mask) != 0) {
        err = "the layer mask names a layer the directions bundle does not carry";
        return false;
    }
    g_strength = strength;
    g_mask = mask;
    return true;
}

bool active() { return g_loaded && g_strength > 0.0f; }

void apply(int il, float* out, int64_t T, void* stream) {
    if (!g_loaded || g_strength == 0.0f || T < 1 || il < 0 || il >= g_bundle.n_layers) return;
    if (((g_mask >> il) & 1) == 0) return;
    float* r = upload_here() + (size_t) il * (size_t) g_bundle.hidden;
    directions_kernel<<<(unsigned int) T, THREADS, 0, (cudaStream_t) stream>>>(out, r, g_strength,
                                                                               (int) g_bundle.hidden, (unsigned int) T);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("directions: launch failed");
}

}  // namespace basalt::models::deepseek41::directions
