// src/models/ernie55/kernel_test.cpp - parity between the HCA CUDA attention kernel and the CPU reference.
// Loads a converted tiny pack (argv[1]), prepares the attention inputs with ref::hca_prepare, runs the
// kernel's hca_attention_forward (GPU on CUDA builds, the identical-math CPU fallback otherwise) and the
// reference's per-(token, head) hca_attention_ref, and compares every output element. Prints which path
// ran so a GPU box confirms it exercised the CUDA kernel.
#include "models/ernie55/kernels/hca_attention.hpp"
#include "models/ernie55/ref.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace basalt::models::ernie55;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    check(kernels::hca_attention_forward(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr) ==
              false,
          "forward rejects empty inputs");
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s PACK - pure checks only without a pack\n", argv[0]);
        if (failures) return 1;
        std::fprintf(stderr, "ernie55 kernel: all checks passed (pure only)\n");
        return 0;
    }

    Pack p;
    std::string err;
    check(open_pack(argv[1], p, err), "open_pack: " + err);
    if (failures) return 1;

    ref::HcaWeights hw;
    check(ref::load_hca(argv[1], p, 0, hw, err), "load_hca: " + err);
    if (failures) return 1;

    const Geometry& g = p.geo;
    const int T = 200;  // same span as the reference test: covers window edges and one compressed block
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<ref::Vec> x(T, ref::Vec(g.n_embd));
    for (auto& v : x)
        for (float& f : v) f = nd(rng);

    const ref::HcaPrepared prep = ref::hca_prepare(g, hw, x);
    check(prep.T == T && prep.nb == T / 128, "prepared shape");
    check((int) prep.q.size() == T * g.n_head * g.head_dim && (int) prep.kv.size() == T * g.head_dim &&
              (int) prep.comp.size() == prep.nb * g.head_dim,
          "prepared flat buffers");

    // the kernel (CUDA on toolkit builds, the identical-math CPU fallback otherwise)
    std::vector<float> got((size_t) T * g.n_head * g.head_dim, -1.f);
    const bool ran = kernels::hca_attention_forward(prep.q.data(), prep.kv.data(), prep.comp.data(),
                                                    hw.sinks.data(), T, g.n_head, g.head_dim, g.window,
                                                    g.compress_ratio[0] ? g.compress_ratio[0] : 128,
                                                    got.data());
    check(ran, "hca_attention_forward ran");
    std::fprintf(stderr, "path: %s\n", kernels::hca_attention_forward_is_cuda() ? "cuda" : "cpu fallback");
    if (failures) return 1;

    // the reference core, one (token, head) at a time
    double max_rel = 0.0;
    std::vector<float> want(g.head_dim);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < g.n_head; ++h) {
            ref::hca_attention_ref(g, hw, prep, t, h, want.data());
            const float* o = got.data() + ((size_t) t * g.n_head + h) * g.head_dim;
            for (int d = 0; d < g.head_dim; ++d) {
                check(std::isfinite(o[d]), "kernel output finite");
                const double rel = std::fabs((double) o[d] - want[d]) / (1.0 + std::fabs((double) want[d]));
                if (rel > max_rel) max_rel = rel;
            }
        }
    check(max_rel < 1e-5, "kernel matches the CPU reference");
    std::fprintf(stderr, "kernel vs ref max rel |diff|: %.3g\n", max_rel);

    if (failures) {
        std::fprintf(stderr, "ernie55 kernel: %d check(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "ernie55 kernel: all checks passed\n");
    return 0;
}
