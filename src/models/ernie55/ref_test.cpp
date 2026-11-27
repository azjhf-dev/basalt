// src/models/ernie55/ref_test.cpp - pure rope/norm/dequant checks plus an end-to-end run of the CPU-reference
// HCA and MLA-DSA layers on a converted tiny pack (argv[1]), with an optional numpy-oracle comparison
// (argv[2], written by tools/convert/ernie55_ref_oracle.py: i32 T, T*n_embd f32 hca, i32 T, T*n_embd f32 mla,
// T*T f32 indexer scores). Host-only: no CUDA, no runtime beyond the pack loader.
#include "models/ernie55/ref.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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

bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }

std::vector<ref::Vec> random_x(int T, int n_embd, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<ref::Vec> x(T, ref::Vec(n_embd));
    for (auto& v : x)
        for (float& f : v) f = nd(rng);
    return x;
}

bool all_finite(const std::vector<ref::Vec>& m) {
    for (const auto& v : m)
        for (float f : v)
            if (!std::isfinite(f)) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // ---- pure: dequant
    check(near(ref::fp16_to_f32(0x3c00), 1.f), "fp16 one");
    check(near(ref::fp16_to_f32(0xc000), -2.f), "fp16 minus two");
    check(near(ref::bf16_to_f32(0x3f80), 1.f), "bf16 one");

    // ---- pure: rope
    {
        ref::RopeParams p;
        p.rot = 8;
        p.base = 10000.f;
        float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        float y[8];
        std::memcpy(y, x, sizeof x);
        ref::rope_inplace(x, 0, 0, p);
        check(std::memcmp(x, y, sizeof x) == 0, "rope at position 0 is the identity");
        ref::rope_inplace(x, 0, 17, p);
        float n0 = 0.f;
        for (float f : x) n0 += f * f;
        check(near(n0, 204.f), "rope preserves the norm");
        ref::rope_back_inplace(x, 0, 17, p);
        for (int i = 0; i < 8; ++i) check(near(x[i], y[i]), "rope then de-rope round-trips");
    }
    {
        // the YaRN ramp: long-wavelength pairs get scaled, short ones do not
        ref::RopeParams p;
        p.rot = 64;
        p.base = 160000.f;
        p.freq_scale = 1.f / 16.f;
        p.n_ctx_orig = 65536;
        p.beta_fast = 32.f;
        p.beta_slow = 1.f;
        const float plain0 = std::pow(p.base, 0.f);
        check(near(ref::rope_inv_freq(p, 0), 1.f), "yarn keeps the highest frequency");
        check(near(ref::rope_inv_freq(p, 31), plain0 * std::pow(p.base, -62.f / 64.f) * p.freq_scale, 1e-3f),
              "yarn scales the lowest frequency");
        ref::RopeParams q = p;
        q.n_ctx_orig = 0;
        check(ref::rope_inv_freq(q, 31) > ref::rope_inv_freq(p, 31), "yarn lowers low frequencies");
    }
    {
        // NEOX pairs (i, i + rot/2)
        ref::RopeParams p;
        p.rot = 4;
        p.base = 10000.f;
        float x[4] = {1, 0, 0, 1};
        ref::rope_neox_inplace(x, 1, p);
        const float a1 = ref::rope_inv_freq(p, 0), a2 = ref::rope_inv_freq(p, 1);
        check(near(x[0], std::cos(a1)) && near(x[2], std::sin(a1)), "neox rotates pair (0, 2)");
        check(near(x[1], -std::sin(a2)) && near(x[3], std::cos(a2)), "neox rotates pair (1, 3)");
    }

    if (argc < 2) {
        std::fprintf(stderr, "usage: %s PACK [ORACLE]\n", argv[0]);
        return 2;
    }

    // ---- end-to-end on a converted tiny pack
    Pack p;
    std::string err;
    const bool opened = open_pack(argv[1], p, err);
    check(opened, "open_pack: " + err);
    if (failures) return 1;
    check(p.geo.n_embd == 256 && p.geo.n_head == 2 && p.geo.head_dim == 128, "tiny geometry");
    check(p.geo.mla_qk == 128 && p.geo.mla_v == 64 && p.geo.rope_dim == 32, "tiny MLA dims");

    ref::HcaWeights hw;
    const bool hca_ok = ref::load_hca(argv[1], p, 0, hw, err);
    check(hca_ok, "load_hca: " + err);
    ref::MlaWeights mw;
    const bool mla_ok = ref::load_mla(argv[1], p, 2, mw, err);
    check(mla_ok, "load_mla: " + err);
    if (failures) return 1;

    const int TH = 200;   // 1 complete 128-block + 72 raw
    const std::vector<ref::Vec> xh = random_x(TH, p.geo.n_embd, 42);
    const std::vector<ref::Vec> hout = ref::hca_forward(p.geo, hw, xh);
    check((int) hout.size() == TH && (int) hout[0].size() == p.geo.n_embd, "hca output shape");
    check(all_finite(hout), "hca output finite");
    {
        // causality: x[100] sits in the one completed block; nothing at t <= 100 may change
        std::vector<ref::Vec> x2 = xh;
        x2[100][0] += 3.f;
        const std::vector<ref::Vec> h2 = ref::hca_forward(p.geo, hw, x2);
        bool same = true, later_differs = false;
        for (int t = 0; t < 100; ++t)
            for (int d = 0; d < p.geo.n_embd; ++d) same &= near(hout[t][d], h2[t][d], 1e-5f);
        for (int d = 0; d < p.geo.n_embd; ++d) later_differs |= !near(hout[150][d], h2[150][d], 1e-5f);
        check(same, "hca causality: t < 100 unchanged");
        check(later_differs, "hca sees the completed-block change through the compressor");
    }
    {
        // a change inside the current (incomplete) block only travels through the raw window
        std::vector<ref::Vec> x2 = xh;
        x2[150][1] += 3.f;
        const std::vector<ref::Vec> h2 = ref::hca_forward(p.geo, hw, x2);
        bool same = true;
        for (int t = 0; t < 150; ++t)
            for (int d = 0; d < p.geo.n_embd; ++d) same &= near(hout[t][d], h2[t][d], 1e-5f);
        check(same, "hca causality: t < 150 unchanged");
    }

    const int TM = 64;   // < top_k + window: attention is dense
    const std::vector<ref::Vec> xm = random_x(TM, p.geo.n_embd, 43);
    const ref::MlaResult m = ref::mla_forward(p.geo, mw, xm);
    check((int) m.out.size() == TM && (int) m.out[0].size() == p.geo.n_embd, "mla output shape");
    check(all_finite(m.out) && all_finite(m.idx_scores) == false, "mla output finite, scores masked above diag");
    {
        bool masked = true, finite_below = true;
        for (int t = 0; t < TM; ++t)
            for (int j = 0; j < TM; ++j) {
                if (j > t) masked &= m.idx_scores[t][j] == -std::numeric_limits<float>::infinity();
                else finite_below &= std::isfinite(m.idx_scores[t][j]);
            }
        check(masked && finite_below, "indexer scores causal");
        std::vector<ref::Vec> x2 = xm;
        x2[50][0] += 3.f;
        const ref::MlaResult m2 = ref::mla_forward(p.geo, mw, x2);
        bool same = true;
        for (int t = 0; t < 50; ++t)
            for (int d = 0; d < p.geo.n_embd; ++d) same &= near(m.out[t][d], m2.out[t][d], 1e-5f);
        check(same, "mla causality: t < 50 unchanged");
    }

    // ---- optional numpy oracle comparison: the oracle carries its own inputs so both sides run identical x
    if (argc >= 3) {
        std::ifstream f(argv[2], std::ios::binary);
        check((bool) f, "oracle file opens");
        auto read_i32 = [&]() {
            int32_t v;
            f.read((char*) &v, 4);
            return v;
        };
        auto read_f32 = [&](int n) {
            std::vector<float> v(n);
            f.read((char*) v.data(), 4 * n);
            return v;
        };
        auto as_x = [&](const std::vector<float>& flat, int T) {
            std::vector<ref::Vec> x(T, ref::Vec(p.geo.n_embd));
            for (int t = 0; t < T; ++t)
                std::copy(flat.begin() + t * p.geo.n_embd, flat.begin() + (t + 1) * p.geo.n_embd, x[t].begin());
            return x;
        };
        check(read_i32() == TH, "oracle hca T");
        const std::vector<ref::Vec> ohx = as_x(read_f32(TH * p.geo.n_embd), TH);
        const std::vector<float> oh = read_f32(TH * p.geo.n_embd);
        check(read_i32() == TM, "oracle mla T");
        const std::vector<ref::Vec> omx = as_x(read_f32(TM * p.geo.n_embd), TM);
        const std::vector<float> om = read_f32(TM * p.geo.n_embd);
        const std::vector<float> os = read_f32(TM * TM);
        check((bool) f, "oracle file complete");
        const std::vector<ref::Vec> hrun = ref::hca_forward(p.geo, hw, ohx);
        const ref::MlaResult mrun = ref::mla_forward(p.geo, mw, omx);
        float max_h = 0.f, max_m = 0.f, max_s = 0.f;
        for (int t = 0; t < TH; ++t)
            for (int d = 0; d < p.geo.n_embd; ++d) {
                const float e = oh[t * p.geo.n_embd + d];
                max_h = std::max(max_h, std::fabs(hrun[t][d] - e) / (1.f + std::fabs(e)));
            }
        for (int t = 0; t < TM; ++t)
            for (int d = 0; d < p.geo.n_embd; ++d) {
                const float e = om[t * p.geo.n_embd + d];
                max_m = std::max(max_m, std::fabs(mrun.out[t][d] - e) / (1.f + std::fabs(e)));
            }
        for (int t = 0; t < TM; ++t)
            for (int j = 0; j <= t; ++j) {
                const float e = os[t * TM + j];
                max_s = std::max(max_s, std::fabs(mrun.idx_scores[t][j] - e) / (1.f + std::fabs(e)));
            }
        std::printf("oracle max rel |diff|: hca %.3g  mla %.3g  indexer %.3g\n", max_h, max_m, max_s);
        check(max_h < 5e-4f, "hca matches the numpy oracle");
        check(max_m < 5e-4f, "mla matches the numpy oracle");
        check(max_s < 5e-4f, "indexer scores match the numpy oracle");
    }

    if (failures) {
        std::fprintf(stderr, "ernie55 ref: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("ernie55 ref: all checks passed\n");
    return 0;
}
