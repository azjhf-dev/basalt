// src/models/deepseek41/directions.cpp - the BSLDIR01 bundle parser: CPU-only, so a test can link it without CUDA.
#include "basalt/models/deepseek41/directions.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace basalt::models::deepseek41::directions {

namespace {

uint32_t u32le(const uint8_t* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

uint64_t u64le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

}  // namespace

bool parse(const void* data, size_t bytes, int64_t expect_layers, int64_t expect_hidden, Bundle& out,
           std::string& err) {
    const uint8_t* p = (const uint8_t*) data;
    if (bytes < 24 || std::memcmp(p, kMagic, 8) != 0) {
        err = "not a BSLDIR01 directions bundle (tools/deepseek41_directions.py writes one)";
        return false;
    }
    const int64_t n_layers = u32le(p + 8), hidden = u32le(p + 12);
    const uint64_t mask = u64le(p + 16);
    if (n_layers < 1 || n_layers > kMaxLayers || hidden < 1) {
        err = "directions bundle: bad geometry";
        return false;
    }
    if (n_layers != expect_layers || hidden != expect_hidden) {
        err = "directions bundle is " + std::to_string((long long) n_layers) + " layers x " +
              std::to_string((long long) hidden) + ", the model is " + std::to_string((long long) expect_layers) +
              " x " + std::to_string((long long) expect_hidden);
        return false;
    }
    if (mask == 0 || (n_layers < kMaxLayers && (mask >> n_layers) != 0)) {
        err = "directions bundle: a layer mask outside the model's layers";
        return false;
    }
    const size_t need = 24 + (size_t) n_layers * (size_t) hidden * 4;
    if (bytes != need) {
        err = "directions bundle: " + std::to_string(bytes) + " bytes, expected " + std::to_string(need);
        return false;
    }
    std::vector<float> vectors((size_t) n_layers * (size_t) hidden);
    std::memcpy(vectors.data(), p + 24, vectors.size() * 4);
    for (int64_t l = 0; l < n_layers; ++l) {
        const float* row = vectors.data() + (size_t) l * (size_t) hidden;
        double norm = 0.0;
        for (int64_t j = 0; j < hidden; ++j) {
            if (!std::isfinite(row[j])) {
                err = "directions bundle: a non-finite value in layer " + std::to_string((long long) l);
                return false;
            }
            norm += (double) row[j] * row[j];
        }
        const bool present = (mask >> l) & 1;
        if (present && std::abs(norm - 1.0) > 1e-5) {
            err = "directions bundle: layer " + std::to_string((long long) l) + " is not a unit vector";
            return false;
        }
        if (!present && norm != 0.0) {
            err = "directions bundle: layer " + std::to_string((long long) l) + " is outside the mask but nonzero";
            return false;
        }
    }
    out.n_layers = n_layers;
    out.hidden = hidden;
    out.mask = mask;
    out.vectors = std::move(vectors);
    return true;
}

bool read(const std::string& path, int64_t expect_layers, int64_t expect_hidden, Bundle& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size < 0 || (unsigned long long) size > (24ull + kMaxLayers * 65536ull * 4)) {
        std::fclose(f);
        err = path + ": implausible size for a directions bundle";
        return false;
    }
    std::vector<uint8_t> data((size_t) size);
    const bool got = data.empty() || std::fread(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    if (!got) {
        err = "cannot read " + path;
        return false;
    }
    return parse(data.data(), data.size(), expect_layers, expect_hidden, out, err);
}

}  // namespace basalt::models::deepseek41::directions
