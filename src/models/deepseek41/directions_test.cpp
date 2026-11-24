// src/models/deepseek41/directions_test.cpp - the BSLDIR01 bundle parser on synthetic bundles (no GPU).
#include "basalt/models/deepseek41/directions.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt::models::deepseek41::directions;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                         \
        }                                                                     \
    } while (0)

namespace {

void put32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}
void put64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}
void putf(std::vector<uint8_t>& b, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    put32(b, u);
}

// A bundle of `layers` rows of width `hidden`: layer `mask` rows carry a unit vector (e0 = 1), the rest are zero.
std::vector<uint8_t> bundle(uint32_t layers, uint32_t hidden, uint64_t mask, float row0 = 1.0f) {
    std::vector<uint8_t> b(kMagic, kMagic + 8);
    put32(b, layers);
    put32(b, hidden);
    put64(b, mask);
    for (uint32_t l = 0; l < layers; ++l) {
        const bool on = (mask >> l) & 1;
        for (uint32_t j = 0; j < hidden; ++j) putf(b, on && j == 0 ? row0 : 0.0f);
    }
    return b;
}

}  // namespace

int main() {
    Bundle out;
    std::string err;

    // a well-formed bundle parses, with its geometry, mask and vectors intact
    {
        const std::vector<uint8_t> b = bundle(40, 5120, 0xFFFFFFFCULL);   // layers 2..39, like the heretic set
        CHECK(parse(b.data(), b.size(), 40, 5120, out, err));
        CHECK(out.n_layers == 40 && out.hidden == 5120 && out.mask == 0xFFFFFFFCULL);
        CHECK(out.vectors.size() == 40u * 5120u);
        CHECK(out.vectors[0] == 0.0f);                          // layer 0 masked out: a zero row
        CHECK(out.vectors[(size_t) 2 * 5120] == 1.0f);          // layer 2's direction: e0
        CHECK(out.vectors[(size_t) 2 * 5120 + 1] == 0.0f);
    }

    // geometry must match the model
    for (auto [L, H] : {std::pair{39, 5120}, {41, 5120}, {40, 4096}}) {
        const std::vector<uint8_t> b = bundle(40, 5120, 0xFULL);
        err.clear();
        CHECK(!parse(b.data(), b.size(), L, H, out, err) && !err.empty());
    }

    // bad magic, truncated file, zero mask, a mask bit past the layers
    {
        std::vector<uint8_t> b = bundle(40, 5120, 0xFULL);
        b[0] = 'X';
        CHECK(!parse(b.data(), b.size(), 40, 5120, out, err));
        b = bundle(40, 5120, 0xFULL);
        CHECK(!parse(b.data(), b.size() - 4, 40, 5120, out, err));
        b = bundle(40, 5120, 0x0ULL);
        CHECK(!parse(b.data(), b.size(), 40, 5120, out, err));
        b = bundle(40, 5120, 1ULL << 40);
        CHECK(!parse(b.data(), b.size(), 40, 5120, out, err));
    }

    // masked rows must be unit vectors; unmasked rows must be zero; every value finite
    {
        std::vector<uint8_t> b = bundle(4, 8, 0x2ULL, 0.5f);   // layer 1's row is not a unit vector
        CHECK(!parse(b.data(), b.size(), 4, 8, out, err));
        b = bundle(4, 8, 0x2ULL);
        float one = 1.0f;
        std::memcpy(b.data() + 24, &one, 4);                   // a nonzero value in a masked-out (zero) row
        CHECK(!parse(b.data(), b.size(), 4, 8, out, err));
        b = bundle(4, 8, 0x2ULL);
        const float nan = std::nanf("");
        std::memcpy(b.data() + 24 + 4, &nan, 4);
        CHECK(!parse(b.data(), b.size(), 4, 8, out, err));
    }

    // a layer count over the mask's width is refused
    {
        const std::vector<uint8_t> b = bundle(65, 8, 0xFULL);
        CHECK(!parse(b.data(), b.size(), 65, 8, out, err));
    }

    if (g_fail == 0) std::printf("directions_test: OK\n");
    return g_fail == 0 ? 0 : 1;
}
