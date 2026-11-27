// src/dense/lora_test.cpp - the dense LoRA loader's contract test (CPU, no GPU). Builds tiny adapter GGUF
// fixtures and checks parsing, validation, name resolution, the scale math and the CPU reference application
// (basalt/dense/lora.hpp, docs/lora.md).  With a path argument it instead checks the published super-adapter
// file: dense_lora_test /path/to/Qwen3.8-27B-super-adapter-lora-F16.gguf
#include "basalt/dense/lora.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using namespace basalt;
using namespace basalt::dense;

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                               \
        }                                                             \
    } while (0)

// ---- a tiny GGUF v3 writer for fixtures ----

static uint16_t f32_to_f16(float v) {
    uint32_t f;
    std::memcpy(&f, &v, 4);
    const uint32_t s = (f >> 16) & 0x8000;
    const int e = (int) ((f >> 23) & 0xff) - 112;
    const uint32_t m = f & 0x7fffff;
    if (e <= 0) return (uint16_t) s;                 // underflows to zero (fixtures avoid it)
    if (e >= 31) return (uint16_t) (s | 0x7c00);     // inf
    return (uint16_t) (s | ((uint32_t) e << 10) | (m >> 13));
}

struct FixtureTensor {
    std::string name;
    uint64_t ne0, ne1;       // GGUF order: dim 0 fastest
    std::vector<uint16_t> f16;
};

struct Fixture {
    std::string arch = "qwen35";
    bool adapter_meta = true;
    bool write_alpha = true;
    float alpha = 16.0f;
    std::vector<FixtureTensor> tensors;

    std::string write(const std::string& path) const {
        std::vector<uint8_t> b;
        auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i))); };
        auto u64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i))); };
        auto str = [&](const std::string& s) { u64(s.size()); b.insert(b.end(), s.begin(), s.end()); };
        u32(0x46554747u);   // "GGUF"
        u32(3);
        u64(tensors.size());
        u64(adapter_meta ? 4 : 1);
        str("general.architecture");
        u32(8);
        str(arch);
        if (adapter_meta) {
            str("general.type");
            u32(8);
            str("adapter");
            str("adapter.type");
            u32(8);
            str("lora");
            if (write_alpha) {
                str("adapter.lora.alpha");
                u32(6);   // F32
                float af = alpha;
                uint32_t ab;
                std::memcpy(&ab, &af, 4);
                u32(ab);
            } else {
                // keep the KV count honest: 4 entries promised, so write a filler key
                str("adapter.lora.note");
                u32(8);
                str("none");
            }
        }
        // tensor directory with offsets, then aligned payloads
        uint64_t off = 0;
        for (const FixtureTensor& t : tensors) {
            str(t.name);
            u32(2);
            u64(t.ne0);
            u64(t.ne1);
            u32(1);   // F16
            u64(off);
            const uint64_t bytes = t.ne0 * t.ne1 * 2;
            off += (bytes + 31) & ~31ull;
        }
        while (b.size() % 32) b.push_back(0);
        for (const FixtureTensor& t : tensors) {
            for (uint16_t h : t.f16) {
                b.push_back((uint8_t) h);
                b.push_back((uint8_t) (h >> 8));
            }
            while (b.size() % 32) b.push_back(0);
        }
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return "cannot write " + path;
        const size_t n = std::fwrite(b.data(), 1, b.size(), f);
        std::fclose(f);
        return n == b.size() ? std::string() : "short write of " + path;
    }
};

static Geometry test_geometry() {
    Geometry g;
    g.n_layers = 4;
    g.n_embd = 8;
    g.n_ff = 16;
    g.n_head = 1;
    g.head_dim = 8;
    g.n_kv = 1;
    g.kinds = {Mixer::Gdn, Mixer::Gdn, Mixer::Gdn, Mixer::Attention};
    return g;
}

// A [8][4] with A[i + 8k] = (i == k), and a [4][16] with B[k + 4j] = (k == j % 4) ? 0.5 : 0.
static FixtureTensor gate_a(const char* name = "blk.1.ffn_gate.weight.lora_a") {
    FixtureTensor t{name, 8, 4, {}};
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 8; ++i) t.f16.push_back(f32_to_f16(i == k ? 1.0f : 0.0f));
    return t;
}
static FixtureTensor gate_b(const char* name = "blk.1.ffn_gate.weight.lora_b") {
    FixtureTensor t{name, 4, 16, {}};
    for (int j = 0; j < 16; ++j)
        for (int k = 0; k < 4; ++k) t.f16.push_back(f32_to_f16(k == j % 4 ? 0.5f : 0.0f));
    return t;
}

int main(int argc, char** argv) {
    if (argc > 1) {   // the published super-adapter file
        Geometry g;
        g.n_layers = 64;
        g.n_embd = 5120;
        g.n_ff = 17408;
        g.n_head = 48;
        g.head_dim = 128;
        g.n_kv = 8;
        for (int i = 0; i < 64; ++i) g.kinds.push_back(i % 4 == 3 ? Mixer::Attention : Mixer::Gdn);
        LoraAdapter la;
        std::string err;
        CHECK(la.load(argv[1], "qwen35", g, 1.0f, err), err.c_str());
        if (!err.empty()) std::printf("note: %s\n", err.c_str());
        CHECK(la.rank() == 96, "super-adapter rank is 96");
        CHECK(!la.alpha_defaulted() && la.alpha() == 192.0, "super-adapter alpha is 192");
        CHECK(std::fabs(la.scale() - 2.0f) < 1e-6f, "super-adapter scale is 2.0");
        int pairs = 0, attn = 0;
        for (int i = 0; i < 64; ++i)
            for (int t = 0; t < kLoraTargets; ++t)
                if (la.find(i, (LoraTarget) t)) {
                    ++pairs;
                    if (t <= (int) LoraTarget::AttnOutput) ++attn;
                }
        CHECK(pairs == 256, "256 resolved pairs");
        CHECK(attn == 64, "4 attention targets on each of 16 attention layers");
        CHECK(la.bytes() == 956301312ull, "payload bytes match the file's tensor bytes");
        std::printf(failures ? "real-file checks FAILED\n" : "real-file checks passed\n");
        return failures ? 1 : 0;
    }

    const std::string dir = std::string("/tmp/basalt_lora_test_") + std::to_string((long long) ::getpid());
    ::mkdir(dir.c_str(), 0755);
    const Geometry g = test_geometry();

    {   // 1. a good fixture loads; names, shapes, scale resolve
        Fixture fx;
        fx.tensors = {gate_a(), gate_b(), gate_a("blk.3.attn_q.weight.lora_a"), gate_b("blk.3.attn_q.weight.lora_b")};
        const std::string p = dir + "/good.gguf";
        CHECK(fx.write(p).empty(), "fixture written");
        LoraAdapter la;
        std::string err;
        CHECK(la.load(p, "qwen35", g, 1.0f, err), err.c_str());
        CHECK(la.rank() == 4 && la.alpha() == 16.0 && !la.alpha_defaulted(), "rank and alpha parsed");
        CHECK(std::fabs(la.scale() - 4.0f) < 1e-6f, "scale is alpha/rank");
        CHECK(la.find(1, LoraTarget::FfnGate) && la.find(1, LoraTarget::FfnGate)->in == 8 &&
                  la.find(1, LoraTarget::FfnGate)->out == 16,
              "ffn_gate resolved");
        CHECK(la.find(3, LoraTarget::AttnQ) && !la.find(1, LoraTarget::AttnQ), "attn_q on layer 3 only");
        CHECK(!la.find(0, LoraTarget::FfnGate), "layer 0 carries nothing");

        // the CPU reference: y[j] += 4 * 0.5 * x[j % 4] (the fixture's factors above)
        float x[8];
        float y[16] = {};
        for (int i = 0; i < 8; ++i) x[i] = (float) (i + 1);   // 1..8
        lora_apply_ref(*la.find(1, LoraTarget::FfnGate), la.rank(), la.scale(), x, 1, y);
        bool ok = true;
        for (int j = 0; j < 16; ++j) ok = ok && std::fabs(y[j] - 2.0f * (float) (j % 4 + 1)) < 1e-4f;
        CHECK(ok, "lora_apply_ref matches the hand-computed result");
        // a zero scale leaves y alone
        float y2[16] = {};
        lora_apply_ref(*la.find(1, LoraTarget::FfnGate), la.rank(), 0.0f, x, 1, y2);
        ok = true;
        for (float v : y2) ok = ok && v == 0.0f;
        CHECK(ok, "zero scale adds nothing");
    }
    {   // 2. --lora-scale multiplies
        Fixture fx;
        fx.tensors = {gate_a(), gate_b()};
        const std::string p = dir + "/scaled.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(la.load(p, "qwen35", g, 0.5f, err), err.c_str());
        CHECK(std::fabs(la.scale() - 2.0f) < 1e-6f, "user scale multiplies alpha/rank");
    }
    {   // 3. a missing alpha defaults the scale to 1.0 x user
        Fixture fx;
        fx.write_alpha = false;
        fx.tensors = {gate_a(), gate_b()};
        const std::string p = dir + "/noalpha.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(la.load(p, "qwen35", g, 3.0f, err), err.c_str());
        CHECK(la.alpha_defaulted() && std::fabs(la.scale() - 3.0f) < 1e-6f, "missing alpha: scale is the user scale");
    }
    {   // 4. architecture mismatch is refused
        Fixture fx;
        fx.arch = "qwen4exp";
        fx.tensors = {gate_a(), gate_b()};
        const std::string p = dir + "/arch.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(!la.load(p, "qwen35", g, 1.0f, err) && err.find("architecture") != std::string::npos,
              "wrong architecture refused");
    }
    {   // 5. not an adapter file is refused
        Fixture fx;
        fx.adapter_meta = false;
        fx.tensors = {gate_a(), gate_b()};
        const std::string p = dir + "/model.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(!la.load(p, "qwen35", g, 1.0f, err), "non-adapter GGUF refused");
    }
    {   // 6. an attention projection on a GDN layer is refused
        Fixture fx;
        fx.tensors = {gate_a("blk.1.attn_q.weight.lora_a"), gate_b("blk.1.attn_q.weight.lora_b")};
        const std::string p = dir + "/gdn.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(!la.load(p, "qwen35", g, 1.0f, err) && err.find("delta-net") != std::string::npos,
              "attention tensor on a GDN layer refused");
    }
    {   // 7. a shape that disagrees with the base projection is refused
        Fixture fx;
        FixtureTensor bad = gate_b();
        bad.name = "blk.1.ffn_gate.weight.lora_b";
        bad.ne1 = 8;   // n_ff is 16
        bad.f16.resize(4 * 8);
        fx.tensors = {gate_a(), bad};
        const std::string p = dir + "/shape.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(!la.load(p, "qwen35", g, 1.0f, err), "wrong shape refused");
    }
    {   // 8. an unpaired factor is refused
        Fixture fx;
        fx.tensors = {gate_a()};
        const std::string p = dir + "/unpaired.gguf";
        fx.write(p);
        LoraAdapter la;
        std::string err;
        CHECK(!la.load(p, "qwen35", g, 1.0f, err), "unpaired factor refused");
    }

    std::printf(failures ? "lora_test: FAILED (%d)\n" : "lora_test: all checks passed\n", failures);
    return failures ? 1 : 0;
}
