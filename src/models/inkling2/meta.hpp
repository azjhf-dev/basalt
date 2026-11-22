// src/models/inkling2/meta.hpp - Inkling 2 Mini (arch "inkling2_loop") pack metadata checks.
// Mirrors src/models/qwen4exp/meta.hpp. Free of the runtime so a CPU test can link it alone.
//
// STATUS: scaffolding, UNCOMPILED (no CUDA toolchain in the authoring sandbox). Key names follow
// llama.cpp upstream #32694 ("inkling2_loop.*"), which the ggml-org GGUFs carry; the converter
// (tools/convert/checkpoint_inkling2.py) writes the same keys into the .basalt pack.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/model/adapter.hpp"

#include <string>

namespace basalt::models::inkling2 {

// fixed geometry of Inkling 2 Mini (Thinking Machines, 2026-11-20 model card):
// 24 physical decoder layers, layers 6..17 looped x3 (48 effective), 128 experts top-8,
// hybrid attention: SWA window 1024 on most layers, full attention on the rest,
// relative position bias tables + learned sinks (no RoPE), short conv state per block.
struct Inkling2Geometry {
    static constexpr int n_layers      = 24;   // physical; effective = 24 + 2*(loop_end-loop_start)
    static constexpr int loop_start    = 6;    // first looped physical layer
    static constexpr int loop_end      = 17;   // last looped physical layer (inclusive)
    static constexpr int loop_passes   = 3;
    static constexpr int n_expert      = 128;
    static constexpr int n_expert_used = 8;
    static constexpr int swa_window    = 1024;
};

inline std::string validate_meta(const GgufFile& g) {
    struct Req { const char* key; uint64_t want; };
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "inkling2_loop") return "architecture is '" + arch->s + "', this adapter requires 'inkling2_loop'";
    const Req reqs[] = {
        {"inkling2_loop.block_count",           (uint64_t) Inkling2Geometry::n_layers},
        {"inkling2_loop.embedding_length",      0},
        {"inkling2_loop.expert_count",          (uint64_t) Inkling2Geometry::n_expert},
        {"inkling2_loop.expert_used_count",     (uint64_t) Inkling2Geometry::n_expert_used},
        {"inkling2_loop.attention.head_count",  0},
        {"inkling2_loop.attention.head_count_kv", 0},
        {"inkling2_loop.attention.sliding_window", (uint64_t) Inkling2Geometry::swa_window},
        {"inkling2_loop.loop.start",            (uint64_t) Inkling2Geometry::loop_start},
        {"inkling2_loop.loop.end",              (uint64_t) Inkling2Geometry::loop_end},
        {"inkling2_loop.loop.count",            (uint64_t) Inkling2Geometry::loop_passes},
        // loop.gate_rank and loop.varying are read by the loader, presence checked but any value accepted
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (r.want && v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    if (!g.get("inkling2_loop.loop.gate_rank")) return "missing inkling2_loop.loop.gate_rank";
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = "inkling2_loop";
    d.n_layers = Inkling2Geometry::n_layers + (Inkling2Geometry::loop_end - Inkling2Geometry::loop_start + 1) *
                                              (Inkling2Geometry::loop_passes - 1); // 48 effective
    d.n_expert = Inkling2Geometry::n_expert;
    d.n_expert_used = Inkling2Geometry::n_expert_used;
    if (const MetaValue* v = g.get("inkling2_loop.embedding_length")) d.n_embd = (int64_t) v->u;
    if (const MetaValue* v = g.get("inkling2_loop.expert_feed_forward_length")) d.n_ff_exp = (int64_t) v->u;
    return d;
}

}  // namespace basalt::models::inkling2
