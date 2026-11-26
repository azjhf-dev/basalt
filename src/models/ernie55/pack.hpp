// src/models/ernie55/pack.hpp - the tensor contract of an ERNIE 5.5 Flash Preview .basalt pack: which tensors
// exist, under which names and types, per layer class. The converter (tools/convert/checkpoint_ernie55.py)
// writes exactly this; the loader (next iteration) must require exactly this, and the contract test holds the
// generation logic together. Pure std, no runtime includes: the contract test compiles and runs without CUDA.
#pragma once

#include "models/ernie55/meta.hpp"

#include "basalt/core/model_file.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::ernie55 {

/// Pack section names (the converter's build()).
inline constexpr const char* kSectionGguf = "gguf";
inline constexpr const char* kSectionExperts = "experts";
inline constexpr const char* kSectionExpertsLayout = "experts.layout";
inline constexpr const char* kSectionDsparkRaw = "dspark/raw";
inline constexpr const char* kSectionDsparkManifest = "dspark/manifest.json";
inline constexpr const char* kSectionMeta = "meta";

/// ggml type ids as the converter writes them (TYPE_ID in checkpoint_ernie55.py).
inline constexpr int kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBF16 = 30, kTypeMXFP4 = 39;

struct TensorSpec {
    std::string name;      ///< pack tensor name (`blk.N.` expanded per layer)
    int type = 0;          ///< a kType* id
};

/// The tensors of one layer. `q8` lists Q8_0 suffixes, `f32` F32, `bf16` BF16; the caller prefixes `blk.N.`.
struct LayerTensors {
    std::vector<const char*> q8, f32, bf16;
};

/// Every layer: norms, the gated-attention gate, the folded VHA mix, both hyper-connection blocks.
inline LayerTensors layer_common() {
    return {
        .q8 = {"attn_gate.weight"},
        .f32 = {"attn_norm.weight", "ffn_norm.weight", "attn_vha.weight",
                "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight"},
        .bf16 = {},
    };
}

/// An HCA layer (compress_ratio 128): low-rank q, single-KV cache projection, grouped low-rank output,
/// sink bias, compressor.
inline LayerTensors layer_hca() {
    return {
        .q8 = {"attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
               "attn_output_a.weight", "attn_output_b.weight"},
        .f32 = {"attn_sinks.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                "attn_compressor_ape.weight", "attn_compressor_norm.weight"},
        .bf16 = {"attn_compressor_kv.weight", "attn_compressor_gate.weight"},
    };
}

/// An MLA-DSA layer (compress_ratio 0): absorbed latent path, whole kv_b (the loader splits it when the
/// kernel contract lands), lightning indexer.
inline LayerTensors layer_mla() {
    return {
        .q8 = {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_b.weight",
               "attn_output.weight", "indexer.attn_q_b.weight"},
        .f32 = {"attn_sinks.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                "indexer.k_norm.weight", "indexer.k_norm.bias"},
        .bf16 = {"indexer.attn_k.weight", "indexer.proj.weight"},
    };
}

/// The leading dense layer's FFN.
inline LayerTensors layer_dense_ffn() {
    return {.q8 = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}, .f32 = {}, .bf16 = {}};
}

/// A MoE layer's dense tensors; the routed experts themselves live in the experts section (MXFP4 rows,
/// `experts.layout` v6), not here.
inline LayerTensors layer_moe() {
    return {
        .q8 = {"ffn_latent_down.weight", "ffn_latent_up.weight",
               "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"},
        .f32 = {"ffn_gate_inp.weight", "exp_probs_b.bias", "exp_scale.weight", "ffn_latent_norm.weight"},
        .bf16 = {},
    };
}

/// Global (no layer) tensors.
inline std::vector<TensorSpec> global_tensors() {
    return {
        {"token_embd.weight", kTypeQ8_0},
        {"output.weight", kTypeQ8_0},
        {"output_norm.weight", kTypeF32},
        {"output_hc_base.weight", kTypeF32},
        {"output_hc_scale.weight", kTypeF32},
        {"output_hc_fn.weight", kTypeF32},
    };
}

/// The full tensor table of a pack for `g`: names and types, the loader's require-list. Order: globals, then
/// per layer (common, attention class, FFN class).
inline std::vector<TensorSpec> expected_tensors(const Geometry& g) {
    std::vector<TensorSpec> out = global_tensors();
    const auto add = [&](int layer, const LayerTensors& t) {
        const std::string pre = "blk." + std::to_string(layer) + ".";
        for (const char* n : t.q8) out.push_back({pre + n, kTypeQ8_0});
        for (const char* n : t.f32) out.push_back({pre + n, kTypeF32});
        for (const char* n : t.bf16) out.push_back({pre + n, kTypeBF16});
    };
    const LayerTensors common = layer_common(), hca = layer_hca(), mla = layer_mla(),
                       dense = layer_dense_ffn(), moe = layer_moe();
    for (int l = 0; l < g.n_layers; ++l) {
        add(l, common);
        add(l, is_hca(g, l) ? hca : mla);
        add(l, is_moe(g, l) ? moe : dense);
    }
    return out;
}

/// The sections a pack must carry beyond `meta` and `tokenizer/*` (the drafter is raw until its own stage).
inline std::vector<const char*> required_sections() {
    return {kSectionGguf, kSectionExperts, kSectionExpertsLayout, kSectionDsparkRaw, kSectionDsparkManifest};
}

/// Sanity between the geometry and the contract: counts that catch a misread config before any loading.
inline std::string check_geometry(const Geometry& g) {
    int hca = 0, mla = 0;
    for (int r : g.compress_ratio) (r != 0 ? hca : mla)++;
    if (hca == 0 || mla == 0) return "a Flash Preview pack mixes HCA and MLA-DSA layers; this one is not hybrid";
    if (g.dspark_targets.size() != 3)
        return "the Preview DSpark drafter has 3 stages; this pack has " + std::to_string(g.dspark_targets.size());
    if (g.dspark_block != 7) return "the Preview DSpark block is 7, not " + std::to_string(g.dspark_block);
    if (g.latent != g.n_ff_exp) return "latent MoE expects moe_latent_size == moe_intermediate_size";
    return {};
}

// ---------------------------------------------------------------- the loader's view (pack.cpp)
// Below the contract: what open_pack fills from a converted pack. The pure helpers are here so the pack test
// exercises them without CUDA.

/// Bytes of one ggml MXFP4 row of `n` values (17 per 32), as the converter's experts section stores them.
inline uint64_t mxfp4_row_bytes(uint64_t n) { return n / 32 * 17; }

/// Bytes of one routed expert's blob: gate | up | down MXFP4 rows plus the FP32 tail {1, 1, 1, 0}.
inline uint64_t expert_blob_bytes(const Geometry& g) {
    return 2 * (uint64_t) g.n_ff_exp * mxfp4_row_bytes((uint64_t) g.latent) +
           (uint64_t) g.latent * mxfp4_row_bytes((uint64_t) g.n_ff_exp) + 16;
}

/// Expected bytes of a contract tensor of `type` holding `values` values.
inline uint64_t tensor_bytes(int type, uint64_t values) {
    switch (type) {
    case kTypeF32: return 4 * values;
    case kTypeBF16: return 2 * values;
    case kTypeQ8_0: return values / 32 * 34;
    default: return 0;
    }
}

struct ExpertRow {
    int layer = -1;
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, blob_bytes = 0;   ///< offset inside the experts section
};

struct TensorRef {
    std::string name;
    int type = 0;
    std::vector<uint64_t> shape;   ///< pack order (innermost first, as the converter writes it)
    uint64_t offset = 0;           ///< absolute file offset
    uint64_t bytes = 0;
};

/// Parses the experts.layout v6 text ("layer gu_type d_type offset blob_bytes" per MoE layer). `rows` comes out
/// one per MoE layer, offsets ascending in the packed order. Pure: the pack test feeds it strings.
inline std::string parse_experts_layout(const std::string& text, const Geometry& g, std::vector<ExpertRow>& rows) {
    rows.clear();
    uint64_t expect_off = 0;
    const uint64_t blob = expert_blob_bytes(g);
    const int n_moe = g.n_layers - g.first_dense;
    for (size_t pos = 0; pos < text.size();) {
        const size_t nl = text.find('\n', pos);
        const std::string line = text.substr(pos, (nl == std::string::npos ? text.size() : nl) - pos);
        pos = nl == std::string::npos ? text.size() : nl + 1;
        if (line.empty() || line[0] == '#') continue;
        ExpertRow r;
        unsigned long long gu, d, off, by;
        if (std::sscanf(line.c_str(), "%d %llu %llu %llu %llu", &r.layer, &gu, &d, &off, &by) != 5)
            return "experts.layout: bad line '" + line + "'";
        r.gu_type = (int) gu;
        r.d_type = (int) d;
        r.offset = off;
        r.blob_bytes = by;
        if (r.layer < g.first_dense || r.layer >= g.n_layers)
            return "experts.layout: layer " + std::to_string(r.layer) + " is not a MoE layer";
        if (r.gu_type != kTypeMXFP4 || r.d_type != kTypeMXFP4)
            return "experts.layout: layer " + std::to_string(r.layer) + " is not MXFP4";
        if (r.blob_bytes != blob)
            return "experts.layout: layer " + std::to_string(r.layer) + " blob " + std::to_string(r.blob_bytes) +
                   " != " + std::to_string(blob);
        if (r.offset != expect_off)
            return "experts.layout: layer " + std::to_string(r.layer) + " offset " + std::to_string(r.offset) +
                   " breaks the packed order (expected " + std::to_string(expect_off) + ")";
        expect_off += r.blob_bytes * (uint64_t) g.n_expert;
        rows.push_back(r);
    }
    if ((int) rows.size() != n_moe)
        return "experts.layout lists " + std::to_string(rows.size()) + " MoE layers, the model has " +
               std::to_string(n_moe);
    return {};
}

struct Pack {
    core::ModelFile file;
    Geometry geo;
    std::map<std::string, TensorRef> tensors;
    std::vector<ExpertRow> experts;      ///< one per MoE layer, packed order
    core::FileRange experts_range;
    core::FileRange dspark_raw;          ///< the raw drafter bytes (folded by the drafter's own stage)
    core::Json dspark_manifest;          ///< name -> dtype, shape, offset, bytes

    const TensorRef* find(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
};

/// Opens a converted ERNIE 5.5 pack and enforces the whole contract: metadata geometry, exact tensor table,
/// sections, experts layout, drafter manifest. False with `err` on the first violation. No device work: this is
/// the structural loader the compute path builds on, and it refuses anything that is not the contract.
bool open_pack(const std::string& path, Pack& p, std::string& err);

}  // namespace basalt::models::ernie55
