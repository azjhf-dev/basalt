// src/models/ernie55/pack.hpp - the tensor contract of an ERNIE 5.5 Flash Preview .basalt pack: which tensors
// exist, under which names and types, per layer class. The converter (tools/convert/checkpoint_ernie55.py)
// writes exactly this; the loader (next iteration) must require exactly this, and the contract test holds the
// generation logic together. Pure std, no runtime includes: the contract test compiles and runs without CUDA.
#pragma once

#include "models/ernie55/meta.hpp"

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

}  // namespace basalt::models::ernie55
