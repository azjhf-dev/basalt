// src/models/ernie55/meta.hpp - the ERNIE 5.5 Flash Preview pack's metadata as the runtime reads it: the
// hparams contract every converted pack must satisfy. Free of the runtime: the pure reader takes key getters,
// so a CPU test can check it without linking the model or opening a file. The GgufFile wrappers at the bottom
// are what the adapter uses.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::models::ernie55 {

/// Reads one metadata value. `u` reads an integer, `f` a float, `u_arr` / `f_arr` an array; false when absent.
struct Kv {
    std::function<bool(const std::string& key, uint64_t& out)> u;
    std::function<bool(const std::string& key, double& out)> f;
    std::function<bool(const std::string& key, std::vector<uint64_t>& out)> u_arr;
    std::function<bool(const std::string& key, std::vector<double>& out)> f_arr;
};

/// Everything the runtime needs to size itself, from the pack's `ernie55.*` metadata (the converter's
/// model_kv). Defaults are invalid on purpose: a key the pack lacks leaves 0 and validate_meta fails.
struct Geometry {
    int n_layers = 0, n_embd = 0, n_vocab = 0;
    int64_t context_length = 0;
    // attention, shared
    int n_head = 0, n_head_kv = 0, head_dim = 0, v_head_dim = 0, q_lora = 0, kv_lora = 0;
    int mla_qk = 0, mla_v = 0;   ///< MLA qk head dim (nope + rope) and v head dim
    int window = 0, o_groups = 0, o_lora = 0;
    float rms_eps = 0.f;
    std::vector<int> compress_ratio;          ///< per layer; 0 marks an MLA-DSA layer, 128 an HCA layer
    float comp_rope_base = 0.f;
    // the DSA indexer (MLA layers)
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;
    // rope
    int rope_dim = 0;
    float rope_base = 0.f, yarn_factor = 0.f, yarn_beta_fast = 0.f, yarn_beta_slow = 0.f;
    int64_t yarn_orig_ctx = 0;
    // hyper-connections
    int hc_mult = 0, hc_sinkhorn = 0;
    float hc_eps = 0.f;
    // MoE
    int first_dense = 0, n_expert = 0, n_used = 0, n_shared = 0, n_ff_exp = 0, latent = 0;
    bool expert_norm = false;
    float expert_scale = 0.f;
    // SiTU activation, MultiMax head
    float situ_beta = 0.f, situ_linear_beta = 0.f;
    std::vector<float> multimax;              ///< ranges then ts
    // DSpark drafter (the raw section; the drafter stage folds it)
    int dspark_block = 0, dspark_markov_rank = 0, dspark_noise_token = 0;
    std::vector<int> dspark_targets;
};

/// Fills `g` from `kv`. Returns the first problem, "" when every required key is present and sane. Pure: no
/// files, no runtime.
inline std::string geometry_from(const Kv& kv, Geometry& g) {
    g = Geometry{};
    const std::string A = "ernie55.";
    uint64_t u = 0;
    double d = 0;
    std::vector<uint64_t> ua;
    std::vector<double> fa;
    const auto get_u = [&](const char* k, int& out) {
        if (!kv.u(A + k, u) || u == 0) return std::string("missing or zero ernie55.") + k;
        out = (int) u;
        return std::string();
    };
    const auto get_i64 = [&](const char* k, int64_t& out) {
        if (!kv.u(A + k, u) || u == 0) return std::string("missing or zero ernie55.") + k;
        out = (int64_t) u;
        return std::string();
    };
    const auto get_f = [&](const char* k, float& out) {
        if (!kv.f(A + k, d) || d == 0.0) return std::string("missing or zero ernie55.") + k;
        out = (float) d;
        return std::string();
    };
    std::string e;
    if (!(e = get_u("block_count", g.n_layers)).empty()) return e;
    if (!(e = get_u("embedding_length", g.n_embd)).empty()) return e;
    if (!(e = get_u("vocab_size", g.n_vocab)).empty()) return e;
    if (!(e = get_i64("context_length", g.context_length)).empty()) return e;
    if (!(e = get_u("attention.head_count", g.n_head)).empty()) return e;
    if (!(e = get_u("attention.head_count_kv", g.n_head_kv)).empty()) return e;
    if (!(e = get_u("attention.key_length", g.head_dim)).empty()) return e;
    if (!(e = get_u("attention.value_length", g.v_head_dim)).empty()) return e;
    if (!(e = get_u("attention.q_lora_rank", g.q_lora)).empty()) return e;
    if (!(e = get_u("attention.kv_lora_rank", g.kv_lora)).empty()) return e;
    if (!(e = get_u("attention.key_length_mla", g.mla_qk)).empty()) return e;
    if (!(e = get_u("attention.value_length_mla", g.mla_v)).empty()) return e;
    if (!(e = get_u("attention.sliding_window", g.window)).empty()) return e;
    if (!(e = get_u("attention.output_group_count", g.o_groups)).empty()) return e;
    if (!(e = get_u("attention.output_lora_rank", g.o_lora)).empty()) return e;
    if (!(e = get_f("attention.layer_norm_rms_epsilon", g.rms_eps)).empty()) return e;
    if (!kv.u_arr(A + "attention.compress_ratios", ua)) return "missing ernie55.attention.compress_ratios";
    if ((int) ua.size() != g.n_layers) return "ernie55.attention.compress_ratios has " +
        std::to_string(ua.size()) + " entries for " + std::to_string(g.n_layers) + " layers";
    for (uint64_t r : ua) {
        if (r != 0 && r != 128) return "ernie55.attention.compress_ratios: unsupported ratio " + std::to_string(r)
            + " (the runtime contract knows 0 = MLA-DSA and 128 = HCA)";
        g.compress_ratio.push_back((int) r);
    }
    if (!(e = get_f("attention.compress_rope_freq_base", g.comp_rope_base)).empty()) return e;
    if (!(e = get_u("attention.indexer.head_count", g.idx_heads)).empty()) return e;
    if (!(e = get_u("attention.indexer.key_length", g.idx_dim)).empty()) return e;
    if (!(e = get_u("attention.indexer.top_k", g.idx_top_k)).empty()) return e;
    if (!(e = get_u("rope.dimension_count", g.rope_dim)).empty()) return e;
    if (!(e = get_f("rope.freq_base", g.rope_base)).empty()) return e;
    if (!(e = get_f("rope.scaling.factor", g.yarn_factor)).empty()) return e;
    if (!(e = get_f("rope.scaling.beta_fast", g.yarn_beta_fast)).empty()) return e;
    if (!(e = get_f("rope.scaling.beta_slow", g.yarn_beta_slow)).empty()) return e;
    if (!(e = get_i64("rope.scaling.original_context_length", g.yarn_orig_ctx)).empty()) return e;
    if (!(e = get_u("hyper_connection.count", g.hc_mult)).empty()) return e;
    if (!(e = get_u("hyper_connection.sinkhorn_iterations", g.hc_sinkhorn)).empty()) return e;
    if (!(e = get_f("hyper_connection.epsilon", g.hc_eps)).empty()) return e;
    if (!(e = get_u("expert_count", g.n_expert)).empty()) return e;
    if (!(e = get_u("expert_used_count", g.n_used)).empty()) return e;
    if (!(e = get_u("expert_shared_count", g.n_shared)).empty()) return e;
    if (!(e = get_u("expert_feed_forward_length", g.n_ff_exp)).empty()) return e;
    if (!(e = get_u("expert_latent_length", g.latent)).empty()) return e;
    if (!(e = get_f("expert_weights_scale", g.expert_scale)).empty()) return e;
    if (kv.u(A + "expert_weights_norm", u)) g.expert_norm = u != 0;
    if (!(e = get_u("leading_dense_block_count", g.first_dense)).empty()) return e;
    if (g.first_dense >= g.n_layers) return "ernie55.leading_dense_block_count covers every layer";
    if (!(e = get_f("activation.situ_beta", g.situ_beta)).empty()) return e;
    if (!(e = get_f("activation.situ_linear_beta", g.situ_linear_beta)).empty()) return e;
    if (!kv.f_arr(A + "logit_multimax", fa) || fa.size() != 8)
        return "missing ernie55.logit_multimax (8 floats: 4 ranges, 4 ts)";
    for (double v : fa) g.multimax.push_back((float) v);
    if (!(e = get_u("dspark.block_size", g.dspark_block)).empty()) return e;
    if (!(e = get_u("dspark.markov_rank", g.dspark_markov_rank)).empty()) return e;
    if (!(e = get_u("dspark.noise_token_id", g.dspark_noise_token)).empty()) return e;
    if (!kv.u_arr(A + "dspark.target_layers", ua) || ua.empty()) return "missing ernie55.dspark.target_layers";
    for (uint64_t t : ua) {
        if (t >= (uint64_t) g.n_layers) return "ernie55.dspark.target_layers names layer " + std::to_string(t)
            + " of " + std::to_string(g.n_layers);
        g.dspark_targets.push_back((int) t);
    }
    return {};
}

/// Layer classes the tensor contract switches on.
inline bool is_hca(const Geometry& g, int layer) { return g.compress_ratio[layer] != 0; }
inline bool is_moe(const Geometry& g, int layer) { return layer >= g.first_dense; }

}  // namespace basalt::models::ernie55
