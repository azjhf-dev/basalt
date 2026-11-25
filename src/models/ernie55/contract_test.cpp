// src/models/ernie55/contract_test.cpp - the ERNIE 5.5 hparams and tensor contract, checked on the CPU with no
// CUDA and no pack file: geometry parsing from the converter's metadata keys, the per-layer-class tensor table,
// and the fail-closed guards. Run: ernie55_contract_test (exit 1 on the first failure, printing what).
#include "models/ernie55/meta.hpp"
#include "models/ernie55/pack.hpp"

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace basalt::models::ernie55;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/// The real ERNIE-5.5-Flash-Preview metadata, as the converter's model_kv writes it.
Kv preview_kv(const std::map<std::string, uint64_t>& ui, const std::map<std::string, double>& fl,
              const std::map<std::string, std::vector<uint64_t>>& uarr,
              const std::map<std::string, std::vector<double>>& farr) {
    Kv kv;
    kv.u = [&](const std::string& k, uint64_t& out) {
        auto it = ui.find(k);
        if (it == ui.end()) return false;
        out = it->second;
        return true;
    };
    kv.f = [&](const std::string& k, double& out) {
        auto it = fl.find(k);
        if (it == fl.end()) return false;
        out = it->second;
        return true;
    };
    kv.u_arr = [&](const std::string& k, std::vector<uint64_t>& out) {
        auto it = uarr.find(k);
        if (it == uarr.end()) return false;
        out = it->second;
        return true;
    };
    kv.f_arr = [&](const std::string& k, std::vector<double>& out) {
        auto it = farr.find(k);
        if (it == farr.end()) return false;
        out = it->second;
        return true;
    };
    return kv;
}

struct PreviewMaps {
    std::map<std::string, uint64_t> ui;
    std::map<std::string, double> fl;
    std::map<std::string, std::vector<uint64_t>> uarr;
    std::map<std::string, std::vector<double>> farr;
};

// baidu/ERNIE-5.5-Flash-Preview config.json, through the converter's model_kv (2026-11-24).
PreviewMaps preview() {
    PreviewMaps m;
    auto& [ui, fl, uarr, farr] = m;
    const std::string A = "ernie55.";
    ui[A + "block_count"] = 43;
    ui[A + "context_length"] = 1048576;
    ui[A + "embedding_length"] = 4096;
    ui[A + "vocab_size"] = 201216;
    ui[A + "attention.head_count"] = 64;
    ui[A + "attention.head_count_kv"] = 1;
    ui[A + "attention.key_length"] = 512;
    ui[A + "attention.value_length"] = 512;
    ui[A + "attention.q_lora_rank"] = 1024;
    ui[A + "attention.kv_lora_rank"] = 512;
    ui[A + "attention.sliding_window"] = 128;
    ui[A + "attention.output_group_count"] = 8;
    ui[A + "attention.output_lora_rank"] = 1024;
    fl[A + "attention.layer_norm_rms_epsilon"] = 1e-6;
    // 37 HCA (128) + 6 MLA-DSA (0 after the converter's max(r, 0)); MLA layers 7, 14, 21, 28, 35, 42.
    std::vector<uint64_t> ratios(43, 128);
    for (int l : {7, 14, 21, 28, 35, 42}) ratios[l] = 0;
    uarr[A + "attention.compress_ratios"] = ratios;
    fl[A + "attention.compress_rope_freq_base"] = 160000.0;
    ui[A + "attention.indexer.head_count"] = 64;
    ui[A + "attention.indexer.key_length"] = 128;
    ui[A + "attention.indexer.top_k"] = 2048;
    ui[A + "rope.dimension_count"] = 64;
    fl[A + "rope.freq_base"] = 8000000.0;
    fl[A + "rope.scaling.factor"] = 16.0;
    fl[A + "rope.scaling.beta_fast"] = 32.0;
    fl[A + "rope.scaling.beta_slow"] = 1.0;
    ui[A + "rope.scaling.original_context_length"] = 65536;
    ui[A + "hyper_connection.count"] = 4;
    ui[A + "hyper_connection.sinkhorn_iterations"] = 20;
    fl[A + "hyper_connection.epsilon"] = 1e-6;
    ui[A + "expert_count"] = 512;
    ui[A + "expert_used_count"] = 10;
    ui[A + "expert_shared_count"] = 1;
    ui[A + "expert_feed_forward_length"] = 2048;
    ui[A + "expert_latent_length"] = 2048;
    ui[A + "expert_weights_norm"] = 1;
    fl[A + "expert_weights_scale"] = 3.0;
    ui[A + "leading_dense_block_count"] = 1;
    fl[A + "activation.situ_beta"] = 4.0;
    fl[A + "activation.situ_linear_beta"] = 25.0;
    farr[A + "logit_multimax"] = {0, 0.5, 0.9, 1.0, 1, 1, 1, 1};
    ui[A + "dspark.block_size"] = 7;
    ui[A + "dspark.markov_rank"] = 256;
    ui[A + "dspark.noise_token_id"] = 1024;
    uarr[A + "dspark.target_layers"] = {40, 41, 42};
    return m;
}

}  // namespace

int main() {
    {
        PreviewMaps m = preview();
        Geometry g;
        const std::string e = geometry_from(preview_kv(m.ui, m.fl, m.uarr, m.farr), g);
        check(e.empty(), "preview geometry parses: " + e);
        check(g.n_layers == 43 && g.n_embd == 4096 && g.n_vocab == 201216, "preview headline dims");
        check(g.n_expert == 512 && g.n_used == 10 && g.n_shared == 1 && g.latent == 2048, "preview MoE dims");
        check(g.dspark_block == 7 && g.dspark_markov_rank == 256 && g.dspark_noise_token == 1024,
              "preview dspark dims");
        check(g.dspark_targets == std::vector<int>({40, 41, 42}), "preview dspark targets");
        check(check_geometry(g).empty(), "preview check_geometry");
        int hca = 0, mla = 0;
        for (int r : g.compress_ratio) (r != 0 ? hca : mla)++;
        check(hca == 37 && mla == 6, "preview is 37 HCA + 6 MLA layers");
        check(is_hca(g, 0) && !is_hca(g, 7) && is_moe(g, 1) && !is_moe(g, 0), "layer classes");

        // the tensor contract: names unique, counts exact, the right class on the right layer
        const std::vector<TensorSpec> t = expected_tensors(g);
        std::set<std::string> names;
        for (const TensorSpec& s : t) check(names.insert(s.name).second, "duplicate tensor " + s.name);
        // 6 globals + 43 common (10) + 37 HCA (12) + 6 MLA (13) + 42 MoE (9) + 1 dense FFN (3)
        check((int) t.size() == 6 + 430 + 444 + 78 + 378 + 3, "tensor count " + std::to_string(t.size()));
        const auto has = [&](const char* n) { return names.count(n) != 0; };
        check(has("blk.0.ffn_gate.weight") && !has("blk.0.ffn_gate_inp.weight"), "layer 0 is the dense FFN");
        check(has("blk.1.ffn_gate_inp.weight") && has("blk.1.attn_kv.weight") && !has("blk.1.attn_kv_b.weight"),
              "layer 1 is HCA MoE");
        check(has("blk.7.attn_kv_a_mqa.weight") && has("blk.7.indexer.attn_q_b.weight") &&
                  !has("blk.7.attn_compressor_kv.weight"),
              "layer 7 is MLA-DSA");
        check(has("token_embd.weight") && has("output_hc_fn.weight") && has("output.weight"), "globals");
        check(has("blk.42.attn_kv_b.weight"), "last layer is MLA-DSA");
    }
    {
        // fail closed on a broken pack
        PreviewMaps m = preview();
        Geometry g;
        m.ui.erase("ernie55.expert_count");
        check(!geometry_from(preview_kv(m.ui, m.fl, m.uarr, m.farr), g).empty(), "missing expert_count fails");
        m = preview();
        m.uarr["ernie55.attention.compress_ratios"].resize(42);
        check(!geometry_from(preview_kv(m.ui, m.fl, m.uarr, m.farr), g).empty(), "short compress_ratios fails");
        m = preview();
        m.uarr["ernie55.attention.compress_ratios"][3] = 64;
        check(!geometry_from(preview_kv(m.ui, m.fl, m.uarr, m.farr), g).empty(), "unknown ratio fails");
        m = preview();
        m.uarr["ernie55.attention.compress_ratios"].assign(43, 128);
        check(geometry_from(preview_kv(m.ui, m.fl, m.uarr, m.farr), g).empty(), "all-HCA parses");
        check(!check_geometry(g).empty(), "all-HCA is not a Flash Preview hybrid");
        m = preview();
        m.uarr["ernie55.dspark.target_layers"] = {41, 42};
        check(geometry_from(preview_kv(m.ui, m.fl, m.uarr, m.farr), g).empty(), "2-stage dspark parses");
        check(!check_geometry(g).empty(), "2-stage dspark is not the Preview drafter");
    }
    if (failures == 0) std::printf("ernie55 contract: all checks passed\n");
    return failures ? 1 : 0;
}
