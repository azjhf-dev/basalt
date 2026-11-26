// src/models/ernie55/pack_test.cpp - the pack loader: pure layout/geometry checks, and (with a pack path as
// argv[1]) an end-to-end open of a converted pack on the CPU. No CUDA. Exit 1 on the first failure.
#include "models/ernie55/pack.hpp"

#include <cstdio>
#include <string>

using namespace basalt::models::ernie55;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

Geometry tiny_geometry() {
    Geometry g;
    g.n_layers = 3;
    g.first_dense = 1;
    g.n_expert = 4;
    g.n_ff_exp = 128;
    g.latent = 128;
    g.compress_ratio = {128, 128, 0};
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    {
        // the blob math: 2 gate/up rows + 1 down row of 128 values (4 blocks of 32, 68 bytes a row) + FP32 tail
        Geometry g = tiny_geometry();
        check(mxfp4_row_bytes(128) == 68, "mxfp4 row bytes");
        check(expert_blob_bytes(g) == 2 * 128 * 68 + 128 * 68 + 16, "expert blob bytes");
        check(tensor_bytes(kTypeQ8_0, 1024) == 1024 / 32 * 34, "q8_0 bytes");
        check(tensor_bytes(kTypeF32, 3) == 12 && tensor_bytes(kTypeBF16, 5) == 10, "float bytes");
    }
    {
        // the layout text: good, then every way it can lie
        Geometry g = tiny_geometry();
        const std::string head = "# experts v6: layer gu_type d_type offset blob_bytes (n_expert 4, total 104512)\n";
        const uint64_t blob = expert_blob_bytes(g);
        const uint64_t layer_bytes = blob * 4;
        const std::string good = head + "1 39 39 0 " + std::to_string(blob) + "\n2 39 39 " +
                                 std::to_string(layer_bytes) + " " + std::to_string(blob) + "\n";
        std::vector<ExpertRow> rows;
        check(parse_experts_layout(good, g, rows).empty() && rows.size() == 2 && rows[0].layer == 1 &&
                  rows[1].offset == layer_bytes,
              "good layout parses");
        check(!parse_experts_layout("# v6\n0 39 39 0 " + std::to_string(blob) + "\n", g, rows).empty(),
              "the dense layer is not a MoE row");
        check(!parse_experts_layout("1 8 39 0 " + std::to_string(blob) + "\n2 39 39 " +
                                    std::to_string(layer_bytes) + " " + std::to_string(blob) + "\n", g, rows).empty(),
              "a non-MXFP4 row is refused");
        check(!parse_experts_layout("1 39 39 0 42\n2 39 39 " + std::to_string(layer_bytes) + " " +
                                    std::to_string(blob) + "\n", g, rows).empty(),
              "a wrong blob size is refused");
        check(!parse_experts_layout("1 39 39 7 " + std::to_string(blob) + "\n2 39 39 " +
                                    std::to_string(layer_bytes) + " " + std::to_string(blob) + "\n", g, rows).empty(),
              "a gap in the packed order is refused");
        check(!parse_experts_layout("2 39 39 0 " + std::to_string(blob) + "\n", g, rows).empty(),
              "a missing MoE layer is refused");
        check(!parse_experts_layout("1 39 39 0\n", g, rows).empty(), "a short line is refused");
    }
    if (argc > 1) {
        Pack p;
        std::string err;
        if (!open_pack(argv[1], p, err)) {
            check(false, "open_pack: " + err);
        } else {
            check(p.geo.n_layers == 3 && p.geo.n_embd == 256 && p.geo.n_vocab == 512, "pack geometry dims");
            check(p.geo.n_expert == 4 && p.geo.latent == 128 && p.geo.n_ff_exp == 128, "pack MoE dims");
            check(p.geo.compress_ratio == std::vector<int>({128, 128, 0}), "pack ratios");
            check(p.tensors.size() == expected_tensors(p.geo).size(), "tensor table size");
            check(p.find("blk.2.attn_kv_a_mqa.weight") != nullptr &&
                      p.find("blk.2.attn_kv_a_mqa.weight")->type == kTypeQ8_0,
                  "MLA tensor present with its type");
            check(p.find("blk.0.ffn_gate.weight") != nullptr && p.find("blk.0.ffn_gate_inp.weight") == nullptr,
                  "dense layer 0");
            check(p.experts.size() == 2 && p.experts[0].layer == 1 && p.experts[1].layer == 2, "expert rows");
            check(p.dspark_manifest.members.count("model.mtp.markov_head.markov_w1.weight") == 1,
                  "drafter manifest entry");
            check(!p.dspark_raw.empty() && p.dspark_raw.size() > 0, "drafter raw range");
        }
    } else {
        std::printf("note: no pack path given; ran the pure checks only\n");
    }
    if (failures == 0) std::printf("ernie55 pack: all checks passed\n");
    return failures ? 1 : 0;
}
