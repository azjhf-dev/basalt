// src/models/ernie55/adapter.cpp - the ERNIE 5.5 Flash Preview adapter: registers the architecture, validates
// and describes a converted pack from its `ernie55.*` metadata and the tensor contract, and refuses to load.
// The compute path (HCA / MLA-DSA attention, latent MoE, mHC, MultiMax, DSpark) does not exist yet, so load
// fails closed instead of letting a converted pack into a session that would compute garbage.
#include "models/ernie55/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "models/ernie55/meta.hpp"
#include "models/ernie55/pack.hpp"

#include <string>

namespace basalt::models::ernie55 {
namespace {

Kv kv_of(const GgufFile& g) {
    Kv kv;
    kv.u = [&](const std::string& k, uint64_t& out) {
        const MetaValue* v = g.get(k);
        if (!v || !v->is_num()) return false;
        out = v->u;
        return true;
    };
    kv.f = [&](const std::string& k, double& out) {
        const MetaValue* v = g.get(k);
        if (!v || !v->is_num()) return false;
        out = v->num();
        return true;
    };
    kv.u_arr = [&](const std::string& k, std::vector<uint64_t>& out) {
        const MetaValue* v = g.get(k);
        if (!v) return false;
        for (const MetaValue& it : v->items) out.push_back(it.u);
        return true;
    };
    kv.f_arr = [&](const std::string& k, std::vector<double>& out) {
        const MetaValue* v = g.get(k);
        if (!v) return false;
        for (const MetaValue& it : v->items) out.push_back(it.num());
        return true;
    };
    return kv;
}

class Adapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "ernie55"; }

    std::string validate(const GgufFile& g) const override {
        const MetaValue* arch = g.get("general.architecture");
        if (!arch) return "missing general.architecture";
        if (arch->s != "ernie55") return "architecture is '" + arch->s + "', this adapter requires 'ernie55'";
        Geometry geo;
        if (std::string e = geometry_from(kv_of(g), geo); !e.empty()) return e;
        return check_geometry(geo);
    }

    model::ModelDesc describe(const GgufFile& g) const override {
        Geometry geo;
        geometry_from(kv_of(g), geo);
        model::ModelDesc d;
        d.arch = "ernie55";
        d.n_layers = geo.n_layers;
        d.n_embd = geo.n_embd;
        d.n_vocab = geo.n_vocab;
        d.n_expert = geo.n_expert;
        d.n_expert_used = geo.n_used;
        d.n_ff_exp = geo.n_ff_exp;
        d.n_expert_shared = geo.n_shared;
        d.context_length = geo.context_length;
        // the experts section is MXFP4 gate | up | down in every MoE layer
        d.expert_gate_up_type.assign(geo.n_layers - geo.first_dense, kTypeMXFP4);
        d.expert_down_type.assign(geo.n_layers - geo.first_dense, kTypeMXFP4);
        return d;
    }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        (void) req;
        (void) out;
        // Fail closed: a converted pack is a real artifact, but there is no compute path for it yet. Loading
        // it into a session would produce plausible-looking garbage, which is worse than a clear refusal.
        return {model::Status::NotImplemented,
                "ernie55: the pack converts, but the runtime (HCA / MLA-DSA attention, latent MoE, mHC, "
                "MultiMax, DSpark) is not implemented yet. Track docs/models/ernie55-flash-converter.md "
                "and docs/models/ernie55-runtime-contract.md."};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "ernie5"}; }

    // The chat/tool dialect and the C++ pre-tokenizer land with the tokenizer stage. Until then the model
    // cannot load, so these are never exercised.
    program::grammar::ToolForm tool_form() const override { return {}; }
    PreTokenizer pretokenizer() const override { return {"ernie5", nullptr}; }
    int max_expert_cards() const override { return 1; }
    const char* model_name() const override { return "ERNIE-5.5-Flash-Preview"; }
    model::TokenNames token_names() const override { return {}; }
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() { return std::make_unique<Adapter>(); }

}  // namespace basalt::models::ernie55
