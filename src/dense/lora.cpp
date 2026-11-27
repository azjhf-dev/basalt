// src/dense/lora.cpp - host side of the dense stack's LoRA adapters (basalt/dense/lora.hpp, docs/lora.md).
#include "basalt/dense/lora.hpp"

#include <cstring>

namespace basalt::dense {
namespace {

struct TargetName {
    const char* base;        // the projection's name in the pack (and the adapter GGUF)
    LoraTarget target;
    bool attention;          // only a Mixer::Attention layer may carry it
};

constexpr TargetName kNames[kLoraTargets] = {
    {"attn_q", LoraTarget::AttnQ, true},       {"attn_k", LoraTarget::AttnK, true},
    {"attn_v", LoraTarget::AttnV, true},       {"attn_output", LoraTarget::AttnOutput, true},
    {"ffn_gate", LoraTarget::FfnGate, false},  {"ffn_up", LoraTarget::FfnUp, false},
    {"ffn_down", LoraTarget::FfnDown, false},
};

constexpr uint32_t kF32 = 0, kF16 = 1;

/// The base projection's (in, out) for a target under `g`.
void base_shape(const Geometry& g, LoraTarget t, int64_t& in, int64_t& out) {
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    switch (t) {
    case LoraTarget::AttnQ: in = H, out = 2 * AQ; return;
    case LoraTarget::AttnK:
    case LoraTarget::AttnV: in = H, out = KV; return;
    case LoraTarget::AttnOutput: in = AQ, out = H; return;
    case LoraTarget::FfnGate:
    case LoraTarget::FfnUp: in = H, out = g.n_ff; return;
    case LoraTarget::FfnDown: in = g.n_ff, out = H; return;
    }
    in = out = 0;
}

uint64_t payload_bytes(const TensorInfo& t) {
    return t.elements() * (t.type == kF32 ? 4 : 2);
}

float read_f(const void* p, uint32_t type, uint64_t i) {
    if (type == kF32) return ((const float*) p)[i];
    const uint16_t h = ((const uint16_t*) p)[i];
    // f16 -> f32
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) return s ? -0.0f : 0.0f;
        float v = (float) m / 1024.0f * 0x1p-14f;
        return s ? -v : v;
    }
    if (e == 0x1f) {
        f = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        f = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    float v;
    std::memcpy(&v, &f, 4);
    return v;
}

}  // namespace

const LoraPair* LoraAdapter::find(int layer, LoraTarget t) const {
    const size_t i = (size_t) layer * kLoraTargets + (size_t) t;
    return i < pairs_.size() && pairs_[i].a ? &pairs_[i] : nullptr;
}

bool LoraAdapter::load(const std::string& path, const std::string& arch, const Geometry& g, float user_scale,
                       std::string& err) {
    try {
        gg_ = std::make_unique<GgufFile>(path);
    } catch (const std::exception& e) {
        err = "lora: cannot read " + path + ": " + e.what();
        return false;
    }
    path_ = path;
    const MetaValue* type = gg_->get("general.type");
    const MetaValue* atype = gg_->get("adapter.type");
    if (!type || type->s != "adapter" || !atype || atype->s != "lora") {
        err = "lora: " + path + " is not a LoRA adapter GGUF (general.type/adapter.type)";
        return false;
    }
    const MetaValue* a = gg_->get("general.architecture");
    if (!a || a->s != arch) {
        err = "lora: " + path + " is for architecture '" + (a ? a->s : std::string("?")) + "', this pack is '" + arch + "'";
        return false;
    }
    const MetaValue* al = gg_->get("adapter.lora.alpha");
    alpha_ = al && al->is_num() ? al->num() : 0.0;
    alpha_defaulted_ = alpha_ <= 0.0;

    // gather and pair: for each blk.N.<base>.weight.lora_a there must be its lora_b
    struct Half {
        const TensorInfo* t = nullptr;
        const TargetName* n = nullptr;
        int layer = 0;
    };
    std::vector<Half> as, bs;
    int64_t rank = -1;
    for (const TensorInfo& t : gg_->tensors()) {
        int layer = -1;
        char base[64];
        char ab = 0;
        if (std::sscanf(t.name.c_str(), "blk.%d.%63[^.].weight.lora_%c", &layer, base, &ab) != 3 || (ab != 'a' && ab != 'b')) {
            err = "lora: unexpected tensor '" + t.name + "' (want blk.N.<base>.weight.lora_a/b)";
            return false;
        }
        if (layer < 0 || layer >= g.n_layers) {
            err = "lora: tensor '" + t.name + "' is past the model's " + std::to_string(g.n_layers) + " layers";
            return false;
        }
        const TargetName* tn = nullptr;
        for (const TargetName& k : kNames)
            if (std::strcmp(base, k.base) == 0) tn = &k;
        if (!tn) {
            err = "lora: '" + t.name + "' targets '" + base + "', not a qwen35 dense linear this build adapts";
            return false;
        }
        if (t.shape.size() != 2 || (t.type != kF16 && t.type != kF32)) {
            err = "lora: '" + t.name + "' must be a 2-d F16 or F32 tensor, is " + std::string(t.type_name());
            return false;
        }
        const int64_t r = ab == 'a' ? (int64_t) t.shape[1] : (int64_t) t.shape[0];
        if (rank < 0) rank = r;
        if (r != rank) {
            err = "lora: '" + t.name + "' has rank " + std::to_string(r) + ", the adapter's other tensors " +
                  std::to_string(rank);
            return false;
        }
        (ab == 'a' ? as : bs).push_back({&t, tn, layer});
    }
    if (as.empty()) {
        err = "lora: " + path + " holds no adapter tensors";
        return false;
    }
    if (as.size() != bs.size()) {
        err = "lora: " + path + " has unpaired lora_a/lora_b tensors";
        return false;
    }
    rank_ = (int) rank;
    if (alpha_defaulted_) alpha_ = (double) rank_;   // no alpha in the file: scale 1.0 before --lora-scale
    scale_ = (float) (alpha_ / (double) rank_) * user_scale;

    pairs_.assign((size_t) g.n_layers * kLoraTargets, LoraPair{});
    for (const Half& h : as) {
        // find its b
        const Half* other = nullptr;
        for (const Half& c : bs)
            if (c.layer == h.layer && c.n == h.n) other = &c;
        if (!other) {
            err = "lora: '" + h.t->name + "' has no lora_b partner";
            return false;
        }
        if (h.n->attention && g.kinds[(size_t) h.layer] != Mixer::Attention) {
            err = "lora: '" + h.t->name + "' is an attention projection on a gated delta-net layer";
            return false;
        }
        int64_t in = 0, out = 0;
        base_shape(g, h.n->target, in, out);
        if ((int64_t) h.t->shape[0] != in || (int64_t) other->t->shape[1] != out) {
            err = "lora: '" + h.t->name + "' and partner do not match the base projection (" + std::to_string(in) +
                  " -> " + std::to_string(out) + ")";
            return false;
        }
        if (other->t->type != h.t->type) {
            err = "lora: '" + h.t->name + "' and partner differ in type";
            return false;
        }
        LoraPair p;
        p.target = h.n->target;
        p.layer = h.layer;
        p.in = in;
        p.out = out;
        p.a = gg_->tensor_data(*h.t);
        p.b = gg_->tensor_data(*other->t);
        p.type = h.t->type;
        bytes_ += payload_bytes(*h.t) + payload_bytes(*other->t);
        pairs_[(size_t) h.layer * kLoraTargets + (size_t) p.target] = p;
    }
    return true;
}

void lora_apply_ref(const LoraPair& p, int rank, float scale, const float* x, int rows, float* y) {
    std::vector<float> h((size_t) rank);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t) r * p.in;
        float* yr = y + (size_t) r * p.out;
        for (int k = 0; k < rank; ++k) {
            double s = 0;
            for (int64_t i = 0; i < p.in; ++i) s += (double) xr[i] * read_f(p.a, p.type, (uint64_t) i + (uint64_t) k * p.in);
            h[(size_t) k] = (float) s;
        }
        for (int64_t j = 0; j < p.out; ++j) {
            double s = 0;
            for (int k = 0; k < rank; ++k) s += (double) h[(size_t) k] * read_f(p.b, p.type, (uint64_t) k + (uint64_t) j * rank);
            yr[j] += scale * (float) s;
        }
    }
}

}  // namespace basalt::dense
