#pragma once
// basalt/dense/lora.hpp - runtime LoRA adapters for the dense stack (docs/lora.md). Host side: parse and validate a
// llama.cpp-format adapter GGUF (general.type=adapter, adapter.type=lora), resolve its tensors against the model's
// geometry and keep them mapped for the device upload. The device products hook in iteration 3; lora_apply_ref is
// the CPU reference the contract test checks and the parity harness compares against.

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/dense/model.hpp"

#include <memory>
#include <string>
#include <vector>

namespace basalt::dense {

/// The base projections an adapter tensor may target.
enum class LoraTarget { AttnQ, AttnK, AttnV, AttnOutput, FfnGate, FfnUp, FfnDown };

/// One adapted projection: the two factors of one layer's base linear, in the GGUF's own storage
/// (A as [in][rank] with `in` fastest, B as [rank][out] with `rank` fastest), mapped in the file.
struct LoraPair {
    LoraTarget target;
    int layer = 0;
    int64_t in = 0, out = 0;     ///< the base projection's width (its input and output sizes)
    const void* a = nullptr;     ///< [in][rank]
    const void* b = nullptr;     ///< [rank][out]
    uint32_t type = 0;           ///< ggml type of both factors (F16 or F32)
};

class LoraAdapter {
public:
    LoraAdapter() = default;
    LoraAdapter(const LoraAdapter&) = delete;
    LoraAdapter& operator=(const LoraAdapter&) = delete;

    /// Parses the adapter GGUF at `path`, refuses anything that is not a LoRA adapter for `arch`, and resolves
    /// every tensor against `g`: a tensor whose base name is not a qwen35 dense linear, sits on a layer of the
    /// wrong kind, pairs with no partner, or disagrees with the base projection's shape fails the load with a
    /// reason.  `user_scale` multiplies the adapter's own alpha/rank scale (--lora-scale).
    bool load(const std::string& path, const std::string& arch, const Geometry& g, float user_scale, std::string& err);

    int rank() const { return rank_; }
    double alpha() const { return alpha_; }
    bool alpha_defaulted() const { return alpha_defaulted_; }
    /// The scale applied to every product: (alpha / rank) * user_scale.
    float scale() const { return scale_; }
    const std::vector<LoraPair>& pairs() const { return pairs_; }
    /// The pair for one layer's target, or nullptr.
    const LoraPair* find(int layer, LoraTarget t) const;
    const std::string& path() const { return path_; }
    /// The adapter's payload bytes (the factors only).
    uint64_t bytes() const { return bytes_; }

private:
    std::unique_ptr<GgufFile> gg_;
    std::string path_;
    std::vector<LoraPair> pairs_;              // [layer * kLoraTargets + target]
    int rank_ = 0;
    double alpha_ = 0.0;
    bool alpha_defaulted_ = false;
    float scale_ = 1.0f;
    uint64_t bytes_ = 0;
};

constexpr int kLoraTargets = 7;

/// The CPU reference application: y[r][out] += scale * (x[r][in] . A) . B for each of `rows` rows, fp32 in and
/// out, the factors decoded from their storage type.  This is the ordering the device products follow: the LoRA
/// reads the pre-quantization activations and adds into the product's fp32 output.
void lora_apply_ref(const LoraPair& p, int rank, float scale, const float* x, int rows, float* y);

}  // namespace basalt::dense
