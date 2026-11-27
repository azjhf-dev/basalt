// src/models/ernie55/ref.hpp - CPU-reference forward math for one HCA layer and one MLA-DSA layer of an
// ERNIE 5.5 Flash Preview pack. This is the ground truth the CUDA kernels will be held to: plain fp32 loops,
// no runtime, no CUDA, reading weights straight out of a converted pack. The semantics mirror the unmerged
// llama.cpp draft PR ggml-org/llama.cpp#31925 (Flash Base; the Preview shares the architecture), cross-checked
// against the checkpoint's own tensor shapes:
//   - HCA (compress_ratio 128): low-rank q (q_a + rmsnorm + q_b), a per-head plain RMS norm, partial
//     interleaved YaRN rope at the compress base (offset head_dim - rot); a single roped KV vector per token;
//     a compressor that reduces each completed 128-token block to one entry (per-dim softmax over the block of
//     the gate scores + APE, weighted sum of the compressor kv, rmsnorm, rope at the block's last position);
//     attention over [raw window of 128 | completed compressed blocks] with per-head sinks; inverse rope; the
//     VHA head mix (folded I + V U^T, out[h] = sum_g M[h][g] in[g]); grouped low-rank output with a sigmoid
//     gate from the normalized q latent.
//   - MLA-DSA (compress_ratio 0): absorbed form - q_a + rmsnorm, gate from q_c, q_b split [nope | rope], kv_a
//     split [kv latent | rope] with rmsnorm on the latent, plain interleaved rope at rope_theta on the rope
//     parts, k_b absorbed into the query, attention over the [latent | rope] keys with the latent as value and
//     v_b applied per head after the weighted sum, sinks, VHA, gate, o_proj. The lightning indexer
//     (LayerNorm + NEOX rope keys, per-head relu scores weighted by a projection) is computed and returned so
//     tests can check it; on a sequence shorter than top_k + window its top-k selects everything and attention
//     is dense.
// Assumptions flagged for the eventual HF-parity run on real weights (no HF modeling code is published yet):
//   - the compressed entry of a block is roped at the block's LAST token position;
//   - kv_b rows are per-head [qk_nope | v] contiguous (the DeepSeek convention);
//   - rope pair order is interleaved (config rope_interleave=true; llama rope type NORM) with the indexer the
//     only NEOX (half-split) consumer.
// Pure std, no CUDA: the ref test compiles and runs without the runtime.
#pragma once

#include "models/ernie55/pack.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::ernie55::ref {

/// A dense fp32 matrix, [rows=out, cols=in] row-major (the checkpoint's layout).
struct Mat {
    int rows = 0, cols = 0;
    std::vector<float> w;

    const float* row(int r) const { return w.data() + (size_t) r * cols; }
};

using Vec = std::vector<float>;

// ------------------------------------------------------------ dequant (pure)

float fp16_to_f32(uint16_t h);
float bf16_to_f32(uint16_t h);

/// Dequantizes a pack tensor to a Mat (2D) or Vec (1D). Q8_0 rows must be multiples of 32 values, as the
/// converter guarantees. False with `err` on a type/shape the reader does not support.
bool read_mat(const std::string& path, const TensorRef& t, Mat& out, std::string& err);
bool read_vec(const std::string& path, const TensorRef& t, Vec& out, std::string& err);

// ------------------------------------------------------------ rope (pure)

struct RopeParams {
    int rot = 0;             ///< rotated dims at the tail of each head vector
    float base = 0.f;        ///< theta
    float freq_scale = 1.f;  ///< 1 / yarn factor (1 = no YaRN)
    int64_t n_ctx_orig = 0;  ///< YaRN original context (0 = no YaRN)
    float beta_fast = 0.f, beta_slow = 0.f;
};

/// The YaRN-ramped inverse frequency of pair i, per ggml's yarn with ext_factor 0 (attention factor 1).
float rope_inv_freq(const RopeParams& p, int i);

/// Interleaved (GPT-J style) rope on x[off, off+rot): pairs (2i, 2i+1), angle pos * inv_freq_i.
void rope_inplace(float* x, int off, int64_t pos, const RopeParams& p);
/// The exact inverse of rope_inplace with the same arguments.
void rope_back_inplace(float* x, int off, int64_t pos, const RopeParams& p);
/// NEOX (half-split) rope on x[0, rot): pairs (i, i + rot/2). The indexer only.
void rope_neox_inplace(float* x, int64_t pos, const RopeParams& p);

// ------------------------------------------------------------ layers

struct HcaWeights {
    Mat q_a, q_b, kv, o_a, o_b, gate;
    Vec q_a_norm, kv_norm, sinks, vha, comp_norm;
    Mat comp_kv, comp_gate, ape;   ///< ape is [ratio][head_dim]
};

struct MlaWeights {
    Mat q_a, q_b, kv_a, kv_b, o, gate, idx_q_b;
    Vec q_a_norm, kv_a_norm, sinks, vha;
    Mat idx_k, idx_proj;
    Vec idx_k_norm, idx_k_norm_b;
};

/// Loads and shape-checks one layer's weights from an opened pack. Every check compares the tensor table
/// against the geometry, so a mismatch between meta and shapes fails here, not in the math.
bool load_hca(const std::string& path, const Pack& p, int layer, HcaWeights& w, std::string& err);
bool load_mla(const std::string& path, const Pack& p, int layer, MlaWeights& w, std::string& err);

/// One HCA attention block over x[0..T) (post-attn_norm hidden states), returning out[0..T) (pre hc-post).
std::vector<Vec> hca_forward(const Geometry& g, const HcaWeights& w, const std::vector<Vec>& x);

struct MlaResult {
    std::vector<Vec> out;          ///< [T][n_embd]
    std::vector<Vec> idx_scores;   ///< [T][T]: the lightning indexer scores, -inf above the diagonal
};

/// One MLA-DSA attention block over x[0..T). Dense attention is correct while T <= top_k + window; the
/// indexer scores are returned for inspection regardless.
MlaResult mla_forward(const Geometry& g, const MlaWeights& w, const std::vector<Vec>& x);

}  // namespace basalt::models::ernie55::ref
