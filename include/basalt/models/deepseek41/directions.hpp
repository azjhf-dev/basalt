// include/basalt/models/deepseek41/directions.hpp - per-request runtime directions (refusal-vector ablation) at the
// attention output projection.
//
// One direction per block, applied on the wo_b output y of every token row as
//
//     y <- y - strength * r * dot(r, y)
//
// where r is the block's unit vector.  strength 0 is an exact bypass: no kernel, no arithmetic, baseline output.
// The idea and the reference vectors are pqhaz's (DeepSeek-V4.1-Flash-1017-heretic); the per-request runtime switch
// follows anvarazizov's TensorFold TP2 recipe (runtime-directions.patch), ported to Basalt's single-engine layout.
//
// The bundle file (BSLDIR01) is written by tools/deepseek41_directions.py:
//   magic 8 bytes "BSLDIR01"
//   u32 LE n_layers, u32 LE hidden, u64 LE layer mask (bit L: layer L carries a direction)
//   f32[n_layers][hidden] unit vectors (zero rows where the mask bit is clear)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41::directions {

inline constexpr char kMagic[8] = {'B', 'S', 'L', 'D', 'I', 'R', '0', '1'};
inline constexpr int kMaxLayers = 64;   // the u64 mask's width

struct Bundle {
    int64_t n_layers = 0;
    int64_t hidden = 0;
    uint64_t mask = 0;                 // the layers with a direction
    std::vector<float> vectors;        // n_layers * hidden, zero where the mask is clear
};

/// Validates `bytes` as a BSLDIR01 bundle for a model of `expect_layers` layers of width `expect_hidden`: sizes,
/// the mask, finite values and unit norms on the masked rows.  False with `err` on any mismatch.
bool parse(const void* data, size_t bytes, int64_t expect_layers, int64_t expect_hidden, Bundle& out, std::string& err);

/// Reads and parses a bundle file.  False with `err` when the file cannot be read or fails parse().
bool read(const std::string& path, int64_t expect_layers, int64_t expect_hidden, Bundle& out, std::string& err);

// ---- runtime (the CUDA half, kernels/directions.cu) -----------------------------------------------------------

/// Loads the bundle at `path` for a model of `n_layers` x `n_embd`.  Once per engine.  The vectors stay on the host
/// and are replicated to a device on its first apply.
bool open(const std::string& path, int64_t n_layers, int64_t n_embd, std::string& err);
bool loaded();
std::string summary();

/// The state the next prompts and decode windows run with: strength in [0,1] and the layers to ablate.
/// `mask` of ~0 selects every layer the bundle carries.  False with `err` when a bit names a layer the bundle lacks
/// or strength is outside [0,1].  strength 0 (or no directions loaded) is the exact baseline.
bool set(float strength, uint64_t mask, std::string& err);
bool active();

/// Applies the active direction on `out` ([T][hidden] f32, the layer's wo_b output) when a direction is loaded, the
/// strength is nonzero and the mask names layer `il`; otherwise an exact bypass.  Throws std::runtime_error when the
/// kernel cannot be launched (attn_out's own contract).
void apply(int il, float* out, int64_t T, void* stream);

}  // namespace basalt::models::deepseek41::directions
