// src/models/ernie55/kernels/hca_attention.hpp - the first ERNIE 5.5 compute kernel: the HCA attention
// core. For every (token, head) it computes softmax-with-sink over the sliding raw KV window plus the
// completed compressed blocks and the weighted sum of the same vectors, matching ref::hca_attention_ref
// exactly. Inputs are the prepared fp32 buffers from ref::hca_prepare (q/kv roped, comp normed + roped);
// output is per head, pre-inverse-rope. On CUDA builds hca_attention_forward runs on the GPU; without CUDA
// a sequential CPU fallback runs the identical math, which is how the parity test runs on machines without
// a toolkit. Correctness-first: one thread per (token, head), dots recomputed per pass, staging copies per
// call - tuning (block-per-head reductions, fused dequant, persistent caches) is a later iteration.
#pragma once

namespace basalt::models::ernie55::kernels {

/// q: [T*nh*hd] roped queries; kv: [T*hd] roped raw KV; comp: [(T/ratio)*hd] compressed entries;
/// sinks: [nh]; out: [T*nh*hd]. ratio is the HCA compress ratio (0 disables compressed blocks).
/// Returns false on a CUDA failure (the CPU fallback cannot fail).
bool hca_attention_forward(const float* q, const float* kv, const float* comp, const float* sinks, int T,
                           int nh, int hd, int window, int ratio, float* out);

/// True when this build runs hca_attention_forward on the GPU. The parity test prints the path it checked.
bool hca_attention_forward_is_cuda();

}  // namespace basalt::models::ernie55::kernels
