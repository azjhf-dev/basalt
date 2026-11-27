# ERNIE 5.5 Flash Preview: CPU math reference (one HCA layer, one MLA-DSA layer)

Iteration 5, 2026-11-27. The math for both attention flavors now exists as plain CPU C++, pinned against an
independent numpy oracle on a converted pack. Still no decode, no kernels, no GPU code.

## What landed

- `src/models/ernie55/ref.hpp` / `ref.cpp` - a deliberately slow, readable reference implementation that
  loads one HCA layer (layer 0) and one MLA-DSA layer (layer 7 by default, the first non-HCA layer) out of a
  converted `.basalt` pack and runs a short teacher-forced forward through each:
  - FP16 / BF16 / Q8_0 dequant, RMS norm, softmax with attention sinks, and all three rope styles the
    Preview uses (HCA: YaRN at `compress_rope_freq_base` 160000, factor 16; MLA-DSA q_pe/k_pe: plain
    interleaved at rope_theta 8M; indexer: NEOX half-split), with exact inverse rope where the llama.cpp
    reference applies it.
  - HCA: low-rank query (q_a, q_b, per-head RMS norm), grouped low-rank output (o_a per group, sigmoid gate
    from the low-rank query, o_b), sliding-window raw KV plus 128-token blocks compressed by the per-dim
    softmax-over-gate-scores mechanism with the APE bias, each compressed block roped at its last position,
    softmax denominator fattened by the per-head sink only (no extra logit term), VHA output mixing with the
    I + V U^T matrix folded at conversion.
  - MLA-DSA: q absorbed through the kv_b no-partial rows per head, K = [kv_c | k_pe] with rope applied to
    the absorbed query and k_pe after absorption, V = kv_c followed by the per-head v_b rows, then o.
    Indexer: per-head LayerNorm-ed queries, score = sum over heads of relu(q_h . k_j) . w_h with w scaled
    by 1/sqrt(idx_dim * idx_heads), causal.
  - Internally accumulates dot products in double so the reference stays tight against float64 oracle
    math (see below).
- `src/models/ernie55/ref_test.cpp` - pure unit checks (dequant round-trips, rope + inverse, softmax with
  sinks) plus an end-to-end forward on the tiny converted pack: output shapes, finiteness, strict causality
  for both layers and the indexer, and - when given the oracle dump as argv[2] - a cross-check against
  `tools/convert/ernie55_ref_oracle.py`, an independent numpy implementation that reads the same pack and
  writes inputs plus expected outputs.
- Geometry grew `mla_qk` / `mla_v` (from `attention.key_length_mla` / `value_length_mla`); the contract test
  asserts both (256/256 on the real Preview metadata).
- Wired into CMake: `ref.cpp` is part of `model_ernie55`, `ernie55_ref_test` is a host-only `basalt_cpu_test`.

## Verification (all CPU, here)

- Tiny fixture: rebuilt math-consistent (was internally inconsistent - head counts, rope dims, VHA/idx
  shapes now line up with the contract), converted with the actual converter CLI, forwarded through the C++
  reference. All checks pass: shapes, finiteness, causality, oracle cross-check.
- Oracle agreement: max relative difference over every output is 1.1e-4 (HCA), 1.2e-4 (MLA-DSA), 2.2e-5
  (indexer) - float32 storage against the oracle's float64 math, at tolerance 5e-4. A first float32-only
  pass measured 2.6e-3 on MLA; switching the reference's accumulators to double closed it to 1.2e-4, which
  is the point of having two independent implementations: precision and semantics separate cleanly.
- `ernie55_contract_test` and `ernie55_pack_test` still pass on the rebuilt pack. Converter suite: 121
  passed. Full CMake build still unverified (no nvcc here).

## Where the math comes from, and what is still assumed

There is no published HF modeling code for ERNIE 5.5 Flash Preview (the HF repo ships only tokenizer
files). The op-level semantics here mirror the unmerged llama.cpp draft PR
[ggml-org/llama.cpp#31925](https://github.com/ggml-org/llama.cpp/pull/31925) - which targets the Flash
**Base** architecture - cross-checked against the Base model card and the Preview's real shard shapes.

Two load-bearing assumptions stay flagged until a parity run against real weights on an actual box:

1. Compressed HCA blocks are roped at the block's **last** position (the draft's wording is ambiguous
   between last and first).
2. `kv_b`'s per-head rows are `[nope | v]` contiguous per head, in that order.

Both are isolated in one function each (`hca_forward`'s compressor block, `mla_forward`'s absorb step), so a
parity failure on real weights is a small, local fix.

Still not done: full-stack forward (embeddings, MoE, mHC, MultiMax, LM head), the C++ pre-tokenizer, the
serve dialect, tier/cache sizing, and every kernel. No generated tokens, no speed numbers.

## Next gate

First CUDA kernel with its parity test against this reference (the HCA attention path is the natural
start), then the C++ `pretok_ernie5` splitter and the serve dialect.
