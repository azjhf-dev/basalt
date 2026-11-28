# ERNIE 5.5 Flash Preview: first CUDA kernel (HCA attention core), parity-checked

Iteration 6, 2026-11-28. The first real kernel exists: the HCA attention core. It is held to the
iteration-5 CPU reference at 4e-7 max relative difference. Still no decode; one kernel is not a model.

## What landed

- `src/models/ernie55/kernels/hca_attention.hpp` / `.cu` - `hca_attention_forward()`: for every (token,
  head), softmax-with-sink over the sliding raw KV window plus the completed compressed blocks, then the
  weighted sum of the same vectors. Inputs are the prepared fp32 buffers (q/kv roped, compressed entries
  normed + roped); output is per head, pre-inverse-rope.
  - The head math lives in one `BASALT_HD` function compiled by both nvcc and the host compiler, so the
    CUDA kernel and the CPU fallback differ only in how (token, head) pairs are iterated - the fallback is
    not a second implementation that can drift.
  - CUDA path: one thread per (token, head), staging copies per call. Correctness-first, deliberately not
    tuned: dots are recomputed across the two softmax passes, nothing is cached on device, no warp
    reductions. The optimization list (block-per-head with shared-memory reduction, online single-pass
    softmax, fused dequant, persistent KV caches) is future work once more of the model exists.
- `ref.cpp` refactor, no math change: `hca_forward` split into `hca_prepare` (everything before attention,
  flattened the way the kernel consumes it) and `hca_attention_ref` (one token-head of the core), so the
  parity test feeds kernel and reference the exact same buffers. The oracle cross-check reproduces the
  pre-refactor numbers bit-for-bit (1.1e-4 / 1.2e-4 / 2.2e-5 against the float64 numpy oracle).
- `src/models/ernie55/kernel_test.cpp` - loads a converted pack, prepares inputs with `hca_prepare`, runs
  both paths over all 200 tokens x 2 heads and compares elementwise. Prints `path: cuda` or
  `path: cpu fallback` so a GPU box confirms it exercised the real kernel.
- CMake: `hca_attention.cu` is part of `model_ernie55`; `ernie55_kernel_test` is wired as a
  `basalt_cpu_test` (on a toolkit build it runs the GPU path).

## Verification

- **Here (no CUDA toolkit, no GPU):** the kernel file compiled as plain C++ (`g++ -x c++`) takes the CPU
  fallback path - the same device function the CUDA kernel runs. Against `hca_attention_ref` on the tiny
  converted pack: max relative diff **4.11e-7** (tolerance 1e-5; the gap is fp32 rounding of the
  normalize-at-the-end vs normalize-per-term order). `ernie55_ref_test` (with oracle), `contract_test`,
  `pack_test` all still pass; converter suite 121 passed.
- **On a GPU box:** `cmake --build` compiles the same file with nvcc and `ernie55_kernel_test PACK` takes
  the CUDA path (`path: cuda`) against the same reference and tolerance. **Not yet run anywhere** - the
  first GPU execution happens on Ash's box.

Honest status: one kernel, parity-checked on CPU. The GPU path is compiled logic that has never executed;
kernel-launch bugs (indexing, copies) can only show up on a real card. No end-to-end forward, no MoE, no
tokenizer, no serve path, no tokens generated, no speed numbers.

## Next gate

The MLA-DSA attention kernel with the same treatment (absorbed-q core + causal indexer), then the C++
`pretok_ernie5` splitter so a prompt can actually reach the layers.
