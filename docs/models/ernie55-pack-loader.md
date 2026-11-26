# ERNIE 5.5 Flash Preview: pack loader (structural, still no decode)

Iteration 4, 2026-11-26. A converted pack can now be opened and validated by the runtime; it still cannot
generate a token.

## What landed

- `src/models/ernie55/pack.cpp` - `open_pack()`: opens a converted `.basalt` and enforces the whole contract
  before anything is trusted:
  - `meta.arch` is `ernie55`; geometry parses from `meta.gguf_kv` and passes the hybrid/Preview checks.
  - All required sections exist (`gguf`, `experts`, `experts.layout`, `dspark/raw`, `dspark/manifest.json`).
  - The tensor table is **exactly** the iteration-3 contract: no missing, no extra, matching ggml types,
    per-tensor bytes consistent with type and shape, and every tensor inside its section.
  - `experts.layout` v6 rows: one per MoE layer, MXFP4 gate/up and down, blob bytes derived from the geometry
    (2 gate/up + 1 down MXFP4 rows + FP32 tail), offsets in the packed order with no gaps, and the section's
    total size equal to layout x experts.
  - The DSpark manifest's entries are complete, in packed order, and total exactly the `dspark/raw` section.
- The adapter's `load()` now runs this loader: a malformed pack is an **error**; a valid pack is a clear
  **NotImplemented** refusal (the compute path does not exist). No silent partial state.
- `src/models/ernie55/pack_test.cpp` - pure layout/geometry checks plus an end-to-end open of a real converted
  pack (path as argv[1]), wired as a host-only `basalt_cpu_test`.

## Verification (all CPU, here)

- End-to-end: converted the synthetic 3-layer checkpoint with the actual converter CLI, then opened the pack
  with the C++ loader - **all checks pass** (geometry, 94-tensor table, 2 expert rows, drafter manifest).
- The end-to-end run caught one real bug before commit: the metadata array getters appended into a reused
  buffer, so `dspark.target_layers` arrived prefixed with `compress_ratios` (`128...`). Both Kv adapters
  (pack.cpp, adapter.cpp) now clear first. This is exactly the class of bug the loader test exists for.
- Pure layout tests: good layout parses; dense-layer row, non-MXFP4 row, wrong blob size, offset gap, missing
  layer and short line all refused.
- `ernie55_contract_test`, `adapter.cpp`, `pack.cpp` compile and pass with plain g++ (no CUDA). Converter
  suite: 121 passed. Full CMake build still unverified (no nvcc here).

Still not done: device upload of dense tensors, tier/cache sizing, the C++ pre-tokenizer, the serve dialect,
and every kernel. No generated tokens, no speed numbers.

## Next gate

CPU-reference math for one HCA layer and one MLA-DSA layer against a HF forward (small context), then the
first CUDA kernel with its parity test. The loader gives both the validated pack to read from.
