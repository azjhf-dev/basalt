# ERNIE 5.5 Flash Preview: runtime contract and fail-closed skeleton

Iteration 3, 2026-11-25. The model now exists as an architecture in the build, but deliberately refuses to run.

## What landed

- `src/models/ernie55/meta.hpp` - the hparams contract: the `ernie55.*` pack metadata (as the converter writes
  it) parsed into a `Geometry`, with a pure getter-based reader so the parser is testable without files or CUDA.
  Unknown compress ratios, wrong ratio counts and out-of-range drafter targets fail closed. MLA layers are
  7, 14, 21, 28, 35, 42 (every 7th); the other 37 are HCA.
- `src/models/ernie55/pack.hpp` - the tensor contract: per layer class (HCA / MLA-DSA, dense / MoE) the exact
  tensor names and types the converter writes and the loader will require. MLA `kv_b` stays whole; the loader
  splits it once the kernel contract is fixed. The routed experts stay in the MXFP4 experts section.
- `src/models/ernie55/adapter.cpp` - registers `ernie55`, validates and describes a pack, then **fails closed
  at load** (`NotImplemented`): HCA / MLA-DSA attention, latent MoE, mHC, MultiMax and DSpark compute do not
  exist yet, and a loaded-but-wrong model is worse than a refusal.
- Build: `ernie55` in `BASALT_BUILTIN_MODELS`; `model_ernie55` and the `ernie55_contract_test` CPU test are
  gated under `if(BASALT_MODEL_ERNIE55)`, so builds without it compile none of it. No CUDA sources.
- `tools/convert/test_checkpoint_ernie55.py::test_contract_matches_cpp_pack_hpp` - the C++ contract's name
  table, re-derived by the same layer-class rules in Python, must equal the converter's tensor table. This
  holds the two languages' contracts together.

## Verification

- `ernie55_contract_test` compiled with plain g++ and **run here: all checks pass** - Preview geometry parses
  (43 layers, 37 HCA + 6 MLA, 512/10/1 experts, DSpark block 7 rank 256 targets 40-42), the tensor table has
  no duplicates and the exact expected count (6 + 430 + 444 + 78 + 378 + 3 = 1339 for the real config), class
  placement is right, and every malformed-metadata case fails closed.
- The test caught one real bug before commit: `geometry_from` appended into a reused `Geometry`; it now resets.
- `adapter.cpp` passes `g++ -fsyntax-only` with the repo's include paths. A full CMake build needs CUDA, which
  this sandbox does not have - the library and test wiring is unverified beyond that.
- Converter suite: **121 tests pass** (the 120 from iteration 2 plus the contract reconciliation test).
  Existing tokenizer suite unaffected.

Not done: the C++ pre-tokenizer (`pretok_ernie5`, mirroring `pretok_joyai.cpp` with the 510-digit rule - note
the Python oracle found Oniguruma `{3}+` needs explicit grouping), the serve dialect / tool form, the loader
that enforces this contract, and every kernel. The adapter's `pretokenizer()` and `tool_form()` return empty
stubs and are unreachable until load exists.

## Next gate

The loader: read a converted pack, require exactly this tensor table, upload dense tensors, and size the
tier/cache geometry - still refusing to decode. Then CPU-reference attention math, then the first kernel.
No GPU, no generated tokens, no speed claims anywhere yet.
