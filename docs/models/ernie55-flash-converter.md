# ERNIE 5.5 Flash Preview: converter checkpoint

Iteration 2, 2026-11-24. Experimental conversion code, not a runnable model port.

## What landed

`tools/convert/checkpoint_ernie55.py` is selected by `model_type: ernie5_5_moe`:

```sh
python3 tools/convert/convert.py --checkpoint /path/to/ERNIE-5.5-Flash-Preview \
    --out ernie55-preview.basalt --workers 4
```

Use the default MXFP4-expert checkpoint, not the separate FP8 or BF16 repositories. All shards must be present.
No streaming download/deletion mode is implemented. Budget disk for both the checkpoint and output.
`--sections-from` and an external `--drafter` are rejected.

- Routed experts: packed E2M1 pairs plus UE8M0 block scales repacked to ggml MXFP4 blocks. No requantization.
- Dense FP8 projections: E4M3 values multiplied by their F32 128x128 block scales, then Q8_0.
- Embedding/output: BF16 to Q8_0. Norms, routing, hyper-connections and VHA: F32.
- VHA folds `I + V U^T`. The two routers are stacked. Hyper-connection mappings are transposed.
- MLA `kv_b` is kept whole. The later runtime loader must split/absorb it, or the converter must add that
  transform when the kernel contract is fixed. It is not yet the absorbed layout expected by an existing model.
- Preview MTP is preserved in `dspark/raw` and `dspark/manifest.json`, not yet executable drafter sections.
- ERNIE tokenizer metadata and pre-tokenizer sequence are exported. Python regex requires explicit grouping
  for the source Oniguruma digit expression: `{3}+` cannot be copied literally without changing behavior.

## Verification

- 9 new tests: synthetic three-layer checkpoint, actual CLI conversion and pack readback, MXFP4 byte layout,
  expert tail, FP8/Q8_0 values, metadata, coverage, VHA/router transforms, orphan-scale rejection and digit grouping.
- Full converter suite: **119 tests passed**.
- Existing tokenizer suite: **6 tests run, 1 skipped** because no real tokenizer directory was supplied.
- Source pre-tokenizer versus Python implementation: **2,106 cases passed**, including long digit runs,
  Unicode digits, mixed scripts, whitespace and seeded random text. Oracle used the original pre-tokenizer
  sequence through the `tokenizers` library, before ByteLevel byte mapping. This is not full-vocabulary parity.
- The downloaded checkpoint index has **140,457 names**, all classifiable by the current tensor map.
  That is name coverage, not full-checkpoint dtype/shape validation. Sampled shard headers informed the map.

Not tested: complete weight conversion, a GPU build, loading this architecture in Basalt, generated tokens,
full-tokenizer parity, numerical forward parity or performance. No GPU or CUDA compiler is available here.
The model adapter and CUDA kernels do not exist yet. Existing Basalt builds should reject this architecture.

## Corrections to the first plan

- Preview has three MTP stages plus auxiliary heads. The initial plan's single-layer description came from
  the Base-model reference and does not describe Preview's full drafter.
- The initial SSD traffic estimate omitted the 42 MoE layers. Ten selected experts with three 2048x2048
  matrices each at 17 bytes per 32 weights are about 66.8 MB **per MoE layer**, or **2.81 GB per token**
  across 42 layers if every selected expert is cold. Shared experts, activations and metadata are additional.
  Partial RAM/VRAM residency reduces this, but SSD-heavy decode must not be promised to be practical.
- The original KV estimate was a rough placeholder, not an allocation model. HCA has a fixed window plus
  compressed history, rather than one full HCA row per historical token. Runtime cache sizing is still open.
- Converter code lives in `tools/convert/checkpoint_ernie55.py`, not the originally proposed `arch/` path.

## Next gate

Define and validate the runtime hparams/tensor contract under `src/models/ernie55/`, with a fail-closed
unsupported-compute path. Then implement CPU-reference math tests and GPU kernels. Do not use successful
pack writing as evidence that inference works.

Sources used: [checkpoint](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) and
[llama.cpp draft PR #31925](https://github.com/ggml-org/llama.cpp/pull/31925). The draft targets Base and
is a reference, not a verified Preview implementation.
