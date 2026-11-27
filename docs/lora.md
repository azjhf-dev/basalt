# LoRA adapters on the dense stack (qwen35 / Qwen3.8-27B)

Fork-only feature branch: `lora-adapter` on azjhf-dev/basalt. Nothing here is proposed upstream.
Pinned base: jesdga95/basalt @ `d0f35a8f` (`git describe`: v0.5.7-157-gd0f35a8f, 2026-11-26).

Status: **iteration 3 — the adapter is applied: device upload (attach_lora), the two factor products as CUDA kernels, and injection into every decode-window, batch-window and prompt-chunk path of the dense stack, plus the parity harness scaffold. Compiles and runs need Ash's 5090 (no local CUDA toolchain); the parity run gates iteration 4.**

Upstream asks for this in jesdga95/basalt#418 (open since 2026-11-18, no maintainer reply):
`--lora adapter.gguf --lora-scale 0.8`, multiple adapters, swap without reloading the base.

## Target adapter: harness-hub's super-adapter

The concrete adapter this work is built and tested against:

- PEFT repo `harness-hub/Qwen3.8-27B-super-adapter` — `adapter_config.json`: PEFT LORA, r=96,
  lora_alpha=192 (scale alpha/r = 2.0), bias none, base `Qwen/Qwen3.8-27B`, targets
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Single F32 safetensors.
- GGUF repo `harness-hub/Qwen3.8-27B-super-adapter-GGUF` — `Qwen3.8-27B-super-adapter-lora-F16.gguf`,
  956,336,640 bytes, GGUF v3, `general.architecture=qwen35`, `general.type=adapter`,
  `adapter.type=lora`, `adapter.lora.alpha=192`, 512 F16 tensors named
  `blk.N.<proj>.weight.lora_a` / `.lora_b`. This is the llama.cpp `--lora` format and the file v1 loads
  directly, no conversion step.

Measured tensor anatomy (both published artifacts agree):

| Basalt tensor | HF module | in -> out | layers |
| --- | --- | --- | --- |
| `attn_q` | `q_proj` (includes the output gate: Basalt's fused 2*AQ = 12,288) | 5,120 -> 12,288 | the 16 attention layers |
| `attn_k` | `k_proj` | 5,120 -> 1,024 | 16 attention layers |
| `attn_v` | `v_proj` | 5,120 -> 1,024 | 16 attention layers |
| `attn_output` | `o_proj` | 6,144 -> 5,120 | 16 attention layers |
| `ffn_gate` | `gate_proj` | 5,120 -> 17,408 | all 64 layers |
| `ffn_up` | `up_proj` | 5,120 -> 17,408 | all 64 layers |
| `ffn_down` | `down_proj` | 17,408 -> 5,120 | all 64 layers |

Not covered by the adapter: the GDN mixers (`attn_qkv`, `attn_gate`, `ssm_out`, ...), all norms,
the token embedding, the output head, and both drafters. Basalt's tensor names in the pack match the
GGUF adapter names one to one, so the loader is a name lookup, not a mapping table.

## Why runtime LoRA (and why the merge fallback is weak here)

For llama.cpp the fallback is easy: merge into BF16 with PEFT, reconvert to any GGUF quant. For
Basalt it is not: the pack moves NVIDIA's NVFP4 (MLP, head) and FP8 E4M3 (GDN/attention projections)
bit for bit, and the tree has no BF16 -> NVFP4 quantizer with calibration. A merge would mean
dequantizing the NVFP4 base to BF16, merging, and requantizing with calibration data — lossy,
heavy, and unverifiable against the published evaluation. Runtime LoRA leaves the pack untouched.

## Design

Semantics, matching PEFT and llama.cpp: for each targeted projection with base weight W,

```
y = x·Wᵀ + s · (x·A)·B        s = (lora_alpha / r) · user_scale = 2.0 · user_scale
```

with A/B the adapter's two factors (A: in×96, B: 96×out as stored in the GGUF). The adapter GGUF
ships F16; keep it bit for bit and run the two small products in BF16/FP32 accumulation, the same
rounding class as the stack's other FP32-accumulated products.

### Loader and CLI (iteration 2)

- `--lora path.gguf` loads a `general.type=adapter`, `adapter.type=lora` GGUF whose
  `general.architecture` must match the pack's (`qwen35`). Refuse anything else with a clear error.
- `--lora-scale S` (default 1.0) multiplies the alpha/r scale; kept in pinned device-visible memory
  so changing it never recaptures decode graphs.
- Loader walks the adapter's tensors, looks each name up in the pack's geometry
  (`blk.N.attn_q` -> layer N's `Linear q`, ...), skips nothing silently: a tensor whose base is not a
  qwen35 dense linear is a hard error, so a Flash-Next or GDN-targeting adapter cannot load half-way.
- Weights live in their own device allocation (~0.96 GB F16), separate from the pack's arena, so a
  later hot-swap reloads the adapter without touching the base.

### Injection points (upstream main, src/dense/forward.cpp)

All targeted products funnel through four helpers; the adapter math hooks there, not in the kernels:

- `prod_group` — the attention q/k/v products (line ~280)
- `prod_swiglu` — `ffn_gate` + `ffn_up` (line ~314)
- `prod_add` — `attn_output` and `ffn_down`, folding into the residual (lines ~245, ~315)
- the prompt-chunk products — `fp4_dense::product_bf16` / FP8 segment path (lines ~344-444)

The hook computes `h = x·A` into a scratch row set (decode window: ≤16×96 floats; prompt chunk:
4096×96 floats = 1.6 MB) from the *normed, pre-quantization* activations (`b.xn` / the chunk's `xn`),
then `y += s·h·B` on the product's FP32 output before `prod_add` folds it into the residual. This is
exactly llama.cpp's ordering: LoRA reads the float activation, not the quantized product input.

Because the single-window and batched-window decode paths share these helpers, covering the helpers
covers both. Decode windows are captured as CUDA graphs per row count; the LoRA products are captured
inside them, legal because adapter weights and the scale are device constants at capture time. A
scale change or adapter swap updates device memory only — no recapture.

### Where the delta joins (iteration 3)

`attach_lora` (src/dense/model.cpp) uploads the factors fp16 to the card, one `LoraDev` per layer and target
(Model::lora_w), plus scratch (one rank-wide buffer per path, one fp32 row set for the residual-adding targets).
The products are two small kernels (src/kernels/cuda/lora_ops.cu): `tmp = x·A` a warp per (row, k), `y += scale·tmp·B`
a thread per (row, n), fp32 accumulation, same ordering as `lora_apply_ref`. Injection points (src/dense/forward.cpp):

- q/k/v: into the product's fp32 rows right after prod_group / gemm, from the normed input (`lora_add`).
- gate/up: with an adapter the fused SwiGLU epilogue has no room for the delta, so the run takes the two-product
  path (scaled products, deltas, then SwiGLU) — window and chunk alike.
- attention out / FFN down (outputs that land in the BF16 residual): the delta is built in fp32 scratch and added
  with add_scaled — one more rounding than the fused epilogue's single add, the documented v1 difference.
  The attention merge runs ungated when the adapter targets o_proj (the delta needs the gated fp32 rows).
- prompt chunks: with an adapter the F8/NVFP4 fused chunk paths (BF16 intermediates, norm fused in the quantizer)
  give way to the FP32 path, whose rows the deltas join; prompt throughput drops, decode is untouched. Bringing
  the delta into the fused chunk kernels is an iteration-4+ perf item.
- drafters (MTP / block): no adapter tensors, run the base weights as before (target verifies; output stays exact).

The memory planner's figures (shared_bytes et al.) do not yet include the adapter (~0.9 GiB) and its scratch (~90
MiB); plan with that headroom until they do.

### Cost estimate

Per targeted projection the two factor products add `2·r·(in+out)` FLOPs against `in·out` for the
base, e.g. q_proj ≈ 5.3%, ffn_down ≈ 4.9%. Across a full layer: roughly 4-5% more decode math,
before the fixed cost of two small kernel launches per product (dominant at window size 1-16;
batched windows and drafted windows amortize it). Prompt chunks take the unfused FP32 path while an
adapter is on (see above), the bigger cost; both to be measured, not guessed, in iteration 4.

### Drafters, vision, multi-card

- MTP and the block drafter carry no adapter tensors and run the base weights: drafts are base-model
  proposals the adapted target still verifies, so output stays exact for the adapted model. Expect a
  small acceptance-rate drop (the drafter no longer predicts the target quite as well); measure.
- Vision: untouched (no adapter tensors in the tower or projector).
- Second-card stage (new since v0.5.7): v1 refuses `--lora` together with a layer split; the stage's
  card needs its own adapter copy, queued for a later iteration. Single-card dense serving is the
  documented qwen35 setup anyway.

### Out of scope for v1

Multiple simultaneous adapters, per-request adapter selection, and a serve-time
`POST /lora-adapters`-style API (all asked for in #418) — the loader and allocation layout are
chosen so these are additive later. Hot-swap of one adapter is the v2 candidate.

## Parity and tests

Reference implementation: llama.cpp at a pinned commit, serving the card's exact recipe —
`ggml-org/Qwen3.8-27B-GGUF` Q8_0 base + the F16 adapter GGUF, `-c 131072`, temp 0.

1. **Contract tests, CPU, no GPU needed (iteration 2).** Parse the adapter GGUF; check every tensor
   name resolves against the qwen35 geometry, every shape matches its base linear, and alpha parses.
   Cross-check values against the PEFT safetensors (both published): same math, two containers.
   Landed: `src/dense/lora_test.cpp` (8 fixture checks plus real-file mode) and
   `tools/lora/verify_adapter.py` (GGUF↔PEFT container cross-check; all 512 tensors match, max abs
   diff 7.1e-06 = F16 rounding).
2. **Delta parity on GPU (iteration 3's run, on Ash's 5090).** llama.cpp's own numbers on an NVFP4/FP8
   base do not exist, so absolute logits against it are dominated by the base quant difference.
   Instead compare *deltas*: for a fixed prompt set, the logit change from adding the adapter must
   match between engines — cosine similarity and KL of (logits_lora − logits_base) per position,
   plus top-1 agreement of the adapted models, plus exact-match greedy decodes (256 tokens) on a
   handful of code and prose prompts. The NVFP4/FP8 base will sit further from the evaluated BF16
   behavior than llama.cpp's Q8_0 base does (the card warns lower-bit quants drift); the delta
   comparison isolates our implementation from that known drift. Harness: `tools/lora/parity.py`
   (builds llama.cpp at the pinned commit, runs both engines base and adapted, compares greedy
   decodes and reported logits).
3. **Optional gold run:** PEFT merge on CPU (Ash's 128 GB box can hold BF16) for a couple of prompts
   as the outer anchor.

## Iterations

1. This document: feasibility, anatomy, injection points, plan. (Done; posted to #inference.)
2. GGUF adapter loader + `--lora` / `--lora-scale` + CPU contract tests. Compile-checked only.
3. Decode-window products (plain + batched windows) + parity harness scripts. Gate: first real run
   on Ash's RTX 5090.
4. Prompt-chunk products, drafter-interaction measurements, perf numbers vs the estimate above.
5. Hot-swap / scale-at-runtime, dual-card stage support, docs and limits. Revisit multi-adapter.

## References

- Adapter card: https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter
- GGUF artifact: https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter-GGUF
- Upstream request: https://github.com/jesdga95/basalt/issues/418
