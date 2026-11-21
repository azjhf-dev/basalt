# Inkling 2 Mini on Basalt (design draft, iteration 1)

Status: design + scaffolding only. **Nothing in this branch is compiled or run** - the dev
sandbox has no CUDA toolchain, so first compile happens on a Blackwell (sm_120) host.
Target: Thinking Machines Inkling 2 Mini (26.4B MoE, 5.4B active, looped transformer,
NVFP4-trained, 1M ctx, hybrid SWA-1024 + full attention, multimodal).

## Why Basalt can solve what llama.cpp day-one could not

The r/LocalLLaMA day-one problems (Lurksome-Lurker comment, 2026-11-21):

1. "llama.cpp does not support relative position bias with flash attention, so FA is
   ignored." In llama.cpp the bias is materialized as a per-head F16 *mask* (n_kv x
   n_tokens x n_head) added before FA, because the fattn kernels take no bias input -
   that is also the 17-38 GiB compute-buffer blowup on long-context prefill.
   Basalt's attention kernels are hand-written sm_120 CUDA: the decode/prefill attention
   kernel can take the relative-bias *table* (n_head x table_len, tiny) plus a per-cell
   distance index and add the gathered bias in-kernel. No materialized bias matrix, no
   blowup, FA stays on. This is the core kernel work of this port.
2. "KV cache has to be f16 or VRAM blows up." Basalt already has quantized KV as
   first-class kernels (`kv_q8.cu`, `kv_q4.cu` in qwen4exp; `"kv": "int8"` in config).
   The inkling2 attention kernel reads quantized KV directly, like the existing models.

## Architecture deltas vs existing Basalt models (what is genuinely new)

- Looped core: physical layers 6-17 are visited 3x (48 effective layers), with per-pass
  gate tensors and per-physical-layer loop slots (GGUF arch `inkling2_loop`, block_count
  24, re-shared loop slots - llama.cpp upstream #32694 + #32704). No Basalt model loops
  layers; the layer runner must iterate passes and source the per-pass loop tensors.
- Relative position bias tables + learned sinks (projected from the token embedding per
  layer) instead of RoPE. (qwen4exp has rope.cu; inkling2 has none.)
- Hybrid SWA(1024)/full attention across 24 physical layers.
- Short conv state (conv1d) on attention and FFN outputs, with rollback slots - closest
  precedent: qwen4exp `gdn.cu` recurrent state handling.
- 128 experts, top-8, sigmoid routing with expert weight scale; expert sizes are
  1152-wide (NVFP4-friendly). Precedent: qwen4exp `dec_moe*.cu`, `native_moe.cu`.
- NVFP4 expert weights + MXFP8 elsewhere: pack format support exists for NVFP4
  (Qwen3.8-27B-NVFP4 pack), dequant in `src/artifact/dequant.cpp`.

## Plan (week 1: scaffolding, compile-on-host)

1. `tools/convert/checkpoint_inkling2.py` - read the ggml-org GGUF (arch inkling2_loop)
   and HF safetensors, emit a `.basalt` pack: `tools/gguf_reader.py` + `model_file.py`
   patterns, following `checkpoint_qwen35.py` / `checkpoint_deepseek41.py`.
2. `src/models/inkling2/` skeleton per docs/adding-a-model.md:
   - `inkling2.cmake` (model_inkling2 + model_inkling2_kernels libs),
   - adapter (`make_adapter()`), meta.hpp with the hparams (incl. loop passes, gate
     ranks, swa pattern, bias table geometry),
   - decode/prefill plan structs, KV+conv running state (checkpoint_transfer).
   - registration: id `inkling2` in `cmake/models.cmake` BASALT_BUILTIN_MODELS,
     `if(BASALT_MODEL_INKLING2)` blocks in CMakeLists.txt.
3. Kernel list (parity test + microbench each, per adding-a-model):
   - `attn_rel_bias.cu` - decode FA with in-kernel bias gather + sinks (THE kernel);
   - `loop_gate.cu` - pass-gate mixing between loop iterations;
   - `conv_state.cu` - short conv with rollback slots (adapt gdn patterns);
   - reuse: kv_q8/kv_q4 patterns, MoE decode patterns, router.
4. `serve/models/inkling2.py` dialect (prompt render, stop texts) - check model card
   for the chat format.
5. Build script for the 5090 host: `docs/models/inkling2.md` with
   `-DBASALT_MODELS=inkling2` and a smoke run.

## Open questions (marked gaps)

- Exact loop-gate math and per-pass tensor shapes: take from upstream llama.cpp
  inkling2-loop.cpp (ported in azjhf-dev/Avalon branch inkling2-mini) and the model card.
- Chat template / tokenizer form: model card pending re-read; Unsloth GGUF discussions
  note template pitfalls.
- Multimodal tower: deferred (Basalt vision path is per-model; text-only first).
- 1M ctx on a 24 GB 5090: KV at int8 for 24 layers is feasible-ish for SWA layers; full
  layers dominate. Real capacity numbers belong in the design doc after meta.hpp exists.

## References

- Model card: https://thinkingmachines.ai/model-card/inkling-2-mini/
- llama.cpp chain: #32217 #32429 #32657 #32694 #32704 (Avalon port: azjhf-dev/Avalon
  branch inkling2-mini), #32695 (Unsloth GGUF registration)
- Basalt: docs/adding-a-model.md, src/models/qwen4exp (template), tools/convert/
