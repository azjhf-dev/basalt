# Inkling 2 Mini targets. STATUS: scaffolding - model_inkling2 builds the adapter skeleton only.
# Registration: add `inkling2` to BASALT_BUILTIN_MODELS in cmake/models.cmake once make_adapter()
# is implemented, then everything below sits under if(BASALT_MODEL_INKLING2) in CMakeLists.txt.

add_library(model_inkling2 STATIC
  src/models/inkling2/adapter.cpp
)
target_include_directories(model_inkling2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Planned kernel library (each kernel gets a parity test + microbench per docs/adding-a-model.md):
# add_library(model_inkling2_kernels STATIC
#   src/models/inkling2/kernels/attn_rel_bias.cu    # decode FA: in-kernel bias-table gather + sinks, int8 KV
#   src/models/inkling2/kernels/attn_rel_bias_prefill.cu
#   src/models/inkling2/kernels/loop_gate.cu        # per-pass gate mixing across the 3 loop visits
#   src/models/inkling2/kernels/conv_state.cu       # short conv + rollback slots (pattern: qwen4exp gdn.cu)
#   src/models/inkling2/kernels/moe_decode.cu       # 128 experts top-8 sigmoid, 1152-wide NVFP4 (pattern: dec_moe.cu)
#   src/models/inkling2/kernels/kv_int8.cu          # adapt qwen4exp kv_q8.cu to per-head-bias layout
# )
