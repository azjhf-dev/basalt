// src/models/inkling2/adapter.cpp - scaffolding. UNCOMPILED in the authoring sandbox (no CUDA).
//
// TODO(iteration 3): implement ModelAdapter over meta.hpp:
//   validate  -> validate_meta(g)
//   describe  -> describe_meta(g)
//   load      -> pack read into a Weights struct; NVFP4 expert planes + MXFP8 projections follow
//                src/artifact/dequant.cpp and the qwen35 converter's NVFP4_PLANES form
//   session   -> ModelSession with decode_window/prefill_chunk; running_state lists KV (int8) +
//                conv rollback slots + loop pass state, in a fixed order, for checkpoint_transfer
// The engine fast path derives from serve::Runtime / serve::Request like src/models/qwen4exp/.
#include "adapter.hpp"

namespace basalt::models::inkling2 {

std::unique_ptr<model::ModelAdapter> make_adapter() {
    return nullptr;  // not implemented: registration stays commented out of cmake/models.cmake
}

}  // namespace basalt::models::inkling2
