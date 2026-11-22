// src/models/inkling2/adapter.hpp - model entry points, mirroring src/models/qwen4exp/adapter.hpp.
// STATUS: scaffolding, UNCOMPILED. make_adapter() returns nullptr until the ModelAdapter methods
// (validate/describe/load/tokens/session) are implemented over include/basalt/model/adapter.hpp.
#pragma once
#include "basalt/model/adapter.hpp"

#include <memory>
#include <vector>

namespace basalt::models::inkling2 {
struct Options;  // decode/prefill options: kv quant (int8 default), max context, resident experts
}

namespace basalt::models::inkling2 {
std::unique_ptr<model::ModelAdapter> make_adapter();
int run_engine(Options& o, std::vector<program::GpuCard>& cards);
bool parse_flag(const std::string& name, model::FlagArgs& args, Options& o, int& code);
std::optional<int> finish_flags(Options& o);
}
