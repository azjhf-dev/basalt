// src/models/ernie55/adapter.hpp - the ERNIE 5.5 Flash Preview adapter.
#pragma once
#include "basalt/model/adapter.hpp"

namespace basalt::models::ernie55 {
std::unique_ptr<model::ModelAdapter> make_adapter();
}
