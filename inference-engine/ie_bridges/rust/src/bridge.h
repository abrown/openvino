#include <memory>
#include "inference_engine.hpp"
#include "rust/cxx.h"

#ifndef OPENVINO_BRIDGE_H
#define OPENVINO_BRIDGE_H

std::unique_ptr<InferenceEngine::Core> core_new(rust::Str xmlConfigFile = {}) {
    return std::make_unique<InferenceEngine::Core>(InferenceEngine::Core(std::string(xmlConfigFile)));
}

std::unique_ptr<InferenceEngine::Core> core_new_default() {
    return std::make_unique<InferenceEngine::Core>(InferenceEngine::Core());
}

#endif //OPENVINO_BRIDGE_H
