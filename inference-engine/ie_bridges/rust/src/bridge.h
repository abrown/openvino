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

std::unique_ptr<InferenceEngine::CNNNetwork> read_network(std::unique_ptr<InferenceEngine::Core> core, rust::Str modelPath, rust::Str binPath) {
    return std::make_unique<InferenceEngine::CNNNetwork>(core->ReadNetwork(std::string(modelPath), std::string(binPath)));
}

#endif //OPENVINO_BRIDGE_H
