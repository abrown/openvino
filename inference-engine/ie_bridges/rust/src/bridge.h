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

std::unique_ptr<InferenceEngine::CNNNetwork> read_network(InferenceEngine::Core &core, rust::Str modelPath, rust::Str binPath) {
    return std::make_unique<InferenceEngine::CNNNetwork>(core.ReadNetwork(std::string(modelPath), std::string(binPath)));
}

std::unique_ptr<InferenceEngine::ExecutableNetwork> load_network(InferenceEngine::Core &core, std::unique_ptr<InferenceEngine::CNNNetwork> network, rust::Str device) {
    return std::make_unique<InferenceEngine::ExecutableNetwork>(core.LoadNetwork(*network, std::string(device)));
}

std::unique_ptr<InferenceEngine::InferRequest> create_infer_request(InferenceEngine::ExecutableNetwork &network) {
    return std::make_unique<InferenceEngine::InferRequest>(network.CreateInferRequest());
}

#endif //OPENVINO_BRIDGE_H
