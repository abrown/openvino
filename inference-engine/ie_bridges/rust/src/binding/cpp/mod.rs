/// This module uses the `cxx` library to safely bridge the gap to the C++ API exposed by OpenVINO.
#[cxx::bridge(namespace = InferenceEngine)]
mod ffi {
    extern "C" {
        include!("src/binding/cpp/bridge.h");

        type Core;
        pub fn core_new(xml_config_file: &str) -> UniquePtr<Core>;
        pub fn core_new_default() -> UniquePtr<Core>;
        pub fn read_network(
            core: &mut Core,
            model_path: &str,
            bin_path: &str,
        ) -> UniquePtr<CNNNetwork>;
        pub fn load_network(
            core: &mut Core,
            network: UniquePtr<CNNNetwork>,
            device: &str,
        ) -> UniquePtr<ExecutableNetwork>;

        type CNNNetwork;
        pub fn setBatchSize(self: &mut CNNNetwork, size: usize);

        type ExecutableNetwork;
        pub fn create_infer_request(network: &mut ExecutableNetwork) -> UniquePtr<InferRequest>;

        type InferRequest;
    }
}

/// Re-publicize the cxx-generated structures under the cpp namespace.
pub use ffi::*;
