//! Define the interface between Rust and OpenVINO's C++ [API](https://docs.openvinotoolkit.org/latest/usergroup16.html).

mod c_api;

use cxx::UniquePtr;

/// This module uses the `cxx` library to safely bridge the gap to the C++ API exposed by OpenVINO.
#[cxx::bridge(namespace = InferenceEngine)]
mod ffi {
    extern "C" {
        include!("src/bridge.h");

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

/// See [Core](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html).
pub struct Core {
    instance: UniquePtr<ffi::Core>,
}

/// Unfortunately, the OpenVINO APIs return objects that we must wrap in a [UniquePtr] in `bridge.h`
/// before using in Rust.
impl Core {
    pub fn new(xml_config_file: Option<&str>) -> Core {
        let instance = match xml_config_file {
            None => ffi::core_new_default(),
            Some(f) => ffi::core_new(f),
        };
        Core { instance }
    }

    pub fn read_network(&mut self, model_path: &str, bin_path: &str) -> CNNNetwork {
        let instance = ffi::read_network(&mut self.instance, model_path, bin_path);
        CNNNetwork { instance }
    }

    pub fn load_network(&mut self, network: CNNNetwork, device: &str) -> ExecutableNetwork {
        let instance = ffi::load_network(&mut self.instance, network.instance, device);
        ExecutableNetwork { instance }
    }
}

pub struct CNNNetwork {
    instance: UniquePtr<ffi::CNNNetwork>,
}

impl CNNNetwork {
    pub fn set_batch_size(&mut self, size: usize) {
        self.instance.setBatchSize(size)
    }
    pub fn get_input_name(&self, index: usize) -> Result<String, c_api::IEStatusCode> {
        let network =
            self.instance.as_ref().unwrap() as *const ffi::CNNNetwork as *const c_api::ie_network_t;
        let mut name: *mut std::os::raw::c_char = std::ptr::null_mut();
        let name_ptr: *mut *mut std::os::raw::c_char = &mut name;
        let result = unsafe { c_api::ie_network_get_input_name(network, index as u64, name_ptr) };
        if result == c_api::IEStatusCode_OK {
            let returned_name = unsafe { std::ffi::CStr::from_ptr(name) };
            Ok(returned_name.to_string_lossy().to_string())
        } else {
            Err(result)
        }
    }
}

pub struct ExecutableNetwork {
    instance: UniquePtr<ffi::ExecutableNetwork>,
}

impl ExecutableNetwork {
    pub fn create_infer_request(&mut self) -> InferRequest {
        let instance = ffi::create_infer_request(&mut self.instance);
        InferRequest { instance }
    }
}

pub struct InferRequest {
    instance: UniquePtr<ffi::InferRequest>,
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;

    // FIXME this test relies on a plugins.xml file being moved to a default location; see build.rs.
    #[test]
    fn construct_core() {
        Core::new(None);
    }

    // FIXME this test relies on a pre-built model in the filesystem--avoid this.
    #[test]
    fn read_network() {
        let mut core = Core::new(None);
        let dir = Path::new("../../../../test-openvino/");
        core.read_network(
            &dir.join("frozen_inference_graph.xml").to_string_lossy(),
            &dir.join("frozen_inference_graph.bin").to_string_lossy(),
        );
    }

    // FIXME this test relies on a pre-built model in the filesystem--avoid this.
    #[test]
    fn demo() {
        let mut core = Core::new(None);
        let dir = Path::new("../../../../test-openvino/");
        let mut network = core.read_network(
            &dir.join("frozen_inference_graph.xml").to_string_lossy(),
            &dir.join("frozen_inference_graph.bin").to_string_lossy(),
        );
        network.set_batch_size(1);
        assert_eq!(network.get_input_name(0).unwrap(), "image_tensor");

        let mut executable_network = core.load_network(network, "CPU");
        let infer_request = executable_network.create_infer_request();
    }
}
