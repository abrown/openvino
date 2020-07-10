//! Define the interface between Rust and OpenVINO's C++ [API](https://docs.openvinotoolkit.org/latest/usergroup16.html).

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
            core: UniquePtr<Core>,
            model_path: &str,
            bin_path: &str,
        ) -> UniquePtr<CNNNetwork>;

        type CNNNetwork;
        pub fn setBatchSize(self: &mut CNNNetwork, size: usize);
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

    pub fn read_network(self, model_path: &str, bin_path: &str) -> CNNNetwork {
        let instance = ffi::read_network(self.instance, model_path, bin_path);
        CNNNetwork { instance }
    }
}

pub struct CNNNetwork {
    instance: UniquePtr<ffi::CNNNetwork>,
}

impl CNNNetwork {
    pub fn set_batch_size(&mut self, size: usize) {
        self.instance.setBatchSize(size)
    }
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
        let core = Core::new(None);
        let dir = Path::new("../../../../test-openvino/");
        core.read_network(
            &dir.join("frozen_inference_graph.xml").to_string_lossy(),
            &dir.join("frozen_inference_graph.bin").to_string_lossy(),
        );
    }

    // FIXME this test relies on a pre-built model in the filesystem--avoid this.
    #[test]
    fn set_batch_size() {
        let core = Core::new(None);
        let dir = Path::new("../../../../test-openvino/");
        let mut network = core.read_network(
            &dir.join("frozen_inference_graph.xml").to_string_lossy(),
            &dir.join("frozen_inference_graph.bin").to_string_lossy(),
        );
        network.set_batch_size(1);
    }
}
