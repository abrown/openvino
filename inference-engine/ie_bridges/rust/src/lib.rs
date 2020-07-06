//! Define the interface between Rust and OpenVINO's C++ [API](https://docs.openvinotoolkit.org/latest/usergroup16.html).

use cxx::CxxVector;
use cxx::UniquePtr;

#[cxx::bridge(namespace = InferenceEngine)]
mod ffi {
    extern "C" {
        include!("src/bridge.h");
        type Core;
        pub fn core_new(xml_config_file: &str) -> UniquePtr<Core>;
        pub fn core_new_default() -> UniquePtr<Core>;
    }
}

/// See [Core](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html).
pub struct Core {
    instance: UniquePtr<ffi::Core>,
}

impl Core {
    pub fn new(xml_config_file: Option<&str>) -> Core {
        let instance = match xml_config_file {
            None => unsafe { ffi::core_new_default() },
            Some(f) => unsafe { ffi::core_new(f) },
        };
        Core { instance }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn construct_core() {
        Core::new(None);
    }
}
