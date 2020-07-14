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
    pub fn as_ptr(&self) -> *const c_api::ie_network_t {
        // FIXME likely will cause bugs; relies on the pointer to the network being at offset 0 in both UniquePtr and ie_network_t.
        &*self.instance as *const ffi::CNNNetwork as *const c_api::ie_network_t
    }
    pub fn as_mut(&mut self) -> *mut c_api::ie_network_t {
        // FIXME likely will cause bugs; relies on the pointer to the network being at offset 0 in both UniquePtr and ie_network_t.
        &mut *self.instance as *mut ffi::CNNNetwork as *mut c_api::ie_network_t
    }
    pub fn set_batch_size(&mut self, size: usize) {
        self.instance.setBatchSize(size)
    }
    pub fn get_input_name(&self, index: usize) -> Result<String, InferenceError> {
        let network = self.as_ptr();
        let mut name: *mut std::os::raw::c_char = std::ptr::null_mut();
        let name_ptr: *mut *mut std::os::raw::c_char = &mut name;
        let result = unsafe { c_api::ie_network_get_input_name(network, index as u64, name_ptr) };
        InferenceError::from(result).and(Ok(unsafe { std::ffi::CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()))
    }
    pub fn get_output_name(&self, index: usize) -> Result<String, InferenceError> {
        let network = self.as_ptr();
        let mut name: *mut std::os::raw::c_char = std::ptr::null_mut();
        let name_ptr: *mut *mut std::os::raw::c_char = &mut name;
        let result = unsafe { c_api::ie_network_get_output_name(network, index as u64, name_ptr) };
        InferenceError::from(result).and(Ok(unsafe { std::ffi::CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()))
    }

    // TODO split into separate methods
    pub fn prep_inputs_and_outputs(
        &mut self,
        input_name: &str,
        output_name: &str,
    ) -> Result<(), InferenceError> {
        let network = self.as_mut();
        let input_name = std::ffi::CString::new(input_name).unwrap().into_raw();
        let output_name = std::ffi::CString::new(output_name).unwrap().into_raw();
        let mut status = c_api::IEStatusCode_OK;
        unsafe {
            status |= c_api::ie_network_set_input_resize_algorithm(
                network,
                input_name,
                c_api::resize_alg_e_RESIZE_BILINEAR,
            );
            status |= c_api::ie_network_set_input_layout(network, input_name, c_api::layout_e_NHWC);
            status |=
                c_api::ie_network_set_input_precision(network, input_name, c_api::precision_e_U8);

            status |= c_api::ie_network_set_output_precision(
                network,
                output_name,
                c_api::precision_e_FP32,
            );
        }
        InferenceError::from(status)
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

impl InferRequest {
    pub fn set_blob(&mut self, name: &str, blob: Blob) -> Result<(), InferenceError> {
        let infer_request = self.instance.as_mut().unwrap() as *mut ffi::InferRequest
            as *mut c_api::ie_infer_request; // FIXME
        let input_name = std::ffi::CString::new(name).unwrap().into_raw();
        let result =
            unsafe { c_api::ie_infer_request_set_blob(infer_request, input_name, blob.internal) };
        InferenceError::from(result)
    }
}

pub struct Blob {
    internal: *const c_api::ie_blob_t,
}

impl Blob {
    pub fn new(description: TensorDescription, data: &mut [u8]) -> Result<Self, InferenceError> {
        let mut blob: *mut c_api::ie_blob_t = std::ptr::null_mut();
        let blob_ptr: *mut *mut c_api::ie_blob_t = &mut blob;
        let data_ptr = data as *mut [u8] as *mut std::os::raw::c_void;
        let result = unsafe {
            c_api::ie_blob_make_memory_from_preallocated(
                description.as_ptr(),
                data_ptr,
                data.len() as u64,
                blob_ptr,
            )
        };
        InferenceError::from(result).and(Ok(Self { internal: blob }))
    }
}

pub struct TensorDescription {
    internal: c_api::tensor_desc_t,
}

impl TensorDescription {
    pub fn new(layout: c_api::layout_e, dimensions: &[u64], precision: c_api::precision_e) -> Self {
        // Setup dimensions.
        assert!(dimensions.len() < 8);
        let mut dims = [0; 8];
        dims[..dimensions.len()].copy_from_slice(dimensions);

        // Create the description structure.
        Self {
            internal: c_api::tensor_desc_t {
                layout,
                dims: c_api::dimensions_t {
                    ranks: dimensions.len() as u64,
                    dims,
                },
                precision,
            },
        }
    }

    pub fn as_ptr(&self) -> *const c_api::tensor_desc_t {
        &self.internal as *const _
    }
}

#[derive(Debug)]
pub enum InferenceError {
    GeneralError,
    NotImplemented,
    NetworkNotLoaded,
    ParameterMismatch,
    NotFound,
    OutOfBounds,
    Unexpected,
    RequestBusy,
    ResultNotReady,
    NotAllocated,
    InferNotStarted,
    NetworkNotReady,
    Undefined,
}

impl InferenceError {
    pub fn from(e: i32) -> Result<(), InferenceError> {
        use InferenceError::*;
        match e {
            // TODO use enum constants from c_api: e.g. c_api::IEStatusCode_OK => ...
            0 => Ok(()),
            -1 => Err(GeneralError),
            -2 => Err(NotImplemented),
            -3 => Err(NetworkNotLoaded),
            -4 => Err(ParameterMismatch),
            -5 => Err(NotFound),
            -6 => Err(OutOfBounds),
            -7 => Err(Unexpected),
            -8 => Err(RequestBusy),
            -9 => Err(ResultNotReady),
            -10 => Err(NotAllocated),
            -11 => Err(InferNotStarted),
            -12 => Err(NetworkNotReady),
            _ => Err(Undefined),
        }
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

        let input_name = network.get_input_name(0).unwrap();
        assert_eq!(input_name, "image_tensor");
        let output_name = network.get_output_name(0).unwrap();
        assert_eq!(output_name, "DetectionOutput");
        network.prep_inputs_and_outputs(&input_name, &output_name);

        let mut executable_network = core.load_network(network, "CPU");
        let mut infer_request = executable_network.create_infer_request();

        // FIXME the dimensions for this file are hard-coded
        // $ file val2017/000000062808.jpg
        // val2017/000000062808.jpg: JPEG image data, JFIF standard 1.01, resolution (DPI), density 72x72, segment length 16, baseline, precision 8, 640x481, components 3
        let desc = TensorDescription::new(
            c_api::layout_e_NHWC,
            &[1, 3, 481, 640], // {1, (size_t)img.mat_channels, (size_t)img.mat_height, (size_t)img.mat_width}
            c_api::precision_e_U8,
        );
        let mut bytes = std::fs::read(dir.join("val2017/000000062808.jpg")).unwrap();
        let blob = Blob::new(desc, &mut bytes).unwrap();
        // TODO infer_request.set_blob(&input_name, blob).unwrap();
    }
}
