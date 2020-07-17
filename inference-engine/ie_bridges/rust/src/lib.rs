//! Define the interface between Rust and OpenVINO's C++ [API](https://docs.openvinotoolkit.org/latest/usergroup16.html).

mod c_api;

use cxx::UniquePtr;
use std::convert::TryFrom;

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
    pub fn as_ptr(&self) -> *const c_api::ie_infer_request_t {
        // FIXME likely will cause bugs; relies on the pointer to the request being at offset 0 in both UniquePtr and ie_network_t.
        &*self.instance as *const ffi::InferRequest as *const c_api::ie_infer_request_t
    }
    pub fn as_mut(&mut self) -> *mut c_api::ie_infer_request_t {
        // FIXME likely will cause bugs; relies on the pointer to the request being at offset 0 in both UniquePtr and ie_network_t.
        &mut *self.instance as *mut ffi::InferRequest as *mut c_api::ie_infer_request_t
    }
    pub fn set_blob(&mut self, name: &str, blob: Blob) -> Result<(), InferenceError> {
        let infer_request_ptr = self.as_mut();
        let name_ptr = std::ffi::CString::new(name).unwrap().into_raw();
        let blob_ptr = blob.internal;
        let result =
            unsafe { c_api::ie_infer_request_set_blob(infer_request_ptr, name_ptr, blob_ptr) };
        InferenceError::from(result)
    }
    pub fn get_blob(&mut self, name: &str) -> Result<Blob, InferenceError> {
        let name_ptr = std::ffi::CString::new(name).unwrap().into_raw();
        let mut blob: *mut c_api::ie_blob_t = std::ptr::null_mut();
        let blob_ptr: *mut *mut c_api::ie_blob_t = &mut blob;
        let result = unsafe { c_api::ie_infer_request_get_blob(self.as_mut(), name_ptr, blob_ptr) };
        InferenceError::from(result).and(Ok(Blob::from(blob)))
    }
    pub fn infer(&mut self) -> Result<(), InferenceError> {
        let result = unsafe { c_api::ie_infer_request_infer(self.as_mut()) };
        InferenceError::from(result)
    }
}

pub struct Blob {
    internal: *mut c_api::ie_blob_t,
}

impl Blob {
    pub fn from(pointer: *mut c_api::ie_blob_t) -> Self {
        Self { internal: pointer }
    }

    /// Allocate space in OpenVINO for an empty Blob.
    pub fn allocate(description: TensorDescription) -> Result<Self, InferenceError> {
        let mut blob: *mut c_api::ie_blob_t = std::ptr::null_mut();
        let blob_ptr: *mut *mut c_api::ie_blob_t = &mut blob;
        let result = unsafe { c_api::ie_blob_make_memory(description.as_ptr(), blob_ptr) };
        InferenceError::from(result).and(Ok(Self { internal: blob }))
    }

    /// Create a new blob by copying data in to the OpenVINO-allocated memory.
    pub fn new(description: TensorDescription, data: &[u8]) -> Result<Self, InferenceError> {
        let mut blob = Self::allocate(description)?;
        let blob_len = blob.byte_len()?;
        assert_eq!(
            blob_len,
            data.len(),
            "The data to initialize ({} bytes) must be the same as the blob size ({} bytes).",
            data.len(),
            blob_len
        );

        // Copy the incoming data into the buffer.
        let buffer = blob.buffer()?;
        buffer.copy_from_slice(data);

        Ok(blob)
    }

    pub fn tensor_desc(&self) -> Result<TensorDescription, InferenceError> {
        let blob = self.internal as *const c_api::ie_blob_t;

        let mut layout: c_api::layout_e = 0;
        let layout_ptr = &mut layout as *mut c_api::layout_e;
        let result = unsafe { c_api::ie_blob_get_layout(blob, layout_ptr) };
        InferenceError::from(result)?;

        let mut dimensions = c_api::dimensions_t {
            ranks: 0,
            dims: [0; 8usize],
        };
        let dimensions_ptr = &mut dimensions as *mut c_api::dimensions_t;
        let result = unsafe { c_api::ie_blob_get_dims(blob, dimensions_ptr) };
        InferenceError::from(result)?;

        let mut precision: c_api::precision_e = 0;
        let precision_ptr = &mut precision as *mut c_api::layout_e;
        let result = unsafe { c_api::ie_blob_get_precision(blob, precision_ptr) };
        InferenceError::from(result)?;

        Ok(TensorDescription::new(
            precision,
            &dimensions.dims,
            precision,
        ))
    }

    /// Get the number of elements contained in the Blob.
    pub fn len(&mut self) -> Result<usize, InferenceError> {
        let mut size = 0;
        let size_ptr = &mut size as *mut std::os::raw::c_int;
        let result = unsafe { c_api::ie_blob_size(self.internal, size_ptr) };
        InferenceError::from(result).and(Ok(usize::try_from(size).unwrap()))
    }

    /// Get the size of the current Blob in bytes.
    pub fn byte_len(&mut self) -> Result<usize, InferenceError> {
        let mut size = 0;
        let size_ptr = &mut size as *mut std::os::raw::c_int;
        let result = unsafe { c_api::ie_blob_byte_size(self.internal, size_ptr) };
        InferenceError::from(result).and(Ok(usize::try_from(size).unwrap()))
    }

    pub fn buffer<T>(&mut self) -> Result<&mut [T], InferenceError> {
        let mut buffer = Blob::empty_buffer();
        let buffer_ptr = &mut buffer as *mut c_api::ie_blob_buffer_t;
        let result = unsafe { c_api::ie_blob_get_buffer(self.internal, buffer_ptr) };
        InferenceError::from(result)?;
        let size = self.len()?;
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer as *mut T, size)
        };
        Ok(slice)
    }

    fn empty_buffer() -> c_api::ie_blob_buffer_t {
        c_api::ie_blob_buffer_t {
            __bindgen_anon_1: c_api::ie_blob_buffer__bindgen_ty_1 {
                buffer: std::ptr::null_mut(),
            },
        }
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

    pub fn len(&self) -> usize {
        self.internal.dims.dims[..self.internal.dims.ranks as usize]
            .iter()
            .fold(1, |a, &b| a * b as usize)
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
    use opencv::core::{MatTrait, MatTraitManual};
    use std::path::PathBuf;

    // FIXME these tests rely on a pre-built model and images in the filesystem--avoid this.
    struct Fixture;
    impl Fixture {
        fn dir() -> PathBuf {
            PathBuf::from("../../../../test-openvino/")
        }
        fn graph() -> PathBuf {
            Fixture::dir().join("frozen_inference_graph.xml")
        }
        fn weights() -> PathBuf {
            Fixture::dir().join("frozen_inference_graph.bin")
        }
        fn image() -> PathBuf {
            Fixture::dir().join("val2017/000000062808.jpg")
        }
    }

    // FIXME this test relies on a plugins.xml file being moved to a default location; see build.rs.
    #[test]
    fn construct_core() {
        Core::new(None);
    }

    #[test]
    fn read_network() {
        let mut core = Core::new(None);
        core.read_network(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        );
    }

    #[test]
    fn read_image() {
        let mat = opencv::imgcodecs::imread(
            &*Fixture::image().to_string_lossy(),
            opencv::imgcodecs::IMREAD_COLOR,
        )
        .unwrap();

        assert_eq!(mat.channels().unwrap(), 3);
        assert_eq!(mat.typ().unwrap(), opencv::core::CV_8UC3);
    }

    #[test]
    fn decode_image() {
        let bytes = std::fs::read(Fixture::image()).unwrap();
        let mut bytes_mat = opencv::core::Mat::from_slice::<u8>(&bytes).unwrap();
        let mat =
            opencv::imgcodecs::imdecode(&mut bytes_mat, opencv::imgcodecs::IMREAD_COLOR).unwrap();

        assert_eq!(mat.channels().unwrap(), 3);
        assert_eq!(mat.typ().unwrap(), opencv::core::CV_8UC3);
    }

    #[test]
    fn demo() {
        let mut core = Core::new(None);
        let mut network = core.read_network(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        );
        network.set_batch_size(1);

        let input_name = network.get_input_name(0).unwrap();
        assert_eq!(input_name, "image_tensor");
        let output_name = network.get_output_name(0).unwrap();
        assert_eq!(output_name, "DetectionOutput");
        network
            .prep_inputs_and_outputs(&input_name, &output_name)
            .unwrap();

        let mut executable_network = core.load_network(network, "CPU");
        let mut infer_request = executable_network.create_infer_request();

        // Read the image.
        let mat = opencv::imgcodecs::imread(
            &*Fixture::image().to_string_lossy(),
            opencv::imgcodecs::IMREAD_COLOR,
        )
        .unwrap();
        let desc = TensorDescription::new(
            c_api::layout_e_NHWC,
            &[
                1,
                mat.channels().unwrap() as u64,
                mat.size().unwrap().height as u64,
                mat.size().unwrap().width as u64, // TODO .try_into().unwrap()
            ], // {1, (size_t)img.mat_channels, (size_t)img.mat_height, (size_t)img.mat_width}
            c_api::precision_e_U8,
        );

        // Extract the OpenCV mat bytes and place them in an OpenVINO blob.
        let data =
            unsafe { std::slice::from_raw_parts(mat.data().unwrap() as *const u8, desc.len()) };
        let blob = Blob::new(desc, data).unwrap();

        infer_request.set_blob(&input_name, blob).unwrap();
        infer_request.infer().unwrap();
        let mut results = infer_request.get_blob(&output_name).unwrap();
        let buffer = results.buffer::<f32>().unwrap().to_vec();

        // Sort results.
        #[derive(Debug, PartialEq)]
        struct Result(usize, f32); // (category, probability)
        type Results = Vec<Result>;
        let mut results: Results = buffer
            .iter()
            .enumerate()
            .map(|(c, p)| Result(c, *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(
            &results[..5],
            &[
                Result(15, 59.0),
                Result(1, 1.0),
                Result(8, 1.0),
                Result(12, 1.0),
                Result(16, 0.9939936),
            ][..]
        )
    }
}
