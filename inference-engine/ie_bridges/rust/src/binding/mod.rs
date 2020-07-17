//! This module contains the bindings to OpenVINO's native functions. Though a single C++ binding
//! would be preferred, limitations in the `cxx` library (see, e.g. https://github.com/dtolnay/cxx/issues/228)
//! prevent us from using the `cxx` as-is. The missing functionality is supplemented using OpenVINO's
//! C API, which itself is a wrapper of the C++ API. At some point in the future, this module may
//! contain only one of these bindings.
pub mod c;
pub mod cpp;
