fn main() {
    let openvino_lib_dir = "../../../bin/intel64/Release/lib";

    // Statically link in OpenVINO's inference engine (and dependencies).
    println!("cargo:rustc-link-search={}", openvino_lib_dir);
    println!("cargo:rustc-link-lib=static=inference_engine_s");
    println!("cargo:rustc-link-lib=static=pugixml");

    // Dynamically link in libtbb (currently required on system). (TODO make static, see inference-engine/cmake/ie_parallel.cmake:51)
    println!("cargo:rustc-link-lib=tbb");

    // Statically link in libittnotify. TODO find this more conveniently, see inference-engine/cmake/FindITT.cmake
    println!("cargo:rustc-link-search=/opt/intel/vtune_profiler_2020.1.0.607630/lib64");
    println!("cargo:rustc-link-lib=static=ittnotify");

    // Use cxx to compile the library; see https://github.com/dtolnay/cxx/blob/master/demo-rs/build.rs.
    cxx_build::bridge("src/lib.rs")
        .include("../../include") // The location of the OpenVINO headers.
        .flag_if_supported("-std=c++14")
        .compile("ffi");

    // TODO More needed.
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/bridge.h");

    // Copy the necessary plugin file to the correct location.
    let profile = std::env::var("PROFILE").unwrap();
    let deps_lib_dir = format!("target/{}/deps/lib", profile);
    dbg!(&deps_lib_dir);
    std::fs::create_dir_all(&deps_lib_dir).expect("to create the directory");
    std::fs::copy(
        format!("{}/plugins.xml", &openvino_lib_dir),
        format!("{}/plugins.xml", &deps_lib_dir),
    )
    .expect("to copy the plugins.xml file");
}
