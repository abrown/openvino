use std::path::{Path, PathBuf};
use std::{fs, io};

fn main() {
    // Note which files will trigger a rebuild.
    mark_rerun_files();

    // Link libraries.
    let openvino_lib_dir = "../../../bin/intel64/Release/lib";
    link_libraries(openvino_lib_dir);

    // Use cxx to compile the library; see https://github.com/dtolnay/cxx/blob/master/demo-rs/build.rs.
    cxx_build::bridge("src/lib.rs")
        .include("../../include") // The location of the OpenVINO headers.
        .flag_if_supported("-std=c++14")
        .compile("ffi");

    // Copy in the plugins.xml file.
    copy_openvino_plugin_file(openvino_lib_dir);
}

/// Helper to mark which files trigger a rerun of the build.
fn mark_rerun_files() {
    // Trigger rebuild on changes to build.rs and Cargo.toml...
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");

    // ...as well as every source file.
    let cb = |p: PathBuf| println!("cargo:rerun-if-changed={}", p.display());
    visit_dirs(Path::new("src"), &cb).expect("to visit source files");
}

/// Helper for linking the libraries necessary to build.
fn link_libraries(openvino_lib_dir: &str) {
    add_library_path(openvino_lib_dir);

    // Statically link in OpenVINO's inference engine (and dependencies).
    println!("cargo:rustc-link-lib=static=pugixml");
    //println!("cargo:rustc-link-lib=static=inference_engine_s");

    // Dynamically link in OpenVINO's inference engine (and dependencies).
    println!("cargo:rustc-link-lib=inference_engine");
    println!("cargo:rustc-link-lib=inference_engine_legacy");
    println!("cargo:rustc-link-lib=inference_engine_transformations");
    println!("cargo:rustc-link-lib=ngraph_backend");
    println!("cargo:rustc-link-lib=ngraph_test_util");
    println!("cargo:rustc-link-lib=ngraph");
    // is this needed?: println!("cargo:rustc-link-args=-Wl,-rpath,{}", openvino_lib_dir);

    // Dynamically link in libtbb (currently required on system). (TODO make static, see inference-engine/cmake/ie_parallel.cmake:51)
    add_library_path("../../temp/tbb/lib");
    println!("cargo:rustc-link-lib=tbb");
    println!("cargo:rustc-link-lib=tbbmalloc");

    // Statically link in libittnotify. TODO find this more conveniently, see inference-engine/cmake/FindITT.cmake
    add_library_path("/opt/intel/vtune_profiler_2020.1.0.607630/lib64");
    println!("cargo:rustc-link-lib=ittnotify");
}

/// Add a path to the set of searchable paths Cargo uses for finding libraries to link to.
fn add_library_path<P: AsRef<Path>>(path: P) {
    let canonicalized = path
        .as_ref()
        .canonicalize()
        .expect("to be able to canonicalize the library path");
    if !canonicalized.exists() || !canonicalized.is_dir() {
        panic!("Unable to find directory: {}", canonicalized.display())
    }
    println!("cargo:rustc-link-search={}", canonicalized.display());
}

/// Helper for recursively visiting the files in this directory; see https://doc.rust-lang.org/std/fs/fn.read_dir.html.
fn visit_dirs(dir: &Path, cb: &dyn Fn(PathBuf)) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(entry.path());
            }
        }
    }
    Ok(())
}

/// Copy the necessary OpenVINO plugin file to the correct location.
fn copy_openvino_plugin_file(openvino_lib_dir: &str) {
    // Create a lib directory next to the built binary; this is the default location OpenVINO expects.
    let profile = std::env::var("PROFILE").unwrap();
    let deps_lib_dir = format!("target/{}/deps/lib", profile);
    std::fs::create_dir_all(&deps_lib_dir).expect("to create the directory");

    // Copy the plugins.xml file.
    std::fs::copy(
        format!("{}/plugins.xml", &openvino_lib_dir),
        format!("{}/plugins.xml", &deps_lib_dir),
    )
    .expect("to copy the plugins.xml file");
}
