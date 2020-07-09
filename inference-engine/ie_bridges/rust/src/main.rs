use openvino::Core;
use std::path::Path;

fn main() {
    // FIXME this relies on the OpenVINO file structure to retrieve a plugins.xml file.
    let core = Core::new(Some("../../../bin/intel64/Release/lib/plugins.xml"));
    // FIXME this relies on a pre-built model currently stored in the filesystem.
    let dir = Path::new("../../../../test-openvino/")
        .canonicalize()
        .expect("a canonical version of this path");
    if dir.exists() {
        let network = core.read_network(
            &dir.join("frozen_inference_graph.xml").to_string_lossy(),
            &dir.join("frozen_inference_graph.bin").to_string_lossy(),
        );
        println!("instantiated network: {:p}", &network);
    } else {
        panic!("Could not find model directory: {}", dir.display());
    }
}
