use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

fn main() {
    #[cfg(feature = "bindgen")]
    generate_rust_types_from_shader_types();
    compile_shaders();
}

// xcrun -sdk macosx metal -c shaders.metal -o shaders.air
// xcrun -sdk macosx metallib shaders.air -o shaders.metallib
fn compile_shaders() {
    println!("cargo:rerun-if-changed=shaders");
    let paths = fs::read_dir("shaders").unwrap();

    let mut i = 0;

    for path in paths {
        let path = path.unwrap().path();
        //panic!("{:?}", path.extension().unwrap());
        if !path.extension().unwrap().eq("metal") {
            continue;
        }
        let output = Command::new("xcrun")
            .arg("-sdk")
            .arg("macosx")
            .arg("metal")
            .args(["-frecord-sources", "-gline-tables-only"])
            .args(["-c", path.to_str().unwrap()])
            .args(["-o", format!("shaders/shaders{i}.air").as_str()])
            .spawn()
            .unwrap()
            .wait_with_output()
            .unwrap();
        if !output.status.success() {
            panic!("shader compilation failed");
        }
        //fs::remove_file(format!("shaders/shaders{i}.air").as_str()).expect("Failed to remove shader");
        i += 1;
    }

    let output = Command::new("xcrun")
        .arg("-sdk")
        .arg("macosx")
        .arg("metallib")
        //.args("shaders/shaders.air")
        //.args(["-frecord-sources"])//, "-gline-tables-only"])
        .args((0..i).map(|i| format!("shaders/shaders{}.air", i)))
        .args(["-o", "shaders/shaders.metallib"])
        .spawn()
        .unwrap()
        .wait_with_output()
        .expect("xcrun failed");
    if !output.status.success() {
        panic!("failed to aggregate air files")
    }
}

#[cfg(feature = "bindgen")]
fn generate_rust_types_from_shader_types() {
    println!("cargo:rerun-if-changed=shaders/shader_types.h");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out = out.join("shader_bindings.rs");

    let bindings = bindgen::Builder::default()
        .header("shaders/shader_types.h")
        .allowlist_type("Particle")
        .allowlist_type("simd_float3")
        .allowlist_type("ComputeArguments")
        .detect_include_paths(true)
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to enerate bindings");

    bindings.write_to_file(out).unwrap();
    bindings.write_to_file("src/generated_header.rs").unwrap();
}
