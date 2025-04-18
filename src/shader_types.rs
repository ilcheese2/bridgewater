#[cfg(feature = "bindgen")]
include!(concat!(env!("OUT_DIR"), "/shader_bindings.rs"));
#[cfg(not(feature = "bindgen"))]
include!(concat!("../generated_header.rs"));
