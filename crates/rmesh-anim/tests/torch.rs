#[test]
fn embedded_torch_loads() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rmesh-viewer-web/web/assets/flare.gltf");
    let data = std::fs::read(path).unwrap();
    let scene = rmesh_anim::gltf_loader::load_gltf_bytes(&data, "Torch")
        .unwrap_or_else(|e| panic!("{e:#}"));
    assert!(!scene.meshes.is_empty());
}
