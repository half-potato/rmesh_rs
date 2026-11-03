use rmesh_anim::gltf_loader::{load_gltf, load_gltf_bytes};

#[test]
fn embedded_gltf_matches_file_loading_and_seeks() {
    let bytes = include_bytes!("fixtures/slide.gltf");
    let mut embedded = load_gltf_bytes(bytes, "slide").unwrap();
    let disk = load_gltf(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/slide.gltf"
    )))
    .unwrap();
    assert_eq!(embedded.meshes[0].vertices, disk.meshes[0].vertices);
    assert_eq!(embedded.scene.clips.len(), 1);
    assert_eq!(embedded.scene.current_duration(), 1.0);
    embedded.scene.playback.playing = false;
    embedded.scene.playback.looping = false;
    embedded.scene.playback.time = 0.0;
    embedded.scene.update(0.0);
    let start = embedded.scene.render_primitives()[0].transform.position;
    let primitive = &embedded.scene.render_primitives()[0];
    let vertex = glam::Vec3::from(embedded.meshes[0].vertices[0]);
    let world = primitive.transform.model_matrix().transform_point3(vertex);
    assert!(
        (world - glam::Vec3::new(99.2, -2.0, 0.0)).length() < 1e-5,
        "{world:?}"
    );
    embedded.scene.playback.time = 1.0;
    embedded.scene.update(0.0);
    let end = embedded.scene.render_primitives()[0].transform.position;
    assert!((end.x - start.x - 0.8).abs() < 1e-6, "{start:?} -> {end:?}");
}

#[test]
fn memory_import_rejects_external_images_and_buffers() {
    let fixture = std::str::from_utf8(include_bytes!("fixtures/slide.gltf")).unwrap();
    let external_buffer = fixture.replace("data:application/octet-stream;base64,", "external.bin#");
    let error = load_gltf_bytes(external_buffer.as_bytes(), "external")
        .err()
        .unwrap();
    assert!(error.to_string().contains("Buffer must be embedded"));
    let external_image = fixture.replacen("{", r#"{"images":[{"uri":"external.png"}],"#, 1);
    let error = load_gltf_bytes(external_image.as_bytes(), "external")
        .err()
        .unwrap();
    assert!(error.to_string().contains("Image must be embedded"));
}

#[test]
fn node_edits_survive_animation_and_preview_restores_children() {
    let mut loaded = load_gltf_bytes(include_bytes!("fixtures/slide.gltf"), "slide").unwrap();
    let scene = &mut loaded.scene;
    scene.playback.playing = false;
    scene.playback.looping = false;
    scene.update(0.0);
    let original = scene.nodes[1].world_transform;
    let mut edited = original;
    edited.position.y += 2.0;
    scene.set_node_world_transform(1, edited);
    scene.playback.time = 1.0;
    scene.update(0.0);
    assert!((scene.nodes[1].world_transform.position.y - original.position.y - 2.0).abs() < 1e-5);
    assert!((scene.nodes[1].world_transform.position.x - original.position.x - 0.8).abs() < 1e-5);
    let before_preview = scene.nodes[1].world_transform.position;
    let mut root = scene.nodes[0].world_transform;
    root.position.z += 3.0;
    let preview = scene.render_node_preview(0, root);
    assert!((preview[0].transform.position.z - before_preview.z - 3.0).abs() < 1e-5);
    assert_eq!(scene.nodes[1].world_transform.position, before_preview);
    scene.remove_node_and_descendants(0);
    assert!(scene.nodes.is_empty());
    assert!(scene.clips[0].channels.is_empty());
}
