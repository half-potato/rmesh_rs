//! Common glTF geometry/material upload for desktop and browser viewers.
use rmesh_anim::GltfScene;
use rmesh_compositor::{
    MaterialDef, MaterialRegistry, PrimitiveGeometry, PrimitiveVertex, TextureData,
};

/// Upload packs in order, returning each pack's mesh/material base indices.
pub fn upload_models(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    geometry: &mut PrimitiveGeometry,
    registry: &mut MaterialRegistry,
    models: &[&GltfScene],
) -> Vec<(usize, usize)> {
    let mut meshes = Vec::new();
    let mut textures = Vec::new();
    let mut materials = Vec::new();
    let mut bases = Vec::new();
    for model in models {
        bases.push((meshes.len(), materials.len()));
        meshes.extend(model.meshes.iter().map(|m| {
            (0..m.vertices.len())
                .map(|i| PrimitiveVertex {
                    position: m.vertices[i],
                    normal: m.normals[i],
                    uv: m.uvs.get(i).copied().unwrap_or([0.0; 2]),
                    tangent: m.tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
                })
                .collect::<Vec<_>>()
        }));
        let base = textures.len();
        textures.extend(model.textures.iter().map(|t| TextureData {
            width: t.width,
            height: t.height,
            pixels: t.pixels.clone(),
        }));
        materials.extend(model.materials.iter().map(|m| MaterialDef {
            base_color_factor: m.base_color_factor,
            roughness_factor: m.roughness_factor,
            metallic_factor: m.metallic_factor,
            occlusion_strength: m.occlusion_strength,
            normal_scale: m.normal_scale,
            base_color_texture: m.base_color_texture.map(|i| i + base),
            metallic_roughness_texture: m.metallic_roughness_texture.map(|i| i + base),
            normal_texture: m.normal_texture.map(|i| i + base),
            occlusion_texture: m.occlusion_texture.map(|i| i + base),
        }));
    }
    geometry.set_custom_meshes(device, &meshes);
    registry.upload(device, queue, &textures, &materials);
    bases
}

pub fn offset_scene(scene: &mut rmesh_anim::AnimatedScene, mesh_base: usize, material_base: usize) {
    for node in &mut scene.nodes {
        if let Some(rmesh_interact::PrimitiveKind::CustomMesh(ref mut index)) = node.mesh_kind {
            *index += mesh_base;
        }
        if let Some(ref mut index) = node.material_index {
            *index += material_base;
        }
    }
}
