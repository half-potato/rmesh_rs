//! Load glTF/GLB files into [`AnimatedScene`] + raw mesh vertex data + material/texture data.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use glam::{Quat, Vec3};

use rmesh_interact::PrimitiveKind;

use crate::{
    AnimatedScene, AnimationChannel, AnimationClip, Interpolation, PlaybackState, SceneNode,
    TargetProperty,
};

/// A loaded mesh: flat triangle list with positions, normals, UVs, and tangents (COLMAP coordinates).
pub struct LoadedMesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
}

/// A loaded texture image (always RGBA8).
pub struct LoadedTexture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// A loaded PBR material definition referencing textures by index.
pub struct LoadedMaterial {
    pub base_color_factor: [f32; 4],
    pub roughness_factor: f32,
    pub metallic_factor: f32,
    pub occlusion_strength: f32,
    pub normal_scale: f32,
    pub base_color_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
}

/// Complete result of loading a glTF file.
pub struct GltfScene {
    pub scene: AnimatedScene,
    /// Custom meshes indexed by `PrimitiveKind::CustomMesh(id)`.
    pub meshes: Vec<LoadedMesh>,
    /// Loaded texture images (RGBA8).
    pub textures: Vec<LoadedTexture>,
    /// PBR material definitions.
    pub materials: Vec<LoadedMaterial>,
}

/// Convert a Y-up coordinate to COLMAP (-Z up): `[x, y, z] → [x, z, -y]`.
fn yup_to_colmap(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], -v[1]]
}

/// Compute a flat (face) normal for a triangle.
fn flat_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = Vec3::from(b) - Vec3::from(a);
    let v = Vec3::from(c) - Vec3::from(a);
    let n = u.cross(v);
    let len = n.length();
    if len < 1e-10 {
        [0.0, 0.0, 1.0]
    } else {
        (n / len).to_array()
    }
}

/// Compute tangent for a triangle from positions and UVs.
fn compute_tangent(
    p0: [f32; 3], p1: [f32; 3], p2: [f32; 3],
    uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2],
    normal: [f32; 3],
) -> [f32; 4] {
    let dp1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let dp2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
    let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

    let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
    let inv_det = if det.abs() < 1e-10 { 1.0 } else { 1.0 / det };

    let t = [
        inv_det * (duv2[1] * dp1[0] - duv1[1] * dp2[0]),
        inv_det * (duv2[1] * dp1[1] - duv1[1] * dp2[1]),
        inv_det * (duv2[1] * dp1[2] - duv1[1] * dp2[2]),
    ];
    let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let t_norm = if len < 1e-10 { [1.0, 0.0, 0.0] } else { [t[0] / len, t[1] / len, t[2] / len] };

    let b = [
        inv_det * (-duv2[0] * dp1[0] + duv1[0] * dp2[0]),
        inv_det * (-duv2[0] * dp1[1] + duv1[0] * dp2[1]),
        inv_det * (-duv2[0] * dp1[2] + duv1[0] * dp2[2]),
    ];
    let cross = [
        normal[1] * t_norm[2] - normal[2] * t_norm[1],
        normal[2] * t_norm[0] - normal[0] * t_norm[2],
        normal[0] * t_norm[1] - normal[1] * t_norm[0],
    ];
    let dot = cross[0] * b[0] + cross[1] * b[1] + cross[2] * b[2];
    let w = if dot >= 0.0 { 1.0 } else { -1.0 };

    [t_norm[0], t_norm[1], t_norm[2], w]
}

/// Convert gltf image data to RGBA8.
fn image_to_rgba8(data: &gltf::image::Data) -> Vec<u8> {
    match data.format {
        gltf::image::Format::R8 => {
            data.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect()
        }
        gltf::image::Format::R8G8 => {
            data.pixels.chunks(2).flat_map(|rg| [rg[0], rg[1], 0, 255]).collect()
        }
        gltf::image::Format::R8G8B8 => {
            data.pixels.chunks(3).flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255]).collect()
        }
        gltf::image::Format::R8G8B8A8 => data.pixels.clone(),
        gltf::image::Format::R16 => {
            data.pixels.chunks(2).flat_map(|c| {
                let v = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
                [v, v, v, 255]
            }).collect()
        }
        gltf::image::Format::R16G16 => {
            data.pixels.chunks(4).flat_map(|c| {
                let r = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8;
                [r, g, 0, 255]
            }).collect()
        }
        gltf::image::Format::R16G16B16 => {
            data.pixels.chunks(6).flat_map(|c| {
                let r = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8;
                let b = (u16::from_le_bytes([c[4], c[5]]) >> 8) as u8;
                [r, g, b, 255]
            }).collect()
        }
        gltf::image::Format::R16G16B16A16 => {
            data.pixels.chunks(8).flat_map(|c| {
                let r = (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([c[2], c[3]]) >> 8) as u8;
                let b = (u16::from_le_bytes([c[4], c[5]]) >> 8) as u8;
                let a = (u16::from_le_bytes([c[6], c[7]]) >> 8) as u8;
                [r, g, b, a]
            }).collect()
        }
        gltf::image::Format::R32G32B32FLOAT => {
            data.pixels.chunks(12).flat_map(|c| {
                let r = (f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(0.0, 1.0) * 255.0) as u8;
                let g = (f32::from_le_bytes([c[4], c[5], c[6], c[7]]).clamp(0.0, 1.0) * 255.0) as u8;
                let b = (f32::from_le_bytes([c[8], c[9], c[10], c[11]]).clamp(0.0, 1.0) * 255.0) as u8;
                [r, g, b, 255]
            }).collect()
        }
        gltf::image::Format::R32G32B32A32FLOAT => {
            data.pixels.chunks(16).flat_map(|c| {
                let r = (f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(0.0, 1.0) * 255.0) as u8;
                let g = (f32::from_le_bytes([c[4], c[5], c[6], c[7]]).clamp(0.0, 1.0) * 255.0) as u8;
                let b = (f32::from_le_bytes([c[8], c[9], c[10], c[11]]).clamp(0.0, 1.0) * 255.0) as u8;
                let a = (f32::from_le_bytes([c[12], c[13], c[14], c[15]]).clamp(0.0, 1.0) * 255.0) as u8;
                [r, g, b, a]
            }).collect()
        }
    }
}

/// Load a glTF/GLB file from disk.
pub fn load_gltf(path: &Path) -> Result<GltfScene> {
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("Failed to import {}", path.display()))?;

    let file_name = path
        .file_stem()
        .map_or("scene".into(), |n| n.to_string_lossy().into_owned());

    // --- Load textures ---
    let textures: Vec<LoadedTexture> = images
        .iter()
        .map(|img| LoadedTexture {
            width: img.width,
            height: img.height,
            pixels: image_to_rgba8(img),
        })
        .collect();

    // --- Load materials ---
    // Map glTF material index → our material index
    let mut material_map: HashMap<usize, usize> = HashMap::new();
    let mut materials: Vec<LoadedMaterial> = Vec::new();

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let mat_idx = materials.len();
        if let Some(index) = material.index() {
            material_map.insert(index, mat_idx);
        }

        let base_color_texture = pbr.base_color_texture()
            .map(|info| info.texture().source().index());
        let metallic_roughness_texture = pbr.metallic_roughness_texture()
            .map(|info| info.texture().source().index());
        let normal_texture = material.normal_texture()
            .map(|info| info.texture().source().index());
        let occlusion_texture = material.occlusion_texture()
            .map(|info| info.texture().source().index());

        let normal_scale = material.normal_texture()
            .map(|info| info.scale())
            .unwrap_or(1.0);
        let occlusion_strength = material.occlusion_texture()
            .map(|info| info.strength())
            .unwrap_or(1.0);

        materials.push(LoadedMaterial {
            base_color_factor: pbr.base_color_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_factor: pbr.metallic_factor(),
            occlusion_strength,
            normal_scale,
            base_color_texture,
            metallic_roughness_texture,
            normal_texture,
            occlusion_texture,
        });
    }

    // --- Load meshes ---
    let mut mesh_map: HashMap<(usize, usize), usize> = HashMap::new();
    // Map (gltf_mesh_index, prim_index) → material index
    let mut mesh_material_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut meshes: Vec<LoadedMesh> = Vec::new();

    for mesh in document.meshes() {
        for (prim_idx, primitive) in mesh.primitives().enumerate() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .with_context(|| {
                    format!("Mesh '{}' primitive {} has no positions", mesh.name().unwrap_or("?"), prim_idx)
                })?
                .collect();

            let normals_raw: Option<Vec<[f32; 3]>> = reader.read_normals().map(|iter| iter.collect());
            let uvs_raw: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|iter| iter.into_f32().collect());
            let tangents_raw: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|iter| iter.collect());
            let indices: Option<Vec<u32>> = reader.read_indices().map(|iter| iter.into_u32().collect());

            // Track material for this mesh primitive
            if let Some(mat_index) = primitive.material().index() {
                if let Some(&our_mat_idx) = material_map.get(&mat_index) {
                    mesh_material_map.insert((mesh.index(), prim_idx), our_mat_idx);
                }
            }

            // Expand to flat triangle list
            let vert_count;
            let (flat_pos, flat_nor, flat_uv, flat_tan) = if let Some(ref idx) = indices {
                vert_count = idx.len();
                let mut pos = Vec::with_capacity(vert_count);
                let mut nor = Vec::with_capacity(vert_count);
                let mut uv = Vec::with_capacity(vert_count);
                let mut tan = Vec::with_capacity(vert_count);

                for &i in idx {
                    let ii = i as usize;
                    pos.push(positions[ii]);
                    if let Some(ref n) = normals_raw { nor.push(n[ii]); }
                    if let Some(ref u) = uvs_raw { uv.push(u[ii]); }
                    if let Some(ref t) = tangents_raw { tan.push(t[ii]); }
                }

                // Compute flat normals if missing
                if normals_raw.is_none() {
                    for tri in pos.chunks(3) {
                        if tri.len() < 3 { continue; }
                        let n = flat_normal(tri[0], tri[1], tri[2]);
                        nor.extend_from_slice(&[n, n, n]);
                    }
                }

                (pos, nor, uv, tan)
            } else {
                vert_count = positions.len();
                let nor = normals_raw.unwrap_or_else(|| {
                    let mut computed = Vec::with_capacity(vert_count);
                    for tri in positions.chunks(3) {
                        if tri.len() < 3 { continue; }
                        let n = flat_normal(tri[0], tri[1], tri[2]);
                        computed.extend_from_slice(&[n, n, n]);
                    }
                    computed
                });
                let uv = uvs_raw.unwrap_or_default();
                let tan = tangents_raw.unwrap_or_default();
                (positions, nor, uv, tan)
            };

            // Fill missing UVs with zeros
            let flat_uv = if flat_uv.len() < vert_count {
                let mut uv = flat_uv;
                uv.resize(vert_count, [0.0, 0.0]);
                uv
            } else {
                flat_uv
            };

            // Compute tangents from UVs if missing
            let flat_tan = if flat_tan.len() < vert_count {
                let mut tan = Vec::with_capacity(vert_count);
                for tri_idx in (0..vert_count).step_by(3) {
                    if tri_idx + 2 >= vert_count { break; }
                    let t = compute_tangent(
                        flat_pos[tri_idx], flat_pos[tri_idx + 1], flat_pos[tri_idx + 2],
                        flat_uv[tri_idx], flat_uv[tri_idx + 1], flat_uv[tri_idx + 2],
                        flat_nor[tri_idx],
                    );
                    tan.extend_from_slice(&[t, t, t]);
                }
                tan.resize(vert_count, [1.0, 0.0, 0.0, 1.0]);
                tan
            } else {
                flat_tan
            };

            // Convert to COLMAP coordinates
            let vertices: Vec<[f32; 3]> = flat_pos.iter().map(|p| yup_to_colmap(*p)).collect();
            let normals: Vec<[f32; 3]> = flat_nor.iter().map(|n| yup_to_colmap(*n)).collect();
            // Tangents: transform xyz by yup_to_colmap, keep w
            let tangents: Vec<[f32; 4]> = flat_tan.iter().map(|t| {
                let tc = yup_to_colmap([t[0], t[1], t[2]]);
                [tc[0], tc[1], tc[2], t[3]]
            }).collect();

            let mesh_id = meshes.len();
            mesh_map.insert((mesh.index(), prim_idx), mesh_id);
            meshes.push(LoadedMesh { vertices, normals, uvs: flat_uv, tangents });
        }
    }

    // --- Build node hierarchy ---
    let gltf_scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .with_context(|| "glTF has no scenes")?;

    let mut nodes: Vec<SceneNode> = Vec::new();
    let mut node_index_map: HashMap<usize, usize> = HashMap::new();

    // Insert root coordinate-conversion node (Y-up → COLMAP)
    {
        let mut root = SceneNode::new(format!("{}_root", file_name));
        root.local_transform.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        root.parent = None;
        root.visible = true;
        nodes.push(root);
    }
    let coord_root_idx = 0usize;

    struct QueueEntry {
        gltf_node_index: usize,
        parent_idx: usize,
    }

    let mut queue: std::collections::VecDeque<QueueEntry> = std::collections::VecDeque::new();
    for root_node in gltf_scene.nodes() {
        queue.push_back(QueueEntry { gltf_node_index: root_node.index(), parent_idx: coord_root_idx });
    }

    while let Some(entry) = queue.pop_front() {
        let gltf_node = document.nodes().nth(entry.gltf_node_index).expect("invalid node index");

        let our_idx = nodes.len();
        node_index_map.insert(gltf_node.index(), our_idx);

        let (translation, rotation, scale) = gltf_node.transform().decomposed();
        let local_transform = rmesh_interact::Transform {
            position: Vec3::from(translation),
            rotation: Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
            scale: Vec3::from(scale),
        };

        let gltf_mesh = gltf_node.mesh();
        let prim_count = gltf_mesh.as_ref().map_or(0, |m| m.primitives().count());

        let name = gltf_node.name().map(|s| s.to_string())
            .unwrap_or_else(|| format!("Node_{}", gltf_node.index()));

        /// Apply material properties from glTF to a SceneNode.
        fn apply_material(
            node: &mut SceneNode,
            material: &gltf::Material,
            material_map: &HashMap<usize, usize>,
        ) {
            let pbr = material.pbr_metallic_roughness();
            node.color = pbr.base_color_factor();
            node.roughness = pbr.roughness_factor();
            node.metallic = pbr.metallic_factor();
            if let Some(mat_idx) = material.index() {
                node.material_index = material_map.get(&mat_idx).copied();
            }
            if let Some(info) = material.normal_texture() {
                node.normal_scale = info.scale();
            }
            if let Some(info) = material.occlusion_texture() {
                node.occlusion_strength = info.strength();
            }
        }

        if prim_count <= 1 {
            let mut node = SceneNode::new(&name);
            node.parent = Some(entry.parent_idx);
            node.local_transform = local_transform;

            if let Some(ref mesh) = gltf_mesh {
                if let Some(prim) = mesh.primitives().next() {
                    if let Some(&mesh_id) = mesh_map.get(&(mesh.index(), 0)) {
                        node.mesh_kind = Some(PrimitiveKind::CustomMesh(mesh_id));
                    }
                    apply_material(&mut node, &prim.material(), &material_map);
                }
            }

            nodes[entry.parent_idx].children.push(our_idx);
            nodes.push(node);
        } else {
            let mut group = SceneNode::new(&name);
            group.parent = Some(entry.parent_idx);
            group.local_transform = local_transform;
            nodes[entry.parent_idx].children.push(our_idx);
            nodes.push(group);

            let mesh = gltf_mesh.as_ref().unwrap();
            for (pi, prim) in mesh.primitives().enumerate() {
                let child_idx = nodes.len();
                let mut child = SceneNode::new(format!("{}_{}", name, pi));
                child.parent = Some(our_idx);
                if let Some(&mesh_id) = mesh_map.get(&(mesh.index(), pi)) {
                    child.mesh_kind = Some(PrimitiveKind::CustomMesh(mesh_id));
                }
                apply_material(&mut child, &prim.material(), &material_map);
                nodes[our_idx].children.push(child_idx);
                nodes.push(child);
            }
        }

        for child in gltf_node.children() {
            queue.push_back(QueueEntry { gltf_node_index: child.index(), parent_idx: our_idx });
        }
    }

    // --- Load animations ---
    let mut clips: Vec<AnimationClip> = Vec::new();

    for anim in document.animations() {
        let anim_name = anim.name().map(|s| s.to_string())
            .unwrap_or_else(|| format!("Animation_{}", anim.index()));

        let mut channels: Vec<AnimationChannel> = Vec::new();
        let mut max_time = 0.0f32;

        for channel in anim.channels() {
            let target = channel.target();
            let gltf_node_idx = target.node().index();
            let our_node_idx = match node_index_map.get(&gltf_node_idx) {
                Some(&idx) => idx,
                None => continue,
            };

            let property = match target.property() {
                gltf::animation::Property::Translation => TargetProperty::Translation,
                gltf::animation::Property::Rotation => TargetProperty::Rotation,
                gltf::animation::Property::Scale => TargetProperty::Scale,
                gltf::animation::Property::MorphTargetWeights => continue,
            };

            let sampler = channel.sampler();
            let interpolation = match sampler.interpolation() {
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };

            let reader = channel.reader(|buf| Some(&buffers[buf.index()]));
            let times: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            if let Some(last) = times.last() { max_time = max_time.max(*last); }

            let values: Vec<f32> = match reader.read_outputs() {
                Some(outputs) => match outputs {
                    gltf::animation::util::ReadOutputs::Translations(iter) => iter.flat_map(|v| v).collect(),
                    gltf::animation::util::ReadOutputs::Rotations(iter) => iter.into_f32().flat_map(|v| v).collect(),
                    gltf::animation::util::ReadOutputs::Scales(iter) => iter.flat_map(|v| v).collect(),
                    gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => continue,
                },
                None => continue,
            };

            channels.push(AnimationChannel { target_node: our_node_idx, property, interpolation, times, values });
        }

        clips.push(AnimationClip { name: anim_name, duration: max_time, channels });
    }

    // --- Assemble scene ---
    let mut scene = AnimatedScene::new(&file_name);
    scene.nodes = nodes;
    scene.clips = clips;
    scene.playback = PlaybackState {
        clip_index: if scene.clips.is_empty() { None } else { Some(0) },
        time: 0.0,
        playing: !scene.clips.is_empty(),
        looping: true,
        speed: 1.0,
    };

    Ok(GltfScene { scene, meshes, textures, materials })
}
