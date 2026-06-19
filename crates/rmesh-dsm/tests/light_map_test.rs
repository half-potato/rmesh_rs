//! Integration tests for point-light deep-shadow-map ("light map") generation.
//!
//! These verify that the per-light cubemap atlas produced by
//! [`rmesh_dsm::generate_dsm_for_lights`] is computed correctly for both
//! **solid meshes** (opaque primitives — cube/sphere/plane/cylinder) and
//! **tetrahedral volumes**.
//!
//! What the light map stores (one RGBA16F moment texture per face, see
//! `dsm_moment_fragment.wgsl` / `dsm_primitive.wgsl`):
//!   * `.r` = α·E[z]    — expected (premultiplied) termination depth
//!   * `.g` = α·E[z²]   — second depth moment (premultiplied)
//!   * `.b` = unused
//!   * `.a` = α         — accumulated occlusion (1 - T)
//! where depth is the linear view-space distance normalised to `[0,1]` over
//! `[near, far]`.
//!
//! Predictions exercised here:
//!   * Solid mesh    → α == 1 (full occlusion), depth == surface distance.
//!   * Tet volume    → 0 < α < 1, α == 1 - exp(-density · path_length).
//!   * Solid in front of a tet → solid wins (α == 1, depth == solid's).
//!   * Empty region / wrong cubemap face → α == 0 (full transmittance).
//!
//! GPU tests gracefully skip (return early) when no adapter is available.

use glam::Vec3;
use rmesh_compositor::PrimitiveGeometry;
use rmesh_data::SceneData;
use rmesh_dsm::{
    build_light_vp, generate_dsm_for_lights, DsmAtlas, DsmPipeline, DsmPrimitivePipeline,
    DsmProjectPipeline,
};
use rmesh_interact::{Primitive, PrimitiveKind};
use rmesh_render::GpuLight;
use rmesh_util::camera::TET_FACES;
use rmesh_util::test_util::build_test_scene;

const NEAR: f32 = 0.05;
const FAR: f32 = 20.0;
const RES: u32 = 128;

/// Forward (look-to) directions of the 6 cubemap faces, matching the private
/// `CUBEMAP_DIRS` table in `rmesh_dsm`: +X, -X, +Y, -Y, +Z, -Z.
const FACE_FORWARD: [Vec3; 6] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];

fn point_light(position: Vec3) -> GpuLight {
    GpuLight {
        position: position.into(),
        light_type: 0, // point
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        direction: [0.0, 0.0, -1.0],
        inner_cos: 1.0,
        outer_cos: 1.0,
        _pad: [0.0; 3],
    }
}

/// A non-degenerate (regular) tetrahedron centred at `center`, half-size `s`.
fn single_tet_scene(center: Vec3, s: f32, density: f32) -> SceneData {
    // Four alternating cube corners → regular tetrahedron.
    let base = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    let mut verts = Vec::with_capacity(12);
    for b in base {
        let p = center + b * s;
        verts.extend_from_slice(&[p.x, p.y, p.z]);
    }
    build_test_scene(verts, vec![0, 1, 2, 3], vec![density], vec![0.0; 3])
}

fn tet_world_verts(verts: &[f32]) -> [Vec3; 4] {
    std::array::from_fn(|i| Vec3::new(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]))
}

/// Analytic opacity along a single ray through the tet, mirroring the CPU
/// reference / WGSL volume integral. `dir` must be a unit vector.
///
/// Returns `(alpha, t_enter, t_exit)` or `None` if the ray misses the tet.
fn ray_tet_alpha(
    origin: Vec3,
    dir: Vec3,
    verts: &[Vec3; 4],
    density: f32,
) -> Option<(f32, f32, f32)> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for face in &TET_FACES {
        let va = verts[face[0]];
        let vb = verts[face[1]];
        let vc = verts[face[2]];
        let v_opp = verts[face[3]];
        let mut n = (vc - va).cross(vb - va);
        if n.dot(v_opp - va) < 0.0 {
            n = -n; // point inward
        }
        let num = n.dot(va - origin);
        let den = n.dot(dir);
        if den.abs() < 1e-20 {
            if num > 0.0 {
                return None; // ray parallel and outside this face
            }
            continue;
        }
        let t = num / den;
        if den > 0.0 {
            t_min = t_min.max(t);
        } else {
            t_max = t_max.min(t);
        }
    }
    if t_max <= t_min {
        return None;
    }
    let path = t_max - t_min;
    let alpha = 1.0 - (-density * path).exp();
    Some((alpha, t_min, t_max))
}

// ---------------------------------------------------------------------------
// CPU-only: cubemap light-map projection matrices
// ---------------------------------------------------------------------------

/// `build_light_vp` must give each cubemap face a frustum looking down its own
/// axis: a point on that axis projects to screen-centre and in front of the
/// camera, while the opposite axis is behind it. This validates the geometry
/// of the point-light light map without needing a GPU.
#[test]
fn point_light_cubemap_vp_matrices() {
    let light = point_light(Vec3::ZERO);

    for face in 0..6 {
        let (vp, _c2w) = build_light_vp(&light, face, NEAR, FAR);
        let fwd = FACE_FORWARD[face];

        // A point straight ahead on this face's axis → centre, visible, valid depth.
        let p = fwd * 3.0;
        let clip = vp * p.extend(1.0);
        assert!(
            clip.w > 0.0,
            "face {face}: on-axis point should be in front of the light (w={})",
            clip.w
        );
        let ndc = clip.truncate() / clip.w;
        assert!(
            ndc.x.abs() < 1e-3 && ndc.y.abs() < 1e-3,
            "face {face}: on-axis point should land at screen centre, got ndc=({}, {})",
            ndc.x,
            ndc.y
        );
        assert!(
            ndc.z > 0.0 && ndc.z < 1.0,
            "face {face}: on-axis depth should be inside [0,1], got {}",
            ndc.z
        );

        // The opposite axis must be behind this face's camera.
        let behind = vp * (-fwd * 3.0).extend(1.0);
        assert!(
            behind.w <= 0.0,
            "face {face}: opposite-axis point should be behind the light (w={})",
            behind.w
        );
    }
}

// ---------------------------------------------------------------------------
// GPU harness
// ---------------------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn try_gpu() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu { device, queue })
    })
}

/// Run the full DSM generation for one point light and return the populated atlas.
fn generate(
    gpu: &Gpu,
    scene: &SceneData,
    primitives: &[Primitive],
    light: GpuLight,
    render_tets: bool,
) -> DsmAtlas {
    let device = &gpu.device;
    let queue = &gpu.queue;

    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, _fwd_pipelines, _targets, _cbg, _rbg) = rmesh_render::setup_forward(
        device,
        queue,
        scene,
        &base_colors,
        &scene.color_grads,
        RES,
        RES,
    );

    let ci_pipelines =
        rmesh_render::ComputeIntervalPipelines::new(device, wgpu::TextureFormat::Rgba16Float);

    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(queue);

    let dsm_pipeline = DsmPipeline::new(device, wgpu::TextureFormat::Rgba16Float);
    let dsm_prim_pipeline = DsmPrimitivePipeline::new(device);
    let dsm_project_pipeline = DsmProjectPipeline::new(device);
    let prim_geometry = PrimitiveGeometry::new(device);

    let atlas = DsmAtlas::new(device, RES, &[light.light_type]);
    let lights = [light];

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dsm_test"),
    });
    generate_dsm_for_lights(
        &atlas,
        &mut encoder,
        device,
        queue,
        &dsm_pipeline,
        &dsm_prim_pipeline,
        &dsm_project_pipeline,
        &prim_geometry,
        primitives,
        &ci_pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &material,
        &lights,
        lights.len() as u32,
        scene.tet_count,
        NEAR,
        FAR,
        render_tets,
    );
    // Flush the final face's work (generate leaves it un-submitted) into the cubemaps.
    queue.submit(std::iter::once(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    atlas
}

/// Read one cubemap face (light `li`, face layer `face`) back to the CPU as
/// `[r, g, b, a]` pixels in row-major order.
fn read_face(gpu: &Gpu, atlas: &DsmAtlas, li: usize, face: u32) -> Vec<[f32; 4]> {
    let device = &gpu.device;
    let queue = &gpu.queue;

    let bytes_per_pixel = 8u32; // Rgba16Float
    let bytes_per_row = RES * bytes_per_pixel;
    let aligned_bytes_per_row = (bytes_per_row + 255) & !255;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dsm_face_readback"),
        size: (aligned_bytes_per_row * RES) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dsm_face_copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas.cubemaps[li],
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row),
                rows_per_image: Some(RES),
            },
        },
        wgpu::Extent3d {
            width: RES,
            height: RES,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().expect("map failed");

    let data = slice.get_mapped_range();
    let mut image = vec![[0.0f32; 4]; (RES * RES) as usize];
    for row in 0..RES {
        let row_start = (row * aligned_bytes_per_row) as usize;
        for col in 0..RES {
            let p = row_start + (col * bytes_per_pixel) as usize;
            let r = half::f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
            let g = half::f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
            let b = half::f16::from_le_bytes([data[p + 4], data[p + 5]]).to_f32();
            let a = half::f16::from_le_bytes([data[p + 6], data[p + 7]]).to_f32();
            image[(row * RES + col) as usize] = [r, g, b, a];
        }
    }
    drop(data);
    readback.unmap();
    image
}

fn at(image: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    image[(y * RES + x) as usize]
}

fn center(image: &[[f32; 4]]) -> [f32; 4] {
    at(image, RES / 2, RES / 2)
}

fn normalize_depth(view_z: f32) -> f32 {
    (view_z - NEAR) / (FAR - NEAR)
}

// ---------------------------------------------------------------------------
// GPU: solid mesh light map
// ---------------------------------------------------------------------------

/// A solid cube at a known distance must occlude fully (α == 1) and store the
/// linear depth of its near surface in the correct cubemap face.
#[test]
fn solid_mesh_light_map_depth_and_occlusion() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU adapter — skipping solid_mesh_light_map_depth_and_occlusion");
        return;
    };

    // Light at origin; unit cube centred 3 units along +X → face 0 (+X).
    let light = point_light(Vec3::ZERO);
    let dist = 3.0f32;
    let mut cube = Primitive::new(PrimitiveKind::Cube, "occluder");
    cube.transform.position = Vec3::new(dist, 0.0, 0.0);
    let primitives = [cube];

    // A dummy far-away tet keeps the scene valid; render_tets=false skips it.
    let scene = single_tet_scene(Vec3::new(0.0, 0.0, -100.0), 0.1, 1.0);
    let atlas = generate(&gpu, &scene, &primitives, light, false);

    let face = read_face(&gpu, &atlas, 0, 0); // light 0, face +X

    // Near face of the unit cube ([-0.5,0.5]^3) sits at dist - 0.5.
    let expected_depth = normalize_depth(dist - 0.5);

    let covered: Vec<&[f32; 4]> = face.iter().filter(|p| p[3] > 0.5).collect();
    assert!(
        !covered.is_empty(),
        "cube should cover some pixels of the +X face"
    );
    for p in &covered {
        assert!(
            (p[3] - 1.0).abs() < 0.02,
            "solid mesh must be fully opaque (alpha≈1), got {}",
            p[3]
        );
        // .r stores α·E[z]; α==1 → depth directly.
        assert!(
            (p[0] - expected_depth).abs() < 0.03,
            "stored depth {} should match cube near-face depth {}",
            p[0],
            expected_depth
        );
        // .g stores α·E[z²]; for an opaque surface E[z²] == E[z]².
        assert!(
            (p[1] - p[0] * p[0]).abs() < 0.02,
            "second moment {} should equal depth² {} for an opaque surface",
            p[1],
            p[0] * p[0]
        );
    }

    // The cube sits on +X only — the -X face must be empty.
    let opposite = read_face(&gpu, &atlas, 0, 1);
    let max_alpha = opposite.iter().map(|p| p[3]).fold(0.0f32, f32::max);
    assert!(
        max_alpha < 0.01,
        "opposite cubemap face should be empty, max alpha was {max_alpha}"
    );
}

// ---------------------------------------------------------------------------
// GPU: tetrahedral volume light map
// ---------------------------------------------------------------------------

/// A single translucent tet must produce partial occlusion matching the
/// analytic volume integral `α = 1 - exp(-density·path)` along the central ray.
#[test]
fn tet_volume_light_map_partial_occlusion() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU adapter — skipping tet_volume_light_map_partial_occlusion");
        return;
    };

    let light = point_light(Vec3::ZERO);
    let center_pos = Vec3::new(0.0, 0.0, -3.0); // along -Z → face 5
    let density = 1.5f32;
    let scene = single_tet_scene(center_pos, 0.7, density);
    let verts = tet_world_verts(&scene.vertices);

    // Analytic prediction for the central ray (light origin, looking -Z).
    let (alpha_exp, t_enter, t_exit) =
        ray_tet_alpha(Vec3::ZERO, Vec3::NEG_Z, &verts, density).expect("central ray must hit tet");
    assert!(
        alpha_exp > 0.05 && alpha_exp < 0.95,
        "test tet should be partially transparent, analytic alpha={alpha_exp}"
    );

    let atlas = generate(&gpu, &scene, &[], light, true);
    let face = read_face(&gpu, &atlas, 0, 5); // light 0, face -Z
    let c = center(&face);

    // Partial occlusion, matching the analytic integral.
    assert!(
        c[3] > 0.01 && c[3] < 0.99,
        "tet must yield partial occlusion 0<alpha<1, got {}",
        c[3]
    );
    assert!(
        (c[3] - alpha_exp).abs() < 0.06,
        "central alpha {} should match analytic {} (1-exp(-density*path))",
        c[3],
        alpha_exp
    );

    // Expected termination depth E[z] = .r / alpha must fall inside the tet's
    // depth span along the central ray.
    let e_depth = c[0] / c[3];
    let lo = normalize_depth(t_enter);
    let hi = normalize_depth(t_exit);
    assert!(
        e_depth > lo - 0.02 && e_depth < hi + 0.02,
        "expected depth {} should lie within tet span [{}, {}]",
        e_depth,
        lo,
        hi
    );

    // Second moment .g = α·E[z²]: variance must be non-negative, i.e.
    // E[z²] ≥ E[z]². (For a volume — unlike an opaque surface — it is > 0.)
    let e_depth_sq = c[1] / c[3];
    assert!(
        e_depth_sq >= e_depth * e_depth - 1e-3,
        "second moment E[z²]={} must be ≥ E[z]²={}",
        e_depth_sq,
        e_depth * e_depth
    );

    // A corner pixel sees empty space → full transmittance (alpha 0).
    let corner = at(&face, 2, 2);
    assert!(
        corner[3] < 0.01,
        "empty region should have alpha≈0, got {}",
        corner[3]
    );

    // The tet is on -Z only; the +X face must be empty.
    let other = read_face(&gpu, &atlas, 0, 0);
    let max_alpha = other.iter().map(|p| p[3]).fold(0.0f32, f32::max);
    assert!(
        max_alpha < 0.01,
        "tet should not leak onto other cubemap faces, max alpha {max_alpha}"
    );
}

/// Increasing density must increase the stored occlusion (monotonicity of the
/// volume integral in the light map).
#[test]
fn tet_light_map_density_monotonic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU adapter — skipping tet_light_map_density_monotonic");
        return;
    };

    let light = point_light(Vec3::ZERO);
    let center_pos = Vec3::new(0.0, 0.0, -3.0);

    let scene_lo = single_tet_scene(center_pos, 0.7, 1.0);
    let scene_hi = single_tet_scene(center_pos, 0.7, 5.0);

    let alpha_lo = center(&read_face(
        &gpu,
        &generate(&gpu, &scene_lo, &[], light, true),
        0,
        5,
    ))[3];
    let alpha_hi = center(&read_face(
        &gpu,
        &generate(&gpu, &scene_hi, &[], light, true),
        0,
        5,
    ))[3];

    assert!(
        alpha_hi > alpha_lo + 0.05,
        "higher density should occlude more: lo={alpha_lo}, hi={alpha_hi}"
    );
    assert!(
        alpha_hi <= 1.01 && alpha_lo >= -0.01,
        "alpha must stay in [0,1]"
    );
}

// ---------------------------------------------------------------------------
// GPU: solid mesh + tet interaction
// ---------------------------------------------------------------------------

/// A solid mesh between the light and a tet must win the depth test: the light
/// map shows full occlusion at the solid's depth, not the tet behind it.
#[test]
fn solid_mesh_occludes_tet_behind_it() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU adapter — skipping solid_mesh_occludes_tet_behind_it");
        return;
    };

    let light = point_light(Vec3::ZERO);
    // Tet far down -Z; cube between it and the light.
    let scene = single_tet_scene(Vec3::new(0.0, 0.0, -3.0), 0.7, 1.5);

    let cube_dist = 1.5f32; // near face at 1.0, well in front of the tet (~2.3+)
    let mut cube = Primitive::new(PrimitiveKind::Cube, "blocker");
    cube.transform.position = Vec3::new(0.0, 0.0, -cube_dist);
    let primitives = [cube];

    let atlas = generate(&gpu, &scene, &primitives, light, true);
    let face = read_face(&gpu, &atlas, 0, 5); // face -Z
    let c = center(&face);

    // Solid occluder in front → full occlusion at the cube's depth.
    assert!(
        (c[3] - 1.0).abs() < 0.02,
        "solid occluder in front should give alpha≈1, got {}",
        c[3]
    );
    let cube_depth = normalize_depth(cube_dist - 0.5);
    let stored_depth = c[0] / c[3];
    assert!(
        (stored_depth - cube_depth).abs() < 0.03,
        "depth should be the cube's ({cube_depth}), not the tet behind it; got {stored_depth}"
    );
}

// ---------------------------------------------------------------------------
// GPU: end-to-end transmittance reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct transmittance T(z) from a moment pixel via the Cantelli
/// (one-sided Chebyshev) bound — a Rust mirror of `evaluate_transmittance` in
/// deferred_shade_frag.wgsl and `dsm_resolve.wgsl`. Constants must match.
fn reconstruct_transmittance(m: [f32; 4], z_norm: f32) -> f32 {
    const MIN_VARIANCE: f32 = 4.0e-4;
    const LBR: f32 = 0.15;
    let alpha = m[3];
    if alpha < 0.01 {
        return 1.0;
    }
    let inv_alpha = 1.0 / alpha;
    let mean = m[0] * inv_alpha; // E[z]
    let e_z2 = m[1] * inv_alpha; // E[z²]
    if z_norm <= mean {
        return 1.0;
    }
    let variance = (e_z2 - mean * mean).max(MIN_VARIANCE);
    let d = z_norm - mean;
    let p_max = variance / (variance + d * d);
    let p_lbr = ((p_max - LBR) / (1.0 - LBR)).clamp(0.0, 1.0); // light-bleeding reduction
    p_lbr.max(1.0 - alpha)
}

/// The stored moments must be usable as a shadow: a receiver in front of the
/// tet reads fully lit (T≈1), one behind reads attenuated (T<1) but never
/// darker than the `1 − α` translucency floor.
#[test]
fn tet_transmittance_reconstruction() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU adapter — skipping tet_transmittance_reconstruction");
        return;
    };

    let light = point_light(Vec3::ZERO);
    let scene = single_tet_scene(Vec3::new(0.0, 0.0, -3.0), 0.7, 2.0);
    let verts = tet_world_verts(&scene.vertices);
    let (_alpha, t_enter, t_exit) =
        ray_tet_alpha(Vec3::ZERO, Vec3::NEG_Z, &verts, 2.0).expect("central ray must hit tet");

    let atlas = generate(&gpu, &scene, &[], light, true);
    let m = center(&read_face(&gpu, &atlas, 0, 5));
    let alpha = m[3];
    assert!(
        alpha > 0.05 && alpha < 0.95,
        "expected a partial occluder, got α={alpha}"
    );

    // Receiver in front of the absorber → fully lit.
    let t_front = reconstruct_transmittance(m, normalize_depth(t_enter - 0.5));
    assert!(
        (t_front - 1.0).abs() < 1e-3,
        "receiver in front of the tet should be fully lit, got T={t_front}"
    );

    // Receiver well behind the absorber → shadowed, but not below the 1−α floor.
    let t_behind = reconstruct_transmittance(m, normalize_depth(t_exit + 5.0));
    assert!(
        t_behind < 0.999,
        "receiver behind the tet should be shadowed, got T={t_behind}"
    );
    assert!(
        t_behind >= (1.0 - alpha) - 1e-3 && t_behind <= 1.0,
        "transmittance {t_behind} must respect the 1−α floor ({}) and stay ≤ 1",
        1.0 - alpha
    );
}
