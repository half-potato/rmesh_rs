//! Cubemap (DSM light-map) face-orientation test.
//!
//! Runs the real GPU DSM pipeline (`generate_dsm_for_lights`) over a small,
//! deliberately asymmetric scene, reads back each cubemap face, and compares it
//! against an independent CPU ray-traced ground truth.
//!
//! Why this catches a "rotated face": the deferred shader samples the cubemap by
//! direction — `cube[select_cubemap_face(dir)][uv(dir)]` — using the fixed WebGPU
//! major-axis convention. So the texel `(u,v)` of face `f` is read for the
//! direction `d(f, u, v)` defined by that convention. The render side fills that
//! texel via `build_light_vp(face f)` (the function under test). This test renders
//! the scene on the CPU with the **canonical** per-face camera derived from the
//! sampling convention (NOT from `build_light_vp`), so it is the ground truth: if
//! face `f`'s `build_light_vp` orientation disagrees (rotated / flipped /
//! transposed), the GPU face and the CPU face diverge and the per-face assertion
//! fails by name.
//!
//! The scene places one tet near each of the 8 cube corners, offset outward from
//! the light at the origin, each with a **distinct size and radius**. That breaks
//! the rotational / mirror symmetry of a face's four corners — without it, an
//! identical-corner scene would be invariant under a 90°/180°/flip and the bug
//! would go undetected.
//!
//! GPU-gated: if no adapter is available the test prints a skip notice and returns.

use glam::{Mat3, Mat4, Vec3, Vec4};
use rmesh_util::camera::perspective_matrix;
use rmesh_util::test_util::{build_test_scene, cpu_render_scene, TestDeviceConfig};
use std::f32::consts::FRAC_PI_2;

const RES: u32 = 128;
const NEAR: f32 = 0.1;
const FAR: f32 = 15.0;

// Face order: +X, -X, +Y, -Y, +Z, -Z (matches CUBEMAP_DIRS / select_cubemap_face).
const FACE_NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

/// Canonical per-face camera basis `(S, T, M)` derived from the WebGPU cube-map
/// sampling convention (the sc / tc / major-axis table documented in
/// `rmesh-dsm/src/lib.rs`):
///
/// ```text
/// +X: sc=-z tc=-y   -X: sc=+z tc=-y
/// +Y: sc=+x tc=+z   -Y: sc=+x tc=-z
/// +Z: sc=+x tc=-y   -Z: sc=-x tc=-y
/// ```
///
/// `S` is the world axis along which texture `u` increases, `T` the axis along
/// which `v` increases, and `M` the (signed) major / forward axis. With camera
/// intrinsics `cx=cy=fx=fy=RES/2`, pixel `(px,py)` builds the camera-space ray
/// `(2u-1, 2v-1, 1)`, so `c2w = [S | T | M]` maps it to
/// `M + (2u-1)·S + (2v-1)·T`, which is exactly the convention's direction for
/// texel `(u,v)` of that face. **This table is the ground truth and must not be
/// taken from `build_light_vp` / `CUBEMAP_DIRS`.**
fn canonical_face_basis(face: usize) -> (Vec3, Vec3, Vec3) {
    let x = Vec3::X;
    let y = Vec3::Y;
    let z = Vec3::Z;
    match face {
        0 => (-z, -y, x),  // +X: sc=-z, tc=-y, ma=+x
        1 => (z, -y, -x),  // -X: sc=+z, tc=-y, ma=-x
        2 => (x, z, y),    // +Y: sc=+x, tc=+z, ma=+y
        3 => (x, -z, -y),  // -Y: sc=+x, tc=-z, ma=-y
        4 => (x, -y, z),   // +Z: sc=+x, tc=-y, ma=+z
        5 => (-x, -y, -z), // -Z: sc=-x, tc=-y, ma=-z
        _ => unreachable!(),
    }
}

/// Build a regular tetrahedron of the given full-extent `size` centered at
/// `center`, returned as four world-space vertices with a winding (caller uses
/// indices `[0,1,3,2]`) that yields positive orientation.
fn make_tet(center: Vec3, size: f32) -> [Vec3; 4] {
    let h = size * 0.5;
    [
        center + Vec3::new(1.0, 1.0, 1.0) * h,
        center + Vec3::new(1.0, -1.0, -1.0) * h,
        center + Vec3::new(-1.0, 1.0, -1.0) * h,
        center + Vec3::new(-1.0, -1.0, 1.0) * h,
    ]
}

/// 8 corner tets, offset outward from the light at the origin, each with a
/// distinct size and radius (asymmetric so a rotated face is detectable).
fn corner_scene() -> rmesh_data::SceneData {
    let signs = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ];

    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut densities: Vec<f32> = Vec::new();
    let mut color_grads: Vec<f32> = Vec::new();

    for (i, &(sx, sy, sz)) in signs.iter().enumerate() {
        let dir = Vec3::new(sx, sy, sz).normalize();
        let radius = 4.0 + 0.5 * i as f32; // distinct radius: 4.0 .. 7.5
        let size = 0.6 + 0.3 * i as f32; // distinct size:   0.6 .. 2.7
        let verts = make_tet(dir * radius, size);

        let base = (i * 4) as u32;
        for v in verts {
            vertices.extend_from_slice(&[v.x, v.y, v.z]);
        }
        // [0,1,3,2] winding → positive orientation for this base tet.
        indices.extend_from_slice(&[base, base + 1, base + 3, base + 2]);
        densities.push(5.0); // high → near-opaque coverage
        color_grads.extend_from_slice(&[0.0, 0.0, 0.0]);
    }

    build_test_scene(vertices, indices, densities, color_grads)
}

/// Read back one layer (`face`) of an `Rgba16Float` cube texture as RGBA f32.
fn read_cube_face(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    face: u32,
    res: u32,
) -> Vec<[f32; 4]> {
    let bytes_per_pixel = 8u32; // Rgba16Float
    let unpadded = res * bytes_per_pixel;
    let aligned = (unpadded + 255) & !255;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dsm_cube_readback"),
        size: (aligned * res) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dsm_cube_copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: face },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(res),
            },
        },
        wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let mut out = vec![[0.0f32; 4]; (res * res) as usize];
    for row in 0..res {
        let row_start = (row * aligned) as usize;
        for col in 0..res {
            let p = row_start + (col * bytes_per_pixel) as usize;
            let r = half::f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
            let g = half::f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
            let b = half::f16::from_le_bytes([data[p + 4], data[p + 5]]).to_f32();
            let a = half::f16::from_le_bytes([data[p + 6], data[p + 7]]).to_f32();
            out[(row * res + col) as usize] = [r, g, b, a];
        }
    }
    drop(data);
    readback.unmap();
    out
}

/// Generate the DSM cubemap for the corner scene and read back all 6 faces'
/// alpha (coverage) channels. Returns `None` if no GPU adapter is available.
fn gpu_cubemap_coverage(scene: &rmesh_data::SceneData) -> Option<[Vec<f32>; 6]> {
    let (device, queue) = rmesh_util::test_util::create_test_device(TestDeviceConfig {
        backends: None, // all backends (CI may not have Vulkan/Metal)
        extra_features: wgpu::Features::empty(),
        base_limits: wgpu::Limits::default(),
    })?;

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];

    // SceneBuffers (incl. interval buffers) + MaterialBuffers + ForwardPipelines.
    let (buffers, material, fwd_pipelines, _targets, _compute_bg, _render_bg) =
        rmesh_render::setup_forward(
            &device,
            &queue,
            scene,
            &zero_base_colors,
            &scene.color_grads,
            RES,
            RES,
        );

    let ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, color_format);

    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    // Dummy SH buffer (project runs at sh_degree 0 → coefficients unused).
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let dsm_pipeline = rmesh_dsm::DsmPipeline::new(&device, color_format);
    let dsm_prim_pipeline = rmesh_dsm::DsmPrimitivePipeline::new(&device);
    let prim_geometry = rmesh_compositor::PrimitiveGeometry::new(&device);
    let atlas = rmesh_dsm::DsmAtlas::new(&device, RES, &[0]); // one point light

    let light = rmesh_render::GpuLight {
        position: [0.0, 0.0, 0.0],
        light_type: 0, // point
        ..Default::default()
    };
    let lights = [light];

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dsm_generate"),
    });
    rmesh_dsm::generate_dsm_for_lights(
        &atlas,
        &mut encoder,
        &device,
        &queue,
        &dsm_pipeline,
        &dsm_prim_pipeline,
        &prim_geometry,
        &[], // no opaque primitives
        &fwd_pipelines,
        &ci_pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &material,
        &dummy_sh,
        &lights,
        1,
        scene.tet_count,
        NEAR,
        FAR,
        true, // render tets
    );
    // generate_dsm_for_lights submits its internal sort encoders but leaves the
    // final face's render + copy commands in `encoder`.
    queue.submit(std::iter::once(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let coverage: [Vec<f32>; 6] = std::array::from_fn(|f| {
        read_cube_face(&device, &queue, &atlas.cubemaps[0], f as u32, RES)
            .into_iter()
            .map(|px| px[3])
            .collect()
    });
    Some(coverage)
}

/// CPU ground-truth coverage for `face`, rendered with the canonical camera.
fn cpu_face_coverage(scene: &rmesh_data::SceneData, face: usize) -> Vec<f32> {
    let (s, t, m) = canonical_face_basis(face);
    let half = RES as f32 / 2.0;
    let intrinsics = [half, half, half, half];
    let c2w = Mat3::from_cols(s, t, m);

    // view maps world → (S·v, T·v, M·v) = (right, up, forward); used only for
    // frustum culling in cpu_render_scene (ray dirs come from c2w + intrinsics).
    let view = Mat4::from_cols(
        Vec4::new(s.x, t.x, m.x, 0.0),
        Vec4::new(s.y, t.y, m.y, 0.0),
        Vec4::new(s.z, t.z, m.z, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );
    let vp = perspective_matrix(FRAC_PI_2, 1.0, NEAR, FAR) * view;

    cpu_render_scene(scene, Vec3::ZERO, vp, c2w, intrinsics, RES, RES)
        .into_iter()
        .map(|px| px[3])
        .collect()
}

/// Intersection-over-union of two coverage masks (alpha > `tau`), ignoring a
/// 1-texel border. Returns `(iou, gpu_covered, cpu_covered)`.
fn coverage_iou(gpu: &[f32], cpu: &[f32], tau: f32) -> (f32, usize, usize) {
    let mut inter = 0usize;
    let mut union = 0usize;
    let mut gpu_cov = 0usize;
    let mut cpu_cov = 0usize;
    for row in 1..(RES - 1) {
        for col in 1..(RES - 1) {
            let i = (row * RES + col) as usize;
            let g = gpu[i] > tau;
            let c = cpu[i] > tau;
            if g {
                gpu_cov += 1;
            }
            if c {
                cpu_cov += 1;
            }
            if g && c {
                inter += 1;
            }
            if g || c {
                union += 1;
            }
        }
    }
    let iou = if union == 0 { 1.0 } else { inter as f32 / union as f32 };
    (iou, gpu_cov, cpu_cov)
}

#[test]
fn point_light_cubemap_face_orientation() {
    let scene = corner_scene();

    let gpu_cov = match gpu_cubemap_coverage(&scene) {
        Some(c) => c,
        None => {
            eprintln!("point_light_cubemap_face_orientation: no GPU adapter — skipping");
            return;
        }
    };

    const TAU: f32 = 0.3; // coverage threshold (high density ⇒ solid blobs)
    const MIN_IOU: f32 = 0.5; // tolerant of f16 + raster/analytic edge noise
    const MIN_COVER: usize = ((RES * RES) as usize) / 400; // ≥0.25% must be covered

    let mut failures: Vec<String> = Vec::new();
    for face in 0..6 {
        let cpu_cov = cpu_face_coverage(&scene, face);
        let (iou, g_cov, c_cov) = coverage_iou(&gpu_cov[face], &cpu_cov, TAU);

        eprintln!(
            "face {} ({}): IoU={:.3}  gpu_covered={}  cpu_covered={}",
            face, FACE_NAMES[face], iou, g_cov, c_cov
        );

        // Sanity: the canonical ground truth must actually see content on this
        // face, otherwise the comparison is vacuous.
        assert!(
            c_cov >= MIN_COVER,
            "face {} ({}): CPU ground truth has almost no coverage ({} texels) — \
             scene geometry does not illuminate this face; test cannot validate it",
            face,
            FACE_NAMES[face],
            c_cov
        );

        if iou < MIN_IOU {
            failures.push(format!(
                "face {} ({}): coverage IoU {:.3} < {:.2} \
                 (gpu_covered={}, cpu_covered={}) — render side (build_light_vp / \
                 CUBEMAP_DIRS) disagrees with the cube-map sampling convention; \
                 this face appears rotated/flipped",
                face, FACE_NAMES[face], iou, MIN_IOU, g_cov, c_cov
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "cubemap face orientation mismatch:\n{}",
        failures.join("\n")
    );
}
