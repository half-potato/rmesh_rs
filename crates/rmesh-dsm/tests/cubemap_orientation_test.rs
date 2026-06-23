//! Cubemap (DSM light-map) all-face orientation probe — one off-axis tet per cone.
//!
//! Renders the real GPU DSM pipeline (`generate_dsm_for_lights`) for a scene
//! with one near-opaque tet placed off-center inside each of the six face cones,
//! reads back all six cube layers, and checks WHERE each tet's coverage blob
//! lands against the fixed WebGPU major-axis sampling convention (u=col, v=row,
//! top=0):
//!
//! ```text
//! +X: u=0.5*(-z/|x|+1) v=0.5*(-y/|x|+1)   -X: u=0.5*(+z/|x|+1) v=0.5*(-y/|x|+1)
//! +Y: u=0.5*(+x/|y|+1) v=0.5*(+z/|y|+1)   -Y: u=0.5*(+x/|y|+1) v=0.5*(-z/|y|+1)
//! +Z: u=0.5*(+x/|z|+1) v=0.5*(-y/|z|+1)   -Z: u=0.5*(-x/|z|+1) v=0.5*(-y/|z|+1)
//! ```
//!
//! Each tet sits in exactly one cone, so it only projects onto its own face;
//! the other faces' VPs put it behind the camera (culled). If a face's GPU write
//! disagrees with the convention (rotated/flipped — the suspected build_light_vp
//! / CUBEMAP_DIRS bug), that face's blob lands in a different quadrant and the
//! assertion names the face.
//!
//! GPU-gated: prints a skip notice and returns if no adapter is available.

use glam::Vec3;
use rmesh_util::test_util::build_test_scene;

const RES: u32 = 128;
const NEAR: f32 = 0.1;
const FAR: f32 = 15.0;

const FACE_NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

/// One off-axis tet center per face cone (dominant axis ±6, distinct tangential
/// offsets so every face's expected (u,v) is a different, asymmetric quadrant).
const CENTERS: [Vec3; 6] = [
    Vec3::new(6.0, 2.0, 1.0),   // +X
    Vec3::new(-6.0, 2.0, 1.0),  // -X
    Vec3::new(1.0, 6.0, 2.0),   // +Y
    Vec3::new(1.0, -6.0, 2.0),  // -Y
    Vec3::new(2.0, 1.0, 6.0),   // +Z
    Vec3::new(2.0, 1.0, -6.0),  // -Z
];

/// Convention-predicted (u, v) for `center` on cube `face` (u=col/RES, v=row/RES,
/// top=0). Mirrors `select_cubemap_face` + the sc/tc table in `rmesh-dsm`.
fn expected_uv(face: usize, c: Vec3) -> (f32, f32) {
    let (sc, tc, ma) = match face {
        0 => (-c.z, -c.y, c.x.abs()), // +X
        1 => (c.z, -c.y, c.x.abs()),  // -X
        2 => (c.x, c.z, c.y.abs()),   // +Y
        3 => (c.x, -c.z, c.y.abs()),  // -Y
        4 => (c.x, -c.y, c.z.abs()),  // +Z
        5 => (-c.x, -c.y, c.z.abs()), // -Z
        _ => unreachable!(),
    };
    (0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0))
}

/// Build a regular tetrahedron of full-extent `size` centered at `center`.
fn make_tet(center: Vec3, size: f32) -> [Vec3; 4] {
    let h = size * 0.5;
    [
        center + Vec3::new(1.0, 1.0, 1.0) * h,
        center + Vec3::new(1.0, -1.0, -1.0) * h,
        center + Vec3::new(-1.0, 1.0, -1.0) * h,
        center + Vec3::new(-1.0, -1.0, 1.0) * h,
    ]
}

/// Six near-opaque tets, one centered in each face cone.
fn six_cone_scene(size: f32) -> rmesh_data::SceneData {
    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut densities: Vec<f32> = Vec::new();
    let mut color_grads: Vec<f32> = Vec::new();
    for (i, &center) in CENTERS.iter().enumerate() {
        let verts = make_tet(center, size);
        let base = (i * 4) as u32;
        for v in verts {
            vertices.extend_from_slice(&[v.x, v.y, v.z]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 3, base + 2]); // +orientation
        densities.push(5.0);
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

/// Generate the DSM cubemap for `scene` and read back all six faces' alpha.
/// Returns `None` if no GPU adapter is available.
fn gpu_all_faces_coverage(scene: &rmesh_data::SceneData) -> Option<[Vec<f32>; 6]> {
    let (device, queue) =
        rmesh_util::test_util::create_test_device(rmesh_util::test_util::TestDeviceConfig {
            backends: None,
            extra_features: wgpu::Features::empty(),
            base_limits: wgpu::Limits::default(),
        })?;

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];

    let (buffers, material, _fwd_pipelines, _targets, _compute_bg, _render_bg) =
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
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(&device, n_pow2.max(1), 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    let dsm_pipeline = rmesh_dsm::DsmPipeline::new(&device, color_format);
    let dsm_prim_pipeline = rmesh_dsm::DsmPrimitivePipeline::new(&device);
    let dsm_project_pipeline = rmesh_dsm::DsmProjectPipeline::new(&device);
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
        &dsm_project_pipeline,
        &prim_geometry,
        &[], // no opaque primitives
        &ci_pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &material,
        &lights,
        1,
        scene.tet_count,
        NEAR,
        FAR,
        true, // render tets
    );
    queue.submit(std::iter::once(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    Some(std::array::from_fn(|f| {
        read_cube_face(&device, &queue, &atlas.cubemaps[0], f as u32, RES)
            .into_iter()
            .map(|px| px[3])
            .collect()
    }))
}

/// Centroid (col, row) of coverage > `tau`, plus covered-pixel count.
fn coverage_centroid(cov: &[f32], tau: f32) -> (f32, f32, usize) {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut n = 0usize;
    for row in 0..RES {
        for col in 0..RES {
            if cov[(row * RES + col) as usize] > tau {
                sx += col as f32;
                sy += row as f32;
                n += 1;
            }
        }
    }
    if n == 0 {
        (f32::NAN, f32::NAN, 0)
    } else {
        (sx / n as f32, sy / n as f32, n)
    }
}

#[test]
fn point_light_cubemap_all_face_orientation() {
    let scene = six_cone_scene(2.0);

    let cov = match gpu_all_faces_coverage(&scene) {
        Some(c) => c,
        None => {
            eprintln!("point_light_cubemap_all_face_orientation: no GPU adapter — skipping");
            return;
        }
    };

    const TAU: f32 = 0.3;
    let min_px = ((RES * RES) as usize) / 800;
    let tol = RES as f32 * 0.30; // quadrant-level: which half, not sub-pixel
    let mut failures: Vec<String> = Vec::new();

    for face in 0..6 {
        let c = CENTERS[face];
        let (cx, cy, n) = coverage_centroid(&cov[face], TAU);
        let (u, v) = expected_uv(face, c);
        let (col_e, row_e) = (u * RES as f32, v * RES as f32);

        eprintln!(
            "face {} ({}): expected (col,row)=({:.1},{:.1})  actual=({:.1},{:.1})  n={}",
            face, FACE_NAMES[face], col_e, row_e, cx, cy, n
        );

        if n < min_px {
            failures.push(format!(
                "face {} ({}): blob too small ({} px) — tet did not render on this face",
                face, FACE_NAMES[face], n
            ));
            continue;
        }
        if (cx - col_e).abs() >= tol || (cy - row_e).abs() >= tol {
            failures.push(format!(
                "face {} ({}): blob at (col={:.1},row={:.1}) but convention predicts \
                 (col={:.1},row={:.1}) — GPU write disagrees with the cube sampling \
                 convention (rotated/flipped face)",
                face, FACE_NAMES[face], cx, cy, col_e, row_e
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "cubemap face orientation mismatch:\n{}",
        failures.join("\n")
    );
}
