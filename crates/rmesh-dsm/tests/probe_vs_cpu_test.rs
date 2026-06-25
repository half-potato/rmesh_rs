//! DSM light-probe vs CPU ray-traced transmittance comparison.
//!
//! Renders a 2-moment DSM cubemap for a point light at the origin around an
//! 8-tet scene, then compares the Cantelli reconstruction against exact CPU
//! ray-traced T(z) for every cubemap texel at several query depths. Writes
//! per-face PNGs to `target/test_output/dsm_probe/` so the bound's failure
//! pattern across the probe surface is visible.
//!
//! PNG channels per pixel: R = T_cpu, G = T_dsm, B = 4·|error|. Magenta-tinted
//! pixels mean CPU is brighter, cyan means DSM is brighter, bright blue means
//! large error.
//!
//! The test is purely diagnostic — it never asserts on the Cantelli output
//! itself, because the bound is conservative at every z (including z=far,
//! whenever `p_max > 1−α`). We do assert that the stored α matches the CPU's
//! `1 − T_total` along the same ray within FP/blend precision — that's a
//! rasterization sanity check, independent of reconstruction.

use glam::Vec3;
use half::f16;
use image::Rgb;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rmesh_compositor::PrimitiveGeometry;
use rmesh_data::SceneData;
use rmesh_dsm::{
    build_light_vp, generate_dsm_for_lights, DsmAtlas, DsmPipeline, DsmPrimitivePipeline,
};
use rmesh_render::{ComputeIntervalPipelines, GpuLight};
use rmesh_sort::{RadixSortPipelines, RadixSortState, SortBackend};
use rmesh_util::camera::TET_FACES;
use rmesh_util::test_util::{build_test_scene, create_test_device_default, random_tet_vertices};

/// Exact transmittance T(z_query) along a unit ray from `origin`.
/// Mirrors `crates/rmesh-render/tests/common/mod.rs::cpu_transmittance_along_ray`.
fn cpu_transmittance_along_ray(scene: &SceneData, origin: Vec3, dir: Vec3, z_query: f32) -> f32 {
    let n = scene.tet_count as usize;
    let mut absorbance = 0.0f32;

    for ti in 0..n {
        let i0 = scene.indices[ti * 4] as usize;
        let i1 = scene.indices[ti * 4 + 1] as usize;
        let i2 = scene.indices[ti * 4 + 2] as usize;
        let i3 = scene.indices[ti * 4 + 3] as usize;
        let verts = [
            Vec3::new(
                scene.vertices[i0 * 3],
                scene.vertices[i0 * 3 + 1],
                scene.vertices[i0 * 3 + 2],
            ),
            Vec3::new(
                scene.vertices[i1 * 3],
                scene.vertices[i1 * 3 + 1],
                scene.vertices[i1 * 3 + 2],
            ),
            Vec3::new(
                scene.vertices[i2 * 3],
                scene.vertices[i2 * 3 + 1],
                scene.vertices[i2 * 3 + 2],
            ),
            Vec3::new(
                scene.vertices[i3 * 3],
                scene.vertices[i3 * 3 + 1],
                scene.vertices[i3 * 3 + 2],
            ),
        ];
        let density = scene.densities[ti];

        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;
        let mut valid = true;

        for face in TET_FACES.iter() {
            let va = verts[face[0]];
            let vb = verts[face[1]];
            let vc = verts[face[2]];
            let v_opp = verts[face[3]];
            let mut nrm = (vc - va).cross(vb - va);
            if nrm.dot(v_opp - va) < 0.0 {
                nrm = -nrm;
            }
            let num = nrm.dot(va - origin);
            let den = nrm.dot(dir);
            if den.abs() < 1e-20 {
                if num > 0.0 {
                    valid = false;
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

        if !valid {
            continue;
        }

        let lo = t_min.max(0.0);
        let hi = t_max.min(z_query);
        if hi > lo {
            absorbance += density * (hi - lo);
        }
    }

    (-absorbance).exp()
}

const RES: u32 = 64;
const NEAR: f32 = 0.05;
const FAR: f32 = 4.0;

fn make_light() -> GpuLight {
    GpuLight {
        position: [0.0, 0.0, 0.0],
        light_type: 0,
        color: [1.0; 3],
        intensity: 1.0,
        direction: [0.0, 0.0, 1.0],
        inner_cos: 1.0,
        outer_cos: 1.0,
        _pad: [0.0; 3],
    }
}

fn build_scene(rng: &mut impl Rng) -> SceneData {
    // Densely overlapping tets in two concentric shells around the light so
    // most directions hit ≥2 absorbers at well-separated depths — that is
    // where Cantelli's variance term grows and the bound becomes loose.
    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut densities: Vec<f32> = Vec::new();
    let mut color_grads: Vec<f32> = Vec::new();

    let shells: [(f32, usize); 2] = [(0.8, 120), (1.8, 200)];
    for &(shell_r, n) in &shells {
        for _ in 0..n {
            // Uniform-on-sphere direction
            let z = rng.random::<f32>() * 2.0 - 1.0;
            let phi = rng.random::<f32>() * std::f32::consts::TAU;
            let r2d = (1.0_f32 - z * z).max(0.0).sqrt();
            let dir = Vec3::new(r2d * phi.cos(), r2d * phi.sin(), z);
            let jitter_r = shell_r * (0.85 + 0.30 * rng.random::<f32>());
            let center = dir * jitter_r;

            let radius = 0.18 + 0.10 * rng.random::<f32>();
            let (mut tv, ti) = random_tet_vertices(rng, radius);
            for k in 0..4 {
                tv[k * 3] += center.x;
                tv[k * 3 + 1] += center.y;
                tv[k * 3 + 2] += center.z;
            }
            let base = (vertices.len() / 3) as u32;
            vertices.extend_from_slice(&tv);
            for k in 0..4 {
                indices.push(base + ti[k]);
            }
            // Modest density: keep T_total well above 0 so the bound has room
            // to be wrong (a fully-opaque ray pegs both CPU and DSM at T≈0).
            densities.push(rng.random::<f32>() * 0.8 + 0.4);
            color_grads.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }

    build_test_scene(vertices, indices, densities, color_grads)
}

/// World-space ray direction for the (face, u, v) texel center on a light
/// cubemap face — matches the per-face rasterizer used by `build_light_vp`.
fn cubemap_dir(face: usize, u_norm: f32, v_norm: f32) -> Vec3 {
    let light = make_light();
    let (_vp, c2w) = build_light_vp(&light, face, NEAR, FAR);
    // Intrinsics fx = fy = cx = cy = res / 2 → camera-space ray
    // (2u-1, 2v-1, 1) (matches the dsm-render intrinsics in generate_dsm_for_lights).
    let cam_ray = Vec3::new(2.0 * u_norm - 1.0, 2.0 * v_norm - 1.0, 1.0);
    (c2w * cam_ray).normalize_or_zero()
}

/// Cantelli (one-sided Chebyshev) bound on transmittance — pure Rust port of
/// `deferred_shade_frag.wgsl::evaluate_transmittance` for a known texel.
fn cantelli_eval(m: [f32; 4], z_query: f32, near: f32, far: f32) -> f32 {
    let shadow_alpha = m[3];
    if shadow_alpha < 0.01 {
        return 1.0;
    }
    let inv_alpha = 1.0 / shadow_alpha;
    let mean = m[0] * inv_alpha;
    let z = ((z_query - near) / (far - near)).clamp(0.0, 1.0);
    if z <= mean {
        return 1.0;
    }
    let variance = ((m[1] - m[0] * m[0]) / shadow_alpha).max(3e-5);
    let d = z - mean;
    let p_max = variance / (variance + d * d);
    let t_total = 1.0 - shadow_alpha;
    p_max.max(t_total)
}

#[test]
fn probe_vs_cpu() {
    let Some((device, queue)) = create_test_device_default() else {
        eprintln!("Skipping probe_vs_cpu: no GPU adapter available");
        return;
    };

    let mut rng = ChaCha8Rng::seed_from_u64(0x70BE);
    let scene = build_scene(&mut rng);

    // --- Forward + interval + sort pipelines (mirrors what the viewer does) ---
    let color_format = wgpu::TextureFormat::Rgba16Float;
    let zero_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, _compute_bg, _render_bg) =
        rmesh_render::setup_forward(
            &device,
            &queue,
            &scene,
            &zero_colors,
            &scene.color_grads,
            RES,
            RES,
        );

    let ci_pipelines = ComputeIntervalPipelines::new(&device, color_format);
    let sort_pipelines = RadixSortPipelines::new(&device, 1, SortBackend::Drs);
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_state = RadixSortState::new(&device, n_pow2, 32, 1, SortBackend::Drs);
    sort_state.upload_configs(&queue);

    let dsm_pipeline = DsmPipeline::new(&device, color_format);
    let dsm_prim_pipeline = DsmPrimitivePipeline::new(&device);
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let prim_geometry = PrimitiveGeometry::new(&device);

    let lights = vec![make_light()];
    let atlas = DsmAtlas::new(&device, RES, &[0]);

    // --- Render DSM ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_test_dsm"),
    });
    generate_dsm_for_lights(
        &atlas,
        &mut encoder,
        &device,
        &queue,
        &dsm_pipeline,
        &dsm_prim_pipeline,
        &fwd_pipelines,
        &prim_geometry,
        &[],
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
        true,
    );
    queue.submit(std::iter::once(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    // --- Read back the 6-layer cubemap as Rgba16Float ---
    let bytes_per_texel: u32 = 8;
    let row_bytes = RES * bytes_per_texel;
    assert!(
        row_bytes.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
        "row_bytes ({row_bytes}) must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT"
    );
    let total_bytes = (row_bytes as u64) * RES as u64 * 6;

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dsm_readback"),
        size: total_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas.cubemaps[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(RES),
            },
        },
        wgpu::Extent3d {
            width: RES,
            height: RES,
            depth_or_array_layers: 6,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let half_data: &[u16] = bytemuck::cast_slice(&data);

    // Decode to 6 × (RES·RES) × [f32; 4]
    let face_texels = (RES * RES) as usize;
    let mut moments: Vec<Vec<[f32; 4]>> = Vec::with_capacity(6);
    for face in 0..6 {
        let mut face_data = vec![[0.0f32; 4]; face_texels];
        let face_offset = face * face_texels * 4;
        for i in 0..face_texels {
            for c in 0..4 {
                face_data[i][c] = f16::from_bits(half_data[face_offset + i * 4 + c]).to_f32();
            }
        }
        moments.push(face_data);
    }
    drop(data);
    readback_buf.unmap();

    // --- Per-texel comparison + PNG output ---
    let out_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test_output/dsm_probe");
    std::fs::create_dir_all(&out_dir).ok();

    let light_pos = Vec3::ZERO;
    let z_queries: [(f32, &str); 4] = [
        (0.25, "0.25"),
        (0.50, "0.50"),
        (0.75, "0.75"),
        (1.00, "far"),
    ];

    // Rasterization sanity: stored α should equal 1 − T_cpu(z=far) per texel,
    // independent of Cantelli. Loop once before the visualization to fail fast
    // if rasterization itself is broken.
    let mut alpha_max_err: f32 = 0.0;
    let mut alpha_mean_err = 0.0f64;
    let mut alpha_n = 0u64;
    for face in 0..6 {
        for v in 0..RES {
            for u in 0..RES {
                let s = (u as f32 + 0.5) / RES as f32;
                let t = (v as f32 + 0.5) / RES as f32;
                let dir = cubemap_dir(face, s, t);
                if dir.length_squared() < 0.5 {
                    continue;
                }
                let t_cpu = cpu_transmittance_along_ray(&scene, light_pos, dir, FAR);
                let alpha = moments[face][(v * RES + u) as usize][3];
                let err = ((1.0 - alpha) - t_cpu).abs();
                alpha_max_err = alpha_max_err.max(err);
                alpha_mean_err += err as f64;
                alpha_n += 1;
            }
        }
    }
    let alpha_mean_err = (alpha_mean_err / alpha_n as f64) as f32;
    eprintln!(
        "\n--- Rasterization sanity: stored α vs CPU (1 - T_total) ---\n\
         α mean error: {alpha_mean_err:.4}   α max error: {alpha_max_err:.4}"
    );

    eprintln!("\n--- DSM probe vs CPU error (Cantelli reconstruction) ---");
    for face in 0..6 {
        for &(zn, label) in &z_queries {
            let z_query = NEAR + zn * (FAR - NEAR);
            let mut img = image::ImageBuffer::<Rgb<u8>, _>::new(RES, RES);
            let mut errors = Vec::with_capacity(face_texels);

            for v in 0..RES {
                for u in 0..RES {
                    let s = (u as f32 + 0.5) / RES as f32;
                    let t = (v as f32 + 0.5) / RES as f32;
                    let dir = cubemap_dir(face, s, t);

                    let t_cpu = cpu_transmittance_along_ray(&scene, light_pos, dir, z_query);

                    let m = moments[face][(v * RES + u) as usize];
                    let t_dsm = cantelli_eval(m, z_query, NEAR, FAR);

                    let err = (t_cpu - t_dsm).abs();
                    errors.push(err);

                    let r = (t_cpu.clamp(0.0, 1.0) * 255.0) as u8;
                    let g = (t_dsm.clamp(0.0, 1.0) * 255.0) as u8;
                    let b = ((err * 4.0).min(1.0) * 255.0) as u8;
                    img.put_pixel(u, v, Rgb([r, g, b]));
                }
            }

            let path = out_dir.join(format!("face{face}_z{label}.png"));
            img.save(&path).expect("PNG write failed");

            let n = errors.len() as f32;
            let mean = errors.iter().sum::<f32>() / n;
            let max = errors.iter().cloned().fold(0.0f32, f32::max);
            eprintln!(
                "face {face} z={label:>4}: mean={mean:.4} max={max:.4} → {}",
                path.file_name().unwrap().to_string_lossy()
            );

            let _ = (zn, max);
        }
    }

    assert!(
        alpha_mean_err < 0.05,
        "α-channel rasterization sanity check failed: mean error {alpha_mean_err:.4} > 0.05 \
         (stored α should match CPU (1 - T_total) within FP/blend precision)"
    );
}
