//! Per-kernel unit tests for the forward rendering pipeline.
//!
//! Tests each shader/kernel in isolation:
//!   - project_compute.wgsl: color eval, cull, depth key generation
//!   - radix sort (5-pass): tet depth sorting
//!   - rasterize_compute.wgsl: subgroup-based tiled forward rendering
//!   - tex_to_buffer.wgsl: texture → storage buffer conversion
//!
//! Each test creates a GPU device with SUBGROUP feature, dispatches a single
//! kernel, reads back the result, and validates correctness.
//!
//! Run: `cargo test -p rmesh-render --test kernel_tests -- --nocapture`

mod common;

use common::*;
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rmesh_util::test_util::{create_test_device_default as create_test_device, read_buffer};

const SEED: u64 = 42424242;
const W: u32 = 16;
const H: u32 = 16;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat3, [f32; 4]) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) =
        test_camera_c2w_intrinsics(eye, target, std::f32::consts::FRAC_PI_2, W as f32, H as f32);
    (vp, c2w, intrinsics)
}

// ---------------------------------------------------------------------------
// Test: project_compute.wgsl (color eval + cull + depth keys)
// ---------------------------------------------------------------------------

/// Dispatches only the forward compute kernel (no sort, no render).
/// Verifies that the compute pass:
///   - Sets instance_count > 0 in indirect args (tet is visible)
///   - Does not crash or produce GPU errors
#[test]
fn test_project_compute_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_project_compute_kernel (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let buffers = rmesh_render::SceneBuffers::upload(&device, &queue, &scene);
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = rmesh_render::MaterialBuffers::upload(
        &device,
        &zero_base_colors,
        &scene.color_grads,
        scene.tet_count,
    );
    let pipelines = rmesh_render::ForwardPipelines::new(&device, rmesh_render::GBufferFormats::default());
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let (_shell_hiz_stub, shell_hiz_stub_view) = rmesh_render::dummy_shell_hiz(&device, &queue);
    let compute_bg = rmesh_render::create_compute_bind_group(
        &device,
        &pipelines,
        &buffers,
        &material,
        &dummy_sh,
        &shell_hiz_stub_view,
    );

    // Camera looking at tet from outside
    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0,
        12,
        0.0,
        0,
        0.01,
        1000.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Reset indirect args
    let reset_cmd = rmesh_render::DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // Dispatch compute only
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.compute_pipeline);
        pass.set_bind_group(0, &compute_bg.bg0, &[]);
        pass.set_bind_group(1, &compute_bg.bg1, &[]);
        let n_pow2 = scene.tet_count.next_power_of_two();
        pass.dispatch_workgroups(n_pow2.div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // Read back indirect args to verify instance_count > 0
    let args: Vec<u32> = read_buffer(&device, &queue, &buffers.indirect_args, 4);
    let instance_count = args[1];
    eprintln!("project_compute: instance_count = {instance_count}");
    assert!(
        instance_count > 0,
        "Tet should be visible but instance_count=0"
    );
}

/// Equivalent of `test_project_compute_kernel` but for the HW-compute project
/// path (`project_compute_hw.wgsl`). After hoisting the per-view cull out, the
/// shader publishes `instance_count = uniforms.tet_count` from thread 0;
/// this test makes sure that write actually lands.
#[test]
fn test_project_compute_hw_kernel() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_project_compute_hw_kernel (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let buffers = rmesh_render::SceneBuffers::upload(&device, &queue, &scene);
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = rmesh_render::MaterialBuffers::upload(
        &device,
        &zero_base_colors,
        &scene.color_grads,
        scene.tet_count,
    );
    let pipelines = rmesh_render::ForwardPipelines::new(&device, rmesh_render::GBufferFormats::default());
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let (_shell_hiz_stub, shell_hiz_stub_view) = rmesh_render::dummy_shell_hiz(&device, &queue);
    let hw_bg = rmesh_render::create_hw_compute_bind_group(
        &device,
        &pipelines,
        &buffers,
        &material,
        &dummy_sh,
        &shell_hiz_stub_view,
    );

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0,
        12,
        0.0,
        0,
        0.01,
        1000.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let reset_cmd = rmesh_render::DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute_hw"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.hw_compute_pipeline);
        pass.set_bind_group(0, &hw_bg.bg, &[]);
        let n_pow2 = scene.tet_count.next_power_of_two();
        pass.dispatch_workgroups(n_pow2.div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let args: Vec<u32> = read_buffer(&device, &queue, &buffers.indirect_args, 4);
    eprintln!(
        "project_compute_hw: vertex_count={}, instance_count={}",
        args[0], args[1]
    );
    assert_eq!(
        args[1], scene.tet_count,
        "project_compute_hw should publish instance_count = tet_count ({}), got {}",
        scene.tet_count, args[1]
    );

    // The projection compacts: slots [0, instance_count) must hold each visible
    // tet exactly once, with nothing written past that. Every tet is visible in
    // this scene, so the surviving slots are a permutation of all tet ids.
    let vis = args[1] as usize;
    let sort_vals: Vec<u32> = read_buffer(&device, &queue, &buffers.sort_values, vis);
    let mut seen = vec![false; scene.tet_count as usize];
    for (slot, &tet) in sort_vals.iter().enumerate() {
        assert!(
            (tet as usize) < seen.len(),
            "slot {slot} holds tet id {tet}, out of range for {} tets",
            scene.tet_count
        );
        assert!(
            !std::mem::replace(&mut seen[tet as usize], true),
            "tet {tet} claimed more than one compacted slot"
        );
    }
    assert!(
        seen.iter().all(|&s| s),
        "every tet is visible here, so all should appear in the compacted range"
    );
}

// ---------------------------------------------------------------------------
// Test: rasterize_compute.wgsl (subgroup shader compilation)
// ---------------------------------------------------------------------------

/// Validates that the forward tiled compute pipeline (which requires
/// `enable subgroups;` in WGSL) can be compiled successfully.
///
/// This is the primary test for verifying that wgpu's SUBGROUP feature
/// is properly hooked up. If subgroups aren't supported, pipeline
/// creation will fail.
#[test]
fn test_rasterize_pipeline_creation() {
    let (device, _queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_rasterize_pipeline_creation (no GPU)");
            return;
        }
    };

    // This will panic if the shader fails to compile (e.g., subgroups not supported)
    let pipeline = rmesh_trainable::RasterizeComputePipeline::new(&device, W, H, 0);

    // Verify the output buffer was created with correct size
    assert_eq!(pipeline.width, W);
    assert_eq!(pipeline.height, H);
    eprintln!("rasterize_compute.wgsl compiled successfully with subgroup support");
}

// ---------------------------------------------------------------------------
// Test: full forward pass (compute + sort + render)
// ---------------------------------------------------------------------------

/// Validates that the interval shading pipelines (mesh shader + fragment)
/// can be compiled successfully.
///
/// Requires `EXPERIMENTAL_MESH_SHADER` feature. Skips if not available.
#[test]
fn test_interval_pipeline_creation() {
    let (device, _queue) = match pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        if !adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
        {
            return None;
        }

        let supported = adapter.limits();
        let mut limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 20,
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 30,
            ..wgpu::Limits::default()
        };
        limits.max_mesh_invocations_per_workgroup = supported.max_mesh_invocations_per_workgroup;
        limits.max_mesh_invocations_per_dimension = supported.max_mesh_invocations_per_dimension;
        limits.max_mesh_output_vertices = supported.max_mesh_output_vertices;
        limits.max_mesh_output_primitives = supported.max_mesh_output_primitives;
        limits.max_mesh_output_layers = supported.max_mesh_output_layers;
        limits.max_mesh_multiview_view_count = supported.max_mesh_multiview_view_count;
        limits.max_task_mesh_workgroup_total_count = supported.max_task_mesh_workgroup_total_count;
        limits.max_task_mesh_workgroups_per_dimension =
            supported.max_task_mesh_workgroups_per_dimension;

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::EXPERIMENTAL_MESH_SHADER,
                required_limits: limits,
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .ok()
    }) {
        Some(dq) => dq,
        None => {
            eprintln!(
                "Skipping test_interval_pipeline_creation (no GPU or no mesh shader support)"
            );
            return;
        }
    };

    let _interval = rmesh_render::IntervalPipelines::new(&device, rmesh_render::GBufferFormats::default());
    eprintln!("interval_mesh.wgsl + interval_fragment.wgsl compiled successfully");
}

/// End-to-end interval shading pass: compute → sort → interval mesh render.
/// Compares GPU output against CPU reference.
#[test]
fn test_interval_forward_e2e() {
    const E2E_W: u32 = 64;
    const E2E_H: u32 = 64;

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);

    let aspect = E2E_W as f32 / E2E_H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, centroid, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) = test_camera_c2w_intrinsics(
        eye,
        centroid,
        std::f32::consts::FRAC_PI_2,
        E2E_W as f32,
        E2E_H as f32,
    );

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, E2E_W, E2E_H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) =
        gpu_interval_render_scene(&scene, eye, vp, c2w, intrinsics, E2E_W, E2E_H)
    {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("interval_forward_e2e: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(mean_diff < 0.1, "mean_diff {mean_diff} >= 0.1");
    } else {
        eprintln!("Skipping GPU comparison (no adapter or no mesh shader support)");
    }
}

/// Compute-interval pipeline creation test (no mesh shader needed).
#[test]
fn test_compute_interval_pipeline_creation() {
    let (device, _queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_compute_interval_pipeline_creation (no GPU)");
            return;
        }
    };

    let _ci = rmesh_render::ComputeIntervalPipelines::new(&device, rmesh_render::GBufferFormats::default());
    eprintln!("interval_compute.wgsl + interval_vertex.wgsl + interval_fragment.wgsl compiled successfully");
}

/// End-to-end compute-interval shading pass (no mesh shader needed).
/// Compares GPU output against CPU reference.
#[test]
fn test_compute_interval_forward_e2e() {
    const E2E_W: u32 = 64;
    const E2E_H: u32 = 64;

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);

    let aspect = E2E_W as f32 / E2E_H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, centroid, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) = test_camera_c2w_intrinsics(
        eye,
        centroid,
        std::f32::consts::FRAC_PI_2,
        E2E_W as f32,
        E2E_H as f32,
    );

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, E2E_W, E2E_H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) =
        gpu_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, E2E_W, E2E_H)
    {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("compute_interval_forward_e2e: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(mean_diff < 0.1, "mean_diff {mean_diff} >= 0.1");
    } else {
        eprintln!("Skipping GPU comparison (no adapter)");
    }
}

/// End-to-end forward pass: compute → hardware rasterize.
/// Verifies the legacy (non-tiled) pipeline produces output matching CPU.
#[test]
fn test_legacy_forward_e2e() {
    // Use larger resolution (64×64) so the tet covers enough pixels to be visible.
    // 16×16 can miss pixel centers for small tets at distance.
    const E2E_W: u32 = 64;
    const E2E_H: u32 = 64;

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);

    let aspect = E2E_W as f32 / E2E_H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, centroid, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) = test_camera_c2w_intrinsics(
        eye,
        centroid,
        std::f32::consts::FRAC_PI_2,
        E2E_W as f32,
        E2E_H as f32,
    );

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, E2E_W, E2E_H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, c2w, intrinsics, E2E_W, E2E_H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("legacy_forward_e2e: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(mean_diff < 0.1, "mean_diff {mean_diff} >= 0.1");
    } else {
        eprintln!("Skipping GPU comparison (no adapter)");
    }
}

// ===========================================================================
// Density-shell z-cull
// ===========================================================================

/// Build a max-Hi-Z-shaped pyramid whose every texel holds `depth`.
///
/// The real one comes from `record_shell_hiz` over a rendered shell; a constant
/// stands in fine here, because `shell_occluded` only ever reads a max over some
/// footprint and a constant pyramid makes that max exactly `depth` at every mip.
fn constant_shell_hiz(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    depth: f32,
) -> wgpu::TextureView {
    let mip_count = 32 - width.max(height).leading_zeros();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_shell_hiz"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for mip in 0..mip_count {
        let w = (width >> mip).max(1);
        let h = (height >> mip).max(1);
        let data = vec![depth; (w * h) as usize];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Run `project_compute` against a constant shell bound and report how many tets
/// survived into the sort.
fn project_visible_count_with_shell(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &rmesh_data::SceneData,
    shell_depth: f32,
    cull_enabled: bool,
) -> u32 {
    let buffers = rmesh_render::SceneBuffers::upload(device, queue, scene);
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let material = rmesh_render::MaterialBuffers::upload(
        device,
        &base_colors,
        &scene.color_grads,
        scene.tet_count,
    );
    let pipelines = rmesh_render::ForwardPipelines::new(device, rmesh_render::GBufferFormats::default());
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let hiz_view = constant_shell_hiz(device, queue, W, H, shell_depth);
    let compute_bg = rmesh_render::create_compute_bind_group(
        device,
        &pipelines,
        &buffers,
        &material,
        &dummy_sh,
        &hiz_view,
    );

    let verts = load_tet_verts(scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    let mut uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0,
        12,
        0.0,
        0,
        0.01,
        1000.0,
    );
    uniforms.shell_cull_flags = u32::from(cull_enabled);
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let reset_cmd = rmesh_render::DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute_shell_cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.compute_pipeline);
        pass.set_bind_group(0, &compute_bg.bg0, &[]);
        pass.set_bind_group(1, &compute_bg.bg1, &[]);
        let n_pow2 = scene.tet_count.next_power_of_two();
        pass.dispatch_workgroups(n_pow2.div_ceil(64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    read_buffer::<u32>(device, queue, &buffers.indirect_args, 4)[1]
}

/// A shell in front of the tet must remove it from the visible set — and
/// crucially from `instance_count`, which is what the radix sort is sized by.
/// Culling only at the fragment level would leave this count unchanged.
#[test]
fn test_shell_cull_removes_occluded_tets() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_shell_cull_removes_occluded_tets (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    // Far plane everywhere: nothing occludes, so this is the baseline.
    let baseline = project_visible_count_with_shell(&device, &queue, &scene, 1.0, true);
    assert!(
        baseline > 0,
        "tet should be visible with no occluder (baseline={baseline})"
    );

    // Shell hard against the near plane: everything is behind it.
    let occluded = project_visible_count_with_shell(&device, &queue, &scene, 0.0, true);
    assert_eq!(
        occluded, 0,
        "tet behind the shell should never reach the sort"
    );

    // Same bound, cull switched off — proves the drop above came from the cull
    // and not from some incidental side effect of the pyramid being bound.
    let disabled = project_visible_count_with_shell(&device, &queue, &scene, 0.0, false);
    assert_eq!(
        disabled, baseline,
        "disabling the cull must restore the baseline exactly"
    );
}

/// The pyramid is cleared to 1.0 where nothing was drawn, so a footprint that
/// touches only background must never cull. This is the property that keeps the
/// cull conservative rather than merely approximate.
#[test]
fn test_shell_cull_is_inert_against_background() {
    let (device, queue) = match create_test_device() {
        Some(dq) => dq,
        None => {
            eprintln!("Skipping test_shell_cull_is_inert_against_background (no GPU)");
            return;
        }
    };

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let with_cull = project_visible_count_with_shell(&device, &queue, &scene, 1.0, true);
    let without_cull = project_visible_count_with_shell(&device, &queue, &scene, 1.0, false);
    assert_eq!(
        with_cull, without_cull,
        "a cleared (far-plane) pyramid must cull nothing"
    );
}

/// Compiles `shell.wgsl` and `shell_hiz_max.wgsl` and builds every pipeline
/// that consumes them.
///
/// Nothing else in the suite touches these shaders: the cull tests drive
/// `project_compute*` against a synthetic Hi-Z texture, so a WGSL error in the
/// shell itself would only surface when the viewer ran. Both shell entry points
/// (`fs_main` for the G-buffer, `fs_dsm_moment` for the shadow atlas) are
/// validated here because the DSM one is reached only after a light exists.
#[test]
fn test_shell_pipeline_creation() {
    let Some((device, queue)) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        // Adapter limits, not `Limits::default()`. The shell binds 9 storage
        // buffers per stage and the default cap is 8; both viewers request
        // adapter limits (native raises it to 20, web takes what the adapter
        // reports — 10 on Chrome), so this matches the real envelope.
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .ok()
    }) else {
        eprintln!("Skipping test_shell_pipeline_creation (no GPU)");
        return;
    };

    // Panics on a naga validation error, which is the point of the test.
    let _shell = rmesh_render::ShellPipeline::new(
        &device,
        rmesh_render::GBufferFormats::default(),
        wgpu::TextureFormat::Rgba16Float, // DSM moment target
        wgpu::TextureFormat::Depth32Float,
    );
    let hiz_pipelines = rmesh_render::ShellHiZPipelines::new(&device);

    // The pyramid needs a depth texture carrying TEXTURE_BINDING. The DSM's
    // staging depth lacked it and only blew up once a light forced the atlas to
    // allocate, so build a real one here rather than trusting the flags.
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shell_pipeline_test_depth"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let hiz = rmesh_render::ShellHiZ::new(&device, &hiz_pipelines, &depth_view, 64, 64);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    rmesh_render::record_shell_hiz(&mut encoder, &hiz_pipelines, &hiz);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    assert_eq!(hiz.mip_count, 7, "64x64 should give mips 64..1");
}
