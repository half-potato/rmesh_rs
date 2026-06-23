//! Trainable pipeline benchmark: compute-based tiled forward + loss + backward.
//!
//! This is the path `rmesh-train` and `rmesh-python` actually exercise. Measures
//! the three stages individually and end-to-end at ~10M tets, 1920x1080.
//!
//! Requires SUBGROUP + TIMESTAMP_QUERY. Skips gracefully otherwise.
//!
//! Run: `cargo bench -p rmesh-trainable --bench trainable_pipeline`

use criterion::{criterion_group, criterion_main, Criterion};
use glam::{Mat3, Mat4, Vec3};
use rmesh_train::{create_loss_bind_group, record_loss_pass, LossBuffers, LossPipeline};
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::shared::{LossUniforms, TileUniforms};
use rmesh_util::test_util::{grid_tet_scene, print_timestamp_table, TimestampRecorder};

const W: u32 = 1920;
const H: u32 = 1080;
const TILE_SIZE: u32 = 8;
// 126^3 cells × 5 tets/cell ≈ 10M tets.
const GRID_SIZE: u32 = 126;

fn setup_camera() -> (Mat4, Mat3, [f32; 4], Vec3) {
    let fov_y = std::f32::consts::FRAC_PI_4;
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(fov_y, aspect, 0.01, 100.0);
    let eye = Vec3::new(0.5, 0.5, 3.0);
    let target = Vec3::new(0.5, 0.5, 0.5);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let view = look_at(eye, target, up);
    let vp = proj * view;
    let f = (target - eye).normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);
    let c2w = Mat3::from_cols(r, -u, f);
    let f_val = 1.0 / (fov_y / 2.0).tan();
    let intrinsics = [
        f_val * H as f32 / 2.0,
        f_val * H as f32 / 2.0,
        W as f32 / 2.0,
        H as f32 / 2.0,
    ];
    (vp, c2w, intrinsics, eye)
}

#[allow(dead_code)]
struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tet_count: u32,
    // Forward project + scene state
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    compute_bg: rmesh_render::ForwardComputeBindGroups,
    // Tile + sort + scan
    tile_pipelines: rmesh_trainable::TilePipelines,
    radix_pipelines: rmesh_trainable::RadixSortPipelines,
    tile_buffers: rmesh_trainable::TileBuffers,
    radix_state: rmesh_trainable::RadixSortState,
    scan_pipelines: rmesh_trainable::ScanPipelines,
    scan_buffers: rmesh_trainable::ScanBuffers,
    prepare_dispatch_bg: wgpu::BindGroup,
    rts_bg: wgpu::BindGroup,
    tile_fill_bg: wgpu::BindGroup,
    tile_gen_scan_bg: wgpu::BindGroup,
    tile_ranges_bg_a: wgpu::BindGroup,
    tile_ranges_bg_b: wgpu::BindGroup,
    // Tiled compute forward
    rasterize: rmesh_trainable::RasterizeComputePipeline,
    rasterize_bg_a: wgpu::BindGroup,
    rasterize_bg_b: wgpu::BindGroup,
    // Loss
    loss_pipeline: LossPipeline,
    loss_buffers: LossBuffers,
    loss_bg: wgpu::BindGroup,
    // Backward tiled
    bwd_pipelines: rmesh_trainable::BackwardTiledPipelines,
    grad_buffers: rmesh_trainable::GradientBuffers,
    mat_grad_buffers: rmesh_trainable::MaterialGradBuffers,
    bwd_bg0_a: wgpu::BindGroup,
    bwd_bg0_b: wgpu::BindGroup,
    bwd_bg1: wgpu::BindGroup,
}

fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

        let supported_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 20,
            ..supported_limits
        };

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC
                    | wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()
    })
}

fn create_state() -> Option<State> {
    let (device, queue) = create_device()?;

    eprintln!("Generating grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, c2w, intrinsics, eye) = setup_camera();

    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward(
            &device,
            &queue,
            &scene,
            &base_colors,
            &scene.color_grads,
            W,
            H,
        );

    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0u32,
        TILE_SIZE,
        0.0,
        0,
        0.01,
        100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Tile / scan / sort infrastructure.
    let tile_pipelines = rmesh_trainable::TilePipelines::new(&device);
    let radix_pipelines =
        rmesh_trainable::RadixSortPipelines::new(&device, 2, rmesh_trainable::SortBackend::Drs);
    let tile_buffers = rmesh_trainable::TileBuffers::new(&device, scene.tet_count, W, H, TILE_SIZE);
    let sorting_bits = rmesh_trainable::sorting_bits_for_tiles(
        tile_buffers.num_tiles,
        rmesh_trainable::SortBackend::Drs,
    );
    let radix_state = rmesh_trainable::RadixSortState::new(
        &device,
        tile_buffers.max_pairs_pow2,
        sorting_bits,
        2,
        rmesh_trainable::SortBackend::Drs,
    );
    radix_state.upload_configs(&queue);

    let scan_pipelines = rmesh_trainable::ScanPipelines::new(&device);
    let scan_buffers = rmesh_trainable::ScanBuffers::new(&device, scene.tet_count);

    let tile_uni = TileUniforms {
        screen_width: W,
        screen_height: H,
        tile_size: TILE_SIZE,
        tiles_x: tile_buffers.tiles_x,
        tiles_y: tile_buffers.tiles_y,
        num_tiles: tile_buffers.num_tiles,
        visible_tet_count: 0,
        _pad: [0; 5],
    };
    queue.write_buffer(
        &tile_buffers.tile_uniforms,
        0,
        bytemuck::bytes_of(&tile_uni),
    );

    let prepare_dispatch_bg = rmesh_trainable::create_prepare_dispatch_bind_group(
        &device,
        &scan_pipelines,
        &buffers.indirect_args,
        &scan_buffers,
    );
    let rts_bg = rmesh_trainable::create_rts_bind_group(
        &device,
        &scan_pipelines,
        &buffers.tiles_touched,
        &scan_buffers,
    );
    let tile_fill_bg =
        rmesh_trainable::create_tile_fill_bind_group(&device, &tile_pipelines, &tile_buffers);
    let tile_gen_scan_bg = rmesh_trainable::create_tile_gen_scan_bind_group(
        &device,
        &scan_pipelines,
        &tile_buffers,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &buffers.compact_tet_ids,
        &buffers.circumdata,
        &buffers.tiles_touched,
        &scan_buffers,
        radix_state.num_keys_buf(),
    );

    let tile_ranges_bg_a = rmesh_trainable::create_tile_ranges_bind_group_with_keys(
        &device,
        &tile_pipelines,
        &tile_buffers.tile_sort_keys,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        radix_state.num_keys_buf(),
    );
    let tile_ranges_bg_b = rmesh_trainable::create_tile_ranges_bind_group_with_keys(
        &device,
        &tile_pipelines,
        radix_state.keys_b(),
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
        radix_state.num_keys_buf(),
    );

    // Tiled compute forward.
    let rasterize = rmesh_trainable::RasterizeComputePipeline::new(&device, W, H, 0);
    let rasterize_bg_a = rmesh_trainable::create_rasterize_bind_group(
        &device,
        &rasterize,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &material.colors,
        &buffers.densities,
        &material.color_grads,
        &tile_buffers.tile_sort_values,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );
    let rasterize_bg_b = rmesh_trainable::create_rasterize_bind_group(
        &device,
        &rasterize,
        &buffers.uniforms,
        &buffers.vertices,
        &buffers.indices,
        &material.colors,
        &buffers.densities,
        &material.color_grads,
        radix_state.values_b(),
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );

    // Loss.
    let loss_pipeline = LossPipeline::new(&device);
    let loss_buffers = LossBuffers::new(&device, W, H);
    let n_pixels = (W * H) as usize;
    let gt_data: Vec<f32> = vec![0.5; n_pixels * 3];
    queue.write_buffer(
        &loss_buffers.ground_truth,
        0,
        bytemuck::cast_slice(&gt_data),
    );
    let loss_uni = LossUniforms {
        width: W,
        height: H,
        loss_type: 1, // L2
        lambda_ssim: 0.0,
    };
    queue.write_buffer(
        &loss_buffers.loss_uniforms,
        0,
        bytemuck::bytes_of(&loss_uni),
    );
    let loss_bg = create_loss_bind_group(
        &device,
        &loss_pipeline,
        &loss_buffers,
        &rasterize.rendered_image,
    );

    // Backward tiled.
    let grad_buffers =
        rmesh_trainable::GradientBuffers::new(&device, scene.vertex_count, scene.tet_count);
    let mat_grad_buffers = rmesh_trainable::MaterialGradBuffers::new(&device, scene.tet_count);
    let bwd_pipelines = rmesh_trainable::BackwardTiledPipelines::new(&device);
    let (bwd_bg0_a, bwd_bg1) = rmesh_trainable::create_backward_tiled_bind_groups(
        &device,
        &bwd_pipelines,
        &buffers.uniforms,
        &loss_buffers.dl_d_image,
        &rasterize.rendered_image,
        &buffers.vertices,
        &buffers.indices,
        &buffers.densities,
        &material.color_grads,
        &material.colors,
        &tile_buffers.tile_sort_values,
        &grad_buffers.d_vertices,
        &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads,
        &mat_grad_buffers.d_base_colors,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );
    let (bwd_bg0_b, _) = rmesh_trainable::create_backward_tiled_bind_groups(
        &device,
        &bwd_pipelines,
        &buffers.uniforms,
        &loss_buffers.dl_d_image,
        &rasterize.rendered_image,
        &buffers.vertices,
        &buffers.indices,
        &buffers.densities,
        &material.color_grads,
        &material.colors,
        radix_state.values_b(),
        &grad_buffers.d_vertices,
        &grad_buffers.d_densities,
        &mat_grad_buffers.d_color_grads,
        &mat_grad_buffers.d_base_colors,
        &tile_buffers.tile_ranges,
        &tile_buffers.tile_uniforms,
    );

    Some(State {
        device,
        queue,
        tet_count: scene.tet_count,
        buffers,
        material,
        fwd_pipelines,
        compute_bg,
        tile_pipelines,
        radix_pipelines,
        tile_buffers,
        radix_state,
        scan_pipelines,
        scan_buffers,
        prepare_dispatch_bg,
        rts_bg,
        tile_fill_bg,
        tile_gen_scan_bg,
        tile_ranges_bg_a,
        tile_ranges_bg_b,
        rasterize,
        rasterize_bg_a,
        rasterize_bg_b,
        loss_pipeline,
        loss_buffers,
        loss_bg,
        bwd_pipelines,
        grad_buffers,
        mat_grad_buffers,
        bwd_bg0_a,
        bwd_bg0_b,
        bwd_bg1,
    })
}

/// Record the tiled forward path (project → scan tile pipeline → sort →
/// tile_ranges → tiled raster).
fn record_forward_tiled(encoder: &mut wgpu::CommandEncoder, s: &State) -> bool {
    rmesh_render::record_project_compute(
        encoder,
        &s.fwd_pipelines,
        &s.buffers,
        &s.compute_bg,
        s.tet_count,
        &s.queue,
    );
    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);

    rmesh_trainable::record_scan_tile_pipeline(
        encoder,
        &s.scan_pipelines,
        &s.tile_pipelines,
        &s.prepare_dispatch_bg,
        &s.rts_bg,
        &s.tile_fill_bg,
        &s.tile_gen_scan_bg,
        &s.scan_buffers,
        &s.tile_buffers,
    );

    let result_in_b = rmesh_trainable::record_radix_sort(
        encoder,
        &s.device,
        &s.radix_pipelines,
        &s.radix_state,
        &s.tile_buffers.tile_sort_keys,
        &s.tile_buffers.tile_sort_values,
    );

    {
        let ranges_bg = if result_in_b {
            &s.tile_ranges_bg_b
        } else {
            &s.tile_ranges_bg_a
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = s.tile_buffers.max_pairs_pow2.div_ceil(256);
        pass.dispatch_workgroups(wgs.min(65535), wgs.div_ceil(65535).max(1), 1);
    }

    let fwd_bg = if result_in_b {
        &s.rasterize_bg_b
    } else {
        &s.rasterize_bg_a
    };
    rmesh_trainable::record_rasterize_compute(
        encoder,
        &s.rasterize,
        fwd_bg,
        s.tile_buffers.num_tiles,
    );
    result_in_b
}

fn run_forward(s: &State) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    record_forward_tiled(&mut encoder, s);
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn run_loss(s: &State) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);
    record_loss_pass(&mut encoder, &s.loss_pipeline, &s.loss_bg, W, H);
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn run_backward(s: &State) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    encoder.clear_buffer(&s.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_base_colors, 0, None);
    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);

    let result_in_b = record_forward_tiled(&mut encoder, s);
    record_loss_pass(&mut encoder, &s.loss_pipeline, &s.loss_bg, W, H);

    let bwd_bg0 = if result_in_b {
        &s.bwd_bg0_b
    } else {
        &s.bwd_bg0_a
    };
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("backward_tiled"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&s.bwd_pipelines.pipeline);
    pass.set_bind_group(0, bwd_bg0, &[]);
    pass.set_bind_group(1, &s.bwd_bg1, &[]);
    let n = s.tile_buffers.num_tiles;
    pass.dispatch_workgroups(n.min(65535), n.div_ceil(65535).max(1), 1);
    drop(pass);

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn print_breakdown(s: &State) {
    let mut ts = TimestampRecorder::new(&s.device, &s.queue, 12);

    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // project_compute (instrumented)
    {
        let reset_cmd = rmesh_render::DrawIndirectCommand {
            vertex_count: 12,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        s.queue
            .write_buffer(&s.buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
        encoder.clear_buffer(&s.buffers.tiles_touched, 0, None);

        let (b, e) = ts.allocate("project_compute");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        cpass.set_pipeline(&s.fwd_pipelines.compute_pipeline);
        cpass.set_bind_group(0, &s.compute_bg.bg0, &[]);
        cpass.set_bind_group(1, &s.compute_bg.bg1, &[]);
        let n_pow2 = s.tet_count.next_power_of_two();
        let total_workgroups = n_pow2.div_ceil(64);
        cpass.dispatch_workgroups(
            total_workgroups.min(65535),
            total_workgroups.div_ceil(65535),
            1,
        );
    }
    encoder.clear_buffer(&s.rasterize.rendered_image, 0, None);
    encoder.clear_buffer(&s.tile_buffers.tile_ranges, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_vertices, 0, None);
    encoder.clear_buffer(&s.grad_buffers.d_densities, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_color_grads, 0, None);
    encoder.clear_buffer(&s.mat_grad_buffers.d_base_colors, 0, None);
    encoder.clear_buffer(&s.loss_buffers.loss_value, 0, None);

    // Scan tile pipeline (single bucket — internals are small)
    rmesh_trainable::record_scan_tile_pipeline(
        &mut encoder,
        &s.scan_pipelines,
        &s.tile_pipelines,
        &s.prepare_dispatch_bg,
        &s.rts_bg,
        &s.tile_fill_bg,
        &s.tile_gen_scan_bg,
        &s.scan_buffers,
        &s.tile_buffers,
    );

    // Radix sort (bucket — owns its own passes)
    let result_in_b = rmesh_trainable::record_radix_sort(
        &mut encoder,
        &s.device,
        &s.radix_pipelines,
        &s.radix_state,
        &s.tile_buffers.tile_sort_keys,
        &s.tile_buffers.tile_sort_values,
    );

    // tile_ranges
    {
        let ranges_bg = if result_in_b {
            &s.tile_ranges_bg_b
        } else {
            &s.tile_ranges_bg_a
        };
        let (b, e) = ts.allocate("tile_ranges");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_ranges"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.tile_pipelines.tile_ranges_pipeline);
        pass.set_bind_group(0, ranges_bg, &[]);
        let wgs = s.tile_buffers.max_pairs_pow2.div_ceil(256);
        pass.dispatch_workgroups(wgs.min(65535), wgs.div_ceil(65535).max(1), 1);
    }

    // rasterize_compute
    {
        let fwd_bg = if result_in_b {
            &s.rasterize_bg_b
        } else {
            &s.rasterize_bg_a
        };
        let (b, e) = ts.allocate("rasterize_compute");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rasterize_compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(s.rasterize.pipeline());
        pass.set_bind_group(0, fwd_bg, &[]);
        pass.set_bind_group(1, &s.rasterize.aux_bind_group, &[]);
        let (x, y) = rmesh_tile::dispatch_2d(s.tile_buffers.num_tiles);
        pass.dispatch_workgroups(x, y, 1);
    }

    // loss
    {
        let (b, e) = ts.allocate("loss");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("loss"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.loss_pipeline.pipeline);
        pass.set_bind_group(0, &s.loss_bg, &[]);
        pass.dispatch_workgroups(W.div_ceil(16), H.div_ceil(16), 1);
    }

    // backward_tiled
    {
        let bwd_bg0 = if result_in_b {
            &s.bwd_bg0_b
        } else {
            &s.bwd_bg0_a
        };
        let (b, e) = ts.allocate("backward_tiled");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_tiled"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        pass.set_pipeline(&s.bwd_pipelines.pipeline);
        pass.set_bind_group(0, bwd_bg0, &[]);
        pass.set_bind_group(1, &s.bwd_bg1, &[]);
        let n = s.tile_buffers.num_tiles;
        pass.dispatch_workgroups(n.min(65535), n.div_ceil(65535).max(1), 1);
    }

    ts.resolve(&mut encoder);
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());

    let results = ts.read_results(&s.device, &s.queue);
    eprintln!("\n=== GPU Timestamp Breakdown (trainable forward+loss+backward) ===");
    print_timestamp_table(&results);
}

fn bench_trainable(c: &mut Criterion) {
    let state = match create_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping trainable_pipeline bench (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    eprintln!(
        "Tiles: {}x{} = {}, max_pairs_pow2 = {}",
        state.tile_buffers.tiles_x,
        state.tile_buffers.tiles_y,
        state.tile_buffers.num_tiles,
        state.tile_buffers.max_pairs_pow2
    );

    // Warmup populates rendered_image / loss / grads so the "*_only" benches
    // see realistic state.
    run_backward(&state);

    c.bench_function("forward_tiled_10M", |b| {
        b.iter(|| run_forward(&state));
    });
    c.bench_function("loss_compute_10M", |b| {
        b.iter(|| run_loss(&state));
    });
    c.bench_function("backward_tiled_10M", |b| {
        b.iter(|| run_backward(&state));
    });

    print_breakdown(&state);
}

criterion_group! {
    name = trainable_pipeline;
    config = Criterion::default().sample_size(15);
    targets = bench_trainable
}
criterion_main!(trainable_pipeline);
