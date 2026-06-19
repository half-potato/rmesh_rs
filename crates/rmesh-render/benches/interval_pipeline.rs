//! Live interval-shading forward path benchmark.
//!
//! Benches the path shipped by the viewer (`RenderMode::IntervalShader`,
//! GPU-sort branch): project_compute → radix sort → indirect_convert →
//! interval_gen → interval_render (HW vertex/fragment). MRT disabled so the
//! measurement focuses on the forward path itself, not deferred shading.
//!
//! Requires a GPU with SUBGROUP + TIMESTAMP_QUERY. Skips gracefully otherwise.
//!
//! Run: `cargo bench -p rmesh-render --bench interval_pipeline`

use criterion::{criterion_group, criterion_main, Criterion};
use glam::{Mat3, Mat4, Vec3};
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::test_util::{grid_tet_scene, print_timestamp_table, TimestampRecorder};

const W: u32 = 1920;
const H: u32 = 1080;
// 100^3 cells × 5 tets ≈ 5M tets.
const GRID_SIZE: u32 = 100;

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
    buffers: rmesh_render::SceneBuffers,
    material: rmesh_render::MaterialBuffers,
    fwd_pipelines: rmesh_render::ForwardPipelines,
    targets: rmesh_render::RenderTargets,
    compute_bg: rmesh_render::ForwardComputeBindGroups,
    ci_pipelines: rmesh_render::ComputeIntervalPipelines,
    sort_pipelines: rmesh_sort::RadixSortPipelines,
    sort_state: rmesh_sort::RadixSortState,
    gen_bg_a: wgpu::BindGroup,
    gen_bg_b: wgpu::BindGroup,
    ci_render_bg: wgpu::BindGroup,
    ci_convert_bg: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
}

fn create_state() -> Option<State> {
    let (device, queue) = pollster::block_on(async {
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
    })?;

    eprintln!("Generating grid scene (grid_size={GRID_SIZE})...");
    let scene = grid_tet_scene(GRID_SIZE);
    eprintln!(
        "Scene: {} vertices, {} tets",
        scene.vertex_count, scene.tet_count
    );

    let (vp, c2w, intrinsics, eye) = setup_camera();

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, targets, compute_bg, _render_bg) =
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
        0, // tile_size=0: skip the tile-counting scanline loop in project_compute.
        //                The interval forward path doesn't read tiles_touched.
        0.0,
        0,
        0.01,
        100.0,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, color_format);

    // 32-bit tet-level sort (1 payload).
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    let gen_bg_a = rmesh_render::create_compute_interval_gen_bind_group(
        &device,
        &ci_pipelines,
        &buffers,
        &material,
    );
    let gen_bg_b = rmesh_render::create_compute_interval_gen_bind_group_with_sort_values(
        &device,
        &ci_pipelines,
        &buffers,
        &material,
        sort_state.values_b(),
    );
    let ci_render_bg =
        rmesh_render::create_compute_interval_render_bind_group(&device, &ci_pipelines, &buffers);
    let ci_convert_bg = rmesh_render::create_compute_interval_indirect_convert_bind_group(
        &device,
        &ci_pipelines,
        &buffers,
    );

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench_interval_depth"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    Some(State {
        device,
        queue,
        tet_count: scene.tet_count,
        buffers,
        material,
        fwd_pipelines,
        targets,
        compute_bg,
        ci_pipelines,
        sort_pipelines,
        sort_state,
        gen_bg_a,
        gen_bg_b,
        ci_render_bg,
        ci_convert_bg,
        depth_view,
    })
}

fn record_frame(s: &State) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // Clear color + depth (the recorder relies on the caller to do this; same
    // contract as the viewer's IntervalShader branch).
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench_interval_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &s.targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &s.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }

    rmesh_render::record_sorted_compute_interval_forward_pass(
        &mut encoder,
        &s.device,
        &s.fwd_pipelines,
        &s.ci_pipelines,
        &s.sort_pipelines,
        &s.sort_state,
        &s.buffers,
        &s.targets,
        &s.compute_bg,
        &s.gen_bg_a,
        &s.gen_bg_b,
        &s.ci_render_bg,
        &s.ci_convert_bg,
        s.tet_count,
        &s.queue,
        &s.depth_view,
        None,   // hw_compute_bg
        None,   // profiler
        false,  // use_16bit_sort
        false,  // mrt_enabled — color-only keeps the bench focused on forward
    );

    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_forward_interval(c: &mut Criterion) {
    let state = match create_state() {
        Some(s) => s,
        None => {
            eprintln!("Skipping interval_pipeline bench (no GPU with SUBGROUP + TIMESTAMP_QUERY)");
            return;
        }
    };

    // Warmup.
    record_frame(&state);

    c.bench_function("forward_interval_5M", |b| {
        b.iter(|| record_frame(&state));
    });

    // GPU-timestamp breakdown (single instrumented pass).
    print_breakdown(&state);
}

fn print_breakdown(s: &State) {
    // 4 instrumented passes: project_compute, indirect_convert, interval_gen,
    // interval_render. Radix sort is treated as one bucket because
    // `record_radix_sort` opens its own passes that we don't own.
    let mut ts = TimestampRecorder::new(&s.device, &s.queue, 8);

    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // project_compute
    {
        let reset_cmd = rmesh_render::DrawIndirectCommand {
            vertex_count: 12,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        s.queue
            .write_buffer(&s.buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
        s.queue.write_buffer(
            &s.buffers.interval_args_buf,
            0,
            bytemuck::cast_slice(&[0u32; 8]),
        );
        let n_pow2 = s.tet_count.next_power_of_two();
        s.queue
            .write_buffer(s.sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));
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
        let dispatch_x = total_workgroups.min(65535);
        let dispatch_y = total_workgroups.div_ceil(65535);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Radix sort (5 passes; no timestamps — record_radix_sort owns its own).
    let result_in_b = rmesh_sort::record_radix_sort(
        &mut encoder,
        &s.device,
        &s.sort_pipelines,
        &s.sort_state,
        &s.buffers.sort_keys,
        &s.buffers.sort_values,
    );

    let gen_bg = if result_in_b {
        &s.gen_bg_b
    } else {
        &s.gen_bg_a
    };

    // indirect_convert
    {
        let (b, e) = ts.allocate("indirect_convert");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("indirect_convert"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        cpass.set_pipeline(&s.ci_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, &s.ci_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // interval_gen (indirect dispatch driven by indirect_convert output)
    {
        let (b, e) = ts.allocate("interval_gen");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interval_gen"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
        });
        cpass.set_pipeline(&s.ci_pipelines.gen_pipeline);
        cpass.set_bind_group(0, gen_bg, &[]);
        cpass.dispatch_workgroups_indirect(&s.buffers.interval_args_buf, 0);
    }

    // interval_render (HW raster — color-only pipeline still declares 4 color
    // targets; the unused slots must be present as `None` in the pass).
    {
        let (b, e) = ts.allocate("interval_render");
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("interval_render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &s.targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                None,
                None,
                None,
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &s.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                query_set: ts.query_set(),
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            }),
            ..Default::default()
        });
        rpass.set_viewport(
            0.0,
            0.0,
            s.targets.width as f32,
            s.targets.height as f32,
            0.0,
            1.0,
        );
        rpass.set_scissor_rect(0, 0, s.targets.width, s.targets.height);
        rpass.set_pipeline(&s.ci_pipelines.render_pipeline_color_only);
        rpass.set_bind_group(0, &s.ci_render_bg, &[]);
        rpass.set_index_buffer(
            s.buffers.interval_fan_index_buf.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        // draw-indexed-indirect args start at byte offset 12 (skip 3 dispatch u32s).
        rpass.draw_indexed_indirect(&s.buffers.interval_args_buf, 12);
    }

    ts.resolve(&mut encoder);
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());

    let results = ts.read_results(&s.device, &s.queue);
    eprintln!("\n=== GPU Timestamp Breakdown (forward interval) ===");
    print_timestamp_table(&results);
}

criterion_group! {
    name = interval_pipeline;
    config = Criterion::default().sample_size(20);
    targets = bench_forward_interval
}
criterion_main!(interval_pipeline);
