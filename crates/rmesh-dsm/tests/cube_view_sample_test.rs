//! Cube-VIEW sampling probe — exercises the exact path the deferred shader uses.
//!
//! The orientation test reads raw cube layers via `copy_texture_to_buffer`. The
//! deferred shader instead reads through a `texture_cube` VIEW with
//! `textureSample(cube, dir)`. If cube-view sampling is rotated/flipped relative
//! to how the layers were written, the raw-layer probe can't see it but the
//! shader can. This test generates the DSM for ONE occluder tet, then samples
//! the cube VIEW (same `atlas.cubemap_views[0]` the deferred pass binds) at a set
//! of candidate directions and reports which direction actually returns the
//! occluder's alpha.
//!
//! If `textureSample(cube, normalize(center))` returns the occluder (high alpha)
//! and the rotated candidates return ~0, identity sampling is correct. If a
//! ROTATED direction returns the occluder instead, the cube-view read is rotated
//! vs the write — the real bug — and this test names the winning transform.
//!
//! GPU-gated: prints a skip notice and returns if no adapter is available.

use glam::Vec3;
use rmesh_util::test_util::build_test_scene;

const RES: u32 = 256;
const NEAR: f32 = 0.1;
const FAR: f32 = 15.0;

const SAMPLE_WGSL: &str = r#"
@group(0) @binding(0) var cube: texture_cube<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    outp[i] = textureSampleLevel(cube, samp, dirs[i].xyz, 0.0);
}
"#;

fn make_tet(center: Vec3, size: f32) -> [Vec3; 4] {
    let h = size * 0.5;
    [
        center + Vec3::new(1.0, 1.0, 1.0) * h,
        center + Vec3::new(1.0, -1.0, -1.0) * h,
        center + Vec3::new(-1.0, 1.0, -1.0) * h,
        center + Vec3::new(-1.0, -1.0, 1.0) * h,
    ]
}

fn single_tet_scene(center: Vec3, size: f32) -> rmesh_data::SceneData {
    let verts = make_tet(center, size);
    let mut vertices = Vec::new();
    for v in verts {
        vertices.extend_from_slice(&[v.x, v.y, v.z]);
    }
    build_test_scene(vertices, vec![0, 1, 3, 2], vec![5.0], vec![0.0, 0.0, 0.0])
}

/// Generate the DSM, then sample the cube VIEW at `dirs`; return per-dir RGBA.
fn sample_cube_view(scene: &rmesh_data::SceneData, dirs: &[Vec3]) -> Option<Vec<[f32; 4]>> {
    let (device, queue) =
        rmesh_util::test_util::create_test_device(rmesh_util::test_util::TestDeviceConfig {
            backends: None,
            extra_features: wgpu::Features::empty(),
            base_limits: wgpu::Limits::default(),
        })?;

    let color_format = wgpu::TextureFormat::Rgba16Float;
    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd_pipelines, _t, _c, _r) = rmesh_render::setup_forward(
        &device,
        &queue,
        scene,
        &zero_base_colors,
        &scene.color_grads,
        RES,
        RES,
    );
    let ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, color_format);
    let n_pow2 = scene.tet_count.next_power_of_two().max(1);
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    let dsm_pipeline = rmesh_dsm::DsmPipeline::new(&device, color_format);
    let dsm_prim_pipeline = rmesh_dsm::DsmPrimitivePipeline::new(&device);
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let prim_geometry = rmesh_compositor::PrimitiveGeometry::new(&device);
    let atlas = rmesh_dsm::DsmAtlas::new(&device, RES, &[0]);

    let lights = [rmesh_render::GpuLight {
        position: [0.0, 0.0, 0.0],
        light_type: 0,
        ..Default::default()
    }];

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

    // --- Sample the cube VIEW (the same view the deferred pass binds) ---
    let cube_view = &atlas.cubemap_views[0];
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("cube_probe_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let mut dir_data = Vec::with_capacity(dirs.len() * 4);
    for d in dirs {
        let n = d.normalize();
        dir_data.extend_from_slice(&[n.x, n.y, n.z, 0.0]);
    }
    let dirs_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dirs"),
        size: (dir_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&dirs_buf, 0, bytemuck::cast_slice(&dir_data));

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (dirs.len() * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (dirs.len() * 16) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cube_sample"),
        source: wgpu::ShaderSource::Wgsl(SAMPLE_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cube_sample_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cube_sample_pl"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cube_sample_pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cube_sample_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(cube_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dirs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cube_sample_enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cube_sample"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(dirs.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (dirs.len() * 16) as u64);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, f32>(&data)
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    drop(data);
    readback.unmap();
    Some(out)
}

#[test]
fn cube_view_sample_hits_occluder_direction() {
    // One occluder tet, +X dominant, tilted +Y and +Z (fully asymmetric).
    let center = Vec3::new(6.0, 2.0, 1.0);
    let scene = single_tet_scene(center, 2.5);

    // Candidate lookup directions, with the transforms in play:
    //   R   = SAMPLE_DIR_ROT(+90° about X) ∘ x-flip  = (x,y,z) -> (-x,-z, y)
    //   R⁻¹                                            = (x,y,z) -> (-x, z,-y)
    let r = |d: Vec3| Vec3::new(-d.x, -d.z, d.y);
    let r_inv = |d: Vec3| Vec3::new(-d.x, d.z, -d.y);
    let labels = [
        "identity   dir",
        "x-flip     (-x,y,z)",
        "R   (-x,-z,y)  [user fix]",
        "R⁻¹ (-x,z,-y)",
        "+X axis",
        "+Y axis",
        "+Z axis",
    ];
    let dirs = vec![
        center,
        Vec3::new(-center.x, center.y, center.z),
        r(center),
        r_inv(center),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
    ];

    let out = match sample_cube_view(&scene, &dirs) {
        Some(o) => o,
        None => {
            eprintln!("cube_view_sample_hits_occluder_direction: no GPU adapter — skipping");
            return;
        }
    };

    eprintln!("occluder tet center {center:?}  (only +X face is generated)");
    let mut best = (0usize, -1.0f32);
    for (i, (lbl, px)) in labels.iter().zip(out.iter()).enumerate() {
        eprintln!("  sample[{lbl:>26}] = rgba({:.3},{:.3},{:.3}, a={:.3})", px[0], px[1], px[2], px[3]);
        if px[3] > best.1 {
            best = (i, px[3]);
        }
    }
    eprintln!(
        "==> highest occluder alpha at: {}  (a={:.3})",
        labels[best.0], best.1
    );

    // The occluder must be found at the IDENTITY direction if cube-view sampling
    // matches the write. If it's found at R instead, the deferred path is rotated.
    assert!(
        best.0 == 0,
        "cube VIEW returns the occluder at '{}', NOT at identity 'dir' — the \
         deferred textureSample path is rotated/flipped relative to the cube write \
         (raw-layer probe cannot see this)",
        labels[best.0]
    );
}
