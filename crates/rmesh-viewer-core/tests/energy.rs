use rmesh_viewer_core::energy;
use wgpu::util::DeviceExt;
#[test]
fn direct_energy_matches_pytorch() {
    validate_energy(false);
}
#[test]
fn filtered_energy_matches_pytorch() {
    validate_energy(true);
}
fn validate_energy(filtered: bool) {
    let Some((device, queue)) = eight_buffer_device() else {
        panic!("GPU required for energy parity")
    };
    if filtered
        && !device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE)
    {
        eprintln!("Float32 filtering unavailable; manual fallback tested separately");
        return;
    }
    let bytes = include_bytes!("fixtures/energy_vectors.bin");
    let input: Vec<[f32; 4]> = bytes
        .chunks_exact(16)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let count = input.len() / 6;
    let packed: Vec<[f32; 4]> = input
        .chunks_exact(6)
        .flat_map(|v| v[..5].iter().copied())
        .collect();
    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: count as u64 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let lut = energy::upload(&device, &queue);
    let shader=device.create_shader_module(wgpu::ShaderModuleDescriptor {label:Some("energy parity"),source:wgpu::ShaderSource::Wgsl(format!("{}\n{}",format!("{}\n{}",energy::shader_source(filtered),if filtered {"@group(0) @binding(3) var energy_sampler:sampler;"} else {""}),r#"
@group(0) @binding(0) var<storage,read> inputs:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
@group(0) @binding(2) var energy_lut:texture_2d<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=arrayLength(&output)){return;}let b=i*5u;let n=inputs[b];let l=inputs[b+1];
 output[i]=vec4f(energy_source(n.xyz,l.xyz,inputs[b+2].xyz,n.w,inputs[b+3].xyz,inputs[b+4].xyz,l.w),0.);
}"#).into())});
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: data.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: output.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&lut),
        },
    ];
    if filtered {
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Sampler(&sampler),
        });
    }
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    queue.submit([enc.finish()]);
    let actual = rmesh_util::test_util::read_buffer::<[f32; 4]>(&device, &queue, &output, count);
    let mut max = 0f32;
    let mut worst = 0;
    for (i, (got, want)) in actual
        .iter()
        .zip(input.chunks_exact(6).map(|v| v[5]))
        .enumerate()
    {
        for c in 0..3 {
            assert!(got[c].is_finite());
            let delta = (got[c] - want[c]).abs();
            if delta > max {
                max = delta;
                worst = i;
            }
        }
    }
    eprintln!("PyTorch energy filtered={filtered} cases={count}, max absolute source error={max}, worst={worst}");
    assert!(max <= 0.01, "energy source differs from PyTorch: {max}");
}

fn eight_buffer_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 8,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .ok()
    })
}

#[test]
fn disconnected_occluder_blocks_diffuse_baker() {
    use rmesh_data::{MaterialBuffers, PbrData, SceneBuffers, ShCoeffs};
    use rmesh_util::test_util::{build_test_scene, read_buffer};
    use rmesh_viewer_core::light_bake::{default_light, LightBake};
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .ok()
    })
    .expect("GPU required");
    {
        // Receiver and dense blocker share no vertices or faces. The ray must
        // leave the receiver, cross a vacuum gap, then enter the blocker.
        let scene = build_test_scene(
            vec![
                -0.1, -0.1, 0., 0.1, -0.1, 0., 0., 0.1, 0., 0., 0., 0.1, -2., -2., 1., 2., -2., 1.,
                0., 2., 1., 0., 0., 3.,
            ],
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![0., 0.],
            vec![0.; 6],
        );
        let pbr = PbrData {
            energy_material: vec![],
            energy_lut: energy::LUT
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
            geom_normals: vec![0., 0., -1.].repeat(8),
            render_min_t: None,
            empty_indices: vec![],
            face_textures: None,
            roughness: vec![0.4; 2],
            env_feature: vec![],
            vertex_normals: vec![0., 0., -1.].repeat(8),
            tone_curve: vec![],
            metallic: vec![],
            f0_dielectric: vec![],
            albedo: vec![0.5; 6],
        };
        let buffers = SceneBuffers::upload(&device, &queue, &scene);
        queue.write_buffer(
            &buffers.vertex_normals,
            0,
            bytemuck::cast_slice(&pbr.vertex_normals),
        );
        let material = MaterialBuffers::upload(&device, &[0.; 6], &[0.; 6], 2);
        let mut bake = LightBake::new(
            &device,
            &queue,
            &scene,
            &ShCoeffs {
                degree: 0,
                coeffs: vec![2.; 6],
            },
            Some(&pbr),
            &buffers,
        );
        let mut lamp = default_light(&scene);
        lamp.transform.position = glam::Vec3::new(0., 0., 4.);
        let run = |bake: &mut LightBake, lamp: &rmesh_interact::Primitive| {
            let mut enc = device.create_command_encoder(&Default::default());
            bake.record(&queue, &mut enc, &material, &[lamp.clone()], [0.; 3]);
            queue.submit([enc.finish()]);
            read_buffer::<f32>(&device, &queue, &material.base_colors, 6)[0]
        };
        let lit = run(&mut bake, &lamp);
        queue.write_buffer(&buffers.densities, 0, bytemuck::cast_slice(&[0f32, 20.]));
        bake.invalidate();
        let blocked = run(&mut bake, &lamp);
        eprintln!("unblocked={lit}, blocked={blocked}");
        assert!(lit > 0.0);
        assert!(blocked < lit * 0.01, "Disconnected blocker ignored");
        lamp.transform.position.z = 0.5;
        let before_blocker = run(&mut bake, &lamp);
        assert!(
            before_blocker > lit,
            "Geometry beyond the light must not shadow it"
        );
    }
}

#[test]
fn entry_face_texture_shades_subtriangle_albedo_and_fallback() {
    use glam::{Mat3, Vec3, Vec4};
    use rmesh_data::{FaceTextureData, PbrData, ShCoeffs};
    use rmesh_util::test_util::build_test_scene;
    use rmesh_viewer_core::light_bake::{default_light, LightBake};

    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .ok()
    })
    .expect("GPU required");
    // Tet 0: front face in XZ plane facing -Y (camera's Z-up convention), stored in permuted
    // local order [2, 0, 1, 3] to verify canonical ascending global-vertex-ID ordering.
    // Tet 1: untextured tet to the side verifying fallback albedo fast path.
    let verts = vec![
        -1.2, 0.0, -0.8, // v0 -> canonical 0 (RED)
         0.2, 0.0, -0.8, // v1 -> canonical 1 (GREEN)
        -0.5, 0.0,  0.9, // v2 -> canonical 2 (BLUE)
        -0.5, 1.5,  0.0, // v3 -> back apex (+Y, opposite = local slot 3)
         0.5, 0.0, -0.6, // v4 -> untextured tet
         1.5, 0.0, -0.6, // v5
         1.0, 0.0,  0.7, // v6
         1.0, 1.5,  0.0, // v7
    ];
    let scene = build_test_scene(
        verts,
        vec![2, 0, 1, 3, 4, 5, 6, 7],
        vec![20.0, 20.0],
        vec![0.0; 6],
    );
    let pbr = PbrData {
        roughness: vec![0.9, 0.9],
        albedo: vec![0.0, 0.0, 0.0, 0.9, 0.9, 0.0], // Tet 0 black fallback; Tet 1 yellow fallback
        energy_material: vec![
            0.9, 0.0, 0.0, 0.0, 0.0, // pure diffuse so RGB ratios match texture albedo
            0.9, 0.0, 0.0, 0.0, 0.0,
        ],
        energy_lut: energy::LUT
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
        geom_normals: vec![0.0, 1.0, 0.0].repeat(8), // inward density gradient -> outward normal -Y
        render_min_t: Some(0.0),
        empty_indices: vec![],
        face_textures: Some(FaceTextureData {
            face_ids: vec![-1, -1, -1, 0, -1, -1, -1, -1],
            charts: vec![0, 2], // res=2 -> 3 sites: v0=R, v1=G, v2=B
            albedo_rgb8: vec![0x0000ff, 0x00ff00, 0xff0000],
        }),
        env_feature: vec![],
        vertex_normals: vec![0.0, 1.0, 0.0].repeat(8),
        tone_curve: vec![],
        metallic: vec![],
        f0_dielectric: vec![],
    };

    let (w, h) = (128u32, 128u32);
    let eye = Vec3::new(0.0, -3.5, 0.0);
    let mut camera = rmesh_util::camera::Camera::new(eye);
    let view = camera.view_matrix();
    let vp = camera.projection_matrix(w as f32 / h as f32) * view;
    let inv = view.inverse();
    let c2w = Mat3::from_cols(
        inv.x_axis.truncate(),
        -inv.y_axis.truncate(),
        -inv.z_axis.truncate(),
    );
    let f = (h as f32 * 0.5) / (camera.fov_y * 0.5).tan();
    let intrinsics = [f, f, w as f32 * 0.5, h as f32 * 0.5];

    let formats = rmesh_render::GBufferFormats {
        color: wgpu::TextureFormat::Rgba16Float,
        ..Default::default()
    };
    let (buffers, material, pipelines, targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward_with_formats(
            &device,
            &queue,
            &scene,
            &vec![0.0f32; 6],
            &scene.color_grads,
            w,
            h,
            formats,
        );
    rmesh_viewer_core::light_bake::upload_material(&device, &queue, &scene, Some(&pbr), &buffers);

    let mut ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(&device, formats);
    let sh = ShCoeffs { degree: 0, coeffs: vec![0.0; 6] };
    let mut bake = LightBake::new(&device, &queue, &scene, &sh, Some(&pbr), &buffers);
    bake.configure_interval(&device, &mut ci_pipelines);

    let mut lamp = default_light(&scene);
    lamp.transform.position = Vec3::new(0.0, -2.5, 0.0);
    lamp.color = Some([1.0, 1.0, 1.0, 10.0]);
    loop {
        let mut enc = device.create_command_encoder(&Default::default());
        bake.record(&queue, &mut enc, &material, &[lamp.clone()], eye.to_array());
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        if bake.ready() {
            break;
        }
    }

    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Basic);
    let sort_state =
        rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Basic);
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

    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        eye,
        w as f32,
        h as f32,
        scene.tet_count,
        0,
        16,
        0.0,
        0,
        camera.near_z,
        camera.far_z,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ft_test_depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&Default::default());

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let _clear = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ft_test_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
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
        &mut enc,
        &device,
        &pipelines,
        &ci_pipelines,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &targets,
        &compute_bg,
        &gen_bg_a,
        &gen_bg_b,
        &ci_render_bg,
        &ci_convert_bg,
        scene.tet_count,
        &queue,
        &depth_view,
        None,
        None,
        false,
        false,
        None,
    );
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // Readback Rgba16Float color target
    let bpp = 8u32;
    let padded = (w * bpp).div_ceil(256) * 256;
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ft_test_rb"),
        size: padded as u64 * h as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        targets.color_texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        targets.color_texture.size(),
    );
    queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    rb.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = rb.slice(..).get_mapped_range();
    let mut img = vec![[0f32; 4]; (w * h) as usize];
    for y in 0..h as usize {
        let row = &mapped[y * padded as usize..y * padded as usize + w as usize * 8];
        for (x, px) in row.chunks_exact(8).enumerate() {
            img[y * w as usize + x] = [
                half::f16::from_le_bytes([px[0], px[1]]).to_f32(),
                half::f16::from_le_bytes([px[2], px[3]]).to_f32(),
                half::f16::from_le_bytes([px[4], px[5]]).to_f32(),
                half::f16::from_le_bytes([px[6], px[7]]).to_f32(),
            ];
        }
    }
    drop(mapped);
    rb.unmap();
    assert!(img.iter().flatten().all(|v| v.is_finite()), "non-finite pixel");

    let sample_world = |p: Vec3| -> [f32; 4] {
        let clip = vp * Vec4::new(p.x, p.y, p.z, 1.0);
        let ndc = clip.truncate() / clip.w;
        let px = ((ndc.x * 0.5 + 0.5) * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let py = ((0.5 - ndc.y * 0.5) * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        img[py * w as usize + px]
    };
    let v0 = Vec3::new(-1.2, 0.0, -0.8);
    let v1 = Vec3::new(0.2, 0.0, -0.8);
    let v2 = Vec3::new(-0.5, 0.0, 0.9);
    let p_r = sample_world(v0 * 0.8 + v1 * 0.1 + v2 * 0.1);
    let p_g = sample_world(v0 * 0.1 + v1 * 0.8 + v2 * 0.1);
    let p_b = sample_world(v0 * 0.1 + v1 * 0.1 + v2 * 0.8);
    let p_fallback = sample_world(Vec3::new(1.0, 0.0, -0.1));
    eprintln!("FT GPU test: near_v0={p_r:?} near_v1={p_g:?} near_v2={p_b:?} fallback={p_fallback:?}");
    assert!(p_r[0] > 0.01 && p_r[0] > 4.0 * p_r[1] && p_r[0] > 4.0 * p_r[2], "v0 corner not red: {p_r:?}");
    assert!(p_g[1] > 0.01 && p_g[1] > 4.0 * p_g[0] && p_g[1] > 4.0 * p_g[2], "v1 corner not green: {p_g:?}");
    assert!(p_b[2] > 0.01 && p_b[2] > 4.0 * p_b[0] && p_b[2] > 4.0 * p_b[1], "v2 corner not blue: {p_b:?}");
    assert!(
        p_fallback[0] > 0.01 && p_fallback[1] > 0.01 && p_fallback[2] < 0.1 * p_fallback[0],
        "untextured fallback tet not yellow: {p_fallback:?}"
    );
}
