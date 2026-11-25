//! Opt-in matched full-frame diagnostic, not a manuscript accuracy experiment.
use glam::{Mat3, Mat4, Vec3};
use rmesh_data::SceneData;
#[test]
#[ignore = "requires RMESH_TEST_ASSET, RMESH_TEST_OUTPUT, GPU and several GB of memory"]
fn render_probe_asset() {
    let path = std::env::var("RMESH_TEST_ASSET").unwrap();
    let prefix = std::env::var("RMESH_TEST_OUTPUT").unwrap();
    let bytes = std::fs::read(path).unwrap();
    let (scene, sh, pbr) = rmesh_data::load_rmesh(&bytes).unwrap();
    drop(bytes);
    let pbr = pbr.expect("energy asset");
    assert!(!pbr.energy_material.is_empty());
    let eye = if let Ok(v) = std::env::var("RMESH_TEST_CAMERA") {
        let c: Vec<f32> = v.split(',').map(|x| x.trim().parse().expect("cam coord")).collect();
        assert_eq!(c.len(), 3, "RMESH_TEST_CAMERA expects x,y,z");
        Vec3::from_slice(&c)
    } else {
        Vec3::from_slice(&scene.start_pose[..3])
    };
    let (w, h) = if let Ok(v) = std::env::var("RMESH_TEST_SIZE") {
        let p: Vec<u32> = v.split('x').map(|x| x.trim().parse().expect("size")).collect();
        assert_eq!(p.len(), 2, "RMESH_TEST_SIZE expects WxH");
        (p[0], p[1])
    } else {
        (512u32, 320u32)
    };
    let mut camera = rmesh_util::camera::Camera::new(eye);
    camera.near_z = test_near();
    let view = camera.view_matrix();
    let vp = camera.projection_matrix(w as f32 / h as f32) * view;
    let inv = view.inverse();
    let c2w = Mat3::from_cols(
        inv.x_axis.truncate(),
        -inv.y_axis.truncate(),
        -inv.z_axis.truncate(),
    );
    let f = (h as f32 * 0.5) / (camera.fov_y / 2.).tan();
    if std::env::var_os("RMESH_TEST_MATERIALS").is_some() {
        inspect_materials(&scene, &pbr, eye, c2w, f);
        return;
    }
    let image = pollster::block_on(gpu_compute_interval_render_scene_async(
        &scene,
        &sh,
        &pbr,
        &prefix,
        eye,
        vp,
        c2w,
        [f, f, w as f32 * 0.5, h as f32 * 0.5],
        w,
        h,
        true,
    ))
    .expect("GPU render");
    assert!(image.iter().flatten().all(|x| x.is_finite()));
    assert!(image.iter().any(|p| p[0] > 0.001), "black frame");
    std::fs::write(format!("{prefix}.rgba"), bytemuck::cast_slice(&image)).unwrap();
}
async fn gpu_compute_interval_render_scene_async(
    scene: &SceneData,
    sh: &rmesh_data::ShCoeffs,
    pbr: &rmesh_data::PbrData,
    prefix: &str,
    cam_pos: Vec3,
    vp: Mat4,
    c2w: Mat3,
    intrinsics: [f32; 4],
    w: u32,
    h: u32,
    use_hw_compute: bool,
) -> Option<Vec<[f32; 4]>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        ..Default::default()
    });
    let adapter = if let Ok(index) = std::env::var("RMESH_TEST_GPU") {
        instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .await
            .into_iter()
            .nth(index.parse().expect("GPU index"))
            .expect("requested GPU")
    } else {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("GPU adapter/device creation")
    };
    eprintln!("Adapter: {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 20,
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        })
        .await
        .expect("GPU adapter/device creation");

    let zero_base_colors = vec![0.5f32; scene.tet_count as usize * 3];
    let (buffers, material, pipelines, targets, compute_bg, _render_bg) =
        rmesh_render::setup_forward_with_formats(
            &device,
            &queue,
            scene,
            &zero_base_colors,
            &scene.color_grads,
            w,
            h,
            rmesh_render::GBufferFormats {
                color: wgpu::TextureFormat::Rgba16Float,
                ..Default::default()
            },
        );

    // Compute-interval pipelines (no mesh shader needed)
    let mut ci_pipelines = rmesh_render::ComputeIntervalPipelines::new(
        &device,
        rmesh_render::GBufferFormats {
            color: wgpu::TextureFormat::Rgba16Float,
            ..Default::default()
        },
    );

    rmesh_viewer_core::light_bake::upload_material(&device, &queue, scene, Some(pbr), &buffers);
    let mut bake = rmesh_viewer_core::light_bake::LightBake::new(
        &device,
        &queue,
        scene,
        sh,
        Some(pbr),
        &buffers,
    );
    bake.configure_interval(&device, &mut ci_pipelines);
    let mut light = rmesh_viewer_core::light_bake::default_light(scene);
    if std::env::var_os("RMESH_TEST_FLASH").is_some() {
        light.transform.position = cam_pos;
    }
    let bake_start = std::time::Instant::now();
    loop {
        let mut encoder = device.create_command_encoder(&Default::default());
        bake.record(
            &queue,
            &mut encoder,
            &material,
            &[light.clone()],
            cam_pos.to_array(),
        );
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        if bake.ready() {
            break;
        }
    }
    eprintln!("Initial bake: {:.3}s", bake_start.elapsed().as_secs_f64());
    // Sort infrastructure
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(&device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(&queue);

    // Bind groups
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

    // For the HW-compute path the project pass uses uniform-buffer uniforms
    // and a different bind group layout (project_compute_hw.wgsl). Built only
    // when needed so we keep the legacy path's setup unchanged.
    let hw_compute_bg = if use_hw_compute {
        let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hw_dummy_sh_coeffs"),
            size: 4, // arrayLength==1 → shader takes the base_colors branch
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let (_shell_hiz_stub, shell_hiz_stub_view) = rmesh_render::dummy_shell_hiz(&device, &queue);
        Some(rmesh_render::create_hw_compute_bind_group(
            &device,
            &pipelines,
            &buffers,
            &material,
            &dummy_sh,
            &shell_hiz_stub_view,
        ))
    } else {
        None
    };

    // Write uniforms
    let mut uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        cam_pos,
        w as f32,
        h as f32,
        scene.tet_count,
        0,
        16,
        0.0,
        0,
        test_near(),
        1000.0,
    );
    uniforms.min_t = pbr.render_min_t.unwrap_or(0.0);
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    // Depth texture
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_ci_depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
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

    let mut timings = String::from("frame,camera_update,ms\n");
    for frame in 0..std::env::var("RMESH_TEST_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(24)
    {
        let start = std::time::Instant::now();
        // Alternate a tiny camera displacement to include the per-view shade pass.
        let moving = frame >= 12;
        let eye = cam_pos + Vec3::new(if moving && frame % 2 == 0 { 0.001 } else { 0.0 }, 0., 0.);
        // Record and submit
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_ci"),
        });

        bake.record(
            &queue,
            &mut encoder,
            &material,
            &[light.clone()],
            eye.to_array(),
        );
        // Clear color and depth
        {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test_ci_clear"),
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
            &mut encoder,
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
            hw_compute_bg.as_ref(),
            None,
            false,
            false,
            None,
        );

        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        timings.push_str(&format!(
            "{frame},{moving},{}\n",
            start.elapsed().as_secs_f64() * 1000.
        ));
    }
    std::fs::write(format!("{prefix}.csv"), timings).unwrap();
    if let Some(positions) = bake.reflection_probe_positions() {
        std::fs::write(format!("{prefix}.probes.txt"), format!("{positions:?}\n")).unwrap();
    }
    if let Some(probe) = bake.reflection_probe() {
        let size = probe.size();
        let row = (size.width * 8 + 255) & !255;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe readback"),
            size: (row * size.height * size.depth_or_array_layers) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: probe,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        std::fs::write(format!("{prefix}.probe.rgba16"), &*slice.get_mapped_range()).unwrap();
        buffer.unmap();
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    // Copy color texture to readback buffer
    let color_format = targets.formats.color;
    let bytes_per_pixel = rmesh_util::shared::bytes_per_texel(color_format);
    let bytes_per_row = w * bytes_per_pixel;
    let aligned_bytes_per_row = (bytes_per_row + 255) & !255;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (aligned_bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &targets.color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Map and read back
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().ok()?;

    let data = slice.get_mapped_range();
    let mut image = vec![[0.0f32; 4]; (w * h) as usize];

    for row in 0..h {
        let row_start = (row * aligned_bytes_per_row) as usize;
        for col in 0..w {
            let pixel_start = row_start + (col * bytes_per_pixel) as usize;
            image[(row * w + col) as usize] =
                rmesh_util::shared::decode_texel(color_format, &data[pixel_start..]);
        }
    }

    drop(data);
    readback.unmap();

    Some(image)
}

fn test_near() -> f32 {
    std::env::var("RMESH_TEST_NEAR")
        .ok()
        .map(|v| v.parse().expect("near plane"))
        .unwrap_or(1.6)
}

// CPU-only diagnostic: identify opacity-dominant cells along selected image rays.
fn inspect_materials(scene: &SceneData, pbr: &rmesh_data::PbrData, eye: Vec3, c2w: Mat3, f: f32) {
    let vertex = |i: u32| Vec3::from_slice(&scene.vertices[3 * i as usize..3 * i as usize + 3]);
    for (px, py) in [
        (100., 40.),
        (256., 30.),
        (440., 40.),
        (320., 100.),
        (256., 200.),
    ] {
        let d = (c2w * Vec3::new((px - 256.) / f, (py - 160.) / f, 1.)).normalize();
        let mut hits = Vec::new();
        for (t, ids) in scene.indices.chunks_exact(4).enumerate() {
            let vs = [
                vertex(ids[0]),
                vertex(ids[1]),
                vertex(ids[2]),
                vertex(ids[3]),
            ];
            let mut near = pbr.render_min_t.unwrap_or(0.);
            let mut far = f32::INFINITY;
            for face in [[0, 2, 1, 3], [1, 2, 3, 0], [0, 3, 2, 1], [3, 0, 1, 2]] {
                let a = vs[face[0]];
                let mut n = (vs[face[1]] - a).cross(vs[face[2]] - a);
                if n.dot(vs[face[3]] - a) > 0. {
                    n = -n;
                }
                let den = n.dot(d);
                let offset = n.dot(a - eye);
                if den.abs() < 1e-20 {
                    if offset < 0. {
                        far = -1.;
                        break;
                    }
                } else if den > 0. {
                    far = far.min(offset / den);
                } else {
                    near = near.max(offset / den);
                }
            }
            if far > near && far.is_finite() {
                hits.push((near, far, t));
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut trans = 1.;
        let mut contributions = Vec::new();
        for (near, far, t) in hits {
            let alpha = 1. - (-scene.densities[t] * (far - near)).exp();
            contributions.push((trans * alpha, t, near, far));
            trans *= 1. - alpha;
        }
        contributions.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("PIXEL {px},{py} residual_trans={trans}");
        for (weight, t, near, far) in contributions.into_iter().take(3) {
            println!("  cell={t} opacity_weight={weight:.5} range={near:.3}..{far:.3} density={} material[roughness,F0rgb,retro]={:?}",scene.densities[t],&pbr.energy_material[5*t..5*t+5]);
            for &i in &scene.indices[t * 4..t * 4 + 4] {
                let owner = t;
                let n = -Vec3::from_slice(&pbr.geom_normals[3 * i as usize..3 * i as usize + 3])
                    .normalize_or_zero();
                println!(
                    "    vertex={i} cell_material={owner} normal_dot_view={:.3} material={:?}",
                    n.dot(-d),
                    &pbr.energy_material[5 * owner..5 * owner + 5]
                );
            }
        }
    }
}
