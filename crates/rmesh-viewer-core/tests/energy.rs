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
    let (device, queue) = eight_buffer_device().expect("GPU required");
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
