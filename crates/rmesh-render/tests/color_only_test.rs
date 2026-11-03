//! The fast viewer path must integrate radiance even when albedo is present.
use rmesh_util::test_util::{create_test_device_default, read_buffer};
use wgpu::util::DeviceExt;

#[test]
fn color_only_preserves_hdr_radiance_with_pbr_materials() {
    let Some((device, queue)) = create_test_device_default() else {
        eprintln!("Skipping color-only GPU regression (no GPU)");
        return;
    };
    let formats = rmesh_render::GBufferFormats {
        color: wgpu::TextureFormat::Rgba16Float,
        ..Default::default()
    };
    let targets = rmesh_render::RenderTargets::new_color_only(&device, 32, 16, formats);
    assert_eq!(targets.color_texture.width(), 32);
    assert_eq!(targets.color_texture.height(), 16);
    for texture in [
        &targets.aux0_texture,
        &targets.normals_texture,
        &targets.depth_texture,
        &targets.ssgi_texture,
        &targets.ssr_texture,
        &targets.ao_texture,
    ] {
        assert_eq!((texture.width(), texture.height()), (1, 1));
    }
    // Also validate the real fragment pipelines, including the new shell entry.
    let _interval = rmesh_render::ComputeIntervalPipelines::new(&device, formats);
    let _shell = rmesh_render::ShellPipeline::new(
        &device,
        formats,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Depth32Float,
    );
    let source = format!(
        "{}\n{}",
        include_str!("../src/wgsl/interval_fragment.wgsl"),
        r#"
@group(0) @binding(6) var<storage, read_write> result: array<vec4f>;
@compute @workgroup_size(1)
fn check_radiance() {
    var input: FragmentInput;
    input.depths = vec2f(0.0, 0.5);
    input.density = 4.0;
    input.base_color = vec3f(4.0, 2.0, 1.0);
    input.coverage = 1.0;
    result[0] = shade_common(vec4f(0.5, 0.5, 0.0, 1.0), input, false).color;
    result[1] = shade_common(vec4f(0.5, 0.5, 0.0, 1.0), input, true).color;
}
"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("radiance_regression"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("check_radiance"),
        compilation_options: Default::default(),
        cache: None,
    });
    let uniforms = rmesh_render::make_uniforms(
        glam::Mat4::IDENTITY,
        glam::Mat3::IDENTITY,
        [1.0, 1.0, 0.5, 0.5],
        glam::Vec3::ZERO,
        1.0,
        1.0,
        1,
        0,
        16,
        0.0,
        0,
        1.0,
        3.0,
    );
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let aux = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0.5f32, 0.04, 0.0, 0.1, 0.2, 0.3]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: aux.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let values: Vec<f32> = read_buffer(&device, &queue, &output, 8);
    let alpha = 1.0 - (-2.0f32).exp();
    for (i, expected) in [
        4.0 * alpha,
        2.0 * alpha,
        alpha,
        alpha,
        0.1 * alpha,
        0.2 * alpha,
        0.3 * alpha,
        alpha,
    ]
    .iter()
    .enumerate()
    {
        assert!(
            (values[i] - expected).abs() < 1e-4,
            "channel {i}: {} != {expected}",
            values[i]
        );
    }
}
