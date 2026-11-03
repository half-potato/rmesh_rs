//! The portable diagnostic path must retain the full ray tracer's RGB integral.
use wgpu::util::DeviceExt;
use glam::{Mat3, Mat4, Vec3};
use rmesh_util::test_util::{build_test_scene, create_test_device_default, read_buffer};

#[test]
fn camera_color_only_matches_full_raytrace() {
    let Some((device, queue)) = create_test_device_default() else { return; };
    let scene = build_test_scene(
        vec![-1., -1., 2., 1., -1., 2., 0., 1., 2., 0., 0., 4.],
        vec![0, 1, 2, 3], vec![2.], vec![0.1, 0.2, 0.3],
    );
    let buffers = rmesh_data::SceneBuffers::upload(&device, &queue, &scene);
    let mut material = rmesh_data::MaterialBuffers::upload(&device, &[2., 1., 0.5], &scene.color_grads, 1);
    material.colors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cached baked radiance"), contents: bytemuck::cast_slice(&[2f32,1.,0.5]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let neighbors = rmesh_raytrace::compute_tet_neighbors(&scene.indices, 1);
    let bvh = rmesh_raytrace::build_boundary_bvh(&scene.vertices, &scene.indices, &neighbors, 1);
    let rt_buffers = rmesh_raytrace::RayTraceBuffers::new(&device, &neighbors, &bvh);
    queue.write_buffer(&rt_buffers.start_tet, 0, bytemuck::bytes_of(&-1i32));
    let uniforms = rmesh_render::make_uniforms(
        Mat4::IDENTITY, Mat3::IDENTITY, [16.,16.,16.,16.], Vec3::ZERO,
        32.,32.,1,0,12,0.,0,0.01,100.,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));
    let mut images = Vec::new();
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("opaque depth"), size: wgpu::Extent3d { width:32,height:32,depth_or_array_layers:1 },
        mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,format:wgpu::TextureFormat::Depth32Float,
        usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::RENDER_ATTACHMENT,view_formats:&[],
    });
    let depth_view=depth.create_view(&Default::default());
    for variant in 0..4 {
        let color_only=variant>0;
        let pipeline = if variant>=2 { rmesh_raytrace::RayTracePipeline::new_color_only_with_depth(&device,32,32,&depth_view) }
            else if color_only { rmesh_raytrace::RayTracePipeline::new_color_only(&device,32,32) }
            else { rmesh_raytrace::RayTracePipeline::new(&device,32,32,0) };
        if color_only { assert_eq!(pipeline.aux_image.size(), 4); }
        let bg = rmesh_raytrace::create_raytrace_bind_group(&device,&pipeline,&buffers,&material,&rt_buffers);
        let mut encoder = device.create_command_encoder(&Default::default());
        if variant>=2 {
            let _pass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment {view:&depth_view,
                    depth_ops:Some(wgpu::Operations {load:wgpu::LoadOp::Clear(if variant==3 {0.0} else {1.0}),store:wgpu::StoreOp::Store}),stencil_ops:None}),
                ..Default::default()
            });
        }
        rmesh_raytrace::record_raytrace(&mut encoder,&pipeline,&bg,32,32);
        queue.submit([encoder.finish()]);
        images.push(read_buffer::<f32>(&device,&queue,&pipeline.rendered_image,32*32*4));
    }
    assert!(images[0].iter().any(|v| *v > 1.));
    assert!(images[0].chunks_exact(4).any(|p| p[3] == 0.));
    assert!(images[3].iter().all(|v| *v == 0.0), "Opaque near depth must block all radiance");
    for image in &images[1..3] {
    for (a,b) in images[0].iter().zip(image) {
        assert!(a.is_finite() && b.is_finite() && (a-b).abs() <= 1e-6, "{a} != {b}");
    }
    }
}

#[test]
fn color_only_pipeline_fits_browser_limits() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else { return; };
        let (device, _) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: 10, ..Default::default() },
            ..Default::default()
        }).await.expect("ten-storage-buffer device");
        let _pipeline = rmesh_raytrace::RayTracePipeline::new_color_only(&device, 32, 32);
    });
}
