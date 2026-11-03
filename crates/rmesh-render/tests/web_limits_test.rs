//! Validate the baked viewer pipelines against WebGPU's baseline storage limit.
#[test]
fn baked_viewer_pipelines_fit_eight_storage_buffers() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else { return; };
        let (device, _) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: 8, ..Default::default() },
            ..Default::default()
        }).await.expect("baseline eight-buffer device");
        let formats = rmesh_render::GBufferFormats { color:wgpu::TextureFormat::Rgba16Float, ..Default::default() };
        let _forward=rmesh_render::ForwardPipelines::new_baked(&device,formats);
        let _interval=rmesh_render::ComputeIntervalPipelines::new_baked(&device,formats);
        let _shell=rmesh_render::ShellPipeline::new_baked(&device,formats,wgpu::TextureFormat::Rgba16Float,wgpu::TextureFormat::Depth32Float);
    });
}
