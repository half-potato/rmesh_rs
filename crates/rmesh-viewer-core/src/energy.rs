//! Direct point-light material matching sdir's `material_model="energy"`.
//! The LUT is copied losslessly from sdir; no fitted/analytic replacement.
pub const WGSL: &str = include_str!("wgsl/energy.wgsl");
pub const LUT: &[u8] = include_bytes!("assets/ggx_energy.rgba32f");

pub fn layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    upload_bytes(device, queue, LUT)
}

pub fn upload_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::TextureView {
    assert_eq!(bytes.len(), 65 * 65 * 16);
    use wgpu::util::DeviceExt;
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("sdir GGX energy LUT"),
            size: wgpu::Extent3d {
                width: 65,
                height: 65,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytes,
    );
    texture.create_view(&Default::default())
}
