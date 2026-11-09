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

/// Same float32 table and bilinear coordinates; use hardware interpolation
/// only when the device supports FLOAT32_FILTERABLE. Caller binds energy_sampler.
pub fn shader_source(filtered: bool) -> String {
    if !filtered {
        return WGSL.to_owned();
    }
    format!(
        "{}\n{}",
        WGSL.replace("fn energy_lookup(", "fn energy_lookup_manual("),
        r#"
fn energy_lookup(roughness:f32,cosine:f32)->vec4f {
    let xy=vec2f(clamp(cosine,0.0,1.0),(clamp(roughness,0.04,1.0)-0.04)/0.96)*64.0;
    return textureSampleLevel(energy_lut,energy_sampler,(xy+vec2f(0.5))/65.0,0.0);
}
"#
    )
}
