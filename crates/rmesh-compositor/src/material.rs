//! PBR material texture management for primitives.
//!
//! Each material has up to 4 textures (base_color, metallic_roughness, normal, occlusion)
//! bound as a single bind group. Materials without a given texture use a 1x1 default.

use wgpu::util::DeviceExt;

/// Per-material PBR properties (scalar factors + texture flags).
#[derive(Debug, Clone)]
pub struct PbrMaterial {
    pub base_color_factor: [f32; 4],
    pub roughness_factor: f32,
    pub metallic_factor: f32,
    pub occlusion_strength: f32,
    pub normal_scale: f32,
    pub has_base_color_tex: bool,
    pub has_metallic_roughness_tex: bool,
    pub has_normal_tex: bool,
    pub has_occlusion_tex: bool,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            roughness_factor: 1.0,
            metallic_factor: 0.0,
            occlusion_strength: 1.0,
            normal_scale: 1.0,
            has_base_color_tex: false,
            has_metallic_roughness_tex: false,
            has_normal_tex: false,
            has_occlusion_tex: false,
        }
    }
}

impl PbrMaterial {
    /// Pack texture presence flags into a u32 bitmask.
    pub fn tex_flags(&self) -> u32 {
        let mut flags = 0u32;
        if self.has_base_color_tex {
            flags |= 1;
        }
        if self.has_metallic_roughness_tex {
            flags |= 2;
        }
        if self.has_normal_tex {
            flags |= 4;
        }
        if self.has_occlusion_tex {
            flags |= 8;
        }
        flags
    }
}

/// GPU bind group for a single material's textures.
pub struct MaterialTextures {
    pub bind_group: wgpu::BindGroup,
    pub properties: PbrMaterial,
}

/// Loaded texture data for GPU upload (always RGBA8).
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Material definition referencing textures by index.
pub struct MaterialDef {
    pub base_color_factor: [f32; 4],
    pub roughness_factor: f32,
    pub metallic_factor: f32,
    pub occlusion_strength: f32,
    pub normal_scale: f32,
    pub base_color_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
}

/// Registry of GPU materials with their texture bind groups.
pub struct MaterialRegistry {
    pub materials: Vec<MaterialTextures>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub default_bind_group: wgpu::BindGroup,
    // Default 1x1 textures
    default_base_color: wgpu::TextureView,
    default_metallic_roughness: wgpu::TextureView,
    default_normal: wgpu::TextureView,
    default_occlusion: wgpu::TextureView,
}

fn create_1x1_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    rgba: [u8; 4],
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &rgba,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_1x1_texture_linear(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    rgba: [u8; 4],
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &rgba,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl MaterialRegistry {
    /// Create a new registry with default textures and bind group layout.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_bgl"),
            entries: &[
                // binding 0: base_color texture (sRGB)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: metallic_roughness texture (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: normal texture (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: occlusion texture (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Default 1x1 textures
        let (_bc_tex, default_base_color) =
            create_1x1_texture(device, queue, "default_base_color", [255, 255, 255, 255]);
        // metallic_roughness: G=roughness(1.0=255), B=metallic(0.0=0)
        let (_mr_tex, default_metallic_roughness) =
            create_1x1_texture_linear(device, queue, "default_metal_rough", [255, 255, 0, 255]);
        // Normal: tangent-space up = (0.5, 0.5, 1.0) encoded as (128, 128, 255)
        let (_nm_tex, default_normal) =
            create_1x1_texture_linear(device, queue, "default_normal", [128, 128, 255, 255]);
        // Occlusion: R=1.0 = no occlusion
        let (_ao_tex, default_occlusion) =
            create_1x1_texture_linear(device, queue, "default_occlusion", [255, 255, 255, 255]);

        let default_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("default_material_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_base_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&default_metallic_roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&default_normal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&default_occlusion),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            materials: Vec::new(),
            bind_group_layout,
            sampler,
            default_bind_group,
            default_base_color,
            default_metallic_roughness,
            default_normal,
            default_occlusion,
        }
    }

    /// Upload textures and create per-material bind groups.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &[TextureData],
        materials: &[MaterialDef],
    ) {
        // Upload all textures to GPU
        let gpu_views: Vec<wgpu::TextureView> = textures
            .iter()
            .enumerate()
            .map(|(i, tex)| {
                // Determine if this texture is used as base_color (sRGB) or linear data.
                // For simplicity, use Rgba8Unorm for all — the shader handles sRGB conversion
                // via the base_color_factor multiplication. This avoids needing to track which
                // format each image should be.
                let gpu_tex = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some(&format!("material_tex_{}", i)),
                        size: wgpu::Extent3d {
                            width: tex.width,
                            height: tex.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &tex.pixels,
                );
                gpu_tex.create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        // Create per-material bind groups
        self.materials = materials
            .iter()
            .map(|mat| {
                let bc_view = mat
                    .base_color_texture
                    .and_then(|i| gpu_views.get(i))
                    .unwrap_or(&self.default_base_color);
                let mr_view = mat
                    .metallic_roughness_texture
                    .and_then(|i| gpu_views.get(i))
                    .unwrap_or(&self.default_metallic_roughness);
                let nm_view = mat
                    .normal_texture
                    .and_then(|i| gpu_views.get(i))
                    .unwrap_or(&self.default_normal);
                let ao_view = mat
                    .occlusion_texture
                    .and_then(|i| gpu_views.get(i))
                    .unwrap_or(&self.default_occlusion);

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("material_bg"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(bc_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(mr_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(nm_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(ao_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                let properties = PbrMaterial {
                    base_color_factor: mat.base_color_factor,
                    roughness_factor: mat.roughness_factor,
                    metallic_factor: mat.metallic_factor,
                    occlusion_strength: mat.occlusion_strength,
                    normal_scale: mat.normal_scale,
                    has_base_color_tex: mat.base_color_texture.is_some(),
                    has_metallic_roughness_tex: mat.metallic_roughness_texture.is_some(),
                    has_normal_tex: mat.normal_texture.is_some(),
                    has_occlusion_tex: mat.occlusion_texture.is_some(),
                };

                MaterialTextures {
                    bind_group,
                    properties,
                }
            })
            .collect();
    }

    /// Clear all uploaded materials.
    pub fn clear(&mut self) {
        self.materials.clear();
    }

    /// Get the bind group for a material index, or the default.
    pub fn bind_group_for(&self, material_index: Option<usize>) -> &wgpu::BindGroup {
        material_index
            .and_then(|i| self.materials.get(i))
            .map(|m| &m.bind_group)
            .unwrap_or(&self.default_bind_group)
    }

    /// Get PBR properties for a material index, or default.
    pub fn properties_for(&self, material_index: Option<usize>) -> PbrMaterial {
        material_index
            .and_then(|i| self.materials.get(i))
            .map(|m| m.properties.clone())
            .unwrap_or_default()
    }
}
