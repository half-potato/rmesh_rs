//! Canonical point material evaluated inside each camera-ray interval.
use rmesh_data::{PbrData, SceneBuffers};
use wgpu::util::DeviceExt;

pub struct EnergyForward {
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    filtered: bool,
}
impl EnergyForward {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pbr: &PbrData,
        buffers: &SceneBuffers,
        transport: &wgpu::Buffer,
        snapshot: &wgpu::Buffer,
    ) -> Self {
        let mut material = Vec::<f32>::with_capacity(pbr.energy_material.len() / 5 * 8);
        for (i, m) in pbr.energy_material.chunks_exact(5).enumerate() {
            material.extend_from_slice(&pbr.albedo[3 * i..3 * i + 3]);
            material.push(m[0]);
            material.extend_from_slice(&m[1..]);
        }
        let material = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("canonical energy materials"),
            contents: bytemuck::cast_slice(&material),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lut = crate::energy::upload_bytes(device, queue, bytemuck::cast_slice(&pbr.energy_lut));
        let filtered = device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let mut entries = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 3 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let mut entry = crate::energy::layout_entry(4);
        entry.visibility = wgpu::ShaderStages::FRAGMENT;
        if filtered {
            entry.ty = wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            };
        }
        entries.push(entry);
        if filtered {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("energy interval"),
            entries: &entries,
        });
        let mut resources = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: material.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: transport.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: snapshot.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&lut),
            },
        ];
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        if filtered {
            resources.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&sampler),
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("energy interval"),
            layout: &layout,
            entries: &resources,
        });
        Self {
            layout,
            group,
            filtered,
        }
    }
    pub fn configure(
        &self,
        device: &wgpu::Device,
        pipelines: &mut rmesh_render::ComputeIntervalPipelines,
    ) {
        let source = format!(
            "{}\n{}\n{}",
            crate::energy::shader_source(self.filtered),
            if self.filtered {
                "@group(1) @binding(5) var energy_sampler:sampler;"
            } else {
                ""
            },
            include_str!("wgsl/energy_interval.wgsl")
        );
        pipelines.set_point_samples(device, true);
        pipelines.set_color_shader(device, &self.layout, &source, self.group.clone());
    }
}
