//! Per-pixel overdraw probe and heatmap.
//!
//! `FRAGMENT_SHADER_INVOCATIONS` gives the frame total; this gives the spatial
//! distribution behind it, which is what says whether the overdraw is a few
//! deep hot spots (a cull can reach it) or spread evenly over the frame (it
//! cannot).
//!
//! The count pass replays the interval draw with the same vertex stage, depth
//! state and indirect args, swapping only the fragment entry point, so what it
//! measures is exactly what the real pass shades. It runs only while the view is
//! on, and costs one extra 32-bpp pass over the same geometry.

/// Fragment count + summed per-fragment alpha. Rg16Float is blendable (R32Float
/// is not, in WebGPU) and represents integers exactly to 2048.
pub const OVERDRAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

static OVERDRAW_VIZ_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/overdraw_viz.wgsl");

/// Which quantity the heatmap shows. Matches `VizUniforms.mode` in the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OverdrawView {
    /// Fragments shaded per pixel, against `scale`.
    #[default]
    Count,
    /// Final blended alpha — how opaque the ray actually ended up.
    Alpha,
    /// Fraction of deposited opacity the ray could not use.
    Wasted,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VizUniforms {
    pub scale: f32,
    pub mode: u32,
    pub _pad: [u32; 2],
}

pub struct OverdrawPipelines {
    pub viz_pipeline: wgpu::RenderPipeline,
    pub viz_bgl: wgpu::BindGroupLayout,
}

impl OverdrawPipelines {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overdraw_viz.wgsl"),
            source: wgpu::ShaderSource::Wgsl(OVERDRAW_VIZ_WGSL.as_str().into()),
        });

        let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let viz_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overdraw_viz_bgl"),
            entries: &[
                tex(0),
                tex(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let viz_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overdraw_viz_layout"),
            bind_group_layouts: &[&viz_bgl],
            immediate_size: 0,
        });

        let viz_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overdraw_viz_pipeline"),
            layout: Some(&viz_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            viz_pipeline,
            viz_bgl,
        }
    }
}

/// Screen-sized probe target. Recreate on resize, along with its bind group.
pub struct OverdrawTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub viz_bg: wgpu::BindGroup,
    pub uniforms: wgpu::Buffer,
}

impl OverdrawTarget {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &OverdrawPipelines,
        gbuf_color_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overdraw_probe"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OVERDRAW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overdraw_viz_uniforms"),
            size: std::mem::size_of::<VizUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let viz_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overdraw_viz_bg"),
            layout: &pipelines.viz_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(gbuf_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });

        Self {
            texture,
            view,
            viz_bg,
            uniforms,
        }
    }
}

/// Record the heatmap over `target_view`. The probe pass must already have run
/// this frame.
pub fn record_overdraw_viz(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &OverdrawPipelines,
    target: &OverdrawTarget,
    target_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("overdraw_viz"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        ..Default::default()
    });
    rpass.set_pipeline(&pipelines.viz_pipeline);
    rpass.set_bind_group(0, &target.viz_bg, &[]);
    rpass.draw(0..3, 0..1);
}

/// Replay the interval draw into the probe target. Record immediately after the
/// interval render pass, with the same `ci_render_bg` and depth view, so the
/// depth test rejects exactly what it rejected there.
pub fn record_overdraw_count(
    encoder: &mut wgpu::CommandEncoder,
    ci_pipelines: &crate::ComputeIntervalPipelines,
    buffers: &crate::SceneBuffers,
    targets: &crate::RenderTargets,
    ci_render_bg: &wgpu::BindGroup,
    depth_view: &wgpu::TextureView,
    probe: &OverdrawTarget,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("overdraw_count"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &probe.view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    rpass.set_viewport(
        0.0,
        0.0,
        targets.width as f32,
        targets.height as f32,
        0.0,
        1.0,
    );
    rpass.set_scissor_rect(0, 0, targets.width, targets.height);
    rpass.set_pipeline(&ci_pipelines.overdraw_pipeline);
    rpass.set_bind_group(0, ci_render_bg, &[]);
    rpass.set_index_buffer(
        buffers.interval_fan_index_buf.slice(..),
        wgpu::IndexFormat::Uint32,
    );
    rpass.draw_indexed_indirect(&buffers.interval_args_buf, 12);
}
