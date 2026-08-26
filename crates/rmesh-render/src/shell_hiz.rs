//! Max-reduce Hi-Z pyramid over the shell (and opaque-primitive) depth buffer.
//!
//! `project_compute*` uses this to drop tets **before** the radix sort, which is
//! where the win is — the sort is the single largest slice of the interval
//! frame, and a fragment-level depth test does nothing to shrink it.
//!
//! The reduction is `max`, not `min`: culling is only safe when a tet's nearest
//! point is behind the *farthest* occluder anywhere in its screen footprint.
//! Cleared texels stay at 1.0 and max() to 1.0, so any footprint touching
//! background culls nothing.
//!
//! This is a deliberate near-duplicate of `rmesh-postprocess`'s Hi-Z. That one
//! is built after the volume pass (and linearizes to view-space Z for SSGI/SSR),
//! so it is both a frame late and the wrong reduction for this use. Crate order
//! also forbids depending on it from here.

static SHELL_HIZ_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/shell_hiz_max.wgsl");

pub struct ShellHiZPipelines {
    pub init_pipeline: wgpu::RenderPipeline,
    pub downsample_pipeline: wgpu::RenderPipeline,
    pub init_bgl: wgpu::BindGroupLayout,
    pub downsample_bgl: wgpu::BindGroupLayout,
}

const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

impl ShellHiZPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shell_hiz_max.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHELL_HIZ_WGSL.as_str().into()),
        });

        let init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shell_hiz_init_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shell_hiz_downsample_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let make = |label: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
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
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HIZ_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        Self {
            init_pipeline: make("shell_hiz_init", &init_bgl, "fs_init"),
            downsample_pipeline: make("shell_hiz_downsample", &downsample_bgl, "fs_downsample"),
            init_bgl,
            downsample_bgl,
        }
    }
}

/// The pyramid itself, plus the bind groups that build it.
///
/// Size-dependent — recreate on resize, and rebuild every bind group that holds
/// [`ShellHiZ::full_view`] along with it.
pub struct ShellHiZ {
    pub texture: wgpu::Texture,
    /// All-mips view. This is what `project_compute*` samples.
    pub full_view: wgpu::TextureView,
    mip_views: Vec<wgpu::TextureView>,
    init_bg: wgpu::BindGroup,
    downsample_bgs: Vec<wgpu::BindGroup>,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
}

impl ShellHiZ {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &ShellHiZPipelines,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mip_count = 32 - width.max(height).leading_zeros();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shell_hiz"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shell_hiz_full"),
            ..Default::default()
        });

        let mip_views: Vec<_> = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shell_hiz_mip"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let init_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shell_hiz_init_bg"),
            layout: &pipelines.init_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            }],
        });

        // downsample_bgs[i] reads mip i; the pass writes mip i+1.
        let downsample_bgs: Vec<_> = (0..mip_count.saturating_sub(1))
            .map(|mip| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shell_hiz_downsample_bg"),
                    layout: &pipelines.downsample_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mip_views[mip as usize]),
                    }],
                })
            })
            .collect();

        Self {
            texture,
            full_view,
            mip_views,
            init_bg,
            downsample_bgs,
            width,
            height,
            mip_count,
        }
    }
}

/// A 1x1 pyramid holding the far plane, for paths that never build a real one
/// (tests, the training harness, the forward-only helper).
///
/// It reads 1.0, so `shell_occluded` finds no occluder and culls nothing even if
/// the enable flag is set. Note this must be written explicitly: a fresh texture
/// zero-initializes, and 0.0 is the *near* plane — it would cull everything.
pub fn dummy_shell_hiz(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shell_hiz_dummy"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HIZ_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::bytes_of(&1.0f32),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Build the pyramid. Record after the shell pass and before
/// `record_compute_interval_project`, so the cull uses this frame's bound.
pub fn record_shell_hiz(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ShellHiZPipelines,
    hiz: &ShellHiZ,
) {
    // Clearing to 1.0 (the far plane) means any texel a pass fails to write
    // culls nothing.
    let clear = wgpu::Operations {
        load: wgpu::LoadOp::Clear(wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }),
        store: wgpu::StoreOp::Store,
    };

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shell_hiz_init"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &hiz.mip_views[0],
                resolve_target: None,
                ops: clear,
                depth_slice: None,
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipelines.init_pipeline);
        pass.set_bind_group(0, &hiz.init_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    for (i, bg) in hiz.downsample_bgs.iter().enumerate() {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shell_hiz_downsample"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &hiz.mip_views[i + 1],
                resolve_target: None,
                ops: clear,
                depth_slice: None,
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipelines.downsample_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
