//! Screen-space post-processing for the deferred renderer.
//!
//! Deferred PBR shading plus the screen-space effects layered on top of the
//! G-buffer: GTAO ambient occlusion, a Hi-Z depth pyramid, SSGI, SSR, temporal
//! denoise, and the bilateral blurs. Every pass operates on loose
//! `wgpu::TextureView`s supplied by the caller (the renderer's G-buffer and this
//! crate's own ping-pong targets), so this crate depends only on `rmesh-util`.
//! Extracted from `rmesh-render`.

#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use rmesh_util::gpu_helpers::{record_fullscreen_pass, tex_entry};
use rmesh_util::shared::{GpuLight, MAX_LIGHTS};

// WGSL shader sources, embedded from crate-local files.
static DEFERRED_SHADE_FRAG_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/deferred_shade_frag.wgsl");
static GTAO_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/gtao.wgsl");
static HIZ_LINEARIZE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/hiz_linearize.wgsl");
static HIZ_DOWNSAMPLE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/hiz_downsample.wgsl");
static AO_BILATERAL_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/ao_bilateral.wgsl");
static SSGI_COMPUTE_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/ssgi_compute.wgsl");
static SSGI_BILATERAL_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/ssgi_bilateral.wgsl");
static SSR_COMPUTE_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/ssr_compute.wgsl");
static TEMPORAL_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/temporal.wgsl");

// ---------------------------------------------------------------------------
// PBR deferred shading types
// ---------------------------------------------------------------------------

/// Uniforms for deferred shading and shadow ray generation.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeferredUniforms {
    pub inv_vp: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    pub num_lights: u32,
    pub width: u32,
    pub height: u32,
    pub ambient: f32,
    pub debug_mode: u32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub dsm_enabled: u32,
    pub exposure: f32,
    pub ao_strength: f32,
    pub ssgi_strength: f32,
    pub _pad: [f32; 2],
}

/// Uniforms for the GTAO ambient-occlusion pass. Matches WGSL `GtaoUniforms`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GtaoUniforms {
    pub inv_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub width: u32,
    pub height: u32,
    pub radius_world: f32,
    pub thickness: f32,
    pub proj_scale: f32,
    pub near: f32,
    pub far: f32,
    pub max_mip: u32,
}
// ===========================================================================
// Deferred Shading Pipeline
// ===========================================================================

/// Fullscreen render pass that reads MRT textures (plaster, aux0, normals, depth+albedo)
/// and computes per-pixel lighting, writing the lit result to color_view.
pub struct DeferredShadePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dsm_bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
    pub light_buf: wgpu::Buffer,
}

impl DeferredShadePipeline {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deferred_shade_frag.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DEFERRED_SHADE_FRAG_WGSL.as_str().into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred_shade_bgl"),
            entries: &[
                // 0: DeferredUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1-4: MRT textures
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: lights storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: hardware depth buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 7: AO texture (R8Unorm, written by GTAO pass)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 8: SSGI radiance (Rgba16Float, denoised indirect-diffuse)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 9: lit_history (Rgba16Float, previous frame's deferred output —
                // exposed for debug visualization to verify the SSGI feedback chain)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 10: SSR radiance (Rgba16Float, sampled along reflect(V, N))
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Group 1: DSM shadow atlas (moments cubemap + metadata buffer + sampler)
        let dsm_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("deferred_dsm_bgl"),
                entries: &[
                    // 0: moments cubemap (E[α·z], E[α·z²], 0, α)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 1: per-light shadow metadata
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 2: bilinear sampler for moment textures
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deferred_shade_pl"),
            bind_group_layouts: &[&bind_group_layout, &dsm_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("deferred_shade_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    // location(0): display target — debug-overridable.
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(1): lit_history feedback — always the true lit
                    // value, even when a debug mode is active. Lets SSGI keep
                    // sampling real lighting while the user is staring at debug.
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deferred_uniforms"),
            size: std::mem::size_of::<DeferredUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deferred_lights"),
            size: (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            dsm_bind_group_layout,
            uniforms_buf,
            light_buf,
        }
    }
}

/// Create the deferred shading bind group from MRT texture views.
#[allow(clippy::too_many_arguments)]
pub fn create_deferred_bind_group(
    device: &wgpu::Device,
    deferred: &DeferredShadePipeline,
    color_view: &wgpu::TextureView,
    aux0_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    hw_depth_view: &wgpu::TextureView,
    ao_view: &wgpu::TextureView,
    ssgi_view: &wgpu::TextureView,
    lit_history_view: &wgpu::TextureView,
    ssr_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deferred_shade_bg"),
        layout: &deferred.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: deferred.uniforms_buf.as_entire_binding(),
            },
            tex_entry(1, color_view),
            tex_entry(2, aux0_view),
            tex_entry(3, normals_view),
            tex_entry(4, depth_view),
            wgpu::BindGroupEntry {
                binding: 5,
                resource: deferred.light_buf.as_entire_binding(),
            },
            tex_entry(6, hw_depth_view),
            tex_entry(7, ao_view),
            tex_entry(8, ssgi_view),
            tex_entry(9, lit_history_view),
            tex_entry(10, ssr_view),
        ],
    })
}

/// Create the DSM shadow bind group (group 1) for deferred shading.
pub fn create_deferred_dsm_bind_group(
    device: &wgpu::Device,
    deferred: &DeferredShadePipeline,
    moments_cube_view: &wgpu::TextureView,
    meta_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("dsm_bilinear_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deferred_dsm_bg"),
        layout: &deferred.dsm_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(moments_cube_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    })
}

/// Record the deferred shading render pass: reads MRT textures, writes lit color to `target_view`.
pub fn record_deferred_shade(
    encoder: &mut wgpu::CommandEncoder,
    deferred: &DeferredShadePipeline,
    bind_group: &wgpu::BindGroup,
    dsm_bind_group: &wgpu::BindGroup,
    display_view: &wgpu::TextureView,
    lit_current_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("deferred_shade"),
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view: display_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }),
            // lit_current = this frame's true lit value (location 1). Copied
            // into lit_history after the pass so SSGI samples it next frame.
            // We can't render directly into lit_history because it's also
            // sampled in this same pass for DBG_LIT_HISTORY (and read by SSGI
            // earlier in the frame).
            Some(wgpu::RenderPassColorAttachment {
                view: lit_current_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }),
        ],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    rpass.set_pipeline(&deferred.pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.set_bind_group(1, dsm_bind_group, &[]);
    rpass.draw(0..3, 0..1); // fullscreen triangle
}

// ===========================================================================
// GTAO Pipeline
// ===========================================================================

/// Fullscreen-fragment horizon-based AO pass. Reads hw_depth + normals,
/// writes a single-channel R8Unorm AO factor.
pub struct GtaoPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
}

impl GtaoPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gtao.wgsl"),
            source: wgpu::ShaderSource::Wgsl(GTAO_WGSL.as_str().into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gtao_bgl"),
            entries: &[
                // 0: GtaoUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: Hi-Z (full mip chain, R32Float linear view-space Z)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: world-space normals (raw, premul-alpha)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: volume expected-depth MRT (Rgba16Float; .r = depth*a, .a = a)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gtao_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gtao_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gtao_uniforms"),
            size: std::mem::size_of::<GtaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniforms_buf,
        }
    }
}

/// Create the GTAO bind group from the depth + normals views.
pub fn create_gtao_bind_group(
    device: &wgpu::Device,
    gtao: &GtaoPipeline,
    hiz_full_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
    volume_depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gtao_bg"),
        layout: &gtao.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gtao.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hiz_full_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(normals_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(volume_depth_view),
            },
        ],
    })
}

/// Record the GTAO render pass: reads depth+normals, writes AO into `ao_view`.
pub fn record_gtao_pass(
    encoder: &mut wgpu::CommandEncoder,
    gtao: &GtaoPipeline,
    bind_group: &wgpu::BindGroup,
    ao_view: &wgpu::TextureView,
) {
    // Clear to AO=1 (no occlusion); fragments overwrite where they shade.
    record_fullscreen_pass(
        encoder,
        "gtao_pass",
        &gtao.pipeline,
        bind_group,
        ao_view,
        wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
    );
}

// ===========================================================================
// Hi-Z (hierarchical depth pyramid) over the fused linear-Z buffer.
// Mip 0 fuses hw depth (opaque primitives) and volume's expected-termination
// depth into linear view-space Z. Each coarser mip stores the min() of its
// 2x2 parent — the tightest possible occluder distance for any pixel covered
// by the mip cell. Used by GTAO for cone-style stepping and (later) by SSGI
// for ray-march traversal.
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HizUniforms {
    pub near: f32,
    pub far: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

pub struct HizPipelines {
    pub linearize_pipeline: wgpu::RenderPipeline,
    pub linearize_bgl: wgpu::BindGroupLayout,
    pub downsample_pipeline: wgpu::RenderPipeline,
    pub downsample_bgl: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
}

impl HizPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let linearize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_linearize.wgsl"),
            source: wgpu::ShaderSource::Wgsl(HIZ_LINEARIZE_WGSL.as_str().into()),
        });
        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_downsample.wgsl"),
            source: wgpu::ShaderSource::Wgsl(HIZ_DOWNSAMPLE_WGSL.as_str().into()),
        });

        let linearize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_linearize_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_downsample_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let r32_target = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R32Float,
            blend: None,
            write_mask: wgpu::ColorWrites::RED,
        };

        let lin_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_linearize_pl"),
            bind_group_layouts: &[&linearize_bgl],
            immediate_size: 0,
        });
        let linearize_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hiz_linearize_pipeline"),
            layout: Some(&lin_pl),
            vertex: wgpu::VertexState {
                module: &linearize_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &linearize_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(r32_target.clone())],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let dn_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_downsample_pl"),
            bind_group_layouts: &[&downsample_bgl],
            immediate_size: 0,
        });
        let downsample_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hiz_downsample_pipeline"),
            layout: Some(&dn_pl),
            vertex: wgpu::VertexState {
                module: &downsample_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &downsample_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(r32_target)],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz_uniforms"),
            size: std::mem::size_of::<HizUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            linearize_pipeline,
            linearize_bgl,
            downsample_pipeline,
            downsample_bgl,
            uniforms_buf,
        }
    }
}

/// A hierarchical Z buffer: R32Float, mip-chained.
/// `full_view` exposes all mips for read-side sampling.
/// `mip_views` are single-mip views — one read-only per mip, plus per-mip
/// render-attachment views (same descriptor — wgpu allows binding either
/// role from the same view as long as not simultaneously).
pub struct HizTexture {
    pub texture: wgpu::Texture,
    pub full_view: wgpu::TextureView,
    pub mip_views: Vec<wgpu::TextureView>,
    pub mip_count: u32,
    pub width: u32,
    pub height: u32,
}

impl HizTexture {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // mip count = floor(log2(max(w, h))) + 1, at least 1.
        let max_dim = width.max(height).max(1);
        let mip_count = 32 - max_dim.leading_zeros();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_full_view"),
            ..Default::default()
        });
        let mip_views: Vec<wgpu::TextureView> = (0..mip_count)
            .map(|m| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hiz_mip_view"),
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Self {
            texture,
            full_view,
            mip_views,
            mip_count,
            width,
            height,
        }
    }
}

pub fn create_hiz_linearize_bind_group(
    device: &wgpu::Device,
    hiz: &HizPipelines,
    hw_depth_view: &wgpu::TextureView,
    volume_depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hiz_linearize_bg"),
        layout: &hiz.linearize_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: hiz.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hw_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(volume_depth_view),
            },
        ],
    })
}

pub fn create_hiz_downsample_bind_group(
    device: &wgpu::Device,
    hiz: &HizPipelines,
    parent_mip_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hiz_downsample_bg"),
        layout: &hiz.downsample_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(parent_mip_view),
        }],
    })
}

/// Run the full Hi-Z build: linearize into mip 0, then min-downsample each
/// descendant mip. Caller has already uploaded HizUniforms (near/far) and
/// constructed the linearize bind group + per-mip downsample bind groups.
pub fn record_hiz_pass(
    encoder: &mut wgpu::CommandEncoder,
    hiz: &HizPipelines,
    tex: &HizTexture,
    linearize_bg: &wgpu::BindGroup,
    downsample_bgs: &[wgpu::BindGroup],
) {
    // Mip 0: linearize + fuse hw + volume depth.
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hiz_linearize_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tex.mip_views[0],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0e20,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_pipeline(&hiz.linearize_pipeline);
        rpass.set_bind_group(0, linearize_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    // Mips 1..mip_count: min-downsample from parent.
    // downsample_bgs[i] binds mip i (parent), pass writes to mip i+1.
    for (i, downsample_bg) in downsample_bgs
        .iter()
        .enumerate()
        .take(tex.mip_count as usize - 1)
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hiz_downsample_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tex.mip_views[i + 1],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0e20,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_pipeline(&hiz.downsample_pipeline);
        rpass.set_bind_group(0, downsample_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

// ===========================================================================
// AO bilateral blur — separable depth+normal-aware blur for the GTAO output.
// Two passes (H, V) with the same pipeline, different bind groups (different
// I/O views and a `direction` uniform).
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AoBlurUniforms {
    pub dir_x: i32, // (1, 0) for horizontal pass, (0, 1) for vertical
    pub dir_y: i32,
    pub sigma_z: f32, // depth tolerance (world units)
    pub sigma_n: f32, // normal cosine power (e.g. 8 = cos^8)
}

pub struct AoBlurPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_h: wgpu::Buffer,
    pub uniforms_v: wgpu::Buffer,
}

impl AoBlurPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ao_bilateral.wgsl"),
            source: wgpu::ShaderSource::Wgsl(AO_BILATERAL_WGSL.as_str().into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ao_blur_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ao_blur_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ao_blur_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let mk_uniforms = |label: &'static str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<AoBlurUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniforms_h = mk_uniforms("ao_blur_uniforms_h");
        let uniforms_v = mk_uniforms("ao_blur_uniforms_v");

        Self {
            pipeline,
            bind_group_layout: bgl,
            uniforms_h,
            uniforms_v,
        }
    }
}

pub fn create_ao_blur_bind_group(
    device: &wgpu::Device,
    blur: &AoBlurPipeline,
    uniforms_buf: &wgpu::Buffer,
    ao_in_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ao_blur_bg"),
        layout: &blur.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(ao_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(normals_view),
            },
        ],
    })
}

/// Run one bilateral blur pass: read AO from `bind_group`'s input, write to
/// `out_view`. Caller has already uploaded the per-pass `BlurUniforms`.
pub fn record_ao_blur_pass(
    encoder: &mut wgpu::CommandEncoder,
    blur: &AoBlurPipeline,
    bind_group: &wgpu::BindGroup,
    out_view: &wgpu::TextureView,
) {
    record_fullscreen_pass(
        encoder,
        "ao_blur_pass",
        &blur.pipeline,
        bind_group,
        out_view,
        wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
    );
}

// ===========================================================================
// SSGI — diffuse screen-space global illumination.
// Ray-marches Hi-Z, samples lit_history at hits, falls back to hemi color
// for misses. Output is averaged radiance L_in (Rgba16Float). Caller's
// deferred shader multiplies by `kd · albedo` to complete the bounce.
// Includes a separate Rgba16Float bilateral denoiser (mirrors AO blur).
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsgiUniforms {
    pub inv_proj: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub width: u32,
    pub height: u32,
    pub near: f32,
    pub far: f32,
    pub max_mip: u32,
    pub frame: u32,
    pub radius_world: f32,
    pub thickness: f32,
    pub sky_color: [f32; 3],
    pub _pad0: f32,
    pub ground_color: [f32; 3],
    pub _pad1: f32,
}

pub struct SsgiPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
}

impl SsgiPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SSGI_COMPUTE_WGSL.as_str().into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssgi_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi_uniforms"),
            size: std::mem::size_of::<SsgiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout: bgl,
            uniforms_buf,
        }
    }
}

pub fn create_ssgi_bind_group(
    device: &wgpu::Device,
    ssgi: &SsgiPipeline,
    hiz_full_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
    lit_history_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssgi_bg"),
        layout: &ssgi.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ssgi.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hiz_full_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(normals_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(lit_history_view),
            },
        ],
    })
}

pub fn record_ssgi_pass(
    encoder: &mut wgpu::CommandEncoder,
    ssgi: &SsgiPipeline,
    bind_group: &wgpu::BindGroup,
    out_view: &wgpu::TextureView,
) {
    record_fullscreen_pass(
        encoder,
        "ssgi_pass",
        &ssgi.pipeline,
        bind_group,
        out_view,
        wgpu::Color::TRANSPARENT,
    );
}

// ---- SSGI bilateral (Rgba16Float; same shape as AoBlurPipeline) ----

/// The SSGI bilateral blur uses the same direction + sigma layout as the AO blur.
pub type SsgiBlurUniforms = AoBlurUniforms;

pub struct SsgiBlurPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_h: wgpu::Buffer,
    pub uniforms_v: wgpu::Buffer,
}

impl SsgiBlurPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_bilateral.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SSGI_BILATERAL_WGSL.as_str().into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_blur_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_blur_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssgi_blur_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let mk = |label: &'static str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<SsgiBlurUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniforms_h = mk("ssgi_blur_uniforms_h");
        let uniforms_v = mk("ssgi_blur_uniforms_v");

        Self {
            pipeline,
            bind_group_layout: bgl,
            uniforms_h,
            uniforms_v,
        }
    }
}

pub fn create_ssgi_blur_bind_group(
    device: &wgpu::Device,
    blur: &SsgiBlurPipeline,
    uniforms_buf: &wgpu::Buffer,
    ssgi_in_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssgi_blur_bg"),
        layout: &blur.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(ssgi_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(normals_view),
            },
        ],
    })
}

pub fn record_ssgi_blur_pass(
    encoder: &mut wgpu::CommandEncoder,
    blur: &SsgiBlurPipeline,
    bind_group: &wgpu::BindGroup,
    out_view: &wgpu::TextureView,
) {
    record_fullscreen_pass(
        encoder,
        "ssgi_blur_pass",
        &blur.pipeline,
        bind_group,
        out_view,
        wgpu::Color::TRANSPARENT,
    );
}

// ===========================================================================
// SSR — single Hi-Z ray-march along reflect(view, N). Sharp specular
// reflections for low-roughness materials; rougher surfaces fall back to
// the SSGI hemi-average via the deferred shader's smoothness² mix.
// ===========================================================================

/// SSR shares the SSGI uniform layout byte-for-byte (same camera matrices,
/// ray-march params, and sky/ground fallback colors), so it reuses the type.
pub type SsrUniforms = SsgiUniforms;

pub struct SsrPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
}

impl SsrPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SSR_COMPUTE_WGSL.as_str().into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: SSGI view — used as the soft miss fallback so single-ray
                // SSR misses don't visually pop against hits.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: Volume depth MRT (E[z]·α, 0, E[z²]·α, α) — drives the
                // per-pixel σ used to jitter the reflection origin.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssr_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr_uniforms"),
            size: std::mem::size_of::<SsrUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout: bgl,
            uniforms_buf,
        }
    }
}

pub fn create_ssr_bind_group(
    device: &wgpu::Device,
    ssr: &SsrPipeline,
    hiz_full_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
    lit_history_view: &wgpu::TextureView,
    ssgi_view: &wgpu::TextureView,
    volume_depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssr_bg"),
        layout: &ssr.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ssr.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hiz_full_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(normals_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(lit_history_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(ssgi_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(volume_depth_view),
            },
        ],
    })
}

pub fn record_ssr_pass(
    encoder: &mut wgpu::CommandEncoder,
    ssr: &SsrPipeline,
    bind_group: &wgpu::BindGroup,
    out_view: &wgpu::TextureView,
) {
    record_fullscreen_pass(
        encoder,
        "ssr_pass",
        &ssr.pipeline,
        bind_group,
        out_view,
        wgpu::Color::TRANSPARENT,
    );
}

// ===========================================================================
// Temporal accumulation — shared shader for SSGI and AO.
// Reprojects history via prev_vp, neighborhood-clamps, blends with current.
// One pipeline per output format (Rgba16Float for SSGI, R8Unorm for AO).
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TemporalUniforms {
    pub inv_vp: [[f32; 4]; 4],
    pub prev_vp: [[f32; 4]; 4],
    pub width: u32,
    pub height: u32,
    pub near: f32,
    pub far: f32,
    pub max_mip: u32,
    pub alpha: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

pub struct TemporalPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
}

impl TemporalPipeline {
    /// Build a temporal-accumulation pipeline. `target_format` is
    /// `Rgba16Float` for SSGI, `R8Unorm` for AO.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("temporal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TEMPORAL_WGSL.as_str().into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("temporal_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: current
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
                // 2: history (sampled bilinearly via the sampler in slot 4)
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
                // 3: Hi-Z (for current-pixel depth → world-pos reconstruction)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: bilinear sampler for history reprojection
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("temporal_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("temporal_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temporal_uniforms"),
            size: std::mem::size_of::<TemporalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("temporal_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout: bgl,
            uniforms_buf,
            sampler,
        }
    }
}

pub fn create_temporal_bind_group(
    device: &wgpu::Device,
    pipeline: &TemporalPipeline,
    current_view: &wgpu::TextureView,
    history_view: &wgpu::TextureView,
    hiz_full_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("temporal_bg"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pipeline.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(current_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(history_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(hiz_full_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&pipeline.sampler),
            },
        ],
    })
}

pub fn record_temporal_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &TemporalPipeline,
    bind_group: &wgpu::BindGroup,
    out_view: &wgpu::TextureView,
) {
    record_fullscreen_pass(
        encoder,
        "temporal_pass",
        &pipeline.pipeline,
        bind_group,
        out_view,
        wgpu::Color::TRANSPARENT,
    );
}

/// Clear a texture to a constant color via a load-op-only render pass.
/// Used at startup to make history textures (lit_history, ssgi_history,
/// ao_history) deterministic on the first frame.
pub fn clear_texture_view(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    color: wgpu::Color,
) {
    let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear_history_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
}
