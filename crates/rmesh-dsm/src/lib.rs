//! Deep shadow map rendering for tetrahedral radiance meshes.
//!
//! Stores a 2-moment α-weighted depth distribution per cubemap texel in a
//! single `Rgba16Float` target per face:
//!
//! ```text
//!   .r = alpha * E[z]    (mean depth,    normalized to [0,1] over [near,far])
//!   .g = alpha * E[z^2]  (second moment)
//!   .b = unused
//!   .a = alpha           (occlusion; accumulates to 1 - T_total)
//! ```
//!
//! Hardware premultiplied-alpha blending (back-to-front) accumulates each
//! channel. The deferred shading pass reconstructs transmittance `T(z)` via the
//! Cantelli (one-sided Chebyshev) bound (see `evaluate_transmittance` in
//! `deferred_shade_frag.wgsl`); [`DsmResolvePipeline`] does the same for debug
//! visualization. See `MSM.md` for the math.
//!
//! # Pipeline flow
//!
//! The DSM pipeline reuses the compute-interval generation and indirect-convert
//! stages from [`rmesh_render::ComputeIntervalPipelines`], then draws with its
//! own lightweight render pipeline:
//!
//! 1. `interval_compute.wgsl` — tet → screen triangles (reused)
//! 2. `interval_indirect_convert.wgsl` — dispatch/draw args (reused)
//! 3. `dsm_moment_fragment.wgsl` — α-weighted moments (this crate)
//!
//! # Output
//!
//! A single Rgba16Float MRT per face holds `(E[α·z], E[α·z²], 0, α)` in
//! normalized depth space `[0,1]`, premul-alpha blended back-to-front. The
//! deferred sampler reads `m.a` (= 1 − T_total), `m.r` (= α·μ), `m.g` for the
//! variance, and clamps T at `(1 − α)` to bound by total transmittance.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use rmesh_compositor::{PrimitiveGeometry, PrimitiveVertex};
use rmesh_interact::Primitive;
use rmesh_render::{
    dispatch_2d, ComputeIntervalPipelines, ForwardPipelines, GpuLight, MaterialBuffers,
    SceneBuffers, Uniforms,
};

static INTERVAL_VERTEX_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/interval_vertex.wgsl");
static DSM_MOMENT_FRAGMENT_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/dsm_moment_fragment.wgsl");
static DSM_PRIMITIVE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/dsm_primitive.wgsl");
static DSM_RESOLVE_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/dsm_resolve.wgsl");

/// DSM moments texture format. Stores `(E[α·z], E[α·z²], 0, α)` per texel.
pub const DSM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Shorthand for a full-buffer bind group entry.
fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Render pipeline for deep shadow map generation.
///
/// Uses `interval_vertex.wgsl` (shared with the full interval pipeline) and
/// `dsm_moment_fragment.wgsl` to write the single-MRT (E[α·z], E[α·z²], 0, α)
/// moments texture.
pub struct DsmPipeline {
    pub render_pipeline: wgpu::RenderPipeline,
    pub render_bg_layout: wgpu::BindGroupLayout,
}

impl DsmPipeline {
    /// Create the DSM render pipeline.
    ///
    /// `color_format` should match the output texture (e.g. `Rgba16Float`).
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // 3 read-only storage bindings: uniforms, verts, tet_data
        let render_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3u32)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let render_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dsm_render_bgl"),
            entries: &render_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dsm_render_pl"),
            bind_group_layouts: &[&render_bg_layout],
            immediate_size: 0,
        });

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_interval_vertex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_VERTEX_WGSL.as_str().into()),
        });

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_moment_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_MOMENT_FRAGMENT_WGSL.as_str().into()),
        });

        // Premultiplied alpha blend for α-weighted moment compositing.
        let premul_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let moment_target = Some(wgpu::ColorTargetState {
            format: color_format,
            blend: Some(premul_blend),
            write_mask: wgpu::ColorWrites::ALL,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dsm_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("main"),
                targets: &[moment_target], // (alpha*E[z], alpha*E[z²], _, alpha)
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            render_pipeline,
            render_bg_layout,
        }
    }
}

/// Create the DSM render bind group (3 storage bindings).
///
/// Binding order matches `interval_vertex.wgsl`:
///   0: uniforms, 1: interval_vertex_buf, 2: interval_tet_data_buf
pub fn create_dsm_render_bind_group(
    device: &wgpu::Device,
    pipeline: &DsmPipeline,
    uniforms: &wgpu::Buffer,
    interval_vertex_buf: &wgpu::Buffer,
    interval_tet_data_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_render_bg"),
        layout: &pipeline.render_bg_layout,
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, interval_vertex_buf),
            buf_entry(2, interval_tet_data_buf),
        ],
    })
}

/// Record a DSM render pass.
///
/// Draws the interval triangles with premultiplied-alpha blending into the
/// moment target (loading over the primitive pre-pass). After the pass, the
/// alpha channel of `moment_view` contains `1 - T_total`.
///
/// * `index_buf` — the static fan index buffer (`interval_fan_index_buf`, 12 u32s)
/// * `indirect_args_buf` — the `interval_args_buf`; draw-indexed-indirect args
///   start at byte offset 12 (skipping the 3 dispatch u32s)
/// * `moment_view` — texture view for the DSM moment output
/// * `width`, `height` — output dimensions (for viewport/scissor)
pub fn record_dsm_render(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &DsmPipeline,
    bind_group: &wgpu::BindGroup,
    index_buf: &wgpu::Buffer,
    indirect_args_buf: &wgpu::Buffer,
    moments_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    width: u32,
    height: u32,
) {
    let load_ops = wgpu::Operations {
        load: wgpu::LoadOp::Load, // preserve primitive-pass moments
        store: wgpu::StoreOp::Store,
    };
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_render"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: moments_view,
            resolve_target: None,
            ops: load_ops,
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
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    rpass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    rpass.set_scissor_rect(0, 0, width, height);
    rpass.set_pipeline(&pipeline.render_pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
    // Draw-indexed-indirect args at byte offset 12 (skip 3 dispatch u32s).
    rpass.draw_indexed_indirect(indirect_args_buf, 12);
}

// ---------------------------------------------------------------------------
// DSM Primitive Pipeline
// ---------------------------------------------------------------------------

/// Uniform data for one DSM primitive draw call, matching `PrimitiveUniformsPadded`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DsmPrimUniform {
    vp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    color: [f32; 4],
    _pad: [f32; 28],
}

const DSM_PRIM_UNIFORM_ALIGN: u64 = 256;
const DSM_PRIM_MAX: u64 = 256;

/// Pipeline for rendering opaque primitives into DSM (depth + full occlusion).
pub struct DsmPrimitivePipeline {
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl DsmPrimitivePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_primitive.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_PRIMITIVE_WGSL.as_str().into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dsm_prim_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<DsmPrimUniform>() as u64
                    ),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dsm_prim_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dsm_prim_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PrimitiveVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DSM_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Y-flip in VP reverses winding
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_prim_uniforms"),
            size: DSM_PRIM_UNIFORM_ALIGN * DSM_PRIM_MAX,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dsm_prim_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<DsmPrimUniform>() as u64),
                }),
            }],
        });

        Self {
            render_pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
        }
    }
}

/// Record a primitive pre-pass for one DSM face.
///
/// Clears color to TRANSPARENT and depth to 1.0, then draws opaque primitives
/// from the light's viewpoint. Primitives write alpha=1 (full occlusion) and
/// their depth, so subsequent tet intervals are culled behind them.
pub fn record_dsm_primitive_pass(
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    pipeline: &DsmPrimitivePipeline,
    geometry: &PrimitiveGeometry,
    moments_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    primitives: &[Primitive],
    light_vp: &Mat4,
    near: f32,
    far: f32,
    width: u32,
    height: u32,
) {
    let clear_color = wgpu::Operations {
        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        store: wgpu::StoreOp::Store,
    };
    let clear_depth = wgpu::Operations {
        load: wgpu::LoadOp::Clear(1.0),
        store: wgpu::StoreOp::Store,
    };

    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
        view: moments_view,
        resolve_target: None,
        ops: clear_color,
        depth_slice: None,
    })];

    if primitives.is_empty() {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dsm_prim_clear"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(clear_depth),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        return;
    }

    // Write uniforms
    let vp_cols = light_vp.to_cols_array_2d();
    let count = primitives.len().min(DSM_PRIM_MAX as usize);

    for (i, prim) in primitives.iter().take(count).enumerate() {
        let model = prim.transform.model_matrix();
        let u = DsmPrimUniform {
            vp: vp_cols,
            model: model.to_cols_array_2d(),
            color: [near, far, width as f32, 0.0],
            _pad: [0.0; 28],
        };
        queue.write_buffer(
            &pipeline.uniform_buffer,
            i as u64 * DSM_PRIM_UNIFORM_ALIGN,
            bytemuck::bytes_of(&u),
        );
    }

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_primitives"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(clear_depth),
            stencil_ops: None,
        }),
        ..Default::default()
    });

    rpass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    rpass.set_scissor_rect(0, 0, width, height);
    rpass.set_pipeline(&pipeline.render_pipeline);
    rpass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
    let mut using_custom_vb = false;

    for (i, prim) in primitives.iter().take(count).enumerate() {
        let offset = i as u32 * DSM_PRIM_UNIFORM_ALIGN as u32;
        rpass.set_bind_group(0, &pipeline.bind_group, &[offset]);

        if let rmesh_interact::PrimitiveKind::CustomMesh(mesh_id) = prim.kind {
            if let (Some(ref cvb), Some(slice)) = (
                &geometry.custom_vertex_buffer,
                geometry.custom_meshes.get(mesh_id),
            ) {
                if !using_custom_vb {
                    rpass.set_vertex_buffer(0, cvb.slice(..));
                    using_custom_vb = true;
                }
                rpass.draw(slice.offset..slice.offset + slice.count, 0..1);
            }
        } else {
            if using_custom_vb {
                rpass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
                using_custom_vb = false;
            }
            let slice = geometry.kinds[prim.kind.index()];
            rpass.draw(slice.offset..slice.offset + slice.count, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------
// DSM Resolve Pipeline (Cantelli reconstruction → T(z_query))
// ---------------------------------------------------------------------------

/// Uniform for the resolve pass: just a query depth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DsmResolveUniforms {
    pub z_query: f32,
    pub near: f32,
    pub far: f32,
    pub _pad: f32,
}

/// Fullscreen resolve: reads the single moments texture and reconstructs
/// T(z_query) via Cantelli, mirroring the deferred shader's logic.
pub struct DsmResolvePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms_buf: wgpu::Buffer,
}

impl DsmResolvePipeline {
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dsm_resolve.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DSM_RESOLVE_WGSL.as_str().into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dsm_resolve_bgl"),
            entries: &[
                // 0: uniforms (z_query)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            DsmResolveUniforms,
                        >() as u64),
                    },
                    count: None,
                },
                // 1: moments texture
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dsm_resolve_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dsm_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_resolve_uniforms"),
            size: std::mem::size_of::<DsmResolveUniforms>() as u64,
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

/// Create the resolve bind group (uniform + moments texture).
pub fn create_dsm_resolve_bind_group(
    device: &wgpu::Device,
    resolve: &DsmResolvePipeline,
    moments_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_resolve_bg"),
        layout: &resolve.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: resolve.uniforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(moments_view),
            },
        ],
    })
}

/// Record resolve passes for all 6 cubemap faces in a cross layout.
///
/// Layout (4×3 grid, each cell = face_size × face_size):
/// ```text
///        [+Y]
///  [-X]  [+Z]  [+X]  [-Z]
///        [-Y]
/// ```
pub fn record_dsm_resolve_cubemap_cross(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    resolve: &DsmResolvePipeline,
    queue: &wgpu::Queue,
    atlas: &DsmAtlas,
    light_index: usize,
    z_query: f32,
    near: f32,
    far: f32,
    output_view: &wgpu::TextureView,
    output_width: u32,
    output_height: u32,
) {
    // Simple 3x2 grid: top row = faces 0,1,2; bottom row = faces 3,4,5
    // Face 0 (+X), Face 1 (-X), Face 2 (+Y), Face 3 (-Y), Face 4 (+Z), Face 5 (-Z)
    let face_w = output_width / 3;
    let face_h = output_height / 2;
    let face_count = if light_index < atlas.cubemaps.len() {
        6usize
    } else {
        0
    };

    let positions: [(u32, u32); 6] = [
        (0, 0),
        (1, 0),
        (2, 0), // +X, -X, +Y
        (0, 1),
        (1, 1),
        (2, 1), // -Y, +Z, -Z
    ];

    queue.write_buffer(
        &resolve.uniforms_buf,
        0,
        bytemuck::bytes_of(&DsmResolveUniforms {
            z_query,
            near,
            far,
            _pad: 0.0,
        }),
    );

    // Clear output
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dsm_cross_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.2,
                        g: 0.2,
                        b: 0.2,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
    }

    // Render each face into its cross position
    for fi in 0..face_count.min(6) {
        // Create a per-face D2 view from the cubemap texture
        let face_view = atlas.cubemaps[light_index].create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: fi as u32,
            array_layer_count: Some(1),
            ..Default::default()
        });

        let resolve_bg = create_dsm_resolve_bind_group(device, resolve, &face_view);

        let (col, row) = positions[fi];
        let x = col * face_w;
        let y = row * face_h;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dsm_cross_face"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        rpass.set_viewport(x as f32, y as f32, face_w as f32, face_h as f32, 0.0, 1.0);
        rpass.set_pipeline(&resolve.pipeline);
        rpass.set_bind_group(0, &resolve_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

/// Record the resolve pass: reads the moment texture, writes T(z_query) to output.
pub fn record_dsm_resolve(
    encoder: &mut wgpu::CommandEncoder,
    resolve: &DsmResolvePipeline,
    bind_group: &wgpu::BindGroup,
    output_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("dsm_resolve"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    rpass.set_pipeline(&resolve.pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Per-Light DSM Cache
// ---------------------------------------------------------------------------

/// Cubemap face view matrices for look_to_rh, matching WebGPU cubemap sampler.
///
/// Standard cubemap face coordinates (OpenGL/WebGPU):
///   Face +X: sc=-z, tc=-y    Face -X: sc=+z, tc=-y
///   Face +Y: sc=+x, tc=+z   Face -Y: sc=+x, tc=-z
///   Face +Z: sc=+x, tc=-y   Face -Z: sc=-x, tc=-y
///
/// For look_to_rh(pos, forward, up):
///   right = normalize(cross(up, -forward))
///   actual_up = cross(-forward, right)
///
/// We need: right aligns with sc direction, actual_up aligns with -tc direction
/// (because tc increases downward in texture, but actual_up points upward in view).
const CUBEMAP_DIRS: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::NEG_Y), // +X: right=-Z(sc=-z✓), up=-Y → actual_up=-Y(tc=-y✓)
    (Vec3::NEG_X, Vec3::NEG_Y), // -X: right=+Z(sc=+z✓), up=-Y → actual_up=-Y(tc=-y✓)
    (Vec3::Y, Vec3::Z),     // +Y: right=+X(sc=+x✓), up=+Z → actual_up=+Z(tc=+z✓)
    (Vec3::NEG_Y, Vec3::NEG_Z), // -Y: right=+X(sc=+x✓), up=-Z → actual_up=-Z(tc=-z✓)
    (Vec3::Z, Vec3::NEG_Y), // +Z: right=+X(sc=+x✓), up=-Y → actual_up=-Y(tc=-y✓)
    (Vec3::NEG_Z, Vec3::NEG_Y), // -Z: right=-X(sc=-x✓), up=-Y → actual_up=-Y(tc=-y✓)
];

/// Per-light shadow metadata for the deferred shader (matches WGSL `ShadowLight`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowLightMeta {
    /// W row of the VP matrix for each cubemap face — the deferred shader
    /// only needs `clip.w = dot(vp_w, vec4(world_pos, 1))` to recover the
    /// receiver depth. Storing only the W row drops this struct from
    /// 416 bytes to 128 bytes and lets the shader avoid a 6-way branch.
    pub vp_w_rows: [[f32; 4]; 6],
    pub face_offset: u32,
    pub face_count: u32,
    pub near: f32,
    pub far: f32,
    pub light_type: u32,
    pub _pad: [u32; 3],
}

/// Cached DSM atlas: one moments cubemap per light + per-light metadata.
///
/// Each cubemap is a 6-layer `Rgba16Float` texture holding the α-weighted
/// 2-moment depth distribution. DSM generation renders one face at a time
/// into a staging 2D texture, then copies into the cubemap layer.
pub struct DsmAtlas {
    /// One cubemap (6 faces) per light, holding (E[α·z], E[α·z²], 0, α).
    pub cubemaps: Vec<wgpu::Texture>,
    /// Cube views for shader binding (one per light).
    pub cubemap_views: Vec<wgpu::TextureView>,
    /// Staging 2D texture for rendering one face at a time, copied into cubemap layers.
    pub staging_moments: wgpu::Texture,
    pub staging_moments_view: wgpu::TextureView,
    pub staging_depth: wgpu::Texture,
    pub staging_depth_view: wgpu::TextureView,
    /// Per-light shadow metadata storage buffer.
    pub meta_buf: wgpu::Buffer,
    pub num_lights: u32,
    pub resolution: u32,
    /// Scratch uniform buffer for light viewpoint.
    pub scratch_uniforms: wgpu::Buffer,
}

impl DsmAtlas {
    /// Allocate atlas textures for the given light types.
    ///
    /// `light_types[i]`: 0 = point (6 faces), else = 1 face.
    pub fn new(device: &wgpu::Device, resolution: u32, light_types: &[u32]) -> Self {
        let num_lights = light_types.len() as u32;

        // One moments cubemap (6 faces) per light
        let mut cubemaps = Vec::with_capacity(light_types.len());
        let mut cubemap_views = Vec::with_capacity(light_types.len());
        for i in 0..light_types.len() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("dsm_cube_light{i}")),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 6,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DSM_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });
            cubemaps.push(tex);
            cubemap_views.push(view);
        }

        // Staging texture: single 2D for rendering one face at a time
        let staging_moments = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dsm_staging_moments"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DSM_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let staging_moments_view =
            staging_moments.create_view(&wgpu::TextureViewDescriptor::default());
        let staging_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dsm_staging_depth"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let staging_depth_view = staging_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_shadow_meta"),
            size: (16 * std::mem::size_of::<ShadowLightMeta>()) as u64, // MAX_LIGHTS
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dsm_scratch_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            // UNIFORM so this buffer can bind as `var<uniform>` in
            // project_compute_hw (now required because that shader uses
            // a uniform binding for `uniforms` to stay under WebGPU's
            // 10 storage-buffers-per-stage cap).
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cubemaps,
            cubemap_views,
            staging_moments,
            staging_moments_view,
            staging_depth,
            staging_depth_view,
            meta_buf,
            num_lights,
            resolution,
            scratch_uniforms,
        }
    }

    /// Create a dummy atlas (1x1 cubemap) for when no DSM is available.
    pub fn new_dummy(device: &wgpu::Device) -> Self {
        Self::new(device, 1, &[0]) // one dummy point light
    }

    /// Fill metadata buffer with per-face W rows and face offsets for all lights.
    pub fn populate_metadata(&self, queue: &wgpu::Queue, lights: &[GpuLight], near: f32, far: f32) {
        let mut metas = vec![ShadowLightMeta::zeroed(); 16]; // MAX_LIGHTS
        for (li, light) in lights.iter().enumerate() {
            if li >= 16 {
                break;
            }
            let fc = 6usize; // always 6 faces for cubemap
            let mut vp_w_rows = [[0.0f32; 4]; 6];
            for fi in 0..fc {
                let (vp, _c2w) = build_light_vp(light, fi, near, far);
                // Shader only reads clip.w = dot(row3, vec4(world_pos, 1)).
                vp_w_rows[fi] = vp.row(3).to_array();
            }
            metas[li] = ShadowLightMeta {
                vp_w_rows,
                face_offset: 0, // not used with cubemaps
                face_count: fc as u32,
                near,
                far,
                light_type: light.light_type,
                _pad: [0; 3],
            };
        }
        queue.write_buffer(&self.meta_buf, 0, bytemuck::cast_slice(&metas));
    }
}

/// Build a view-projection matrix for a light face.
///
/// For point lights (type 0), `face_index` selects the cubemap face (0..6).
/// For spot lights (type 1), uses the light's direction and outer cone angle.
/// For directional lights (type 2), uses an orthographic projection along direction.
pub fn build_light_vp(light: &GpuLight, face_index: usize, near: f32, far: f32) -> (Mat4, Mat3) {
    let pos = Vec3::from(light.position);

    let (forward, up) = match light.light_type {
        0 => {
            // Point light cubemap face
            CUBEMAP_DIRS[face_index]
        }
        1 => {
            // Spot light — look along direction
            let dir = Vec3::from(light.direction).normalize_or_zero();
            let tentative_up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
            (dir, tentative_up)
        }
        _ => {
            // Directional light — look along direction
            let dir = Vec3::from(light.direction).normalize_or_zero();
            let tentative_up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
            (dir, tentative_up)
        }
    };

    let view = Mat4::look_to_rh(pos, forward, up);

    let proj = match light.light_type {
        0 => {
            // Cubemap face: 90° square FOV
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far)
        }
        1 => {
            // Spot light: FOV = 2 * outer_angle. light.outer_cos is precomputed
            // for the deferred shader, so recover the angle here via acos.
            let fov = (2.0 * light.outer_cos.acos()).max(0.01);
            Mat4::perspective_rh(fov, 1.0, near, far)
        }
        _ => {
            // Directional: orthographic (scene-sized box)
            Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, near, far)
        }
    };

    // Flip Y in clip space: look_to_rh renders Y-down, cubemap expects Y-up.
    let flip = Mat4::from_cols(glam::Vec4::X, -glam::Vec4::Y, glam::Vec4::Z, glam::Vec4::W);

    // Camera-to-world rotation = inverse of the 3×3 part of view
    let view3 = Mat3::from_cols(
        view.col(0).truncate(),
        view.col(1).truncate(),
        view.col(2).truncate(),
    );
    let c2w = view3.transpose();

    let mut vp = flip * proj * view;

    // DEBUG: rotate face 0 (+X) image 90° CCW about its center in clip space.
    // A clip-space Z-rotation only mixes X/Y, so the W row (receiver depth read
    // by populate_metadata) is untouched. Negate the angle if it comes out CW.

    (vp, c2w)
}

/// Generate deep shadow maps for all active lights.
///
/// Each cubemap face is sorted independently from the light's perspective:
/// projecting the tets through that face's view-projection yields depth keys
/// whose back-to-front order is correct for compositing the α-weighted
/// termination-depth moments. (A single per-light radial sort is *not* a valid
/// per-face depth order off-axis, so each face gets its own project + sort.)
///
/// Per light, per face (×6):
///   1. Project compute — reuse the forward `project_compute_hw` pipeline to
///      project from the face VP and emit depth sort keys (frustum-culled)
///   2. Radix sort by depth from the light
///   3. Indirect convert — build dispatch/draw args from the visible count
///   4. Interval gen + DSM render into staging, then copy to the cubemap face
#[allow(clippy::too_many_arguments)]
pub fn generate_dsm_for_lights(
    atlas: &DsmAtlas,
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dsm_pipeline: &DsmPipeline,
    dsm_prim_pipeline: &DsmPrimitivePipeline,
    fwd_pipelines: &ForwardPipelines,
    prim_geometry: &PrimitiveGeometry,
    primitives: &[Primitive],
    ci_pipelines: &ComputeIntervalPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
    lights: &[GpuLight],
    num_lights: u32,
    tet_count: u32,
    near: f32,
    far: f32,
    render_tets: bool,
) {
    let res = atlas.resolution;
    let n_pow2 = tet_count.next_power_of_two();
    let half = res as f32 / 2.0;
    let intrinsics = [half, half, half, half];

    for li in 0..num_lights.min(lights.len() as u32) {
        let light = &lights[li as usize];
        let face_count = 6usize;
        let pos = Vec3::from(light.position);

        // --- Per face: project + sort + convert + interval gen + render + copy ---
        for fi in 0..face_count {
            let (face_vp, face_c2w) = build_light_vp(light, fi, near, far);
            let face_uniforms = rmesh_render::make_uniforms(
                face_vp, face_c2w, intrinsics, pos, res as f32, res as f32, tet_count, 0, 4, 0.0,
                0, near, far,
            );

            // Flush prior GPU work so the face_uniforms write takes effect
            // before this face's project pass (and so the previous face's
            // commands have consumed the shared buffers we reset below).
            let old_encoder = std::mem::replace(
                encoder,
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dsm_face_sort"),
                }),
            );
            queue.submit(std::iter::once(old_encoder.finish()));
            queue.write_buffer(
                &atlas.scratch_uniforms,
                0,
                bytemuck::bytes_of(&face_uniforms),
            );

            let mut result_in_b = false;
            if render_tets {
                // Reset draw/dispatch state: `project_compute_hw` frustum-culls
                // and counts visible tets via atomicAdd on
                // `indirect_args.instance_count`, so it must start at 0.
                queue.write_buffer(
                    &buffers.indirect_args,
                    0,
                    bytemuck::cast_slice(&[12u32, 0u32, 0u32, 0u32]),
                );
                queue.write_buffer(
                    &buffers.interval_args_buf,
                    0,
                    bytemuck::cast_slice(&[0u32; 8]),
                );
                queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

                // Project tets from this face's VP → depth sort keys, reusing
                // the forward HW project compute (its color/SH outputs are
                // unused here but the bind group layout requires them).
                let compute_bg = create_dsm_hw_compute_bg(
                    device,
                    fwd_pipelines,
                    buffers,
                    material,
                    sh_coeffs_buf,
                    &atlas.scratch_uniforms,
                );
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dsm_project_hw"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
                    cpass.set_bind_group(0, &compute_bg, &[]);
                    let total_workgroups = n_pow2.div_ceil(64u32);
                    let (dx, dy) = dispatch_2d(total_workgroups);
                    cpass.dispatch_workgroups(dx, dy, 1);
                }

                result_in_b = rmesh_sort::record_radix_sort(
                    encoder,
                    device,
                    sort_pipelines,
                    sort_state,
                    &buffers.sort_keys,
                    &buffers.sort_values,
                );

                let convert_bg = create_dsm_indirect_convert_bg(device, ci_pipelines, buffers);
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dsm_indirect_convert"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&ci_pipelines.indirect_convert_pipeline);
                    cpass.set_bind_group(0, &convert_bg, &[]);
                    cpass.dispatch_workgroups(1, 1, 1);
                }

                // Submit the sort work, then re-write face uniforms so the
                // render pass below sees the correct per-face VP.
                let sort_encoder = std::mem::replace(
                    encoder,
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("dsm_face_render"),
                    }),
                );
                queue.submit(std::iter::once(sort_encoder.finish()));
                queue.write_buffer(
                    &atlas.scratch_uniforms,
                    0,
                    bytemuck::bytes_of(&face_uniforms),
                );
            }

            record_dsm_primitive_pass(
                encoder,
                queue,
                dsm_prim_pipeline,
                prim_geometry,
                &atlas.staging_moments_view,
                &atlas.staging_depth_view,
                primitives,
                &face_vp,
                near,
                far,
                res,
                res,
            );

            if render_tets {
                let sort_vals = if result_in_b {
                    sort_state.values_b()
                } else {
                    &buffers.sort_values
                };
                let gen_bg = create_dsm_interval_gen_bg(
                    device,
                    ci_pipelines,
                    buffers,
                    material,
                    sort_vals,
                    &atlas.scratch_uniforms,
                );
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("dsm_interval_gen"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&ci_pipelines.gen_pipeline);
                    cpass.set_bind_group(0, &gen_bg, &[]);
                    cpass.dispatch_workgroups_indirect(&buffers.interval_args_buf, 0);
                }

                let render_bg = create_dsm_render_bind_group(
                    device,
                    dsm_pipeline,
                    &atlas.scratch_uniforms,
                    &buffers.interval_vertex_buf,
                    &buffers.interval_tet_data_buf,
                );
                record_dsm_render(
                    encoder,
                    dsm_pipeline,
                    &render_bg,
                    &buffers.interval_fan_index_buf,
                    &buffers.interval_args_buf,
                    &atlas.staging_moments_view,
                    &atlas.staging_depth_view,
                    res,
                    res,
                );
            }

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas.staging_moments,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas.cubemaps[li as usize],
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: fi as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// Create the forward HW projection compute bind group with scratch uniforms.
///
/// Mirrors `rmesh_render::create_hw_compute_bind_group` but binds the DSM
/// per-face `scratch_uniforms` at binding 0 instead of `buffers.uniforms`.
/// Layout matches `project_compute_hw.wgsl` bindings 0–10.
fn create_dsm_hw_compute_bg(
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
    scratch_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_hw_compute_bg"),
        layout: &fwd_pipelines.hw_compute_bind_group_layout,
        entries: &[
            buf_entry(0, scratch_uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &buffers.circumdata),
            buf_entry(4, &buffers.sort_keys),
            buf_entry(5, &buffers.sort_values),
            buf_entry(6, &buffers.indirect_args),
            buf_entry(7, &material.colors),
            buf_entry(8, &material.base_colors),
            buf_entry(9, &material.color_grads),
            buf_entry(10, sh_coeffs_buf),
        ],
    })
}

/// Create the indirect-convert bind group.
fn create_dsm_indirect_convert_bg(
    device: &wgpu::Device,
    ci_pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_indirect_convert_bg"),
        layout: &ci_pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.interval_args_buf),
        ],
    })
}

/// Create the interval-gen bind group with scratch uniforms and explicit sort_values.
fn create_dsm_interval_gen_bg(
    device: &wgpu::Device,
    ci_pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
    scratch_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dsm_interval_gen_bg"),
        layout: &ci_pipelines.gen_bg_layout,
        entries: &[
            buf_entry(0, scratch_uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
            buf_entry(8, &buffers.interval_vertex_buf),
            buf_entry(9, &buffers.interval_tet_data_buf),
            buf_entry(10, &buffers.vertex_normals),
        ],
    })
}
