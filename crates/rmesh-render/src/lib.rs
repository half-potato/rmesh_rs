//! Forward rendering pipeline orchestration (wgpu).
//!
//! Sets up the wgpu compute and render pipelines for the forward pass:
//!   1. Compute pass: SH eval, cull, depth key generation
//!   2. Render pass: Hardware rasterization with MRT output
//!
//! All GPU buffer management and bind group creation lives here.

#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use glam::Mat4;
use rmesh_data::SceneData;
use wgpu::util::DeviceExt;

// Re-export shared types for CPU-side use.
pub use rmesh_util::shared::{BVHNode, DrawIndirectCommand, GpuLight, Uniforms, MAX_LIGHTS};

// Shared wgpu helpers (used unqualified throughout this crate).
use rmesh_util::gpu_helpers::{buf_entry, storage_entries};

// Scene/material GPU buffers now live in rmesh-data; re-export for existing callers.
pub use rmesh_data::{MaterialBuffers, SceneBuffers};

// `SceneBuffers` sizes its uniform buffer with a local constant because rmesh-data
// cannot depend on rmesh-util. Assert here (where both are visible) that it matches.
const _UNIFORMS_SIZE_CHECK: () = assert!(
    std::mem::size_of::<Uniforms>() == rmesh_data::UNIFORMS_BUFFER_SIZE as usize,
    "rmesh_data::UNIFORMS_BUFFER_SIZE must equal size_of::<Uniforms>()"
);

// Re-export tile types (moved to rmesh-tile crate).
pub use rmesh_tile::{
    create_prepare_dispatch_bind_group, create_rts_bind_group, create_tile_fill_bind_group,
    create_tile_gen_scan_bind_group, create_tile_ranges_bind_group,
    create_tile_ranges_bind_group_with_keys, dispatch_2d, record_scan_tile_pipeline, ScanBuffers,
    ScanPipelines, TileBuffers, TilePipelines,
};

// WGSL shader sources, embedded from crate-local files.
const PROJECT_COMPUTE_WGSL: &str = include_str!("wgsl/project_compute.wgsl");
const PROJECT_COMPUTE_HW_WGSL: &str = include_str!("wgsl/project_compute_hw.wgsl");
const FORWARD_VERTEX_QUAD_WGSL: &str = include_str!("wgsl/forward_vertex_quad.wgsl");
const FORWARD_PREPASS_COMPUTE_WGSL: &str = include_str!("wgsl/forward_prepass_compute.wgsl");
const FORWARD_VERTEX_WGSL: &str = include_str!("wgsl/forward_vertex.wgsl");
const FORWARD_FRAGMENT_WGSL: &str = include_str!("wgsl/forward_fragment.wgsl");
const TEX_TO_BUFFER_WGSL: &str = include_str!("wgsl/tex_to_buffer.wgsl");
const BLIT_WGSL: &str = include_str!("wgsl/blit.wgsl");
const FORWARD_MESH_WGSL: &str = include_str!("wgsl/forward_mesh.wgsl");
const INDIRECT_CONVERT_WGSL: &str = include_str!("wgsl/indirect_convert.wgsl");
const INTERVAL_MESH_WGSL: &str = include_str!("wgsl/interval_mesh.wgsl");
const INTERVAL_FRAGMENT_WGSL: &str = include_str!("wgsl/interval_fragment.wgsl");
const INTERVAL_COMPUTE_WGSL: &str = include_str!("wgsl/interval_compute.wgsl");
const INTERVAL_VERTEX_WGSL: &str = include_str!("wgsl/interval_vertex.wgsl");
const INTERVAL_INDIRECT_CONVERT_WGSL: &str = include_str!("wgsl/interval_indirect_convert.wgsl");
const PROJECT_COMPUTE_16BIT_WGSL: &str = include_str!("wgsl/project_compute_16bit.wgsl");

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

/// A pair of bind groups for the forward project_compute shader.
/// Group 0 holds the 8 read-only inputs; group 1 holds the 6 read-write buffers.
/// Split because WebGPU caps `maxStorageBuffersPerShaderStage` at 10.
pub struct ForwardComputeBindGroups {
    pub bg0: wgpu::BindGroup,
    pub bg1: wgpu::BindGroup,
}

/// Bind group for the HW-only project_compute_hw shader.
/// Held in a struct rather than a bare `wgpu::BindGroup` for symmetry with
/// `ForwardComputeBindGroups` and so callers can store a single field.
pub struct ForwardHwComputeBindGroups {
    pub bg: wgpu::BindGroup,
}

/// Compiled pipelines for the forward pass.
pub struct ForwardPipelines {
    pub compute_pipeline: wgpu::ComputePipeline,
    /// 16-bit linear sort key variant of compute_pipeline (same layout).
    pub compute_pipeline_16bit: wgpu::ComputePipeline,
    pub compute_bg0_layout: wgpu::BindGroupLayout,
    pub compute_bg1_layout: wgpu::BindGroupLayout,
    /// Lean HW-only projection compute (no tile counting)
    pub hw_compute_pipeline: wgpu::ComputePipeline,
    pub hw_compute_bind_group_layout: wgpu::BindGroupLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    /// Color-only render pipeline (no MRT — skips aux/normals/depth targets for perf)
    pub render_pipeline_color_only: wgpu::RenderPipeline,
    pub render_bind_group_layout: wgpu::BindGroupLayout,
    /// Quad-based render pipeline (4 verts/tet via triangle strip, reads precomputed buffer)
    pub quad_render_pipeline: wgpu::RenderPipeline,
    /// Color-only quad render pipeline (no MRT)
    pub quad_render_pipeline_color_only: wgpu::RenderPipeline,
    /// Bind group layout for quad render (6 bindings: uniforms, precomputed, sorted_indices, colors, densities, color_grads)
    pub quad_render_bg_layout: wgpu::BindGroupLayout,
    /// Compute prepass pipeline (precomputes clip positions + normals for quad path)
    pub prepass_compute_pipeline: wgpu::ComputePipeline,
    pub prepass_bg_layout: wgpu::BindGroupLayout,
}

/// Helper: create N storage buffer layout entries for the given visibility.
impl ForwardPipelines {
    /// Create all three pipelines from WGSL shader sources.
    ///
    /// `color_format`: texture format for all color attachments (Rgba16Float).
    /// Total bytes per sample must not exceed 32 bytes (4 * 8 = 32 for Rgba16Float).
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Compute pipeline (split into 2 bind groups, 14 storage total) -----
        // Split along read-only / read-write boundary for code organization:
        //   group 0 (8 read-only): uniforms, vertices, indices, densities,
        //                          color_grads, circumdata, base_colors_buf, sh_coeffs
        //   group 1 (6 read-write): colors, sort_keys, sort_values, indirect_args,
        //                           tiles_touched, compact_tet_ids
        //
        // Note: WebGPU's maxStorageBuffersPerShaderStage is 10 in Chrome and the
        // per-stage cap is checked across ALL bind groups, so this pipeline is
        // INVALID on web (14 > 10). The web viewer dodges by passing
        // `hw_compute_bg = Some(...)` to `record_sorted_forward_pass`, which
        // uses `hw_compute_pipeline` (10 storage total via uniform-buffer trick)
        // instead of `compute_pipeline`.
        let compute_bg0_entries = storage_entries(8, wgpu::ShaderStages::COMPUTE, &[true; 8]);
        let compute_bg0_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bg0_layout"),
                entries: &compute_bg0_entries,
            });
        let compute_bg1_entries = storage_entries(6, wgpu::ShaderStages::COMPUTE, &[false; 6]);
        let compute_bg1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bg1_layout"),
                entries: &compute_bg1_entries,
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&compute_bg0_layout, &compute_bg1_layout],
                immediate_size: 0,
            });
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PROJECT_COMPUTE_WGSL.into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("project_compute_pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ----- 16-bit linear sort key variant (same layout, different shader) -----
        let compute_shader_16bit = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project_compute_16bit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PROJECT_COMPUTE_16BIT_WGSL.into()),
        });
        let compute_pipeline_16bit =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("project_compute_16bit_pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader_16bit,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- HW projection compute pipeline (11 bindings, single group) -----
        // Bindings: uniforms(r), vertices(r), indices(r), circumdata(r),
        //           sort_keys(rw), sort_values(rw), indirect_args(rw),
        //           colors(rw), base_colors(r), color_grads(r), sh_coeffs(r)
        // Binding 0 (uniforms) is bound as a UNIFORM buffer so the COMPUTE-stage
        // storage-buffer count is 10, fitting WebGPU's per-stage cap.
        let hw_compute_storage_read_only = [
            true, true, true, // 1-3 read-only (vertices, indices, circumdata)
            false, false, false, // 4-6 read-write (sort_keys, sort_values, indirect_args)
            false, true, true,
            true, // 7=colors(rw), 8=base_colors(r), 9=color_grads(r), 10=sh_coeffs(r)
        ];
        let mut hw_compute_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, &read_only) in hw_compute_storage_read_only.iter().enumerate() {
            hw_compute_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let hw_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hw_compute_bind_group_layout"),
                entries: &hw_compute_entries,
            });
        let hw_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hw_compute_pipeline_layout"),
                bind_group_layouts: &[&hw_compute_bind_group_layout],
                immediate_size: 0,
            });
        let hw_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project_compute_hw.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PROJECT_COMPUTE_HW_WGSL.into()),
        });
        let hw_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("project_compute_hw_pipeline"),
                layout: Some(&hw_compute_pipeline_layout),
                module: &hw_compute_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Render pipeline (7 bindings) -----
        // Bindings: uniforms, vertices, indices, colors, densities, color_grads, sorted_indices
        // All read-only from the vertex/fragment perspective.
        let render_read_only = [true; 7];
        let render_entries = storage_entries(
            7,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &render_read_only,
        );
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bind_group_layout"),
                entries: &render_entries,
            });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                immediate_size: 0,
            });
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_vertex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_VERTEX_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_FRAGMENT_WGSL.into()),
        });

        // Premultiplied alpha blend for color attachment 0
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward_render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                buffers: &[], // No vertex buffers -- all data from storage
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), // Only front-facing → 1 fragment per pixel
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
                targets: &[
                    // Color attachment 0: premultiplied alpha blend
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Color attachment 1 (aux): premultiplied alpha blend
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Color attachment 2 (normals): premultiplied alpha blend.
                    // Rgba8Unorm matches normals_texture (halved bandwidth vs
                    // Rgba16Float; bias-encoded as (n*0.5+0.5)*alpha).
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Color attachment 3 (depth): premultiplied alpha blend
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ----- Prepass compute pipeline (9 bindings) -----
        // Bindings: uniforms(r), vertices(r), indices(r), sorted_indices(r),
        //           indirect_args(r), precomputed(rw), colors(r), densities(r), color_grads(r)
        let prepass_read_only = [
            true, true, true, true, true,  // 0-4 read-only
            false, // 5 read-write (precomputed)
            true, true, true, // 6-8 read-only (colors, densities, color_grads)
        ];
        let prepass_entries = storage_entries(9, wgpu::ShaderStages::COMPUTE, &prepass_read_only);
        let prepass_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prepass_bg_layout"),
            entries: &prepass_entries,
        });
        let prepass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prepass_pipeline_layout"),
                bind_group_layouts: &[&prepass_bg_layout],
                immediate_size: 0,
            });
        let prepass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_prepass_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_PREPASS_COMPUTE_WGSL.into()),
        });
        let prepass_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prepass_compute_pipeline"),
                layout: Some(&prepass_pipeline_layout),
                module: &prepass_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ----- Quad render bind group layout (2 bindings, reads precomputed buffer) -----
        // Bindings: uniforms(r), precomputed(r)
        // Vertex shader reads only precomputed. Fragment shader reads uniforms for intrinsics.
        let quad_render_read_only = [true; 2];
        let quad_render_entries = storage_entries(
            2,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &quad_render_read_only,
        );
        let quad_render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("quad_render_bg_layout"),
                entries: &quad_render_entries,
            });
        let quad_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quad_render_pipeline_layout"),
                bind_group_layouts: &[&quad_render_bg_layout],
                immediate_size: 0,
            });

        // Quad-based render pipeline (triangle strip, 4 verts/tet)
        // Shares forward_fragment.wgsl — vertex output matches exactly.
        let quad_vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_vertex_quad.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_VERTEX_QUAD_WGSL.into()),
        });
        let quad_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward_quad_render_pipeline"),
            layout: Some(&quad_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &quad_vertex_shader,
                entry_point: Some("main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No face culling — quad is screen-aligned
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
                targets: &[
                    // location(0) color
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(1) aux0
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(2) normals — Rgba8Unorm bias-encoded.
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(3) depth
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ----- Color-only pipeline variants (no MRT — single color target) -----
        let color_only_targets: &[Option<wgpu::ColorTargetState>] = &[
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(premul_blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            None,
            None,
            None,
        ];

        let render_pipeline_color_only =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("forward_render_pipeline_color_only"),
                layout: Some(&render_pipeline_layout),
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
                    cull_mode: Some(wgpu::Face::Back),
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
                    targets: color_only_targets,
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        let quad_render_pipeline_color_only =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("forward_quad_render_pipeline_color_only"),
                layout: Some(&quad_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &quad_vertex_shader,
                    entry_point: Some("main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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
                    targets: color_only_targets,
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        Self {
            compute_pipeline,
            compute_pipeline_16bit,
            compute_bg0_layout,
            compute_bg1_layout,
            hw_compute_pipeline,
            hw_compute_bind_group_layout,
            render_pipeline,
            render_pipeline_color_only,
            render_bind_group_layout,
            quad_render_pipeline,
            quad_render_pipeline_color_only,
            quad_render_bg_layout,
            prepass_compute_pipeline,
            prepass_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh Shader Pipelines (optional, requires EXPERIMENTAL_MESH_SHADER)
// ---------------------------------------------------------------------------

/// Compiled pipelines for the mesh shader forward pass.
pub struct MeshForwardPipelines {
    pub mesh_render_pipeline: wgpu::RenderPipeline,
    /// Color-only mesh render pipeline (no MRT)
    pub mesh_render_pipeline_color_only: wgpu::RenderPipeline,
    pub mesh_render_bg_layout: wgpu::BindGroupLayout,
    pub indirect_convert_pipeline: wgpu::ComputePipeline,
    pub indirect_convert_bg_layout: wgpu::BindGroupLayout,
}

impl MeshForwardPipelines {
    /// Create mesh shader pipelines. Requires `Features::EXPERIMENTAL_MESH_SHADER`.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Mesh render pipeline (8 read-only storage bindings) -----
        // Bindings 0-6: same as vertex shader render bind group
        // Binding 7: indirect_args (read visible_count)
        let mesh_read_only = [true; 8];
        let mesh_entries = storage_entries(8, wgpu::ShaderStages::MESH, &mesh_read_only);
        let mesh_render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_render_bg_layout"),
                entries: &mesh_entries,
            });
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&mesh_render_bg_layout],
            immediate_size: 0,
        });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_MESH_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(FORWARD_FRAGMENT_WGSL.into()),
        });

        // Same blend state as ForwardPipelines
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

        let mesh_render_pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("mesh_forward_render_pipeline"),
            layout: Some(&mesh_pipeline_layout),
            task: None,
            mesh: wgpu::MeshState {
                module: &mesh_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
                targets: &[
                    // location(0) color
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(1) aux0
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(2) normals — Rgba8Unorm bias-encoded.
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(3) depth
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // ----- Indirect convert compute pipeline (2 bindings) -----
        let indirect_convert_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("indirect_convert_bg_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let indirect_convert_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("indirect_convert_pipeline_layout"),
                bind_group_layouts: &[&indirect_convert_bg_layout],
                immediate_size: 0,
            });
        let indirect_convert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("indirect_convert.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INDIRECT_CONVERT_WGSL.into()),
        });
        let indirect_convert_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("indirect_convert_pipeline"),
                layout: Some(&indirect_convert_layout),
                module: &indirect_convert_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let color_only_targets: &[Option<wgpu::ColorTargetState>] = &[
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(premul_blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            None,
            None,
            None,
        ];

        let mesh_render_pipeline_color_only =
            device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
                label: Some("mesh_forward_render_pipeline_color_only"),
                layout: Some(&mesh_pipeline_layout),
                task: None,
                mesh: wgpu::MeshState {
                    module: &mesh_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
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
                    targets: color_only_targets,
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        Self {
            mesh_render_pipeline,
            mesh_render_pipeline_color_only,
            mesh_render_bg_layout,
            indirect_convert_pipeline,
            indirect_convert_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Interval Shading Pipelines (requires EXPERIMENTAL_MESH_SHADER)
// ---------------------------------------------------------------------------

/// Compiled pipelines for the interval shading forward pass.
///
/// Decomposes each tet into non-overlapping screen-space triangles with
/// interpolated front/back NDC depths. Single color output (no MRT).
pub struct IntervalPipelines {
    pub mesh_render_pipeline: wgpu::RenderPipeline,
    pub mesh_render_bg_layout: wgpu::BindGroupLayout,
    pub indirect_convert_pipeline: wgpu::ComputePipeline,
    pub indirect_convert_bg_layout: wgpu::BindGroupLayout,
}

impl IntervalPipelines {
    /// Create interval shading pipelines. Requires `Features::EXPERIMENTAL_MESH_SHADER`.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Mesh render pipeline (8 read-only storage bindings) -----
        // Same bindings as MeshForwardPipelines: uniforms, vertices, indices, colors,
        // densities, color_grads, sorted_indices, indirect_args
        let mesh_read_only = [true; 8];
        let mesh_entries = storage_entries(
            8,
            wgpu::ShaderStages::MESH | wgpu::ShaderStages::FRAGMENT,
            &mesh_read_only,
        );
        let mesh_render_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("interval_mesh_render_bg_layout"),
                entries: &mesh_entries,
            });
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interval_mesh_pipeline_layout"),
            bind_group_layouts: &[&mesh_render_bg_layout],
            immediate_size: 0,
        });
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_mesh.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_MESH_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_FRAGMENT_WGSL.into()),
        });

        // Premultiplied alpha blend for single color attachment
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

        let mesh_render_pipeline = device.create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
            label: Some("interval_mesh_render_pipeline"),
            layout: Some(&mesh_pipeline_layout),
            task: None,
            mesh: wgpu::MeshState {
                module: &mesh_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No face culling — interval triangles are screen-aligned
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
                targets: &[
                    // Single color attachment: premultiplied alpha blend
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // ----- Indirect convert compute pipeline (2 bindings) -----
        // Reuses indirect_convert.wgsl with TETS_PER_GROUP overridden to 16
        let indirect_convert_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("interval_indirect_convert_bg_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let indirect_convert_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("interval_indirect_convert_pipeline_layout"),
                bind_group_layouts: &[&indirect_convert_bg_layout],
                immediate_size: 0,
            });
        let indirect_convert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_indirect_convert.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INDIRECT_CONVERT_WGSL.into()),
        });
        // Override TETS_PER_GROUP to 16 for interval path
        let indirect_convert_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interval_indirect_convert_pipeline"),
                layout: Some(&indirect_convert_layout),
                module: &indirect_convert_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("TETS_PER_GROUP", 16.0)],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            mesh_render_pipeline,
            mesh_render_bg_layout,
            indirect_convert_pipeline,
            indirect_convert_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Compute-Based Interval Shading Pipelines (no mesh shader required)
// ---------------------------------------------------------------------------

/// Compiled pipelines for compute-based interval shading.
///
/// Replaces mesh shader with compute → vertex/fragment draw, making interval
/// shading available on all GPUs.
pub struct ComputeIntervalPipelines {
    pub gen_pipeline: wgpu::ComputePipeline,
    pub gen_bg_layout: wgpu::BindGroupLayout,
    pub render_pipeline: wgpu::RenderPipeline,
    /// Color-only render pipeline (no MRT)
    pub render_pipeline_color_only: wgpu::RenderPipeline,
    pub render_bg_layout: wgpu::BindGroupLayout,
    pub indirect_convert_pipeline: wgpu::ComputePipeline,
    pub indirect_convert_bg_layout: wgpu::BindGroupLayout,
}

impl ComputeIntervalPipelines {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // ----- Gen compute pipeline (11 bindings) -----
        // Binding 0 (uniforms) is bound as a UNIFORM buffer so the COMPUTE-stage
        // storage-buffer count is 10, fitting WebGPU's per-stage cap.
        //   1-7 read-only, 8-9 read-write (out_vertices, out_tet_data),
        //   10 read-only (vertex_normals).
        let gen_storage_read_only = [
            true, true, true, true, true, true, true, // 1-7 read-only
            false, false, // 8-9 read-write
            true,  // 10 read-only
        ];
        let mut gen_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, &read_only) in gen_storage_read_only.iter().enumerate() {
            gen_entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let gen_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_interval_gen_bg_layout"),
            entries: &gen_entries,
        });
        let gen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_interval_gen_pipeline_layout"),
            bind_group_layouts: &[&gen_bg_layout],
            immediate_size: 0,
        });
        let gen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_COMPUTE_WGSL.into()),
        });
        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_interval_gen_pipeline"),
            layout: Some(&gen_pipeline_layout),
            module: &gen_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ----- Render pipeline (6 read-only storage bindings) -----
        // 0: uniforms, 1: interval_vertex_buf, 2: interval_tet_data_buf
        // 3: aux_data, 4: vertex_normals, 5: tet_indices
        let render_read_only = [true; 6];
        let render_entries = storage_entries(
            6,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &render_read_only,
        );
        let render_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_interval_render_bg_layout"),
            entries: &render_entries,
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_interval_render_pipeline_layout"),
                bind_group_layouts: &[&render_bg_layout],
                immediate_size: 0,
            });

        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_vertex.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_VERTEX_WGSL.into()),
        });
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_fragment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_FRAGMENT_WGSL.into()),
        });

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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compute_interval_render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("main"),
                buffers: &[], // No vertex buffers — reads from storage
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No face culling — interval triangles are screen-aligned
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
                targets: &[
                    // location(0): color (plaster RGBA)
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(1): aux0 (roughness, env_feat[0..2])
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(2): normals (normal.xyz, env_feat[3]) — Rgba8Unorm bias-encoded.
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // location(3): depth_albedo (depth, albedo.rgb)
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(premul_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // ----- Indirect convert compute pipeline (2 bindings) -----
        let indirect_convert_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_interval_indirect_convert_bg_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let indirect_convert_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_interval_indirect_convert_pipeline_layout"),
                bind_group_layouts: &[&indirect_convert_bg_layout],
                immediate_size: 0,
            });
        let indirect_convert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_indirect_convert.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_INDIRECT_CONVERT_WGSL.into()),
        });
        let indirect_convert_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_interval_indirect_convert_pipeline"),
                layout: Some(&indirect_convert_layout),
                module: &indirect_convert_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let color_only_targets: &[Option<wgpu::ColorTargetState>] = &[
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(premul_blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            None,
            None,
            None,
        ];

        let render_pipeline_color_only =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("compute_interval_render_pipeline_color_only"),
                layout: Some(&render_pipeline_layout),
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
                    targets: color_only_targets,
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });

        Self {
            gen_pipeline,
            gen_bg_layout,
            render_pipeline,
            render_pipeline_color_only,
            render_bg_layout,
            indirect_convert_pipeline,
            indirect_convert_bg_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// Render Targets
// ---------------------------------------------------------------------------

/// The render target textures for MRT output.
pub struct RenderTargets {
    /// Main color output texture (e.g. Rgba16Float)
    pub color_texture: wgpu::Texture,
    /// Auxiliary output texture (Rgba16Float)
    pub aux0_texture: wgpu::Texture,
    /// Normals output texture (Rgba16Float, premultiplied alpha blend)
    pub normals_texture: wgpu::Texture,
    /// Depth output texture (Rgba16Float, premultiplied alpha blend)
    pub depth_texture: wgpu::Texture,
    /// AO output texture (R8Unorm) — written by GTAO pass, read by deferred.
    pub ao_texture: wgpu::Texture,
    /// AO bilateral-blur intermediate (R8Unorm). H pass writes here, V pass
    /// reads it and writes back to `ao_texture`.
    pub ao_blur_temp_texture: wgpu::Texture,
    /// SSGI radiance (Rgba16Float). Written by ssgi_compute, denoised in
    /// place via the SSGI bilateral, read by deferred.
    pub ssgi_texture: wgpu::Texture,
    /// SSGI bilateral H-pass intermediate.
    pub ssgi_blur_temp_texture: wgpu::Texture,
    /// SSGI temporal-pass output. Sits between ssgi_compute and the bilateral.
    pub ssgi_temporal_texture: wgpu::Texture,
    /// SSGI history (Rgba16Float). Updated each frame via copy from ssgi_view
    /// (the post-blur final). Sampled by the SSGI temporal pass.
    pub ssgi_history_texture: wgpu::Texture,
    /// AO temporal-pass output. Sits between GTAO and the AO bilateral.
    pub ao_temporal_texture: wgpu::Texture,
    /// AO history (R8Unorm). Same role as ssgi_history but for the GTAO output.
    pub ao_history_texture: wgpu::Texture,
    /// Current-frame "true lit" output from deferred shade location(1).
    /// Written each deferred pass; copied into lit_history afterward so SSGI
    /// has feedback data immune to debug-mode overrides on location(0).
    pub lit_current_texture: wgpu::Texture,
    /// Previous frame's true lit ("lit history") for SSGI to sample. Updated
    /// each frame via texture-to-texture copy from `lit_current_texture`.
    pub lit_history_texture: wgpu::Texture,
    /// SSR radiance (Rgba16Float, RGB = sampled radiance along reflection
    /// direction). Final post-temporal value lands here and is consumed by
    /// the deferred shader.
    pub ssr_texture: wgpu::Texture,
    /// SSR temporal-pass output. Sits between SSR compute and the per-frame
    /// copy-back into `ssr_texture`.
    pub ssr_temporal_texture: wgpu::Texture,
    /// SSR history (previous frame's post-temporal SSR). Sampled by the
    /// SSR temporal pass for reprojection.
    pub ssr_history_texture: wgpu::Texture,
    /// View into color texture
    pub color_view: wgpu::TextureView,
    /// View into aux texture
    pub aux0_view: wgpu::TextureView,
    /// View into normals texture
    pub normals_view: wgpu::TextureView,
    /// View into depth texture
    pub depth_view: wgpu::TextureView,
    pub ao_blur_temp_view: wgpu::TextureView,
    pub ssgi_blur_temp_view: wgpu::TextureView,
    pub ssgi_temporal_view: wgpu::TextureView,
    pub ao_temporal_view: wgpu::TextureView,
    pub ssr_temporal_view: wgpu::TextureView,
    /// Ping-pong pairs: index 0 = primary texture's view, 1 = secondary.
    /// Never index directly — use the `*_current(parity)` / `*_history(parity)`
    /// helpers below. Replaces the old current/history pattern that needed a
    /// per-frame copy_texture_to_texture; a single parity flip rotates roles.
    pub ao_views: [wgpu::TextureView; 2],
    pub ssgi_views: [wgpu::TextureView; 2],
    pub ssr_views: [wgpu::TextureView; 2],
    pub lit_views: [wgpu::TextureView; 2],
    pub width: u32,
    pub height: u32,
}

impl RenderTargets {
    /// AO output for this frame (write target + same-frame deferred read).
    #[inline]
    pub fn ao_current(&self, parity: u32) -> &wgpu::TextureView {
        &self.ao_views[(parity & 1) as usize]
    }
    /// Previous frame's AO output, sampled by this frame's AO temporal pass.
    #[inline]
    pub fn ao_history(&self, parity: u32) -> &wgpu::TextureView {
        &self.ao_views[((parity ^ 1) & 1) as usize]
    }
    #[inline]
    pub fn ssgi_current(&self, parity: u32) -> &wgpu::TextureView {
        &self.ssgi_views[(parity & 1) as usize]
    }
    #[inline]
    pub fn ssgi_history(&self, parity: u32) -> &wgpu::TextureView {
        &self.ssgi_views[((parity ^ 1) & 1) as usize]
    }
    #[inline]
    pub fn ssr_current(&self, parity: u32) -> &wgpu::TextureView {
        &self.ssr_views[(parity & 1) as usize]
    }
    #[inline]
    pub fn ssr_history(&self, parity: u32) -> &wgpu::TextureView {
        &self.ssr_views[((parity ^ 1) & 1) as usize]
    }
    /// Deferred shader's location(1) target this frame.
    #[inline]
    pub fn lit_current(&self, parity: u32) -> &wgpu::TextureView {
        &self.lit_views[(parity & 1) as usize]
    }
    /// Previous frame's lit value — sampled by SSGI/SSR and by DBG_LIT_HISTORY.
    #[inline]
    pub fn lit_history(&self, parity: u32) -> &wgpu::TextureView {
        &self.lit_views[((parity ^ 1) & 1) as usize]
    }

    /// Create render target textures at the given resolution.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let color_format = wgpu::TextureFormat::Rgba16Float;

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let aux0_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aux0_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Normals: Rgba8Unorm with bias encoding (n*0.5+0.5) halves bandwidth
        // vs Rgba16Float in the forward MRT write and every consumer. Producers
        // pre-normalize and bias the unit normal; consumers undo the bias and
        // re-normalize. Volume's additive alpha-blend still works because the
        // bias is linear (Σα·biased / Σα = biased of weighted-average direction).
        let normals_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normals_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // ao_texture is half of the ping-pong AO pair; copy flags are no longer
        // needed because the per-frame copy is gone (parity flip rotates roles).
        let ao_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ao_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let ao_blur_temp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ao_blur_temp"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let make_target =
            |label: &'static str, format: wgpu::TextureFormat, extra: wgpu::TextureUsages| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | extra,
                    view_formats: &[],
                })
            };
        let none = wgpu::TextureUsages::empty();
        let cdst = wgpu::TextureUsages::COPY_DST;
        let csrc = wgpu::TextureUsages::COPY_SRC;
        // Ping-pong pairs no longer need COPY flags — the parity flip replaces
        // the per-frame copies. Only ssr_texture keeps COPY_DST because the
        // ssr_temporal → ssr copy is still in place (out of scope here).
        let ssgi_texture = make_target("ssgi", wgpu::TextureFormat::Rgba16Float, none);
        let ssgi_blur_temp_texture =
            make_target("ssgi_blur_temp", wgpu::TextureFormat::Rgba16Float, none);
        let ssgi_temporal_texture =
            make_target("ssgi_temporal", wgpu::TextureFormat::Rgba16Float, none);
        let ssgi_history_texture =
            make_target("ssgi_history", wgpu::TextureFormat::Rgba16Float, none);
        let lit_current_texture =
            make_target("lit_current", wgpu::TextureFormat::Rgba16Float, none);
        let lit_history_texture =
            make_target("lit_history", wgpu::TextureFormat::Rgba16Float, none);
        let ao_temporal_texture = make_target("ao_temporal", wgpu::TextureFormat::R8Unorm, none);
        let ao_history_texture = make_target("ao_history", wgpu::TextureFormat::R8Unorm, none);
        // Both halves of the SSR ping-pong need COPY_DST because the still-in-
        // place ssr_temporal → ssr_current copy can target either slot depending
        // on parity. Killing that copy is the remaining SSR cleanup, out of
        // scope here.
        let ssr_texture = make_target("ssr", wgpu::TextureFormat::Rgba16Float, cdst);
        let ssr_temporal_texture =
            make_target("ssr_temporal", wgpu::TextureFormat::Rgba16Float, csrc);
        let ssr_history_texture =
            make_target("ssr_history", wgpu::TextureFormat::Rgba16Float, cdst);

        let v = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let color_view = v(&color_texture);
        let aux0_view = v(&aux0_texture);
        let normals_view = v(&normals_texture);
        let depth_view = v(&depth_texture);
        let ao_blur_temp_view = v(&ao_blur_temp_texture);
        let ssgi_blur_temp_view = v(&ssgi_blur_temp_texture);
        let ssgi_temporal_view = v(&ssgi_temporal_texture);
        let ao_temporal_view = v(&ao_temporal_texture);
        let ssr_temporal_view = v(&ssr_temporal_texture);
        let ao_views = [v(&ao_texture), v(&ao_history_texture)];
        let ssgi_views = [v(&ssgi_texture), v(&ssgi_history_texture)];
        let ssr_views = [v(&ssr_texture), v(&ssr_history_texture)];
        let lit_views = [v(&lit_current_texture), v(&lit_history_texture)];
        Self {
            color_texture,
            aux0_texture,
            normals_texture,
            depth_texture,
            ao_texture,
            ao_blur_temp_texture,
            ssgi_texture,
            ssgi_blur_temp_texture,
            ssgi_temporal_texture,
            ssgi_history_texture,
            ao_temporal_texture,
            ao_history_texture,
            lit_current_texture,
            lit_history_texture,
            ssr_texture,
            ssr_temporal_texture,
            ssr_history_texture,
            color_view,
            aux0_view,
            normals_view,
            depth_view,
            ao_blur_temp_view,
            ssgi_blur_temp_view,
            ssgi_temporal_view,
            ao_temporal_view,
            ssr_temporal_view,
            ao_views,
            ssgi_views,
            ssr_views,
            lit_views,
            width,
            height,
        }
    }
}

// ---------------------------------------------------------------------------
// Tex-to-Buffer Pipeline
// ---------------------------------------------------------------------------

/// Pipeline that converts the Rgba16Float render target to an f32 storage buffer.
///
/// The forward render pass outputs to a texture, but the loss and backward shaders
/// need a flat `array<f32>` storage buffer. This compute pipeline bridges the gap.
pub struct TexToBufferPipeline {
    pipeline: wgpu::ComputePipeline,
    _bind_group_layout: wgpu::BindGroupLayout,
    _params_buffer: wgpu::Buffer,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl TexToBufferPipeline {
    /// Create the pipeline and allocate the rendered_image buffer.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tex_to_buffer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(TEX_TO_BUFFER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex_to_buffer_bind_group_layout"),
            entries: &[
                // @binding(0) texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1) output buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) params (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex_to_buffer_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tex_to_buffer_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4, // RGBA f32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_data: [u32; 2] = [width, height];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tex_to_buffer_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex_to_buffer_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rendered_image.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            _bind_group_layout: bind_group_layout,
            _params_buffer: params_buffer,
            rendered_image,
            bind_group,
            width,
            height,
        }
    }
}

/// Record the tex-to-buffer conversion dispatch.
///
/// Should be called after the render pass finishes, within the same command encoder.
pub fn record_tex_to_buffer(encoder: &mut wgpu::CommandEncoder, ttb: &TexToBufferPipeline) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("tex_to_buffer"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&ttb.pipeline);
    pass.set_bind_group(0, &ttb.bind_group, &[]);
    pass.dispatch_workgroups(ttb.width.div_ceil(16), ttb.height.div_ceil(16), 1);
}

// ---------------------------------------------------------------------------
// Bind Groups
// ---------------------------------------------------------------------------

/// Create the compute bind groups (split 8 read-only + 6 read-write).
///
/// Binding order matches `project_compute.wgsl`:
///   group 0: 0=uniforms, 1=vertices, 2=indices, 3=densities,
///            4=color_grads, 5=circumdata, 6=base_colors_buf, 7=sh_coeffs
///   group 1: 0=colors, 1=sort_keys, 2=sort_values, 3=indirect_args,
///            4=tiles_touched, 5=compact_tet_ids
pub fn create_compute_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
) -> ForwardComputeBindGroups {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group_0"),
        layout: &pipelines.compute_bg0_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &buffers.densities),
            buf_entry(4, &material.color_grads),
            buf_entry(5, &buffers.circumdata),
            buf_entry(6, &material.base_colors),
            buf_entry(7, sh_coeffs_buf),
        ],
    });
    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group_1"),
        layout: &pipelines.compute_bg1_layout,
        entries: &[
            buf_entry(0, &material.colors),
            buf_entry(1, &buffers.sort_keys),
            buf_entry(2, &buffers.sort_values),
            buf_entry(3, &buffers.indirect_args),
            buf_entry(4, &buffers.tiles_touched),
            buf_entry(5, &buffers.compact_tet_ids),
        ],
    });
    ForwardComputeBindGroups { bg0, bg1 }
}

/// Create the HW projection compute bind group (11 bindings, single group).
///
/// Binding order matches `project_compute_hw.wgsl`:
///   0=uniforms, 1=vertices, 2=indices, 3=circumdata,
///   4=sort_keys, 5=sort_values, 6=indirect_args, 7=colors,
///   8=base_colors, 9=color_grads, 10=sh_coeffs
pub fn create_hw_compute_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
) -> ForwardHwComputeBindGroups {
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hw_compute_bind_group"),
        layout: &pipelines.hw_compute_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
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
    });
    ForwardHwComputeBindGroups { bg }
}

/// Create the render bind group (7 bindings).
///
/// Binding order matches `forward_vertex.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices (= sort_values)
pub fn create_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    create_render_bind_group_with_sort_values(
        device,
        pipelines,
        buffers,
        material,
        &buffers.sort_values,
    )
}

/// Create a render bind group with an explicit sort_values buffer.
///
/// Same as [`create_render_bind_group`] but binding 6 uses a caller-provided
/// `sort_values` buffer instead of `buffers.sort_values`. This is needed when
/// the radix sort result ends up in the alternate (B) buffer.
pub fn create_render_bind_group_with_sort_values(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group_sort_b"),
        layout: &pipelines.render_bind_group_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
        ],
    })
}

/// Create a prepass compute bind group (9 bindings).
///
/// Binding order matches `forward_prepass_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: sorted_indices,
///   4: indirect_args, 5: precomputed, 6: colors, 7: densities, 8: color_grads
pub fn create_prepass_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sorted_indices: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prepass_bind_group"),
        layout: &pipelines.prepass_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, sorted_indices),
            buf_entry(4, &buffers.indirect_args),
            buf_entry(5, &buffers.precomputed),
            buf_entry(6, &material.colors),
            buf_entry(7, &buffers.densities),
            buf_entry(8, &material.color_grads),
        ],
    })
}

/// Create a quad render bind group (2 bindings).
///
/// Binding order matches `forward_vertex_quad.wgsl`:
///   0: uniforms, 1: precomputed
pub fn create_quad_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("quad_render_bind_group"),
        layout: &pipelines.quad_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.precomputed),
        ],
    })
}

/// Create the mesh shader render bind group (8 bindings).
///
/// Binding order matches `forward_mesh.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices, 7: indirect_args
pub fn create_mesh_render_bind_group(
    device: &wgpu::Device,
    mesh_pipelines: &MeshForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    create_mesh_render_bind_group_with_sort_values(
        device,
        mesh_pipelines,
        buffers,
        material,
        &buffers.sort_values,
    )
}

/// Create a mesh render bind group with an explicit sort_values buffer.
///
/// Same as [`create_mesh_render_bind_group`] but binding 6 uses a caller-provided
/// `sort_values` buffer (the radix sort alternate B buffer).
pub fn create_mesh_render_bind_group_with_sort_values(
    device: &wgpu::Device,
    mesh_pipelines: &MeshForwardPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh_render_bind_group_sort_b"),
        layout: &mesh_pipelines.mesh_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
        ],
    })
}

/// Create the indirect convert bind group (2 bindings).
pub fn create_indirect_convert_bind_group(
    device: &wgpu::Device,
    mesh_pipelines: &MeshForwardPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("indirect_convert_bind_group"),
        layout: &mesh_pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.mesh_indirect_args),
        ],
    })
}

/// Create the interval mesh shader render bind group (8 bindings).
///
/// Binding order matches `interval_mesh.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: sorted_indices, 7: indirect_args
pub fn create_interval_render_bind_group(
    device: &wgpu::Device,
    interval_pipelines: &IntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    create_interval_render_bind_group_with_sort_values(
        device,
        interval_pipelines,
        buffers,
        material,
        &buffers.sort_values,
    )
}

/// Create an interval render bind group with an explicit sort_values buffer.
pub fn create_interval_render_bind_group_with_sort_values(
    device: &wgpu::Device,
    interval_pipelines: &IntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_render_bind_group_sort_b"),
        layout: &interval_pipelines.mesh_render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.vertices),
            buf_entry(2, &buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, sort_values),
            buf_entry(7, &buffers.indirect_args),
        ],
    })
}

/// Create the interval indirect convert bind group (2 bindings).
pub fn create_interval_indirect_convert_bind_group(
    device: &wgpu::Device,
    interval_pipelines: &IntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_indirect_convert_bind_group"),
        layout: &interval_pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.mesh_indirect_args),
        ],
    })
}

/// Create the compute-interval gen bind group (10 bindings).
///
/// Binding order matches `interval_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors, 4: densities,
///   5: color_grads, 6: sorted_indices (sort_values), 7: indirect_args,
///   8: interval_vertex_buf, 9: interval_tet_data_buf
pub fn create_compute_interval_gen_bind_group(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
) -> wgpu::BindGroup {
    create_compute_interval_gen_bind_group_with_sort_values(
        device,
        pipelines,
        buffers,
        material,
        &buffers.sort_values,
    )
}

/// Create a compute-interval gen bind group with an explicit sort_values buffer (B swap).
pub fn create_compute_interval_gen_bind_group_with_sort_values(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sort_values: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_gen_bg_sort_b"),
        layout: &pipelines.gen_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
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

/// Create the compute-interval render bind group (3 bindings).
///
/// Binding order matches `interval_vertex.wgsl`:
///   0: uniforms, 1: interval_vertex_buf, 2: interval_tet_data_buf
pub fn create_compute_interval_render_bind_group(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    // Dummy buffers for aux_data, vertex_normals, tet_indices when no PBR data
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ci_render_dummy"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_render_bg"),
        layout: &pipelines.render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.interval_vertex_buf),
            buf_entry(2, &buffers.interval_tet_data_buf),
            buf_entry(3, &dummy),                  // aux_data
            buf_entry(4, &buffers.vertex_normals), // vertex_normals
            buf_entry(5, &dummy),                  // tet_indices (use dummy, tet_id will be 0)
        ],
    })
}

/// Create the compute-interval render bind group with PBR material buffers.
pub fn create_compute_interval_render_bind_group_pbr(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    aux_data: &wgpu::Buffer,
    indices: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_render_bg_pbr"),
        layout: &pipelines.render_bg_layout,
        entries: &[
            buf_entry(0, &buffers.uniforms),
            buf_entry(1, &buffers.interval_vertex_buf),
            buf_entry(2, &buffers.interval_tet_data_buf),
            buf_entry(3, aux_data),
            buf_entry(4, &buffers.vertex_normals),
            buf_entry(5, indices),
        ],
    })
}

/// Create the compute-interval indirect convert bind group (2 bindings).
pub fn create_compute_interval_indirect_convert_bind_group(
    device: &wgpu::Device,
    pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_interval_indirect_convert_bg"),
        layout: &pipelines.indirect_convert_bg_layout,
        entries: &[
            buf_entry(0, &buffers.indirect_args),
            buf_entry(1, &buffers.interval_args_buf),
        ],
    })
}

// ---------------------------------------------------------------------------
// Command Recording
// ---------------------------------------------------------------------------

/// Record only the forward compute pass (no sort).
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys + tiles_touched + compact_tet_ids)
///
/// Use this with the scan-based tile pipeline which reads compact_tet_ids
/// directly instead of relying on sorted sort_values.
pub fn record_project_compute(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    compute_bg: &ForwardComputeBindGroups,
    tet_count: u32,
    queue: &wgpu::Queue,
) {
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // Clear tiles_touched so RTS vec4 padding elements are zero
    encoder.clear_buffer(&buffers.tiles_touched, 0, None);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.compute_pipeline);
        cpass.set_bind_group(0, &compute_bg.bg0, &[]);
        cpass.set_bind_group(1, &compute_bg.bg1, &[]);

        let workgroup_size = 64u32;
        let n_pow2 = tet_count.next_power_of_two();
        let total_workgroups = n_pow2.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

/// Record just the forward compute pass (no sort, no hardware rasterization).
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys)
pub fn record_project_compute_and_sort(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    compute_bg: &ForwardComputeBindGroups,
    tet_count: u32,
    queue: &wgpu::Queue,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.compute_pipeline);
        cpass.set_bind_group(0, &compute_bg.bg0, &[]);
        cpass.set_bind_group(1, &compute_bg.bg1, &[]);

        let workgroup_size = 64u32;
        let n_pow2 = tet_count.next_power_of_two();
        let total_workgroups = n_pow2.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

/// Record the full forward pass into a command encoder.
///
/// Stages:
///   1. Reset indirect args (vertex_count=12, instance_count=0)
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Render pass (hardware rasterization with MRT, draw_indirect)
///
/// The caller must have already written the Uniforms into `buffers.uniforms`
/// via `queue.write_buffer` before calling this function.
pub fn record_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ForwardPipelines,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &ForwardComputeBindGroups,
    render_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
) {
    // ----- 1. Reset indirect args -----
    // vertex_count=12 (4 tri faces), instance_count=0 (compute pass will atomicAdd)
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipelines.compute_pipeline);
        cpass.set_bind_group(0, &compute_bg.bg0, &[]);
        cpass.set_bind_group(1, &compute_bg.bg1, &[]);

        // Dispatch for n_pow2 threads: tet_count threads do real work,
        // padding threads (tet_count..n_pow2-1) initialize sort buffers.
        let workgroup_size = 64u32;
        let n_pow2 = tet_count.next_power_of_two();
        let total_workgroups = n_pow2.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Render pass -----
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward_render"),
            color_attachments: &[
                // Attachment 0: main color (premultiplied alpha blend, load to preserve primitives)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                // Attachment 1: auxiliary (no blending, overwrite)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.aux0_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                // Attachment 2: normals (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.normals_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                // Attachment 3: depth (premultiplied alpha blend)
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
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

        rpass.set_viewport(
            0.0,
            0.0,
            targets.width as f32,
            targets.height as f32,
            0.0,
            1.0,
        );
        rpass.set_scissor_rect(0, 0, targets.width, targets.height);
        rpass.set_pipeline(&pipelines.render_pipeline);
        rpass.set_bind_group(0, render_bg, &[]);

        // Draw indirect -- instance_count comes from compute pass atomicAdd
        rpass.draw_indirect(&buffers.indirect_args, 0);
    }
}

/// Record a sorted forward pass: project_compute → radix sort → render.
///
/// Unlike [`record_forward_pass`], this inserts a radix sort between the
/// compute and render passes so that tets are drawn back-to-front.
/// The sort uses ascending order on `~depth_bits` keys written by
/// `project_compute`, which gives correct back-to-front compositing with
/// the existing premultiplied alpha blend state (src=One, dst=OneMinusSrcAlpha).
///
/// The caller must provide two render bind groups:
/// - `render_bg_a`: uses `buffers.sort_values` (primary A buffer)
/// - `render_bg_b`: uses `sort_state.values_b` (alternate B buffer)
///
/// The function selects the correct bind group based on which buffer
/// holds the sorted result (depends on number of radix sort passes).
pub fn record_sorted_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipelines: &ForwardPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &ForwardComputeBindGroups,
    render_bg_a: &wgpu::BindGroup,
    render_bg_b: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&ForwardHwComputeBindGroups>,
    use_quad: bool,
    prepass_bg_a: Option<&wgpu::BindGroup>,
    prepass_bg_b: Option<&wgpu::BindGroup>,
    quad_render_bg: Option<&wgpu::BindGroup>,
    profiler: Option<&wgpu::QuerySet>,
    mrt_enabled: bool,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: if use_quad { 4 } else { 12 },
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));

    // Write sort element count to radix sort's num_keys_buf.
    // project_compute dispatches n_pow2 threads; padding threads write
    // sort_keys = 0xFFFFFFFF which sorts to the end.
    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass -----
    // Use lean HW projection shader when available (no tile counting work)
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, &hw_bg.bg, &[]);
        } else {
            cpass.set_pipeline(&pipelines.compute_pipeline);
            cpass.set_bind_group(0, &compute_bg.bg0, &[]);
            cpass.set_bind_group(1, &compute_bg.bg1, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = n_pow2.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort (back-to-front via ascending ~depth_bits) -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder,
        device,
        sort_pipelines,
        sort_state,
        &buffers.sort_keys,
        &buffers.sort_values,
    );

    // ----- 3.5. Compute prepass (quad path only) -----
    if use_quad {
        let prepass_bg = if result_in_b {
            prepass_bg_b.expect("prepass_bg_b required for quad path")
        } else {
            prepass_bg_a.expect("prepass_bg_a required for quad path")
        };
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_prepass"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&pipelines.prepass_compute_pipeline);
        cpass.set_bind_group(0, prepass_bg, &[]);
        // Dispatch enough threads: tet_count is upper bound on visible tets
        let workgroup_size = 64u32;
        let total_workgroups = tet_count.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 4. Render pass -----
    let render_bg = if use_quad {
        quad_render_bg.expect("quad_render_bg required for quad path")
    } else if result_in_b {
        render_bg_b
    } else {
        render_bg_a
    };

    let color_attachment = Some(wgpu::RenderPassColorAttachment {
        view: &targets.color_view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    });

    // MRT targets use LoadOp::Load to preserve primitive MRT data written earlier
    let mrt_load = wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    };

    let mrt_attachments;
    let color_only_attachments;
    let color_attachments: &[Option<wgpu::RenderPassColorAttachment>] = if mrt_enabled {
        mrt_attachments = [
            color_attachment,
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.aux0_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normals_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.depth_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
        ];
        &mrt_attachments
    } else {
        color_only_attachments = [color_attachment, None, None, None];
        &color_only_attachments
    };

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("forward_render"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
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
        if use_quad {
            if mrt_enabled {
                rpass.set_pipeline(&pipelines.quad_render_pipeline);
            } else {
                rpass.set_pipeline(&pipelines.quad_render_pipeline_color_only);
            }
        } else if mrt_enabled {
            rpass.set_pipeline(&pipelines.render_pipeline);
        } else {
            rpass.set_pipeline(&pipelines.render_pipeline_color_only);
        }
        rpass.set_bind_group(0, render_bg, &[]);

        rpass.draw_indirect(&buffers.indirect_args, 0);
    }
}

/// Record a sorted forward pass using mesh shaders instead of hardware vertex rasterization.
///
/// Steps 1-3 are identical to [`record_sorted_forward_pass`]:
///   1. Reset indirect args
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Radix sort
///   3.5. Indirect convert: turns `indirect_args.instance_count` into mesh dispatch args
///   4. Render pass with `draw_mesh_tasks_indirect`
pub fn record_sorted_mesh_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    mesh_pipelines: &MeshForwardPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &ForwardComputeBindGroups,
    mesh_render_bg_a: &wgpu::BindGroup,
    mesh_render_bg_b: &wgpu::BindGroup,
    indirect_convert_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&ForwardHwComputeBindGroups>,
    profiler: Option<&wgpu::QuerySet>,
    mrt_enabled: bool,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    // Reset mesh dispatch args to safe values (0 workgroups)
    queue.write_buffer(
        &buffers.mesh_indirect_args,
        0,
        bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
    );

    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, &hw_bg.bg, &[]);
        } else {
            cpass.set_pipeline(&fwd_pipelines.compute_pipeline);
            cpass.set_bind_group(0, &compute_bg.bg0, &[]);
            cpass.set_bind_group(1, &compute_bg.bg1, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = n_pow2.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder,
        device,
        sort_pipelines,
        sort_state,
        &buffers.sort_keys,
        &buffers.sort_values,
    );

    // ----- 3.5. Indirect convert: instance_count → mesh dispatch args -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("indirect_convert"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&mesh_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, indirect_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // ----- 4. Mesh shader render pass -----
    let mesh_bg = if result_in_b {
        mesh_render_bg_b
    } else {
        mesh_render_bg_a
    };

    let color_attachment = Some(wgpu::RenderPassColorAttachment {
        view: &targets.color_view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    });

    // MRT targets use LoadOp::Load to preserve primitive MRT data written earlier
    let mrt_load = wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    };

    let mrt_attachments;
    let color_only_attachments;
    let color_attachments: &[Option<wgpu::RenderPassColorAttachment>] = if mrt_enabled {
        mrt_attachments = [
            color_attachment,
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.aux0_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normals_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.depth_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
        ];
        &mrt_attachments
    } else {
        color_only_attachments = [color_attachment, None, None, None];
        &color_only_attachments
    };

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_forward_render"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
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
        if mrt_enabled {
            rpass.set_pipeline(&mesh_pipelines.mesh_render_pipeline);
        } else {
            rpass.set_pipeline(&mesh_pipelines.mesh_render_pipeline_color_only);
        }
        rpass.set_bind_group(0, mesh_bg, &[]);

        rpass.draw_mesh_tasks_indirect(&buffers.mesh_indirect_args, 0);
    }
}

/// Record a sorted forward pass using the interval shading path.
///
/// Steps 1-3 are identical to [`record_sorted_mesh_forward_pass`]:
///   1. Reset indirect args
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Radix sort
///   3.5. Indirect convert (TETS_PER_GROUP=16)
///   4. Render pass with interval mesh shader + single color output
pub fn record_sorted_interval_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    interval_pipelines: &IntervalPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &ForwardComputeBindGroups,
    interval_render_bg_a: &wgpu::BindGroup,
    interval_render_bg_b: &wgpu::BindGroup,
    indirect_convert_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&ForwardHwComputeBindGroups>,
    profiler: Option<&wgpu::QuerySet>,
) {
    // ----- 1. Reset indirect args -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    queue.write_buffer(
        &buffers.mesh_indirect_args,
        0,
        bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
    );

    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        if let Some(hw_bg) = hw_compute_bg {
            cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
            cpass.set_bind_group(0, &hw_bg.bg, &[]);
        } else {
            cpass.set_pipeline(&fwd_pipelines.compute_pipeline);
            cpass.set_bind_group(0, &compute_bg.bg0, &[]);
            cpass.set_bind_group(1, &compute_bg.bg1, &[]);
        }

        let workgroup_size = 64u32;
        let total_workgroups = n_pow2.div_ceil(workgroup_size);
        let max_per_dim = 65535u32;
        let dispatch_x = total_workgroups.min(max_per_dim);
        let dispatch_y = total_workgroups.div_ceil(max_per_dim);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ----- 3. Radix sort -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder,
        device,
        sort_pipelines,
        sort_state,
        &buffers.sort_keys,
        &buffers.sort_values,
    );

    // ----- 3.5. Indirect convert: instance_count → mesh dispatch args (TETS_PER_GROUP=16) -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("interval_indirect_convert"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&interval_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, indirect_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // ----- 4. Interval shading render pass (single color output) -----
    let render_bg = if result_in_b {
        interval_render_bg_b
    } else {
        interval_render_bg_a
    };
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("interval_forward_render"),
            color_attachments: &[
                // Single color attachment: premultiplied alpha blend
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
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
        rpass.set_pipeline(&interval_pipelines.mesh_render_pipeline);
        rpass.set_bind_group(0, render_bg, &[]);

        rpass.draw_mesh_tasks_indirect(&buffers.mesh_indirect_args, 0);
    }
}

/// Record a sorted forward pass using compute-based interval shading (no mesh shader needed).
///
/// Steps:
///   1. Reset indirect args + interval_args_buf
///   2. Compute pass (SH eval + cull + depth keys)
///   3. Radix sort
///   4. Indirect convert: writes both compute dispatch + draw-indexed-indirect args
///   5. Compute pass: interval_compute generates vertices + per-tet data
///   6. Render pass with instanced indexed draw + interval_fragment (single color output)
/// Record steps 1 + 2 of the compute-interval pipeline: reset the indirect
/// args and dispatch the project compute (SH eval + frustum cull + depth
/// keys + `atomicAdd` on `indirect_args.instance_count`).
///
/// Extracted so callers that want to substitute a CPU sort can `queue.submit`
/// after this call, overwrite `sort_values` / `indirect_args.instance_count`
/// via `queue.write_buffer`, and then call
/// [`record_compute_interval_after_sort`] in a second encoder.
/// `record_sorted_compute_interval_forward_pass` calls both halves with the
/// GPU radix sort in between.
pub fn record_compute_interval_project(
    encoder: &mut wgpu::CommandEncoder,
    fwd_pipelines: &ForwardPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    compute_bg: &ForwardComputeBindGroups,
    hw_compute_bg: Option<&ForwardHwComputeBindGroups>,
    tet_count: u32,
    queue: &wgpu::Queue,
    profiler: Option<&wgpu::QuerySet>,
    use_16bit_sort: bool,
) {
    // ----- 1. Reset indirect args + interval_args_buf -----
    let reset_cmd = DrawIndirectCommand {
        vertex_count: 12,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    queue.write_buffer(&buffers.indirect_args, 0, bytemuck::bytes_of(&reset_cmd));
    queue.write_buffer(
        &buffers.interval_args_buf,
        0,
        bytemuck::cast_slice(&[0u32; 8]),
    );

    let n_pow2 = tet_count.next_power_of_two();
    queue.write_buffer(sort_state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    // ----- 2. Compute pass (projection + SH eval) -----
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("project_compute"),
        timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }),
    });
    if use_16bit_sort {
        cpass.set_pipeline(&fwd_pipelines.compute_pipeline_16bit);
        cpass.set_bind_group(0, &compute_bg.bg0, &[]);
        cpass.set_bind_group(1, &compute_bg.bg1, &[]);
    } else if let Some(hw_bg) = hw_compute_bg {
        cpass.set_pipeline(&fwd_pipelines.hw_compute_pipeline);
        cpass.set_bind_group(0, &hw_bg.bg, &[]);
    } else {
        cpass.set_pipeline(&fwd_pipelines.compute_pipeline);
        cpass.set_bind_group(0, &compute_bg.bg0, &[]);
        cpass.set_bind_group(1, &compute_bg.bg1, &[]);
    }

    let workgroup_size = 64u32;
    let total_workgroups = n_pow2.div_ceil(workgroup_size);
    let max_per_dim = 65535u32;
    let dispatch_x = total_workgroups.min(max_per_dim);
    let dispatch_y = total_workgroups.div_ceil(max_per_dim);
    cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
}

/// Record steps 4–6 of the compute-interval pipeline: indirect-convert (turns
/// `indirect_args.instance_count` into dispatch+draw args), gen compute
/// (writes interval vertex/tet-data buffers), and the interval render pass.
///
/// `gen_bg` selects which sort-values buffer the gen step reads (A or B).
/// For the GPU-sort path the caller picks based on
/// [`rmesh_sort::record_radix_sort`]'s `result_in_b` return; for the CPU-sort
/// path it's always A because the caller wrote the sorted indices into
/// `buffers.sort_values` (which is what the A bind group references).
pub fn record_compute_interval_after_sort(
    encoder: &mut wgpu::CommandEncoder,
    ci_pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    gen_bg: &wgpu::BindGroup,
    ci_render_bg: &wgpu::BindGroup,
    ci_convert_bg: &wgpu::BindGroup,
    depth_view: &wgpu::TextureView,
    profiler: Option<&wgpu::QuerySet>,
    mrt_enabled: bool,
) {
    record_compute_interval_after_sort_impl(
        encoder,
        ci_pipelines,
        buffers,
        targets,
        gen_bg,
        ci_render_bg,
        ci_convert_bg,
        depth_view,
        profiler,
        mrt_enabled,
    );
}

pub fn record_sorted_compute_interval_forward_pass(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    fwd_pipelines: &ForwardPipelines,
    ci_pipelines: &ComputeIntervalPipelines,
    sort_pipelines: &rmesh_sort::RadixSortPipelines,
    sort_state: &rmesh_sort::RadixSortState,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    compute_bg: &ForwardComputeBindGroups,
    gen_bg_a: &wgpu::BindGroup,
    gen_bg_b: &wgpu::BindGroup,
    ci_render_bg: &wgpu::BindGroup,
    ci_convert_bg: &wgpu::BindGroup,
    tet_count: u32,
    queue: &wgpu::Queue,
    depth_view: &wgpu::TextureView,
    hw_compute_bg: Option<&ForwardHwComputeBindGroups>,
    profiler: Option<&wgpu::QuerySet>,
    use_16bit_sort: bool,
    mrt_enabled: bool,
) {
    // Steps 1 + 2: reset + project compute
    record_compute_interval_project(
        encoder,
        fwd_pipelines,
        sort_state,
        buffers,
        compute_bg,
        hw_compute_bg,
        tet_count,
        queue,
        profiler,
        use_16bit_sort,
    );

    // ----- 3. Radix sort -----
    let result_in_b = rmesh_sort::record_radix_sort(
        encoder,
        device,
        sort_pipelines,
        sort_state,
        &buffers.sort_keys,
        &buffers.sort_values,
    );

    // Steps 4 + 5 + 6: indirect convert + gen + render
    let gen_bg = if result_in_b { gen_bg_b } else { gen_bg_a };
    record_compute_interval_after_sort_impl(
        encoder,
        ci_pipelines,
        buffers,
        targets,
        gen_bg,
        ci_render_bg,
        ci_convert_bg,
        depth_view,
        profiler,
        mrt_enabled,
    );
}

/// Shared body of steps 4-6 (factored so the wrapper and the public
/// `record_compute_interval_after_sort` both go through one implementation).
fn record_compute_interval_after_sort_impl(
    encoder: &mut wgpu::CommandEncoder,
    ci_pipelines: &ComputeIntervalPipelines,
    buffers: &SceneBuffers,
    targets: &RenderTargets,
    gen_bg: &wgpu::BindGroup,
    ci_render_bg: &wgpu::BindGroup,
    ci_convert_bg: &wgpu::BindGroup,
    depth_view: &wgpu::TextureView,
    profiler: Option<&wgpu::QuerySet>,
    mrt_enabled: bool,
) {
    // ----- 4. Indirect convert → combined dispatch + draw args -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_interval_indirect_convert"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2),
                end_of_pass_write_index: Some(3),
            }),
        });
        cpass.set_pipeline(&ci_pipelines.indirect_convert_pipeline);
        cpass.set_bind_group(0, ci_convert_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // ----- 5. Compute pass: generate interval vertices + indices -----
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_interval_gen"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ci_pipelines.gen_pipeline);
        cpass.set_bind_group(0, gen_bg, &[]);
        cpass.dispatch_workgroups_indirect(&buffers.interval_args_buf, 0);
    }

    // ----- 6. Render pass (MRT: color + aux0 + normals + depth_albedo) -----
    let color_ops = wgpu::Operations {
        load: wgpu::LoadOp::Load, // preserve primitive pass output
        store: wgpu::StoreOp::Store,
    };
    // MRT targets use LoadOp::Load to preserve primitive MRT data written earlier
    let mrt_load = wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    };

    let color_attachment = Some(wgpu::RenderPassColorAttachment {
        view: &targets.color_view,
        resolve_target: None,
        ops: color_ops,
        depth_slice: None,
    });

    let mrt_attachments;
    let color_only_attachments;
    let color_attachments: &[Option<wgpu::RenderPassColorAttachment>] = if mrt_enabled {
        mrt_attachments = [
            color_attachment,
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.aux0_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.normals_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &targets.depth_view,
                resolve_target: None,
                ops: mrt_load,
                depth_slice: None,
            }),
        ];
        &mrt_attachments
    } else {
        color_only_attachments = [color_attachment, None, None, None];
        &color_only_attachments
    };

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("compute_interval_render"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: profiler.map(|qs| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
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
        if mrt_enabled {
            rpass.set_pipeline(&ci_pipelines.render_pipeline);
        } else {
            rpass.set_pipeline(&ci_pipelines.render_pipeline_color_only);
        }
        rpass.set_bind_group(0, ci_render_bg, &[]);
        rpass.set_index_buffer(
            buffers.interval_fan_index_buf.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        // draw-indexed-indirect args start at byte offset 12 (skip 3 dispatch u32s)
        rpass.draw_indexed_indirect(&buffers.interval_args_buf, 12);
    }
}

// ---------------------------------------------------------------------------
// High-level helpers
// ---------------------------------------------------------------------------

/// Convenience: set up everything needed for a forward frame.
///
/// Returns (SceneBuffers, MaterialBuffers, ForwardPipelines, RenderTargets, compute_bg, render_bg).
pub fn setup_forward(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneData,
    base_colors: &[f32],
    color_grads: &[f32],
    width: u32,
    height: u32,
) -> (
    SceneBuffers,
    MaterialBuffers,
    ForwardPipelines,
    RenderTargets,
    ForwardComputeBindGroups,
    wgpu::BindGroup,
) {
    let color_format = wgpu::TextureFormat::Rgba16Float;

    let buffers = SceneBuffers::upload(device, queue, scene);
    let material = MaterialBuffers::upload(device, base_colors, color_grads, scene.tet_count);
    let pipelines = ForwardPipelines::new(device, color_format);
    let targets = RenderTargets::new(device, width, height);

    // Dummy sh_coeffs buffer (sh_degree=0 path uses base_colors instead)
    let dummy_sh = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_sh_coeffs"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let compute_bg = create_compute_bind_group(device, &pipelines, &buffers, &material, &dummy_sh);
    let render_bg = create_render_bind_group(device, &pipelines, &buffers, &material);

    (buffers, material, pipelines, targets, compute_bg, render_bg)
}

/// Build a `Uniforms` struct from camera matrices and scene metadata.
pub fn make_uniforms(
    vp: Mat4,
    c2w: glam::Mat3,
    intrinsics: [f32; 4],
    cam_pos: glam::Vec3,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size: u32,
    min_t: f32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
) -> Uniforms {
    Uniforms {
        vp_col0: vp.col(0).into(),
        vp_col1: vp.col(1).into(),
        vp_col2: vp.col(2).into(),
        vp_col3: vp.col(3).into(),
        c2w_col0: [c2w.col(0).x, c2w.col(0).y, c2w.col(0).z, 0.0],
        c2w_col1: [c2w.col(1).x, c2w.col(1).y, c2w.col(1).z, 0.0],
        c2w_col2: [c2w.col(2).x, c2w.col(2).y, c2w.col(2).z, 0.0],
        intrinsics,
        cam_pos_pad: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
        screen_width,
        screen_height,
        tet_count,
        step,
        tile_size_u: tile_size,
        ray_mode: 0,
        min_t,
        sh_degree,
        near_plane,
        far_plane,
        _pad1: [0; 2],
    }
}

// ---------------------------------------------------------------------------
// Blit Pipeline (Rgba16Float → sRGB swapchain)
// ---------------------------------------------------------------------------

/// Pipeline that blits the Rgba16Float render target to the sRGB swapchain
/// via a fullscreen triangle.
pub struct BlitPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
}

impl BlitPipeline {
    /// Create the blit pipeline targeting `target_format` (e.g. Bgra8UnormSrgb).
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
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
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }
}

const BLIT_NF_WGSL: &str = "
// Fullscreen triangle blit using textureLoad (no sampler, works with non-filterable formats).

@group(0) @binding(0) var src_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn linear_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.pos.xy);
    let color = textureLoad(src_tex, coord, 0);
    return vec4<f32>(
        linear_to_srgb(color.r),
        linear_to_srgb(color.g),
        linear_to_srgb(color.b),
        1.0,
    );
}
";

/// A blit pipeline for non-filterable textures (e.g. Rgba32Float) using `textureLoad`.
pub struct BlitPipelineNonFiltering {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl BlitPipelineNonFiltering {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit_nf.wgsl"),
            source: wgpu::ShaderSource::Wgsl(BLIT_NF_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_nf_bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_nf_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_nf_pipeline"),
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
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Create a bind group for the non-filtering blit pipeline.
pub fn create_blit_nf_bind_group(
    device: &wgpu::Device,
    blit: &BlitPipelineNonFiltering,
    source_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_nf_bg"),
        layout: &blit.bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(source_view),
        }],
    })
}

/// Create a bind group for the blit pipeline from a source texture view.
pub fn create_blit_bind_group(
    device: &wgpu::Device,
    blit: &BlitPipeline,
    source_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bg"),
        layout: &blit.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&blit.sampler),
            },
        ],
    })
}

/// Record a blit render pass: fullscreen triangle sampling `source` to `target_view`.
pub fn record_blit(
    encoder: &mut wgpu::CommandEncoder,
    blit: &BlitPipeline,
    bind_group: &wgpu::BindGroup,
    target_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("blit"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
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
    rpass.set_pipeline(&blit.pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.draw(0..3, 0..1); // fullscreen triangle
}

/// Record a blit render pass using the non-filtering pipeline.
pub fn record_blit_nf(
    encoder: &mut wgpu::CommandEncoder,
    blit: &BlitPipelineNonFiltering,
    bind_group: &wgpu::BindGroup,
    target_view: &wgpu::TextureView,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("blit_nf"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
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
    rpass.set_pipeline(&blit.pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.draw(0..3, 0..1);
}


// Interval tiled pipeline (`IntervalTiledBuffers`, `IntervalGeneratePipeline`,
// `IntervalTiledRasterizePipeline`) moved to `rmesh-trainable`.
