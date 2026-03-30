//! Backward pass orchestration (wgpu port).
//!
//! Manages:
//!   - Backward compute dispatch (2 bind groups for WebGPU 8-buffer limit)
//!   - Backward tiled compute dispatch
//!   - Gradient buffer management

use rmesh_util::shared;
use wgpu;

// Re-export shared uniform types for downstream crates.
pub use shared::TileUniforms;

// Re-export sort types (moved to rmesh-sort crate).
pub use rmesh_sort::{
    RadixSortPipelines, RadixSortState, SortBackend,
    record_radix_sort, sorting_bits_for_tiles,
};

// Re-export tile types (moved to rmesh-tile crate).
pub use rmesh_tile::{
    TileBuffers, TilePipelines,
    ScanPipelines, ScanBuffers,
    create_tile_fill_bind_group, create_tile_ranges_bind_group,
    create_tile_ranges_bind_group_with_keys,
    create_prepare_dispatch_bind_group, create_rts_bind_group,
    create_tile_gen_scan_bind_group,
    record_scan_tile_pipeline,
};

// WGSL shader sources, embedded from crate-local files.
const BACKWARD_TILED_WGSL: &str = include_str!("wgsl/backward_tiled_compute.wgsl");
const BACKWARD_INTERVAL_TILED_WGSL: &str = include_str!("wgsl/backward_interval_tiled.wgsl");
const INTERVAL_CHAIN_BACK_WGSL: &str = include_str!("wgsl/interval_chain_back.wgsl");

/// Gradient buffers for scene geometry parameters.
pub struct GradientBuffers {
    pub d_vertices: wgpu::Buffer,
    pub d_densities: wgpu::Buffer,
}

/// Gradient buffers for material/appearance parameters.
pub struct MaterialGradBuffers {
    pub d_base_colors: wgpu::Buffer,
    pub d_color_grads: wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// Buffer creation helpers
// ---------------------------------------------------------------------------

/// Create a zero-initialized storage buffer with COPY_DST (for clearing) and COPY_SRC (for readback).
fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// GradientBuffers
// ---------------------------------------------------------------------------

impl GradientBuffers {
    /// Allocate zero-initialized geometry gradient buffers.
    pub fn new(
        device: &wgpu::Device,
        vertex_count: u32,
        tet_count: u32,
    ) -> Self {
        Self {
            d_vertices: create_storage_buffer(
                device,
                "d_vertices",
                (vertex_count as u64) * 3 * 4,
            ),
            d_densities: create_storage_buffer(
                device,
                "d_densities",
                (tet_count as u64) * 4,
            ),
        }
    }
}

impl MaterialGradBuffers {
    /// Allocate zero-initialized material gradient buffers.
    pub fn new(
        device: &wgpu::Device,
        tet_count: u32,
    ) -> Self {
        Self {
            d_base_colors: create_storage_buffer(
                device,
                "d_base_colors",
                (tet_count as u64) * 3 * 4,
            ),
            d_color_grads: create_storage_buffer(
                device,
                "d_color_grads",
                (tet_count as u64) * 3 * 4,
            ),
        }
    }
}

// ===========================================================================
// Backward tiled pipeline
// ===========================================================================

/// Pipeline for the backward tiled pass (warp-per-tile gradient computation).
pub struct BackwardTiledPipelines {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout_0: wgpu::BindGroupLayout,
    pub bg_layout_1: wgpu::BindGroupLayout,
}

impl BackwardTiledPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backward_tiled_compute"),
            source: wgpu::ShaderSource::Wgsl(BACKWARD_TILED_WGSL.into()),
        });

        // Group 0: 9 read-only bindings
        let bg_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("backward_tiled_bgl_0"),
                entries: &[
                    storage_entry(0, true),  // uniforms
                    storage_entry(1, true),  // dl_d_image
                    storage_entry(2, true),  // rendered_image
                    storage_entry(3, true),  // vertices
                    storage_entry(4, true),  // indices
                    storage_entry(5, true),  // densities
                    storage_entry(6, true),  // color_grads
                    storage_entry(7, true),  // colors_buf
                    storage_entry(8, true),  // tile_sort_values
                ],
            });

        // Group 1: 6 bindings (3 rw + 2 read + 1 rw)
        let bg_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("backward_tiled_bgl_1"),
                entries: &[
                    storage_entry(0, false), // d_vertices
                    storage_entry(1, false), // d_densities
                    storage_entry(2, false), // d_color_grads
                    storage_entry(3, true),  // tile_ranges
                    storage_entry(4, true),  // tile_uniforms
                    storage_entry(5, false), // d_base_colors
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("backward_tiled_pl"),
                bind_group_layouts: &[&bg_layout_0, &bg_layout_1],
                immediate_size: 0,
            });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("backward_tiled_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self { pipeline, bg_layout_0, bg_layout_1 }
    }
}

// ---------------------------------------------------------------------------
// Bind group creation
// ---------------------------------------------------------------------------

/// Create the backward tiled bind groups.
///
/// Returns `(bg0, bg1)`.
pub fn create_backward_tiled_bind_groups(
    device: &wgpu::Device,
    pipelines: &BackwardTiledPipelines,
    uniforms: &wgpu::Buffer,
    dl_d_image: &wgpu::Buffer,
    rendered_image: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    densities: &wgpu::Buffer,
    color_grads: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    tile_sort_values: &wgpu::Buffer,
    d_vertices: &wgpu::Buffer,
    d_densities: &wgpu::Buffer,
    d_color_grads: &wgpu::Buffer,
    d_base_colors: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_tiled_bg_0"),
        layout: &pipelines.bg_layout_0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dl_d_image.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: rendered_image.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: colors.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: tile_sort_values.as_entire_binding() },
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_tiled_bg_1"),
        layout: &pipelines.bg_layout_1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: d_vertices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: d_densities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: d_color_grads.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tile_ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: tile_uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: d_base_colors.as_entire_binding() },
        ],
    });

    (bg0, bg1)
}

// ---------------------------------------------------------------------------
// Recording functions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Helper to create a storage buffer bind group layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Shorthand for a full-buffer bind group entry.
fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

// ===========================================================================
// Interval gradient buffers
// ===========================================================================

/// Gradient buffers for interval tiled backward pass (screen-space gradients).
pub struct IntervalGradBuffers {
    /// Per-vertex gradients: [max_visible × 5 × 10] f32
    /// (d_zf, d_zb, d_off_f, d_off_b, d_nf.xyz, d_nb.xyz)
    pub d_interval_verts: wgpu::Buffer,
    /// Per-tet gradients: [max_visible × 4] f32 (d_base_color.rgb, d_density)
    pub d_interval_tet_data: wgpu::Buffer,
    /// Per-vertex normal gradients: [N × 3] f32
    pub d_vertex_normals: wgpu::Buffer,
    /// Per-tet aux gradients: [tet_count × aux_dim] f32
    pub d_aux_data: wgpu::Buffer,
}

impl IntervalGradBuffers {
    pub fn new(device: &wgpu::Device, max_visible: u32, vertex_count: u32, tet_count: u32, aux_dim: usize) -> Self {
        let m = max_visible as u64;
        let n = vertex_count as u64;
        let aux_size = if aux_dim > 0 { (tet_count as u64) * (aux_dim as u64) * 4 } else { 4 };
        Self {
            d_interval_verts: create_storage_buffer(
                device,
                "d_interval_verts",
                m * 5 * 10 * 4, // 5 verts × 10 floats × 4 bytes
            ),
            d_interval_tet_data: create_storage_buffer(
                device,
                "d_interval_tet_data",
                m * 4 * 4, // 4 floats × 4 bytes
            ),
            d_vertex_normals: create_storage_buffer(
                device,
                "d_vertex_normals",
                n * 3 * 4, // N × 3 f32
            ),
            d_aux_data: create_storage_buffer(
                device,
                "d_aux_data",
                aux_size,
            ),
        }
    }
}

// ===========================================================================
// Backward interval tiled pipeline
// ===========================================================================

/// Pipeline for the backward interval tiled pass.
pub struct BackwardIntervalTiledPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout_0: wgpu::BindGroupLayout,
    pub bg_layout_1: wgpu::BindGroupLayout,
    pub aux_dim: usize,
}

impl BackwardIntervalTiledPipeline {
    pub fn new(device: &wgpu::Device, aux_dim: usize) -> Self {
        let aux_dim_max = if aux_dim > 0 { aux_dim } else { 1 };
        let source = BACKWARD_INTERVAL_TILED_WGSL
            .replace("/*AUX_DIM*/0u", &format!("{}u", aux_dim))
            .replace("/*AUX_DIM_MAX*/1u", &format!("{}u", aux_dim_max));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backward_interval_tiled"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Group 0: 13 read-only bindings (11 original + 2 aux)
        let bg_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backward_interval_tiled_bgl_0"),
            entries: &[
                storage_entry(0, true),  // uniforms
                storage_entry(1, true),  // dl_d_image
                storage_entry(2, true),  // rendered_image
                storage_entry(3, true),  // interval_verts
                storage_entry(4, true),  // interval_tet_data
                storage_entry(5, true),  // interval_meta
                storage_entry(6, true),  // tile_sort_values
                storage_entry(7, true),  // dl_d_xyzd
                storage_entry(8, true),  // dl_d_distortion
                storage_entry(9, true),  // xyzd_image
                storage_entry(10, true), // distortion_image
                storage_entry(11, true), // aux_data
                storage_entry(12, true), // dl_d_aux
            ],
        });

        // Group 1: 3 rw + 2 read
        let bg_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backward_interval_tiled_bgl_1"),
            entries: &[
                storage_entry(0, false), // d_interval_verts
                storage_entry(1, false), // d_interval_tet_data
                storage_entry(2, true),  // tile_ranges
                storage_entry(3, true),  // tile_uniforms
                storage_entry(4, false), // d_aux_data
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("backward_interval_tiled_pl"),
            bind_group_layouts: &[&bg_layout_0, &bg_layout_1],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("backward_interval_tiled_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bg_layout_0, bg_layout_1, aux_dim }
    }
}

/// Create the backward interval tiled bind groups.
///
/// Returns `(bg0, bg1)`.
pub fn create_backward_interval_tiled_bind_groups(
    device: &wgpu::Device,
    pipelines: &BackwardIntervalTiledPipeline,
    uniforms: &wgpu::Buffer,
    dl_d_image: &wgpu::Buffer,
    rendered_image: &wgpu::Buffer,
    interval_verts: &wgpu::Buffer,
    interval_tet_data: &wgpu::Buffer,
    interval_meta: &wgpu::Buffer,
    tile_sort_values: &wgpu::Buffer,
    d_interval_verts: &wgpu::Buffer,
    d_interval_tet_data: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
    dl_d_xyzd: &wgpu::Buffer,
    dl_d_distortion: &wgpu::Buffer,
    xyzd_image: &wgpu::Buffer,
    distortion_image: &wgpu::Buffer,
    aux_data: &wgpu::Buffer,
    dl_d_aux: &wgpu::Buffer,
    d_aux_data: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_interval_tiled_bg_0"),
        layout: &pipelines.bg_layout_0,
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, dl_d_image),
            buf_entry(2, rendered_image),
            buf_entry(3, interval_verts),
            buf_entry(4, interval_tet_data),
            buf_entry(5, interval_meta),
            buf_entry(6, tile_sort_values),
            buf_entry(7, dl_d_xyzd),
            buf_entry(8, dl_d_distortion),
            buf_entry(9, xyzd_image),
            buf_entry(10, distortion_image),
            buf_entry(11, aux_data),
            buf_entry(12, dl_d_aux),
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backward_interval_tiled_bg_1"),
        layout: &pipelines.bg_layout_1,
        entries: &[
            buf_entry(0, d_interval_verts),
            buf_entry(1, d_interval_tet_data),
            buf_entry(2, tile_ranges),
            buf_entry(3, tile_uniforms),
            buf_entry(4, d_aux_data),
        ],
    });

    (bg0, bg1)
}

/// Record the backward interval tiled compute pass dispatch.
pub fn record_backward_interval_tiled(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &BackwardIntervalTiledPipeline,
    bg0: &wgpu::BindGroup,
    bg1: &wgpu::BindGroup,
    num_tiles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("backward_interval_tiled"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, bg0, &[]);
    pass.set_bind_group(1, bg1, &[]);
    let max_per_dim = 65535u32;
    let x = num_tiles.min(max_per_dim);
    let y = (num_tiles + max_per_dim - 1) / max_per_dim;
    pass.dispatch_workgroups(x, y, 1);
}

// ===========================================================================
// Interval chain-back pipeline
// ===========================================================================

/// Pipeline for the interval chain-back pass (per-tet, maps screen grads to tet params).
pub struct IntervalChainBackPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout_0: wgpu::BindGroupLayout,
    pub bg_layout_1: wgpu::BindGroupLayout,
}

impl IntervalChainBackPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_chain_back"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_CHAIN_BACK_WGSL.into()),
        });

        // Group 0: 10 read-only bindings
        let bg_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interval_chain_back_bgl_0"),
            entries: &[
                storage_entry(0, true),  // uniforms
                storage_entry(1, true),  // vertices
                storage_entry(2, true),  // indices
                storage_entry(3, true),  // color_grads
                storage_entry(4, true),  // compact_tet_ids
                storage_entry(5, true),  // indirect_args
                storage_entry(6, true),  // interval_meta
                storage_entry(7, true),  // d_interval_verts
                storage_entry(8, true),  // d_interval_tet_data
                storage_entry(9, true),  // vertex_normals
            ],
        });

        // Group 1: 5 rw gradient outputs
        let bg_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interval_chain_back_bgl_1"),
            entries: &[
                storage_entry(0, false), // d_vertices
                storage_entry(1, false), // d_densities
                storage_entry(2, false), // d_color_grads
                storage_entry(3, false), // d_base_colors
                storage_entry(4, false), // d_vertex_normals
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interval_chain_back_pl"),
            bind_group_layouts: &[&bg_layout_0, &bg_layout_1],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("interval_chain_back_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bg_layout_0, bg_layout_1 }
    }
}

/// Create the interval chain-back bind groups.
///
/// Returns `(bg0, bg1)`.
pub fn create_interval_chain_back_bind_groups(
    device: &wgpu::Device,
    pipelines: &IntervalChainBackPipeline,
    uniforms: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    color_grads: &wgpu::Buffer,
    compact_tet_ids: &wgpu::Buffer,
    indirect_args: &wgpu::Buffer,
    interval_meta: &wgpu::Buffer,
    d_interval_verts: &wgpu::Buffer,
    d_interval_tet_data: &wgpu::Buffer,
    vertex_normals: &wgpu::Buffer,
    d_vertices: &wgpu::Buffer,
    d_densities: &wgpu::Buffer,
    d_color_grads: &wgpu::Buffer,
    d_base_colors: &wgpu::Buffer,
    d_vertex_normals: &wgpu::Buffer,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_chain_back_bg_0"),
        layout: &pipelines.bg_layout_0,
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, vertices),
            buf_entry(2, indices),
            buf_entry(3, color_grads),
            buf_entry(4, compact_tet_ids),
            buf_entry(5, indirect_args),
            buf_entry(6, interval_meta),
            buf_entry(7, d_interval_verts),
            buf_entry(8, d_interval_tet_data),
            buf_entry(9, vertex_normals),
        ],
    });

    let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_chain_back_bg_1"),
        layout: &pipelines.bg_layout_1,
        entries: &[
            buf_entry(0, d_vertices),
            buf_entry(1, d_densities),
            buf_entry(2, d_color_grads),
            buf_entry(3, d_base_colors),
            buf_entry(4, d_vertex_normals),
        ],
    });

    (bg0, bg1)
}

/// Record the interval chain-back compute pass dispatch.
pub fn record_interval_chain_back(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &IntervalChainBackPipeline,
    bg0: &wgpu::BindGroup,
    bg1: &wgpu::BindGroup,
    tet_count: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("interval_chain_back"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, bg0, &[]);
    pass.set_bind_group(1, bg1, &[]);
    let workgroups = (tet_count + 63) / 64;
    let max_per_dim = 65535u32;
    let x = workgroups.min(max_per_dim);
    let y = (workgroups + max_per_dim - 1) / max_per_dim;
    pass.dispatch_workgroups(x, y, 1);
}

