//! Compute-based forward tiled interval rasterizer.
//!
//! Two-stage pipeline:
//!   1. `IntervalGeneratePipeline` — per-tet screen-triangle decomposition
//!      (writes into `IntervalTiledBuffers`).
//!   2. `IntervalTiledRasterizePipeline` — per-tile compute rasterization.
//!
//! Forward half of the *interval-based* trainable pipeline. Paired with the
//! backward kernel in `backward_interval_tiled.wgsl`.

use crate::buf_entry;
use rmesh_render::{MaterialBuffers, SceneBuffers};
use rmesh_tile::dispatch_2d;

const INTERVAL_GENERATE_WGSL: &str = include_str!("wgsl/interval_generate.wgsl");
const INTERVAL_TILED_RASTERIZE_WGSL: &str = include_str!("wgsl/interval_tiled_rasterize.wgsl");

// ---------------------------------------------------------------------------
// Shared buffers
// ---------------------------------------------------------------------------

/// Buffers for the tiled interval pipeline (shared between forward and backward).
pub struct IntervalTiledBuffers {
    /// Screen triangle vertices: [max_visible × 5 × 4] vec4<f32>
    pub interval_verts: wgpu::Buffer,
    /// Per-tet base_color + density: [max_visible] vec4<f32>
    pub interval_tet_data: wgpu::Buffer,
    /// Per-tet metadata (num_silhouette | tet_id << 4): [max_visible] u32
    pub interval_meta: wgpu::Buffer,
}

impl IntervalTiledBuffers {
    pub fn new(device: &wgpu::Device, max_visible: u32) -> Self {
        let m = max_visible as u64;
        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let interval_verts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_verts"),
            size: m * 5 * 4 * 16,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let interval_tet_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_tet_data"),
            size: m * 16,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let interval_meta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_meta"),
            size: m * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        Self {
            interval_verts,
            interval_tet_data,
            interval_meta,
        }
    }
}

// ---------------------------------------------------------------------------
// IntervalGeneratePipeline
// ---------------------------------------------------------------------------

/// Pipeline for the interval generate pass (per-tet screen triangle decomposition).
pub struct IntervalGeneratePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout: wgpu::BindGroupLayout,
}

impl IntervalGeneratePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let read_only = [
            true, true, true, true, true, true, true, true, // 0-7
            false, false, false, // 8-10
            true,  // 11 (vertex_normals)
        ];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interval_generate_bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interval_generate_pl"),
            bind_group_layouts: &[&bg_layout],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("interval_generate.wgsl"),
            source: wgpu::ShaderSource::Wgsl(INTERVAL_GENERATE_WGSL.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("interval_generate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bg_layout,
        }
    }
}

/// Create the interval generate bind group (12 bindings).
///
/// Binding order matches `interval_generate.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors, 4: densities,
///   5: color_grads, 6: compact_tet_ids, 7: indirect_args,
///   8: interval_verts, 9: interval_tet_data, 10: interval_meta,
///   11: vertex_normals
pub fn create_interval_generate_bind_group(
    device: &wgpu::Device,
    pipeline: &IntervalGeneratePipeline,
    scene: &SceneBuffers,
    material: &MaterialBuffers,
    interval: &IntervalTiledBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_generate_bg"),
        layout: &pipeline.bg_layout,
        entries: &[
            buf_entry(0, &scene.uniforms),
            buf_entry(1, &scene.vertices),
            buf_entry(2, &scene.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &scene.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &scene.compact_tet_ids),
            buf_entry(7, &scene.indirect_args),
            buf_entry(8, &interval.interval_verts),
            buf_entry(9, &interval.interval_tet_data),
            buf_entry(10, &interval.interval_meta),
            buf_entry(11, &scene.vertex_normals),
        ],
    })
}

/// Record the interval generate compute pass.
///
/// Dispatches ceil(visible_count / 64) workgroups.
pub fn record_interval_generate(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &IntervalGeneratePipeline,
    bind_group: &wgpu::BindGroup,
    tet_count: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("interval_generate"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    let workgroups = tet_count.div_ceil(64);
    let (x, y) = dispatch_2d(workgroups);
    pass.dispatch_workgroups(x, y, 1);
}

// ---------------------------------------------------------------------------
// IntervalTiledRasterizePipeline
// ---------------------------------------------------------------------------

/// Pipeline for the interval tiled rasterize pass (forward, per-tile).
pub struct IntervalTiledRasterizePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W × H × 4] f32
    pub rendered_image: wgpu::Buffer,
    /// Normal+depth output: [W × H × 4] f32 (normal.xyz + depth)
    pub xyzd_image: wgpu::Buffer,
    /// Distortion state output: [W × H × 5] f32
    pub distortion_image: wgpu::Buffer,
    /// Custom aux output: [W × H × aux_dim] f32
    pub aux_image: wgpu::Buffer,
    /// Number of custom aux channels per tet (0 = no aux)
    pub aux_dim: usize,
    /// Dummy buffer for aux_data binding when no scene data is provided
    pub aux_data_dummy: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
}

impl IntervalTiledRasterizePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, aux_dim: usize) -> Self {
        let sm_aux_size = if aux_dim > 0 { 256 * aux_dim } else { 1 };
        let source = INTERVAL_TILED_RASTERIZE_WGSL
            .replace("/*AUX_DIM*/0u", &format!("{}u", aux_dim))
            .replace("/*SM_AUX_SIZE*/1u", &format!("{}u", sm_aux_size));
        let shader = rmesh_util::compose::create_shader_module(
            device,
            "interval_tiled_rasterize.wgsl",
            &source,
        )
        .expect("Failed to compose interval_tiled_rasterize.wgsl");

        let read_only = [
            true, true, true, true, true, true, true, false, false, false, true, false,
        ];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("interval_tiled_rasterize_bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("interval_tiled_rasterize_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("interval_tiled_rasterize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let n_pixels = (width as u64) * (height as u64);
        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_rendered_image"),
            size: n_pixels * 4 * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let xyzd_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_xyzd_image"),
            size: n_pixels * 4 * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let distortion_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_distortion_image"),
            size: n_pixels * 5 * 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let aux_image_size = if aux_dim > 0 {
            n_pixels * (aux_dim as u64) * 4
        } else {
            4
        };
        let aux_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_aux_image"),
            size: aux_image_size,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        let aux_data_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tiled_aux_data_dummy"),
            size: 4,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            rendered_image,
            xyzd_image,
            distortion_image,
            aux_image,
            aux_dim,
            aux_data_dummy,
            width,
            height,
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// Create the interval tiled rasterize bind group (12 bindings).
///
/// Binding order matches `interval_tiled_rasterize.wgsl`:
///   0: uniforms, 1: interval_verts, 2: interval_tet_data, 3: interval_meta,
///   4: tile_sort_values, 5: tile_ranges, 6: tile_uniforms, 7: rendered_image,
///   8: xyzd_image, 9: distortion_image, 10: aux_data, 11: aux_image
pub fn create_interval_tiled_rasterize_bind_group(
    device: &wgpu::Device,
    rasterize: &IntervalTiledRasterizePipeline,
    uniforms: &wgpu::Buffer,
    interval: &IntervalTiledBuffers,
    tile_sort_values: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
    aux_data: &wgpu::Buffer,
    aux_image: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("interval_tiled_rasterize_bg"),
        layout: rasterize.bind_group_layout(),
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, &interval.interval_verts),
            buf_entry(2, &interval.interval_tet_data),
            buf_entry(3, &interval.interval_meta),
            buf_entry(4, tile_sort_values),
            buf_entry(5, tile_ranges),
            buf_entry(6, tile_uniforms),
            buf_entry(7, &rasterize.rendered_image),
            buf_entry(8, &rasterize.xyzd_image),
            buf_entry(9, &rasterize.distortion_image),
            buf_entry(10, aux_data),
            buf_entry(11, aux_image),
        ],
    })
}

/// Record the interval tiled rasterize compute pass dispatch.
pub fn record_interval_tiled_rasterize(
    encoder: &mut wgpu::CommandEncoder,
    rasterize: &IntervalTiledRasterizePipeline,
    bind_group: &wgpu::BindGroup,
    num_tiles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("interval_tiled_rasterize"),
        timestamp_writes: None,
    });
    pass.set_pipeline(rasterize.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    let (x, y) = dispatch_2d(num_tiles);
    pass.dispatch_workgroups(x, y, 1);
}
