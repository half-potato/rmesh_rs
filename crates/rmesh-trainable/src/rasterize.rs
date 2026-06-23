//! Compute-based forward tiled rasterizer.
//!
//! Software rasterizer matched to the backward kernel. The forward half of the
//! trainable pipeline.
//!
//! Requires `wgpu::Features::SUBGROUPS`. Renders directly to an f32 storage
//! buffer (no texture intermediate).

use crate::buf_entry;
use rmesh_tile::dispatch_2d;

static RASTERIZE_COMPUTE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/rasterize_compute.wgsl");

/// Compute-based forward renderer using tiles with warp-per-tile.
pub struct RasterizeComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    aux_bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    /// Auxiliary output buffer: [W x H x AUX_STRIDE] f32
    pub aux_image: wgpu::Buffer,
    /// Debug stats output buffer: [W x H x 4] u32 (ray_miss, ghost, occluded, useful)
    pub debug_image: wgpu::Buffer,
    /// Default aux bind group (group 1) with dummy aux_data + debug_image
    pub aux_bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
    pub aux_dim: u32,
    pub aux_stride: u32,
}

impl RasterizeComputePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, aux_dim: u32) -> Self {
        let aux_stride = 8 + aux_dim;

        let source = RASTERIZE_COMPUTE_WGSL
            .replace("/*AUX_DIM*/0u", &format!("{}u", aux_dim))
            .replace("/*SM_AUX_SIZE*/1u", &format!("{}u", (256 * aux_dim).max(1)));

        let shader =
            rmesh_util::compose::create_shader_module(device, "rasterize_compute.wgsl", &source)
                .expect("Failed to compose rasterize_compute.wgsl");

        let read_only = [true, true, true, true, true, true, true, true, true, false];
        let entries: Vec<wgpu::BindGroupLayoutEntry> = read_only
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rasterize_compute_bgl"),
            entries: &entries,
        });

        let aux_entries: Vec<wgpu::BindGroupLayoutEntry> = [false, true, false]
            .iter()
            .enumerate()
            .map(|(i, &ro)| rmesh_tile::storage_entry(i as u32, ro))
            .collect();
        let aux_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rasterize_aux_bgl"),
                entries: &aux_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rasterize_compute_pl"),
            bind_group_layouts: &[&bind_group_layout, &aux_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rasterize_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let aux_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_aux_image"),
            size: (width as u64) * (height as u64) * (aux_stride as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let aux_data_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_aux_data_dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let debug_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rasterize_debug_image"),
            size: (width as u64) * (height as u64) * 4 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let aux_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rasterize_aux_bg_default"),
            layout: &aux_bind_group_layout,
            entries: &[
                buf_entry(0, &aux_image),
                buf_entry(1, &aux_data_dummy),
                buf_entry(2, &debug_image),
            ],
        });

        Self {
            pipeline,
            bind_group_layout,
            aux_bind_group_layout,
            rendered_image,
            aux_image,
            debug_image,
            aux_bind_group,
            width,
            height,
            aux_dim,
            aux_stride,
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// Create the forward tiled bind group.
///
/// Binding order matches `rasterize_compute.wgsl`:
///   0: uniforms, 1: vertices, 2: indices, 3: colors,
///   4: densities, 5: color_grads, 6: tile_sort_values,
///   7: tile_ranges, 8: tile_uniforms, 9: rendered_image
pub fn create_rasterize_bind_group(
    device: &wgpu::Device,
    rasterize: &RasterizeComputePipeline,
    uniforms: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    densities: &wgpu::Buffer,
    color_grads: &wgpu::Buffer,
    tile_sort_values: &wgpu::Buffer,
    tile_ranges: &wgpu::Buffer,
    tile_uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rasterize_compute_bg"),
        layout: rasterize.bind_group_layout(),
        entries: &[
            buf_entry(0, uniforms),
            buf_entry(1, vertices),
            buf_entry(2, indices),
            buf_entry(3, colors),
            buf_entry(4, densities),
            buf_entry(5, color_grads),
            buf_entry(6, tile_sort_values),
            buf_entry(7, tile_ranges),
            buf_entry(8, tile_uniforms),
            buf_entry(9, &rasterize.rendered_image),
        ],
    })
}

/// Record the forward tiled compute pass dispatch.
///
/// Dispatches one workgroup per tile (32 threads each).
pub fn record_rasterize_compute(
    encoder: &mut wgpu::CommandEncoder,
    rasterize: &RasterizeComputePipeline,
    bind_group: &wgpu::BindGroup,
    num_tiles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rasterize_compute"),
        timestamp_writes: None,
    });
    pass.set_pipeline(rasterize.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_bind_group(1, &rasterize.aux_bind_group, &[]);
    let (x, y) = dispatch_2d(num_tiles);
    pass.dispatch_workgroups(x, y, 1);
}
