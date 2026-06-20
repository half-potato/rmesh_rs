//! GPU buffer containers for scene geometry and per-tet material data.
//!
//! These hold the `wgpu::Buffer`s that the rendering crates bind. They live in
//! `rmesh-data` (alongside [`SceneData`](crate::SceneData), which `SceneBuffers::upload`
//! consumes) so that the renderer, raytracer, and trainer can all share them
//! without depending on each other.

use crate::SceneData;
use wgpu::util::DeviceExt;

/// Byte size of the per-frame uniforms buffer.
///
/// Must equal `std::mem::size_of::<rmesh_util::shared::Uniforms>()`. `rmesh-data`
/// cannot depend on `rmesh-util` (that would form a dependency cycle through the
/// `test-util` feature), so the value is duplicated here and asserted to match in
/// `rmesh-render` (see `UNIFORMS_SIZE_CHECK`).
pub const UNIFORMS_BUFFER_SIZE: u64 = 192;

/// GPU buffers for scene geometry (independent of material/appearance model).
pub struct SceneBuffers {
    /// Vertex positions [N x 3] f32
    pub vertices: wgpu::Buffer,
    /// Tet vertex indices [M x 4] u32
    pub indices: wgpu::Buffer,
    /// Per-tet density [M] f32
    pub densities: wgpu::Buffer,
    /// Circumsphere data [M x 4] f32 (cx, cy, cz, r^2)
    pub circumdata: wgpu::Buffer,
    /// Sort keys [M] u32 (written by compute, sorted in place)
    pub sort_keys: wgpu::Buffer,
    /// Sort values [M] u32 (written by compute, sorted in place)
    pub sort_values: wgpu::Buffer,
    /// Indirect draw arguments (DrawIndirectCommand, 16 bytes)
    pub indirect_args: wgpu::Buffer,
    /// Per-frame uniforms (Uniforms struct)
    pub uniforms: wgpu::Buffer,
    /// Tiles touched per visible tet [M] u32 (written by compute at vis_idx)
    pub tiles_touched: wgpu::Buffer,
    /// Compact visible tet IDs [M] u32 (written by compute at vis_idx)
    pub compact_tet_ids: wgpu::Buffer,
    /// Mesh shader indirect dispatch args [3] u32 (x, y, z workgroup counts)
    pub mesh_indirect_args: wgpu::Buffer,
    /// Precomputed per-tet data for quad renderer [M × 8] vec4<f32> (128 bytes/tet)
    pub precomputed: wgpu::Buffer,
    /// Compute-interval vertex buffer [M × 5 × 3] vec4<f32> (240 bytes/tet)
    pub interval_vertex_buf: wgpu::Buffer,
    /// Compute-interval per-tet flat data [M] vec4<f32> (16 bytes/tet)
    pub interval_tet_data_buf: wgpu::Buffer,
    /// Static fan index buffer [12] u32 — shared across all tets via instanced draw
    pub interval_fan_index_buf: wgpu::Buffer,
    /// Combined compute dispatch + draw-indexed-indirect args [8] u32 (32 bytes)
    pub interval_args_buf: wgpu::Buffer,
    /// Per-vertex normals [N × 3] f32 (optional, for interval tiled xyzd output)
    pub vertex_normals: wgpu::Buffer,
}

/// GPU buffers for per-tet material/appearance data (pluggable per rendering mode).
pub struct MaterialBuffers {
    /// Per-tet base color [M x 3] f32 (uploaded from model, input to softplus)
    pub base_colors: wgpu::Buffer,
    /// Per-tet color gradient [M x 3] f32
    pub color_grads: wgpu::Buffer,
    /// Evaluated per-tet color [M x 3] f32 (written by project_compute)
    pub colors: wgpu::Buffer,
}

impl SceneBuffers {
    /// Upload scene geometry to GPU buffers.
    pub fn upload(device: &wgpu::Device, _queue: &wgpu::Queue, scene: &SceneData) -> Self {
        let storage_copy = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let trainable = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(&scene.vertices),
            usage: trainable,
        });

        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&scene.indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let densities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("densities"),
            contents: bytemuck::cast_slice(&scene.densities),
            usage: trainable,
        });

        let circumdata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("circumdata"),
            contents: bytemuck::cast_slice(&scene.circumdata),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let m = scene.tet_count as u64;
        let n_pow2 = (scene.tet_count as u64).next_power_of_two();

        let sort_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_keys"),
            size: n_pow2 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_values"),
            size: n_pow2 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect_args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: UNIFORMS_BUFFER_SIZE,
            // UNIFORM in addition to STORAGE so the same buffer can be bound
            // as `var<uniform>` in shaders that need to reduce their storage-
            // buffer count under WebGPU's 10-per-stage cap.
            usage: storage_copy | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let tiles_touched = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiles_touched"),
            size: m.div_ceil(4) * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let compact_tet_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact_tet_ids"),
            size: m * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mesh_indirect_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_indirect_args"),
            contents: bytemuck::cast_slice(&[0u32, 1u32, 1u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        });

        // Precomputed buffer: 10 × vec4<f32> = 160 bytes per tet
        let precomputed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("precomputed"),
            size: m * 10 * 16, // 160 bytes per tet
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Compute-interval vertex buffer: 5 verts × 3 vec4s × 16 bytes = 240 bytes/tet
        let interval_vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_vertex_buf"),
            size: m * 5 * 4 * 16, // 5 verts × 4 vec4 × 16 bytes (pos+z, offsets, n_front, n_back)
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Compute-interval per-tet flat data: 2 vec4 × 16 bytes = 32 bytes/tet
        // Slot 0: (base_color.rgb, density), Slot 1: (tet_id, 0, 0, 0)
        let interval_tet_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_tet_data_buf"),
            size: m * 32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Static fan index buffer: M × 12 u32s, created once (never written per frame).
        // Fan pattern per tet i: (i*5+0,i*5+1,i*5+4), (i*5+1,i*5+2,i*5+4),
        //                        (i*5+2,i*5+3,i*5+4), (i*5+3,i*5+0,i*5+4)
        let fan_indices: Vec<u32> = (0..m as u32)
            .flat_map(|i| {
                let b = i * 5;
                [
                    b,
                    b + 1,
                    b + 4,
                    b + 1,
                    b + 2,
                    b + 4,
                    b + 2,
                    b + 3,
                    b + 4,
                    b + 3,
                    b,
                    b + 4,
                ]
            })
            .collect();
        let interval_fan_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("interval_fan_index_buf"),
            contents: bytemuck::cast_slice(&fan_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Combined dispatch + draw-indexed-indirect args: 8 × u32 = 32 bytes
        let interval_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interval_args_buf"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-vertex normals: N × 3 f32, initialized to zeros
        let n = scene.vertex_count as u64;
        let vertex_normals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_normals"),
            size: n * 3 * 4,
            usage: trainable,
            mapped_at_creation: false,
        });

        Self {
            vertices,
            indices,
            densities,
            circumdata,
            sort_keys,
            sort_values,
            indirect_args,
            uniforms,
            tiles_touched,
            compact_tet_ids,
            mesh_indirect_args,
            precomputed,
            interval_vertex_buf,
            interval_tet_data_buf,
            interval_fan_index_buf,
            interval_args_buf,
            vertex_normals,
        }
    }
}

impl MaterialBuffers {
    /// Upload material data to GPU buffers.
    ///
    /// * `base_colors` — per-tet base color `[M × 3]` f32 (pre-softplus)
    /// * `color_grads` — per-tet color gradient `[M × 3]` f32
    /// * `tet_count` — number of tetrahedra
    pub fn upload(
        device: &wgpu::Device,
        base_colors: &[f32],
        color_grads: &[f32],
        tet_count: u32,
    ) -> Self {
        let trainable = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let base_colors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("base_colors"),
            contents: bytemuck::cast_slice(base_colors),
            usage: trainable,
        });

        let color_grads_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color_grads"),
            contents: bytemuck::cast_slice(color_grads),
            usage: trainable,
        });

        let colors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colors"),
            size: (tet_count as u64) * 3 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            base_colors: base_colors_buf,
            color_grads: color_grads_buf,
            colors,
        }
    }
}
