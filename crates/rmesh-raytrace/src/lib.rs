//! GPU ray tracing and point-in-tet location for tetrahedral meshes.
//!
//! Software ray–tet intersection (`RayTracePipeline`) used by the viewer's
//! diagnostic ray-trace mode, plus adjacency-walk point location
//! (`LocatePipeline`, `find_containing_tet*`) and the boundary BVH the walk
//! and ray tracer rely on. Extracted from `rmesh-render`.

#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

use rmesh_data::{MaterialBuffers, SceneBuffers};
use rmesh_util::gpu_helpers::{buf_entry, storage_entries};
use rmesh_util::shared::BVHNode;

// WGSL shader sources, embedded from crate-local files.
static RAYTRACE_COMPUTE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/raytrace_compute.wgsl");
static LOCATE_COMPUTE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/locate_compute.wgsl");

// ---------------------------------------------------------------------------
// Ray Tracing: Tet Neighbors
// ---------------------------------------------------------------------------

/// Tet neighbor adjacency: `neighbors[tet_id * 4 + face_idx]` = neighbor tet or -1.
pub fn compute_tet_neighbors(indices: &[u32], tet_count: usize) -> Vec<i32> {
    use rmesh_util::shared::TET_FACE_INDICES;

    let mut neighbors = vec![-1i32; tet_count * 4];
    let mut face_map: HashMap<[u32; 3], (usize, usize)> = HashMap::with_capacity(tet_count * 4);

    for tet_id in 0..tet_count {
        for face_idx in 0..4usize {
            let fi_base = face_idx * 3;
            let vi0 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base] as usize];
            let vi1 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 1] as usize];
            let vi2 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 2] as usize];

            let mut key = [vi0, vi1, vi2];
            key.sort();

            if let Some(&(other_tet, other_face)) = face_map.get(&key) {
                neighbors[tet_id * 4 + face_idx] = other_tet as i32;
                neighbors[other_tet * 4 + other_face] = tet_id as i32;
                face_map.remove(&key);
            } else {
                face_map.insert(key, (tet_id, face_idx));
            }
        }
    }

    neighbors
}

// ---------------------------------------------------------------------------
// Ray Tracing: BVH Builder
// ---------------------------------------------------------------------------

/// BVH data: nodes + packed boundary face array.
pub struct BVHData {
    pub nodes: Vec<BVHNode>,
    pub boundary_faces: Vec<u32>,
}

/// 30-bit Morton code for 10-bit x,y,z.
fn morton_3d(x: u32, y: u32, z: u32) -> u32 {
    fn expand_bits(mut v: u32) -> u32 {
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        v
    }
    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Build a BVH over boundary faces (faces where `neighbors[...] == -1`).
pub fn build_boundary_bvh(
    vertices: &[f32],
    indices: &[u32],
    neighbors: &[i32],
    tet_count: usize,
) -> BVHData {
    use rmesh_util::shared::TET_FACE_INDICES;

    // Collect boundary faces
    let mut boundary_faces: Vec<u32> = Vec::new();
    let mut centroids: Vec<[f32; 3]> = Vec::new();
    let mut aabbs: Vec<([f32; 3], [f32; 3])> = Vec::new();

    for tet_id in 0..tet_count {
        for face_idx in 0..4usize {
            if neighbors[tet_id * 4 + face_idx] != -1 {
                continue;
            }
            let packed = ((tet_id as u32) << 2) | (face_idx as u32);
            boundary_faces.push(packed);

            let fi_base = face_idx * 3;
            let vi0 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base] as usize] as usize;
            let vi1 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 1] as usize] as usize;
            let vi2 = indices[tet_id * 4 + TET_FACE_INDICES[fi_base + 2] as usize] as usize;

            let v0 = [
                vertices[vi0 * 3],
                vertices[vi0 * 3 + 1],
                vertices[vi0 * 3 + 2],
            ];
            let v1 = [
                vertices[vi1 * 3],
                vertices[vi1 * 3 + 1],
                vertices[vi1 * 3 + 2],
            ];
            let v2 = [
                vertices[vi2 * 3],
                vertices[vi2 * 3 + 1],
                vertices[vi2 * 3 + 2],
            ];

            centroids.push([
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ]);
            aabbs.push((
                [
                    v0[0].min(v1[0]).min(v2[0]),
                    v0[1].min(v1[1]).min(v2[1]),
                    v0[2].min(v1[2]).min(v2[2]),
                ],
                [
                    v0[0].max(v1[0]).max(v2[0]),
                    v0[1].max(v1[1]).max(v2[1]),
                    v0[2].max(v1[2]).max(v2[2]),
                ],
            ));
        }
    }

    let n_faces = boundary_faces.len();
    if n_faces == 0 {
        return BVHData {
            nodes: vec![BVHNode {
                aabb_min: [0.0; 3],
                left_or_face: -1,
                aabb_max: [0.0; 3],
                right_or_count: 0,
            }],
            boundary_faces,
        };
    }

    // Sort faces by Morton code of centroid
    let mut scene_min = [f32::INFINITY; 3];
    let mut scene_max = [f32::NEG_INFINITY; 3];
    for c in &centroids {
        for i in 0..3 {
            scene_min[i] = scene_min[i].min(c[i]);
            scene_max[i] = scene_max[i].max(c[i]);
        }
    }
    let scene_extent = [
        (scene_max[0] - scene_min[0]).max(1e-10),
        (scene_max[1] - scene_min[1]).max(1e-10),
        (scene_max[2] - scene_min[2]).max(1e-10),
    ];

    let mut sorted_indices: Vec<usize> = (0..n_faces).collect();
    sorted_indices.sort_by_key(|&i| {
        let nx = ((centroids[i][0] - scene_min[0]) / scene_extent[0] * 1023.0) as u32;
        let ny = ((centroids[i][1] - scene_min[1]) / scene_extent[1] * 1023.0) as u32;
        let nz = ((centroids[i][2] - scene_min[2]) / scene_extent[2] * 1023.0) as u32;
        morton_3d(nx.min(1023), ny.min(1023), nz.min(1023))
    });

    // Reorder boundary_faces and aabbs by sorted order
    let orig_faces = boundary_faces.clone();
    let orig_aabbs = aabbs.clone();
    for (new_i, &old_i) in sorted_indices.iter().enumerate() {
        boundary_faces[new_i] = orig_faces[old_i];
        aabbs[new_i] = orig_aabbs[old_i];
    }

    // Build binary BVH
    struct BuildTask {
        start: usize,
        end: usize,
        node_idx: usize,
    }

    let mut nodes: Vec<BVHNode> = Vec::with_capacity(2 * n_faces);
    // Reserve root
    nodes.push(BVHNode {
        aabb_min: [0.0; 3],
        left_or_face: 0,
        aabb_max: [0.0; 3],
        right_or_count: 0,
    });
    let mut stack = vec![BuildTask {
        start: 0,
        end: n_faces,
        node_idx: 0,
    }];

    while let Some(task) = stack.pop() {
        let count = task.end - task.start;
        let mut amin = [f32::INFINITY; 3];
        let mut amax = [f32::NEG_INFINITY; 3];
        for aabb in aabbs.iter().take(task.end).skip(task.start) {
            for d in 0..3 {
                amin[d] = amin[d].min(aabb.0[d]);
                amax[d] = amax[d].max(aabb.1[d]);
            }
        }

        if count <= 4 {
            nodes[task.node_idx] = BVHNode {
                aabb_min: amin,
                left_or_face: -(task.start as i32 + 1),
                aabb_max: amax,
                right_or_count: count as i32,
            };
        } else {
            let mid = task.start + count / 2;
            let left_idx = nodes.len();
            nodes.push(BVHNode {
                aabb_min: [0.0; 3],
                left_or_face: 0,
                aabb_max: [0.0; 3],
                right_or_count: 0,
            });
            let right_idx = nodes.len();
            nodes.push(BVHNode {
                aabb_min: [0.0; 3],
                left_or_face: 0,
                aabb_max: [0.0; 3],
                right_or_count: 0,
            });

            nodes[task.node_idx] = BVHNode {
                aabb_min: amin,
                left_or_face: left_idx as i32,
                aabb_max: amax,
                right_or_count: right_idx as i32,
            };

            stack.push(BuildTask {
                start: task.start,
                end: mid,
                node_idx: left_idx,
            });
            stack.push(BuildTask {
                start: mid,
                end: task.end,
                node_idx: right_idx,
            });
        }
    }

    BVHData {
        nodes,
        boundary_faces,
    }
}

// ---------------------------------------------------------------------------
// Ray Tracing: Containment Test
// ---------------------------------------------------------------------------

/// Find the tet containing `point`, or None if outside the mesh.
/// Brute-force barycentric test — O(N_tets), single query per frame.
pub fn find_containing_tet(
    vertices: &[f32],
    indices: &[u32],
    tet_count: usize,
    point: Vec3,
) -> Option<u32> {
    for tet_id in 0..tet_count {
        let vi = [
            indices[tet_id * 4] as usize,
            indices[tet_id * 4 + 1] as usize,
            indices[tet_id * 4 + 2] as usize,
            indices[tet_id * 4 + 3] as usize,
        ];
        let v = [
            Vec3::new(
                vertices[vi[0] * 3],
                vertices[vi[0] * 3 + 1],
                vertices[vi[0] * 3 + 2],
            ),
            Vec3::new(
                vertices[vi[1] * 3],
                vertices[vi[1] * 3 + 1],
                vertices[vi[1] * 3 + 2],
            ),
            Vec3::new(
                vertices[vi[2] * 3],
                vertices[vi[2] * 3 + 1],
                vertices[vi[2] * 3 + 2],
            ),
            Vec3::new(
                vertices[vi[3] * 3],
                vertices[vi[3] * 3 + 1],
                vertices[vi[3] * 3 + 2],
            ),
        ];

        let d = v[1] - v[0];
        let e = v[2] - v[0];
        let f = v[3] - v[0];
        let p = point - v[0];

        let det = d.dot(e.cross(f));
        if det.abs() < 1e-20 {
            continue;
        }
        let inv_det = 1.0 / det;

        let u = p.dot(e.cross(f)) * inv_det;
        let v_coord = d.dot(p.cross(f)) * inv_det;
        let w = d.dot(e.cross(p)) * inv_det;

        let eps = -1e-6;
        if u >= eps && v_coord >= eps && w >= eps && (u + v_coord + w) <= 1.0 + 1e-6 {
            return Some(tet_id as u32);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Ray Tracing: Pipeline + Buffers
// ---------------------------------------------------------------------------

/// Compute-based ray tracing pipeline with adjacency traversal.
pub struct RayTracePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    aux_bind_group_layout: wgpu::BindGroupLayout,
    /// The output storage buffer: [W x H x 4] f32
    pub rendered_image: wgpu::Buffer,
    /// Auxiliary output buffer: [W x H x AUX_STRIDE] f32
    pub aux_image: wgpu::Buffer,
    /// Default aux bind group (group 1) with dummy aux_data
    pub aux_bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
    pub aux_dim: u32,
    pub aux_stride: u32,
}

impl RayTracePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, aux_dim: u32) -> Self {
        let aux_stride = 8 + aux_dim;

        // String-substitute AUX_DIM and AUX_ACC_SIZE in shader source
        let source = RAYTRACE_COMPUTE_WGSL
            .replace("/*AUX_DIM*/0u", &format!("{}u", aux_dim))
            .replace("/*AUX_ACC_SIZE*/1u", &format!("{}u", aux_dim.max(1)));

        let shader =
            rmesh_util::compose::create_shader_module(device, "raytrace_compute.wgsl", &source)
                .expect("Failed to compose raytrace_compute.wgsl");

        // Group 0: 13 bindings (0-6 read, 7 rw, 8-12 read)
        let read_only = [
            true, true, true, true, true, true, true, false, true, true, true, true, true,
        ];
        let entries = storage_entries(13, wgpu::ShaderStages::COMPUTE, &read_only);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raytrace_bgl"),
            entries: &entries,
        });

        // Group 1: aux_image (rw), aux_data (read)
        let aux_entries = storage_entries(2, wgpu::ShaderStages::COMPUTE, &[false, true]);
        let aux_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raytrace_aux_bgl"),
                entries: &aux_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raytrace_pl"),
            bind_group_layouts: &[&bind_group_layout, &aux_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raytrace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rendered_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raytrace_rendered_image"),
            size: (width as u64) * (height as u64) * 4 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let aux_image = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raytrace_aux_image"),
            size: (width as u64) * (height as u64) * (aux_stride as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dummy aux_data buffer (4 bytes minimum)
        let aux_data_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raytrace_aux_data_dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let aux_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raytrace_aux_bg_default"),
            layout: &aux_bind_group_layout,
            entries: &[buf_entry(0, &aux_image), buf_entry(1, &aux_data_dummy)],
        });

        Self {
            pipeline,
            bind_group_layout,
            aux_bind_group_layout,
            rendered_image,
            aux_image,
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

/// GPU buffers for ray trace adjacency data.
pub struct RayTraceBuffers {
    pub tet_neighbors: wgpu::Buffer,
    pub bvh_nodes: wgpu::Buffer,
    pub boundary_faces: wgpu::Buffer,
    pub start_tet: wgpu::Buffer,
    /// Per-pixel ray origins [W*H*3] f32 (used when ray_mode=1)
    pub ray_origins: wgpu::Buffer,
    /// Per-pixel ray directions [W*H*3] f32 (used when ray_mode=1)
    pub ray_dirs: wgpu::Buffer,
}

impl RayTraceBuffers {
    pub fn new(device: &wgpu::Device, neighbors: &[i32], bvh: &BVHData) -> Self {
        let tet_neighbors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet_neighbors"),
            contents: bytemuck::cast_slice(neighbors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bvh_data = if bvh.nodes.is_empty() {
            vec![BVHNode {
                aabb_min: [0.0; 3],
                left_or_face: -1,
                aabb_max: [0.0; 3],
                right_or_count: 0,
            }]
        } else {
            bvh.nodes.clone()
        };
        let bvh_nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_nodes"),
            contents: bytemuck::cast_slice(&bvh_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bf_data = if bvh.boundary_faces.is_empty() {
            vec![0u32]
        } else {
            bvh.boundary_faces.clone()
        };
        let boundary_faces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boundary_faces"),
            contents: bytemuck::cast_slice(&bf_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let start_tet = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("start_tet"),
            contents: bytemuck::cast_slice(&[-1i32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Placeholder buffers for ray_origins and ray_dirs (4 bytes each, minimum valid)
        let ray_origins = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ray_origins"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let ray_dirs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ray_dirs"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            tet_neighbors,
            bvh_nodes,
            boundary_faces,
            start_tet,
            ray_origins,
            ray_dirs,
        }
    }
}

/// Create the ray trace bind group.
pub fn create_raytrace_bind_group(
    device: &wgpu::Device,
    rt_pipeline: &RayTracePipeline,
    scene_buffers: &SceneBuffers,
    material: &MaterialBuffers,
    rt_buffers: &RayTraceBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raytrace_bg"),
        layout: rt_pipeline.bind_group_layout(),
        entries: &[
            buf_entry(0, &scene_buffers.uniforms),
            buf_entry(1, &scene_buffers.vertices),
            buf_entry(2, &scene_buffers.indices),
            buf_entry(3, &material.colors),
            buf_entry(4, &scene_buffers.densities),
            buf_entry(5, &material.color_grads),
            buf_entry(6, &rt_buffers.tet_neighbors),
            buf_entry(7, &rt_pipeline.rendered_image),
            buf_entry(8, &rt_buffers.bvh_nodes),
            buf_entry(9, &rt_buffers.boundary_faces),
            buf_entry(10, &rt_buffers.start_tet),
            buf_entry(11, &rt_buffers.ray_origins),
            buf_entry(12, &rt_buffers.ray_dirs),
        ],
    })
}

/// Record the ray trace compute pass dispatch.
pub fn record_raytrace(
    encoder: &mut wgpu::CommandEncoder,
    rt_pipeline: &RayTracePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("raytrace"),
        timestamp_writes: None,
    });
    pass.set_pipeline(rt_pipeline.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_bind_group(1, &rt_pipeline.aux_bind_group, &[]);
    pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
}

// Forward tiled pipeline (`RasterizeComputePipeline`) moved to `rmesh-trainable`.

// ===========================================================================
// Point Location Pipeline (adjacency walking)
// ===========================================================================

/// Uniforms for the locate compute shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocateUniforms {
    pub num_queries: u32,
    pub hint_tet: i32, // global hint, or -1 to use per-query hint_tets
    pub tet_count: u32,
    pub _pad: u32,
}

/// GPU pipeline for point-in-tet location via adjacency walking.
pub struct LocatePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl LocatePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("locate_compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(LOCATE_COMPUTE_WGSL.as_str().into()),
        });

        // 7 bindings: 0-5 read, 6 read_write
        let read_only = [true, true, true, true, true, true, false];
        let entries = storage_entries(7, wgpu::ShaderStages::COMPUTE, &read_only);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("locate_bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("locate_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("locate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

/// Create the locate bind group.
///
/// Binding order matches `locate_compute.wgsl`:
///   0: locate_uniforms, 1: vertices, 2: indices, 3: tet_neighbors,
///   4: query_points, 5: hint_tets, 6: result_tets
pub fn create_locate_bind_group(
    device: &wgpu::Device,
    pipeline: &LocatePipeline,
    locate_uniforms_buf: &wgpu::Buffer,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    tet_neighbors: &wgpu::Buffer,
    query_points: &wgpu::Buffer,
    hint_tets: &wgpu::Buffer,
    result_tets: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("locate_bg"),
        layout: pipeline.bind_group_layout(),
        entries: &[
            buf_entry(0, locate_uniforms_buf),
            buf_entry(1, vertices),
            buf_entry(2, indices),
            buf_entry(3, tet_neighbors),
            buf_entry(4, query_points),
            buf_entry(5, hint_tets),
            buf_entry(6, result_tets),
        ],
    })
}

/// Record the locate compute pass dispatch.
pub fn record_locate(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &LocatePipeline,
    bind_group: &wgpu::BindGroup,
    num_queries: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("locate"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline.pipeline());
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(num_queries.div_ceil(64), 1, 1);
}

// ---------------------------------------------------------------------------
// Point Location: CPU Walking
// ---------------------------------------------------------------------------

/// Find the tet containing `point` by walking from `hint_tet`.
///
/// Same algorithm as the GPU shader but on CPU. Falls back to brute-force
/// `find_containing_tet()` if the walk exceeds 512 steps.
///
/// Vertex-to-face mapping: vertex k -> face opposite vertex k.
///   vertex 0 -> face 1, vertex 1 -> face 2,
///   vertex 2 -> face 3, vertex 3 -> face 0.
pub fn find_containing_tet_walk(
    vertices: &[f32],
    indices: &[u32],
    neighbors: &[i32],
    tet_count: usize,
    point: Vec3,
    hint_tet: usize,
) -> Option<u32> {
    const VERTEX_TO_FACE: [usize; 4] = [1, 2, 3, 0];
    const MAX_ITERS: usize = 512;
    const EPS: f32 = -1e-6;

    let mut current = if hint_tet < tet_count { hint_tet } else { 0 };

    for _ in 0..MAX_ITERS {
        let vi = [
            indices[current * 4] as usize,
            indices[current * 4 + 1] as usize,
            indices[current * 4 + 2] as usize,
            indices[current * 4 + 3] as usize,
        ];
        let v = [
            Vec3::new(
                vertices[vi[0] * 3],
                vertices[vi[0] * 3 + 1],
                vertices[vi[0] * 3 + 2],
            ),
            Vec3::new(
                vertices[vi[1] * 3],
                vertices[vi[1] * 3 + 1],
                vertices[vi[1] * 3 + 2],
            ),
            Vec3::new(
                vertices[vi[2] * 3],
                vertices[vi[2] * 3 + 1],
                vertices[vi[2] * 3 + 2],
            ),
            Vec3::new(
                vertices[vi[3] * 3],
                vertices[vi[3] * 3 + 1],
                vertices[vi[3] * 3 + 2],
            ),
        ];

        let d = v[1] - v[0];
        let e = v[2] - v[0];
        let f = v[3] - v[0];
        let p = point - v[0];

        let det = d.dot(e.cross(f));
        if det.abs() < 1e-20 {
            // Degenerate tet — fall back to brute force
            return find_containing_tet(vertices, indices, tet_count, point);
        }
        let inv_det = 1.0 / det;

        let u = p.dot(e.cross(f)) * inv_det; // bary for v1
        let vc = d.dot(p.cross(f)) * inv_det; // bary for v2
        let w = d.dot(e.cross(p)) * inv_det; // bary for v3
        let s = 1.0 - u - vc - w; // bary for v0

        // Check containment
        if s >= EPS && u >= EPS && vc >= EPS && w >= EPS {
            return Some(current as u32);
        }

        // Find most negative barycentric
        let barys = [s, u, vc, w];
        let mut min_idx = 0;
        for k in 1..4 {
            if barys[k] < barys[min_idx] {
                min_idx = k;
            }
        }

        let face_idx = VERTEX_TO_FACE[min_idx];
        let neighbor = neighbors[current * 4 + face_idx];

        if neighbor < 0 {
            return None; // Outside mesh
        }

        current = neighbor as usize;
    }

    // Walk exhausted — fall back to brute force
    find_containing_tet(vertices, indices, tet_count, point)
}
