//! Shared test utilities for GPU kernel tests.
//!
//! Gated behind the `test-util` feature. Provides:
//!   - GPU device creation
//!   - Buffer readback helpers
//!   - Random scene generators
//!   - Circumsphere computation

use glam::Vec3;
use rand::Rng;
use rmesh_data::SceneData;

// ---------------------------------------------------------------------------
// GPU device helpers
// ---------------------------------------------------------------------------

/// Configuration for test device creation.
pub struct TestDeviceConfig {
    /// Which backends to use. `None` means wgpu default.
    pub backends: Option<wgpu::Backends>,
    /// Extra required features (SUBGROUP is always requested).
    pub extra_features: wgpu::Features,
    /// Base limits to start from (`Limits::default()` or `Limits::downlevel_defaults()`).
    pub base_limits: wgpu::Limits,
}

impl Default for TestDeviceConfig {
    fn default() -> Self {
        Self {
            backends: Some(wgpu::Backends::VULKAN | wgpu::Backends::METAL),
            extra_features: wgpu::Features::empty(),
            base_limits: wgpu::Limits::default(),
        }
    }
}

/// Create a GPU device with SUBGROUP feature. Returns None if no adapter is found.
pub fn create_test_device(config: TestDeviceConfig) -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance_desc = match config.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..Default::default()
            },
            None => wgpu::InstanceDescriptor::default(),
        };
        let instance = wgpu::Instance::new(&instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP | config.extra_features,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..config.base_limits
                },
                ..Default::default()
            })
            .await
            .ok()
    })
}

/// Create a test device with default config (Vulkan|Metal, default limits).
pub fn create_test_device_default() -> Option<(wgpu::Device, wgpu::Queue)> {
    create_test_device(TestDeviceConfig::default())
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Read back a GPU buffer as a `Vec<T>`. Source buffer must have `COPY_SRC`.
pub fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let size = (count * std::mem::size_of::<T>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback.unmap();
    result
}

/// Create a read-write storage buffer with COPY_DST and COPY_SRC.
pub fn create_rw_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------

/// Compute circumspheres from flat vertex/index arrays.
pub fn compute_circumspheres(vertices: &[f32], indices: &[u32]) -> Vec<f32> {
    let tet_count = indices.len() / 4;
    let mut circumdata = vec![0.0f32; tet_count * 4];
    for i in 0..tet_count {
        let i0 = indices[i * 4] as usize;
        let i1 = indices[i * 4 + 1] as usize;
        let i2 = indices[i * 4 + 2] as usize;
        let i3 = indices[i * 4 + 3] as usize;
        let v0 = Vec3::new(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        let v1 = Vec3::new(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        let v2 = Vec3::new(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
        let v3 = Vec3::new(vertices[i3 * 3], vertices[i3 * 3 + 1], vertices[i3 * 3 + 2]);
        let a = v1 - v0;
        let b = v2 - v0;
        let c = v3 - v0;
        let (aa, bb, cc) = (a.dot(a), b.dot(b), c.dot(c));
        let cross_bc = b.cross(c);
        let cross_ca = c.cross(a);
        let cross_ab = a.cross(b);
        let mut denom = 2.0 * a.dot(cross_bc);
        if denom.abs() < 1e-12 {
            denom = 1.0;
        }
        let r = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denom;
        let center = v0 + r;
        circumdata[i * 4] = center.x;
        circumdata[i * 4 + 1] = center.y;
        circumdata[i * 4 + 2] = center.z;
        circumdata[i * 4 + 3] = r.dot(r);
    }
    circumdata
}

/// Generate a random tetrahedron centered roughly at origin.
/// Ensures positive orientation (det > 0) so face winding produces inward normals.
pub fn random_tet_vertices<R: Rng>(rng: &mut R, radius: f32) -> ([f32; 12], [u32; 4]) {
    let mut verts = [0.0f32; 12];
    for i in 0..4 {
        verts[i * 3] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 1] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
        verts[i * 3 + 2] = (rng.random::<f32>() - 0.5) * 2.0 * radius;
    }
    // Ensure positive orientation: det(v1-v0, v2-v0, v3-v0) > 0
    let v0 = Vec3::new(verts[0], verts[1], verts[2]);
    let v1 = Vec3::new(verts[3], verts[4], verts[5]);
    let v2 = Vec3::new(verts[6], verts[7], verts[8]);
    let v3 = Vec3::new(verts[9], verts[10], verts[11]);
    let det = (v1 - v0).dot((v2 - v0).cross(v3 - v0));
    if det < 0.0 {
        // Swap vertices 2 and 3 to flip orientation
        return (verts, [0, 1, 3, 2]);
    }
    (verts, [0, 1, 2, 3])
}

/// Build a `SceneData` from raw vertices, indices, and per-tet parameters.
pub fn build_test_scene(
    vertices: Vec<f32>,
    indices: Vec<u32>,
    densities: Vec<f32>,
    color_grads: Vec<f32>,
) -> SceneData {
    let vertex_count = vertices.len() as u32 / 3;
    let tet_count = indices.len() as u32 / 4;
    let circumdata = compute_circumspheres(&vertices, &indices);

    SceneData {
        vertices,
        indices,
        densities,
        color_grads,
        circumdata,
        start_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vertex_count,
        tet_count,
    }
}

/// Build a single-tet test scene with random parameters.
pub fn random_single_tet_scene<R: Rng>(rng: &mut R, radius: f32) -> SceneData {
    let (verts, indices) = random_tet_vertices(rng, radius);
    let density = vec![rng.random::<f32>() * 5.0 + 0.5];
    let color_grads = vec![
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
        (rng.random::<f32>() - 0.5) * 0.2,
    ];

    build_test_scene(verts.to_vec(), indices.to_vec(), density, color_grads)
}
