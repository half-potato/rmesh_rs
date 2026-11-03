//! Shared vertex picking and CPU preparation for GPU PBD grabs.
use glam::Mat4;
use rmesh_data::SceneData;
use rmesh_pbd::{build_island, color_constraints, ConstraintColoring, Island, MeshTopology};

/// Rich result from [`pick_vertex`]. The hover preview uses
/// `screen_pos` to draw a marker; the click handler uses the other fields
/// for the diagnostic log.
#[derive(Clone, Copy, Debug)]
pub struct PickInfo {
    pub idx: u32,
    pub screen_pos: [f32; 2],
    pub screen_dist: f32,
    pub view_depth: f32,
    pub candidates_in_radius: u32,
    pub candidates_passing_depth: u32,
    pub visible_depth: Option<f32>,
    pub depth_tol: f32,
}

/// Per-vertex maximum incident density, used to reject empty-space vertices.
pub fn vertex_max_density(scene: &SceneData) -> Vec<f32> {
    let mut result = vec![0.0_f32; scene.vertex_count as usize];
    for (ids, &density) in scene.indices.chunks_exact(4).zip(&scene.densities) {
        for &id in ids {
            result[id as usize] = result[id as usize].max(density);
        }
    }
    result
}

/// Nearest dense vertex in the cursor radius. An optional view-depth sample
/// can constrain selection when a caller actually has a rendered depth image.
/// The color-only viewers pass None; no screen-space depth channel is needed.
#[allow(clippy::too_many_arguments)]
pub fn pick_vertex(
    verts: &[f32],
    max_density: Option<&[f32]>,
    view: Mat4,
    vp: Mat4,
    viewport: [f32; 2],
    mouse_px: [f32; 2],
    radius_px: f32,
    visible_depth: Option<f32>,
) -> Option<PickInfo> {
    let [w, h] = viewport;
    if w <= 0.0
        || h <= 0.0
        || !mouse_px.iter().all(|v| v.is_finite())
        || !radius_px.is_finite()
        || radius_px <= 0.0
    {
        return None;
    }
    let r_sq = radius_px * radius_px;

    let depth_tol = 0.15_f32;
    // Density floor: in .rmesh's exp((val-100)/20) encoding, 1.0 = the
    // middle value (val=100). Vertices touching only sub-1.0 tets are
    // basically in empty fog space — reject.
    let min_density = 1.0_f32;

    let mut best: Option<(u32, f32, f32, f32, f32)> = None;
    let mut candidates_in_radius = 0u32;
    let mut candidates_passing_depth = 0u32;
    let mut candidates_passing_density = 0u32;
    let n = verts.len() / 3;
    for i in 0..n {
        let b = i * 3;
        let p = glam::Vec4::new(verts[b], verts[b + 1], verts[b + 2], 1.0);
        let view_pos = view * p;
        if view_pos.z >= -0.001 {
            continue;
        }
        let view_depth = -view_pos.z;
        let clip = vp * p;
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;
        let sx = (ndc.x * 0.5 + 0.5) * w;
        let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * h;
        let dx = sx - mouse_px[0];
        let dy = sy - mouse_px[1];
        let d_sq = dx * dx + dy * dy;
        if d_sq >= r_sq {
            continue;
        }
        candidates_in_radius += 1;
        if let Some(vd) = visible_depth {
            if (view_depth - vd).abs() > depth_tol {
                continue;
            }
        }
        candidates_passing_depth += 1;
        if let Some(md) = max_density {
            if md[i] < min_density {
                continue;
            }
        }
        candidates_passing_density += 1;
        match best {
            Some((_, _, bd, _, _)) if view_depth >= bd => {}
            _ => best = Some((i as u32, d_sq, view_depth, sx, sy)),
        }
    }
    let _ = candidates_passing_density; // tracked here for symmetry; could be surfaced
    best.map(|(idx, d_sq, view_depth, sx, sy)| PickInfo {
        idx,
        screen_pos: [sx, sy],
        screen_dist: d_sq.sqrt(),
        view_depth,
        candidates_in_radius,
        candidates_passing_depth,
        visible_depth,
        depth_tol,
    })
}

pub struct PreparedGrab {
    pub island: Island,
    pub coloring: ConstraintColoring,
    pub handle_initial: Vec<(u32, [f32; 3])>,
}

/// Refresh depth-sort bounds after committing a deformation.
pub fn refresh_bounds(
    scene: &mut SceneData,
    queue: &wgpu::Queue,
    buffers: &rmesh_data::SceneBuffers,
) {
    scene.circumdata = rmesh_data::compute_circumspheres_parallel(
        &scene.vertices,
        &scene.indices,
        scene.tet_count as usize,
    );
    queue.write_buffer(
        &buffers.circumdata,
        0,
        bytemuck::cast_slice(&scene.circumdata),
    );
}

pub fn prepare_grab(
    scene: &SceneData,
    topology: &MeshTopology,
    handles: &[u32],
    radius: f32,
) -> PreparedGrab {
    let island = build_island(topology, &scene.indices, &scene.vertices, handles, radius);
    let coloring = color_constraints(island.particles.len(), &island.distance_constraints);
    let handle_initial = handles
        .iter()
        .map(|&h| {
            let b = h as usize * 3;
            (
                h,
                [
                    scene.vertices[b],
                    scene.vertices[b + 1],
                    scene.vertices[b + 2],
                ],
            )
        })
        .collect();
    PreparedGrab {
        island,
        coloring,
        handle_initial,
    }
}

/// Convert screen displacement to world-space handle positions at each depth.
pub fn drag_positions(
    handles: &[(u32, [f32; 3])],
    view: Mat4,
    projection: Mat4,
    height: f32,
    delta: [f32; 2],
) -> Vec<[f32; 4]> {
    let focal = projection.col(1).y * height * 0.5;
    let inverse = view.inverse();
    let right = inverse.col(0).truncate();
    let up = inverse.col(1).truncate();
    handles
        .iter()
        .map(|(_, p)| {
            let p = glam::Vec3::from(*p);
            let scale = (-(view * p.extend(1.0)).z).max(0.01) / focal;
            let q = p + (right * delta[0] - up * delta[1]) * scale;
            [q.x, q.y, q.z, 0.0]
        })
        .collect()
}

/// Nonblocking readback: browser callers poll completion on subsequent frames.
pub struct VertexReadback {
    buffer: wgpu::Buffer,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

impl VertexReadback {
    pub fn start(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &wgpu::Buffer,
        count: usize,
    ) -> Self {
        let size = (count * 4) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_grab_readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(vertices, 0, &buffer, 0, size);
        queue.submit([encoder.finish()]);
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
        let ready = state.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                ready.store(
                    if result.is_ok() { 1 } else { 2 },
                    std::sync::atomic::Ordering::Release,
                );
            });
        Self { buffer, state }
    }

    pub fn poll(&self) -> Option<Result<Vec<f32>, &'static str>> {
        match self.state.load(std::sync::atomic::Ordering::Acquire) {
            0 | 3 => None,
            1 => {
                let data = self.buffer.slice(..).get_mapped_range();
                let vertices = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
                drop(data);
                self.buffer.unmap();
                self.state.store(3, std::sync::atomic::Ordering::Release);
                Some(Ok(vertices))
            }
            _ => Some(Err("Vertex readback failed")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_readback_returns_vertices_once() {
        use wgpu::util::DeviceExt;
        let Some((device, queue)) = rmesh_util::test_util::create_test_device_default() else {
            eprintln!("No GPU: skipping vertex readback test");
            return;
        };
        let expected = [1.0_f32, 2.0, 3.0, -4.0, 5.0, 6.0];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("readback_fixture"),
            contents: bytemuck::cast_slice(&expected),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let readback = VertexReadback::start(&device, &queue, &buffer, expected.len());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(readback.poll().unwrap().unwrap(), expected);
        assert!(readback.poll().is_none());
    }
    #[test]
    fn picking_ignores_fog_and_behind_camera() {
        let vertices = [0., 0., -3., 0., 0., -2., 0., 0., -1., 0., 0., 1.];
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let pick = pick_vertex(
            &vertices,
            Some(&[2., 2., 0.1, 2.]),
            Mat4::IDENTITY,
            proj,
            [100., 100.],
            [50., 50.],
            10.,
            None,
        )
        .unwrap();
        assert_eq!(pick.idx, 1);
        assert!(pick_vertex(
            &vertices,
            None,
            Mat4::IDENTITY,
            proj,
            [100., 100.],
            [0., 0.],
            2.,
            None
        )
        .is_none());
    }
    #[test]
    fn drag_scales_with_depth_and_flips_screen_y() {
        let projection = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.);
        let positions = drag_positions(
            &[(0, [0., 0., -2.]), (1, [0., 0., -4.])],
            Mat4::IDENTITY,
            projection,
            100.,
            [10., 10.],
        );
        assert!(positions[0][0] > 0. && positions[0][1] < 0.);
        assert!((positions[1][0] - 2. * positions[0][0]).abs() < 1e-6);
    }
}
