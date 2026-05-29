//! Unit mesh generation for primitives (cube, sphere, plane, cylinder).
//! All meshes are centered at origin, fitting within a unit bounding box [-0.5, 0.5]^3.
//!
//! Geometry is generated in the scene's COLMAP coordinate system where -Z is up:
//!   X = right, Y = forward, Z = -up.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// A single vertex with position, normal, UV, and tangent.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PrimitiveVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4], // xyz = tangent direction, w = handedness sign
}

/// Convert a Y-up coordinate to the scene's COLMAP coordinate system (-Z up).
/// (x, y, z)_yup → (x, z, -y)_colmap
fn yup_to_colmap(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], -v[1]]
}

/// Compute tangent for a triangle from position and UV gradients.
/// Returns `(tangent_dir, handedness_sign)`.
fn compute_tangent(
    p0: [f32; 3], p1: [f32; 3], p2: [f32; 3],
    uv0: [f32; 2], uv1: [f32; 2], uv2: [f32; 2],
    normal: [f32; 3],
) -> [f32; 4] {
    let dp1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let dp2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
    let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

    let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
    let inv_det = if det.abs() < 1e-10 { 1.0 } else { 1.0 / det };

    let t = [
        inv_det * (duv2[1] * dp1[0] - duv1[1] * dp2[0]),
        inv_det * (duv2[1] * dp1[1] - duv1[1] * dp2[1]),
        inv_det * (duv2[1] * dp1[2] - duv1[1] * dp2[2]),
    ];

    let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let t_norm = if len < 1e-10 {
        [1.0, 0.0, 0.0]
    } else {
        [t[0] / len, t[1] / len, t[2] / len]
    };

    // Compute bitangent for handedness
    let b = [
        inv_det * (-duv2[0] * dp1[0] + duv1[0] * dp2[0]),
        inv_det * (-duv2[0] * dp1[1] + duv1[0] * dp2[1]),
        inv_det * (-duv2[0] * dp1[2] + duv1[0] * dp2[2]),
    ];
    // handedness = sign(dot(cross(normal, tangent), bitangent))
    let cross = [
        normal[1] * t_norm[2] - normal[2] * t_norm[1],
        normal[2] * t_norm[0] - normal[0] * t_norm[2],
        normal[0] * t_norm[1] - normal[1] * t_norm[0],
    ];
    let dot = cross[0] * b[0] + cross[1] * b[1] + cross[2] * b[2];
    let w = if dot >= 0.0 { 1.0 } else { -1.0 };

    [t_norm[0], t_norm[1], t_norm[2], w]
}

/// Offset and count into the shared vertex buffer for one primitive kind.
#[derive(Copy, Clone, Debug)]
pub struct MeshSlice {
    pub offset: u32,
    pub count: u32,
}

/// GPU vertex buffer containing all four unit meshes plus optional custom meshes.
pub struct PrimitiveGeometry {
    pub vertex_buffer: wgpu::Buffer,
    pub kinds: [MeshSlice; 4],
    /// Separate vertex buffer for custom meshes loaded from glTF etc.
    pub custom_vertex_buffer: Option<wgpu::Buffer>,
    /// Mesh slices into `custom_vertex_buffer`, indexed by `PrimitiveKind::CustomMesh(id)`.
    pub custom_meshes: Vec<MeshSlice>,
    /// Separate vertex buffer for collision debug mesh overlay.
    pub collision_vertex_buffer: Option<wgpu::Buffer>,
    pub collision_vertex_count: u32,
}

impl PrimitiveGeometry {
    /// Generate all unit meshes and upload to the device.
    pub fn new(device: &wgpu::Device) -> Self {
        let cube = generate_cube();
        let sphere = generate_sphere(16, 8);
        let plane = generate_plane();
        let cylinder = generate_cylinder(24);

        let mut all_verts = Vec::new();
        let mut kinds = [MeshSlice { offset: 0, count: 0 }; 4];

        for (i, mesh) in [&cube, &sphere, &plane, &cylinder].iter().enumerate() {
            kinds[i] = MeshSlice {
                offset: all_verts.len() as u32,
                count: mesh.len() as u32,
            };
            all_verts.extend_from_slice(mesh);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("primitive_geometry"),
            contents: bytemuck::cast_slice(&all_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            kinds,
            custom_vertex_buffer: None,
            custom_meshes: Vec::new(),
            collision_vertex_buffer: None,
            collision_vertex_count: 0,
        }
    }

    /// Upload custom mesh vertex data. Each entry is a flat triangle list.
    /// Replaces any previously loaded custom meshes.
    pub fn set_custom_meshes(&mut self, device: &wgpu::Device, meshes: &[Vec<PrimitiveVertex>]) {
        if meshes.is_empty() || meshes.iter().all(|m| m.is_empty()) {
            self.custom_vertex_buffer = None;
            self.custom_meshes.clear();
            return;
        }

        let mut all_verts = Vec::new();
        let mut slices = Vec::new();
        for mesh in meshes {
            slices.push(MeshSlice {
                offset: all_verts.len() as u32,
                count: mesh.len() as u32,
            });
            all_verts.extend_from_slice(mesh);
        }

        self.custom_vertex_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("custom_mesh_geometry"),
                contents: bytemuck::cast_slice(&all_verts),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.custom_meshes = slices;
    }

    /// Upload collision debug mesh vertex data. Pass empty slice to clear.
    pub fn set_collision_mesh(&mut self, device: &wgpu::Device, vertices: &[PrimitiveVertex]) {
        if vertices.is_empty() {
            self.collision_vertex_buffer = None;
            self.collision_vertex_count = 0;
            return;
        }
        self.collision_vertex_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("collision_debug_mesh"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));
        self.collision_vertex_count = vertices.len() as u32;
    }
}

fn generate_cube() -> Vec<PrimitiveVertex> {
    let mut verts = Vec::with_capacity(36);

    // Each quad's vertices must be ordered so that (v1-v0)×(v2-v0) = outward normal (CCW).
    // Generated in Y-up then converted to COLMAP space.
    // UVs: each face gets a standard [0,1]^2 mapping.
    let faces_yup: [([f32; 3], [[f32; 3]; 4], [[f32; 2]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0],
         [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]],
         [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        // -X
        ([-1.0, 0.0, 0.0],
         [[-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]],
         [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        // +Y
        ([0.0, 1.0, 0.0],
         [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]],
         [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        // -Y
        ([0.0, -1.0, 0.0],
         [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]],
         [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        // +Z
        ([0.0, 0.0, 1.0],
         [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
         [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
        // -Z
        ([0.0, 0.0, -1.0],
         [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]],
         [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
    ];

    for (normal, quad, uvs) in &faces_yup {
        let n = yup_to_colmap(*normal);
        let p: Vec<[f32; 3]> = quad.iter().map(|v| yup_to_colmap(*v)).collect();
        for &[a, b, c] in &[[0, 1, 2], [0, 2, 3]] {
            let tan = compute_tangent(p[a], p[b], p[c], uvs[a], uvs[b], uvs[c], n);
            verts.push(PrimitiveVertex { position: p[a], normal: n, uv: uvs[a], tangent: tan });
            verts.push(PrimitiveVertex { position: p[b], normal: n, uv: uvs[b], tangent: tan });
            verts.push(PrimitiveVertex { position: p[c], normal: n, uv: uvs[c], tangent: tan });
        }
    }

    verts
}

fn generate_sphere(slices: u32, stacks: u32) -> Vec<PrimitiveVertex> {
    let mut verts = Vec::new();
    let r = 0.5;
    let pi = std::f32::consts::PI;
    let two_pi = 2.0 * pi;

    // Helper: spherical coords → position, normal, UV (all in Y-up, then convert)
    let sphere_pt = |theta: f32, phi: f32| -> ([f32; 3], [f32; 3], [f32; 2]) {
        let (st, ct) = (theta.sin(), theta.cos());
        let (sp, cp) = (phi.sin(), phi.cos());
        let n = yup_to_colmap([st * cp, ct, st * sp]);
        let p = [n[0] * r, n[1] * r, n[2] * r];
        let u = phi / two_pi;
        let v = theta / pi;
        (p, n, [u, v])
    };

    let north_n = yup_to_colmap([0.0, 1.0, 0.0]);
    let north_p = [north_n[0] * r, north_n[1] * r, north_n[2] * r];
    let south_n = yup_to_colmap([0.0, -1.0, 0.0]);
    let south_p = [south_n[0] * r, south_n[1] * r, south_n[2] * r];

    let theta1 = pi / stacks as f32;
    let theta_last = pi * (stacks - 1) as f32 / stacks as f32;

    for j in 0..slices {
        let phi0 = two_pi * j as f32 / slices as f32;
        let phi1 = two_pi * (j + 1) as f32 / slices as f32;

        // North pole fan
        let (p0, n0, uv0) = sphere_pt(theta1, phi0);
        let (p1, n1, uv1) = sphere_pt(theta1, phi1);
        let north_uv = [(phi0 + phi1) / (2.0 * two_pi), 0.0];
        let tan = compute_tangent(north_p, p1, p0, north_uv, uv1, uv0, north_n);
        verts.push(PrimitiveVertex { position: north_p, normal: north_n, uv: north_uv, tangent: tan });
        verts.push(PrimitiveVertex { position: p1, normal: n1, uv: uv1, tangent: tan });
        verts.push(PrimitiveVertex { position: p0, normal: n0, uv: uv0, tangent: tan });

        // South pole fan
        let (p0, n0, uv0) = sphere_pt(theta_last, phi0);
        let (p1, n1, uv1) = sphere_pt(theta_last, phi1);
        let south_uv = [(phi0 + phi1) / (2.0 * two_pi), 1.0];
        let tan = compute_tangent(south_p, p0, p1, south_uv, uv0, uv1, south_n);
        verts.push(PrimitiveVertex { position: south_p, normal: south_n, uv: south_uv, tangent: tan });
        verts.push(PrimitiveVertex { position: p0, normal: n0, uv: uv0, tangent: tan });
        verts.push(PrimitiveVertex { position: p1, normal: n1, uv: uv1, tangent: tan });
    }

    // Middle bands
    for i in 1..stacks - 1 {
        let theta0 = pi * i as f32 / stacks as f32;
        let theta1 = pi * (i + 1) as f32 / stacks as f32;

        for j in 0..slices {
            let phi0 = two_pi * j as f32 / slices as f32;
            let phi1 = two_pi * (j + 1) as f32 / slices as f32;

            let (p00, n00, uv00) = sphere_pt(theta0, phi0);
            let (p10, n10, uv10) = sphere_pt(theta1, phi0);
            let (p01, n01, uv01) = sphere_pt(theta0, phi1);
            let (p11, n11, uv11) = sphere_pt(theta1, phi1);

            let tan0 = compute_tangent(p00, p11, p10, uv00, uv11, uv10, n00);
            verts.push(PrimitiveVertex { position: p00, normal: n00, uv: uv00, tangent: tan0 });
            verts.push(PrimitiveVertex { position: p11, normal: n11, uv: uv11, tangent: tan0 });
            verts.push(PrimitiveVertex { position: p10, normal: n10, uv: uv10, tangent: tan0 });

            let tan1 = compute_tangent(p00, p01, p11, uv00, uv01, uv11, n00);
            verts.push(PrimitiveVertex { position: p00, normal: n00, uv: uv00, tangent: tan1 });
            verts.push(PrimitiveVertex { position: p01, normal: n01, uv: uv01, tangent: tan1 });
            verts.push(PrimitiveVertex { position: p11, normal: n11, uv: uv11, tangent: tan1 });
        }
    }

    verts
}

fn generate_plane() -> Vec<PrimitiveVertex> {
    let n = yup_to_colmap([0.0, 1.0, 0.0]);
    let q0 = yup_to_colmap([-0.5, 0.0, 0.5]);
    let q1 = yup_to_colmap([0.5, 0.0, 0.5]);
    let q2 = yup_to_colmap([0.5, 0.0, -0.5]);
    let q3 = yup_to_colmap([-0.5, 0.0, -0.5]);

    let uv0 = [0.0, 0.0];
    let uv1 = [1.0, 0.0];
    let uv2 = [1.0, 1.0];
    let uv3 = [0.0, 1.0];

    let tan0 = compute_tangent(q0, q1, q2, uv0, uv1, uv2, n);
    let tan1 = compute_tangent(q0, q2, q3, uv0, uv2, uv3, n);

    vec![
        PrimitiveVertex { position: q0, normal: n, uv: uv0, tangent: tan0 },
        PrimitiveVertex { position: q1, normal: n, uv: uv1, tangent: tan0 },
        PrimitiveVertex { position: q2, normal: n, uv: uv2, tangent: tan0 },
        PrimitiveVertex { position: q0, normal: n, uv: uv0, tangent: tan1 },
        PrimitiveVertex { position: q2, normal: n, uv: uv2, tangent: tan1 },
        PrimitiveVertex { position: q3, normal: n, uv: uv3, tangent: tan1 },
    ]
}

fn generate_cylinder(segments: u32) -> Vec<PrimitiveVertex> {
    let mut verts = Vec::new();
    let r = 0.5;
    let h = 0.5;
    let two_pi = 2.0 * std::f32::consts::PI;

    for i in 0..segments {
        let a0 = two_pi * i as f32 / segments as f32;
        let a1 = two_pi * (i + 1) as f32 / segments as f32;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        let u0 = i as f32 / segments as f32;
        let u1 = (i + 1) as f32 / segments as f32;

        // Y-up values, then convert
        let n0 = yup_to_colmap([c0, 0.0, s0]);
        let n1 = yup_to_colmap([c1, 0.0, s1]);
        let p0b = yup_to_colmap([c0 * r, -h, s0 * r]);
        let p1b = yup_to_colmap([c1 * r, -h, s1 * r]);
        let p0t = yup_to_colmap([c0 * r, h, s0 * r]);
        let p1t = yup_to_colmap([c1 * r, h, s1 * r]);

        // Side quad UVs: u = angle fraction, v = height (0=bottom, 1=top)
        let uv_0b = [u0, 0.0];
        let uv_1b = [u1, 0.0];
        let uv_0t = [u0, 1.0];
        let uv_1t = [u1, 1.0];

        let tan_s0 = compute_tangent(p0b, p1t, p1b, uv_0b, uv_1t, uv_1b, n0);
        verts.push(PrimitiveVertex { position: p0b, normal: n0, uv: uv_0b, tangent: tan_s0 });
        verts.push(PrimitiveVertex { position: p1t, normal: n1, uv: uv_1t, tangent: tan_s0 });
        verts.push(PrimitiveVertex { position: p1b, normal: n1, uv: uv_1b, tangent: tan_s0 });

        let tan_s1 = compute_tangent(p0b, p0t, p1t, uv_0b, uv_0t, uv_1t, n0);
        verts.push(PrimitiveVertex { position: p0b, normal: n0, uv: uv_0b, tangent: tan_s1 });
        verts.push(PrimitiveVertex { position: p0t, normal: n0, uv: uv_0t, tangent: tan_s1 });
        verts.push(PrimitiveVertex { position: p1t, normal: n1, uv: uv_1t, tangent: tan_s1 });

        // Top cap: circular projection UVs
        let nt = yup_to_colmap([0.0, 1.0, 0.0]);
        let top_center = yup_to_colmap([0.0, h, 0.0]);
        let uv_tc = [0.5, 0.5];
        let uv_t0 = [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5];
        let uv_t1 = [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5];
        let tan_t = compute_tangent(top_center, p1t, p0t, uv_tc, uv_t1, uv_t0, nt);
        verts.push(PrimitiveVertex { position: top_center, normal: nt, uv: uv_tc, tangent: tan_t });
        verts.push(PrimitiveVertex { position: p1t, normal: nt, uv: uv_t1, tangent: tan_t });
        verts.push(PrimitiveVertex { position: p0t, normal: nt, uv: uv_t0, tangent: tan_t });

        // Bottom cap
        let nb = yup_to_colmap([0.0, -1.0, 0.0]);
        let bot_center = yup_to_colmap([0.0, -h, 0.0]);
        let uv_bc = [0.5, 0.5];
        let uv_b0 = [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5];
        let uv_b1 = [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5];
        let tan_b = compute_tangent(bot_center, p0b, p1b, uv_bc, uv_b0, uv_b1, nb);
        verts.push(PrimitiveVertex { position: bot_center, normal: nb, uv: uv_bc, tangent: tan_b });
        verts.push(PrimitiveVertex { position: p0b, normal: nb, uv: uv_b0, tangent: tan_b });
        verts.push(PrimitiveVertex { position: p1b, normal: nb, uv: uv_b1, tangent: tan_b });
    }

    verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_vertex_counts() {
        let cube = generate_cube();
        assert_eq!(cube.len(), 36);

        let sphere = generate_sphere(16, 8);
        assert!(sphere.len() > 400, "sphere should have many vertices, got {}", sphere.len());

        let plane = generate_plane();
        assert_eq!(plane.len(), 6);

        let cylinder = generate_cylinder(24);
        assert!(cylinder.len() > 100, "cylinder should have many vertices, got {}", cylinder.len());
    }

    #[test]
    fn all_normals_unit_length() {
        for mesh in [generate_cube(), generate_sphere(16, 8), generate_plane(), generate_cylinder(24)] {
            for v in &mesh {
                let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
                assert!((len - 1.0).abs() < 0.01, "normal not unit length: {:?} (len={})", v.normal, len);
            }
        }
    }

    #[test]
    fn vertex_size_48_bytes() {
        assert_eq!(std::mem::size_of::<PrimitiveVertex>(), 48);
    }
}
