//! Unit mesh generation for primitives (cube, sphere, plane, cylinder).
//! All meshes are centered at origin, fitting within a unit bounding box [-0.5, 0.5]^3.
//!
//! Geometry is generated in the scene's COLMAP coordinate system where -Z is up:
//!   X = right, Y = forward, Z = -up.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// A single vertex with position and normal.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PrimitiveVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Convert a Y-up coordinate to the scene's COLMAP coordinate system (-Z up).
/// (x, y, z)_yup → (x, z, -y)_colmap
fn yup_to_colmap(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], -v[1]]
}

/// Offset and count into the shared vertex buffer for one primitive kind.
#[derive(Copy, Clone, Debug)]
pub struct MeshSlice {
    pub offset: u32,
    pub count: u32,
}

/// GPU vertex buffer containing all four unit meshes.
pub struct PrimitiveGeometry {
    pub vertex_buffer: wgpu::Buffer,
    pub kinds: [MeshSlice; 4],
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
        }
    }
}

fn generate_cube() -> Vec<PrimitiveVertex> {
    let mut verts = Vec::with_capacity(36);

    // Each quad's vertices must be ordered so that (v1-v0)×(v2-v0) = outward normal (CCW).
    // Generated in Y-up then converted to COLMAP space.
    let faces_yup: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]),
        // -Y
        ([0.0, -1.0, 0.0], [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]),
        // -Z
        ([0.0, 0.0, -1.0], [[0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]]),
    ];

    for (normal, quad) in &faces_yup {
        let n = yup_to_colmap(*normal);
        for &[a, b, c] in &[[0, 1, 2], [0, 2, 3]] {
            verts.push(PrimitiveVertex { position: yup_to_colmap(quad[a]), normal: n });
            verts.push(PrimitiveVertex { position: yup_to_colmap(quad[b]), normal: n });
            verts.push(PrimitiveVertex { position: yup_to_colmap(quad[c]), normal: n });
        }
    }

    verts
}

fn generate_sphere(slices: u32, stacks: u32) -> Vec<PrimitiveVertex> {
    let mut verts = Vec::new();
    let r = 0.5;

    // Helper: Y-up spherical coords to COLMAP position and normal
    let sphere_pt = |theta: f32, phi: f32| -> ([f32; 3], [f32; 3]) {
        let (st, ct) = (theta.sin(), theta.cos());
        let (sp, cp) = (phi.sin(), phi.cos());
        let n = yup_to_colmap([st * cp, ct, st * sp]);
        let p = [n[0] * r, n[1] * r, n[2] * r];
        (p, n)
    };

    let north_n = yup_to_colmap([0.0, 1.0, 0.0]);
    let north_p = [north_n[0] * r, north_n[1] * r, north_n[2] * r];
    let south_n = yup_to_colmap([0.0, -1.0, 0.0]);
    let south_p = [south_n[0] * r, south_n[1] * r, south_n[2] * r];

    let theta1 = std::f32::consts::PI / stacks as f32;
    let theta_last = std::f32::consts::PI * (stacks - 1) as f32 / stacks as f32;

    for j in 0..slices {
        let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
        let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;

        // North pole fan
        let (p0, n0) = sphere_pt(theta1, phi0);
        let (p1, n1) = sphere_pt(theta1, phi1);
        verts.push(PrimitiveVertex { position: north_p, normal: north_n });
        verts.push(PrimitiveVertex { position: p1, normal: n1 });
        verts.push(PrimitiveVertex { position: p0, normal: n0 });

        // South pole fan
        let (p0, n0) = sphere_pt(theta_last, phi0);
        let (p1, n1) = sphere_pt(theta_last, phi1);
        verts.push(PrimitiveVertex { position: south_p, normal: south_n });
        verts.push(PrimitiveVertex { position: p0, normal: n0 });
        verts.push(PrimitiveVertex { position: p1, normal: n1 });
    }

    // Middle bands (i=1..stacks-1)
    for i in 1..stacks - 1 {
        let theta0 = std::f32::consts::PI * i as f32 / stacks as f32;
        let theta1 = std::f32::consts::PI * (i + 1) as f32 / stacks as f32;

        for j in 0..slices {
            let phi0 = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let phi1 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / slices as f32;

            let (p00, n00) = sphere_pt(theta0, phi0);
            let (p10, n10) = sphere_pt(theta1, phi0);
            let (p01, n01) = sphere_pt(theta0, phi1);
            let (p11, n11) = sphere_pt(theta1, phi1);

            verts.push(PrimitiveVertex { position: p00, normal: n00 });
            verts.push(PrimitiveVertex { position: p11, normal: n11 });
            verts.push(PrimitiveVertex { position: p10, normal: n10 });

            verts.push(PrimitiveVertex { position: p00, normal: n00 });
            verts.push(PrimitiveVertex { position: p01, normal: n01 });
            verts.push(PrimitiveVertex { position: p11, normal: n11 });
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

    vec![
        PrimitiveVertex { position: q0, normal: n },
        PrimitiveVertex { position: q1, normal: n },
        PrimitiveVertex { position: q2, normal: n },
        PrimitiveVertex { position: q0, normal: n },
        PrimitiveVertex { position: q2, normal: n },
        PrimitiveVertex { position: q3, normal: n },
    ]
}

fn generate_cylinder(segments: u32) -> Vec<PrimitiveVertex> {
    let mut verts = Vec::new();
    let r = 0.5;
    let h = 0.5;

    for i in 0..segments {
        let a0 = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
        let a1 = 2.0 * std::f32::consts::PI * (i + 1) as f32 / segments as f32;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        // Y-up values, then convert
        let n0 = yup_to_colmap([c0, 0.0, s0]);
        let n1 = yup_to_colmap([c1, 0.0, s1]);
        let p0b = yup_to_colmap([c0 * r, -h, s0 * r]);
        let p1b = yup_to_colmap([c1 * r, -h, s1 * r]);
        let p0t = yup_to_colmap([c0 * r, h, s0 * r]);
        let p1t = yup_to_colmap([c1 * r, h, s1 * r]);

        // Side quad
        verts.push(PrimitiveVertex { position: p0b, normal: n0 });
        verts.push(PrimitiveVertex { position: p1t, normal: n1 });
        verts.push(PrimitiveVertex { position: p1b, normal: n1 });

        verts.push(PrimitiveVertex { position: p0b, normal: n0 });
        verts.push(PrimitiveVertex { position: p0t, normal: n0 });
        verts.push(PrimitiveVertex { position: p1t, normal: n1 });

        // Top cap
        let nt = yup_to_colmap([0.0, 1.0, 0.0]);
        let top_center = yup_to_colmap([0.0, h, 0.0]);
        verts.push(PrimitiveVertex { position: top_center, normal: nt });
        verts.push(PrimitiveVertex { position: p1t, normal: nt });
        verts.push(PrimitiveVertex { position: p0t, normal: nt });

        // Bottom cap
        let nb = yup_to_colmap([0.0, -1.0, 0.0]);
        let bot_center = yup_to_colmap([0.0, -h, 0.0]);
        verts.push(PrimitiveVertex { position: bot_center, normal: nb });
        verts.push(PrimitiveVertex { position: p0b, normal: nb });
        verts.push(PrimitiveVertex { position: p1b, normal: nb });
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
}
