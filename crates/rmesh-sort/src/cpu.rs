//! CPU frustum culling + depth sort for rmesh-viewer-web.
//!
//! WebGPU on wgpu 28 doesn't expose the `subgroups` feature, so the fast
//! [`crate::drs`] backend isn't reachable. The fallback [`crate::basic`]
//! backend produces visibly out-of-order rendering at scale (4.8M tets), so
//! the web viewer can't trust GPU sort. This module replaces it with a
//! per-frame sphere-vs-frustum cull + back-to-front sort done on the CPU,
//! mirroring brush_rs.
//!
//! The output is a tightly-packed list of *visible* tet IDs in back-to-front
//! order, ready to write into `SceneBuffers::sort_values` and feed to the
//! interval-shading pipeline. The caller also overwrites
//! `indirect_args.instance_count` with the returned visible count.

use glam::{Mat4, Vec3, Vec4};

/// Reusable per-frame workspace for CPU cull + sort.
pub struct CpuSorter {
    /// `(sort_key << 32) | tet_id` — radsort sorts u64 ascending; with the
    /// key in the high 32 bits, the sort orders by key with tet_id as the
    /// piggy-backed value.
    packed: Vec<u64>,
    /// Tet IDs in back-to-front order. Length = visible count from the most
    /// recent `cull_and_sort` call.
    pub sorted_indices: Vec<u32>,
}

impl CpuSorter {
    /// `tet_count` is an upper bound used to pre-allocate the workspace.
    pub fn new(tet_count: usize) -> Self {
        Self {
            packed: Vec::with_capacity(tet_count),
            sorted_indices: Vec::with_capacity(tet_count),
        }
    }

    /// Cull tets against the camera frustum and sort visible ones back-to-front.
    ///
    /// Arguments mirror what `project_compute_hw.wgsl` reads from GPU buffers:
    /// - `vertices`: `[V * 3]` f32 (per-vertex world-space position)
    /// - `indices`:  `[M * 4]` u32 (per-tet vertex indices)
    /// - `circumdata`: `[M * 4]` f32 — `[cx, cy, cz, radius²]` per tet (see
    ///   `compute_circumspheres_parallel` in rmesh-data)
    /// - `cam_pos`: camera world-space position (`uniforms.cam_pos_pad.xyz`)
    /// - `vp`: view-projection matrix (`uniforms.vp_col0..3` as columns)
    ///
    /// Returns the number of visible tets. After return,
    /// `self.sorted_indices[..visible_count]` is the back-to-front list,
    /// matching what the GPU's project + radix-sort path would have produced
    /// — but correctly, on Chrome WebGPU.
    pub fn cull_and_sort(
        &mut self,
        vertices: &[f32],
        indices: &[u32],
        circumdata: &[f32],
        cam_pos: Vec3,
        vp: Mat4,
    ) -> usize {
        let tet_count = indices.len() / 4;
        self.packed.clear();
        self.packed.reserve(tet_count);

        // Gribb-Hartmann frustum planes extracted from VP, with wgpu/Metal
        // NDC z range [0, 1] (near = r2, NOT r3 + r2).
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);
        let raw = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r2,      // near
            r3 - r2, // far
        ];
        // Normalize so `dot(plane.xyz, p) + plane.w` is Euclidean distance.
        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in raw.iter().enumerate() {
            let len = p.truncate().length();
            planes[i] = if len > 1e-10 { *p / len } else { *p };
        }

        for tet in 0..tet_count {
            let base = tet * 4;
            let center = Vec3::new(circumdata[base], circumdata[base + 1], circumdata[base + 2]);
            // circumdata[base + 3] is radius² (see rmesh-data
            // compute_circumspheres_parallel). Take sqrt for the plane test.
            let radius = circumdata[base + 3].sqrt();

            // Sphere-vs-frustum: reject if the sphere is fully outside any
            // single plane. Conservative — keeps slightly more tets than the
            // GPU's per-vertex test, but never culls a visible tet.
            let mut culled = false;
            for plane in &planes {
                let dist = plane.truncate().dot(center) + plane.w;
                if dist + radius < 0.0 {
                    culled = true;
                    break;
                }
            }
            if culled {
                continue;
            }

            // Depth key — same formula as project_compute_hw.wgsl line 153:
            //   depth = (cam - v0) · (cam + v0 - 2c)
            // Bit-cast to u32, then bitwise NOT so radix sort ascending
            // produces back-to-front order (farthest = smallest key).
            let i0 = indices[base] as usize;
            let v0 = Vec3::new(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
            let depth = (cam_pos - v0)
                .dot(cam_pos + v0 - 2.0 * center)
                .clamp(-1e20, 1e20);
            let key = !depth.to_bits();

            // Pack key in high bits, tet_id in low bits — std `sort_unstable`
            // (pdqsort) sorts u64 ascending = sorts by key with tet_id as
            // tiebreaker. Using pdqsort instead of radsort because radsort
            // allocates ~8N bytes of scratch per call (~38MB at 4.8M tets);
            // pdqsort is in-place and only ~2-3× slower for this workload.
            self.packed.push(((key as u64) << 32) | (tet as u64));
        }

        self.packed.sort_unstable();

        self.sorted_indices.clear();
        self.sorted_indices.reserve(self.packed.len());
        for &p in &self.packed {
            self.sorted_indices.push((p & 0xFFFF_FFFF) as u32);
        }
        self.sorted_indices.len()
    }
}
