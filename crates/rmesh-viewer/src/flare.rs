//! Flare gun: shoot light projectiles into the scene.
//!
//! Extracts a boundary surface mesh from dense tets (density ≥ threshold),
//! builds a BVH for fast ray-segment collision, and renders a debug overlay.

use glam::{Quat, Vec3};
use rmesh_interact::{Primitive, PrimitiveKind};
use std::collections::HashMap;
use std::path::Path;

const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, 9.81); // +Z is down in COLMAP coords (-Z up)
const INITIAL_SPEED: f32 = 15.0;
const TIME_SCALE: f32 = 0.1; // slow motion
const MAX_LIFETIME: f32 = 30.0;
const FADE_DURATION: f32 = 5.0;
const FLARE_COLOR: [f32; 3] = [1.0, 0.85, 0.5];
const FLARE_INTENSITY: f32 = 2.0;
const FLARE_RADIUS: f32 = 0.2;
/// The model is ~17 units tall along its local axis.
const FLARE_MODEL_HEIGHT: f32 = 17.0;
const FLARE_MODEL_SCALE: f32 = 2.0 * FLARE_RADIUS / FLARE_MODEL_HEIGHT;
const BVH_MAX_LEAF: usize = 4;

// -----------------------------------------------------------------------
// Tet face winding — matches rmesh-util TET_FACES
// -----------------------------------------------------------------------

/// Tet face vertex indices: [a, b, c, opposite].
const TET_FACES: [[usize; 4]; 4] = [[0, 2, 1, 3], [1, 2, 3, 0], [0, 3, 2, 1], [3, 0, 1, 2]];

// -----------------------------------------------------------------------
// Collision Mesh: boundary surface from dense tets + BVH
// -----------------------------------------------------------------------

struct BvhNode {
    aabb_min: Vec3,
    aabb_max: Vec3,
    /// Leaf: start index into `tri_order`. Internal: left child node index.
    first: u32,
    /// Leaf: triangle count (>0). Internal: right child node index (first field is left).
    count_or_right: u32,
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        // We ensure internal nodes always have count_or_right set to a node index
        // and leaves always have count > 0. We use a sentinel: internal nodes are
        // flagged by having the high bit set on count_or_right.
        self.count_or_right & (1 << 31) == 0
    }
}

pub struct CollisionMesh {
    /// Boundary triangles: [v0, v1, v2] per triangle.
    triangles: Vec<[Vec3; 3]>,
    /// Per-triangle outward normal.
    normals: Vec<Vec3>,
    /// BVH nodes (flat array).
    bvh: Vec<BvhNode>,
    /// Triangle indices ordered by BVH traversal.
    tri_order: Vec<u32>,
}

impl CollisionMesh {
    /// Extract the boundary surface of the dense region and build a BVH.
    pub fn build(scene_data: &rmesh_data::SceneData, threshold: f32) -> Self {
        let tet_count = scene_data.tet_count as usize;
        let verts = &scene_data.vertices;
        let idxs = &scene_data.indices;

        // Collect faces from dense tets. Key = sorted (i, j, k).
        // Value = (tet_index, face_index, count).
        let mut face_map: HashMap<[u32; 3], Vec<(usize, usize)>> = HashMap::new();

        for t in 0..tet_count {
            if scene_data.densities[t] < threshold {
                continue;
            }
            let ti = [
                idxs[t * 4],
                idxs[t * 4 + 1],
                idxs[t * 4 + 2],
                idxs[t * 4 + 3],
            ];
            for (fi, face) in TET_FACES.iter().enumerate() {
                let mut key = [ti[face[0]], ti[face[1]], ti[face[2]]];
                key.sort_unstable();
                face_map.entry(key).or_default().push((t, fi));
            }
        }

        // Boundary faces appear exactly once (not shared by two dense tets).
        let mut triangles = Vec::new();
        let mut normals = Vec::new();

        for entries in face_map.values() {
            if entries.len() != 1 {
                continue; // interior face (shared by two dense tets)
            }
            let (t, fi) = entries[0];
            let ti = [
                idxs[t * 4],
                idxs[t * 4 + 1],
                idxs[t * 4 + 2],
                idxs[t * 4 + 3],
            ];
            let face = &TET_FACES[fi];

            let v0 = v3_from(verts, ti[face[0]] as usize);
            let v1 = v3_from(verts, ti[face[1]] as usize);
            let v2 = v3_from(verts, ti[face[2]] as usize);

            let normal = (v1 - v0).cross(v2 - v0);
            let len = normal.length();
            if len < 1e-12 {
                continue; // degenerate face
            }

            // Orient outward: normal should point away from the opposite vertex
            let opp = v3_from(verts, ti[face[3]] as usize);
            let center = (v0 + v1 + v2) / 3.0;
            let outward = if normal.dot(center - opp) >= 0.0 {
                normal / len
            } else {
                -normal / len
            };

            triangles.push([v0, v1, v2]);
            normals.push(outward);
        }

        log::info!(
            "Collision mesh: {} boundary triangles from {} dense tets (threshold={:.3})",
            triangles.len(),
            face_map.len(),
            threshold,
        );

        // Build BVH
        let (bvh, tri_order) = build_bvh(&triangles);

        Self {
            triangles,
            normals,
            bvh,
            tri_order,
        }
    }

    /// Ray-segment intersection. Returns the closest hit distance t ∈ (0, t_max], or None.
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3, t_max: f32) -> Option<f32> {
        if self.bvh.is_empty() {
            return None;
        }

        let inv_dir = Vec3::new(
            if dir.x.abs() > 1e-20 {
                1.0 / dir.x
            } else {
                f32::copysign(1e20, dir.x)
            },
            if dir.y.abs() > 1e-20 {
                1.0 / dir.y
            } else {
                f32::copysign(1e20, dir.y)
            },
            if dir.z.abs() > 1e-20 {
                1.0 / dir.z
            } else {
                f32::copysign(1e20, dir.z)
            },
        );

        let mut closest = t_max;
        let mut hit = false;
        let mut stack = [0u32; 64];
        stack[0] = 0;
        let mut sp = 1;

        while sp > 0 {
            sp -= 1;
            let node_idx = stack[sp] as usize;
            let node = &self.bvh[node_idx];

            if !ray_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, closest) {
                continue;
            }

            if node.is_leaf() {
                let start = node.first as usize;
                let count = node.count_or_right as usize;
                for i in start..start + count {
                    let ti = self.tri_order[i] as usize;
                    let [v0, v1, v2] = self.triangles[ti];
                    if let Some(t) = ray_triangle(origin, dir, v0, v1, v2) {
                        if t > 0.0 && t <= closest {
                            closest = t;
                            hit = true;
                        }
                    }
                }
            } else {
                let left = node.first;
                let right = node.count_or_right & !(1 << 31);
                if sp + 2 <= stack.len() {
                    stack[sp] = left;
                    sp += 1;
                    stack[sp] = right;
                    sp += 1;
                }
            }
        }

        if hit {
            Some(closest)
        } else {
            None
        }
    }

    /// Convert to PrimitiveVertex triangles for debug visualization.
    pub fn to_debug_vertices(&self) -> Vec<rmesh_compositor::PrimitiveVertex> {
        let mut out = Vec::with_capacity(self.triangles.len() * 3);
        for (i, &[v0, v1, v2]) in self.triangles.iter().enumerate() {
            let n = self.normals[i];
            for v in [v0, v1, v2] {
                out.push(rmesh_compositor::PrimitiveVertex {
                    position: v.to_array(),
                    normal: n.to_array(),
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
            }
        }
        out
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

fn v3_from(verts: &[f32], vi: usize) -> Vec3 {
    Vec3::new(verts[vi * 3], verts[vi * 3 + 1], verts[vi * 3 + 2])
}

// -----------------------------------------------------------------------
// BVH construction
// -----------------------------------------------------------------------

fn build_bvh(triangles: &[[Vec3; 3]]) -> (Vec<BvhNode>, Vec<u32>) {
    let n = triangles.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Compute centroids and AABBs per triangle
    let mut centroids: Vec<Vec3> = Vec::with_capacity(n);
    let mut tri_mins: Vec<Vec3> = Vec::with_capacity(n);
    let mut tri_maxs: Vec<Vec3> = Vec::with_capacity(n);
    for &[v0, v1, v2] in triangles {
        centroids.push((v0 + v1 + v2) / 3.0);
        tri_mins.push(v0.min(v1).min(v2));
        tri_maxs.push(v0.max(v1).max(v2));
    }

    let mut tri_order: Vec<u32> = (0..n as u32).collect();
    let mut nodes: Vec<BvhNode> = Vec::new();

    build_bvh_recursive(
        &mut nodes,
        &mut tri_order,
        &centroids,
        &tri_mins,
        &tri_maxs,
        0,
        n,
    );

    (nodes, tri_order)
}

fn build_bvh_recursive(
    nodes: &mut Vec<BvhNode>,
    tri_order: &mut [u32],
    centroids: &[Vec3],
    tri_mins: &[Vec3],
    tri_maxs: &[Vec3],
    start: usize,
    end: usize,
) -> u32 {
    let count = end - start;

    // Compute AABB of this range
    let mut aabb_min = Vec3::splat(f32::INFINITY);
    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);
    for i in start..end {
        let ti = tri_order[i] as usize;
        aabb_min = aabb_min.min(tri_mins[ti]);
        aabb_max = aabb_max.max(tri_maxs[ti]);
    }

    // Leaf
    if count <= BVH_MAX_LEAF {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb_min,
            aabb_max,
            first: start as u32,
            count_or_right: count as u32, // high bit clear → leaf
        });
        return node_idx;
    }

    // Find longest axis of centroid extent
    let mut cmin = Vec3::splat(f32::INFINITY);
    let mut cmax = Vec3::splat(f32::NEG_INFINITY);
    for i in start..end {
        let ti = tri_order[i] as usize;
        cmin = cmin.min(centroids[ti]);
        cmax = cmax.max(centroids[ti]);
    }
    let extent = cmax - cmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Sort by centroid along chosen axis, split at median
    let mid = start + count / 2;
    tri_order[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
        let ca = centroids[a as usize][axis];
        let cb = centroids[b as usize][axis];
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Allocate this node (placeholder, will fill children after recursion)
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        aabb_min,
        aabb_max,
        first: 0,
        count_or_right: 0,
    });

    let left = build_bvh_recursive(nodes, tri_order, centroids, tri_mins, tri_maxs, start, mid);
    let right = build_bvh_recursive(nodes, tri_order, centroids, tri_mins, tri_maxs, mid, end);

    // Fill in internal node: left child, right child with high bit set
    nodes[node_idx as usize].first = left;
    nodes[node_idx as usize].count_or_right = right | (1 << 31);

    node_idx
}

// -----------------------------------------------------------------------
// Ray intersection primitives
// -----------------------------------------------------------------------

/// Ray-AABB slab intersection test. Returns true if ray hits the box within [0, t_max].
fn ray_aabb(origin: Vec3, inv_dir: Vec3, aabb_min: Vec3, aabb_max: Vec3, t_max: f32) -> bool {
    let t1 = (aabb_min - origin) * inv_dir;
    let t2 = (aabb_max - origin) * inv_dir;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let t_enter = tmin.x.max(tmin.y).max(tmin.z).max(0.0);
    let t_exit = tmax.x.min(tmax.y).min(tmax.z).min(t_max);
    t_enter <= t_exit
}

/// Möller-Trumbore ray-triangle intersection. Returns t if hit.
fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-10 {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > 1e-6 {
        Some(t)
    } else {
        None
    }
}

// -----------------------------------------------------------------------
// Flare
// -----------------------------------------------------------------------

struct Flare {
    position: Vec3,
    velocity: Vec3,
    /// Normalized velocity direction (preserved when stuck, for model orientation).
    last_dir: Vec3,
    lifetime: f32,
    stuck: bool,
    /// Indices into the App's primitives vec (one per mesh in the model).
    primitive_indices: Vec<usize>,
}

// -----------------------------------------------------------------------
// FlareSystem
// -----------------------------------------------------------------------

pub struct FlareSystem {
    flares: Vec<Flare>,
    pub density_threshold: f32,
    pub force_dsm_recompute: bool,
    pub show_collision_mesh: bool,
    collision_mesh: Option<CollisionMesh>,
    last_built_threshold: f32,
    pub collision_mesh_dirty: bool,
    /// Number of custom meshes that belong to the flare model.
    pub flare_mesh_count: usize,
    /// Base custom mesh index for the flare model meshes.
    pub flare_mesh_base: usize,
    /// True when a flare model has been loaded.
    pub has_flare_model: bool,
    /// Material indices for each mesh in the flare model.
    pub flare_material_indices: Vec<usize>,
}

impl Default for FlareSystem {
    fn default() -> Self {
        Self {
            flares: Vec::new(),
            density_threshold: 0.05,
            force_dsm_recompute: false,
            show_collision_mesh: false,
            collision_mesh: None,
            last_built_threshold: -1.0,
            collision_mesh_dirty: false,
            flare_mesh_count: 0,
            flare_mesh_base: 0,
            has_flare_model: false,
            flare_material_indices: Vec::new(),
        }
    }
}

impl FlareSystem {
    pub fn shoot(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
        primitives: &mut Vec<Primitive>,
        next_id: &mut u32,
    ) {
        let flare_id = *next_id;
        *next_id += 1;

        let mesh_count = if self.has_flare_model {
            self.flare_mesh_count.max(1)
        } else {
            1
        };
        let mut indices = Vec::with_capacity(mesh_count);

        for mesh_i in 0..mesh_count {
            let name = format!("Flare.{:03}.{}", flare_id, mesh_i);
            let kind = if self.has_flare_model {
                PrimitiveKind::CustomMesh(self.flare_mesh_base + mesh_i)
            } else {
                PrimitiveKind::PointLight
            };

            let mut prim = Primitive::new(kind, name);
            prim.transform.position = camera_pos;
            prim.transform.scale = Vec3::splat(FLARE_MODEL_SCALE);
            prim.transform.rotation = dir_to_rotation(camera_forward);
            prim.color = Some([
                FLARE_COLOR[0],
                FLARE_COLOR[1],
                FLARE_COLOR[2],
                FLARE_INTENSITY,
            ]);

            if mesh_i < self.flare_material_indices.len() {
                prim.material_index = Some(self.flare_material_indices[mesh_i]);
            }

            indices.push(primitives.len());
            primitives.push(prim);
        }

        self.flares.push(Flare {
            position: camera_pos,
            velocity: camera_forward * INITIAL_SPEED,
            last_dir: camera_forward,
            lifetime: MAX_LIFETIME,
            stuck: false,
            primitive_indices: indices,
        });
    }

    /// Force-rebuild the collision mesh from the given scene. Call this right
    /// after a new scene is loaded so the BVH is cached up front, avoiding a
    /// hitch the first time a flare is fired or the collision mesh is shown.
    /// Also clears the stale mesh left over from a previously loaded scene.
    pub fn rebuild_collision_mesh(&mut self, scene_data: &rmesh_data::SceneData) {
        self.collision_mesh = None;
        self.last_built_threshold = -1.0;
        self.ensure_collision_mesh(scene_data);
    }

    /// Rebuild the collision mesh if the threshold changed or it hasn't been built yet.
    pub fn ensure_collision_mesh(&mut self, scene_data: &rmesh_data::SceneData) {
        if scene_data.tet_count == 0 {
            return;
        }
        if self.collision_mesh.is_some()
            && (self.last_built_threshold - self.density_threshold).abs() < 1e-6
        {
            return;
        }
        self.collision_mesh = Some(CollisionMesh::build(scene_data, self.density_threshold));
        self.last_built_threshold = self.density_threshold;
        self.collision_mesh_dirty = true;
    }

    pub fn tick(
        &mut self,
        dt: f32,
        primitives: &mut Vec<Primitive>,
        scene_data: &rmesh_data::SceneData,
    ) {
        if self.flares.is_empty() || dt <= 0.0 {
            return;
        }

        let dt = dt * TIME_SCALE;

        // Ensure collision mesh exists
        self.ensure_collision_mesh(scene_data);

        for flare in &mut self.flares {
            if !flare.stuck {
                let new_vel = flare.velocity + GRAVITY * dt;
                let new_pos = flare.position + new_vel * dt;

                // Track direction for model orientation
                if new_vel.length_squared() > 1e-8 {
                    flare.last_dir = new_vel.normalize();
                }

                // Ray-segment collision: test from old position toward new position
                let segment = new_pos - flare.position;
                let seg_len = segment.length();

                let hit = if seg_len > 1e-8 {
                    let dir = segment / seg_len;
                    self.collision_mesh
                        .as_ref()
                        .and_then(|mesh| mesh.ray_intersect(flare.position, dir, seg_len))
                } else {
                    None
                };

                if let Some(t) = hit {
                    let dir = segment / seg_len;
                    flare.position += dir * (t - FLARE_RADIUS).max(0.0);
                    flare.velocity = Vec3::ZERO;
                    flare.stuck = true;
                } else {
                    flare.position = new_pos;
                    flare.velocity = new_vel;
                }
            }

            // Decay lifetime
            flare.lifetime -= dt;

            // Update all primitive transforms and fade intensity
            let fade = if flare.lifetime < FADE_DURATION {
                (flare.lifetime / FADE_DURATION).max(0.0)
            } else {
                1.0
            };
            let intensity = FLARE_INTENSITY * fade;
            let rotation = dir_to_rotation(flare.last_dir);

            for &prim_idx in &flare.primitive_indices {
                if let Some(prim) = primitives.get_mut(prim_idx) {
                    prim.transform.position = flare.position;
                    prim.transform.rotation = rotation;
                    prim.color = Some([FLARE_COLOR[0], FLARE_COLOR[1], FLARE_COLOR[2], intensity]);
                }
            }
        }

        // Remove expired flares and their primitives.
        let mut to_remove: Vec<usize> = Vec::new();
        for (i, flare) in self.flares.iter().enumerate() {
            if flare.lifetime <= 0.0 {
                to_remove.push(i);
            }
        }

        to_remove.sort_unstable();
        for &flare_idx in to_remove.iter().rev() {
            // Remove primitives in reverse order to keep indices stable within this flare
            let mut prim_indices = self.flares[flare_idx].primitive_indices.clone();
            prim_indices.sort_unstable();
            for &prim_idx in prim_indices.iter().rev() {
                if prim_idx < primitives.len() {
                    primitives.remove(prim_idx);

                    // Adjust all other flare primitive indices
                    for f in &mut self.flares {
                        for idx in &mut f.primitive_indices {
                            if *idx > prim_idx {
                                *idx -= 1;
                            }
                        }
                    }
                }
            }

            self.flares.remove(flare_idx);
        }
    }

    pub fn clear(&mut self, primitives: &mut Vec<Primitive>) {
        let mut prim_indices: Vec<usize> = self
            .flares
            .iter()
            .flat_map(|f| f.primitive_indices.iter().copied())
            .collect();
        prim_indices.sort_unstable();
        prim_indices.dedup();
        for &idx in prim_indices.iter().rev() {
            if idx < primitives.len() {
                primitives.remove(idx);
            }
        }
        self.flares.clear();
    }

    pub fn has_moving_flares(&self) -> bool {
        self.flares.iter().any(|f| !f.stuck && f.lifetime > 0.0)
    }

    pub fn flare_count(&self) -> usize {
        self.flares.len()
    }

    /// Get debug vertices for the collision mesh (if built).
    pub fn collision_debug_vertices(&self) -> Option<Vec<rmesh_compositor::PrimitiveVertex>> {
        self.collision_mesh.as_ref().map(|m| m.to_debug_vertices())
    }

    pub fn collision_triangle_count(&self) -> usize {
        self.collision_mesh
            .as_ref()
            .map_or(0, |m| m.triangle_count())
    }

    /// Load the flare 3D model from a glTF file. Returns the converted mesh vertices
    /// and material data for upload to the GPU. Call this once at startup.
    pub fn load_model(&mut self, path: &Path) -> Option<rmesh_anim::gltf_loader::GltfScene> {
        match rmesh_anim::gltf_loader::load_gltf(path) {
            Ok(scene) => {
                self.flare_mesh_count = scene.meshes.len();
                self.has_flare_model = true;
                // Each mesh maps to a material by index (matching the glTF primitive order)
                self.flare_material_indices = (0..scene.meshes.len())
                    .map(|i| if i < scene.materials.len() { i } else { 0 })
                    .collect();
                log::info!(
                    "Loaded flare model: {} meshes, {} textures, {} materials",
                    scene.meshes.len(),
                    scene.textures.len(),
                    scene.materials.len(),
                );
                Some(scene)
            }
            Err(e) => {
                log::error!("Failed to load flare model from {:?}: {}", path, e);
                None
            }
        }
    }
}

/// Compute a rotation quaternion that orients the model's -Z axis (its long axis
/// in COLMAP coords after yup_to_colmap) along the given direction vector.
fn dir_to_rotation(dir: Vec3) -> Quat {
    if dir.length_squared() < 1e-8 {
        return Quat::IDENTITY;
    }
    let dir = dir.normalize();
    // Model long axis is -Z in COLMAP; flip to +Z so the head faces forward.
    let from = Vec3::Z;
    Quat::from_rotation_arc(from, dir)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal SceneData with the given tets.
    fn make_scene(tets: &[([Vec3; 4], f32)]) -> rmesh_data::SceneData {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut densities = Vec::new();

        for (i, (verts, density)) in tets.iter().enumerate() {
            let base = (i * 4) as u32;
            for v in verts {
                vertices.push(v.x);
                vertices.push(v.y);
                vertices.push(v.z);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 3]);
            densities.push(*density);
        }

        let tet_count = tets.len() as u32;
        let vertex_count = (tets.len() * 4) as u32;
        let circumdata = compute_test_circumdata(&vertices, &indices, tet_count as usize);

        rmesh_data::SceneData {
            vertices,
            indices,
            densities,
            color_grads: vec![0.0; tets.len() * 3],
            circumdata,
            start_pose: [0.0; 7],
            vertex_count,
            tet_count,
        }
    }

    fn compute_test_circumdata(vertices: &[f32], indices: &[u32], tet_count: usize) -> Vec<f32> {
        let mut circumdata = vec![0.0f32; tet_count * 4];
        for t in 0..tet_count {
            let i0 = indices[t * 4] as usize;
            let i1 = indices[t * 4 + 1] as usize;
            let i2 = indices[t * 4 + 2] as usize;
            let i3 = indices[t * 4 + 3] as usize;
            let v0 = glam::DVec3::new(
                vertices[i0 * 3] as f64,
                vertices[i0 * 3 + 1] as f64,
                vertices[i0 * 3 + 2] as f64,
            );
            let v1 = glam::DVec3::new(
                vertices[i1 * 3] as f64,
                vertices[i1 * 3 + 1] as f64,
                vertices[i1 * 3 + 2] as f64,
            );
            let v2 = glam::DVec3::new(
                vertices[i2 * 3] as f64,
                vertices[i2 * 3 + 1] as f64,
                vertices[i2 * 3 + 2] as f64,
            );
            let v3 = glam::DVec3::new(
                vertices[i3 * 3] as f64,
                vertices[i3 * 3 + 1] as f64,
                vertices[i3 * 3 + 2] as f64,
            );
            let a = v1 - v0;
            let b = v2 - v0;
            let c = v3 - v0;
            let denom = 2.0 * a.dot(b.cross(c));
            if denom.abs() < 1e-10 {
                let center = 0.25 * (v0 + v1 + v2 + v3);
                circumdata[t * 4] = center.x as f32;
                circumdata[t * 4 + 1] = center.y as f32;
                circumdata[t * 4 + 2] = center.z as f32;
                circumdata[t * 4 + 3] = 1e20;
            } else {
                let aa = a.dot(a);
                let bb = b.dot(b);
                let cc = c.dot(c);
                let rel = (aa * b.cross(c) + bb * c.cross(a) + cc * a.cross(b)) / denom;
                let center = v0 + rel;
                circumdata[t * 4] = center.x as f32;
                circumdata[t * 4 + 1] = center.y as f32;
                circumdata[t * 4 + 2] = center.z as f32;
                circumdata[t * 4 + 3] = rel.dot(rel) as f32;
            }
        }
        circumdata
    }

    fn unit_tet() -> [Vec3; 4] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    // -----------------------------------------------------------------------
    // CollisionMesh extraction
    // -----------------------------------------------------------------------

    #[test]
    fn collision_mesh_single_tet_has_4_faces() {
        let scene = make_scene(&[(unit_tet(), 1.0)]);
        let mesh = CollisionMesh::build(&scene, 0.1);
        // A single dense tet has 4 boundary faces (no shared faces)
        assert_eq!(mesh.triangle_count(), 4);
    }

    #[test]
    fn collision_mesh_skips_low_density() {
        let scene = make_scene(&[(unit_tet(), 0.05)]);
        let mesh = CollisionMesh::build(&scene, 0.1);
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn collision_mesh_two_adjacent_tets_share_face() {
        // Two tets sharing face (v0, v1, v2), different apex vertices.
        // They share vertices 0,1,2. Tet A has apex at (0,0,1), tet B at (0,0,-1).
        // Two tets sharing face (v0,v1,v2). Must use same vertex indices
        // for the shared face so the face map detects the match.
        let vertices = vec![
            0.0, 0.0, 0.0, // v0
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
            0.0, 0.0, 1.0, // v3 (apex A)
            0.0, 0.0, -1.0, // v4 (apex B)
        ];
        let indices = vec![
            0, 1, 2, 3, // tet A
            0, 1, 2, 4, // tet B — shares face (0,1,2) with tet A
        ];
        let densities = vec![1.0, 1.0];
        let circumdata = compute_test_circumdata(&vertices, &indices, 2);

        let scene = rmesh_data::SceneData {
            vertices,
            indices,
            densities,
            color_grads: vec![0.0; 6],
            circumdata,
            start_pose: [0.0; 7],
            vertex_count: 5,
            tet_count: 2,
        };

        let mesh = CollisionMesh::build(&scene, 0.1);
        // Each tet has 4 faces = 8 total. 1 face is shared → 8 - 2 = 6 boundary faces.
        assert_eq!(mesh.triangle_count(), 6);
    }

    // -----------------------------------------------------------------------
    // Ray-segment collision (the tunneling fix)
    // -----------------------------------------------------------------------

    #[test]
    fn ray_hits_single_tet_surface() {
        let scene = make_scene(&[(unit_tet(), 1.0)]);
        let mesh = CollisionMesh::build(&scene, 0.1);

        // Shoot ray from outside toward the tet
        let origin = Vec3::new(-1.0, 0.1, 0.1);
        let dir = Vec3::X; // +X
        let hit = mesh.ray_intersect(origin, dir, 10.0);
        assert!(hit.is_some(), "ray should hit the tet surface");
        let t = hit.unwrap();
        assert!(t > 0.9 && t < 1.1, "hit distance should be ~1.0, got {}", t);
    }

    #[test]
    fn ray_misses_tet() {
        let scene = make_scene(&[(unit_tet(), 1.0)]);
        let mesh = CollisionMesh::build(&scene, 0.1);

        // Shoot ray that clearly misses
        let origin = Vec3::new(-1.0, 5.0, 5.0);
        let dir = Vec3::X;
        let hit = mesh.ray_intersect(origin, dir, 10.0);
        assert!(hit.is_none(), "ray should miss");
    }

    #[test]
    fn ray_catches_thin_tet_surface() {
        // This is the tunneling test: a thin tet that the old point-check missed.
        let thin_tet = [
            Vec3::new(5.0, -0.5, -0.5),
            Vec3::new(5.02, -0.5, -0.5),
            Vec3::new(5.0, 0.5, -0.5),
            Vec3::new(5.0, 0.0, 0.5),
        ];
        let scene = make_scene(&[(thin_tet, 1.0)]);
        let mesh = CollisionMesh::build(&scene, 0.1);
        assert!(mesh.triangle_count() > 0, "should have boundary faces");

        // Ray from origin going +X, simulating a flare trajectory
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::X;
        let hit = mesh.ray_intersect(origin, dir, 10.0);
        assert!(
            hit.is_some(),
            "ray should catch the thin tet surface (tunneling fix)"
        );
    }

    // -----------------------------------------------------------------------
    // Full tick integration with ray-segment collision
    // -----------------------------------------------------------------------

    #[test]
    fn tick_flare_hits_small_tet_surface() {
        // Small tet that the old point-in-tet approach would tunnel through.
        // Placed at x=0.3 so gravity drift stays small even with TIME_SCALE.
        let small_tet = [
            Vec3::new(0.3, -0.5, -0.5),
            Vec3::new(0.32, -0.5, -0.5),
            Vec3::new(0.3, 0.5, -0.5),
            Vec3::new(0.3, 0.0, 0.5),
        ];
        let scene = make_scene(&[(small_tet, 1.0)]);

        let mut system = FlareSystem::default();
        system.density_threshold = 0.1;
        let mut primitives = Vec::new();
        let mut next_id = 0u32;

        // Shoot directly at it
        let cam_pos = Vec3::new(0.0, 0.0, 0.0);
        let cam_fwd = Vec3::X;
        system.shoot(cam_pos, cam_fwd, &mut primitives, &mut next_id);

        let dt = 1.0 / 60.0;
        let mut stuck = false;
        for _ in 0..6000 {
            system.tick(dt, &mut primitives, &scene);
            if system.flares.iter().any(|f| f.stuck) {
                stuck = true;
                break;
            }
        }

        assert!(
            stuck,
            "flare should hit the small tet surface (ray-segment collision)"
        );
    }

    #[test]
    fn tick_flare_hits_big_tet() {
        let big_tet = [
            Vec3::new(-5.0, -5.0, -5.0),
            Vec3::new(15.0, -5.0, -5.0),
            Vec3::new(-5.0, 15.0, -5.0),
            Vec3::new(-5.0, -5.0, 15.0),
        ];
        let scene = make_scene(&[(big_tet, 1.0)]);

        let mut system = FlareSystem::default();
        let mut primitives = Vec::new();
        let mut next_id = 0u32;

        system.shoot(
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::X,
            &mut primitives,
            &mut next_id,
        );

        let dt = 1.0 / 60.0;
        let mut stuck = false;
        for _ in 0..6000 {
            system.tick(dt, &mut primitives, &scene);
            if system.flares.iter().any(|f| f.stuck) {
                stuck = true;
                break;
            }
        }

        assert!(stuck, "flare should hit the big tet");
    }

    // -----------------------------------------------------------------------
    // BVH ray intersection primitives
    // -----------------------------------------------------------------------

    #[test]
    fn ray_triangle_basic() {
        let v0 = Vec3::ZERO;
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        // Hit from -Z side
        let hit = ray_triangle(Vec3::new(0.2, 0.2, -1.0), Vec3::Z, v0, v1, v2);
        assert!(hit.is_some());
        assert!((hit.unwrap() - 1.0).abs() < 1e-4);

        // Miss (ray parallel to triangle)
        let miss = ray_triangle(Vec3::new(0.2, 0.2, 0.0), Vec3::X, v0, v1, v2);
        assert!(miss.is_none());
    }
}
