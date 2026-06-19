//! CPU island construction: BFS expansion from user-selected handles, plus
//! constraint generation. Direct port of `initPBD` from DelTetRenderer's
//! `PBDMove.hpp`.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::mesh_topology::MeshTopology;

/// One particle in the simulation island.
///
/// `inv_mass == 0` means the particle is kinematic (handles) or pinned
/// (boundary). The solver leaves its position alone except for the explicit
/// per-frame handle override.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub global_index: u32,
    pub position: [f32; 3],
    pub inv_mass: f32,
}

/// XPBD edge-distance constraint, identifying particles by their *local* index
/// in the [`Island::particles`] vector.
#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    pub p1_local: u32,
    pub p2_local: u32,
    pub rest_length: f32,
    pub alpha: f32,
}

/// CPU representation of one PBD island, ready to be uploaded to the GPU.
pub struct Island {
    pub particles: Vec<Particle>,
    /// Local indices into [`Self::particles`] for the user-handle vertices,
    /// in the same order as the `handles` slice passed to [`build_island`].
    pub handle_local_indices: Vec<u32>,
    pub distance_constraints: Vec<DistanceConstraint>,
    /// Maps a scene-global vertex index to its local index in `particles`.
    pub global_to_local: HashMap<u32, u32>,
}

/// BFS from the handle vertices through `MeshTopology::adjacency`. Neighbors
/// within `radius` of any handle become active particles; the first neighbor
/// outside the radius becomes a boundary particle and stops the BFS there.
pub fn build_island(
    topology: &MeshTopology,
    indices: &[u32],
    vertices: &[f32],
    handles: &[u32],
    radius: f32,
) -> Island {
    if handles.is_empty() {
        return Island {
            particles: Vec::new(),
            handle_local_indices: Vec::new(),
            distance_constraints: Vec::new(),
            global_to_local: HashMap::new(),
        };
    }

    let vertex_count = topology.adjacency.len();
    let pos = |i: u32| {
        let b = i as usize * 3;
        [vertices[b], vertices[b + 1], vertices[b + 2]]
    };
    let dist = |a: [f32; 3], b: [f32; 3]| {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let radius_sq = radius * radius;

    // Spatial hash of handle positions; cell size = radius. The 3×3×3
    // neighborhood around any vertex's cell covers all handles that could be
    // within `radius`.
    let cell_size = radius.max(1e-6);
    let cell_of = |p: [f32; 3]| -> [i32; 3] {
        [
            (p[0] / cell_size).floor() as i32,
            (p[1] / cell_size).floor() as i32,
            (p[2] / cell_size).floor() as i32,
        ]
    };
    let mut handle_grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for &h in handles {
        handle_grid.entry(cell_of(pos(h))).or_default().push(h);
    }

    let mut visited = vec![false; vertex_count];
    let mut active: HashSet<u32> = HashSet::new();
    let mut boundary: HashSet<u32> = HashSet::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &h in handles {
        if !visited[h as usize] {
            visited[h as usize] = true;
            queue.push_back(h);
        }
    }

    while let Some(current) = queue.pop_front() {
        for &n in &topology.adjacency[current as usize] {
            if visited[n as usize] {
                continue;
            }
            let np = pos(n);
            let cc = cell_of(np);
            let mut inside = false;
            'cells: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = [cc[0] + dx, cc[1] + dy, cc[2] + dz];
                        if let Some(bucket) = handle_grid.get(&key) {
                            for &h in bucket {
                                let dp = pos(h);
                                let d = {
                                    let ddx = np[0] - dp[0];
                                    let ddy = np[1] - dp[1];
                                    let ddz = np[2] - dp[2];
                                    ddx * ddx + ddy * ddy + ddz * ddz
                                };
                                if d < radius_sq {
                                    inside = true;
                                    break 'cells;
                                }
                            }
                        }
                    }
                }
            }
            visited[n as usize] = true;
            if inside {
                active.insert(n);
                queue.push_back(n);
            } else {
                boundary.insert(n);
            }
        }
    }

    // Allocate particles. Order: sorted by global index for determinism.
    let handle_set: HashSet<u32> = handles.iter().copied().collect();
    let mut sim_vertices: Vec<u32> = active
        .iter()
        .chain(boundary.iter())
        .chain(handles.iter())
        .copied()
        .collect();
    sim_vertices.sort_unstable();
    sim_vertices.dedup();

    let mut particles: Vec<Particle> = Vec::with_capacity(sim_vertices.len());
    let mut global_to_local: HashMap<u32, u32> = HashMap::with_capacity(sim_vertices.len());
    for &gi in &sim_vertices {
        let li = particles.len() as u32;
        global_to_local.insert(gi, li);
        let inv_mass = if handle_set.contains(&gi) || boundary.contains(&gi) {
            0.0
        } else {
            1.0
        };
        particles.push(Particle {
            global_index: gi,
            position: pos(gi),
            inv_mass,
        });
    }

    let handle_local_indices: Vec<u32> = handles
        .iter()
        .filter_map(|h| global_to_local.get(h).copied())
        .collect();

    // Build distance constraints: one per unique edge of any tet whose 4
    // vertices are all in the island.
    let mut distance_constraints: Vec<DistanceConstraint> = Vec::new();
    let mut existing_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut processed_tets: HashSet<u32> = HashSet::new();

    for &gi in &sim_vertices {
        let Some(tets) = topology.vertex_to_tets.get(gi as usize) else {
            continue;
        };
        for &tet in tets {
            if !processed_tets.insert(tet) {
                continue;
            }
            let base = tet as usize * 4;
            let v = [
                indices[base],
                indices[base + 1],
                indices[base + 2],
                indices[base + 3],
            ];
            if !v.iter().all(|gv| global_to_local.contains_key(gv)) {
                continue;
            }
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let (mut u, mut w) = (v[i], v[j]);
                    if u > w {
                        std::mem::swap(&mut u, &mut w);
                    }
                    if !existing_edges.insert((u, w)) {
                        continue;
                    }
                    let pu = pos(u);
                    let pw = pos(w);
                    let rest_length = dist(pu, pw);
                    // Compliance follows PBDMove.hpp:237–248: near a handle
                    // (min half-sum distance < 0.05) use 0.0001 (stiff), else 0.01.
                    let mut k_x = f32::MAX;
                    for &h in handles {
                        let hp = pos(h);
                        let k_xi = 0.5 * (dist(pu, hp) + dist(pw, hp));
                        if k_xi < k_x {
                            k_x = k_xi;
                        }
                    }
                    let alpha = if k_x < 0.05 { 0.0001 } else { 0.01 };
                    distance_constraints.push(DistanceConstraint {
                        p1_local: global_to_local[&u],
                        p2_local: global_to_local[&w],
                        rest_length,
                        alpha,
                    });
                }
            }
        }
    }

    Island {
        particles,
        handle_local_indices,
        distance_constraints,
        global_to_local,
    }
}
