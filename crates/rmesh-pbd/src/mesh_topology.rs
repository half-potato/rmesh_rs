//! Mesh topology helpers (CPU): vertex one-ring adjacency and vertex→tets map.
//!
//! Both are needed for PBD island construction (BFS over `adjacency`,
//! constraint generation over `vertex_to_tets`). They depend only on the tet
//! index buffer, so we compute them once when a scene loads.

use std::collections::BTreeSet;

/// CPU-side mesh topology, computed once per scene.
pub struct MeshTopology {
    /// `adjacency[v]` = sorted set of vertex indices sharing at least one tet with `v`.
    pub adjacency: Vec<Vec<u32>>,
    /// `vertex_to_tets[v]` = list of tetrahedra containing `v`.
    pub vertex_to_tets: Vec<Vec<u32>>,
}

impl MeshTopology {
    /// Build adjacency + vertex_to_tets from a flat `indices` buffer (4 u32 per tet).
    pub fn build(indices: &[u32], vertex_count: u32, tet_count: u32) -> Self {
        let n = vertex_count as usize;
        let m = tet_count as usize;
        assert_eq!(indices.len(), m * 4, "indices.len() must equal tet_count*4");

        let mut adj_sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
        let mut vertex_to_tets: Vec<Vec<u32>> = vec![Vec::new(); n];
        for tet in 0..m {
            let v = [
                indices[tet * 4],
                indices[tet * 4 + 1],
                indices[tet * 4 + 2],
                indices[tet * 4 + 3],
            ];
            for &vi in &v {
                vertex_to_tets[vi as usize].push(tet as u32);
                for &vj in &v {
                    if vi != vj {
                        adj_sets[vi as usize].insert(vj);
                    }
                }
            }
        }
        let adjacency: Vec<Vec<u32>> = adj_sets.into_iter().map(|s| s.into_iter().collect()).collect();
        Self {
            adjacency,
            vertex_to_tets,
        }
    }
}
