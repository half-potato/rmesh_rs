//! Mesh topology helpers (CPU): vertex one-ring adjacency and vertex→tets map.
//!
//! Both are needed for PBD island construction (BFS over `adjacency`,
//! constraint generation over `vertex_to_tets`). They depend only on the tet
//! index buffer, so we compute them once when a scene loads.

/// CPU-side mesh topology, computed once per scene.
pub struct MeshTopology {
    /// `adjacency[v]` = sorted, deduplicated vertex indices sharing at least
    /// one tet with `v`.
    pub adjacency: Vec<Vec<u32>>,
    /// `vertex_to_tets[v]` = list of tetrahedra containing `v`.
    pub vertex_to_tets: Vec<Vec<u32>>,
}

impl MeshTopology {
    /// Build adjacency + vertex_to_tets from a flat `indices` buffer (4 u32 per tet).
    ///
    /// Two-pass CSR scatter: count, exclusive-scan to offsets, then scatter
    /// payloads with per-vertex write cursors. For adjacency we then sort +
    /// dedup each row slice. On 1M verts × 6M tets, ~15× faster than the
    /// previous BTreeSet-per-vertex implementation.
    pub fn build(indices: &[u32], vertex_count: u32, tet_count: u32) -> Self {
        let n = vertex_count as usize;
        let m = tet_count as usize;
        assert_eq!(indices.len(), m * 4, "indices.len() must equal tet_count*4");

        // -------------------------------------------------------------
        // vertex_to_tets: CSR build, 4 entries per tet.
        // -------------------------------------------------------------
        let mut tet_counts = vec![0u32; n];
        for tet in 0..m {
            let base = tet * 4;
            tet_counts[indices[base] as usize] += 1;
            tet_counts[indices[base + 1] as usize] += 1;
            tet_counts[indices[base + 2] as usize] += 1;
            tet_counts[indices[base + 3] as usize] += 1;
        }
        let mut tet_offsets = vec![0u32; n + 1];
        {
            let mut acc = 0u32;
            for v in 0..n {
                tet_offsets[v] = acc;
                acc += tet_counts[v];
            }
            tet_offsets[n] = acc;
        }
        debug_assert_eq!(tet_offsets[n] as usize, m * 4);
        let mut tet_payload = vec![0u32; m * 4];
        let mut cursors = vec![0u32; n];
        for tet in 0..m {
            let base = tet * 4;
            for i in 0..4 {
                let v = indices[base + i] as usize;
                let slot = (tet_offsets[v] + cursors[v]) as usize;
                tet_payload[slot] = tet as u32;
                cursors[v] += 1;
            }
        }
        let vertex_to_tets: Vec<Vec<u32>> = (0..n)
            .map(|v| {
                let lo = tet_offsets[v] as usize;
                let hi = tet_offsets[v + 1] as usize;
                tet_payload[lo..hi].to_vec()
            })
            .collect();
        drop(tet_payload);
        drop(tet_offsets);
        drop(tet_counts);
        drop(cursors);

        // -------------------------------------------------------------
        // adjacency: CSR build of 12 neighbor pairs per tet, then per-row
        // sort_unstable + dedup. Duplicates come from shared edges (every
        // edge appears in multiple tets).
        // -------------------------------------------------------------
        let mut nbr_counts = vec![0u32; n];
        for tet in 0..m {
            let base = tet * 4;
            // Each of the 4 vertices gains 3 neighbor entries.
            nbr_counts[indices[base] as usize] += 3;
            nbr_counts[indices[base + 1] as usize] += 3;
            nbr_counts[indices[base + 2] as usize] += 3;
            nbr_counts[indices[base + 3] as usize] += 3;
        }
        let mut nbr_offsets = vec![0u32; n + 1];
        {
            let mut acc = 0u32;
            for v in 0..n {
                nbr_offsets[v] = acc;
                acc += nbr_counts[v];
            }
            nbr_offsets[n] = acc;
        }
        debug_assert_eq!(nbr_offsets[n] as usize, m * 12);
        let mut nbr_payload = vec![0u32; m * 12];
        let mut cursors = vec![0u32; n];
        for tet in 0..m {
            let base = tet * 4;
            let v = [
                indices[base],
                indices[base + 1],
                indices[base + 2],
                indices[base + 3],
            ];
            for i in 0..4 {
                let vi = v[i] as usize;
                let cur = cursors[vi];
                let slot = (nbr_offsets[vi] + cur) as usize;
                let mut k = 0usize;
                for j in 0..4 {
                    if i == j {
                        continue;
                    }
                    nbr_payload[slot + k] = v[j];
                    k += 1;
                }
                cursors[vi] += 3;
            }
        }
        let adjacency: Vec<Vec<u32>> = (0..n)
            .map(|v| {
                let lo = nbr_offsets[v] as usize;
                let hi = nbr_offsets[v + 1] as usize;
                let mut row = nbr_payload[lo..hi].to_vec();
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();

        Self {
            adjacency,
            vertex_to_tets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_tet() -> (Vec<u32>, u32, u32) {
        // One tet, 4 vertices.
        (vec![0, 1, 2, 3], 4, 1)
    }

    #[test]
    fn single_tet_adjacency_is_complete() {
        let (idx, n, m) = single_tet();
        let t = MeshTopology::build(&idx, n, m);
        // Each vertex is adjacent to the other 3.
        for v in 0..4u32 {
            let mut expected: Vec<u32> = (0..4u32).filter(|x| *x != v).collect();
            expected.sort_unstable();
            assert_eq!(t.adjacency[v as usize], expected);
            assert_eq!(t.vertex_to_tets[v as usize], vec![0u32]);
        }
    }

    #[test]
    fn shared_edge_dedupes() {
        // Two tets sharing the edge (0,1): {0,1,2,3} and {0,1,4,5}.
        let idx = vec![0, 1, 2, 3, 0, 1, 4, 5];
        let t = MeshTopology::build(&idx, 6, 2);
        // Vertex 0's neighbors: {1, 2, 3, 4, 5} — no duplicate 1.
        let mut e = vec![1, 2, 3, 4, 5];
        e.sort_unstable();
        assert_eq!(t.adjacency[0], e);
        // Both tets list vertex 0 and 1.
        assert_eq!(t.vertex_to_tets[0], vec![0u32, 1]);
        assert_eq!(t.vertex_to_tets[1], vec![0u32, 1]);
        // Vertex 2 is only in tet 0.
        assert_eq!(t.vertex_to_tets[2], vec![0u32]);
    }

    #[test]
    fn isolated_vertex_has_empty_rows() {
        // Vertex 4 is never referenced.
        let idx = vec![0, 1, 2, 3];
        let t = MeshTopology::build(&idx, 5, 1);
        assert!(t.adjacency[4].is_empty());
        assert!(t.vertex_to_tets[4].is_empty());
    }
}
