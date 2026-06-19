//! Greedy edge coloring for parallel Gauss-Seidel XPBD.
//!
//! Two distance constraints can be solved in the same GPU dispatch *iff* they
//! share no vertex (otherwise their writes to `particles[i].predicted` race).
//! We greedy-color the constraint set so each color is a vertex-disjoint
//! matching, then run one dispatch per color per solver iteration.

use crate::island::DistanceConstraint;

/// A reordering of `constraints` into vertex-disjoint color batches.
pub struct ConstraintColoring {
    /// Constraints sorted by color. Each contiguous slice is one color.
    pub constraints: Vec<DistanceConstraint>,
    /// `color_offsets[c] .. color_offsets[c+1]` indexes color `c`'s slice in
    /// `constraints`. `color_offsets.len() == num_colors + 1`.
    pub color_offsets: Vec<u32>,
}

impl ConstraintColoring {
    pub fn num_colors(&self) -> usize {
        self.color_offsets.len().saturating_sub(1)
    }

    pub fn color_len(&self, c: usize) -> u32 {
        self.color_offsets[c + 1] - self.color_offsets[c]
    }
}

/// Greedy first-fit edge coloring. Each constraint picks the smallest color
/// not yet used by either of its endpoint vertices in any earlier constraint.
pub fn color_constraints(
    particle_count: usize,
    constraints: &[DistanceConstraint],
) -> ConstraintColoring {
    if constraints.is_empty() {
        return ConstraintColoring {
            constraints: Vec::new(),
            color_offsets: vec![0],
        };
    }

    let mut vertex_colors: Vec<Vec<bool>> = vec![Vec::new(); particle_count];
    let mut assigned: Vec<u32> = Vec::with_capacity(constraints.len());
    let mut num_colors: u32 = 0;

    for c in constraints {
        let v1 = c.p1_local as usize;
        let v2 = c.p2_local as usize;
        // Find the smallest color used by neither endpoint.
        let mut color = 0u32;
        loop {
            let used1 = vertex_colors[v1].get(color as usize).copied().unwrap_or(false);
            let used2 = vertex_colors[v2].get(color as usize).copied().unwrap_or(false);
            if !used1 && !used2 {
                break;
            }
            color += 1;
        }
        // Mark color used by both endpoints; grow the row as needed.
        for v in [v1, v2] {
            if vertex_colors[v].len() <= color as usize {
                vertex_colors[v].resize(color as usize + 1, false);
            }
            vertex_colors[v][color as usize] = true;
        }
        assigned.push(color);
        if color + 1 > num_colors {
            num_colors = color + 1;
        }
    }

    // Bucket constraints by color and concatenate.
    let nc = num_colors as usize;
    let mut buckets: Vec<Vec<DistanceConstraint>> = vec![Vec::new(); nc];
    for (i, &c) in assigned.iter().enumerate() {
        buckets[c as usize].push(constraints[i]);
    }
    let mut out_constraints: Vec<DistanceConstraint> = Vec::with_capacity(constraints.len());
    let mut color_offsets: Vec<u32> = Vec::with_capacity(nc + 1);
    color_offsets.push(0);
    for bucket in buckets {
        out_constraints.extend(bucket);
        color_offsets.push(out_constraints.len() as u32);
    }

    ConstraintColoring {
        constraints: out_constraints,
        color_offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dc(a: u32, b: u32) -> DistanceConstraint {
        DistanceConstraint {
            p1_local: a,
            p2_local: b,
            rest_length: 1.0,
            alpha: 0.01,
        }
    }

    #[test]
    fn empty_input_returns_zero_colors() {
        let c = color_constraints(0, &[]);
        assert_eq!(c.num_colors(), 0);
        assert_eq!(c.color_offsets, vec![0]);
    }

    #[test]
    fn disjoint_edges_get_one_color() {
        // (0-1), (2-3), (4-5) share no vertices — all fit in one color.
        let c = color_constraints(6, &[dc(0, 1), dc(2, 3), dc(4, 5)]);
        assert_eq!(c.num_colors(), 1);
        assert_eq!(c.color_len(0), 3);
    }

    #[test]
    fn star_of_n_needs_n_colors() {
        // Center vertex 0 connected to 1, 2, 3, 4 — every edge shares vertex 0,
        // so each needs its own color.
        let c = color_constraints(5, &[dc(0, 1), dc(0, 2), dc(0, 3), dc(0, 4)]);
        assert_eq!(c.num_colors(), 4);
        for i in 0..4 {
            assert_eq!(c.color_len(i), 1);
        }
    }

    #[test]
    fn within_a_color_endpoints_are_disjoint() {
        // Tet 0-1-2-3 has 6 edges; greedy coloring should produce 3 colors
        // (each color = a perfect matching of the K4).
        let edges = [
            dc(0, 1), dc(0, 2), dc(0, 3),
            dc(1, 2), dc(1, 3), dc(2, 3),
        ];
        let c = color_constraints(4, &edges);
        for ci in 0..c.num_colors() {
            let start = c.color_offsets[ci] as usize;
            let end = c.color_offsets[ci + 1] as usize;
            let mut seen = std::collections::HashSet::new();
            for con in &c.constraints[start..end] {
                assert!(seen.insert(con.p1_local), "duplicate vertex in color");
                assert!(seen.insert(con.p2_local), "duplicate vertex in color");
            }
        }
    }
}
