// Binned surface-area heuristic: minimize overlapping boxes, rather than
// splitting the Morton sequence in half regardless of spatial distribution.
pub(super) fn partition(bounds: &mut [([f32; 3], [f32; 3])], faces: &mut [u32]) -> Option<usize> {
    const B: usize = 12;
    let area = |lo: [f32; 3], hi: [f32; 3]| {
        let d = std::array::from_fn::<_, 3, _>(|i| (hi[i] - lo[i]).max(0.));
        d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
    };
    let mut best = f32::INFINITY;
    let mut choice = None;
    for axis in 0..3 {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &(a, b) in bounds.iter() {
            let c = (a[axis] + b[axis]) * 0.5;
            lo = lo.min(c);
            hi = hi.max(c);
        }
        if hi - lo <= 1e-12 {
            continue;
        }
        let scale = B as f32 / (hi - lo);
        let bin = |a: [f32; 3], b: [f32; 3]| {
            ((((a[axis] + b[axis]) * 0.5 - lo) * scale) as usize).min(B - 1)
        };
        let mut counts = [0usize; B];
        let mut mins = [[f32::INFINITY; 3]; B];
        let mut maxs = [[f32::NEG_INFINITY; 3]; B];
        for &(a, b) in bounds.iter() {
            let i = bin(a, b);
            counts[i] += 1;
            for d in 0..3 {
                mins[i][d] = mins[i][d].min(a[d]);
                maxs[i][d] = maxs[i][d].max(b[d]);
            }
        }
        let mut right = [0f32; B];
        let mut rcount = [0usize; B];
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        let mut n = 0;
        for i in (0..B).rev() {
            n += counts[i];
            for d in 0..3 {
                mn[d] = mn[d].min(mins[i][d]);
                mx[d] = mx[d].max(maxs[i][d]);
            }
            right[i] = area(mn, mx) * n as f32;
            rcount[i] = n;
        }
        mn = [f32::INFINITY; 3];
        mx = [f32::NEG_INFINITY; 3];
        n = 0;
        for i in 0..B - 1 {
            n += counts[i];
            for d in 0..3 {
                mn[d] = mn[d].min(mins[i][d]);
                mx[d] = mx[d].max(maxs[i][d]);
            }
            if n == 0 || rcount[i + 1] == 0 {
                continue;
            }
            let cost = area(mn, mx) * n as f32 + right[i + 1];
            if cost < best {
                best = cost;
                choice = Some((axis, lo, scale, i));
            }
        }
    }
    let (axis, lo, scale, split) = choice?;
    let mut mid = 0;
    for i in 0..bounds.len() {
        let (a, b) = bounds[i];
        let bin = ((((a[axis] + b[axis]) * 0.5 - lo) * scale) as usize).min(B - 1);
        if bin <= split {
            bounds.swap(mid, i);
            faces.swap(mid, i);
            mid += 1;
        }
    }
    (mid > 0 && mid < bounds.len()).then_some(mid)
}

#[cfg(test)]
mod tests {
    #[test]
    fn hierarchy_preserves_faces_bounds_and_stack_limit() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for i in 0..512u32 {
            let x = (i % 16) as f32;
            let y = (i / 16) as f32;
            vertices.extend_from_slice(&[x, y, 0., x + 0.25, y, 0., x, y + 0.25, 0., x, y, 0.25]);
            indices.extend_from_slice(&[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]);
        }
        let neighbors = vec![-1; 2048];
        let original = crate::build_boundary_bvh(&vertices, &indices, &neighbors, 512);
        let optimized = crate::build_boundary_bvh_sah(&vertices, &indices, &neighbors, 512);
        let mut a = original.boundary_faces;
        let mut b = optimized.boundary_faces.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        let mut stack = vec![(0usize, 0)];
        let mut leaf_faces = Vec::new();
        while let Some((i, depth)) = stack.pop() {
            assert!(depth < 64);
            let node = &optimized.nodes[i];
            if node.left_or_face < 0 {
                let begin = (-node.left_or_face - 1) as usize;
                leaf_faces.extend_from_slice(
                    &optimized.boundary_faces[begin..begin + node.right_or_count as usize],
                );
            } else {
                for child in [node.left_or_face, node.right_or_count] {
                    let c = &optimized.nodes[child as usize];
                    for d in 0..3 {
                        assert!(c.aabb_min[d] >= node.aabb_min[d]);
                        assert!(c.aabb_max[d] <= node.aabb_max[d]);
                    }
                    stack.push((child as usize, depth + 1));
                }
            }
        }
        leaf_faces.sort_unstable();
        assert_eq!(leaf_faces, a);
    }
}
