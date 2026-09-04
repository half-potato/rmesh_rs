//! View-independent importance cull.
//!
//! A tet's weight on a ray is `w = α · T`, with `T` the transmittance the ray
//! arrives with. From any camera in free space, a ray reaching tet `t` has
//! crossed at least `d_t` of optical depth — the least any path through the
//! mesh accumulates between free space and `t` — so `T ≤ exp(−d_t)` there, and
//! `w ≤ α_max(t) · exp(−d_t)` with `α_max` the tet's opacity along its longest
//! edge. Tets whose bound falls under a threshold contribute at most that much
//! to any pixel from any free-space viewpoint, so they can be left out of every
//! frame: no sort, no interval, no fragments. Computed once per scene.
//!
//! Measured against the per-pixel oracle (`RMESH_ORACLE`), the bound holds
//! (`w_max` exceeds it by more than 0.003 on 26–534 of 6.5M tets, all grazing
//! rays through big tets). Where the cut comes from depends on the scene: in
//! stump and bicycle the dense set is only a few tets thick, so almost every
//! culled tet goes on `α_max` alone (which `project_compute_hw.wgsl` had no
//! cull for at all); in room, with real walls, 72% of the culled tets go on the
//! distance term. Numbers per threshold are on `ShellSystem::default`.
//!
//! This is the view-independent half of the story. The oracle puts the
//! per-pixel floor at ~25 fragments/px for 1% error and shows a view-dependent
//! importance mask (per-tet `max w` from a low-res ray march) reaching ~50/px;
//! this bound is what can be had without knowing the camera.

use glam::Vec3;
use rayon::prelude::*;

/// Per-tet `density · longest_edge`, the quantity `project_compute` culls on.
pub fn tet_max_od(verts: &[f32], idxs: &[u32], dens: &[f32], tet_count: usize) -> Vec<f32> {
    (0..tet_count)
        .into_par_iter()
        .map(|t| {
            let p = |k: usize| {
                let b = idxs[t * 4 + k] as usize * 3;
                Vec3::new(verts[b], verts[b + 1], verts[b + 2])
            };
            let v = [p(0), p(1), p(2), p(3)];
            let mut m = 0.0f32;
            for i in 0..4 {
                for j in i + 1..4 {
                    m = m.max(v[i].distance_squared(v[j]));
                }
            }
            dens[t] * m.sqrt()
        })
        .collect()
}

/// Per tet, `d_t`: the least optical depth any path through the mesh
/// accumulates between free space and the tet (node-weighted Dijkstra over
/// the face adjacency, tet cost `od[t]`).
///
/// A ray from a camera in free space that reaches tet `t` has crossed at least
/// `d_t` of optical depth — provided `od[t]` under-estimates what any ray
/// crossing `t` picks up — so `T ≤ exp(−d_t)` there and the tet's weight on
/// that ray is at most `α_t · exp(−d_t)`. That bounds the oracle's `w_max`
/// from above for every free-space viewpoint at once. With the mean chord as
/// `od` the bound is only statistical (a grazing ray sees less); the caller
/// scales the chord down to trade tightness for how often it holds.
///
/// Seeds are tets whose own opacity is negligible (`α < 0.005`) and hull tets
/// (a face with no neighbour), both at distance zero.
pub fn optical_distance(
    neighbors: &[i32],
    od: &[f32],
    alphas: &[f32],
    tet_count: usize,
) -> Vec<f32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct Key(f32);
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Key {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0
                .partial_cmp(&o.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    const ALPHA_FREE: f32 = 0.005;
    let mut dist = vec![f32::INFINITY; tet_count];
    let mut heap = BinaryHeap::new();
    for t in 0..tet_count {
        let hull = (0..4).any(|f| neighbors[t * 4 + f] < 0);
        if alphas[t] < ALPHA_FREE || hull {
            dist[t] = 0.0;
        }
    }
    // Only seeds bordering a non-seed can relax anything.
    for t in 0..tet_count {
        if dist[t] == 0.0
            && (0..4).any(|f| {
                let nb = neighbors[t * 4 + f];
                nb >= 0 && dist[nb as usize] > 0.0
            })
        {
            heap.push(Reverse((Key(0.0), t as u32)));
        }
    }
    while let Some(Reverse((Key(d), t))) = heap.pop() {
        let t = t as usize;
        if d > dist[t] {
            continue;
        }
        let nd = d + od[t];
        for f in 0..4 {
            let nb = neighbors[t * 4 + f];
            if nb < 0 {
                continue;
            }
            let nb = nb as usize;
            if nd < dist[nb] {
                dist[nb] = nd;
                heap.push(Reverse((Key(nd), nb as u32)));
            }
        }
    }
    dist
}

/// Fraction of the mean chord used as a tet's path cost in the distance
/// transform. A grazing ray sees far less than the mean chord, so this trades
/// how tight the bound is for how often it holds; 0.5 was the measured knee.
pub const CHORD_SCALE: f32 = 0.5;

/// Per-tet weight bound `α_max · exp(−d)`. `alphas` are mean-chord opacities
/// (the shell's), `max_od` is `density · longest_edge`.
pub fn weight_bound(
    neighbors: &[i32],
    alphas: &[f32],
    max_od: &[f32],
    tet_count: usize,
) -> Vec<f32> {
    let od_path: Vec<f32> = alphas
        .iter()
        .map(|&a| -(1.0 - a.min(0.999_999)).ln() * CHORD_SCALE)
        .collect();
    let d = optical_distance(neighbors, &od_path, alphas, tet_count);
    (0..tet_count)
        .into_par_iter()
        .map(|t| (1.0 - (-max_od[t]).exp()) * (-d[t]).exp())
        .collect()
}

/// Bit-packed keep mask (`bit t` set ⇒ draw) for `w_bound >= threshold`, with
/// the kept count and how many tets the distance term removed beyond what the
/// opacity factor alone would have.
pub fn build_mask(
    w_bound: &[f32],
    max_od: &[f32],
    threshold: f32,
    tet_count: usize,
) -> (Vec<u32>, usize, usize) {
    let mut mask = vec![0u32; tet_count.div_ceil(32)];
    let (mut kept, mut by_distance) = (0usize, 0usize);
    for t in 0..tet_count {
        if w_bound[t] >= threshold {
            mask[t >> 5] |= 1u32 << (t & 31);
            kept += 1;
        } else if 1.0 - (-max_od[t]).exp() >= threshold {
            by_distance += 1;
        }
    }
    (mask, kept, by_distance)
}
