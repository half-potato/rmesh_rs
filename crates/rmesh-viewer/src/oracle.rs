//! Overdraw oracle: how few fragments per pixel could this view get away with?
//!
//! The renderer composites every tet a ray crosses. At ~108 fragments per pixel
//! and a mean per-fragment alpha of ~0.11, most of that work lands on rays that
//! are already opaque. The question this module answers is what the floor is —
//! per pixel, exactly, for this view — and then how close a per-tet keep mask
//! (the only lever the renderer actually has) gets to it.
//!
//! # Theory
//!
//! Along one ray with fragments `i = 0..N` in depth order, alpha `α_i` and
//! premultiplied colour `c_i`, the pixel is
//!
//! ```text
//!   C = Σ w_i · c_i,   w_i = α_i · T_i,   T_i = Π_{j<i} (1 − α_j),   A = 1 − T_N
//! ```
//!
//! `w_i` is the fragment's *weight* — the share of the pixel it owns. Dropping
//! fragment `j` alone changes the pixel by exactly
//!
//! ```text
//!   ΔC_j = w_j · (B_j − c_j),   ΔA_j = w_j · (1 − A_j^behind)
//! ```
//!
//! where `B_j` is what the ray composites *behind* `j` (over transparent black)
//! and `A_j^behind` its opacity. So a fragment is free to drop when its weight
//! is small (`w_j → 0`: it is behind something opaque) **or** when it looks
//! like what is behind it (`B_j ≈ c_j`, `A^behind ≈ 1`: a homogeneous interior).
//!
//! The first kind needs no assumption about the volume: `w_j ≤ e^{−τ}` for any
//! tet whose optical distance from free space exceeds `τ`, so those tets can be
//! culled from every viewpoint at a bounded per-pixel error. The second kind is
//! only free *one at a time* — drop a whole homogeneous run and the earlier
//! fragments now see the background, so the error returns. Recovering that
//! overdraw needs the dropped optical depth put back somewhere (a coarser
//! mesh), which is an assumption about the volume being locally uniform.
//!
//! Hence the four per-pixel strategies measured here, each giving the minimum
//! fragment count `k(ε)` for a per-pixel L∞ error `ε` on (premultiplied rgb, α):
//!
//! * `ftb`  — front-to-back with a transmittance early-out. What a compute
//!            compositor gets for free. `k = #fragments before T ≤ ε`.
//! * `w`    — keep the top-k by weight `w_i`, drop the rest, recomposite.
//!            The "draw only the important tets" idea, taken literally.
//! * `cost` — keep the top-k by marginal drop cost `w_j·max(|B_j−c_j|, 1−A_j^behind)`.
//! * `comp` — top-k by weight, but each dropped fragment's optical depth is
//!            folded into the next kept one. The coarsening bound: what a
//!            locally-homogeneous volume could be simplified to.
//!
//! The per-pixel numbers are a lower bound on any per-tet scheme, since a tet
//! must be kept for every pixel that needs it. The sweep then applies real
//! per-tet masks — ranked by the oracle's own `w_max` / `cost_max`, and by the
//! view-independent optical-depth bound — to the GPU renderer and reports what
//! each actually buys in overdraw, fps and PSNR.
//!
//! Entry points: [`OracleConfig::from_env`] (`RMESH_ORACLE=<dir>`), and the
//! `pre_frame` / `post_frame` hooks called from the render loop.

use glam::{Mat3, Vec3};
use rayon::prelude::*;
use rmesh_util::shared::TET_FACE_INDICES;
use std::sync::atomic::{AtomicU32, Ordering};

/// Per-pixel error tolerances the `k(ε)` columns are evaluated at.
pub const EPS: [f32; 4] = [0.005, 0.01, 0.02, 0.05];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub struct OracleConfig {
    pub out_dir: String,
    /// Frame on which the oracle runs (lets the shell finish building first).
    pub at_frame: usize,
    /// Pixel stride for the per-pixel subset analysis. The walk and the
    /// per-tet importance always run at full resolution.
    pub stride: usize,
    /// Weight thresholds for the importance sweeps.
    pub thresholds: Vec<f32>,
    /// Frames rendered per sweep candidate (the first is excluded from timing).
    pub frames: usize,
    pub exit_when_done: bool,
}

impl OracleConfig {
    pub fn from_env() -> Option<Self> {
        let out_dir = std::env::var("RMESH_ORACLE").ok()?;
        let num = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let list = |k: &str, d: &str| -> Vec<f32> {
            std::env::var(k)
                .unwrap_or_else(|_| d.to_string())
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        };
        Some(Self {
            out_dir,
            at_frame: num("RMESH_ORACLE_FRAME", 40),
            stride: num("RMESH_ORACLE_STRIDE", 2).max(1),
            thresholds: list("RMESH_ORACLE_SWEEP", "0.1,0.03,0.01,0.003,0.001"),
            frames: num("RMESH_ORACLE_FRAMES", 20).max(2),
            exit_when_done: std::env::var("RMESH_ORACLE_EXIT").map_or(true, |v| v != "0"),
        })
    }
}

/// Pinhole camera in the renderer's convention (`c2w` columns = right, down,
/// forward; pixel `(px, py)` has row 0 at the top).
#[derive(Clone, Copy)]
pub struct CamParams {
    pub eye: Vec3,
    pub c2w: Mat3,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub w: u32,
    pub h: u32,
}

impl CamParams {
    fn ray_dir(&self, px: u32, py: u32) -> Vec3 {
        let d = Vec3::new(
            (px as f32 + 0.5 - self.cx) / self.fx,
            (py as f32 + 0.5 - self.cy) / self.fy,
            1.0,
        )
        .normalize();
        (self.c2w * d).normalize()
    }
}

// ---------------------------------------------------------------------------
// Ray walk
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Frag {
    tet: u32,
    alpha: f32,
    od: f32,
    /// α_i · T_i: the share of the pixel this fragment owns.
    w: f32,
}

/// Walk one ray through the mesh front-to-back, replicating `eval_tet_rt` in
/// `raytrace_compute.wgsl` (entry/exit planes, `od = clamp(ρ·dist, 1e-8, 88)`)
/// and the raster path's `max_od < 0.01` cull, so the fragment list is what
/// `interval_fragment.wgsl` shades for this pixel with the shell off.
/// Where rays start: at the camera's containing tet when it is inside the
/// mesh, else through the nearest boundary face. The boundary BVH (the same
/// one the GPU tracer uses) is needed either way, because pruned meshes have
/// voids a ray falls out of and must re-enter.
pub struct RayStart<'a> {
    pub inside: Option<u32>,
    pub bvh: &'a rmesh_raytrace::BVHData,
}

fn ray_tri(o: Vec3, d: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    let (e1, e2) = (v1 - v0, v2 - v0);
    let h = d.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-10 {
        return -1.0;
    }
    let f = 1.0 / a;
    let s = o - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return -1.0;
    }
    let q = s.cross(e1);
    let v = f * d.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return -1.0;
    }
    let t = f * e2.dot(q);
    if t < 0.0 {
        -1.0
    } else {
        t
    }
}

/// CPU port of `bvh_trace` in `raytrace_compute.wgsl`: nearest boundary-face
/// hit along the ray beyond `t_min`, as `(t, tet)`.
fn bvh_entry(
    bvh: &rmesh_raytrace::BVHData,
    verts: &[f32],
    idxs: &[u32],
    o: Vec3,
    d: Vec3,
    t_min: f32,
) -> Option<(f32, u32)> {
    if bvh.nodes.is_empty() {
        return None;
    }
    let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    let v3 = |vi: u32| {
        let b = vi as usize * 3;
        Vec3::new(verts[b], verts[b + 1], verts[b + 2])
    };
    let mut best = (f32::INFINITY, u32::MAX);
    let mut stack = vec![0u32];
    while let Some(ni) = stack.pop() {
        let n = &bvh.nodes[ni as usize];
        let t1 = (Vec3::from(n.aabb_min) - o) * inv;
        let t2 = (Vec3::from(n.aabb_max) - o) * inv;
        let lo = t1.min(t2);
        let hi = t1.max(t2);
        let t_enter = lo.x.max(lo.y).max(lo.z);
        let t_exit = hi.x.min(hi.y).min(hi.z);
        if !(t_enter <= t_exit && t_exit >= t_min && t_enter < best.0) {
            continue;
        }
        if n.left_or_face < 0 {
            let start = (-(n.left_or_face + 1)) as usize;
            for &packed in &bvh.boundary_faces[start..start + n.right_or_count as usize] {
                let tet = packed >> 2;
                let b = (packed & 3) as usize * 3;
                let t = ray_tri(
                    o,
                    d,
                    v3(idxs[tet as usize * 4 + TET_FACE_INDICES[b] as usize]),
                    v3(idxs[tet as usize * 4 + TET_FACE_INDICES[b + 1] as usize]),
                    v3(idxs[tet as usize * 4 + TET_FACE_INDICES[b + 2] as usize]),
                );
                if t > t_min && t < best.0 {
                    best = (t, tet);
                }
            }
        } else {
            stack.push(n.left_or_face as u32);
            stack.push(n.right_or_count as u32);
        }
    }
    (best.1 != u32::MAX).then_some(best)
}

#[allow(clippy::too_many_arguments)]
fn walk_ray(
    verts: &[f32],
    idxs: &[u32],
    dens: &[f32],
    max_od: &[f32],
    neighbors: &[i32],
    start: &RayStart,
    eye: Vec3,
    dir: Vec3,
    out: &mut Vec<Frag>,
) {
    const MAX_ITERS: usize = 16384;
    out.clear();
    // Pruned meshes are not solid: a ray can fall out of the tetrahedralized
    // region into a void and re-enter further on. The GPU tracer stops at the
    // first exit; the raster path does not care, so neither can we — on every
    // exit, look for the next boundary face along the ray.
    let reenter = |t_from: f32| bvh_entry(start.bvh, verts, idxs, eye, dir, t_from);
    let (mut cur, mut t_cursor) = match start.inside {
        Some(t) => (t as i64, 0.0f32),
        None => match reenter(0.0) {
            Some((t, tet)) => (tet as i64, t),
            None => return,
        },
    };
    let mut trans = 1.0f32;
    let v3 = |vi: u32| {
        let b = vi as usize * 3;
        Vec3::new(verts[b], verts[b + 1], verts[b + 2])
    };

    for _ in 0..MAX_ITERS {
        if cur < 0 {
            break;
        }
        let t = cur as usize;
        let vs = [
            v3(idxs[t * 4]),
            v3(idxs[t * 4 + 1]),
            v3(idxs[t * 4 + 2]),
            v3(idxs[t * 4 + 3]),
        ];
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;
        let mut exit_face = 4usize;
        let mut valid = true;
        for fi in 0..4 {
            let b = fi * 3;
            let va = vs[TET_FACE_INDICES[b] as usize];
            let vb = vs[TET_FACE_INDICES[b + 1] as usize];
            let vc = vs[TET_FACE_INDICES[b + 2] as usize];
            // Inward normal: toward the opposite vertex.
            let v_opp = vs[(0..4)
                .find(|&k| {
                    k != TET_FACE_INDICES[b] as usize
                        && k != TET_FACE_INDICES[b + 1] as usize
                        && k != TET_FACE_INDICES[b + 2] as usize
                })
                .unwrap()];
            let mut n = (vc - va).cross(vb - va);
            if n.dot(v_opp - va) < 0.0 {
                n = -n;
            }
            let num = n.dot(va - eye);
            let den = n.dot(dir);
            if den.abs() < 1e-20 {
                if num > 0.0 {
                    valid = false;
                }
                continue;
            }
            let tt = num / den;
            if den > 0.0 {
                if tt > t_min {
                    t_min = tt;
                }
            } else if tt < t_max {
                t_max = tt;
                exit_face = fi;
            }
        }
        t_min = t_min.max(t_cursor);
        if !valid || t_min >= t_max || exit_face == 4 {
            break;
        }

        if max_od[t] >= 0.01 {
            let od = (dens[t] * (t_max - t_min)).clamp(1e-8, 88.0);
            let alpha_t = (-od).exp();
            let alpha = 1.0 - alpha_t;
            out.push(Frag {
                tet: t as u32,
                alpha,
                od,
                w: alpha * trans,
            });
            trans *= alpha_t;
        }

        cur = neighbors[t * 4 + exit_face] as i64;
        t_cursor = t_max;
        if cur < 0 {
            // Fell out of the mesh; nudge past this face so the search does
            // not return it.
            let Some((t, tet)) = reenter(t_max * (1.0 + 1e-6) + 1e-6) else {
                break;
            };
            cur = tet as i64;
            t_cursor = t;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-pixel analysis
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
pub struct PixelStats {
    /// Fragments on the ray (shell off).
    pub n: u32,
    /// Final alpha.
    pub alpha: f32,
    pub k_ftb: [u16; 4],
    pub k_w: [u16; 4],
    pub k_cost: [u16; 4],
    pub k_comp: [u16; 4],
    /// L∞ error of keeping exactly `K_FIXED` fragments: [ftb, w, comp].
    pub e_fixed: [f32; 3],
    /// Same, as rgb squared error (for a PSNR-equivalent).
    pub se_fixed: [f32; 3],
}

/// The fragment budget the user asked about.
pub const K_FIXED: u32 = 10;

#[inline]
fn se_rgb(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>() / 3.0
}

/// L∞ distance between two (premultiplied rgb, alpha) composites.
#[inline]
fn err4(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..4).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max)
}

/// Recomposite the ray keeping fragments with `rank[i] < k`. With `compensate`,
/// a dropped fragment's optical depth is carried forward into the next kept
/// fragment (any left over at the end is lost, like the dropped fog it was).
fn composite_subset(
    frags: &[Frag],
    colors: &[Vec3],
    rank: &[u32],
    k: u32,
    compensate: bool,
) -> [f32; 4] {
    let mut trans = 1.0f32;
    let mut c = Vec3::ZERO;
    let mut pending = 0.0f32;
    for (i, f) in frags.iter().enumerate() {
        if rank[i] < k {
            let od = if compensate { f.od + pending } else { f.od };
            pending = 0.0;
            let alpha = if compensate {
                1.0 - (-od).exp()
            } else {
                f.alpha
            };
            c += colors[i] * (alpha * trans);
            trans *= 1.0 - alpha;
        } else if compensate {
            pending += f.od;
        }
    }
    [c.x, c.y, c.z, 1.0 - trans]
}

/// Smallest `k` whose subset composite is within `eps` of `full`, assuming the
/// error is monotone in `k` (true up to the odd lucky cancellation).
fn min_k(
    frags: &[Frag],
    colors: &[Vec3],
    rank: &[u32],
    full: [f32; 4],
    eps: f32,
    compensate: bool,
) -> u16 {
    let n = frags.len() as u32;
    // The background (k = 0) is a valid answer on near-empty rays.
    if err4(composite_subset(frags, colors, rank, 0, compensate), full) <= eps {
        return 0;
    }
    let (mut lo, mut hi) = (0u32, n); // err(lo) > eps, err(hi) <= eps
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if err4(composite_subset(frags, colors, rank, mid, compensate), full) <= eps {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi.min(u16::MAX as u32) as u16
}

fn rank_by_desc(scores: &[f32], order: &mut Vec<u32>, rank: &mut Vec<u32>) {
    let n = scores.len();
    order.clear();
    order.extend(0..n as u32);
    order.sort_unstable_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rank.clear();
    rank.resize(n, 0);
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u32;
    }
}

// ---------------------------------------------------------------------------
// Oracle run
// ---------------------------------------------------------------------------

pub struct Ranking {
    pub name: String,
    pub scores: Vec<f32>,
    pub thresholds: Vec<f32>,
    /// Keep when `score > threshold` (budgets) rather than `>=` (weights).
    pub strict: bool,
}

pub struct OracleOutput {
    pub stats: Vec<PixelStats>,
    pub stats_w: usize,
    pub stats_h: usize,
    /// Per tet: max over pixels of the fragment weight `w`.
    pub w_max: Vec<f32>,
    /// Per tet: `α_t · exp(−d_t)` with `d_t` the optical distance from free
    /// space (half mean chord) — the view-independent stand-in for `w_max`.
    pub w_bound: Vec<f32>,
    /// Every per-tet score the sweep will threshold, with its thresholds:
    /// * `w_max`, `cost_max` — max over pixels of the weight / marginal drop cost;
    /// * `w_max/4`, `+dil` — `w_max` seen by every 4th ray, with one ring of
    ///   adjacency dilation;
    /// * `drop_w`, `drop_cost` — the largest, over pixels, of the budget a
    ///   pixel would spend to lose the tet (its ascending cumulative weight
    ///   there). Dropping every tet with `drop_w <= ε` leaves every pixel with
    ///   at most `ε` of dropped weight — a per-tet mask with a per-pixel
    ///   guarantee, which a plain `w_max` threshold does not give;
    /// * `w_bound c{1,0.5,0.25}` — the optical-distance bound at three chord
    ///   scales.
    pub rankings: Vec<Ranking>,
    pub secs_walk: f64,
    pub secs_dt: f64,
}

/// Per-tet colour as the fragment shader composites it: PBR albedo when
/// present, else the SH DC term through the same `softplus(C0·sh0 + 0.5)`.
pub fn tet_colors(
    sh: &rmesh_data::ShCoeffs,
    pbr: Option<&rmesh_data::PbrData>,
    tet_count: usize,
) -> Vec<Vec3> {
    if let Some(p) = pbr {
        if p.albedo.len() >= tet_count * 3 {
            return (0..tet_count)
                .map(|t| Vec3::new(p.albedo[t * 3], p.albedo[t * 3 + 1], p.albedo[t * 3 + 2]))
                .collect();
        }
    }
    const C0: f32 = 0.282_094_8;
    let nc = sh.num_coeffs() as usize;
    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    (0..tet_count)
        .map(|t| {
            let b = t * nc * 3;
            let ch = |c: usize| sh.coeffs.get(b + c * nc).copied().unwrap_or(0.0);
            Vec3::new(
                softplus(C0 * ch(0) + 0.5),
                softplus(C0 * ch(1) + 0.5),
                softplus(C0 * ch(2) + 0.5),
            )
        })
        .collect()
}

fn atomic_max_f32(slot: &AtomicU32, v: f32) {
    // Non-negative floats order like their bit patterns.
    let bits = v.max(0.0).to_bits();
    let mut cur = slot.load(Ordering::Relaxed);
    while bits > cur {
        match slot.compare_exchange_weak(cur, bits, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(c) => cur = c,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_oracle(
    cam: CamParams,
    scene: &rmesh_data::SceneData,
    neighbors: &[i32],
    alphas: &[f32],
    colors: &[Vec3],
    start: &RayStart,
    stride: usize,
    thresholds: &[f32],
) -> OracleOutput {
    let tet_count = scene.tet_count as usize;
    let (verts, idxs, dens) = (
        &scene.vertices[..],
        &scene.indices[..],
        &scene.densities[..],
    );
    let max_od = crate::importance::tet_max_od(verts, idxs, dens, tet_count);

    let w_max: Vec<AtomicU32> = (0..tet_count).map(|_| AtomicU32::new(0)).collect();
    let cost_max: Vec<AtomicU32> = (0..tet_count).map(|_| AtomicU32::new(0)).collect();
    let drop_w: Vec<AtomicU32> = (0..tet_count).map(|_| AtomicU32::new(0)).collect();
    let drop_cost: Vec<AtomicU32> = (0..tet_count).map(|_| AtomicU32::new(0)).collect();
    // Importance seen only by every LOWRES-th ray in x and y: what a GPU
    // ray-march at reduced resolution could measure per frame.
    const LOWRES: u32 = 4;
    let w_max_lo: Vec<AtomicU32> = (0..tet_count).map(|_| AtomicU32::new(0)).collect();

    let sw = (cam.w as usize).div_ceil(stride);
    let sh = (cam.h as usize).div_ceil(stride);
    let mut stats = vec![PixelStats::default(); sw * sh];

    let t0 = std::time::Instant::now();
    stats.par_chunks_mut(sw).enumerate().for_each(|(sy, row)| {
        let mut frags: Vec<Frag> = Vec::with_capacity(512);
        let mut cols: Vec<Vec3> = Vec::with_capacity(512);
        let mut scores: Vec<f32> = Vec::with_capacity(512);
        let mut order: Vec<u32> = Vec::with_capacity(512);
        let mut rank: Vec<u32> = Vec::with_capacity(512);
        let mut behind_c: Vec<Vec3> = Vec::with_capacity(512);
        let mut behind_a: Vec<f32> = Vec::with_capacity(512);

        // Every pixel row in this stride band walks (for the per-tet
        // importance); only the band's stride pixel gets the subset analysis.
        for py in (sy * stride) as u32..((sy + 1) * stride).min(cam.h as usize) as u32 {
            for px in 0..cam.w {
                let dir = cam.ray_dir(px, py);
                walk_ray(
                    verts, idxs, dens, &max_od, neighbors, start, cam.eye, dir, &mut frags,
                );
                let n = frags.len();

                // Marginal drop cost needs what is composited behind each
                // fragment: one backward recurrence.
                cols.clear();
                cols.extend(frags.iter().map(|f| colors[f.tet as usize]));
                behind_c.clear();
                behind_c.resize(n, Vec3::ZERO);
                behind_a.clear();
                behind_a.resize(n, 0.0);
                let (mut bc, mut ba) = (Vec3::ZERO, 0.0f32);
                for i in (0..n).rev() {
                    behind_c[i] = bc;
                    behind_a[i] = ba;
                    let f = &frags[i];
                    bc = cols[i] * f.alpha + bc * (1.0 - f.alpha);
                    ba = f.alpha + ba * (1.0 - f.alpha);
                }
                scores.clear();
                for i in 0..n {
                    let f = &frags[i];
                    let contrast = (behind_c[i] - cols[i])
                        .abs()
                        .max_element()
                        .max(1.0 - behind_a[i]);
                    let cost = f.w * contrast;
                    scores.push(cost);
                    atomic_max_f32(&w_max[f.tet as usize], f.w);
                    atomic_max_f32(&cost_max[f.tet as usize], cost);
                    if px % LOWRES == 0 && py % LOWRES == 0 {
                        atomic_max_f32(&w_max_lo[f.tet as usize], f.w);
                    }
                }

                // Per-pixel ascending cumulative sums: the budget at which
                // this fragment (and all lighter ones) can go.
                let wts: Vec<f32> = frags.iter().map(|f| f.w).collect();
                rank_by_desc(&wts, &mut order, &mut rank);
                let mut cum = 0.0f32;
                for &i in order.iter().rev() {
                    cum += wts[i as usize];
                    atomic_max_f32(&drop_w[frags[i as usize].tet as usize], cum);
                }
                let mut order_c = Vec::new();
                let mut rank_c = Vec::new();
                rank_by_desc(&scores, &mut order_c, &mut rank_c);
                cum = 0.0;
                for &i in order_c.iter().rev() {
                    cum += scores[i as usize];
                    atomic_max_f32(&drop_cost[frags[i as usize].tet as usize], cum);
                }

                if px as usize % stride != 0 || py as usize % stride != 0 {
                    continue;
                }
                let sx = px as usize / stride;
                let full = composite_subset(&frags, &cols, &vec![0; n], u32::MAX, false);
                let mut ps = PixelStats {
                    n: n as u32,
                    alpha: full[3],
                    ..Default::default()
                };

                // ftb: fragments before T drops to eps (the tail is then ≤ eps).
                for (e, &eps) in EPS.iter().enumerate() {
                    let mut trans = 1.0f32;
                    let mut k = n;
                    for (i, f) in frags.iter().enumerate() {
                        trans *= 1.0 - f.alpha;
                        if trans <= eps {
                            k = i + 1;
                            break;
                        }
                    }
                    ps.k_ftb[e] = k.min(u16::MAX as usize) as u16;
                }

                // Top-k by weight (plain and compensated); `rank` is still the
                // weight ranking from above.
                for (e, &eps) in EPS.iter().enumerate() {
                    ps.k_w[e] = min_k(&frags, &cols, &rank, full, eps, false);
                    ps.k_comp[e] = min_k(&frags, &cols, &rank, full, eps, true);
                }
                // Exactly K_FIXED fragments: front-to-back, by weight, compensated.
                let depth_rank: Vec<u32> = (0..n as u32).collect();
                let fixed = [
                    composite_subset(&frags, &cols, &depth_rank, K_FIXED, false),
                    composite_subset(&frags, &cols, &rank, K_FIXED, false),
                    composite_subset(&frags, &cols, &rank, K_FIXED, true),
                ];
                for (j, f) in fixed.iter().enumerate() {
                    ps.e_fixed[j] = err4(*f, full);
                    ps.se_fixed[j] = se_rgb(*f, full);
                }
                // Top-k by marginal drop cost.
                for (e, &eps) in EPS.iter().enumerate() {
                    ps.k_cost[e] = min_k(&frags, &cols, &rank_c, full, eps, false);
                }
                row[sx] = ps;
            }
        }
    });
    let secs_walk = t0.elapsed().as_secs_f64();

    let w_max: Vec<f32> = w_max
        .into_iter()
        .map(|a| f32::from_bits(a.into_inner()))
        .collect();
    let cost_max: Vec<f32> = cost_max
        .into_iter()
        .map(|a| f32::from_bits(a.into_inner()))
        .collect();
    let drop_w: Vec<f32> = drop_w
        .into_iter()
        .map(|a| f32::from_bits(a.into_inner()))
        .collect();
    let drop_cost: Vec<f32> = drop_cost
        .into_iter()
        .map(|a| f32::from_bits(a.into_inner()))
        .collect();

    let w_max_lo: Vec<f32> = w_max_lo
        .into_iter()
        .map(|a| f32::from_bits(a.into_inner()))
        .collect();
    // One ring of face-adjacency dilation, to cover the tets the sparse rays
    // slipped between.
    let w_max_lo_dil: Vec<f32> = (0..tet_count)
        .into_par_iter()
        .map(|t| {
            (0..4).fold(w_max_lo[t], |m, f| {
                let nb = neighbors[t * 4 + f];
                if nb >= 0 {
                    m.max(w_max_lo[nb as usize])
                } else {
                    m
                }
            })
        })
        .collect();

    let t1 = std::time::Instant::now();
    // Tet opacity for the worst-case chord (the longest edge), for the bound's
    // own-alpha factor; the transform's path cost uses a fraction of the mean
    // chord, since a ray can cross a tet far more briefly than its mean chord.
    let alpha_max: Vec<f32> = max_od.iter().map(|&od| 1.0 - (-od).exp()).collect();
    let mut w_bounds = Vec::new();
    for &chord in &[1.0f32, 0.5, 0.25] {
        let od_path: Vec<f32> = alphas
            .iter()
            .map(|&a| -(1.0 - a.min(0.999_999)).ln() * chord)
            .collect();
        let d = crate::importance::optical_distance(neighbors, &od_path, alphas, tet_count);
        let wb: Vec<f32> = (0..tet_count)
            .map(|t| alpha_max[t] * (-d[t]).exp())
            .collect();
        let violations = (0..tet_count).filter(|&t| w_max[t] > wb[t] + 0.003).count();
        log::info!(
            "[oracle] w_bound chord×{chord}: w_max exceeds the bound by >0.003 on {violations} tets"
        );
        w_bounds.push((chord, wb));
    }
    let secs_dt = t1.elapsed().as_secs_f64();
    let w_bound = w_bounds[1].1.clone();

    let budgets: Vec<f32> = EPS.iter().copied().chain([0.1]).collect();
    let mut rankings = vec![
        Ranking {
            name: "w_max".into(),
            scores: w_max.clone(),
            thresholds: thresholds.to_vec(),
            strict: false,
        },
        Ranking {
            name: "cost_max".into(),
            scores: cost_max,
            thresholds: thresholds.to_vec(),
            strict: false,
        },
        Ranking {
            name: format!("w_max/{LOWRES}"),
            scores: w_max_lo,
            thresholds: thresholds.to_vec(),
            strict: false,
        },
        Ranking {
            name: format!("w_max/{LOWRES}+dil"),
            scores: w_max_lo_dil,
            thresholds: thresholds.to_vec(),
            strict: false,
        },
        Ranking {
            name: "drop_w".into(),
            scores: drop_w,
            thresholds: budgets.clone(),
            strict: true,
        },
        Ranking {
            name: "drop_cost".into(),
            scores: drop_cost,
            thresholds: budgets,
            strict: true,
        },
    ];
    for (chord, wb) in w_bounds {
        rankings.push(Ranking {
            name: format!("w_bound c{chord}"),
            scores: wb,
            thresholds: thresholds.to_vec(),
            strict: false,
        });
    }

    OracleOutput {
        stats,
        stats_w: sw,
        stats_h: sh,
        w_max,
        w_bound,
        rankings,
        secs_walk,
        secs_dt,
    }
}

// ---------------------------------------------------------------------------
// View-independent bound: optical distance from free space
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn quantiles(vals: &mut [f32]) -> (f32, f32, f32, f32, f32) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| vals[((vals.len() - 1) as f64 * p) as usize];
    let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    (mean as f32, q(0.5), q(0.9), q(0.99), *vals.last().unwrap())
}

/// Google Turbo, 5th-order polynomial fit.
fn turbo(x: f32) -> [u8; 3] {
    let x = x.clamp(0.0, 1.0);
    let r = 0.135_721_38
        + x * (4.615_392_6
            + x * (-42.660_32 + x * (132.131_08 + x * (-152.942_4 + x * 59.286_38))));
    let g = 0.091_402_61
        + x * (2.194_188_4
            + x * (4.842_966_6 + x * (-14.185_033 + x * (4.277_298_6 + x * 2.829_566))));
    let b = 0.106_673_3
        + x * (12.641_946 + x * (-60.582_05 + x * (110.362_77 + x * (-89.903_11 + x * 27.348_25))));
    [
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

fn write_heatmap(path: &str, w: usize, h: usize, vals: impl Fn(usize) -> f32, scale: f32) {
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, p) in img.pixels_mut().enumerate() {
        *p = image::Rgb(turbo(vals(i) / scale));
    }
    if let Err(e) = img.save(path) {
        log::error!("[oracle] {path}: {e}");
    }
}

pub fn report(out: &OracleOutput, dir: &str, tet_count: usize) {
    let st = &out.stats;
    let col = |f: &dyn Fn(&PixelStats) -> f32| -> Vec<f32> { st.iter().map(f).collect() };
    let mut lines = Vec::new();
    let mut row = |name: &str, mut v: Vec<f32>| {
        let (mean, p50, p90, p99, max) = quantiles(&mut v);
        lines.push(format!(
            "{name:<14} {mean:>8.1} {p50:>8.1} {p90:>8.1} {p99:>8.1} {max:>8.0}"
        ));
    };
    row("N (all)", col(&|p| p.n as f32));
    row("alpha", col(&|p| p.alpha));
    for (e, &eps) in EPS.iter().enumerate() {
        row(&format!("ftb  eps={eps}"), col(&|p| p.k_ftb[e] as f32));
        row(&format!("w    eps={eps}"), col(&|p| p.k_w[e] as f32));
        row(&format!("cost eps={eps}"), col(&|p| p.k_cost[e] as f32));
        row(&format!("comp eps={eps}"), col(&|p| p.k_comp[e] as f32));
    }
    let header = format!(
        "{:<14} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "per-pixel k", "mean", "p50", "p90", "p99", "max"
    );
    // Error of a hard budget of K_FIXED fragments per pixel.
    let mut fixed_lines = Vec::new();
    for (j, name) in ["ftb", "w", "comp"].iter().enumerate() {
        let mut e: Vec<f32> = st.iter().map(|p| p.e_fixed[j]).collect();
        let mse = st.iter().map(|p| p.se_fixed[j] as f64).sum::<f64>() / st.len().max(1) as f64;
        let (mean, p50, p90, p99, max) = quantiles(&mut e);
        fixed_lines.push(format!(
            "{:<14} {mean:>8.4} {p50:>8.4} {p90:>8.4} {p99:>8.4} {max:>8.3}   PSNR-equiv {:.2} dB",
            format!("k={K_FIXED} {name}"),
            if mse > 0.0 {
                10.0 * (1.0 / mse).log10()
            } else {
                f64::INFINITY
            }
        ));
    }
    let fixed_header = format!(
        "{:<14} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Linf error", "mean", "p50", "p90", "p99", "max"
    );
    lines.push(String::new());
    lines.push(fixed_header);
    lines.extend(fixed_lines);
    log::info!(
        "[oracle] walk {:.1}s, distance transform {:.1}s, {}x{} pixels analysed\n{header}\n{}",
        out.secs_walk,
        out.secs_dt,
        out.stats_w,
        out.stats_h,
        lines.join("\n")
    );

    // How many tets each ranking keeps at each threshold.
    let count_ge = |v: &[f32], th: f32| v.iter().filter(|&&x| x >= th).count();
    let seen = out.w_max.iter().filter(|&&x| x > 0.0).count();
    log::info!(
        "[oracle] tets touched by any ray: {seen} of {tet_count}; w_bound >= w_max holds for {} of them",
        (0..tet_count).filter(|&t| out.w_bound[t] >= out.w_max[t] * 0.999).count()
    );
    for r in &out.rankings {
        let kept: Vec<String> = r
            .thresholds
            .iter()
            .map(|&th| {
                let n = if r.strict {
                    r.scores.iter().filter(|&&x| x > th).count()
                } else {
                    count_ge(&r.scores, th)
                };
                format!("{th}:{n}")
            })
            .collect();
        log::info!("[oracle] {:<14} keeps {}", r.name, kept.join("  "));
    }

    let _ = std::fs::create_dir_all(dir);
    std::fs::write(
        format!("{dir}/oracle_pixels.txt"),
        format!("{header}\n{}\n", lines.join("\n")),
    )
    .ok();
    let (w, h) = (out.stats_w, out.stats_h);
    write_heatmap(&format!("{dir}/n_all.png"), w, h, |i| st[i].n as f32, 256.0);
    write_heatmap(
        &format!("{dir}/k_ftb_0.01.png"),
        w,
        h,
        |i| st[i].k_ftb[1] as f32,
        64.0,
    );
    write_heatmap(
        &format!("{dir}/k_w_0.01.png"),
        w,
        h,
        |i| st[i].k_w[1] as f32,
        64.0,
    );
    write_heatmap(
        &format!("{dir}/k_cost_0.01.png"),
        w,
        h,
        |i| st[i].k_cost[1] as f32,
        64.0,
    );
    write_heatmap(
        &format!("{dir}/k_comp_0.01.png"),
        w,
        h,
        |i| st[i].k_comp[1] as f32,
        64.0,
    );
}

// ---------------------------------------------------------------------------
// Image metrics (tonemapped LDR, values in [0,1])
// ---------------------------------------------------------------------------

/// The deferred output is linear; the swapchain is sRGB. Scoring in sRGB
/// matches what a screenshot of the presented frame scores, which is how the
/// project's 35 dB / 0.99 bar has been judged, and it weights the dark regions
/// where culling errors actually show.
#[inline]
fn srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub fn psnr(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    let mut mse = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        for c in 0..3 {
            let d = (srgb(x[c]) - srgb(y[c])) as f64;
            mse += d * d;
        }
    }
    mse /= (a.len() * 3) as f64;
    if mse == 0.0 {
        f32::INFINITY
    } else {
        (10.0 * (1.0 / mse).log10()) as f32
    }
}

/// SSIM on Rec.709 luminance, 11×11 Gaussian σ=1.5, C1=1e-4, C2=9e-4.
pub fn ssim(a: &[[f32; 4]], b: &[[f32; 4]], w: usize, h: usize) -> f32 {
    const R: usize = 5;
    let luma = |p: &[f32; 4]| 0.2126 * srgb(p[0]) + 0.7152 * srgb(p[1]) + 0.0722 * srgb(p[2]);
    let ya: Vec<f32> = a.iter().map(luma).collect();
    let yb: Vec<f32> = b.iter().map(luma).collect();
    let mut k = [0.0f32; 11];
    let mut ks = 0.0;
    for (i, kv) in k.iter_mut().enumerate() {
        let d = i as f32 - R as f32;
        *kv = (-(d * d) / 4.5).exp();
        ks += *kv;
    }
    for kv in k.iter_mut() {
        *kv /= ks;
    }
    let blur = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0f32; w * h];
        for y in 0..h {
            for x in R..w.saturating_sub(R) {
                tmp[y * w + x] = k
                    .iter()
                    .enumerate()
                    .map(|(i, kv)| kv * src[y * w + x + i - R])
                    .sum();
            }
        }
        let mut out = vec![0.0f32; w * h];
        for y in R..h.saturating_sub(R) {
            for x in 0..w {
                out[y * w + x] = k
                    .iter()
                    .enumerate()
                    .map(|(i, kv)| kv * tmp[(y + i - R) * w + x])
                    .sum();
            }
        }
        out
    };
    let mul = |p: &[f32], q: &[f32]| -> Vec<f32> { p.iter().zip(q).map(|(x, y)| x * y).collect() };
    let (mu_a, mu_b) = (blur(&ya), blur(&yb));
    let (saa, sbb, sab) = (
        blur(&mul(&ya, &ya)),
        blur(&mul(&yb, &yb)),
        blur(&mul(&ya, &yb)),
    );
    let (c1, c2) = (1e-4f32, 9e-4f32);
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for y in R..h.saturating_sub(R) {
        for x in R..w.saturating_sub(R) {
            let i = y * w + x;
            let (ma, mb) = (mu_a[i], mu_b[i]);
            let va = saa[i] - ma * ma;
            let vb = sbb[i] - mb * mb;
            let cov = sab[i] - ma * mb;
            let s = ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            sum += s as f64;
            cnt += 1;
        }
    }
    if cnt == 0 {
        1.0
    } else {
        (sum / cnt as f64) as f32
    }
}

/// Blocking readback of a 2D texture's mip 0 as tightly packed bytes.
pub fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> anyhow::Result<Vec<u8>> {
    let (w, h) = (texture.width(), texture.height());
    let bpp = texture
        .format()
        .block_copy_size(None)
        .ok_or_else(|| anyhow::anyhow!("unsupported format {:?}", texture.format()))?;
    let unpadded = w * bpp;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oracle_readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let view = buffer.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * h) as usize);
    for row in 0..h {
        let s = (row * padded) as usize;
        out.extend_from_slice(&view[s..s + unpadded as usize]);
    }
    drop(view);
    buffer.unmap();
    Ok(out)
}

fn f16s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Rgba16Float texture → per-pixel rgba.
pub fn read_rgba16(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
) -> anyhow::Result<Vec<[f32; 4]>> {
    anyhow::ensure!(
        tex.format() == wgpu::TextureFormat::Rgba16Float,
        "expected Rgba16Float, got {:?}",
        tex.format()
    );
    let v = f16s(&read_texture(device, queue, tex)?);
    Ok(v.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect())
}

/// Overdraw probe (Rg16Float: count, Σα) → (mean fragments/px, mean α per fragment).
pub fn read_overdraw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
) -> anyhow::Result<(f32, f32)> {
    anyhow::ensure!(
        tex.format() == wgpu::TextureFormat::Rg16Float,
        "expected Rg16Float, got {:?}",
        tex.format()
    );
    let v = f16s(&read_texture(device, queue, tex)?);
    let (mut n, mut a) = (0.0f64, 0.0f64);
    for c in v.chunks_exact(2) {
        n += c[0] as f64;
        a += c[1] as f64;
    }
    let px = (v.len() / 2).max(1) as f64;
    Ok(((n / px) as f32, if n > 0.0 { (a / n) as f32 } else { 0.0 }))
}

// ---------------------------------------------------------------------------
// Sweep harness (driven from the render loop)
// ---------------------------------------------------------------------------

pub struct Candidate {
    pub name: String,
    pub shell: bool,
    /// Bit-packed keep mask, or `None` for all tets.
    pub mask: Option<Vec<u32>>,
    pub kept: usize,
}

pub struct SweepRow {
    pub name: String,
    pub kept: usize,
    pub ms: f32,
    pub overdraw: f32,
    pub mean_alpha: f32,
    pub psnr: f32,
    pub ssim: f32,
}

pub enum Stage {
    Waiting,
    Running { cand: usize, frame: usize },
    Done,
}

pub struct OracleRun {
    pub cfg: OracleConfig,
    pub stage: Stage,
    pub candidates: Vec<Candidate>,
    pub reference: Option<Vec<[f32; 4]>>,
    pub rows: Vec<SweepRow>,
    pub last_frame: Option<std::time::Instant>,
    pub frame_ms: Vec<f32>,
}

impl OracleRun {
    pub fn new(cfg: OracleConfig) -> Self {
        Self {
            cfg,
            stage: Stage::Waiting,
            candidates: Vec::new(),
            reference: None,
            rows: Vec::new(),
            last_frame: None,
            frame_ms: Vec::new(),
        }
    }

    /// Build the candidate list from the oracle's rankings. `base_mask` is the
    /// PVS mask the frame already applies, so every candidate is ANDed with it.
    pub fn build_candidates(&mut self, out: &OracleOutput, base_mask: &[u32], tet_count: usize) {
        let pack = |score: &[f32], th: f32| -> (Vec<u32>, usize) {
            let mut m = base_mask.to_vec();
            let mut kept = 0usize;
            for (t, &s) in score.iter().enumerate().take(tet_count) {
                if s < th {
                    m[t >> 5] &= !(1u32 << (t & 31));
                } else if m[t >> 5] & (1u32 << (t & 31)) != 0 {
                    kept += 1;
                }
            }
            (m, kept)
        };
        let all = base_mask
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>()
            .min(tet_count);
        self.candidates.push(Candidate {
            name: "reference (shell off)".into(),
            shell: false,
            mask: None,
            kept: all,
        });
        self.candidates.push(Candidate {
            name: "baseline (shell on)".into(),
            shell: true,
            mask: None,
            kept: all,
        });
        for r in &out.rankings {
            for &th in &r.thresholds {
                // `>` for the budgets: a tet is droppable at exactly its budget.
                let eff = if r.strict { th.next_up() } else { th };
                let (m, kept) = pack(&r.scores, eff);
                self.candidates.push(Candidate {
                    name: format!("{} {} {th}", r.name, if r.strict { ">" } else { ">=" }),
                    shell: true,
                    mask: Some(m),
                    kept,
                });
            }
        }
    }

    pub fn finish_candidate(
        &mut self,
        overdraw: (f32, f32),
        frame: Option<Vec<[f32; 4]>>,
        w: usize,
        h: usize,
    ) {
        let Stage::Running { cand, .. } = self.stage else {
            return;
        };
        let c = &self.candidates[cand];
        let ms = if self.frame_ms.is_empty() {
            0.0
        } else {
            self.frame_ms.iter().sum::<f32>() / self.frame_ms.len() as f32
        };
        let (p, s) = match (&self.reference, &frame) {
            (Some(r), Some(f)) if r.len() == f.len() => (psnr(r, f), ssim(r, f, w, h)),
            _ => (f32::NAN, f32::NAN),
        };
        if cand == 0 {
            self.reference = frame;
        }
        log::info!(
            "[oracle sweep] {:<22} kept {:>9}  {:>6.2} ms ({:>5.1} fps)  overdraw {:>6.1}/px  alpha {:.3}  PSNR {:>6.2}  SSIM {:.5}",
            c.name, c.kept, ms, if ms > 0.0 { 1000.0 / ms } else { 0.0 }, overdraw.0, overdraw.1, p, s
        );
        self.rows.push(SweepRow {
            name: c.name.clone(),
            kept: c.kept,
            ms,
            overdraw: overdraw.0,
            mean_alpha: overdraw.1,
            psnr: p,
            ssim: s,
        });
        self.frame_ms.clear();
        self.last_frame = None;
        self.stage = if cand + 1 < self.candidates.len() {
            Stage::Running {
                cand: cand + 1,
                frame: 0,
            }
        } else {
            Stage::Done
        };
    }

    pub fn write_csv(&self) {
        let mut s =
            String::from("candidate,kept_tets,frame_ms,fps,overdraw_per_px,mean_alpha,psnr,ssim\n");
        for r in &self.rows {
            s.push_str(&format!(
                "{},{},{:.3},{:.2},{:.2},{:.4},{:.3},{:.5}\n",
                r.name,
                r.kept,
                r.ms,
                if r.ms > 0.0 { 1000.0 / r.ms } else { 0.0 },
                r.overdraw,
                r.mean_alpha,
                r.psnr,
                r.ssim
            ));
        }
        let path = format!("{}/sweep.csv", self.cfg.out_dir);
        if let Err(e) = std::fs::write(&path, s) {
            log::error!("[oracle] {path}: {e}");
        } else {
            log::info!("[oracle] wrote {path}");
        }
    }
}

// ---------------------------------------------------------------------------
// Render-loop hooks
// ---------------------------------------------------------------------------

impl crate::App {
    /// Before the frame's GPU work: run the oracle once the scene has settled,
    /// then step the sweep — install the candidate's mask and shell state, time
    /// the previous frame, and arm the overdraw probe on a candidate's last frame.
    pub(crate) fn oracle_pre_frame(&mut self, view_mat: glam::Mat4, w: u32, h: u32) {
        let Some(run) = self.oracle.as_mut() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else { return };

        if let Stage::Waiting = run.stage {
            if gpu.ts_frame < run.cfg.at_frame || self.shell_system.rebuild_pending() {
                return;
            }
            // The sweep measures masks against "draw everything", so the
            // importance cull must be off; give the visibility mask a frame to
            // be rebuilt without it.
            if self.shell_system.importance_enabled {
                self.shell_system.importance_enabled = false;
                return;
            }
            // The overdraw probe only exists on the compute interval path.
            self.use_mesh_shader = false;
            let tet_count = self.scene_data.tet_count as usize;
            let cache = match self.shell_system.scene_cache() {
                Some(c) => c,
                None => {
                    log::info!("[oracle] no shell cache; building adjacency + opacity");
                    let sd = &self.scene_data;
                    let neighbors = rmesh_raytrace::compute_tet_neighbors(&sd.indices, tet_count);
                    let alphas = crate::flare::tet_alphas(
                        &sd.vertices,
                        &sd.indices,
                        &sd.densities,
                        tet_count,
                    );
                    let max_od = crate::importance::tet_max_od(
                        &sd.vertices,
                        &sd.indices,
                        &sd.densities,
                        tet_count,
                    );
                    std::sync::Arc::new(crate::shell::SceneCache {
                        neighbors,
                        alphas,
                        max_od,
                    })
                }
            };
            // The walk can leave the hull on a concave boundary even for an
            // interior point, so confirm "outside" by brute force before
            // falling back to boundary-face entry.
            let hint = if self.rt_start_tet_hint >= 0 {
                self.rt_start_tet_hint as usize
            } else {
                0
            };
            let located = rmesh_raytrace::find_containing_tet_walk(
                &self.scene_data.vertices,
                &self.scene_data.indices,
                &cache.neighbors,
                tet_count,
                self.camera.position,
                hint,
            )
            .or_else(|| {
                rmesh_raytrace::find_containing_tet(
                    &self.scene_data.vertices,
                    &self.scene_data.indices,
                    tet_count,
                    self.camera.position,
                )
            });
            if let Some(t) = located {
                self.rt_start_tet_hint = t as i32;
            }
            let bvh = rmesh_raytrace::build_boundary_bvh(
                &self.scene_data.vertices,
                &self.scene_data.indices,
                &cache.neighbors,
                tet_count,
            );
            log::info!(
                "[oracle] camera {} the mesh; boundary BVH over {} faces",
                if located.is_some() {
                    "inside"
                } else {
                    "outside"
                },
                bvh.boundary_faces.len()
            );
            let start = RayStart {
                inside: located,
                bvh: &bvh,
            };

            let inv_view = view_mat.inverse();
            let c2w = Mat3::from_cols(
                inv_view.col(0).truncate(),
                -inv_view.col(1).truncate(),
                -inv_view.col(2).truncate(),
            );
            let f_val = 1.0 / (self.camera.fov_y / 2.0).tan();
            let cam = CamParams {
                eye: self.camera.position,
                c2w,
                fx: f_val * h as f32 / 2.0,
                fy: f_val * h as f32 / 2.0,
                cx: w as f32 / 2.0,
                cy: h as f32 / 2.0,
                w,
                h,
            };
            let colors = tet_colors(&self.sh_coeffs, self.pbr_data.as_ref(), tet_count);
            log::info!(
                "[oracle] walking {}x{} rays ({} tets, stride {})...",
                w,
                h,
                tet_count,
                run.cfg.stride
            );
            let out = run_oracle(
                cam,
                &self.scene_data,
                &cache.neighbors,
                &cache.alphas,
                &colors,
                &start,
                run.cfg.stride,
                &run.cfg.thresholds,
            );
            report(&out, &run.cfg.out_dir, tet_count);
            run.build_candidates(&out, &self.active_mask_scratch, tet_count);
            if !self.shell_system.has_shell() {
                log::warn!("[oracle] no shell built; every candidate runs with the shell off");
                for c in run.candidates.iter_mut() {
                    c.shell = false;
                }
            }
            run.stage = Stage::Running { cand: 0, frame: 0 };
        }

        let Stage::Running { cand, frame } = run.stage else {
            return;
        };
        let c = &run.candidates[cand];
        if frame == 0 {
            self.shell_system.enabled = c.shell;
            let mask: &[u32] = c.mask.as_deref().unwrap_or(&self.active_mask_scratch);
            gpu.queue
                .write_buffer(&gpu.buffers.tet_visibility, 0, bytemuck::cast_slice(mask));
            run.frame_ms.clear();
            run.last_frame = None;
        } else if frame >= 2 {
            if let Some(t) = run.last_frame {
                run.frame_ms.push(t.elapsed().as_secs_f32() * 1000.0);
            }
        }
        run.last_frame = Some(std::time::Instant::now());
        self.overdraw_view = if frame + 1 == run.cfg.frames {
            Some(rmesh_render::OverdrawView::Count)
        } else {
            None
        };
        run.stage = Stage::Running {
            cand,
            frame: frame + 1,
        };
    }
}

/// After the frame's submit: on a candidate's last frame, read the probe and
/// the deferred output back and score it. A free function over the fields it
/// touches, because the render loop holds `gpu` borrowed for the whole frame.
pub(crate) fn oracle_post_frame(
    oracle: &mut Option<OracleRun>,
    gpu: &crate::gpu_state::GpuState,
    overdraw_view: &mut Option<rmesh_render::OverdrawView>,
    active_mask_dirty: &mut bool,
) {
    let Some(run) = oracle.as_mut() else { return };
    let Stage::Running { frame, .. } = run.stage else {
        return;
    };
    if frame < run.cfg.frames {
        return;
    }
    let overdraw = match read_overdraw(&gpu.device, &gpu.queue, &gpu.overdraw_target.texture) {
        Ok(v) => v,
        Err(e) => {
            log::error!("[oracle] overdraw readback: {e}");
            (f32::NAN, f32::NAN)
        }
    };
    let img = match gpu.deferred_output.as_ref() {
        Some(t) => read_rgba16(&gpu.device, &gpu.queue, t)
            .map_err(|e| log::error!("[oracle] frame readback: {e}"))
            .ok(),
        None => {
            log::warn!("[oracle] no deferred output; PSNR unavailable");
            None
        }
    };
    let (w, h) = (
        gpu.surface_config.width as usize,
        gpu.surface_config.height as usize,
    );
    run.finish_candidate(overdraw, img, w, h);
    if let Stage::Done = run.stage {
        run.write_csv();
        *overdraw_view = None;
        *active_mask_dirty = true;
        if run.cfg.exit_when_done {
            std::process::exit(0);
        }
    }
}
