//! Multi-tet forward rendering tests.
//!
//! Ported from delaunay_splatting/tests/multi_tet_test.py.
//! Uses hand-crafted multi-tet scenes (no Delaunay dependency).
//!
//! Run: `cargo test -p rmesh-render --test multi_tet_test -- --nocapture`

mod common;

use common::*;
use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 189710234;
const W: u32 = 64;
const H: u32 = 64;
/// Tolerance for CPU vs GPU comparison on outside views.
/// Outside views typically show mean_diff < 0.001.
const ATOL: f32 = 0.02;

fn setup_camera(eye: Vec3, target: Vec3) -> (glam::Mat4, glam::Mat3, [f32; 4]) {
    let aspect = W as f32 / H as f32;
    let proj = perspective_matrix(std::f32::consts::FRAC_PI_2, aspect, 0.01, 100.0);
    let view = look_at(eye, target, Vec3::new(0.0, 0.0, 1.0));
    let vp = proj * view;
    let (c2w, intrinsics) =
        test_camera_c2w_intrinsics(eye, target, std::f32::consts::FRAC_PI_2, W as f32, H as f32);
    (vp, c2w, intrinsics)
}

/// Two tetrahedra sharing a face (5 unique vertices).
/// Tests sorting correctness and multi-tet compositing.
fn two_tet_scene(rng: &mut ChaCha8Rng) -> SceneData {
    // 5 vertices forming two tets sharing face (0,1,2)
    // Tet 0: [0,1,2,3]  Tet 1: [0,2,1,4] (flipped to maintain outward orientation)
    let vertices = vec![
        0.0, 0.0, 0.0, // v0
        1.0, 0.0, 0.0, // v1
        0.5, 1.0, 0.0, // v2
        0.5, 0.3, 0.8, // v3 (above)
        0.5, 0.3, -0.8, // v4 (below)
    ];
    let indices = vec![
        0, 1, 2, 3, // tet 0
        0, 2, 1, 4, // tet 1
    ];

    let densities = vec![
        rng.random::<f32>() * 3.0 + 0.5,
        rng.random::<f32>() * 3.0 + 0.5,
    ];
    let color_grads = vec![
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
        (rng.random::<f32>() - 0.5) * 0.1,
    ];

    build_test_scene(vertices, indices, densities, color_grads)
}

/// Four tetrahedra forming a larger shape (8 vertices at cube corners + center).
fn four_tet_scene(rng: &mut ChaCha8Rng) -> SceneData {
    // Center vertex plus 4 base corners of a cube, forming 4 tets
    let s = 0.5f32;
    let vertices = vec![
        0.0, 0.0, 0.0, // v0: center
        s, s, s, // v1
        -s, s, s, // v2
        -s, -s, s, // v3
        s, -s, s, // v4
        s, s, -s, // v5
        -s, s, -s, // v6
        -s, -s, -s, // v7
        s, -s, -s, // v8
    ];
    let indices = vec![
        0, 1, 2, 3, // tet 0
        0, 1, 4, 5, // tet 1
        0, 2, 6, 3, // tet 2
        0, 5, 8, 7, // tet 3
    ];

    let tet_count = 4;
    let densities: Vec<f32> = (0..tet_count)
        .map(|_| rng.random::<f32>() * 3.0 + 0.5)
        .collect();
    let color_grads: Vec<f32> = (0..tet_count * 3)
        .map(|_| (rng.random::<f32>() - 0.5) * 0.1)
        .collect();

    build_test_scene(vertices, indices, densities, color_grads)
}

/// Camera inside the two-tet scene, looking outward.
/// Interior views have higher CPU-vs-GPU divergence because hardware rasterization
/// clips geometry to the near plane, while CPU ray casting intersects the full tet.
#[test]
fn test_two_tet_center_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = two_tet_scene(&mut rng);

    let eye = Vec3::new(0.5, 0.4, 0.1); // inside one of the tets, off the shared face
    let target = eye + Vec3::new(1.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("two_tet_center: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        // Very relaxed tolerance: camera inside multi-tet scene causes major
        // near-plane clipping divergence on GPU (the hardware rasterizer clips
        // to the near plane while CPU ray casting intersects the full tet).
        // With two overlapping tets this compounds significantly.
        assert!(
            mean_diff < 0.5,
            "two_tet_center: mean_diff {mean_diff} >= 0.5"
        );
    } else {
        eprintln!("Skipping GPU test (no adapter)");
    }
}

/// Camera outside the two-tet scene, verifying sorting correctness.
#[test]
fn test_two_tet_outside_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = two_tet_scene(&mut rng);

    // View from outside. Avoid looking along Z axis since
    // up=(0,0,1) creates degenerate look_at when forward ∥ up.
    let eye = Vec3::new(3.0, 0.4, 1.0);
    let target = Vec3::new(0.5, 0.4, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

    if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("two_tet_outside: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(
            mean_diff < ATOL,
            "two_tet_outside: mean_diff {mean_diff} >= {ATOL}"
        );
    } else {
        eprintln!("Skipping GPU test (no adapter)");
    }
}

/// Four-tet scene viewed from outside at multiple angles.
#[test]
fn test_four_tet_multi_angle() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);

    let viewpoints = [
        (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
        (Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO),
    ];

    for (i, (eye, target)) in viewpoints.iter().enumerate() {
        let (vp, c2w, intrinsics) = setup_camera(*eye, *target);
        let cpu_image = cpu_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H);

        if let Some(gpu_image) = gpu_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!("four_tet angle {i}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
            assert!(
                mean_diff < ATOL,
                "four_tet angle {i}: mean_diff {mean_diff} >= {ATOL}"
            );
        } else {
            eprintln!("Skipping GPU test (no adapter)");
            break;
        }
    }
}

/// Two-tet scene, camera outside — interval shading path.
#[test]
fn test_interval_two_tet_outside_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = two_tet_scene(&mut rng);

    let eye = Vec3::new(3.0, 0.4, 1.0);
    let target = Vec3::new(0.5, 0.4, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

    if let Some(gpu_image) = gpu_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("interval_two_tet_outside: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(
            mean_diff < ATOL,
            "interval_two_tet_outside: mean_diff {mean_diff} >= {ATOL}"
        );
    } else {
        eprintln!("Skipping GPU interval test (no adapter or no mesh shader support)");
    }
}

/// Four-tet scene, multiple angles — interval shading path.
#[test]
fn test_interval_four_tet_multi_angle() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);

    let viewpoints = [
        (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO),
        (Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO),
    ];

    for (i, (eye, target)) in viewpoints.iter().enumerate() {
        let (vp, c2w, intrinsics) = setup_camera(*eye, *target);
        let cpu_image = cpu_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H);

        if let Some(gpu_image) = gpu_interval_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H)
        {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!(
                "interval_four_tet angle {i}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}"
            );
            assert!(
                mean_diff < ATOL,
                "interval_four_tet angle {i}: mean_diff {mean_diff} >= {ATOL}"
            );
        } else {
            eprintln!("Skipping GPU interval test (no adapter or no mesh shader support)");
            break;
        }
    }
}

/// Four-tet scene, multiple angles — compute-interval shading path (no mesh shader).
#[test]
fn test_compute_interval_four_tet_multi_angle() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);

    let viewpoints = [
        (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO),
        (Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO),
    ];

    for (i, (eye, target)) in viewpoints.iter().enumerate() {
        let (vp, c2w, intrinsics) = setup_camera(*eye, *target);
        let cpu_image = cpu_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H);

        if let Some(gpu_image) =
            gpu_compute_interval_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H)
        {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!("compute_interval_four_tet angle {i}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
            assert!(
                mean_diff < ATOL,
                "compute_interval_four_tet angle {i}: mean_diff {mean_diff} >= {ATOL}"
            );
        } else {
            eprintln!("Skipping GPU compute-interval test (no adapter)");
            break;
        }
    }
}

/// Build a procedural many-tet scene: an N×N×N grid of axis-aligned tetrahedra.
///
/// Used to exercise the GPU sort path at viewer-realistic scale, where the
/// camera's frustum captures only a fraction of the tets. Each cell contributes
/// one tetrahedron (4 vertices forming a regular-ish shape). The resulting
/// scene has `n^3` tets and `4·n^3` vertices.
fn grid_tet_scene(n: u32, rng: &mut ChaCha8Rng) -> SceneData {
    let nf = n as f32;
    let mut vertices: Vec<f32> = Vec::with_capacity(4 * 3 * (n * n * n) as usize);
    let mut indices: Vec<u32> = Vec::with_capacity(4 * (n * n * n) as usize);
    let mut densities: Vec<f32> = Vec::with_capacity((n * n * n) as usize);
    let mut color_grads: Vec<f32> = Vec::with_capacity(3 * (n * n * n) as usize);

    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let cx = (i as f32 - nf * 0.5) * 0.2;
                let cy = (j as f32 - nf * 0.5) * 0.2;
                let cz = (k as f32 - nf * 0.5) * 0.2;
                let s = 0.08;
                let base = vertices.len() as u32 / 3;
                vertices.extend_from_slice(&[
                    cx + s, cy + s, cz + s,
                    cx + s, cy - s, cz - s,
                    cx - s, cy + s, cz - s,
                    cx - s, cy - s, cz + s,
                ]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base + 3]);
                densities.push(rng.random::<f32>() * 2.0 + 0.5);
                color_grads.extend_from_slice(&[
                    (rng.random::<f32>() - 0.5) * 0.1,
                    (rng.random::<f32>() - 0.5) * 0.1,
                    (rng.random::<f32>() - 0.5) * 0.1,
                ]);
            }
        }
    }

    build_test_scene(vertices, indices, densities, color_grads)
}

/// Scaled-up HW-compute interval render test. Mirrors the viewer's GPU-sort
/// path on a scene where the camera frustum sees only a fraction of the tets,
/// so `instance_count = tet_count` (this codebase's current behavior) is much
/// larger than `visible_count`. If something in the GPU sort → indirect_convert
/// → interval_compute → render chain breaks at scale, the small 4-tet test
/// passes while this one catches it.
#[test]
fn test_hw_compute_interval_grid_scene() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = grid_tet_scene(50, &mut rng); // 50^3 = 125k tets, viewer-scale

    let eye = Vec3::new(2.5, 0.0, 0.0);
    let target = Vec3::ZERO;
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);
    let cpu_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(
        cpu_alpha > 1.0,
        "CPU reference is essentially black ({cpu_alpha}) — bad test setup"
    );

    let Some(hw_image) =
        gpu_hw_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H)
    else {
        eprintln!("Skipping hw_compute_interval_grid_scene (no GPU adapter)");
        return;
    };
    let hw_alpha: f32 = hw_image.iter().map(|p| p[3]).sum();

    let legacy_image = gpu_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H)
        .expect("legacy path should also be runnable");
    let legacy_alpha: f32 = legacy_image.iter().map(|p| p[3]).sum();

    eprintln!(
        "grid({}) cpu_alpha={cpu_alpha:.2}, hw_alpha={hw_alpha:.2}, legacy_alpha={legacy_alpha:.2}",
        scene.tet_count
    );

    assert!(
        hw_alpha > cpu_alpha * 0.5,
        "HW-compute on {}-tet grid is (near-)black (hw_alpha={hw_alpha}, legacy_alpha={legacy_alpha}, cpu_alpha={cpu_alpha}); \
         GPU sort path is broken at scale",
        scene.tet_count
    );

    let (max_diff, mean_diff, _) = compare_images(&cpu_image, &hw_image);
    eprintln!("  HW vs CPU: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
    assert!(
        mean_diff < ATOL * 2.0,
        "grid HW mean_diff {mean_diff} >= {} (max_diff={max_diff})",
        ATOL * 2.0
    );
}

/// HW-compute interval shading on a four-tet scene from several angles.
///
/// Exercises `project_compute_hw.wgsl` (uniform-buffer uniforms — the path
/// rmesh-viewer takes when sort_mode != CPU). Before this test existed, every
/// interval/compute-interval test in the suite passed `hw_compute_bg = None`
/// and fell through to the legacy `project_compute.wgsl` storage-uniforms
/// path; the viewer's actual project shader was never exercised, so changes
/// that broke only it could pass the test suite while rendering the viewer
/// all-black.
///
/// Compares against both the CPU reference and the legacy GPU path: the HW
/// and legacy GPU paths share their entire downstream (sort, indirect_convert,
/// interval_compute, render). They must produce the same image, and that
/// image must match the CPU reference.
#[test]
fn test_hw_compute_interval_four_tet_multi_angle() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);

    let viewpoints = [
        (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO),
        (Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO),
    ];

    for (i, (eye, target)) in viewpoints.iter().enumerate() {
        let (vp, c2w, intrinsics) = setup_camera(*eye, *target);
        let cpu_image = cpu_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H);
        let cpu_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
        assert!(
            cpu_alpha > 0.1,
            "angle {i}: CPU reference is all-zero — bad test scene"
        );

        let Some(hw_image) =
            gpu_hw_compute_interval_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H)
        else {
            eprintln!("Skipping hw_compute_interval_four_tet_multi_angle (no GPU adapter)");
            return;
        };
        let hw_alpha: f32 = hw_image.iter().map(|p| p[3]).sum();

        let legacy_image =
            gpu_compute_interval_render_scene(&scene, *eye, vp, c2w, intrinsics, W, H)
                .expect("legacy path should also be runnable");
        let legacy_alpha: f32 = legacy_image.iter().map(|p| p[3]).sum();

        eprintln!(
            "hw_compute_interval_four_tet angle {i}: cpu_alpha={cpu_alpha:.3}, hw_alpha={hw_alpha:.3}, legacy_alpha={legacy_alpha:.3}"
        );

        assert!(
            hw_alpha > 0.1,
            "angle {i}: HW-compute interval render is all-black \
             (hw_alpha={hw_alpha}, legacy_alpha={legacy_alpha}); \
             project_compute_hw.wgsl → sort → interval_compute chain is broken"
        );

        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &hw_image);
        eprintln!("  HW vs CPU: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(
            mean_diff < ATOL,
            "angle {i}: HW-compute mean_diff {mean_diff} >= {ATOL} (max_diff={max_diff})"
        );
    }
}

/// Verify CPU sorting produces the same order as GPU.
#[test]
fn test_sort_order_matches() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = four_tet_scene(&mut rng);
    let cam = Vec3::new(3.0, 0.0, 0.0);

    let sorted = sort_tets_back_to_front(&scene, cam);

    // Back-to-front: first in the list should be the farthest tet
    // Verify it's a valid permutation
    let n = scene.tet_count as usize;
    assert_eq!(sorted.len(), n);
    let mut seen = vec![false; n];
    for &idx in &sorted {
        assert!((idx as usize) < n);
        assert!(!seen[idx as usize], "duplicate index in sort");
        seen[idx as usize] = true;
    }

    // Verify depth ordering: each successive tet should be closer (or equal)
    for i in 1..sorted.len() {
        let prev = sorted[i - 1] as usize;
        let curr = sorted[i] as usize;
        let depth_prev = {
            let cx = scene.circumdata[prev * 4];
            let cy = scene.circumdata[prev * 4 + 1];
            let cz = scene.circumdata[prev * 4 + 2];
            let r2 = scene.circumdata[prev * 4 + 3];
            let d = Vec3::new(cx, cy, cz) - cam;
            d.dot(d) - r2
        };
        let depth_curr = {
            let cx = scene.circumdata[curr * 4];
            let cy = scene.circumdata[curr * 4 + 1];
            let cz = scene.circumdata[curr * 4 + 2];
            let r2 = scene.circumdata[curr * 4 + 3];
            let d = Vec3::new(cx, cy, cz) - cam;
            d.dot(d) - r2
        };
        assert!(
            depth_prev >= depth_curr - 1e-6,
            "sort order wrong: depth[{prev}]={depth_prev} < depth[{curr}]={depth_curr}"
        );
    }
}
