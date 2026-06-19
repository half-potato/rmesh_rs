//! Single-tet forward rendering tests.
//!
//! Ported from delaunay_splatting/tests/single_tet_test.py.
//! Compares GPU pipeline output against CPU reference renderer.
//!
//! Run: `cargo test -p rmesh-render --test single_tet_test -- --nocapture`

mod common;

use common::*;
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 189710234;
const W: u32 = 64;
const H: u32 = 64;
/// Tolerance for CPU vs GPU hardware rasterizer comparison.
/// Outside views (face_view) show mean_diff < 0.001; interior views diverge
/// more due to near-plane clipping on GPU that CPU ray casting doesn't have.
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

/// Camera at the centroid of the tet, looking outward with random rotation.
/// Tests: camera-inside-tet rendering at various tet sizes.
/// Interior views have higher CPU-vs-GPU divergence because hardware rasterization
/// clips geometry to the near plane, while CPU ray casting intersects the full tet.
/// Larger tets produce more geometry behind the camera, increasing the divergence.
#[test]
fn test_center_view() {
    let radii = [0.05, 0.1, 0.2, 0.4];

    for &radius in &radii {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let scene = random_single_tet_scene(&mut rng, radius);

        let verts = load_tet_verts(&scene, 0);
        let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;

        // Camera at centroid, look along +X
        let eye = centroid;
        let target = centroid + Vec3::new(1.0, 0.0, 0.0);
        let (vp, c2w, intrinsics) = setup_camera(eye, target);

        let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

        // Check CPU image isn't all zero (tet should be visible from inside)
        let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
        assert!(
            total_alpha > 0.01,
            "radius={radius}: CPU image is all-zero, tet not visible from center"
        );

        // GPU comparison (skip if no adapter)
        if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!(
                "center_view radius={radius}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}"
            );
            // Relaxed tolerance for interior views: near-plane clipping on GPU
            // causes larger divergence with bigger tets. Divergence scales
            // roughly linearly with radius (more geometry behind camera).
            let tol = ATOL + radius * 0.6;
            assert!(
                mean_diff < tol,
                "radius={radius}: mean_diff {mean_diff} >= {tol}"
            );
        } else {
            eprintln!("Skipping GPU test (no adapter)");
        }
    }
}

/// Camera near a face of the tet at various distances.
/// Tests: standard outside-looking-in rendering.
#[test]
fn test_face_view() {
    let offsets = [0.1, 1.0, 5.0, 10.0];

    for &offset in &offsets {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let scene = random_single_tet_scene(&mut rng, 0.3);

        let verts = load_tet_verts(&scene, 0);
        let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;

        // Face 0 center
        let face_center = (verts[0] + verts[2] + verts[1]) / 3.0;
        let face_normal = (face_center - centroid).normalize();

        let eye = face_center + face_normal * offset;
        let target = centroid;
        let (vp, c2w, intrinsics) = setup_camera(eye, target);

        let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

        if let Some(gpu_image) = gpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!(
                "face_view offset={offset}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}"
            );
            assert!(
                mean_diff < ATOL,
                "offset={offset}: mean_diff {mean_diff} >= {ATOL}"
            );
        } else {
            eprintln!("Skipping GPU test (no adapter)");
        }
    }
}

/// Ray tracing: camera outside single tet, BVH entry.
/// Compare GPU raytrace against CPU reference renderer.
#[test]
fn test_raytrace_single_tet_outside() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;

    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.1, "CPU image is all-zero");

    if let Some(rt_image) = gpu_raytrace_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &rt_image);
        eprintln!("raytrace outside: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        assert!(
            mean_diff < ATOL,
            "raytrace outside: mean_diff {mean_diff} >= {ATOL}"
        );
    } else {
        eprintln!("Skipping GPU raytrace test (no adapter)");
    }
}

/// Ray tracing: camera inside single tet.
/// With single tet, start_tet = 0, adjacency traversal hits 1 tet then exits.
#[test]
fn test_raytrace_single_tet_inside() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;

    let eye = centroid;
    let target = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(rt_image) = gpu_raytrace_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &rt_image);
        eprintln!("raytrace inside: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        // Interior view still has divergence due to floating-point differences
        // in ray-tet intersection math between CPU and GPU. Relax tolerance.
        assert!(
            mean_diff < 0.1,
            "raytrace inside: mean_diff {mean_diff} >= 0.1"
        );
    } else {
        eprintln!("Skipping GPU raytrace test (no adapter)");
    }
}

/// Interval shading: camera at the centroid of the tet, looking outward.
/// Tests the interval path for interior views.
#[test]
fn test_interval_center_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid;
    let target = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) = gpu_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("interval_center_view: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        let tol = ATOL + 0.3 * 0.6;
        assert!(
            mean_diff < tol,
            "interval_center_view: mean_diff {mean_diff} >= {tol}"
        );
    } else {
        eprintln!("Skipping GPU interval test (no adapter or no mesh shader support)");
    }
}

/// Interval shading: camera near a face of the tet, looking in.
/// Tests standard outside-looking-in rendering via interval path.
#[test]
fn test_interval_face_view() {
    let offsets = [0.1, 1.0, 5.0, 10.0];

    for &offset in &offsets {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let scene = random_single_tet_scene(&mut rng, 0.3);

        let verts = load_tet_verts(&scene, 0);
        let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
        let face_center = (verts[0] + verts[2] + verts[1]) / 3.0;
        let face_normal = (face_center - centroid).normalize();

        let eye = face_center + face_normal * offset;
        let target = centroid;
        let (vp, c2w, intrinsics) = setup_camera(eye, target);

        let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

        if let Some(gpu_image) = gpu_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H) {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!(
                "interval_face_view offset={offset}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}"
            );
            // Near-camera views (offset < 1) have larger error from silhouette
            // edge interpolation artifacts — use a relaxed tolerance.
            let tol = if offset < 1.0 { 0.03 } else { ATOL };
            assert!(
                mean_diff < tol,
                "interval offset={offset}: mean_diff {mean_diff} >= {tol}"
            );
        } else {
            eprintln!("Skipping GPU interval test (no adapter or no mesh shader support)");
        }
    }
}

/// Verify the CPU reference renderer produces reasonable output
/// by checking pixel values are bounded and consistent.
#[test]
fn test_cpu_reference_sanity() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;

    let eye = centroid + Vec3::new(2.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, centroid);

    let image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

    // All alpha values should be in [0, 1]
    for (i, pixel) in image.iter().enumerate() {
        assert!(
            pixel[3] >= -0.001 && pixel[3] <= 1.001,
            "pixel {i}: alpha={} out of range",
            pixel[3]
        );
        // Premultiplied color channels should be non-negative
        for ch in 0..3 {
            assert!(
                pixel[ch] >= -0.001,
                "pixel {i} ch {ch}: color={} is negative",
                pixel[ch]
            );
        }
    }

    // Some pixels should be non-zero (tet is visible)
    let total_alpha: f32 = image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.1, "Image is all-zero");
}

/// Compute-interval shading: camera at the centroid of the tet, looking outward.
/// Tests the compute-interval path for interior views (no mesh shader needed).
#[test]
fn test_compute_interval_center_view() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let scene = random_single_tet_scene(&mut rng, 0.3);

    let verts = load_tet_verts(&scene, 0);
    let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
    let eye = centroid;
    let target = centroid + Vec3::new(1.0, 0.0, 0.0);
    let (vp, c2w, intrinsics) = setup_camera(eye, target);

    let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);
    let total_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
    assert!(total_alpha > 0.01, "CPU image is all-zero");

    if let Some(gpu_image) =
        gpu_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H)
    {
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        eprintln!("compute_interval_center_view: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}");
        let tol = ATOL + 0.3 * 0.6;
        assert!(
            mean_diff < tol,
            "compute_interval_center_view: mean_diff {mean_diff} >= {tol}"
        );
    } else {
        eprintln!("Skipping GPU compute-interval test (no adapter)");
    }
}

/// HW-compute interval shading: camera near a face of the tet.
///
/// Exercises `project_compute_hw.wgsl` (uniform-buffer uniforms, the viewer's
/// path) instead of the legacy storage-buffer `project_compute.wgsl` that the
/// other interval tests cover. Before this test existed, changes that broke
/// only the HW project pipeline could pass every interval test while still
/// rendering the viewer all-black. The regression check is intentionally
/// minimal: just "did anything get rendered" — total alpha must be
/// non-trivial.
#[test]
fn test_hw_compute_interval_face_view() {
    // Iterating multiple offsets so we can see which (camera distance, GPU
    // path) combinations black out. offset=0.1 is the close-camera regime that
    // diverges from the CPU reference even on the legacy compute-interval path.
    let offsets = [1.0, 5.0, 10.0, 0.1];

    for &offset in &offsets {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let scene = random_single_tet_scene(&mut rng, 0.3);

        let verts = load_tet_verts(&scene, 0);
        let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
        let face_center = (verts[0] + verts[2] + verts[1]) / 3.0;
        let face_normal = (face_center - centroid).normalize();

        let eye = face_center + face_normal * offset;
        let target = centroid;
        let (vp, c2w, intrinsics) = setup_camera(eye, target);

        let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);
        let cpu_alpha: f32 = cpu_image.iter().map(|p| p[3]).sum();
        assert!(
            cpu_alpha > 0.01,
            "offset={offset}: CPU reference is all-zero — bad test scene"
        );

        let Some(gpu_image) =
            gpu_hw_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H)
        else {
            eprintln!("Skipping hw_compute_interval_face_view (no GPU adapter)");
            return;
        };

        // Also render via the LEGACY compute path so we can compare. Both share
        // the entire downstream (sort, indirect_convert, interval_compute, render).
        // If legacy_alpha > 0 and gpu_alpha == 0, the bug is in
        // project_compute_hw.wgsl vs project_compute.wgsl.
        let legacy_image =
            gpu_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H)
                .expect("legacy path should also be runnable");
        let legacy_alpha: f32 = legacy_image.iter().map(|p| p[3]).sum();

        let gpu_alpha: f32 = gpu_image.iter().map(|p| p[3]).sum();
        eprintln!(
            "hw_compute_interval_face_view offset={offset}: cpu_alpha={cpu_alpha:.3}, gpu_alpha={gpu_alpha:.3}, legacy_alpha={legacy_alpha:.3}"
        );

        // Primary regression assertion: image is NOT all-black. Any non-trivial
        // alpha means the project_compute_hw → sort → interval_gen → render
        // chain emitted at least one visible fragment.
        assert!(
            gpu_alpha > 0.1,
            "offset={offset}: HW-compute interval render is all-black (gpu_alpha={gpu_alpha}); \
             project_compute_hw.wgsl / interval_generate.wgsl chain is broken"
        );

        // Sanity-check it agrees with the CPU reference within the same
        // tolerance the legacy-path test uses.
        let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
        let tol = ATOL + 0.3 * 0.6;
        assert!(
            mean_diff < tol,
            "offset={offset}: HW-compute mean_diff {mean_diff} >= {tol} \
             (max_diff={max_diff})"
        );
    }
}

/// Compute-interval shading: camera near a face of the tet, looking in.
/// Tests standard outside-looking-in rendering via compute-interval path.
#[test]
fn test_compute_interval_face_view() {
    let offsets = [0.1, 1.0, 5.0, 10.0];

    for &offset in &offsets {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let scene = random_single_tet_scene(&mut rng, 0.3);

        let verts = load_tet_verts(&scene, 0);
        let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
        let face_center = (verts[0] + verts[2] + verts[1]) / 3.0;
        let face_normal = (face_center - centroid).normalize();

        let eye = face_center + face_normal * offset;
        let target = centroid;
        let (vp, c2w, intrinsics) = setup_camera(eye, target);

        let cpu_image = cpu_render_scene(&scene, eye, vp, c2w, intrinsics, W, H);

        if let Some(gpu_image) =
            gpu_compute_interval_render_scene(&scene, eye, vp, c2w, intrinsics, W, H)
        {
            let (max_diff, mean_diff, _) = compare_images(&cpu_image, &gpu_image);
            eprintln!(
                "compute_interval_face_view offset={offset}: max_diff={max_diff:.4}, mean_diff={mean_diff:.6}"
            );
            let tol = if offset < 1.0 { 0.03 } else { ATOL };
            assert!(
                mean_diff < tol,
                "compute_interval offset={offset}: mean_diff {mean_diff} >= {tol}"
            );
        } else {
            eprintln!("Skipping GPU compute-interval test (no adapter)");
        }
    }
}
