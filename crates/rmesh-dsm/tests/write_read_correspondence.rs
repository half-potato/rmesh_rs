//! Write/read orientation correspondence for the DSM cubemap.
//!
//! The DSM is *written* by rasterizing geometry with `build_light_vp` (view +
//! perspective + clip-space Y-flip) into a per-face 2D texture, then copied to
//! cube layer `fi`. It is *read* by `textureSample(cube, dir)`, where the GPU
//! maps `dir` → (face, u, v) via the fixed WebGPU/OpenGL major-axis convention.
//!
//! If those two disagree for any face, shadows on that face land at the wrong
//! texel — a genuine orientation discrepancy. This test reproduces both sides
//! analytically (no GPU) and asserts they agree. It calls the REAL
//! `build_light_vp`, so it tracks whatever that function currently does.
//!
//! Texel convention (both sides): u right, v down, v=0 = top row.
//!  - write: a fragment at NDC (x,y) lands at u=(x+1)/2, v=(1-y)/2
//!  - read : u=(sc/|ma|+1)/2, v=(tc/|ma|+1)/2  (cube sc/tc per face)

use glam::{Vec3, Vec4};
use rmesh_render::GpuLight;

const NEAR: f32 = 0.05;
const FAR: f32 = 50.0;

/// WebGPU/OpenGL cube convention: direction → (face, u, v). Mirrors the table
/// in `CUBEMAP_DIRS` / `select_cubemap_face`. v=0 is the top row.
fn cube_read(d: Vec3) -> (usize, f32, f32) {
    let a = d.abs();
    let (face, sc, tc, ma);
    if a.x >= a.y && a.x >= a.z {
        ma = a.x;
        if d.x > 0.0 {
            (face, sc, tc) = (0, -d.z, -d.y); // +X
        } else {
            (face, sc, tc) = (1, d.z, -d.y); // -X
        }
    } else if a.y >= a.z {
        ma = a.y;
        if d.y > 0.0 {
            (face, sc, tc) = (2, d.x, d.z); // +Y
        } else {
            (face, sc, tc) = (3, d.x, -d.z); // -Y
        }
    } else {
        ma = a.z;
        if d.z > 0.0 {
            (face, sc, tc) = (4, d.x, -d.y); // +Z
        } else {
            (face, sc, tc) = (5, -d.x, -d.y); // -Z
        }
    }
    let u = 0.5 * (sc / ma + 1.0);
    let v = 0.5 * (tc / ma + 1.0);
    (face, u, v)
}

/// Where `build_light_vp(face)` actually stores a world direction `d` (light at
/// origin): project to clip, perspective-divide, map NDC → texel (v=0 top).
fn cube_write(light: &GpuLight, face: usize, d: Vec3) -> (f32, f32) {
    let (vp, _) = rmesh_dsm::build_light_vp(light, face, NEAR, FAR);
    let clip = vp * Vec4::new(d.x, d.y, d.z, 1.0);
    let ndc = Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    let u = 0.5 * (ndc.x + 1.0);
    let v = 0.5 * (1.0 - ndc.y);
    (u, v)
}

#[test]
fn write_matches_read_all_faces() {
    let light = GpuLight {
        position: [0.0, 0.0, 0.0],
        light_type: 0,
        ..Default::default()
    };

    // Per face: forward axis + its two tangential world axes.
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::Y, Vec3::Z),         // +X
        (Vec3::NEG_X, Vec3::Y, Vec3::Z),     // -X
        (Vec3::Y, Vec3::X, Vec3::Z),         // +Y
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),     // -Y
        (Vec3::Z, Vec3::X, Vec3::Y),         // +Z
        (Vec3::NEG_Z, Vec3::X, Vec3::Y),     // -Z
    ];

    let offsets = [-0.5f32, -0.25, 0.0, 0.25, 0.5];
    let mut worst = 0.0f32;
    let mut failures = 0usize;

    for (fi, (fwd, ta, tb)) in faces.iter().enumerate() {
        for &s in &offsets {
            for &t in &offsets {
                // Asymmetric off-center direction inside this face's cone.
                let d = *fwd + *ta * s + *tb * t;

                // Sanity: the convention must agree this direction is on face fi
                // (forward component dominates, so it always should).
                let (rface, ru, rv) = cube_read(d);
                assert_eq!(
                    rface, fi,
                    "convention put dir {d:?} on face {rface}, expected {fi}"
                );

                let (wu, wv) = cube_write(&light, fi, d);
                let du = (wu - ru).abs();
                let dv = (wv - rv).abs();
                worst = worst.max(du).max(dv);
                if du > 1e-4 || dv > 1e-4 {
                    failures += 1;
                    eprintln!(
                        "face {fi} s={s:+.2} t={t:+.2}: write=({wu:.4},{wv:.4}) read=({ru:.4},{rv:.4})  Δ=({du:.4},{dv:.4})"
                    );
                }
            }
        }
    }

    eprintln!("worst |write-read| texel error = {worst:.6}");
    assert_eq!(
        failures, 0,
        "{failures} write/read mismatches — orientation discrepancy exists"
    );
}
