//! Inspect authoritative energy data, optionally export full-precision CLI input.
use anyhow::{Context, Result};
use std::io::Write;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (scene, _, pbr) = rmesh_data::load_rmesh(&std::fs::read(
        args.get(1)
            .context("Usage: inspect_energy scene.rmesh [output.rmlit]")?,
    )?)?;
    let pbr = pbr.context("Missing material")?;
    anyhow::ensure!(!pbr.energy_material.is_empty(), "Not an energy export");
    println!(
        "vertices={} tets={} zero_density={} min_t={:?} start_pose={:?}",
        scene.vertex_count,
        scene.tet_count,
        scene.densities.iter().filter(|&&d| d == 0.).count(),
        pbr.render_min_t,
        scene.start_pose
    );
    println!(
        "energy_rows={} geom_normals={} LUT={} density_sum={:.9}",
        pbr.energy_material.len() / 5,
        pbr.geom_normals.len() / 3,
        pbr.energy_lut.len(),
        scene.densities.iter().map(|&x| x as f64).sum::<f64>()
    );
    if let Some(path) = args.get(2) {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for v in scene.vertices.chunks_exact(3) {
            for c in 0..3 {
                lo[c] = lo[c].min(v[c]);
                hi[c] = hi[c].max(v[c]);
            }
        }
        let eps = ((0..3).map(|c| (hi[c] - lo[c]).powi(2)).sum::<f32>().sqrt() * 1e-7).max(1e-7);
        f.write_all(b"RMLIT001")?;
        for word in [scene.vertex_count, scene.tet_count, 4, eps.to_bits()] {
            f.write_all(&word.to_le_bytes())?;
        }
        for data in [&scene.vertices, &pbr.geom_normals] {
            f.write_all(bytemuck::cast_slice(data))?;
        }
        f.write_all(bytemuck::cast_slice(&scene.indices))?;
        f.write_all(bytemuck::cast_slice(&scene.densities))?;
        for i in 0..scene.tet_count as usize {
            let mut row = [0f32; 9];
            row[0] = pbr.roughness[i];
            row[1..4].copy_from_slice(&pbr.albedo[3 * i..3 * i + 3]);
            row[4..].copy_from_slice(&pbr.energy_material[5 * i..5 * i + 5]);
            f.write_all(bytemuck::bytes_of(&row))?;
        }
        f.flush()?;
    }
    Ok(())
}
