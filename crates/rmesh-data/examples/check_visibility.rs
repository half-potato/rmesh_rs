//! Headless load-check for the visibility (occlusion PVS) tagged sections.
//! Usage: cargo run -p rmesh-data --example check_visibility -- <file.rmesh>
fn main() {
    let path = std::env::args().nth(1).expect("usage: check_visibility <file.rmesh>");
    let data = std::fs::read(&path).expect("read file");
    let (scene, _sh, pbr) = rmesh_data::load_rmesh(&data).expect("load_rmesh");
    println!(
        "loaded: tets={} verts={} pbr={}",
        scene.tet_count, scene.vertex_count, pbr.is_some()
    );
    match scene.visibility {
        Some(v) => {
            let ones: u64 = v.masks.iter().map(|&b| b as u64).sum();
            println!(
                "visibility OK: regions={} tet_count={} masks_len={} (expect {}*{}={}) ones={} \
                 centroids_len={} radii_len={}",
                v.num_regions,
                v.tet_count,
                v.masks.len(),
                v.num_regions,
                v.tet_count,
                v.num_regions as usize * v.tet_count as usize,
                ones,
                v.region_centroids.len(),
                v.region_radii.len()
            );
            assert_eq!(v.tet_count, scene.tet_count, "vis tet_count must match scene");
            assert_eq!(
                v.masks.len(),
                v.num_regions as usize * v.tet_count as usize,
                "mask size must be K*M"
            );
            // Per-region visible fraction (sanity: should vary across regions).
            let m = v.tet_count as usize;
            for k in 0..v.num_regions as usize {
                let c: u64 = v.masks[k * m..(k + 1) * m].iter().map(|&b| b as u64).sum();
                println!("  region {k}: {:.3} visible", c as f64 / m as f64);
            }
        }
        None => panic!("NO visibility data parsed"),
    }
}
