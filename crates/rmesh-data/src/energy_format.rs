//! Strict decoder for the versioned energy export. Legacy readers must not
//! silently render a newer energy model with old PBR semantics.
use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;

pub struct EnergyData {
    pub density: Vec<f32>,
    pub material: Vec<f32>,
    pub geom_normals: Vec<f32>,
    pub lut: Vec<f32>,
    pub min_t: f32,
}
struct Section<'a> {
    dtype: u32,
    shape: Vec<usize>,
    bytes: &'a [u8],
}
pub fn parse(data: &[u8], offset: usize, nv: usize, nt: usize) -> Result<Option<EnergyData>> {
    let mut p = offset;
    fn word(data: &[u8], p: &mut usize) -> Result<u32> {
        let b = data
            .get(*p..*p + 4)
            .context("Truncated energy section header")?;
        *p += 4;
        Ok(u32::from_le_bytes(b.try_into()?))
    }
    ensure!(word(data, &mut p)? == 0x524d5458, "Invalid tagged magic");
    let count = word(data, &mut p)?;
    let mut sections = BTreeMap::new();
    for _ in 0..count {
        let tag = std::str::from_utf8(data.get(p..p + 16).context("Truncated section tag")?)?
            .trim_end_matches('\0');
        p += 16;
        let dtype = word(data, &mut p)?;
        let rank = word(data, &mut p)?;
        ensure!(rank <= 8, "Invalid rank for {tag}");
        let shape = (0..rank)
            .map(|_| word(data, &mut p).map(|v| v as usize))
            .collect::<Result<Vec<_>>>()?;
        let len = word(data, &mut p)? as usize;
        let bytes = data
            .get(p..p.checked_add(len).context("Section size overflow")?)
            .context("Truncated section payload")?;
        p = (p + len + 3) & !3;
        // Unknown sections are skipped by byte count, even for future dtypes.
        if dtype <= 3 {
            let width = match dtype {
                0 => 2,
                2 => 1,
                _ => 4,
            };
            let elements = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .context("Shape overflow")?;
            ensure!(
                elements.checked_mul(width) == Some(len),
                "Section size/shape mismatch: {tag}"
            );
        }
        ensure!(
            sections
                .insert(
                    tag,
                    Section {
                        dtype,
                        shape,
                        bytes
                    }
                )
                .is_none(),
            "Duplicate section: {tag}"
        );
    }
    let Some(version) = sections.get("energy_version") else {
        return Ok(None);
    };
    ensure!(
        version.dtype == 3 && version.shape == [3],
        "Invalid energy_version section"
    );
    let version: Vec<i32> = version
        .bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        version == [1, 4, 1],
        "Unsupported energy version {version:?}; expected [1,4,1]"
    );
    let values = |tag: &str, shape: &[usize]| -> Result<Vec<f32>> {
        let s = sections
            .get(tag)
            .with_context(|| format!("Missing energy section {tag}"))?;
        ensure!(
            s.dtype == 1 && s.shape == shape,
            "Invalid dtype/shape for {tag}"
        );
        let v: Vec<f32> = s
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(v.iter().all(|x| x.is_finite()), "Nonfinite {tag}");
        Ok(v)
    };
    let density = values("density_f32", &[nt])?;
    ensure!(density.iter().all(|&x| x >= 0.0), "Negative density");
    let roughness = values("roughness", &[nt])?;
    let f0 = values("fresnel_rgb", &[nt, 3])?;
    let retro = values("backscatter", &[nt])?;
    let albedo = values("albedo", &[nt, 3])?;
    ensure!(
        roughness.iter().all(|x| (0.04..=1.0).contains(x)),
        "Invalid energy roughness"
    );
    ensure!(
        f0.iter()
            .chain(&retro)
            .chain(&albedo)
            .all(|x| (0.0..=1.0).contains(x)),
        "Invalid energy material range"
    );
    let geom_normals = values("geom_normals", &[nv, 3])?;
    values("vertex_normals", &[nv, 3])?;
    values("env_feature", &[nt, 4])?;
    values("plaster_rgb", &[nt, 3])?;
    values("plaster_grad", &[nt, 3])?;
    let near = values("min_t", &[2])?;
    ensure!(near.iter().all(|&x| x >= 0.0), "Invalid min_t");
    let table = values("ggx_table", &[2, 65, 65])?;
    let avg = values("ggx_average", &[2, 65])?;
    let mut lut = Vec::with_capacity(65 * 65 * 4);
    for y in 0..65 {
        for x in 0..65 {
            lut.extend_from_slice(&[
                table[y * 65 + x],
                table[65 * 65 + y * 65 + x],
                avg[y],
                avg[65 + y],
            ]);
        }
    }
    if !sections.contains_key("energy_meta") {
        bail!("Missing energy_meta");
    }
    let mut material = Vec::with_capacity(nt * 5);
    for i in 0..nt {
        material.extend_from_slice(&[
            roughness[i],
            f0[3 * i],
            f0[3 * i + 1],
            f0[3 * i + 2],
            retro[i],
        ]);
    }
    Ok(Some(EnergyData {
        density,
        material,
        geom_normals,
        lut,
        min_t: near[1],
    }))
}

#[cfg(test)]
mod tests {
    fn fixture(version: [i32; 3], density: f32) -> Vec<u8> {
        let mut bytes = vec![];
        for x in [4u32, 1, 0, 0] {
            bytes.extend(x.to_le_bytes());
        }
        bytes.extend([0u8; 32]);
        for x in [0f32, 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.] {
            bytes.extend(x.to_le_bytes());
        }
        for x in [0u32, 1, 2, 3] {
            bytes.extend(x.to_le_bytes());
        }
        bytes.extend([0u8; 4]); // Compatibility density is deliberately wrong.
        let mut tags: Vec<(&str, u32, Vec<u32>, Vec<u8>)> = vec![];
        tags.push((
            "energy_version",
            3,
            vec![3],
            version.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        for (tag, shape, values) in [
            ("density_f32", vec![1], vec![density]),
            ("roughness", vec![1], vec![0.04]),
            ("fresnel_rgb", vec![1, 3], vec![0.1, 0.2, 0.3]),
            ("backscatter", vec![1], vec![0.2]),
            ("albedo", vec![1, 3], vec![0.5; 3]),
            ("geom_normals", vec![4, 3], vec![0.; 12]),
            ("vertex_normals", vec![4, 3], vec![0.; 12]),
            ("env_feature", vec![1, 4], vec![0.; 4]),
            ("plaster_rgb", vec![1, 3], vec![1.; 3]),
            ("plaster_grad", vec![1, 3], vec![0.; 3]),
            ("min_t", vec![2], vec![1.6; 2]),
            ("ggx_table", vec![2, 65, 65], vec![0.; 2 * 65 * 65]),
            ("ggx_average", vec![2, 65], vec![0.; 130]),
        ] {
            tags.push((
                tag,
                1,
                shape,
                values.iter().flat_map(|v: &f32| v.to_le_bytes()).collect(),
            ));
        }
        tags.push(("energy_meta", 2, vec![2], b"{}".to_vec()));
        tags.push(("future_scalar", 3, vec![], 7i32.to_le_bytes().to_vec()));
        bytes.extend(0x524d5458u32.to_le_bytes());
        bytes.extend((tags.len() as u32).to_le_bytes());
        for (tag, dtype, shape, data) in tags {
            let mut name = [0u8; 16];
            name[..tag.len()].copy_from_slice(tag.as_bytes());
            bytes.extend(name);
            bytes.extend(dtype.to_le_bytes());
            bytes.extend((shape.len() as u32).to_le_bytes());
            for d in shape {
                bytes.extend(d.to_le_bytes());
            }
            bytes.extend((data.len() as u32).to_le_bytes());
            bytes.extend(data);
            while bytes.len() % 4 != 0 {
                bytes.push(0);
            }
        }
        bytes
    }
    #[test]
    fn energy_overrides_density_without_reordering() {
        for density in [0., 1e-9, 123.456] {
            let (scene, _, pbr) = crate::load_rmesh_raw(&fixture([1, 4, 1], density)).unwrap();
            assert_eq!(scene.densities, [density]);
            assert_eq!(scene.indices, [0, 1, 2, 3]);
            let pbr = pbr.unwrap();
            assert_eq!(pbr.energy_material, [0.04, 0.1, 0.2, 0.3, 0.2]);
            assert_eq!(pbr.render_min_t, Some(1.6));
        }
    }
    #[test]
    fn rejects_unknown_semantics_and_invalid_density() {
        assert!(crate::load_rmesh_raw(&fixture([2, 4, 1], 1.))
            .err()
            .unwrap()
            .to_string()
            .contains("Unsupported energy version"));
        assert!(crate::load_rmesh_raw(&fixture([1, 3, 1], 1.)).is_err());
        assert!(crate::load_rmesh_raw(&fixture([1, 4, 1], f32::NAN)).is_err());
        assert!(crate::load_rmesh_raw(&fixture([1, 4, 1], -1.)).is_err());
        let mut broken = fixture([1, 4, 1], 1.);
        broken.truncate(broken.len() - 1);
        assert!(crate::load_rmesh_raw(&broken).is_err());
    }
}
