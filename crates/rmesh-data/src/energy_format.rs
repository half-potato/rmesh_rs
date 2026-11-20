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
    if data.starts_with(b"RMESH002") {
        ensure!(p == data.len(), "Trailing or truncated RMESH002 data");
        for tag in ["vertex_normals", "env_feature", "plaster_rgb", "plaster_grad", "metallic", "f0_dielectric"] {
            ensure!(!sections.contains_key(tag), "Legacy field {tag} is not part of RMESH002");
        }
        ensure!(sections.contains_key("tone_curve"), "Missing tone_curve");
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
        version == [1, 4, 1] || (version == [1, 4, 2] && data.starts_with(b"RMESH002")),
        "Unsupported energy version {version:?}; expected [1,4,1] or RMESH002 [1,4,2]"
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
    let material_values = |tag: &str, shape: &[usize]| -> Result<Vec<f32>> {
        if version == [1, 4, 1] { return values(tag, shape); }
        let s = sections.get(tag).with_context(|| format!("Missing energy section {tag}"))?;
        ensure!(s.dtype == 2 && s.shape == shape, "Invalid UNORM8 dtype/shape for {tag}");
        Ok(s.bytes.iter().map(|&x| {
            let value = x as f32 / 255.0;
            if tag == "roughness" { 0.04 + 0.96 * value } else { value }
        }).collect())
    };
    let roughness = material_values("roughness", &[nt])?;
    let f0 = material_values("fresnel_rgb", &[nt, 3])?;
    let retro = material_values("backscatter", &[nt])?;
    let albedo = material_values("albedo", &[nt, 3])?;
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
    if sections.contains_key("vertex_normals") {
        values("vertex_normals", &[nv, 3])?;
    }
    // Older exports include captured-illumination fields unused by the
    // explicit-light energy renderer. Compact exports omit these sections.
    // Still validate them when present, preserving malformed-file checks.
    for (tag, channels) in [("env_feature", 4), ("plaster_rgb", 3), ("plaster_grad", 3)] {
        if sections.contains_key(tag) {
            values(tag, &[nt, channels])?;
        }
    }
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
        fixture_with_optional_fields(version, density, true)
    }
    fn fixture_with_optional_fields(version: [i32; 3], density: f32, full: bool) -> Vec<u8> {
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
        if !full {
            tags.retain(|(tag, _, _, _)| !matches!(*tag, "env_feature" | "plaster_rgb" | "plaster_grad"));
        }
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
    fn compact_energy_preserves_render_inputs() {
        let (old_scene, _, old_pbr) = crate::load_rmesh_raw(&fixture([1, 4, 1], 2.)).unwrap();
        let (scene, _, pbr) = crate::load_rmesh_raw(
            &fixture_with_optional_fields([1, 4, 1], 2., false)).unwrap();
        let old = old_pbr.unwrap();
        let new = pbr.unwrap();
        assert_eq!(scene.densities, old_scene.densities);
        assert_eq!(scene.indices, old_scene.indices);
        assert_eq!(new.energy_material, old.energy_material);
        assert_eq!(new.albedo, old.albedo);
        assert_eq!(new.geom_normals, old.geom_normals);
        assert_eq!(new.render_min_t, old.render_min_t);
        assert!(new.env_feature.is_empty());
    }
    fn v2_fixture(quantized: bool) -> Vec<u8> {
        let legacy = fixture([1, 4, 1], 2.);
        // Fixture: 48-byte header, four vertices, one tet, four density bytes.
        let mut v2 = b"RMESH002".to_vec();
        v2.extend(&legacy[..8]);
        v2.extend(&legacy[16..112]);
        let mut p = 116;
        let count = u32::from_le_bytes(legacy[p+4..p+8].try_into().unwrap());
        p += 8;
        let mut records = vec![];
        for _ in 0..count {
            let start = p;
            let tag = std::str::from_utf8(&legacy[p..p+16]).unwrap().trim_end_matches('\0');
            let rank = u32::from_le_bytes(legacy[p+20..p+24].try_into().unwrap()) as usize;
            p += 24 + 4*rank;
            let size = u32::from_le_bytes(legacy[p..p+4].try_into().unwrap()) as usize;
            p = (p + 4 + size + 3) & !3;
            if !matches!(tag, "vertex_normals" | "env_feature" | "plaster_rgb" | "plaster_grad") {
                let mut record = legacy[start..p].to_vec();
                if quantized && matches!(tag, "albedo" | "roughness" | "fresnel_rgb" | "backscatter") {
                    let payload: &[u8] = match tag {
                        "roughness" => &[0], "backscatter" => &[255], _ => &[0,128,255],
                    };
                    record.truncate(24+4*rank);
                    record[16..20].copy_from_slice(&2u32.to_le_bytes());
                    record.extend((payload.len() as u32).to_le_bytes());
                    record.extend(payload);
                    while record.len()%4 != 0 { record.push(0); }
                }
                if quantized && tag == "energy_version" { record[40..44].copy_from_slice(&2i32.to_le_bytes()); }
                records.push(record);
            }
        }
        let mut tone = b"tone_curve\0\0\0\0\0\0".to_vec();
        for x in [0u32, 1, 5, 10] { tone.extend(x.to_le_bytes()); }
        tone.extend([0u8; 12]);
        records.push(tone);
        v2.extend(0x524d5458u32.to_le_bytes());
        v2.extend((records.len() as u32).to_le_bytes());
        for record in records { v2.extend(record); }
        v2
    }
    #[test]
    fn v2_unorm8_materials_decode_and_reject_mismatched_encoding() {
        let bytes = v2_fixture(true);
        let (scene, _, pbr) = crate::load_rmesh_raw(&bytes).unwrap();
        let pbr = pbr.unwrap();
        assert_eq!(scene.densities, [2.]);
        assert_eq!(pbr.albedo, [0., 128./255., 1.]);
        assert_eq!(pbr.roughness, [0.04]);
        assert_eq!(pbr.energy_material, [0.04, 0., 128./255., 1., 1.]);
        let mut wrong = bytes;
        let tag = wrong.windows(16).position(|w| w == b"energy_version\0\0").unwrap();
        wrong[tag+40..tag+44].copy_from_slice(&1i32.to_le_bytes());
        assert!(crate::load_rmesh_raw(&wrong).is_err());
    }
    #[test]
    fn v2_container_has_only_authoritative_fields() {
        let v2 = v2_fixture(false);
        let (scene, sh, pbr) = crate::load_rmesh_raw(&v2).unwrap();
        assert_eq!(scene.densities, [2.]);
        assert!(scene.color_grads.iter().all(|&x| x == 0.));
        assert!(sh.coeffs.iter().all(|&x| x == 0.));
        let pbr = pbr.unwrap();
        assert_eq!(pbr.vertex_normals, pbr.geom_normals);
        assert!(pbr.env_feature.is_empty());
        assert!(pbr.metallic.is_empty());
        let mut bad = v2.clone();
        bad[7] = b'3';
        assert!(crate::load_rmesh_raw(&bad).is_err());
        assert!(crate::load_rmesh_raw(&v2[..v2.len()-1]).is_err());
        // Geometry indices are rejected before circumsphere computation.
        let mut bad = v2;
        bad[96..100].copy_from_slice(&99u32.to_le_bytes());
        assert!(crate::load_rmesh_raw(&bad).is_err());
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
