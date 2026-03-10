//! naga_oil Composer helper for shared WGSL modules.
//!
//! Provides shared WGSL modules (structs, math, SH constants, intersection helpers)
//! and a helper to create composed shader modules via naga_oil's Composer.

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderLanguage, ShaderType,
};

const COMMON_WGSL: &str = include_str!("wgsl/common.wgsl");
const MATH_WGSL: &str = include_str!("wgsl/math.wgsl");
const SH_WGSL: &str = include_str!("wgsl/sh.wgsl");
const INTERSECT_WGSL: &str = include_str!("wgsl/intersect.wgsl");

/// Create a Composer pre-loaded with all shared rmesh WGSL modules.
///
/// Modules available for import:
/// - `rmesh::common` — Uniforms, TileUniforms, DrawIndirectArgs
/// - `rmesh::math` — softplus, dsoftplus, phi, dphi_dx, project_to_ndc
/// - `rmesh::sh` — SH basis constants + eval_sh
/// - `rmesh::intersect` — TET_FACES constant, vertex loaders
pub fn create_composer() -> Result<Composer, String> {
    let mut composer = Composer::default();

    let modules = [
        ("rmesh::common", COMMON_WGSL),
        ("rmesh::math", MATH_WGSL),
        ("rmesh::sh", SH_WGSL),
        ("rmesh::intersect", INTERSECT_WGSL),
    ];

    for (name, source) in modules {
        composer
            .add_composable_module(ComposableModuleDescriptor {
                source,
                file_path: name,
                language: ShaderLanguage::Wgsl,
                ..Default::default()
            })
            .map_err(|e| format!("Failed to add module {name}: {e:?}"))?;
    }

    Ok(composer)
}

/// Compose a WGSL shader source (which may `#import` shared modules) into a
/// `wgpu::ShaderModule`.
///
/// The source can use `#import rmesh::common`, `#import rmesh::math`, etc.
pub fn create_shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> Result<wgpu::ShaderModule, String> {
    let mut composer = create_composer()?;

    let module = composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: label,
            shader_type: ShaderType::Wgsl,
            ..Default::default()
        })
        .map_err(|e| format!("Failed to compose shader {label}: {e:?}"))?;

    // Convert the composed naga module back to WGSL text, since wgpu 28
    // no longer accepts ShaderSource::Naga directly.
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("Validation error in {label}: {e:?}"))?;

    let wgsl_text = naga::back::wgsl::write_string(
        &module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|e| format!("WGSL write error in {label}: {e:?}"))?;

    Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl_text.into()),
    }))
}
