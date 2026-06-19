// DSM primitive pass: projects opaque primitives from the light viewpoint and
// writes α=1 absorber moments at the primitive's NDC depth. Single MRT output
// matching dsm_moment_fragment.wgsl: (E[α·z], E[α·z²], 0, α).

struct PrimitiveUniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    model_col0: vec4<f32>,
    model_col1: vec4<f32>,
    model_col2: vec4<f32>,
    model_col3: vec4<f32>,
    color: vec4<f32>,  // (near, far, width, _)
}

@group(0) @binding(0) var<uniform> u: PrimitiveUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VsOut {
    let model = mat4x4<f32>(u.model_col0, u.model_col1, u.model_col2, u.model_col3);
    let vp = mat4x4<f32>(u.vp_col0, u.vp_col1, u.vp_col2, u.vp_col3);
    var out: VsOut;
    out.pos = vp * model * vec4<f32>(position, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert NDC depth to linear view-space Z, normalize to [0,1]
    let near = u.color.x;
    let far = u.color.y;
    let z_linear = near * far / (far - frag_coord.z * (far - near));
    let z = clamp((z_linear - near) / (far - near), 0.0, 1.0);

    // Opaque blocker: α=1 absorber concentrated at depth z.
    return vec4<f32>(z, z * z, 0.0, 1.0);
}
