// DSM resolve (debug visualization): reads a single 2-moment shadow map texel
// and reconstructs T(z_query) via the Cantelli (one-sided Chebyshev) bound,
// matching deferred_shade_frag.wgsl::evaluate_transmittance.
//
// Storage layout (from dsm_moment_fragment.wgsl):
//   .r = E[α·z], .g = E[α·z²], .b = 0, .a = α

struct ResolveUniforms {
    z_query: f32,  // metric (linear) depth
    near: f32,
    far: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: ResolveUniforms;
@group(0) @binding(1) var moments_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<u32>(u32(frag_coord.x), u32(frag_coord.y));
    let m = textureLoad(moments_tex, coords, 0);

    let shadow_alpha = m.a;
    if (shadow_alpha < 0.01) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let inv_alpha = 1.0 / shadow_alpha;
    let mean = m.r * inv_alpha;

    // z_query normalized to the same [0,1] space the fragment shader stored.
    let z = clamp((params.z_query - params.near) / (params.far - params.near), 0.0, 1.0);
    if (z <= mean) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let variance = max(m.g - m.r * m.r, 3e-5) * inv_alpha;
    let d = z - mean;
    let p_max = variance / (variance + d * d);

    let T_total = 1.0 - shadow_alpha;
    let T = max(p_max, T_total);
    return vec4<f32>(T, T, T, 1.0);
}
