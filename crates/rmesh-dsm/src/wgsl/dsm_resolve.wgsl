// DSM resolve: reconstructs transmittance T(z_query) from the stored depth
// moments for debug visualization. Uses the same Cantelli (one-sided
// Chebyshev) bound as `evaluate_transmittance` in deferred_shade_frag.wgsl, so
// this preview matches the shadows actually cast.
//
// Reads one moment texture (.r = alpha*E[z], .g = alpha*E[z^2], .a = alpha)
// and a uniform z_query depth, outputs grayscale T.

struct ResolveUniforms {
    z_query: f32,  // metric (linear) depth
    near: f32,
    far: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: ResolveUniforms;
@group(0) @binding(1) var moment_tex: texture_2d<f32>;

// Must match the tunables in deferred_shade_frag.wgsl::evaluate_transmittance.
const SHADOW_MIN_VARIANCE: f32 = 4.0e-4;
const SHADOW_LBR: f32          = 0.15;

fn linstep(lo: f32, hi: f32, v: f32) -> f32 {
    return clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<u32>(u32(frag_coord.x), u32(frag_coord.y));
    let m = textureLoad(moment_tex, coords, 0);

    let alpha = m.a;
    if (alpha < 0.01) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0); // no occluder → fully lit
    }

    // Normalize query depth to [0,1].
    let z = clamp((params.z_query - params.near) / (params.far - params.near), 0.0, 1.0);

    let inv_alpha = 1.0 / alpha;
    let mean = m.r * inv_alpha;        // E[z]
    let e_z2 = m.g * inv_alpha;        // E[z²]
    if (z <= mean) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let variance = max(e_z2 - mean * mean, SHADOW_MIN_VARIANCE);
    let d = z - mean;
    var p_max = variance / (variance + d * d);
    p_max = linstep(SHADOW_LBR, 1.0, p_max);
    let t_total = 1.0 - alpha;
    let t = max(p_max, t_total);

    return vec4<f32>(t, t, t, 1.0);
}
