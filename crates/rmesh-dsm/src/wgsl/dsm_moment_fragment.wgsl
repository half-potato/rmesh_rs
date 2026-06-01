// Deep-shadow-map moment fragment shader (tet intervals).
//
// Accumulates the two depth moments of the alpha-weighted light-side
// termination distribution into a single RGBA16Float target, composited via
// hardware premultiplied-alpha blending (back-to-front). The deferred pass
// reconstructs transmittance from these via a Cantelli/Chebyshev bound — see
// `evaluate_transmittance` in deferred_shade_frag.wgsl.
//
// Output channel layout (premultiplied by per-fragment alpha):
//   .r = alpha * E[z]    (mean depth,    normalized to [0,1] over [near,far])
//   .g = alpha * E[z^2]  (second moment, same normalization)
//   .b = unused
//   .a = alpha           (occlusion; accumulates to 1 - T_total)

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    c2w_col0: vec4<f32>,
    c2w_col1: vec4<f32>,
    c2w_col2: vec4<f32>,
    intrinsics: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    step: u32,
    tile_size_u: u32,
    ray_mode: u32,
    min_t: f32,
    sh_degree: u32,
    near_plane: f32,
    far_plane: f32,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;

struct FragmentInput {
    @location(0) depths: vec2<f32>,
    @location(1) color_offsets: vec2<f32>,
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
};

// phi(x) = (1 - exp(-x)) / x
// Taylor with 4 terms for |x| < 0.02 avoids catastrophic cancellation.
fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return (1.0 - exp(-x)) / x;
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>, in: FragmentInput) -> @location(0) vec4<f32> {
    let near = uniforms.near_plane;
    let far = uniforms.far_plane;
    let range = far - near;

    // Linearize NDC depths to view-space Z
    let z_f = near * far / (far - clamp(in.depths.x, 0.0, 1.0) * range);
    let z_b = near * far / (far - clamp(in.depths.y, 0.0, 1.0) * range);

    // Ray direction scale factor
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let x_cam = (frag_coord.x - uniforms.intrinsics.z) / fx;
    let y_cam = (frag_coord.y - uniforms.intrinsics.w) / fy;
    let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

    // Opacity of this interval
    let dist = abs(z_b - z_f) * ray_scale;
    let od = clamp(in.density * dist, 0.0, 88.0);
    let alpha = 1.0 - exp(-od);

    // Normalize view-space Z to [0,1] (same as forward pass)
    let za = clamp((z_f - near) / range, 0.0, 1.0);
    let zb = clamp((z_b - near) / range, 0.0, 1.0);

    // Expected termination depth (volume-rendering weighted, same as forward pass)
    let alpha_t = exp(-od);
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;  // back weight
    let w1 = 1.0 - phi_val;      // front weight
    let depth_premul = w0 * zb + w1 * za;          // alpha * E[z]
    let depth_sq_premul = w0 * zb * zb + w1 * za * za; // alpha * E[z^2]

    return vec4<f32>(depth_premul, depth_sq_premul, 0.0, alpha);
}
