// Interval shading fragment shader.
//
// Linearizes interpolated NDC depths to view-space Z, computes ray segment
// distance, and evaluates the volume rendering integral.
//
// MRT outputs (all Rgba16Float, .a = alpha for correct hardware blend):
//   location(0): albedo.rgb * a, a
//   location(1): roughness * a, f0_dielectric * a, metallic * a, a
//   location(2): field_gradient.xyz * a, a  — raw, normalized only in deferred
//   location(3): expected_depth * a, 0, expected_z2 * a, a
//
// Channel order in slot 1 matches the glTF primitive_mrt convention so the
// deferred shader can read metallic from .b uniformly across volumes and
// primitives.

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
@group(0) @binding(3) var<storage, read> aux_data: array<f32>;        // [M * AUX_DIM]
@group(0) @binding(4) var<storage, read> vertex_normals: array<f32>;  // [V * 3]
@group(0) @binding(5) var<storage, read> tet_indices: array<u32>;     // [M * 4]

const AUX_DIM: u32 = 6u;

struct FragmentInput {
    @location(0) depths: vec2<f32>,                  // (z_front_ndc, z_back_ndc)
    @location(1) color_offsets: vec2<f32>,            // (offset_front, offset_back)
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
    @location(5) field_gradient: vec3<f32>,          // interpolated raw gradient, normalize in pixel
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
    @location(2) normals: vec4<f32>,
    @location(3) expected_depth: vec4<f32>,
};

// phi(x) = (1 - exp(-x)) / x
// Taylor with 4 terms for |x| < 0.02 avoids catastrophic cancellation.
fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return (1.0 - exp(-x)) / x;
}

// Volume rendering integral for a ray segment through a tet.
fn compute_integral(c0: vec3<f32>, c1: vec3<f32>, optical_depth: f32) -> vec4<f32> {
    let alpha = exp(-optical_depth);
    let phi_val = phi(optical_depth);
    let w0 = phi_val - alpha;
    let w1 = 1.0 - phi_val;
    let c = c0 * w0 + c1 * w1;
    return vec4<f32>(c, 1.0 - alpha);
}

@fragment
fn main(@builtin(position) frag_coord: vec4<f32>, in: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    let near = uniforms.near_plane;
    let far = uniforms.far_plane;

    // Linearize interpolated NDC depths -> view-space Z (positive = into screen)
    let range = far - near;
    let z_front_clamped = clamp(in.depths.x, 0.0, 1.0);
    let z_back_clamped = clamp(in.depths.y, 0.0, 1.0);
    let z_f = near * far / (far - z_front_clamped * range);
    let z_b = near * far / (far - z_back_clamped * range);

    // Reconstruct ray direction scale factor from pixel position
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let cx = uniforms.intrinsics.z;
    let cy = uniforms.intrinsics.w;

    let x_cam = (frag_coord.x - cx) / fx;
    let y_cam = (frag_coord.y - cy) / fy;
    let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

    // Distance through tet = |z_b - z_f| * ray_scale
    let dist = abs(z_b - z_f) * ray_scale;

    // Colors at entry/exit from interpolated offsets
    let c_front = max(in.base_color + vec3<f32>(in.color_offsets.x), vec3<f32>(0.0));
    let c_back = max(in.base_color + vec3<f32>(in.color_offsets.y), vec3<f32>(0.0));

    // Volume rendering integral
    let od = clamp(in.density * dist, 0.0, 88.0);
    let alpha_t = exp(-od);
    let alpha = 1.0 - alpha_t;

    // Per-tet aux channels lookup (parametric BRDF layout, AUX_DIM=6).
    // For non-PBR scenes, aux_data is bound to a 4-byte dummy buffer (see
    // `create_compute_interval_render_bind_group`); detect that and use the
    // proper volume-integrated SH color for `out.color` so the color-only
    // path matches what the Regular forward pass produces.
    let has_pbr = arrayLength(&aux_data) > 1u;
    let aux_base = in.tet_id * AUX_DIM;
    let roughness     = select(0.5,  aux_data[aux_base + 0u], has_pbr);
    let metallic      = select(0.0,  aux_data[aux_base + 1u], has_pbr);
    let f0_dielectric = select(0.04, aux_data[aux_base + 2u], has_pbr);
    let albedo        = select(in.base_color,
                               vec3f(aux_data[aux_base + 3u],
                                     aux_data[aux_base + 4u],
                                     aux_data[aux_base + 5u]),
                               has_pbr);

    // Slot 0: color (premultiplied alpha).
    //   PBR: flat per-tet albedo × α (deferred shader applies lighting).
    //   non-PBR: volume integral of the SH color through the tet, using the
    //   same w0/w1 transmittance weighting as `forward_fragment.wgsl` so the
    //   interval render matches the Regular path.
    // Shared w0/w1 across the color (slot 0) and depth-moment (slot 3) outputs —
    // both blends use the same exponential-transmittance weights, so phi(od)
    // only needs evaluating once.
    let phi_val = phi(od);
    let w0 = phi_val - alpha_t;   // back weight (z_b)
    let w1 = 1.0 - phi_val;       // front weight (z_f)
    let integrated = c_back * w0 + c_front * w1;  // already × α
    out.color = select(vec4f(integrated, alpha),
                       vec4f(albedo * alpha, alpha),
                       has_pbr);

    // Slot 1: PBR material (roughness, f0_dielectric, metallic) — metallic in
    // .b to match primitive_mrt's glTF convention.
    out.aux0 = vec4f(roughness * alpha, f0_dielectric * alpha, metallic * alpha, alpha);

    // Slot 2: normals — Rgba8Unorm bias-encoded.
    // Pre-normalize the per-fragment gradient direction, then bias to [0,1] for
    // the Unorm target: (n*0.5+0.5)*alpha premultiplies. Hardware blend gives
    // Σ α_i·biased_i in rgb and Σ α_i in a; consumer divides, un-biases (*2-1),
    // and re-normalizes. Zero-gradient fragments contribute (0.5,0.5,0.5)*α
    // which decodes to (0,0,0) — a "no direction" vote that dilutes magnitude
    // without skewing direction. Guard normalize() against zero gradients.
    let g = in.field_gradient;
    let g_len = length(g);
    var n_unit = vec3f(0.0);
    if (g_len > 1e-6) {
        n_unit = g / g_len;
    }
    let n_biased = n_unit * 0.5 + 0.5;
    out.normals = vec4f(n_biased * alpha, alpha);

    // Slot 3: expected termination depth + E[z²] (for std). .g is unused
    // (formerly retro * alpha; left at 0). .b holds the second-moment
    // contribution, composited via the same premul-alpha blend as .r.
    //
    // Exact within-segment α·E[z²] for the exponential termination distribution
    // on [z_f, z_b] (PDF f(z) = σ_eff · exp(−σ_eff(z−z_f))):
    //   α·E[z²] = w0·z_b² + w1·z_f² − L² · correction(od)
    // where correction(od) = w0 + α_t − 2·w0/od ≥ 0 measures how much the
    // 2-point upper bound at the extremes overshoots the true exponential.
    // For small od the direct form has catastrophic cancellation; Taylor:
    //   correction(od) ≈ od/6 − od²/12 + od³/40
    let depth_premul = w0 * z_b + w1 * z_f;

    var correction: f32;
    if (abs(od) < 0.02) {
        correction = od * ((1.0/6.0) - od * ((1.0/12.0) - od * (1.0/40.0)));
    } else {
        correction = w0 + alpha_t - 2.0 * w0 / od;
    }
    let L = z_b - z_f;
    let depth2_premul = w0 * z_b * z_b + w1 * z_f * z_f - L * L * correction;

    out.expected_depth = vec4f(depth_premul, 0.0, depth2_premul, alpha);

    return out;
}
