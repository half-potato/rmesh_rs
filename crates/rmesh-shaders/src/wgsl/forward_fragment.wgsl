// Forward fragment shader: volume rendering integral with MRT output.
//
// Computes ray-tet intersection, evaluates the volume integral with linear color
// interpolation, and outputs premultiplied alpha for back-to-front blending.
//
// MRT outputs:
//   location(0): premultiplied color (RGBA)
//   location(1): auxiliary float4 (t_enter, t_exit, optical_depth, dist)

struct FragmentInput {
    @location(0) @interpolate(flat) tet_density: f32,
    @location(1) @interpolate(flat) base_color: vec3<f32>,
    @location(2) plane_numerators: vec4<f32>,
    @location(3) plane_denominators: vec4<f32>,
    @location(4) ray_dir: vec3<f32>,
    @location(5) @interpolate(flat) dc_dt: f32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux0: vec4<f32>,
};

// φ(x) = (1 - exp(-x)) / x
// Numerically stable: if |x| < 1e-6, use Taylor expansion φ ≈ 1 - x/2
fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) {
        return 1.0 - x * 0.5;
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
fn main(in: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    let base_color = in.base_color;
    let d = length(in.ray_dir);
    let plane_denom = in.plane_denominators / d;
    let dc_dt = in.dc_dt / d;
    let all_t = in.plane_numerators / plane_denom;

    let neg_inf = vec4<f32>(-3.402823e38);
    let pos_inf = vec4<f32>(3.402823e38);

    // Classify planes: denom > 0 → entering, denom < 0 → exiting
    let t_enter = vec4<f32>(
        select(neg_inf.x, all_t.x, plane_denom.x > 0.0),
        select(neg_inf.y, all_t.y, plane_denom.y > 0.0),
        select(neg_inf.z, all_t.z, plane_denom.z > 0.0),
        select(neg_inf.w, all_t.w, plane_denom.w > 0.0),
    );
    let t_exit = vec4<f32>(
        select(pos_inf.x, all_t.x, plane_denom.x < 0.0),
        select(pos_inf.y, all_t.y, plane_denom.y < 0.0),
        select(pos_inf.z, all_t.z, plane_denom.z < 0.0),
        select(pos_inf.w, all_t.w, plane_denom.w < 0.0),
    );

    let t_min = max(max(t_enter.x, t_enter.y), max(t_enter.z, t_enter.w));
    let t_max = min(min(t_exit.x, t_exit.y), min(t_exit.z, t_exit.w));

    let dist = max(t_max - t_min, 0.0);
    let od = in.tet_density * dist;

    let c_start = max(base_color + vec3<f32>(dc_dt * t_min), vec3<f32>(0.0));
    let c_end = max(base_color + vec3<f32>(dc_dt * t_max), vec3<f32>(0.0));

    // Note: (c_end, c_start) — exit color first, matching webrm convention
    out.color = compute_integral(c_end, c_start, od);
    out.aux0 = vec4<f32>(t_min, t_max, od, dist);

    return out;
}
