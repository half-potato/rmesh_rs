// Integrate perspective-weighted cell-corner radiance pre-interpolated at the
// interval vertices in interval_compute_energy.wgsl. Zero fragment storage reads.
// `Color3` is aliased to `vec3<f16>` when SHADER_F16 is enabled, else `vec3<f32>`.

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
    shell_cull_bias: f32,
    shell_cull_flags: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tet_data: array<vec4<f32>>;

struct EnergyIntervalVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) inv_z_back: f32,
    @location(1) c_front_s: Color3,
    @location(2) c_back_s: Color3,
    // (inv_w_min, density, coverage)
    @location(3) @interpolate(flat) params: vec3<f32>,
};

@vertex
fn vs_energy(@builtin(vertex_index) vid: u32) -> EnergyIntervalVsOut {
    let tet = vid / 5u;
    let pos_depth = verts[vid * 2u + 0u];
    let packed = bitcast<vec4<u32>>(verts[vid * 2u + 1u]);
    let td = tet_data[tet];

    let c0 = unpack2x16float(packed.x);
    let c1 = unpack2x16float(packed.y);
    let c2 = unpack2x16float(packed.z);

    let inv_near = 1.0 / uniforms.near_plane;
    let inv_far = 1.0 / uniforms.far_plane;

    var out: EnergyIntervalVsOut;
    out.position = vec4<f32>(pos_depth.xy, pos_depth.z, 1.0);
    out.inv_z_back = fma(pos_depth.w, inv_far - inv_near, inv_near);
    out.c_front_s = Color3(vec3<f32>(c0.x, c0.y, c1.x));
    out.c_back_s = Color3(vec3<f32>(c1.y, c2.x, c2.y));
    out.params = td.xyz;
    return out;
}

fn phi_from_alpha(x: f32, alpha: f32) -> f32 {
    if (x < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return alpha / x;
}

@fragment
fn main_color_only(in: EnergyIntervalVsOut) -> @location(0) vec4<f32> {
    let inv_near = 1.0 / uniforms.near_plane;
    let inv_far = 1.0 / uniforms.far_plane;

    // Hardware clips 0 <= position.z <= 1, so z_f >= near always.
    let z_f_raw = 1.0 / fma(in.position.z, inv_far - inv_near, inv_near);
    let z_b = 1.0 / max(in.inv_z_back, inv_far);

    let ray_xy = (in.position.xy - uniforms.intrinsics.zw) / uniforms.intrinsics.xy;
    let ray_scale = sqrt(dot(ray_xy, ray_xy) + 1.0);

    var z_f = z_f_raw;
    var c_front = in.c_front_s * ScalarF(z_f_raw * in.params.x);
    let c_back = in.c_back_s * ScalarF(z_b * in.params.x);

    if (uniforms.min_t > 0.0 && z_f < uniforms.min_t) {
        let start = min(z_b, max(z_f, uniforms.min_t / ray_scale));
        if (start > z_f) {
            let frac = ScalarF(clamp((start - z_f) / max(z_b - z_f, 1e-20), 0.0, 1.0));
            c_front = mix(c_front, c_back, frac);
            z_f = start;
        }
    }

    let dist = max(z_b - z_f, 0.0) * ray_scale;
    let od = min(in.params.y * dist, 88.0);
    if (od <= 0.0) {
        return vec4<f32>(0.0);
    }
    let alpha_t = exp(-od);
    let alpha = 1.0 - alpha_t;

    let phi_val = phi_from_alpha(od, alpha);
    let w0 = ScalarF(phi_val - alpha_t);
    let w1 = ScalarF(1.0 - phi_val);
    let rgb = max(c_back * w0 + c_front * w1, Color3(0.0));
    return vec4<f32>(vec3<f32>(rgb), alpha) * in.params.z;
}

@fragment
fn fs_overdraw(in: EnergyIntervalVsOut) -> @location(0) vec4<f32> {
    let inv_near = 1.0 / uniforms.near_plane;
    let inv_far = 1.0 / uniforms.far_plane;
    var z_f = 1.0 / fma(in.position.z, inv_far - inv_near, inv_near);
    let z_b = 1.0 / max(in.inv_z_back, inv_far);
    let ray_xy = (in.position.xy - uniforms.intrinsics.zw) / uniforms.intrinsics.xy;
    let ray_scale = sqrt(dot(ray_xy, ray_xy) + 1.0);
    if (uniforms.min_t > 0.0 && z_f < uniforms.min_t) {
        z_f = min(z_b, max(z_f, uniforms.min_t / ray_scale));
    }
    let od = min(in.params.y * max(z_b - z_f, 0.0) * ray_scale, 88.0);
    let alpha = (1.0 - exp(-od)) * in.params.z;
    return vec4<f32>(1.0, alpha, 0.0, 0.0);
}
