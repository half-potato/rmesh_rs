// Energy interval shading with sparse entry-face linear albedo textures.
// Integrates perspective-weighted albedo-free diffuse irradiance and specular radiance
// over the ray segment, then evaluates the exact entry-face triangular texture stencil
// (matching face_textures.slang) with a fast-path skip for untextured tetrahedra.

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

// Group 1: fragment-only entry-face texture atlas + fallback per-tet albedo tail.
// Storage buffers per stage: vertex=3 (uniforms, verts, tet_data),
// fragment=6 (uniforms, ft_vertices, ft_indices, ft_face_ids, ft_charts, ft_albedo) <= 8.
@group(1) @binding(0) var<storage, read> ft_vertices: array<f32>;
@group(1) @binding(1) var<storage, read> ft_indices: array<u32>;
@group(1) @binding(2) var<storage, read> ft_face_ids: array<vec4<i32>>;
@group(1) @binding(3) var<storage, read> ft_charts: array<vec2<i32>>;
@group(1) @binding(4) var<storage, read> ft_albedo: array<u32>;
@group(1) @binding(5) var<uniform> ft_info: vec4<u32>; // .x = fallback_offset (S')

struct EnergyIntervalFtVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) inv_z_back: f32,
    @location(1) d_front_s: Color3,
    @location(2) d_back_s: Color3,
    @location(3) s_front_s: Color3,
    @location(4) s_back_s: Color3,
    // (inv_w_min, density, coverage)
    @location(5) @interpolate(flat) params: vec3<f32>,
    @location(6) @interpolate(flat) tet_id: u32,
};

@vertex
fn vs_energy(@builtin(vertex_index) vid: u32) -> EnergyIntervalFtVsOut {
    let tet = vid / 5u;
    let pos_depth = verts[vid * 3u + 0u];
    let packed_d = bitcast<vec4<u32>>(verts[vid * 3u + 1u]);
    let packed_s = bitcast<vec4<u32>>(verts[vid * 3u + 2u]);
    let td = tet_data[tet];

    let d0 = unpack2x16float(packed_d.x);
    let d1 = unpack2x16float(packed_d.y);
    let d2 = unpack2x16float(packed_d.z);

    let s0 = unpack2x16float(packed_s.x);
    let s1 = unpack2x16float(packed_s.y);
    let s2 = unpack2x16float(packed_s.z);

    let inv_near = 1.0 / uniforms.near_plane;
    let inv_far = 1.0 / uniforms.far_plane;

    var out: EnergyIntervalFtVsOut;
    out.position = vec4<f32>(pos_depth.xy, pos_depth.z, 1.0);
    out.inv_z_back = fma(pos_depth.w, inv_far - inv_near, inv_near);
    out.d_front_s = Color3(vec3<f32>(d0.x, d0.y, d1.x));
    out.d_back_s = Color3(vec3<f32>(d1.y, d2.x, d2.y));
    out.s_front_s = Color3(vec3<f32>(s0.x, s0.y, s1.x));
    out.s_back_s = Color3(vec3<f32>(s1.y, s2.x, s2.y));
    out.params = td.xyz;
    out.tet_id = bitcast<u32>(td.w);
    return out;
}

fn phi_from_alpha(x: f32, alpha: f32) -> f32 {
    if (x < 0.02) {
        return 1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0)));
    }
    return alpha / x;
}

fn unpack_rgb8(w: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(w & 0xffu),
        f32((w >> 8u) & 0xffu),
        f32((w >> 16u) & 0xffu),
    ) * (1.0 / 255.0);
}

fn load_vert(idx: u32) -> vec3<f32> {
    let b = idx * 3u;
    return vec3<f32>(ft_vertices[b], ft_vertices[b + 1u], ft_vertices[b + 2u]);
}

fn sample_entry_albedo(tet_id: u32, ray_xy: vec2<f32>, z_f_raw: f32) -> vec3<f32> {
    let fids = ft_face_ids[tet_id];
    // Fast path: untextured tet skips vertex loads, barycentrics, and chart lookup.
    if (fids.x < 0 && fids.y < 0 && fids.z < 0 && fids.w < 0) {
        return unpack_rgb8(ft_albedo[ft_info.x + tet_id]);
    }

    let ib = tet_id * 4u;
    let vi = vec4<u32>(
        ft_indices[ib],
        ft_indices[ib + 1u],
        ft_indices[ib + 2u],
        ft_indices[ib + 3u],
    );
    let v0 = load_vert(vi.x);
    let v1 = load_vert(vi.y);
    let v2 = load_vert(vi.z);
    let v3 = load_vert(vi.w);

    let a = v1 - v0;
    let b = v2 - v0;
    let c = v3 - v0;
    let bc = cross(b, c);
    let det = dot(a, bc);
    if (abs(det) < 1e-30) {
        return unpack_rgb8(ft_albedo[ft_info.x + tet_id]);
    }
    let ca = cross(c, a);
    let ab = cross(a, b);
    let inv_det = 1.0 / det;

    let c2w = mat3x3<f32>(
        uniforms.c2w_col0.xyz,
        uniforms.c2w_col1.xyz,
        uniforms.c2w_col2.xyz,
    );
    let ray_dir = c2w * vec3<f32>(ray_xy, 1.0);
    let p_enter = uniforms.cam_pos_pad.xyz + ray_dir * z_f_raw;
    let p = p_enter - v0;

    let u = vec3<f32>(dot(p, bc), dot(p, ca), dot(p, ab)) * inv_det;
    let bary = vec4<f32>(1.0 - u.x - u.y - u.z, u.x, u.y, u.z);
    let du = vec3<f32>(dot(ray_dir, bc), dot(ray_dir, ca), dot(ray_dir, ab)) * inv_det;
    let dbary = vec4<f32>(-du.x - du.y - du.z, du.x, du.y, du.z);

    // Entry face: bary[j] increases along the ray (dbary[j] > 0) and vanishes at p_enter.
    var opposite = -1;
    var best_abs = 1e30;
    for (var j = 0; j < 4; j++) {
        if (dbary[j] > 0.0) {
            let abv = abs(bary[j]);
            if (abv < best_abs) {
                best_abs = abv;
                opposite = j;
            }
        }
    }
    if (opposite < 0) {
        opposite = 0;
        for (var j = 1; j < 4; j++) {
            if (abs(bary[j]) < abs(bary[opposite])) {
                opposite = j;
            }
        }
    }

    let chart = fids[opposite];
    if (chart < 0) {
        return unpack_rgb8(ft_albedo[ft_info.x + tet_id]);
    }

    let ch = ft_charts[u32(chart)];
    let offset = ch.x;
    let n = ch.y;
    if (n <= 1) {
        return unpack_rgb8(ft_albedo[u32(offset)]);
    }

    // Canonical ascending global vertex-ID order agrees across both tets sharing the face.
    var c0 = 0u;
    var c1 = 1u;
    var c2 = 2u;
    if (opposite == 0) { c0 = 1u; c1 = 2u; c2 = 3u; }
    else if (opposite == 1) { c0 = 0u; c1 = 2u; c2 = 3u; }
    else if (opposite == 2) { c0 = 0u; c1 = 1u; c2 = 3u; }

    if (vi[c1] < vi[c0]) { let t = c0; c0 = c1; c1 = t; }
    if (vi[c2] < vi[c0]) { let t = c0; c0 = c2; c2 = t; }
    if (vi[c2] < vi[c1]) { let t = c1; c1 = c2; c2 = t; }

    var uvw = max(vec3<f32>(bary[c0], bary[c1], bary[c2]), vec3<f32>(0.0));
    uvw /= max(uvw.x + uvw.y + uvw.z, 1e-20);

    let xy = uvw.yz * f32(n - 1);
    let ii = min(i32(floor(xy.x)), n - 2);
    var jj = min(i32(floor(xy.y)), n - 2);
    if (ii + jj >= n - 1 && jj > 0) {
        jj -= 1;
    }
    let fx = xy.x - f32(ii);
    let fy = xy.y - f32(jj);
    var si = vec3<i32>(ii, ii + 1, ii);
    var sj = vec3<i32>(jj, jj, jj + 1);
    var w = vec3<f32>(1.0 - fx - fy, fx, fy);
    if (fx + fy > 1.0 && ii + jj < n - 2) {
        si = vec3<i32>(ii + 1, ii, ii + 1);
        sj = vec3<i32>(jj, jj + 1, jj + 1);
        w = vec3<f32>(1.0 - fy, 1.0 - fx, fx + fy - 1.0);
    }

    let s0 = u32(offset + sj.x * n - (sj.x * (sj.x - 1)) / 2 + si.x);
    let s1 = u32(offset + sj.y * n - (sj.y * (sj.y - 1)) / 2 + si.y);
    let s2 = u32(offset + sj.z * n - (sj.z * (sj.z - 1)) / 2 + si.z);
    return unpack_rgb8(ft_albedo[s0]) * w.x
         + unpack_rgb8(ft_albedo[s1]) * w.y
         + unpack_rgb8(ft_albedo[s2]) * w.z;
}

@fragment
fn main_color_only(in: EnergyIntervalFtVsOut) -> @location(0) vec4<f32> {
    let inv_near = 1.0 / uniforms.near_plane;
    let inv_far = 1.0 / uniforms.far_plane;

    let z_f_raw = 1.0 / fma(in.position.z, inv_far - inv_near, inv_near);
    let z_b = 1.0 / max(in.inv_z_back, inv_far);

    let ray_xy = (in.position.xy - uniforms.intrinsics.zw) / uniforms.intrinsics.xy;
    let ray_scale = sqrt(dot(ray_xy, ray_xy) + 1.0);

    var z_f = z_f_raw;
    let sf_f = ScalarF(z_f_raw * in.params.x);
    let sf_b = ScalarF(z_b * in.params.x);
    var d_front = in.d_front_s * sf_f;
    let d_back = in.d_back_s * sf_b;
    var s_front = in.s_front_s * sf_f;
    let s_back = in.s_back_s * sf_b;

    if (uniforms.min_t > 0.0 && z_f < uniforms.min_t) {
        let start = min(z_b, max(z_f, uniforms.min_t / ray_scale));
        if (start > z_f) {
            let frac = ScalarF(clamp((start - z_f) / max(z_b - z_f, 1e-20), 0.0, 1.0));
            d_front = mix(d_front, d_back, frac);
            s_front = mix(s_front, s_back, frac);
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
    let diff_int = vec3<f32>(max(d_back * w0 + d_front * w1, Color3(0.0)));
    let spec_int = vec3<f32>(max(s_back * w0 + s_front * w1, Color3(0.0)));

    let albedo = sample_entry_albedo(in.tet_id, ray_xy, z_f_raw);
    let rgb = diff_int * albedo + spec_int;
    return vec4<f32>(rgb, alpha) * in.params.z;
}

@fragment
fn fs_overdraw(in: EnergyIntervalFtVsOut) -> @location(0) vec4<f32> {
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
