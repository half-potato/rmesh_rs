// Backward interval tiled: 1 warp (32 threads) per tile.
//
// Same tile structure and state-undo pattern as backward_tiled_compute.wgsl.
// Reads pre-computed screen triangles from interval_generate.
// Gradients accumulate to per-vertex interval attributes (d_interval_verts)
// and per-tet data (d_interval_tet_data) via warp reduce + atomics.
//
// One workgroup per tile. Dispatch: (num_tiles, 1, 1).

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

struct TileUniforms {
    screen_width: u32,
    screen_height: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_tiles: u32,
    visible_tet_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
};

// --- Safe math utilities ---
const TINY_VAL: f32 = 1.0754944e-20;
const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;
const LOG_MAX_VAL: f32 = 46.0517;

fn safe_clip_f32(v: f32, minv: f32, maxv: f32) -> f32 {
    return max(min(v, maxv), minv);
}

fn safe_clip_v3f(v: vec3<f32>, minv: f32, maxv: f32) -> vec3<f32> {
    return vec3<f32>(
        max(min(v.x, maxv), minv),
        max(min(v.y, maxv), minv),
        max(min(v.z, maxv), minv)
    );
}

fn safe_div_f32(a: f32, b: f32) -> f32 {
    if (abs(b) < TINY_VAL) {
        return clamp(a / TINY_VAL, MIN_VAL, MAX_VAL);
    } else {
        return clamp(a / b, MIN_VAL, MAX_VAL);
    }
}

fn safe_exp_f32(v: f32) -> f32 {
    return exp(clamp(v, -88.0, LOG_MAX_VAL));
}

fn phi(x: f32) -> f32 {
    if (abs(x) < 0.02) { return 1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0))); }
    return safe_div_f32(1.0 - exp(-x), x);
}

fn dphi_dx(x: f32) -> f32 {
    if (abs(x) < 0.02) { return -0.5 + x * (1.0/3.0 + x * (-1.0/8.0 + x * (1.0/30.0))); }
    let ex = exp(-x);
    return safe_div_f32(ex * (x + 1.0) - 1.0, x * x);
}
// --- End safe math utilities ---

// Group 0: read-only scene + interval data
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> dl_d_image: array<f32>;
@group(0) @binding(2) var<storage, read> rendered_image: array<f32>;
@group(0) @binding(3) var<storage, read> interval_verts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> interval_tet_data: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> interval_meta: array<u32>;
@group(0) @binding(6) var<storage, read> tile_sort_values: array<u32>;
@group(0) @binding(7) var<storage, read> dl_d_xyzd: array<f32>;          // [W*H*4]
@group(0) @binding(8) var<storage, read> dl_d_distortion: array<f32>;    // [W*H*5]
@group(0) @binding(9) var<storage, read> xyzd_image: array<f32>;         // [W*H*4] (unused, reserved)
@group(0) @binding(10) var<storage, read> distortion_image: array<f32>;  // [W*H*5]
@group(0) @binding(11) var<storage, read> aux_data: array<f32>;         // [M * AUX_DIM]
@group(0) @binding(12) var<storage, read> dl_d_aux: array<f32>;         // [W*H*AUX_DIM]

const AUX_DIM: u32 = /*AUX_DIM*/0u;
const AUX_DIM_MAX: u32 = /*AUX_DIM_MAX*/1u;

// Group 1: gradient outputs + tile metadata
@group(1) @binding(0) var<storage, read_write> d_interval_verts: array<atomic<f32>>;
@group(1) @binding(1) var<storage, read_write> d_interval_tet_data: array<atomic<f32>>;
@group(1) @binding(2) var<storage, read> tile_ranges: array<u32>;
@group(1) @binding(3) var<storage, read> tile_uniforms: TileUniforms;
@group(1) @binding(4) var<storage, read_write> d_aux_data: array<atomic<f32>>; // [M * AUX_DIM]

// Workgroup shared memory
var<workgroup> sm_state: array<vec4<f32>, 256>;   // .xyz = color, .w = log_t
var<workgroup> sm_d_log_t: array<f32, 256>;       // running d(log_t) per pixel
// Distortion state undo + gradient (z,w components only)
var<workgroup> sm_dist_z: array<f32, 256>;         // running distortion.z (m0 accumulator)
var<workgroup> sm_dist_w: array<f32, 256>;         // running distortion.w (m1 accumulator)
var<workgroup> sm_d_dist_z: array<f32, 256>;       // running gradient d(loss)/d(distortion.z)
var<workgroup> sm_d_dist_w: array<f32, 256>;       // running gradient d(loss)/d(distortion.w)
var<workgroup> sm_xl: array<i32, 16>;
var<workgroup> sm_xr: array<i32, 16>;
var<workgroup> sm_prefix: array<u32, 17>;

// Barycentric coordinates
fn bary2d(pt: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = pt - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let inv_denom = select(0.0, 1.0 / denom, abs(denom) > 1e-10);
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

fn point_in_triangle(pt: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let d0 = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    let d1 = (c.x - b.x) * (pt.y - b.y) - (c.y - b.y) * (pt.x - b.x);
    let d2 = (a.x - c.x) * (pt.y - c.y) - (a.y - c.y) * (pt.x - c.x);
    let has_neg = (d0 < 0.0) || (d1 < 0.0) || (d2 < 0.0);
    let has_pos = (d0 > 0.0) || (d1 > 0.0) || (d2 > 0.0);
    return !(has_neg && has_pos);
}

// Warp reduction
fn warp_reduce(val: f32) -> f32 {
    var v = val;
    v += subgroupShuffleXor(v, 16u);
    v += subgroupShuffleXor(v, 8u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
}

@compute @workgroup_size(32)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_id = wg_id.x;
    if (tile_id >= tile_uniforms.num_tiles) {
        return;
    }

    let tile_x = tile_id % tile_uniforms.tiles_x;
    let tile_y = tile_id / tile_uniforms.tiles_x;
    let w = tile_uniforms.screen_width;
    let h = tile_uniforms.screen_height;
    let W = f32(w);
    let H = f32(h);
    let TS = tile_uniforms.tile_size;
    let tile_ox = f32(tile_x * TS);
    let tile_oy = f32(tile_y * TS);

    let range_start = tile_ranges[tile_id * 2u];
    let range_end = tile_ranges[tile_id * 2u + 1u];

    let near = uniforms.near_plane;
    let far = uniforms.far_plane;
    let range_nf = far - near;
    let fx = uniforms.intrinsics.x;
    let fy = uniforms.intrinsics.y;
    let cx_cam = uniforms.intrinsics.z;
    let cy_cam = uniforms.intrinsics.w;

    // Initialize state from rendered_image (final forward state) and dl_d_image
    for (var i = lane; i < TS * TS; i += 32u) {
        let row = i / TS;
        let col = i % TS;
        let px = tile_x * TS + col;
        let py = tile_y * TS + row;
        if (px < w && py < h) {
            let idx = py * w + px;
            let a = rendered_image[idx * 4u + 3u];
            sm_state[i] = vec4<f32>(
                rendered_image[idx * 4u],
                rendered_image[idx * 4u + 1u],
                rendered_image[idx * 4u + 2u],
                log(max(1.0 - a, 1e-20)));
            sm_d_log_t[i] = -dl_d_image[idx * 4u + 3u] * (1.0 - a);
            // Initialize distortion state from forward output
            sm_dist_z[i] = distortion_image[idx * 5u + 2u];
            sm_dist_w[i] = distortion_image[idx * 5u + 3u];
            // Initialize distortion gradients from upstream loss gradient
            sm_d_dist_z[i] = dl_d_distortion[idx * 5u + 2u];
            sm_d_dist_w[i] = dl_d_distortion[idx * 5u + 3u];
        } else {
            sm_state[i] = vec4<f32>(0.0);
            sm_d_log_t[i] = 0.0;
            sm_dist_z[i] = 0.0;
            sm_dist_w[i] = 0.0;
            sm_d_dist_z[i] = 0.0;
            sm_d_dist_w[i] = 0.0;
        }
    }
    workgroupBarrier();

    // Process tets back-to-front (reverse of forward order)
    for (var cursor = range_start; cursor < range_end; cursor += 1u) {
        let vis_idx = tile_sort_values[cursor];

        // Read meta
        let meta_val = interval_meta[vis_idx];
        let num_sil = meta_val & 0xFu;

        if (num_sil == 0u) {
            continue;
        }

        // Load pre-computed silhouette vertices (4 vec4 per vertex)
        let vb = vis_idx * 5u;
        var sv_pix: array<vec2<f32>, 5>;
        var sv_zf: array<f32, 5>;
        var sv_zb: array<f32, 5>;
        var sv_off_f: array<f32, 5>;
        var sv_off_b: array<f32, 5>;
        var sv_nf: array<vec3<f32>, 5>;
        var sv_nb: array<vec3<f32>, 5>;
        for (var i = 0u; i < 5u; i++) {
            let d0 = interval_verts[(vb + i) * 4u];
            let d1 = interval_verts[(vb + i) * 4u + 1u];
            let d2 = interval_verts[(vb + i) * 4u + 2u];
            let d3 = interval_verts[(vb + i) * 4u + 3u];
            sv_pix[i] = d0.xy;
            sv_zf[i] = d0.z;
            sv_zb[i] = d0.w;
            sv_off_f[i] = d1.x;
            sv_off_b[i] = d1.y;
            sv_nf[i] = d2.xyz;
            sv_nb[i] = d3.xyz;
        }

        // Per-tet material
        let tet_data = interval_tet_data[vis_idx];
        let base_color = tet_data.xyz;
        let density_raw = tet_data.w;

        // Convert silhouette to tile-local coords
        var proj: array<vec2<f32>, 4>;
        for (var i = 0u; i < min(num_sil, 4u); i++) {
            proj[i] = vec2<f32>(sv_pix[i].x - tile_ox, sv_pix[i].y - tile_oy);
        }

        // Scanline fill
        if (lane < TS) {
            var xl_f: f32 = 1e10;
            var xr_f: f32 = -1e10;
            let yc = f32(lane) + 0.5;

            for (var e = 0u; e < num_sil; e++) {
                let ei = e;
                let ej = (e + 1u) % num_sil;
                let yi = proj[ei].y;
                let yj = proj[ej].y;
                if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
                    let t = (yc - yi) / (yj - yi);
                    let x = proj[ei].x + t * (proj[ej].x - proj[ei].x);
                    xl_f = min(xl_f, x);
                    xr_f = max(xr_f, x);
                }
            }
            for (var v = 0u; v < num_sil; v++) {
                let vy = proj[v].y;
                if (vy >= yc - 0.5 && vy < yc + 0.5) {
                    xl_f = min(xl_f, proj[v].x);
                    xr_f = max(xr_f, proj[v].x);
                }
            }

            if (xl_f <= xr_f) {
                let eps = 0.001;
                let xl_i = max(i32(ceil(xl_f - 0.5 - eps)), 0);
                let xr_i = min(i32(floor(xr_f - 0.5 + eps)), i32(TS) - 1);
                if (xl_i <= xr_i) {
                    sm_xl[lane] = xl_i;
                    sm_xr[lane] = xr_i;
                } else {
                    sm_xl[lane] = 0;
                    sm_xr[lane] = -1;
                }
            } else {
                sm_xl[lane] = 0;
                sm_xr[lane] = -1;
            }
        }
        workgroupBarrier();

        if (lane == 0u) {
            sm_prefix[0] = 0u;
            for (var r = 0u; r < TS; r++) {
                let row_w = u32(max(sm_xr[r] - sm_xl[r] + 1, 0));
                sm_prefix[r + 1u] = sm_prefix[r] + row_w;
            }
        }
        workgroupBarrier();

        let total = sm_prefix[TS];
        if (total == 0u) {
            continue;
        }

        // Per-thread gradient accumulators for interval vertex attributes.
        // 5 vertices × 4 gradient channels (d_zf, d_zb, d_off_f, d_off_b)
        var d_sv: array<vec4<f32>, 5>;
        d_sv[0] = vec4<f32>(0.0);
        d_sv[1] = vec4<f32>(0.0);
        d_sv[2] = vec4<f32>(0.0);
        d_sv[3] = vec4<f32>(0.0);
        d_sv[4] = vec4<f32>(0.0);
        // Normal gradient accumulators: 5 vertices × (d_nf.xyz, d_nb.xyz)
        var d_sv_nf: array<vec3<f32>, 5>;
        var d_sv_nb: array<vec3<f32>, 5>;
        for (var vi = 0u; vi < 5u; vi++) {
            d_sv_nf[vi] = vec3<f32>(0.0);
            d_sv_nb[vi] = vec3<f32>(0.0);
        }
        // Per-tet: d_base_color, d_density
        var d_base_color_accum = vec3<f32>(0.0);
        var d_density_accum: f32 = 0.0;
        // Per-tet aux gradient accumulator (no-op when AUX_DIM=0)
        var d_aux_local: array<f32, AUX_DIM_MAX>;
        for (var axi = 0u; axi < AUX_DIM; axi++) {
            d_aux_local[axi] = 0.0;
        }

        // Process covered pixels
        for (var idx = lane; idx < total; idx += 32u) {
            var row = 0u;
            for (var r = 0u; r < TS; r++) {
                if (idx < sm_prefix[r + 1u]) {
                    row = r;
                    break;
                }
            }
            let col = u32(sm_xl[row]) + (idx - sm_prefix[row]);
            let pixel_local = row * TS + col;
            let px = tile_x * TS + col;
            let py = tile_y * TS + row;

            if (px >= w || py >= h) {
                continue;
            }

            let pixel_idx = py * w + px;
            let pix_center = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

            // Find containing triangle
            var found = false;
            var tri_ai: u32 = 0u;
            var tri_bi: u32 = 0u;
            let tri_ci: u32 = 4u;
            var bary = vec3<f32>(0.0);
            var z_front_ndc: f32 = 0.0;
            var z_back_ndc: f32 = 0.0;
            var off_front: f32 = 0.0;
            var off_back: f32 = 0.0;

            for (var tri = 0u; tri < num_sil; tri++) {
                let ai = tri;
                let bi = (tri + 1u) % num_sil;

                if (point_in_triangle(pix_center, sv_pix[ai], sv_pix[bi], sv_pix[tri_ci])) {
                    tri_ai = ai;
                    tri_bi = bi;
                    bary = bary2d(pix_center, sv_pix[ai], sv_pix[bi], sv_pix[tri_ci]);

                    z_front_ndc = bary.x * sv_zf[ai] + bary.y * sv_zf[bi] + bary.z * sv_zf[tri_ci];
                    z_back_ndc = bary.x * sv_zb[ai] + bary.y * sv_zb[bi] + bary.z * sv_zb[tri_ci];
                    off_front = bary.x * sv_off_f[ai] + bary.y * sv_off_f[bi] + bary.z * sv_off_f[tri_ci];
                    off_back = bary.x * sv_off_b[ai] + bary.y * sv_off_b[bi] + bary.z * sv_off_b[tri_ci];
                    found = true;
                    break;
                }
            }

            if (!found) {
                continue;
            }

            // Forward recomputation
            let z_front_clamped = clamp(z_front_ndc, 0.0, 1.0);
            let z_back_clamped = clamp(z_back_ndc, 0.0, 1.0);
            let z_f = near * far / (far - z_front_clamped * range_nf);
            let z_b = near * far / (far - z_back_clamped * range_nf);

            let x_cam = (f32(px) + 0.5 - cx_cam) / fx;
            let y_cam = (f32(py) + 0.5 - cy_cam) / fy;
            let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

            let dist = abs(z_b - z_f) * ray_scale;

            let c_front_raw = base_color + vec3<f32>(off_front);
            let c_back_raw = base_color + vec3<f32>(off_back);
            let c_front = max(c_front_raw, vec3<f32>(0.0));
            let c_back = max(c_back_raw, vec3<f32>(0.0));

            let od = clamp(density_raw * dist, 1e-8, 88.0);
            let alpha_t = safe_exp_f32(-od);
            let phi_val = phi(od);
            let w0 = phi_val - alpha_t;
            let w1 = 1.0 - phi_val;
            let c_premul = safe_clip_v3f(c_back * w0 + c_front * w1, MIN_VAL, MAX_VAL);

            // Read state, undo contribution
            let state = sm_state[pixel_local];
            let color = state.xyz;
            let log_t = state.w;
            let d_log_t = sm_d_log_t[pixel_local];

            let prev_log_t = min(log_t + od, 0.0);
            let t_prev = safe_exp_f32(prev_log_t);
            let prev_color = color - c_premul * t_prev;

            sm_state[pixel_local] = vec4<f32>(prev_color, prev_log_t);

            if (t_prev < 1e-6) {
                sm_d_log_t[pixel_local] = d_log_t;
                continue;
            }

            // Upstream color gradient
            let d_color = vec3<f32>(
                dl_d_image[pixel_idx * 4u],
                dl_d_image[pixel_idx * 4u + 1u],
                dl_d_image[pixel_idx * 4u + 2u],
            );

            // === Backward computation ===
            let d_c_premul = safe_clip_v3f(d_color * t_prev, MIN_VAL, MAX_VAL);
            let d_od_state = -d_log_t;
            let d_old_log_t = safe_clip_f32(d_log_t + dot(d_color, c_premul) * t_prev, MIN_VAL, MAX_VAL);
            sm_d_log_t[pixel_local] = d_old_log_t;

            let dphi_val = dphi_dx(od);
            let dw0_dod = dphi_val + alpha_t;
            let dw1_dod = -dphi_val;

            let d_c_back_integral = d_c_premul * w0;
            let d_c_front_integral = d_c_premul * w1;
            let d_od_integral = dot(d_c_premul, c_back * dw0_dod + c_front * dw1_dod);
            let d_od = safe_clip_f32(d_od_state + d_od_integral, MIN_VAL, MAX_VAL);

            // Respect the clamp
            let od_raw = density_raw * dist;
            let od_active = (od_raw >= 1e-8) && (od_raw <= 88.0);
            let d_od_raw = select(0.0, d_od, od_active);
            let d_density_local = safe_clip_f32(d_od_raw * dist, MIN_VAL, MAX_VAL);
            let d_dist = safe_clip_f32(d_od_raw * density_raw, MIN_VAL, MAX_VAL);

            // d_dist → d_z_f, d_z_b
            let sign_zb_minus_zf = select(-1.0, 1.0, z_b >= z_f);
            let d_z_f_view = -d_dist * ray_scale * sign_zb_minus_zf;
            let d_z_b_view = d_dist * ray_scale * sign_zb_minus_zf;

            // Linearize backward: z_view = near*far / (far - z_ndc * range_nf)
            // d(z_view)/d(z_ndc) = near * far * range_nf / (far - z_ndc * range_nf)^2
            let denom_f = far - z_front_clamped * range_nf;
            let denom_b = far - z_back_clamped * range_nf;
            let dz_dzn_f = near * far * range_nf / (denom_f * denom_f);
            let dz_dzn_b = near * far * range_nf / (denom_b * denom_b);

            // Respect the clamp on z_ndc
            let zf_active = (z_front_ndc >= 0.0) && (z_front_ndc <= 1.0);
            let zb_active = (z_back_ndc >= 0.0) && (z_back_ndc <= 1.0);
            let d_z_front_ndc = select(0.0, d_z_f_view * dz_dzn_f, zf_active);
            let d_z_back_ndc = select(0.0, d_z_b_view * dz_dzn_b, zb_active);

            // Color chain through ReLU
            let d_c_back_raw = vec3<f32>(
                select(0.0, d_c_back_integral.x, c_back_raw.x > 0.0),
                select(0.0, d_c_back_integral.y, c_back_raw.y > 0.0),
                select(0.0, d_c_back_integral.z, c_back_raw.z > 0.0),
            );
            let d_c_front_raw = vec3<f32>(
                select(0.0, d_c_front_integral.x, c_front_raw.x > 0.0),
                select(0.0, d_c_front_integral.y, c_front_raw.y > 0.0),
                select(0.0, d_c_front_integral.z, c_front_raw.z > 0.0),
            );

            // d_base_color = d_c_front_raw + d_c_back_raw (broadcast scalar from each)
            let d_bc = d_c_front_raw + d_c_back_raw;
            // d_off_front = sum(d_c_front_raw), d_off_back = sum(d_c_back_raw)
            let d_off_front = d_c_front_raw.x + d_c_front_raw.y + d_c_front_raw.z;
            let d_off_back = d_c_back_raw.x + d_c_back_raw.y + d_c_back_raw.z;

            // Scatter gradients to triangle vertices via barycentrics
            // The interpolated value was: val = bary.x * v_a + bary.y * v_b + bary.z * v_c
            // d_zf for each vertex
            let d_zf_a = bary.x * d_z_front_ndc;
            let d_zf_b = bary.y * d_z_front_ndc;
            let d_zf_c = bary.z * d_z_front_ndc;
            // d_zb for each vertex
            let d_zb_a = bary.x * d_z_back_ndc;
            let d_zb_b = bary.y * d_z_back_ndc;
            let d_zb_c = bary.z * d_z_back_ndc;
            // d_off_f for each vertex
            let d_of_a = bary.x * d_off_front;
            let d_of_b = bary.y * d_off_front;
            let d_of_c = bary.z * d_off_front;
            // d_off_b for each vertex
            let d_ob_a = bary.x * d_off_back;
            let d_ob_b = bary.y * d_off_back;
            let d_ob_c = bary.z * d_off_back;

            d_sv[tri_ai] += vec4<f32>(d_zf_a, d_zb_a, d_of_a, d_ob_a);
            d_sv[tri_bi] += vec4<f32>(d_zf_b, d_zb_b, d_of_b, d_ob_b);
            d_sv[tri_ci] += vec4<f32>(d_zf_c, d_zb_c, d_of_c, d_ob_c);

            d_base_color_accum += safe_clip_v3f(d_bc, MIN_VAL, MAX_VAL);
            d_density_accum += safe_clip_f32(d_density_local, MIN_VAL, MAX_VAL);

            // === xyzd backward (linear accumulation, no state undo needed) ===
            let d_xyzd = vec4<f32>(
                dl_d_xyzd[pixel_idx * 4u],
                dl_d_xyzd[pixel_idx * 4u + 1u],
                dl_d_xyzd[pixel_idx * 4u + 2u],
                dl_d_xyzd[pixel_idx * 4u + 3u],
            );

            // Recompute xyzd contribution
            let n_f = bary.x * sv_nf[tri_ai] + bary.y * sv_nf[tri_bi] + bary.z * sv_nf[tri_ci];
            let n_b = bary.x * sv_nb[tri_ai] + bary.y * sv_nb[tri_bi] + bary.z * sv_nb[tri_ci];
            let normal_contrib = n_b * w0 + n_f * w1;
            let depth_contrib = w0 * z_b + w1 * z_f;
            let xyzd_contrib = vec4<f32>(normal_contrib, depth_contrib);

            // d_xyzd_contrib = d_xyzd * T_prev (linear accumulation: output += T * contrib)
            let d_xyzd_contrib = d_xyzd * t_prev;
            // contribution to d_od from xyzd: -dot(d_xyzd, xyzd_contrib) * T_prev
            let d_od_xyzd = -dot(d_xyzd, xyzd_contrib) * t_prev;

            // Chain to normal weights
            let d_normal = d_xyzd_contrib.xyz;
            let d_depth_raw = d_xyzd_contrib.w;

            // d_n_b += d_normal * w0, d_n_f += d_normal * w1
            let d_n_b = d_normal * w0;
            let d_n_f = d_normal * w1;

            // Additional d_od from xyzd through volume weights
            let d_od_xyzd_weights = dot(d_normal, n_b * dw0_dod + n_f * dw1_dod)
                                  + d_depth_raw * (z_b * dw0_dod + z_f * dw1_dod);

            // Additional d_z_f/d_z_b from depth contribution
            let d_z_f_xyzd = d_depth_raw * w1;
            let d_z_b_xyzd = d_depth_raw * w0;

            // Add xyzd contributions to d_od (already clamped upstream)
            d_density_accum += safe_clip_f32((d_od_xyzd + d_od_xyzd_weights) * dist, MIN_VAL, MAX_VAL);

            // Convert d_z_f/d_z_b from view-space to NDC and add to vertex grads
            let d_zf_ndc_xyzd = select(0.0, d_z_f_xyzd * dz_dzn_f, zf_active);
            let d_zb_ndc_xyzd = select(0.0, d_z_b_xyzd * dz_dzn_b, zb_active);
            d_sv[tri_ai] += vec4<f32>(bary.x * d_zf_ndc_xyzd, bary.x * d_zb_ndc_xyzd, 0.0, 0.0);
            d_sv[tri_bi] += vec4<f32>(bary.y * d_zf_ndc_xyzd, bary.y * d_zb_ndc_xyzd, 0.0, 0.0);
            d_sv[tri_ci] += vec4<f32>(bary.z * d_zf_ndc_xyzd, bary.z * d_zb_ndc_xyzd, 0.0, 0.0);

            // Scatter normal gradients to silhouette vertices via barycentrics
            d_sv_nf[tri_ai] += bary.x * d_n_f;
            d_sv_nf[tri_bi] += bary.y * d_n_f;
            d_sv_nf[tri_ci] += bary.z * d_n_f;
            d_sv_nb[tri_ai] += bary.x * d_n_b;
            d_sv_nb[tri_bi] += bary.y * d_n_b;
            d_sv_nb[tri_ci] += bary.z * d_n_b;

            // === distortion backward (state undo needed for z, w) ===
            // Recompute distortion forward values
            let enter = z_f * ray_scale;
            let exit_d = z_b * ray_scale;
            let dt_d = exit_d - enter;
            let u = density_raw * dt_d;
            let exp_neg_u = safe_exp_f32(-u);
            let alpha_dist = max(1.0 - exp_neg_u, 0.0);
            let m0 = t_prev * alpha_dist;
            let m1 = t_prev * (enter * alpha_dist + safe_div_f32(1.0 - (1.0 + u) * exp_neg_u, density_raw));
            let sd = t_prev * t_prev * safe_div_f32(1.0 - 2.0 * u * exp_neg_u - exp_neg_u * exp_neg_u, density_raw);

            // State undo: remove current tet's contribution
            let old_z = sm_dist_z[pixel_local] - m0;
            let old_w = sm_dist_w[pixel_local] - m1;
            sm_dist_z[pixel_local] = old_z;
            sm_dist_w[pixel_local] = old_w;

            // Read constant upstream gradients
            let d_x = dl_d_distortion[pixel_idx * 5u];
            let d_y = dl_d_distortion[pixel_idx * 5u + 1u];
            let d_z = sm_d_dist_z[pixel_local];
            let d_w = sm_d_dist_w[pixel_local];
            let d_v = dl_d_distortion[pixel_idx * 5u + 4u];

            // Gradients through update equations
            let d_m0 = d_z + 2.0 * old_w * d_y;
            let d_m1 = d_w + 2.0 * old_z * d_x;
            let d_sd = d_v;

            // Update running gradients for next (earlier) tet
            sm_d_dist_z[pixel_local] = d_z + 2.0 * m1 * d_x;
            sm_d_dist_w[pixel_local] = d_w + 2.0 * m0 * d_y;

            // Chain d_m0, d_m1, d_sd → d_density, d_enter, d_exit
            // m0 = T * alpha_dist, m1 = T * (enter * alpha_dist + integral_term), sd = T^2 * ...
            // For simplicity, chain through d_density via d_alpha_dist and d_u
            let d_alpha_dist_m0 = d_m0 * t_prev;
            let d_u_from_alpha = d_alpha_dist_m0 * exp_neg_u; // d(alpha)/d(u) = exp(-u)
            let d_density_dist = safe_clip_f32(d_u_from_alpha * dt_d, MIN_VAL, MAX_VAL);
            d_density_accum += d_density_dist;

            // === aux backward (linear accumulation, no state undo needed) ===
            // Forward: aux_output[ai] += T_j * alpha * aux_data[tet_id * AUX_DIM + ai]
            // Backward: d_aux_data[ai] += dl_d_aux[ai] * T_j * alpha
            //           d_od_extra += dl_d_aux[ai] * T_prev * aux_data[ai]  (from d(T*alpha*aux)/d(od))
            let tet_id = meta_val >> 4u;
            let alpha = 1.0 - alpha_t;
            for (var axi = 0u; axi < AUX_DIM; axi++) {
                let dl_ai = dl_d_aux[pixel_idx * AUX_DIM + axi];
                let a_val = aux_data[tet_id * AUX_DIM + axi];
                // d_od contribution from aux: -T_prev * (dl * aux_val) (same chain as color)
                d_density_accum += safe_clip_f32(-dl_ai * t_prev * a_val * dist, MIN_VAL, MAX_VAL);
                d_aux_local[axi] += dl_ai * t_prev * alpha;
            }
        }

        // ===== Warp reduce + flush =====
        let reduced_d_density = warp_reduce(d_density_accum);
        let reduced_d_bc = vec3<f32>(
            warp_reduce(d_base_color_accum.x),
            warp_reduce(d_base_color_accum.y),
            warp_reduce(d_base_color_accum.z),
        );

        var reduced_d_sv: array<vec4<f32>, 5>;
        var reduced_d_sv_nf: array<vec3<f32>, 5>;
        var reduced_d_sv_nb: array<vec3<f32>, 5>;
        for (var vi = 0u; vi < 5u; vi++) {
            reduced_d_sv[vi] = vec4<f32>(
                warp_reduce(d_sv[vi].x),
                warp_reduce(d_sv[vi].y),
                warp_reduce(d_sv[vi].z),
                warp_reduce(d_sv[vi].w),
            );
            reduced_d_sv_nf[vi] = vec3<f32>(
                warp_reduce(d_sv_nf[vi].x),
                warp_reduce(d_sv_nf[vi].y),
                warp_reduce(d_sv_nf[vi].z),
            );
            reduced_d_sv_nb[vi] = vec3<f32>(
                warp_reduce(d_sv_nb[vi].x),
                warp_reduce(d_sv_nb[vi].y),
                warp_reduce(d_sv_nb[vi].z),
            );
        }

        // Warp reduce d_aux_local (no-op when AUX_DIM=0)
        var reduced_d_aux: array<f32, AUX_DIM_MAX>;
        for (var axi = 0u; axi < AUX_DIM; axi++) {
            reduced_d_aux[axi] = warp_reduce(d_aux_local[axi]);
        }

        // Lane 0 writes to global memory via atomics
        if (lane == 0u) {
            // d_interval_tet_data[vis_idx] = vec4(d_base_color.rgb, d_density)
            let td_base = vis_idx * 4u;
            atomicAdd(&d_interval_tet_data[td_base], reduced_d_bc.x);
            atomicAdd(&d_interval_tet_data[td_base + 1u], reduced_d_bc.y);
            atomicAdd(&d_interval_tet_data[td_base + 2u], reduced_d_bc.z);
            atomicAdd(&d_interval_tet_data[td_base + 3u], reduced_d_density);

            // d_interval_verts: 10 floats per vertex (d_zf, d_zb, d_off_f, d_off_b, d_nf.xyz, d_nb.xyz)
            for (var vi = 0u; vi < 5u; vi++) {
                let sv_base = (vis_idx * 5u + vi) * 10u;
                atomicAdd(&d_interval_verts[sv_base], reduced_d_sv[vi].x);
                atomicAdd(&d_interval_verts[sv_base + 1u], reduced_d_sv[vi].y);
                atomicAdd(&d_interval_verts[sv_base + 2u], reduced_d_sv[vi].z);
                atomicAdd(&d_interval_verts[sv_base + 3u], reduced_d_sv[vi].w);
                atomicAdd(&d_interval_verts[sv_base + 4u], reduced_d_sv_nf[vi].x);
                atomicAdd(&d_interval_verts[sv_base + 5u], reduced_d_sv_nf[vi].y);
                atomicAdd(&d_interval_verts[sv_base + 6u], reduced_d_sv_nf[vi].z);
                atomicAdd(&d_interval_verts[sv_base + 7u], reduced_d_sv_nb[vi].x);
                atomicAdd(&d_interval_verts[sv_base + 8u], reduced_d_sv_nb[vi].y);
                atomicAdd(&d_interval_verts[sv_base + 9u], reduced_d_sv_nb[vi].z);
            }

            // d_aux_data: per-tet aux gradients (no-op when AUX_DIM=0)
            let tet_id_flush = meta_val >> 4u;
            for (var axi = 0u; axi < AUX_DIM; axi++) {
                atomicAdd(&d_aux_data[tet_id_flush * AUX_DIM + axi], reduced_d_aux[axi]);
            }
        }
        workgroupBarrier();
    }
}
