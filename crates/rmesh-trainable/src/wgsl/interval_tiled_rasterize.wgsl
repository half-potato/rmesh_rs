// Interval tiled rasterize: 1 warp (32 threads) per tile.
//
// Same tile infrastructure as rasterize_compute.wgsl (scanline fill, prefix sum).
// Reads pre-computed screen triangles from interval_generate instead of doing
// ray-tet intersection. For each covered pixel: find containing triangle,
// bary interp → z_front/z_back NDC, linearize → dist, volume integral, composite.
//
// One workgroup per tile. Dispatch: (num_tiles, 1, 1).

#import rmesh::math::{MAX_VAL, safe_clip_v3f, safe_exp_f32, safe_div_f32, phi}

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

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> interval_verts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> interval_tet_data: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> interval_meta: array<u32>;
@group(0) @binding(4) var<storage, read> tile_sort_values: array<u32>;
@group(0) @binding(5) var<storage, read> tile_ranges: array<u32>;
@group(0) @binding(6) var<storage, read> tile_uniforms: TileUniforms;
@group(0) @binding(7) var<storage, read_write> rendered_image: array<f32>;
@group(0) @binding(8) var<storage, read_write> xyzd_image: array<f32>;       // [W*H*4]
@group(0) @binding(9) var<storage, read_write> distortion_image: array<f32>; // [W*H*5]
@group(0) @binding(10) var<storage, read> aux_data: array<f32>;             // [M * AUX_DIM]
@group(0) @binding(11) var<storage, read_write> aux_image: array<f32>;      // [W*H*AUX_DIM]

const AUX_DIM: u32 = /*AUX_DIM*/0u;
const SM_AUX_SIZE: u32 = /*SM_AUX_SIZE*/1u;

// Workgroup shared memory
var<workgroup> sm_color: array<vec4<f32>, 256>;    // .xyz = color_accum, .w = log_t
var<workgroup> sm_xyzd: array<vec4<f32>, 256>;     // normal.xyz + depth
var<workgroup> sm_dist: array<array<f32, 5>, 256>;  // DistortionState5 (x,y,z,w,v)
var<workgroup> sm_aux: array<f32, SM_AUX_SIZE>;     // variable aux [256 * AUX_DIM]
var<workgroup> sm_xl: array<i32, 16>;               // scanline left x per row
var<workgroup> sm_xr: array<i32, 16>;               // scanline right x per row
var<workgroup> sm_prefix: array<u32, 17>;            // prefix sum of row widths

// Barycentric coordinates of point pt in triangle (a, b, c) in 2D.
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

// Test if point is inside triangle (2D, using cross products).
fn point_in_triangle(pt: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let d0 = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    let d1 = (c.x - b.x) * (pt.y - b.y) - (c.y - b.y) * (pt.x - b.x);
    let d2 = (a.x - c.x) * (pt.y - c.y) - (a.y - c.y) * (pt.x - c.x);
    let has_neg = (d0 < 0.0) || (d1 < 0.0) || (d2 < 0.0);
    let has_pos = (d0 > 0.0) || (d1 > 0.0) || (d2 > 0.0);
    return !(has_neg && has_pos);
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

    // Initialize per-pixel state (32 threads x pixels each)
    for (var i = lane; i < TS * TS; i += 32u) {
        sm_color[i] = vec4<f32>(0.0);
        sm_xyzd[i] = vec4<f32>(0.0);
        sm_dist[i] = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    }
    // Initialize aux shared memory
    for (var i = lane; i < TS * TS * AUX_DIM; i += 32u) {
        sm_aux[i] = 0.0;
    }
    workgroupBarrier();

    // Process tets front-to-back (nearest first)
    var cursor = range_end;
    while (cursor > range_start) {
        cursor -= 1u;
        let vis_idx = tile_sort_values[cursor];

        // Read meta: low 4 bits = num_silhouette_verts, rest = tet_id
        let meta_val = interval_meta[vis_idx];
        let num_sil = meta_val & 0xFu;

        // Skip degenerate tets (num_sil == 0)
        if (num_sil == 0u) {
            continue;
        }

        // Load pre-computed silhouette vertices (pixel coords, global)
        // Layout: 4 vec4 per vertex (pos+z, off, n_front, n_back)
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

        // Load per-tet material
        let tet_data = interval_tet_data[vis_idx];
        let base_color = tet_data.xyz;
        let density_raw = tet_data.w;

        // Convert silhouette to tile-local coords for scanline
        var proj: array<vec2<f32>, 4>;
        for (var i = 0u; i < min(num_sil, 4u); i++) {
            proj[i] = vec2<f32>(sv_pix[i].x - tile_ox, sv_pix[i].y - tile_oy);
        }

        // Scanline fill on the silhouette polygon (3 or 4 verts)
        if (lane < TS) {
            var xl_f: f32 = 1e10;
            var xr_f: f32 = -1e10;

            let yc = f32(lane) + 0.5;

            // Test all edges of the silhouette polygon
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

            // Include vertices within this pixel row
            for (var v = 0u; v < num_sil; v++) {
                let vy = proj[v].y;
                if (vy >= yc - 0.5 && vy < yc + 0.5) {
                    xl_f = min(xl_f, proj[v].x);
                    xr_f = max(xr_f, proj[v].x);
                }
            }

            if (xl_f <= xr_f) {
                let eps = 0.001;
                var xl_i = max(i32(ceil(xl_f - 0.5 - eps)), 0);
                var xr_i = min(i32(floor(xr_f - 0.5 + eps)), i32(TS) - 1);

                // Trim saturated pixels
                while (xl_i <= xr_i && sm_color[lane * TS + u32(xl_i)].w < -13.8) {
                    xl_i++;
                }
                while (xr_i >= xl_i && sm_color[lane * TS + u32(xr_i)].w < -13.8) {
                    xr_i--;
                }

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

        // Prefix sum of row widths (thread 0)
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

        // Process covered pixels
        for (var idx = lane; idx < total; idx += 32u) {
            // Map linear index to (row, col) via prefix sum
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

            // Skip saturated pixels
            if (sm_color[pixel_local].w < -13.8) {
                continue;
            }

            // Pixel center in global coords
            let pix_center = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

            // Find which triangle of the fan contains this pixel.
            // Fan triangles: (sv[0], sv[1], sv[4]), (sv[1], sv[2], sv[4]),
            //                (sv[2], sv[3], sv[4]), (sv[3], sv[0], sv[4])
            // For num_sil=3, triangle 2 is (sv[2], sv[3], sv[4]) where sv[3]=sv[0] — degenerate.
            var found = false;
            var z_front_ndc: f32 = 0.0;
            var z_back_ndc: f32 = 0.0;
            var off_front: f32 = 0.0;
            var off_back: f32 = 0.0;
            var bary = vec3<f32>(0.0);
            var ai: u32 = 0u;
            var bi: u32 = 0u;
            let ci: u32 = 4u; // center vertex (always slot 4)

            for (var tri = 0u; tri < num_sil; tri++) {
                let tri_a = tri;
                let tri_b = (tri + 1u) % num_sil;

                if (point_in_triangle(pix_center, sv_pix[tri_a], sv_pix[tri_b], sv_pix[ci])) {
                    ai = tri_a;
                    bi = tri_b;
                    bary = bary2d(pix_center, sv_pix[ai], sv_pix[bi], sv_pix[ci]);

                    z_front_ndc = bary.x * sv_zf[ai] + bary.y * sv_zf[bi] + bary.z * sv_zf[ci];
                    z_back_ndc = bary.x * sv_zb[ai] + bary.y * sv_zb[bi] + bary.z * sv_zb[ci];
                    off_front = bary.x * sv_off_f[ai] + bary.y * sv_off_f[bi] + bary.z * sv_off_f[ci];
                    off_back = bary.x * sv_off_b[ai] + bary.y * sv_off_b[bi] + bary.z * sv_off_b[ci];
                    found = true;
                    break;
                }
            }

            if (!found) {
                continue;
            }

            // Linearize NDC depths -> view-space Z
            let z_front_clamped = clamp(z_front_ndc, 0.0, 1.0);
            let z_back_clamped = clamp(z_back_ndc, 0.0, 1.0);
            let z_f = near * far / (far - z_front_clamped * range_nf);
            let z_b = near * far / (far - z_back_clamped * range_nf);

            // Ray direction scale factor from pixel position
            let x_cam = (f32(px) + 0.5 - cx_cam) / fx;
            let y_cam = (f32(py) + 0.5 - cy_cam) / fy;
            let ray_scale = length(vec3<f32>(x_cam, y_cam, 1.0));

            // Distance through tet
            let dist = abs(z_b - z_f) * ray_scale;

            // Colors at entry/exit from interpolated offsets
            let c_front = max(base_color + vec3<f32>(off_front), vec3<f32>(0.0));
            let c_back = max(base_color + vec3<f32>(off_back), vec3<f32>(0.0));

            // Volume rendering integral
            let od = clamp(density_raw * dist, 1e-8, 88.0);
            let alpha_t = safe_exp_f32(-od);
            let phi_val = phi(od);
            let w0 = phi_val - alpha_t;
            let w1 = 1.0 - phi_val;
            // c_back = exit color first (matching interval_fragment convention)
            let c_premul = safe_clip_v3f(c_back * w0 + c_front * w1, 0.0, MAX_VAL);

            // Composite color into shared memory
            let state = sm_color[pixel_local];
            let T_j = safe_exp_f32(state.w);

            if (T_j >= 1e-6) {
                sm_color[pixel_local] = vec4<f32>(
                    state.xyz + c_premul * T_j,
                    state.w - od,
                );

                // --- xyzd: interpolate normals + depth via volume integral ---
                let n_f = bary.x * sv_nf[ai] + bary.y * sv_nf[bi] + bary.z * sv_nf[ci];
                let n_b = bary.x * sv_nb[ai] + bary.y * sv_nb[bi] + bary.z * sv_nb[ci];
                let normal_contrib = n_b * w0 + n_f * w1;
                let depth_contrib = w0 * z_b + w1 * z_f;
                sm_xyzd[pixel_local] = sm_xyzd[pixel_local] + T_j * vec4<f32>(normal_contrib, depth_contrib);

                // --- distortion: Mip-NeRF 360 distortion state update ---
                let enter = z_f * ray_scale;
                let exit_d = z_b * ray_scale;
                let dt_d = exit_d - enter;
                let u = density_raw * dt_d;
                let exp_neg_u = safe_exp_f32(-u);
                let alpha_dist = max(1.0 - exp_neg_u, 0.0);
                let m0 = T_j * alpha_dist;
                let m1 = T_j * (enter * alpha_dist + safe_div_f32(1.0 - (1.0 + u) * exp_neg_u, density_raw));
                let sd = T_j * T_j * safe_div_f32(1.0 - 2.0 * u * exp_neg_u - exp_neg_u * exp_neg_u, density_raw);

                var ds = sm_dist[pixel_local];
                ds[0] += 2.0 * m1 * ds[2];  // x += 2*m1*z
                ds[1] += 2.0 * m0 * ds[3];  // y += 2*m0*w
                ds[2] += m0;                  // z += m0
                ds[3] += m1;                  // w += m1
                ds[4] += sd;                  // v += self_dist
                sm_dist[pixel_local] = ds;

                // --- aux: linear accumulation (no-op when AUX_DIM=0) ---
                let alpha = 1.0 - alpha_t;
                let tet_id = meta_val >> 4u;
                for (var axi = 0u; axi < AUX_DIM; axi++) {
                    sm_aux[pixel_local * AUX_DIM + axi] += T_j * alpha * aux_data[tet_id * AUX_DIM + axi];
                }
            }
        }
        workgroupBarrier();
    }

    // Write output (32 threads x pixels each)
    for (var i = lane; i < TS * TS; i += 32u) {
        let row = i / TS;
        let col = i % TS;
        let px = tile_x * TS + col;
        let py = tile_y * TS + row;
        if (px < w && py < h) {
            let pixel_idx = py * w + px;
            let state = sm_color[i];
            let T_final = safe_exp_f32(state.w);
            rendered_image[pixel_idx * 4u] = state.x;
            rendered_image[pixel_idx * 4u + 1u] = state.y;
            rendered_image[pixel_idx * 4u + 2u] = state.z;
            rendered_image[pixel_idx * 4u + 3u] = 1.0 - T_final;

            // xyzd output (normal.xyz + depth)
            let xyzd = sm_xyzd[i];
            xyzd_image[pixel_idx * 4u] = xyzd.x;
            xyzd_image[pixel_idx * 4u + 1u] = xyzd.y;
            xyzd_image[pixel_idx * 4u + 2u] = xyzd.z;
            xyzd_image[pixel_idx * 4u + 3u] = xyzd.w;

            // distortion output (5 channels)
            let ds = sm_dist[i];
            distortion_image[pixel_idx * 5u] = ds[0];
            distortion_image[pixel_idx * 5u + 1u] = ds[1];
            distortion_image[pixel_idx * 5u + 2u] = ds[2];
            distortion_image[pixel_idx * 5u + 3u] = ds[3];
            distortion_image[pixel_idx * 5u + 4u] = ds[4];

            // aux output (no-op when AUX_DIM=0)
            for (var axi = 0u; axi < AUX_DIM; axi++) {
                aux_image[pixel_idx * AUX_DIM + axi] = sm_aux[i * AUX_DIM + axi];
            }
        }
    }
}
