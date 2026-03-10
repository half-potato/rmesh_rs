// Backward compute shader: gradient computation for all trainable parameters.
//
// Per-pixel backward pass: iterates all sorted tets (furthest→nearest),
// forward-replays intersection + integral, undoes pixel state, computes
// gradients via chain rule, and atomically scatters to global gradient buffers.

struct Uniforms {
    vp_col0: vec4<f32>,
    vp_col1: vec4<f32>,
    vp_col2: vec4<f32>,
    vp_col3: vec4<f32>,
    inv_vp_col0: vec4<f32>,
    inv_vp_col1: vec4<f32>,
    inv_vp_col2: vec4<f32>,
    inv_vp_col3: vec4<f32>,
    cam_pos_pad: vec4<f32>,
    screen_width: f32,
    screen_height: f32,
    tet_count: u32,
    sh_degree: u32,
    step: u32,
    _pad1: vec3<u32>,
};

// Group 0: read-only scene data + sorted indices
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> dl_d_image: array<f32>;
@group(0) @binding(2) var<storage, read> rendered_image: array<f32>;
@group(0) @binding(3) var<storage, read> vertices: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(6) var<storage, read> densities: array<f32>;
@group(0) @binding(7) var<storage, read> color_grads_buf: array<f32>;
@group(0) @binding(8) var<storage, read> circumdata: array<f32>;
@group(0) @binding(9) var<storage, read> colors_buf: array<f32>;
@group(0) @binding(10) var<storage, read> sorted_indices: array<u32>;

// Group 1: read-write gradient outputs
@group(1) @binding(0) var<storage, read_write> d_sh_coeffs: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> d_vertices: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> d_densities: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> d_color_grads: array<atomic<u32>>;

const TINY_VAL: f32 = 1e-20;

// SH Constants
const C0: f32 = 0.28209479177387814;
const C1: f32 = 0.4886025119029199;
const C2_0: f32 = 1.0925484305920792;
const C2_1: f32 = -1.0925484305920792;
const C2_2: f32 = 0.31539156525252005;
const C2_3: f32 = -1.0925484305920792;
const C2_4: f32 = 0.5462742152960396;
const C3_0: f32 = -0.5900435899266435;
const C3_1: f32 = 2.890611442640554;
const C3_2: f32 = -0.4570457994644658;
const C3_3: f32 = 0.3731763325901154;
const C3_4: f32 = -0.4570457994644658;
const C3_5: f32 = 1.445305721320277;
const C3_6: f32 = -0.5900435899266435;

// Face winding
const FACES: array<vec3<u32>, 4> = array<vec3<u32>, 4>(
    vec3<u32>(0u, 2u, 1u),
    vec3<u32>(1u, 2u, 3u),
    vec3<u32>(0u, 3u, 2u),
    vec3<u32>(3u, 0u, 1u),
);

fn phi(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return 1.0 - x * 0.5; }
    return (1.0 - exp(-x)) / x;
}

fn dphi_dx(x: f32) -> f32 {
    if (abs(x) < 1e-6) { return -0.5 + x / 3.0; }
    return (exp(-x) * (1.0 + x) - 1.0) / (x * x);
}

fn softplus(x: f32) -> f32 {
    if (x > 8.0) { return x; }
    return 0.1 * log(1.0 + exp(10.0 * x));
}

fn dsoftplus(x: f32) -> f32 {
    if (x > 8.0) { return 1.0; }
    let e = exp(10.0 * x);
    return e / (1.0 + e);
}

fn load_f32x3_v(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

fn load_f32x3_c(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors_buf[idx * 3u], colors_buf[idx * 3u + 1u], colors_buf[idx * 3u + 2u]);
}

fn load_f32x3_g(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads_buf[idx * 3u], color_grads_buf[idx * 3u + 1u], color_grads_buf[idx * 3u + 2u]);
}

// Atomic float add via compare-and-swap — inlined at call sites
// (naga does not allow ptr<storage, atomic<u32>, read_write> as function arguments)

fn eval_sh(dir: vec3<f32>, sh_degree: u32, base: u32) -> f32 {
    let x = dir.x; let y = dir.y; let z = dir.z;
    var val = C0 * sh_coeffs[base];
    if (sh_degree >= 1u) {
        val += -C1 * y * sh_coeffs[base + 1u];
        val += C1 * z * sh_coeffs[base + 2u];
        val += -C1 * x * sh_coeffs[base + 3u];
    }
    if (sh_degree >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        val += C2_0 * x * y * sh_coeffs[base + 4u];
        val += C2_1 * y * z * sh_coeffs[base + 5u];
        val += C2_2 * (2.0 * zz - xx - yy) * sh_coeffs[base + 6u];
        val += C2_3 * x * z * sh_coeffs[base + 7u];
        val += C2_4 * (xx - yy) * sh_coeffs[base + 8u];
        if (sh_degree >= 3u) {
            val += C3_0 * y * (3.0 * xx - yy) * sh_coeffs[base + 9u];
            val += C3_1 * x * y * z * sh_coeffs[base + 10u];
            val += C3_2 * y * (4.0 * zz - xx - yy) * sh_coeffs[base + 11u];
            val += C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[base + 12u];
            val += C3_4 * x * (4.0 * zz - xx - yy) * sh_coeffs[base + 13u];
            val += C3_5 * z * (xx - yy) * sh_coeffs[base + 14u];
            val += C3_6 * x * (xx - 3.0 * yy) * sh_coeffs[base + 15u];
        }
    }
    return val;
}

fn scatter_sh_grads(dir: vec3<f32>, sh_degree: u32, d_sh_result: vec3<f32>, sh_base: u32, nc: u32) {
    let x = dir.x; let y = dir.y; let z = dir.z;
    var basis: array<f32, 16>;
    basis[0] = C0;
    var n_basis = 1u;
    if (sh_degree >= 1u) {
        basis[1] = -C1 * y;
        basis[2] = C1 * z;
        basis[3] = -C1 * x;
        n_basis = 4u;
    }
    if (sh_degree >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        basis[4] = C2_0 * x * y;
        basis[5] = C2_1 * y * z;
        basis[6] = C2_2 * (2.0 * zz - xx - yy);
        basis[7] = C2_3 * x * z;
        basis[8] = C2_4 * (xx - yy);
        n_basis = 9u;
        if (sh_degree >= 3u) {
            basis[9] = C3_0 * y * (3.0 * xx - yy);
            basis[10] = C3_1 * x * y * z;
            basis[11] = C3_2 * y * (4.0 * zz - xx - yy);
            basis[12] = C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            basis[13] = C3_4 * x * (4.0 * zz - xx - yy);
            basis[14] = C3_5 * z * (xx - yy);
            basis[15] = C3_6 * x * (xx - 3.0 * yy);
            n_basis = 16u;
        }
    }
    let d_channels = array<f32, 3>(d_sh_result.x, d_sh_result.y, d_sh_result.z);
    for (var c = 0u; c < 3u; c++) {
        for (var k = 0u; k < n_basis; k++) {
            let idx = sh_base + c * nc + k;
            let add_val = d_channels[c] * basis[k];
            var ob = atomicLoad(&d_sh_coeffs[idx]);
            loop {
                let nv = bitcast<u32>(bitcast<f32>(ob) + add_val);
                let r = atomicCompareExchangeWeak(&d_sh_coeffs[idx], ob, nv);
                if (r.exchanged) { break; }
                ob = r.old_value;
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let px = global_id.x;
    let py = global_id.y;
    let w = u32(uniforms.screen_width);
    let h = u32(uniforms.screen_height);

    if (px >= w || py >= h) { return; }

    let pixel_idx = py * w + px;

    // 1. Load final pixel state
    let final_r = rendered_image[pixel_idx * 4u];
    let final_g = rendered_image[pixel_idx * 4u + 1u];
    let final_b = rendered_image[pixel_idx * 4u + 2u];
    let final_a = rendered_image[pixel_idx * 4u + 3u];

    var color = vec3<f32>(final_r, final_g, final_b);
    let transmittance = max(1.0 - final_a, TINY_VAL);
    var log_t = log(transmittance);

    // 2. Load loss gradient
    var d_color = vec3<f32>(
        dl_d_image[pixel_idx * 4u],
        dl_d_image[pixel_idx * 4u + 1u],
        dl_d_image[pixel_idx * 4u + 2u],
    );
    var d_log_t = dl_d_image[pixel_idx * 4u + 3u];

    // 3. Compute ray from pixel coordinates via inverse VP
    let ndc_x = (2.0 * (f32(px) + 0.5) / uniforms.screen_width) - 1.0;
    // wgpu framebuffer y=0 is top, but NDC y=+1 is top → flip
    let ndc_y = 1.0 - (2.0 * (f32(py) + 0.5) / uniforms.screen_height);

    let inv_vp = mat4x4<f32>(uniforms.inv_vp_col0, uniforms.inv_vp_col1, uniforms.inv_vp_col2, uniforms.inv_vp_col3);
    let near_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_clip = inv_vp * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_clip.xyz / near_clip.w;
    let far_world = far_clip.xyz / far_clip.w;

    let cam = uniforms.cam_pos_pad.xyz;
    let ray_dir = normalize(far_world - near_world);

    let num_coeffs = (uniforms.sh_degree + 1u) * (uniforms.sh_degree + 1u);
    let sh_stride = num_coeffs * 3u;
    let nc = num_coeffs;
    let tet_count = uniforms.tet_count;

    // 4. Iterate tets: furthest → nearest
    for (var tet_offset = 0u; tet_offset < tet_count; tet_offset++) {
        let tet_id = sorted_indices[tet_offset];

        // 4a. Load tet geometry
        let ti0 = indices[tet_id * 4u];
        let ti1 = indices[tet_id * 4u + 1u];
        let ti2 = indices[tet_id * 4u + 2u];
        let ti3 = indices[tet_id * 4u + 3u];

        var verts: array<vec3<f32>, 4>;
        verts[0] = load_f32x3_v(ti0);
        verts[1] = load_f32x3_v(ti1);
        verts[2] = load_f32x3_v(ti2);
        verts[3] = load_f32x3_v(ti3);

        let density_raw = densities[tet_id];
        let colors_tet = load_f32x3_c(tet_id);
        let grad = load_f32x3_g(tet_id);

        // 4b. Forward replay: ray-tet intersection
        var t_min_val = -3.402823e38;
        var t_max_val = 3.402823e38;
        var min_face = 0u;
        var max_face = 0u;
        var valid = true;

        for (var fi = 0u; fi < 4u; fi++) {
            let f = FACES[fi];
            let va = verts[f[0]];
            let vb = verts[f[1]];
            let vc = verts[f[2]];
            let n = cross(vc - va, vb - va);

            let num = dot(n, va - cam);
            let den = dot(n, ray_dir);

            if (abs(den) < 1e-20) {
                if (num > 0.0) { valid = false; }
                continue;
            }

            let t = num / den;

            if (den > 0.0) {
                if (t > t_min_val) { t_min_val = t; min_face = fi; }
            } else {
                if (t < t_max_val) { t_max_val = t; max_face = fi; }
            }
        }

        if (!valid || t_min_val >= t_max_val) { continue; }

        // 4c. Forward replay: colors & integral
        let base_offset = dot(grad, cam - verts[0]);
        let base_color = colors_tet + vec3<f32>(base_offset);
        let dc_dt = dot(grad, ray_dir);

        let c_start_raw = base_color + vec3<f32>(dc_dt * t_min_val);
        let c_end_raw = base_color + vec3<f32>(dc_dt * t_max_val);
        let c_start = max(c_start_raw, vec3<f32>(0.0));
        let c_end = max(c_end_raw, vec3<f32>(0.0));

        let dist = t_max_val - t_min_val;
        let od = max(density_raw * dist, 1e-8);

        let alpha_t = exp(-od);
        let phi_val = phi(od);
        let w0 = phi_val - alpha_t;
        let w1 = 1.0 - phi_val;
        let c_premul = c_end * w0 + c_start * w1;

        // 4d. Undo pixel state
        let prev_log_t = log_t + od;
        let t_prev = exp(prev_log_t);
        let prev_color = color - c_premul * t_prev;

        // 4e. Backward: pixel state
        let d_c_premul = d_color * t_prev;
        let d_od_state = -d_log_t;
        let d_old_color = d_color;
        let d_old_log_t = d_log_t + dot(d_color, c_premul) * t_prev;

        // 4f. Backward: compute_integral
        let dphi_val = dphi_dx(od);
        let dw0_dod = dphi_val + alpha_t;
        let dw1_dod = -dphi_val;

        let d_c_end_integral = d_c_premul * w0;
        let d_c_start_integral = d_c_premul * w1;
        let d_od_integral = dot(d_c_premul, c_end * dw0_dod + c_start * dw1_dod);

        let d_od = d_od_state + d_od_integral;

        // 4g. Backward: od and dist
        let d_density_local = d_od * dist;
        let d_dist = d_od * density_raw;
        var d_t_min = -d_dist;
        var d_t_max = d_dist;

        // 4h. Backward: color chain (max clamp → base_color → dc_dt)
        let d_c_start_raw = vec3<f32>(
            select(0.0, d_c_start_integral.x, c_start_raw.x > 0.0),
            select(0.0, d_c_start_integral.y, c_start_raw.y > 0.0),
            select(0.0, d_c_start_integral.z, c_start_raw.z > 0.0),
        );
        let d_c_end_raw = vec3<f32>(
            select(0.0, d_c_end_integral.x, c_end_raw.x > 0.0),
            select(0.0, d_c_end_integral.y, c_end_raw.y > 0.0),
            select(0.0, d_c_end_integral.z, c_end_raw.z > 0.0),
        );

        let d_base_color = d_c_start_raw + d_c_end_raw;
        let d_dc_dt = (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * t_min_val
            + (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * t_max_val;
        d_t_min += (d_c_start_raw.x + d_c_start_raw.y + d_c_start_raw.z) * dc_dt;
        d_t_max += (d_c_end_raw.x + d_c_end_raw.y + d_c_end_raw.z) * dc_dt;

        // 4h-cont. base_color = colors + grad.(cam - v0)
        let d_base_offset_scalar = d_base_color.x + d_base_color.y + d_base_color.z;
        var d_grad = (cam - verts[0]) * d_base_offset_scalar;
        let d_v0_from_base = -grad * d_base_offset_scalar;

        // dc_dt = grad . ray_dir
        d_grad += ray_dir * d_dc_dt;

        // 4i. Backward: softplus
        let centroid = (verts[0] + verts[1] + verts[2] + verts[3]) * 0.25;
        let sh_dir = normalize(centroid - cam);

        let sh_base = tet_id * sh_stride;
        var sh_result = vec3<f32>(0.0);
        sh_result.x = eval_sh(sh_dir, uniforms.sh_degree, sh_base);
        sh_result.y = eval_sh(sh_dir, uniforms.sh_degree, sh_base + nc);
        sh_result.z = eval_sh(sh_dir, uniforms.sh_degree, sh_base + 2u * nc);

        let offset_val = dot(grad, verts[0] - centroid);
        let sp_input = sh_result + vec3<f32>(0.5 + offset_val);

        let d_sp_input = vec3<f32>(
            d_base_color.x * dsoftplus(sp_input.x),
            d_base_color.y * dsoftplus(sp_input.y),
            d_base_color.z * dsoftplus(sp_input.z),
        );

        // 4j. sp_input = sh_result + 0.5 + offset
        let d_sh_result = d_sp_input;
        let d_offset_scalar = d_sp_input.x + d_sp_input.y + d_sp_input.z;

        d_grad += (verts[0] - centroid) * d_offset_scalar;
        let d_v0_from_offset = grad * d_offset_scalar * 0.75;
        let d_vother_from_offset = -grad * d_offset_scalar * 0.25;

        // 4k. SH gradients
        scatter_sh_grads(sh_dir, uniforms.sh_degree, d_sh_result, sh_base, nc);

        // 4l. Intersection gradients (dt/d(vertices))
        var d_vert_i: array<vec3<f32>, 4>;

        // t_min face
        {
            let f = FACES[min_face];
            let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
            let n = cross(vc - va, vb - va);
            let den = dot(n, ray_dir);
            let hit = cam + ray_dir * t_min_val;

            let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
            let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
            let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);

            d_vert_i[f[0]] += dt_dva * d_t_min;
            d_vert_i[f[1]] += dt_dvb * d_t_min;
            d_vert_i[f[2]] += dt_dvc * d_t_min;
        }

        // t_max face
        {
            let f = FACES[max_face];
            let va = verts[f[0]]; let vb = verts[f[1]]; let vc = verts[f[2]];
            let n = cross(vc - va, vb - va);
            let den = dot(n, ray_dir);
            let hit = cam + ray_dir * t_max_val;

            let dt_dva = (cross(va - hit, vb - vc) + n) * (1.0 / den);
            let dt_dvb = cross(va - hit, vc - va) * (1.0 / den);
            let dt_dvc = cross(va - hit, va - vb) * (1.0 / den);

            d_vert_i[f[0]] += dt_dva * d_t_max;
            d_vert_i[f[1]] += dt_dvb * d_t_max;
            d_vert_i[f[2]] += dt_dvc * d_t_max;
        }

        // 4m. Combine vertex gradients and scatter
        d_vert_i[0] += d_v0_from_base + d_v0_from_offset;
        d_vert_i[1] += d_vother_from_offset;
        d_vert_i[2] += d_vother_from_offset;
        d_vert_i[3] += d_vother_from_offset;

        let vert_indices = array<u32, 4>(ti0, ti1, ti2, ti3);
        for (var vi = 0u; vi < 4u; vi++) {
            let vidx = vert_indices[vi];
            let dv = d_vert_i[vi];
            for (var ax = 0u; ax < 3u; ax++) {
                let comp = select(select(dv.z, dv.y, ax == 1u), dv.x, ax == 0u);
                let gi = vidx * 3u + ax;
                var ob_v = atomicLoad(&d_vertices[gi]);
                loop {
                    let nv_v = bitcast<u32>(bitcast<f32>(ob_v) + comp);
                    let r_v = atomicCompareExchangeWeak(&d_vertices[gi], ob_v, nv_v);
                    if (r_v.exchanged) { break; }
                    ob_v = r_v.old_value;
                }
            }
        }

        {
            var ob_d = atomicLoad(&d_densities[tet_id]);
            loop {
                let nv_d = bitcast<u32>(bitcast<f32>(ob_d) + d_density_local);
                let r_d = atomicCompareExchangeWeak(&d_densities[tet_id], ob_d, nv_d);
                if (r_d.exchanged) { break; }
                ob_d = r_d.old_value;
            }
        }

        let dg = d_grad;
        for (var gi2 = 0u; gi2 < 3u; gi2++) {
            let gc = select(select(dg.z, dg.y, gi2 == 1u), dg.x, gi2 == 0u);
            let gidx = tet_id * 3u + gi2;
            var ob_g = atomicLoad(&d_color_grads[gidx]);
            loop {
                let nv_g = bitcast<u32>(bitcast<f32>(ob_g) + gc);
                let r_g = atomicCompareExchangeWeak(&d_color_grads[gidx], ob_g, nv_g);
                if (r_g.exchanged) { break; }
                ob_g = r_g.old_value;
            }
        }

        // 4n. Update pixel state for next iteration
        color = prev_color;
        log_t = prev_log_t;
        d_color = d_old_color;
        d_log_t = d_old_log_t;
    }
}
