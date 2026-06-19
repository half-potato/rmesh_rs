// Interval chain-back: per-tet compute shader.
//
// Maps screen triangle attribute gradients back to tet parameters.
// 1 thread per visible tet.
//
// Reads:
//   d_interval_verts[vis_idx * 5 + i] = vec4(d_zf, d_zb, d_off_f, d_off_b)
//   d_interval_tet_data[vis_idx] = vec4(d_base_color.rgb, d_density)
//   Original tet data: vertices, indices, VP matrix, cam_pos, color_grads
//
// Writes (via atomics):
//   d_vertices, d_densities, d_color_grads, d_base_colors

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

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

const MIN_VAL: f32 = -1e+20;
const MAX_VAL: f32 = 1e+20;

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

// Group 0: read-only inputs
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> color_grads: array<f32>;
@group(0) @binding(4) var<storage, read> compact_tet_ids: array<u32>;
@group(0) @binding(5) var<storage, read> indirect_args: DrawIndirectArgs;
@group(0) @binding(6) var<storage, read> interval_meta: array<u32>;
@group(0) @binding(7) var<storage, read> d_interval_verts: array<f32>;
@group(0) @binding(8) var<storage, read> d_interval_tet_data: array<f32>;
@group(0) @binding(9) var<storage, read> vertex_normals: array<f32>;

// Group 1: gradient outputs
@group(1) @binding(0) var<storage, read_write> d_vertices: array<atomic<f32>>;
@group(1) @binding(1) var<storage, read_write> d_densities: array<atomic<f32>>;
@group(1) @binding(2) var<storage, read_write> d_color_grads: array<atomic<f32>>;
@group(1) @binding(3) var<storage, read_write> d_base_colors: array<atomic<f32>>;
@group(1) @binding(4) var<storage, read_write> d_vertex_normals: array<atomic<f32>>;

// Maps vertex index -> TET_FACES index where that vertex is face[3] (opposite)
const OPPOSITE_FACE: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 0u);

// Face table: (a, b, c, opposite_vertex)
const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Edge table
const TET_EDGES: array<vec2<u32>, 6> = array<vec2<u32>, 6>(
    vec2<u32>(0u, 1u),
    vec2<u32>(0u, 2u),
    vec2<u32>(0u, 3u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 3u),
    vec2<u32>(2u, 3u),
);

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

fn load_normal(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertex_normals[idx * 3u], vertex_normals[idx * 3u + 1u], vertex_normals[idx * 3u + 2u]);
}

fn classify_silhouette(p: array<vec2<f32>, 4>) -> u32 {
    for (var v = 0u; v < 4u; v++) {
        let face = TET_FACES[v];
        let a = p[face[0]];
        let b = p[face[1]];
        let c = p[face[2]];
        let pt = p[face[3]];
        let d0 = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
        let d1 = (c.x - b.x) * (pt.y - b.y) - (c.y - b.y) * (pt.x - b.x);
        let d2 = (a.x - c.x) * (pt.y - c.y) - (a.y - c.y) * (pt.x - c.x);
        let all_pos = d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0;
        let all_neg = d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0;
        if all_pos || all_neg { return face[3]; }
    }
    return 4u;
}

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

fn line_intersect_t(a1: vec2<f32>, a2: vec2<f32>, b1: vec2<f32>, b2: vec2<f32>) -> f32 {
    let d1 = a2 - a1;
    let d2 = b2 - b1;
    let denom = d1.x * d2.y - d1.y * d2.x;
    if abs(denom) < 1e-10 { return -1.0; }
    let d = b1 - a1;
    return (d.x * d2.y - d.y * d2.x) / denom;
}

fn find_crossing_edges(p: array<vec2<f32>, 4>) -> vec2<u32> {
    let pairs = array<vec2<u32>, 3>(
        vec2<u32>(0u, 5u),
        vec2<u32>(1u, 4u),
        vec2<u32>(2u, 3u),
    );
    for (var i = 0u; i < 3u; i++) {
        let ea = TET_EDGES[pairs[i].x];
        let eb = TET_EDGES[pairs[i].y];
        let a1 = p[ea.x]; let a2 = p[ea.y];
        let b1 = p[eb.x]; let b2 = p[eb.y];
        let ta = line_intersect_t(a1, a2, b1, b2);
        let tb = line_intersect_t(b1, b2, a1, a2);
        if ta > 0.0 && ta < 1.0 && tb > 0.0 && tb < 1.0 {
            return pairs[i];
        }
    }
    return pairs[0];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vis_idx = global_id.x;
    let visible_count = indirect_args.instance_count;

    if vis_idx >= visible_count {
        return;
    }

    let meta_val = interval_meta[vis_idx];
    let num_sil = meta_val & 0xFu;
    let tet_id = meta_val >> 4u;

    if (num_sil == 0u) {
        return;
    }

    // Read accumulated screen-space gradients
    // d_interval_verts: 5 vertices × 10 floats (d_zf, d_zb, d_off_f, d_off_b, d_nf.xyz, d_nb.xyz)
    var d_sv: array<vec4<f32>, 5>;
    var d_sv_nf: array<vec3<f32>, 5>;
    var d_sv_nb: array<vec3<f32>, 5>;
    for (var i = 0u; i < 5u; i++) {
        let base = (vis_idx * 5u + i) * 10u;
        d_sv[i] = vec4<f32>(
            d_interval_verts[base],
            d_interval_verts[base + 1u],
            d_interval_verts[base + 2u],
            d_interval_verts[base + 3u],
        );
        d_sv_nf[i] = vec3<f32>(
            d_interval_verts[base + 4u],
            d_interval_verts[base + 5u],
            d_interval_verts[base + 6u],
        );
        d_sv_nb[i] = vec3<f32>(
            d_interval_verts[base + 7u],
            d_interval_verts[base + 8u],
            d_interval_verts[base + 9u],
        );
    }

    // d_interval_tet_data: vec4(d_base_color.rgb, d_density)
    let td_base = vis_idx * 4u;
    let d_bc = vec3<f32>(
        d_interval_tet_data[td_base],
        d_interval_tet_data[td_base + 1u],
        d_interval_tet_data[td_base + 2u],
    );
    let d_density_direct = d_interval_tet_data[td_base + 3u];

    // Load tet geometry
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];
    let vidx = array<u32, 4>(i0, i1, i2, i3);

    var v_world: array<vec3<f32>, 4>;
    v_world[0] = load_f32x3(i0 * 3u);
    v_world[1] = load_f32x3(i1 * 3u);
    v_world[2] = load_f32x3(i2 * 3u);
    v_world[3] = load_f32x3(i3 * 3u);

    // Load per-vertex normals
    var vn: array<vec3<f32>, 4>;
    vn[0] = load_normal(i0);
    vn[1] = load_normal(i1);
    vn[2] = load_normal(i2);
    vn[3] = load_normal(i3);

    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let cam_pos = uniforms.cam_pos_pad.xyz;
    let grad = load_grad(tet_id);
    let W = uniforms.screen_width;
    let H = uniforms.screen_height;

    // Clip space
    var clip: array<vec4<f32>, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        clip[vi] = vp * vec4<f32>(v_world[vi], 1.0);
    }

    // NDC + pixel
    var ndc_z: array<f32, 4>;
    var pixel_xy: array<vec2<f32>, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        let inv_w = select(0.0, 1.0 / clip[vi].w, abs(clip[vi].w) > 1e-10);
        let ndc_xy = clip[vi].xy * inv_w;
        ndc_z[vi] = clip[vi].z * inv_w;
        pixel_xy[vi] = vec2<f32>(
            (ndc_xy.x + 1.0) * 0.5 * W,
            (1.0 - ndc_xy.y) * 0.5 * H,
        );
    }

    // Accumulate gradients for tet parameters
    var d_vert: array<vec3<f32>, 4>;
    d_vert[0] = vec3<f32>(0.0);
    d_vert[1] = vec3<f32>(0.0);
    d_vert[2] = vec3<f32>(0.0);
    d_vert[3] = vec3<f32>(0.0);
    var d_vn: array<vec3<f32>, 4>;
    d_vn[0] = vec3<f32>(0.0);
    d_vn[1] = vec3<f32>(0.0);
    d_vn[2] = vec3<f32>(0.0);
    d_vn[3] = vec3<f32>(0.0);
    var d_grad_accum = vec3<f32>(0.0);

    // 1. d_density: directly from d_interval_tet_data.w
    // (will be written at end)

    // 2. d_base_colors: directly from d_interval_tet_data.xyz
    // base_color = color + dot(grad, cam_pos - v_world[0])
    // d_color (=d_base_colors) += d_bc
    // d_grad += d_bc_scalar * (cam_pos - v_world[0])
    // d_v_world[0] -= d_bc_scalar * grad
    let d_bc_scalar = d_bc.x + d_bc.y + d_bc.z;
    d_grad_accum += (cam_pos - v_world[0]) * d_bc_scalar;
    d_vert[0] -= grad * d_bc_scalar;

    // 3. Chain gradients from screen-space vertex attributes back to tet params.
    //
    // For silhouette vertices (slots 0..num_sil-1):
    //   ndc_z at silhouette vertex i = ndc_z[tet_vtx_i]
    //   offset at silhouette vertex i = dot(grad, v_world[tet_vtx_i] - cam_pos)
    //
    // For center vertex (slot 4):
    //   ndc_z_front/back and offset_front/back depend on case.
    //
    // ndc_z = clip_z / clip_w where clip = vp * (v_world, 1)
    // d(ndc_z)/d(v_world) = (vp_z_row - ndc_z * vp_w_row) / clip_w (row = col of column-major)

    let vp_z_row = vec3<f32>(vp[0].z, vp[1].z, vp[2].z); // z component of each column
    let vp_w_row = vec3<f32>(vp[0].w, vp[1].w, vp[2].w); // w component of each column

    let interior_vtx = classify_silhouette(pixel_xy);

    if (interior_vtx < 4u) {
        // Case 1: 3 silhouette + center
        let face = TET_FACES[OPPOSITE_FACE[interior_vtx]];
        let si = array<u32, 3>(face[0], face[1], face[2]);
        let center = face[3];

        let bary = bary2d(pixel_xy[center], pixel_xy[si[0]], pixel_xy[si[1]], pixel_xy[si[2]]);

        let z_face = bary.x * ndc_z[si[0]] + bary.y * ndc_z[si[1]] + bary.z * ndc_z[si[2]];
        let z_center = ndc_z[center];
        let face_is_front = z_face <= z_center;

        // Silhouette vertices (slots 0, 1, 2): z_front = z_back = ndc_z[si[i]]
        for (var i = 0u; i < 3u; i++) {
            let vi = si[i];
            let d_ndc_z_total = d_sv[i].x + d_sv[i].y;
            let inv_w = select(0.0, 1.0 / clip[vi].w, abs(clip[vi].w) > 1e-10);
            let d_v = (vp_z_row - ndc_z[vi] * vp_w_row) * inv_w * d_ndc_z_total;
            d_vert[vi] += safe_clip_v3f(d_v, MIN_VAL, MAX_VAL);

            let d_off_total = d_sv[i].z + d_sv[i].w;
            d_vert[vi] += safe_clip_v3f(grad * d_off_total, MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f((v_world[vi] - cam_pos) * d_off_total, MIN_VAL, MAX_VAL);

            // Normal chain-back for silhouette: n_front = n_back = vn[vi]
            d_vn[vi] += d_sv_nf[i] + d_sv_nb[i];
        }

        // Slot 3 is a copy of slot 0
        {
            let vi = si[0];
            let d_ndc_z_total = d_sv[3].x + d_sv[3].y;
            let inv_w = select(0.0, 1.0 / clip[vi].w, abs(clip[vi].w) > 1e-10);
            let d_v = (vp_z_row - ndc_z[vi] * vp_w_row) * inv_w * d_ndc_z_total;
            d_vert[vi] += safe_clip_v3f(d_v, MIN_VAL, MAX_VAL);
            let d_off_total = d_sv[3].z + d_sv[3].w;
            d_vert[vi] += safe_clip_v3f(grad * d_off_total, MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f((v_world[vi] - cam_pos) * d_off_total, MIN_VAL, MAX_VAL);
            d_vn[vi] += d_sv_nf[3] + d_sv_nb[3];
        }

        // Center vertex (slot 4)
        let d_center = d_sv[4];

        var d_z_face: f32;
        var d_z_center_ndc: f32;
        var d_face_offset: f32;
        var d_center_offset: f32;
        if (face_is_front) {
            d_z_face = d_center.x;
            d_z_center_ndc = d_center.y;
            d_face_offset = d_center.z;
            d_center_offset = d_center.w;
        } else {
            d_z_center_ndc = d_center.x;
            d_z_face = d_center.y;
            d_center_offset = d_center.z;
            d_face_offset = d_center.w;
        }

        let bary_arr = array<f32, 3>(bary.x, bary.y, bary.z);
        for (var i = 0u; i < 3u; i++) {
            let vi = si[i];
            let d_ndc_z_i = bary_arr[i] * d_z_face;
            let inv_w = select(0.0, 1.0 / clip[vi].w, abs(clip[vi].w) > 1e-10);
            let d_v = (vp_z_row - ndc_z[vi] * vp_w_row) * inv_w * d_ndc_z_i;
            d_vert[vi] += safe_clip_v3f(d_v, MIN_VAL, MAX_VAL);
        }

        {
            let inv_w = select(0.0, 1.0 / clip[center].w, abs(clip[center].w) > 1e-10);
            let d_v = (vp_z_row - ndc_z[center] * vp_w_row) * inv_w * d_z_center_ndc;
            d_vert[center] += safe_clip_v3f(d_v, MIN_VAL, MAX_VAL);
        }

        for (var i = 0u; i < 3u; i++) {
            let vi = si[i];
            let d_off_i = bary_arr[i] * d_face_offset;
            d_vert[vi] += safe_clip_v3f(grad * d_off_i, MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f((v_world[vi] - cam_pos) * d_off_i, MIN_VAL, MAX_VAL);
        }

        d_vert[center] += safe_clip_v3f(grad * d_center_offset, MIN_VAL, MAX_VAL);
        d_grad_accum += safe_clip_v3f((v_world[center] - cam_pos) * d_center_offset, MIN_VAL, MAX_VAL);

        // Normal chain-back for center vertex (slot 4)
        // face_is_front: n_front=n_face, n_back=n_center; else swap
        var d_n_face: vec3<f32>;
        var d_n_center: vec3<f32>;
        if (face_is_front) {
            d_n_face = d_sv_nf[4];
            d_n_center = d_sv_nb[4];
        } else {
            d_n_center = d_sv_nf[4];
            d_n_face = d_sv_nb[4];
        }
        // n_face = bary.x * vn[si[0]] + bary.y * vn[si[1]] + bary.z * vn[si[2]]
        for (var i = 0u; i < 3u; i++) {
            d_vn[si[i]] += bary_arr[i] * d_n_face;
        }
        // n_center = vn[center]
        d_vn[center] += d_n_center;

    } else {
        // Case 2: 4 silhouette + center (edge crossing)
        let crossing_edges = find_crossing_edges(pixel_xy);
        let ea = TET_EDGES[crossing_edges.x];
        let eb = TET_EDGES[crossing_edges.y];

        let ta = line_intersect_t(pixel_xy[ea.x], pixel_xy[ea.y], pixel_xy[eb.x], pixel_xy[eb.y]);
        let tb = line_intersect_t(pixel_xy[eb.x], pixel_xy[eb.y], pixel_xy[ea.x], pixel_xy[ea.y]);

        let z_a = mix(ndc_z[ea.x], ndc_z[ea.y], ta);
        let z_b = mix(ndc_z[eb.x], ndc_z[eb.y], tb);
        let a_is_front = z_a <= z_b;

        // Silhouette layout: sv[0]=ea.x, sv[1]=eb.x, sv[2]=ea.y, sv[3]=eb.y
        let sv_map = array<u32, 4>(ea.x, eb.x, ea.y, eb.y);
        for (var i = 0u; i < 4u; i++) {
            let vi = sv_map[i];
            let d_ndc_z_total = d_sv[i].x + d_sv[i].y;
            let inv_w = select(0.0, 1.0 / clip[vi].w, abs(clip[vi].w) > 1e-10);
            let d_v = (vp_z_row - ndc_z[vi] * vp_w_row) * inv_w * d_ndc_z_total;
            d_vert[vi] += safe_clip_v3f(d_v, MIN_VAL, MAX_VAL);

            let d_off_total = d_sv[i].z + d_sv[i].w;
            d_vert[vi] += safe_clip_v3f(grad * d_off_total, MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f((v_world[vi] - cam_pos) * d_off_total, MIN_VAL, MAX_VAL);

            // Normal chain-back for silhouette: n_front = n_back = vn[vi]
            d_vn[vi] += d_sv_nf[i] + d_sv_nb[i];
        }

        // Center vertex (slot 4)
        let d_center = d_sv[4];
        var d_z_a: f32;
        var d_z_b: f32;
        var d_off_a: f32;
        var d_off_b: f32;
        if (a_is_front) {
            d_z_a = d_center.x;
            d_z_b = d_center.y;
            d_off_a = d_center.z;
            d_off_b = d_center.w;
        } else {
            d_z_b = d_center.x;
            d_z_a = d_center.y;
            d_off_b = d_center.z;
            d_off_a = d_center.w;
        }

        {
            let d_ndc_ea_x = (1.0 - ta) * d_z_a;
            let d_ndc_ea_y = ta * d_z_a;
            let inv_w_x = select(0.0, 1.0 / clip[ea.x].w, abs(clip[ea.x].w) > 1e-10);
            let inv_w_y = select(0.0, 1.0 / clip[ea.y].w, abs(clip[ea.y].w) > 1e-10);
            d_vert[ea.x] += safe_clip_v3f((vp_z_row - ndc_z[ea.x] * vp_w_row) * inv_w_x * d_ndc_ea_x, MIN_VAL, MAX_VAL);
            d_vert[ea.y] += safe_clip_v3f((vp_z_row - ndc_z[ea.y] * vp_w_row) * inv_w_y * d_ndc_ea_y, MIN_VAL, MAX_VAL);
        }

        {
            let d_ndc_eb_x = (1.0 - tb) * d_z_b;
            let d_ndc_eb_y = tb * d_z_b;
            let inv_w_x = select(0.0, 1.0 / clip[eb.x].w, abs(clip[eb.x].w) > 1e-10);
            let inv_w_y = select(0.0, 1.0 / clip[eb.y].w, abs(clip[eb.y].w) > 1e-10);
            d_vert[eb.x] += safe_clip_v3f((vp_z_row - ndc_z[eb.x] * vp_w_row) * inv_w_x * d_ndc_eb_x, MIN_VAL, MAX_VAL);
            d_vert[eb.y] += safe_clip_v3f((vp_z_row - ndc_z[eb.y] * vp_w_row) * inv_w_y * d_ndc_eb_y, MIN_VAL, MAX_VAL);
        }

        {
            let d_off_ea_x = (1.0 - ta) * d_off_a;
            let d_off_ea_y = ta * d_off_a;
            d_vert[ea.x] += safe_clip_v3f(grad * d_off_ea_x, MIN_VAL, MAX_VAL);
            d_vert[ea.y] += safe_clip_v3f(grad * d_off_ea_y, MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f((v_world[ea.x] - cam_pos) * d_off_ea_x + (v_world[ea.y] - cam_pos) * d_off_ea_y, MIN_VAL, MAX_VAL);
        }

        {
            let d_off_eb_x = (1.0 - tb) * d_off_b;
            let d_off_eb_y = tb * d_off_b;
            d_vert[eb.x] += safe_clip_v3f(grad * d_off_eb_x, MIN_VAL, MAX_VAL);
            d_vert[eb.y] += safe_clip_v3f(grad * d_off_eb_y, MIN_VAL, MAX_VAL);
            d_grad_accum += safe_clip_v3f((v_world[eb.x] - cam_pos) * d_off_eb_x + (v_world[eb.y] - cam_pos) * d_off_eb_y, MIN_VAL, MAX_VAL);
        }

        // Normal chain-back for center vertex (slot 4)
        // n_ea = mix(vn[ea.x], vn[ea.y], ta), n_eb = mix(vn[eb.x], vn[eb.y], tb)
        // a_is_front: n_front=n_ea, n_back=n_eb; else swap
        var d_n_a: vec3<f32>;
        var d_n_b_edge: vec3<f32>;
        if (a_is_front) {
            d_n_a = d_sv_nf[4];
            d_n_b_edge = d_sv_nb[4];
        } else {
            d_n_b_edge = d_sv_nf[4];
            d_n_a = d_sv_nb[4];
        }
        // n_ea = (1-ta)*vn[ea.x] + ta*vn[ea.y]
        d_vn[ea.x] += (1.0 - ta) * d_n_a;
        d_vn[ea.y] += ta * d_n_a;
        // n_eb = (1-tb)*vn[eb.x] + tb*vn[eb.y]
        d_vn[eb.x] += (1.0 - tb) * d_n_b_edge;
        d_vn[eb.y] += tb * d_n_b_edge;
    }

    // Flush to global via atomics
    atomicAdd(&d_densities[tet_id], d_density_direct);

    for (var vi = 0u; vi < 4u; vi++) {
        let gi = vidx[vi] * 3u;
        let dv = safe_clip_v3f(d_vert[vi], MIN_VAL, MAX_VAL);
        atomicAdd(&d_vertices[gi], dv.x);
        atomicAdd(&d_vertices[gi + 1u], dv.y);
        atomicAdd(&d_vertices[gi + 2u], dv.z);
    }

    let d_grad = safe_clip_v3f(d_grad_accum, MIN_VAL, MAX_VAL);
    atomicAdd(&d_color_grads[tet_id * 3u], d_grad.x);
    atomicAdd(&d_color_grads[tet_id * 3u + 1u], d_grad.y);
    atomicAdd(&d_color_grads[tet_id * 3u + 2u], d_grad.z);

    let d_bc_out = safe_clip_v3f(d_bc, MIN_VAL, MAX_VAL);
    atomicAdd(&d_base_colors[tet_id * 3u], d_bc_out.x);
    atomicAdd(&d_base_colors[tet_id * 3u + 1u], d_bc_out.y);
    atomicAdd(&d_base_colors[tet_id * 3u + 2u], d_bc_out.z);

    // Flush vertex normal gradients
    for (var vi = 0u; vi < 4u; vi++) {
        let gi = vidx[vi] * 3u;
        let dvn = safe_clip_v3f(d_vn[vi], MIN_VAL, MAX_VAL);
        atomicAdd(&d_vertex_normals[gi], dvn.x);
        atomicAdd(&d_vertex_normals[gi + 1u], dvn.y);
        atomicAdd(&d_vertex_normals[gi + 2u], dvn.z);
    }
}
