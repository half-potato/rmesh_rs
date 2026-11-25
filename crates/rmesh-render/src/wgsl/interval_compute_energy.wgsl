// Compute-based interval shading for energy materials: generates screen-space
// triangles per tet with perspective-weighted cell-corner radiance (C / z_view)
// so the fragment shader integrates color with zero storage reads.

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

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

// Binding 3 carries packed corner_radiance (2 u32 = f16x3 per corner, 4 corners
// per tet) instead of legacy flat colors, staying within WebGPU's 8-storage-buffer cap.
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> corner_radiance: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read> indirect_args: DrawIndirectArgs;
@group(0) @binding(8) var<storage, read_write> out_vertices: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> out_tet_data: array<vec4<f32>>;

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_corner(idx: u32) -> vec3<f32> {
    let packed = corner_radiance[idx];
    return vec3<f32>(unpack2x16float(packed.x), unpack2x16float(packed.y).x);
}

const OPPOSITE_FACE: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 0u);

const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

const TET_EDGES: array<vec2<u32>, 6> = array<vec2<u32>, 6>(
    vec2<u32>(0u, 1u),
    vec2<u32>(0u, 2u),
    vec2<u32>(0u, 3u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 3u),
    vec2<u32>(2u, 3u),
);

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

        if all_pos || all_neg {
            return face[3];
        }
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
    if abs(denom) < 1e-10 {
        return -1.0;
    }
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

// Compact layout (half the memory traffic of the 4-vec4 legacy layout):
//   out_vertices[slot*2+0] = (ndc_xy, z_front_ndc, z_back_ndc)
//   out_vertices[slot*2+1] = packed 6x f16: (c_front_s.rgb, c_back_s.rgb), pad
//   out_tet_data[tid]      = (inv_w_min, density, coverage, 0)
// where c_s = corner_radiance * (w_min / clip.w), keeping magnitudes in
// [0, max_radiance] so f16 never overflows or underflows at large depths.
fn write_vertex(slot: u32, ndc_xy: vec2<f32>, z_front: f32, z_back: f32,
                c_front_s: vec3<f32>, c_back_s: vec3<f32>) {
    out_vertices[slot * 2u + 0u] = vec4<f32>(ndc_xy, z_front, z_back);
    out_vertices[slot * 2u + 1u] = bitcast<vec4<f32>>(vec4<u32>(
        pack2x16float(c_front_s.xy),
        pack2x16float(vec2<f32>(c_front_s.z, c_back_s.x)),
        pack2x16float(c_back_s.yz),
        0u,
    ));
}

fn write_degenerate_vertex(slot: u32) {
    out_vertices[slot * 2u + 0u] = vec4<f32>(0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.y * 65535u * 64u + global_id.x;
    let visible_count = indirect_args.instance_count;
    if tid >= visible_count {
        return;
    }

    let v_base = tid * 5u;
    let tet_id = sorted_indices[tid];

    let tet_density = densities[tet_id];
    if tet_density < 1e-6 {
        for (var i = 0u; i < 5u; i++) { write_degenerate_vertex(v_base + i); }
        out_tet_data[tid] = vec4<f32>(0.0);
        return;
    }

    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);

    var clip: array<vec4<f32>, 4>;
    clip[0] = vp * vec4<f32>(load_f32x3(i0 * 3u), 1.0);
    clip[1] = vp * vec4<f32>(load_f32x3(i1 * 3u), 1.0);
    clip[2] = vp * vec4<f32>(load_f32x3(i2 * 3u), 1.0);
    clip[3] = vp * vec4<f32>(load_f32x3(i3 * 3u), 1.0);

    if clip[0].w <= 0.0 || clip[1].w <= 0.0 || clip[2].w <= 0.0 || clip[3].w <= 0.0 {
        for (var i = 0u; i < 5u; i++) { write_degenerate_vertex(v_base + i); }
        out_tet_data[tid] = vec4<f32>(0.0);
        return;
    }

    var ndc_xy: array<vec2<f32>, 4>;
    var ndc_z: array<f32, 4>;
    var inv_w: array<f32, 4>;
    for (var vi = 0u; vi < 4u; vi++) {
        let w = clip[vi].w;
        let iw = select(0.0, 1.0 / w, abs(w) > 1e-10);
        inv_w[vi] = iw;
        ndc_xy[vi] = clip[vi].xy * iw;
        ndc_z[vi] = clip[vi].z * iw;
    }

    let z_range = max(max(ndc_z[0], ndc_z[1]), max(ndc_z[2], ndc_z[3]))
                - min(min(ndc_z[0], ndc_z[1]), min(ndc_z[2], ndc_z[3]));
    if z_range < 1e-7 {
        for (var i = 0u; i < 5u; i++) { write_degenerate_vertex(v_base + i); }
        out_tet_data[tid] = vec4<f32>(0.0);
        return;
    }

    let ndc_min = min(min(ndc_xy[0], ndc_xy[1]), min(ndc_xy[2], ndc_xy[3]));
    let ndc_max = max(max(ndc_xy[0], ndc_xy[1]), max(ndc_xy[2], ndc_xy[3]));
    let ext_px = (ndc_max - ndc_min)
        * vec2<f32>(uniforms.screen_width, uniforms.screen_height) * 0.5;
    let coverage = min(ext_px.x * ext_px.y, 1.0);

    let w_min = min(min(clip[0].w, clip[1].w), min(clip[2].w, clip[3].w));
    let inv_w_min = select(0.0, 1.0 / w_min, w_min > 1e-10);
    out_tet_data[tid] = vec4<f32>(inv_w_min, tet_density, coverage, 0.0);

    let cbase = tet_id * 4u;
    var c_oz: array<vec3<f32>, 4>;
    c_oz[0] = load_corner(cbase) * (w_min * inv_w[0]);
    c_oz[1] = load_corner(cbase + 1u) * (w_min * inv_w[1]);
    c_oz[2] = load_corner(cbase + 2u) * (w_min * inv_w[2]);
    c_oz[3] = load_corner(cbase + 3u) * (w_min * inv_w[3]);

    let interior_vtx = classify_silhouette(ndc_xy);

    if interior_vtx < 4u {
        let face = TET_FACES[OPPOSITE_FACE[interior_vtx]];
        let si0 = face[0];
        let si1 = face[1];
        let si2 = face[2];
        let center = face[3];

        var bary = bary2d(ndc_xy[center], ndc_xy[si0], ndc_xy[si1], ndc_xy[si2]);
        bary = max(bary, vec3<f32>(0.0));
        bary /= max(bary.x + bary.y + bary.z, 1e-30);

        let z_face = bary.x * ndc_z[si0] + bary.y * ndc_z[si1] + bary.z * ndc_z[si2];
        let z_center = ndc_z[center];
        let z_front = min(z_face, z_center);
        let z_back = max(z_face, z_center);

        let c_face_oz = bary.x * c_oz[si0] + bary.y * c_oz[si1] + bary.z * c_oz[si2];
        let c_center_oz = c_oz[center];
        var c_front_oz: vec3<f32>;
        var c_back_oz: vec3<f32>;
        if z_face <= z_center {
            c_front_oz = c_face_oz;
            c_back_oz = c_center_oz;
        } else {
            c_front_oz = c_center_oz;
            c_back_oz = c_face_oz;
        }

        let sv_idx = array<u32, 3>(si0, si1, si2);
        for (var i = 0u; i < 3u; i++) {
            let vi = sv_idx[i];
            write_vertex(v_base + i, ndc_xy[vi], ndc_z[vi], ndc_z[vi], c_oz[vi], c_oz[vi]);
        }
        write_vertex(v_base + 3u, ndc_xy[si0], ndc_z[si0], ndc_z[si0], c_oz[si0], c_oz[si0]);
        write_vertex(v_base + 4u, ndc_xy[center], z_front, z_back, c_front_oz, c_back_oz);
    } else {
        let crossing_edges = find_crossing_edges(ndc_xy);
        let ea = TET_EDGES[crossing_edges.x];
        let eb = TET_EDGES[crossing_edges.y];

        let ta = clamp(line_intersect_t(ndc_xy[ea.x], ndc_xy[ea.y], ndc_xy[eb.x], ndc_xy[eb.y]), 0.0, 1.0);
        let tb = clamp(line_intersect_t(ndc_xy[eb.x], ndc_xy[eb.y], ndc_xy[ea.x], ndc_xy[ea.y]), 0.0, 1.0);

        let center_xy = mix(ndc_xy[ea.x], ndc_xy[ea.y], ta);
        let z_a = mix(ndc_z[ea.x], ndc_z[ea.y], ta);
        let z_b = mix(ndc_z[eb.x], ndc_z[eb.y], tb);
        let z_front = min(z_a, z_b);
        let z_back = max(z_a, z_b);

        let c_a_oz = mix(c_oz[ea.x], c_oz[ea.y], ta);
        let c_b_oz = mix(c_oz[eb.x], c_oz[eb.y], tb);
        var c_front_oz: vec3<f32>;
        var c_back_oz: vec3<f32>;
        if z_a <= z_b {
            c_front_oz = c_a_oz;
            c_back_oz = c_b_oz;
        } else {
            c_front_oz = c_b_oz;
            c_back_oz = c_a_oz;
        }

        let sv = array<u32, 4>(ea.x, eb.x, ea.y, eb.y);
        for (var i = 0u; i < 4u; i++) {
            let vi = sv[i];
            write_vertex(v_base + i, ndc_xy[vi], ndc_z[vi], ndc_z[vi], c_oz[vi], c_oz[vi]);
        }
        write_vertex(v_base + 4u, center_xy, z_front, z_back, c_front_oz, c_back_oz);
    }
}
