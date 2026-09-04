// Interval shading mesh shader.
//
// Same geometry as `interval_compute.wgsl` + `interval_vertex.wgsl`, in one
// stage: each tet becomes 5 vertices and a 4-triangle fan of non-overlapping
// screen-space triangles carrying front/back NDC depth, so each fragment is
// shaded exactly once. Nothing is written to a vertex buffer and no
// indirect draw is needed; the mesh workgroup emits straight to the
// rasterizer. Keep the two generators in sync — the fragment shader
// (`interval_fragment.wgsl`) is shared, so its inputs must match exactly.
//
// TETS_PER_GROUP threads per workgroup, one tet each, fixed 5 verts / 4 prims
// per slot (a degenerate tet emits a collapsed fan), so no prefix sum and no
// divergent early return — returning from a subset of threads before the
// output counts are set hangs mesh shaders on some drivers.

enable wgpu_mesh_shader;

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

// Bindings 0, 3, 4, 5 are shared with `interval_fragment.wgsl` (uniforms,
// aux_data, vertex_normals, tet_indices) and must keep those slots and types;
// the rest are this stage's own.
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> vertex_normals: array<f32>; // [V * 3]
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> densities: array<f32>;
@group(0) @binding(7) var<storage, read> color_grads: array<f32>;
@group(0) @binding(8) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(9) var<storage, read> indirect_args: DrawIndirectArgs;

// Must match `FragmentInput` in interval_fragment.wgsl.
struct IntervalVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) depths: vec2<f32>,                  // (z_front_ndc, z_back_ndc)
    @location(1) color_offsets: vec2<f32>,            // (offset_front, offset_back)
    @location(2) @interpolate(flat) density: f32,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) tet_id: u32,
    @location(5) field_gradient: vec3<f32>,
    @location(6) @interpolate(flat) coverage: f32,
};

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
};

const TETS_PER_GROUP: u32 = 32u;
const VERTS_PER_TET: u32 = 5u;
const PRIMS_PER_TET: u32 = 4u;
const MAX_VERTS: u32 = 160u;  // 32 × 5
const MAX_PRIMS: u32 = 128u;  // 32 × 4

struct MeshOutput {
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
    @builtin(vertices) verts: array<IntervalVertexOutput, 160>,
    @builtin(primitives) prims: array<PrimitiveOutput, 128>,
};

var<workgroup> mesh_out: MeshOutput;

fn load_f32x3(buf_base: u32) -> vec3<f32> {
    return vec3<f32>(vertices[buf_base], vertices[buf_base + 1u], vertices[buf_base + 2u]);
}

fn load_color(idx: u32) -> vec3<f32> {
    return vec3<f32>(colors[idx * 3u], colors[idx * 3u + 1u], colors[idx * 3u + 2u]);
}

fn load_grad(idx: u32) -> vec3<f32> {
    return vec3<f32>(color_grads[idx * 3u], color_grads[idx * 3u + 1u], color_grads[idx * 3u + 2u]);
}

fn load_vnormal(vi: u32) -> vec3<f32> {
    return vec3<f32>(vertex_normals[vi * 3u], vertex_normals[vi * 3u + 1u], vertex_normals[vi * 3u + 2u]);
}

fn interp_gradient_3(n: array<vec3f, 4>, face: vec4u, bary: vec3f) -> vec3f {
    return bary.x * n[face[0]] + bary.y * n[face[1]] + bary.z * n[face[2]];
}

// Maps vertex index -> TET_FACES index where that vertex is face[3] (opposite)
const OPPOSITE_FACE: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 0u);

// Face table: (a, b, c, opposite_vertex)
const TET_FACES: array<vec4<u32>, 4> = array<vec4<u32>, 4>(
    vec4<u32>(0u, 2u, 1u, 3u),
    vec4<u32>(1u, 2u, 3u, 0u),
    vec4<u32>(0u, 3u, 2u, 1u),
    vec4<u32>(3u, 0u, 1u, 2u),
);

// Edge table: 6 edges of a tet (a, b)
const TET_EDGES: array<vec2<u32>, 6> = array<vec2<u32>, 6>(
    vec2<u32>(0u, 1u),
    vec2<u32>(0u, 2u),
    vec2<u32>(0u, 3u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 3u),
    vec2<u32>(2u, 3u),
);

// Which vertex projects inside the opposite face's screen triangle: its index,
// or 4 when none (two edges cross instead).
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

fn emit_vertex(slot: u32, ndc_xy: vec2<f32>, z_front: f32, z_back: f32,
               off_front: f32, off_back: f32, gradient: vec3f,
               density: f32, base_color: vec3f, tet_id: u32, coverage: f32) {
    var out: IntervalVertexOutput;
    out.position = vec4<f32>(ndc_xy, z_front, 1.0);
    out.depths = vec2<f32>(z_front, z_back);
    out.color_offsets = vec2<f32>(off_front, off_back);
    out.field_gradient = gradient;
    out.density = density;
    out.base_color = base_color;
    out.tet_id = tet_id;
    out.coverage = coverage;
    mesh_out.verts[slot] = out;
}

fn emit_degenerate(slot: u32) {
    var out: IntervalVertexOutput;
    out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.depths = vec2<f32>(0.0);
    out.color_offsets = vec2<f32>(0.0);
    out.field_gradient = vec3<f32>(0.0);
    out.density = 0.0;
    out.base_color = vec3<f32>(0.0);
    out.tet_id = 0u;
    out.coverage = 0.0;
    mesh_out.verts[slot] = out;
}

@mesh(mesh_out) @workgroup_size(32)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let visible_count = indirect_args.instance_count;
    let wg_first = wg_id.x * TETS_PER_GROUP;
    let wg_count = min(TETS_PER_GROUP, select(0u, visible_count - wg_first, visible_count > wg_first));
    let tid = wg_first + lid;

    if lid == 0u {
        mesh_out.vertex_count = wg_count * VERTS_PER_TET;
        mesh_out.primitive_count = wg_count * PRIMS_PER_TET;
    }

    let v_base = lid * VERTS_PER_TET;
    let p_base = lid * PRIMS_PER_TET;

    // The fan is the same for every slot, valid or not.
    if lid < wg_count {
        mesh_out.prims[p_base + 0u].indices = vec3<u32>(v_base + 0u, v_base + 1u, v_base + 4u);
        mesh_out.prims[p_base + 1u].indices = vec3<u32>(v_base + 1u, v_base + 2u, v_base + 4u);
        mesh_out.prims[p_base + 2u].indices = vec3<u32>(v_base + 2u, v_base + 3u, v_base + 4u);
        mesh_out.prims[p_base + 3u].indices = vec3<u32>(v_base + 3u, v_base + 0u, v_base + 4u);
    }

    // `valid` is cleared instead of returning early, see the header.
    var valid = lid < wg_count;
    var tet_id = 0u;
    var v_world: array<vec3<f32>, 4>;
    var ndc_xy: array<vec2<f32>, 4>;
    var ndc_z: array<f32, 4>;
    var vn: array<vec3f, 4>;
    var i0 = 0u; var i1 = 0u; var i2 = 0u; var i3 = 0u;

    if valid {
        tet_id = sorted_indices[tid];
        i0 = indices[tet_id * 4u];
        i1 = indices[tet_id * 4u + 1u];
        i2 = indices[tet_id * 4u + 2u];
        i3 = indices[tet_id * 4u + 3u];
        v_world[0] = load_f32x3(i0 * 3u);
        v_world[1] = load_f32x3(i1 * 3u);
        v_world[2] = load_f32x3(i2 * 3u);
        v_world[3] = load_f32x3(i3 * 3u);

        let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
        var clip: array<vec4<f32>, 4>;
        for (var vi = 0u; vi < 4u; vi++) {
            clip[vi] = vp * vec4<f32>(v_world[vi], 1.0);
        }
        // A vertex behind the camera makes NDC meaningless: drop the tet, as
        // the compute generator does.
        if clip[0].w <= 0.0 || clip[1].w <= 0.0 || clip[2].w <= 0.0 || clip[3].w <= 0.0 {
            valid = false;
        } else {
            for (var vi = 0u; vi < 4u; vi++) {
                let w = clip[vi].w;
                let inv_w = select(0.0, 1.0 / w, abs(w) > 1e-10);
                ndc_xy[vi] = clip[vi].xy * inv_w;
                ndc_z[vi] = clip[vi].z * inv_w;
            }
        }
    }

    var tet_density = 0.0;
    var coverage = 0.0;
    var area_px = 0.0;
    var base_color = vec3<f32>(0.0);
    var grad = vec3<f32>(0.0);
    let cam_pos = uniforms.cam_pos_pad.xyz;

    if valid {
        var ndc_min = ndc_xy[0];
        var ndc_max = ndc_xy[0];
        for (var vi = 1u; vi < 4u; vi++) {
            ndc_min = min(ndc_min, ndc_xy[vi]);
            ndc_max = max(ndc_max, ndc_xy[vi]);
        }
        let ext_px = (ndc_max - ndc_min)
            * vec2<f32>(uniforms.screen_width, uniforms.screen_height) * 0.5;
        area_px = ext_px.x * ext_px.y;
        coverage = min(area_px, 1.0);

        tet_density = densities[tet_id];
        grad = load_grad(tet_id);
        base_color = load_color(tet_id) + vec3<f32>(dot(grad, cam_pos - v_world[0]));

        vn[0] = load_vnormal(i0);
        vn[1] = load_vnormal(i1);
        vn[2] = load_vnormal(i2);
        vn[3] = load_vnormal(i3);

        let z_range = max(max(ndc_z[0], ndc_z[1]), max(ndc_z[2], ndc_z[3]))
                    - min(min(ndc_z[0], ndc_z[1]), min(ndc_z[2], ndc_z[3]));
        if z_range < 1e-7 || tet_density < 1e-6 {
            valid = false;
        }
    }

    if !valid {
        if lid < wg_count {
            for (var i = 0u; i < VERTS_PER_TET; i++) { emit_degenerate(v_base + i); }
        }
    } else if area_px < 1.0 {
        // Sub-pixel tet: one pixel-snapped splat, weighted by `coverage` in
        // the fragment shader.
        let scr = vec2<f32>(uniforms.screen_width, uniforms.screen_height);
        let cen = 0.25 * (ndc_xy[0] + ndc_xy[1] + ndc_xy[2] + ndc_xy[3]);
        let px = (cen * 0.5 + 0.5) * scr;
        let px_c = floor(px) + 0.5;
        let cen_snap = (px_c / scr) * 2.0 - 1.0;
        let hx = 1.0 / uniforms.screen_width;
        let hy = 1.0 / uniforms.screen_height;

        let zf = min(min(ndc_z[0], ndc_z[1]), min(ndc_z[2], ndc_z[3]));
        let zb = max(max(ndc_z[0], ndc_z[1]), max(ndc_z[2], ndc_z[3]));

        let cen_world = 0.25 * (v_world[0] + v_world[1] + v_world[2] + v_world[3]);
        let cen_off = dot(grad, cen_world - cam_pos);
        let gn = 0.25 * (vn[0] + vn[1] + vn[2] + vn[3]);

        let corners = array<vec2<f32>, 4>(
            cen_snap + vec2<f32>(-hx, -hy),
            cen_snap + vec2<f32>(-hx,  hy),
            cen_snap + vec2<f32>( hx,  hy),
            cen_snap + vec2<f32>( hx, -hy),
        );
        for (var i = 0u; i < 4u; i++) {
            emit_vertex(v_base + i, corners[i], zf, zb, cen_off, cen_off, gn,
                        tet_density, base_color, tet_id, coverage);
        }
        emit_vertex(v_base + 4u, cen_snap, zf, zb, cen_off, cen_off, gn,
                    tet_density, base_color, tet_id, coverage);
    } else {
        let interior_vtx = classify_silhouette(ndc_xy);

        if interior_vtx < 4u {
            // Case 1: one vertex projects inside the opposite face:
            // 3 silhouette verts + centre; slot 3 repeats slot 0 so the
            // fourth fan triangle collapses.
            let face = TET_FACES[OPPOSITE_FACE[interior_vtx]];
            let si0 = face[0];
            let si1 = face[1];
            let si2 = face[2];
            let center = face[3];

            let bary = bary2d(ndc_xy[center], ndc_xy[si0], ndc_xy[si1], ndc_xy[si2]);

            let z_face = bary.x * ndc_z[si0] + bary.y * ndc_z[si1] + bary.z * ndc_z[si2];
            let z_center = ndc_z[center];
            let z_front = min(z_face, z_center);
            let z_back = max(z_face, z_center);

            var sv_offset: array<f32, 3>;
            let sv_idx = array<u32, 3>(si0, si1, si2);
            for (var i = 0u; i < 3u; i++) {
                sv_offset[i] = dot(grad, v_world[sv_idx[i]] - cam_pos);
            }
            let center_offset = dot(grad, v_world[center] - cam_pos);
            let face_offset = bary.x * sv_offset[0] + bary.y * sv_offset[1] + bary.z * sv_offset[2];

            var offset_front: f32;
            var offset_back: f32;
            var grad_center: vec3f;
            if z_face <= z_center {
                offset_front = face_offset;
                offset_back = center_offset;
                grad_center = interp_gradient_3(vn, face, bary);
            } else {
                offset_front = center_offset;
                offset_back = face_offset;
                grad_center = vn[center];
            }

            for (var i = 0u; i < 3u; i++) {
                emit_vertex(v_base + i, ndc_xy[sv_idx[i]],
                            ndc_z[sv_idx[i]], ndc_z[sv_idx[i]],
                            sv_offset[i], sv_offset[i], vn[sv_idx[i]],
                            tet_density, base_color, tet_id, coverage);
            }
            emit_vertex(v_base + 3u, ndc_xy[sv_idx[0]],
                        ndc_z[sv_idx[0]], ndc_z[sv_idx[0]],
                        sv_offset[0], sv_offset[0], vn[sv_idx[0]],
                        tet_density, base_color, tet_id, coverage);
            emit_vertex(v_base + 4u, ndc_xy[center], z_front, z_back,
                        offset_front, offset_back, grad_center,
                        tet_density, base_color, tet_id, coverage);
        } else {
            // Case 2: two opposite edges cross: 4 silhouette verts + centre.
            let crossing_edges = find_crossing_edges(ndc_xy);
            let ea = TET_EDGES[crossing_edges.x];
            let eb = TET_EDGES[crossing_edges.y];

            let ta = line_intersect_t(ndc_xy[ea.x], ndc_xy[ea.y], ndc_xy[eb.x], ndc_xy[eb.y]);
            let tb = line_intersect_t(ndc_xy[eb.x], ndc_xy[eb.y], ndc_xy[ea.x], ndc_xy[ea.y]);

            let center_xy = mix(ndc_xy[ea.x], ndc_xy[ea.y], ta);

            let z_a = mix(ndc_z[ea.x], ndc_z[ea.y], ta);
            let z_b = mix(ndc_z[eb.x], ndc_z[eb.y], tb);
            let z_front = min(z_a, z_b);
            let z_back = max(z_a, z_b);

            let off_ea_x = dot(grad, v_world[ea.x] - cam_pos);
            let off_ea_y = dot(grad, v_world[ea.y] - cam_pos);
            let off_eb_x = dot(grad, v_world[eb.x] - cam_pos);
            let off_eb_y = dot(grad, v_world[eb.y] - cam_pos);

            let off_a = mix(off_ea_x, off_ea_y, ta);
            let off_b = mix(off_eb_x, off_eb_y, tb);
            let g_a = mix(vn[ea.x], vn[ea.y], ta);
            let g_b = mix(vn[eb.x], vn[eb.y], tb);

            var offset_front: f32;
            var offset_back: f32;
            var grad_center: vec3f;
            if z_a <= z_b {
                offset_front = off_a;
                offset_back = off_b;
                grad_center = g_a;
            } else {
                offset_front = off_b;
                offset_back = off_a;
                grad_center = g_b;
            }

            let sv = array<u32, 4>(ea.x, eb.x, ea.y, eb.y);
            let sv_off = array<f32, 4>(off_ea_x, off_eb_x, off_ea_y, off_eb_y);
            for (var i = 0u; i < 4u; i++) {
                emit_vertex(v_base + i, ndc_xy[sv[i]],
                            ndc_z[sv[i]], ndc_z[sv[i]],
                            sv_off[i], sv_off[i], vn[sv[i]],
                            tet_density, base_color, tet_id, coverage);
            }
            emit_vertex(v_base + 4u, center_xy, z_front, z_back,
                        offset_front, offset_back, grad_center,
                        tet_density, base_color, tet_id, coverage);
        }
    }
}
