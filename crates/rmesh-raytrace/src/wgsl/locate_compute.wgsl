// Point location via adjacency walking.
//
// For each query point, walks through the tet mesh starting from a hint tet
// until finding the containing tet or determining the point is outside.

struct LocateUniforms {
    num_queries: u32,
    hint_tet: i32,    // global hint, or -1 to use per-query hint_tets
    tet_count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> locate_uniforms: LocateUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> tet_neighbors: array<i32>;
@group(0) @binding(4) var<storage, read> query_points: array<f32>;
@group(0) @binding(5) var<storage, read> hint_tets: array<i32>;
@group(0) @binding(6) var<storage, read_write> result_tets: array<i32>;

// Vertex k -> face index that is opposite vertex k.
// Face 0 opposite = vertex 3, Face 1 opposite = vertex 0,
// Face 2 opposite = vertex 1, Face 3 opposite = vertex 2.
const VERTEX_TO_FACE: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 0u);

const MAX_WALK_ITERS: u32 = 512u;
const BARY_EPS: f32 = -1e-6;

fn load_vertex(idx: u32) -> vec3<f32> {
    return vec3<f32>(vertices[idx * 3u], vertices[idx * 3u + 1u], vertices[idx * 3u + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    if (query_idx >= locate_uniforms.num_queries) {
        return;
    }

    // Load query point
    let p = vec3<f32>(
        query_points[query_idx * 3u],
        query_points[query_idx * 3u + 1u],
        query_points[query_idx * 3u + 2u],
    );

    // Determine start tet
    var current_tet: i32;
    if (locate_uniforms.hint_tet >= 0) {
        current_tet = locate_uniforms.hint_tet;
    } else {
        current_tet = hint_tets[query_idx];
    }

    // Clamp to valid range
    if (current_tet < 0 || u32(current_tet) >= locate_uniforms.tet_count) {
        current_tet = 0;
    }

    // Walk loop
    for (var iter = 0u; iter < MAX_WALK_ITERS; iter++) {
        let tet_id = u32(current_tet);

        // Load 4 vertices
        let vi0 = indices[tet_id * 4u];
        let vi1 = indices[tet_id * 4u + 1u];
        let vi2 = indices[tet_id * 4u + 2u];
        let vi3 = indices[tet_id * 4u + 3u];

        let v0 = load_vertex(vi0);
        let v1 = load_vertex(vi1);
        let v2 = load_vertex(vi2);
        let v3 = load_vertex(vi3);

        // Barycentric coordinates via Cramer's rule
        let d = v1 - v0;
        let e = v2 - v0;
        let f = v3 - v0;
        let q = p - v0;

        let det = dot(d, cross(e, f));
        if (abs(det) < 1e-20) {
            // Degenerate tet
            result_tets[query_idx] = -1;
            return;
        }
        let inv_det = 1.0 / det;

        let u = dot(q, cross(e, f)) * inv_det;  // bary for v1
        let v = dot(d, cross(q, f)) * inv_det;  // bary for v2
        let w = dot(d, cross(e, q)) * inv_det;  // bary for v3
        let s = 1.0 - u - v - w;                // bary for v0

        // Check containment
        if (s >= BARY_EPS && u >= BARY_EPS && v >= BARY_EPS && w >= BARY_EPS) {
            result_tets[query_idx] = current_tet;
            return;
        }

        // Find most negative barycentric -> walk through opposite face
        var min_bary = s;
        var min_vertex = 0u;
        if (u < min_bary) { min_bary = u; min_vertex = 1u; }
        if (v < min_bary) { min_bary = v; min_vertex = 2u; }
        if (w < min_bary) { min_bary = w; min_vertex = 3u; }

        let face_idx = VERTEX_TO_FACE[min_vertex];
        let neighbor = tet_neighbors[tet_id * 4u + face_idx];

        if (neighbor < 0) {
            // Walked outside the mesh
            result_tets[query_idx] = -1;
            return;
        }

        current_tet = neighbor;
    }

    // Exhausted iterations
    result_tets[query_idx] = -1;
}
