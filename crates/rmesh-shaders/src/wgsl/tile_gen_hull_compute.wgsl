// Tile-tet pair generation with convex hull overlap test for 4x4 tiles.
//
// For each visible tet, project vertices to screen, compute tile AABB,
// then for each tile in AABB, test convex hull overlap before emitting.
// Emit (tile_id << 15 | depth) key for overlapping tiles.
//
// Key encoding: 17-bit tile_id (supports 131K tiles), 15-bit depth.
// Dispatched for visible_tet_count threads.

struct TileUniforms {
    screen_width: u32,
    screen_height: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_tiles: u32,
    visible_tet_count: u32,
    max_pairs: u32,
    max_pairs_pow2: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

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

@group(0) @binding(0) var<storage, read> tile_uniforms: TileUniforms;
@group(0) @binding(1) var<storage, read> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> vertices: array<f32>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(5) var<storage, read> circumdata: array<f32>;
@group(0) @binding(6) var<storage, read_write> tile_sort_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_sort_values: array<u32>;
@group(0) @binding(8) var<storage, read_write> tile_pair_count: array<atomic<u32>>;

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

fn cross2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    return ax * by - ay * bx;
}

// Test if point (px, py) is inside triangle (ax,ay)-(bx,by)-(cx,cy).
fn point_in_tri(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> bool {
    let d1 = cross2d(bx - ax, by - ay, px - ax, py - ay);
    let d2 = cross2d(cx - bx, cy - by, px - bx, py - by);
    let d3 = cross2d(ax - cx, ay - cy, px - cx, py - cy);
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    return !(has_neg && has_pos);
}

// Test if point is inside the 2D convex hull of 4 projected tet vertices.
// Tests all C(4,3)=4 triangles.
fn point_in_hull(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) -> bool {
    return point_in_tri(px, py, x0, y0, x1, y1, x2, y2)
        || point_in_tri(px, py, x0, y0, x1, y1, x3, y3)
        || point_in_tri(px, py, x0, y0, x2, y2, x3, y3)
        || point_in_tri(px, py, x1, y1, x2, y2, x3, y3);
}

// Test if a projected vertex falls inside a tile rectangle.
fn point_in_rect(px: f32, py: f32, rect_left: f32, rect_top: f32, rect_right: f32, rect_bottom: f32) -> bool {
    return px >= rect_left && px <= rect_right && py >= rect_top && py <= rect_bottom;
}

// Test if a line segment between two projected vertices intersects a tile rectangle.
// Uses separating axis theorem on the 4 rectangle edges.
fn segment_intersects_rect(
    x0: f32, y0: f32, x1: f32, y1: f32,
    rect_left: f32, rect_top: f32, rect_right: f32, rect_bottom: f32
) -> bool {
    // Clip the segment against the rect using Cohen-Sutherland-style bounds
    let seg_min_x = min(x0, x1);
    let seg_max_x = max(x0, x1);
    let seg_min_y = min(y0, y1);
    let seg_max_y = max(y0, y1);

    // Quick rejection: segment's bounding box doesn't overlap tile
    if (seg_max_x < rect_left || seg_min_x > rect_right ||
        seg_max_y < rect_top || seg_min_y > rect_bottom) {
        return false;
    }

    // Check if the segment's line crosses through the rectangle
    // using the edge normal test
    let dx = x1 - x0;
    let dy = y1 - y0;

    // Test all 4 corners of the rect against the line
    let c0 = cross2d(dx, dy, rect_left - x0, rect_top - y0);
    let c1 = cross2d(dx, dy, rect_right - x0, rect_top - y0);
    let c2 = cross2d(dx, dy, rect_right - x0, rect_bottom - y0);
    let c3 = cross2d(dx, dy, rect_left - x0, rect_bottom - y0);

    let all_pos = (c0 > 0.0) && (c1 > 0.0) && (c2 > 0.0) && (c3 > 0.0);
    let all_neg = (c0 < 0.0) && (c1 < 0.0) && (c2 < 0.0) && (c3 < 0.0);

    // If all corners are on the same side of the line, no crossing
    return !(all_pos || all_neg);
}

// Test if the convex hull of 4 projected vertices overlaps a tile.
// Checks: (1) tile center inside hull, (2) any hull vertex inside tile,
// (3) any hull edge crosses tile rect (catches thin tets crossing tile).
fn hull_overlaps_tile(
    proj: array<vec2<f32>, 4>,
    rect_left: f32, rect_top: f32, rect_right: f32, rect_bottom: f32
) -> bool {
    // Test 1: tile center inside hull
    let cx = (rect_left + rect_right) * 0.5;
    let cy = (rect_top + rect_bottom) * 0.5;
    if (point_in_hull(cx, cy,
            proj[0].x, proj[0].y, proj[1].x, proj[1].y,
            proj[2].x, proj[2].y, proj[3].x, proj[3].y)) {
        return true;
    }

    // Test 2: any hull vertex inside tile rect
    for (var i = 0u; i < 4u; i++) {
        if (point_in_rect(proj[i].x, proj[i].y, rect_left, rect_top, rect_right, rect_bottom)) {
            return true;
        }
    }

    // Test 3: any hull edge crosses tile rect
    // 6 edges of the tetrahedron's 2D hull (all pairs of 4 vertices)
    let edges = array<vec2<u32>, 6>(
        vec2<u32>(0u, 1u), vec2<u32>(0u, 2u), vec2<u32>(0u, 3u),
        vec2<u32>(1u, 2u), vec2<u32>(1u, 3u), vec2<u32>(2u, 3u),
    );
    for (var e = 0u; e < 6u; e++) {
        let a = edges[e].x;
        let b = edges[e].y;
        if (segment_intersects_rect(
                proj[a].x, proj[a].y, proj[b].x, proj[b].y,
                rect_left, rect_top, rect_right, rect_bottom)) {
            return true;
        }
    }

    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let vis_idx = gid.x + gid.y * nwg.x * 64u;
    if (vis_idx >= tile_uniforms.visible_tet_count) {
        return;
    }

    let tet_id = sorted_indices[vis_idx];

    // Load tet vertices
    let i0 = indices[tet_id * 4u];
    let i1 = indices[tet_id * 4u + 1u];
    let i2 = indices[tet_id * 4u + 2u];
    let i3 = indices[tet_id * 4u + 3u];

    let v0 = load_vertex(i0);
    let v1 = load_vertex(i1);
    let v2 = load_vertex(i2);
    let v3 = load_vertex(i3);

    // Project to clip space
    let vp = mat4x4<f32>(uniforms.vp_col0, uniforms.vp_col1, uniforms.vp_col2, uniforms.vp_col3);
    let c0 = vp * vec4<f32>(v0, 1.0);
    let c1 = vp * vec4<f32>(v1, 1.0);
    let c2 = vp * vec4<f32>(v2, 1.0);
    let c3 = vp * vec4<f32>(v3, 1.0);

    let any_behind = (c0.w <= 0.0) || (c1.w <= 0.0) || (c2.w <= 0.0) || (c3.w <= 0.0);

    let W = f32(tile_uniforms.screen_width);
    let H = f32(tile_uniforms.screen_height);
    let ts = f32(tile_uniforms.tile_size);

    var tile_min_x: u32;
    var tile_min_y: u32;
    var tile_max_x: u32;
    var tile_max_y: u32;

    // Projected 2D screen coordinates (only valid when !any_behind)
    var proj: array<vec2<f32>, 4>;

    if (any_behind) {
        // Conservative: cover all tiles
        tile_min_x = 0u;
        tile_min_y = 0u;
        tile_max_x = tile_uniforms.tiles_x - 1u;
        tile_max_y = tile_uniforms.tiles_y - 1u;
    } else {
        // NDC to pixel coords
        let n0 = c0.xyz / c0.w;
        let n1 = c1.xyz / c1.w;
        let n2 = c2.xyz / c2.w;
        let n3 = c3.xyz / c3.w;

        proj[0] = vec2<f32>((n0.x + 1.0) * 0.5 * W, (1.0 - n0.y) * 0.5 * H);
        proj[1] = vec2<f32>((n1.x + 1.0) * 0.5 * W, (1.0 - n1.y) * 0.5 * H);
        proj[2] = vec2<f32>((n2.x + 1.0) * 0.5 * W, (1.0 - n2.y) * 0.5 * H);
        proj[3] = vec2<f32>((n3.x + 1.0) * 0.5 * W, (1.0 - n3.y) * 0.5 * H);

        let pix_min_x = min(min(proj[0].x, proj[1].x), min(proj[2].x, proj[3].x));
        let pix_max_x = max(max(proj[0].x, proj[1].x), max(proj[2].x, proj[3].x));
        let pix_min_y = min(min(proj[0].y, proj[1].y), min(proj[2].y, proj[3].y));
        let pix_max_y = max(max(proj[0].y, proj[1].y), max(proj[2].y, proj[3].y));

        // Clamp to screen bounds
        let clamped_left = max(pix_min_x, 0.0);
        let clamped_right = min(pix_max_x, W - 1.0);
        let clamped_top = max(pix_min_y, 0.0);
        let clamped_bottom = min(pix_max_y, H - 1.0);

        if (clamped_left > clamped_right || clamped_top > clamped_bottom) {
            return; // Off-screen
        }

        tile_min_x = u32(clamped_left / ts);
        tile_max_x = min(u32(clamped_right / ts), tile_uniforms.tiles_x - 1u);
        tile_min_y = u32(clamped_top / ts);
        tile_max_y = min(u32(clamped_bottom / ts), tile_uniforms.tiles_y - 1u);
    }

    // Depth key from circumsphere
    let cx = circumdata[tet_id * 4u];
    let cy = circumdata[tet_id * 4u + 1u];
    let cz = circumdata[tet_id * 4u + 2u];
    let r2 = circumdata[tet_id * 4u + 3u];
    let cam = uniforms.cam_pos_pad.xyz;
    let diff = vec3<f32>(cx, cy, cz) - cam;
    let depth = dot(diff, diff) - r2;
    let inv_depth = ~bitcast<u32>(depth);
    let depth_bits = inv_depth >> 17u; // 15 bits of depth

    // Emit tile-tet pairs with hull overlap test
    for (var ty = tile_min_y; ty <= tile_max_y; ty++) {
        for (var tx = tile_min_x; tx <= tile_max_x; tx++) {
            // If any vertex is behind camera, skip hull test (conservative emit)
            if (!any_behind) {
                let rect_left = f32(tx) * ts;
                let rect_top = f32(ty) * ts;
                let rect_right = rect_left + ts;
                let rect_bottom = rect_top + ts;

                if (!hull_overlaps_tile(proj, rect_left, rect_top, rect_right, rect_bottom)) {
                    continue;
                }
            }

            let tile_id = ty * tile_uniforms.tiles_x + tx;
            let key = (tile_id << 15u) | (depth_bits & 0x7FFFu);

            let pair_idx = atomicAdd(&tile_pair_count[0], 1u);
            if (pair_idx < tile_uniforms.max_pairs) {
                tile_sort_keys[pair_idx] = key;
                tile_sort_values[pair_idx] = tet_id;
            }
        }
    }
}
