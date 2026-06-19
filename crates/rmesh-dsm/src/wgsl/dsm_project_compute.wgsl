// DSM project compute: power-of-point sort key generation.
//
// Used to build the back-to-front order for a point/spot light's DSM. Because
// the Radiance Meshes power sort is a function of the camera *origin* only,
// one dispatch per light is enough for all six cubemap faces (see
// RADIANCE_MESHES.md). All tets are emitted; per-face visibility is handled
// downstream by interval_generate + hardware rasterizer clipping.
//
// Differences from project_compute_hw.wgsl:
//   * No view-frustum culling (we'd need the union of 6 face frusta).
//   * No SH evaluation (DSM stores depth moments; colors unused).
//   * instance_count := tet_count (set once by thread 0 via atomicStore).

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
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> circumdata: array<f32>;
@group(0) @binding(4) var<storage, read_write> sort_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_values: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect_args: DrawIndirectArgs;

fn load_vertex(idx: u32) -> vec3<f32> {
    let i = idx * 3u;
    return vec3<f32>(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let tet_id = global_id.x + global_id.y * num_workgroups.x * 64u;

    // Thread 0 publishes the total visible count for indirect_convert.
    if (tet_id == 0u) {
        atomicStore(&indirect_args.instance_count, uniforms.tet_count);
    }

    // Padding threads (beyond tet_count): radix sort needs sentinel keys.
    if (tet_id >= uniforms.tet_count) {
        if (tet_id < arrayLength(&sort_keys)) {
            sort_keys[tet_id] = 0xFFFFFFFFu;
            sort_values[tet_id] = tet_id;
        }
        return;
    }

    // Power of the camera origin w.r.t. tet's circumsphere, in the
    // numerically-stable form pow(P) = (P - v0) . (P + v0 - 2C).
    let i0 = indices[tet_id * 4u];
    let v0 = load_vertex(i0);
    let center = vec3<f32>(
        circumdata[tet_id * 4u],
        circumdata[tet_id * 4u + 1u],
        circumdata[tet_id * 4u + 2u],
    );
    let cam = uniforms.cam_pos_pad.xyz;
    let depth_raw = dot(cam - v0, cam + v0 - 2.0 * center);
    let depth = clamp(depth_raw, -1e20, 1e20);
    sort_keys[tet_id] = ~bitcast<u32>(depth); // invert for back-to-front
    sort_values[tet_id] = tet_id;
}
