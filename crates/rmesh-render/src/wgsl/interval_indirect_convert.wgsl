// Converts DrawIndirectArgs (instance_count = visible tet count) to
// a combined buffer with compute dispatch args + draw-indexed-indirect args.
//
// Output layout (8 x u32 = 32 bytes):
//   [0..2]: dispatch_workgroups(X, Y, Z) for interval_compute
//   [3..7]: draw_indexed_indirect(index_count, instance_count, first_index, base_vertex, first_instance)

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read> indirect_args: DrawIndirectArgs;
@group(0) @binding(1) var<storage, read_write> combined_args: array<u32, 8>;

override WG_SIZE: u32 = 64u;

// maxComputeWorkgroupsPerDimension. A dispatch exceeding this is invalid; an
// *indirect* dispatch that exceeds it is silently dropped rather than raising a
// validation error, so the gen pass would do nothing and the draw would
// rasterize the previous frame's stale interval vertices (frozen image) or the
// zero-initialized buffer (black).
const MAX_WG_PER_DIM: u32 = 65535u;

@compute @workgroup_size(1)
fn main() {
    let count = indirect_args.instance_count;
    // Compute dispatch args. Past 65535 workgroups (= 4,194,240 tets at
    // WG_SIZE 64) spill into Y — interval_compute reconstructs the flat id as
    // `global_id.y * 65535 * 64 + global_id.x`, which requires X to be pinned
    // to exactly 65535 whenever Y > 1.
    let wg = (count + WG_SIZE - 1u) / WG_SIZE;
    combined_args[0] = min(wg, MAX_WG_PER_DIM);
    combined_args[1] = (wg + MAX_WG_PER_DIM - 1u) / MAX_WG_PER_DIM;
    combined_args[2] = 1u;
    // Draw-indexed-indirect args (static fan index buffer, 12 indices per tet)
    combined_args[3] = count * 12u;  // index_count
    combined_args[4] = 1u;           // instance_count
    combined_args[5] = 0u;           // first_index
    combined_args[6] = 0u;           // base_vertex
    combined_args[7] = 0u;           // first_instance
}
