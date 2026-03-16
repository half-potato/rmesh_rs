// Fill tile sort buffers with sentinel values (0xFFFFFFFF).
// Each thread fills one key pair (2 u32s) and one value.
// Dispatched for max_pairs_pow2 threads.

@group(0) @binding(0) var<storage, read> tile_uniforms: array<u32>; // unused but kept for bind group compat
@group(0) @binding(1) var<storage, read_write> tile_sort_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_sort_values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx >= arrayLength(&tile_sort_keys) / 2u) {
        return;
    }
    tile_sort_keys[idx * 2u] = 0xFFFFFFFFu;
    tile_sort_keys[idx * 2u + 1u] = 0xFFFFFFFFu;
    tile_sort_values[idx] = 0u;
}
