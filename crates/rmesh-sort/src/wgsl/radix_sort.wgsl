// Bitonic sort step compute shader.
//
// Performs one step of the bitonic merge network.
// Must be dispatched log2(N) * (log2(N)+1) / 2 times with
// different (stage, step) parameters.

struct SortUniforms {
    count: u32,
    stage: u32,
    step_size: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> uniforms: SortUniforms;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.count) {
        return;
    }

    let step = uniforms.step_size;
    let stage = uniforms.stage;

    let pair_distance = 1u << step;
    let block_size = 1u << (stage + 1u);

    let pos = idx;
    let partner = pos ^ pair_distance;

    if (partner > pos && partner < uniforms.count) {
        // Sort direction: ascending in first half of block, descending in second
        let ascending = ((pos / block_size) % 2u) == 0u;

        let key_a = keys[pos];
        let key_b = keys[partner];

        var should_swap = false;
        if (ascending) {
            should_swap = key_a > key_b;
        } else {
            should_swap = key_a < key_b;
        }

        if (should_swap) {
            keys[pos] = key_b;
            keys[partner] = key_a;

            let val_a = values[pos];
            let val_b = values[partner];
            values[pos] = val_b;
            values[partner] = val_a;
        }
    }
}
