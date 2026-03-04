// Radix sort - count pass: per-workgroup histogram of 4-bit digit.
// Adapted from brush-sort (Apache-2.0 / MIT, Google LLC)

const WG: u32 = 256;
const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 16; // 1 << BITS_PER_PASS
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = 1024; // WG * ELEMENTS_PER_THREAD

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

struct Uniforms {
    shift: u32,
}

@group(0) @binding(0) var<storage, read> config: Uniforms;
@group(0) @binding(1) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

var<workgroup> histogram: array<atomic<u32>, BIN_COUNT>;

@compute
@workgroup_size(WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let group_id = wid.x + wid.y * num_workgroups.x;

    if (group_id >= num_wgs) {
        return;
    }

    if (local_id.x < BIN_COUNT) {
        histogram[local_id.x] = 0u;
    }
    workgroupBarrier();

    let wg_block_start = BLOCK_SIZE * group_id;
    let shift_bit = config.shift;
    var data_index = wg_block_start + local_id.x;

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        if (data_index < num_keys) {
            let local_key = (src[data_index] >> shift_bit) & 0xfu;
            atomicAdd(&histogram[local_key], 1u);
        }
        data_index += WG;
    }
    workgroupBarrier();
    if (local_id.x < BIN_COUNT) {
        let num_wgs2 = div_ceil(num_keys, BLOCK_SIZE);
        counts[local_id.x * num_wgs2 + group_id] = histogram[local_id.x];
    }
}
