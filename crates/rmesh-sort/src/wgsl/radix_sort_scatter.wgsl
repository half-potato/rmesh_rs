// Radix sort - scatter pass: local radix sort + scatter to global output.
// Adapted from brush-sort (Apache-2.0 / MIT, Google LLC)

const WG: u32 = 256;
const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 16;
const ELEMENTS_PER_THREAD: u32 = 4;
const BLOCK_SIZE: u32 = 1024;
const KEY_STRIDE: u32 = /*KEY_STRIDE*/1u;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

struct Uniforms {
    shift: u32,
}

@group(0) @binding(0) var<storage, read> config: Uniforms;
@group(0) @binding(1) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<u32>;
@group(0) @binding(4) var<storage, read> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;

var<workgroup> lds_sums: array<u32, WG>;
var<workgroup> lds_scratch: array<u32, WG>;
var<workgroup> bin_offset_cache: array<u32, WG>;
var<workgroup> local_histogram: array<atomic<u32>, BIN_COUNT>;

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
        bin_offset_cache[local_id.x] = counts[local_id.x * num_wgs + group_id];
    }
    workgroupBarrier();
    let wg_block_start = BLOCK_SIZE * group_id;
    let block_index = wg_block_start + local_id.x;
    var data_index = block_index;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        if (local_id.x < BIN_COUNT) {
            local_histogram[local_id.x] = 0u;
        }
        var local_key_lo = ~0u;
        var local_key_hi = ~0u;
        var local_value = 0u;

        if (data_index < num_keys) {
            local_key_lo = src[data_index * KEY_STRIDE];
            if (KEY_STRIDE > 1u) {
                local_key_hi = src[data_index * KEY_STRIDE + 1u];
            }
            local_value = values[data_index];
        }

        let word = select(0u, 1u, KEY_STRIDE > 1u && config.shift >= 32u);
        let eff_shift = config.shift - word * 32u;

        for (var bit_shift = 0u; bit_shift < BITS_PER_PASS; bit_shift += 2u) {
            let key_word = select(local_key_lo, local_key_hi, word == 1u);
            let key_index = (key_word >> eff_shift) & 0xfu;
            let bit_key = (key_index >> bit_shift) & 3u;
            var packed_histogram = 1u << (bit_key * 8u);
            // workgroup prefix sum
            var sum = packed_histogram;
            lds_scratch[local_id.x] = sum;
            for (var j = 0u; j < 8u; j++) {
                workgroupBarrier();
                if (local_id.x >= (1u << j)) {
                    sum += lds_scratch[local_id.x - (1u << j)];
                }
                workgroupBarrier();
                lds_scratch[local_id.x] = sum;
            }
            workgroupBarrier();
            packed_histogram = lds_scratch[WG - 1u];
            packed_histogram = (packed_histogram << 8u) + (packed_histogram << 16u) + (packed_histogram << 24u);
            var local_sum = packed_histogram;
            if (local_id.x > 0u) {
                local_sum += lds_scratch[local_id.x - 1u];
            }
            let key_offset = (local_sum >> (bit_key * 8u)) & 0xffu;

            lds_sums[key_offset] = local_key_lo;
            workgroupBarrier();
            local_key_lo = lds_sums[local_id.x];
            workgroupBarrier();

            if (KEY_STRIDE > 1u) {
                lds_sums[key_offset] = local_key_hi;
                workgroupBarrier();
                local_key_hi = lds_sums[local_id.x];
                workgroupBarrier();
            }

            lds_sums[key_offset] = local_value;
            workgroupBarrier();
            local_value = lds_sums[local_id.x];
            workgroupBarrier();
        }
        let key_word2 = select(local_key_lo, local_key_hi, word == 1u);
        let key_index = (key_word2 >> eff_shift) & 0xfu;
        atomicAdd(&local_histogram[key_index], 1u);
        workgroupBarrier();
        var histogram_local_sum = 0u;
        if (local_id.x < BIN_COUNT) {
            histogram_local_sum = local_histogram[local_id.x];
        }
        // workgroup prefix sum of histogram
        var histogram_prefix_sum = histogram_local_sum;
        if (local_id.x < BIN_COUNT) {
            lds_scratch[local_id.x] = histogram_prefix_sum;
        }
        for (var j = 0u; j < 4u; j++) {
            workgroupBarrier();
            if (local_id.x >= (1u << j) && local_id.x < BIN_COUNT) {
                histogram_prefix_sum += lds_scratch[local_id.x - (1u << j)];
            }
            workgroupBarrier();
            if (local_id.x < BIN_COUNT) {
                lds_scratch[local_id.x] = histogram_prefix_sum;
            }
        }
        let global_offset = bin_offset_cache[key_index];
        workgroupBarrier();
        var local_offset = local_id.x;
        if (key_index > 0u) {
            local_offset -= lds_scratch[key_index - 1u];
        }
        let total_offset = global_offset + local_offset;
        if (total_offset < num_keys) {
            out[total_offset * KEY_STRIDE] = local_key_lo;
            if (KEY_STRIDE > 1u) {
                out[total_offset * KEY_STRIDE + 1u] = local_key_hi;
            }
            out_values[total_offset] = local_value;
        }
        if (local_id.x < BIN_COUNT) {
            bin_offset_cache[local_id.x] += local_histogram[local_id.x];
        }
        workgroupBarrier();
        data_index += WG;
    }
}
