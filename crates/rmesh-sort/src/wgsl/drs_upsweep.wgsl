// DeviceRadixSort — Upsweep kernel (WGE16 path only)
// Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)

const US_DIM: u32 = 128u;
const RADIX: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const RADIX_LOG: u32 = 8u;
const PART_SIZE: u32 = 3840u;
const KEY_STRIDE: u32 = /*KEY_STRIDE*/1u;

struct Config {
    numKeys: u32,
    radixShift: u32,
    threadBlocks: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read> b_sort: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_globalHist: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> b_passHist: array<u32>;

var<workgroup> g_us: array<atomic<u32>, 512>;
var<private> subgroup_size_p: u32;
var<private> subgroup_invocation_id_p: u32;

fn extractDigitFromIndex(idx: u32) -> u32 {
    let shift = config.radixShift;
    if KEY_STRIDE > 1u && shift >= 32u {
        return (b_sort[idx * KEY_STRIDE + 1u] >> (shift - 32u)) & RADIX_MASK;
    }
    return (b_sort[idx * KEY_STRIDE] >> shift) & RADIX_MASK;
}

fn globalHistOffset() -> u32 {
    return (config.radixShift / RADIX_LOG) * RADIX;
}

@compute
@workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    subgroup_size_p = sg_size;
    subgroup_invocation_id_p = sg_id;
    let gtid = local_id.x;
    let gid = wid.x + wid.y * nwg.x;

    // Dynamic bounds: compute effective threadBlocks from numKeys
    let effectiveTB = (config.numKeys + PART_SIZE - 1u) / PART_SIZE;
    if gid >= effectiveTB {
        return;
    }

    // Clear shared memory (512 entries = RADIX * 2)
    for (var i = gtid; i < 512u; i += US_DIM) {
        atomicStore(&g_us[i], 0u);
    }
    workgroupBarrier();

    // HistogramDigitCounts: 64 threads per sub-histogram
    let histOffset = (gtid / 64u) * RADIX;
    let partitionEnd = select(
        (gid + 1u) * PART_SIZE,
        config.numKeys,
        gid == effectiveTB - 1u
    );
    for (var i = gtid + gid * PART_SIZE; i < partitionEnd; i += US_DIM) {
        let digit = extractDigitFromIndex(i);
        atomicAdd(&g_us[digit + histOffset], 1u);
    }
    workgroupBarrier();

    // ReduceWriteDigitCounts: reduce 2 sub-histograms, write to passHist
    // config.threadBlocks is the passHist stride (= max_thread_blocks)
    for (var i = gtid; i < RADIX; i += US_DIM) {
        let v0 = atomicLoad(&g_us[i]);
        let v1 = atomicLoad(&g_us[i + RADIX]);
        let combined = v0 + v1;
        b_passHist[i * config.threadBlocks + gid] = combined;
        atomicStore(&g_us[i], combined + subgroupExclusiveAdd(combined));
    }

    // GlobalHistExclusiveScanWGE16
    workgroupBarrier();

    let laneCount = subgroup_size_p;
    if gtid < (RADIX / laneCount) {
        let idx = (gtid + 1u) * laneCount - 1u;
        let v = atomicLoad(&g_us[idx]);
        atomicStore(&g_us[idx], v + subgroupExclusiveAdd(v));
    }
    workgroupBarrier();

    let gHistOffset = globalHistOffset();
    let laneMask = laneCount - 1u;
    let circularLaneShift = (subgroup_invocation_id_p + 1u) & laneMask;

    for (var i = gtid; i < RADIX; i += US_DIM) {
        let index = circularLaneShift + (i & ~laneMask);
        let val = atomicLoad(&g_us[i]);
        var prev = 0u;
        if i >= laneCount {
            prev = subgroupShuffle(atomicLoad(&g_us[i - 1u]), 0u);
        }
        let contribution = select(val, 0u, subgroup_invocation_id_p == laneMask) + prev;
        atomicAdd(&b_globalHist[index + gHistOffset], contribution);
    }
}
