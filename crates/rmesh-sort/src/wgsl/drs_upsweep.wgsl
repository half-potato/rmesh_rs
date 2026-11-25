// DeviceRadixSort — Upsweep kernel (WGE16 path only)
// Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)
// Optimized: 4 sub-histograms (1 per 32-lane subgroup) to halve atomic contention.

const US_DIM: u32 = 128u;
const RADIX: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const RADIX_LOG: u32 = 8u;
const PART_SIZE: u32 = 2048u;
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

// 4 sub-histograms (1024 atomic entries = 4 KB; atomic avoids naga's single-thread zero-init).
var<workgroup> g_us: array<atomic<u32>, 1024>;

@compute
@workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let gtid = local_id.x;
    let gid = wid.x + wid.y * 65535u;

    let effectiveTB = (config.numKeys + PART_SIZE - 1u) / PART_SIZE;
    if gid >= effectiveTB {
        return;
    }

    // Clear 4 sub-histograms.
    for (var i = gtid; i < 1024u; i += US_DIM) {
        atomicStore(&g_us[i], 0u);
    }
    workgroupBarrier();

    // HistogramDigitCounts: 32 threads per sub-histogram (1 per 32-lane wave).
    let histOffset = (gtid >> 5u) * RADIX;
    let isLast = gid == effectiveTB - 1u;
    let rawShift = config.radixShift;
    let hiWord = KEY_STRIDE > 1u && rawShift >= 32u;
    let shift = select(rawShift, rawShift - 32u, hiWord);
    let wordOff = select(0u, 1u, hiWord);

    if !isLast {
        let base = gtid + gid * PART_SIZE;
        for (var k = 0u; k < PART_SIZE; k += US_DIM) {
            let i = base + k;
            let key = select(b_sort[i], b_sort[i * KEY_STRIDE + wordOff], KEY_STRIDE > 1u);
            let digit = (key >> shift) & RADIX_MASK;
            let d0 = subgroupBroadcastFirst(digit);
            if subgroupAll(digit == d0) {
                if sg_id == 0u {
                    atomicAdd(&g_us[d0 + histOffset], sg_size);
                }
            } else {
                atomicAdd(&g_us[digit + histOffset], 1u);
            }
        }
    } else {
        for (var i = gtid + gid * PART_SIZE; i < config.numKeys; i += US_DIM) {
            let key = select(b_sort[i], b_sort[i * KEY_STRIDE + wordOff], KEY_STRIDE > 1u);
            let digit = (key >> shift) & RADIX_MASK;
            atomicAdd(&g_us[digit + histOffset], 1u);
        }
    }
    workgroupBarrier();

    // Reduce 4 sub-histograms and write passHist.
    for (var i = gtid; i < RADIX; i += US_DIM) {
        let combined = atomicLoad(&g_us[i])
            + atomicLoad(&g_us[i + RADIX])
            + atomicLoad(&g_us[i + 2u * RADIX])
            + atomicLoad(&g_us[i + 3u * RADIX]);
        b_passHist[i * config.threadBlocks + gid] = combined;
        atomicStore(&g_us[i], combined + subgroupExclusiveAdd(combined));
    }
    workgroupBarrier();

    let laneCount = sg_size;
    if gtid < (RADIX / laneCount) {
        let idx = (gtid + 1u) * laneCount - 1u;
        let v = atomicLoad(&g_us[idx]);
        atomicStore(&g_us[idx], v + subgroupExclusiveAdd(v));
    }
    workgroupBarrier();

    let gHistOffset = (rawShift / RADIX_LOG) * RADIX;
    let laneMask = laneCount - 1u;
    let circularLaneShift = (sg_id + 1u) & laneMask;

    for (var i = gtid; i < RADIX; i += US_DIM) {
        let index = circularLaneShift + (i & ~laneMask);
        let val = atomicLoad(&g_us[i]);
        var prev = 0u;
        if i >= laneCount {
            prev = subgroupShuffle(atomicLoad(&g_us[i - 1u]), 0u);
        }
        let contribution = select(val, 0u, sg_id == laneMask) + prev;
        atomicAdd(&b_globalHist[index + gHistOffset], contribution);
    }
}
