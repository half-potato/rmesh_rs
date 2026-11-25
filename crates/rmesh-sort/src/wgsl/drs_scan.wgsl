// DeviceRadixSort — Scan kernel (WGE16 path only)
// Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)
//
// Scans only [0, effectiveTB) per digit (using config.threadBlocks as row stride),
// eliminating both passHist clearing and dead-partition scans.

const SCAN_DIM: u32 = 128u;
const RADIX: u32 = 256u;
const PART_SIZE: u32 = 2048u;

struct Config {
    numKeys: u32,
    radixShift: u32,
    threadBlocks: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read_write> b_passHist: array<u32>;

var<workgroup> g_scan: array<u32, 128>;

@compute
@workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let gtid = local_id.x;
    let gid = wid.x; // one workgroup per digit (0..255)

    let effectiveTB = min((config.numKeys + PART_SIZE - 1u) / PART_SIZE, config.threadBlocks);
    if effectiveTB == 0u {
        return;
    }

    let laneCount = sg_size;
    let laneMask = laneCount - 1u;
    let circularLaneShift = (sg_id + 1u) & laneMask;

    var reduction = 0u;
    let partitionsEnd = (effectiveTB / SCAN_DIM) * SCAN_DIM;
    let deviceOffset = gid * config.threadBlocks;

    // Full partitions
    for (var k = 0u; k < partitionsEnd; k += SCAN_DIM) {
        g_scan[gtid] = b_passHist[k + gtid + deviceOffset];
        g_scan[gtid] += subgroupExclusiveAdd(g_scan[gtid]);
        workgroupBarrier();

        if gtid < SCAN_DIM / laneCount {
            let idx = (gtid + 1u) * laneCount - 1u;
            g_scan[idx] += subgroupExclusiveAdd(g_scan[idx]);
        }
        workgroupBarrier();

        let i = k + gtid;
        let index = circularLaneShift + (i & ~laneMask);
        let val = select(g_scan[gtid], 0u, sg_id == laneMask);
        var prev = 0u;
        if gtid >= laneCount {
            prev = subgroupShuffle(g_scan[gtid - 1u], 0u);
        }
        b_passHist[index + deviceOffset] = val + prev + reduction;

        reduction += g_scan[SCAN_DIM - 1u];
        workgroupBarrier();
    }

    // Partial partition (remainder)
    let remainder = effectiveTB - partitionsEnd;
    if remainder > 0u {
        if gtid < remainder {
            g_scan[gtid] = b_passHist[partitionsEnd + gtid + deviceOffset];
        } else {
            g_scan[gtid] = 0u;
        }
        g_scan[gtid] += subgroupExclusiveAdd(g_scan[gtid]);
        workgroupBarrier();

        if gtid < SCAN_DIM / laneCount {
            let idx = (gtid + 1u) * laneCount - 1u;
            g_scan[idx] += subgroupExclusiveAdd(g_scan[idx]);
        }
        workgroupBarrier();

        let i = partitionsEnd + gtid;
        let index = circularLaneShift + (i & ~laneMask);
        if index < effectiveTB {
            let val = select(g_scan[gtid], 0u, sg_id == laneMask);
            var prev = 0u;
            if gtid >= laneCount {
                prev = subgroupShuffle(g_scan[(gtid & ~laneMask) - 1u], 0u);
            }
            b_passHist[index + deviceOffset] = val + prev + reduction;
        }
    }
}
