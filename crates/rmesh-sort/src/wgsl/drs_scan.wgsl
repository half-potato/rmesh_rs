// DeviceRadixSort — Scan kernel (WGE16 path only)
// Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)
//
// Each workgroup scans one digit's partition-level counts in b_passHist.
// Dispatched with RADIX (256) workgroups.

const SCAN_DIM: u32 = 128u;
const RADIX: u32 = 256u;

struct Config {
    numKeys: u32,
    radixShift: u32,
    threadBlocks: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read_write> b_passHist: array<u32>;

var<workgroup> g_scan: array<u32, 128>;
var<private> subgroup_size_p: u32;
var<private> subgroup_invocation_id_p: u32;

@compute
@workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    subgroup_size_p = sg_size;
    subgroup_invocation_id_p = sg_id;
    let gtid = local_id.x;
    let gid = wid.x; // one workgroup per digit (0..255)

    let laneCount = subgroup_size_p;
    let laneMask = laneCount - 1u;
    let circularLaneShift = (subgroup_invocation_id_p + 1u) & laneMask;

    var reduction = 0u;
    let partitionsEnd = (config.threadBlocks / SCAN_DIM) * SCAN_DIM;
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
        let val = select(g_scan[gtid], 0u, subgroup_invocation_id_p == laneMask);
        var prev = 0u;
        if gtid >= laneCount {
            prev = subgroupShuffle(g_scan[gtid - 1u], 0u);
        }
        b_passHist[index + deviceOffset] = val + prev + reduction;

        reduction += g_scan[SCAN_DIM - 1u];
        workgroupBarrier();
    }

    // Partial partition (remainder)
    let remainder = config.threadBlocks - partitionsEnd;
    if remainder > 0u {
        let i_local = gtid;
        if i_local < remainder {
            g_scan[gtid] = b_passHist[partitionsEnd + i_local + deviceOffset];
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
        if index < config.threadBlocks {
            let val = select(g_scan[gtid], 0u, subgroup_invocation_id_p == laneMask);
            var prev = 0u;
            if gtid >= laneCount {
                prev = subgroupShuffle(g_scan[(gtid & ~laneMask) - 1u], 0u);
            }
            b_passHist[index + deviceOffset] = val + prev + reduction;
        }
    }
}
