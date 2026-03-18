// DeviceRadixSort — Downsweep kernel (WGE16 path only)
// Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)

const D_DIM: u32 = 256u;
const KEYS_PER_THREAD: u32 = 15u;
const PART_SIZE: u32 = 3840u;
const D_TOTAL_SMEM: u32 = 4096u;
const RADIX: u32 = 256u;
const RADIX_MASK: u32 = 255u;
const RADIX_LOG: u32 = 8u;
const KEY_STRIDE: u32 = /*KEY_STRIDE*/1u;

struct Config {
    numKeys: u32,
    radixShift: u32,
    threadBlocks: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read> b_sort: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_alt: array<u32>;
@group(0) @binding(3) var<storage, read> b_sortPayload: array<u32>;
@group(0) @binding(4) var<storage, read_write> b_altPayload: array<u32>;
@group(0) @binding(5) var<storage, read> b_globalHist: array<u32>;
@group(0) @binding(6) var<storage, read> b_passHist: array<u32>;

var<workgroup> g_d: array<atomic<u32>, 4096>;
var<private> subgroup_size_p: u32;
var<private> subgroup_invocation_id_p: u32;

// ---- Helpers ----

fn effShift() -> u32 {
    if KEY_STRIDE > 1u && config.radixShift >= 32u {
        return config.radixShift - 32u;
    }
    return config.radixShift;
}

fn extractDigit(key: u32) -> u32 {
    return (key >> effShift()) & RADIX_MASK;
}

fn sortWordIdx() -> u32 {
    return select(0u, 1u, KEY_STRIDE > 1u && config.radixShift >= 32u);
}

fn otherWordIdx() -> u32 {
    return select(1u, 0u, KEY_STRIDE > 1u && config.radixShift >= 32u);
}

fn getWaveIndex(gtid: u32) -> u32 {
    return gtid / subgroup_size_p;
}

fn subPartSize() -> u32 {
    return KEYS_PER_THREAD * subgroup_size_p;
}

fn sharedOffset(gtid: u32) -> u32 {
    return subgroup_invocation_id_p + getWaveIndex(gtid) * subPartSize();
}

fn deviceOffset(gtid: u32, partIndex: u32) -> u32 {
    return sharedOffset(gtid) + partIndex * PART_SIZE;
}

fn waveHistsSize() -> u32 {
    return (D_DIM / subgroup_size_p) * RADIX;
}

fn globalHistOff() -> u32 {
    return (config.radixShift / RADIX_LOG) * RADIX;
}

fn waveFlags() -> u32 {
    if (subgroup_size_p & 31u) != 0u {
        return (1u << subgroup_size_p) - 1u;
    }
    return 0xffffffffu;
}

// Count peer bits (set bits at positions < laneIdx) and total bits in waveFlags
fn countPeersTotal(wf: vec4<u32>) -> vec2<u32> {
    let li = subgroup_invocation_id_p;
    var peer = 0u;
    var total = countOneBits(wf.x);

    if li < 32u {
        peer += countOneBits(wf.x & ((1u << (li & 31u)) - 1u));
    } else {
        peer += countOneBits(wf.x);
    }

    if subgroup_size_p > 32u {
        total += countOneBits(wf.y);
        if li >= 32u {
            if li < 64u {
                peer += countOneBits(wf.y & ((1u << (li & 31u)) - 1u));
            } else {
                peer += countOneBits(wf.y);
            }
        }
    }
    if subgroup_size_p > 64u {
        total += countOneBits(wf.z);
        if li >= 64u {
            if li < 96u {
                peer += countOneBits(wf.z & ((1u << (li & 31u)) - 1u));
            } else {
                peer += countOneBits(wf.z);
            }
        }
    }
    if subgroup_size_p > 96u {
        total += countOneBits(wf.w);
        if li >= 96u {
            peer += countOneBits(wf.w & ((1u << (li & 31u)) - 1u));
        }
    }
    return vec2(peer, total);
}

fn findLowestPeer(wf: vec4<u32>) -> u32 {
    var fbl = firstTrailingBit(wf.x);
    if fbl != 0xffffffffu { return fbl; }
    if subgroup_size_p > 32u {
        fbl = firstTrailingBit(wf.y);
        if fbl != 0xffffffffu { return 32u + fbl; }
    }
    if subgroup_size_p > 64u {
        fbl = firstTrailingBit(wf.z);
        if fbl != 0xffffffffu { return 64u + fbl; }
    }
    if subgroup_size_p > 96u {
        fbl = firstTrailingBit(wf.w);
        if fbl != 0xffffffffu { return 96u + fbl; }
    }
    return 0u;
}

@compute
@workgroup_size(256, 1, 1)
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

    let laneCount = subgroup_size_p;
    let laneIdx = subgroup_invocation_id_p;
    let waveIdx = getWaveIndex(gtid);
    let isLast = (gid == effectiveTB - 1u);

    // ==== ClearWaveHists ====
    let histEnd = waveHistsSize();
    for (var i = gtid; i < histEnd; i += D_DIM) {
        atomicStore(&g_d[i], 0u);
    }
    workgroupBarrier();

    // ==== LoadKeys ====
    var sort_keys: array<u32, 15>; // the word used for digit extraction
    var other_keys: array<u32, 15>; // the other word (KEY_STRIDE > 1 only)

    for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
        let t = deviceOffset(gtid, gid) + ki * laneCount;
        if isLast && t >= config.numKeys {
            sort_keys[ki] = 0xffffffffu;
            other_keys[ki] = 0xffffffffu;
        } else {
            sort_keys[ki] = b_sort[t * KEY_STRIDE + sortWordIdx()];
            if KEY_STRIDE > 1u {
                other_keys[ki] = b_sort[t * KEY_STRIDE + otherWordIdx()];
            }
        }
    }

    // ==== RankKeysWGE16 ====
    var offsets: array<u32, 15>;

    for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
        // WarpLevelMultiSplit
        var wf = vec4(waveFlags(), waveFlags(), waveFlags(), waveFlags());

        for (var bit = 0u; bit < RADIX_LOG; bit++) {
            let t = bool((sort_keys[ki] >> (bit + effShift())) & 1u);
            let ballot = subgroupBallot(t);
            let flip = select(0xffffffffu, 0u, t);
            wf.x &= flip ^ ballot.x;
            wf.y &= flip ^ ballot.y;
            wf.z &= flip ^ ballot.z;
            wf.w &= flip ^ ballot.w;
        }

        let digit = extractDigit(sort_keys[ki]);
        let histIdx = digit + waveIdx * RADIX;
        let lowestPeer = findLowestPeer(wf);
        let pt = countPeersTotal(wf);
        let peerBits = pt.x;
        let totalBits = pt.y;

        var preIncr = 0u;
        if peerBits == 0u {
            preIncr = atomicAdd(&g_d[histIdx], totalBits);
        }
        offsets[ki] = subgroupShuffle(preIncr, lowestPeer) + peerBits;
    }
    workgroupBarrier();

    // ==== WaveHistInclusiveScanCircularShiftWGE16 ====
    var histReduction = 0u;
    if gtid < RADIX {
        histReduction = atomicLoad(&g_d[gtid]);
        for (var i = gtid + RADIX; i < histEnd; i += RADIX) {
            let v = atomicLoad(&g_d[i]);
            histReduction += v;
            atomicStore(&g_d[i], histReduction - v);
        }
        histReduction += subgroupExclusiveAdd(histReduction);
    }
    workgroupBarrier();

    // ==== WaveHistReductionExclusiveScanWGE16 ====
    if gtid < RADIX {
        let laneMask = laneCount - 1u;
        let destIdx = ((laneIdx + 1u) & laneMask) + (gtid & ~laneMask);
        atomicStore(&g_d[destIdx], histReduction);
    }
    workgroupBarrier();

    if gtid < RADIX / laneCount {
        let v = atomicLoad(&g_d[gtid * laneCount]);
        atomicStore(&g_d[gtid * laneCount], subgroupExclusiveAdd(v));
    }
    workgroupBarrier();

    if gtid < RADIX && laneIdx != 0u {
        let v = atomicLoad(&g_d[gtid]);
        let prev = subgroupShuffle(atomicLoad(&g_d[gtid - 1u]), 1u);
        atomicStore(&g_d[gtid], v + prev);
    }
    workgroupBarrier();

    // ==== UpdateOffsetsWGE16 ====
    var exclusiveHistReduction = 0u;
    if gtid >= laneCount {
        let t = waveIdx * RADIX;
        for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
            let t2 = extractDigit(sort_keys[ki]);
            offsets[ki] += u32(atomicLoad(&g_d[t2 + t])) + u32(atomicLoad(&g_d[t2]));
        }
    } else {
        for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
            offsets[ki] += u32(atomicLoad(&g_d[extractDigit(sort_keys[ki])]));
        }
    }
    if gtid < RADIX {
        exclusiveHistReduction = atomicLoad(&g_d[gtid]);
    }
    workgroupBarrier();

    // ==== ScatterKeysShared (sort words) ====
    for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
        atomicStore(&g_d[offsets[ki]], sort_keys[ki]);
    }

    // ==== LoadThreadBlockReductions ====
    if gtid < RADIX {
        atomicStore(&g_d[gtid + PART_SIZE],
            b_globalHist[gtid + globalHistOff()]
            + b_passHist[gtid * config.threadBlocks + gid]
            - exclusiveHistReduction);
    }
    workgroupBarrier();

    // ==== ScatterSortWordsDevice + save digits ====
    let finalPartSize = select(PART_SIZE, config.numKeys - gid * PART_SIZE, isLast);
    var digits: array<u32, 15>;

    for (var ti = 0u; ti < KEYS_PER_THREAD; ti++) {
        let t = gtid + ti * D_DIM;
        if t < finalPartSize {
            let key = atomicLoad(&g_d[t]);
            let d = extractDigit(key);
            digits[ti] = d;
            let dst = atomicLoad(&g_d[d + PART_SIZE]) + t;
            if KEY_STRIDE > 1u {
                b_alt[dst * KEY_STRIDE + sortWordIdx()] = key;
            } else {
                b_alt[dst] = key;
            }
        }
    }

    // ==== Scatter other key words (KEY_STRIDE > 1 only) ====
    if KEY_STRIDE > 1u {
        workgroupBarrier();
        for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
            atomicStore(&g_d[offsets[ki]], other_keys[ki]);
        }
        workgroupBarrier();

        for (var ti = 0u; ti < KEYS_PER_THREAD; ti++) {
            let t = gtid + ti * D_DIM;
            if t < finalPartSize {
                let val = atomicLoad(&g_d[t]);
                let dst = atomicLoad(&g_d[digits[ti] + PART_SIZE]) + t;
                b_alt[dst * KEY_STRIDE + otherWordIdx()] = val;
            }
        }
    }
    workgroupBarrier();

    // ==== Scatter payloads ====
    // Reload payloads from device
    var payloads: array<u32, 15>;
    for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
        let t = deviceOffset(gtid, gid) + ki * laneCount;
        if !isLast || t < config.numKeys {
            payloads[ki] = b_sortPayload[t];
        }
    }

    // Scatter payloads to shared
    for (var ki = 0u; ki < KEYS_PER_THREAD; ki++) {
        atomicStore(&g_d[offsets[ki]], payloads[ki]);
    }
    workgroupBarrier();

    // Write payloads to device
    for (var ti = 0u; ti < KEYS_PER_THREAD; ti++) {
        let t = gtid + ti * D_DIM;
        if t < finalPartSize {
            let payload = atomicLoad(&g_d[t]);
            let dst = atomicLoad(&g_d[digits[ti] + PART_SIZE]) + t;
            b_altPayload[dst] = payload;
        }
    }
}
