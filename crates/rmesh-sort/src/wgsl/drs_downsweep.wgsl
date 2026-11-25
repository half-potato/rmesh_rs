// DeviceRadixSort — Downsweep kernel (WGE16 path only)
// Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)
// Optimized for Metal / WebGPU:
//   - KEYS_PER_THREAD = 8 (PART_SIZE = 2048), fully scalarized in registers
//     (zero loop-indexed local `array<u32, N>` — prevents Metal AGX stack spilling).
//   - Fast scalar 32-lane WarpLevelMultiSplit with subgroupAll uniform-digit fast path.
//   - Plain non-atomic `g_d: array<u32, 4096>` (paired with `zero_initialize_workgroup_memory: false`).

const D_DIM: u32 = 256u;
const KEYS_PER_THREAD: u32 = 8u;
const PART_SIZE: u32 = 2048u;
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

var<workgroup> g_d: array<u32, 4096>;
var<private> subgroup_size_p: u32;
var<private> subgroup_invocation_id_p: u32;

fn waveFlags() -> u32 {
    if (subgroup_size_p & 31u) != 0u {
        return (1u << subgroup_size_p) - 1u;
    }
    return 0xffffffffu;
}

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

fn rank_key32(digit: u32, waveHistBase: u32, laneIdx: u32, peerMask: u32) -> u32 {
    let histIdx = digit + waveHistBase;
    let firstDigit = subgroupBroadcastFirst(digit);
    if subgroupAll(digit == firstDigit) {
        var preIncr = 0u;
        if laneIdx == 0u {
            preIncr = g_d[histIdx];
            g_d[histIdx] = preIncr + 32u;
        }
        return subgroupBroadcastFirst(preIncr) + laneIdx;
    } else {
        let t0 = (digit & 1u) != 0u;
        var wf = select(0xffffffffu, 0u, t0) ^ subgroupBallot(t0).x;
        let t1 = (digit & 2u) != 0u;
        wf &= select(0xffffffffu, 0u, t1) ^ subgroupBallot(t1).x;
        let t2 = (digit & 4u) != 0u;
        wf &= select(0xffffffffu, 0u, t2) ^ subgroupBallot(t2).x;
        let t3 = (digit & 8u) != 0u;
        wf &= select(0xffffffffu, 0u, t3) ^ subgroupBallot(t3).x;
        let t4 = (digit & 16u) != 0u;
        wf &= select(0xffffffffu, 0u, t4) ^ subgroupBallot(t4).x;
        let t5 = (digit & 32u) != 0u;
        wf &= select(0xffffffffu, 0u, t5) ^ subgroupBallot(t5).x;
        let t6 = (digit & 64u) != 0u;
        wf &= select(0xffffffffu, 0u, t6) ^ subgroupBallot(t6).x;
        let t7 = (digit & 128u) != 0u;
        wf &= select(0xffffffffu, 0u, t7) ^ subgroupBallot(t7).x;
        let lowestPeer = firstTrailingBit(wf);
        let peerBits = countOneBits(wf & peerMask);
        let totalBits = countOneBits(wf);

        var preIncr = 0u;
        if peerBits == 0u {
            preIncr = g_d[histIdx];
            g_d[histIdx] = preIncr + totalBits;
        }
        return subgroupShuffle(preIncr, lowestPeer) + peerBits;
    }
}

fn rank_key_generic(digit: u32, waveHistBase: u32, initMask: u32) -> u32 {
    var wf = vec4(initMask, initMask, initMask, initMask);
    for (var bit = 0u; bit < RADIX_LOG; bit++) {
        let t = ((digit >> bit) & 1u) != 0u;
        let ballot = subgroupBallot(t);
        let flip = select(0xffffffffu, 0u, t);
        wf.x &= flip ^ ballot.x;
        wf.y &= flip ^ ballot.y;
        wf.z &= flip ^ ballot.z;
        wf.w &= flip ^ ballot.w;
    }
    let histIdx = digit + waveHistBase;
    let lowestPeer = findLowestPeer(wf);
    let pt = countPeersTotal(wf);
    let peerBits = pt.x;
    let totalBits = pt.y;

    var preIncr = 0u;
    if peerBits == 0u {
        preIncr = g_d[histIdx];
        g_d[histIdx] = preIncr + totalBits;
    }
    return subgroupShuffle(preIncr, lowestPeer) + peerBits;
}

fn load_sort_word(t: u32, numKeys: u32, sw: u32) -> u32 {
    if t >= numKeys {
        return 0xffffffffu;
    }
    if KEY_STRIDE > 1u {
        return b_sort[t * KEY_STRIDE + sw];
    }
    return b_sort[t];
}

fn load_other_word(t: u32, numKeys: u32, ow: u32) -> u32 {
    if t >= numKeys {
        return 0xffffffffu;
    }
    return b_sort[t * KEY_STRIDE + ow];
}

fn scatter_key_dev(t: u32, shift: u32, sw: u32) -> u32 {
    let key = g_d[t];
    let d = (key >> shift) & RADIX_MASK;
    let dst = g_d[d + PART_SIZE] + t;
    if KEY_STRIDE > 1u {
        b_alt[dst * KEY_STRIDE + sw] = key;
    } else {
        b_alt[dst] = key;
    }
    return dst;
}

@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    subgroup_size_p = sg_size;
    subgroup_invocation_id_p = sg_id;
    let gtid = local_id.x;
    let gid = wid.x + wid.y * 65535u;

    let effectiveTB = (config.numKeys + PART_SIZE - 1u) / PART_SIZE;
    if gid >= effectiveTB {
        return;
    }

    let laneCount = subgroup_size_p;
    let laneIdx = subgroup_invocation_id_p;
    let waveIdx = gtid / laneCount;
    let isLast = (gid == effectiveTB - 1u);

    let rawShift = config.radixShift;
    let hiWord = KEY_STRIDE > 1u && rawShift >= 32u;
    let shift = select(rawShift, rawShift - 32u, hiWord);
    let sw = select(0u, 1u, hiWord);
    let ow = select(1u, 0u, hiWord);
    let gHistOff = (rawShift / RADIX_LOG) * RADIX;

    // ==== ClearWaveHists ====
    let histEnd = (D_DIM / laneCount) * RADIX;
    for (var i = gtid; i < histEnd; i += D_DIM) {
        g_d[i] = 0u;
    }
    workgroupBarrier();

    // ==== LoadKeys (scalar registers k0..k7, zero array spill) ====
    let baseOff = laneIdx + waveIdx * (KEYS_PER_THREAD * laneCount) + gid * PART_SIZE;
    var k0: u32; var k1: u32; var k2: u32; var k3: u32;
    var k4: u32; var k5: u32; var k6: u32; var k7: u32;
    var ok0 = 0u; var ok1 = 0u; var ok2 = 0u; var ok3 = 0u;
    var ok4 = 0u; var ok5 = 0u; var ok6 = 0u; var ok7 = 0u;

    if !isLast && KEY_STRIDE == 1u {
        k0 = b_sort[baseOff];
        k1 = b_sort[baseOff + laneCount];
        k2 = b_sort[baseOff + 2u * laneCount];
        k3 = b_sort[baseOff + 3u * laneCount];
        k4 = b_sort[baseOff + 4u * laneCount];
        k5 = b_sort[baseOff + 5u * laneCount];
        k6 = b_sort[baseOff + 6u * laneCount];
        k7 = b_sort[baseOff + 7u * laneCount];
    } else {
        let nk = config.numKeys;
        k0 = load_sort_word(baseOff, nk, sw);
        k1 = load_sort_word(baseOff + laneCount, nk, sw);
        k2 = load_sort_word(baseOff + 2u * laneCount, nk, sw);
        k3 = load_sort_word(baseOff + 3u * laneCount, nk, sw);
        k4 = load_sort_word(baseOff + 4u * laneCount, nk, sw);
        k5 = load_sort_word(baseOff + 5u * laneCount, nk, sw);
        k6 = load_sort_word(baseOff + 6u * laneCount, nk, sw);
        k7 = load_sort_word(baseOff + 7u * laneCount, nk, sw);
        if KEY_STRIDE > 1u {
            ok0 = load_other_word(baseOff, nk, ow);
            ok1 = load_other_word(baseOff + laneCount, nk, ow);
            ok2 = load_other_word(baseOff + 2u * laneCount, nk, ow);
            ok3 = load_other_word(baseOff + 3u * laneCount, nk, ow);
            ok4 = load_other_word(baseOff + 4u * laneCount, nk, ow);
            ok5 = load_other_word(baseOff + 5u * laneCount, nk, ow);
            ok6 = load_other_word(baseOff + 6u * laneCount, nk, ow);
            ok7 = load_other_word(baseOff + 7u * laneCount, nk, ow);
        }
    }

    // ==== RankKeysWGE16 ====
    let waveHistBase = waveIdx * RADIX;
    var o0: u32; var o1: u32; var o2: u32; var o3: u32;
    var o4: u32; var o5: u32; var o6: u32; var o7: u32;

    if laneCount == 32u {
        let peerMask = (1u << laneIdx) - 1u;
        o0 = rank_key32((k0 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o1 = rank_key32((k1 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o2 = rank_key32((k2 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o3 = rank_key32((k3 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o4 = rank_key32((k4 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o5 = rank_key32((k5 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o6 = rank_key32((k6 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
        o7 = rank_key32((k7 >> shift) & RADIX_MASK, waveHistBase, laneIdx, peerMask);
    } else {
        let initMask = waveFlags();
        o0 = rank_key_generic((k0 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o1 = rank_key_generic((k1 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o2 = rank_key_generic((k2 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o3 = rank_key_generic((k3 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o4 = rank_key_generic((k4 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o5 = rank_key_generic((k5 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o6 = rank_key_generic((k6 >> shift) & RADIX_MASK, waveHistBase, initMask);
        o7 = rank_key_generic((k7 >> shift) & RADIX_MASK, waveHistBase, initMask);
    }
    workgroupBarrier();

    // ==== WaveHistInclusiveScanCircularShiftWGE16 ====
    var histReduction = 0u;
    if gtid < RADIX {
        histReduction = g_d[gtid];
        for (var i = gtid + RADIX; i < histEnd; i += RADIX) {
            let v = g_d[i];
            histReduction += v;
            g_d[i] = histReduction - v;
        }
        histReduction += subgroupExclusiveAdd(histReduction);
    }
    workgroupBarrier();

    // ==== WaveHistReductionExclusiveScanWGE16 ====
    if gtid < RADIX {
        let laneMask = laneCount - 1u;
        let destIdx = ((laneIdx + 1u) & laneMask) + (gtid & ~laneMask);
        g_d[destIdx] = histReduction;
    }
    workgroupBarrier();

    if gtid < RADIX / laneCount {
        let v = g_d[gtid * laneCount];
        g_d[gtid * laneCount] = subgroupExclusiveAdd(v);
    }
    workgroupBarrier();

    if gtid < RADIX && laneIdx != 0u {
        let v = g_d[gtid];
        let prev = subgroupShuffle(g_d[gtid - 1u], 1u);
        g_d[gtid] = v + prev;
    }
    workgroupBarrier();

    // ==== UpdateOffsetsWGE16 ====
    if gtid >= laneCount {
        let d0 = (k0 >> shift) & RADIX_MASK; o0 += g_d[d0 + waveHistBase] + g_d[d0];
        let d1 = (k1 >> shift) & RADIX_MASK; o1 += g_d[d1 + waveHistBase] + g_d[d1];
        let d2 = (k2 >> shift) & RADIX_MASK; o2 += g_d[d2 + waveHistBase] + g_d[d2];
        let d3 = (k3 >> shift) & RADIX_MASK; o3 += g_d[d3 + waveHistBase] + g_d[d3];
        let d4 = (k4 >> shift) & RADIX_MASK; o4 += g_d[d4 + waveHistBase] + g_d[d4];
        let d5 = (k5 >> shift) & RADIX_MASK; o5 += g_d[d5 + waveHistBase] + g_d[d5];
        let d6 = (k6 >> shift) & RADIX_MASK; o6 += g_d[d6 + waveHistBase] + g_d[d6];
        let d7 = (k7 >> shift) & RADIX_MASK; o7 += g_d[d7 + waveHistBase] + g_d[d7];
    } else {
        o0 += g_d[(k0 >> shift) & RADIX_MASK];
        o1 += g_d[(k1 >> shift) & RADIX_MASK];
        o2 += g_d[(k2 >> shift) & RADIX_MASK];
        o3 += g_d[(k3 >> shift) & RADIX_MASK];
        o4 += g_d[(k4 >> shift) & RADIX_MASK];
        o5 += g_d[(k5 >> shift) & RADIX_MASK];
        o6 += g_d[(k6 >> shift) & RADIX_MASK];
        o7 += g_d[(k7 >> shift) & RADIX_MASK];
    }
    var exclusiveHistReduction = 0u;
    if gtid < RADIX {
        exclusiveHistReduction = g_d[gtid];
    }
    workgroupBarrier();

    // ==== ScatterKeysShared ====
    g_d[o0] = k0;
    g_d[o1] = k1;
    g_d[o2] = k2;
    g_d[o3] = k3;
    g_d[o4] = k4;
    g_d[o5] = k5;
    g_d[o6] = k6;
    g_d[o7] = k7;

    // ==== LoadThreadBlockReductions ====
    if gtid < RADIX {
        g_d[gtid + PART_SIZE] =
            b_globalHist[gtid + gHistOff]
            + b_passHist[gtid * config.threadBlocks + gid]
            - exclusiveHistReduction;
    }
    workgroupBarrier();

    // ==== ScatterSortWordsDevice + save global destination indices ====
    let finalPartSize = select(PART_SIZE, config.numKeys - gid * PART_SIZE, isLast);
    var dst0 = 0u; var dst1 = 0u; var dst2 = 0u; var dst3 = 0u;
    var dst4 = 0u; var dst5 = 0u; var dst6 = 0u; var dst7 = 0u;

    if !isLast {
        dst0 = scatter_key_dev(gtid, shift, sw);
        dst1 = scatter_key_dev(gtid + 256u, shift, sw);
        dst2 = scatter_key_dev(gtid + 512u, shift, sw);
        dst3 = scatter_key_dev(gtid + 768u, shift, sw);
        dst4 = scatter_key_dev(gtid + 1024u, shift, sw);
        dst5 = scatter_key_dev(gtid + 1280u, shift, sw);
        dst6 = scatter_key_dev(gtid + 1536u, shift, sw);
        dst7 = scatter_key_dev(gtid + 1792u, shift, sw);
    } else {
        if gtid < finalPartSize { dst0 = scatter_key_dev(gtid, shift, sw); }
        if gtid + 256u < finalPartSize { dst1 = scatter_key_dev(gtid + 256u, shift, sw); }
        if gtid + 512u < finalPartSize { dst2 = scatter_key_dev(gtid + 512u, shift, sw); }
        if gtid + 768u < finalPartSize { dst3 = scatter_key_dev(gtid + 768u, shift, sw); }
        if gtid + 1024u < finalPartSize { dst4 = scatter_key_dev(gtid + 1024u, shift, sw); }
        if gtid + 1280u < finalPartSize { dst5 = scatter_key_dev(gtid + 1280u, shift, sw); }
        if gtid + 1536u < finalPartSize { dst6 = scatter_key_dev(gtid + 1536u, shift, sw); }
        if gtid + 1792u < finalPartSize { dst7 = scatter_key_dev(gtid + 1792u, shift, sw); }
    }

    if KEY_STRIDE > 1u {
        workgroupBarrier();
        g_d[o0] = ok0;
        g_d[o1] = ok1;
        g_d[o2] = ok2;
        g_d[o3] = ok3;
        g_d[o4] = ok4;
        g_d[o5] = ok5;
        g_d[o6] = ok6;
        g_d[o7] = ok7;
        workgroupBarrier();

        if !isLast {
            b_alt[dst0 * KEY_STRIDE + ow] = g_d[gtid];
            b_alt[dst1 * KEY_STRIDE + ow] = g_d[gtid + 256u];
            b_alt[dst2 * KEY_STRIDE + ow] = g_d[gtid + 512u];
            b_alt[dst3 * KEY_STRIDE + ow] = g_d[gtid + 768u];
            b_alt[dst4 * KEY_STRIDE + ow] = g_d[gtid + 1024u];
            b_alt[dst5 * KEY_STRIDE + ow] = g_d[gtid + 1280u];
            b_alt[dst6 * KEY_STRIDE + ow] = g_d[gtid + 1536u];
            b_alt[dst7 * KEY_STRIDE + ow] = g_d[gtid + 1792u];
        } else {
            if gtid < finalPartSize { b_alt[dst0 * KEY_STRIDE + ow] = g_d[gtid]; }
            if gtid + 256u < finalPartSize { b_alt[dst1 * KEY_STRIDE + ow] = g_d[gtid + 256u]; }
            if gtid + 512u < finalPartSize { b_alt[dst2 * KEY_STRIDE + ow] = g_d[gtid + 512u]; }
            if gtid + 768u < finalPartSize { b_alt[dst3 * KEY_STRIDE + ow] = g_d[gtid + 768u]; }
            if gtid + 1024u < finalPartSize { b_alt[dst4 * KEY_STRIDE + ow] = g_d[gtid + 1024u]; }
            if gtid + 1280u < finalPartSize { b_alt[dst5 * KEY_STRIDE + ow] = g_d[gtid + 1280u]; }
            if gtid + 1536u < finalPartSize { b_alt[dst6 * KEY_STRIDE + ow] = g_d[gtid + 1536u]; }
            if gtid + 1792u < finalPartSize { b_alt[dst7 * KEY_STRIDE + ow] = g_d[gtid + 1792u]; }
        }
    }
    workgroupBarrier();

    // ==== Scatter payloads ====
    if !isLast {
        g_d[o0] = b_sortPayload[baseOff];
        g_d[o1] = b_sortPayload[baseOff + laneCount];
        g_d[o2] = b_sortPayload[baseOff + 2u * laneCount];
        g_d[o3] = b_sortPayload[baseOff + 3u * laneCount];
        g_d[o4] = b_sortPayload[baseOff + 4u * laneCount];
        g_d[o5] = b_sortPayload[baseOff + 5u * laneCount];
        g_d[o6] = b_sortPayload[baseOff + 6u * laneCount];
        g_d[o7] = b_sortPayload[baseOff + 7u * laneCount];
    } else {
        let nk = config.numKeys;
        if baseOff < nk { g_d[o0] = b_sortPayload[baseOff]; }
        if baseOff + laneCount < nk { g_d[o1] = b_sortPayload[baseOff + laneCount]; }
        if baseOff + 2u * laneCount < nk { g_d[o2] = b_sortPayload[baseOff + 2u * laneCount]; }
        if baseOff + 3u * laneCount < nk { g_d[o3] = b_sortPayload[baseOff + 3u * laneCount]; }
        if baseOff + 4u * laneCount < nk { g_d[o4] = b_sortPayload[baseOff + 4u * laneCount]; }
        if baseOff + 5u * laneCount < nk { g_d[o5] = b_sortPayload[baseOff + 5u * laneCount]; }
        if baseOff + 6u * laneCount < nk { g_d[o6] = b_sortPayload[baseOff + 6u * laneCount]; }
        if baseOff + 7u * laneCount < nk { g_d[o7] = b_sortPayload[baseOff + 7u * laneCount]; }
    }
    workgroupBarrier();

    if !isLast {
        b_altPayload[dst0] = g_d[gtid];
        b_altPayload[dst1] = g_d[gtid + 256u];
        b_altPayload[dst2] = g_d[gtid + 512u];
        b_altPayload[dst3] = g_d[gtid + 768u];
        b_altPayload[dst4] = g_d[gtid + 1024u];
        b_altPayload[dst5] = g_d[gtid + 1280u];
        b_altPayload[dst6] = g_d[gtid + 1536u];
        b_altPayload[dst7] = g_d[gtid + 1792u];
    } else {
        if gtid < finalPartSize { b_altPayload[dst0] = g_d[gtid]; }
        if gtid + 256u < finalPartSize { b_altPayload[dst1] = g_d[gtid + 256u]; }
        if gtid + 512u < finalPartSize { b_altPayload[dst2] = g_d[gtid + 512u]; }
        if gtid + 768u < finalPartSize { b_altPayload[dst3] = g_d[gtid + 768u]; }
        if gtid + 1024u < finalPartSize { b_altPayload[dst4] = g_d[gtid + 1024u]; }
        if gtid + 1280u < finalPartSize { b_altPayload[dst5] = g_d[gtid + 1280u]; }
        if gtid + 1536u < finalPartSize { b_altPayload[dst6] = g_d[gtid + 1536u]; }
        if gtid + 1792u < finalPartSize { b_altPayload[dst7] = g_d[gtid + 1792u]; }
    }
}
