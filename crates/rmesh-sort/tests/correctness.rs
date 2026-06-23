//! GPU radix sort correctness tests.
//!
//! Catches the Chrome WebGPU sort bug noted in `src/cpu.rs::cull_and_sort`:
//! the GPU radix sort produces incorrect output on Chrome WebGPU, which is
//! why `CpuSorter` exists as a fallback. The tests pass on native (Metal /
//! Vulkan) where the sort is correct and fail on platforms where it isn't.
//!
//! Coverage:
//!   * Keys come back monotonically non-decreasing (ascending order).
//!   * Values form a permutation of the input — every input value index
//!     present exactly once.
//!   * (key, value) pairs are preserved: sorted_values[i] is the index of
//!     the i-th smallest input key (with ties broken arbitrarily by the
//!     sort, so we only require key-rank consistency).
//!
//! Skips gracefully when no GPU adapter is available, or for the DRS backend
//! when the adapter lacks `Features::SUBGROUP`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rmesh_sort::{record_radix_sort, RadixSortPipelines, RadixSortState, SortBackend};
use rmesh_util::test_util::{create_test_device, TestDeviceConfig};

const SEED: u64 = 0xA1B2_C3D4;
/// Sizes covering small / cache-shaped / large-enough-to-stress-the-spine.
/// 4_800_000 is the threshold cpu.rs cites for the Basic backend going
/// out-of-order; 8_388_608 is roughly the viewer's working set on the
/// `room_pbr_refined` scene (6.1M tets, next_pow2 = 8M).
const SIZES: &[u32] = &[1, 64, 1_024, 65_536, 1_048_576, 4_800_000, 8_388_608];

fn try_gpu(require_subgroup: bool) -> Option<(wgpu::Device, wgpu::Queue)> {
    let extra = if require_subgroup {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    create_test_device(TestDeviceConfig {
        extra_features: extra,
        ..Default::default()
    })
}

/// Run one sort and read the result back to CPU.
fn run_sort(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backend: SortBackend,
    keys_in: &[u32],
    values_in: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n = keys_in.len() as u32;
    let n_pow2 = n.next_power_of_two().max(1);
    assert_eq!(values_in.len() as u32, n);

    let pipelines = RadixSortPipelines::new(device, 1, backend);
    let state = RadixSortState::new(device, n_pow2, 32, 1, backend);
    state.upload_configs(queue);

    let buf = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let keys_a = buf("test_keys_a");
    let values_a = buf("test_values_a");

    // Upload real data into [0, n) and sentinel keys into the padding tail so
    // padding always sorts to the end (matches the convention used by
    // `project_compute_hw.wgsl`).
    let mut keys_padded = vec![u32::MAX; n_pow2 as usize];
    let mut values_padded = vec![0u32; n_pow2 as usize];
    keys_padded[..n as usize].copy_from_slice(keys_in);
    values_padded[..n as usize].copy_from_slice(values_in);
    for i in (n as usize)..(n_pow2 as usize) {
        values_padded[i] = i as u32; // padding identity, ignored after sort
    }
    queue.write_buffer(&keys_a, 0, bytemuck::cast_slice(&keys_padded));
    queue.write_buffer(&values_a, 0, bytemuck::cast_slice(&values_padded));
    queue.write_buffer(state.num_keys_buf(), 0, bytemuck::bytes_of(&n_pow2));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let result_in_b =
        record_radix_sort(&mut encoder, device, &pipelines, &state, &keys_a, &values_a);

    // Copy the destination buffers to readback staging.
    let readback = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let rb_keys = readback("rb_keys");
    let rb_values = readback("rb_values");
    let (src_keys, src_values) = if result_in_b {
        (state.keys_b(), state.values_b())
    } else {
        (&keys_a, &values_a)
    };
    encoder.copy_buffer_to_buffer(src_keys, 0, &rb_keys, 0, (n_pow2 as u64) * 4);
    encoder.copy_buffer_to_buffer(src_values, 0, &rb_values, 0, (n_pow2 as u64) * 4);
    queue.submit(std::iter::once(encoder.finish()));

    let map = |buf: &wgpu::Buffer| -> Vec<u32> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().expect("map failed");
        let data = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buf.unmap();
        out
    };
    let sorted_keys_full = map(&rb_keys);
    let sorted_values_full = map(&rb_values);

    // Drop the sentinel-padding tail before returning.
    (
        sorted_keys_full[..n as usize].to_vec(),
        sorted_values_full[..n as usize].to_vec(),
    )
}

fn check_correctness(
    backend: SortBackend,
    n: u32,
    keys_in: &[u32],
    sorted_keys: &[u32],
    sorted_values: &[u32],
) {
    // 1. Ascending order.
    for i in 1..sorted_keys.len() {
        assert!(
            sorted_keys[i - 1] <= sorted_keys[i],
            "{backend:?} n={n}: keys not sorted at i={i}: {} > {}",
            sorted_keys[i - 1],
            sorted_keys[i],
        );
    }

    // 2. Values form a permutation of [0, n).
    let mut seen = vec![false; n as usize];
    for &v in sorted_values {
        assert!(
            (v as usize) < n as usize,
            "{backend:?} n={n}: value {v} out of range [0, {n})",
        );
        assert!(
            !seen[v as usize],
            "{backend:?} n={n}: duplicate value {v} in sorted output",
        );
        seen[v as usize] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "{backend:?} n={n}: sorted values not a permutation of [0, n)",
    );

    // 3. (key, value) pairs preserved: sorted_keys[i] == input_keys[sorted_values[i]].
    for (i, &v) in sorted_values.iter().enumerate() {
        assert_eq!(
            sorted_keys[i], keys_in[v as usize],
            "{backend:?} n={n}: (key,value) pair broken at i={i} — sorted_keys[i]={}, but \
             input_keys[sorted_values[i]={v}]={}",
            sorted_keys[i], keys_in[v as usize],
        );
    }
}

fn run_correctness(backend: SortBackend, require_subgroup: bool) {
    let Some((device, queue)) = try_gpu(require_subgroup) else {
        eprintln!("skipping correctness for {backend:?}: no compatible GPU adapter");
        return;
    };

    for &n in SIZES {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ (n as u64));
        let keys: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
        let values: Vec<u32> = (0..n).collect();

        let (sorted_keys, sorted_values) = run_sort(&device, &queue, backend, &keys, &values);
        check_correctness(backend, n, &keys, &sorted_keys, &sorted_values);
        eprintln!("{backend:?} n={n}: OK");
    }
}

#[test]
fn drs_radix_sort_correctness() {
    run_correctness(SortBackend::Drs, true);
}

#[test]
fn basic_radix_sort_correctness() {
    run_correctness(SortBackend::Basic, false);
}
