//! GPU radix-sort throughput bench.
//!
//! Benches the two backends end-to-end on synthetic random keys, across
//! realistic tet-count scales:
//!   - `basic` — 5-pass, 4-bit radix (works without subgroup ops)
//!   - `drs`   — 3-kernel DeviceRadixSort, 8-bit radix (needs `SUBGROUP`)
//!
//! Skips gracefully if no GPU adapter is found, or for the `drs` group if
//! the adapter lacks `SUBGROUP`. Throughput is reported in keys/sec via
//! `Criterion`'s element-throughput.
//!
//! Run: `cargo bench -p rmesh-sort --bench radix_sort`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rmesh_sort::{record_radix_sort, RadixSortPipelines, RadixSortState, SortBackend};
use rmesh_util::test_util::{create_test_device, TestDeviceConfig};

const SIZES: &[u32] = &[1_000_000, 4_000_000, 16_000_000];
const SORTING_BITS: u32 = 32;
const SEED: u64 = 0xA1B2_C3D4;

struct SortState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    keys_a: wgpu::Buffer,
    values_a: wgpu::Buffer,
    pipelines: RadixSortPipelines,
    state: RadixSortState,
}

fn make_device(require_subgroup: bool) -> Option<(wgpu::Device, wgpu::Queue)> {
    let extra = if require_subgroup {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    // Pass an explicit config so we don't auto-require SUBGROUP for the
    // basic backend (which doesn't need it).
    create_test_device(TestDeviceConfig {
        extra_features: extra,
        ..Default::default()
    })
}

fn make_sort_state(
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: SortBackend,
    n: u32,
) -> SortState {
    let n_pow2 = n.next_power_of_two();
    let pipelines = RadixSortPipelines::new(&device, 1, backend);
    let state = RadixSortState::new(&device, n_pow2, SORTING_BITS, 1, backend);
    state.upload_configs(&queue);

    let buf = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_pow2 as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let keys_a = buf("bench_keys_a");
    let values_a = buf("bench_values_a");

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let keys: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let values: Vec<u32> = (0..n).collect();
    queue.write_buffer(&keys_a, 0, bytemuck::cast_slice(&keys));
    queue.write_buffer(&values_a, 0, bytemuck::cast_slice(&values));
    queue.write_buffer(state.num_keys_buf(), 0, bytemuck::bytes_of(&n));

    SortState {
        device,
        queue,
        keys_a,
        values_a,
        pipelines,
        state,
    }
}

fn record_and_submit(s: &SortState) {
    let mut encoder = s
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let _ = record_radix_sort(
        &mut encoder,
        &s.device,
        &s.pipelines,
        &s.state,
        &s.keys_a,
        &s.values_a,
    );
    s.queue.submit(std::iter::once(encoder.finish()));
    let _ = s.device.poll(wgpu::PollType::wait_indefinitely());
}

fn bench_backend(c: &mut Criterion, backend: SortBackend, label: &str) {
    let require_subgroup = matches!(backend, SortBackend::Drs);
    let mut group = c.benchmark_group(label);
    group.sample_size(20);

    for &n in SIZES {
        let (device, queue) = match make_device(require_subgroup) {
            Some(dq) => dq,
            None => {
                eprintln!(
                    "Skipping {label} (no GPU with required features: subgroup={require_subgroup})"
                );
                group.finish();
                return;
            }
        };
        let s = make_sort_state(device, queue, backend, n);
        // Warmup pass so first-iteration shader compile / buffer allocation
        // costs don't pollute the sample window.
        record_and_submit(&s);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &s, |b, s| {
            b.iter(|| record_and_submit(s));
        });
    }
    group.finish();
}

fn bench_basic(c: &mut Criterion) {
    bench_backend(c, SortBackend::Basic, "radix_sort/basic");
}

fn bench_drs(c: &mut Criterion) {
    bench_backend(c, SortBackend::Drs, "radix_sort/drs");
}

criterion_group! {
    name = radix_sort;
    config = Criterion::default();
    targets = bench_basic, bench_drs
}
criterion_main!(radix_sort);
