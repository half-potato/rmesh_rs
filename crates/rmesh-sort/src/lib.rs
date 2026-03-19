//! Sort infrastructure shared between forward and backward passes.
//!
//! Two backends:
//!   - `basic` — 5-pass radix sort (4-bit radix, no subgroup ops, web-safe)
//!   - `drs`   — 3-kernel DeviceRadixSort (8-bit radix, requires SUBGROUP feature)

pub mod basic;
pub mod drs;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Helper to create a storage buffer bind group layout entry.
pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Helper to create a compute pipeline from a single shader and bind group layouts.
pub fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    bgls: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: bgls,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Compute 2D dispatch dimensions that stay within the 65535 limit per dimension.
pub fn dispatch_2d(total_workgroups: u32) -> (u32, u32) {
    if total_workgroups <= 65535 {
        (total_workgroups, 1)
    } else {
        let x = 65535u32;
        let y = (total_workgroups + x - 1) / x;
        (x, y)
    }
}

/// Create a zero-initialized storage buffer with COPY_DST and COPY_SRC.
pub fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// Legacy constant aliases (backward compat)
// ---------------------------------------------------------------------------

pub const RADIX_WG: u32 = 256;
pub const RADIX_BLOCK_SIZE: u32 = basic::RADIX_BLOCK_SIZE;
pub const RADIX_BIN_COUNT: u32 = basic::RADIX_BIN_COUNT;

// ---------------------------------------------------------------------------
// Unified enum API
// ---------------------------------------------------------------------------

/// Which sort backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortBackend {
    /// 5-pass, 4-bit radix. No subgroup ops — works on WebGPU/web targets.
    Basic,
    /// 3-kernel b0nes164 DeviceRadixSort, 8-bit radix. Requires `wgpu::Features::SUBGROUP`.
    Drs,
}

/// Unified pipelines enum dispatching to the chosen backend.
pub enum RadixSortPipelines {
    Basic(basic::RadixSortPipelines),
    Drs(drs::RadixSortPipelines),
}

impl RadixSortPipelines {
    pub fn new(device: &wgpu::Device, key_stride: u32, backend: SortBackend) -> Self {
        match backend {
            SortBackend::Basic => RadixSortPipelines::Basic(basic::RadixSortPipelines::new(device, key_stride)),
            SortBackend::Drs => RadixSortPipelines::Drs(drs::RadixSortPipelines::new(device, key_stride)),
        }
    }
}

/// Unified state enum dispatching to the chosen backend.
pub enum RadixSortState {
    Basic(basic::RadixSortState),
    Drs(drs::RadixSortState),
}

impl RadixSortState {
    pub fn new(device: &wgpu::Device, sort_buf_size: u32, sorting_bits: u32, key_stride: u32, backend: SortBackend) -> Self {
        match backend {
            SortBackend::Basic => RadixSortState::Basic(basic::RadixSortState::new(device, sort_buf_size, sorting_bits, key_stride)),
            SortBackend::Drs => RadixSortState::Drs(drs::RadixSortState::new(device, sort_buf_size, sorting_bits, key_stride)),
        }
    }

    pub fn upload_configs(&self, queue: &wgpu::Queue) {
        match self {
            RadixSortState::Basic(s) => s.upload_configs(queue),
            RadixSortState::Drs(s) => s.upload_configs(queue),
        }
    }

    pub fn num_keys_buf(&self) -> &wgpu::Buffer {
        match self {
            RadixSortState::Basic(s) => &s.num_keys_buf,
            RadixSortState::Drs(s) => &s.num_keys_buf,
        }
    }

    pub fn keys_b(&self) -> &wgpu::Buffer {
        match self {
            RadixSortState::Basic(s) => &s.keys_b,
            RadixSortState::Drs(s) => &s.keys_b,
        }
    }

    pub fn values_b(&self) -> &wgpu::Buffer {
        match self {
            RadixSortState::Basic(s) => &s.values_b,
            RadixSortState::Drs(s) => &s.values_b,
        }
    }
}

/// Compute sorting_bits for 64-bit tile sort keys, delegating to the backend.
pub fn sorting_bits_for_tiles(num_tiles: u32, backend: SortBackend) -> u32 {
    match backend {
        SortBackend::Basic => basic::sorting_bits_for_tiles(num_tiles),
        SortBackend::Drs => drs::sorting_bits_for_tiles(num_tiles),
    }
}

/// Record a complete radix sort of keys/values.
///
/// Panics if `pipelines` and `state` use different backends.
/// Returns `true` if result ended up in the B (alternate) buffers.
pub fn record_radix_sort(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipelines: &RadixSortPipelines,
    state: &RadixSortState,
    keys_a: &wgpu::Buffer,
    values_a: &wgpu::Buffer,
) -> bool {
    match (pipelines, state) {
        (RadixSortPipelines::Basic(p), RadixSortState::Basic(s)) => {
            basic::record_radix_sort(encoder, device, p, s, keys_a, values_a)
        }
        (RadixSortPipelines::Drs(p), RadixSortState::Drs(s)) => {
            drs::record_radix_sort(encoder, device, p, s, keys_a, values_a)
        }
        _ => panic!("mismatched sort backend: pipelines and state must use the same backend"),
    }
}
