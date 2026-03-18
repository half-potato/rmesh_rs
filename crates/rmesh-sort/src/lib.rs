//! Sort infrastructure shared between forward and backward passes.
//!
//! Contains:
//!   - 3-kernel DeviceRadixSort pipeline (drs_upsweep, drs_scan, drs_downsweep)
//!   - Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024)
const DRS_UPSWEEP_WGSL: &str = include_str!("wgsl/drs_upsweep.wgsl");
const DRS_SCAN_WGSL: &str = include_str!("wgsl/drs_scan.wgsl");
const DRS_DOWNSWEEP_WGSL: &str = include_str!("wgsl/drs_downsweep.wgsl");

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

// ===========================================================================
// DeviceRadixSort (8-bit radix, 3 kernels per digit pass)
// ===========================================================================

/// Constants matching the WGSL shaders.
pub const RADIX: u32 = 256;
pub const RADIX_LOG: u32 = 8;
pub const D_DIM: u32 = 256;
pub const KEYS_PER_THREAD: u32 = 15;
pub const PART_SIZE: u32 = D_DIM * KEYS_PER_THREAD; // 3840
pub const US_DIM: u32 = 128;
pub const SCAN_DIM: u32 = 128;

// Keep old names as aliases for callers that reference them
pub const RADIX_WG: u32 = D_DIM;
pub const RADIX_BLOCK_SIZE: u32 = PART_SIZE;
pub const RADIX_BIN_COUNT: u32 = RADIX;

/// Compute sorting_bits for 64-bit tile sort keys: 32 bits of depth + enough
/// bits to distinguish all tile IDs, rounded up to a multiple of 8.
pub fn sorting_bits_for_tiles(num_tiles: u32) -> u32 {
    let tile_bits = if num_tiles <= 1 { 1 } else { 32 - (num_tiles - 1).leading_zeros() };
    (32 + tile_bits + 7) & !7
}

/// Pipelines for the 3-kernel DeviceRadixSort.
pub struct RadixSortPipelines {
    pub upsweep_pipeline: wgpu::ComputePipeline,
    pub upsweep_bgl: wgpu::BindGroupLayout,
    pub scan_pipeline: wgpu::ComputePipeline,
    pub scan_bgl: wgpu::BindGroupLayout,
    pub downsweep_pipeline: wgpu::ComputePipeline,
    pub downsweep_bgl: wgpu::BindGroupLayout,
}

impl RadixSortPipelines {
    pub fn new(device: &wgpu::Device, key_stride: u32) -> Self {
        let upsweep_src = DRS_UPSWEEP_WGSL.replace("/*KEY_STRIDE*/1u", &format!("/*KEY_STRIDE*/{key_stride}u"));
        let downsweep_src = DRS_DOWNSWEEP_WGSL.replace("/*KEY_STRIDE*/1u", &format!("/*KEY_STRIDE*/{key_stride}u"));

        // Upsweep: config(r), b_sort(r), b_globalHist(rw), b_passHist(rw)
        let upsweep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drs_upsweep"),
            source: wgpu::ShaderSource::Wgsl(upsweep_src.into()),
        });
        let upsweep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("drs_upsweep_bgl"),
            entries: &[
                storage_entry(0, true),  // config
                storage_entry(1, true),  // b_sort
                storage_entry(2, false), // b_globalHist
                storage_entry(3, false), // b_passHist
            ],
        });
        let upsweep_pipeline = make_compute_pipeline(device, "drs_upsweep", &upsweep_shader, &[&upsweep_bgl]);

        // Scan: config(r), b_passHist(rw)
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drs_scan"),
            source: wgpu::ShaderSource::Wgsl(DRS_SCAN_WGSL.into()),
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("drs_scan_bgl"),
            entries: &[
                storage_entry(0, true),  // config
                storage_entry(1, false), // b_passHist
            ],
        });
        let scan_pipeline = make_compute_pipeline(device, "drs_scan", &scan_shader, &[&scan_bgl]);

        // Downsweep: config(r), b_sort(r), b_alt(rw), b_sortPayload(r), b_altPayload(rw), b_globalHist(r), b_passHist(r)
        let downsweep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drs_downsweep"),
            source: wgpu::ShaderSource::Wgsl(downsweep_src.into()),
        });
        let downsweep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("drs_downsweep_bgl"),
            entries: &[
                storage_entry(0, true),  // config
                storage_entry(1, true),  // b_sort
                storage_entry(2, false), // b_alt
                storage_entry(3, true),  // b_sortPayload
                storage_entry(4, false), // b_altPayload
                storage_entry(5, true),  // b_globalHist
                storage_entry(6, true),  // b_passHist
            ],
        });
        let downsweep_pipeline = make_compute_pipeline(device, "drs_downsweep", &downsweep_shader, &[&downsweep_bgl]);

        Self {
            upsweep_pipeline,
            upsweep_bgl,
            scan_pipeline,
            scan_bgl,
            downsweep_pipeline,
            downsweep_bgl,
        }
    }
}

/// Buffers and bind groups for the radix sort.
pub struct RadixSortState {
    pub keys_b: wgpu::Buffer,
    pub values_b: wgpu::Buffer,
    pub global_hist: wgpu::Buffer,
    pub pass_hist: wgpu::Buffer,
    pub num_keys_buf: wgpu::Buffer,
    pub config_buffers: Vec<wgpu::Buffer>,
    pub max_thread_blocks: u32,
    pub sorting_bits: u32,
    pub key_stride: u32,
}

impl RadixSortState {
    pub fn new(device: &wgpu::Device, sort_buf_size: u32, sorting_bits: u32, key_stride: u32) -> Self {
        let max_thread_blocks = (sort_buf_size + PART_SIZE - 1) / PART_SIZE;
        let num_passes = (sorting_bits + 7) / 8;

        let keys_b = create_storage_buffer(device, "radix_keys_b", (sort_buf_size as u64) * (key_stride as u64) * 4);
        let values_b = create_storage_buffer(device, "radix_values_b", (sort_buf_size as u64) * 4);

        // Global histogram: RADIX * num_passes entries
        let global_hist = create_storage_buffer(
            device,
            "radix_global_hist",
            (RADIX as u64) * (num_passes as u64) * 4,
        );

        // Pass histogram: RADIX * max_thread_blocks entries
        let pass_hist = create_storage_buffer(
            device,
            "radix_pass_hist",
            (RADIX as u64) * (max_thread_blocks as u64) * 4,
        );

        let num_keys_buf = create_storage_buffer(device, "radix_num_keys", 4);

        // Config buffer per pass: {numKeys, radixShift, threadBlocks, _pad}
        let config_buffers: Vec<wgpu::Buffer> = (0..num_passes)
            .map(|pass| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("radix_config_{pass}")),
                    size: 16, // 4 u32s
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            keys_b,
            values_b,
            global_hist,
            pass_hist,
            num_keys_buf,
            config_buffers,
            max_thread_blocks,
            sorting_bits,
            key_stride,
        }
    }

    /// Write per-pass config to GPU buffers. Call once at init.
    /// numKeys is set to 0 here and updated per-frame via queue.write_buffer.
    pub fn upload_configs(&self, queue: &wgpu::Queue) {
        let num_passes = (self.sorting_bits + 7) / 8;
        for pass in 0..num_passes {
            let config = [0u32, pass * 8, self.max_thread_blocks, 0u32];
            queue.write_buffer(&self.config_buffers[pass as usize], 0, bytemuck::cast_slice(&config));
        }
    }
}

/// Record a complete radix sort of keys/values.
///
/// After sorting, the result ends up in either the primary (A) or alternate (B) buffers
/// depending on the number of passes. Returns `true` if result is in B buffers.
pub fn record_radix_sort(
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    pipelines: &RadixSortPipelines,
    state: &RadixSortState,
    keys_a: &wgpu::Buffer,
    values_a: &wgpu::Buffer,
) -> bool {
    let num_passes = (state.sorting_bits + 7) / 8;
    let thread_blocks = state.max_thread_blocks;
    let (us_dx, us_dy) = dispatch_2d(thread_blocks);
    let (ds_dx, ds_dy) = dispatch_2d(thread_blocks);

    // Copy numKeys from num_keys_buf into the first u32 of each config buffer.
    // This supports both CPU-written and GPU-written (e.g. scan shader) numKeys.
    for config_buf in &state.config_buffers {
        encoder.copy_buffer_to_buffer(&state.num_keys_buf, 0, config_buf, 0, 4);
    }

    // Clear global histogram and pass histogram
    encoder.clear_buffer(&state.global_hist, 0, None);
    encoder.clear_buffer(&state.pass_hist, 0, None);

    for pass in 0..num_passes {
        let even = pass % 2 == 0;
        let (src_keys, src_vals, dst_keys, dst_vals) = if even {
            (keys_a, values_a, &state.keys_b, &state.values_b)
        } else {
            (&state.keys_b, &state.values_b, keys_a, values_a)
        };

        let config_buf = &state.config_buffers[pass as usize];

        // 1. Upsweep
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("drs_upsweep_bg"),
                layout: &pipelines.upsweep_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: config_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.global_hist.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: state.pass_hist.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("drs_upsweep"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.upsweep_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(us_dx, us_dy, 1);
        }

        // 2. Scan (256 workgroups — one per digit)
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("drs_scan_bg"),
                layout: &pipelines.scan_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: config_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.pass_hist.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("drs_scan"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.scan_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(RADIX, 1, 1);
        }

        // 3. Downsweep
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("drs_downsweep_bg"),
                layout: &pipelines.downsweep_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: config_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dst_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: src_vals.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: dst_vals.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: state.global_hist.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: state.pass_hist.as_entire_binding() },
                ],
            });
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("drs_downsweep"),
                timestamp_writes: None,
            });
            p.set_pipeline(&pipelines.downsweep_pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(ds_dx, ds_dy, 1);
        }
    }

    // Return true if result ended up in B buffers (odd number of passes)
    num_passes % 2 != 0
}
