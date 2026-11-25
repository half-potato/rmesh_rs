//! 3-kernel DeviceRadixSort (8-bit radix, uses subgroup ops).
//!
//! Ported from b0nes164/GPUSorting (MIT, Thomas Smith 2024).
//! Kernels: setup (indirect args) → [upsweep → scan → downsweep] × passes

use std::sync::Mutex;

use crate::{create_storage_buffer, storage_entry};

/// Create a DRS compute pipeline without naga's single-threaded workgroup zero-init.
/// Safe because every DRS kernel explicitly initializes all workgroup elements before reading.
fn make_drs_pipeline(
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
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    })
}

static DRS_UPSWEEP_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/drs_upsweep.wgsl");
static DRS_SCAN_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/drs_scan.wgsl");
static DRS_DOWNSWEEP_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("wgsl/drs_downsweep.wgsl");

const DRS_SETUP_WGSL: &str = r#"
const PART_SIZE: u32 = 2048u;
@group(0) @binding(0) var<storage, read> num_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_args: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let n = num_keys[0];
    let tb = (n + PART_SIZE - 1u) / PART_SIZE;
    if tb <= 65535u {
        dispatch_args[0] = tb;
        dispatch_args[1] = 1u;
    } else {
        dispatch_args[0] = 65535u;
        dispatch_args[1] = (tb + 65534u) / 65535u;
    }
    dispatch_args[2] = 1u;
}
"#;

/// Constants matching the WGSL shaders.
pub const RADIX: u32 = 256;
pub const RADIX_LOG: u32 = 8;
pub const D_DIM: u32 = 256;
pub const KEYS_PER_THREAD: u32 = 8;
pub const PART_SIZE: u32 = D_DIM * KEYS_PER_THREAD; // 2048
pub const US_DIM: u32 = 128;
pub const SCAN_DIM: u32 = 128;

/// Compute sorting_bits for 64-bit tile sort keys: 32 bits of depth + enough
/// bits to distinguish all tile IDs, rounded up to a multiple of 8.
pub fn sorting_bits_for_tiles(num_tiles: u32) -> u32 {
    let tile_bits = if num_tiles <= 1 {
        1
    } else {
        32 - (num_tiles - 1).leading_zeros()
    };
    (32 + tile_bits + 7) & !7
}

/// Pipelines for the 3-kernel DeviceRadixSort (+ 1-thread indirect dispatch setup).
pub struct RadixSortPipelines {
    pub setup_pipeline: wgpu::ComputePipeline,
    pub setup_bgl: wgpu::BindGroupLayout,
    pub upsweep_pipeline: wgpu::ComputePipeline,
    pub upsweep_bgl: wgpu::BindGroupLayout,
    pub scan_pipeline: wgpu::ComputePipeline,
    pub scan_bgl: wgpu::BindGroupLayout,
    pub downsweep_pipeline: wgpu::ComputePipeline,
    pub downsweep_bgl: wgpu::BindGroupLayout,
}

impl RadixSortPipelines {
    pub fn new(device: &wgpu::Device, key_stride: u32) -> Self {
        let setup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drs_setup"),
            source: wgpu::ShaderSource::Wgsl(DRS_SETUP_WGSL.into()),
        });
        let setup_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("drs_setup_bgl"),
            entries: &[
                storage_entry(0, true),  // num_keys
                storage_entry(1, false), // dispatch_args
            ],
        });
        let setup_pipeline =
            make_drs_pipeline(device, "drs_setup", &setup_shader, &[&setup_bgl]);

        let upsweep_src =
            DRS_UPSWEEP_WGSL.replace("/*KEY_STRIDE*/1u", &format!("/*KEY_STRIDE*/{key_stride}u"));
        let downsweep_src = DRS_DOWNSWEEP_WGSL
            .replace("/*KEY_STRIDE*/1u", &format!("/*KEY_STRIDE*/{key_stride}u"));

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
        let upsweep_pipeline =
            make_drs_pipeline(device, "drs_upsweep", &upsweep_shader, &[&upsweep_bgl]);

        // Scan: config(r), b_passHist(rw)
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drs_scan"),
            source: wgpu::ShaderSource::Wgsl(DRS_SCAN_WGSL.as_str().into()),
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("drs_scan_bgl"),
            entries: &[
                storage_entry(0, true),  // config
                storage_entry(1, false), // b_passHist
            ],
        });
        let scan_pipeline = make_drs_pipeline(device, "drs_scan", &scan_shader, &[&scan_bgl]);

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
        let downsweep_pipeline = make_drs_pipeline(
            device,
            "drs_downsweep",
            &downsweep_shader,
            &[&downsweep_bgl],
        );

        Self {
            setup_pipeline,
            setup_bgl,
            upsweep_pipeline,
            upsweep_bgl,
            scan_pipeline,
            scan_bgl,
            downsweep_pipeline,
            downsweep_bgl,
        }
    }
}

struct CachedPassBgs {
    upsweep: wgpu::BindGroup,
    scan: wgpu::BindGroup,
    downsweep: wgpu::BindGroup,
}

struct CachedBindGroups {
    keys_a: wgpu::Buffer,
    values_a: wgpu::Buffer,
    setup: wgpu::BindGroup,
    passes: Vec<CachedPassBgs>,
}

/// Buffers and bind groups for the radix sort.
pub struct RadixSortState {
    pub keys_b: wgpu::Buffer,
    pub values_b: wgpu::Buffer,
    pub global_hist: wgpu::Buffer,
    pub pass_hist: wgpu::Buffer,
    pub num_keys_buf: wgpu::Buffer,
    pub dispatch_args: wgpu::Buffer,
    pub config_buffers: Vec<wgpu::Buffer>,
    pub max_thread_blocks: u32,
    pub sorting_bits: u32,
    pub key_stride: u32,
    cached_bgs: Mutex<Option<CachedBindGroups>>,
}

impl RadixSortState {
    pub fn new(
        device: &wgpu::Device,
        sort_buf_size: u32,
        sorting_bits: u32,
        key_stride: u32,
    ) -> Self {
        let max_thread_blocks = sort_buf_size.div_ceil(PART_SIZE);
        let num_passes = sorting_bits.div_ceil(8);

        let keys_b = create_storage_buffer(
            device,
            "radix_keys_b",
            (sort_buf_size as u64) * (key_stride as u64) * 4,
        );
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
        let dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix_dispatch_args"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

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
            dispatch_args,
            config_buffers,
            max_thread_blocks,
            sorting_bits,
            key_stride,
            cached_bgs: Mutex::new(None),
        }
    }

    /// Write per-pass config to GPU buffers. Call once at init.
    /// numKeys is set to 0 here and updated per-frame via queue.write_buffer.
    pub fn upload_configs(&self, queue: &wgpu::Queue) {
        let num_passes = self.sorting_bits.div_ceil(8);
        for pass in 0..num_passes {
            let config = [0u32, pass * 8, self.max_thread_blocks, 0u32];
            queue.write_buffer(
                &self.config_buffers[pass as usize],
                0,
                bytemuck::cast_slice(&config),
            );
        }
    }
}

fn build_cached_bgs(
    device: &wgpu::Device,
    pipelines: &RadixSortPipelines,
    state: &RadixSortState,
    keys_a: &wgpu::Buffer,
    values_a: &wgpu::Buffer,
) -> CachedBindGroups {
    let num_passes = state.sorting_bits.div_ceil(8);
    let setup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("drs_setup_bg"),
        layout: &pipelines.setup_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.num_keys_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: state.dispatch_args.as_entire_binding(),
            },
        ],
    });
    let mut passes = Vec::with_capacity(num_passes as usize);
    for pass in 0..num_passes {
        let even = pass % 2 == 0;
        let (src_keys, src_vals, dst_keys, dst_vals) = if even {
            (keys_a, values_a, &state.keys_b, &state.values_b)
        } else {
            (&state.keys_b, &state.values_b, keys_a, values_a)
        };
        let config_buf = &state.config_buffers[pass as usize];
        let upsweep = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("drs_upsweep_bg"),
            layout: &pipelines.upsweep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.global_hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.pass_hist.as_entire_binding(),
                },
            ],
        });
        let scan = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("drs_scan_bg"),
            layout: &pipelines.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.pass_hist.as_entire_binding(),
                },
            ],
        });
        let downsweep = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("drs_downsweep_bg"),
            layout: &pipelines.downsweep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: src_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: state.global_hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state.pass_hist.as_entire_binding(),
                },
            ],
        });
        passes.push(CachedPassBgs {
            upsweep,
            scan,
            downsweep,
        });
    }
    CachedBindGroups {
        keys_a: keys_a.clone(),
        values_a: values_a.clone(),
        setup,
        passes,
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
    let num_passes = state.sorting_bits.div_ceil(8);

    // Copy numKeys from num_keys_buf into the first u32 of each config buffer.
    for config_buf in &state.config_buffers {
        encoder.copy_buffer_to_buffer(&state.num_keys_buf, 0, config_buf, 0, 4);
    }

    // Only global_hist needs clearing (upsweep atomicAdds into it); pass_hist is
    // overwritten across [0, effectiveTB) by upsweep and scanned over [0, effectiveTB).
    encoder.clear_buffer(&state.global_hist, 0, None);

    let mut guard = state.cached_bgs.lock().unwrap();
    let needs_rebuild = match guard.as_ref() {
        Some(c) => c.keys_a != *keys_a || c.values_a != *values_a,
        None => true,
    };
    if needs_rebuild {
        *guard = Some(build_cached_bgs(device, pipelines, state, keys_a, values_a));
    }
    let bgs = guard.as_ref().unwrap();

    // Record setup + all passes inside one ComputePass (avoids 12 Metal
    // command-encoder transitions; wgpu inserts storage barriers per dispatch).
    {
        let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("drs_sort"),
            timestamp_writes: None,
        });
        p.set_pipeline(&pipelines.setup_pipeline);
        p.set_bind_group(0, &bgs.setup, &[]);
        p.dispatch_workgroups(1, 1, 1);

        for pass_bgs in &bgs.passes {
            p.set_pipeline(&pipelines.upsweep_pipeline);
            p.set_bind_group(0, &pass_bgs.upsweep, &[]);
            p.dispatch_workgroups_indirect(&state.dispatch_args, 0);

            p.set_pipeline(&pipelines.scan_pipeline);
            p.set_bind_group(0, &pass_bgs.scan, &[]);
            p.dispatch_workgroups(RADIX, 1, 1);

            p.set_pipeline(&pipelines.downsweep_pipeline);
            p.set_bind_group(0, &pass_bgs.downsweep, &[]);
            p.dispatch_workgroups_indirect(&state.dispatch_args, 0);
        }
    }

    // Return true if result ended up in B buffers (odd number of passes)
    !num_passes.is_multiple_of(2)
}
