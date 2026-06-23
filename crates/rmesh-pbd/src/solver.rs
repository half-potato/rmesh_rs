//! GPU PBD solver: takes an [`Island`] + coloring, builds GPU pipelines and
//! per-color bind groups, drives one solver iteration as a sequence of
//! compute dispatches.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::coloring::ConstraintColoring;
use crate::island::Island;

static APPLY_HANDLES_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("shaders/apply_handles.wgsl");
static PREDICT_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("shaders/predict.wgsl");
static SOLVE_DISTANCE_WGSL: rmesh_util::HotShader =
    rmesh_util::hot_shader!("shaders/solve_distance.wgsl");
static FINALIZE_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("shaders/finalize.wgsl");

/// GPU layout for one particle. Matches the `Particle` WGSL struct: three
/// vec4<f32>s, 48 bytes total, std430-natural.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuParticle {
    /// xyz = position, w = inv_mass
    position: [f32; 4],
    /// xyz = predicted position (overwritten by predict / solve_distance), w = pad
    predicted: [f32; 4],
    /// xyz = velocity, w = pad
    velocity: [f32; 4],
}

/// GPU layout for one distance constraint. 16 bytes, std430-natural.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuDistanceConstraint {
    p1: u32,
    p2: u32,
    rest_length: f32,
    alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PbdUniforms {
    dt: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// All compute pipelines and bind-group layouts shared across grabs. Cache
/// once per device.
pub struct PbdPipelines {
    apply_handles_layout: wgpu::BindGroupLayout,
    apply_handles_pipeline: wgpu::ComputePipeline,
    predict_layout: wgpu::BindGroupLayout,
    predict_pipeline: wgpu::ComputePipeline,
    solve_layout: wgpu::BindGroupLayout,
    solve_pipeline: wgpu::ComputePipeline,
    finalize_layout: wgpu::BindGroupLayout,
    finalize_pipeline: wgpu::ComputePipeline,
}

impl PbdPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let apply_handles_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbd_apply_handles"),
            source: wgpu::ShaderSource::Wgsl(APPLY_HANDLES_WGSL.as_str().into()),
        });
        let predict_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbd_predict"),
            source: wgpu::ShaderSource::Wgsl(PREDICT_WGSL.as_str().into()),
        });
        let solve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbd_solve_distance"),
            source: wgpu::ShaderSource::Wgsl(SOLVE_DISTANCE_WGSL.as_str().into()),
        });
        let finalize_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbd_finalize"),
            source: wgpu::ShaderSource::Wgsl(FINALIZE_WGSL.as_str().into()),
        });

        let storage_rw_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let apply_handles_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pbd_apply_handles_bgl"),
                entries: &[
                    storage_rw_entry(0),
                    storage_ro_entry(1),
                    storage_ro_entry(2),
                ],
            });
        let predict_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbd_predict_bgl"),
            entries: &[storage_rw_entry(0), uniform_entry(1)],
        });
        let solve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbd_solve_bgl"),
            entries: &[storage_rw_entry(0), storage_ro_entry(1), uniform_entry(2)],
        });
        let finalize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbd_finalize_bgl"),
            entries: &[
                storage_rw_entry(0),
                storage_ro_entry(1),
                storage_rw_entry(2),
                uniform_entry(3),
            ],
        });

        let mk_pipeline = |label, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            apply_handles_pipeline: mk_pipeline(
                "pbd_apply_handles_pl",
                &apply_handles_module,
                &apply_handles_layout,
            ),
            apply_handles_layout,
            predict_pipeline: mk_pipeline("pbd_predict_pl", &predict_module, &predict_layout),
            predict_layout,
            solve_pipeline: mk_pipeline("pbd_solve_pl", &solve_module, &solve_layout),
            solve_layout,
            finalize_pipeline: mk_pipeline("pbd_finalize_pl", &finalize_module, &finalize_layout),
            finalize_layout,
        }
    }
}

/// One grab's worth of GPU state: particle/constraint buffers + per-color
/// bind groups. Created in [`PbdSolver::init_grab`], lives until the grab
/// is released.
pub struct PbdSolver {
    pub solver_iterations: u32,
    pub num_particles: u32,
    pub num_handles: u32,

    // Held to keep the buffers alive for the lifetime of the bind groups.
    #[allow(dead_code)]
    particles_buf: wgpu::Buffer,
    #[allow(dead_code)]
    constraints_buf: wgpu::Buffer,
    #[allow(dead_code)]
    global_indices_buf: wgpu::Buffer,
    #[allow(dead_code)]
    handle_local_indices_buf: wgpu::Buffer,
    handle_positions_buf: wgpu::Buffer,
    uniforms_buf: wgpu::Buffer,

    apply_handles_bg: wgpu::BindGroup,
    predict_bg: wgpu::BindGroup,
    /// One bind group per coloring batch, each pointing at its slice of
    /// `constraints_buf`. We dispatch them in order, repeated per iteration.
    solve_bgs: Vec<(wgpu::BindGroup, u32)>, // (bind group, constraint count)
    finalize_bg: wgpu::BindGroup,
}

impl PbdSolver {
    pub fn init_grab(
        device: &wgpu::Device,
        pipelines: &PbdPipelines,
        scene_vertices_buf: &wgpu::Buffer,
        island: &Island,
        coloring: &ConstraintColoring,
        solver_iterations: u32,
    ) -> Self {
        let num_particles = island.particles.len() as u32;
        let num_handles = island.handle_local_indices.len() as u32;

        let gpu_particles: Vec<GpuParticle> = island
            .particles
            .iter()
            .map(|p| GpuParticle {
                position: [p.position[0], p.position[1], p.position[2], p.inv_mass],
                predicted: [p.position[0], p.position[1], p.position[2], 0.0],
                velocity: [0.0, 0.0, 0.0, 0.0],
            })
            .collect();
        let particles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbd_particles"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Concatenated constraints across all colors. We need each color's
        // buffer-binding offset to satisfy `min_storage_buffer_offset_alignment`
        // (typically 256 bytes) so we pad between colors. The 16-byte stride
        // means alignment_in_constraints = 256 / 16 = 16. We track each
        // color's *padded* start so the bind groups can use real offsets.
        let stride = std::mem::size_of::<GpuDistanceConstraint>();
        let align_bytes = device.limits().min_storage_buffer_offset_alignment as usize;
        assert!(
            align_bytes % stride == 0,
            "constraint stride {stride} must divide storage alignment {align_bytes}"
        );
        let align_in_constraints = align_bytes / stride;

        let mut padded: Vec<GpuDistanceConstraint> = Vec::new();
        let mut padded_offsets: Vec<u32> = Vec::with_capacity(coloring.num_colors() + 1);
        padded_offsets.push(0);
        for c in 0..coloring.num_colors() {
            let start = coloring.color_offsets[c] as usize;
            let end = coloring.color_offsets[c + 1] as usize;
            for src in &coloring.constraints[start..end] {
                padded.push(GpuDistanceConstraint {
                    p1: src.p1_local,
                    p2: src.p2_local,
                    rest_length: src.rest_length,
                    alpha: src.alpha,
                });
            }
            // Pad up to alignment so the *next* color's offset is aligned.
            let rem = padded.len() % align_in_constraints;
            if rem != 0 {
                let pad = align_in_constraints - rem;
                // Self-edges (p1=p2): the shader's `len < 1e-6` early-out skips
                // them. Never dispatched anyway — bind size excludes padding —
                // but defensive.
                padded.extend(std::iter::repeat_n(
                    GpuDistanceConstraint {
                        p1: 0,
                        p2: 0,
                        rest_length: 0.0,
                        alpha: 0.0,
                    },
                    pad,
                ));
            }
            padded_offsets.push(padded.len() as u32);
        }

        // wgpu refuses zero-sized buffers — pad to at least one constraint.
        let constraints_bytes: Vec<u8> = if padded.is_empty() {
            vec![0u8; stride]
        } else {
            bytemuck::cast_slice(&padded).to_vec()
        };
        let constraints_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbd_constraints"),
            contents: &constraints_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let global_indices: Vec<u32> = island.particles.iter().map(|p| p.global_index).collect();
        let global_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbd_global_indices"),
            contents: bytemuck::cast_slice(&global_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let handle_indices_bytes: Vec<u8> = if island.handle_local_indices.is_empty() {
            vec![0u8; 4]
        } else {
            bytemuck::cast_slice(&island.handle_local_indices).to_vec()
        };
        let handle_local_indices_buf =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbd_handle_local_indices"),
                contents: &handle_indices_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Initial handle positions = handle particles' current positions.
        let initial_handle_positions: Vec<[f32; 4]> = island
            .handle_local_indices
            .iter()
            .map(|&li| {
                let p = &island.particles[li as usize];
                [p.position[0], p.position[1], p.position[2], 0.0]
            })
            .collect();
        let handle_positions_bytes: Vec<u8> = if initial_handle_positions.is_empty() {
            vec![0u8; 16]
        } else {
            bytemuck::cast_slice(&initial_handle_positions).to_vec()
        };
        let handle_positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbd_handle_positions"),
            contents: &handle_positions_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbd_uniforms"),
            size: std::mem::size_of::<PbdUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let apply_handles_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbd_apply_handles_bg"),
            layout: &pipelines.apply_handles_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: handle_local_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: handle_positions_buf.as_entire_binding(),
                },
            ],
        });

        let predict_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbd_predict_bg"),
            layout: &pipelines.predict_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniforms_buf.as_entire_binding(),
                },
            ],
        });

        let finalize_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbd_finalize_bg"),
            layout: &pipelines.finalize_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: global_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_vertices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms_buf.as_entire_binding(),
                },
            ],
        });

        // Per-color solve bind groups: each binds a slice of constraints_buf
        // covering only that color's real (unpadded) constraints, with the
        // padded offset that satisfies storage-buffer alignment.
        let stride_u64 = stride as u64;
        let mut solve_bgs = Vec::with_capacity(coloring.num_colors());
        for c in 0..coloring.num_colors() {
            let count = coloring.color_len(c);
            if count == 0 {
                continue;
            }
            let offset = padded_offsets[c] as u64 * stride_u64;
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pbd_solve_color_bg"),
                layout: &pipelines.solve_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particles_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &constraints_buf,
                            offset,
                            size: wgpu::BufferSize::new(count as u64 * stride_u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniforms_buf.as_entire_binding(),
                    },
                ],
            });
            solve_bgs.push((bg, count));
        }

        Self {
            solver_iterations,
            num_particles,
            num_handles,
            particles_buf,
            constraints_buf,
            global_indices_buf,
            handle_local_indices_buf,
            handle_positions_buf,
            uniforms_buf,
            apply_handles_bg,
            predict_bg,
            solve_bgs,
            finalize_bg,
        }
    }

    /// Number of bytes a step expects to find in the `handle_positions` slice.
    pub fn handle_positions_byte_size(&self) -> u64 {
        (self.num_handles.max(1) as u64) * 16
    }

    /// Upload new handle positions (world-space, one vec4 per handle, w=0) and
    /// record one PBD step into `encoder`. Caller is responsible for
    /// submitting and for ordering any read of `scene_vertices_buf` after
    /// this encoder.
    pub fn step(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &PbdPipelines,
        dt: f32,
        handle_positions: &[[f32; 4]],
    ) {
        if self.num_particles == 0 {
            return;
        }
        let uniforms = PbdUniforms {
            dt,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

        if !handle_positions.is_empty() {
            queue.write_buffer(
                &self.handle_positions_buf,
                0,
                bytemuck::cast_slice(handle_positions),
            );
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pbd_step"),
            timestamp_writes: None,
        });

        // 1. Apply handle positions (kinematic write into particles).
        if self.num_handles > 0 {
            pass.set_pipeline(&pipelines.apply_handles_pipeline);
            pass.set_bind_group(0, &self.apply_handles_bg, &[]);
            pass.dispatch_workgroups(self.num_handles.div_ceil(64), 1, 1);
        }

        // 2. Predict.
        pass.set_pipeline(&pipelines.predict_pipeline);
        pass.set_bind_group(0, &self.predict_bg, &[]);
        pass.dispatch_workgroups(self.num_particles.div_ceil(64), 1, 1);

        // 3. Solver: iterate Gauss-Seidel by color.
        pass.set_pipeline(&pipelines.solve_pipeline);
        for _iter in 0..self.solver_iterations {
            for (bg, count) in &self.solve_bgs {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            }
        }

        // 4. Finalize + scatter into the global vertices buffer.
        pass.set_pipeline(&pipelines.finalize_pipeline);
        pass.set_bind_group(0, &self.finalize_bg, &[]);
        pass.dispatch_workgroups(self.num_particles.div_ceil(64), 1, 1);
    }
}
