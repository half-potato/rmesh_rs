//! GPU SH evaluation compute pipeline.
//!
//! Evaluates spherical harmonics per-tet on the GPU, replacing the CPU
//! `evaluate_sh_colors()` path. One compute dispatch per frame with the
//! current camera position.

use bytemuck::{Pod, Zeroable};

const SH_EVAL_WGSL: &str = include_str!("wgsl/sh_eval_compute.wgsl");

/// Uniforms for the SH evaluation compute shader (32 bytes, std140-compatible).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShEvalUniforms {
    pub cam_pos: [f32; 4],
    pub tet_count: u32,
    pub sh_degree: u32,
    pub _pad: [u32; 2],
}

/// Reusable compute pipeline + bind group layout for SH evaluation.
pub struct ShEvalPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout: wgpu::BindGroupLayout,
}

impl ShEvalPipeline {
    /// Create the SH eval compute pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_module = crate::compose::create_shader_module(
            device,
            "sh_eval_compute.wgsl",
            SH_EVAL_WGSL,
        )
        .expect("Failed to compose sh_eval_compute shader");

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sh_eval bg_layout"),
            entries: &[
                // @binding(0) uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) vertices (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) indices (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3) sh_coeffs (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4) color_grads (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(5) base_colors (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sh_eval pipeline_layout"),
            bind_group_layouts: &[&bg_layout],
            ..Default::default()
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sh_eval pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bg_layout,
        }
    }

    /// Create a bind group for the SH eval pipeline.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        uniforms_buf: &wgpu::Buffer,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        sh_coeffs: &wgpu::Buffer,
        color_grads: &wgpu::Buffer,
        base_colors: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sh_eval bind_group"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sh_coeffs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: color_grads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: base_colors.as_entire_binding(),
                },
            ],
        })
    }

    /// Record a compute dispatch for SH evaluation.
    ///
    /// Uses a 2D dispatch grid to stay within the 65535 workgroup-per-dimension
    /// limit (4.2M tets @ 256 threads/group = ~16k groups, but we handle up to
    /// 256 * 65535 * 65535 ≈ 1.1 trillion tets).
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        tet_count: u32,
        profiler: Option<&wgpu::QuerySet>,
    ) {
        const WG_SIZE: u32 = 256;
        const MAX_DIM: u32 = 65535;

        let total_wgs = (tet_count + WG_SIZE - 1) / WG_SIZE;
        let wg_x = total_wgs.min(MAX_DIM);
        let wg_y = (total_wgs + wg_x - 1) / wg_x;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sh_eval compute"),
            timestamp_writes: profiler.map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(6),
                end_of_pass_write_index: Some(7),
            }),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
