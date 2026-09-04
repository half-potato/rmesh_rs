//! Density shell: the boundary surface of the density-thresholded region,
//! drawn as opaque geometry before the volume.
//!
//! The shell exists to answer one question cheaply: *how far can light get
//! before it runs out?* Its back faces are where a ray leaves the dense region,
//! so the depth written here bounds transmission. Two consumers use that bound:
//!
//! * the interval render pass, which already depth-tests `LessEqual` against
//!   this buffer — so occluded fragments die with no shader change at all;
//! * [`crate::shell_hiz`], which reduces it to a max-pyramid so
//!   `project_compute*` can drop whole tets *before* the radix sort.
//!
//! It also writes color into the same MRT G-buffer the volume blends over, so
//! residual transmittance reveals the model's own surface instead of the
//! background. That is what keeps a high density threshold from washing the
//! image out.
//!
//! Face ids arrive packed as `(tet_id << 2) | face_idx` from
//! `rmesh_viewer::flare::extract_boundary_faces`; positions and normals are
//! fetched live from the scene buffers, so the shell tracks animation and PBD
//! edits without re-extraction.

use wgpu::util::DeviceExt;

static SHELL_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/shell.wgsl");

/// Per-pass view data. Separate from the shared [`rmesh_util::shared::Uniforms`]
/// so the DSM can draw the same shell through a cube-face VP with the light as
/// the eye.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShellView {
    /// Column-major view-projection matrix.
    pub vp: [[f32; 4]; 4],
    /// xyz = eye (camera or light) position. Back-facing is decided against
    /// this in world space, so no NDC winding convention is involved.
    pub eye: [f32; 4],
    /// x = face count, y = near plane, z = far plane, w = face orientation
    /// (0 = exit/back faces, 1 = entry/front faces).
    pub params: [f32; 4],
}

pub const SHELL_VIEW_SIZE: u64 = std::mem::size_of::<ShellView>() as u64;

/// GPU-side shell: the packed face-id buffer plus its count.
///
/// The buffer is grown on demand and never shrunk — thresholds get dragged back
/// and forth, and a realloc per drag step is worse than holding the high-water
/// mark.
/// Which side of the dense region the shell draws. See the header comment in
/// `shell.wgsl` for why the choice is a correctness/speed trade rather than a
/// preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ShellFace {
    /// Back faces: where a ray leaves the dense region. The volume in front of
    /// the shell still renders in full, so the tets that actually accumulate the
    /// image are never culled — the shell only backs them. The default.
    #[default]
    Exit,
    /// Front faces: where a ray enters. Culls the region's own interior, which
    /// means culling the tets whose accumulated radiance *is* the image and
    /// replacing them with one flat surface.
    ///
    /// Measured, this is not a free win: at a fixed threshold and erosion it is
    /// faster and worse (on `stump_conserv`, 67fps/29.5dB against exit's
    /// 60fps/32.1dB), and once quality is matched the two sit on the same
    /// speed/quality frontier. Erosion, not face orientation, is what makes the
    /// shell safe. Kept because a genuinely opaque object gets there cheaper
    /// this way; prefer `Exit` unless measurement says otherwise.
    Entry,
}

pub struct ShellBuffers {
    pub faces: wgpu::Buffer,
    /// Faces actually live in `faces` (may be less than its capacity).
    pub face_count: u32,
    capacity: u32,
    pub view_buf: wgpu::Buffer,
    /// Written into `params.w` by both view writers, so the camera pass and the
    /// DSM cube faces cannot disagree about which surface the shell is.
    pub face: ShellFace,
}

impl ShellBuffers {
    pub fn new(device: &wgpu::Device) -> Self {
        // One dummy face so the binding is never zero-sized (WebGPU rejects
        // zero-sized storage bindings) and `arrayLength` stays well-defined.
        let faces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shell_faces"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shell_view"),
            size: SHELL_VIEW_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            faces,
            face_count: 0,
            capacity: 1,
            view_buf,
            face: ShellFace::default(),
        }
    }

    /// Upload a freshly extracted face list.
    ///
    /// Returns `true` when the underlying buffer was reallocated, meaning every
    /// bind group holding it must be rebuilt.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, faces: &[u32]) -> bool {
        self.face_count = faces.len() as u32;
        if faces.is_empty() {
            return false;
        }

        let mut realloc = false;
        if faces.len() as u32 > self.capacity {
            self.faces = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shell_faces"),
                size: (faces.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = faces.len() as u32;
            realloc = true;
        }
        queue.write_buffer(&self.faces, 0, bytemuck::cast_slice(faces));
        realloc
    }

    /// Write the per-pass view. `face_count` is baked in so the vertex shader
    /// can drop trailing triangles when the draw is over-sized.
    pub fn write_view(&self, queue: &wgpu::Queue, vp: [[f32; 4]; 4], eye: [f32; 3]) {
        self.write_view_with_planes(queue, vp, eye, 0.0, 0.0);
    }

    /// As [`write_view`](Self::write_view), plus the near/far planes the DSM
    /// entry point needs to normalize depth into moment space.
    pub fn write_view_with_planes(
        &self,
        queue: &wgpu::Queue,
        vp: [[f32; 4]; 4],
        eye: [f32; 3],
        near: f32,
        far: f32,
    ) {
        let view = ShellView {
            vp,
            eye: [eye[0], eye[1], eye[2], 0.0],
            params: [
                self.face_count as f32,
                near,
                far,
                match self.face {
                    ShellFace::Exit => 0.0,
                    ShellFace::Entry => 1.0,
                },
            ],
        };
        queue.write_buffer(&self.view_buf, 0, bytemuck::bytes_of(&view));
    }

    /// Vertex count for the draw (3 per boundary face).
    #[inline]
    pub fn vertex_count(&self) -> u32 {
        self.face_count * 3
    }
}

pub struct ShellPipeline {
    pub mrt_pipeline: wgpu::RenderPipeline,
    /// Writes depth plus an alpha=1 absorber moment, for the DSM cube faces.
    pub dsm_pipeline: wgpu::RenderPipeline,
    pub bg_layout: wgpu::BindGroupLayout,
}

impl ShellPipeline {
    /// `formats` drives the four G-buffer MRT attachments; `dsm_format` drives
    /// the separate DSM moment target. These were a single `color_format`
    /// parameter, which meant narrowing the G-buffer silently narrowed DSM
    /// moments too — they are unrelated targets and must be named separately.
    pub fn new(
        device: &wgpu::Device,
        formats: rmesh_util::shared::GBufferFormats,
        dsm_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shell.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHELL_WGSL.as_str().into()),
        });

        // binding 0 = uniform (ShellView), 1..=9 = read-only storage.
        // That is 9 storage buffers per stage, inside WebGPU's cap of 10.
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for binding in 1..=9u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shell_bg_layout"),
            entries: &entries,
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shell_pipeline_layout"),
            bind_group_layouts: &[&bg_layout],
            immediate_size: 0,
        });

        // Facing is decided in world space in the vertex shader (unwanted faces
        // collapse to a point), so the rasterizer must not second-guess it.
        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        };

        // Push the shell very slightly away from the eye. Its own faces are the
        // exit plane of the tets sitting right in front of it, and those must
        // still pass the interval pass's LessEqual test.
        let depth_stencil = wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 16,
                slope_scale: 1.0,
                clamp: 0.0,
            },
        };

        let mrt_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shell_mrt_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive,
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    // Opaque: replace, don't blend. The volume blends over this.
                    Some(wgpu::ColorTargetState {
                        format: formats.color,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: formats.aux0,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: formats.normals,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: formats.depth,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // DSM cube faces: same geometry, but the target is the moment atlas and
        // the blend must match dsm_primitive / dsm_moment_fragment so the shell
        // composites into the same alpha-weighted moment stack.
        let premul_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let dsm_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shell_dsm_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 16,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_dsm_moment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: dsm_format,
                    blend: Some(premul_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            mrt_pipeline,
            dsm_pipeline,
            bg_layout,
        }
    }
}

/// Bind group for the shell pass.
///
/// Bindings: 0 = shell view (uniform), 1 = uniforms, 2 = vertices, 3 = indices,
/// 4 = shell faces, 5 = vertex normals, 6 = sh coeffs, 7 = color grads,
/// 8 = aux data, 9 = base colors.
#[allow(clippy::too_many_arguments)]
pub fn create_shell_bind_group(
    device: &wgpu::Device,
    pipeline: &ShellPipeline,
    shell: &ShellBuffers,
    buffers: &rmesh_data::SceneBuffers,
    material: &rmesh_data::MaterialBuffers,
    sh_coeffs: &wgpu::Buffer,
    aux_data: &wgpu::Buffer,
) -> wgpu::BindGroup {
    use rmesh_util::gpu_helpers::buf_entry;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shell_bg"),
        layout: &pipeline.bg_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: shell.view_buf.as_entire_binding(),
            },
            buf_entry(1, &buffers.uniforms),
            buf_entry(2, &buffers.vertices),
            buf_entry(3, &buffers.indices),
            buf_entry(4, &shell.faces),
            buf_entry(5, &buffers.vertex_normals),
            buf_entry(6, sh_coeffs),
            buf_entry(7, &material.color_grads),
            buf_entry(8, aux_data),
            buf_entry(9, &material.base_colors),
        ],
    })
}

/// Draw the shell into an existing G-buffer + depth pair.
///
/// Attachments are loaded, not cleared — the primitive pass runs first and this
/// composites into the same targets. Must be recorded *before* the volume pass,
/// since that pass depth-tests against what this writes.
#[allow(clippy::too_many_arguments)]
pub fn record_shell_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ShellPipeline,
    bind_group: &wgpu::BindGroup,
    shell: &ShellBuffers,
    color_view: &wgpu::TextureView,
    aux0_view: &wgpu::TextureView,
    normals_view: &wgpu::TextureView,
    depth_albedo_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
) {
    if shell.face_count == 0 {
        return;
    }

    let load_store = wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    };
    fn attach(
        view: &wgpu::TextureView,
        ops: wgpu::Operations<wgpu::Color>,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops,
            depth_slice: None,
        })
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("shell_pass"),
        color_attachments: &[
            attach(color_view, load_store),
            attach(aux0_view, load_store),
            attach(normals_view, load_store),
            attach(depth_albedo_view, load_store),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    pass.set_pipeline(&pipeline.mrt_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..shell.vertex_count(), 0..1);
}

/// Draw the shell into one DSM cube face: depth plus an alpha=1 absorber moment.
///
/// Record right after `record_dsm_primitive_pass` for that face, so it lands on
/// top of the opaque primitives and before the tet intervals — which the depth
/// it writes then rejects.
pub fn record_shell_dsm_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ShellPipeline,
    bind_group: &wgpu::BindGroup,
    shell: &ShellBuffers,
    moment_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
) {
    if shell.face_count == 0 {
        return;
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("shell_dsm_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: moment_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    pass.set_pipeline(&pipeline.dsm_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..shell.vertex_count(), 0..1);
}
