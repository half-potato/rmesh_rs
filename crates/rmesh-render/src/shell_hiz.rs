//! Max-reduce Hi-Z pyramid over the shell (and opaque-primitive) depth buffer.
//!
//! `project_compute*` uses this to drop tets **before** the radix sort, which is
//! where the win is — the sort is the single largest slice of the interval
//! frame, and a fragment-level depth test does nothing to shrink it.
//!
//! Built with multi-mip compute passes (up to 4 mip levels per dispatch, staying
//! within WebGPU's 4-storage-texture limit) instead of one render pass per mip.
//!
//! The reduction is `max`, not `min`: culling is only safe when a tet's nearest
//! point is behind the *farthest* occluder anywhere in its screen footprint.
//! Cleared texels stay at 1.0 and max() to 1.0, so any footprint touching
//! background culls nothing.

use wgpu::util::DeviceExt;

static SHELL_HIZ_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/shell_hiz_max.wgsl");

pub struct ShellHiZPipelines {
    pub init_pipeline: wgpu::ComputePipeline,
    pub downsample_pipeline: wgpu::ComputePipeline,
    pub tail_pipeline: wgpu::ComputePipeline,
    pub init_bgl: wgpu::BindGroupLayout,
    pub downsample_bgl: wgpu::BindGroupLayout,
}

const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

fn storage_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HIZ_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

impl ShellHiZPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shell_hiz_max.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHELL_HIZ_WGSL.as_str().into()),
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shell_hiz_init_bgl"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_tex_entry(2),
                storage_tex_entry(3),
                storage_tex_entry(4),
                storage_tex_entry(5),
            ],
        });

        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shell_hiz_downsample_bgl"),
            entries: &[
                uniform_entry,
                storage_tex_entry(2),
                storage_tex_entry(3),
                storage_tex_entry(4),
                storage_tex_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let make = |label: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            })
        };

        Self {
            init_pipeline: make("shell_hiz_init", &init_bgl, "cs_init"),
            downsample_pipeline: make("shell_hiz_downsample", &downsample_bgl, "cs_downsample"),
            tail_pipeline: make("shell_hiz_tail", &downsample_bgl, "cs_tail"),
            init_bgl,
            downsample_bgl,
        }
    }
}

#[derive(Clone, Copy)]
enum HiZPassKind {
    Init,
    Downsample,
    Tail,
}

struct HiZPass {
    kind: HiZPassKind,
    bg: wgpu::BindGroup,
    dispatch: [u32; 3],
    _params: wgpu::Buffer,
}

/// The pyramid itself, plus the bind groups that build it.
///
/// Size-dependent — recreate on resize, and rebuild every bind group that holds
/// [`ShellHiZ::full_view`] along with it.
pub struct ShellHiZ {
    pub texture: wgpu::Texture,
    /// All-mips view. This is what `project_compute*` samples.
    pub full_view: wgpu::TextureView,
    _dummy_out: wgpu::Texture,
    passes: Vec<HiZPass>,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
}

fn mip_dim(base: u32, mip: u32) -> u32 {
    (base >> mip).max(1)
}

impl ShellHiZ {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &ShellHiZPipelines,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mip_count = 32 - width.max(height).leading_zeros();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shell_hiz"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shell_hiz_full"),
            ..Default::default()
        });

        let mip_views: Vec<_> = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shell_hiz_mip"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // 1x1 dummy storage texture for unused output slots when a pass writes < 4 mips.
        let dummy_out = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shell_hiz_dummy_out"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let dummy_out_view = dummy_out.create_view(&wgpu::TextureViewDescriptor::default());

        let make_params = |src_w: u32, src_h: u32, levels: u32| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shell_hiz_params"),
                contents: bytemuck::cast_slice(&[src_w, src_h, levels, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        let mut passes = Vec::new();

        // Pass 1: copy src_depth -> mip 0 and fuse up to 3 reductions while parent dims are even.
        let mut init_levels = 1u32;
        while init_levels < 4 && init_levels < mip_count {
            let pw = mip_dim(width, init_levels - 1);
            let ph = mip_dim(height, init_levels - 1);
            if pw % 2 == 0 && ph % 2 == 0 {
                init_levels += 1;
            } else {
                break;
            }
        }
        let out_view = |mip: u32, count: u32| -> &wgpu::TextureView {
            if mip < count {
                &mip_views[mip as usize]
            } else {
                &dummy_out_view
            }
        };
        {
            let params_buf = make_params(width, height, init_levels);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shell_hiz_init_bg"),
                layout: &pipelines.init_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(out_view(0, init_levels)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(out_view(1, init_levels)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(out_view(2, init_levels)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(out_view(3, init_levels)),
                    },
                ],
            });
            passes.push(HiZPass {
                kind: HiZPassKind::Init,
                bg,
                dispatch: [width.div_ceil(16), height.div_ceil(16), 1],
                _params: params_buf,
            });
        }

        // Subsequent passes: reduce from `cur - 1` into `cur..cur+levels`.
        let mut cur = init_levels;
        while cur < mip_count {
            let src_mip = cur - 1;
            let sw = mip_dim(width, src_mip);
            let sh = mip_dim(height, src_mip);
            let d0w = mip_dim(width, cur);
            let d0h = mip_dim(height, cur);
            let remaining = mip_count - cur;

            // Small mips fit in one 12 KB workgroup (tail_a[2048], tail_b[1024]),
            // handling arbitrary odd borders across up to 4 levels in 1 workgroup.
            let (kind, levels, dispatch) = if d0w * d0h <= 2048 && (d0w / 2).max(1) * (d0h / 2).max(1) <= 1024 {
                (HiZPassKind::Tail, remaining.min(4), [1, 1, 1])
            } else {
                let mut k = 1u32;
                if sw % 2 == 0 && sh % 2 == 0 {
                    while k < 4 && k < remaining {
                        let pw = mip_dim(width, cur + k - 1);
                        let ph = mip_dim(height, cur + k - 1);
                        if pw % 2 == 0 && ph % 2 == 0 {
                            k += 1;
                        } else {
                            break;
                        }
                    }
                }
                (
                    HiZPassKind::Downsample,
                    k,
                    [d0w.div_ceil(8), d0h.div_ceil(8), 1],
                )
            };

            let end_mip = cur + levels;
            let params_buf = make_params(sw, sh, levels);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shell_hiz_downsample_bg"),
                layout: &pipelines.downsample_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(out_view(cur, end_mip)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(out_view(cur + 1, end_mip)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(out_view(cur + 2, end_mip)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(out_view(cur + 3, end_mip)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&mip_views[src_mip as usize]),
                    },
                ],
            });
            passes.push(HiZPass {
                kind,
                bg,
                dispatch,
                _params: params_buf,
            });
            cur = end_mip;
        }

        Self {
            texture,
            full_view,
            _dummy_out: dummy_out,
            passes,
            width,
            height,
            mip_count,
        }
    }
}

/// A 1x1 pyramid holding the far plane, for paths that never build a real one
/// (tests, the training harness, the forward-only helper).
///
/// It reads 1.0, so `shell_occluded` finds no occluder and culls nothing even if
/// the enable flag is set. Note this must be written explicitly: a fresh texture
/// zero-initializes, and 0.0 is the *near* plane — it would cull everything.
pub fn dummy_shell_hiz(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shell_hiz_dummy"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HIZ_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::bytes_of(&1.0f32),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Build the pyramid. Record after the shell pass and before
/// `record_compute_interval_project`, so the cull uses this frame's bound.
pub fn record_shell_hiz(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &ShellHiZPipelines,
    hiz: &ShellHiZ,
) {
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("shell_hiz"),
        timestamp_writes: None,
    });
    for p in &hiz.passes {
        let pipeline = match p.kind {
            HiZPassKind::Init => &pipelines.init_pipeline,
            HiZPassKind::Downsample => &pipelines.downsample_pipeline,
            HiZPassKind::Tail => &pipelines.tail_pipeline,
        };
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &p.bg, &[]);
        cpass.dispatch_workgroups(p.dispatch[0], p.dispatch[1], p.dispatch[2]);
    }
}
