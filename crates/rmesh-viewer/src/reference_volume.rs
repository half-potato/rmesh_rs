//! Full-precision, per-ray reference lighting. Never substitutes a coarse bake.
use anyhow::{Context, Result, ensure};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
    samples: u32,
    limit: u32,
    mode: u32,
    eps: f32,
    near: f32,
    offset: u32,
    pad: u32,
    light: [f32; 4],
    color: [f32; 4],
}

pub struct ReferenceVolume {
    pipeline: wgpu::ComputePipeline,
    geometry: wgpu::BindGroup,
    io_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    pub mode: u32,
    eps: f32,
}

impl ReferenceVolume {
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        ensure!(
            data.len() >= 24 && &data[..8] == b"RMLIT001",
            "Not an RMLIT001 reference scene"
        );
        let word = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        let nv = word(8) as usize;
        let nt = word(12) as usize;
        let mode = word(16);
        let eps = f32::from_bits(word(20));
        ensure!(
            nv > 0 && nt > 0 && (mode == 0 || mode == 3 || mode == 4) && eps.is_finite() && eps > 0.0,
            "Invalid reference scene header"
        );
        ensure!(
            data.len() == 24 + nv * 24 + nt * 56,
            "Reference scene size mismatch"
        );
        let mut cursor = 24;
        let mut take = |len: usize| {
            let a = &data[cursor..cursor + len];
            cursor += len;
            a
        };
        let vertices = take(nv * 12);
        let normals = take(nv * 12);
        let indices = take(nt * 16);
        let density = take(nt * 4);
        let materials = take(nt * 36);
        let verts: Vec<f32> = vertices
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let idx: Vec<u32> = indices
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            idx.iter().all(|&i| (i as usize) < nv),
            "Invalid reference vertex index"
        );
        ensure!(verts.iter().all(|v| v.is_finite()), "Nonfinite geometry");
        let neighbors = rmesh_raytrace::compute_tet_neighbors(&idx, nt);
        // Reuse the BVH builder with one representative of EVERY face,
        // including interior faces, for sdir's recovery/re-entry traversal.
        let selection: Vec<i32> = neighbors
            .iter()
            .enumerate()
            .map(|(i, &n)| if n < 0 || i / 4 < n as usize { -1 } else { 0 })
            .collect();
        log::info!("Building full reference face BVH for {nt} cells");
        let bvh = rmesh_raytrace::build_boundary_bvh(&verts, &idx, &selection, nt);
        ensure!(!bvh.nodes.is_empty(), "Empty reference BVH");
        let upload = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let buffers = [
            upload("ref_vertices", vertices),
            upload("ref_indices", indices),
            upload("ref_density", density),
            upload("ref_normals", normals),
            upload("ref_materials", materials),
            upload("ref_neighbors", bytemuck::cast_slice(&neighbors)),
            upload("ref_bvh", bytemuck::cast_slice(&bvh.nodes)),
            upload("ref_faces", bytemuck::cast_slice(&bvh.boundary_faces)),
        ];
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference_params"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for binding in 1..=8 {
            entries.push(storage_entry(binding, true));
        }
        entries.push(rmesh_viewer_core::energy::layout_entry(9));
        let energy_lut = rmesh_viewer_core::energy::upload(device, queue);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reference_geometry"),
            entries: &entries,
        });
        let mut bindings = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }];
        for (i, b) in buffers.iter().enumerate() {
            bindings.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            });
        }
        bindings.push(wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&energy_lut) });
        let geometry = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reference_geometry"),
            layout: &layout,
            entries: &bindings,
        });
        let io_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reference_io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reference_volume"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", rmesh_viewer_core::energy::WGSL, include_str!("wgsl/reference_volume.wgsl")).into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reference_volume"),
            bind_group_layouts: &[&layout, &io_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reference_volume"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            geometry,
            io_layout,
            params,
            mode,
            eps,
        })
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rays: &[[f32; 6]],
        light: [f32; 3],
        color: [f32; 3],
        near: f32,
        samples: u32,
    ) -> Result<Vec<[f32; 4]>> {
        ensure!(
            !rays.is_empty() && samples > 0 && samples <= 64,
            "Invalid ray count or quadrature count"
        );
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reference_rays"),
            contents: bytemuck::cast_slice(rays),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference_output"),
            size: rays.len() as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let failures = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference_failures"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reference_io"),
            layout: &self.io_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ray_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: failures.as_entire_binding(),
                },
            ],
        });
        // Separate submissions make each uniform update apply to its own batch.
        for offset in (0..rays.len()).step_by(1024) {
            let end = (offset + 1024).min(rays.len());
            let p = Params {
                count: end as u32,
                samples,
                limit: 4096,
                mode: self.mode,
                eps: self.eps,
                near,
                offset: offset as u32,
                pad: 0,
                light: [light[0], light[1], light[2], 0.],
                color: [color[0], color[1], color[2], 0.],
            };
            queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.geometry, &[]);
                pass.set_bind_group(1, &bg, &[]);
                pass.dispatch_workgroups(((end - offset) as u32).div_ceil(64), 1, 1);
            }
            queue.submit(Some(enc.finish()));
        }
        let errors = readback(device, queue, &failures, 12)?;
        let counts: Vec<u32> = errors
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            counts.iter().all(|&v| v == 0),
            "Incomplete reference traversal (BVH/shadow/camera): {counts:?}"
        );
        let raw = readback(device, queue, &output, rays.len() as u64 * 16)?;
        let pixels: Vec<[f32; 4]> = raw
            .chunks_exact(16)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        ensure!(
            pixels.iter().flatten().all(|v| v.is_finite()),
            "Nonfinite reference output"
        );
        Ok(pixels)
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>> {
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reference_readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &dst, 0, size);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    dst.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = dst.slice(..).get_mapped_range().to_vec();
    dst.unmap();
    Ok(bytes)
}

pub fn run_cli(args: &[String]) -> Result<()> {
    ensure!(
        args.len() >= 5,
        "Usage: rmesh-viewer --reference-volume scene.rmlit rays.bin output.rgba [light_x light_y light_z intensity samples near red green blue]"
    );
    let number = |i: usize, default: f32| -> Result<f32> {
        args.get(i)
            .map(|v| v.parse().context("Invalid numeric reference argument"))
            .unwrap_or(Ok(default))
    };
    let light = [
        number(5, 0.04947496)?,
        number(6, -1.5245742)?,
        number(7, -1.0272096)?,
    ];
    let intensity = number(8, 5.8619976)?;
    let sample_count = number(9, 4.)?;
    ensure!(
        sample_count.is_finite()
            && sample_count.fract() == 0.0
            && (1.0..=64.0).contains(&sample_count),
        "Samples must be an integer in 1..=64"
    );
    let samples = sample_count as u32;
    let near = number(10, 0.1)?;
    let color = [
        number(11, 1.0)? * intensity,
        number(12, 0.88)? * intensity,
        number(13, 0.72)? * intensity,
    ];
    ensure!(
        light.iter().chain(color.iter()).all(|v| v.is_finite())
            && near.is_finite()
            && near >= 0.0
            && intensity.is_finite()
            && intensity >= 0.0,
        "Invalid light or near plane"
    );
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        log::info!("Reference GPU: {:?}", adapter.get_info());
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 12,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .map_err(anyhow::Error::from)
    })?;
    let scene = ReferenceVolume::load(&device, &queue, std::path::Path::new(&args[2]))?;
    let bytes = std::fs::read(&args[3])?;
    ensure!(bytes.len() % 24 == 0, "Invalid rays file");
    let rays: Vec<[f32; 6]> = bytes
        .chunks_exact(24)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let result = scene.render(&device, &queue, &rays, light, color, near, samples)?;
    std::fs::write(&args[4], bytemuck::cast_slice(&result))?;
    log::info!(
        "Reference render wrote {} rays to {}",
        result.len(),
        args[4]
    );
    Ok(())
}
