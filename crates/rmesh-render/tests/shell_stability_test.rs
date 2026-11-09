//! Exercise the real shell raster pipeline on a 4K offscreen attachment.
use rmesh_util::test_util::{create_test_device_default, read_buffer};
use wgpu::util::DeviceExt;

#[test]
fn shell_color_and_depth_are_stable_at_4k() {
    let Some((device, queue)) = create_test_device_default() else {
        return;
    };
    let formats = rmesh_render::GBufferFormats {
        color: wgpu::TextureFormat::Rgba8Unorm,
        ..Default::default()
    };
    let pipeline = rmesh_render::ShellPipeline::new(
        &device,
        formats,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Depth32Float,
    );
    let mut shell = rmesh_render::ShellBuffers::new(&device);
    // Many rejected and overlapping faces stress the same vertex/raster path
    // as a dense extracted shell, without requiring an external asset.
    let faces: Vec<u32> = (0..4096).map(|i| i % 4).collect();
    shell.upload(&device, &queue, &faces);
    shell.write_view(
        &queue,
        glam::Mat4::IDENTITY.to_cols_array_2d(),
        [0., 0., -2.],
    );
    let upload = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let buffers = [
        upload(&[0u8; std::mem::size_of::<rmesh_render::Uniforms>()]),
        upload(bytemuck::cast_slice(&[
            -0.8f32, -0.8, 0.3, 0.8, -0.8, 0.3, 0., 0.8, 0.3, 0., 0., 0.8,
        ])),
        upload(bytemuck::cast_slice(&[0u32, 1, 2, 3])),
        shell.faces.clone(),
        upload(&[0u8; 4]),
        upload(&[0u8; 4]),
        upload(&[0u8; 12]),
        upload(&[0u8; 4]),
        upload(bytemuck::cast_slice(&[0.8f32, 0.4, 0.2])),
    ];
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: shell.view_buf.as_entire_binding(),
    }];
    entries.extend(
        buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            }),
    );
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.bg_layout,
        entries: &entries,
    });
    let texture = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 3840,
                height: 2160,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = texture(
        formats.color,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let depth = texture(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let cv = color.create_view(&Default::default());
    let dv = depth.create_view(&Default::default());
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 6 * (64 * 64 * 4 + 3840 * 2160 * 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for frame in 0..6u64 {
        if frame == 3 {
            let uniforms = rmesh_render::make_uniforms(
                glam::Mat4::IDENTITY, glam::Mat3::IDENTITY,
                [1920., 1920., 1920., 1080.], glam::Vec3::ZERO,
                3840., 2160., 1, 0, 16, 4., 0, 1., 3.,
            );
            queue.write_buffer(&buffers[0], 0, bytemuck::bytes_of(&uniforms));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cv,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
        rmesh_render::record_shell_color_pass(&mut encoder, &pipeline, &bg, &shell, &cv, &dv);
        for (channel, tex) in [&color, &depth].into_iter().enumerate() {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: if channel == 1 {
                        wgpu::Origin3d::ZERO
                    } else {
                        wgpu::Origin3d {
                            x: 1888,
                            y: 1048,
                            z: 0,
                        }
                    },
                    aspect: if channel == 1 {
                        wgpu::TextureAspect::DepthOnly
                    } else {
                        wgpu::TextureAspect::All
                    },
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &result,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: frame * (64 * 64 * 4 + 3840 * 2160 * 4)
                            + channel as u64 * 64 * 64 * 4,
                        bytes_per_row: Some(if channel == 1 { 3840 * 4 } else { 256 }),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: if channel == 1 { 3840 } else { 64 },
                    height: if channel == 1 { 2160 } else { 64 },
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit([encoder.finish()]);
    }
    let bytes: Vec<u8> = read_buffer(
        &device,
        &queue,
        &result,
        6 * (64 * 64 * 4 + 3840 * 2160 * 4),
    );
    for (frame, data) in bytes
        .chunks_exact(64 * 64 * 4 + 3840 * 2160 * 4)
        .enumerate()
    {
        for pixel in data[..64 * 64 * 4].chunks_exact(4) {
            assert_eq!(
                pixel,
                if frame < 3 { [204, 102, 51, 255] } else { [0, 0, 0, 255] },
                "frame {frame}: unexpected shell color with camera cutoff"
            );
        }
        for z in data
            [64 * 64 * 4 + (1080 * 3840 + 1888) * 4..64 * 64 * 4 + (1080 * 3840 + 1952) * 4]
            .chunks_exact(4)
        {
            let z = f32::from_le_bytes(z.try_into().unwrap());
            assert!(if frame < 3 { z > 0.3 && z < 0.9 } else { z == 1.0 },
                "frame {frame}: excluded shell must not write depth: {z}");
        }
        assert_eq!(
            data,
            &bytes[(if frame < 3 { 0 } else { 3 }) * (64 * 64 * 4 + 3840 * 2160 * 4)
                ..(if frame < 3 { 1 } else { 4 }) * (64 * 64 * 4 + 3840 * 2160 * 4)],
            "frame {frame}: stationary shell changed"
        );
    }
}
