//! Small wgpu helpers shared across the rendering crates (render, raytrace,
//! postprocess). These are generic over wgpu only — no project-specific types.

/// Bind-group entry for a whole storage/uniform buffer.
pub fn buf_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Bind-group entry for a texture view.
pub fn tex_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// Build `count` consecutive storage-buffer layout entries, one per `read_only` flag.
pub fn storage_entries(
    count: u32,
    visibility: wgpu::ShaderStages,
    read_only: &[bool],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: read_only[i as usize],
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

/// Record a fullscreen triangle pass writing a single color target.
///
/// Captures the shape shared by the post-processing passes (GTAO, AO/SSGI blur,
/// SSGI, SSR, temporal): bind group 0, a 3-vertex draw, one color attachment
/// cleared to `clear`.
pub fn record_fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    out_view: &wgpu::TextureView,
    clear: wgpu::Color,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: out_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    rpass.set_pipeline(pipeline);
    rpass.set_bind_group(0, bind_group, &[]);
    rpass.draw(0..3, 0..1);
}
