//! G-buffer format A/B harness.
//!
//! Renders the same scene twice on one device — once with a baseline
//! `GBufferFormats` and once with a candidate — and reports how much the
//! narrower format changed the final tonemapped deferred frame.
//!
//! Why this shape:
//!
//! - **One process, one device.** `GBufferFormats` is a runtime parameter, so
//!   both configurations are built against the same adapter, the same uploaded
//!   scene and the same camera. A cross-run PNG comparison would fold in
//!   whatever else differs between two processes.
//! - **No GTAO / SSGI / SSR, no temporal history, no frame counter.** Those
//!   passes carry per-frame jitter and history feedback, which would swamp a
//!   quantization delta with denoiser noise. Leaving them out makes the render
//!   deterministic, which `determinism_selftest` asserts before any metric is
//!   believed.
//! - **No density shell.** The shell writes `alpha = 1` into all four slots,
//!   so omitting it leaves *more* low-alpha pixels than the viewer has. Since
//!   quantization error is amplified by `1/alpha`, that biases this harness
//!   toward over-reporting error — the conservative direction.
//!
//! Metrics are computed on the tonemapped frame because that is what the
//! acceptance bar refers to, and they are resolution-dependent — every run in a
//! campaign must use the same `RMESH_AB_RES`.
//!
//! Run with:
//! ```text
//! RMESH_AB_CANDIDATE=aux0=rgba8 \
//!   cargo test --release -p rmesh-postprocess --test gbuffer_format_ab -- --ignored --nocapture
//! ```

use glam::{Mat3, Vec3};
use rmesh_data::{PbrData, SceneData};
use rmesh_render::GBufferFormats;
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::shared::{bytes_per_texel, decode_texel, GpuLight};

// ---------------------------------------------------------------------------
// GPU setup
// ---------------------------------------------------------------------------

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Requests the adapter's maximum buffer sizes — a real 6.5M-tet asset needs
/// multi-GB interval buffers that the default limits will not cover.
fn try_gpu_big() -> Option<Gpu> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        let al = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gbuffer_ab"),
                required_features: wgpu::Features::SUBGROUP
                    | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
                    max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
                    max_buffer_size: al.max_buffer_size,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu { device, queue })
    })
}

fn make_tex(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    fmt: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage,
        view_formats: &[],
    });
    let v = t.create_view(&wgpu::TextureViewDescriptor::default());
    (t, v)
}

fn cam_c2w_intrinsics(eye: Vec3, target: Vec3, fov_y: f32, w: u32, h: u32) -> (Mat3, [f32; 4]) {
    let view = look_at(eye, target, Vec3::Z);
    let inv_view = view.inverse();
    let right = inv_view.col(0).truncate();
    let up = inv_view.col(1).truncate();
    let back = inv_view.col(2).truncate();
    let c2w = Mat3::from_cols(right, -up, -back);
    let f = 1.0 / (fov_y / 2.0).tan();
    let fy = f * h as f32 / 2.0;
    (c2w, [fy, fy, w as f32 / 2.0, h as f32 / 2.0])
}

// ---------------------------------------------------------------------------
// Format parsing
// ---------------------------------------------------------------------------

fn parse_format(name: &str) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as F;
    match name.trim().to_ascii_lowercase().as_str() {
        "rgba16f" | "rgba16float" => F::Rgba16Float,
        "rgba8" | "rgba8unorm" => F::Rgba8Unorm,
        "rgba8srgb" | "rgba8unormsrgb" => F::Rgba8UnormSrgb,
        "rgb10a2" | "rgb10a2unorm" => F::Rgb10a2Unorm,
        "rg16f" | "rg16float" => F::Rg16Float,
        other => panic!("unknown format {other:?}"),
    }
}

/// `RMESH_AB_CANDIDATE="aux0=rgba8,color=rgba8srgb"` — unnamed slots keep the
/// baseline format, so a run always isolates exactly what it names.
fn candidate_from_env(base: GBufferFormats) -> GBufferFormats {
    let mut f = base;
    let Ok(spec) = std::env::var("RMESH_AB_CANDIDATE") else {
        return f;
    };
    for part in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (slot, val) = part.split_once('=').expect("expected slot=format");
        let fmt = parse_format(val);
        match slot.trim().to_ascii_lowercase().as_str() {
            "color" => f.color = fmt,
            "aux0" => f.aux0 = fmt,
            "normals" => f.normals = fmt,
            "depth" => f.depth = fmt,
            other => panic!("unknown slot {other:?}"),
        }
    }
    f
}

fn describe(f: GBufferFormats) -> String {
    format!(
        "color={:?} aux0={:?} normals={:?} depth={:?} ({} B/px, {} ROP slots)",
        f.color,
        f.aux0,
        f.normals,
        f.depth,
        f.bytes_per_pixel(),
        f.rop_slots()
    )
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

struct AbFrame {
    /// Deferred location(0), linear HDR.
    display: Vec<[f32; 4]>,
    /// Slot-0 accumulated alpha, decoded per-format. Quantization error is
    /// amplified by 1/alpha, so every metric is also reported bucketed by this.
    alpha: Vec<f32>,
}

struct View {
    eye: Vec3,
    target: Vec3,
    fov: f32,
    near: f32,
    far: f32,
}

/// Read a whole texture back and decode it with the format it was created with.
fn readback(
    gpu: &Gpu,
    tex: &wgpu::Texture,
    fmt: wgpu::TextureFormat,
    w: u32,
    h: u32,
) -> Vec<[f32; 4]> {
    let bpp = bytes_per_texel(fmt);
    let aligned = (w * bpp + 255) & !255;
    let rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ab_readback"),
        size: (aligned * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut e = gpu.device.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(e.finish()));
    let slice = rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let mut out = vec![[0.0f32; 4]; (w * h) as usize];
    for y in 0..h {
        let row = (y * aligned) as usize;
        for x in 0..w {
            let p = row + (x * bpp) as usize;
            out[(y * w + x) as usize] = decode_texel(fmt, &data[p..]);
        }
    }
    drop(data);
    rb.unmap();
    out
}

#[allow(clippy::too_many_arguments)]
fn render_gbuf(
    gpu: &Gpu,
    scene: &SceneData,
    pbr: &PbrData,
    view: &View,
    light: GpuLight,
    gbuf: GBufferFormats,
    w: u32,
    h: u32,
) -> AbFrame {
    let device = &gpu.device;
    let queue = &gpu.queue;

    let vmat = look_at(view.eye, view.target, Vec3::Z);
    let proj = perspective_matrix(view.fov, w as f32 / h as f32, view.near, view.far);
    let vp = proj * vmat;
    let (c2w, intrinsics) = cam_c2w_intrinsics(view.eye, view.target, view.fov, w, h);

    let base_colors = vec![0.6f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd, targets, cbg, _rbg) = rmesh_render::setup_forward_with_formats(
        device,
        queue,
        scene,
        &base_colors,
        &scene.color_grads,
        w,
        h,
        gbuf,
    );

    // Real learned vertex normals — without these the deferred normal is flat
    // and the whole shading comparison is degenerate.
    if pbr.vertex_normals.len() == scene.vertex_count as usize * 3 {
        queue.write_buffer(
            &buffers.vertex_normals,
            0,
            bytemuck::cast_slice(&pbr.vertex_normals),
        );
    }

    let ci = rmesh_render::ComputeIntervalPipelines::new(device, gbuf);
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(queue);

    // Real per-tet material, packed exactly as the viewer does
    // (crates/rmesh-viewer/src/main.rs): stride 6,
    // [roughness, metallic, f0_dielectric, albedo.rgb]. A constant fixture here
    // would make every slot-1 measurement read zero error regardless of format.
    let tc = scene.tet_count as usize;
    let mut aux_flat = vec![0.0f32; tc * 6];
    for t in 0..tc {
        aux_flat[t * 6] = pbr.roughness[t];
        aux_flat[t * 6 + 1] = pbr.metallic[t];
        aux_flat[t * 6 + 2] = pbr.f0_dielectric[t];
        for c in 0..3 {
            aux_flat[t * 6 + 3 + c] = pbr.albedo[t * 3 + c];
        }
    }
    use wgpu::util::DeviceExt;
    let aux_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ab_aux"),
        contents: bytemuck::cast_slice(&aux_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let gen_bg_a =
        rmesh_render::create_compute_interval_gen_bind_group(device, &ci, &buffers, &material);
    let gen_bg_b = rmesh_render::create_compute_interval_gen_bind_group_with_sort_values(
        device,
        &ci,
        &buffers,
        &material,
        sort_state.values_b(),
    );
    let ci_render_bg = rmesh_render::create_compute_interval_render_bind_group_pbr(
        device,
        &ci,
        &buffers,
        &aux_buf,
        &buffers.indices,
    );
    let ci_convert_bg =
        rmesh_render::create_compute_interval_indirect_convert_bind_group(device, &ci, &buffers);

    let uniforms = rmesh_render::make_uniforms(
        vp,
        c2w,
        intrinsics,
        view.eye,
        w as f32,
        h as f32,
        scene.tet_count,
        0,
        16,
        0.0,
        0,
        view.near,
        view.far,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let (_hw, hw_v) = make_tex(
        device,
        w,
        h,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        "ab_hw_depth",
    );

    fn clear_att(v: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
        wgpu::RenderPassColorAttachment {
            view: v,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        }
    }
    let mut enc = device.create_command_encoder(&Default::default());
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("ab_clear_mrt"),
        color_attachments: &[
            Some(clear_att(&targets.color_view)),
            Some(clear_att(&targets.aux0_view)),
            Some(clear_att(&targets.normals_view)),
            Some(clear_att(&targets.depth_view)),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &hw_v,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    rmesh_render::record_sorted_compute_interval_forward_pass(
        &mut enc,
        device,
        &fwd,
        &ci,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &targets,
        &cbg,
        &gen_bg_a,
        &gen_bg_b,
        &ci_render_bg,
        &ci_convert_bg,
        scene.tet_count,
        queue,
        &hw_v,
        None,
        None,
        false,
        true,
        None,
    );
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    // Deferred shade. DSM off: shadows add a second cubemap render whose own
    // formats are unrelated to the G-buffer, and would only add variance.
    let display_format = wgpu::TextureFormat::Rgba16Float;
    let atlas = rmesh_dsm::DsmAtlas::new_dummy(device);
    let deferred = rmesh_postprocess::DeferredShadePipeline::new(device, display_format);
    let ra_tb = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let (_ao, ao_v) = make_tex(device, w, h, wgpu::TextureFormat::R8Unorm, ra_tb, "ab_ao");
    let (_ssgi, ssgi_v) = make_tex(device, w, h, display_format, ra_tb, "ab_ssgi");
    let (_lit, lit_v) = make_tex(device, w, h, display_format, ra_tb, "ab_lit");
    let (_ssr, ssr_v) = make_tex(device, w, h, display_format, ra_tb, "ab_ssr");
    {
        // AO=1 everywhere; SSGI/SSR stay at their transparent clear.
        let mut e = device.create_command_encoder(&Default::default());
        for (v, clear) in [
            (&ao_v, wgpu::Color::WHITE),
            (&ssgi_v, wgpu::Color::TRANSPARENT),
            (&lit_v, wgpu::Color::TRANSPARENT),
            (&ssr_v, wgpu::Color::TRANSPARENT),
        ] {
            e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ab_clear_aux"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: v,
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
        }
        queue.submit(std::iter::once(e.finish()));
    }

    let bg = rmesh_postprocess::create_deferred_bind_group(
        device,
        &deferred,
        &targets.color_view,
        &targets.aux0_view,
        &targets.normals_view,
        &targets.depth_view,
        &hw_v,
        &ao_v,
        &ssgi_v,
        &lit_v,
        &ssr_v,
    );
    let dsm_bg = rmesh_postprocess::create_deferred_dsm_bind_group(
        device,
        &deferred,
        &atlas.cubemap_views[0],
        &atlas.meta_buf,
    );
    let u = rmesh_postprocess::DeferredUniforms {
        inv_vp: vp.inverse().to_cols_array_2d(),
        cam_pos: view.eye.to_array(),
        num_lights: 1,
        width: w,
        height: h,
        ambient: 0.015,
        debug_mode: 0,
        near_plane: view.near,
        far_plane: view.far,
        dsm_enabled: 0,
        exposure: 1.0,
        ao_strength: 1.0,
        ssgi_strength: 0.0,
        _pad0: [0.0; 2],
        sky_color: [0.0; 3],
        _pad1: 0.0,
        ground_color: [0.0; 3],
        _pad2: 0.0,
    };
    queue.write_buffer(&deferred.uniforms_buf, 0, bytemuck::bytes_of(&u));
    let mut lights = [GpuLight::default(); rmesh_render::MAX_LIGHTS];
    lights[0] = light;
    queue.write_buffer(&deferred.light_buf, 0, bytemuck::cast_slice(&lights));

    let (disp, disp_v) = make_tex(
        device,
        w,
        h,
        display_format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "ab_disp",
    );
    let (_litc, litc_v) = make_tex(
        device,
        w,
        h,
        display_format,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "ab_litc",
    );
    let mut e = device.create_command_encoder(&Default::default());
    rmesh_postprocess::record_deferred_shade(&mut e, &deferred, &bg, &dsm_bg, &disp_v, &litc_v);
    queue.submit(std::iter::once(e.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let display = readback(gpu, &disp, display_format, w, h);
    let alpha = readback(gpu, &targets.color_texture, gbuf.color, w, h)
        .iter()
        .map(|p| p[3])
        .collect();
    AbFrame { display, alpha }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

mod metrics {
    /// Reinhard + gamma 2.2, matching how these frames are written to PNG.
    /// Metrics run on this, not on linear HDR, because the acceptance bar is
    /// about what the frame looks like.
    pub fn tonemap(c: f32) -> f32 {
        (c / (1.0 + c)).clamp(0.0, 1.0).powf(1.0 / 2.2)
    }

    pub fn luma(p: [f32; 4]) -> f32 {
        0.2126 * tonemap(p[0]) + 0.7152 * tonemap(p[1]) + 0.0722 * tonemap(p[2])
    }

    /// PSNR over tonemapped RGB in [0,1]. Kept in float rather than quantizing
    /// to u8 first, which would floor the result near 48 dB.
    pub fn psnr(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
        let mut mse = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            for c in 0..3 {
                let d = (tonemap(x[c]) - tonemap(y[c])) as f64;
                mse += d * d;
            }
        }
        mse /= (a.len() * 3) as f64;
        if mse == 0.0 {
            f32::INFINITY
        } else {
            (10.0 * (1.0f64 / mse).log10()) as f32
        }
    }

    pub fn max_mean_abs(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f32, f32) {
        let (mut mx, mut sum) = (0.0f32, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            for c in 0..3 {
                let d = (tonemap(x[c]) - tonemap(y[c])).abs();
                mx = mx.max(d);
                sum += d as f64;
            }
        }
        (mx, (sum / (a.len() * 3) as f64) as f32)
    }

    /// SSIM on luminance with the standard Wang et al. 11x11 Gaussian
    /// (sigma = 1.5), so the number is comparable to published figures. A box
    /// window would read systematically higher and hide blocky artifacts.
    /// Edge windows are cropped rather than clamped.
    pub fn ssim(a: &[[f32; 4]], b: &[[f32; 4]], w: usize, h: usize) -> f32 {
        const R: usize = 5; // 11x11
        let ya: Vec<f32> = a.iter().map(|p| luma(*p)).collect();
        let yb: Vec<f32> = b.iter().map(|p| luma(*p)).collect();

        let mut k = [0.0f32; 11];
        let sigma = 1.5f32;
        let mut ksum = 0.0;
        for (i, kv) in k.iter_mut().enumerate() {
            let d = i as f32 - R as f32;
            *kv = (-(d * d) / (2.0 * sigma * sigma)).exp();
            ksum += *kv;
        }
        for kv in k.iter_mut() {
            *kv /= ksum;
        }

        // Separable Gaussian; only the fully-covered interior is produced.
        let blur = |src: &[f32]| -> Vec<f32> {
            let mut tmp = vec![0.0f32; w * h];
            for y in 0..h {
                for x in R..w.saturating_sub(R) {
                    let mut s = 0.0;
                    for (i, kv) in k.iter().enumerate() {
                        s += kv * src[y * w + x + i - R];
                    }
                    tmp[y * w + x] = s;
                }
            }
            let mut out = vec![0.0f32; w * h];
            for y in R..h.saturating_sub(R) {
                for x in R..w.saturating_sub(R) {
                    let mut s = 0.0;
                    for (i, kv) in k.iter().enumerate() {
                        s += kv * tmp[(y + i - R) * w + x];
                    }
                    out[y * w + x] = s;
                }
            }
            out
        };

        let sq = |v: &[f32]| v.iter().map(|x| x * x).collect::<Vec<_>>();
        let mul = |u: &[f32], v: &[f32]| {
            u.iter().zip(v).map(|(x, y)| x * y).collect::<Vec<_>>()
        };

        let mu_a = blur(&ya);
        let mu_b = blur(&yb);
        let aa = blur(&sq(&ya));
        let bb = blur(&sq(&yb));
        let ab = blur(&mul(&ya, &yb));

        const C1: f32 = 1e-4; // (0.01 * L)^2, L = 1
        const C2: f32 = 9e-4; // (0.03 * L)^2

        let (mut sum, mut n) = (0.0f64, 0usize);
        for y in R..h.saturating_sub(R) {
            for x in R..w.saturating_sub(R) {
                let i = y * w + x;
                let (ma, mb) = (mu_a[i], mu_b[i]);
                let va = aa[i] - ma * ma;
                let vb = bb[i] - mb * mb;
                let cab = ab[i] - ma * mb;
                let s = ((2.0 * ma * mb + C1) * (2.0 * cab + C2))
                    / ((ma * ma + mb * mb + C1) * (va + vb + C2));
                sum += s as f64;
                n += 1;
            }
        }
        if n == 0 {
            1.0
        } else {
            (sum / n as f64) as f32
        }
    }
}

fn save_png(img: &[[f32; 4]], w: u32, h: u32, name: &str) {
    let mut buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = img[(y * w + x) as usize];
            let e = |c: f32| (metrics::tonemap(c) * 255.0) as u8;
            buf.put_pixel(x, y, image::Rgb([e(p[0]), e(p[1]), e(p[2])]));
        }
    }
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/test_output/{name}"));
    std::fs::create_dir_all(out.parent().unwrap()).ok();
    buf.save(&out).unwrap();
    eprintln!("saved {}", out.display());
}

/// Amplified |difference| image. The plain tonemap would crush a 1/255 step to
/// invisibility; at the default gain of 16 it lands around mid grey.
fn save_diff_png(a: &[[f32; 4]], b: &[[f32; 4]], w: u32, h: u32, gain: f32, name: &str) {
    let mut buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let e = |c: usize| {
                let d = (metrics::tonemap(a[i][c]) - metrics::tonemap(b[i][c])).abs();
                ((d * gain).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
            };
            buf.put_pixel(x, y, image::Rgb([e(0), e(1), e(2)]));
        }
    }
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/test_output/{name}"));
    std::fs::create_dir_all(out.parent().unwrap()).ok();
    buf.save(&out).unwrap();
    eprintln!("saved {}", out.display());
}

/// Error bucketed by accumulated alpha. The deferred shader divides every
/// G-buffer channel by alpha (early-out below 0.01), so quantization error is
/// amplified by up to 100x at silhouettes while interior pixels sit near
/// alpha = 1. A global PSNR can pass while the fringe sparkles.
fn report_alpha_buckets(base: &AbFrame, cand: &AbFrame) {
    const EDGES: [(f32, f32); 5] = [
        (0.01, 0.05),
        (0.05, 0.15),
        (0.15, 0.40),
        (0.40, 0.80),
        (0.80, 1.01),
    ];
    eprintln!("  {:>12}  {:>9}  {:>10}  {:>10}  {:>8}", "alpha", "pixels", "mean|dY|", "max|dY|", "PSNR");
    for (lo, hi) in EDGES {
        let (mut n, mut sum, mut mx, mut mse) = (0usize, 0.0f64, 0.0f32, 0.0f64);
        for i in 0..base.display.len() {
            let a = base.alpha[i];
            if a < lo || a >= hi {
                continue;
            }
            let d = (metrics::luma(base.display[i]) - metrics::luma(cand.display[i])).abs();
            n += 1;
            sum += d as f64;
            mx = mx.max(d);
            mse += (d * d) as f64;
        }
        if n == 0 {
            eprintln!("  [{lo:.2},{hi:.2})  {:>9}  {:>10}  {:>10}  {:>8}", 0, "-", "-", "-");
            continue;
        }
        let p = if mse == 0.0 {
            f32::INFINITY
        } else {
            (10.0 * (n as f64 / mse).log10()) as f32
        };
        eprintln!(
            "  [{lo:.2},{hi:.2})  {n:>9}  {:>10.6}  {mx:>10.6}  {p:>8.2}",
            sum / n as f64
        );
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs ~/Downloads/room_pbr_refined.rmesh; run with --ignored"]
fn gbuffer_format_ab() {
    let Some(gpu) = try_gpu_big() else {
        eprintln!("no GPU adapter — skipping");
        return;
    };

    let path = format!(
        "{}/Downloads/room_pbr_refined.rmesh",
        std::env::var("HOME").unwrap_or_default()
    );
    let path = std::env::var("RMESH_AB_SCENE").unwrap_or(path);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e} — skipping");
            return;
        }
    };
    let (scene, _sh, pbr) = rmesh_data::load_rmesh(&bytes).expect("load_rmesh");
    let Some(pbr) = pbr else {
        eprintln!("{path} has no PBR sections — skipping");
        return;
    };

    // Resolution is a parameter because PSNR/SSIM are resolution-dependent:
    // every run in one campaign must use the same value.
    let res = std::env::var("RMESH_AB_RES").unwrap_or_else(|_| "640x360".into());
    let (w, h) = res.split_once('x').expect("RMESH_AB_RES=WxH");
    let (w, h): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());

    // Camera: viewer convention is eye=start_pose, target=origin, with a
    // bounds-framing fallback for a degenerate pose.
    let mut mn = Vec3::splat(f32::INFINITY);
    let mut mx = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..scene.vertex_count as usize {
        let p = Vec3::new(
            scene.vertices[i * 3],
            scene.vertices[i * 3 + 1],
            scene.vertices[i * 3 + 2],
        );
        mn = mn.min(p);
        mx = mx.max(p);
    }
    let center = (mn + mx) * 0.5;
    let size = (mx - mn).length();
    let sp = Vec3::new(scene.start_pose[0], scene.start_pose[1], scene.start_pose[2]);
    let (eye, target) = if sp.length() > 1e-3 {
        (sp, Vec3::ZERO)
    } else {
        (center + Vec3::new(0.0, -size * 0.6, size * 0.1), center)
    };
    let view = View {
        eye,
        target,
        fov: 0.8,
        near: (size * 0.001).max(0.01),
        far: size * 3.0,
    };
    let light = GpuLight {
        position: (eye + Vec3::new(0.0, 0.0, size * 0.2)).to_array(),
        light_type: 0,
        color: [1.0, 1.0, 1.0],
        intensity: size * size * 0.5,
        ..Default::default()
    };

    // The baseline is the PRE-CAMPAIGN all-f16 set, not `Default`, so every
    // measurement is cumulative against one fixed reference even as the shipping
    // default narrows underneath us. Overridable for incremental comparisons.
    let pristine = GBufferFormats {
        color: wgpu::TextureFormat::Rgba16Float,
        aux0: wgpu::TextureFormat::Rgba16Float,
        normals: wgpu::TextureFormat::Rgba8Unorm,
        depth: wgpu::TextureFormat::Rgba16Float,
    };
    let baseline_fmt = match std::env::var("RMESH_AB_BASELINE").ok().as_deref() {
        Some("default") => GBufferFormats::default(),
        Some(spec) => {
            let mut f = pristine;
            for part in spec.split(',').filter(|s| !s.trim().is_empty()) {
                let (slot, val) = part.split_once('=').expect("expected slot=format");
                let fmt = parse_format(val);
                match slot.trim().to_ascii_lowercase().as_str() {
                    "color" => f.color = fmt,
                    "aux0" => f.aux0 = fmt,
                    "normals" => f.normals = fmt,
                    "depth" => f.depth = fmt,
                    other => panic!("unknown slot {other:?}"),
                }
            }
            f
        }
        None => pristine,
    };
    // With no RMESH_AB_CANDIDATE, compare whatever is shipping against pristine.
    let candidate_fmt = match std::env::var("RMESH_AB_CANDIDATE") {
        Ok(_) => candidate_from_env(baseline_fmt),
        Err(_) => GBufferFormats::default(),
    };

    eprintln!("\n=== G-buffer format A/B ===");
    eprintln!("scene     {path}");
    eprintln!("          {} verts, {} tets", scene.vertex_count, scene.tet_count);
    eprintln!("res       {w}x{h}   (PSNR/SSIM are resolution-dependent)");
    eprintln!("baseline  {}", describe(baseline_fmt));
    eprintln!("candidate {}", describe(candidate_fmt));

    let base = render_gbuf(&gpu, &scene, &pbr, &view, light, baseline_fmt, w, h);

    // Determinism precondition. If the same formats twice are not bit-identical,
    // every delta below is noise and the campaign cannot proceed.
    {
        let again = render_gbuf(&gpu, &scene, &pbr, &view, light, baseline_fmt, w, h);
        let (mx, _) = metrics::max_mean_abs(&base.display, &again.display);
        assert_eq!(
            mx, 0.0,
            "baseline is not deterministic (max tonemapped delta {mx:e}) — \
             fix that before trusting any format comparison"
        );
        eprintln!("determinism  OK (baseline reproduces bit-identically)");
    }

    if candidate_fmt == baseline_fmt {
        eprintln!(
            "\ncandidate == baseline; set RMESH_AB_CANDIDATE=\"aux0=rgba8\" to compare.\n\
             (A no-op change should still be verified here: it must read PSNR = inf.)"
        );
    }

    let cand = render_gbuf(&gpu, &scene, &pbr, &view, light, candidate_fmt, w, h);

    let psnr = metrics::psnr(&base.display, &cand.display);
    let ssim = metrics::ssim(&base.display, &cand.display, w as usize, h as usize);
    let (mxd, mean) = metrics::max_mean_abs(&base.display, &cand.display);

    eprintln!("\n--- global (tonemapped) ---");
    eprintln!("  PSNR      {psnr:.2} dB     (bar: >= 35)");
    eprintln!("  SSIM      {ssim:.5}        (bar: >= 0.99)");
    eprintln!("  max |d|   {mxd:.6}");
    eprintln!("  mean |d|  {mean:.6}");

    eprintln!("\n--- by accumulated alpha (error is amplified by 1/alpha) ---");
    report_alpha_buckets(&base, &cand);

    let gain: f32 = std::env::var("RMESH_DIFF_GAIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16.0);
    save_png(&base.display, w, h, "gbuf_ab_baseline.png");
    save_png(&cand.display, w, h, "gbuf_ab_candidate.png");
    save_diff_png(
        &base.display,
        &cand.display,
        w,
        h,
        gain,
        &format!("gbuf_ab_diff_x{}.png", gain as u32),
    );

    assert!(psnr >= 35.0, "PSNR {psnr:.2} dB below the 35 dB bar");
    assert!(ssim >= 0.99, "SSIM {ssim:.5} below the 0.99 bar");
    eprintln!("\nPASS\n");
}
