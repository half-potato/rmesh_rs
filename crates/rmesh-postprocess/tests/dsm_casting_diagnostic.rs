//! DSM CASTING diagnostic (visual tool, not a pass/fail test).
//!
//! The user's localization: "the problem is only the volumetric CREATION of the
//! shadow map — it looks fine on the compositor primitives." Neither existing
//! e2e test casts with volumetric geometry (the shadow test uses a primitive
//! cube), so this is the untested path.
//!
//! This casts the SAME occluder into a light's DSM cubemap two ways —
//! (1) a volumetric tet, (2) a primitive cube at the same place — and dumps all
//! 6 cube faces of each as a horizontal strip PNG (R=normalized depth mean,
//! G=occlusion α, B=moment weight). If the volumetric strip differs from the
//! primitive strip in WHERE/HOW the occluder lands on a face, that difference IS
//! the casting bug. PNGs land in target/test_output/.

use glam::{Mat3, Vec3};
use half::f16;
use rmesh_data::SceneData;
use rmesh_interact::{Primitive, PrimitiveKind};
use rmesh_render::GpuLight;
use rmesh_util::camera::{look_at, perspective_matrix};
use rmesh_util::test_util::build_test_scene;

const NEAR: f32 = 0.05;
const FAR: f32 = 50.0;
const RES: u32 = 128;
const W: u32 = 320;
const H: u32 = 320;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn try_gpu() -> Option<Gpu> {
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 20,
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu { device, queue })
    })
}

/// Like try_gpu but requests the adapter's MAX buffer sizes (real meshes have
/// multi-GB interval buffers). Returns None if the adapter can't provide them.
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
        eprintln!(
            "adapter max_buffer_size={:.2}GB max_storage_binding={:.2}GB",
            al.max_buffer_size as f64 / 1e9,
            al.max_storage_buffer_binding_size as f64 / 1e9
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP | wgpu::Features::SHADER_FLOAT32_ATOMIC,
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

/// A regular tet centered at `center`, OUTWARD per-vertex normals.
fn tet_scene(center: Vec3, s: f32, density: f32) -> (SceneData, Vec<f32>) {
    let base = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    let mut verts = Vec::new();
    let mut vn = Vec::new();
    for b in base {
        let p = center + b * s;
        verts.extend_from_slice(&[p.x, p.y, p.z]);
        let n = b.normalize();
        vn.extend_from_slice(&[n.x, n.y, n.z]);
    }
    (
        build_test_scene(verts, vec![0, 1, 2, 3], vec![density], vec![0.0; 3]),
        vn,
    )
}

/// Build a DSM for one light, casting `scene` tets (if `render_tets`) and
/// `prim_occluders`. Returns the atlas for face readback.
fn build_dsm(
    gpu: &Gpu,
    scene: &SceneData,
    vnormals: &[f32],
    prim_occluders: &[Primitive],
    light: GpuLight,
    render_tets: bool,
) -> rmesh_dsm::DsmAtlas {
    build_dsm_nf(
        gpu,
        scene,
        vnormals,
        prim_occluders,
        light,
        render_tets,
        NEAR,
        FAR,
        RES,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_dsm_nf(
    gpu: &Gpu,
    scene: &SceneData,
    vnormals: &[f32],
    prim_occluders: &[Primitive],
    light: GpuLight,
    render_tets: bool,
    near: f32,
    far: f32,
    res: u32,
) -> rmesh_dsm::DsmAtlas {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let color_format = wgpu::TextureFormat::Rgba16Float;

    let base_colors = vec![0.6f32; scene.tet_count as usize * 3];
    let (buffers, material, _fwd, _targets, _cbg, _rbg) = rmesh_render::setup_forward(
        device,
        queue,
        scene,
        &base_colors,
        &scene.color_grads,
        res,
        res,
    );
    queue.write_buffer(&buffers.vertex_normals, 0, bytemuck::cast_slice(vnormals));

    let ci = rmesh_render::ComputeIntervalPipelines::new(device, color_format);
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(queue);

    let dsm_pipeline = rmesh_dsm::DsmPipeline::new(device, color_format);
    let dsm_prim = rmesh_dsm::DsmPrimitivePipeline::new(device);
    let dsm_proj = rmesh_dsm::DsmProjectPipeline::new(device);
    let prim_geo = rmesh_compositor::PrimitiveGeometry::new(device);
    let atlas = rmesh_dsm::DsmAtlas::new(device, res, &[light.light_type]);
    let lights_arr = [light];

    let mut e = device.create_command_encoder(&Default::default());
    rmesh_dsm::generate_dsm_for_lights(
        &atlas,
        &mut e,
        device,
        queue,
        &dsm_pipeline,
        &dsm_prim,
        &dsm_proj,
        &prim_geo,
        prim_occluders,
        &ci,
        &sort_pipelines,
        &sort_state,
        &buffers,
        &material,
        &lights_arr,
        1,
        scene.tet_count,
        near,
        far,
        render_tets,
    );
    queue.submit(std::iter::once(e.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    atlas.populate_metadata(queue, &lights_arr, near, far);
    atlas
}

/// Read all 6 faces of the atlas's first cubemap and write a horizontal strip
/// PNG: faces 0..5 left→right, each RES×RES. R=depth mean (.r/.b), G=occ (.a),
/// B=moment weight (.b). Prints per-face occluded-pixel counts + depth range.
fn dump_cube_strip(gpu: &Gpu, atlas: &rmesh_dsm::DsmAtlas, name: &str) {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let res = atlas.resolution;
    let bpp = 8u32; // Rgba16Float
    let aligned = (res * bpp + 255) & !255;

    let mut strip = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(res * 6, res);

    for face in 0..6u32 {
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("face_rb"),
            size: (aligned * res) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut e = device.create_command_encoder(&Default::default());
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas.cubemaps[0],
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &rb,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
                    rows_per_image: Some(res),
                },
            },
            wgpu::Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(e.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();

        let mut occ_px = 0u32;
        let mut dmin = f32::INFINITY;
        let mut dmax = f32::NEG_INFINITY;
        for y in 0..res {
            let row = (y * aligned) as usize;
            for x in 0..res {
                let p = row + (x * bpp) as usize;
                let rr = f16::from_le_bytes([data[p], data[p + 1]]).to_f32();
                let gg = f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32();
                let bb = f16::from_le_bytes([data[p + 4], data[p + 5]]).to_f32();
                let aa = f16::from_le_bytes([data[p + 6], data[p + 7]]).to_f32();
                let mean = if bb > 1e-4 {
                    (rr / bb).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let occ = aa.clamp(0.0, 1.0);
                if occ > 0.05 {
                    occ_px += 1;
                    dmin = dmin.min(mean);
                    dmax = dmax.max(mean);
                }
                let px = image::Rgb([
                    (mean * 255.0) as u8,
                    (occ * 255.0) as u8,
                    (bb.clamp(0.0, 1.0) * 255.0) as u8,
                ]);
                strip.put_pixel(face * res + x, y, px);
            }
        }
        drop(data);
        rb.unmap();
        let face_name = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"][face as usize];
        if occ_px > 0 {
            eprintln!(
                "  face {face} ({face_name}): occluded_px={occ_px} depth_mean∈[{dmin:.3},{dmax:.3}]"
            );
        } else {
            eprintln!("  face {face} ({face_name}): empty");
        }
    }

    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/test_output/{name}"));
    std::fs::create_dir_all(out.parent().unwrap()).ok();
    strip.save(&out).unwrap();
    eprintln!("saved {}", out.display());
}

/// Cast an identical occluder into a point-light DSM two ways and dump both.
/// Light at origin; occluder offset into +X, lifted +Y and pushed +Z so it lands
/// OFF-CENTER on the +X face — any mirror/rotation moves the blob to a different
/// corner than the (trusted) primitive reference.
#[test]
fn dsm_casting_volumetric_vs_primitive() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU — skipping dsm_casting_volumetric_vs_primitive");
        return;
    };

    let light = GpuLight {
        position: [0.0, 0.0, 0.0],
        light_type: 0,
        color: [1.0, 1.0, 1.0],
        intensity: 30.0,
        direction: [0.0, 0.0, -1.0],
        inner_cos: 1.0,
        outer_cos: 1.0,
        _pad: [0.0; 3],
    };

    // Occluder center: +X dominant (lands on +X face), offset +Y and +Z so it's
    // off-center on that face. Asymmetric in all three axes.
    let center = Vec3::new(4.0, 1.5, 2.5);

    // --- (1) volumetric tet caster ---
    let (scene, vn) = tet_scene(center, 0.8, 40.0);
    eprintln!("VOLUMETRIC tet cast (center={center:?}):");
    let atlas_v = build_dsm(&gpu, &scene, &vn, &[], light, true);
    dump_cube_strip(&gpu, &atlas_v, "cast_volumetric.png");

    // --- (2) primitive cube caster (trusted reference) at the same place ---
    // Dummy zero-density tet so setup_forward has a scene; render_tets=false so
    // only the primitive casts.
    let (dummy, dvn) = tet_scene(Vec3::new(0.0, 0.0, -40.0), 0.1, 0.0);
    let mut cube = Primitive::new(PrimitiveKind::Cube, "occ");
    cube.transform.position = center;
    cube.transform.scale = Vec3::splat(1.4);
    eprintln!("PRIMITIVE cube cast (center={center:?}):");
    let atlas_p = build_dsm(
        &gpu,
        &dummy,
        &dvn,
        std::slice::from_ref(&cube),
        light,
        false,
    );
    dump_cube_strip(&gpu, &atlas_p, "cast_primitive.png");

    eprintln!(
        "\nCompare target/test_output/cast_volumetric.png vs cast_primitive.png.\n\
         Both should show the occluder on the +X face (face 0, leftmost), in the\n\
         SAME corner. A different corner/face on the volumetric strip = casting bug."
    );
}

// ===========================================================================
// End-to-end render: VOLUMETRIC occluder casting onto a volumetric wall, swept
// across densities. Dumps the lit image AND the pure-transmittance debug view
// (DBG_SHADOW=9), so we can SEE whether the α>0.1 moment gate erases the shadow
// for low/medium-density occluders (real radiance-mesh case).
// ===========================================================================

fn cam_c2w_intrinsics(eye: Vec3, target: Vec3, fov_y: f32) -> (Mat3, [f32; 4]) {
    let f = (target - eye).normalize();
    let r = f.cross(Vec3::Z).normalize();
    let u = r.cross(f);
    let c2w = Mat3::from_cols(r, -u, f);
    let fval = 1.0 / (fov_y / 2.0).tan();
    let fx = fval * H as f32 / 2.0;
    (c2w, [fx, fx, W as f32 / 2.0, H as f32 / 2.0])
}

fn make_tex(
    device: &wgpu::Device,
    fmt: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: W,
            height: H,
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

fn save_png(img: &[[f32; 4]], name: &str) {
    let mut buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let p = img[(y * W + x) as usize];
            let tm = |c: f32| ((c / (1.0 + c)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
            buf.put_pixel(x, y, image::Rgb([tm(p[0]), tm(p[1]), tm(p[2])]));
        }
    }
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/test_output/{name}"));
    std::fs::create_dir_all(out.parent().unwrap()).ok();
    buf.save(&out).unwrap();
    eprintln!("saved {}", out.display());
}

/// Default-near/far wrapper (NEAR/FAR) used by the synthetic tests.
#[allow(clippy::too_many_arguments)]
fn render_full(
    gpu: &Gpu,
    scene: &SceneData,
    vnormals: &[f32],
    prim_occluders: &[Primitive],
    dsm_on: bool,
    debug_mode: u32,
    light: GpuLight,
    eye: Vec3,
    target: Vec3,
    fov: f32,
) -> Vec<[f32; 4]> {
    render_full_nf(
        gpu,
        scene,
        vnormals,
        prim_occluders,
        dsm_on,
        debug_mode,
        light,
        eye,
        target,
        fov,
        NEAR,
        FAR,
    )
}

/// Full forward → DSM → deferred render. `scene` tets are BOTH G-buffer
/// receivers and DSM casters; `prim_occluders` cast only. `dsm_on` forces the
/// shadow path even with no primitive occluders (volumetric caster in `scene`).
#[allow(clippy::too_many_arguments)]
fn render_full_nf(
    gpu: &Gpu,
    scene: &SceneData,
    vnormals: &[f32],
    prim_occluders: &[Primitive],
    dsm_on: bool,
    debug_mode: u32,
    light: GpuLight,
    eye: Vec3,
    target: Vec3,
    fov: f32,
    near: f32,
    far: f32,
) -> Vec<[f32; 4]> {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let color_format = wgpu::TextureFormat::Rgba16Float;

    let view = look_at(eye, target, Vec3::Z);
    let proj = perspective_matrix(fov, W as f32 / H as f32, near, far);
    let vp = proj * view;
    let (c2w, intrinsics) = cam_c2w_intrinsics(eye, target, fov);

    let base_colors = vec![0.6f32; scene.tet_count as usize * 3];
    let (buffers, material, fwd, targets, cbg, _rbg) =
        rmesh_render::setup_forward(device, queue, scene, &base_colors, &scene.color_grads, W, H);
    queue.write_buffer(&buffers.vertex_normals, 0, bytemuck::cast_slice(vnormals));

    let ci = rmesh_render::ComputeIntervalPipelines::new(device, color_format);
    let n_pow2 = scene.tet_count.next_power_of_two();
    let sort_pipelines =
        rmesh_sort::RadixSortPipelines::new(device, 1, rmesh_sort::SortBackend::Drs);
    let sort_state =
        rmesh_sort::RadixSortState::new(device, n_pow2, 32, 1, rmesh_sort::SortBackend::Drs);
    sort_state.upload_configs(queue);

    let aux_flat: Vec<f32> = (0..scene.tet_count)
        .flat_map(|_| [0.7f32, 0.04, 0.04, 0.04, 0.6, 0.6, 0.6])
        .collect();
    use wgpu::util::DeviceExt;
    let aux_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("aux"),
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
        eye,
        W as f32,
        H as f32,
        scene.tet_count,
        0,
        16,
        0.0,
        0,
        near,
        far,
    );
    queue.write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

    let (_hw, hw_v) = make_tex(
        device,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        "hw_depth",
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
        label: Some("clear_mrt"),
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
    );
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let lights_arr = [light];
    let atlas = if dsm_on {
        let dsm_pipeline = rmesh_dsm::DsmPipeline::new(device, color_format);
        let dsm_prim = rmesh_dsm::DsmPrimitivePipeline::new(device);
        let dsm_proj = rmesh_dsm::DsmProjectPipeline::new(device);
        let prim_geo = rmesh_compositor::PrimitiveGeometry::new(device);
        let atlas = rmesh_dsm::DsmAtlas::new(device, 512, &[light.light_type]);
        let mut e = device.create_command_encoder(&Default::default());
        rmesh_dsm::generate_dsm_for_lights(
            &atlas,
            &mut e,
            device,
            queue,
            &dsm_pipeline,
            &dsm_prim,
            &dsm_proj,
            &prim_geo,
            prim_occluders,
            &ci,
            &sort_pipelines,
            &sort_state,
            &buffers,
            &material,
            &lights_arr,
            1,
            scene.tet_count,
            near,
            far,
            true,
        );
        queue.submit(std::iter::once(e.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        atlas.populate_metadata(queue, &lights_arr, near, far);
        atlas
    } else {
        rmesh_dsm::DsmAtlas::new_dummy(device)
    };

    let deferred = rmesh_postprocess::DeferredShadePipeline::new(device, color_format);
    let ra_tb = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let (_ao, ao_v) = make_tex(device, wgpu::TextureFormat::R8Unorm, ra_tb, "ao");
    let (_ssgi, ssgi_v) = make_tex(device, color_format, ra_tb, "ssgi");
    let (_lit, lit_v) = make_tex(device, color_format, ra_tb, "lit");
    let (_ssr, ssr_v) = make_tex(device, color_format, ra_tb, "ssr");
    {
        let mut e = device.create_command_encoder(&Default::default());
        e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_ao"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ao_v,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
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
        cam_pos: eye.to_array(),
        num_lights: 1,
        width: W,
        height: H,
        ambient: 0.015,
        debug_mode,
        near_plane: near,
        far_plane: far,
        dsm_enabled: if dsm_on { 1 } else { 0 },
        exposure: 1.0,
        ao_strength: 1.0,
        ssgi_strength: 0.0,
        _pad: [0.0; 2],
    };
    queue.write_buffer(&deferred.uniforms_buf, 0, bytemuck::bytes_of(&u));
    let mut lights = [GpuLight::default(); rmesh_render::MAX_LIGHTS];
    lights[0] = light;
    queue.write_buffer(&deferred.light_buf, 0, bytemuck::cast_slice(&lights));

    let (disp, disp_v) = make_tex(
        device,
        color_format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "disp",
    );
    let (_litc, litc_v) = make_tex(
        device,
        color_format,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "litc",
    );
    let mut enc = device.create_command_encoder(&Default::default());
    rmesh_postprocess::record_deferred_shade(&mut enc, &deferred, &bg, &dsm_bg, &disp_v, &litc_v);
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let bpp = 8u32;
    let aligned = (W * bpp + 255) & !255;
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb"),
        size: (aligned * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut e = device.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &disp,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(e.finish()));
    let slice = rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let mut img = vec![[0.0f32; 4]; (W * H) as usize];
    for y in 0..H {
        let row = (y * aligned) as usize;
        for x in 0..W {
            let p = row + (x * bpp) as usize;
            img[(y * W + x) as usize] = [
                f16::from_le_bytes([data[p], data[p + 1]]).to_f32(),
                f16::from_le_bytes([data[p + 2], data[p + 3]]).to_f32(),
                f16::from_le_bytes([data[p + 4], data[p + 5]]).to_f32(),
                f16::from_le_bytes([data[p + 6], data[p + 7]]).to_f32(),
            ];
        }
    }
    drop(data);
    rb.unmap();
    img
}

/// Build the wall + a volumetric occluder tet between light and wall.
/// Returns (scene, vnormals). Wall is the receiver (faces -Y); occluder casts.
fn wall_with_occluder_tet(
    occ_center: Vec3,
    occ_size: f32,
    occ_density: f32,
) -> (SceneData, Vec<f32>) {
    // Wall: large triangle in XZ at y=0, normals -Y (toward camera/light side).
    let mut verts = vec![
        -6.0, 0.0, -6.0, 6.0, 0.0, -6.0, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0, // apex behind wall
    ];
    let mut vn = vec![
        0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let mut indices = vec![0u32, 1, 2, 3];
    let mut densities = vec![10.0f32];

    // Occluder tet.
    let base = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    let v_off = (verts.len() / 3) as u32;
    for b in base {
        let p = occ_center + b * occ_size;
        verts.extend_from_slice(&[p.x, p.y, p.z]);
        let n = b.normalize();
        vn.extend_from_slice(&[n.x, n.y, n.z]);
    }
    indices.extend_from_slice(&[v_off, v_off + 1, v_off + 2, v_off + 3]);
    densities.push(occ_density);

    let color_grads = vec![0.0f32; densities.len() * 3];
    (build_test_scene(verts, indices, densities, color_grads), vn)
}

/// Wall + a CLUSTER of `n` jittered tets around `occ_center`, each of size
/// `occ_size` and density `per_density`. Individually semi-transparent, together
/// opaque — mimics a real radiance-mesh occluder.
fn wall_with_occluder_cluster(
    occ_center: Vec3,
    occ_size: f32,
    n: usize,
    per_density: f32,
) -> (SceneData, Vec<f32>) {
    let mut verts = vec![
        -6.0, 0.0, -6.0, 6.0, 0.0, -6.0, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0,
    ];
    let mut vn = vec![
        0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let mut indices = vec![0u32, 1, 2, 3];
    let mut densities = vec![10.0f32];

    let base = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ];
    // Deterministic jitter (no rng dep): small offsets on a lattice.
    for k in 0..n {
        let fk = k as f32;
        let jitter = Vec3::new(
            ((fk * 1.3).sin()) * 0.35,
            ((fk * 0.7).cos()) * 0.35,
            ((fk * 1.9).sin()) * 0.35,
        );
        let c = occ_center + jitter;
        let v_off = (verts.len() / 3) as u32;
        for b in base {
            let p = c + b * occ_size;
            verts.extend_from_slice(&[p.x, p.y, p.z]);
            let nrm = b.normalize();
            vn.extend_from_slice(&[nrm.x, nrm.y, nrm.z]);
        }
        indices.extend_from_slice(&[v_off, v_off + 1, v_off + 2, v_off + 3]);
        densities.push(per_density);
    }
    let color_grads = vec![0.0f32; densities.len() * 3];
    (build_test_scene(verts, indices, densities, color_grads), vn)
}

/// End-to-end: a VOLUMETRIC tet occluder casting a shadow onto a volumetric
/// wall, across densities. Compares against the trusted primitive-cube caster.
/// Dumps lit + pure-transmittance (DBG_SHADOW) images for each.
#[test]
fn dsm_volumetric_caster_endtoend() {
    let Some(gpu) = try_gpu() else {
        eprintln!("no GPU — skipping dsm_volumetric_caster_endtoend");
        return;
    };

    // Geometry: light in front of wall (+X offset), occluder between them; the
    // shadow lands on the wall near world origin. Camera looks at the wall.
    let light = GpuLight {
        position: [2.0, -5.0, 0.0],
        light_type: 0,
        color: [1.0, 1.0, 1.0],
        intensity: 60.0,
        direction: [0.0, 0.0, -1.0],
        inner_cos: 1.0,
        outer_cos: 1.0,
        _pad: [0.0; 3],
    };
    let eye = Vec3::new(0.0, -8.0, 0.0);
    let target = Vec3::ZERO;
    let fov = 60f32.to_radians();
    let occ_center = Vec3::new(1.0, -2.5, 0.0); // light→occ→(0,0,0) on wall

    // (A) trusted reference: primitive cube caster onto wall-only scene.
    let wall_only = {
        let verts = vec![
            -6.0, 0.0, -6.0, 6.0, 0.0, -6.0, 0.0, 0.0, 6.0, 0.0, 3.0, 0.0,
        ];
        let vn = vec![
            0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        (
            build_test_scene(verts, vec![0, 1, 2, 3], vec![10.0], vec![0.0; 3]),
            vn,
        )
    };
    let mut cube = Primitive::new(PrimitiveKind::Cube, "occ");
    cube.transform.position = occ_center;
    cube.transform.scale = Vec3::splat(1.2);
    let prim_lit = render_full(
        &gpu,
        &wall_only.0,
        &wall_only.1,
        std::slice::from_ref(&cube),
        true,
        0,
        light,
        eye,
        target,
        fov,
    );
    let prim_shadow = render_full(
        &gpu,
        &wall_only.0,
        &wall_only.1,
        std::slice::from_ref(&cube),
        true,
        9,
        light,
        eye,
        target,
        fov,
    );
    save_png(&prim_lit, "e2e_prim_caster_lit.png");
    save_png(&prim_shadow, "e2e_prim_caster_shadow.png");

    // (B) volumetric tet caster across densities.
    for &dens in &[20.0f32, 4.0, 1.0, 0.3] {
        let (scene, vn) = wall_with_occluder_tet(occ_center, 0.7, dens);
        let lit = render_full(&gpu, &scene, &vn, &[], true, 0, light, eye, target, fov);
        let shadow = render_full(&gpu, &scene, &vn, &[], true, 9, light, eye, target, fov);
        let tag = format!("{dens:.1}").replace('.', "p");
        save_png(&lit, &format!("e2e_vol_caster_d{tag}_lit.png"));
        save_png(&shadow, &format!("e2e_vol_caster_d{tag}_shadow.png"));
    }

    // (C) realistic case: a CLUSTER of individually semi-transparent tets
    // (per-tet density 0.8, α≈0.55 each) that together are opaque. This is how a
    // real radiance mesh occludes — cumulative .a across blended intervals — and
    // must still cast a strong shadow even though no single interval is opaque.
    {
        let (scene, vn) = wall_with_occluder_cluster(occ_center, 0.55, 8, 0.8);
        let lit = render_full(&gpu, &scene, &vn, &[], true, 0, light, eye, target, fov);
        let shadow = render_full(&gpu, &scene, &vn, &[], true, 9, light, eye, target, fov);
        save_png(&lit, "e2e_vol_cluster_lit.png");
        save_png(&shadow, "e2e_vol_cluster_shadow.png");
    }

    eprintln!(
        "\nDBG_SHADOW images: BLACK = fully shadowed (T=0), WHITE = lit (T=1).\n\
         The primitive caster shows a dark shadow disc near screen center.\n\
         If the volumetric-caster shadow fades/vanishes as density drops, the\n\
         α>0.1 moment gate is erasing thin-interval occluders."
    );
}

fn project_px(vp: glam::Mat4, world: Vec3) -> (u32, u32) {
    let clip = vp * world.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let px = ((ndc.x * 0.5 + 0.5) * W as f32).clamp(0.0, (W - 1) as f32) as u32;
    let py = ((1.0 - (ndc.y * 0.5 + 0.5)) * H as f32).clamp(0.0, (H - 1) as f32) as u32;
    (px, py)
}

/// Min transmittance over a window (the darkest shadow value found nearby).
fn min_t(img: &[[f32; 4]], px: u32, py: u32, r: i32) -> f32 {
    let mut m = 1.0f32;
    for dy in -r..=r {
        for dx in -r..=r {
            let x = px as i32 + dx;
            let y = py as i32 + dy;
            if x >= 0 && x < W as i32 && y >= 0 && y < H as i32 {
                m = m.min(img[(y as u32 * W + x as u32) as usize][1]);
            }
        }
    }
    m
}

fn avg_t(img: &[[f32; 4]], px: u32, py: u32, r: i32) -> f32 {
    let mut s = 0.0f64;
    let mut n = 0u32;
    for dy in -r..=r {
        for dx in -r..=r {
            let x = px as i32 + dx;
            let y = py as i32 + dy;
            if x >= 0 && x < W as i32 && y >= 0 && y < H as i32 {
                s += img[(y as u32 * W + x as u32) as usize][1] as f64;
                n += 1;
            }
        }
    }
    (s / n.max(1) as f64) as f32
}

// ===========================================================================
// REAL MESH diagnostic: load ~/Downloads/room_pbr_refined.rmesh, render with the
// real geometry + learned vertex normals + densities, a placed point light, and
// dump lit / shadow(DBG_SHADOW) / no-shadow images so we can SEE the actual
// self-shadow + shadow behaviour on the user's data. Visual tool — no asserts.
// ===========================================================================
#[test]
#[ignore = "needs ~/Downloads/room_pbr_refined.rmesh; run with --ignored"]
fn real_room_shadow_diagnostic() {
    let Some(gpu) = try_gpu_big() else {
        eprintln!("no GPU (or insufficient buffer limits) — skipping real_room_shadow_diagnostic");
        return;
    };
    let path = format!(
        "{}/Downloads/room_pbr_refined.rmesh",
        std::env::var("HOME").unwrap_or_default()
    );
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e} — skipping");
            return;
        }
    };
    let (scene, _sh, pbr) = rmesh_data::load_rmesh(&bytes).expect("load_rmesh");
    eprintln!(
        "loaded: {} verts, {} tets, start_pose={:?}",
        scene.vertex_count, scene.tet_count, scene.start_pose
    );

    // Real learned vertex normals (critical for the normal-offset self-shadow
    // path); fall back to zeros if absent.
    let vn = pbr
        .as_ref()
        .map(|p| p.vertex_normals.clone())
        .filter(|v| v.len() == scene.vertex_count as usize * 3)
        .unwrap_or_else(|| vec![0.0; scene.vertex_count as usize * 3]);
    eprintln!(
        "vertex_normals: {}",
        if vn.iter().any(|&x| x != 0.0) {
            "real (PBR)"
        } else {
            "ZERO"
        }
    );

    // Scene bounds.
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
    eprintln!("bounds min={mn:?} max={mx:?} center={center:?} diag={size:.3}");

    // Camera: viewer convention is eye=start_pose, target=ZERO. Fall back to
    // framing the bounds if start_pose is degenerate.
    let sp = Vec3::new(
        scene.start_pose[0],
        scene.start_pose[1],
        scene.start_pose[2],
    );
    let (eye, target) = if sp.length() > 1e-3 {
        (sp, Vec3::ZERO)
    } else {
        (center + Vec3::new(0.0, -size * 0.6, size * 0.1), center)
    };
    let fov = 50f32.to_radians();
    let near = (size * 0.002).max(0.02);
    let far = size * 4.0;
    eprintln!("camera eye={eye:?} target={target:?} near={near:.3} far={far:.2}");

    // HEADLIGHT: light at the camera (+ tiny offset). Everything the camera sees
    // is then definitionally lit — no occluder between light and visible surface
    // — so any darkening under DSM is SELF-shadow, isolating it from occlusion.
    // Intensity ~ (eye→target dist)² so the framed surfaces are well exposed.
    let view_dist = (target - eye).length().max(1.0);
    let lpos = eye + Vec3::new(view_dist * 0.05, view_dist * 0.05, 0.0);
    let intensity = view_dist.powi(2) * 1.5;
    let light = GpuLight {
        position: lpos.to_array(),
        light_type: 0,
        color: [1.0, 1.0, 1.0],
        intensity,
        direction: [0.0, 0.0, -1.0],
        inner_cos: 1.0,
        outer_cos: 1.0,
        _pad: [0.0; 3],
    };
    eprintln!("light pos={lpos:?} intensity={intensity:.2}");

    let noshadow = render_full_nf(
        &gpu,
        &scene,
        &vn,
        &[],
        false,
        0,
        light,
        eye,
        target,
        fov,
        near,
        far,
    );
    save_png(&noshadow, "real_room_noshadow.png");
    let lit = render_full_nf(
        &gpu,
        &scene,
        &vn,
        &[],
        true,
        0,
        light,
        eye,
        target,
        fov,
        near,
        far,
    );
    save_png(&lit, "real_room_lit.png");
    let shadow = render_full_nf(
        &gpu,
        &scene,
        &vn,
        &[],
        true,
        9,
        light,
        eye,
        target,
        fov,
        near,
        far,
    );
    save_png(&shadow, "real_room_shadow.png");

    // OFF-AXIS light (flare-like): a headlight only exposes self-shadow, never
    // cast-shadow DIRECTION. Put the light strongly to one side of the camera so
    // real cast shadows appear; render lit + DBG_SHADOW for left and right
    // placements. Correct behaviour: shadows fall AWAY from the light, so the two
    // placements must produce shadows on OPPOSITE sides of objects.
    let fdir = (target - eye).normalize();
    let right = fdir.cross(Vec3::Z).normalize();
    for (tag, sgn) in [("right", 1.0f32), ("left", -1.0f32)] {
        // IN-ROOM light: small offset from the look target (origin), to the side
        // and pulled toward the camera so it actually illuminates the visible
        // interior. Correct: right/left must shadow OPPOSITE sides of geometry.
        let lp = target + right * (sgn * 8.0) - fdir * 3.0;
        let mut sl = light;
        sl.position = lp.to_array();
        sl.intensity = 120.0;
        eprintln!("in-room light {tag}: pos={lp:?}");
        let off = render_full_nf(
            &gpu,
            &scene,
            &vn,
            &[],
            false,
            0,
            sl,
            eye,
            target,
            fov,
            near,
            far,
        );
        save_png(&off, &format!("real_room_in_{tag}_nodsm.png"));
        let lit_s = render_full_nf(
            &gpu,
            &scene,
            &vn,
            &[],
            true,
            0,
            sl,
            eye,
            target,
            fov,
            near,
            far,
        );
        save_png(&lit_s, &format!("real_room_in_{tag}_lit.png"));
        let sh_s = render_full_nf(
            &gpu,
            &scene,
            &vn,
            &[],
            true,
            9,
            sl,
            eye,
            target,
            fov,
            near,
            far,
        );
        save_png(&sh_s, &format!("real_room_in_{tag}_shadow.png"));
    }

    // Dump the raw DSM cube content for inspection (depth mean / occlusion / wt).
    let atlas = build_dsm_nf(&gpu, &scene, &vn, &[], light, true, near, far, 256);
    dump_cube_strip(&gpu, &atlas, "real_room_cube.png");

    // Overall self-shadow / darkening factor over lit pixels.
    let mut son = 0.0f64;
    let mut soff = 0.0f64;
    for i in 0..(W * H) as usize {
        if noshadow[i][1] as f64 > 0.05 {
            soff += noshadow[i][1] as f64;
            son += lit[i][1] as f64;
        }
    }
    eprintln!(
        "real room: lit energy kept with DSM on = {:.4} (1.0 = no darkening)",
        if soff > 0.0 { son / soff } else { 1.0 }
    );
}
