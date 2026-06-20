//! rmesh-viewer: Interactive wgpu viewer for .rmesh files.
//!
//! Usage:
//!   rmesh-viewer <input.rmesh>
//!
//! Controls:
//!   Left-drag: Orbit
//!   Middle-drag: Pan
//!   Right-drag (vertical): Zoom
//!   Scroll: Zoom
//!   Escape: Quit

use anyhow::{Context, Result};
use glam::Vec3;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use rmesh_anim::{AnimatedScene, AnimationClock};
use rmesh_interact::{
    InteractContext, InteractEvent, InteractKey, InteractResult, Primitive, TransformInteraction,
    VertexSelectInteraction, VertexSelectResult,
};
use rmesh_pbd::{
    build_island, color_constraints, ConstraintColoring, Island, MeshTopology, PbdSolver,
};
use rmesh_util::camera::Camera;
use wgpu::util::DeviceExt;

use rmesh_compositor::{PrimitiveGeometry, PrimitivePipeline, PrimitiveTargets};
use rmesh_render::{
    create_blit_bind_group, create_compute_bind_group, create_compute_interval_gen_bind_group,
    create_compute_interval_gen_bind_group_with_sort_values,
    create_compute_interval_indirect_convert_bind_group, create_compute_interval_render_bind_group,
    create_hw_compute_bind_group, create_indirect_convert_bind_group,
    create_interval_indirect_convert_bind_group, create_interval_render_bind_group,
    create_interval_render_bind_group_with_sort_values, create_mesh_render_bind_group,
    create_mesh_render_bind_group_with_sort_values, create_prepass_bind_group,
    create_quad_render_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values, BlitPipeline, ComputeIntervalPipelines,
    ForwardPipelines, IntervalPipelines, MaterialBuffers, MeshForwardPipelines, RenderTargets,
    SceneBuffers,
};

mod flare;
mod gpu_state;
mod render;
use flare::FlareSystem;
use gpu_state::*;

/// Create an Rgba32Float texture for copying the raytrace buffer output to blit.
/// Hard-fail when the loaded PbrData is missing the parametric BRDF channels
/// (metallic / f0_dielectric / albedo / roughness). This is what we'd see
/// for a .rmesh file produced by an old `convert.py` (env_feature/MLP era);
/// the deferred shader has no way to render those, so refuse to upload garbage.
fn require_parametric_pbr(pbr: &rmesh_data::PbrData, tet_count: usize) {
    let lens = [
        ("roughness", pbr.roughness.len()),
        ("metallic", pbr.metallic.len()),
        ("f0_dielectric", pbr.f0_dielectric.len()),
        ("albedo", pbr.albedo.len() / 3),
    ];
    let bad: Vec<_> = lens.iter().filter(|(_, l)| *l != tet_count).collect();
    if !bad.is_empty() {
        let detail = bad
            .iter()
            .map(|(name, l)| format!("{}={}", name, l))
            .collect::<Vec<_>>()
            .join(", ");
        panic!(
            "PBR data missing parametric BRDF channels for {} tets ({}). \
             This .rmesh was likely produced with an older convert.py — \
             regenerate it with the parametric-BRDF convert.py that emits \
             metallic / f0_dielectric tagged sections.",
            tet_count, detail,
        );
    }
}

fn create_rt_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt_output_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Rich result from [`App::pick_vertex_info`]. The hover preview uses
/// `screen_pos` to draw a marker; the click handler uses the other fields
/// for the diagnostic log.
#[derive(Clone, Copy, Debug)]
pub struct PickInfo {
    pub idx: u32,
    pub screen_pos: [f32; 2],
    pub screen_dist: f32,
    pub view_depth: f32,
    pub candidates_in_radius: u32,
    pub candidates_passing_depth: u32,
    pub visible_depth: Option<f32>,
    pub depth_tol: f32,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    camera: Camera,
    scene_data: rmesh_data::SceneData,
    sh_coeffs: rmesh_data::ShCoeffs,
    left_pressed: bool,
    middle_pressed: bool,
    right_pressed: bool,
    shift_pressed: bool,
    last_mouse: (f64, f64),
    // egui + FPS
    egui_ctx: egui::Context,
    frame_times: VecDeque<std::time::Instant>,
    fps: f64,
    loaded_path: Option<PathBuf>,
    pending_load: Option<PathBuf>,
    vsync: bool,
    render_mode: RenderMode,
    mesh_shader_supported: bool,
    // Rendering options
    show_primitives: bool,
    show_scene: bool,
    sort_mode: SortMode,
    // Transform interaction
    interaction: TransformInteraction,
    // Vertex-pick + drag-to-PBD interaction
    vertex_select: VertexSelectInteraction,
    /// CPU mesh topology (adjacency + vertex→tets), built lazily on first
    /// vertex-select grab and rebuilt on scene reload.
    mesh_topology: Option<MeshTopology>,
    /// Cached pre-grab vertex positions (flat f32 array, scene-global) so
    /// CancelGrab can restore them on the GPU.
    pbd_pre_grab_vertices: Option<Vec<f32>>,
    /// World-space initial handle positions captured at BeginGrab. Used with
    /// the per-frame mouse delta to compute new handle positions.
    pbd_handle_initial: Vec<(u32, [f32; 3])>,
    /// Number of PBD steps dispatched in the current grab (reset on init).
    pbd_step_counter: u32,
    /// Soft-body grab island radius in world units. Drives the BFS extent
    /// in [`init_pbd_grab`] — larger = more vertices pulled along, softer
    /// falloff. Exposed via the left side panel.
    pbd_radius: f32,
    /// Vertex that would be picked if user clicked now — refreshed each
    /// frame while VertexSelect mode is on and not grabbing. Rendered as
    /// a small overlay circle.
    pbd_hover_vertex: Option<u32>,
    /// Per-vertex max incident tet density (lazy; built once per scene
    /// alongside topology). Used to reject picks on vertices that only
    /// touch low-density (fog/empty) tets.
    pbd_vertex_max_density: Option<Vec<f32>>,
    primitives: Vec<Primitive>,
    next_primitive_id: u32,
    // Animation
    anim_clock: AnimationClock,
    animated_scene: Option<AnimatedScene>,
    // Deferred PBR shading
    deferred_enabled: bool,
    deferred_debug_mode: u32,
    ambient: f32,
    sky_color: [f32; 3],
    ground_color: [f32; 3],
    exposure: f32,
    ao_enabled: bool,
    ao_strength: f32,
    gtao_radius: f32,
    ssgi_enabled: bool,
    ssgi_strength: f32,
    ssgi_radius: f32,
    /// History blend factor: 1.0 = no temporal accumulation, 0.0 = frozen.
    ssgi_temporal_alpha: f32,
    ao_temporal_alpha: f32,
    ssr_temporal_alpha: f32,
    /// Previous frame's view-projection (for inline camera motion vectors).
    prev_view_proj: glam::Mat4,
    dsm_query_depth: f32,
    /// Cached light state for DSM dirty detection.
    cached_dsm_lights: Vec<rmesh_render::GpuLight>,
    cached_dsm_num_lights: u32,
    pbr_data: Option<rmesh_data::PbrData>,
    // Ray trace CPU-side state
    rt_neighbors_cpu: Vec<i32>,
    rt_start_tet_hint: i32,
    rt_locate_ms: f32,
    // Flare gun
    flare_system: FlareSystem,
}

impl App {
    fn new(
        scene: rmesh_data::SceneData,
        sh: rmesh_data::ShCoeffs,
        pbr: Option<rmesh_data::PbrData>,
    ) -> Self {
        let pos = Vec3::new(
            scene.start_pose[0],
            scene.start_pose[1],
            scene.start_pose[2],
        );
        let cam_pos = if pos.length() < 0.001 {
            Vec3::new(0.0, 3.0, -2.0)
        } else {
            pos
        };

        Self {
            window: None,
            gpu: None,
            camera: Camera::new(cam_pos),
            scene_data: scene,
            sh_coeffs: sh,
            left_pressed: false,
            middle_pressed: false,
            right_pressed: false,
            shift_pressed: false,
            last_mouse: (0.0, 0.0),
            egui_ctx: egui::Context::default(),
            frame_times: VecDeque::with_capacity(120),
            fps: 0.0,
            loaded_path: None,
            pending_load: None,
            vsync: true,
            // Default to IntervalShader when PBR data is loaded — only this
            // path runs interval_fragment.wgsl, which writes the metallic/F0
            // channels the deferred shader expects. Other modes write
            // legacy (t_min, t_max, od, dist) into aux0 and the deferred shader
            // would misread them as PBR values (metallic ≈ t_max → all green).
            render_mode: if pbr.is_some() {
                RenderMode::IntervalShader
            } else {
                RenderMode::Regular
            },
            mesh_shader_supported: false,
            show_primitives: true,
            show_scene: true,
            sort_mode: SortMode::Gpu32,
            interaction: TransformInteraction::new(),
            vertex_select: VertexSelectInteraction::new(),
            mesh_topology: None,
            pbd_pre_grab_vertices: None,
            pbd_handle_initial: Vec::new(),
            pbd_step_counter: 0,
            pbd_radius: 0.25,
            pbd_hover_vertex: None,
            pbd_vertex_max_density: None,
            primitives: Vec::new(),
            next_primitive_id: 1,
            anim_clock: AnimationClock::new(),
            animated_scene: None,
            deferred_enabled: pbr.is_some(),
            deferred_debug_mode: 0,
            ambient: 0.3,
            sky_color: [0.0, 0.0, 0.0],
            ground_color: [0.0, 0.0, 0.0],
            exposure: 1.0,
            ao_enabled: true,
            ao_strength: 1.0,
            gtao_radius: 0.5,
            ssgi_enabled: true,
            ssgi_strength: 1.0,
            ssgi_radius: 1.5,
            ssgi_temporal_alpha: 0.2,
            ao_temporal_alpha: 0.2,
            ssr_temporal_alpha: 0.2,
            prev_view_proj: glam::Mat4::IDENTITY,
            dsm_query_depth: 1.0,
            cached_dsm_lights: Vec::new(),
            cached_dsm_num_lights: 0,
            pbr_data: pbr,
            rt_neighbors_cpu: Vec::new(),
            rt_start_tet_hint: -1,
            rt_locate_ms: 0.0,
            flare_system: FlareSystem::default(),
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let t_total = std::time::Instant::now();
        let size = window.inner_size();

        log::info!("Requesting GPU adapter...");
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue, mesh_shader_supported, subgroup_supported) =
            pollster::block_on(async {
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: Some(&surface),
                        force_fallback_adapter: false,
                    })
                    .await
                    .expect("No suitable GPU adapter found");

                let info = adapter.get_info();
                log::info!(
                    "GPU: {:?} (backend: {:?}, driver: {:?})",
                    info.name,
                    info.backend,
                    info.driver_info
                );

                let adapter_features = adapter.features();
                let _backend = info.backend;
                let subgroup_supported = adapter_features.contains(wgpu::Features::SUBGROUP);
                let mesh_shader_supported = false;
                //subgroup_supported
                // && adapter_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
                // && backend != wgpu::Backend::Metal; // naga MSL backend doesn't implement mesh shaders
                log::info!("Subgroup support: {}", subgroup_supported);
                log::info!("Mesh shader support: {}", mesh_shader_supported);

                let mut required_features =
                    wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::TIMESTAMP_QUERY;
                if subgroup_supported {
                    required_features |= wgpu::Features::SUBGROUP;
                }
                if mesh_shader_supported {
                    required_features |= wgpu::Features::EXPERIMENTAL_MESH_SHADER;
                }

                let mut limits = wgpu::Limits::default();
                limits.max_storage_buffers_per_shader_stage = 20;
                limits.max_storage_buffer_binding_size = 2 * 1024 * 1024 * 1024 - 4; // 1 GB
                limits.max_buffer_size = 2 * 1024 * 1024 * 1024 - 4; // 1 GB

                // Copy mesh shader limits from adapter (they default to 0 = disabled)
                if mesh_shader_supported {
                    let supported = adapter.limits();
                    limits.max_mesh_invocations_per_workgroup =
                        supported.max_mesh_invocations_per_workgroup;
                    limits.max_mesh_invocations_per_dimension =
                        supported.max_mesh_invocations_per_dimension;
                    limits.max_mesh_output_vertices = supported.max_mesh_output_vertices;
                    limits.max_mesh_output_primitives = supported.max_mesh_output_primitives;
                    limits.max_mesh_output_layers = supported.max_mesh_output_layers;
                    limits.max_mesh_multiview_view_count = supported.max_mesh_multiview_view_count;
                    limits.max_task_mesh_workgroup_total_count =
                        supported.max_task_mesh_workgroup_total_count;
                    limits.max_task_mesh_workgroups_per_dimension =
                        supported.max_task_mesh_workgroups_per_dimension;
                }

                // SAFETY: We opt into experimental features (mesh shaders) and accept
                // that the API surface may change in future wgpu releases.
                let experimental = if mesh_shader_supported {
                    unsafe { wgpu::ExperimentalFeatures::enabled() }
                } else {
                    wgpu::ExperimentalFeatures::disabled()
                };

                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("rmesh device"),
                        required_features,
                        required_limits: limits,
                        experimental_features: experimental,
                        ..Default::default()
                    })
                    .await
                    .expect("Failed to create device");

                (
                    adapter,
                    device,
                    queue,
                    mesh_shader_supported,
                    subgroup_supported,
                )
            });
        self.mesh_shader_supported = mesh_shader_supported;
        self.render_mode = RenderMode::IntervalShader;

        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer non-sRGB surface — blit shader applies linear→sRGB manually.
        // Using an sRGB surface would double-gamma.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        };
        surface.configure(&device, &surface_config);

        let color_format = wgpu::TextureFormat::Rgba16Float;

        log::info!("Compiling shader pipelines...");
        let t0 = std::time::Instant::now();
        let pipelines = ForwardPipelines::new(&device, color_format);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);
        let mesh_pipelines = if mesh_shader_supported {
            log::info!("Compiling mesh shader pipelines...");
            Some(MeshForwardPipelines::new(&device, color_format))
        } else {
            None
        };
        let interval_pipelines = if mesh_shader_supported {
            log::info!("Compiling interval shading pipelines...");
            Some(IntervalPipelines::new(&device, color_format))
        } else {
            None
        };
        let compute_interval_pipelines = ComputeIntervalPipelines::new(&device, color_format);
        log::info!("Pipelines compiled: {:.2}s", t0.elapsed().as_secs_f64());

        log::info!(
            "Uploading scene buffers ({} tets)...",
            self.scene_data.tet_count
        );
        let t0 = std::time::Instant::now();
        let buffers = SceneBuffers::upload(&device, &queue, &self.scene_data);

        // Allocate base_colors with zeros — GPU SH eval will fill them
        let zero_colors = vec![0.0f32; self.scene_data.tet_count as usize * 3];
        let material = MaterialBuffers::upload(
            &device,
            &zero_colors,
            &self.scene_data.color_grads,
            self.scene_data.tet_count,
        );

        // Upload SH coefficients to GPU as f16-packed u32 array
        let sh_total_dims =
            ((self.sh_coeffs.degree + 1) * (self.sh_coeffs.degree + 1)) as usize * 3;
        let sh_coeffs_packed = pack_sh_coeffs_f16(&self.sh_coeffs.coeffs, sh_total_dims);
        let sh_coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh_coeffs"),
            contents: bytemuck::cast_slice(&sh_coeffs_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sh_degree = self.sh_coeffs.degree;

        // Upload PBR aux data if available: [M * 6] f32 packed as
        //   [roughness, metallic, f0_dielectric, albedo.r, albedo.g, albedo.b]
        // Matches the parametric BRDF format produced by convert.py.
        let aux_data_buf = if let Some(ref pbr) = self.pbr_data {
            let tet_count = self.scene_data.tet_count as usize;
            require_parametric_pbr(pbr, tet_count);
            let mut aux = vec![0.0f32; tet_count * 6];
            for t in 0..tet_count {
                aux[t * 6] = pbr.roughness[t];
                aux[t * 6 + 1] = pbr.metallic[t];
                aux[t * 6 + 2] = pbr.f0_dielectric[t];
                for c in 0..3 {
                    aux[t * 6 + 3 + c] = pbr.albedo[t * 3 + c];
                }
            }
            // Override vertex normals from PBR data
            if !pbr.vertex_normals.is_empty() {
                queue.write_buffer(
                    &buffers.vertex_normals,
                    0,
                    bytemuck::cast_slice(&pbr.vertex_normals),
                );
            }
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("aux_data"),
                contents: bytemuck::cast_slice(&aux),
                usage: wgpu::BufferUsages::STORAGE,
            });
            log::info!("Uploaded PBR aux data: {} tets × 8 channels", tet_count);
            Some(buf)
        } else {
            None
        };

        log::info!("Buffers uploaded: {:.2}s", t0.elapsed().as_secs_f64());

        let targets = RenderTargets::new(&device, size.width.max(1), size.height.max(1));

        // Radix sort state — use DRS (subgroup-based) when available, else Basic
        let sort_backend = if subgroup_supported {
            rmesh_sort::SortBackend::Drs
        } else {
            rmesh_sort::SortBackend::Basic
        };
        log::info!(
            "Creating radix sort pipelines (backend: {:?})...",
            sort_backend
        );
        let t0 = std::time::Instant::now();
        let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device, 1, sort_backend);
        let n_pow2 = self.scene_data.tet_count.next_power_of_two();
        let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32, 1, sort_backend);
        sort_state.upload_configs(&queue);
        let sort_state_16bit =
            rmesh_sort::RadixSortState::new(&device, n_pow2, 16, 1, sort_backend);
        sort_state_16bit.upload_configs(&queue);
        log::info!("Sort pipelines: {:.2}s", t0.elapsed().as_secs_f64());

        // Fluid simulation: lazily initialized when first enabled (saves ~500MB GPU memory)

        let compute_bg =
            create_compute_bind_group(&device, &pipelines, &buffers, &material, &sh_coeffs_buf);
        let hw_compute_bg =
            create_hw_compute_bind_group(&device, &pipelines, &buffers, &material, &sh_coeffs_buf);
        let render_bg = create_render_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg_b = create_render_bind_group_with_sort_values(
            &device,
            &pipelines,
            &buffers,
            &material,
            sort_state.values_b(),
        );

        // Mesh shader bind groups (if supported)
        let (mesh_render_bg_a, mesh_render_bg_b, indirect_convert_bg) =
            if let Some(ref mp) = mesh_pipelines {
                let a = create_mesh_render_bind_group(&device, mp, &buffers, &material);
                let b = create_mesh_render_bind_group_with_sort_values(
                    &device,
                    mp,
                    &buffers,
                    &material,
                    sort_state.values_b(),
                );
                let ic = create_indirect_convert_bind_group(&device, mp, &buffers);
                (Some(a), Some(b), Some(ic))
            } else {
                (None, None, None)
            };

        // Interval shading bind groups (if supported)
        let (interval_render_bg_a, interval_render_bg_b, interval_indirect_convert_bg) =
            if let Some(ref ip) = interval_pipelines {
                let a = create_interval_render_bind_group(&device, ip, &buffers, &material);
                let b = create_interval_render_bind_group_with_sort_values(
                    &device,
                    ip,
                    &buffers,
                    &material,
                    sort_state.values_b(),
                );
                let ic = create_interval_indirect_convert_bind_group(&device, ip, &buffers);
                (Some(a), Some(b), Some(ic))
            } else {
                (None, None, None)
            };

        // Compute-interval bind groups (always available)
        let compute_interval_gen_bg_a = create_compute_interval_gen_bind_group(
            &device,
            &compute_interval_pipelines,
            &buffers,
            &material,
        );
        let compute_interval_gen_bg_b = create_compute_interval_gen_bind_group_with_sort_values(
            &device,
            &compute_interval_pipelines,
            &buffers,
            &material,
            sort_state.values_b(),
        );
        // 16-bit sort variant: gen bind groups use sort_state_16bit's values_b
        let compute_interval_gen_bg_a_16bit = create_compute_interval_gen_bind_group(
            &device,
            &compute_interval_pipelines,
            &buffers,
            &material,
        );
        let compute_interval_gen_bg_b_16bit =
            create_compute_interval_gen_bind_group_with_sort_values(
                &device,
                &compute_interval_pipelines,
                &buffers,
                &material,
                sort_state_16bit.values_b(),
            );
        let compute_interval_render_bg = if let Some(ref aux_buf) = aux_data_buf {
            rmesh_render::create_compute_interval_render_bind_group_pbr(
                &device,
                &compute_interval_pipelines,
                &buffers,
                aux_buf,
                &buffers.indices,
            )
        } else {
            create_compute_interval_render_bind_group(
                &device,
                &compute_interval_pipelines,
                &buffers,
            )
        };
        let compute_interval_convert_bg = create_compute_interval_indirect_convert_bind_group(
            &device,
            &compute_interval_pipelines,
            &buffers,
        );

        // Quad prepass + render bind groups (A/B for sort result location)
        let prepass_bg_a = create_prepass_bind_group(
            &device,
            &pipelines,
            &buffers,
            &material,
            &buffers.sort_values,
        );
        let prepass_bg_b = create_prepass_bind_group(
            &device,
            &pipelines,
            &buffers,
            &material,
            sort_state.values_b(),
        );
        let quad_render_bg = create_quad_render_bind_group(&device, &pipelines, &buffers);

        let blit_bg = create_blit_bind_group(&device, &blit_pipeline, &targets.color_view);

        // Primitive setup (depth used for hardware early-z culling in forward pass)
        let mut primitive_geometry = PrimitiveGeometry::new(&device);
        let mut material_registry = rmesh_compositor::MaterialRegistry::new(&device, &queue);
        let primitive_pipeline =
            PrimitivePipeline::new(&device, &material_registry.bind_group_layout);
        let primitive_targets =
            PrimitiveTargets::new(&device, size.width.max(1), size.height.max(1));

        // Instance count readback buffer
        let instance_count_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_count_readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // GPU timestamp profiling
        let ts_period_ns = queue.get_timestamp_period();
        let ts_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: TS_QUERY_COUNT,
        });
        let ts_resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_resolve"),
            size: (TS_QUERY_COUNT as u64) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ts_readback = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if i == 0 {
                    "ts_readback_a"
                } else {
                    "ts_readback_b"
                }),
                size: (TS_QUERY_COUNT as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let ts_readback_ready: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2] =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let ts_readback_mapped: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2] =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));

        // egui setup
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // Hi-Z mip pyramid over fused linear-Z. Built each frame; feeds GTAO
        // (and later SSGI). Pipelines are global; texture + bind groups are
        // resized when the surface size changes.
        let hiz_pipelines = rmesh_postprocess::HizPipelines::new(&device);
        let hiz_texture =
            rmesh_postprocess::HizTexture::new(&device, size.width.max(1), size.height.max(1));
        let hiz_linearize_bg = rmesh_postprocess::create_hiz_linearize_bind_group(
            &device,
            &hiz_pipelines,
            &primitive_targets.depth_view,
            &targets.depth_view,
        );
        let hiz_downsample_bgs: Vec<wgpu::BindGroup> = (0..(hiz_texture.mip_count as usize - 1))
            .map(|i| {
                rmesh_postprocess::create_hiz_downsample_bind_group(
                    &device,
                    &hiz_pipelines,
                    &hiz_texture.mip_views[i],
                )
            })
            .collect();

        // GTAO pass — always created. AO target is in `targets`. Reads Hi-Z
        // for fused depth (sample any mip) and the volume MRT for std.
        let gtao_pipeline = rmesh_postprocess::GtaoPipeline::new(&device);
        let gtao_bg = rmesh_postprocess::create_gtao_bind_group(
            &device,
            &gtao_pipeline,
            &hiz_texture.full_view,
            &targets.normals_view,
            &targets.depth_view,
        );

        // AO bilateral blur (separable, depth+normal aware). With temporal
        // accumulation in front of it, H reads ao_temporal_view (the
        // temporal-blended AO) and V writes back to ao_view (final).
        let ao_blur_pipeline = rmesh_postprocess::AoBlurPipeline::new(&device);
        let ao_blur_bg_h = rmesh_postprocess::create_ao_blur_bind_group(
            &device,
            &ao_blur_pipeline,
            &ao_blur_pipeline.uniforms_h,
            &targets.ao_temporal_view,
            &hiz_texture.full_view,
            &targets.normals_view,
        );
        let ao_blur_bg_v = rmesh_postprocess::create_ao_blur_bind_group(
            &device,
            &ao_blur_pipeline,
            &ao_blur_pipeline.uniforms_v,
            &targets.ao_blur_temp_view,
            &hiz_texture.full_view,
            &targets.normals_view,
        );

        // SSGI: compute (Hi-Z ray march, samples lit_history) + bilateral denoise.
        // lit_history rotates → bind group doubled, indexed by parity.
        let ssgi_pipeline = rmesh_postprocess::SsgiPipeline::new(&device);
        let ssgi_bg: [wgpu::BindGroup; 2] = std::array::from_fn(|p| {
            rmesh_postprocess::create_ssgi_bind_group(
                &device,
                &ssgi_pipeline,
                &hiz_texture.full_view,
                &targets.normals_view,
                targets.lit_history(p as u32),
            )
        });
        // SSGI blur reads the temporal-blended SSGI (ssgi_temporal_view), not
        // the raw ssgi_view. V still writes back to ssgi_view (final).
        let ssgi_blur_pipeline = rmesh_postprocess::SsgiBlurPipeline::new(&device);
        let ssgi_blur_bg_h = rmesh_postprocess::create_ssgi_blur_bind_group(
            &device,
            &ssgi_blur_pipeline,
            &ssgi_blur_pipeline.uniforms_h,
            &targets.ssgi_temporal_view,
            &hiz_texture.full_view,
            &targets.normals_view,
        );
        let ssgi_blur_bg_v = rmesh_postprocess::create_ssgi_blur_bind_group(
            &device,
            &ssgi_blur_pipeline,
            &ssgi_blur_pipeline.uniforms_v,
            &targets.ssgi_blur_temp_view,
            &hiz_texture.full_view,
            &targets.normals_view,
        );

        // Temporal accumulation pipelines (one per output format). Bind groups
        // doubled for parity: each variant binds (current, history) in the
        // orientation appropriate for that frame.
        let ssgi_temporal_pipeline =
            rmesh_postprocess::TemporalPipeline::new(&device, wgpu::TextureFormat::Rgba16Float);
        let ssgi_temporal_bg: [wgpu::BindGroup; 2] = std::array::from_fn(|p| {
            rmesh_postprocess::create_temporal_bind_group(
                &device,
                &ssgi_temporal_pipeline,
                targets.ssgi_current(p as u32),
                targets.ssgi_history(p as u32),
                &hiz_texture.full_view,
            )
        });
        let ao_temporal_pipeline =
            rmesh_postprocess::TemporalPipeline::new(&device, wgpu::TextureFormat::R8Unorm);
        let ao_temporal_bg: [wgpu::BindGroup; 2] = std::array::from_fn(|p| {
            rmesh_postprocess::create_temporal_bind_group(
                &device,
                &ao_temporal_pipeline,
                targets.ao_current(p as u32),
                targets.ao_history(p as u32),
                &hiz_texture.full_view,
            )
        });

        // SSR pipeline + temporal (third TemporalPipeline instance, Rgba16Float).
        // lit_history + ssgi_current rotate → ssr_bg doubled. ssr_current /
        // ssr_history rotate → ssr_temporal_bg doubled.
        let ssr_pipeline = rmesh_postprocess::SsrPipeline::new(&device);
        let ssr_bg: [wgpu::BindGroup; 2] = std::array::from_fn(|p| {
            rmesh_postprocess::create_ssr_bind_group(
                &device,
                &ssr_pipeline,
                &hiz_texture.full_view,
                &targets.normals_view,
                targets.lit_history(p as u32),
                targets.ssgi_current(p as u32),
                &targets.depth_view,
            )
        });
        let ssr_temporal_pipeline =
            rmesh_postprocess::TemporalPipeline::new(&device, wgpu::TextureFormat::Rgba16Float);
        let ssr_temporal_bg: [wgpu::BindGroup; 2] = std::array::from_fn(|p| {
            rmesh_postprocess::create_temporal_bind_group(
                &device,
                &ssr_temporal_pipeline,
                targets.ssr_current(p as u32),
                targets.ssr_history(p as u32),
                &hiz_texture.full_view,
            )
        });

        // Clear history textures on first frame so SSGI/AO/SSR temporal reads
        // are deterministic. Without this, contents are vendor-defined.
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("history_init_clear"),
            });
            let black = wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            };
            // Clear both slots of each ping-pong pair so whichever slot is
            // "history" on frame 0 is deterministic regardless of parity.
            for p in 0..2u32 {
                rmesh_postprocess::clear_texture_view(&mut enc, targets.lit_history(p), black);
                rmesh_postprocess::clear_texture_view(&mut enc, targets.ssgi_history(p), black);
                rmesh_postprocess::clear_texture_view(&mut enc, targets.ao_history(p), black);
                rmesh_postprocess::clear_texture_view(&mut enc, targets.ssr_history(p), black);
                rmesh_postprocess::clear_texture_view(&mut enc, targets.ssr_current(p), black);
            }
            queue.submit(std::iter::once(enc.finish()));
        }

        // Deferred PBR shading pipeline (only when PBR data is loaded)
        let has_pbr = self.pbr_data.is_some();
        let (
            deferred_pipeline,
            deferred_bg,
            deferred_output,
            deferred_output_view,
            deferred_blit_bg,
            deferred_dsm_dummy_bg,
        ) = if has_pbr {
            log::info!("Creating deferred PBR shading pipeline...");
            let dp = rmesh_postprocess::DeferredShadePipeline::new(&device, color_format);
            // Deferred BG doubled for parity: ao/ssgi/ssr_current rotate, lit_history rotates.
            let bg: [wgpu::BindGroup; 2] = std::array::from_fn(|p| {
                rmesh_postprocess::create_deferred_bind_group(
                    &device,
                    &dp,
                    &targets.color_view,
                    &targets.aux0_view,
                    &targets.normals_view,
                    &targets.depth_view,
                    &primitive_targets.depth_view,
                    targets.ao_current(p as u32),
                    targets.ssgi_current(p as u32),
                    targets.lit_history(p as u32),
                    targets.ssr_current(p as u32),
                )
            });
            // Dummy DSM bind group (1x1 atlas, no lights)
            let dummy_atlas = rmesh_dsm::DsmAtlas::new_dummy(&device);
            let dummy_dsm_bg = rmesh_postprocess::create_deferred_dsm_bind_group(
                &device,
                &dp,
                &dummy_atlas.cubemap_views[0],
                &dummy_atlas.meta_buf,
            );
            // Separate output texture (can't read+write color_view in same pass)
            let out_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("deferred_output"),
                size: wgpu::Extent3d {
                    width: size.width.max(1),
                    height: size.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let d_blit_bg = create_blit_bind_group(&device, &blit_pipeline, &out_view);
            (
                Some(dp),
                Some(bg),
                Some(out_tex),
                Some(out_view),
                Some(d_blit_bg),
                Some(dummy_dsm_bg),
            )
        } else {
            (None, None, None, None, None, None)
        };

        // Ray trace pipeline — DIAGNOSTIC: stubbed to keep init fast; real path is RenderMode::RayTrace only
        log::info!("Building ray trace data... (STUBBED FOR DIAGNOSTIC)");
        let t0 = std::time::Instant::now();
        let rt_neighbors: Vec<i32> = vec![-1];
        let rt_bvh = rmesh_raytrace::BVHData {
            nodes: Vec::new(),
            boundary_faces: Vec::new(),
        };
        let rt_pipeline =
            rmesh_raytrace::RayTracePipeline::new(&device, size.width.max(1), size.height.max(1), 0);
        let rt_buffers = rmesh_raytrace::RayTraceBuffers::new(&device, &rt_neighbors, &rt_bvh);
        let start_tet: i32 = -1; // DIAGNOSTIC: skip find_containing_tet
        queue.write_buffer(&rt_buffers.start_tet, 0, bytemuck::cast_slice(&[start_tet]));
        self.rt_neighbors_cpu = rt_neighbors;
        self.rt_start_tet_hint = start_tet;
        let rt_bg = rmesh_raytrace::create_raytrace_bind_group(
            &device,
            &rt_pipeline,
            &buffers,
            &material,
            &rt_buffers,
        );
        let (rt_texture, rt_texture_view) =
            create_rt_texture(&device, size.width.max(1), size.height.max(1));
        let rt_blit_pipeline = rmesh_render::BlitPipelineNonFiltering::new(&device, surface_format);
        let rt_blit_bg =
            rmesh_render::create_blit_nf_bind_group(&device, &rt_blit_pipeline, &rt_texture_view);
        log::info!("Ray trace data: {:.2}s", t0.elapsed().as_secs_f64());

        // DSM debug view (2-moment deep shadow map from camera perspective)
        let dsm_pipeline = rmesh_dsm::DsmPipeline::new(&device, color_format);
        let dsm_prim_pipeline = rmesh_dsm::DsmPrimitivePipeline::new(&device);
        let dsm_project_pipeline = rmesh_dsm::DsmProjectPipeline::new(&device);
        let dsm_resolve_pipeline = rmesh_dsm::DsmResolvePipeline::new(&device, color_format);
        let (
            dsm_moments_texture,
            dsm_moments_view,
            dsm_depth_texture,
            dsm_depth_view,
            dsm_resolve_output,
            dsm_resolve_output_view,
        ) = {
            let w = size.width.max(1);
            let h = size.height.max(1);
            let mtex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dsm_moments"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let mview = mtex.create_view(&wgpu::TextureViewDescriptor::default());
            let dtex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dsm_debug_depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());
            let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dsm_resolve_output"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let resolve_view = resolve_tex.create_view(&wgpu::TextureViewDescriptor::default());
            (mtex, mview, dtex, dview, resolve_tex, resolve_view)
        };
        let dsm_render_bg = rmesh_dsm::create_dsm_render_bind_group(
            &device,
            &dsm_pipeline,
            &buffers.uniforms,
            &buffers.interval_vertex_buf,
            &buffers.interval_tet_data_buf,
        );
        let dsm_resolve_bg = rmesh_dsm::create_dsm_resolve_bind_group(
            &device,
            &dsm_resolve_pipeline,
            &dsm_moments_view,
        );
        let dsm_blit_bg = create_blit_bind_group(&device, &blit_pipeline, &dsm_resolve_output_view);

        // Load flare 3D model and upload its meshes/materials
        {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()));
            let candidates = [
                std::path::PathBuf::from("assets/flare/scene.gltf"),
                std::path::PathBuf::from("../assets/flare/scene.gltf"),
            ];
            let extra: Vec<std::path::PathBuf> = exe_dir
                .iter()
                .map(|d| d.join("assets/flare/scene.gltf"))
                .collect();
            let flare_path = candidates.iter().chain(extra.iter()).find(|p| p.exists());
            if let Some(path) = flare_path {
                if let Some(gltf_scene) = self.flare_system.load_model(path) {
                    let prim_verts: Vec<Vec<rmesh_compositor::PrimitiveVertex>> = gltf_scene
                        .meshes
                        .iter()
                        .map(|m| {
                            (0..m.vertices.len())
                                .map(|i| rmesh_compositor::PrimitiveVertex {
                                    position: m.vertices[i],
                                    normal: m.normals[i],
                                    uv: if i < m.uvs.len() {
                                        m.uvs[i]
                                    } else {
                                        [0.0, 0.0]
                                    },
                                    tangent: if i < m.tangents.len() {
                                        m.tangents[i]
                                    } else {
                                        [1.0, 0.0, 0.0, 1.0]
                                    },
                                })
                                .collect()
                        })
                        .collect();
                    self.flare_system.flare_mesh_base = 0;
                    primitive_geometry.set_custom_meshes(&device, &prim_verts);

                    // Upload flare material textures
                    let tex_data: Vec<rmesh_compositor::TextureData> = gltf_scene
                        .textures
                        .iter()
                        .map(|t| rmesh_compositor::TextureData {
                            width: t.width,
                            height: t.height,
                            pixels: t.pixels.clone(),
                        })
                        .collect();
                    let mat_defs: Vec<rmesh_compositor::MaterialDef> = gltf_scene
                        .materials
                        .iter()
                        .map(|m| rmesh_compositor::MaterialDef {
                            base_color_factor: m.base_color_factor,
                            roughness_factor: m.roughness_factor,
                            metallic_factor: m.metallic_factor,
                            occlusion_strength: m.occlusion_strength,
                            normal_scale: m.normal_scale,
                            base_color_texture: m.base_color_texture,
                            metallic_roughness_texture: m.metallic_roughness_texture,
                            normal_texture: m.normal_texture,
                            occlusion_texture: m.occlusion_texture,
                        })
                        .collect();
                    material_registry.upload(&device, &queue, &tex_data, &mat_defs);
                }
            } else {
                log::warn!("Flare model not found, using default sphere");
            }
        }

        log::info!("GPU init total: {:.2}s", t_total.elapsed().as_secs_f64());

        // Chunk size = enough for one sort_values write at typical scene
        // scale (4.8M tets × 4 B ≈ 19 MB) with headroom. The belt recycles
        // chunks across frames so this is a one-shot allocation.
        let staging_belt = wgpu::util::StagingBelt::new(device.clone(), 32 * 1024 * 1024);

        let pbd_pipelines = rmesh_pbd::PbdPipelines::new(&device);

        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            surface_config,
            pipelines,
            blit_pipeline,
            sort_pipelines,
            sort_state,
            sort_state_16bit,
            sort_backend,
            cpu_sorter: rmesh_sort::CpuSorter::new(self.scene_data.tet_count as usize),
            staging_belt,
            buffers,
            material_buffers: material,
            targets,
            compute_bg,
            hw_compute_bg,
            render_bg,
            render_bg_b,
            blit_bg,
            tet_count: self.scene_data.tet_count,
            sh_coeffs_buf,
            sh_degree,
            pending_reconfigure: false,
            mesh_pipelines,
            mesh_render_bg_a,
            mesh_render_bg_b,
            indirect_convert_bg,
            interval_pipelines,
            interval_render_bg_a,
            interval_render_bg_b,
            interval_indirect_convert_bg,
            compute_interval_pipelines,
            compute_interval_gen_bg_a,
            compute_interval_gen_bg_b,
            compute_interval_gen_bg_a_16bit,
            compute_interval_gen_bg_b_16bit,
            compute_interval_render_bg,
            compute_interval_convert_bg,
            prepass_bg_a,
            prepass_bg_b,
            quad_render_bg,
            pbd_solver: None,
            pbd_pipelines,
            tet_neighbors_buf: None,
            primitive_geometry,
            material_registry,
            primitive_pipeline,
            primitive_targets,
            egui_renderer,
            egui_state,
            instance_count_readback,
            visible_instance_count: 0,
            ts_query_set,
            ts_resolve_buf,
            ts_readback,
            ts_readback_ready,
            ts_readback_mapped,
            ts_frame: 0,
            ts_period_ns,
            gpu_times_ms: GpuTimings::default(),
            cpu_times_ms: CpuTimings::default(),
            instance_count_ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            instance_count_mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            aux_data_buf,
            deferred_pipeline,
            deferred_bg,
            gtao_pipeline,
            gtao_bg,
            hiz_pipelines,
            hiz_texture,
            hiz_linearize_bg,
            hiz_downsample_bgs,
            ao_blur_pipeline,
            ao_blur_bg_h,
            ao_blur_bg_v,
            ssgi_pipeline,
            ssgi_bg,
            ssgi_blur_pipeline,
            ssgi_blur_bg_h,
            ssgi_blur_bg_v,
            ssgi_temporal_pipeline,
            ssgi_temporal_bg,
            ao_temporal_pipeline,
            ao_temporal_bg,
            ssr_pipeline,
            ssr_bg,
            ssr_temporal_pipeline,
            ssr_temporal_bg,
            frame_counter: 0,
            frame_parity: 0,
            deferred_output,
            deferred_output_view,
            deferred_blit_bg,
            has_pbr_data: has_pbr,
            dsm_pipeline,
            dsm_prim_pipeline,
            dsm_project_pipeline,
            dsm_resolve_pipeline,
            dsm_moments_texture,
            dsm_moments_view,
            dsm_depth_texture,
            dsm_depth_view,
            dsm_resolve_output,
            dsm_resolve_output_view,
            dsm_render_bg,
            dsm_resolve_bg,
            dsm_blit_bg,
            dsm_atlas: None,
            deferred_dsm_bg: None,
            deferred_dsm_dummy_bg,
            rt_pipeline,
            rt_buffers,
            rt_bg,
            rt_texture,
            rt_texture_view,
            rt_blit_pipeline,
            rt_blit_bg,
        });
    }

    // render() and update_fps() are in render.rs
    // (The old ~720-line render body has been deleted from this file)

    fn load_file(&mut self, path: &std::path::Path) {
        log::info!("Loading: {}", path.display());
        let file_data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to read {}: {}", path.display(), e);
                return;
            }
        };

        let is_ply = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("ply"));

        let (scene, sh, pbr) = if is_ply {
            match rmesh_data::load_ply(&file_data) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Failed to parse PLY: {}", e);
                    return;
                }
            }
        } else {
            match rmesh_data::load_rmesh(&file_data)
                .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
            {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Failed to parse scene: {}", e);
                    return;
                }
            }
        };

        log::info!(
            "Loaded: {} vertices, {} tets, SH degree {}",
            scene.vertex_count,
            scene.tet_count,
            sh.degree,
        );

        // Update scene data
        self.scene_data = scene;
        self.sh_coeffs = sh;
        self.pbr_data = pbr;
        // Drop topology/grab state from the previous scene.
        self.mesh_topology = None;
        self.pbd_vertex_max_density = None;
        self.pbd_pre_grab_vertices = None;
        self.pbd_handle_initial.clear();
        self.vertex_select.set_enabled(false);

        // Cache the flare collision mesh up front for the new scene (drops the
        // stale mesh from the previous scene and avoids a first-flare hitch).
        self.flare_system.rebuild_collision_mesh(&self.scene_data);
        self.deferred_enabled = self.pbr_data.is_some();
        self.loaded_path = Some(path.to_path_buf());

        // Reset camera
        let pos = Vec3::new(
            self.scene_data.start_pose[0],
            self.scene_data.start_pose[1],
            self.scene_data.start_pose[2],
        );
        if pos.length() > 0.001 {
            self.camera = Camera::new(pos);
        }

        // Rebuild GPU buffers
        if let Some(gpu) = &mut self.gpu {
            gpu.buffers = SceneBuffers::upload(&gpu.device, &gpu.queue, &self.scene_data);

            let zero_colors = vec![0.0f32; self.scene_data.tet_count as usize * 3];
            gpu.material_buffers = MaterialBuffers::upload(
                &gpu.device,
                &zero_colors,
                &self.scene_data.color_grads,
                self.scene_data.tet_count,
            );
            gpu.tet_count = self.scene_data.tet_count;

            // Recreate SH coeffs buffer (f16-packed)
            let sh_total_dims =
                ((self.sh_coeffs.degree + 1) * (self.sh_coeffs.degree + 1)) as usize * 3;
            let sh_coeffs_packed = pack_sh_coeffs_f16(&self.sh_coeffs.coeffs, sh_total_dims);
            gpu.sh_coeffs_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sh_coeffs"),
                    contents: bytemuck::cast_slice(&sh_coeffs_packed),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            gpu.sh_degree = self.sh_coeffs.degree;

            // Recreate sort state for new tet count
            let n_pow2 = gpu.tet_count.next_power_of_two();
            gpu.sort_state =
                rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 32, 1, gpu.sort_backend);
            gpu.sort_state.upload_configs(&gpu.queue);
            gpu.sort_state_16bit =
                rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 16, 1, gpu.sort_backend);
            gpu.sort_state_16bit.upload_configs(&gpu.queue);

            // Recreate bind groups
            gpu.compute_bg = create_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
            );
            gpu.hw_compute_bg = create_hw_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
            );
            gpu.render_bg = create_render_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
            );
            gpu.render_bg_b = create_render_bind_group_with_sort_values(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                gpu.sort_state.values_b(),
            );

            // Recreate quad prepass + render bind groups
            gpu.prepass_bg_a = create_prepass_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.buffers.sort_values,
            );
            gpu.prepass_bg_b = create_prepass_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                gpu.sort_state.values_b(),
            );
            gpu.quad_render_bg =
                create_quad_render_bind_group(&gpu.device, &gpu.pipelines, &gpu.buffers);

            // Recreate mesh shader bind groups
            if let Some(ref mp) = gpu.mesh_pipelines {
                gpu.mesh_render_bg_a = Some(create_mesh_render_bind_group(
                    &gpu.device,
                    mp,
                    &gpu.buffers,
                    &gpu.material_buffers,
                ));
                gpu.mesh_render_bg_b = Some(create_mesh_render_bind_group_with_sort_values(
                    &gpu.device,
                    mp,
                    &gpu.buffers,
                    &gpu.material_buffers,
                    gpu.sort_state.values_b(),
                ));
                gpu.indirect_convert_bg = Some(create_indirect_convert_bind_group(
                    &gpu.device,
                    mp,
                    &gpu.buffers,
                ));
            }

            // Recreate interval shading bind groups
            if let Some(ref ip) = gpu.interval_pipelines {
                gpu.interval_render_bg_a = Some(create_interval_render_bind_group(
                    &gpu.device,
                    ip,
                    &gpu.buffers,
                    &gpu.material_buffers,
                ));
                gpu.interval_render_bg_b =
                    Some(create_interval_render_bind_group_with_sort_values(
                        &gpu.device,
                        ip,
                        &gpu.buffers,
                        &gpu.material_buffers,
                        gpu.sort_state.values_b(),
                    ));
                gpu.interval_indirect_convert_bg = Some(
                    create_interval_indirect_convert_bind_group(&gpu.device, ip, &gpu.buffers),
                );
            }

            // Recreate compute-interval bind groups
            gpu.compute_interval_gen_bg_a = create_compute_interval_gen_bind_group(
                &gpu.device,
                &gpu.compute_interval_pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
            );
            gpu.compute_interval_gen_bg_b = create_compute_interval_gen_bind_group_with_sort_values(
                &gpu.device,
                &gpu.compute_interval_pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                gpu.sort_state.values_b(),
            );
            gpu.compute_interval_gen_bg_a_16bit = create_compute_interval_gen_bind_group(
                &gpu.device,
                &gpu.compute_interval_pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
            );
            gpu.compute_interval_gen_bg_b_16bit =
                create_compute_interval_gen_bind_group_with_sort_values(
                    &gpu.device,
                    &gpu.compute_interval_pipelines,
                    &gpu.buffers,
                    &gpu.material_buffers,
                    gpu.sort_state_16bit.values_b(),
                );
            // Upload PBR aux data if available
            gpu.has_pbr_data = self.pbr_data.is_some();
            if let Some(ref pbr) = self.pbr_data {
                // Aux layout: [roughness, metallic, f0_dielectric, albedo.r, albedo.g, albedo.b]
                let tc = self.scene_data.tet_count as usize;
                require_parametric_pbr(pbr, tc);
                let mut aux = vec![0.0f32; tc * 6];
                for t in 0..tc {
                    aux[t * 6] = pbr.roughness[t];
                    aux[t * 6 + 1] = pbr.metallic[t];
                    aux[t * 6 + 2] = pbr.f0_dielectric[t];
                    for c in 0..3 {
                        aux[t * 6 + 3 + c] = pbr.albedo[t * 3 + c];
                    }
                }
                if !pbr.vertex_normals.is_empty() {
                    gpu.queue.write_buffer(
                        &gpu.buffers.vertex_normals,
                        0,
                        bytemuck::cast_slice(&pbr.vertex_normals),
                    );
                }
                let aux_buf = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("aux_data"),
                        contents: bytemuck::cast_slice(&aux),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                gpu.compute_interval_render_bg =
                    rmesh_render::create_compute_interval_render_bind_group_pbr(
                        &gpu.device,
                        &gpu.compute_interval_pipelines,
                        &gpu.buffers,
                        &aux_buf,
                        &gpu.buffers.indices,
                    );
                gpu.aux_data_buf = Some(aux_buf);

                // Recreate deferred pipeline
                let color_format = wgpu::TextureFormat::Rgba16Float;
                let dp = rmesh_postprocess::DeferredShadePipeline::new(&gpu.device, color_format);
                gpu.deferred_bg = Some(std::array::from_fn(|p| {
                    rmesh_postprocess::create_deferred_bind_group(
                        &gpu.device,
                        &dp,
                        &gpu.targets.color_view,
                        &gpu.targets.aux0_view,
                        &gpu.targets.normals_view,
                        &gpu.targets.depth_view,
                        &gpu.primitive_targets.depth_view,
                        gpu.targets.ao_current(p as u32),
                        gpu.targets.ssgi_current(p as u32),
                        gpu.targets.lit_history(p as u32),
                        gpu.targets.ssr_current(p as u32),
                    )
                }));
                // Reset DSM state (will be regenerated on next frame with lights)
                let dummy_atlas = rmesh_dsm::DsmAtlas::new_dummy(&gpu.device);
                gpu.deferred_dsm_dummy_bg = Some(rmesh_postprocess::create_deferred_dsm_bind_group(
                    &gpu.device,
                    &dp,
                    &dummy_atlas.cubemap_views[0],
                    &dummy_atlas.meta_buf,
                ));
                gpu.deferred_dsm_bg = None;
                gpu.dsm_atlas = None;
                self.cached_dsm_lights.clear();
                self.cached_dsm_num_lights = 0;
                let w = gpu.surface_config.width;
                let h = gpu.surface_config.height;
                let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("deferred_output"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.deferred_blit_bg = Some(create_blit_bind_group(
                    &gpu.device,
                    &gpu.blit_pipeline,
                    &out_view,
                ));
                gpu.deferred_output = Some(out_tex);
                gpu.deferred_output_view = Some(out_view);
                gpu.deferred_pipeline = Some(dp);
            } else {
                gpu.compute_interval_render_bg = create_compute_interval_render_bind_group(
                    &gpu.device,
                    &gpu.compute_interval_pipelines,
                    &gpu.buffers,
                );
                gpu.aux_data_buf = None;
                gpu.deferred_pipeline = None;
                gpu.deferred_bg = None;
                gpu.deferred_output = None;
                gpu.deferred_output_view = None;
                gpu.deferred_blit_bg = None;
            }
            gpu.compute_interval_convert_bg = create_compute_interval_indirect_convert_bind_group(
                &gpu.device,
                &gpu.compute_interval_pipelines,
                &gpu.buffers,
            );

            // Recreate ray trace neighbors + state
            let neighbors = rmesh_raytrace::compute_tet_neighbors(
                &self.scene_data.indices,
                self.scene_data.tet_count as usize,
            );
            let tet_neighbors_buf =
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("tet_neighbors"),
                        contents: bytemuck::cast_slice(&neighbors),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            gpu.pbd_solver = None;
            gpu.tet_neighbors_buf = Some(tet_neighbors_buf);


            let rt_bvh = rmesh_raytrace::build_boundary_bvh(
                &self.scene_data.vertices,
                &self.scene_data.indices,
                &neighbors,
                self.scene_data.tet_count as usize,
            );
            let w = gpu.surface_config.width;
            let h = gpu.surface_config.height;
            gpu.rt_pipeline = rmesh_raytrace::RayTracePipeline::new(&gpu.device, w, h, 0);
            gpu.rt_buffers = rmesh_raytrace::RayTraceBuffers::new(&gpu.device, &neighbors, &rt_bvh);
            let start_tet = rmesh_raytrace::find_containing_tet(
                &self.scene_data.vertices,
                &self.scene_data.indices,
                self.scene_data.tet_count as usize,
                self.camera.position,
            )
            .map(|t| t as i32)
            .unwrap_or(-1);
            gpu.queue.write_buffer(
                &gpu.rt_buffers.start_tet,
                0,
                bytemuck::cast_slice(&[start_tet]),
            );
            self.rt_neighbors_cpu = neighbors.clone();
            self.rt_start_tet_hint = start_tet;
            gpu.rt_bg = rmesh_raytrace::create_raytrace_bind_group(
                &gpu.device,
                &gpu.rt_pipeline,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.rt_buffers,
            );
            let (rt_tex, rt_view) = create_rt_texture(&gpu.device, w, h);
            gpu.rt_blit_bg = rmesh_render::create_blit_nf_bind_group(
                &gpu.device,
                &gpu.rt_blit_pipeline,
                &rt_view,
            );
            gpu.rt_texture = rt_tex;
            gpu.rt_texture_view = rt_view;
        }

        // Update window title
        if let Some(window) = &self.window {
            let name = path.file_name().map_or("rmesh viewer".into(), |n| {
                format!("rmesh viewer - {}", n.to_string_lossy())
            });
            window.set_title(&name);
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let gpu = match &mut self.gpu {
            Some(g) => g,
            None => return,
        };
        if new_size.width > 0 && new_size.height > 0 {
            gpu.surface_config.width = new_size.width;
            gpu.surface_config.height = new_size.height;
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
            gpu.targets = RenderTargets::new(&gpu.device, new_size.width, new_size.height);
            // Recreate blit bind group since color_view changed
            gpu.blit_bg =
                create_blit_bind_group(&gpu.device, &gpu.blit_pipeline, &gpu.targets.color_view);

            // Recreate primitive depth target
            gpu.primitive_targets =
                PrimitiveTargets::new(&gpu.device, new_size.width, new_size.height);

            // Recreate Hi-Z (texture + per-mip bind groups) since size changed.
            gpu.hiz_texture =
                rmesh_postprocess::HizTexture::new(&gpu.device, new_size.width, new_size.height);
            gpu.hiz_linearize_bg = rmesh_postprocess::create_hiz_linearize_bind_group(
                &gpu.device,
                &gpu.hiz_pipelines,
                &gpu.primitive_targets.depth_view,
                &gpu.targets.depth_view,
            );
            gpu.hiz_downsample_bgs = (0..(gpu.hiz_texture.mip_count as usize - 1))
                .map(|i| {
                    rmesh_postprocess::create_hiz_downsample_bind_group(
                        &gpu.device,
                        &gpu.hiz_pipelines,
                        &gpu.hiz_texture.mip_views[i],
                    )
                })
                .collect();

            // Recreate GTAO bind group: Hi-Z full view + normals + volume depth.
            gpu.gtao_bg = rmesh_postprocess::create_gtao_bind_group(
                &gpu.device,
                &gpu.gtao_pipeline,
                &gpu.hiz_texture.full_view,
                &gpu.targets.normals_view,
                &gpu.targets.depth_view,
            );

            // Recreate AO blur bind groups. H reads ao_temporal_view (post-temporal).
            gpu.ao_blur_bg_h = rmesh_postprocess::create_ao_blur_bind_group(
                &gpu.device,
                &gpu.ao_blur_pipeline,
                &gpu.ao_blur_pipeline.uniforms_h,
                &gpu.targets.ao_temporal_view,
                &gpu.hiz_texture.full_view,
                &gpu.targets.normals_view,
            );
            gpu.ao_blur_bg_v = rmesh_postprocess::create_ao_blur_bind_group(
                &gpu.device,
                &gpu.ao_blur_pipeline,
                &gpu.ao_blur_pipeline.uniforms_v,
                &gpu.targets.ao_blur_temp_view,
                &gpu.hiz_texture.full_view,
                &gpu.targets.normals_view,
            );

            // Recreate SSGI bind groups (doubled for parity).
            gpu.ssgi_bg = std::array::from_fn(|p| {
                rmesh_postprocess::create_ssgi_bind_group(
                    &gpu.device,
                    &gpu.ssgi_pipeline,
                    &gpu.hiz_texture.full_view,
                    &gpu.targets.normals_view,
                    gpu.targets.lit_history(p as u32),
                )
            });
            gpu.ssgi_blur_bg_h = rmesh_postprocess::create_ssgi_blur_bind_group(
                &gpu.device,
                &gpu.ssgi_blur_pipeline,
                &gpu.ssgi_blur_pipeline.uniforms_h,
                &gpu.targets.ssgi_temporal_view,
                &gpu.hiz_texture.full_view,
                &gpu.targets.normals_view,
            );
            gpu.ssgi_blur_bg_v = rmesh_postprocess::create_ssgi_blur_bind_group(
                &gpu.device,
                &gpu.ssgi_blur_pipeline,
                &gpu.ssgi_blur_pipeline.uniforms_v,
                &gpu.targets.ssgi_blur_temp_view,
                &gpu.hiz_texture.full_view,
                &gpu.targets.normals_view,
            );

            // Recreate temporal bind groups (doubled for parity).
            gpu.ssgi_temporal_bg = std::array::from_fn(|p| {
                rmesh_postprocess::create_temporal_bind_group(
                    &gpu.device,
                    &gpu.ssgi_temporal_pipeline,
                    gpu.targets.ssgi_current(p as u32),
                    gpu.targets.ssgi_history(p as u32),
                    &gpu.hiz_texture.full_view,
                )
            });
            gpu.ao_temporal_bg = std::array::from_fn(|p| {
                rmesh_postprocess::create_temporal_bind_group(
                    &gpu.device,
                    &gpu.ao_temporal_pipeline,
                    gpu.targets.ao_current(p as u32),
                    gpu.targets.ao_history(p as u32),
                    &gpu.hiz_texture.full_view,
                )
            });

            // Recreate SSR + SSR temporal bind groups (doubled for parity).
            gpu.ssr_bg = std::array::from_fn(|p| {
                rmesh_postprocess::create_ssr_bind_group(
                    &gpu.device,
                    &gpu.ssr_pipeline,
                    &gpu.hiz_texture.full_view,
                    &gpu.targets.normals_view,
                    gpu.targets.lit_history(p as u32),
                    gpu.targets.ssgi_current(p as u32),
                    &gpu.targets.depth_view,
                )
            });
            gpu.ssr_temporal_bg = std::array::from_fn(|p| {
                rmesh_postprocess::create_temporal_bind_group(
                    &gpu.device,
                    &gpu.ssr_temporal_pipeline,
                    gpu.targets.ssr_current(p as u32),
                    gpu.targets.ssr_history(p as u32),
                    &gpu.hiz_texture.full_view,
                )
            });

            // Re-clear history textures (resize gives new vendor-defined contents).
            {
                let mut enc = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("history_resize_clear"),
                    });
                let black = wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                };
                for p in 0..2u32 {
                    rmesh_postprocess::clear_texture_view(&mut enc, gpu.targets.lit_history(p), black);
                    rmesh_postprocess::clear_texture_view(&mut enc, gpu.targets.ssgi_history(p), black);
                    rmesh_postprocess::clear_texture_view(&mut enc, gpu.targets.ao_history(p), black);
                    rmesh_postprocess::clear_texture_view(&mut enc, gpu.targets.ssr_history(p), black);
                    rmesh_postprocess::clear_texture_view(&mut enc, gpu.targets.ssr_current(p), black);
                }
                gpu.queue.submit(std::iter::once(enc.finish()));
            }

            // Recreate deferred resources since MRT texture views changed
            if let Some(ref dp) = gpu.deferred_pipeline {
                gpu.deferred_bg = Some(std::array::from_fn(|p| {
                    rmesh_postprocess::create_deferred_bind_group(
                        &gpu.device,
                        dp,
                        &gpu.targets.color_view,
                        &gpu.targets.aux0_view,
                        &gpu.targets.normals_view,
                        &gpu.targets.depth_view,
                        &gpu.primitive_targets.depth_view,
                        gpu.targets.ao_current(p as u32),
                        gpu.targets.ssgi_current(p as u32),
                        gpu.targets.lit_history(p as u32),
                        gpu.targets.ssr_current(p as u32),
                    )
                }));
                let out_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("deferred_output"),
                    size: wgpu::Extent3d {
                        width: new_size.width,
                        height: new_size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
                gpu.deferred_blit_bg = Some(create_blit_bind_group(
                    &gpu.device,
                    &gpu.blit_pipeline,
                    &out_view,
                ));
                gpu.deferred_output = Some(out_tex);
                gpu.deferred_output_view = Some(out_view);
            }

            // Recreate DSM debug textures
            {
                let w = new_size.width;
                let h = new_size.height;
                let color_format = wgpu::TextureFormat::Rgba16Float;
                gpu.dsm_moments_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("dsm_moments"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                gpu.dsm_moments_view = gpu
                    .dsm_moments_texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                gpu.dsm_depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("dsm_debug_depth"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                gpu.dsm_depth_view = gpu
                    .dsm_depth_texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                gpu.dsm_resolve_output = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("dsm_resolve_output"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                gpu.dsm_resolve_output_view = gpu
                    .dsm_resolve_output
                    .create_view(&wgpu::TextureViewDescriptor::default());
                gpu.dsm_render_bg = rmesh_dsm::create_dsm_render_bind_group(
                    &gpu.device,
                    &gpu.dsm_pipeline,
                    &gpu.buffers.uniforms,
                    &gpu.buffers.interval_vertex_buf,
                    &gpu.buffers.interval_tet_data_buf,
                );
                gpu.dsm_resolve_bg = rmesh_dsm::create_dsm_resolve_bind_group(
                    &gpu.device,
                    &gpu.dsm_resolve_pipeline,
                    &gpu.dsm_moments_view,
                );
                gpu.dsm_blit_bg = create_blit_bind_group(
                    &gpu.device,
                    &gpu.blit_pipeline,
                    &gpu.dsm_resolve_output_view,
                );
            }

            // Recreate ray trace pipeline + texture for new size
            gpu.rt_pipeline = rmesh_raytrace::RayTracePipeline::new(
                &gpu.device,
                new_size.width,
                new_size.height,
                0,
            );
            gpu.rt_bg = rmesh_raytrace::create_raytrace_bind_group(
                &gpu.device,
                &gpu.rt_pipeline,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.rt_buffers,
            );
            let (rt_tex, rt_view) = create_rt_texture(&gpu.device, new_size.width, new_size.height);
            gpu.rt_blit_bg = rmesh_render::create_blit_nf_bind_group(
                &gpu.device,
                &gpu.rt_blit_pipeline,
                &rt_view,
            );
            gpu.rt_texture = rt_tex;
            gpu.rt_texture_view = rt_view;
        }
    }

    /// World-space transform of whatever is currently selected, or None.
    fn current_selected_transform(&self) -> Option<rmesh_interact::Transform> {
        use rmesh_interact::Selection;
        match self.interaction.selected() {
            Some(Selection::Primitive(i)) => self.primitives.get(i).map(|p| p.transform),
            Some(Selection::Node(i)) => self
                .animated_scene
                .as_ref()
                .and_then(|s| s.nodes.get(i))
                .map(|n| n.world_transform),
            None => None,
        }
    }

    /// Apply a confirmed world-space transform back to the selected entity.
    /// For scene nodes, the world transform is converted to the node's local space
    /// (by composing with the inverse of the parent's world matrix) and the scene's
    /// world transforms are recomputed so children follow.
    fn apply_committed_transform(&mut self, world_t: rmesh_interact::Transform) {
        use rmesh_interact::Selection;
        match self.interaction.selected() {
            Some(Selection::Primitive(i)) => {
                if let Some(p) = self.primitives.get_mut(i) {
                    p.transform = world_t;
                }
            }
            Some(Selection::Node(i)) => {
                if let Some(ref mut scene) = self.animated_scene {
                    if i < scene.nodes.len() {
                        let parent_world = scene.parent_world_matrix(i);
                        let local_mat = parent_world.inverse() * world_t.model_matrix();
                        let (scale, rotation, translation) =
                            local_mat.to_scale_rotation_translation();
                        scene.nodes[i].local_transform = rmesh_interact::Transform {
                            position: translation,
                            rotation,
                            scale,
                        };
                        scene.compute_world_transforms();
                    }
                }
            }
            None => {}
        }
    }

    /// Sync the current selection's transform into the interaction state machine,
    /// dispatch the event, and route a Confirmed transform back to the entity.
    fn process_interact_event(
        &mut self,
        ie: &InteractEvent,
        ctx: &InteractContext,
    ) -> InteractResult {
        let cur_t = self.current_selected_transform();
        self.interaction.set_current_transform(cur_t);
        let result = self.interaction.process_event(ie, ctx);
        if let InteractResult::Confirmed(new_t) = result {
            self.apply_committed_transform(new_t);
        }
        result
    }

    // -------------------------------------------------------------------
    // Vertex-select / PBD wiring
    // -------------------------------------------------------------------

    /// Lazily build the CPU mesh adjacency / vertex→tets map. Cleared on
    /// scene reload (see `load_scene`).
    fn ensure_mesh_topology(&mut self) {
        if self.mesh_topology.is_none() {
            log::info!("[pbd] Building mesh topology for {} vertices, {} tets",
                self.scene_data.vertex_count, self.scene_data.tet_count);
            let t0 = std::time::Instant::now();
            self.mesh_topology = Some(MeshTopology::build(
                &self.scene_data.indices,
                self.scene_data.vertex_count,
                self.scene_data.tet_count,
            ));
            log::info!("[pbd] Topology built in {:.2}s", t0.elapsed().as_secs_f64());
        }
        if self.pbd_vertex_max_density.is_none() {
            // Per-vertex max incident tet density. The volume renderer uses
            // density in `od = density * dist`, so high-density tets are
            // what's visually "there." Vertices that touch only low-density
            // tets are basically in empty space and shouldn't be pickable.
            let vc = self.scene_data.vertex_count as usize;
            let mut max_d = vec![0.0_f32; vc];
            let indices = &self.scene_data.indices;
            let densities = &self.scene_data.densities;
            for (t, dens) in densities.iter().enumerate() {
                let base = t * 4;
                for &vi in &indices[base..base + 4] {
                    let slot = &mut max_d[vi as usize];
                    if *dens > *slot {
                        *slot = *dens;
                    }
                }
            }
            self.pbd_vertex_max_density = Some(max_d);
        }
    }

    /// Project every vertex to screen and return the index of the one closest
    /// to `mouse_px` within `radius_px`, breaking ties by depth (nearest wins).
    /// Returns `None` if no vertex is within the radius. CPU-only for now;
    /// fine for typical mesh sizes (≤ a few hundred k).
    #[allow(dead_code)]
    fn pick_vertex(&mut self, mouse_px: [f32; 2], radius_px: f32) -> Option<u32> {
        self.pick_vertex_info(mouse_px, radius_px).map(|p| p.idx)
    }

    /// Same as [`pick_vertex`] but returns rich info for diagnostic logging
    /// and the hover preview marker.
    fn pick_vertex_info(&mut self, mouse_px: [f32; 2], radius_px: f32) -> Option<PickInfo> {
        // Per-vertex max density is needed for the density filter.
        self.ensure_mesh_topology();

        let gpu = self.gpu.as_ref()?;
        let w = gpu.surface_config.width as f32;
        let h = gpu.surface_config.height as f32;
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(w / h);
        let vp = proj * view;
        let r_sq = radius_px * radius_px;

        // Occlusion test: visible surface depth from the volume renderer.
        let visible_depth = self.read_visible_depth_at(mouse_px);
        let depth_tol = 0.15_f32;
        // Density floor: in .rmesh's exp((val-100)/20) encoding, 1.0 = the
        // middle value (val=100). Vertices touching only sub-1.0 tets are
        // basically in empty fog space — reject.
        let min_density = 1.0_f32;

        let max_density = self.pbd_vertex_max_density.as_deref();
        let mut best: Option<(u32, f32, f32, f32, f32)> = None;
        let mut candidates_in_radius = 0u32;
        let mut candidates_passing_depth = 0u32;
        let mut candidates_passing_density = 0u32;
        let verts = &self.scene_data.vertices;
        let n = self.scene_data.vertex_count as usize;
        for i in 0..n {
            let b = i * 3;
            let p = glam::Vec4::new(verts[b], verts[b + 1], verts[b + 2], 1.0);
            let view_pos = view * p;
            if view_pos.z >= -0.001 {
                continue;
            }
            let view_depth = -view_pos.z;
            let clip = vp * p;
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            let sx = (ndc.x * 0.5 + 0.5) * w;
            let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * h;
            let dx = sx - mouse_px[0];
            let dy = sy - mouse_px[1];
            let d_sq = dx * dx + dy * dy;
            if d_sq >= r_sq {
                continue;
            }
            candidates_in_radius += 1;
            if let Some(vd) = visible_depth {
                if (view_depth - vd).abs() > depth_tol {
                    continue;
                }
            }
            candidates_passing_depth += 1;
            if let Some(md) = max_density {
                if md[i] < min_density {
                    continue;
                }
            }
            candidates_passing_density += 1;
            match best {
                Some((_, _, bd, _, _)) if view_depth >= bd => {}
                _ => best = Some((i as u32, d_sq, view_depth, sx, sy)),
            }
        }
        let _ = candidates_passing_density; // tracked here for symmetry; could be surfaced
        best.map(|(idx, d_sq, view_depth, sx, sy)| PickInfo {
            idx,
            screen_pos: [sx, sy],
            screen_dist: d_sq.sqrt(),
            view_depth,
            candidates_in_radius,
            candidates_passing_depth,
            visible_depth,
            depth_tol,
        })
    }

    /// Read one pixel from the volume renderer's depth attachment
    /// (`gpu.targets.depth_texture`, Rgba16Float, premul: `.r` = α·E[z_view],
    /// `.a` = α). Returns `r / a` (view-space depth at the visible surface),
    /// or `None` if the pixel has no volume contribution. Blocking 1-pixel
    /// readback (~1 ms).
    fn read_visible_depth_at(&self, mouse_px: [f32; 2]) -> Option<f32> {
        let gpu = self.gpu.as_ref()?;
        let tex_w = gpu.surface_config.width;
        let tex_h = gpu.surface_config.height;
        let mx = mouse_px[0] as i32;
        let my = mouse_px[1] as i32;
        if mx < 0 || my < 0 || mx as u32 >= tex_w || my as u32 >= tex_h {
            return None;
        }
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbd_depth_pixel_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbd_depth_pixel_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.targets.depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: mx as u32, y: my as u32, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range();
        let r = half::f16::from_le_bytes([data[0], data[1]]).to_f32();
        let a = half::f16::from_le_bytes([data[6], data[7]]).to_f32();
        drop(data);
        staging.unmap();

        if a < 1e-3 { None } else { Some(r / a) }
    }

    /// React to a [`VertexSelectResult`] from the vertex-select state machine.
    fn handle_vertex_select_result(&mut self, result: VertexSelectResult) {
        match result {
            VertexSelectResult::Pick => {
                let mp = self.vertex_select.mouse_pos();
                let info = self.pick_vertex_info(mp, 30.0);
                if let Some(info) = info {
                    let idx = info.idx;
                    let b = idx as usize * 3;
                    let adj_count = self
                        .mesh_topology
                        .as_ref()
                        .and_then(|t| t.adjacency.get(idx as usize))
                        .map_or(0, |a| a.len());
                    let vd_str = info
                        .visible_depth
                        .map_or("none".to_string(), |d| format!("{d:.3}"));
                    let max_dens = self
                        .pbd_vertex_max_density
                        .as_ref()
                        .map_or(0.0, |d| d[idx as usize]);
                    log::info!(
                        "[pbd] Pick: vertex #{idx} at [{:.3}, {:.3}, {:.3}] (mouse=[{:.1}, {:.1}], {adj_count} neighbors, max_density={max_dens:.3}, {} in radius, {} passed depth, visible={vd_str})",
                        self.scene_data.vertices[b],
                        self.scene_data.vertices[b + 1],
                        self.scene_data.vertices[b + 2],
                        mp[0], mp[1],
                        info.candidates_in_radius,
                        info.candidates_passing_depth,
                    );
                    self.vertex_select.set_selected(vec![idx]);
                    self.vertex_select.begin_grab();
                    self.init_pbd_grab();
                } else {
                    log::info!("[pbd] Pick: no vertex within 30px of mouse=[{:.1}, {:.1}]", mp[0], mp[1]);
                    self.vertex_select.set_selected(Vec::new());
                }
            }
            VertexSelectResult::UpdateGrab => {
                // Encoded in render(): the PBD step dispatches each frame
                // while is_grabbing(). render() handles per-frame drag logging.
            }
            VertexSelectResult::ConfirmGrab => {
                let displacement = self.readback_pbd_vertices();
                log::info!(
                    "[pbd] ConfirmGrab: kept deformation ({} steps, max vertex Δ = {:.4})",
                    self.pbd_step_counter, displacement,
                );
                self.pbd_pre_grab_vertices = None;
                if let Some(gpu) = &mut self.gpu {
                    gpu.pbd_solver = None;
                }
                self.pbd_handle_initial.clear();
                // Force DSM atlas rebuild on next frame — the shadow cubemaps
                // were baked against the old geometry. Cleared via the same
                // idiom as render.rs:663-664 / main.rs:1534-1535.
                self.cached_dsm_lights.clear();
                self.cached_dsm_num_lights = 0;
                // Kick off a background rebuild of the flare collision-mesh
                // BVH so subsequent flares bounce off the deformed surface.
                // Does not block confirm; the new mesh installs once ready.
                self.flare_system.start_async_rebuild(&self.scene_data);
            }
            VertexSelectResult::CancelGrab => {
                log::info!("[pbd] CancelGrab: restoring pre-grab vertices");
                if let (Some(verts), Some(gpu)) =
                    (self.pbd_pre_grab_vertices.take(), self.gpu.as_ref())
                {
                    gpu.queue.write_buffer(&gpu.buffers.vertices, 0, bytemuck::cast_slice(&verts));
                    self.scene_data.vertices = verts;
                }
                if let Some(gpu) = &mut self.gpu {
                    gpu.pbd_solver = None;
                }
                self.pbd_handle_initial.clear();
            }
            _ => {}
        }
    }

    /// Synchronously read the GPU `vertices` buffer back into
    /// `scene_data.vertices`, so subsequent picks/grabs see the deformed
    /// state. Blocks briefly (one submit + poll Wait + map). Returns the
    /// max world-space delta vs the pre-grab snapshot, or 0 if no snapshot.
    fn readback_pbd_vertices(&mut self) -> f32 {
        let Some(gpu) = self.gpu.as_ref() else { return 0.0; };
        let n_floats = self.scene_data.vertices.len();
        let buf_size = (n_floats * std::mem::size_of::<f32>()) as u64;
        if buf_size == 0 {
            return 0.0;
        }
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbd_vertices_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbd_vertices_readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(&gpu.buffers.vertices, 0, &staging, 0, buf_size);
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_res| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let mut max_d_sq = 0.0_f32;
        if let Some(prev) = self.pbd_pre_grab_vertices.as_ref() {
            for (a, b) in prev.chunks_exact(3).zip(floats.chunks_exact(3)) {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                let d_sq = dx * dx + dy * dy + dz * dz;
                if d_sq > max_d_sq {
                    max_d_sq = d_sq;
                }
            }
        }
        self.scene_data.vertices.copy_from_slice(floats);
        drop(data);
        staging.unmap();
        max_d_sq.sqrt()
    }

    /// Build an island from the current selection and upload to GPU.
    fn init_pbd_grab(&mut self) {
        self.ensure_mesh_topology();
        let handles: Vec<u32> = self.vertex_select.selected().to_vec();
        if handles.is_empty() {
            return;
        }
        let topology = self.mesh_topology.as_ref().unwrap();
        let radius = self.pbd_radius;
        let t0 = std::time::Instant::now();
        let island: Island = build_island(
            topology,
            &self.scene_data.indices,
            &self.scene_data.vertices,
            &handles,
            radius,
        );
        let coloring: ConstraintColoring =
            color_constraints(island.particles.len(), &island.distance_constraints);
        let total = island.particles.len();
        let handles_n = island.handle_local_indices.len();
        let fixed_n = island.particles.iter().filter(|p| p.inv_mass == 0.0).count();
        let active_n = total - fixed_n;
        let boundary_n = fixed_n.saturating_sub(handles_n);
        log::info!(
            "[pbd] Grab island: {total} particles ({handles_n} handles, {boundary_n} boundary, {active_n} active), {} edges, {} colors (built in {:.1}ms)",
            island.distance_constraints.len(),
            coloring.num_colors(),
            t0.elapsed().as_secs_f64() * 1e3,
        );
        if island.distance_constraints.is_empty() {
            log::warn!(
                "[pbd] Island has no constraints — picked vertex is isolated. Drag will move only the handle, no soft-body deformation."
            );
        }

        // Cache initial handle world positions for the drag computation.
        self.pbd_handle_initial = handles
            .iter()
            .map(|&h| {
                let b = h as usize * 3;
                (
                    h,
                    [
                        self.scene_data.vertices[b],
                        self.scene_data.vertices[b + 1],
                        self.scene_data.vertices[b + 2],
                    ],
                )
            })
            .collect();

        // Snapshot for cancel.
        self.pbd_pre_grab_vertices = Some(self.scene_data.vertices.clone());
        self.pbd_step_counter = 0;

        if let Some(gpu) = &mut self.gpu {
            let solver = PbdSolver::init_grab(
                &gpu.device,
                &gpu.pbd_pipelines,
                &gpu.buffers.vertices,
                &island,
                &coloring,
                16,
            );
            gpu.pbd_solver = Some(solver);
        }
    }
}

/// Map winit KeyCode to InteractKey.
fn winit_key_to_interact(key: KeyCode) -> Option<InteractKey> {
    match key {
        KeyCode::KeyG => Some(InteractKey::G),
        KeyCode::KeyS => Some(InteractKey::S),
        KeyCode::KeyR => Some(InteractKey::R),
        KeyCode::KeyX => Some(InteractKey::X),
        KeyCode::KeyY => Some(InteractKey::Y),
        KeyCode::KeyZ => Some(InteractKey::Z),
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Some(InteractKey::Shift),
        KeyCode::Enter | KeyCode::NumpadEnter => Some(InteractKey::Enter),
        KeyCode::Escape => Some(InteractKey::Escape),
        KeyCode::Backspace => Some(InteractKey::Backspace),
        KeyCode::Delete => Some(InteractKey::Delete),
        KeyCode::Tab => Some(InteractKey::Tab),
        _ => None,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let title = self
            .loaded_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or("rmesh viewer".into(), |n| {
                format!("rmesh viewer - {}", n.to_string_lossy())
            });
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.init_gpu(window.clone());
        self.window = Some(window);

        // Cache the flare collision mesh on load so the BVH is ready before the
        // first flare is fired or the collision mesh is toggled on.
        self.flare_system.rebuild_collision_mesh(&self.scene_data);

        log::info!(
            "Viewer initialized: {} tets, {} vertices (SH degree {})",
            self.scene_data.tet_count,
            self.scene_data.vertex_count,
            self.sh_coeffs.degree,
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Feed events to egui first
        if let Some(gpu) = &mut self.gpu {
            if let Some(window) = &self.window {
                let _ = gpu.egui_state.on_window_event(window, &event);
            }
        }

        // Check if egui wants keyboard/pointer (from previous frame)
        let egui_wants_pointer = self.egui_ctx.egui_wants_pointer_input();
        let egui_wants_keyboard = self.egui_ctx.egui_wants_keyboard_input();

        // Build interaction context for the state machine
        let interact_ctx = {
            let (w, h) = self.gpu.as_ref().map_or((800, 600), |g| {
                (g.surface_config.width, g.surface_config.height)
            });
            let aspect = w as f32 / h as f32;
            InteractContext {
                view_matrix: self.camera.view_matrix(),
                proj_matrix: self.camera.projection_matrix(aspect),
                viewport_width: w as f32,
                viewport_height: h as f32,
            }
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event: ref key_event,
                ..
            } if !egui_wants_keyboard => {
                let PhysicalKey::Code(code) = key_event.physical_key else {
                    return;
                };

                // Track shift state for shift+left-click pan
                if matches!(code, KeyCode::ShiftLeft | KeyCode::ShiftRight) {
                    self.shift_pressed = key_event.state == ElementState::Pressed;
                }

                // Try interaction system first
                let mut consumed = false;
                if let Some(ikey) = winit_key_to_interact(code) {
                    let ie = match key_event.state {
                        ElementState::Pressed => InteractEvent::KeyDown(ikey),
                        ElementState::Released => InteractEvent::KeyUp(ikey),
                    };
                    // VertexSelect mode owns Tab + (when active) Escape.
                    let was_enabled = self.vertex_select.is_enabled();
                    let vs_result = self.vertex_select.process_event(&ie);
                    if was_enabled != self.vertex_select.is_enabled() {
                        if self.vertex_select.is_enabled() {
                            log::info!("[pbd] VertexSelect mode ON  (Tab to exit, LMB to pick+drag)");
                        } else {
                            log::info!("[pbd] VertexSelect mode OFF");
                        }
                    }
                    if !matches!(vs_result, VertexSelectResult::NotConsumed) {
                        self.handle_vertex_select_result(vs_result);
                        consumed = true;
                    } else {
                        let result = self.process_interact_event(&ie, &interact_ctx);
                        consumed = !matches!(result, InteractResult::NotConsumed);
                    }
                }

                // CharInput for numeric entry (only on press)
                if !consumed && key_event.state == ElementState::Pressed {
                    if let Some(ref text) = key_event.text {
                        for ch in text.chars() {
                            if matches!(ch, '0'..='9' | '.' | '-') {
                                let ie = InteractEvent::CharInput(ch);
                                let result = self.process_interact_event(&ie, &interact_ctx);
                                if !matches!(result, InteractResult::NotConsumed) {
                                    consumed = true;
                                }
                            }
                        }
                    }
                }

                // Flare gun: L key shoots a flare from camera
                if !consumed && code == KeyCode::KeyL && key_event.state == ElementState::Pressed {
                    let fwd = (self.camera.orbit_target - self.camera.position).normalize();
                    self.flare_system.shoot(
                        self.camera.position,
                        fwd,
                        &mut self.primitives,
                        &mut self.next_primitive_id,
                    );
                    consumed = true;
                }

                // Fallback: Escape quits if not consumed by interaction
                if !consumed && code == KeyCode::Escape && key_event.state == ElementState::Pressed
                {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(size) => {
                self.resize(size);
            }

            WindowEvent::MouseInput { state, button, .. } if !egui_wants_pointer => {
                // Feed mouse buttons to interaction system
                let mb = match button {
                    MouseButton::Left => Some(rmesh_interact::MouseButton::Left),
                    MouseButton::Middle => Some(rmesh_interact::MouseButton::Middle),
                    MouseButton::Right => Some(rmesh_interact::MouseButton::Right),
                    _ => None,
                };
                if let Some(mb) = mb {
                    let ie = match state {
                        ElementState::Pressed => InteractEvent::MouseDown { button: mb },
                        ElementState::Released => InteractEvent::MouseUp { button: mb },
                    };
                    // Route to vertex-select first when its mode is on.
                    let mut consumed = false;
                    if self.vertex_select.is_enabled() {
                        // Sync absolute cursor position before LMB-down captures mouse_start.
                        self.vertex_select.set_mouse_pos([
                            self.last_mouse.0 as f32,
                            self.last_mouse.1 as f32,
                        ]);
                        let vs_result = self.vertex_select.process_event(&ie);
                        if !matches!(vs_result, VertexSelectResult::NotConsumed) {
                            self.handle_vertex_select_result(vs_result);
                            consumed = true;
                        }
                    }
                    if !consumed {
                        let result = self.process_interact_event(&ie, &interact_ctx);
                        if matches!(result, InteractResult::NotConsumed) {
                            // Only update camera button state if not consumed
                            match button {
                                MouseButton::Left => self.left_pressed = state == ElementState::Pressed,
                                MouseButton::Middle => {
                                    self.middle_pressed = state == ElementState::Pressed
                                }
                                MouseButton::Right => {
                                    self.right_pressed = state == ElementState::Pressed
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse.0;
                let dy = position.y - self.last_mouse.1;

                if !egui_wants_pointer {
                    if self.vertex_select.is_enabled() {
                        // Snap to absolute pixel coords; ignore delta accumulation.
                        self.vertex_select.set_mouse_pos([position.x as f32, position.y as f32]);
                        // Feed a zero-delta MouseMove to drive the state machine
                        // (emits UpdateGrab while grabbing, Noop otherwise).
                        let ie = InteractEvent::MouseMove { dx: 0.0, dy: 0.0 };
                        let vs_result = self.vertex_select.process_event(&ie);
                        self.handle_vertex_select_result(vs_result);
                    } else if self.interaction.is_active() {
                        // Feed mouse movement to interaction system
                        let ie = InteractEvent::MouseMove {
                            dx: dx as f32,
                            dy: dy as f32,
                        };
                        self.process_interact_event(&ie, &interact_ctx);
                    } else {
                        if self.left_pressed && self.shift_pressed {
                            self.camera.pan(dx as f32, dy as f32);
                        } else if self.left_pressed {
                            self.camera.orbit(dx as f32, dy as f32);
                        }
                        if self.middle_pressed {
                            self.camera.pan(dx as f32, dy as f32);
                        }
                        if self.right_pressed {
                            self.camera.zoom(dy as f32);
                        }
                    }
                }

                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } if !egui_wants_pointer => {
                if !self.interaction.is_active() {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                    };
                    self.camera.zoom(-scroll);
                }
            }

            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rmesh-viewer <input.rmesh|input.ply>");
        std::process::exit(1);
    }

    let scene_path = PathBuf::from(&args[1]);
    log::info!("Loading: {}", scene_path.display());

    let file_data = std::fs::read(&scene_path)
        .with_context(|| format!("Failed to read {}", scene_path.display()))?;

    let is_ply = scene_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ply"));

    let (scene, sh, pbr) = if is_ply {
        rmesh_data::load_ply(&file_data).context("Failed to parse PLY file")?
    } else {
        rmesh_data::load_rmesh(&file_data)
            .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
            .context("Failed to parse scene file")?
    };

    log::info!(
        "Scene: {} vertices, {} tets, SH degree {}, PBR: {}",
        scene.vertex_count,
        scene.tet_count,
        sh.degree,
        if pbr.is_some() { "yes" } else { "no" },
    );

    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    let mut app = App::new(scene, sh, pbr);
    app.loaded_path = Some(scene_path);
    event_loop.run_app(&mut app)?;

    Ok(())
}
