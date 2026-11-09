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
    MeshTopology, PbdSolver,
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

use rmesh_viewer_core::flare;
mod gpu_state;
mod shell;
use rmesh_viewer_core::importance;
mod oracle;
mod render;
use rmesh_viewer_core::light_bake;
mod reference_volume;
mod benchmark;
use flare::FlareSystem;
use gpu_state::*;

pub use rmesh_viewer_core::quality::QualityPreset;

/// The knobs a preset governs, borrowed individually so the render loop can
/// use them while it holds `gpu` mutably.
pub struct QualityKnobs<'a> {
    pub sort_mode: &'a mut SortMode,
    pub shell: &'a mut shell::ShellSystem,
}

impl QualityKnobs<'_> {
    /// Which preset the current knobs match exactly, if any.
    pub fn current(&self) -> Option<QualityPreset> {
        QualityPreset::ALL.into_iter().find(|p| {
            let (sort, shell, imp, thr) = p.knobs();
            *self.sort_mode == sort
                && self.shell.enabled == shell
                && self.shell.importance_enabled == imp
                && (!imp || (self.shell.importance_threshold - thr).abs() < 1e-6)
        })
    }

    pub fn apply(&mut self, preset: QualityPreset) {
        let (sort, shell, imp, thr) = preset.knobs();
        *self.sort_mode = sort;
        if shell && !self.shell.enabled {
            self.shell.mark_dirty();
        }
        self.shell.enabled = shell;
        self.shell.importance_enabled = imp;
        self.shell.importance_threshold = thr;
        log::info!("[quality] preset {:?}", preset);
    }
}

impl App {
    fn quality_knobs(&mut self) -> QualityKnobs<'_> {
        QualityKnobs {
            sort_mode: &mut self.sort_mode,
            shell: &mut self.shell_system,
        }
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("0") | Some("false") => false,
        Some("1") | Some("true") => true,
        _ => default,
    }
}

pub use rmesh_viewer_core::vertices::PickInfo;

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
    /// Drive the interval path with the mesh shader (`IntervalPipelines`)
    /// instead of interval_compute + a vertex stage. Only meaningful when
    /// `mesh_shader_supported`; on by default there.
    use_mesh_shader: bool,
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
    pbr_data: Option<rmesh_data::PbrData>,
    // Flare gun
    flare_system: FlareSystem,
    shell_system: shell::ShellSystem,
    /// Overdraw heatmap view. `RMESH_OVERDRAW=count|alpha|wasted` replaces the
    /// final image with it; off costs nothing (the probe pass is not recorded).
    overdraw_view: Option<rmesh_render::OverdrawView>,
    /// Top of the heatmap's colour ramp, in fragments per pixel.
    overdraw_scale: f32,
    /// `RMESH_SHOT` frame capture, for judging a change on the image.
    screenshot: Option<shell::Screenshot>,
    benchmark: Option<benchmark::Benchmark>,
    /// `RMESH_QUALITY=<preset>`: applied on the first frame, so it overrides
    /// the individual env toggles.
    pending_quality: Option<QualityPreset>,
    /// Ignore mouse and keyboard entirely. Defaults to on for headless runs
    /// (`RMESH_SHOT` / `RMESH_ORACLE`), whose windows pop up on a desktop
    /// where a stray hover over the menu bar or a drag has already ruined a
    /// measurement. `RMESH_NO_INPUT=0|1` overrides.
    ignore_input: bool,
    /// `RMESH_ORACLE=<dir>`: per-pixel overdraw oracle + keep-mask sweep.
    oracle: Option<oracle::OracleRun>,
    // Occlusion culling ('C' toggles). active_region -1 = no-cull (all visible);
    // -2 forces a first rebuild. active_mask_scratch is the bit-packed
    // [ceil(M/32)] u32 mask uploaded to SceneBuffers.tet_visibility on change.
    cull_enabled: bool,
    active_region: i32,
    active_mask_dirty: bool,
    active_mask_scratch: Vec<u32>,
    active_baked_count: u32,
    // One-shot sub-pixel tet measurement (press 'M' to re-run for the current view).
    measure_subpixel: bool,
    // One-shot GPU timing probe: press 'T' to drain the swapchain for a few
    // frames so one completes, then log drawn-tet-count + per-stage GPU ms.
    timing_probe_frames: u32,
}

/// Build the density-shell bind group.
///
/// `aux_data` is optional — scenes without PBR channels get a 4-byte stub, which
/// `shell.wgsl` detects with the same `arrayLength(&aux_data) > 1u` check
/// `interval_fragment.wgsl` uses. The stub is kept alive by the bind group's own
/// reference, so it does not need storing.
fn build_shell_bind_group(
    device: &wgpu::Device,
    pipeline: &rmesh_render::ShellPipeline,
    shell: &rmesh_render::ShellBuffers,
    buffers: &SceneBuffers,
    material: &MaterialBuffers,
    sh_coeffs_buf: &wgpu::Buffer,
    aux_data_buf: Option<&wgpu::Buffer>,
) -> wgpu::BindGroup {
    let stub;
    let aux = match aux_data_buf {
        Some(buf) => buf,
        None => {
            stub = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shell_aux_stub"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &stub
        }
    };
    rmesh_render::create_shell_bind_group(
        device,
        pipeline,
        shell,
        buffers,
        material,
        sh_coeffs_buf,
        aux,
    )
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
        // Reproducible capture views without interactive camera input.
        let cam_pos = if let Ok(value) = std::env::var("RMESH_CAMERA_POSITION") {
            let coords: Vec<f32> = value
                .split(',')
                .map(|v| v.parse().expect("camera coordinate"))
                .collect();
            assert!(
                coords.len() == 3 && coords.iter().all(|v| v.is_finite()),
                "RMESH_CAMERA_POSITION requires three finite coordinates"
            );
            let position = Vec3::new(coords[0], coords[1], coords[2]);
            assert!(
                position.length() > 0.001,
                "camera cannot coincide with orbit target"
            );
            position
        } else {
            cam_pos
        };
        let vis_words = ((scene.tet_count as usize + 31) / 32).max(1);

        let initial_light = light_bake::default_light(&scene);
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
            vsync: std::env::var_os("RMESH_BENCHMARK").is_none() && env_flag("RMESH_VSYNC", true),
            // The production renderer integrates radiance into a single target.
            render_mode: RenderMode::IntervalShader,
            mesh_shader_supported: false,
            use_mesh_shader: true,
            // Matched linear captures exclude light gizmos and opaque helpers.
            show_primitives: std::env::var_os("RMESH_LINEAR_SHOT").is_none(),
            show_scene: true,
            // RMESH_SORT=gpu32|gpu16|cpu pins the sort path for headless A/Bs.
            sort_mode: match std::env::var("RMESH_SORT").ok().as_deref() {
                Some("gpu16") => SortMode::Gpu16,
                Some("cpu") => SortMode::Cpu,
                _ => SortMode::Gpu32,
            },
            interaction: TransformInteraction::new(),
            vertex_select: VertexSelectInteraction::new(),
            mesh_topology: None,
            pbd_pre_grab_vertices: None,
            pbd_handle_initial: Vec::new(),
            pbd_step_counter: 0,
            pbd_radius: 0.25,
            pbd_hover_vertex: None,
            pbd_vertex_max_density: None,
            primitives: vec![initial_light],
            next_primitive_id: 1,
            anim_clock: AnimationClock::new(),
            animated_scene: None,
            pbr_data: pbr,
            flare_system: FlareSystem::default(),
            shell_system: shell::ShellSystem::from_env(),
            overdraw_view: match std::env::var("RMESH_OVERDRAW").ok().as_deref() {
                Some("count") | Some("1") => Some(rmesh_render::OverdrawView::Count),
                Some("alpha") => Some(rmesh_render::OverdrawView::Alpha),
                Some("wasted") => Some(rmesh_render::OverdrawView::Wasted),
                _ => None,
            },
            overdraw_scale: std::env::var("RMESH_OVERDRAW_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256.0),
            screenshot: shell::Screenshot::from_env(),
            benchmark: benchmark::Benchmark::from_env(),
            pending_quality: std::env::var("RMESH_QUALITY").ok().and_then(|v| QualityPreset::from_name(&v)),
            ignore_input: std::env::var_os("RMESH_BENCHMARK").is_some() || env_flag(
                "RMESH_NO_INPUT",
                std::env::var_os("RMESH_SHOT").is_some() || std::env::var_os("RMESH_ORACLE").is_some(),
            ),
            oracle: oracle::OracleConfig::from_env().map(oracle::OracleRun::new),
            cull_enabled: true,
            active_region: -2,
            active_mask_dirty: true,
            active_mask_scratch: vec![0xFFFF_FFFFu32; vis_words],
            active_baked_count: 0,
            // Both probes are opt-in via their keybinds ('M' / 'T'). Auto-running
            // them at startup made every launch stall: the timing probe blocks on
            // `device.poll(Wait)` for up to 120s on each of its frames, and the
            // sub-pixel probe is an O(tet_count) CPU sweep on the first frame.
            // RMESH_OVERDRAW_PROBE=1 fires the 'M' sub-pixel/overdraw probe on the
            // first frame, so it can be captured without a keypress.
            measure_subpixel: env_flag("RMESH_OVERDRAW_PROBE", false),
            timing_probe_frames: 0,
        }
    }

    /// Occlusion-cull mask update: pick the region for the current camera and,
    /// on any change, rebuild the bit-packed per-tet mask and upload it to
    /// `SceneBuffers.tet_visibility`. No-op (all-ones) when cull is off, no
    /// visibility data is baked, or the camera is outside all region coverage.
    /// O(M) only on region/toggle change — cheap and infrequent.
    fn update_cull_mask(&mut self) {
        let cam = self.camera.position;
        let target: i32 = match (&self.scene_data.visibility, self.cull_enabled) {
            (Some(vis), true) => {
                let mut best = -1i32;
                let mut best_d2 = f32::INFINITY;
                for r in 0..vis.num_regions as usize {
                    let cx = vis.region_centroids[r * 3];
                    let cy = vis.region_centroids[r * 3 + 1];
                    let cz = vis.region_centroids[r * 3 + 2];
                    let d2 = (cam.x - cx).powi(2) + (cam.y - cy).powi(2) + (cam.z - cz).powi(2);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = r as i32;
                    }
                }
                if best >= 0 {
                    let radius = vis.region_radii.get(best as usize).copied().unwrap_or(0.0);
                    let margin = 1.25_f32;
                    if best_d2 > (radius * margin).powi(2) {
                        best = -1; // outside coverage => no-cull fallback
                    }
                }
                best
            }
            _ => -1, // cull off or no visibility data => render all
        };

        if target == self.active_region && !self.active_mask_dirty {
            return;
        }

        let m = self.scene_data.tet_count as usize;
        if target < 0 {
            for w in self.active_mask_scratch.iter_mut() {
                *w = 0xFFFF_FFFFu32;
            }
        } else if let Some(vis) = self.scene_data.visibility.as_ref() {
            let scratch = &mut self.active_mask_scratch;
            for w in scratch.iter_mut() {
                *w = 0;
            }
            let row = &vis.masks[(target as usize) * m..(target as usize) * m + m];
            for (t, &v) in row.iter().enumerate() {
                if v != 0 {
                    scratch[t >> 5] |= 1u32 << (t & 31);
                }
            }
        }
        // View-independent importance cull: a tet under the weight bound is
        // dropped whatever region the camera is in.
        if let Some(imp) = self.shell_system.importance_mask() {
            for (w, &m) in self.active_mask_scratch.iter_mut().zip(imp) {
                *w &= m;
            }
        }
        self.active_baked_count = self
            .active_mask_scratch
            .iter()
            .map(|w| w.count_ones())
            .sum::<u32>()
            .min(m as u32);

        if let Some(gpu) = self.gpu.as_ref() {
            gpu.queue.write_buffer(
                &gpu.buffers.tet_visibility,
                0,
                bytemuck::cast_slice(&self.active_mask_scratch),
            );
        }
        self.active_region = target;
        self.active_mask_dirty = false;
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
                // naga's MSL backend does not implement mesh shaders, so
                // Metal is excluded even when the adapter advertises them.
                // RMESH_MESH=0 forces the compute interval path.
                let mesh_shader_supported = subgroup_supported
                    && adapter_features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER)
                    && _backend != wgpu::Backend::Metal
                    && env_flag("RMESH_MESH", true);
                log::info!("Subgroup support: {}", subgroup_supported);
                log::info!("Mesh shader support: {}", mesh_shader_supported);

                let mut required_features =
                    wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::TIMESTAMP_QUERY;
                required_features |= adapter_features & wgpu::Features::FLOAT32_FILTERABLE;
                if subgroup_supported {
                    required_features |= wgpu::Features::SUBGROUP;
                }
                if mesh_shader_supported {
                    required_features |= wgpu::Features::EXPERIMENTAL_MESH_SHADER;
                }
                // Encoder-level timestamps let the postprocess profiler bracket
                // each fullscreen pass without threading timestamp_writes
                // through every record_* helper. Optional — the profiler
                // degrades to "unavailable" when the backend lacks it.
                let encoder_ts =
                    adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
                if encoder_ts {
                    required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
                }
                log::info!("Encoder timestamp support: {}", encoder_ts);
                // Fragment-invocation counter for the interval raster pass.
                let pipe_stats =
                    adapter_features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY);
                if pipe_stats {
                    required_features |= wgpu::Features::PIPELINE_STATISTICS_QUERY;
                }
                log::info!("Pipeline statistics support: {}", pipe_stats);

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
            // COPY_SRC so `RMESH_SHOT` can read the presented frame back. Free
            // on every backend we target; keeps the capture honest by grabbing
            // exactly what was shown rather than an off-screen re-render.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
        // RMESH_GBUF="color=rgba8srgb,aux0=rgba16f" overrides the attachment
        // formats for a perf sweep without a rebuild.
        let gbuffer_formats = rmesh_util::shared::GBufferFormats {
            color: wgpu::TextureFormat::Rgba16Float,
            ..rmesh_util::shared::GBufferFormats::from_env()
        };
        log::info!("Volume output: one Rgba16Float radiance target");
        log::info!("Compiling shader pipelines...");
        let t0 = std::time::Instant::now();
        let pipelines = ForwardPipelines::new(&device, gbuffer_formats);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);
        let mesh_pipelines = if mesh_shader_supported {
            log::info!("Compiling mesh shader pipelines...");
            Some(MeshForwardPipelines::new(&device, gbuffer_formats))
        } else {
            None
        };
        let interval_pipelines = if mesh_shader_supported {
            log::info!("Compiling interval shading pipelines...");
            Some(IntervalPipelines::new(&device, gbuffer_formats))
        } else {
            None
        };
        let mut compute_interval_pipelines = ComputeIntervalPipelines::new(&device, gbuffer_formats);
        log::info!("Pipelines compiled: {:.2}s", t0.elapsed().as_secs_f64());

        log::info!(
            "Uploading scene buffers ({} tets)...",
            self.scene_data.tet_count
        );
        let t0 = std::time::Instant::now();
        let buffers = SceneBuffers::upload(&device, &queue, &self.scene_data);

        let (material, sh_coeffs_buf) = light_bake::upload_material(
            &device, &queue, &self.scene_data, self.pbr_data.as_ref(), &buffers);
        let sh_degree = 0;
        let light_bake = light_bake::LightBake::new(&device, &queue, &self.scene_data, &self.sh_coeffs, self.pbr_data.as_ref(), &buffers);
        light_bake.configure_interval(&device, &mut compute_interval_pipelines);

        let aux_data_buf: Option<wgpu::Buffer> = None;

        log::info!("Buffers uploaded: {:.2}s", t0.elapsed().as_secs_f64());

        let targets = RenderTargets::new_color_only(
            &device,
            size.width.max(1),
            size.height.max(1),
            gbuffer_formats,
        );

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

        // Primitive depth target and the density-shell Hi-Z are built here rather
        // than with the rest of the primitive setup below: the projection bind
        // groups sample the pyramid, so it has to exist first.
        let primitive_targets =
            PrimitiveTargets::new(&device, size.width.max(1), size.height.max(1));
        let shell_pipeline = rmesh_render::ShellPipeline::new(
            &device,
            gbuffer_formats,
            color_format, // DSM moment target — deliberately independent of the G-buffer
            wgpu::TextureFormat::Depth32Float,
        );
        let shell_buffers = rmesh_render::ShellBuffers::new(&device);
        let shell_bg = build_shell_bind_group(
            &device,
            &shell_pipeline,
            &shell_buffers,
            &buffers,
            &material,
            &sh_coeffs_buf,
            aux_data_buf.as_ref(),
        );
        let overdraw_pipelines =
            rmesh_render::OverdrawPipelines::new(&device, surface_config.format);
        let overdraw_target = rmesh_render::OverdrawTarget::new(
            &device,
            &overdraw_pipelines,
            &targets.color_view,
            size.width.max(1),
            size.height.max(1),
        );

        let shell_hiz_pipelines = rmesh_render::ShellHiZPipelines::new(&device);
        let shell_hiz = rmesh_render::ShellHiZ::new(
            &device,
            &shell_hiz_pipelines,
            &primitive_targets.depth_view,
            size.width.max(1),
            size.height.max(1),
        );

        let compute_bg = create_compute_bind_group(
            &device,
            &pipelines,
            &buffers,
            &material,
            &sh_coeffs_buf,
            &shell_hiz.full_view,
        );
        let hw_compute_bg = create_hw_compute_bind_group(
            &device,
            &pipelines,
            &buffers,
            &material,
            &sh_coeffs_buf,
            &shell_hiz.full_view,
        );
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
                let a = create_interval_render_bind_group(
                    &device,
                    ip,
                    &buffers,
                    &material,
                    aux_data_buf.as_ref(),
                );
                let b = create_interval_render_bind_group_with_sort_values(
                    &device,
                    ip,
                    &buffers,
                    &material,
                    sort_state.values_b(),
                    aux_data_buf.as_ref(),
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
            PrimitivePipeline::new(
                &device,
                &material_registry.bind_group_layout,
                gbuffer_formats,
            );

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

        // --- Postprocess-chain profiler ---
        // A second query set, written with encoder-level write_timestamp()
        // calls that bracket the shell, occlusion hierarchy, and diagnostics
        // chain. Slot count is dynamic per frame (passes are gated by toggles),
        // so `pp_ts_used` records how many were actually written and only that
        // prefix is ever resolved — see TS_WRITTEN_COUNT's warning.
        let pp_ts_enabled = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let pp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pp_profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: PP_TS_COUNT,
        });
        let pp_resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pp_ts_resolve"),
            size: (PP_TS_COUNT as u64) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pp_readback = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pp_ts_readback"),
                size: (PP_TS_COUNT as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let pp_readback_ready: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2] =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let pp_readback_mapped: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2] =
            std::array::from_fn(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));

        // --- Fragment-invocation counter for the interval raster pass ---
        // Turns the raster pass's per-fragment cost from an estimate into a
        // measurement: fragments/frame is the denominator of the whole roofline.
        // Optional; the adapter may not expose pipeline statistics.
        let frag_stats_enabled = device
            .features()
            .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY);
        let frag_query_set = frag_stats_enabled.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frag_invocations"),
                ty: wgpu::QueryType::PipelineStatistics(
                    wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
                ),
                count: 1,
            })
        });
        // One statistic, 8 bytes.
        let frag_resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frag_stats_resolve"),
            size: 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let frag_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frag_stats_readback"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let frag_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let frag_mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

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
                    self.flare_system.flare_mesh_base = 0;
                    rmesh_viewer_core::assets::upload_models(&device, &queue, &mut primitive_geometry, &mut material_registry, &[&gltf_scene]);
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
            light_bake,
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
            shell_pipeline,
            shell_buffers,
            shell_bg,
            overdraw_pipelines,
            overdraw_target,
            shell_hiz_pipelines,
            shell_hiz,
            egui_renderer,
            egui_state,
            instance_count_readback,
            visible_instance_count: 0,
            ts_query_set,
            ts_resolve_buf,
            ts_readback,
            ts_readback_ready,
            ts_readback_mapped,
            pp_ts_enabled,
            pp_query_set,
            pp_resolve_buf,
            pp_readback,
            pp_readback_ready,
            pp_readback_mapped,
            pp_labels_pending: [Vec::new(), Vec::new()],
            pp_times_ms: Vec::new(),
            frag_query_set,
            frag_resolve_buf,
            frag_readback,
            frag_ready,
            frag_mapped,
            frag_invocations: 0,
            ts_frame: 0,
            ts_period_ns,
            gpu_times_ms: GpuTimings::default(),
            cpu_times_ms: CpuTimings::default(),
            instance_count_ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            instance_count_mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            aux_data_buf,
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
        self.primitives = vec![light_bake::default_light(&self.scene_data)];
        // Drop topology/grab state from the previous scene.
        self.mesh_topology = None;
        self.pbd_vertex_max_density = None;
        self.pbd_pre_grab_vertices = None;
        self.pbd_handle_initial.clear();
        self.vertex_select.set_enabled(false);

        // Cache the flare collision mesh up front for the new scene (drops the
        // stale mesh from the previous scene and avoids a first-flare hitch).
        self.flare_system.rebuild_collision_mesh(&self.scene_data);
        // Shell face ids index the previous scene's tet array — drop them.
        self.shell_system.reset();

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

            (gpu.material_buffers, gpu.sh_coeffs_buf) = light_bake::upload_material(
                &gpu.device, &gpu.queue, &self.scene_data, self.pbr_data.as_ref(), &gpu.buffers);
            gpu.tet_count = self.scene_data.tet_count;
            gpu.sh_degree = 0;
            gpu.light_bake = light_bake::LightBake::new(&gpu.device, &gpu.queue, &self.scene_data, &self.sh_coeffs, self.pbr_data.as_ref(), &gpu.buffers);
            gpu.light_bake.configure_interval(&gpu.device, &mut gpu.compute_interval_pipelines);

            // Recreate sort state for new tet count
            let n_pow2 = gpu.tet_count.next_power_of_two();
            gpu.sort_state =
                rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 32, 1, gpu.sort_backend);
            gpu.sort_state.upload_configs(&gpu.queue);
            gpu.sort_state_16bit =
                rmesh_sort::RadixSortState::new(&gpu.device, n_pow2, 16, 1, gpu.sort_backend);
            gpu.sort_state_16bit.upload_configs(&gpu.queue);

            // Recreate bind groups
            // New scene: drop the old shell (its face ids index the previous
            // tet array) and rebind against the new buffers.
            gpu.shell_buffers = rmesh_render::ShellBuffers::new(&gpu.device);
            gpu.shell_bg = build_shell_bind_group(
                &gpu.device,
                &gpu.shell_pipeline,
                &gpu.shell_buffers,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
                gpu.aux_data_buf.as_ref(),
            );

            gpu.compute_bg = create_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
                &gpu.shell_hiz.full_view,
            );
            gpu.hw_compute_bg = create_hw_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
                &gpu.shell_hiz.full_view,
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
                    gpu.aux_data_buf.as_ref(),
                ));
                gpu.interval_render_bg_b =
                    Some(create_interval_render_bind_group_with_sort_values(
                        &gpu.device,
                        ip,
                        &gpu.buffers,
                        &gpu.material_buffers,
                        gpu.sort_state.values_b(),
                        gpu.aux_data_buf.as_ref(),
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
            if let Some(pbr) = &self.pbr_data {
                if !pbr.vertex_normals.is_empty() {
                    gpu.queue.write_buffer(&gpu.buffers.vertex_normals, 0, bytemuck::cast_slice(&pbr.vertex_normals));
                }
            }
            gpu.compute_interval_render_bg = create_compute_interval_render_bind_group(
                &gpu.device, &gpu.compute_interval_pipelines, &gpu.buffers,
            );
            gpu.aux_data_buf = None;
            gpu.compute_interval_convert_bg = create_compute_interval_indirect_convert_bind_group(
                &gpu.device,
                &gpu.compute_interval_pipelines,
                &gpu.buffers,
            );

            // Refresh adjacency used by vertex deformation
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
            gpu.targets = RenderTargets::new_color_only(
                &gpu.device,
                new_size.width,
                new_size.height,
                gpu.targets.formats,
            );
            // Recreate blit bind group since color_view changed
            gpu.blit_bg =
                create_blit_bind_group(&gpu.device, &gpu.blit_pipeline, &gpu.targets.color_view);

            // Recreate primitive depth target
            gpu.primitive_targets =
                PrimitiveTargets::new(&gpu.device, new_size.width, new_size.height);

            // Overdraw probe: screen-sized, and its bind group holds the
            // G-buffer colour view that was just replaced.
            gpu.overdraw_target = rmesh_render::OverdrawTarget::new(
                &gpu.device,
                &gpu.overdraw_pipelines,
                &gpu.targets.color_view,
                new_size.width,
                new_size.height,
            );

            // Recreate the density-shell Hi-Z, and with it every bind group that
            // holds its view. The projection bind groups sample the pyramid, and
            // they are NOT otherwise rebuilt on resize — miss this and the cull
            // silently reads a stale, wrongly-sized texture.
            gpu.shell_hiz = rmesh_render::ShellHiZ::new(
                &gpu.device,
                &gpu.shell_hiz_pipelines,
                &gpu.primitive_targets.depth_view,
                new_size.width,
                new_size.height,
            );
            gpu.compute_bg = create_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
                &gpu.shell_hiz.full_view,
            );
            gpu.hw_compute_bg = create_hw_compute_bind_group(
                &gpu.device,
                &gpu.pipelines,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
                &gpu.shell_hiz.full_view,
            );

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
                    scene.set_node_world_transform(i, world_t);
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
            log::info!(
                "[pbd] Building mesh topology for {} vertices, {} tets",
                self.scene_data.vertex_count,
                self.scene_data.tet_count
            );
            let t0 = std::time::Instant::now();
            self.mesh_topology = Some(MeshTopology::build(
                &self.scene_data.indices,
                self.scene_data.vertex_count,
                self.scene_data.tet_count,
            ));
            log::info!("[pbd] Topology built in {:.2}s", t0.elapsed().as_secs_f64());
        }
        if self.pbd_vertex_max_density.is_none() {
            self.pbd_vertex_max_density = Some(rmesh_viewer_core::vertices::vertex_max_density(&self.scene_data));
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
        rmesh_viewer_core::vertices::pick_vertex(
            &self.scene_data.vertices, self.pbd_vertex_max_density.as_deref(),
            view, vp, [w, h], mouse_px, radius_px, None,
        )
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
                    log::info!(
                        "[pbd] Pick: no vertex within 30px of mouse=[{:.1}, {:.1}]",
                        mp[0],
                        mp[1]
                    );
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
                    self.pbd_step_counter,
                    displacement,
                );
                self.pbd_pre_grab_vertices = None;
                self.shell_system.reset();
                self.active_mask_dirty = true;
                if let Some(gpu) = &mut self.gpu {
                    rmesh_viewer_core::vertices::refresh_bounds(&mut self.scene_data, &gpu.queue, &gpu.buffers);
                    gpu.shell_buffers.face_count = 0;
                    gpu.pbd_solver = None;
                    gpu.light_bake.invalidate();
                }
                self.pbd_handle_initial.clear();
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
                    gpu.queue
                        .write_buffer(&gpu.buffers.vertices, 0, bytemuck::cast_slice(&verts));
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
        let Some(gpu) = self.gpu.as_ref() else {
            return 0.0;
        };
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
        let prepared = rmesh_viewer_core::vertices::prepare_grab(&self.scene_data, topology, &handles, radius);
        let island = prepared.island;
        let coloring = prepared.coloring;
        let total = island.particles.len();
        let handles_n = island.handle_local_indices.len();
        let fixed_n = island
            .particles
            .iter()
            .filter(|p| p.inv_mass == 0.0)
            .count();
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
        self.pbd_handle_initial = prepared.handle_initial;

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
            .with_inner_size(
                // RMESH_WIN=WxH overrides the default window size, so the
                // postprocess profiler can be swept across resolutions.
                std::env::var("RMESH_WIN")
                    .ok()
                    .and_then(|v| {
                        let (w, h) = v.split_once('x')?;
                        Some(winit::dpi::PhysicalSize::new(
                            w.trim().parse::<u32>().ok()?,
                            h.trim().parse::<u32>().ok()?,
                        ))
                    })
                    .map_or(
                        winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                        winit::dpi::Size::Physical,
                    ),
            );
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
        if self.ignore_input
            && matches!(
                event,
                WindowEvent::KeyboardInput { .. }
                    | WindowEvent::ModifiersChanged(_)
                    | WindowEvent::CursorMoved { .. }
                    | WindowEvent::CursorEntered { .. }
                    | WindowEvent::CursorLeft { .. }
                    | WindowEvent::MouseWheel { .. }
                    | WindowEvent::MouseInput { .. }
                    | WindowEvent::Touch(_)
                    | WindowEvent::PinchGesture { .. }
                    | WindowEvent::DoubleTapGesture { .. }
                    | WindowEvent::RotationGesture { .. }
            )
        {
            return;
        }

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

                // 'C' toggles occlusion culling (mask rebuilt next frame).
                if code == KeyCode::KeyC && key_event.state == ElementState::Pressed {
                    if self.scene_data.visibility.is_some() {
                        self.cull_enabled = !self.cull_enabled;
                        self.active_mask_dirty = true;
                        log::info!(
                            "Occlusion cull: {}",
                            if self.cull_enabled { "ON" } else { "OFF" }
                        );
                    } else {
                        log::info!("Occlusion cull: no visibility data baked in this .rmesh");
                    }
                }

                // 'M' re-runs the sub-pixel tet probe for the current view.
                if code == KeyCode::KeyM && key_event.state == ElementState::Pressed {
                    self.measure_subpixel = true;
                }

                // 'T' runs the one-shot GPU timing probe (drains a few frames so
                // one completes, then logs drawn count + per-stage GPU ms).
                if code == KeyCode::KeyT && key_event.state == ElementState::Pressed {
                    self.timing_probe_frames = 2;
                    log::info!("[timing probe] capturing 2 frames — may take up to ~1-2 min if a frame is very slow...");
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
                            log::info!(
                                "[pbd] VertexSelect mode ON  (Tab to exit, LMB to pick+drag)"
                            );
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
                        self.vertex_select
                            .set_mouse_pos([self.last_mouse.0 as f32, self.last_mouse.1 as f32]);
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
                                MouseButton::Left => {
                                    self.left_pressed = state == ElementState::Pressed
                                }
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
                        self.vertex_select
                            .set_mouse_pos([position.x as f32, position.y as f32]);
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
                let start = std::time::Instant::now();
                self.render();
                if let Some(mut bench) = self.benchmark.take() {
                    let done = bench.after_frame(self, start);
                    self.benchmark = Some(bench);
                    if done { event_loop.exit(); return; }
                }
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
    if args.get(1).is_some_and(|a| a == "--reference-volume") {
        return reference_volume::run_cli(&args);
    }
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
