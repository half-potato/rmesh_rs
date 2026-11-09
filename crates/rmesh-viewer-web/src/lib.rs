//! rmesh-viewer-web: WebAssembly viewer for .rmesh files.
//!
//! Exposes a `WebViewer` struct to JavaScript via wasm-bindgen.
//! Uses the same rendering pipeline as the native rmesh-viewer.

#![allow(dead_code)]

use rmesh_viewer_core::quality::{QualityPreset, SortMode};
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

use rmesh_compositor::record_primitive_pass;
use rmesh_interact::{InteractEvent, InteractResult, PrimitiveKind};
use rmesh_render::{
    create_blit_bind_group, create_compute_bind_group, create_compute_interval_gen_bind_group,
    create_compute_interval_indirect_convert_bind_group, create_compute_interval_render_bind_group,
    create_hw_compute_bind_group, create_render_bind_group, BlitPipeline, ComputeIntervalPipelines,
    ForwardComputeBindGroups, ForwardHwComputeBindGroups, ForwardPipelines, MaterialBuffers,
    RenderTargets, SceneBuffers,
};
use rmesh_render::{
    record_blit, record_compute_interval_after_sort, record_compute_interval_project,
};
use rmesh_sort::CpuSorter;
use rmesh_util::camera::Camera;
use rmesh_util::shared::{GBufferFormats, Uniforms};
use rmesh_viewer_core::{
    flare::FlareSystem,
    light_bake::{default_light, LightBake},
};
use std::sync::Arc;

use glam::Vec3;

use rmesh_compositor::{MaterialRegistry, PrimitiveGeometry, PrimitivePipeline, PrimitiveTargets};
use rmesh_interact::{InteractContext, InteractKey, Primitive, TransformInteraction};

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

/// Internal state for a loaded scene (separated so `WebViewer` can exist without a scene).
#[allow(dead_code)]
struct SceneState {
    buffers: SceneBuffers,
    material: MaterialBuffers,
    /// Legacy projection bindings required by the shared recorder signature.
    /// Both web sort precisions use the lean HW projection bindings below.
    compute_bg: ForwardComputeBindGroups,
    /// Baked geometry projection bindings fit the baseline eight-buffer cap.
    hw_compute_bg: ForwardHwComputeBindGroups,
    /// Interval gen bind group reading from `buffers.sort_values`. The web
    /// viewer overwrites `sort_values` with CPU-sorted indices between the
    /// project and after-sort submits, so we never need the B (ping-pong)
    /// variant.
    gen_bg_a: wgpu::BindGroup,
    /// Forward render bind groups (legacy HW-raster path). The A variant
    /// references `buffers.sort_values`; in the forward path it also reads
    /// the CPU-sorted indices we write before the render pass.
    render_bg: wgpu::BindGroup,
    quad_render_bg: wgpu::BindGroup,
    prepass_bg: wgpu::BindGroup,
    #[allow(dead_code)]
    /// Interval render bind group (non-PBR: aux_data + tet_indices are dummy).
    ci_render_bg: wgpu::BindGroup,
    /// Indirect convert bind group (turns instance_count into a mesh dispatch).
    ci_convert_bg: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
    sh_coeffs_buf: wgpu::Buffer,
    sh_degree: u32,
    tet_count: u32,
    /// CPU sorter workspace + scene arrays. Per-frame cull+sort produces
    /// `sorted_indices` which we upload to `buffers.sort_values` before the
    /// interval gen step when the CPU fallback is selected.
    cpu_sorter: CpuSorter,
    importance_mask: Vec<u32>,
    importance_kept: u32,
    importance_scores: Vec<f32>,
    max_od: Vec<f32>,
    opacity: Vec<f32>,
    neighbors: Vec<i32>,
    topology: Option<rmesh_pbd::MeshTopology>,
    vertex_density: Option<Vec<f32>>,
    arrays: Arc<rmesh_data::SceneData>,
    light_bake: LightBake,
    material_min_t: f32,
    shell: rmesh_render::ShellBuffers,
    shell_bg: wgpu::BindGroup,
    last_sort_vp: Option<[f32; 16]>,
    visible_count: usize,
    gpu_sort: Option<rmesh_sort::RadixSortState>,
}

struct VertexGrab {
    solver: rmesh_pbd::PbdSolver,
    initial: Vec<(u32, [f32; 3])>,
    original: Vec<f32>,
    delta: [f32; 2],
    readback: Option<rmesh_viewer_core::vertices::VertexReadback>,
    restore_gpu_sort: bool,
}

#[wasm_bindgen]
pub struct WebViewer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    camera: Camera,
    pipelines: ForwardPipelines,
    compute_interval_pipelines: ComputeIntervalPipelines,
    /// Persistent staging-buffer pool for CPU-sort `sort_values` /
    /// `indirect_args` writes. Without this, every frame's `queue.write_buffer`
    /// would allocate a fresh ~19 MB staging buffer, leaking memory.
    staging_belt: wgpu::util::StagingBelt,
    blit_pipeline: BlitPipeline,
    shell_pipeline: rmesh_render::ShellPipeline,
    shell_enabled: bool,
    importance_enabled: bool,
    importance_threshold: f32,
    shell_threshold: f32,
    shell_erosion: u32,
    shell_cull: bool,
    shell_bias: f32,
    shell_only: bool,
    shell_entry: bool,
    show_scene: bool,
    shell_hiz: rmesh_render::ShellHiZ,
    shell_hiz_pipelines: rmesh_render::ShellHiZPipelines,
    targets: RenderTargets,
    scene: Option<SceneState>,
    // Primitives
    primitive_geometry: PrimitiveGeometry,
    primitive_pipeline: PrimitivePipeline,
    primitive_targets: PrimitiveTargets,
    material_registry: MaterialRegistry,
    show_primitives: bool,
    // Interaction
    interaction: TransformInteraction,
    primitives: Vec<Primitive>,
    next_primitive_id: u32,
    /// Render path selection. Defaults to `Interval`; the UI dropdown can
    /// toggle to `Forward` (the legacy HW-raster path, same as the viewer
    /// shipped before interval shading was wired up).
    render_mode: WebRenderMode,
    flares: FlareSystem,
    last_frame: web_time::Instant,
    gpu_sort_pipelines: Option<rmesh_sort::RadixSortPipelines>,
    use_gpu_sort: bool,
    sort_bits: u32,
    flare_model: Option<rmesh_anim::GltfScene>,
    animation: Option<rmesh_anim::GltfScene>,
    animation_offsets: (usize, usize),
    vertex_grab: Option<VertexGrab>,
    pbd_pipelines: Option<rmesh_pbd::PbdPipelines>,
}

/// Render modes share projection, sorting, baked lighting, and compositing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WebRenderMode {
    /// Compute-interval shading (default): one compute pass to emit interval
    /// geometry, then an instanced indexed draw.
    Interval,
    /// Legacy forward HW-raster: one draw_indirect call per visible tet
    /// (12 vertices each). Slower but visually identical when sorted right.
    Forward,
    Quad,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebViewer {
    /// Create a viewer attached to a `<canvas>` element by its DOM id. Async — must `await` in JS.
    pub async fn create(canvas_id: &str) -> Result<WebViewer, JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| format!("no element with id '{canvas_id}'"))?;
        let canvas: web_sys::HtmlCanvasElement =
            canvas.dyn_into().map_err(|_| "element is not a canvas")?;

        let width = canvas.client_width().max(1) as u32;
        let height = canvas.client_height().max(1) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("create_surface: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("request_adapter: {e}"))?;

        log::info!("GPU: {:?}", adapter.get_info().name);

        let adapter_limits = adapter.limits();
        log::info!(
            "Adapter limits: storage_buffers/stage={}, bind_groups={}, storage_buffer_max={}, workgroup_storage={}",
            adapter_limits.max_storage_buffers_per_shader_stage,
            adapter_limits.max_bind_groups,
            adapter_limits.max_storage_buffer_binding_size,
            adapter_limits.max_compute_workgroup_storage_size,
        );

        if adapter_limits.max_storage_buffers_per_shader_stage < 8 {
            return Err(
                "This viewer requires WebGPU with at least 8 storage buffers per shader stage"
                    .into(),
            );
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rmesh-web device"),
                required_features: adapter.features() & wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: 8, ..adapter_limits.clone() },
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let formats = GBufferFormats {
            color: wgpu::TextureFormat::Rgba16Float,
            ..Default::default()
        };

        log::info!("Compiling shader pipelines...");
        let pipelines = ForwardPipelines::new_baked(&device, formats);
        let compute_interval_pipelines = ComputeIntervalPipelines::new_baked(&device, formats);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);
        let targets = RenderTargets::new_color_only(&device, width, height, formats);

        let camera = Camera::new(Vec3::new(0.0, 3.0, -2.0));

        // Primitive setup
        let primitive_geometry = PrimitiveGeometry::new(&device);
        let material_registry = MaterialRegistry::new(&device, &queue);
        let primitive_pipeline =
            PrimitivePipeline::new(&device, &material_registry.bind_group_layout, formats);
        let primitive_targets = PrimitiveTargets::new(&device, width, height);

        let shell_hiz_pipelines = rmesh_render::ShellHiZPipelines::new(&device);
        let shell_hiz = rmesh_render::ShellHiZ::new(
            &device,
            &shell_hiz_pipelines,
            &primitive_targets.depth_view,
            width,
            height,
        );

        log::info!("WebViewer initialized ({width}×{height})");

        let device_for_belt = device.clone();
        Ok(WebViewer {
            device: device.clone(),
            queue,
            surface,
            surface_config,
            camera,
            pipelines,
            compute_interval_pipelines,
            staging_belt: wgpu::util::StagingBelt::new(device_for_belt, 32 * 1024 * 1024),
            shell_pipeline: rmesh_render::ShellPipeline::new_baked(
                &device,
                formats,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureFormat::Depth32Float,
            ),
            shell_enabled: true,
            importance_enabled: true,
            importance_threshold: 0.01,
            shell_threshold: 0.1,
            shell_erosion: 2,
            shell_cull: true,
            shell_bias: 0.0,
            shell_only: false,
            shell_entry: false,
            show_scene: true,
            shell_hiz,
            shell_hiz_pipelines,
            blit_pipeline,
            targets,
            scene: None,
            primitive_geometry,
            primitive_pipeline,
            primitive_targets,
            material_registry,
            show_primitives: true,
            interaction: TransformInteraction::new(),
            primitives: Vec::new(),
            next_primitive_id: 1,
            render_mode: WebRenderMode::Interval,
            flares: FlareSystem::default(),
            last_frame: web_time::Instant::now(),
            gpu_sort_pipelines: None,
            use_gpu_sort: true,
            sort_bits: 32,
            flare_model: None,
            animation: None,
            animation_offsets: (0, 0),
            vertex_grab: None,
            pbd_pipelines: None,
        })
    }

    /// Load a `.rmesh` file from a `Uint8Array`. Call after construction.
    pub fn load_rmesh(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let (scene_data, sh_coeffs, pbr) = rmesh_data::load_rmesh(data)
            .or_else(|_| rmesh_data::load_rmesh_raw(data))
            .map_err(|e| format!("parse error: {e}"))?;

        log::info!(
            "Loaded: {} vertices, {} tets, SH degree {}",
            scene_data.vertex_count,
            scene_data.tet_count,
            sh_coeffs.degree,
        );

        self.upload_scene(scene_data, sh_coeffs, pbr);
        Ok(())
    }

    /// Load a `.ply` file from a `Uint8Array`.
    pub fn load_ply(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let (scene_data, sh_coeffs, pbr) =
            rmesh_data::load_ply(data).map_err(|e| format!("PLY parse error: {e}"))?;

        log::info!(
            "Loaded PLY: {} vertices, {} tets, SH degree {}",
            scene_data.vertex_count,
            scene_data.tet_count,
            sh_coeffs.degree,
        );

        self.upload_scene(scene_data, sh_coeffs, pbr);
        Ok(())
    }

    /// Switch render path. Accepts `"interval"` (compute-interval shading,
    /// default) or `"forward"` (legacy HW raster). Unknown values are ignored.
    pub fn set_render_mode(&mut self, mode: &str) {
        self.render_mode = match mode {
            "forward" => WebRenderMode::Forward,
            "quad" => WebRenderMode::Quad,
            "interval" => WebRenderMode::Interval,
            _ => return,
        };
        log::info!("Render mode set to {:?}", self.render_mode);
    }

    /// Returns the current render mode label (matching the setter strings).
    pub fn get_render_mode(&self) -> String {
        match self.render_mode {
            WebRenderMode::Interval => "interval".to_string(),
            WebRenderMode::Forward => "forward".to_string(),
            WebRenderMode::Quad => "quad".to_string(),
        }
    }

    /// Render one frame. JS calls this from `requestAnimationFrame`.
    pub fn render(&mut self) -> Result<(), JsValue> {
        self.poll_vertex_grab()?;
        let shell_enabled = self.show_scene && self.shell_enabled && self.vertex_grab.is_none()
            && !self.scene.as_ref().is_some_and(|s|s.light_bake.uses_energy());
        if self.scene.as_ref().is_some_and(|s|s.light_bake.uses_energy()) {self.render_mode=WebRenderMode::Interval;self.shell_only=false;}
        // Mutable borrow — CPU sort mutates `scene.cpu_sorter`.
        let scene = match self.scene.as_mut() {
            Some(s) => s,
            None => return Ok(()), // nothing loaded yet
        };

        let dt = self.last_frame.elapsed().as_secs_f32().min(0.1);
        self.last_frame = web_time::Instant::now();
        if self.vertex_grab.is_none() && (self.flares.show_collision_mesh || self.flares.flare_count() > 0) {
            self.flares.ensure_collision_mesh(&scene.arrays);
            self.flares.upload_collision_debug(&self.device, &mut self.primitive_geometry);
        }
        self.flares.tick(dt, &mut self.primitives, &scene.arrays);
        if let Some(animation) = &mut self.animation {
            animation.scene.update(dt);
        }
        let w = self.surface_config.width;
        let h = self.surface_config.height;
        let aspect = w as f32 / h as f32;

        let view_mat = self.camera.view_matrix();
        let proj_mat = self.camera.projection_matrix(aspect);
        let vp = proj_mat * view_mat;
        let ctx = InteractContext {
            view_matrix: view_mat,
            proj_matrix: proj_mat,
            viewport_width: w as f32,
            viewport_height: h as f32,
        };
        let preview = self.interaction.preview_transform(&ctx);
        let selected = self.interaction.selected();
        let mut render_primitives = self.primitives.clone();
        if let (Some(rmesh_interact::Selection::Primitive(i)), Some(transform)) =
            (selected, preview)
        {
            if let Some(p) = render_primitives.get_mut(i) {
                p.transform = transform;
            }
        }
        if let Some(animation) = &mut self.animation {
            let animated = match (selected, preview) {
                (Some(rmesh_interact::Selection::Node(i)), Some(transform)) => {
                    animation.scene.render_node_preview(i, transform)
                }
                _ => animation.scene.render_primitives(),
            };
            render_primitives.extend(animated);
        }

        let inv_view = view_mat.inverse();
        let cam_right = inv_view.col(0).truncate();
        let cam_up = inv_view.col(1).truncate();
        let cam_back = inv_view.col(2).truncate();
        let c2w = glam::Mat3::from_cols(cam_right, -cam_up, -cam_back);

        let f_val = 1.0 / (self.camera.fov_y / 2.0).tan();
        let fx = f_val * h as f32 / 2.0;
        let fy = f_val * h as f32 / 2.0;
        let intrinsics = [fx, fy, w as f32 / 2.0, h as f32 / 2.0];

        let vp_cols = vp.to_cols_array_2d();
        let pos = self.camera.position.to_array();

        let uniforms = Uniforms {
            vp_col0: vp_cols[0],
            vp_col1: vp_cols[1],
            vp_col2: vp_cols[2],
            vp_col3: vp_cols[3],
            c2w_col0: [c2w.col(0).x, c2w.col(0).y, c2w.col(0).z, 0.0],
            c2w_col1: [c2w.col(1).x, c2w.col(1).y, c2w.col(1).z, 0.0],
            c2w_col2: [c2w.col(2).x, c2w.col(2).y, c2w.col(2).z, 0.0],
            intrinsics,
            cam_pos_pad: [pos[0], pos[1], pos[2], 0.0],
            screen_width: w as f32,
            screen_height: h as f32,
            tet_count: scene.tet_count,
            step: 0,
            tile_size_u: 12,
            ray_mode: 0,
            min_t: scene.material_min_t,
            sh_degree: scene.sh_degree,
            near_plane: self.camera.near_z,
            far_plane: 1000.0,
            shell_cull_bias: self.shell_bias,
            shell_cull_flags: u32::from(
                shell_enabled && self.shell_cull && self.use_gpu_sort && scene.shell.face_count > 0,
            ) | if self.vertex_grab.is_some() { 2 } else { 0 }
                | if self.use_gpu_sort && self.sort_bits == 16 { 4 } else { 0 },
        };

        self.queue
            .write_buffer(&scene.buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        // ----- CPU frustum cull + back-to-front sort -----
        // CPU fallback: reuse the sorted list while the view is unchanged.
        // Resize and importance-mask edits invalidate this cache explicitly.
        if self.show_scene && !self.shell_only && !self.use_gpu_sort && scene.last_sort_vp != Some(vp.to_cols_array()) {
            scene.cpu_sorter.preserve_subpixel = scene.light_bake.uses_energy();
            scene.visible_count = scene.cpu_sorter.cull_and_sort_projected(
                &scene.arrays.vertices,
                &scene.arrays.indices,
                &scene.arrays.circumdata,
                self.camera.position,
                vp,
                if self.importance_enabled {
                    Some(&scene.importance_mask)
                } else {
                    None
                },
                [w, h],
            );

            scene.last_sort_vp = Some(vp.to_cols_array());
        }
        let visible_count = scene.visible_count;

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Err(e) => return Err(format!("surface error: {e:?}").into()),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Single-encoder render. wgpu's pipeline barriers serialize:
        //   project_compute (writes sort_values/instance_count) →
        //   StagingBelt copies (overwrite sort_values + instance_count) →
        //   interval gen + render (reads sort_values).
        let mut encoder2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("web render"),
            });

        if let Some(grab) = &self.vertex_grab {
            if grab.readback.is_none() {
                let positions = rmesh_viewer_core::vertices::drag_positions(
                    &grab.initial,
                    view_mat,
                    proj_mat,
                    h as f32,
                    grab.delta,
                );
                grab.solver.step(
                    &self.queue,
                    &mut encoder2,
                    self.pbd_pipelines.as_ref().unwrap(),
                    0.016,
                    &positions,
                );
            }
        } else {
            scene.light_bake.record(
                &self.queue,
                &mut encoder2,
                &scene.material,
                &self.primitives,
                pos,
            );
        }

        record_primitive_pass(
            &mut encoder2,
            &self.queue,
            &self.primitive_pipeline,
            &self.primitive_geometry,
            &self.targets.color_view,
            &self.primitive_targets.depth_view,
            if self.show_primitives {
                render_primitives.as_slice()
            } else {
                &[]
            },
            &vp,
            None,
            Some(&self.material_registry),
        );

        if self.flares.show_collision_mesh && self.vertex_grab.is_none() {
            rmesh_compositor::record_collision_debug_pass(&mut encoder2, &self.queue,
                &self.primitive_pipeline, &self.primitive_geometry, &self.targets.color_view,
                &self.primitive_targets.depth_view, &vp, &self.material_registry);
        }

        if shell_enabled {
            scene.shell.face = if self.shell_entry {
                rmesh_render::ShellFace::Entry
            } else {
                rmesh_render::ShellFace::Exit
            };
            scene.shell.write_view(
                &self.queue,
                vp.to_cols_array_2d(),
                self.camera.position.to_array(),
            );
            rmesh_render::record_shell_color_pass(
                &mut encoder2,
                &self.shell_pipeline,
                &scene.shell_bg,
                &scene.shell,
                &self.targets.color_view,
                &self.primitive_targets.depth_view,
            );
        }

        if shell_enabled && self.use_gpu_sort && !self.shell_only {
            rmesh_render::record_shell_hiz(
                &mut encoder2,
                &self.shell_hiz_pipelines,
                &self.shell_hiz,
            );
        }

        if self.show_scene && !self.shell_only {
            record_compute_interval_project(
                &mut encoder2,
                &self.pipelines,
                if self.use_gpu_sort {
                    scene.gpu_sort.as_ref()
                } else {
                    None
                },
                &scene.buffers,
                &scene.compute_bg,
                Some(&scene.hw_compute_bg),
                scene.tet_count,
                &self.queue,
                None,
                false,
            );

            if self.use_gpu_sort {
                let state = scene.gpu_sort.as_ref().expect("GPU sort initialized");
                let in_b = rmesh_sort::record_radix_sort(
                    &mut encoder2,
                    &self.device,
                    self.gpu_sort_pipelines.as_ref().unwrap(),
                    state,
                    &scene.buffers.sort_keys,
                    &scene.buffers.sort_values,
                );
                if in_b {
                    encoder2.copy_buffer_to_buffer(
                        state.values_b(),
                        0,
                        &scene.buffers.sort_values,
                        0,
                        scene.tet_count as u64 * 4,
                    );
                }
            } else {
                // ----- StagingBelt writes: sort_values + instance_count -----
                // The belt reuses pinned chunks across frames so there's no per-frame
                // allocation. Without the belt the equivalent `queue.write_buffer`
                // calls would each allocate a fresh ~19 MB staging buffer and leak.
                if visible_count > 0 {
                    let size = (visible_count * 4) as wgpu::BufferAddress;
                    let mut view = self.staging_belt.write_buffer(
                        &mut encoder2,
                        &scene.buffers.sort_values,
                        0,
                        wgpu::BufferSize::new(size).unwrap(),
                    );
                    view.copy_from_slice(bytemuck::cast_slice(
                        &scene.cpu_sorter.sorted_indices[..visible_count],
                    ));
                }
                {
                    let mut view = self.staging_belt.write_buffer(
                        &mut encoder2,
                        &scene.buffers.indirect_args,
                        4,
                        wgpu::BufferSize::new(4).unwrap(),
                    );
                    view.copy_from_slice(bytemuck::bytes_of(&(visible_count as u32)));
                }
            }
            match self.render_mode {
                WebRenderMode::Interval => {
                    // Compute-interval: indirect_convert → interval_gen → indexed draw.
                    record_compute_interval_after_sort(
                        &mut encoder2,
                        &self.compute_interval_pipelines,
                        &scene.buffers,
                        &self.targets,
                        &scene.gen_bg_a, // selected sorter output
                        &scene.ci_render_bg,
                        &scene.ci_convert_bg,
                        &self.primitive_targets.depth_view,
                        None,
                        false, // mrt_enabled
                        None,  // pipeline statistics
                    );
                }
                WebRenderMode::Forward | WebRenderMode::Quad => {
                    let quad = self.render_mode == WebRenderMode::Quad;
                    // Projection initializes 12 vertices. Quad strips draw four.
                    if quad {
                        self.queue.write_buffer(&scene.buffers.indirect_args, 0, bytemuck::bytes_of(&4u32));
                    }
                    rmesh_render::record_forward_after_sort(
                        &mut encoder2, &self.pipelines, &scene.buffers, &self.targets,
                        if quad { &scene.quad_render_bg } else { &scene.render_bg },
                        Some(&scene.prepass_bg), scene.tet_count,
                        &self.primitive_targets.depth_view, None, quad, false,
                    );
                }
            }
        }

        record_blit(&mut encoder2, &self.blit_pipeline, &scene.blit_bg, &view);

        self.staging_belt.finish();
        self.queue.submit(std::iter::once(encoder2.finish()));
        self.staging_belt.recall();
        output.present();

        Ok(())
    }

    /// Resize the rendering surface (call when canvas size changes).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.targets = RenderTargets::new_color_only(
            &self.device,
            width,
            height,
            GBufferFormats {
                color: wgpu::TextureFormat::Rgba16Float,
                ..Default::default()
            },
        );

        // Recreate primitive depth target
        self.primitive_targets = PrimitiveTargets::new(&self.device, width, height);

        self.shell_hiz = rmesh_render::ShellHiZ::new(
            &self.device,
            &self.shell_hiz_pipelines,
            &self.primitive_targets.depth_view,
            width,
            height,
        );
        if let Some(scene) = &mut self.scene {
            // Projected subpixel culling depends on pixel dimensions even if
            // the aspect ratio (and therefore VP matrix) did not change.
            scene.last_sort_vp = None;
            scene.hw_compute_bg = create_hw_compute_bind_group(
                &self.device,
                &self.pipelines,
                &scene.buffers,
                &scene.material,
                &scene.sh_coeffs_buf,
                &self.shell_hiz.full_view,
            );
        }
        // Recreate blit bind group since color_view changed
        if let Some(scene) = &mut self.scene {
            scene.blit_bg =
                create_blit_bind_group(&self.device, &self.blit_pipeline, &self.targets.color_view);
        }
    }

    /// Begin a PBD grab at framebuffer coordinates. Returns the picked ID or -1.
    pub fn begin_vertex_grab(&mut self, x: f32, y: f32, radius: f32) -> i32 {
        if self.vertex_grab.is_some() || !radius.is_finite() || radius <= 0.0 {
            return -1;
        }
        let view = self.camera.view_matrix();
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        let projection = self.camera.projection_matrix(w / h);
        let Some(scene) = &mut self.scene else {
            return -1;
        };
        let density = scene
            .vertex_density
            .get_or_insert_with(|| rmesh_viewer_core::vertices::vertex_max_density(&scene.arrays));
        let Some(pick) = rmesh_viewer_core::vertices::pick_vertex(
            &scene.arrays.vertices,
            Some(density),
            view,
            projection * view,
            [w, h],
            [x, y],
            30.0,
            None,
        ) else {
            return -1;
        };
        let topology = scene.topology.get_or_insert_with(|| {
            rmesh_pbd::MeshTopology::build(
                &scene.arrays.indices,
                scene.arrays.vertex_count,
                scene.arrays.tet_count,
            )
        });
        let prepared =
            rmesh_viewer_core::vertices::prepare_grab(&scene.arrays, topology, &[pick.idx], radius);
        let pipelines = self
            .pbd_pipelines
            .get_or_insert_with(|| rmesh_pbd::PbdPipelines::new(&self.device));
        let solver = rmesh_pbd::PbdSolver::init_grab(
            &self.device,
            pipelines,
            &scene.buffers.vertices,
            &prepared.island,
            &prepared.coloring,
            16,
        );
        self.queue.write_buffer(
            &scene.buffers.tet_visibility,
            0,
            bytemuck::cast_slice(&vec![u32::MAX; scene.importance_mask.len()]),
        );
        self.vertex_grab = Some(VertexGrab {
            solver,
            initial: prepared.handle_initial,
            original: scene.arrays.vertices.clone(),
            delta: [0.0; 2],
            readback: None,
            restore_gpu_sort: self.use_gpu_sort,
        });
        self.set_gpu_sort(true);
        pick.idx as i32
    }

    pub fn move_vertex_grab(&mut self, dx: f32, dy: f32) {
        if !dx.is_finite() || !dy.is_finite() {
            return;
        }
        if let Some(grab) = &mut self.vertex_grab {
            if grab.readback.is_none() {
                grab.delta = [dx, dy];
            }
        }
    }

    pub fn vertex_grab_active(&self) -> bool {
        self.vertex_grab.is_some()
    }

    pub fn end_vertex_grab(&mut self, commit: bool) {
        let Some(grab) = &mut self.vertex_grab else {
            return;
        };
        let Some(scene) = &mut self.scene else {
            return;
        };
        if commit {
            if grab.readback.is_none() {
                // Mouseup can arrive before the next animation frame. Apply
                // the latest pointer displacement before capturing vertices.
                let view = self.camera.view_matrix();
                let projection = self.camera.projection_matrix(
                    self.surface_config.width as f32 / self.surface_config.height as f32,
                );
                let positions = rmesh_viewer_core::vertices::drag_positions(
                    &grab.initial,
                    view,
                    projection,
                    self.surface_config.height as f32,
                    grab.delta,
                );
                let mut encoder = self.device.create_command_encoder(&Default::default());
                grab.solver.step(
                    &self.queue,
                    &mut encoder,
                    self.pbd_pipelines.as_ref().unwrap(),
                    0.016,
                    &positions,
                );
                self.queue.submit([encoder.finish()]);
                grab.readback = Some(rmesh_viewer_core::vertices::VertexReadback::start(
                    &self.device,
                    &self.queue,
                    &scene.buffers.vertices,
                    grab.original.len(),
                ));
            }
        } else {
            let grab = self.vertex_grab.take().unwrap();
            self.queue.write_buffer(
                &scene.buffers.vertices,
                0,
                bytemuck::cast_slice(&grab.original),
            );
            scene.last_sort_vp = None;
            self.set_importance_enabled(self.importance_enabled);
            self.set_gpu_sort(grab.restore_gpu_sort);
        }
    }

    pub fn vertex_position(&self, index: u32) -> Vec<f32> {
        self.scene
            .as_ref()
            .filter(|s| index < s.arrays.vertex_count)
            .map(|s| s.arrays.vertices[index as usize * 3..index as usize * 3 + 3].to_vec())
            .unwrap_or_default()
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }
    /// Set an orbit camera pose, useful for bookmarks and reproducible captures.
    pub fn set_camera(&mut self, x: f32, y: f32, z: f32, tx: f32, ty: f32, tz: f32) {
        let position = Vec3::new(x, y, z);
        let target = Vec3::new(tx, ty, tz);
        if position.is_finite() && target.is_finite() && position.distance(target) > 0.001 {
            self.camera = Camera::new(position - target);
            self.camera.orbit_target = target;
            self.camera.position = position;
        }
    }
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy);
    }
    pub fn zoom(&mut self, delta: f32) {
        self.camera.zoom(delta);
    }

    /// Returns true if a scene is currently loaded.
    pub fn has_scene(&self) -> bool {
        self.scene.is_some()
    }

    /// Returns true if the interaction system is active (suppress camera controls in JS).
    pub fn is_interacting(&self) -> bool {
        self.interaction.is_active()
    }

    /// Process a key down event from JS. Returns a string describing the result.
    pub fn on_key_down(&mut self, key: &str) -> String {
        if key == "Delete" && !self.interaction.is_active() {
            self.delete_selected();
            return "Deleted selection".into();
        }
        if matches!(key, "l" | "L") {
            self.shoot_torch();
            return "Torch thrown".into();
        }
        let ctx = self.interact_ctx();
        if let Some(ikey) = js_key_to_interact(key) {
            self.sync_current_transform();
            let result = self
                .interaction
                .process_event(&InteractEvent::KeyDown(ikey), &ctx);
            if let InteractResult::Confirmed(t) = result {
                self.apply_committed_transform(t);
            }
            if !matches!(result, InteractResult::NotConsumed) {
                return format!("{:?}", result);
            }
        }
        if key.len() == 1 {
            let ch = key.chars().next().unwrap();
            if matches!(ch, '0'..='9' | '.' | '-') {
                self.sync_current_transform();
                let result = self
                    .interaction
                    .process_event(&InteractEvent::CharInput(ch), &ctx);
                if let InteractResult::Confirmed(t) = result {
                    self.apply_committed_transform(t);
                }
                return format!("{:?}", result);
            }
        }
        "NotConsumed".to_string()
    }

    pub fn load_flare_model(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let model =
            rmesh_anim::gltf_loader::load_gltf_bytes(data, "Torch").map_err(|e| e.to_string())?;
        // Model installation happens before throwing; clear old indexed instances.
        self.flares.clear(&mut self.primitives);
        self.flares.configure_model(&model);
        self.flare_model = Some(model);
        self.upload_models();
        Ok(())
    }

    pub fn load_animation(&mut self, data: &[u8], name: &str) -> Result<(), JsValue> {
        self.animation =
            Some(rmesh_anim::gltf_loader::load_gltf_bytes(data, name).map_err(|e| e.to_string())?);
        self.animation_offsets = (0, 0);
        self.interaction.set_selected(None);
        if let Some(a) = &mut self.animation {
            a.scene.update(0.0);
        }
        self.upload_models();
        self.show_primitives = true;
        Ok(())
    }
    pub fn animation_node_count(&self) -> usize {
        self.animation.as_ref().map_or(0, |a| a.scene.nodes.len())
    }
    pub fn animation_node_name(&self, index: usize) -> String {
        self.animation
            .as_ref()
            .and_then(|a| a.scene.nodes.get(index))
            .map_or(String::new(), |n| n.name.clone())
    }
    pub fn animation_node_parent(&self, index: usize) -> i32 {
        self.animation
            .as_ref()
            .and_then(|a| a.scene.nodes.get(index))
            .and_then(|n| n.parent)
            .map_or(-1, |p| p as i32)
    }
    pub fn set_selected_node(&mut self, index: i32) {
        self.interaction.set_selected(
            if index >= 0 && (index as usize) < self.animation_node_count() {
                Some(rmesh_interact::Selection::Node(index as usize))
            } else {
                None
            },
        );
    }
    pub fn selected_node(&self) -> i32 {
        match self.interaction.selected() {
            Some(rmesh_interact::Selection::Node(i)) => i as i32,
            _ => -1,
        }
    }

    pub fn animation_clip_count(&self) -> usize {
        self.animation.as_ref().map_or(0, |a| a.scene.clips.len())
    }
    pub fn animation_clip_name(&self, index: usize) -> String {
        self.animation
            .as_ref()
            .and_then(|a| a.scene.clips.get(index))
            .map_or(String::new(), |c| c.name.clone())
    }
    pub fn animation_time(&self) -> f32 {
        self.animation
            .as_ref()
            .map_or(0.0, |a| a.scene.playback.time)
    }
    pub fn animation_duration(&self) -> f32 {
        self.animation
            .as_ref()
            .map_or(0.0, |a| a.scene.current_duration())
    }
    pub fn set_animation_playing(&mut self, playing: bool) {
        if let Some(a) = &mut self.animation {
            a.scene.playback.playing = playing;
        }
    }
    pub fn set_animation_clip(&mut self, index: usize) {
        if let Some(a) = &mut self.animation {
            if index < a.scene.clips.len() {
                a.scene.playback.clip_index = Some(index);
                a.scene.playback.time = 0.0;
                a.scene.update(0.0);
            }
        }
    }
    pub fn seek_animation(&mut self, time: f32) {
        if let Some(a) = &mut self.animation {
            a.scene.playback.time = time.clamp(0.0, a.scene.current_duration());
            a.scene.update(0.0);
        }
    }
    pub fn set_animation_speed(&mut self, speed: f32) {
        if speed.is_finite() {
            if let Some(a) = &mut self.animation {
                a.scene.playback.speed = speed;
            }
        }
    }
    pub fn set_animation_looping(&mut self, looping: bool) {
        if let Some(a) = &mut self.animation {
            a.scene.playback.looping = looping;
        }
    }

    pub fn set_collision_parameters(&mut self, threshold: f32, show: bool) {
        if !threshold.is_finite() { return; }
        self.flares.density_threshold = threshold.clamp(0.0, 1.0);
        self.flares.show_collision_mesh = show;
    }
    pub fn collision_triangle_count(&self) -> usize { self.flares.collision_triangle_count() }

    pub fn shoot_torch(&mut self) {
        let Some(scene) = &self.scene else {
            return;
        };
        self.flares.ensure_collision_mesh(&scene.arrays);
        let forward = -self.camera.view_matrix().inverse().col(2).truncate();
        self.flares.shoot(
            self.camera.position,
            forward,
            &mut self.primitives,
            &mut self.next_primitive_id,
        );
    }

    pub fn clear_torches(&mut self) {
        self.flares.clear(&mut self.primitives);
        self.interaction.set_selected(None);
    }
    pub fn set_sort_mode(&mut self, mode: &str) {
        let bits = match mode {
            "gpu16" => 16,
            "gpu" | "gpu32" | "cpu" => 32,
            _ => return,
        };
        if self.vertex_grab.is_some() { return; }
        if bits != self.sort_bits {
            self.sort_bits = bits;
            if let Some(scene) = &mut self.scene { scene.gpu_sort = None; }
        }
        self.set_gpu_sort(mode != "cpu");
    }

    pub fn sort_mode(&self) -> String {
        if !self.use_gpu_sort { "cpu" }
        else if self.sort_bits == 16 { "gpu16" } else { "gpu" }.into()
    }

    pub fn set_quality_preset(&mut self, name: &str) {
        if self.vertex_grab.is_some() { return; }
        let Some(preset) = QualityPreset::from_name(name) else { return; };
        let (sort, shell, importance, threshold) = preset.knobs();
        self.set_sort_mode(match sort {
            SortMode::Cpu => "cpu", SortMode::Gpu16 => "gpu16", SortMode::Gpu32 => "gpu",
        });
        self.set_shell_enabled(shell);
        self.set_importance_enabled(importance);
        self.set_importance_threshold(threshold);
    }

    pub fn quality_preset(&self) -> String {
        let sort = if !self.use_gpu_sort { SortMode::Cpu }
            else if self.sort_bits == 16 { SortMode::Gpu16 } else { SortMode::Gpu32 };
        QualityPreset::ALL.into_iter().find(|preset| {
            let (s, shell, importance, threshold) = preset.knobs();
            s == sort && shell == self.shell_enabled && importance == self.importance_enabled
                && (!importance || (threshold - self.importance_threshold).abs() < 1e-6)
        }).map_or("Custom", QualityPreset::label).into()
    }

    pub fn importance_threshold(&self) -> f32 { self.importance_threshold }
    pub fn importance_enabled(&self) -> bool { self.importance_enabled }
    pub fn shell_enabled(&self) -> bool { self.shell_enabled }

    pub fn set_gpu_sort(&mut self, enabled: bool) {
        // During a drag the CPU vertex copy still represents the last commit.
        if !enabled && self.vertex_grab.is_some() {
            return;
        }
        if enabled && self.gpu_sort_pipelines.is_none() {
            self.gpu_sort_pipelines = Some(rmesh_sort::RadixSortPipelines::new(
                &self.device,
                1,
                rmesh_sort::SortBackend::Basic,
            ));
        }
        if let Some(scene) = &mut self.scene {
            if enabled && scene.gpu_sort.is_none() {
                let state = rmesh_sort::RadixSortState::new(
                    &self.device,
                    scene.tet_count.next_power_of_two(),
                    self.sort_bits,
                    1,
                    rmesh_sort::SortBackend::Basic,
                );
                state.upload_configs(&self.queue);
                scene.gpu_sort = Some(state);
            }
            scene.last_sort_vp = None;
        }
        self.use_gpu_sort = enabled;
    }

    pub fn set_importance_enabled(&mut self, enabled: bool) {
        self.importance_enabled = enabled;
        if let Some(scene) = &mut self.scene {
            let mask = if enabled && self.vertex_grab.is_none() {
                scene.importance_mask.clone()
            } else {
                vec![u32::MAX; scene.importance_mask.len()]
            };
            self.queue.write_buffer(
                &scene.buffers.tet_visibility,
                0,
                bytemuck::cast_slice(&mask),
            );
            scene.last_sort_vp = None;
        }
    }

    pub fn set_importance_threshold(&mut self, threshold: f32) {
        if !threshold.is_finite() {
            return;
        }
        self.importance_threshold = threshold.clamp(0.001, 0.3);
        if let Some(scene) = &mut self.scene {
            scene.importance_mask = rmesh_viewer_core::importance::build_mask(
                &scene.importance_scores,
                &scene.max_od,
                self.importance_threshold,
                scene.tet_count as usize,
            )
            .0;
        }
        if let Some(scene) = &mut self.scene {
            scene.importance_kept = scene.importance_mask.iter().map(|v| v.count_ones()).sum();
        }
        self.set_importance_enabled(self.importance_enabled);
    }
    pub fn set_shell_parameters(&mut self, threshold: f32, erosion: u32) {
        if !threshold.is_finite() {
            return;
        }
        self.shell_threshold = threshold.clamp(0.0, 0.99);
        self.shell_erosion = erosion.min(24);
        self.rebuild_shell();
    }
    pub fn set_shell_cull(&mut self, enabled: bool, bias: f32) {
        if !bias.is_finite() {
            return;
        }
        self.shell_cull = enabled;
        self.shell_bias = bias.clamp(0.0, 0.01);
    }
    pub fn set_shell_entry(&mut self, entry: bool) {
        self.shell_entry = entry;
    }
    pub fn set_shell_only(&mut self, only: bool) {
        self.shell_only = only;
    }
    pub fn set_show_scene(&mut self, show: bool) {
        self.show_scene = show;
    }
    pub fn shell_face_count(&self) -> u32 {
        self.scene.as_ref().map_or(0, |s| s.shell.face_count)
    }
    pub fn importance_kept_count(&self) -> u32 {
        self.scene.as_ref().map_or(0, |s| s.importance_kept)
    }

    pub fn set_shell_enabled(&mut self, enabled: bool) {
        self.shell_enabled = enabled;
    }
    pub fn bake_ready(&self) -> bool {
        self.scene.as_ref().is_some_and(|s| s.light_bake.ready())
    }
    pub fn bake_progress(&self) -> f32 {
        self.scene.as_ref().map_or(0.0, |s| s.light_bake.progress())
    }
    pub fn bake_generation(&self) -> u32 {
        self.scene
            .as_ref()
            .map_or(0, |s| s.light_bake.generation as u32)
    }
    pub fn light_parameters(&self, index: usize) -> Vec<f32> {
        self.primitives
            .get(index)
            .filter(|p| matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight))
            .map(|p| {
                let c = p.color.unwrap_or([1.0, 1.0, 1.0, 3.0]);
                vec![
                    p.transform.position.x,
                    p.transform.position.y,
                    p.transform.position.z,
                    c[3],
                ]
            })
            .unwrap_or_default()
    }
    pub fn set_light(&mut self, index: usize, x: f32, y: f32, z: f32, intensity: f32) {
        if ![x, y, z, intensity].iter().all(|v| v.is_finite()) {
            return;
        }
        if let Some(p) = self.primitives.get_mut(index) {
            if matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight) {
                p.transform.position = Vec3::new(x, y, z);
                let mut color = p.color.unwrap_or([1.0, 1.0, 1.0, 3.0]);
                color[3] = intensity.max(0.0);
                p.color = Some(color);
            }
        }
    }

    /// Linear RGB, matching the desktop light color and cached transport.
    pub fn light_color(&self, index: usize) -> Vec<f32> {
        self.primitives
            .get(index)
            .filter(|p| matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight))
            .map(|p| p.color.unwrap_or([1.0, 1.0, 1.0, 3.0])[..3].to_vec())
            .unwrap_or_default()
    }

    pub fn set_light_color(&mut self, index: usize, r: f32, g: f32, b: f32) {
        if ![r, g, b].iter().all(|v| v.is_finite()) {
            return;
        }
        if let Some(p) = self.primitives.get_mut(index) {
            if matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight) {
                let intensity = p.color.unwrap_or([1.0, 1.0, 1.0, 3.0])[3];
                p.color = Some([r.max(0.0), g.max(0.0), b.max(0.0), intensity]);
            }
        }
    }

    pub fn light_direction(&self, index: usize) -> Vec<f32> {
        self.primitives
            .get(index)
            .filter(|p| p.kind == PrimitiveKind::SpotLight)
            .map(|p| (p.transform.rotation * Vec3::NEG_Z).to_array().to_vec())
            .unwrap_or_default()
    }

    pub fn set_light_direction(&mut self, index: usize, x: f32, y: f32, z: f32) {
        let direction = Vec3::new(x, y, z);
        if !direction.is_finite() || direction.length_squared() < 1e-12 {
            return;
        }
        if let Some(p) = self.primitives.get_mut(index) {
            if p.kind == PrimitiveKind::SpotLight {
                p.transform.rotation =
                    glam::Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize());
            }
        }
    }

    /// Process a key up event from JS.
    pub fn on_key_up(&mut self, key: &str) {
        if let Some(ikey) = js_key_to_interact(key) {
            let ctx = self.interact_ctx();
            self.sync_current_transform();
            self.interaction
                .process_event(&InteractEvent::KeyUp(ikey), &ctx);
        }
    }

    /// Feed mouse movement to the interaction system (when active).
    pub fn on_interact_mouse_move(&mut self, dx: f32, dy: f32) {
        let ctx = self.interact_ctx();
        self.sync_current_transform();
        self.interaction
            .process_event(&InteractEvent::MouseMove { dx, dy }, &ctx);
    }

    /// Feed mouse button to interaction. button: 0=left, 1=middle, 2=right.
    pub fn on_interact_mouse_down(&mut self, button: u32) -> String {
        let mb = match button {
            0 => rmesh_interact::MouseButton::Left,
            1 => rmesh_interact::MouseButton::Middle,
            2 => rmesh_interact::MouseButton::Right,
            _ => return "NotConsumed".to_string(),
        };
        let ctx = self.interact_ctx();
        self.sync_current_transform();
        let result = self
            .interaction
            .process_event(&InteractEvent::MouseDown { button: mb }, &ctx);
        if let InteractResult::Confirmed(t) = result {
            self.apply_committed_transform(t);
        }
        format!("{:?}", result)
    }

    /// Add a primitive of the given kind. kind: "cube", "sphere", "plane", "cylinder".
    pub fn add_primitive(&mut self, kind: &str) -> i32 {
        let pk = match kind {
            "cube" => PrimitiveKind::Cube,
            "sphere" => PrimitiveKind::Sphere,
            "plane" => PrimitiveKind::Plane,
            "cylinder" => PrimitiveKind::Cylinder,
            "light" => PrimitiveKind::PointLight,
            "spot" => PrimitiveKind::SpotLight,
            _ => return -1,
        };
        let name = format!("{}.{:03}", pk.label(), self.next_primitive_id);
        self.next_primitive_id += 1;
        self.primitives.push(Primitive::new(pk, name));
        let idx = self.primitives.len() - 1;
        self.interaction
            .set_selected(Some(rmesh_interact::Selection::Primitive(idx)));
        idx as i32
    }

    /// Set the selected primitive index (-1 for none).
    pub fn set_selected(&mut self, idx: i32) {
        let sel = if idx < 0 {
            None
        } else {
            Some(rmesh_interact::Selection::Primitive(idx as usize))
        };
        self.interaction.set_selected(sel);
    }

    /// Get the selected primitive index (-1 for none). Returns -1 for non-primitive selections too.
    pub fn get_selected(&self) -> i32 {
        match self.interaction.selected() {
            Some(rmesh_interact::Selection::Primitive(i)) => i as i32,
            _ => -1,
        }
    }

    /// Delete the selected primitive.
    pub fn delete_selected(&mut self) {
        match self.interaction.selected() {
            Some(rmesh_interact::Selection::Primitive(i)) => {
                self.flares.remove_primitive(&mut self.primitives, i)
            }
            Some(rmesh_interact::Selection::Node(i)) => {
                if let Some(a) = &mut self.animation {
                    a.scene.remove_node_and_descendants(i);
                }
            }
            None => {}
        }
        self.interaction.set_selected(None);
    }

    /// Get display info as "mode|axis|numeric" or empty string if idle.
    pub fn display_info(&self) -> String {
        if let Some(info) = self.interaction.display_info() {
            let axis_str = info.axis.label().unwrap_or_default();
            format!("{}|{}|{}", info.mode.label(), axis_str, info.numeric_text)
        } else {
            String::new()
        }
    }

    /// Get primitive count.
    pub fn primitive_count(&self) -> u32 {
        self.primitives.len() as u32
    }

    /// Get primitive name by index.
    pub fn primitive_name(&self, idx: u32) -> String {
        self.primitives
            .get(idx as usize)
            .map_or(String::new(), |p| p.name.clone())
    }

    /// Toggle the primitive compositor overlay.
    pub fn set_show_primitives(&mut self, show: bool) {
        self.show_primitives = show;
    }

    /// Get the current show_primitives state.
    pub fn get_show_primitives(&self) -> bool {
        self.show_primitives
    }
}

/// Map JS `event.key` string to `InteractKey`.
fn js_key_to_interact(key: &str) -> Option<InteractKey> {
    match key {
        "g" | "G" => Some(InteractKey::G),
        "s" | "S" => Some(InteractKey::S),
        "r" | "R" => Some(InteractKey::R),
        "x" | "X" => Some(InteractKey::X),
        "y" | "Y" => Some(InteractKey::Y),
        "z" | "Z" => Some(InteractKey::Z),
        "Shift" => Some(InteractKey::Shift),
        "Enter" => Some(InteractKey::Enter),
        "Escape" => Some(InteractKey::Escape),
        "Backspace" => Some(InteractKey::Backspace),
        "Delete" => Some(InteractKey::Delete),
        "Tab" => Some(InteractKey::Tab),
        _ => None,
    }
}

// Private helpers (not exported to JS)
impl WebViewer {
    #[cfg(target_arch = "wasm32")]
    fn poll_vertex_grab(&mut self) -> Result<(), JsValue> {
        let result = self
            .vertex_grab
            .as_ref()
            .and_then(|g| g.readback.as_ref())
            .and_then(|r| r.poll());
        let Some(result) = result else {
            return Ok(());
        };
        let vertices = match result {
            Ok(v) if v.iter().all(|x| x.is_finite()) => v,
            _ => {
                self.end_vertex_grab(false);
                return Err(JsValue::from_str(
                    "Vertex deformation readback failed; restored original geometry",
                ));
            }
        };
        let grab = self.vertex_grab.take().unwrap();
        if let Some(scene) = &mut self.scene {
            let data = Arc::make_mut(&mut scene.arrays);
            data.vertices = vertices;
            rmesh_viewer_core::vertices::refresh_bounds(data, &self.queue, &scene.buffers);
            let opacity = rmesh_viewer_core::flare::tet_alphas(
                &data.vertices,
                &data.indices,
                &data.densities,
                data.tet_count as usize,
            );
            let max_od = rmesh_viewer_core::importance::tet_max_od(
                &data.vertices,
                &data.indices,
                &data.densities,
                data.tet_count as usize,
            );
            let scores = rmesh_viewer_core::importance::weight_bound(
                &scene.neighbors,
                &opacity,
                &max_od,
                data.tet_count as usize,
            );
            scene.importance_mask = rmesh_viewer_core::importance::build_mask(
                &scores,
                &max_od,
                self.importance_threshold,
                data.tet_count as usize,
            )
            .0;
            let faces = rmesh_viewer_core::flare::extract_boundary_faces_adj(
                &scene.neighbors,
                &opacity,
                data.tet_count as usize,
                self.shell_threshold,
                self.shell_erosion,
            );
            scene.shell.upload(&self.device, &self.queue, &faces);
            let aux = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shell_aux_stub"),
                    contents: bytemuck::bytes_of(&0u32),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            scene.shell_bg = rmesh_render::create_shell_bind_group(
                &self.device,
                &self.shell_pipeline,
                &scene.shell,
                &scene.buffers,
                &scene.material,
                &scene.sh_coeffs_buf,
                &aux,
            );
            scene.importance_kept = scene.importance_mask.iter().map(|v| v.count_ones()).sum();
            scene.opacity = opacity;
            scene.max_od = max_od;
            scene.importance_scores = scores;
            scene.last_sort_vp = None;
            scene.light_bake.invalidate();
        }
        self.flares.invalidate_geometry();
        if self.flares.flare_count() > 0 {
            if let Some(scene) = &self.scene {
                self.flares.ensure_collision_mesh(&scene.arrays);
            }
        }
        self.set_importance_enabled(self.importance_enabled);
        self.set_gpu_sort(grab.restore_gpu_sort);
        Ok(())
    }

    fn rebuild_shell(&mut self) {
        if let Some(scene) = &mut self.scene {
            let faces = rmesh_viewer_core::flare::extract_boundary_faces_adj(
                &scene.neighbors,
                &scene.opacity,
                scene.tet_count as usize,
                self.shell_threshold,
                self.shell_erosion,
            );
            scene.shell.upload(&self.device, &self.queue, &faces);
            let aux = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shell_aux_stub"),
                    contents: bytemuck::bytes_of(&0u32),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            scene.shell_bg = rmesh_render::create_shell_bind_group(
                &self.device,
                &self.shell_pipeline,
                &scene.shell,
                &scene.buffers,
                &scene.material,
                &scene.sh_coeffs_buf,
                &aux,
            );
        }
    }

    fn upload_models(&mut self) {
        let mut models = Vec::new();
        if let Some(flare) = &self.flare_model {
            models.push(flare);
        }
        if let Some(animation) = &self.animation {
            models.push(animation);
        }
        let bases = rmesh_viewer_core::assets::upload_models(
            &self.device,
            &self.queue,
            &mut self.primitive_geometry,
            &mut self.material_registry,
            &models,
        );
        if let Some(animation) = &mut self.animation {
            for node in &mut animation.scene.nodes {
                if let Some(PrimitiveKind::CustomMesh(ref mut i)) = node.mesh_kind {
                    *i -= self.animation_offsets.0;
                }
                if let Some(ref mut i) = node.material_index {
                    *i -= self.animation_offsets.1;
                }
            }
            let (mesh_base, material_base) = *bases.last().unwrap();
            rmesh_viewer_core::assets::offset_scene(&mut animation.scene, mesh_base, material_base);
            self.animation_offsets = (mesh_base, material_base);
        }
    }

    fn sync_current_transform(&mut self) {
        let cur = match self.interaction.selected() {
            Some(rmesh_interact::Selection::Primitive(i)) => {
                self.primitives.get(i).map(|p| p.transform)
            }
            Some(rmesh_interact::Selection::Node(i)) => self
                .animation
                .as_ref()
                .and_then(|a| a.scene.nodes.get(i))
                .map(|n| n.world_transform),
            _ => None,
        };
        self.interaction.set_current_transform(cur);
    }

    fn apply_committed_transform(&mut self, t: rmesh_interact::Transform) {
        match self.interaction.selected() {
            Some(rmesh_interact::Selection::Primitive(i)) => {
                if let Some(p) = self.primitives.get_mut(i) {
                    p.transform = t;
                }
            }
            Some(rmesh_interact::Selection::Node(i)) => {
                if let Some(a) = &mut self.animation {
                    a.scene.set_node_world_transform(i, t);
                }
            }
            None => {}
        }
    }

    fn interact_ctx(&mut self) -> InteractContext {
        let w = self.surface_config.width;
        let h = self.surface_config.height;
        let aspect = w as f32 / h as f32;
        InteractContext {
            view_matrix: self.camera.view_matrix(),
            proj_matrix: self.camera.projection_matrix(aspect),
            viewport_width: w as f32,
            viewport_height: h as f32,
        }
    }

    fn upload_scene(
        &mut self,
        scene_data: rmesh_data::SceneData,
        sh_coeffs: rmesh_data::ShCoeffs,
        pbr: Option<rmesh_data::PbrData>,
    ) {
        self.vertex_grab = None;
        let tet_count = scene_data.tet_count;

        // Reset camera to scene start pose
        let pos = Vec3::new(
            scene_data.start_pose[0],
            scene_data.start_pose[1],
            scene_data.start_pose[2],
        );
        if pos.length() > 0.001 {
            self.camera = Camera::new(pos);
        }

        let material_min_t = pbr.as_ref().and_then(|p|p.render_min_t).unwrap_or(0.0);
        let buffers = SceneBuffers::upload(&self.device, &self.queue, &scene_data);

        let (mut material, sh_coeffs_buf) = rmesh_viewer_core::light_bake::upload_material(
            &self.device,
            &self.queue,
            &scene_data,
            pbr.as_ref(),
            &buffers,
        );
        material.colors = material.base_colors.clone();

        let light_bake = LightBake::new(
            &self.device,
            &self.queue,
            &scene_data,
            &sh_coeffs,
            pbr.as_ref(),
            &buffers,
        );
        light_bake.configure_interval(&self.device, &mut self.compute_interval_pipelines);
        self.flares.clear(&mut self.primitives);
        self.primitives = vec![default_light(&scene_data)];
        self.interaction.set_selected(None);
        self.last_frame = web_time::Instant::now();

        let opacity = rmesh_viewer_core::flare::tet_alphas(
            &scene_data.vertices,
            &scene_data.indices,
            &scene_data.densities,
            tet_count as usize,
        );
        let neighbors =
            rmesh_raytrace::compute_tet_neighbors(&scene_data.indices, tet_count as usize);
        let max_od = rmesh_viewer_core::importance::tet_max_od(
            &scene_data.vertices,
            &scene_data.indices,
            &scene_data.densities,
            tet_count as usize,
        );
        let scores = rmesh_viewer_core::importance::weight_bound(
            &neighbors,
            &opacity,
            &max_od,
            tet_count as usize,
        );
        let (importance_mask, kept, _) = rmesh_viewer_core::importance::build_mask(
            &scores,
            &max_od,
            self.importance_threshold,
            tet_count as usize,
        );
        log::info!("Importance culling keeps {kept}/{tet_count} tets");
        if self.importance_enabled {
            self.queue.write_buffer(
                &buffers.tet_visibility,
                0,
                bytemuck::cast_slice(&importance_mask),
            );
        }
        let faces = rmesh_viewer_core::flare::extract_boundary_faces_adj(
            &neighbors,
            &opacity,
            tet_count as usize,
            self.shell_threshold,
            self.shell_erosion,
        );
        let mut shell = rmesh_render::ShellBuffers::new(&self.device);
        shell.upload(&self.device, &self.queue, &faces);
        let aux_stub = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shell_aux_stub"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let shell_bg = rmesh_render::create_shell_bind_group(
            &self.device,
            &self.shell_pipeline,
            &shell,
            &buffers,
            &material,
            &sh_coeffs_buf,
            &aux_stub,
        );

        // The unused tile-projection fallback gets a far-plane stub; the active
        // HW projection below samples the live shell pyramid.
        let (_shell_hiz_stub, shell_hiz_stub_view) =
            rmesh_render::dummy_shell_hiz(&self.device, &self.queue);
        let compute_bg = create_compute_bind_group(
            &self.device,
            &self.pipelines,
            &buffers,
            &material,
            &sh_coeffs_buf,
            &shell_hiz_stub_view,
        );
        let hw_compute_bg = create_hw_compute_bind_group(
            &self.device,
            &self.pipelines,
            &buffers,
            &material,
            &sh_coeffs_buf,
            &self.shell_hiz.full_view,
        );
        // Both sort paths leave final indices in buffers.sort_values.
        let gen_bg_a = create_compute_interval_gen_bind_group(
            &self.device,
            &self.compute_interval_pipelines,
            &buffers,
            &material,
        );
        let ci_render_bg = create_compute_interval_render_bind_group(
            &self.device,
            &self.compute_interval_pipelines,
            &buffers,
        );
        let ci_convert_bg = create_compute_interval_indirect_convert_bind_group(
            &self.device,
            &self.compute_interval_pipelines,
            &buffers,
        );
        // Both sort backends leave their final indices in buffers.sort_values.
        let render_bg =
            create_render_bind_group(&self.device, &self.pipelines, &buffers, &material);
        let quad_render_bg = rmesh_render::create_quad_render_bind_group(
            &self.device, &self.pipelines, &buffers);
        let prepass_bg = rmesh_render::create_prepass_bind_group(
            &self.device, &self.pipelines, &buffers, &material, &buffers.sort_values);
        let blit_bg =
            create_blit_bind_group(&self.device, &self.blit_pipeline, &self.targets.color_view);

        let arrays = Arc::new(scene_data);
        let cpu_sorter = CpuSorter::new(tet_count as usize);

        self.scene = Some(SceneState {
            buffers,
            material,
            compute_bg,
            hw_compute_bg,
            gen_bg_a,
            render_bg,
            quad_render_bg,
            prepass_bg,
            ci_render_bg,
            ci_convert_bg,
            blit_bg,
            sh_coeffs_buf,
            sh_degree: 0,
            tet_count,
            cpu_sorter,
            importance_mask,
            importance_kept: kept as u32,
            importance_scores: scores,
            max_od,
            opacity,
            neighbors,
            topology: None,
            vertex_density: None,
            arrays,
            light_bake,
            material_min_t,
            shell,
            shell_bg,
            last_sort_vp: None,
            visible_count: 0,
            gpu_sort: None,
        });
        #[cfg(target_arch = "wasm32")]
        if self.use_gpu_sort {
            self.set_gpu_sort(true);
        }
    }
}
