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
use glam::{Mat4, Vec3};
use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use rayon::prelude::*;
use rmesh_render::{
    create_compute_bind_group, create_render_bind_group,
    create_render_bind_group_with_sort_values, BlitPipeline, ForwardPipelines, MaterialBuffers,
    RenderTargets, SceneBuffers, Uniforms, create_blit_bind_group, record_blit,
};

// SH basis function constants (match eval_sh_py.py / webrm)
const C0: f32 = 0.28209479;
const C1: f32 = 0.48860251;
const C2: [f32; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];
const C3: [f32; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];

/// Orbit camera matching the webrm Camera class.
struct Camera {
    position: Vec3,
    orbit_target: Vec3,
    orbit_distance: f32,
    orbit_yaw: f32,
    orbit_pitch: f32,
    fov_y: f32,
    near_z: f32,
    far_z: f32,
}

impl Camera {
    fn new(position: Vec3) -> Self {
        let distance = position.length();
        let pitch = (position.z / distance).asin();
        let yaw = position.y.atan2(position.x);

        Self {
            position,
            orbit_target: Vec3::ZERO,
            orbit_distance: distance,
            orbit_yaw: yaw,
            orbit_pitch: pitch,
            fov_y: 50.0_f32.to_radians(),
            near_z: 0.01,
            far_z: 1000.0,
        }
    }

    fn view_matrix(&mut self) -> Mat4 {
        let d = self.orbit_distance;
        let yaw = self.orbit_yaw;
        let pitch = self.orbit_pitch;

        let eye = self.orbit_target
            + Vec3::new(
                d * pitch.cos() * yaw.cos(),
                d * pitch.cos() * yaw.sin(),
                d * pitch.sin(),
            );

        self.position = eye;

        // Z-up look-at
        let up = Vec3::new(0.0, 0.0, -1.0);
        Mat4::look_at_rh(eye, self.orbit_target, up)
    }

    fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near_z, self.far_z)
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.004;
        self.orbit_yaw -= dx * sensitivity;
        self.orbit_pitch += dy * sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.001;
        self.orbit_pitch = self.orbit_pitch.clamp(-limit, limit);
    }

    fn zoom(&mut self, delta: f32) {
        self.orbit_distance = (self.orbit_distance + delta * 0.01).max(0.1);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.002 * self.orbit_distance;
        let yaw = self.orbit_yaw;
        let pitch = self.orbit_pitch;

        // Right vector (perpendicular to forward in XY plane)
        let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        // Up vector (perpendicular to forward and right)
        let up = Vec3::new(
            -pitch.sin() * yaw.cos(),
            -pitch.sin() * yaw.sin(),
            pitch.cos(),
        );

        self.orbit_target += sensitivity * (-dx * right + dy * up);
    }
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipelines: ForwardPipelines,
    blit_pipeline: BlitPipeline,
    sort_pipelines: rmesh_sort::RadixSortPipelines,
    sort_state: rmesh_sort::RadixSortState,
    buffers: SceneBuffers,
    material_buffers: MaterialBuffers,
    targets: RenderTargets,
    compute_bg: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,
    render_bg_b: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
    tet_count: u32,
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
    last_mouse: (f64, f64),
}

impl App {
    fn new(scene: rmesh_data::SceneData, sh: rmesh_data::ShCoeffs) -> Self {
        let pos = Vec3::new(scene.start_pose[0], scene.start_pose[1], scene.start_pose[2]);
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
            last_mouse: (0.0, 0.0),
        }
    }

    /// Evaluate SH coefficients to base colors for the current camera position (parallel).
    ///
    /// Matches the webrm compute shader (webgpu_rs.js) exactly:
    ///   - Planar memory layout: featIndex = channel * numCoeffs + coeff
    ///   - Direction = centroid - camPos (point FROM camera TO tet)
    ///   - Degree-1 signs: -C1*y, +C1*z, -C1*x
    ///   - Applies softplus(SH + 0.5 + grad_offset_at_v0, beta=10)
    fn evaluate_sh_colors(&self) -> Vec<f32> {
        let t_count = self.scene_data.tet_count as usize;
        let degree = self.sh_coeffs.degree as usize;
        let nc = (degree + 1) * (degree + 1); // numCoeffs per channel
        let stride = nc * 3; // total floats per tet
        let cam_pos = self.camera.position;
        let sh = &self.sh_coeffs.coeffs;
        let verts = &self.scene_data.vertices;
        let indices = &self.scene_data.indices;
        let grads = &self.scene_data.color_grads;

        let mut base_colors = vec![0.0f32; t_count * 3];
        base_colors
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(t, rgb)| {
                let sh_base = t * stride;
                if sh_base + stride > sh.len() {
                    rgb[0] = 0.5;
                    rgb[1] = 0.5;
                    rgb[2] = 0.5;
                    return;
                }

                // Load v0 and centroid for gradient offset (matching webrm)
                let i0 = indices[t * 4] as usize;
                let i1 = indices[t * 4 + 1] as usize;
                let i2 = indices[t * 4 + 2] as usize;
                let i3 = indices[t * 4 + 3] as usize;
                let v0 = Vec3::new(verts[i0*3], verts[i0*3+1], verts[i0*3+2]);
                let v1 = Vec3::new(verts[i1*3], verts[i1*3+1], verts[i1*3+2]);
                let v2 = Vec3::new(verts[i2*3], verts[i2*3+1], verts[i2*3+2]);
                let v3 = Vec3::new(verts[i3*3], verts[i3*3+1], verts[i3*3+2]);
                let centroid = (v0 + v1 + v2 + v3) * 0.25;

                // Direction: centroid - camPos (matches webrm convention)
                let dir = (centroid - cam_pos).normalize_or_zero();
                let (x, y, z) = (dir.x, dir.y, dir.z);

                // Planar layout: sh[sh_base + channel * nc + coeff]
                // Evaluate per-channel SH
                for c in 0..3usize {
                    let ch_base = sh_base + c * nc;

                    // Degree 0: C0 * sh_dc
                    let mut val = C0 * sh[ch_base];

                    // Degree 1: -C1*y, +C1*z, -C1*x
                    if degree >= 1 {
                        val -= C1 * y * sh[ch_base + 1];
                        val += C1 * z * sh[ch_base + 2];
                        val -= C1 * x * sh[ch_base + 3];
                    }

                    // Degree 2
                    if degree >= 2 {
                        let xx = x * x;
                        let yy = y * y;
                        let zz = z * z;
                        let xy = x * y;
                        let yz = y * z;
                        let xz = x * z;
                        val += C2[0] * xy * sh[ch_base + 4];
                        val += C2[1] * yz * sh[ch_base + 5];
                        val += C2[2] * (2.0 * zz - xx - yy) * sh[ch_base + 6];
                        val += C2[3] * xz * sh[ch_base + 7];
                        val += C2[4] * (xx - yy) * sh[ch_base + 8];
                    }

                    // Degree 3
                    if degree >= 3 {
                        let xx = x * x;
                        let yy = y * y;
                        let zz = z * z;
                        val += C3[0] * y * (3.0 * xx - yy) * sh[ch_base + 9];
                        val += C3[1] * x * y * z * sh[ch_base + 10];
                        val += C3[2] * y * (4.0 * zz - xx - yy) * sh[ch_base + 11];
                        val += C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[ch_base + 12];
                        val += C3[4] * x * (4.0 * zz - xx - yy) * sh[ch_base + 13];
                        val += C3[5] * z * (xx - yy) * sh[ch_base + 14];
                        val += C3[6] * x * (xx - 3.0 * yy) * sh[ch_base + 15];
                    }

                    rgb[c] = val + 0.5;
                }

                // Add gradient offset at v0 (matching webrm: dot(grad, v0 - centroid))
                let grad = Vec3::new(grads[t*3], grads[t*3+1], grads[t*3+2]);
                let offset = grad.dot(v0 - centroid);
                rgb[0] += offset;
                rgb[1] += offset;
                rgb[2] += offset;

                // Softplus activation (beta=10), matching webrm:
                //   sp = 0.1 * log(1.0 + exp(10.0 * x))
                for c in 0..3usize {
                    rgb[c] = 0.1 * (1.0 + (10.0 * rgb[c]).exp()).ln();
                }
            });

        base_colors
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

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("No suitable GPU adapter found");

            log::info!("GPU: {:?}", adapter.get_info().name);

            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffers_per_shader_stage = 16;
            limits.max_storage_buffer_binding_size = 256 * 1024 * 1024; // 256 MB
            limits.max_buffer_size = 512 * 1024 * 1024; // 512 MB

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("rmesh device"),
                        required_features: wgpu::Features::SUBGROUP
                            | wgpu::Features::SHADER_FLOAT32_ATOMIC,
                        required_limits: limits,
                        ..Default::default()
                    },
                )
                .await
                .expect("Failed to create device");

            (adapter, device, queue)
        });

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
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let color_format = wgpu::TextureFormat::Rgba16Float;

        log::info!("Compiling shader pipelines...");
        let t0 = std::time::Instant::now();
        let pipelines = ForwardPipelines::new(&device, color_format);
        let blit_pipeline = BlitPipeline::new(&device, surface_format);
        log::info!("Pipelines compiled: {:.2}s", t0.elapsed().as_secs_f64());

        log::info!("Uploading scene buffers ({} tets)...", self.scene_data.tet_count);
        let t0 = std::time::Instant::now();
        let buffers = SceneBuffers::upload(&device, &queue, &self.scene_data);

        // Evaluate SH colors instead of flat gray
        let base_colors = self.evaluate_sh_colors();
        let material = MaterialBuffers::upload(
            &device,
            &base_colors,
            &self.scene_data.color_grads,
            self.scene_data.tet_count,
        );
        log::info!("Buffers uploaded: {:.2}s", t0.elapsed().as_secs_f64());

        let targets = RenderTargets::new(&device, size.width.max(1), size.height.max(1));

        // Radix sort state
        log::info!("Creating radix sort pipelines...");
        let t0 = std::time::Instant::now();
        let sort_pipelines = rmesh_sort::RadixSortPipelines::new(&device);
        let n_pow2 = (self.scene_data.tet_count as u32).next_power_of_two();
        let sort_state = rmesh_sort::RadixSortState::new(&device, n_pow2, 32);
        sort_state.upload_configs(&queue);
        log::info!("Sort pipelines: {:.2}s", t0.elapsed().as_secs_f64());

        let compute_bg = create_compute_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg = create_render_bind_group(&device, &pipelines, &buffers, &material);
        let render_bg_b = create_render_bind_group_with_sort_values(
            &device,
            &pipelines,
            &buffers,
            &material,
            &sort_state.values_b,
        );

        let blit_bg = create_blit_bind_group(&device, &blit_pipeline, &targets.color_view);
        log::info!("GPU init total: {:.2}s", t_total.elapsed().as_secs_f64());

        self.gpu = Some(GpuState {
            device,
            queue,
            surface,
            surface_config,
            pipelines,
            blit_pipeline,
            sort_pipelines,
            sort_state,
            buffers,
            material_buffers: material,
            targets,
            compute_bg,
            render_bg,
            render_bg_b,
            blit_bg,
            tet_count: self.scene_data.tet_count,
        });
    }

    fn render(&mut self) {
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                gpu.surface.configure(&gpu.device, &gpu.surface_config);
                return;
            }
            Err(e) => {
                log::error!("Surface error: {:?}", e);
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let w = gpu.surface_config.width;
        let h = gpu.surface_config.height;
        let aspect = w as f32 / h as f32;

        let view_mat = self.camera.view_matrix();
        let proj_mat = self.camera.projection_matrix(aspect);
        let vp = proj_mat * view_mat;

        // Extract c2w rotation from inverse view matrix
        // RH convention: camera axes are x=right, y=up, z=backward
        // Pinhole convention: x=right, y=DOWN, z=FORWARD
        let inv_view = view_mat.inverse();
        let cam_right = inv_view.col(0).truncate();
        let cam_up = inv_view.col(1).truncate();
        let cam_back = inv_view.col(2).truncate();
        let c2w = glam::Mat3::from_cols(cam_right, -cam_up, -cam_back);

        // Intrinsics from FOV
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
            tet_count: gpu.tet_count,
            step: 0,
            tile_size_u: 12,
            ray_mode: 0,
            min_t: 0.0,
            _pad1: [0; 5],
        };

        gpu.queue
            .write_buffer(&gpu.buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        // Upload view-dependent SH colors
        let base_colors = self.evaluate_sh_colors();
        gpu.queue.write_buffer(
            &gpu.material_buffers.base_colors,
            0,
            bytemuck::cast_slice(&base_colors),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward pass"),
            });

        // Sorted forward pass: compute → radix sort → HW render
        rmesh_render::record_sorted_forward_pass(
            &mut encoder,
            &gpu.device,
            &gpu.pipelines,
            &gpu.sort_pipelines,
            &gpu.sort_state,
            &gpu.buffers,
            &gpu.targets,
            &gpu.compute_bg,
            &gpu.render_bg,
            &gpu.render_bg_b,
            gpu.tet_count,
            &gpu.queue,
        );

        // Blit Rgba16Float render target to sRGB swapchain
        record_blit(&mut encoder, &gpu.blit_pipeline, &gpu.blit_bg, &view);

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
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
            gpu.blit_bg = create_blit_bind_group(
                &gpu.device,
                &gpu.blit_pipeline,
                &gpu.targets.color_view,
            );
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("rmesh viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.init_gpu(window.clone());
        self.window = Some(window);
        log::info!(
            "Viewer initialized: {} tets, {} vertices (SH degree {})",
            self.scene_data.tet_count,
            self.scene_data.vertex_count,
            self.sh_coeffs.degree,
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.resize(size);
            }

            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => {
                    self.left_pressed = state == ElementState::Pressed;
                }
                MouseButton::Middle => {
                    self.middle_pressed = state == ElementState::Pressed;
                }
                MouseButton::Right => {
                    self.right_pressed = state == ElementState::Pressed;
                }
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse.0;
                let dy = position.y - self.last_mouse.1;

                if self.left_pressed {
                    self.camera.orbit(dx as f32, dy as f32);
                }
                if self.middle_pressed {
                    self.camera.pan(dx as f32, dy as f32);
                }
                if self.right_pressed {
                    self.camera.zoom(dy as f32);
                }

                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.zoom(-scroll);
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
        .map_or(false, |ext| ext.eq_ignore_ascii_case("ply"));

    let (scene, sh) = if is_ply {
        rmesh_data::load_ply(&file_data).context("Failed to parse PLY file")?
    } else {
        rmesh_data::load_rmesh(&file_data)
            .or_else(|_| rmesh_data::load_rmesh_raw(&file_data))
            .context("Failed to parse scene file")?
    };

    log::info!(
        "Scene: {} vertices, {} tets, SH degree {}",
        scene.vertex_count,
        scene.tet_count,
        sh.degree,
    );

    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    let mut app = App::new(scene, sh);
    event_loop.run_app(&mut app)?;

    Ok(())
}
