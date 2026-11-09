//! Per-frame rendering: GPU readback, egui UI, command encoding, submit/present.

use rmesh_compositor::record_primitive_pass;
use rmesh_interact::{InteractContext, Primitive, PrimitiveKind};
use rmesh_render::{record_blit, Uniforms};

use crate::gpu_state::*;
use crate::App;

impl App {
    pub(crate) fn update_fps(&mut self) {
        let now = std::time::Instant::now();
        self.frame_times.push_back(now);
        // Keep a 1-second window
        while let Some(&front) = self.frame_times.front() {
            if now.duration_since(front).as_secs_f64() > 1.0 {
                self.frame_times.pop_front();
            } else {
                break;
            }
        }
        self.fps = self.frame_times.len() as f64;
    }

    pub(crate) fn render(&mut self) {
        self.update_fps();

        // Tick animation clock and update animated scene
        let anim_dt = self.anim_clock.tick();
        if let Some(ref mut scene) = self.animated_scene {
            scene.update(anim_dt);
        }

        // Tick flare physics
        self.flare_system
            .tick(anim_dt, &mut self.primitives, &self.scene_data);

        if self.gpu.is_none() {
            return;
        }

        let (w, h) = {
            let g = self.gpu.as_ref().unwrap();
            (g.surface_config.width, g.surface_config.height)
        };
        let aspect = w as f32 / h as f32;

        // Camera matrices (view_matrix updates self.camera.position)
        let view_mat = self.camera.view_matrix();
        let proj_mat = self.camera.projection_matrix(aspect);
        let vp = proj_mat * view_mat;

        if let Some(p) = self.pending_quality.take() {
            self.quality_knobs().apply(p);
        }
        let quality_now = self.quality_knobs().current();

        // Occlusion-cull mask: pick region for the (now-current) camera, upload on change.
        if self.shell_system.refresh_importance() {
            self.active_mask_dirty = true;
        }
        self.update_cull_mask();

        // Overdraw oracle + keep-mask sweep (RMESH_ORACLE): may override the
        // visibility mask, shell state and overdraw probe for this frame.
        self.oracle_pre_frame(view_mat, w, h);

        // One-shot diagnostic: overdraw / visibility breakdown for this exact view.
        if self.measure_subpixel {
            self.measure_subpixel = false;
            let drawn = self.gpu.as_ref().map_or(0, |g| g.visible_instance_count);
            log::info!(
                "[overdraw probe] baked PVS: cull_enabled={} region={} baked_visible={} of {} ({:.1}%) | \
                 GPU actually drew (visible_instance_count) = {} last frame",
                self.cull_enabled,
                self.active_region,
                self.active_baked_count,
                self.scene_data.tet_count,
                100.0 * self.active_baked_count as f32 / self.scene_data.tet_count.max(1) as f32,
                drawn,
            );
            // The region-21 PVS mask (u8 per tet), if a region is active, so the
            // probe can measure overdraw over only the PVS-visible set.
            let m = self.scene_data.tet_count as usize;
            let vis_row: Option<&[u8]> = if self.active_region >= 0 {
                self.scene_data.visibility.as_ref().and_then(|v| {
                    let r = self.active_region as usize;
                    v.masks.get(r * m..r * m + m)
                })
            } else {
                None
            };
            measure_subpixel_fraction(
                &vp,
                w,
                h,
                &self.scene_data.vertices,
                &self.scene_data.indices,
                &self.scene_data.densities,
                m,
                vis_row,
            );
        }
        // Snapshot the previous frame's view-projection BEFORE any state
        // update — the temporal pass needs it to reproject the history sample.

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
        let fps = self.fps;
        let visible_count = self.gpu.as_ref().map_or(0, |g| g.visible_instance_count);
        let cull_on = self.cull_enabled;
        let cull_region = self.active_region;
        let cull_baked = self.active_baked_count;
        let has_vis = self.scene_data.visibility.is_some();
        let gpu_times = self
            .gpu
            .as_ref()
            .map_or(GpuTimings::default(), |g| g.gpu_times_ms.clone());
        let cpu_times = self
            .gpu
            .as_ref()
            .map_or(CpuTimings::default(), |g| g.cpu_times_ms.clone());

        let gpu = self.gpu.as_mut().unwrap();

        let t_frame_start = std::time::Instant::now();

        // --- Non-blocking readback of previous frame's data ---
        {
            use std::sync::atomic::Ordering;
            gpu.device.poll(wgpu::PollType::Poll).ok();

            // Read previous frame's GPU timestamps
            let prev = (gpu.ts_frame + 1) % 2;
            if gpu.ts_readback_ready[prev].load(Ordering::Acquire) {
                let slice = gpu.ts_readback[prev].slice(..);
                let data = slice.get_mapped_range();
                // Mapped byte ranges need not be aligned for u64 on every backend.
                let ts: Vec<u64> = data
                    .chunks_exact(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                let p = gpu.ts_period_ns as f64 / 1_000_000.0; // convert to ms

                let project = (ts[1].wrapping_sub(ts[0])) as f64 * p;
                let prepass = (ts[3].wrapping_sub(ts[2])) as f64 * p;
                let sort = if ts[2] > ts[1] && ts[2] < ts[4] {
                    (ts[2].wrapping_sub(ts[1])) as f64 * p
                } else {
                    (ts[4].wrapping_sub(ts[1])) as f64 * p
                };
                let render = (ts[5].wrapping_sub(ts[4])) as f64 * p;
                // Slots 6..8 (SH eval) are never resolved — see TS_WRITTEN_COUNT.
                // Reading them here would report uninitialized buffer contents.
                let sh = 0.0;
                let total = sh + project + sort + prepass + render;

                gpu.gpu_times_ms = GpuTimings {
                    sh_eval_ms: sh as f32,
                    project_ms: project as f32,
                    sort_ms: sort as f32,
                    prepass_ms: prepass as f32,
                    render_ms: render as f32,
                    total_ms: total as f32,
                };

                drop(data);
                gpu.ts_readback[prev].unmap();
                gpu.ts_readback_ready[prev].store(false, Ordering::Release);
                gpu.ts_readback_mapped[prev].store(false, Ordering::Release);
            }

            // Read previous frame's postprocess-chain timings
            if gpu.pp_readback_ready[prev].load(Ordering::Acquire) {
                let labels = std::mem::take(&mut gpu.pp_labels_pending[prev]);
                {
                    let slice = gpu.pp_readback[prev].slice(..);
                    let data = slice.get_mapped_range();
                    // Mapped byte ranges need not be aligned for u64 on every backend.
                    let ts: Vec<u64> = data
                        .chunks_exact(8)
                        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                        .collect();
                    let p = gpu.ts_period_ns as f64 / 1_000_000.0;
                    let mut out = Vec::with_capacity(labels.len());
                    for (i, label) in labels.iter().enumerate() {
                        if 2 * i + 1 >= ts.len() {
                            break;
                        }
                        let dt = ts[2 * i + 1].wrapping_sub(ts[2 * i]) as f64 * p;
                        out.push((*label, dt as f32));
                    }
                    gpu.pp_times_ms = out;
                }
                gpu.pp_readback[prev].unmap();
                gpu.pp_readback_ready[prev].store(false, Ordering::Release);
                gpu.pp_readback_mapped[prev].store(false, Ordering::Release);
            }

            // RMESH_PP_PROFILE=1 dumps the frame breakdown periodically. Logged
            // outside the readback branch so it also covers frames without
            // auxiliary shell or diagnostic passes.
            static PP_PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *PP_PROFILE.get_or_init(|| std::env::var("RMESH_PP_PROFILE").is_ok())
                && gpu.ts_frame % 60 == 0
            {
                let chain: f32 = gpu.pp_times_ms.iter().map(|(_, t)| *t).sum();
                let parts: Vec<String> = gpu
                    .pp_times_ms
                    .iter()
                    .map(|(l, t)| format!("{l} {t:.3}"))
                    .collect();
                let t = gpu.gpu_times_ms.clone();
                let px = (gpu.surface_config.width as f64) * (gpu.surface_config.height as f64);
                let frags = gpu.frag_invocations as f64;
                log::info!(
                    "[pp {}x{} fps={:.1}] frags={:.1}M ({:.1}/px, {:.1} Gfrag/s) | geom {:.2}ms (proj {:.2} sort {:.2} gen {:.2} raster {:.2}) | chain {:.3}ms = {}",
                    gpu.surface_config.width,
                    gpu.surface_config.height,
                    fps,
                    frags / 1.0e6,
                    frags / px.max(1.0),
                    frags / (t.render_ms.max(1.0e-6) as f64 * 1.0e-3) / 1.0e9,
                    t.total_ms,
                    t.project_ms,
                    t.sort_ms,
                    t.prepass_ms,
                    t.render_ms,
                    chain,
                    parts.join(" + ")
                );
            }

            // Read previous frame's fragment-invocation count
            if gpu.frag_ready.load(Ordering::Acquire) {
                {
                    let slice = gpu.frag_readback.slice(..);
                    let data = slice.get_mapped_range();
                    gpu.frag_invocations = u64::from_le_bytes(data[0..8].try_into().unwrap());
                }
                gpu.frag_readback.unmap();
                gpu.frag_ready.store(false, Ordering::Release);
                gpu.frag_mapped.store(false, Ordering::Release);
            }

            // Read previous frame's instance count
            if gpu.instance_count_ready.load(Ordering::Acquire) {
                let slice = gpu.instance_count_readback.slice(..);
                let data = slice.get_mapped_range();
                gpu.visible_instance_count =
                    u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                drop(data);
                gpu.instance_count_readback.unmap();
                gpu.instance_count_ready.store(false, Ordering::Release);
                gpu.instance_count_mapped.store(false, Ordering::Release);
            }
        }
        let t_after_poll = std::time::Instant::now();

        // Apply deferred surface reconfigure (must happen before get_current_texture)
        if gpu.pending_reconfigure {
            gpu.pending_reconfigure = false;
            gpu.surface_config.present_mode = if self.vsync {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::Immediate
            };
            log::info!(
                "Reconfiguring surface: present_mode={:?} frame_latency={}",
                gpu.surface_config.present_mode,
                gpu.surface_config.desired_maximum_frame_latency
            );
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
        }

        // One-shot GPU timing probe: block (with a hard cap so it can't hang)
        // until the GPU is idle before acquiring, so the swapchain can't jam and
        // the frame completes. Times the wait so we see the real per-frame
        // wall-clock even when it's far past the 2s acquire timeout. Two frames:
        // the first completes a clean frame, the second reads back its drawn
        // count + per-stage timers.
        if self.timing_probe_frames > 0 {
            let t_drain = std::time::Instant::now();
            gpu.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(120)),
                })
                .ok();
            let drain_ms = t_drain.elapsed().as_secs_f64() * 1000.0;
            self.timing_probe_frames -= 1;
            let t = gpu.gpu_times_ms.clone();
            log::info!(
                "[timing probe] frame wall-clock {:.0}ms | GPU drew {} tets | timers: total {:.1}ms = \
                 SH {:.1} + Proj {:.1} + Sort {:.1} + Gen(prepass) {:.1} + Render {:.1}",
                drain_ms,
                gpu.visible_instance_count,
                t.total_ms,
                t.sh_eval_ms,
                t.project_ms,
                t.sort_ms,
                t.prepass_ms,
                t.render_ms,
            );
        }

        let t_before_acquire = std::time::Instant::now();
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                gpu.surface.configure(&gpu.device, &gpu.surface_config);
                return;
            }
            Err(e) => {
                log::error!(
                    "Surface error: {:?} (waited {:.1}ms)",
                    e,
                    t_before_acquire.elapsed().as_secs_f64() * 1000.0
                );
                return;
            }
        };
        let t_after_acquire = std::time::Instant::now();

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
            min_t: self.pbr_data.as_ref().and_then(|p|p.render_min_t).unwrap_or(0.0),
            sh_degree: gpu.sh_degree,
            near_plane: self.camera.near_z,
            far_plane: self.camera.far_z,
            shell_cull_bias: self.shell_system.cull_bias,
            // One frame behind the UI, like every other setting written here
            // (the egui frame runs after this upload). Harmless either way: the
            // pyramid is cleared to the far plane, so a stale-but-set flag finds
            // no occluder rather than an invalid one.
            shell_cull_flags: u32::from(self.shell_system.cull_active() && !gpu.light_bake.uses_energy()),
        };

        gpu.queue
            .write_buffer(&gpu.buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));

        let t_before_egui = std::time::Instant::now();

        // --- Vertex-select hover preview ---
        // Refresh which vertex would be picked if the user clicked now.
        // Skipped while grabbing (the handle is fixed) and when in egui's
        // pointer focus (avoid expensive scan when the cursor is in the UI).
        let hover_marker: Option<egui::Pos2> = if self.vertex_select.is_enabled()
            && !self.vertex_select.is_grabbing()
            && !self.egui_ctx.is_pointer_over_egui()
        {
            let mp = self.vertex_select.mouse_pos();
            let info = self.pick_vertex_info(mp, 30.0);
            self.pbd_hover_vertex = info.as_ref().map(|i| i.idx);
            // Convert from physical surface pixels to egui logical points.
            let ppp = self.egui_ctx.pixels_per_point().max(0.01);
            info.map(|i| egui::pos2(i.screen_pos[0] / ppp, i.screen_pos[1] / ppp))
        } else {
            self.pbd_hover_vertex = None;
            None
        };

        // Borrow `gpu` again (the hover call took &mut self → had to drop it).
        let gpu = self.gpu.as_mut().unwrap();

        // --- egui frame ---
        let window = self.window.as_ref().unwrap();
        let raw_input = gpu.egui_state.take_egui_input(window);
        let tet_count = gpu.tet_count;
        let mut open_file = false;
        let mut vsync = self.vsync;
        let mut render_mode = self.render_mode;
        let mut use_mesh_shader = self.use_mesh_shader;
        let mesh_supported = self.mesh_shader_supported;
        let mesh_shader_supported = self.mesh_shader_supported;
        let mut show_primitives = self.show_primitives;
        let mut show_scene = self.show_scene;
        let mut sort_mode = self.sort_mode;
        let mut pbd_radius = self.pbd_radius;
        let vertex_select_on = self.vertex_select.is_enabled();
        let interact_display = self.interaction.display_info();
        let interact_selected = self.interaction.selected();
        let mut new_selected: Option<rmesh_interact::Selection> = interact_selected;
        let mut add_primitive: Option<PrimitiveKind> = None;
        let mut delete_selected = false;
        let mut add_shadow_test_scene = false;
        let mut shoot_flare = false;
        let mut clear_flares = false;
        let mut density_threshold = self.flare_system.density_threshold;
        let mut show_collision_mesh = self.flare_system.show_collision_mesh;
        let mut shell_enabled = self.shell_system.enabled;
        let mut shell_cull_enabled = self.shell_system.cull_enabled;
        let mut shell_threshold = self.shell_system.density_threshold;
        let mut shell_cull_bias = self.shell_system.cull_bias;
        let mut shell_erosion = self.shell_system.erosion_steps;
        let mut shell_show_only = self.shell_system.show_only;
        let mut shell_face = self.shell_system.face;
        let mut importance_enabled = self.shell_system.importance_enabled;
        let mut importance_threshold = self.shell_system.importance_threshold;
        let shell_face_count = self.shell_system.face_count();
        let shell_rebuilding = self.shell_system.rebuild_pending();
        // Set by the threshold slider on release (see the Density Shell block).
        let mut shell_rebuild = false;
        let flare_count = self.flare_system.flare_count();
        let collision_tri_count = self.flare_system.collision_triangle_count();
        let bake_was_ready = gpu.light_bake.ready();
        let mut light_edits: Vec<_> = self.primitives.iter().enumerate()
            .filter(|(_, p)| matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight))
            .map(|(i, p)| (i, p.transform.position, p.color.unwrap_or([1.0, 1.0, 1.0, 3.0])))
            .collect();
        let primitives_ref = &self.primitives;
        let mut quality_pick: Option<crate::QualityPreset> = None;
        let mut load_glb = false;
        // Animation playback state (read from scene, written back after egui)
        let mut anim_playing = self
            .animated_scene
            .as_ref()
            .is_some_and(|s| s.playback.playing);
        let mut anim_looping = self
            .animated_scene
            .as_ref()
            .is_none_or(|s| s.playback.looping);
        let mut anim_speed = self
            .animated_scene
            .as_ref()
            .map_or(1.0, |s| s.playback.speed);
        let mut anim_time = self
            .animated_scene
            .as_ref()
            .map_or(0.0, |s| s.playback.time);
        let anim_duration = self
            .animated_scene
            .as_ref()
            .map_or(0.0, |s| s.current_duration());
        let mut anim_clip_index = self
            .animated_scene
            .as_ref()
            .and_then(|s| s.playback.clip_index);
        let anim_clip_names: Vec<String> = self.animated_scene.as_ref().map_or(Vec::new(), |s| {
            s.clips.iter().map(|c| c.name.clone()).collect()
        });
        let anim_scene_name = self.animated_scene.as_ref().map(|s| s.name.clone());
        let anim_nodes: Vec<(String, usize, bool, Option<usize>)> =
            self.animated_scene.as_ref().map_or(Vec::new(), |s| {
                s.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.name.clone(), i, n.mesh_kind.is_some(), n.parent))
                    .collect()
            });
        let mut anim_goto_start = false;
        let mut anim_goto_end = false;
        let mut unload_scene = false;
        #[allow(deprecated)]
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Open Scene...").clicked() {
                            open_file = true;
                            ui.close_menu();
                        }
                        if ui.button("Load Animation (.glb)...").clicked() {
                            load_glb = true;
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Settings", |ui| {
                        ui.checkbox(&mut vsync, "Vsync");
                        ui.label("Renderer");
                        ui.radio_value(&mut render_mode, RenderMode::Regular, "Regular");
                        ui.radio_value(&mut render_mode, RenderMode::Quad, "Quad");
                        if mesh_shader_supported {
                            ui.radio_value(&mut render_mode, RenderMode::MeshShader, "Mesh Shader");
                        }
                        ui.radio_value(&mut render_mode, RenderMode::IntervalShader, "Interval Shader");
                        ui.add_enabled(
                            mesh_supported,
                            egui::Checkbox::new(&mut use_mesh_shader, "Interval via Mesh Shader"),
                        )
                        .on_hover_text(if mesh_supported {
                            "Emit interval geometry from a mesh shader instead of a compute pass + vertex stage"
                        } else {
                            "This GPU/backend has no mesh shader support"
                        });
                        ui.checkbox(&mut show_primitives, "Show Primitives");
                        ui.checkbox(&mut show_scene, "Show Scene");
                        ui.separator();
                        ui.label("Flare Gun");
                        ui.add(egui::Slider::new(&mut density_threshold, 0.0..=1.0).text("Collision Density"));
                        ui.checkbox(&mut show_collision_mesh, format!("Show Collision Mesh ({})", collision_tri_count));
                        ui.separator();
                        ui.label("Density Shell");
                        ui.add_enabled(!gpu.light_bake.uses_energy(), egui::Checkbox::new(&mut shell_enabled, "Enable Shell"));
                        ui.add_enabled_ui(shell_enabled, |ui| {
                            // An opacity in [0, 1] -- the alpha one tet
                            // contributes to a crossing ray -- so the range is
                            // bounded and linear, and means the same thing in
                            // every scene. It was raw density, which spanned
                            // decades and had to be retuned per scene.
                            let slider = egui::Slider::new(&mut shell_threshold, 0.0..=0.99)
                                .text("Tet Opacity Threshold");
                            // Rebuild on release, not on drag: extraction is
                            // ~100ms on a multi-million-tet scene, fast enough
                            // to retune but not to hold a drag at frame rate.
                            if ui.add(slider).drag_stopped() {
                                shell_rebuild = true;
                            }
                            // Peels N tet layers off the dense set before
                            // extracting. Structures thinner than N disappear
                            // rather than getting an opaque shell the volume is
                            // far too transparent to justify. Same rebuild cost
                            // as the threshold, so also on release.
                            let erosion = egui::Slider::new(&mut shell_erosion, 0..=24)
                                .text("Erosion (tets)");
                            if ui.add(erosion).drag_stopped() {
                                shell_rebuild = true;
                            }

                            // Which surface of the dense region the shell is.
                            // Entry occludes the region's own interior, which is
                            // where the fragments are; Exit only occludes what
                            // is behind the whole region and so recovers much
                            // less. See the header of shell.wgsl.
                            ui.horizontal(|ui| {
                                ui.label("Surface");
                                ui.radio_value(
                                    &mut shell_face,
                                    rmesh_render::ShellFace::Entry,
                                    "Entry",
                                );
                                ui.radio_value(
                                    &mut shell_face,
                                    rmesh_render::ShellFace::Exit,
                                    "Exit",
                                );
                            });
                            ui.checkbox(&mut shell_cull_enabled, "Z-Cull Behind Shell");
                            ui.add(
                                egui::Slider::new(&mut shell_cull_bias, 0.0..=0.01)
                                    .text("Cull Bias"),
                            );
                            ui.checkbox(&mut shell_show_only, "Show Shell Only");
                            if shell_rebuilding {
                                ui.label("rebuilding...");
                            } else {
                                ui.label(format!("{} faces", shell_face_count));
                            }
                        });
                        ui.separator();
                        ui.checkbox(&mut importance_enabled, "Importance Cull")
                            .on_hover_text("Drop tets whose weight on any ray from free space is bounded under the threshold");
                        ui.add_enabled(
                            importance_enabled,
                            egui::Slider::new(&mut importance_threshold, 0.001..=0.3)
                                .logarithmic(true)
                                .text("Weight Floor"),
                        );
                    });
                    ui.menu_button(
                        format!("Quality: {}", quality_now.map_or("Custom", |p| p.label())),
                        |ui| {
                            for p in crate::QualityPreset::ALL {
                                if ui
                                    .radio(quality_now == Some(p), p.label())
                                    .on_hover_text(match p {
                                        crate::QualityPreset::Lowest => "CPU sort, importance cull 0.03",
                                        crate::QualityPreset::Low => "GPU 16-bit sort, importance cull 0.02",
                                        crate::QualityPreset::Medium => "GPU 16-bit sort, importance cull 0.01",
                                        crate::QualityPreset::High => "GPU 32-bit sort, shell + importance cull 0.01",
                                        crate::QualityPreset::Reference => "GPU 32-bit sort, no shell, no importance cull",
                                    })
                                    .clicked()
                                {
                                    quality_pick = Some(p);
                                }
                            }
                        },
                    );
                    // Sort selector — only the interval-shading path uses sort,
                    // so we visually gray it out otherwise but keep it clickable
                    // (selecting CPU before switching to IntervalShader is fine).
                    ui.menu_button(format!("Sort: {}", sort_mode), |ui| {
                        ui.radio_value(&mut sort_mode, SortMode::Gpu32, "GPU 32-bit");
                        ui.radio_value(&mut sort_mode, SortMode::Gpu16, "GPU 16-bit");
                        ui.radio_value(&mut sort_mode, SortMode::Cpu, "CPU (radsort + cull)");
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let cull_info = if !has_vis {
                            String::new()
                        } else if cull_on {
                            format!("  |  cull:ON r={} baked={}", cull_region, cull_baked)
                        } else {
                            "  |  cull:OFF".to_string()
                        };
                        let timing_details = format!(
                            "FPS: {:.0}  |  {} vis / {} tets  |  GPU: {:.1}ms (SH:{:.1} Proj:{:.1} Sort:{:.1} Pre:{:.1} Rend:{:.1})  |  CPU: {:.1}ms (Poll:{:.1} Acq:{:.1} Egui:{:.1} Enc:{:.1} Sub:{:.1} Pres:{:.1}){}",
                            fps, visible_count, tet_count,
                            gpu_times.total_ms,
                            gpu_times.sh_eval_ms,
                            gpu_times.project_ms,
                            gpu_times.sort_ms,
                            gpu_times.prepass_ms,
                            gpu_times.render_ms,
                            cpu_times.total_ms,
                            cpu_times.poll_readback_ms,
                            cpu_times.acquire_ms,
                            cpu_times.egui_ms,
                            cpu_times.encode_ms,
                            cpu_times.submit_ms,
                            cpu_times.present_ms,
                            cull_info,
                        );
                        ui.label(format!("{fps:.0} FPS · GPU {:.1} ms · CPU {:.1} ms", gpu_times.total_ms, cpu_times.total_ms)).on_hover_text(timing_details);
                    });
                });
            });

            // Primitives panel
            egui::SidePanel::left("primitives_panel").default_width(160.0).show(ctx, |ui| {
                ui.heading("Primitives");
                ui.horizontal(|ui| {
                    if ui.button("Cube").clicked() {
                        add_primitive = Some(PrimitiveKind::Cube);
                    }
                    if ui.button("Sphere").clicked() {
                        add_primitive = Some(PrimitiveKind::Sphere);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Plane").clicked() {
                        add_primitive = Some(PrimitiveKind::Plane);
                    }
                    if ui.button("Cylinder").clicked() {
                        add_primitive = Some(PrimitiveKind::Cylinder);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Point Light").clicked() {
                        add_primitive = Some(PrimitiveKind::PointLight);
                    }
                    if ui.button("Spot Light").clicked() {
                        add_primitive = Some(PrimitiveKind::SpotLight);
                    }
                });
                if ui.button("Add Shadow Test Scene").clicked() {
                    add_shadow_test_scene = true;
                }
                ui.horizontal(|ui| {
                    if ui.button("Shoot Flare (L)").clicked() {
                        shoot_flare = true;
                    }
                    if flare_count > 0
                        && ui.button(format!("Clear Flares ({})", flare_count)).clicked() {
                            clear_flares = true;
                        }
                });
                ui.separator();
                ui.heading("Lighting");
                if gpu.light_bake.ready() { ui.label("Baked · ambient off"); }
                else { ui.add(egui::ProgressBar::new(gpu.light_bake.progress()).text("Baking lighting")); }
                for (i, position, color) in &mut light_edits {
                    ui.label(&primitives_ref[*i].name);
                    ui.horizontal(|ui| {
                        for (label, value) in [("X", &mut position.x), ("Y", &mut position.y), ("Z", &mut position.z)] {
                            ui.add(egui::DragValue::new(value).speed(0.02).prefix(label));
                        }
                    });
                    let mut rgb = [color[0], color[1], color[2]];
                    ui.color_edit_button_rgb(&mut rgb);
                    color[..3].copy_from_slice(&rgb);
                    ui.add(egui::Slider::new(&mut color[3], 0.0..=100.0).text("Intensity"));
                }
                ui.separator();
                for (i, prim) in primitives_ref.iter().enumerate() {
                    let sel = rmesh_interact::Selection::Primitive(i);
                    let selected = new_selected == Some(sel);
                    if ui.selectable_label(selected, &prim.name).clicked() {
                        new_selected = if selected { None } else { Some(sel) };
                    }
                }
                if matches!(new_selected, Some(rmesh_interact::Selection::Primitive(_))) {
                    ui.separator();
                    if ui.button("Delete").clicked() {
                        delete_selected = true;
                    }
                }

                ui.separator();
                ui.heading("PBD Grab");
                let mode_label = if vertex_select_on { "ON (Tab)" } else { "off (Tab)" };
                ui.label(format!("Vertex select: {mode_label}"));
                ui.add(
                    egui::Slider::new(&mut pbd_radius, 0.01..=2.0)
                        .logarithmic(true)
                        .text("Radius"),
                );
                ui.small("Larger radius = wider deformation island.\nApplies on next grab.");
            });

            // Animated scene hierarchy panel (right side, only shown when a scene is loaded)
            if let Some(ref scene_name) = anim_scene_name {
                egui::SidePanel::right("anim_scene_panel").default_width(180.0).show(ctx, |ui| {
                    ui.heading("Animation");
                    ui.label(format!("Scene: {}", scene_name));
                    if ui.small_button("Unload").clicked() {
                        unload_scene = true;
                    }
                    ui.separator();

                    // Clip selector
                    if !anim_clip_names.is_empty() {
                        ui.label("Clip:");
                        let current_label = anim_clip_index
                            .and_then(|i| anim_clip_names.get(i))
                            .map_or("(none)", |s| s.as_str());
                        egui::ComboBox::from_id_salt("anim_clip")
                            .selected_text(current_label)
                            .show_ui(ui, |ui| {
                                for (i, name) in anim_clip_names.iter().enumerate() {
                                    if ui.selectable_value(&mut anim_clip_index, Some(i), name).clicked() {
                                        anim_time = 0.0;
                                    }
                                }
                            });
                    }
                    ui.separator();

                    // Node hierarchy tree
                    ui.label("Nodes:");
                    egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                        for (name, idx, has_mesh, parent) in &anim_nodes {
                            let depth = {
                                let mut d = 0usize;
                                let mut cur: Option<usize> = *parent;
                                while let Some(p) = cur {
                                    d += 1;
                                    cur = anim_nodes.get(p).and_then(|n| n.3);
                                }
                                d
                            };
                            let indent = "  ".repeat(depth);
                            let icon = if *has_mesh { "\u{25A0}" } else { "\u{25CB}" };
                            let sel = rmesh_interact::Selection::Node(*idx);
                            let selected = new_selected == Some(sel);
                            if ui
                                .selectable_label(selected, format!("{}{} {}", indent, icon, name))
                                .clicked()
                            {
                                new_selected = if selected { None } else { Some(sel) };
                            }
                        }
                    });
                    if matches!(new_selected, Some(rmesh_interact::Selection::Node(_))) {
                        ui.separator();
                        if ui.button("Delete Node").clicked() {
                            delete_selected = true;
                        }
                    }
                });
            }

            // Animation timeline (bottom panel, shown when a scene is loaded with clips)
            if anim_scene_name.is_some() && !anim_clip_names.is_empty() {
                egui::TopBottomPanel::bottom("anim_timeline").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        // Go to start
                        if ui.small_button("\u{23EE}").on_hover_text("Go to start").clicked() {
                            anim_goto_start = true;
                        }
                        // Play/Pause toggle
                        let play_label = if anim_playing { "\u{23F8}" } else { "\u{25B6}" };
                        if ui.small_button(play_label).on_hover_text(if anim_playing { "Pause" } else { "Play" }).clicked() {
                            anim_playing = !anim_playing;
                        }
                        // Go to end
                        if ui.small_button("\u{23ED}").on_hover_text("Go to end").clicked() {
                            anim_goto_end = true;
                        }
                        // Loop toggle
                        let loop_label = if anim_looping { "\u{1F501}" } else { "\u{27A1}" };
                        if ui.small_button(loop_label).on_hover_text(if anim_looping { "Looping" } else { "Play once" }).clicked() {
                            anim_looping = !anim_looping;
                        }
                        ui.separator();
                        // Time scrubber
                        let max_t = if anim_duration > 0.0 { anim_duration } else { 1.0 };
                        ui.add(
                            egui::Slider::new(&mut anim_time, 0.0..=max_t)
                                .text("")
                                .show_value(false)
                                .min_decimals(2)
                                .max_decimals(2),
                        );
                        // Time display
                        ui.label(format!("{:.2}s / {:.2}s", anim_time, anim_duration));
                        ui.separator();
                        // Speed
                        ui.label("Speed:");
                        ui.add(egui::DragValue::new(&mut anim_speed).range(0.0..=5.0).speed(0.05).suffix("x"));
                    });
                });
            }

            // VertexSelect hover preview: solid yellow dot at the vertex
            // that would be picked if the user clicked now.
            if let Some(pos) = hover_marker {
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("pbd_hover_marker"),
                ));
                painter.circle_filled(pos, 6.0, egui::Color32::from_rgb(255, 230, 0));
                painter.circle_stroke(
                    pos,
                    6.0,
                    egui::Stroke::new(1.5, egui::Color32::BLACK),
                );
            }

            // Transform HUD overlay (centered at bottom, below timeline if present)
            if let Some(ref info) = interact_display {
                egui::TopBottomPanel::bottom("interact_hud").show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.label(
                            egui::RichText::new(info.mode.label())
                                .size(16.0)
                                .strong(),
                        );
                        if let Some(axis_label) = info.axis.label() {
                            ui.label(
                                egui::RichText::new(axis_label)
                                    .size(16.0)
                                    .color(egui::Color32::from_rgb(100, 180, 255)),
                            );
                        }
                        if !info.numeric_text.is_empty() {
                            ui.label(
                                egui::RichText::new(&info.numeric_text)
                                    .size(16.0)
                                    .monospace(),
                            );
                        }
                    });
                });
            }

        });
        if open_file {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Scene", &["rmesh", "ply"])
                .pick_file()
            {
                self.pending_load = Some(path);
            }
        }
        if vsync != self.vsync {
            self.vsync = vsync;
            gpu.pending_reconfigure = true;
        }
        self.render_mode = render_mode;
        self.use_mesh_shader = use_mesh_shader;
        self.show_primitives = show_primitives;
        for (i, position, color) in light_edits {
            self.primitives[i].transform.position = position;
            self.primitives[i].color = Some(color);
        }
        self.show_scene = show_scene;
        self.sort_mode = sort_mode;
        self.pbd_radius = pbd_radius;
        self.flare_system.density_threshold = density_threshold;
        self.flare_system.show_collision_mesh = show_collision_mesh;

        let shell_was_enabled = self.shell_system.enabled;
        self.shell_system.enabled = shell_enabled;
        self.shell_system.cull_enabled = shell_cull_enabled;
        self.shell_system.density_threshold = shell_threshold;
        self.shell_system.cull_bias = shell_cull_bias;
        self.shell_system.erosion_steps = shell_erosion;
        self.shell_system.show_only = shell_show_only;
        self.shell_system.face = shell_face;
        self.shell_system.importance_enabled = importance_enabled;
        self.shell_system.importance_threshold = importance_threshold;
        if shell_rebuild || (shell_enabled && !shell_was_enabled) {
            self.shell_system.mark_dirty();
        }
        // A preset pick wins over the individual widgets written back above.
        if let Some(p) = quality_pick {
            crate::QualityKnobs {
                sort_mode: &mut self.sort_mode,
                shell: &mut self.shell_system,
            }
            .apply(p);
        }

        if gpu.light_bake.uses_energy() {
            self.render_mode = RenderMode::IntervalShader;
            self.use_mesh_shader = false;
            self.shell_system.enabled = false;
            self.shell_system.show_only = false;
        }

        // Density shell: kick off a background extraction if the threshold moved,
        // and install one if it has finished. The previous shell keeps drawing
        // in the meantime, so dragging the slider never stalls the frame.
        self.shell_system.maybe_start_rebuild(&self.scene_data);
        if self
            .shell_system
            .poll(&gpu.device, &gpu.queue, &mut gpu.shell_buffers)
        {
            // The face buffer was reallocated, so the old bind group points at a
            // dead buffer.
            let bg = crate::build_shell_bind_group(
                &gpu.device,
                &gpu.shell_pipeline,
                &gpu.shell_buffers,
                &gpu.buffers,
                &gpu.material_buffers,
                &gpu.sh_coeffs_buf,
                gpu.aux_data_buf.as_ref(),
            );
            gpu.shell_bg = bg;
        }

        // Install any async-rebuilt collision mesh kicked off by the PBD
        // ConfirmGrab path.
        self.flare_system.poll_rebuild();

        // Build/rebuild collision mesh if threshold changed (or first time showing)
        if show_collision_mesh || self.flare_system.flare_count() > 0 {
            self.flare_system.ensure_collision_mesh(&self.scene_data);
        }

        self.flare_system.upload_collision_debug(&gpu.device, &mut gpu.primitive_geometry);

        if shoot_flare {
            let fwd = (self.camera.orbit_target - self.camera.position).normalize();
            self.flare_system.shoot(
                self.camera.position,
                fwd,
                &mut self.primitives,
                &mut self.next_primitive_id,
            );
        }
        if clear_flares {
            self.flare_system.clear(&mut self.primitives);
        }

        // Write back animation playback state
        if let Some(ref mut scene) = self.animated_scene {
            scene.playback.playing = anim_playing;
            scene.playback.looping = anim_looping;
            scene.playback.speed = anim_speed;
            scene.playback.time = anim_time;
            scene.playback.clip_index = anim_clip_index;
            if anim_goto_start {
                scene.playback.time = 0.0;
            }
            if anim_goto_end {
                scene.playback.time = scene.current_duration();
                scene.playback.playing = false;
            }
        }
        if unload_scene {
            self.animated_scene = None;
            gpu.primitive_geometry.set_custom_meshes(&gpu.device, &[]);
            gpu.material_registry.clear();
        }

        // Handle .glb file load dialog
        if load_glb {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("glTF", &["glb", "gltf"])
                .pick_file()
            {
                log::info!("Loading animation file: {}", path.display());
                match rmesh_anim::gltf_loader::load_gltf(&path) {
                    Ok(mut gltf_scene) => {
                        let flare = if self.flare_system.has_flare_model {
                            rmesh_anim::gltf_loader::load_gltf(std::path::Path::new("assets/flare/scene.gltf")).ok()
                        } else { None };
                        let mut models = Vec::new();
                        if let Some(flare) = &flare { models.push(flare); }
                        models.push(&gltf_scene);
                        let bases = rmesh_viewer_core::assets::upload_models(&gpu.device, &gpu.queue, &mut gpu.primitive_geometry, &mut gpu.material_registry, &models);
                        let (mesh_base, material_base) = *bases.last().unwrap();
                        rmesh_viewer_core::assets::offset_scene(&mut gltf_scene.scene, mesh_base, material_base);

                        let mesh_count = gltf_scene.meshes.len();
                        let node_count = gltf_scene.scene.nodes.len();
                        let clip_count = gltf_scene.scene.clips.len();
                        let tex_count = gltf_scene.textures.len();
                        let mat_count = gltf_scene.materials.len();
                        log::info!(
                            "Loaded glTF: {} meshes, {} nodes, {} clips, {} textures, {} materials",
                            mesh_count,
                            node_count,
                            clip_count,
                            tex_count,
                            mat_count,
                        );
                        self.animated_scene = Some(gltf_scene.scene);
                    }
                    Err(e) => {
                        log::error!("Failed to load glTF: {}", e);
                    }
                }
            }
        }

        // Handle shadow test scene
        if add_shadow_test_scene {
            use rmesh_interact::PrimitiveKind;
            self.primitives.clear();
            self.interaction.set_selected(None);

            // Point light at origin
            let mut light = Primitive::new(PrimitiveKind::PointLight, "TestLight");
            light.transform.position = glam::Vec3::new(0.0, 0.0, 0.0);
            self.primitives.push(light);

            // Small occluder cube in +X direction
            let mut occluder = Primitive::new(PrimitiveKind::Cube, "Occluder_+X");
            occluder.transform.position = glam::Vec3::new(2.0, 0.0, 0.0);
            occluder.transform.scale = glam::Vec3::splat(0.5);
            self.primitives.push(occluder);

            // 6 large surface cubes in each axis direction
            let dirs: [(glam::Vec3, &str); 6] = [
                (glam::Vec3::X, "+X"),
                (glam::Vec3::NEG_X, "-X"),
                (glam::Vec3::Y, "+Y"),
                (glam::Vec3::NEG_Y, "-Y"),
                (glam::Vec3::Z, "+Z"),
                (glam::Vec3::NEG_Z, "-Z"),
            ];
            for (dir, label) in dirs {
                let mut wall = Primitive::new(PrimitiveKind::Cube, format!("Wall_{label}"));
                wall.transform.position = dir * 5.0;
                wall.transform.scale = glam::Vec3::splat(2.0);
                self.primitives.push(wall);
            }

            self.show_primitives = true;
            self.next_primitive_id = 10;
        }

        // Handle primitive add/delete/selection
        if let Some(kind) = add_primitive {
            let name = format!("{}.{:03}", kind.label(), self.next_primitive_id);
            self.next_primitive_id += 1;
            self.primitives.push(Primitive::new(kind, name));
            let idx = self.primitives.len() - 1;
            self.interaction
                .set_selected(Some(rmesh_interact::Selection::Primitive(idx)));
        } else if delete_selected {
            match self.interaction.selected() {
                Some(rmesh_interact::Selection::Primitive(idx)) => {
                    if idx < self.primitives.len() {
                        self.flare_system.remove_primitive(&mut self.primitives, idx);
                        self.interaction.set_selected(None);
                    }
                }
                Some(rmesh_interact::Selection::Node(idx)) => {
                    if let Some(ref mut scene) = self.animated_scene {
                        scene.remove_node_and_descendants(idx);
                        self.interaction.set_selected(None);
                    }
                }
                None => {}
            }
        } else {
            self.interaction.set_selected(new_selected);
        }

        gpu.egui_state
            .handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w, h],
            pixels_per_point: full_output.pixels_per_point,
        };

        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            gpu.egui_renderer
                .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
        }

        let t_after_egui = std::time::Instant::now();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward pass"),
            });

        gpu.light_bake.record(&gpu.queue, &mut encoder, &gpu.material_buffers, &self.primitives, pos);

        gpu.egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );

        // Record a PBD step if a vertex grab is in progress.
        if self.vertex_select.is_grabbing() {
            if let Some(solver) = gpu.pbd_solver.as_ref() {
                let mouse_now = self.vertex_select.mouse_pos();
                let mouse_start = self.vertex_select.mouse_start();
                let surf_w = gpu.surface_config.width as f32;
                let surf_h = gpu.surface_config.height as f32;
                let view = self.camera.view_matrix();
                let proj = self.camera.projection_matrix(surf_w / surf_h);
                let mouse_dx = mouse_now[0] - mouse_start[0];
                let mouse_dy = mouse_now[1] - mouse_start[1];
                let positions = rmesh_viewer_core::vertices::drag_positions(
                    &self.pbd_handle_initial, view, proj, surf_h, [mouse_dx, mouse_dy],
                );
                let max_drag = positions.iter().zip(&self.pbd_handle_initial)
                    .map(|(p, (_, initial))| glam::Vec3::from_slice(&p[..3]).distance(glam::Vec3::from(*initial)))
                    .fold(0.0_f32, f32::max);
                solver.step(
                    &gpu.queue,
                    &mut encoder,
                    &gpu.pbd_pipelines,
                    0.016,
                    &positions,
                );

                // Rate-limited per-step log: first 3 frames, then every 30.
                self.pbd_step_counter = self.pbd_step_counter.wrapping_add(1);
                let n = self.pbd_step_counter;
                if n <= 3 || n.is_multiple_of(30) {
                    log::info!(
                        "[pbd] step #{n} | {} particles, {} handles | mouseΔ=({:.1}, {:.1}) px | world drag={:.4}",
                        solver.num_particles, solver.num_handles, mouse_dx, mouse_dy, max_drag,
                    );
                }
            }
        }

        // 1. Primitive pass first: renders to volume color target + primitive depth target
        //    (clears both even if no primitives, so forward pass can use LoadOp::Load)
        {
            // Temporarily apply preview transform so the grabbed entity follows the mouse.
            // For primitives we patch in place and restore after the pass.
            // The shared node preview also transforms children and restores the scene.
            let mut node_preview = None;
            let preview_restore: Option<(usize, rmesh_interact::Transform)> =
                if self.show_primitives {
                    let ctx = InteractContext {
                        view_matrix: view_mat,
                        proj_matrix: proj_mat,
                        viewport_width: w as f32,
                        viewport_height: h as f32,
                    };
                    if let Some(preview) = self.interaction.preview_transform(&ctx) {
                        match self.interaction.selected() {
                            Some(rmesh_interact::Selection::Primitive(idx))
                                if idx < self.primitives.len() =>
                            {
                                let original = self.primitives[idx].transform;
                                self.primitives[idx].transform = preview;
                                Some((idx, original))
                            }
                            Some(rmesh_interact::Selection::Node(idx)) => {
                                node_preview = Some((idx, preview));
                                None
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

            // Build combined primitive list: manual primitives + animated scene nodes
            let combined_primitives: Vec<Primitive>;
            let render_prims: &[Primitive] = if self.show_primitives {
                let mut prims = self.primitives.clone();
                if let Some(ref mut scene) = self.animated_scene {
                    prims.extend(match node_preview {
                        Some((idx, transform)) => scene.render_node_preview(idx, transform),
                        None => scene.render_primitives(),
                    });
                }
                combined_primitives = prims;
                &combined_primitives
            } else {
                combined_primitives = Vec::new();
                &combined_primitives
            };

            record_primitive_pass(
                &mut encoder,
                &gpu.queue,
                &gpu.primitive_pipeline,
                &gpu.primitive_geometry,
                &gpu.targets.color_view,
                &gpu.primitive_targets.depth_view,
                render_prims,
                &vp,
                None,
                Some(&gpu.material_registry),
            );

            // Collision debug mesh overlay
            if self.flare_system.show_collision_mesh {
                rmesh_compositor::record_collision_debug_pass(
                    &mut encoder,
                    &gpu.queue,
                    &gpu.primitive_pipeline,
                    &gpu.primitive_geometry,
                    &gpu.targets.color_view,
                    &gpu.primitive_targets.depth_view,
                    &vp,
                    &gpu.material_registry,
                );
            }

            // Restore original transform after rendering
            if let Some((idx, original)) = preview_restore {
                self.primitives[idx].transform = original;
            }
        }

        let mrt_enabled = false;
        let skip_volume = !self.show_scene || self.shell_system.show_only;

        // --- GPU pass profiler ---
        // Brackets each pass with encoder-level timestamps. Slot i occupies
        // queries [2i, 2i+1]; `pp_labels` records what was written so only that
        // prefix is resolved (resolving a never-written query wedges the NVIDIA
        // driver). Covers the shell, its Hi-Z, and optional overdraw probe.
        let mut pp_labels: Vec<&'static str> = Vec::new();
        let pp_on = gpu.pp_ts_enabled;
        macro_rules! pp_time {
            ($label:expr, $body:block) => {{
                let slot = pp_labels.len() as u32;
                let armed = pp_on && (slot * 2 + 1) < PP_TS_COUNT;
                if armed {
                    encoder.write_timestamp(&gpu.pp_query_set, slot * 2);
                }
                $body
                if armed {
                    encoder.write_timestamp(&gpu.pp_query_set, slot * 2 + 1);
                    pp_labels.push($label);
                }
            }};
        }

        // 1b. Density shell: the boundary surface of the dense region, drawn as
        //     opaque geometry into the same color + depth targets the primitives
        //     just wrote. Two jobs at once — its color becomes the volume's
        //     backstop (residual transmittance reveals the model instead of the
        //     background), and its depth becomes the bound everything past it is
        //     culled against.
        if self.shell_system.enabled && gpu.shell_buffers.face_count > 0 {
            gpu.shell_buffers.face = self.shell_system.face;
            gpu.shell_buffers
                .write_view(&gpu.queue, vp.to_cols_array_2d(), self.camera.position.into());
            pp_time!("shell", {
                rmesh_render::record_shell_color_pass(
                    &mut encoder,
                    &gpu.shell_pipeline,
                    &gpu.shell_bg,
                    &gpu.shell_buffers,
                    &gpu.targets.color_view,
                    &gpu.primitive_targets.depth_view,
                );
            });
        }

        // Max-reduce pyramid over the depth buffer the shell just wrote. Built
        // HERE — before the projection dispatch — so the bound is this frame's,
        // not last frame's, and only when the cull will actually sample it:
        // `shell_occluded` returns early on a clear flag, so a pyramid built
        // with the shell off is a full mip chain that nothing reads. (That also
        // means opaque primitives alone never drive the cull today; giving them
        // their own occlusion would mean setting the flag for them too.)
        if self.shell_system.cull_active() {
            pp_time!("shell_hiz", {
                rmesh_render::record_shell_hiz(
                    &mut encoder,
                    &gpu.shell_hiz_pipelines,
                    &gpu.shell_hiz,
                );
            });
        }

        // 2. Sorted forward pass: compute → radix sort → render
        //    Color uses LoadOp::Load (preserves primitive colors), depth test culls behind primitives
        if !skip_volume {
            match self.render_mode {
                RenderMode::IntervalShader => {
                    match self.sort_mode {
                        SortMode::Cpu => {
                            // Single-encoder CPU-sort dispatch. wgpu schedules
                            // pipeline barriers between project_compute (writes
                            // sort_values), the StagingBelt copy (overwrites it),
                            // and interval_gen (reads it) so the encoded order is
                            // the executed order.
                            gpu.cpu_sorter.preserve_subpixel = gpu.light_bake.uses_energy();
                            let visible_count = gpu.cpu_sorter.cull_and_sort_projected(
                                &self.scene_data.vertices,
                                &self.scene_data.indices,
                                &self.scene_data.circumdata,
                                self.camera.position,
                                vp,
                                Some(&self.active_mask_scratch),
                                [gpu.targets.width, gpu.targets.height],
                            );
                            rmesh_render::record_compute_interval_project(
                                &mut encoder,
                                &gpu.pipelines,
                                Some(&gpu.sort_state),
                                &gpu.buffers,
                                &gpu.compute_bg,
                                Some(&gpu.hw_compute_bg),
                                gpu.tet_count,
                                &gpu.queue,
                                Some(&gpu.ts_query_set),
                                false, // use_16bit_sort always false on Cpu path
                            );
                            if visible_count > 0 {
                                let size = (visible_count * 4) as wgpu::BufferAddress;
                                let mut view = gpu.staging_belt.write_buffer(
                                    &mut encoder,
                                    &gpu.buffers.sort_values,
                                    0,
                                    wgpu::BufferSize::new(size).unwrap(),
                                );
                                view.copy_from_slice(bytemuck::cast_slice(
                                    &gpu.cpu_sorter.sorted_indices[..visible_count],
                                ));
                            }
                            {
                                let mut view = gpu.staging_belt.write_buffer(
                                    &mut encoder,
                                    &gpu.buffers.indirect_args,
                                    4,
                                    wgpu::BufferSize::new(4).unwrap(),
                                );
                                view.copy_from_slice(bytemuck::bytes_of(&(visible_count as u32)));
                            }
                            rmesh_render::record_compute_interval_after_sort(
                                &mut encoder,
                                &gpu.compute_interval_pipelines,
                                &gpu.buffers,
                                &gpu.targets,
                                &gpu.compute_interval_gen_bg_a,
                                &gpu.compute_interval_render_bg,
                                &gpu.compute_interval_convert_bg,
                                &gpu.primitive_targets.depth_view,
                                Some(&gpu.ts_query_set),
                                mrt_enabled,
                                gpu.frag_query_set.as_ref(),
                            );
                        }
                        _ if self.use_mesh_shader
                            && gpu.interval_pipelines.is_some()
                            && gpu.interval_render_bg_a.is_some() =>
                        {
                            // Mesh-shader interval path: project → sort → mesh
                            // dispatch, no interval buffers. 32-bit sort only;
                            // the 16-bit sort's bind groups hold a different
                            // values buffer that this path does not carry.
                            rmesh_render::record_sorted_interval_forward_pass(
                                &mut encoder,
                                &gpu.device,
                                &gpu.pipelines,
                                gpu.interval_pipelines.as_ref().unwrap(),
                                &gpu.sort_pipelines,
                                &gpu.sort_state,
                                &gpu.buffers,
                                &gpu.targets,
                                &gpu.compute_bg,
                                gpu.interval_render_bg_a.as_ref().unwrap(),
                                gpu.interval_render_bg_b.as_ref().unwrap(),
                                gpu.interval_indirect_convert_bg.as_ref().unwrap(),
                                gpu.tet_count,
                                &gpu.queue,
                                &gpu.primitive_targets.depth_view,
                                Some(&gpu.hw_compute_bg),
                                Some(&gpu.ts_query_set),
                                mrt_enabled,
                                gpu.frag_query_set.as_ref(),
                            );
                        }
                        _ => {
                            let use_16bit = matches!(self.sort_mode, SortMode::Gpu16);
                            let (sort_st, gen_a, gen_b) = if use_16bit {
                                (
                                    &gpu.sort_state_16bit,
                                    &gpu.compute_interval_gen_bg_a_16bit,
                                    &gpu.compute_interval_gen_bg_b_16bit,
                                )
                            } else {
                                (
                                    &gpu.sort_state,
                                    &gpu.compute_interval_gen_bg_a,
                                    &gpu.compute_interval_gen_bg_b,
                                )
                            };
                            rmesh_render::record_sorted_compute_interval_forward_pass(
                                &mut encoder,
                                &gpu.device,
                                &gpu.pipelines,
                                &gpu.compute_interval_pipelines,
                                &gpu.sort_pipelines,
                                sort_st,
                                &gpu.buffers,
                                &gpu.targets,
                                &gpu.compute_bg,
                                gen_a,
                                gen_b,
                                &gpu.compute_interval_render_bg,
                                &gpu.compute_interval_convert_bg,
                                gpu.tet_count,
                                &gpu.queue,
                                &gpu.primitive_targets.depth_view,
                                Some(&gpu.hw_compute_bg),
                                Some(&gpu.ts_query_set),
                                use_16bit,
                                mrt_enabled,
                                gpu.frag_query_set.as_ref(),
                            );
                        }
                    }
                }
                RenderMode::MeshShader
                    if gpu.mesh_pipelines.is_some() && gpu.mesh_render_bg_a.is_some() =>
                {
                    rmesh_render::record_sorted_mesh_forward_pass(
                        &mut encoder,
                        &gpu.device,
                        &gpu.pipelines,
                        gpu.mesh_pipelines.as_ref().unwrap(),
                        &gpu.sort_pipelines,
                        &gpu.sort_state,
                        &gpu.buffers,
                        &gpu.targets,
                        &gpu.compute_bg,
                        gpu.mesh_render_bg_a.as_ref().unwrap(),
                        gpu.mesh_render_bg_b.as_ref().unwrap(),
                        gpu.indirect_convert_bg.as_ref().unwrap(),
                        gpu.tet_count,
                        &gpu.queue,
                        &gpu.primitive_targets.depth_view,
                        Some(&gpu.hw_compute_bg),
                        Some(&gpu.ts_query_set),
                        mrt_enabled,
                    );
                }
                _ => {
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
                        &gpu.primitive_targets.depth_view,
                        Some(&gpu.hw_compute_bg),
                        matches!(self.render_mode, RenderMode::Quad),
                        Some(&gpu.prepass_bg_a),
                        Some(&gpu.prepass_bg_b),
                        Some(&gpu.quad_render_bg),
                        Some(&gpu.ts_query_set),
                        mrt_enabled,
                    );
                }
            }
        } // end skip_volume

        // Copy instance_count to readback — skip if buffer has a pending/completed map
        if !gpu
            .instance_count_mapped
            .load(std::sync::atomic::Ordering::Acquire)
        {
            encoder.copy_buffer_to_buffer(
                &gpu.buffers.indirect_args,
                4,
                &gpu.instance_count_readback,
                0,
                4,
            );
        }

        // 2c. Overdraw probe. Replays the interval draw into a counting target,
        //     so it measures the fragments that actually survived the shell
        //     z-cull and the depth test this frame — not a geometric estimate.
        if self.overdraw_view.is_some() && !skip_volume {
            // The probe replays the compute path's interval buffers, which the
            // mesh path never fills.
            let mesh_active = self.use_mesh_shader && gpu.interval_pipelines.is_some();
            if self.render_mode == RenderMode::IntervalShader && !mesh_active {
                pp_time!("overdraw_probe", {
                    rmesh_render::record_overdraw_count(
                        &mut encoder,
                        &gpu.compute_interval_pipelines,
                        &gpu.buffers,
                        &gpu.targets,
                        &gpu.compute_interval_render_bg,
                        &gpu.primitive_targets.depth_view,
                        &gpu.overdraw_target,
                    );
                });
            }
        }

        // Present integrated radiance (or the diagnostic output).
        if let Some(view_mode) = self.overdraw_view {
            gpu.queue.write_buffer(
                &gpu.overdraw_target.uniforms,
                0,
                bytemuck::bytes_of(&rmesh_render::VizUniforms {
                    scale: self.overdraw_scale,
                    mode: match view_mode {
                        rmesh_render::OverdrawView::Count => 0,
                        rmesh_render::OverdrawView::Alpha => 1,
                        rmesh_render::OverdrawView::Wasted => 2,
                    },
                    _pad: [0; 2],
                }),
            );
            rmesh_render::record_overdraw_viz(
                &mut encoder,
                &gpu.overdraw_pipelines,
                &gpu.overdraw_target,
                &view,
            );
        } else {
            record_blit(&mut encoder, &gpu.blit_pipeline, &gpu.blit_bg, &view);
        }

        // egui render pass (overlay on swapchain)
        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                })
                .forget_lifetime();
            gpu.egui_renderer
                .render(&mut rpass, &paint_jobs, &screen_desc);
        }

        // Resolve GPU timestamps and copy to readback buffer (skip if target still mapped/pending)
        let cur_rb = gpu.ts_frame % 2;
        let ts_buf_free =
            !gpu.ts_readback_mapped[cur_rb].load(std::sync::atomic::Ordering::Acquire);
        if ts_buf_free {
            // Only resolve slots that are actually written this frame — see
            // TS_WRITTEN_COUNT. This used to be gated on `sh_degree > 0`, but the
            // SH-eval pass that would write slots 6..8 is never recorded by the
            // viewer at any degree, so every SH scene resolved two never-written
            // queries and wedged the driver.
            let written_count: u32 = TS_WRITTEN_COUNT;
            encoder.resolve_query_set(&gpu.ts_query_set, 0..written_count, &gpu.ts_resolve_buf, 0);
            encoder.copy_buffer_to_buffer(
                &gpu.ts_resolve_buf,
                0,
                &gpu.ts_readback[cur_rb],
                0,
                (written_count as u64) * 8,
            );
        }

        // Resolve the interval raster pass's fragment-invocation counter.
        // Recorded on every frame the interval path runs, so it's always safe
        // to resolve when the query set exists.
        let frag_free = gpu.frag_query_set.is_some()
            && !skip_volume
            && !gpu.frag_mapped.load(std::sync::atomic::Ordering::Acquire);
        if frag_free {
            let qs = gpu.frag_query_set.as_ref().unwrap();
            encoder.resolve_query_set(qs, 0..1, &gpu.frag_resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&gpu.frag_resolve_buf, 0, &gpu.frag_readback, 0, 8);
        }

        // Resolve the profiler's written prefix. `pp_labels` is exactly the set
        // of brackets recorded above, so 2*len slots were
        // written and no unwritten query is ever resolved.
        let pp_cur = gpu.ts_frame % 2;
        let pp_pairs = pp_labels.len() as u32;
        let pp_free = pp_pairs > 0
            && !gpu.pp_readback_mapped[pp_cur].load(std::sync::atomic::Ordering::Acquire);
        if pp_free {
            encoder.resolve_query_set(
                &gpu.pp_query_set,
                0..(pp_pairs * 2),
                &gpu.pp_resolve_buf,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &gpu.pp_resolve_buf,
                0,
                &gpu.pp_readback[pp_cur],
                0,
                (pp_pairs as u64) * 2 * 8,
            );
            gpu.pp_labels_pending[pp_cur] = pp_labels;
        }

        let t_before_submit = std::time::Instant::now();
        // Close any in-flight staging chunks (CPU sort writes for this
        // frame) and submit.
        gpu.staging_belt.finish();
        gpu.queue.submit(std::iter::once(encoder.finish()));
        // Recycle staging chunks whose copies completed on previous frames.
        gpu.staging_belt.recall();
        let t_after_submit = std::time::Instant::now();

        // Async map of this frame's timestamp readback (will be ready next frame)
        if ts_buf_free {
            use std::sync::atomic::Ordering;
            gpu.ts_readback_mapped[cur_rb].store(true, Ordering::Release);
            let ready = gpu.ts_readback_ready[cur_rb].clone();
            gpu.ts_readback[cur_rb]
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }
        if pp_free {
            use std::sync::atomic::Ordering;
            gpu.pp_readback_mapped[pp_cur].store(true, Ordering::Release);
            let ready = gpu.pp_readback_ready[pp_cur].clone();
            gpu.pp_readback[pp_cur]
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }

        if frag_free {
            use std::sync::atomic::Ordering;
            gpu.frag_mapped.store(true, Ordering::Release);
            let ready = gpu.frag_ready.clone();
            gpu.frag_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }

        // Always advance frame counter so we alternate slots
        gpu.ts_frame += 1;

        // Async map instance count readback (will be ready next frame)
        // Only map if we actually copied data (i.e., buffer wasn't already mapped)
        if !gpu
            .instance_count_mapped
            .load(std::sync::atomic::Ordering::Acquire)
        {
            use std::sync::atomic::Ordering;
            gpu.instance_count_mapped.store(true, Ordering::Release);
            let ready = gpu.instance_count_ready.clone();
            gpu.instance_count_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        }

        // Free egui textures after submit
        for id in &full_output.textures_delta.free {
            gpu.egui_renderer.free_texture(id);
        }

        crate::oracle::oracle_post_frame(
            &mut self.oracle,
            gpu,
            &mut self.overdraw_view,
            &mut self.active_mask_dirty,
        );

        // Capture before present: the surface texture is still ours to copy.
        if let Some(shot) = &self.screenshot {
            if gpu.ts_frame >= shot.at_frame && gpu.light_bake.ready() && bake_was_ready {
                if let Ok(path) = std::env::var("RMESH_LINEAR_SHOT") {
                    let path = if shot.remaining > 1 || shot.captured > 0 {
                        format!("{path}.{}.npy", shot.captured)
                    } else { path };
                    if let Err(e) = crate::shell::dump_linear_rgb(&gpu.device, &gpu.queue, &gpu.targets.color_texture, &path) {
                        log::error!("[linear] {e}");
                        std::process::exit(1);
                    }
                    let light_data: Vec<_> = self.primitives.iter()
                        .filter(|p| matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight))
                        .map(|p| [p.transform.position.to_array(),
                            (p.transform.rotation * glam::Vec3::NEG_Z).to_array(),
                            p.color.unwrap_or([1.0, 1.0, 1.0, 3.0])[..3].try_into().unwrap()])
                        .collect();
                    let intensities: Vec<_> = self.primitives.iter()
                        .filter(|p| matches!(p.kind, PrimitiveKind::PointLight | PrimitiveKind::SpotLight))
                        .map(|p| p.color.unwrap_or([1.0, 1.0, 1.0, 3.0])[3]).collect();
                    let metadata = format!(
                        "{{\"width\":{w},\"height\":{h},\"intrinsics\":{:?},\"camera_position\":{:?},\"c2w_columns\":{:?},\"near\":{},\"far\":{},\"light_position_direction_rgb\":{:?},\"light_intensities\":{:?},\"shell_enabled\":{},\"importance_enabled\":{},\"bake\":\"{}\"}}\n",
                        uniforms.intrinsics, pos, [uniforms.c2w_col0, uniforms.c2w_col1, uniforms.c2w_col2],
                        self.pbr_data.as_ref().and_then(|p|p.render_min_t).unwrap_or(self.camera.near_z), self.camera.far_z, light_data, intensities,
                        self.shell_system.enabled, self.shell_system.importance_enabled,
                        if gpu.light_bake.uses_energy() {"energy_mode4_interpolated_visibility"} else {"vertex_average_diffuse_unverified"},
                    );
                    if let Err(e) = std::fs::write(format!("{path}.json"), metadata) {
                        log::error!("[linear] metadata: {e}");
                        std::process::exit(1);
                    }
                }
                if let Ok(path) = std::env::var("RMESH_OVERDRAW_DUMP") {
                    if let Err(e) = crate::shell::dump_overdraw_probe(
                        &gpu.device,
                        &gpu.queue,
                        &gpu.overdraw_target.texture,
                        &path,
                    ) {
                        log::error!("[overdraw] {e}");
                    }
                }
                if let Err(e) = shot.capture(&gpu.device, &gpu.queue, &output.texture) {
                    log::error!("[shot] {e}");
                }
                if shot.remaining == 1 { std::process::exit(0); }
                let shot = self.screenshot.as_mut().unwrap();
                shot.remaining -= 1;
                shot.captured += 1;
                shot.at_frame = gpu.ts_frame + 1;
            }
        }

        let t_before_present = std::time::Instant::now();
        output.present();
        let t_after_present = std::time::Instant::now();

        // EMA smoothing (alpha = 0.05 for stable readout)
        let alpha = 0.05f32;
        let mix = |old: f32, new: f32| old * (1.0 - alpha) + new * alpha;
        let prev = &gpu.cpu_times_ms;
        gpu.cpu_times_ms = CpuTimings {
            poll_readback_ms: mix(
                prev.poll_readback_ms,
                (t_after_poll - t_frame_start).as_secs_f32() * 1000.0,
            ),
            acquire_ms: mix(
                prev.acquire_ms,
                (t_after_acquire - t_before_acquire).as_secs_f32() * 1000.0,
            ),
            egui_ms: mix(
                prev.egui_ms,
                (t_after_egui - t_before_egui).as_secs_f32() * 1000.0,
            ),
            encode_ms: mix(
                prev.encode_ms,
                (t_before_submit - t_after_egui).as_secs_f32() * 1000.0,
            ),
            submit_ms: mix(
                prev.submit_ms,
                (t_after_submit - t_before_submit).as_secs_f32() * 1000.0,
            ),
            present_ms: mix(
                prev.present_ms,
                (t_after_present - t_before_present).as_secs_f32() * 1000.0,
            ),
            total_ms: mix(
                prev.total_ms,
                (t_after_present - t_frame_start).as_secs_f32() * 1000.0,
            ),
        };

        // Handle deferred file load
        if let Some(path) = self.pending_load.take() {
            self.load_file(&path);
        }
    }
}

/// One-shot CPU replica of the interval shader's screen-area test, for the
/// current view. Projects every tet exactly as `interval_compute.wgsl` does
/// (clip = vp·world, ndc = clip.xy/w, area = ndc-bbox × screen/2) and logs the
/// distribution of projected pixel areas — so we can see what fraction of tets
/// are sub-pixel and thus candidates for the point-splat fast path.
fn measure_subpixel_fraction(
    vp: &glam::Mat4,
    w: u32,
    h: u32,
    vertices: &[f32],
    indices: &[u32],
    densities: &[f32],
    tet_count: usize,
    vis_mask: Option<&[u8]>,
) {
    let sx = w as f32 * 0.5;
    let sy = h as f32 * 0.5;

    let screen_area = (w as f32) * (h as f32);
    let mut behind = 0usize; // any vertex at/behind camera (w <= 0)
    let mut in_front = 0usize; // fully in front, area measured
    let mut on_screen = 0usize; // in-front AND ndc bbox overlaps the frustum
    let mut capped_fill = 0.0f64; // Σ min(area, screen) over on-screen tets = overdraw proxy
    // Same, but restricted to PVS-visible tets (region mask bit set).
    let mut on_screen_pvs = 0usize;
    let mut capped_fill_pvs = 0.0f64;
    // Opacity of PVS-visible on-screen tets: alpha = 1-exp(-density * mean_edge).
    let mut alpha_sum = 0.0f64;
    // alpha buckets: [<0.01, 0.01-0.05, 0.05-0.2, 0.2-0.5, >=0.5]
    let mut alpha_buckets = [0usize; 5];
    // Pixel-area histogram buckets:
    // [<1, 1-16, 16-64, 64-256, 256-1k, 1k-10k, 10k-screen, >=screen]
    let mut buckets = [0usize; 8];
    let mut huge = 0usize; // area >= full screen (near-plane blowup / very close)
    let mut max_area = 0.0f32;
    let mut area_sum = 0.0f64;

    for t in 0..tet_count {
        let base = t * 4;
        let mut any_behind = false;
        let mut mn = glam::Vec2::splat(f32::INFINITY);
        let mut mx = glam::Vec2::splat(f32::NEG_INFINITY);
        let mut wp = [glam::Vec3::ZERO; 4];
        for k in 0..4 {
            let vi = indices[base + k] as usize;
            let world = glam::Vec3::new(vertices[vi * 3], vertices[vi * 3 + 1], vertices[vi * 3 + 2]);
            wp[k] = world;
            let clip = *vp * world.extend(1.0);
            if clip.w <= 1e-6 {
                any_behind = true;
                break;
            }
            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
            mn = mn.min(ndc);
            mx = mx.max(ndc);
        }
        if any_behind {
            behind += 1;
            continue;
        }
        in_front += 1;

        // Frustum test: does the tet's NDC bbox overlap the [-1,1] screen rect?
        let onscreen = mx.x >= -1.0 && mn.x <= 1.0 && mx.y >= -1.0 && mn.y <= 1.0;
        if !onscreen {
            continue; // in front of the camera but entirely off to the side
        }
        on_screen += 1;

        let ext = (mx - mn) * glam::Vec2::new(sx, sy);
        let area = (ext.x * ext.y).max(0.0);
        area_sum += area as f64;
        let capped = area.min(screen_area) as f64;
        capped_fill += capped;
        // PVS-visible subset (mask absent => treat all as visible).
        if vis_mask.map_or(true, |mk| mk[t] != 0) {
            on_screen_pvs += 1;
            capped_fill_pvs += capped;
            // Approx per-tet opacity: alpha = 1 - exp(-density * mean_edge_len).
            let mean_edge = (wp[0].distance(wp[1])
                + wp[0].distance(wp[2])
                + wp[0].distance(wp[3])
                + wp[1].distance(wp[2])
                + wp[1].distance(wp[3])
                + wp[2].distance(wp[3]))
                / 6.0;
            let alpha = 1.0 - (-densities[t] * mean_edge).exp();
            alpha_sum += alpha as f64;
            let ab = if alpha < 0.01 {
                0
            } else if alpha < 0.05 {
                1
            } else if alpha < 0.2 {
                2
            } else if alpha < 0.5 {
                3
            } else {
                4
            };
            alpha_buckets[ab] += 1;
        }
        max_area = max_area.max(area);
        if area >= screen_area {
            huge += 1;
        }
        let b = if area < 1.0 {
            0
        } else if area < 16.0 {
            1
        } else if area < 64.0 {
            2
        } else if area < 256.0 {
            3
        } else if area < 1024.0 {
            4
        } else if area < 10240.0 {
            5
        } else if area < screen_area {
            6
        } else {
            7
        };
        buckets[b] += 1;
    }

    let sub_px = buckets[0]; // area < 1px
    let pct = |n: usize| {
        if on_screen == 0 {
            0.0
        } else {
            100.0 * n as f32 / on_screen as f32
        }
    };
    // Average tets covering each pixel = Σ min(area, screen) / screen. This is
    // the depth-complexity / overdraw factor — the thing that makes fill-bound
    // frames slow.
    let overdraw = capped_fill / screen_area as f64;
    let overdraw_pvs = capped_fill_pvs / screen_area as f64;
    log::info!(
        "[overdraw probe] {}x{} (screen={:.0}px) | tets={} behind_cam={} in_front={} on_screen={} | \
         OVERDRAW={:.0}x tets/pixel | sub-pixel(<1px)={} ({:.1}%) | full-screen+={} ({:.2}%) | max_area={:.0}px",
        w,
        h,
        screen_area,
        tet_count,
        behind,
        in_front,
        on_screen,
        overdraw,
        sub_px,
        pct(sub_px),
        huge,
        pct(huge),
        max_area,
    );
    log::info!(
        "[overdraw probe] PVS-visible on-screen tets={} (of {} on-screen) | OVERDRAW(PVS-only)={:.0}x tets/pixel \
         => applying the region mask would cut overdraw {:.1}x",
        on_screen_pvs,
        on_screen,
        overdraw_pvs,
        if overdraw_pvs > 0.0 { overdraw / overdraw_pvs } else { 0.0 },
    );
    // Opacity: does a pixel saturate early (few opaque tets) or need all of them?
    let mean_alpha = if on_screen_pvs > 0 {
        alpha_sum / on_screen_pvs as f64
    } else {
        0.0
    };
    // Tets to reach T<0.01 assuming mean alpha: N = ln(0.01)/ln(1-mean_alpha).
    let sat_depth = if mean_alpha > 1e-4 && mean_alpha < 1.0 {
        (0.01f64.ln() / (1.0 - mean_alpha).ln()).ceil()
    } else {
        f64::INFINITY
    };
    let apct = |n: usize| {
        if on_screen_pvs == 0 {
            0.0
        } else {
            100.0 * n as f32 / on_screen_pvs as f32
        }
    };
    log::info!(
        "[overdraw probe] per-tet opacity (PVS-visible): mean_alpha={:.3} => pixel saturates after ~{:.0} tets \
         (vs {} stacked). alpha buckets: <0.01={:.0}%  0.01-0.05={:.0}%  0.05-0.2={:.0}%  0.2-0.5={:.0}%  >=0.5={:.0}%",
        mean_alpha,
        sat_depth,
        on_screen_pvs,
        apct(alpha_buckets[0]),
        apct(alpha_buckets[1]),
        apct(alpha_buckets[2]),
        apct(alpha_buckets[3]),
        apct(alpha_buckets[4]),
    );
    log::info!(
        "[overdraw probe] area histogram: <1px={} ({:.1}%)  1-16={} ({:.1}%)  16-64={} ({:.1}%)  \
         64-256={} ({:.1}%)  256-1k={} ({:.1}%)  1k-10k={} ({:.1}%)  10k-screen={} ({:.1}%)  >=screen={} ({:.2}%)",
        buckets[0], pct(buckets[0]),
        buckets[1], pct(buckets[1]),
        buckets[2], pct(buckets[2]),
        buckets[3], pct(buckets[3]),
        buckets[4], pct(buckets[4]),
        buckets[5], pct(buckets[5]),
        buckets[6], pct(buckets[6]),
        buckets[7], pct(buckets[7]),
    );
}
