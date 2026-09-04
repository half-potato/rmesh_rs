//! CPU-side driver for the density shell.
//!
//! Owns the threshold, the extracted face list, and the background rebuild that
//! keeps the UI responsive while the slider moves.
//!
//! Extraction runs off a cached tet adjacency, which depends only on the
//! tetrahedralization and so survives every threshold and erosion change.
//! Building it is the expensive part and happens once per scene; after that a
//! rebuild is two linear scans over the tet array. Both still run off-thread,
//! with the previous shell live until the new one lands.
//!
//! Face ids are packed `(tet_id << 2) | face_idx` and index the tet array, not
//! vertex positions, so an extracted shell survives animation and PBD edits. It
//! only needs rebuilding when the threshold or the scene itself changes.

use std::sync::{mpsc, Arc};

use crate::flare::extract_boundary_faces_adj;

/// What a background rebuild sends back: the new face list, plus the adjacency
/// if this rebuild is the one that had to build it.
type RebuildResult = (Vec<u32>, Option<Arc<Vec<i32>>>);

pub struct ShellSystem {
    /// Draw the shell at all (backstop color + depth bound).
    pub enabled: bool,
    /// Let `project_compute*` drop tets behind the shell before the sort. Split
    /// from `enabled` so the backstop can be judged on its own, and so the cull
    /// can be toggled for an A/B without disturbing the image behind it.
    pub cull_enabled: bool,
    /// Per-tet density floor for "solid". Densities are `exp((u8 - 100) / 20)`,
    /// so this spans several orders of magnitude and wants a log slider.
    pub density_threshold: f32,
    /// NDC slack on the cull test. Larger culls less.
    pub cull_bias: f32,
    /// Morphological erosion applied to the dense set before extraction, in
    /// tets. Structures thinner than this vanish instead of getting an opaque
    /// shell they are far too transparent to deserve. Costs a rebuild.
    pub erosion_steps: u32,
    /// Which side of the dense region the shell surfaces on. `Exit` is always a
    /// valid bound but occludes little; `Entry` occludes the region's interior
    /// and is valid to the extent erosion has made the region genuinely opaque.
    pub face: rmesh_render::ShellFace,
    /// Debug: draw the shell but skip the volume.
    pub show_only: bool,

    faces: Vec<u32>,
    /// (threshold, erosion) the live `faces` were built at; the NaN threshold
    /// never compares equal, so it also triggers the very first build.
    built_key: (f32, u32),
    /// Set when the threshold moved and a rebuild is owed.
    dirty: bool,
    /// In-flight rebuild. At most one; starting a new one drops the old
    /// receiver and its result is discarded on arrival.
    pending: Option<mpsc::Receiver<RebuildResult>>,
    pending_key: (f32, u32),
    /// `neighbors[tet * 4 + face]` = the tet across that face, or -1. Built once
    /// per scene by the first rebuild and reused by every later one; cleared in
    /// [`ShellSystem::reset`] because it indexes the old scene's tets.
    adjacency: Option<Arc<Vec<i32>>>,
}

impl Default for ShellSystem {
    fn default() -> Self {
        Self {
            // On by default. The settings below are the measured operating
            // point rather than a guess: exit faces plus erosion 8 costs a few
            // frames against no shell at all but stays within the project's
            // 35dB / 0.99 SSIM bar, so the speedup is not paid for in image
            // quality. Extraction runs off-thread, so the first ~2s of a scene
            // simply renders without a shell.
            enabled: true,
            cull_enabled: true,
            density_threshold: 1.0,
            cull_bias: 0.0,
            // Twelve layers. One is nowhere near enough: the error the shell
            // introduces is dominated by opaque facets appearing on structures
            // far too transparent to deserve them, and it takes this many peels
            // to clear them. On `stump_conserv` at the default threshold,
            // erosion 0/1/2/4/8 measures 25.4/32.1/35.3/40.7/48.8 dB PSNR at
            // 0.863/0.954/0.971/0.988/0.997 SSIM for 78/60/56/55/54 fps — frame
            // time flattens after two peels while quality keeps climbing.
            //
            // 8 clears the project's 35dB / 0.99 SSIM bar on `stump_conserv` but
            // was overfit to it: `room_conserv` scores 32.8dB / 0.955 there. 12
            // clears the bar on all three scenes measured — stump 54.4dB/0.9992,
            // bicycle 51.6dB/0.9987, room 43.5dB/0.9946 — and costs stump only
            // 1 fps against 8, so it is the honest default. Erosion past this
            // starts deleting the shell outright (room keeps 0.24% of its tets
            // at 16, bicycle 0.92% at 12), which is why the gain is scene
            // dependent: +18% on stump, +9% on room, +5% on bicycle.
            //
            // Costs one linear scan over the tets per step.
            erosion_steps: 12,
            face: rmesh_render::ShellFace::Exit,
            show_only: false,
            faces: Vec::new(),
            built_key: (f32::NAN, 0),
            dirty: false,
            pending: None,
            pending_key: (f32::NAN, 0),
            adjacency: None,
        }
    }
}

impl ShellSystem {
    /// Environment overrides, so the shell can be driven from a headless
    /// profiling run instead of the egui sliders:
    ///
    /// * `RMESH_SHELL=1`             — enable
    /// * `RMESH_SHELL_THRESHOLD=1.0` — density floor
    /// * `RMESH_SHELL_EROSION=1`     — erosion layers
    /// * `RMESH_SHELL_CULL=0|1`      — z-cull before the sort
    /// * `RMESH_SHELL_BIAS=0.0`      — NDC slack on the cull test
    /// * `RMESH_SHELL_FACE=exit|entry` — which surface the shell is
    /// * `RMESH_SHELL_ONLY=1`        — draw the shell, skip the volume
    pub fn from_env() -> Self {
        let mut s = Self::default();
        let f = |n: &str| std::env::var(n).ok().and_then(|v| v.parse::<f32>().ok());
        if let Some(v) = std::env::var("RMESH_SHELL").ok() {
            s.enabled = v != "0" && v != "false";
        }
        if let Some(v) = f("RMESH_SHELL_THRESHOLD") {
            s.density_threshold = v;
        }
        if let Some(v) = std::env::var("RMESH_SHELL_EROSION").ok().and_then(|v| v.parse().ok()) {
            s.erosion_steps = v;
        }
        if let Some(v) = std::env::var("RMESH_SHELL_CULL").ok() {
            s.cull_enabled = v != "0" && v != "false";
        }
        if let Some(v) = f("RMESH_SHELL_BIAS") {
            s.cull_bias = v;
        }
        if let Some(v) = std::env::var("RMESH_SHELL_FACE").ok() {
            s.face = match v.as_str() {
                "entry" | "front" => rmesh_render::ShellFace::Entry,
                _ => rmesh_render::ShellFace::Exit,
            };
        }
        if let Some(v) = std::env::var("RMESH_SHELL_ONLY").ok() {
            s.show_only = v != "0" && v != "false";
        }
        s
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn has_shell(&self) -> bool {
        !self.faces.is_empty()
    }

    /// Whether `project_compute*` will read the Hi-Z pyramid this frame. The
    /// same condition gates building it: `shell_occluded` returns early when the
    /// flag is clear, so a pyramid built with the shell off is never sampled.
    pub fn cull_active(&self) -> bool {
        self.enabled && self.cull_enabled && self.has_shell()
    }

    pub fn rebuild_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Call when the threshold slider is released, or the scene changes.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Forget the current shell — used when a new scene invalidates the tet ids.
    pub fn reset(&mut self) {
        self.faces.clear();
        self.built_key = (f32::NAN, 0);
        self.pending = None;
        self.adjacency = None;
        self.dirty = self.enabled;
    }

    /// Kick off a background extraction if one is owed and none is running.
    pub fn maybe_start_rebuild(&mut self, scene_data: &rmesh_data::SceneData) {
        if !self.enabled || scene_data.tet_count == 0 {
            return;
        }
        // `built_key.0` starts NaN, so this is also what triggers the very
        // first build once the feature is switched on.
        let key = (self.density_threshold, self.erosion_steps);
        let stale = self.dirty || self.built_key != key;
        if !stale || self.pending.is_some() {
            return;
        }

        // Only cloned when the adjacency still has to be built; afterwards the
        // thread needs nothing but the densities.
        let idxs = if self.adjacency.is_none() {
            scene_data.indices.clone()
        } else {
            Vec::new()
        };
        let cached_adj = self.adjacency.clone();
        let densities = scene_data.densities.clone();
        let tet_count = scene_data.tet_count as usize;
        let threshold = self.density_threshold;
        let erosion = self.erosion_steps;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            // Build the adjacency on the first rebuild for a scene and hand it
            // back with the faces; later rebuilds get it for free.
            let (adj, fresh) = match cached_adj {
                Some(a) => (a, None),
                None => {
                    let t = std::time::Instant::now();
                    let a = Arc::new(rmesh_raytrace::compute_tet_neighbors(&idxs, tet_count));
                    log::info!(
                        "[shell] built tet adjacency for {} tets in {:.2}s",
                        tet_count,
                        t.elapsed().as_secs_f64()
                    );
                    (a.clone(), Some(a))
                }
            };

            // RMESH_SHELL_SWEEP="thr:ero,thr:ero,..." reports what each setting
            // selects without needing a run per point. Diagnostic only.
            if let Ok(spec) = std::env::var("RMESH_SHELL_SWEEP") {
                for pt in spec.split(',') {
                    let mut it = pt.split(':');
                    let (Some(th), Some(er)) = (it.next(), it.next()) else {
                        continue;
                    };
                    let (Ok(th), Ok(er)) = (th.parse::<f32>(), er.parse::<u32>()) else {
                        continue;
                    };
                    let t = std::time::Instant::now();
                    let f = extract_boundary_faces_adj(&adj, &densities, tet_count, th, er);
                    log::info!(
                        "[sweep] threshold {th} erosion {er}: {} faces in {:.3}s",
                        f.len(),
                        t.elapsed().as_secs_f64()
                    );
                }
            }

            let t0 = std::time::Instant::now();
            let faces = extract_boundary_faces_adj(&adj, &densities, tet_count, threshold, erosion);
            log::info!(
                "[shell] extracted {} boundary faces at threshold {:.4}, erosion {} in {:.3}s",
                faces.len(),
                threshold,
                erosion,
                t0.elapsed().as_secs_f64()
            );
            // Receiver may be gone if a newer rebuild superseded us.
            let _ = tx.send((faces, fresh));
        });
        self.pending = Some(rx);
        self.pending_key = key;
        self.dirty = false;
    }

    /// Install a finished extraction, if one arrived.
    ///
    /// Returns `true` when the face buffer was reallocated, meaning the shell
    /// bind group must be rebuilt.
    pub fn poll(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &mut rmesh_render::ShellBuffers,
    ) -> bool {
        let Some(rx) = &self.pending else {
            return false;
        };
        let Ok((faces, fresh_adj)) = rx.try_recv() else {
            return false;
        };
        self.pending = None;
        // Keep the adjacency even when the faces themselves are stale — it cost
        // seconds to build and does not depend on the settings that went stale.
        if let Some(adj) = fresh_adj {
            self.adjacency = Some(adj);
        }

        // A rebuild finishing for settings the user has already moved past is
        // stale; drop it and let `maybe_start_rebuild` issue the current one.
        if self.pending_key != (self.density_threshold, self.erosion_steps) {
            self.dirty = true;
            return false;
        }

        self.built_key = self.pending_key;
        let realloc = buffers.upload(device, queue, &faces);
        self.faces = faces;
        realloc
    }
}

/// Headless frame capture, so a change to the shell can be judged on the image
/// and not only on the frame time.
///
/// `RMESH_SHOT=<path>` writes the presented frame once `RMESH_SHOT_FRAME` (default
/// 240) frames have gone by — late enough for the background extraction to have
/// landed and the temporal passes to have converged — then exits.
pub struct Screenshot {
    pub path: String,
    pub at_frame: usize,
}

impl Screenshot {
    pub fn from_env() -> Option<Self> {
        Some(Self {
            path: std::env::var("RMESH_SHOT").ok()?,
            at_frame: std::env::var("RMESH_SHOT_FRAME")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(240),
        })
    }

    /// Copy the presented surface texture into a mapped buffer and write a PNG.
    /// Blocks on the device — this ends the process, so the stall is free.
    pub fn capture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> anyhow::Result<()> {
        let (w, h) = (texture.width(), texture.height());
        // copy_texture_to_buffer requires rows padded to 256 bytes.
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));

        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely())?;

        let view = buffer.slice(..).get_mapped_range();
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h {
            let start = (row * padded) as usize;
            rgba.extend_from_slice(&view[start..start + unpadded as usize]);
        }
        drop(view);
        buffer.unmap();

        // Surface formats here are Bgra8*; PNG wants RGBA.
        if matches!(
            texture.format(),
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for px in rgba.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        for px in rgba.chunks_exact_mut(4) {
            px[3] = 255;
        }

        image::RgbaImage::from_raw(w, h, rgba)
            .ok_or_else(|| anyhow::anyhow!("screenshot buffer size mismatch"))?
            .save(&self.path)?;
        log::info!("[shot] wrote {} ({}x{})", self.path, w, h);
        Ok(())
    }
}
