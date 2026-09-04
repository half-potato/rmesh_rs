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

use crate::flare::{extract_boundary_faces_adj, tet_alphas};

/// What a background rebuild sends back: the new face list, plus the adjacency
/// if this rebuild is the one that had to build it.
type RebuildResult = (Vec<u32>, Option<Arc<SceneCache>>);

/// Per-scene data a rebuild needs that does not depend on the settings: the tet
/// adjacency and the per-tet opacity. Both cost seconds to derive and neither
/// changes when the threshold or erosion moves, so the first rebuild for a scene
/// hands them back and later ones reuse them.
pub struct SceneCache {
    /// `neighbors[tet * 4 + face]` = the tet across that face, or -1.
    pub neighbors: Vec<i32>,
    /// Alpha a single tet contributes to a ray crossing it, in [0, 1].
    pub alphas: Vec<f32>,
}

pub struct ShellSystem {
    /// Draw the shell at all (backstop color + depth bound).
    pub enabled: bool,
    /// Let `project_compute*` drop tets behind the shell before the sort. Split
    /// from `enabled` so the backstop can be judged on its own, and so the cull
    /// can be toggled for an A/B without disturbing the image behind it.
    pub cull_enabled: bool,
    /// Opacity floor for "solid", in [0, 1]: the alpha a single tet must
    /// contribute to a crossing ray. This is `1 - exp(-density * dt)`, not raw
    /// density — a large sparse tet blocks more light than a small dense one, so
    /// thresholding density alone both admits transparent tets and rejects
    /// opaque ones. Being a real opacity, it also means the same thing in every
    /// scene, which raw density did not.
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
    /// Built once per scene by the first rebuild and reused by every later one;
    /// cleared in [`ShellSystem::reset`] because it indexes the old scene's tets.
    cache: Option<Arc<SceneCache>>,
}

impl Default for ShellSystem {
    fn default() -> Self {
        Self {
            // On by default. The settings below are the measured operating
            // point rather than a guess: they stay within the project's
            // 35dB / 0.99 SSIM bar on every scene tested, so the speedup is not
            // paid for in image quality. Extraction runs off-thread, so the
            // first ~2s of a scene simply renders without a shell.
            enabled: true,
            cull_enabled: true,
            density_threshold: 0.1,
            cull_bias: 0.0,
            // Two layers. Erosion removes structures too thin to be opaque,
            // which would otherwise get an opaque shell the volume renders as
            // nearly transparent -- at erosion 0 those facets are blatant and
            // cost 34 dB (22.2 dB at the default threshold, against 56.2 at 2).
            //
            // Two is also near the ceiling here: thresholding on real opacity
            // leaves a set that is only a few tets thick almost everywhere, so
            // erosion 4 deletes it outright (3828 faces left, from 1.7M). That
            // thinness is the honest shape of the data -- mean per-fragment
            // alpha is 0.114, so genuinely opaque regions are rare.
            //
            // (0.1, 2) clears the project's 35dB / 0.99 SSIM bar on all three
            // scenes measured, with the gain varying by how much solid each has:
            // stump 46->51 fps at 56.2dB, room 32->48 at 47.0dB, bicycle
            // 22->26 at 51.9dB. Costs one linear scan over the tets per step.
            erosion_steps: 2,
            face: rmesh_render::ShellFace::Exit,
            show_only: false,
            faces: Vec::new(),
            built_key: (f32::NAN, 0),
            dirty: false,
            pending: None,
            pending_key: (f32::NAN, 0),
            cache: None,
        }
    }
}

impl ShellSystem {
    /// Environment overrides, so the shell can be driven from a headless
    /// profiling run instead of the egui sliders:
    ///
    /// * `RMESH_SHELL=1`             — enable
    /// * `RMESH_SHELL_THRESHOLD=0.1` — per-tet opacity floor, in [0, 1]
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
        self.cache = None;
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

        // Only cloned when the cache still has to be built; afterwards the
        // thread needs nothing but the cache itself.
        let cached = self.cache.clone();
        let (idxs, verts, densities) = if cached.is_none() {
            (
                scene_data.indices.clone(),
                scene_data.vertices.clone(),
                scene_data.densities.clone(),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let tet_count = scene_data.tet_count as usize;
        let threshold = self.density_threshold;
        let erosion = self.erosion_steps;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            // Build the per-scene cache on the first rebuild and hand it back
            // with the faces; later rebuilds get it for free.
            let (cache, fresh) = match cached {
                Some(c) => (c, None),
                None => {
                    let t = std::time::Instant::now();
                    let c = Arc::new(SceneCache {
                        neighbors: rmesh_raytrace::compute_tet_neighbors(&idxs, tet_count),
                        alphas: tet_alphas(&verts, &idxs, &densities, tet_count),
                    });
                    log::info!(
                        "[shell] built adjacency + per-tet opacity for {} tets in {:.2}s",
                        tet_count,
                        t.elapsed().as_secs_f64()
                    );
                    (c.clone(), Some(c))
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
                    let f =
                        extract_boundary_faces_adj(&cache.neighbors, &cache.alphas, tet_count, th, er);
                    log::info!(
                        "[sweep] threshold {th} erosion {er}: {} faces in {:.3}s",
                        f.len(),
                        t.elapsed().as_secs_f64()
                    );
                }
            }

            let t0 = std::time::Instant::now();
            let faces =
                extract_boundary_faces_adj(&cache.neighbors, &cache.alphas, tet_count, threshold, erosion);
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
        if let Some(c) = fresh_adj {
            self.cache = Some(c);
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

/// Dump the raw overdraw probe (Rg16Float: fragment count, summed alpha) so the
/// distribution behind the heatmap can be quantified rather than read off a
/// colour ramp. `RMESH_OVERDRAW_DUMP=<path>` writes it alongside the screenshot.
pub fn dump_overdraw_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    path: &str,
) -> anyhow::Result<()> {
    let (w, h) = (texture.width(), texture.height());
    let unpadded = w * 4; // Rg16Float = 4 bytes
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("overdraw_dump"),
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
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let start = (row * padded) as usize;
        out.extend_from_slice(&view[start..start + unpadded as usize]);
    }
    drop(view);
    buffer.unmap();

    std::fs::write(path, &out)?;
    log::info!("[overdraw] dumped {path} ({w}x{h}, Rg16Float raw)");
    Ok(())
}
