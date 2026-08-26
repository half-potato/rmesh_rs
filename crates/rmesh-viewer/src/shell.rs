//! CPU-side driver for the density shell.
//!
//! Owns the threshold, the extracted face list, and the background rebuild that
//! keeps the UI responsive while the slider moves. Extraction is a hash over
//! every face of every dense tet — seconds on a multi-million-tet scene — so it
//! runs off-thread and the previous shell stays live until the new one lands.
//!
//! Face ids are packed `(tet_id << 2) | face_idx` and index the tet array, not
//! vertex positions, so an extracted shell survives animation and PBD edits. It
//! only needs rebuilding when the threshold or the scene itself changes.

use std::sync::mpsc;

use crate::flare::extract_boundary_faces;

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
    /// Debug: draw the shell but skip the volume.
    pub show_only: bool,

    faces: Vec<u32>,
    /// Threshold the live `faces` were built at; `f32::NAN` until first build.
    built_threshold: f32,
    /// Set when the threshold moved and a rebuild is owed.
    dirty: bool,
    /// In-flight rebuild. At most one; starting a new one drops the old
    /// receiver and its result is discarded on arrival.
    pending: Option<mpsc::Receiver<Vec<u32>>>,
    pending_threshold: f32,
}

impl Default for ShellSystem {
    fn default() -> Self {
        Self {
            // Off by default: the cull trades correctness for speed wherever a
            // dense region is too thin to actually be opaque, and the right
            // threshold is scene-specific. Opt in, then tune.
            enabled: false,
            cull_enabled: true,
            density_threshold: 1.0,
            cull_bias: 0.0,
            show_only: false,
            faces: Vec::new(),
            built_threshold: f32::NAN,
            dirty: false,
            pending: None,
            pending_threshold: 0.0,
        }
    }
}

impl ShellSystem {
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn has_shell(&self) -> bool {
        !self.faces.is_empty()
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
        self.built_threshold = f32::NAN;
        self.pending = None;
        self.dirty = self.enabled;
    }

    /// Kick off a background extraction if one is owed and none is running.
    pub fn maybe_start_rebuild(&mut self, scene_data: &rmesh_data::SceneData) {
        if !self.enabled || scene_data.tet_count == 0 {
            return;
        }
        // `built_threshold` starts NaN, so this is also what triggers the very
        // first build once the feature is switched on.
        let stale = self.dirty || self.built_threshold != self.density_threshold;
        if !stale || self.pending.is_some() {
            return;
        }

        let idxs = scene_data.indices.clone();
        let densities = scene_data.densities.clone();
        let tet_count = scene_data.tet_count as usize;
        let threshold = self.density_threshold;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let faces = extract_boundary_faces(&idxs, &densities, tet_count, threshold);
            log::info!(
                "[shell] extracted {} boundary faces at threshold {:.4} in {:.2}s",
                faces.len(),
                threshold,
                t0.elapsed().as_secs_f64()
            );
            // Receiver may be gone if a newer rebuild superseded us.
            let _ = tx.send(faces);
        });
        self.pending = Some(rx);
        self.pending_threshold = threshold;
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
        let Ok(faces) = rx.try_recv() else {
            return false;
        };
        self.pending = None;

        // A rebuild finishing for a threshold the user has already moved past is
        // stale; drop it and let `maybe_start_rebuild` issue the current one.
        if self.pending_threshold != self.density_threshold {
            self.dirty = true;
            return false;
        }

        self.built_threshold = self.pending_threshold;
        let realloc = buffers.upload(device, queue, &faces);
        self.faces = faces;
        realloc
    }
}
