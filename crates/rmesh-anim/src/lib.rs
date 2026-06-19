//! Animation system: clock, keyframe evaluation, scene hierarchy, playback.

pub mod gltf_loader;

use glam::{Quat, Vec3};
use rmesh_interact::{Primitive, PrimitiveKind, Transform};

pub use gltf_loader::{GltfScene, LoadedMaterial, LoadedMesh, LoadedTexture};

// ---------------------------------------------------------------------------
// Animation clock
// ---------------------------------------------------------------------------

/// Tracks wall-clock delta time between frames.
pub struct AnimationClock {
    last: Option<std::time::Instant>,
}

impl AnimationClock {
    pub fn new() -> Self {
        Self { last: None }
    }

    /// Call once per frame. Returns delta time in seconds.
    pub fn tick(&mut self) -> f32 {
        let now = std::time::Instant::now();
        let dt = match self.last {
            Some(prev) => now.duration_since(prev).as_secs_f32(),
            None => 0.0,
        };
        self.last = Some(now);
        dt
    }
}

impl Default for AnimationClock {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Keyframe interpolation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetProperty {
    Translation,
    Rotation,
    Scale,
}

impl TargetProperty {
    /// Number of floats per keyframe value (3 for T/S, 4 for R).
    /// For CubicSpline, each keyframe stores 3x this (in-tangent, value, out-tangent).
    pub fn components(self) -> usize {
        match self {
            Self::Translation | Self::Scale => 3,
            Self::Rotation => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Animation channel & clip
// ---------------------------------------------------------------------------

/// A single animated property track (e.g. "node 3, rotation").
#[derive(Debug, Clone)]
pub struct AnimationChannel {
    pub target_node: usize,
    pub property: TargetProperty,
    pub interpolation: Interpolation,
    /// Keyframe timestamps in seconds, sorted ascending.
    pub times: Vec<f32>,
    /// Packed keyframe values. For Step/Linear: `components` floats per keyframe.
    /// For CubicSpline: `3 * components` floats per keyframe (in-tangent, value, out-tangent).
    pub values: Vec<f32>,
}

impl AnimationChannel {
    /// Evaluate this channel at time `t` and apply to the target node's local transform.
    pub fn evaluate(&self, t: f32, nodes: &mut [SceneNode]) {
        if self.times.is_empty() {
            return;
        }
        let node = &mut nodes[self.target_node];
        let c = self.property.components();

        // Find surrounding keyframes via binary search
        let n = self.times.len();
        if t <= self.times[0] {
            self.apply_value(node, &self.values[..c]);
            return;
        }
        if t >= self.times[n - 1] {
            let start = match self.interpolation {
                Interpolation::CubicSpline => (n - 1) * 3 * c + c, // skip in-tangent
                _ => (n - 1) * c,
            };
            self.apply_value(node, &self.values[start..start + c]);
            return;
        }

        // Binary search for the interval [times[i], times[i+1]] containing t
        let i = match self.times.binary_search_by(|k| k.partial_cmp(&t).unwrap()) {
            Ok(exact) => exact.min(n - 2),
            Err(insert) => (insert - 1).min(n - 2),
        };

        let t0 = self.times[i];
        let t1 = self.times[i + 1];
        let frac = if (t1 - t0).abs() < 1e-10 {
            0.0
        } else {
            ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        };

        match self.interpolation {
            Interpolation::Step => {
                let start = i * c;
                self.apply_value(node, &self.values[start..start + c]);
            }
            Interpolation::Linear => {
                let a_start = i * c;
                let b_start = (i + 1) * c;
                let a = &self.values[a_start..a_start + c];
                let b = &self.values[b_start..b_start + c];
                match self.property {
                    TargetProperty::Translation | TargetProperty::Scale => {
                        let va = Vec3::from_slice(a);
                        let vb = Vec3::from_slice(b);
                        let v = va.lerp(vb, frac);
                        self.apply_vec3(node, v);
                    }
                    TargetProperty::Rotation => {
                        let qa = Quat::from_slice(a).normalize();
                        let qb = Quat::from_slice(b).normalize();
                        let q = qa.slerp(qb, frac);
                        node.local_transform.rotation = q;
                    }
                }
            }
            Interpolation::CubicSpline => {
                // glTF cubic spline: each keyframe stores [in_tangent, value, out_tangent]
                let stride = 3 * c;
                let dt = t1 - t0;
                let a_val = &self.values[i * stride + c..i * stride + 2 * c];
                let a_out = &self.values[i * stride + 2 * c..i * stride + 3 * c];
                let b_in = &self.values[(i + 1) * stride..(i + 1) * stride + c];
                let b_val = &self.values[(i + 1) * stride + c..(i + 1) * stride + 2 * c];

                let mut result = vec![0.0f32; c];
                let t2 = frac * frac;
                let t3 = t2 * frac;
                for j in 0..c {
                    let p0 = a_val[j];
                    let m0 = a_out[j] * dt;
                    let p1 = b_val[j];
                    let m1 = b_in[j] * dt;
                    // Hermite basis
                    result[j] = (2.0 * t3 - 3.0 * t2 + 1.0) * p0
                        + (t3 - 2.0 * t2 + frac) * m0
                        + (-2.0 * t3 + 3.0 * t2) * p1
                        + (t3 - t2) * m1;
                }
                if self.property == TargetProperty::Rotation {
                    let q = Quat::from_xyzw(result[0], result[1], result[2], result[3]).normalize();
                    node.local_transform.rotation = q;
                } else {
                    self.apply_vec3(node, Vec3::from_slice(&result));
                }
            }
        }
    }

    fn apply_value(&self, node: &mut SceneNode, v: &[f32]) {
        match self.property {
            TargetProperty::Translation => {
                node.local_transform.position = Vec3::from_slice(v);
            }
            TargetProperty::Rotation => {
                node.local_transform.rotation = Quat::from_xyzw(v[0], v[1], v[2], v[3]).normalize();
            }
            TargetProperty::Scale => {
                node.local_transform.scale = Vec3::from_slice(v);
            }
        }
    }

    fn apply_vec3(&self, node: &mut SceneNode, v: Vec3) {
        match self.property {
            TargetProperty::Translation => node.local_transform.position = v,
            TargetProperty::Scale => node.local_transform.scale = v,
            TargetProperty::Rotation => unreachable!(),
        }
    }
}

/// A named collection of animation channels with a fixed duration.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
}

// ---------------------------------------------------------------------------
// Scene hierarchy
// ---------------------------------------------------------------------------

/// A node in the scene hierarchy. May have a mesh, a parent, children, and material properties.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub local_transform: Transform,
    /// Computed each frame from hierarchy. Do not set manually.
    pub world_transform: Transform,
    /// Which primitive mesh to draw (None = group/bone node, no geometry).
    pub mesh_kind: Option<PrimitiveKind>,
    /// RGBA base color factor. Defaults to white.
    pub color: [f32; 4],
    pub visible: bool,
    /// PBR roughness factor (0.0 = mirror, 1.0 = fully rough).
    pub roughness: f32,
    /// PBR metallic factor (0.0 = dielectric, 1.0 = metal).
    pub metallic: f32,
    /// Ambient occlusion strength (0.0 = no effect, 1.0 = full).
    pub occlusion_strength: f32,
    /// Normal map scale factor.
    pub normal_scale: f32,
    /// Index into the MaterialRegistry for texture lookups.
    pub material_index: Option<usize>,
}

impl SceneNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            children: Vec::new(),
            local_transform: Transform::default(),
            world_transform: Transform::default(),
            mesh_kind: None,
            color: [1.0, 1.0, 1.0, 1.0],
            visible: true,
            roughness: 1.0,
            metallic: 0.0,
            occlusion_strength: 1.0,
            normal_scale: 1.0,
            material_index: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Playback state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlaybackState {
    /// Index into `AnimatedScene::clips`, or None if no clip selected.
    pub clip_index: Option<usize>,
    /// Current playback time in seconds.
    pub time: f32,
    pub playing: bool,
    pub looping: bool,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            clip_index: None,
            time: 0.0,
            playing: false,
            looping: true,
            speed: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Animated scene
// ---------------------------------------------------------------------------

/// A complete animated scene: hierarchy of nodes + animation clips + playback.
pub struct AnimatedScene {
    pub name: String,
    pub nodes: Vec<SceneNode>,
    pub clips: Vec<AnimationClip>,
    pub playback: PlaybackState,
}

impl AnimatedScene {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            clips: Vec::new(),
            playback: PlaybackState::default(),
        }
    }

    /// Advance playback time by `dt` seconds, evaluate animation channels,
    /// and recompute world transforms. Call once per frame.
    pub fn update(&mut self, dt: f32) {
        // Advance time
        if self.playback.playing {
            if let Some(clip_idx) = self.playback.clip_index {
                if clip_idx < self.clips.len() {
                    let duration = self.clips[clip_idx].duration;
                    self.playback.time += dt * self.playback.speed;
                    if self.playback.looping && duration > 0.0 {
                        // Wrap around
                        self.playback.time = self.playback.time.rem_euclid(duration);
                    } else {
                        self.playback.time = self.playback.time.clamp(0.0, duration);
                        if self.playback.time >= duration {
                            self.playback.playing = false;
                        }
                    }
                }
            }
        }

        // Evaluate active clip
        if let Some(clip_idx) = self.playback.clip_index {
            if clip_idx < self.clips.len() {
                let t = self.playback.time;
                // Clone channels to avoid borrow conflict (clips is immutable during eval)
                let channels = self.clips[clip_idx].channels.to_vec();
                for channel in &channels {
                    if channel.target_node < self.nodes.len() {
                        channel.evaluate(t, &mut self.nodes);
                    }
                }
            }
        }

        // Recompute world transforms (parent-first ordering assumed)
        self.compute_world_transforms();
    }

    /// Parent world matrix for the node at `idx` (identity if root).
    /// Used by the viewer to convert an interactive world-space transform back
    /// into the node's local space.
    pub fn parent_world_matrix(&self, idx: usize) -> glam::Mat4 {
        match self.nodes.get(idx).and_then(|n| n.parent) {
            Some(p) if p < self.nodes.len() => self.nodes[p].world_transform.model_matrix(),
            _ => glam::Mat4::IDENTITY,
        }
    }

    /// Remove a node and every descendant. Other nodes' indices may change, so
    /// the caller must clear any cached node-indices (e.g. the current selection).
    /// Animation channels targeting removed nodes are dropped; surviving channels
    /// are re-indexed.
    pub fn remove_node_and_descendants(&mut self, root: usize) {
        if root >= self.nodes.len() {
            return;
        }
        // Collect every descendant via BFS.
        let mut to_remove = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if to_remove.insert(i) {
                stack.extend(self.nodes[i].children.iter().copied());
            }
        }

        // Build old-idx -> new-idx remap for the survivors.
        let mut remap = vec![None; self.nodes.len()];
        let mut new_idx = 0usize;
        for (old, _) in self.nodes.iter().enumerate() {
            if !to_remove.contains(&old) {
                remap[old] = Some(new_idx);
                new_idx += 1;
            }
        }

        // Drop the doomed nodes, then rewrite parent/children pointers on survivors.
        let mut survivors = Vec::with_capacity(new_idx);
        for (old, node) in self.nodes.drain(..).enumerate() {
            if remap[old].is_none() {
                continue;
            }
            let mut node = node;
            node.parent = node.parent.and_then(|p| remap[p]);
            node.children = node.children.into_iter().filter_map(|c| remap[c]).collect();
            survivors.push(node);
        }
        self.nodes = survivors;

        // Re-index animation channels; drop any targeting removed nodes.
        for clip in &mut self.clips {
            clip.channels.retain(|ch| remap[ch.target_node].is_some());
            for ch in &mut clip.channels {
                ch.target_node = remap[ch.target_node].unwrap();
            }
        }

        self.compute_world_transforms();
    }

    /// Walk nodes in order, computing world = parent_world * local.
    /// Assumes nodes are ordered so that parents appear before children.
    pub fn compute_world_transforms(&mut self) {
        for i in 0..self.nodes.len() {
            let local_mat = self.nodes[i].local_transform.model_matrix();
            let world_mat = match self.nodes[i].parent {
                Some(p) if p < i => self.nodes[p].world_transform.model_matrix() * local_mat,
                _ => local_mat,
            };
            let (scale, rotation, translation) = world_mat.to_scale_rotation_translation();
            self.nodes[i].world_transform = Transform {
                position: translation,
                rotation,
                scale,
            };
        }
    }

    /// Current clip duration, or 0 if no clip selected.
    pub fn current_duration(&self) -> f32 {
        self.playback
            .clip_index
            .and_then(|i| self.clips.get(i))
            .map_or(0.0, |c| c.duration)
    }

    /// Produce a flat list of `Primitive` instances for rendering.
    /// Uses world transforms and per-node material properties.
    pub fn render_primitives(&self) -> Vec<Primitive> {
        let mut out = Vec::new();
        for node in &self.nodes {
            if !node.visible {
                continue;
            }
            if let Some(kind) = node.mesh_kind {
                let mut prim = Primitive::new(kind, &node.name);
                prim.transform = node.world_transform;
                prim.color = Some(node.color);
                prim.roughness = node.roughness;
                prim.metallic = node.metallic;
                prim.material_index = node.material_index;
                out.push(prim);
            }
        }
        out
    }

    /// Number of visible mesh nodes.
    pub fn visible_mesh_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.visible && n.mesh_kind.is_some())
            .count()
    }
}
