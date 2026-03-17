use glam::{Mat4, Quat, Vec3};

use crate::event::{InteractEvent, InteractKey, MouseButton};
use crate::numeric::NumericInput;
use crate::transform::{Primitive, Transform};

/// Which transform operation is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Grab,
    Scale,
    Rotate,
}

impl TransformMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Grab => "Grab",
            Self::Scale => "Scale",
            Self::Rotate => "Rotate",
        }
    }
}

/// A single world axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn label(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Unit vector for this axis.
    pub fn unit(self) -> Vec3 {
        match self {
            Self::X => Vec3::X,
            Self::Y => Vec3::Y,
            Self::Z => Vec3::Z,
        }
    }

    /// Mask: 1.0 on this axis, 0.0 on others.
    pub fn mask(self) -> Vec3 {
        self.unit()
    }
}

/// Axis constraint for the active transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisConstraint {
    /// No constraint — free transform (uses screen-space mouse movement).
    Free,
    /// Constrained to a single axis.
    SingleAxis(Axis),
    /// Constrained to the plane perpendicular to this axis (Shift+axis).
    Plane(Axis),
}

impl AxisConstraint {
    /// Returns a Vec3 mask: 1.0 for axes affected, 0.0 for locked.
    pub fn mask(self) -> Vec3 {
        match self {
            Self::Free => Vec3::ONE,
            Self::SingleAxis(a) => a.mask(),
            Self::Plane(a) => Vec3::ONE - a.mask(),
        }
    }

    pub fn label(self) -> Option<String> {
        match self {
            Self::Free => None,
            Self::SingleAxis(a) => Some(a.label().to_string()),
            Self::Plane(a) => Some(format!("Shift+{}", a.label())),
        }
    }
}

/// The state of the interaction system.
#[derive(Debug)]
enum InteractState {
    Idle,
    Transforming {
        mode: TransformMode,
        axis: AxisConstraint,
        numeric: NumericInput,
        original_transform: Transform,
        mouse_accum: [f32; 2],
        shift_held: bool,
    },
}

/// Result of processing an input event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractResult {
    /// Event consumed, nothing visually changed.
    Noop,
    /// Event consumed, preview transform updated — re-render.
    PreviewUpdated,
    /// Transform confirmed — commit to primitive.
    Confirmed,
    /// Transform canceled — revert to original.
    Canceled,
    /// Event not consumed — pass to camera / other systems.
    NotConsumed,
}

/// Information for the HUD overlay.
#[derive(Debug, Clone)]
pub struct DisplayInfo {
    pub mode: TransformMode,
    pub axis: AxisConstraint,
    pub numeric_text: String,
}

/// Camera context needed for screen-to-world sensitivity.
pub struct InteractContext {
    pub view_matrix: Mat4,
    pub proj_matrix: Mat4,
    pub viewport_width: f32,
    pub viewport_height: f32,
}

/// Blender-style modal transform interaction state machine.
///
/// Operates on a list of [`Primitive`]s. One primitive may be selected at a time.
/// When a mode key (G/S/R) is pressed with a selection, the system enters
/// `Transforming` state. Axis constraints, numeric input, and mouse accumulation
/// modify the preview transform. Enter/LMB confirms, Escape/RMB cancels.
pub struct TransformInteraction {
    state: InteractState,
    selected: Option<usize>,
}

impl TransformInteraction {
    pub fn new() -> Self {
        Self {
            state: InteractState::Idle,
            selected: None,
        }
    }

    /// Process an input event. Returns how the caller should respond.
    pub fn process_event(
        &mut self,
        event: &InteractEvent,
        primitives: &mut [Primitive],
        ctx: &InteractContext,
    ) -> InteractResult {
        match &mut self.state {
            InteractState::Idle => self.process_idle(event, primitives),
            InteractState::Transforming { .. } => self.process_transforming(event, primitives, ctx),
        }
    }

    fn process_idle(
        &mut self,
        event: &InteractEvent,
        primitives: &[Primitive],
    ) -> InteractResult {
        match event {
            InteractEvent::KeyDown(key) => match key {
                InteractKey::G | InteractKey::S | InteractKey::R => {
                    let Some(idx) = self.selected else {
                        return InteractResult::NotConsumed;
                    };
                    if idx >= primitives.len() {
                        return InteractResult::NotConsumed;
                    }
                    let mode = match key {
                        InteractKey::G => TransformMode::Grab,
                        InteractKey::S => TransformMode::Scale,
                        InteractKey::R => TransformMode::Rotate,
                        _ => unreachable!(),
                    };
                    self.state = InteractState::Transforming {
                        mode,
                        axis: AxisConstraint::Free,
                        numeric: NumericInput::new(),
                        original_transform: primitives[idx].transform,
                        mouse_accum: [0.0, 0.0],
                        shift_held: false,
                    };
                    InteractResult::PreviewUpdated
                }
                InteractKey::Delete => {
                    // Delete selected primitive — signal via Confirmed so caller can handle
                    if self.selected.is_some() {
                        InteractResult::Noop // deletion handled by caller checking key
                    } else {
                        InteractResult::NotConsumed
                    }
                }
                _ => InteractResult::NotConsumed,
            },
            _ => InteractResult::NotConsumed,
        }
    }

    fn process_transforming(
        &mut self,
        event: &InteractEvent,
        primitives: &mut [Primitive],
        ctx: &InteractContext,
    ) -> InteractResult {
        // Extract mutable fields — we need to pattern-match the state
        let InteractState::Transforming {
            mode,
            axis,
            numeric,
            original_transform,
            mouse_accum,
            shift_held,
        } = &mut self.state
        else {
            return InteractResult::NotConsumed;
        };

        match event {
            InteractEvent::KeyDown(key) => match key {
                // Axis constraints
                InteractKey::X => {
                    *axis = if *shift_held {
                        AxisConstraint::Plane(Axis::X)
                    } else {
                        AxisConstraint::SingleAxis(Axis::X)
                    };
                    InteractResult::PreviewUpdated
                }
                InteractKey::Y => {
                    *axis = if *shift_held {
                        AxisConstraint::Plane(Axis::Y)
                    } else {
                        AxisConstraint::SingleAxis(Axis::Y)
                    };
                    InteractResult::PreviewUpdated
                }
                InteractKey::Z => {
                    *axis = if *shift_held {
                        AxisConstraint::Plane(Axis::Z)
                    } else {
                        AxisConstraint::SingleAxis(Axis::Z)
                    };
                    InteractResult::PreviewUpdated
                }
                InteractKey::Shift => {
                    *shift_held = true;
                    InteractResult::Noop
                }
                InteractKey::Backspace => {
                    numeric.backspace();
                    InteractResult::PreviewUpdated
                }
                // Confirm
                InteractKey::Enter => {
                    if let Some(idx) = self.selected {
                        if idx < primitives.len() {
                            primitives[idx].transform = Self::compute_preview(*original_transform, *mode, *axis, numeric, *mouse_accum, ctx);
                        }
                    }
                    self.state = InteractState::Idle;
                    return InteractResult::Confirmed;
                }
                // Cancel
                InteractKey::Escape => {
                    self.state = InteractState::Idle;
                    return InteractResult::Canceled;
                }
                _ => InteractResult::Noop,
            },

            InteractEvent::KeyUp(InteractKey::Shift) => {
                *shift_held = false;
                InteractResult::Noop
            }
            InteractEvent::KeyUp(_) => InteractResult::Noop,

            InteractEvent::CharInput(ch) => {
                if numeric.push(*ch) {
                    InteractResult::PreviewUpdated
                } else {
                    InteractResult::Noop
                }
            }

            InteractEvent::MouseMove { dx, dy } => {
                // Only accumulate mouse movement when no numeric input
                if numeric.is_empty() {
                    mouse_accum[0] += dx;
                    mouse_accum[1] += dy;
                    InteractResult::PreviewUpdated
                } else {
                    InteractResult::Noop
                }
            }

            InteractEvent::MouseDown { button: MouseButton::Left } => {
                // LMB = confirm
                if let Some(idx) = self.selected {
                    if idx < primitives.len() {
                        primitives[idx].transform = Self::compute_preview(*original_transform, *mode, *axis, numeric, *mouse_accum, ctx);
                    }
                }
                self.state = InteractState::Idle;
                return InteractResult::Confirmed;
            }

            InteractEvent::MouseDown { button: MouseButton::Right } => {
                // RMB = cancel
                self.state = InteractState::Idle;
                return InteractResult::Canceled;
            }

            _ => InteractResult::Noop,
        }
    }

    fn compute_preview(
        original: Transform,
        mode: TransformMode,
        axis: AxisConstraint,
        numeric: &NumericInput,
        mouse_accum: [f32; 2],
        ctx: &InteractContext,
    ) -> Transform {
        match mode {
            TransformMode::Grab => {
                if let Some(value) = numeric.value() {
                    let mask = axis.mask();
                    Transform {
                        position: original.position + mask * value,
                        ..original
                    }
                } else {
                    // Project mouse delta to world-space movement
                    let delta = Self::mouse_to_world_grab(
                        mouse_accum, axis, ctx,
                    );
                    Transform {
                        position: original.position + delta,
                        ..original
                    }
                }
            }
            TransformMode::Scale => {
                let value = numeric.value().unwrap_or(
                    (mouse_accum[0] + mouse_accum[1]) * Self::sensitivity(TransformMode::Scale),
                );
                let mask = axis.mask();
                let factor = Vec3::ONE + mask * value;
                Transform {
                    scale: original.scale * factor,
                    ..original
                }
            }
            TransformMode::Rotate => {
                let value = numeric.value().unwrap_or(
                    (mouse_accum[0] + mouse_accum[1]) * Self::sensitivity(TransformMode::Rotate),
                );
                let angle = value.to_radians();
                let rot_axis = match axis {
                    AxisConstraint::Free => Vec3::Z,
                    AxisConstraint::SingleAxis(a) => a.unit(),
                    AxisConstraint::Plane(a) => a.unit(),
                };
                let rot = Quat::from_axis_angle(rot_axis, angle);
                Transform {
                    rotation: rot * original.rotation,
                    ..original
                }
            }
        }
    }

    /// Convert screen-space mouse delta to world-space grab displacement.
    fn mouse_to_world_grab(
        mouse_accum: [f32; 2],
        axis: AxisConstraint,
        ctx: &InteractContext,
    ) -> Vec3 {
        let sensitivity = 0.01;
        let dx = mouse_accum[0] * sensitivity;
        let dy = mouse_accum[1] * sensitivity;

        // Extract camera right and up vectors from the view matrix
        let view = ctx.view_matrix;
        let right = Vec3::new(view.col(0).x, view.col(1).x, view.col(2).x).normalize();
        let up = Vec3::new(view.col(0).y, view.col(1).y, view.col(2).y).normalize();

        match axis {
            AxisConstraint::Free => {
                // Move in the camera's right/up plane
                right * dx - up * dy
            }
            AxisConstraint::SingleAxis(a) => {
                // Project screen delta onto the screen-space direction of the axis
                let world_dir = a.unit();
                let screen_dir = Vec3::new(
                    right.dot(world_dir),
                    up.dot(world_dir),
                    0.0,
                );
                let screen_len = screen_dir.length();
                if screen_len < 1e-6 {
                    return Vec3::ZERO; // axis points at camera, can't project
                }
                let screen_dir = screen_dir / screen_len;
                // Dot the mouse delta with the screen projection of the axis
                let projected = dx * screen_dir.x - dy * screen_dir.y;
                world_dir * (projected / screen_len)
                }
            AxisConstraint::Plane(normal_axis) => {
                // Move in the plane perpendicular to normal_axis,
                // but only along the camera right/up components that lie in that plane
                let n = normal_axis.unit();
                let plane_right = (right - n * n.dot(right)).normalize_or_zero();
                let plane_up = (up - n * n.dot(up)).normalize_or_zero();
                plane_right * dx - plane_up * dy
            }
        }
    }

    fn sensitivity(mode: TransformMode) -> f32 {
        match mode {
            TransformMode::Scale => 0.005,
            TransformMode::Rotate => 0.5, // degrees per pixel
            _ => 0.01,
        }
    }

    /// Compute the preview transform for the currently-transforming primitive.
    /// Returns `None` if not in a transform state.
    pub fn preview_transform(&self, ctx: &InteractContext) -> Option<Transform> {
        if let InteractState::Transforming {
            mode,
            axis,
            numeric,
            original_transform,
            mouse_accum,
            ..
        } = &self.state
        {
            Some(Self::compute_preview(*original_transform, *mode, *axis, numeric, *mouse_accum, ctx))
        } else {
            None
        }
    }

    /// Information for HUD display. Returns `None` if idle.
    pub fn display_info(&self) -> Option<DisplayInfo> {
        if let InteractState::Transforming {
            mode,
            axis,
            numeric,
            ..
        } = &self.state
        {
            Some(DisplayInfo {
                mode: *mode,
                axis: *axis,
                numeric_text: numeric.display().to_string(),
            })
        } else {
            None
        }
    }

    /// Whether the interaction system is actively transforming (suppress camera controls).
    pub fn is_active(&self) -> bool {
        matches!(self.state, InteractState::Transforming { .. })
    }

    /// Currently selected primitive index.
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// Set the selected primitive index.
    pub fn set_selected(&mut self, idx: Option<usize>) {
        self.selected = idx;
    }
}

impl Default for TransformInteraction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::PrimitiveKind;

    fn dummy_ctx() -> InteractContext {
        InteractContext {
            view_matrix: Mat4::IDENTITY,
            proj_matrix: Mat4::IDENTITY,
            viewport_width: 800.0,
            viewport_height: 600.0,
        }
    }

    fn one_cube() -> Vec<Primitive> {
        vec![Primitive::new(PrimitiveKind::Cube, "Cube")]
    }

    #[test]
    fn test_idle_no_selection() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        // G with no selection => NotConsumed
        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);
        assert_eq!(r, InteractResult::NotConsumed);
        assert!(!ti.is_active());
    }

    #[test]
    fn test_grab_confirm_enter() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        // Press G => Transforming
        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);
        assert_eq!(r, InteractResult::PreviewUpdated);
        assert!(ti.is_active());

        // Type X constraint
        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &mut prims, &ctx);
        assert_eq!(r, InteractResult::PreviewUpdated);

        // Type "5"
        let r = ti.process_event(&InteractEvent::CharInput('5'), &mut prims, &ctx);
        assert_eq!(r, InteractResult::PreviewUpdated);

        // Check preview
        let preview = ti.preview_transform(&ctx).unwrap();
        assert!((preview.position.x - 5.0).abs() < 1e-6);
        assert!(preview.position.y.abs() < 1e-6);
        assert!(preview.position.z.abs() < 1e-6);

        // Confirm
        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::Enter), &mut prims, &ctx);
        assert_eq!(r, InteractResult::Confirmed);
        assert!(!ti.is_active());
        assert!((prims[0].transform.position.x - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cancel_reverts() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('9'), &mut prims, &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &mut prims, &ctx);

        // Cancel
        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::Escape), &mut prims, &ctx);
        assert_eq!(r, InteractResult::Canceled);
        assert!(!ti.is_active());
        // Original transform unchanged
        assert!(prims[0].transform.position.x.abs() < 1e-6);
    }

    #[test]
    fn test_scale_mode() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::S), &mut prims, &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::Y), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('2'), &mut prims, &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        // Scale Y should be 1 + 2 = 3
        assert!((preview.scale.y - 3.0).abs() < 1e-6);
        assert!((preview.scale.x - 1.0).abs() < 1e-6);
        assert!((preview.scale.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_mode() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::R), &mut prims, &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::Z), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('9'), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('0'), &mut prims, &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        // 90 degrees around Z
        let expected = Quat::from_axis_angle(Vec3::Z, 90.0_f32.to_radians());
        let dot = preview.rotation.dot(expected).abs();
        assert!(dot > 0.99, "rotation mismatch: dot={dot}");
    }

    #[test]
    fn test_mouse_accumulation() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &mut prims, &ctx);

        // Mouse movement (no numeric entered)
        ti.process_event(
            &InteractEvent::MouseMove { dx: 100.0, dy: 0.0 },
            &mut prims,
            &ctx,
        );

        let preview = ti.preview_transform(&ctx).unwrap();
        // 100 * 0.01 = 1.0 on X
        assert!((preview.position.x - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_shift_plane_constraint() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);

        // Hold shift then press X => plane constraint (YZ)
        ti.process_event(&InteractEvent::KeyDown(InteractKey::Shift), &mut prims, &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &mut prims, &ctx);

        ti.process_event(&InteractEvent::CharInput('3'), &mut prims, &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        // X locked, YZ affected
        assert!(preview.position.x.abs() < 1e-6);
        assert!((preview.position.y - 3.0).abs() < 1e-6);
        assert!((preview.position.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lmb_confirms() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('2'), &mut prims, &ctx);

        let r = ti.process_event(
            &InteractEvent::MouseDown { button: MouseButton::Left },
            &mut prims,
            &ctx,
        );
        assert_eq!(r, InteractResult::Confirmed);
        assert!((prims[0].transform.position.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmb_cancels() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('9'), &mut prims, &ctx);

        let r = ti.process_event(
            &InteractEvent::MouseDown { button: MouseButton::Right },
            &mut prims,
            &ctx,
        );
        assert_eq!(r, InteractResult::Canceled);
        // Position should be unchanged
        assert!(prims[0].transform.position.x.abs() < 1e-6);
    }

    #[test]
    fn test_display_info() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let mut prims = one_cube();
        ti.set_selected(Some(0));

        assert!(ti.display_info().is_none());

        ti.process_event(&InteractEvent::KeyDown(InteractKey::S), &mut prims, &ctx);
        let info = ti.display_info().unwrap();
        assert_eq!(info.mode, TransformMode::Scale);
        assert_eq!(info.axis, AxisConstraint::Free);
        assert!(info.numeric_text.is_empty());

        ti.process_event(&InteractEvent::KeyDown(InteractKey::Z), &mut prims, &ctx);
        ti.process_event(&InteractEvent::CharInput('4'), &mut prims, &ctx);
        let info = ti.display_info().unwrap();
        assert_eq!(info.axis, AxisConstraint::SingleAxis(Axis::Z));
        assert_eq!(info.numeric_text, "4");
    }
}
