use glam::{Mat4, Quat, Vec3};

use crate::event::{InteractEvent, InteractKey, MouseButton};
use crate::numeric::NumericInput;
use crate::transform::Transform;

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

/// What the user currently has selected. The interaction system is agnostic about
/// which kind it is — the caller routes confirmed transforms to the right place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// Index into the user primitive list.
    Primitive(usize),
    /// Index into the animated scene's nodes.
    Node(usize),
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
#[derive(Debug, Clone, Copy)]
pub enum InteractResult {
    /// Event consumed, nothing visually changed.
    Noop,
    /// Event consumed, preview transform updated — re-render.
    PreviewUpdated,
    /// Transform confirmed — caller should apply this transform to the selected entity.
    Confirmed(Transform),
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
/// Tracks one [`Selection`] at a time. The caller is responsible for:
/// - Calling [`set_current_transform`] each frame with the world-space transform
///   of the selected entity (or `None` if nothing is selected).
/// - Routing [`InteractResult::Confirmed`] payloads back to the right entity.
///
/// When a mode key (G/S/R) is pressed with a selection, the system enters
/// `Transforming` state. Axis constraints, numeric input, and mouse accumulation
/// modify the preview transform. Enter/LMB confirms, Escape/RMB cancels.
pub struct TransformInteraction {
    state: InteractState,
    selected: Option<Selection>,
    current_transform: Option<Transform>,
}

impl TransformInteraction {
    pub fn new() -> Self {
        Self {
            state: InteractState::Idle,
            selected: None,
            current_transform: None,
        }
    }

    /// Tell the state machine the current world-space transform of the selected entity.
    /// Called each frame by the viewer before feeding events.
    pub fn set_current_transform(&mut self, t: Option<Transform>) {
        self.current_transform = t;
    }

    /// Process an input event. Returns how the caller should respond.
    pub fn process_event(
        &mut self,
        event: &InteractEvent,
        ctx: &InteractContext,
    ) -> InteractResult {
        match &mut self.state {
            InteractState::Idle => self.process_idle(event),
            InteractState::Transforming { .. } => self.process_transforming(event, ctx),
        }
    }

    fn process_idle(&mut self, event: &InteractEvent) -> InteractResult {
        match event {
            InteractEvent::KeyDown(key) => match key {
                InteractKey::G | InteractKey::S | InteractKey::R => {
                    if self.selected.is_none() {
                        return InteractResult::NotConsumed;
                    }
                    let Some(original) = self.current_transform else {
                        return InteractResult::NotConsumed;
                    };
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
                        original_transform: original,
                        mouse_accum: [0.0, 0.0],
                        shift_held: false,
                    };
                    InteractResult::PreviewUpdated
                }
                InteractKey::Delete => {
                    // Delete selected entity — signal via Noop so caller can handle
                    if self.selected.is_some() {
                        InteractResult::Noop
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
        ctx: &InteractContext,
    ) -> InteractResult {
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
                InteractKey::Enter => {
                    let new_t = Self::compute_preview(*original_transform, *mode, *axis, numeric, *mouse_accum, ctx);
                    self.state = InteractState::Idle;
                    InteractResult::Confirmed(new_t)
                }
                InteractKey::Escape => {
                    self.state = InteractState::Idle;
                    InteractResult::Canceled
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
                if numeric.is_empty() {
                    mouse_accum[0] += dx;
                    mouse_accum[1] += dy;
                    InteractResult::PreviewUpdated
                } else {
                    InteractResult::Noop
                }
            }

            InteractEvent::MouseDown { button: MouseButton::Left } => {
                let new_t = Self::compute_preview(*original_transform, *mode, *axis, numeric, *mouse_accum, ctx);
                self.state = InteractState::Idle;
                InteractResult::Confirmed(new_t)
            }

            InteractEvent::MouseDown { button: MouseButton::Right } => {
                self.state = InteractState::Idle;
                InteractResult::Canceled
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
                    let delta = Self::mouse_to_world_grab(
                        mouse_accum, axis, original.position, ctx,
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
        obj_pos: Vec3,
        ctx: &InteractContext,
    ) -> Vec3 {
        let view_pos = ctx.view_matrix * obj_pos.extend(1.0);
        let depth = (-view_pos.z).max(0.01);

        let focal_px = ctx.proj_matrix.col(1).y * ctx.viewport_height * 0.5;
        let world_per_px = depth / focal_px;

        let dx = mouse_accum[0] * world_per_px;
        let dy = mouse_accum[1] * world_per_px;

        let view = ctx.view_matrix;
        let right = Vec3::new(view.col(0).x, view.col(1).x, view.col(2).x).normalize();
        let up = Vec3::new(view.col(0).y, view.col(1).y, view.col(2).y).normalize();

        match axis {
            AxisConstraint::Free => right * dx - up * dy,
            AxisConstraint::SingleAxis(a) => {
                let world_dir = a.unit();
                let screen_dir = Vec3::new(
                    right.dot(world_dir),
                    up.dot(world_dir),
                    0.0,
                );
                let screen_len = screen_dir.length();
                if screen_len < 1e-6 {
                    return Vec3::ZERO;
                }
                let screen_dir = screen_dir / screen_len;
                let projected = dx * screen_dir.x - dy * screen_dir.y;
                world_dir * (projected / screen_len)
            }
            AxisConstraint::Plane(normal_axis) => {
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
            TransformMode::Rotate => 0.5,
            _ => 0.01,
        }
    }

    /// Compute the preview transform for the currently-transforming entity.
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

    /// Currently selected entity.
    pub fn selected(&self) -> Option<Selection> {
        self.selected
    }

    /// Set the selected entity.
    pub fn set_selected(&mut self, sel: Option<Selection>) {
        // Cancel any in-progress transform if selection changes
        if self.selected != sel && self.is_active() {
            self.state = InteractState::Idle;
        }
        self.selected = sel;
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

    fn dummy_ctx() -> InteractContext {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 800.0 / 600.0, 0.1, 100.0);
        InteractContext {
            view_matrix: view,
            proj_matrix: proj,
            viewport_width: 800.0,
            viewport_height: 600.0,
        }
    }

    fn setup() -> (TransformInteraction, InteractContext) {
        let mut ti = TransformInteraction::new();
        ti.set_selected(Some(Selection::Primitive(0)));
        ti.set_current_transform(Some(Transform::default()));
        (ti, dummy_ctx())
    }

    #[test]
    fn test_idle_no_selection() {
        let mut ti = TransformInteraction::new();
        let ctx = dummy_ctx();
        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        assert!(matches!(r, InteractResult::NotConsumed));
        assert!(!ti.is_active());
    }

    #[test]
    fn test_grab_confirm_enter() {
        let (mut ti, ctx) = setup();

        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        assert!(matches!(r, InteractResult::PreviewUpdated));
        assert!(ti.is_active());

        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &ctx);
        ti.process_event(&InteractEvent::CharInput('5'), &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        assert!((preview.position.x - 5.0).abs() < 1e-6);

        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::Enter), &ctx);
        match r {
            InteractResult::Confirmed(t) => {
                assert!((t.position.x - 5.0).abs() < 1e-6);
            }
            _ => panic!("expected Confirmed, got {:?}", r),
        }
        assert!(!ti.is_active());
    }

    #[test]
    fn test_cancel_reverts() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        ti.process_event(&InteractEvent::CharInput('9'), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &ctx);

        let r = ti.process_event(&InteractEvent::KeyDown(InteractKey::Escape), &ctx);
        assert!(matches!(r, InteractResult::Canceled));
        assert!(!ti.is_active());
    }

    #[test]
    fn test_scale_mode() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::S), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::Y), &ctx);
        ti.process_event(&InteractEvent::CharInput('2'), &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        assert!((preview.scale.y - 3.0).abs() < 1e-6);
        assert!((preview.scale.x - 1.0).abs() < 1e-6);
        assert!((preview.scale.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_mode() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::R), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::Z), &ctx);
        ti.process_event(&InteractEvent::CharInput('9'), &ctx);
        ti.process_event(&InteractEvent::CharInput('0'), &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        let expected = Quat::from_axis_angle(Vec3::Z, 90.0_f32.to_radians());
        let dot = preview.rotation.dot(expected).abs();
        assert!(dot > 0.99, "rotation mismatch: dot={dot}");
    }

    #[test]
    fn test_mouse_accumulation() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &ctx);

        ti.process_event(&InteractEvent::MouseMove { dx: 100.0, dy: 0.0 }, &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        assert!(preview.position.x > 0.5, "x={}", preview.position.x);
        assert!(preview.position.x < 1.0, "x={}", preview.position.x);
        assert!(preview.position.y.abs() < 1e-4);
    }

    #[test]
    fn test_shift_plane_constraint() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::Shift), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &ctx);
        ti.process_event(&InteractEvent::CharInput('3'), &ctx);

        let preview = ti.preview_transform(&ctx).unwrap();
        assert!(preview.position.x.abs() < 1e-6);
        assert!((preview.position.y - 3.0).abs() < 1e-6);
        assert!((preview.position.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lmb_confirms() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        ti.process_event(&InteractEvent::KeyDown(InteractKey::X), &ctx);
        ti.process_event(&InteractEvent::CharInput('2'), &ctx);

        let r = ti.process_event(
            &InteractEvent::MouseDown { button: MouseButton::Left },
            &ctx,
        );
        match r {
            InteractResult::Confirmed(t) => {
                assert!((t.position.x - 2.0).abs() < 1e-6);
            }
            _ => panic!("expected Confirmed, got {:?}", r),
        }
    }

    #[test]
    fn test_rmb_cancels() {
        let (mut ti, ctx) = setup();

        ti.process_event(&InteractEvent::KeyDown(InteractKey::G), &ctx);
        ti.process_event(&InteractEvent::CharInput('9'), &ctx);

        let r = ti.process_event(
            &InteractEvent::MouseDown { button: MouseButton::Right },
            &ctx,
        );
        assert!(matches!(r, InteractResult::Canceled));
    }

    #[test]
    fn test_display_info() {
        let (mut ti, ctx) = setup();

        assert!(ti.display_info().is_none());

        ti.process_event(&InteractEvent::KeyDown(InteractKey::S), &ctx);
        let info = ti.display_info().unwrap();
        assert_eq!(info.mode, TransformMode::Scale);
        assert_eq!(info.axis, AxisConstraint::Free);
        assert!(info.numeric_text.is_empty());

        ti.process_event(&InteractEvent::KeyDown(InteractKey::Z), &ctx);
        ti.process_event(&InteractEvent::CharInput('4'), &ctx);
        let info = ti.display_info().unwrap();
        assert_eq!(info.axis, AxisConstraint::SingleAxis(Axis::Z));
        assert_eq!(info.numeric_text, "4");
    }
}
