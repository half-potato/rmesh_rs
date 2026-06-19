//! Vertex-selection interaction mode.
//!
//! A separate state machine from [`crate::TransformInteraction`]: this one
//! drives vertex pick + grab gestures for the PBD soft-body solver. The viewer
//! routes events to whichever interaction is active, then acts on the
//! returned [`VertexSelectResult`].
//!
//! Flow:
//! 1. User presses Tab → [`VertexSelectInteraction::set_enabled(true)`] →
//!    state = `Hovering`. Camera input is suppressed.
//! 2. CursorMoved updates `mouse_pos` (viewer does live-highlight separately).
//! 3. LMB down on a vertex →
//!    a. result is [`VertexSelectResult::Pick`] — viewer hit-tests and calls
//!       [`set_selected`]. If nothing under the cursor, viewer clears selection.
//!    b. With a non-empty selection, result is [`VertexSelectResult::BeginGrab`].
//!       Viewer calls `PbdSolver::init_grab(selected)`.
//! 4. CursorMoved while `Grabbing` → [`VertexSelectResult::UpdateGrab`].
//!    Viewer unprojects the mouse delta to world space and calls `pbd.step`.
//! 5. LMB up → [`VertexSelectResult::ConfirmGrab`]. Viewer drops the PBD
//!    solver state but keeps the deformed vertices.
//! 6. Escape while `Grabbing` → [`VertexSelectResult::CancelGrab`]. Viewer
//!    restores the pre-grab vertex positions and drops the solver.
//! 7. Tab again → mode disabled.

use crate::event::{InteractEvent, InteractKey, MouseButton};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Inactive,
    Hovering,
    Grabbing,
}

/// What the viewer should do in response to an event fed to
/// [`VertexSelectInteraction::process_event`].
#[derive(Debug, Clone, Copy)]
pub enum VertexSelectResult {
    /// Event consumed; nothing to do.
    Noop,
    /// Event not consumed — pass to camera / other systems.
    NotConsumed,
    /// LMB down while idle: viewer should hit-test current `mouse_pos` and
    /// call [`VertexSelectInteraction::set_selected`].
    Pick,
    /// Begin a grab on the current selection. Viewer initializes its PBD solver.
    BeginGrab,
    /// Mouse moved while grabbing. Viewer feeds new mouse pos to the solver.
    UpdateGrab,
    /// LMB up while grabbing — commit the deformation.
    ConfirmGrab,
    /// Escape while grabbing — viewer should revert vertex positions and
    /// drop the solver.
    CancelGrab,
}

/// State for vertex pick + drag-to-PBD gestures.
///
/// Stateless w.r.t. the actual GPU/vertex data — owns only enough to know
/// whether camera input should be suppressed and what mouse delta to emit on
/// each step.
pub struct VertexSelectInteraction {
    state: State,
    /// Scene-global vertex indices currently selected. Set by viewer in
    /// response to [`VertexSelectResult::Pick`].
    selected: Vec<u32>,
    /// Most recent cursor position in window pixels. Updated on every
    /// CursorMoved while enabled.
    mouse_pos: [f32; 2],
    /// Cursor position at grab start, used to compute mouse delta for
    /// world-space drag in the viewer's unproject step.
    mouse_start: [f32; 2],
}

impl Default for VertexSelectInteraction {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexSelectInteraction {
    pub fn new() -> Self {
        Self {
            state: State::Inactive,
            selected: Vec::new(),
            mouse_pos: [0.0, 0.0],
            mouse_start: [0.0, 0.0],
        }
    }

    /// Whether vertex-select mode is currently on (Tab was pressed).
    pub fn is_enabled(&self) -> bool {
        !matches!(self.state, State::Inactive)
    }

    /// Whether a grab is currently in progress (suppress camera; expect
    /// per-frame PBD steps).
    pub fn is_grabbing(&self) -> bool {
        matches!(self.state, State::Grabbing)
    }

    /// Suppress all camera input while VertexSelect mode is active, not just
    /// while grabbing — so the user can hover-pick without panning the view.
    pub fn suppresses_camera(&self) -> bool {
        self.is_enabled()
    }

    /// Most recent cursor position in window pixels.
    pub fn mouse_pos(&self) -> [f32; 2] {
        self.mouse_pos
    }

    /// Cursor position when the current grab started. Use with [`mouse_pos`]
    /// to compute the screen-space drag vector.
    pub fn mouse_start(&self) -> [f32; 2] {
        self.mouse_start
    }

    /// Currently selected scene-global vertex indices.
    pub fn selected(&self) -> &[u32] {
        &self.selected
    }

    /// Viewer calls this after a [`VertexSelectResult::Pick`] with the
    /// hit-test result (empty = no vertex under cursor).
    pub fn set_selected(&mut self, sel: Vec<u32>) {
        self.selected = sel;
    }

    /// Force enable/disable from outside (e.g. UI checkbox). Equivalent to
    /// the Tab key.
    pub fn set_enabled(&mut self, on: bool) {
        match (on, self.state) {
            (true, State::Inactive) => self.state = State::Hovering,
            (false, _) => {
                self.state = State::Inactive;
                self.selected.clear();
            }
            _ => {}
        }
    }

    /// Process one input event. Returns what the viewer should do.
    pub fn process_event(&mut self, event: &InteractEvent) -> VertexSelectResult {
        match (self.state, event) {
            // Tab toggles the whole mode in/out, regardless of substate.
            (_, InteractEvent::KeyDown(InteractKey::Tab)) => {
                let was_on = self.is_enabled();
                self.set_enabled(!was_on);
                VertexSelectResult::Noop
            }

            (State::Inactive, _) => VertexSelectResult::NotConsumed,

            // CursorMoved — track and (if grabbing) emit UpdateGrab.
            (_, InteractEvent::MouseMove { dx, dy }) => {
                self.mouse_pos[0] += dx;
                self.mouse_pos[1] += dy;
                if matches!(self.state, State::Grabbing) {
                    VertexSelectResult::UpdateGrab
                } else {
                    VertexSelectResult::Noop
                }
            }

            // LMB down: pick → (if selection non-empty) begin grab.
            (State::Hovering, InteractEvent::MouseDown { button: MouseButton::Left }) => {
                self.mouse_start = self.mouse_pos;
                VertexSelectResult::Pick
            }

            // LMB up while grabbing → confirm. Otherwise no-op.
            (State::Grabbing, InteractEvent::MouseUp { button: MouseButton::Left }) => {
                self.state = State::Hovering;
                VertexSelectResult::ConfirmGrab
            }

            // Escape while grabbing → cancel. Escape while just hovering →
            // exit the mode.
            (State::Grabbing, InteractEvent::KeyDown(InteractKey::Escape)) => {
                self.state = State::Hovering;
                VertexSelectResult::CancelGrab
            }
            (State::Hovering, InteractEvent::KeyDown(InteractKey::Escape)) => {
                self.set_enabled(false);
                VertexSelectResult::Noop
            }

            _ => VertexSelectResult::Noop,
        }
    }

    /// Promote the current `Hovering` state to `Grabbing`. Viewer calls this
    /// after [`VertexSelectResult::Pick`] once it has populated `selected`
    /// with at least one vertex.
    pub fn begin_grab(&mut self) {
        if matches!(self.state, State::Hovering) && !self.selected.is_empty() {
            self.state = State::Grabbing;
        }
    }

    /// Set cursor position directly (e.g. when receiving an absolute
    /// CursorMoved before any deltas accumulate). Use this when feeding the
    /// initial mouse coordinates after enabling the mode.
    pub fn set_mouse_pos(&mut self, pos: [f32; 2]) {
        self.mouse_pos = pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_toggles_mode() {
        let mut vs = VertexSelectInteraction::new();
        assert!(!vs.is_enabled());
        let _ = vs.process_event(&InteractEvent::KeyDown(InteractKey::Tab));
        assert!(vs.is_enabled());
        let _ = vs.process_event(&InteractEvent::KeyDown(InteractKey::Tab));
        assert!(!vs.is_enabled());
    }

    #[test]
    fn pick_then_begin_then_drag_then_confirm() {
        let mut vs = VertexSelectInteraction::new();
        vs.set_enabled(true);
        vs.set_mouse_pos([100.0, 100.0]);

        // LMB down → Pick
        let r = vs.process_event(&InteractEvent::MouseDown { button: MouseButton::Left });
        assert!(matches!(r, VertexSelectResult::Pick));
        assert_eq!(vs.mouse_start(), [100.0, 100.0]);

        // Viewer reports a hit
        vs.set_selected(vec![42]);
        vs.begin_grab();
        assert!(vs.is_grabbing());

        // Cursor moves → UpdateGrab
        let r = vs.process_event(&InteractEvent::MouseMove { dx: 10.0, dy: 5.0 });
        assert!(matches!(r, VertexSelectResult::UpdateGrab));
        assert_eq!(vs.mouse_pos(), [110.0, 105.0]);

        // LMB up → ConfirmGrab, drops back to hovering
        let r = vs.process_event(&InteractEvent::MouseUp { button: MouseButton::Left });
        assert!(matches!(r, VertexSelectResult::ConfirmGrab));
        assert!(!vs.is_grabbing());
        assert!(vs.is_enabled());
    }

    #[test]
    fn escape_while_grabbing_cancels_only_the_grab() {
        let mut vs = VertexSelectInteraction::new();
        vs.set_enabled(true);
        vs.set_selected(vec![7]);
        vs.process_event(&InteractEvent::MouseDown { button: MouseButton::Left });
        vs.begin_grab();
        let r = vs.process_event(&InteractEvent::KeyDown(InteractKey::Escape));
        assert!(matches!(r, VertexSelectResult::CancelGrab));
        assert!(!vs.is_grabbing());
        assert!(vs.is_enabled());
    }

    #[test]
    fn escape_while_hovering_exits_mode() {
        let mut vs = VertexSelectInteraction::new();
        vs.set_enabled(true);
        let r = vs.process_event(&InteractEvent::KeyDown(InteractKey::Escape));
        assert!(matches!(r, VertexSelectResult::Noop));
        assert!(!vs.is_enabled());
    }

    #[test]
    fn camera_suppression_active_whenever_mode_enabled() {
        let mut vs = VertexSelectInteraction::new();
        assert!(!vs.suppresses_camera());
        vs.set_enabled(true);
        assert!(vs.suppresses_camera());
    }
}
