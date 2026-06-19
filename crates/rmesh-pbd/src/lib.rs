//! GPU XPBD distance-constraint solver for localized vertex grabs on tet meshes.
//!
//! Algorithm port of DelTetRenderer's `PBDMove.hpp`:
//! - BFS island construction (CPU) — handles + active particles + boundary pin.
//! - Greedy edge coloring (CPU) — partition distance constraints into
//!   vertex-disjoint batches for Gauss-Seidel parallelism.
//! - Per-step GPU pipeline (compute):
//!     1. apply_handles  — overwrite handle particle positions from user input
//!     2. predict        — symplectic Euler (no gravity, matches reference)
//!     3. solve_distance × solver_iterations × num_colors — XPBD edge correction
//!     4. finalize       — recover velocity, commit position, scatter into
//!                          the scene's global `vertices` storage buffer
//!
//! Typical use from a viewer:
//! ```ignore
//! // Once per scene load:
//! let topology = MeshTopology::build(&scene.indices, scene.vertex_count, scene.tet_count);
//! let pipelines = PbdPipelines::new(device);
//!
//! // On BeginGrab (user clicked + holds LMB on selected vertices):
//! let island = build_island(&topology, &scene.indices, &scene.vertices, &handles, 0.25);
//! let coloring = color_constraints(island.particles.len(), &island.distance_constraints);
//! let solver = PbdSolver::init_grab(device, &pipelines, &scene_buffers.vertices,
//!                                   &island, &coloring, 16);
//!
//! // Each drag frame:
//! solver.step(queue, &mut encoder, &pipelines, 0.016, &handle_positions);
//! ```

mod coloring;
mod island;
mod mesh_topology;
mod solver;

pub use coloring::{color_constraints, ConstraintColoring};
pub use island::{build_island, DistanceConstraint, Island, Particle};
pub use mesh_topology::MeshTopology;
pub use solver::{PbdPipelines, PbdSolver};
