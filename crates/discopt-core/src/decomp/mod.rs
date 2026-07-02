//! Graph kernels for the Decomposition Advisor.
//!
//! This module provides the foundational, allocation-light graph primitives
//! the Decomposition Advisor (see `docs/design/decomposition-advisor.md`) runs
//! over the graphs a model induces — the variable–constraint incidence graph,
//! the variable dependency graph, and the directed stage / dual-dependency
//! graphs. Everything is built on one substrate, [`CsrGraph`], a
//! compressed-sparse-row adjacency, so the Jacobian/Hessian/KKT views can share
//! vertex storage without copying.
//!
//! The kernels here are deliberately *pure* (no interior mutability, no global
//! state): a graph in, an owned result out. That is the same discipline that
//! keeps the Rayon MILP driver bit-reproducible (`design/rayon-parallelization.md`),
//! and it lets these run under `rayon` fan-out later without any locking.
//!
//! # Kernels
//! - [`connected_components`] — undirected components via union-find. The
//!   cheapest structural check; detects block-diagonal structure directly.
//! - [`strongly_connected_components`] — Tarjan's algorithm over a *directed*
//!   graph, for stage precedence and dual-dependency cycles.
//! - [`articulation_and_bridges`] — cut vertices and cut edges in one DFS; the
//!   exact, near-linear detector for small separators / single linking edges.
//!
//! All traversals are **iterative** (explicit work stacks) so a deep model — a
//! long constraint chain, a multistage graph — cannot overflow the native
//! stack.

pub mod components;
pub mod csr;
pub mod mincut;

pub use components::{connected_components, strongly_connected_components};
pub use csr::CsrGraph;
pub use mincut::{articulation_and_bridges, ArticulationResult};
