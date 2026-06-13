//! Warm-started revised simplex LP solver for the MILP branch-and-bound.
//!
//! This is the per-node LP engine for **pure MILP**: a bounded-variable revised
//! simplex whose basis factorization is provided by the [`feral`] crate's
//! unsymmetric LU engine (`ftran`/`btran` + product-form column updates). After
//! a B&B branch changes one variable bound, the child re-optimizes from its
//! parent's optimal basis with a few **dual-simplex** pivots — the warm start
//! that makes node throughput competitive, in contrast to the cold
//! interior-point solve POUNCE does per node.
//!
//! POUNCE/IPM remains the engine for MINLP/MIQP/NLP (nonlinear relaxations,
//! differentiability); this module is only reached for linear MILP nodes.
//!
//! Scaffolding (this commit / roadmap P0): the [`linsolve`] abstraction — a
//! [`LinearSolver`] trait with the production [`linsolve::FeralLU`] backend and
//! a dense oracle [`linsolve::DenseLU`]. The primal/dual simplex drivers build
//! on this in subsequent increments.

pub mod linsolve;
pub mod primal;

pub use primal::solve_lp;

use crate::lp::basis::Basis;

/// Outcome of an LP solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    /// Optimal basis found.
    Optimal,
    /// Primal infeasible (certified by phase-1 / dual unboundedness).
    Infeasible,
    /// Primal unbounded.
    Unbounded,
    /// Hit the iteration limit without converging.
    IterLimit,
    /// Numerical breakdown (singular basis, stall) — caller should fall back.
    Numerical,
}

/// Tunable simplex parameters.
#[derive(Debug, Clone)]
pub struct SimplexOptions {
    /// Feasibility/optimality tolerance.
    pub tol: f64,
    /// Maximum pivots before declaring [`LpStatus::IterLimit`].
    pub max_iter: usize,
}

impl Default for SimplexOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_iter: 100_000,
        }
    }
}

/// Result of an LP solve: status plus the primal point, objective, and the
/// optimal [`Basis`] (the warm-start state a child node inherits).
#[derive(Debug, Clone)]
pub struct LpSolve {
    /// Solve status.
    pub status: LpStatus,
    /// Primal solution, length `n` (structural + slack columns).
    pub x: Vec<f64>,
    /// Objective value `cᵀx`.
    pub obj: f64,
    /// Optimal basis (basic columns + nonbasic at-bound status).
    pub basis: Basis,
    /// Simplex pivots performed.
    pub iters: usize,
}
