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

pub mod batch;
pub mod dual;
pub mod linsolve;
pub mod presolve;
pub mod primal;
pub mod scaling;
pub mod sparse;

pub use batch::{solve_lp_batch, solve_lp_multi_rhs, LpInstance};
pub use dual::{
    solve_lp_warm, solve_lp_warm_csc, solve_lp_warm_scaled, solve_lp_warm_scaled_csc, PreparedDual,
};
pub use presolve::{tighten_bounds, tighten_bounds_csc};
pub use primal::{solve_lp, solve_lp_cols, solve_lp_cols_scaled, solve_lp_scaled};
pub use scaling::Scaling;
pub use sparse::SparseCols;

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
    /// Optional absolute wall-clock deadline. When set, the primal and dual
    /// iteration loops poll it every few hundred pivots and bail out with
    /// [`LpStatus::IterLimit`] once it passes. This bounds the cost of a *single*
    /// pathological LP solve — e.g. a dense, degenerate lifted-McCormick
    /// relaxation that would otherwise grind all the way to `max_iter` and blow
    /// past the enclosing MILP/B&B time budget. The bail is reported exactly like
    /// the iteration cap, which every caller already treats soundly: the node
    /// gets a non-pruning bound and the gap is left uncertified, so optimality is
    /// never falsely claimed. `None` (the default) disables the check, leaving
    /// short LP solves bit-identical to before.
    pub deadline: Option<std::time::Instant>,
    /// Warm dual-simplex **stall guard** (discopt F2). When `true` (the default),
    /// a warm dual re-optimization that fails to converge within a *size-derived*
    /// pivot budget — [`warm_stall_cap`](Self::warm_stall_cap) `= K·(m+n)+C`,
    /// clamped to `max_iter` — abandons the warm basis and cold-solves the same LP
    /// (the existing numerical-breakdown fallback), returning the identical
    /// optimum via a different path. This bounds the *tail*: a warm re-solve after
    /// a row-append can enter a floating-point degenerate stall near the
    /// feasibility tolerance and grind all the way to `max_iter` (100 000 pivots,
    /// ~8.7 s on a 118×143 LP for nvs01) when a cold solve of the same LP costs
    /// ~5 ms. Healthy warm solves finish in a handful of pivots (≤25 on the
    /// profiled class), orders of magnitude below the cap, so the guard is inert
    /// on them and the result is bit-identical. Bound-neutral: the cold fallback
    /// is the same trusted path a numerical breakdown already takes, so it returns
    /// the same LP optimum/vertex — only *who* computes it changes on a stall.
    /// Set `false` to restore the pure-`max_iter` legacy bound (e.g. the bare
    /// pure-LP binding, or to exercise the raw loop in a test).
    pub warm_stall_guard: bool,
    /// Explicit override for the warm-dual stall cap, in pivots. `None` (the
    /// default) uses the size-derived `K·(m+n)+C`; `Some(k)` forces exactly `k`
    /// (still clamped to `max_iter`). Only honored when `warm_stall_guard` is set.
    /// Exists so a test can drive a deterministic, small cap without depending on
    /// LP size; production always leaves it `None`.
    pub warm_stall_cap_override: Option<usize>,
}

impl SimplexOptions {
    /// Coefficient `K` and constant `C` of the size-derived warm-dual stall cap
    /// `K·(m+n)+C`. Chosen from the profiled healthy-call distribution (F2 entry
    /// experiment, `docs/dev/bottleneck-profile-2026-07-05.md` §5): warm dual
    /// re-solves on the nvs01/st_e36/nvs09 class converge in ≤25 pivots on LPs of
    /// ~120 rows (m+n≈260), while a stall reaches the full 100 000-pivot `max_iter`.
    /// `20·(m+n)+500` sits ~230× above the observed healthy maximum (so no healthy
    /// solve ever trips) yet ~18× below the stall (m+n≈260 → cap≈5 700 pivots ≈
    /// 0.5 s vs the 8.7 s full grind), recovering the ~5 ms cold cost promptly.
    const WARM_STALL_K: usize = 20;
    const WARM_STALL_C: usize = 500;

    /// The effective warm dual-simplex stall cap for an `m×n` (rows×cols) LP:
    /// the [`warm_stall_cap_override`](Self::warm_stall_cap_override) if set, else
    /// the size-derived `K·(m+n)+C`, both clamped to `max_iter`. Only meaningful
    /// when [`warm_stall_guard`](Self::warm_stall_guard) is set.
    #[inline]
    pub fn warm_stall_cap(&self, m: usize, n: usize) -> usize {
        self.warm_stall_cap_override
            .unwrap_or_else(|| {
                Self::WARM_STALL_K
                    .saturating_mul(m.saturating_add(n))
                    .saturating_add(Self::WARM_STALL_C)
            })
            .min(self.max_iter)
    }
}

impl Default for SimplexOptions {
    fn default() -> Self {
        Self {
            tol: 1e-9,
            max_iter: 100_000,
            deadline: None,
            warm_stall_guard: true,
            warm_stall_cap_override: None,
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
    /// Certificate vector of length `m` (one per row), interpreted by `status`:
    ///
    /// * [`LpStatus::Optimal`] — the row duals `y = B⁻ᵀ c_B`. Feeding these to a
    ///   Neumaier–Shcherbina safe-bound evaluation yields a *rigorous* lower
    ///   bound that holds at any conditioning (it never over-estimates even when
    ///   the reported vertex objective drifts on an ill-conditioned basis), so a
    ///   caller can certify the bound without a second independent solve.
    /// * [`LpStatus::Infeasible`] — a **Farkas dual ray** candidate: a free-sign
    ///   `y` such that `bᵀy` exceeds the box-maximum of `(Aᵀy)ᵀz`, a verifiable
    ///   proof the feasible set is empty. The caller verifies it (trying ±y, with
    ///   a magnitude-scaled margin) before trusting the infeasible verdict — so
    ///   an imperfect candidate only forces a fallback, never an unsound fathom.
    ///
    /// Empty for every other status. Verification is the caller's job; this is a
    /// *candidate*, sound only once independently checked.
    pub dual: Vec<f64>,
    /// Primal unbounded ray candidate of length `n`, populated only for
    /// [`LpStatus::Unbounded`]: a direction `d` with `A d = 0`, box-feasible, and
    /// `cᵀd < 0` along which the objective decreases without bound. Empty
    /// otherwise. Like [`Self::dual`] it is a candidate for the caller to verify.
    pub ray: Vec<f64>,
}
