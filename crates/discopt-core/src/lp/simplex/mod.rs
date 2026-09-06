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
pub mod refine;
pub mod regularized_lu;
pub mod scaling;
pub mod sparse;

pub use batch::{solve_lp_batch, solve_lp_multi_rhs, LpInstance};
pub use dual::{
    solve_lp_warm, solve_lp_warm_csc, solve_lp_warm_scaled, solve_lp_warm_scaled_csc,
    unstable_pivot_recovery_default, PreparedDual,
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
    /// Expel **zero-valued basic artificials** from the final (phase-2 optimal)
    /// basis so the emitted basis is full-rank in real columns whenever the
    /// constraint rows are — the precondition every warm-start entry point
    /// (`solve_lp_warm`, `PreparedDual::prepare`) requires (P1.0,
    /// `docs/dev/scip-parity-kernel-plan.md`). On the slackless equality rows of
    /// the convex big-M relaxations, phase-1 parks a degenerate (value-0) basic
    /// artificial the strict reduced-cost rule never expels; the legacy emission
    /// finds no zero-valued singleton slack to substitute (equality rows have
    /// none) and emits a SHORT, mislabelled basis, so warm-start silently
    /// cold-falls-back (measured: rsyn0805m 90/s vs the ~1 400/s kernel rate the
    /// full basis unlocks). When `true`, a finalization pass drives them out via
    /// `t = 0` degenerate pivots (x/obj/duals invariant on the vertex).
    ///
    /// Default `false`. This is **bound-changing on the default path**, not
    /// neutral: it also fills rows an inequality LP left short (a nonzero slack →
    /// no zero-valued singleton), so warm-start engages where it previously
    /// cold-fell-back, changing the *sequence* of node solves within a time
    /// budget (node_count drifts on ~7/66 vendored instances — all sound: each
    /// node's LP optimum is bit-identical, only the degenerate dual vector, hence
    /// the safe bound value, may differ, always ≤ the true optimum). So it ships
    /// OFF and is opted into only by the branch-and-cut kernel path (which needs
    /// warm-start to succeed), pending the CLAUDE.md §5 graduation panel.
    pub expel_zero_artificials: bool,
    /// Bank the warm dual loop's progress when [`deadline`](Self::deadline) cuts
    /// it short (#928). When `true`, a deadline exit from the dual pivot loop
    /// returns [`LpStatus::IterLimit`] carrying the current (dual-feasible)
    /// basis's row-dual candidate `y = B⁻ᵀc_B` — whose Neumaier–Shcherbina
    /// evaluation is the monotone best-so-far dual objective — instead of falling
    /// back to a cold primal whose spent budget yields a near-useless initial
    /// basis dual. It also enables handing the last exact-refresh duals to the
    /// cold fallback when the loop breaks down and the fallback is then itself
    /// cut — a recovery that exists only to keep that anytime floor alive.
    ///
    /// Default `false`: the MILP B&B driver also sets `deadline` on the default
    /// route, and this changes the pivot path (same audited optimum, possibly a
    /// different degenerate vertex), which is *bound-changing* under the §5
    /// regime. Only the pure-LP warm binding (`solve_lp_warm_csc_py`) sets it,
    /// and only when its caller passed a `time_limit` — i.e. exactly the
    /// `DISCOPT_LP_WARM_DEADLINE` path whose graduation panel judges it.
    pub bank_deadline_duals: bool,
    /// On a near-zero (unstable) dual pivot, refactorize + exact-recompute in
    /// place and re-select instead of abandoning the warm loop for the cold
    /// primal (#1008 R1).
    ///
    /// This used to be gated on [`bank_deadline_duals`](Self::bank_deadline_duals),
    /// which `lp_bindings` sets to `deadline.is_some()`. Nothing about the
    /// recovery is deadline-related: a tiny pivot is usually Forrest–Tomlin
    /// update drift, and refactorize + exact recompute is the loop's soundness
    /// anchor already. The coupling meant a caller who passed no `time_limit`
    /// silently got the cold hand-off on the same LP and the same starting basis.
    /// Measured on QPLIB_2170's root relaxation (`DISCOPT_PROFILE=1`, counters
    /// `DualUnstablePivotRecoveries` / `DualUnstablePivotBails`): at
    /// `time_limit=40` the recovery fires once and the LP solves to `optimal 0`
    /// (HiGHS agrees, 81 pivots); at `time_limit=None` the bail fires once and
    /// the caller gets nothing. One pivot, and the difference is whether a
    /// deadline was supplied.
    ///
    /// Default `false` and opted into by `DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY=1`:
    /// re-selecting after a refactorize can land on a different degenerate
    /// vertex, so this is bound-changing under §5 and needs a corpus panel that
    /// is both cert-clean and net-positive before it graduates. The deadline
    /// path's behavior is unchanged — it keeps the recovery it already had.
    pub recover_unstable_pivot: bool,
    /// Consecutive degenerate dual pivots after which a **stalled** warm re-solve
    /// is abandoned for the cold solve; `0` disables the bail (#1013).
    ///
    /// A degenerate dual pivot raises no dual objective, so a long run of them is
    /// a run with no progress — and the two escapes that were supposed to catch
    /// that cannot: Bland's rule engages at `2·(n+1)` consecutive degenerate
    /// pivots (tens of thousands on a lifted relaxation — measured
    /// `DualBlandActivations` = 0 on all 100 LPs of the in-repo relaxation panel)
    /// and the F2 stall guard only trips at the size-derived pivot cap
    /// (`DualStallTrips` = 0 on all 100, including cells that are 98% degenerate).
    ///
    /// The action on trip is the one every other difficulty in the dual loop
    /// already takes — return to the caller's cold two-phase primal, which
    /// self-verifies its own verdict.
    ///
    /// That was originally documented as "cannot make a solve wrong, only slower
    /// or faster". **That is withdrawn (#1008).** It assumes the cold solve
    /// finishes, which the #1013 panel's three bailing cells all happened to do.
    /// On an LP where it does not, the bail trades a certified optimum for no
    /// bound — and on the captured QPLIB_2170 relaxation the cold path does not
    /// merely refuse, it returns `Unbounded` against a true optimum of 0. The flag
    /// is therefore **default-OFF**; `DISCOPT_LP_DUAL_STALL_BAIL=1` opts in. See
    /// `dual::STALL_PATIENCE` for how the patience was derived and
    /// `dual_stall_bail_can_cost_a_bound_when_the_cold_solve_fails` for the case.
    pub dual_stall_patience: usize,
    /// Relative size of the **cost perturbation** applied to a warm dual start;
    /// `0.0` disables it (#1013).
    ///
    /// The stall this addresses is *dual* degeneracy: the per-pivot trace of the
    /// worst panel cell shows the chosen pivot magnitude at a median of exactly
    /// 1.0 and a primal step that is never zero, while `d_q ≈ 0` — a sparse
    /// objective on a lifted relaxation ties the reduced costs at zero, so the
    /// ratio test takes a zero-length *dual* step however it breaks the tie. That
    /// is why both tie-break arms (a dual Harris pass scoped to the stall, and
    /// Bland at a reachable threshold) were falsified: neither has anything to
    /// discriminate. Perturbing the costs removes the ties themselves, which is
    /// the classical remedy (Koberstein 2005; Huangfu & Hall 2015).
    ///
    /// Only **nonbasic** costs move, each in the sign that preserves the start's
    /// dual feasibility (`AT_LOWER` up, `AT_UPPER` down), so `y = B⁻ᵀc_B` and the
    /// basic reduced costs are untouched and the start stays dual feasible by
    /// construction. Nonbasic *free* columns are skipped — they must price to
    /// exactly zero, and perturbing one makes the start dual infeasible outright.
    ///
    /// The answer the caller receives is always computed on the **true** costs:
    /// the perturbed solve only supplies a starting basis, from which a clean-up
    /// re-solve on the unperturbed `c` produces the returned point, objective,
    /// basis and duals. Anything other than an `Optimal` clean-up is discarded and
    /// the ordinary unperturbed path runs, so the perturbation can change how many
    /// pivots a solve takes but never what it certifies.
    ///
    /// **Default-ON** (1e-5), graduated on the CLAUDE.md §5 panel:
    /// 100 in-repo relaxation LPs x 3 reps, arms interleaved — 0 status
    /// regressions, max relative objective drift 4.9e-11, no objective in the
    /// unsound direction, 77/100 cells bit-identical, total pivots 62 098 →
    /// 35 542 (−42.8 %) and total wall 85.16 s → 70.32 s (0.826x); plus two
    /// tree-level differential panels (16 MINLPLib instances with recorded
    /// optima, bit-identical objective/bound/node count; 9 QPLIB instances, no
    /// unsound bound and no status or certification regression). On the captured
    /// `QPLIB_2170` relaxation the pivot count falls ~25x and its spread over the
    /// refactorization cadence collapses from 6 075 pivots to 416.
    ///
    /// `DISCOPT_LP_DUAL_COST_PERTURB=0` opts out and restores the previous path
    /// exactly; a float sets a specific size.
    pub dual_cost_perturb: f64,
    /// Start a COLD spatial-kernel node LP from the sign-matched dual-feasible
    /// slack basis instead of the cold two-phase primal.
    ///
    /// The kernel's node LP (`bnb::spatial_kernel::solve_spatial_node`) has always
    /// gone straight to `solve_lp_cols_scaled`, i.e. the cold PRIMAL loop — the
    /// one whose own comment reads "a dense, degenerate lifted-McCormick LP can
    /// otherwise grind toward `max_iter` and run uninterruptibly for minutes". It
    /// does exactly that on relaxations rich in linear *equalities*: an equality
    /// reaches the LP layer as two opposing `<=` rows, both tight at every
    /// feasible point, so the LP is massively primal-degenerate. Measured on the
    /// constraint-factor-RLT root relaxation of a continuous nonconvex QP
    /// (QPLIB_1157, 3937 rows): the cold primal exhausted `max_iter` after >150 s
    /// and returned no bound; the dual start returns the same optimum (to 5e-13
    /// against HiGHS) in ~6 s.
    ///
    /// With slacks basic and each structural column nonbasic at the bound its
    /// objective sign selects, `y = B⁻ᵀc_B = 0` and every reduced cost is `c_j`,
    /// so the basis is dual-feasible whenever each selected side is finite.
    /// `PreparedDual::prepare` re-verifies that precondition independently and
    /// falls back to the cold path when it fails, so an accepted basis can never
    /// make the engine converge wrong — this only changes *which* algorithm
    /// reaches the same LP optimum.
    ///
    /// Default `false`: a different optimal basis on a degenerate LP means a
    /// different (still rigorous) safe bound and can move downstream branching,
    /// which is bound-changing under the §5 regime. Set from Python by
    /// `DISCOPT_LP_COLD_DUAL_START`, pending its graduation panel.
    pub cold_dual_start: bool,
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
            expel_zero_artificials: false,
            bank_deadline_duals: false,
            recover_unstable_pivot: false,
            dual_stall_patience: dual::stall_patience_default(),
            dual_cost_perturb: dual::cost_perturb_default(),
            cold_dual_start: false,
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
