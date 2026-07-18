//! Basis factorization behind the revised simplex.
//!
//! The simplex only needs four operations on the current basis matrix `B`
//! (the `m × m` submatrix of basic columns): factorize it, solve `B x = rhs`
//! (`ftran`), solve `Bᵀ y = rhs` (`btran`), and replace one basic column with a
//! product-form update after a pivot. [`LinearSolver`] abstracts these so the
//! simplex is independent of the backend.
//!
//! - [`FeralLU`] — the production backend, wrapping [`feral`]'s sparse
//!   unsymmetric LU (`PLUQ` with `ftran`/`btran` and simplex-style product-form
//!   column updates). This is what makes warm re-solves cheap.
//! - [`DenseLU`] — a dependency-light oracle/fallback built on the crate's
//!   existing dense Gaussian elimination ([`crate::lp::gomory::solve_dense`]);
//!   it refactorizes on every solve, so it is for bring-up, tiny bases, and
//!   cross-checking `FeralLU` in tests — not performance.

use super::refine::residual_matvec_dd;
use feral::{
    should_use_dense_lu, DenseLu, GeneralMatrix, LuParams, LuSingularAction, RefactorCause,
    SparseColMatrix, SparseLu, SparseLuSymbolic,
};

/// Error from a basis factorization/solve.
#[derive(Debug, Clone)]
pub enum LinError {
    /// The basis is singular (or numerically so).
    Singular,
    /// No factorization has been computed yet.
    NotFactorized,
    /// Backend error (message).
    Backend(String),
}

/// Operations the revised simplex needs from a basis factorization of `B`.
pub trait LinearSolver {
    /// Factorize the `m × m` basis whose columns are `cols[0..m]` (each an
    /// `m`-vector of that basic column's dense entries).
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError>;

    /// Factorize the `m × m` basis from its **sparse** columns: `cols[slot]` lists
    /// the `(row, value)` nonzeros of basis slot `slot` (`0..m`). This avoids
    /// materializing the dense `m × m` basis (the O(m²) column build + nnz scan +
    /// dense→sparse conversion that dominated refactorization of row-heavy
    /// lifted-McCormick bases — see feral#87 / discopt#229/#268). The matrix built
    /// here is exactly the one [`factorize`](Self::factorize) would build from the
    /// same columns, so the factorization — and every downstream pivot — is
    /// bit-identical.
    ///
    /// The default scatters to dense and delegates to [`factorize`](Self::factorize)
    /// so backends without a native sparse path keep working; [`FeralLU`] overrides
    /// it to build feral's sparse matrix from the sparse columns directly.
    fn factorize_sparse(&mut self, m: usize, cols: &[Vec<(usize, f64)>]) -> Result<(), LinError> {
        let mut dense = vec![vec![0.0; m]; m];
        for (slot, col) in cols.iter().enumerate() {
            for &(row, v) in col {
                dense[slot][row] = v;
            }
        }
        self.factorize(m, &dense)
    }
    /// Solve `B x = rhs` in place.
    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError>;
    /// Solve `Bᵀ y = rhs` in place.
    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError>;
    /// Replace the basic column in slot `leaving_slot` with `entering_col`
    /// (product-form update; the caller refactorizes when updates accumulate).
    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError>;

    /// Solve `B x = rhs` in place, with iterative refinement when the backend
    /// supports it and it is enabled. The default is a plain [`ftran`](Self::ftran)
    /// — refinement is opt-in and backend-specific (see [`FeralLU::with_refine_steps`]),
    /// so a backend that does not implement it (or has it disabled) degrades to the
    /// unrefined solve with no behavior change.
    fn ftran_refined(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        self.ftran(rhs)
    }

    /// Solve `Bᵀ y = rhs` in place, with iterative refinement when available
    /// (see [`ftran_refined`](Self::ftran_refined)). Defaults to plain
    /// [`btran`](Self::btran).
    fn btran_refined(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        self.btran(rhs)
    }

    /// Solve `B X = RHS` for several right-hand sides at once, each `rhs[k]` an
    /// `m`-vector solved in place. The point is to reuse the single existing
    /// factorization across all `k` solves (sensitivity / strong-branching probes
    /// / multi-rhs LPs) instead of refactorizing per system. The default loops
    /// over [`ftran`](Self::ftran); a backend with a true matrix solve may
    /// override. On the first failing column it stops and returns the error
    /// (later columns are left untouched).
    fn ftran_multi(&mut self, rhs: &mut [&mut [f64]]) -> Result<(), LinError> {
        for col in rhs.iter_mut() {
            self.ftran(col)?;
        }
        Ok(())
    }

    /// Solve `Bᵀ Y = RHS` for several right-hand sides at once (see
    /// [`ftran_multi`](Self::ftran_multi)).
    fn btran_multi(&mut self, rhs: &mut [&mut [f64]]) -> Result<(), LinError> {
        for col in rhs.iter_mut() {
            self.btran(col)?;
        }
        Ok(())
    }
}

/// Production backend: feral's unsymmetric LU, routed dense-or-sparse per basis.
///
/// B&B node bases are small and often dense (design / balance / big-M rows).
/// feral's *dense* LU factorizes in O(m³) with tiny constants and **no**
/// symbolic-analysis phase, while the sparse path re-runs
/// `SparseLuSymbolic::analyze` on every refactorization — measured at
/// ~440 µs/iteration (≈100× too slow) on a ~130-row dense basis, with healthy
/// iteration counts (so the cost was the LU, not pricing/degeneracy). We
/// therefore route small bases (and feral-judged dense ones) to `DenseLu`, and
/// leave genuinely large/sparse bases on `SparseLu`.
///
/// `Clone` duplicates the factorization itself (both feral LUs are `Clone`), so
/// a prepared basis factorization can be cloned and re-optimized from several
/// times — the dual warm-start reuses one factorization across a node's
/// strong-branching probes instead of refactorizing the identical basis per probe.
#[derive(Debug, Clone)]
enum Factored {
    // Both boxed: feral v0.11.2 grew `SparseLu` to ~736 B (vs `DenseLu` ~424 B),
    // so an unboxed enum trips clippy's `large_enum_variant` on the size gap (and
    // reserves the larger size for every `Factored`). Boxing both keeps the enum
    // to two pointers; method calls in ftran/btran/update auto-deref through the
    // boxes, so only the construction sites add `Box::new`.
    Sparse(Box<SparseLu>),
    Dense(Box<DenseLu>),
}

/// Force-dense cutoff: at or below this `m`, a dense LU is always at least as
/// fast as sparse (the O(m³) factor is cheap and there is no symbolic phase),
/// regardless of density — so route there even when feral's density test, which
/// also gates on sparsity, would not. Above it we defer to feral's heuristic.
const FORCE_DENSE_M: usize = 256;

/// Tiny-basis cutoff (mirrors feral's `M_TINY`): at or below this `m` a dense LU
/// always wins and the density-aware route never diverts to sparse.
const M_TINY: usize = 16;

/// Choose the dense vs sparse LU route for an `m`×`m` basis with `nnz` stored
/// entries — the discopt-side policy on top of feral's `should_use_dense_lu`.
///
/// **Default (OFF):** the historical policy — force dense for every `m ≤
/// FORCE_DENSE_M`, else defer to feral's `should_use_dense_lu` (which itself is
/// dense for `m ≤ 16`, dense if `m ≤ dense_threshold` and ≥ 25% dense, else
/// sparse). Byte-identical to prior behavior.
///
/// **Density-aware route (`DISCOPT_LU_DENSITY_ROUTE=1`):** for the band
/// `M_TINY < m ≤ FORCE_DENSE_M`, consult the *same* density gate feral uses
/// (`nnz·4 ≥ m·m`, i.e. ≥ 25% dense) instead of forcing dense. Sparse-but-large
/// bases (`nnz·4 < m·m`) then take feral's sparse LU. Motivation (#557): the
/// wide lifted-McCormick node bases land at `m ≈ 200–256` but are only ~4% dense;
/// forcing them dense pays an O(m³) refactor **and** triggers the dense FT
/// update's `TinyPivot` refactor-every-pivot storm, whereas the sparse LU factors
/// them ~13× faster and its FT update commits (measured on st_e36: a captured
/// m=225 4.4%-dense node LP goes 62 ms → 4.6 ms, `RefacFtTinyPivot` 201/solve →
/// ~0). Truly-dense small bases and `m ≤ M_TINY` still route dense (the density
/// gate keeps them there), so the ~440 µs/iter sparse-symbolic overhead the
/// dense cutoff was built to avoid is not paid on the bases that need dense.
fn want_dense_route(m: usize, nnz: usize, params: &LuParams) -> bool {
    route_dense_decision(m, nnz, params, density_route_enabled())
}

/// Pure routing decision (env flag passed in, so it is deterministically
/// testable). See [`want_dense_route`] for the policy rationale.
fn route_dense_decision(m: usize, nnz: usize, params: &LuParams, density_route: bool) -> bool {
    // Off: preserve the exact historical routing.
    if !density_route {
        return m <= FORCE_DENSE_M || should_use_dense_lu(m, nnz, params);
    }
    // Density-aware: tiny always dense; the (M_TINY, FORCE_DENSE_M] band defers to
    // the density gate; above FORCE_DENSE_M feral's heuristic decides (unchanged).
    if m <= M_TINY {
        return true;
    }
    if m <= FORCE_DENSE_M {
        // feral's exact density criterion: dense iff at least a quarter dense.
        // Saturating so a huge m can't overflow m·m / nnz·4.
        return nnz.saturating_mul(4) >= m.saturating_mul(m);
    }
    should_use_dense_lu(m, nnz, params)
}

/// Whether the `DISCOPT_LU_DENSITY_ROUTE` density-aware LU route (#557) is on
/// for new factorizations: the env flag is not explicitly disabled AND no
/// dense-retry suppression is in effect (#85 — a failed solve is re-run once
/// with the route suppressed; see [`with_density_route_suppressed`]).
///
/// Default ON. Graduated per T2.6 with 3 consecutive green held-out verdicts:
/// BR-3 #602 (verdict 1), FLAG-GRAD #612 (verdict 2), and the P0 SPATIAL-CERT
/// re-run (`docs/dev/p0-spatial-cert-2026-07-10.md`, verdict 3 — incorrect 0,
/// oracle-cross 0, cert-loss 0; the nvs22 loss that broke the first verdict-3
/// attempt was a spatial-driver certification-accounting bug, fixed there).
/// Set `DISCOPT_LU_DENSITY_ROUTE=0` to restore the historical dense-preferring
/// routing byte-identically.
fn density_route_enabled() -> bool {
    !route_suppressed()
        && std::env::var("DISCOPT_LU_DENSITY_ROUTE")
            .map(|v| v != "0")
            .unwrap_or(true)
}

std::thread_local! {
    /// #85 failure-triggered dense retry: while `true`, [`want_dense_route`]
    /// ignores the `DISCOPT_LU_DENSITY_ROUTE` flag and every factorization takes
    /// the historical dense-preferring policy. Set only for the duration of a
    /// retry re-solve (see `dense_retry_wanted` in `primal.rs`); thread-local
    /// because the parallel B&B feature solves independent node LPs on rayon
    /// workers.
    static DENSITY_ROUTE_SUPPRESSED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Whether the density-route suppression (a dense retry in progress) is active
/// on this thread.
fn route_suppressed() -> bool {
    DENSITY_ROUTE_SUPPRESSED.with(|s| s.get())
}

std::thread_local! {
    /// #671 factorization hardening: while `true`, a solve on this thread builds its
    /// basis factor with [`FeralLU::with_singular_perturb`] so a near-singular basis
    /// completes (`PerturbToEps`) instead of aborting, and the refined solves recover
    /// accuracy in double-double. Set only for the duration of a *failure-triggered*
    /// hardened retry (see `hardening_retry` in `primal.rs`/`dual.rs`); thread-local
    /// for the rayon-parallel B&B. Default off — no solve is affected unless a prior
    /// solve numerically failed AND the flag is enabled.
    static FACTORIZATION_HARDENING_ACTIVE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// The `PerturbToEps` floor used by the hardened retry: below `zero_pivot_tol`
/// (1e-13) so genuine small pivots are preserved, above 0 so exactly-singular /
/// sub-tolerance pivots are floored and the factorization completes (the balance
/// mapped by the `regularized_lu.rs` boundary experiment).
pub(crate) const HARDENING_ABS_FLOOR: f64 = 1e-14;

/// Whether the `DISCOPT_LP_FACTORIZATION_HARDENING` flag (default OFF) is enabled
/// *and* no hardened retry is already in progress on this thread (a retry that
/// fails again must fall through to the existing fallback chain, never recurse).
pub(crate) fn hardening_retry_available() -> bool {
    !FACTORIZATION_HARDENING_ACTIVE.with(|s| s.get())
        && std::env::var("DISCOPT_LP_FACTORIZATION_HARDENING")
            .map(|v| v != "0")
            .unwrap_or(false)
}

/// Whether the hardened factorization mode is active on this thread (a hardened
/// retry is in progress). Read at [`FeralLU`] construction in the simplex.
pub(crate) fn hardening_active() -> bool {
    FACTORIZATION_HARDENING_ACTIVE.with(|s| s.get())
}

/// A basis solver for a node solve, hardened iff a hardened retry is in progress:
/// [`FeralLU::with_singular_perturb`] under [`hardening_active`], else the plain
/// `FeralLU::new()` (byte-identical to today). The single construction helper the
/// simplex uses so the mode is honored uniformly.
pub(crate) fn node_feral_lu() -> FeralLU {
    if hardening_active() {
        FeralLU::new().with_singular_perturb(HARDENING_ABS_FLOOR)
    } else {
        FeralLU::new()
    }
}

/// Run `f` with factorization hardening active on this thread (a hardened retry).
/// Restores the previous state on exit, including on unwind.
pub(crate) fn with_hardening_active<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FACTORIZATION_HARDENING_ACTIVE.with(|s| s.set(self.0));
        }
    }
    let _guard = Restore(FACTORIZATION_HARDENING_ACTIVE.with(|s| s.replace(true)));
    f()
}

/// #85: whether a *failed* LP solve may be retried on the dense route — true iff
/// the density route is currently active (so the failure may be
/// sparse-route-specific) and we are not already inside a retry (a retry that
/// fails again must fall through to the existing fallback chain, never recurse;
/// [`density_route_enabled`] is false under suppression, which provides exactly
/// that guarantee).
pub(crate) fn density_route_retry_available() -> bool {
    density_route_enabled()
}

/// Run `f` with the density-aware LU route suppressed: every factorization under
/// `f` takes the historical dense-preferring routing (the robust path the
/// default configuration uses). Restores the previous suppression state on exit,
/// including on unwind (guard-based), so a panicking solve cannot leave the
/// thread stuck in suppressed mode.
pub(crate) fn with_density_route_suppressed<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSITY_ROUTE_SUPPRESSED.with(|s| s.set(self.0));
        }
    }
    let _guard = Restore(DENSITY_ROUTE_SUPPRESSED.with(|s| s.replace(true)));
    f()
}

/// The retained basis matrix, in dense-column form, kept **in sync** with every
/// [`update`](LinearSolver::update) so a refined solve or condition estimate can
/// form the residual `r = b − B·x` against the matrix the factor *currently*
/// represents — not the one it was originally factorized from. Only populated in
/// numeric-focus mode (`refine_steps > 0`); off by default it stays empty, so the
/// hot path pays no retention cost. Dense columns are the canonical form because
/// [`update`](LinearSolver::update) hands us a dense `entering_col`, so keeping it
/// in sync is a trivial `cols[slot] = entering_col` — the sparse/dense feral
/// matrix is rebuilt lazily, only when a refined solve is actually requested.
#[derive(Debug, Clone, Default)]
struct RetainedBasis {
    m: usize,
    cols: Vec<Vec<f64>>,
}

/// A feral basis matrix in whichever form the current factor consumes.
enum BasisMat {
    Sparse(SparseColMatrix),
    Dense(GeneralMatrix),
}

/// Number of iterative-refinement steps used in numeric-focus mode. feral's own
/// refined solve caps at this many `r = b − B·x; B δ = r; x += δ` rounds and
/// early-exits once `‖r‖/‖b‖ < refine_tol`, so 2 is a cheap, effective default
/// (matches feral's internal refined-path default and the hand-rolled 3-round
/// loop the root Gomory solve already used).
const NUMERIC_FOCUS_REFINE_STEPS: usize = 2;

/// Basis factorization backend: feral's LU, dispatched dense-or-sparse per
/// basis at factorize time (see the [`Factored`] docs for the rationale).
#[derive(Default, Clone)]
pub struct FeralLU {
    lu: Option<Factored>,
    /// Iterative-refinement steps baked into the factor's [`LuParams`] at
    /// factorize time. `0` (default) → refinement is a no-op and no basis is
    /// retained, i.e. byte-for-byte the pre-numeric-focus behavior.
    refine_steps: usize,
    /// The in-sync basis, populated iff refinement or singular-perturb mode is on
    /// (see [`RetainedBasis`]).
    retained: Option<RetainedBasis>,
    /// Singular-pivot perturbation floor (issue #671 factorization hardening). When
    /// `Some(abs_floor)`, the factorization uses `LuSingularAction::PerturbToEps`
    /// so a near-singular basis (the hda-class ill-conditioned relaxations) yields
    /// a *completed* factor of a nearby matrix `B'` instead of aborting
    /// (`SingularBasis` → `Numerical`), and the refined solves recover accuracy
    /// against the true `B` in **double-double** precision (feral's own
    /// double-precision refinement was falsified on this class, #671). `None`
    /// (default) → strict `Fail`, byte-identical to today.
    singular_perturb: Option<f64>,
}

impl FeralLU {
    /// A new, unfactorized solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable numeric-focus mode with the default refinement depth: refined
    /// [`ftran`](LinearSolver::ftran_refined)/[`btran`](LinearSolver::btran_refined)
    /// solves and [`condition_estimate`](Self::condition_estimate)/[`growth`](Self::growth)
    /// signals become available. The in-engine analogue of Gurobi's `NumericFocus`
    /// — the principled alternative to falling back to a different solver on an
    /// ill-conditioned basis (discopt#364).
    pub fn with_numeric_focus(self) -> Self {
        self.with_refine_steps(NUMERIC_FOCUS_REFINE_STEPS)
    }

    /// Set the iterative-refinement depth (0 disables; see [`with_numeric_focus`]).
    /// Takes effect at the next factorization (the depth is baked into the
    /// factor's [`LuParams`]). A positive depth turns on basis retention.
    pub fn with_refine_steps(mut self, steps: usize) -> Self {
        self.refine_steps = steps;
        self
    }

    /// Enable **factorization hardening** for near-singular bases (issue #671): the
    /// factorization uses `LuSingularAction::PerturbToEps { abs_floor }` so a
    /// singular / near-singular pivot is floored to `abs_floor` and the factor
    /// *completes* (a nearby matrix `B'`) instead of aborting, and
    /// [`ftran_refined`](LinearSolver::ftran_refined) /
    /// [`btran_refined`](LinearSolver::btran_refined) run **double-double**
    /// iterative refinement against the true retained `B` to recover accuracy.
    ///
    /// `abs_floor` must sit **below** the genuine small pivots of the class (else
    /// the perturbed factor discards a direction the solution needs — see the entry
    /// experiment in `regularized_lu.rs`); a small value like `1e-12` floors the
    /// exactly-zero / sub-`zero_pivot_tol` pivots while preserving genuine ones.
    /// Turns on basis retention (needed for the true-`B` residual). Failure-
    /// triggered by design: callers escalate to this only when the strict factor
    /// aborted, so with it unused every solve is byte-identical to today.
    pub fn with_singular_perturb(mut self, abs_floor: f64) -> Self {
        self.singular_perturb = Some(abs_floor);
        self
    }

    /// Whether basis retention (and hence a refined solve) is active: a positive
    /// refine depth **or** singular-perturb hardening.
    fn refine_enabled(&self) -> bool {
        self.refine_steps > 0 || self.singular_perturb.is_some()
    }

    /// The [`LuParams`] for this solver's factorizations, carrying the configured
    /// refinement depth and, when hardening is on, the `PerturbToEps` singular
    /// action. All other fields are feral's defaults (strict partial pivoting,
    /// `zero_pivot_tol = 1e-13`, etc.).
    fn params(&self) -> LuParams {
        LuParams {
            refine_steps: self.refine_steps,
            on_singular: match self.singular_perturb {
                Some(abs_floor) => LuSingularAction::PerturbToEps { abs_floor },
                None => LuSingularAction::Fail,
            },
            ..LuParams::default()
        }
    }

    /// Rebuild the retained basis as the feral matrix the current factor consumes
    /// (sparse for a `SparseLu`, dense for a `DenseLu`). Returns `None` when not in
    /// numeric-focus mode / unfactorized; propagates a construction error otherwise.
    fn build_basis_matrix(&self) -> Option<Result<BasisMat, LinError>> {
        let rb = self.retained.as_ref()?;
        Some(match self.lu.as_ref()? {
            Factored::Sparse(_) => SparseColMatrix::from_dense_columns(rb.m, &rb.cols)
                .map(BasisMat::Sparse)
                .map_err(feral_err),
            Factored::Dense(_) => GeneralMatrix::from_columns(rb.m, &rb.cols)
                .map(BasisMat::Dense)
                .map_err(feral_err),
        })
    }

    /// One-norm condition estimate `κ₁ ≈ ‖B‖₁·‖B⁻¹‖₁` of the current basis
    /// (Hager–Higham, via feral#94). `None` unless in numeric-focus mode and
    /// factorized. A large value flags an ill-conditioned node — the signal that
    /// drives in-engine recovery (refine / perturb / branch) instead of a solver
    /// swap (discopt#364).
    pub fn condition_estimate(&mut self) -> Option<f64> {
        let mat = self.build_basis_matrix()?.ok()?;
        match (self.lu.as_mut()?, mat) {
            (Factored::Sparse(lu), BasisMat::Sparse(b)) => lu.condition_estimate_1(&b).ok(),
            (Factored::Dense(lu), BasisMat::Dense(b)) => lu.condition_estimate_1(&b).ok(),
            // build_basis_matrix builds to match the live factor kind, so the
            // cross arms are unreachable; degrade to None rather than panic.
            _ => None,
        }
    }

    /// The factor's element-growth high-water ratio `‖U‖∞ / ‖U₀‖∞` (feral#93), or
    /// `None` when unfactorized. A cheap conditioning proxy: growth ≫ 1 signals a
    /// factorization that lost digits. Available regardless of numeric-focus mode
    /// (it is read straight off the factor, needing no retained basis).
    pub fn growth(&self) -> Option<f64> {
        match &self.lu {
            Some(Factored::Sparse(lu)) => Some(lu.growth()),
            Some(Factored::Dense(lu)) => Some(lu.growth()),
            None => None,
        }
    }

    /// Why the last [`update`](LinearSolver::update) demanded a refactorization,
    /// with a cause-specific magnitude (feral#95): `Growth`/`TinyPivot` mean
    /// *ill-conditioning* (refine-and-retry is the right response), `UpdateBudget`
    /// is mere bookkeeping (a plain refactor suffices). `None` if the last update
    /// succeeded or the solver is unfactorized.
    pub fn last_refactor(&self) -> Option<(RefactorCause, f64)> {
        match &self.lu {
            Some(Factored::Sparse(lu)) => lu.last_refactor(),
            Some(Factored::Dense(lu)) => lu.last_refactor(),
            None => None,
        }
    }

    /// Cumulative Forrest–Tomlin bump-update work (feral's `eta_ops`) accumulated
    /// on the current factorization since it was built. `0` for the dense backend
    /// (its updates are not product-form etas) and when unfactorized. The simplex
    /// uses this to refactorize *before* wide-bump updates compound into the
    /// O(bump²) FT blowup that dominates row-heavy McCormick bases
    /// (discopt#268 / feral#87).
    pub fn ft_update_work(&self) -> usize {
        match &self.lu {
            Some(Factored::Sparse(lu)) => lu.eta_ops(),
            _ => 0,
        }
    }

    /// nnz of the current sparse factor (`0` for the dense backend / unfactorized).
    /// The work budget the FT-update accumulation is compared against.
    pub fn factor_nnz(&self) -> usize {
        match &self.lu {
            Some(Factored::Sparse(lu)) => lu.factor_nnz(),
            _ => 0,
        }
    }

    /// Double-double iterative refinement of `B x = rhs` (`transpose=false`) or
    /// `Bᵀ x = rhs` (`transpose=true`) using the retained *true* basis `B` and the
    /// current factor as the correction solver (issue #671 hardening). The residual
    /// `rhs − B x` is accumulated in ≈106-bit double-double
    /// ([`super::refine::residual_dd`]) — that high precision is what recovers
    /// accuracy past a `PerturbToEps` factor's error (feral's own double-precision
    /// refinement could not, #671). Used only when a retained basis is present;
    /// falls back to the plain solve otherwise.
    fn dd_refined(&mut self, rhs: &mut [f64], transpose: bool) -> Result<(), LinError> {
        // A `PerturbToEps` factor of a near-singular basis has κ ≈ 1/abs_floor, so
        // the Wilkinson refinement gains ~`-log10(κ·u)` digits/step; a dozen steps
        // with an early exit is ample and inert on well-conditioned solves (one
        // correction, residual ≤ TOL, break). Failure-triggered path.
        const MAX_STEPS: usize = 12;
        const TOL: f64 = 1e-13;
        let (_m, cols) = match self.retained.as_ref() {
            Some(rb) => (rb.m, rb.cols.clone()),
            None => {
                return if transpose {
                    self.btran_raw(rhs)
                } else {
                    self.ftran_raw(rhs)
                }
            }
        };
        let target = rhs.to_vec();
        let mut x = vec![0.0f64; cols.len()];
        for _ in 0..MAX_STEPS {
            // High-precision residual `target − (B or Bᵀ)·x`, sparse-aware O(nnz).
            let mut r = residual_matvec_dd(&cols, &x, &target, transpose);
            let resnorm = r.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            if resnorm <= TOL {
                break;
            }
            // Correction through the (perturbed) factor: solve `B' dr = r`.
            if transpose {
                self.btran_raw(&mut r)?;
            } else {
                self.ftran_raw(&mut r)?;
            }
            for (xi, ri) in x.iter_mut().zip(&r) {
                *xi += ri;
            }
        }
        rhs.copy_from_slice(&x);
        Ok(())
    }

    /// Raw `B' x = rhs` solve through the current factor (no refinement).
    fn ftran_raw(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        match self.lu.as_mut().ok_or(LinError::NotFactorized)? {
            Factored::Sparse(lu) => lu.ftran(rhs).map_err(feral_err),
            Factored::Dense(lu) => lu.ftran(rhs).map_err(feral_err),
        }
    }

    /// Raw `B'ᵀ x = rhs` solve through the current factor (no refinement).
    fn btran_raw(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        match self.lu.as_mut().ok_or(LinError::NotFactorized)? {
            Factored::Sparse(lu) => lu.btran(rhs).map_err(feral_err),
            Factored::Dense(lu) => lu.btran(rhs).map_err(feral_err),
        }
    }
}

fn feral_err(e: feral::FeralError) -> LinError {
    LinError::Backend(format!("{e:?}"))
}

impl LinearSolver for FeralLU {
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError> {
        debug_assert_eq!(cols.len(), m);
        let params = self.params();
        let nnz: usize = cols
            .iter()
            .map(|c| c.iter().filter(|&&v| v != 0.0).count())
            .sum();
        self.lu = Some(if want_dense_route(m, nnz, &params) {
            Factored::Dense(Box::new(
                DenseLu::factor(cols, m, params).map_err(feral_err)?,
            ))
        } else {
            let a = SparseColMatrix::from_dense_columns(m, cols).map_err(feral_err)?;
            let sym = SparseLuSymbolic::analyze(&a).map_err(feral_err)?;
            Factored::Sparse(Box::new(
                SparseLu::factor(&a, &sym, params).map_err(feral_err)?,
            ))
        });
        self.retained = self.refine_enabled().then(|| RetainedBasis {
            m,
            cols: cols.to_vec(),
        });
        Ok(())
    }

    fn factorize_sparse(&mut self, m: usize, cols: &[Vec<(usize, f64)>]) -> Result<(), LinError> {
        let params = self.params();
        // nnz is the sum of column lengths — O(m), not the O(m²) dense scan the
        // `Vec<Vec<f64>>` path pays.
        let nnz: usize = cols.iter().map(|c| c.len()).sum();
        // Densify once if either the dense LU branch needs it or numeric-focus
        // retention does; reuse the single dense copy for both.
        let want_dense = want_dense_route(m, nnz, &params);
        let dense = (want_dense || self.refine_enabled()).then(|| {
            let mut d = vec![vec![0.0; m]; m];
            for (slot, col) in cols.iter().enumerate() {
                for &(row, v) in col {
                    d[slot][row] = v;
                }
            }
            d
        });
        self.lu = Some(if want_dense {
            // Small/dense basis: feral's dense LU wins and there is no symbolic
            // phase.
            let dense = dense.as_ref().expect("densified for the dense branch");
            Factored::Dense(Box::new(
                DenseLu::factor(dense, m, params).map_err(feral_err)?,
            ))
        } else {
            // Build feral's sparse matrix directly from the sparse columns —
            // O(nnz), no dense m×m intermediate.
            let a = SparseColMatrix::from_sparse_columns(m, cols).map_err(feral_err)?;
            let sym = SparseLuSymbolic::analyze(&a).map_err(feral_err)?;
            Factored::Sparse(Box::new(
                SparseLu::factor(&a, &sym, params).map_err(feral_err)?,
            ))
        });
        self.retained = dense
            .filter(|_| self.refine_enabled())
            .map(|cols| RetainedBasis { m, cols });
        Ok(())
    }

    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        // #671 hardening keeps the hot-loop solve RAW even in singular-perturb mode:
        // the perturbation only floors an already-tiny pivot (~1e-13 → abs_floor), so
        // `B' ≈ B` and the simplex's pricing/ratio decisions are essentially
        // unchanged — while a per-solve double-double refine would be O(m²) per pivot
        // (the retained basis is dense) and dominate. Accuracy is recovered where it
        // is *certified*: the audit path calls [`ftran_refined`], which runs the
        // double-double refinement (`dd_refined`) against the true `B`.
        self.ftran_raw(rhs)
    }

    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        self.btran_raw(rhs)
    }

    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError> {
        let res = match self.lu.as_mut().ok_or(LinError::NotFactorized)? {
            Factored::Sparse(lu) => lu.update(leaving_slot, entering_col).map_err(feral_err),
            Factored::Dense(lu) => lu.update(leaving_slot, entering_col).map_err(feral_err),
        };
        // Keep the retained basis in lock-step with the factor: only on a
        // successful update (a failed one rolls the factor back and the caller
        // refactorizes, which rebuilds `retained` from scratch). Without this the
        // residual `b − B·x` in a refined solve would be formed against the stale
        // pre-update basis and silently corrupt the correction.
        if res.is_ok() {
            if let Some(rb) = self.retained.as_mut() {
                rb.cols[leaving_slot].clear();
                rb.cols[leaving_slot].extend_from_slice(entering_col);
            }
        }
        res
    }

    fn ftran_refined(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        // #671 hardening: in singular-perturb mode the factor is of a nearby `B'`,
        // so recover accuracy against the true `B` in double-double precision.
        if self.singular_perturb.is_some() {
            return self.dd_refined(rhs, false);
        }
        // Build the basis matrix first (immutable borrow, produces an owned
        // matrix), then solve (mutable borrow) — the two borrows don't overlap.
        let mat = match self.build_basis_matrix() {
            Some(m) => m?,
            None => return self.ftran(rhs), // not in numeric-focus mode
        };
        match (self.lu.as_mut().ok_or(LinError::NotFactorized)?, mat) {
            (Factored::Sparse(lu), BasisMat::Sparse(b)) => {
                lu.ftran_refined(&b, rhs).map_err(feral_err)
            }
            (Factored::Dense(lu), BasisMat::Dense(b)) => {
                lu.ftran_refined(&b, rhs).map_err(feral_err)
            }
            _ => self.ftran(rhs),
        }
    }

    fn btran_refined(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        if self.singular_perturb.is_some() {
            return self.dd_refined(rhs, true);
        }
        let mat = match self.build_basis_matrix() {
            Some(m) => m?,
            None => return self.btran(rhs),
        };
        match (self.lu.as_mut().ok_or(LinError::NotFactorized)?, mat) {
            (Factored::Sparse(lu), BasisMat::Sparse(b)) => {
                lu.btran_refined(&b, rhs).map_err(feral_err)
            }
            (Factored::Dense(lu), BasisMat::Dense(b)) => {
                lu.btran_refined(&b, rhs).map_err(feral_err)
            }
            _ => self.btran(rhs),
        }
    }
}

/// Dense oracle/fallback: stores the basis columns and refactorizes on every
/// solve via [`crate::lp::gomory::solve_dense`]. For bring-up / tiny bases /
/// cross-checking `FeralLU`, not performance.
#[derive(Default)]
pub struct DenseLU {
    m: usize,
    cols: Vec<Vec<f64>>,
}

impl DenseLU {
    /// A new, unfactorized solver.
    pub fn new() -> Self {
        Self {
            m: 0,
            cols: Vec::new(),
        }
    }
}

impl LinearSolver for DenseLU {
    fn factorize(&mut self, m: usize, cols: &[Vec<f64>]) -> Result<(), LinError> {
        self.m = m;
        self.cols = cols.to_vec();
        Ok(())
    }

    fn ftran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        // B x = rhs, with B[i][j] = cols[j][i] (column-major basis → row-major B).
        let m = self.m;
        let mut mat = vec![0.0_f64; m * m];
        for (j, col) in self.cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                mat[i * m + j] = v;
            }
        }
        let x = crate::lp::gomory::solve_dense(&mat, m, rhs, 1e-12).ok_or(LinError::Singular)?;
        rhs.copy_from_slice(&x);
        Ok(())
    }

    fn btran(&mut self, rhs: &mut [f64]) -> Result<(), LinError> {
        // Bᵀ y = rhs, with Bᵀ[i][j] = B[j][i] = cols[i][j].
        let m = self.m;
        let mut mat = vec![0.0_f64; m * m];
        for (i, col) in self.cols.iter().enumerate() {
            for (j, &v) in col.iter().enumerate() {
                mat[i * m + j] = v;
            }
        }
        let y = crate::lp::gomory::solve_dense(&mat, m, rhs, 1e-12).ok_or(LinError::Singular)?;
        rhs.copy_from_slice(&y);
        Ok(())
    }

    fn update(&mut self, leaving_slot: usize, entering_col: &[f64]) -> Result<(), LinError> {
        self.cols[leaving_slot] = entering_col.to_vec();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_retry_suppression_scopes_and_restores() {
        // #85: the suppression guard must (a) suppress inside the closure —
        // which also makes density_route_retry_available() false, preventing
        // retry recursion; (b) restore on exit; (c) restore on unwind.
        assert!(!route_suppressed());
        with_density_route_suppressed(|| {
            assert!(route_suppressed());
            // No recursion: inside a retry, a further retry is unavailable even
            // if the env flag were set (density_route_enabled() is false here).
            assert!(!density_route_retry_available());
            // Nesting is harmless and restores to the outer (suppressed) state.
            with_density_route_suppressed(|| assert!(route_suppressed()));
            assert!(route_suppressed());
        });
        assert!(!route_suppressed());
        // Unwind safety: a panicking solve must not leave the thread suppressed.
        let _ = std::panic::catch_unwind(|| {
            with_density_route_suppressed(|| panic!("simulated solve panic"))
        });
        assert!(!route_suppressed());
    }

    #[test]
    fn route_default_off_is_historical() {
        // Default (density_route=false) must reproduce the exact historical policy:
        // dense for every m ≤ FORCE_DENSE_M regardless of density, else feral.
        let p = LuParams::default();
        // Wide, sparse McCormick-like band basis: historically forced dense.
        assert!(route_dense_decision(225, 996 + 225, &p, false));
        // A genuinely large sparse basis: feral routes it sparse either way.
        assert!(!route_dense_decision(1000, 1000 * 4, &p, false));
        // Tiny: dense.
        assert!(route_dense_decision(8, 0, &p, false));
    }

    #[test]
    fn route_density_on_diverts_sparse_band_only() {
        let p = LuParams::default();
        // The #557 target: m=225 at ~4.4% density → now SPARSE (was dense).
        assert!(!route_dense_decision(225, 996 + 225, &p, true));
        // A truly-dense small basis (≥25% dense) in the band → still DENSE.
        let dense_nnz = 100 * 100 / 3; // ~33% dense
        assert!(route_dense_decision(100, dense_nnz, &p, true));
        // Exactly the 25% density boundary (nnz·4 == m·m) → dense (≥ gate).
        assert!(route_dense_decision(100, 100 * 100 / 4, &p, true));
        // Just below 25% → sparse.
        assert!(!route_dense_decision(100, 100 * 100 / 4 - 1, &p, true));
        // Tiny stays dense even at 0 nnz.
        assert!(route_dense_decision(16, 0, &p, true));
        // Above FORCE_DENSE_M: feral heuristic, unchanged by the flag.
        assert_eq!(
            route_dense_decision(300, 300, &p, true),
            should_use_dense_lu(300, 300, &p)
        );
    }

    // Matrix-vector B·x with B[i][j] = cols[j][i].
    fn bmatvec(cols: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let m = cols.len();
        let mut y = vec![0.0; m];
        for (j, col) in cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                y[i] += v * x[j];
            }
        }
        y
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn feral_matches_dense_ftran_btran_and_update() {
        // B columns: [[2,1,0],[1,2,1],[0,1,2]] → SPD-ish nonsingular.
        let cols = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let m = 3;
        let mut fl = FeralLU::new();
        let mut dl = DenseLU::new();
        fl.factorize(m, &cols).unwrap();
        dl.factorize(m, &cols).unwrap();

        // ftran: B x = rhs.
        let rhs = [1.0, 2.0, 3.0];
        let mut xf = rhs;
        let mut xd = rhs;
        fl.ftran(&mut xf).unwrap();
        dl.ftran(&mut xd).unwrap();
        assert!(close(&xf, &xd, 1e-9), "ftran feral {xf:?} vs dense {xd:?}");
        assert!(close(&bmatvec(&cols, &xf), &rhs, 1e-9), "B·x != rhs");

        // btran: Bᵀ y = rhs.
        let mut yf = rhs;
        let mut yd = rhs;
        fl.btran(&mut yf).unwrap();
        dl.btran(&mut yd).unwrap();
        assert!(close(&yf, &yd, 1e-9), "btran feral {yf:?} vs dense {yd:?}");

        // product-form update: replace column 1 with [0,0,1] (keeps B
        // nonsingular — [1,1,1] would make it singular and feral would
        // correctly signal NeedsRefactor), then re-solve ftran.
        let newcol = [0.0, 0.0, 1.0];
        fl.update(1, &newcol).unwrap();
        dl.update(1, &newcol).unwrap();
        let mut xf2 = rhs;
        let mut xd2 = rhs;
        fl.ftran(&mut xf2).unwrap();
        dl.ftran(&mut xd2).unwrap();
        assert!(
            close(&xf2, &xd2, 1e-9),
            "post-update ftran feral {xf2:?} vs dense {xd2:?}"
        );
        let mut updated = cols.clone();
        updated[1] = newcol.to_vec();
        assert!(
            close(&bmatvec(&updated, &xf2), &rhs, 1e-9),
            "post-update B·x != rhs"
        );
    }

    #[test]
    fn multi_rhs_matches_per_column_solves() {
        // ftran_multi / btran_multi over one factorization must reproduce
        // solving each right-hand side individually.
        let cols = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let m = 3;
        let mut fl = FeralLU::new();
        fl.factorize(m, &cols).unwrap();

        let rhs0 = [1.0, 2.0, 3.0];
        let rhs1 = [3.0, 0.0, -1.0];

        // Reference: individual ftran.
        let (mut r0, mut r1) = (rhs0, rhs1);
        fl.ftran(&mut r0).unwrap();
        fl.ftran(&mut r1).unwrap();

        // Batched ftran_multi.
        let (mut b0, mut b1) = (rhs0, rhs1);
        {
            let mut batch: [&mut [f64]; 2] = [&mut b0, &mut b1];
            fl.ftran_multi(&mut batch).unwrap();
        }
        assert!(close(&b0, &r0, 1e-12) && close(&b1, &r1, 1e-12));
        // Each solved system genuinely satisfies B·x = rhs.
        assert!(close(&bmatvec(&cols, &b0), &rhs0, 1e-9));
        assert!(close(&bmatvec(&cols, &b1), &rhs1, 1e-9));

        // btran_multi likewise matches individual btran.
        let (mut y0, mut y1) = (rhs0, rhs1);
        fl.btran(&mut y0).unwrap();
        fl.btran(&mut y1).unwrap();
        let (mut c0, mut c1) = (rhs0, rhs1);
        {
            let mut batch: [&mut [f64]; 2] = [&mut c0, &mut c1];
            fl.btran_multi(&mut batch).unwrap();
        }
        assert!(close(&c0, &y0, 1e-12) && close(&c1, &y1, 1e-12));
    }

    // ---- numeric-focus: iterative refinement + condition/growth signals ----

    // Residual ‖B·x − rhs‖∞ of a candidate solution.
    fn residual_inf(cols: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
        bmatvec(cols, x)
            .iter()
            .zip(rhs)
            .map(|(bx, r)| (bx - r).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn refined_solve_matches_plain_on_well_conditioned() {
        // On a benign basis, refinement must not perturb an already-accurate
        // solution: refined == plain, both exact.
        let cols = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let m = 3;
        let mut fl = FeralLU::new().with_numeric_focus();
        fl.factorize(m, &cols).unwrap();
        let rhs = [1.0, 2.0, 3.0];
        let (mut xr, mut xp) = (rhs, rhs);
        fl.ftran_refined(&mut xr).unwrap();
        fl.ftran(&mut xp).unwrap();
        assert!(close(&xr, &xp, 1e-12), "refined {xr:?} vs plain {xp:?}");
        assert!(residual_inf(&cols, &xr, &rhs) < 1e-9);
    }

    #[test]
    fn refined_solve_valid_after_update_syncs_basis() {
        // The stale-`b` trap: after a product-form update the factor represents
        // the *updated* basis. If the retained basis is not kept in sync, the
        // refined solve forms its residual against the old matrix and corrupts
        // the answer. This asserts the refined solve satisfies the UPDATED system.
        let cols = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let m = 3;
        let mut fl = FeralLU::new().with_numeric_focus();
        fl.factorize(m, &cols).unwrap();

        let newcol = [0.0, 0.0, 1.0]; // keeps B nonsingular
        fl.update(1, &newcol).unwrap();
        let mut updated = cols.clone();
        updated[1] = newcol.to_vec();

        let rhs = [1.0, 2.0, 3.0];
        let mut x = rhs;
        fl.ftran_refined(&mut x).unwrap();
        assert!(
            residual_inf(&updated, &x, &rhs) < 1e-9,
            "refined ftran must satisfy the UPDATED basis; residual too large (stale-b?)"
        );

        // btran against the updated basis transpose likewise.
        let mut y = rhs;
        fl.btran_refined(&mut y).unwrap();
        // Bᵀ y = rhs  ⇔  residual of Bᵀ against y.
        let mut byt = vec![0.0; m];
        for (i, col) in updated.iter().enumerate() {
            byt[i] = col.iter().zip(&y).map(|(a, yj)| a * yj).sum();
        }
        let rt = byt
            .iter()
            .zip(&rhs)
            .map(|(a, r)| (a - r).abs())
            .fold(0.0, f64::max);
        assert!(rt < 1e-9, "refined btran must satisfy updated Bᵀ");
    }

    #[test]
    fn refinement_does_not_worsen_residual_ill_conditioned() {
        // A basis with a large dynamic range across columns. Refinement can only
        // reduce (or leave) the residual, never worsen it — the safety property
        // that makes it a sound in-engine recovery step.
        // Moderate dynamic range: ill-conditioned enough to be a meaningful test,
        // but the smallest pivot (~1e-3) stays well above feral's relative
        // singular floor (zero_pivot_tol · ‖U₀‖∞ ≈ 1e-13 · 1e6), so both paths
        // factorize.
        let cols = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 1e6, 0.0],
            vec![1.0, 1.0, 1e-3],
        ];
        let m = 3;
        let rhs = [1.0, 1.0, 1.0];

        let mut plain = FeralLU::new();
        plain.factorize(m, &cols).unwrap();
        let mut xp = rhs;
        plain.ftran(&mut xp).unwrap();

        let mut nf = FeralLU::new().with_numeric_focus();
        nf.factorize(m, &cols).unwrap();
        let mut xr = rhs;
        nf.ftran_refined(&mut xr).unwrap();

        let rp = residual_inf(&cols, &xp, &rhs);
        let rr = residual_inf(&cols, &xr, &rhs);
        assert!(
            rr <= rp + 1e-18,
            "refined residual {rr:e} must not exceed plain {rp:e}"
        );
    }

    #[test]
    fn condition_estimate_tracks_conditioning() {
        // Identity → κ₁ ≈ 1; a diagonal with a 1e10 dynamic range → κ₁ ≈ 1e10.
        let ident = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut fi = FeralLU::new().with_numeric_focus();
        fi.factorize(2, &ident).unwrap();
        let ki = fi.condition_estimate().expect("numeric-focus + factorized");
        assert!((ki - 1.0).abs() < 1e-6, "κ₁(I) ≈ 1, got {ki}");

        let ill = vec![vec![1.0, 0.0], vec![0.0, 1e-10]];
        let mut fx = FeralLU::new().with_numeric_focus();
        fx.factorize(2, &ill).unwrap();
        let kx = fx.condition_estimate().expect("numeric-focus + factorized");
        assert!(kx > 1e9, "κ₁(diag(1,1e-10)) ≈ 1e10, got {kx}");

        // Off by default: no retained basis, so no estimate.
        let mut plain = FeralLU::new();
        plain.factorize(2, &ident).unwrap();
        assert!(plain.condition_estimate().is_none());
    }

    #[test]
    fn growth_signal_available_after_factorize() {
        let cols = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let mut fl = FeralLU::new();
        assert!(fl.growth().is_none(), "unfactorized → no growth");
        fl.factorize(2, &cols).unwrap();
        let g = fl.growth().expect("factorized");
        assert!(
            g.is_finite() && g >= 1.0,
            "growth is a ≥1 high-water ratio, got {g}"
        );
    }

    #[test]
    fn numeric_focus_off_retains_nothing() {
        // The default path must not pay retention cost or change behavior.
        let cols = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let mut fl = FeralLU::new();
        fl.factorize(2, &cols).unwrap();
        assert!(fl.retained.is_none());
        // ftran_refined with refinement off is exactly ftran.
        let rhs = [1.0, 1.0];
        let (mut a, mut b) = (rhs, rhs);
        fl.ftran_refined(&mut a).unwrap();
        fl.ftran(&mut b).unwrap();
        assert_eq!(a, b);
    }
}
