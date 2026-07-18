//! Bounded-variable two-phase **primal** revised simplex (cold start).
//!
//! Solves `min cᵀx s.t. A x = b, l ≤ x ≤ u` (the crate's standard form via
//! [`LpView`]). Phase 1 adds one artificial per row to form a trivial identity
//! basis and minimizes total artificial value to reach a feasible basis (or
//! certify infeasibility); Phase 2 optimizes the real objective. The basis is
//! factorized by [`FeralLU`] (`ftran`/`btran` + product-form updates).
//!
//! This is the correctness-first cold solver (roadmap P1). It recomputes the
//! basic solution by `ftran` each iteration (clear over fast) and uses Dantzig
//! pricing with a Bland's-rule fallback after a stall to prevent cycling.
//! Warm-start (dual simplex) and performance pricing/factorization come later.
//!
// Several simplex loops index `cost`/`stat`/columns by the same index `j`, so a
// range loop is clearer than zipping multiple slices.
#![allow(clippy::needless_range_loop)]

use super::linsolve::{node_feral_lu, FeralLU, LinearSolver};
use super::scaling::{ScaledLp, Scaling};
use super::sparse::SparseCols;
use super::{LpSolve, LpStatus, SimplexOptions};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};
use crate::lp::crossover::LpView;

const INF: f64 = 1e20;

/// Solve the LP `min cᵀx s.t. A x = b, l ≤ x ≤ u` by two-phase primal simplex.
///
/// Ill-scaled matrices (lifted McCormick LPs whose coefficients span many orders
/// of magnitude) are equilibrated first ([`ScaledLp`]) so the basis
/// factorization stays well-conditioned; the scaled solution is mapped back to
/// the original space. Well-conditioned LPs are solved directly (no copy,
/// bit-identical to the unscaled path).
pub fn solve_lp(lp: &LpView<'_>, b: &[f64], opts: &SimplexOptions) -> LpSolve {
    match ScaledLp::maybe_new(lp, b) {
        Some(scaled) => {
            let view = scaled.view();
            let mut sol = solve_lp_scaled(&view, scaled.b(), opts);
            scaled.unscale_x(&mut sol.x);
            // The certificate vectors are produced in scaled space; map them back
            // so they are consistent with the *original* A/b/bounds the caller
            // verifies against (a scaled dual against an unscaled matrix would make
            // the safe bound / Farkas check spuriously fail — exactly on the
            // ill-scaled LPs where the certificate matters most).
            scaled.unscale_dual(&mut sol.dual);
            scaled.unscale_ray(&mut sol.ray);
            sol
        }
        None => solve_lp_scaled(lp, b, opts),
    }
}

/// #85 failure-triggered dense retry (with the #557 density-aware LU route): is
/// this solve a *failure* that may be sparse-route-specific?
///
/// Entry experiment (docs/dev/performance-plan.md §6, 2026-07-10): with the
/// route ON, a small class of node LPs exits `Numerical`/`IterLimit` on the
/// sparse route where the historical dense-preferring route succeeds (nvs21: 39
/// vs 12 failing solves; the failing node keeps its inherited loose relaxation
/// bound, the final dual bound sticks at that value, and the `optimal`
/// certificate is lost — `feasible`, bound −1.59e7 vs true −5.685). A
/// factorization-time conditioning gate was falsified as the discriminator
/// (inverted populations), so the fix is failure-triggered: detect the failure
/// the solve already reports and re-solve once on the robust path.
fn dense_retry_wanted(sol: &LpSolve) -> bool {
    matches!(sol.status, LpStatus::Numerical | LpStatus::IterLimit)
        && super::linsolve::density_route_retry_available()
}

/// Re-solve a failed LP once, cold, with the density-aware LU route suppressed
/// (every factorization takes the historical dense-preferring policy — the exact
/// robust path the default configuration uses).
///
/// **Soundness invariant:** this only ever *replaces a failure* — a
/// `Numerical`/`IterLimit` exit carries no certificate to preserve — with the
/// robust path's result; a suspect fast-path result is never accepted or
/// blended. A retry that also fails is returned as that failure and falls
/// through to the existing fallback chain exactly as before the retry existed.
/// The retry is **cold**, not warm from the failed run's state: that state
/// (basis, factor, iterate) is precisely what is in doubt after a numerical
/// breakdown (sound-or-refuse). Recursion is impossible: under suppression,
/// `density_route_retry_available()` is false.
fn dense_retry(failed: LpSolve, re_solve: impl FnOnce() -> LpSolve) -> LpSolve {
    debug_assert!(matches!(
        failed.status,
        LpStatus::Numerical | LpStatus::IterLimit
    ));
    drop(failed); // no certificate to keep — the retry result replaces it wholesale
    crate::profile::incr(crate::profile::Ctr::LpDenseRetries);
    let sol = super::linsolve::with_density_route_suppressed(re_solve);
    if !matches!(sol.status, LpStatus::Numerical | LpStatus::IterLimit) {
        crate::profile::incr(crate::profile::Ctr::LpDenseRetryRescues);
    }
    sol
}

/// Two-phase primal simplex on an already-equilibrated (or known well-scaled) LP.
/// The warm dual path's cold fallback and the B&B driver (which equilibrates the
/// working matrix once and shares it across all node solves) call this directly
/// so the matrix is not scaled twice; the caller owns the [`scaling::Scaling`]
/// and unscales the returned `x` itself.
pub fn solve_lp_scaled(lp: &LpView<'_>, b: &[f64], opts: &SimplexOptions) -> LpSolve {
    let sol = Simplex::new(lp, b, opts).run();
    if dense_retry_wanted(&sol) {
        return dense_retry(sol, || Simplex::new(lp, b, opts).run());
    }
    sol
}

/// Cold primal solve from an owned CSC matrix (already scaled by the caller, if at
/// all) instead of a dense [`LpView`] — the sparse-native cold path used by the
/// CSC warm entry and the dual fallback. Never materializes the dense `m×n` matrix.
#[allow(clippy::too_many_arguments)]
pub fn solve_lp_cols(
    cols: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    opts: &SimplexOptions,
) -> LpSolve {
    // #85: `cols` is consumed by the solve, so a possible dense retry needs its
    // own copy up front. O(nnz), paid only with the density route ON (the warm
    // dual path already clones the CSC per solve — see dual.rs — so this is
    // in-family), and nothing extra with the route OFF (default).
    let retry_cols = super::linsolve::density_route_retry_available().then(|| cols.clone());
    let sol = Simplex::new_from_cols(cols, m, n, c, l, u, b, opts).run();
    if let Some(cols2) = retry_cols {
        if dense_retry_wanted(&sol) {
            return dense_retry(sol, || {
                Simplex::new_from_cols(cols2, m, n, c, l, u, b, opts).run()
            });
        }
    }
    sol
}

/// Sparse-native equivalent of [`solve_lp`]: a cold CSC solve with the SAME
/// geometric-mean equilibration [`solve_lp`] applies via [`ScaledLp`]. It is
/// bit-identical to `solve_lp` on the same matrix — [`Scaling::from_sparse`] yields
/// the identical factors as `Scaling::from_matrix` (equilibration is sparse-native;
/// zeros never affect a line's min/max), [`Scaling::scale_cols`] produces the same
/// `R A C` entries as `scale_matrix`, [`solve_lp_cols`] runs the same `Simplex` the
/// node solves already trust, and the certificate vectors are unscaled back to the
/// original space exactly as `solve_lp` does. The point: it NEVER materializes the
/// dense `m×n` matrix, so a large sparse relaxation (docs/dev/sparse-milp-plan.md,
/// T2) is solved from CSC without the dense blow-up while staying pivot-for-pivot
/// identical to the dense root solve.
#[allow(clippy::too_many_arguments)]
pub fn solve_lp_cols_scaled(
    mut cols: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    opts: &SimplexOptions,
) -> LpSolve {
    match Scaling::from_sparse(&cols, m, n) {
        Some(scaling) => {
            scaling.scale_cols(&mut cols);
            let cs = scaling.scale_c(c);
            let ls = scaling.scale_lower(l);
            let us = scaling.scale_upper(u);
            let bs = scaling.scale_b(b);
            let mut sol = solve_lp_cols(cols, m, n, &cs, &ls, &us, &bs, opts);
            // Map the certificate vectors back to the original space (mirrors
            // `solve_lp`), so a caller's safe-bound / Farkas check sees them against
            // the ORIGINAL A/b/bounds rather than the scaled system.
            scaling.unscale_x(&mut sol.x);
            scaling.unscale_dual(&mut sol.dual);
            scaling.unscale_ray(&mut sol.ray);
            sol
        }
        // Well-conditioned: solve unscaled — bit-identical to `solve_lp`'s `None`
        // branch (`solve_lp_scaled` on the raw view).
        None => solve_lp_cols(cols, m, n, c, l, u, b, opts),
    }
}

/// Warm-started primal solve from an inherited `start` basis (cert:T1.4).
///
/// The incremental McCormick B&B changes only the node's box, so a child LP
/// differs from its parent by a few McCormick row *coefficients* over a *subset*
/// box — the parent's basis is usually still **primal-feasible** but no longer
/// dual-optimal. The dual simplex cannot repair that (it requires dual
/// feasibility on entry — see `dual.rs`), so its warm path rejects the basis and
/// cold-refactorizes. This entry instead runs a **primal** re-solve: it ingests
/// `start`, factorizes it, and — iff the basis is nonsingular AND primal-feasible
/// — skips phase 1 and runs primal phase 2 directly (a handful of pivots). On any
/// doubt (malformed/singular basis, artificial basic, primal-infeasible) it falls
/// back to the full cold two-phase solve. **Sound by construction:** phase 2 from
/// a primal-feasible basis converges to the true LP optimum, and every other case
/// takes the trusted cold path — the warm shortcut can only change speed.
#[allow(clippy::too_many_arguments)]
pub fn solve_lp_cols_warm(
    cols: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    start: &Basis,
    opts: &SimplexOptions,
) -> LpSolve {
    // #85: cloned up front for a possible dense retry (see solve_lp_cols).
    let retry_cols = super::linsolve::density_route_retry_available().then(|| cols.clone());
    let s = Simplex::new_from_cols(cols, m, n, c, l, u, b, opts);
    let sol = match s.run_warm(start) {
        Ok(sol) => sol,
        // Unusable warm basis → the genuine cold two-phase solve on the same
        // matrix (handed back so no clone / no stale state is reused).
        Err(cols_back) => Simplex::new_from_cols(cols_back, m, n, c, l, u, b, opts).run(),
    };
    if let Some(cols2) = retry_cols {
        if dense_retry_wanted(&sol) {
            // Cold, never warm from `start`-derived state: after a numerical
            // failure the warm trajectory is exactly what is in doubt.
            return dense_retry(sol, || {
                Simplex::new_from_cols(cols2, m, n, c, l, u, b, opts).run()
            });
        }
    }
    sol
}

/// Whether `x` satisfies the box `l ≤ x ≤ u` and the equalities `A x = b` to a
/// small absolute/relative tolerance. Used as the final feasibility audit before
/// certifying an `Optimal` solve so incremental `x_B` drift (Harris bound
/// excursions, deferred refactorization) cannot return a wrong optimum.
/// Why the final feasibility audit rejected a point — the distinction that
/// decides whether iterative refinement can help (discopt#364).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feasibility {
    /// Within tolerance on both bounds and `Ax=b`.
    Ok,
    /// A variable sits outside `[l, u]` beyond tolerance. On the degenerate lifted
    /// relaxations this is typically a Harris ratio-test bound *excursion* (the
    /// `delta_tol` δ-expansion letting a basic var settle just past a bound), not a
    /// solve-accuracy problem — so recomputing `x_B` more accurately does **not**
    /// fix it. Refinement is skipped for this kind.
    Bounds,
    /// The `Ax=b` row residual exceeds tolerance — a genuine linear-solve
    /// inaccuracy (accumulated Forrest–Tomlin update error / ill-conditioning),
    /// which a fresh refinement-polished factorization *can* recover.
    Rows,
}

/// Classify a candidate point against its bounds and `Ax=b`. Bounds are checked
/// first, so a point violating *both* reports [`Feasibility::Bounds`] — correct
/// for the recovery decision, since refinement cannot repair a bound excursion
/// even when it repairs the residual.
fn audit_feasibility(
    cols: &SparseCols,
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    x: &[f64],
) -> Feasibility {
    const FEAS: f64 = 1e-6;
    for j in 0..n {
        // Relative bound tolerance: a variable whose bound (and hence value) is
        // large carries proportionally larger floating-point drift, so an
        // absolute 1e-6 would spuriously reject a sound optimum on ill-scaled
        // lifted LPs (variable magnitudes ~1e9). Scale the slack by the bound
        // magnitude, mirroring the relative test already used on the row residual.
        let lo_tol = FEAS * (1.0 + l[j].abs().min(INF));
        let hi_tol = FEAS * (1.0 + u[j].abs().min(INF));
        if x[j] < l[j] - lo_tol || x[j] > u[j] + hi_tol {
            return Feasibility::Bounds;
        }
    }
    // Row activity `A x` accumulated over the sparse columns (O(nnz)), not a dense
    // `m·n` matvec — the lifted relaxations are ~0.3% dense, so this is the
    // difference between milliseconds and seconds on the per-solve audit.
    let mut ax = vec![0.0f64; m];
    for j in 0..n {
        let xj = x[j];
        if xj != 0.0 {
            let (rows, vals) = cols.col(j);
            for (k, &i) in rows.iter().enumerate() {
                ax[i] += vals[k] * xj;
            }
        }
    }
    for i in 0..m {
        if (ax[i] - b[i]).abs() > FEAS * (1.0 + b[i].abs()) {
            return Feasibility::Rows;
        }
    }
    Feasibility::Ok
}

/// Recover finite, *valid* upper bounds for the **slack** columns of the standard
/// form (discopt C-39, generalized). A Farkas ray has row multipliers whose reduced
/// cost on a slack `s ∈ [0, ∞)` selects its open upper side, collapsing the
/// Neumaier–Shcherbina `g₀` to `−∞` — so a genuine infeasibility whose ray touches a
/// slack would fail to certify. A slack is a **single-entry** column: one nonzero
/// `a` in some row `i` (`Σ_{k≠j} A_ik x_k + a·s_j = b_i`), so
/// `s_j = (b_i − Σ_{k≠j} A_ik x_k) / a`, and its maximum over the box is
/// `(b_i − min_x Σ_{k≠j} A_ik x_k)/a` for `a > 0` or
/// `(b_i − max_x Σ_{k≠j} A_ik x_k)/a` for `a < 0` (dividing by the negative `a`
/// flips the extreme).
///
/// The coefficient `a` is **general** (not just `±1`): geometric-mean equilibration
/// scales each `≤`/`≥` slack column by its row/col factor (the AMP trig-square rows
/// yield `−0.5` surplus slacks). The "other columns" sum is over **all** columns
/// except `j` — structural *and* the many-entry slack-region columns — because the
/// crate's standard form is not a clean `[A | ±I]`; the sum uses each column's own
/// box (`min_x a·x = a·l` if `a>0` else `a·u`). We track, per row and per side, how
/// many columns are **open** (unbounded on the extreme); a slack's bound is
/// recoverable iff the only open column on the needed side is the slack itself
/// (`open_count == 1` and this `j` is that open column) or there are none
/// (`open_count == 0`). Otherwise `+∞` (the check bails that sign, conservative).
///
/// Exact (one pass, no iterative-FBBT division roundoff) and superset-preserving
/// (removes only already-infeasible points), so it can never make a *feasible* LP
/// certify: sound by construction. Returns the slack upper bounds (`+∞` where not
/// recoverable).
fn slack_upper_bounds(
    cols: &SparseCols,
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> Vec<f64> {
    // Per-row min/max of `Σ_k A_ik x_k` over the box, accumulated over EVERY column
    // (structural and slack-region) from that column's own bounds. `open_min/max`
    // counts how many columns are unbounded on that extreme; `min/max_ax` sums the
    // finite ones. To bound a single-entry slack `j` we need the sum over the OTHER
    // columns, obtained by removing `j`'s own contribution below.
    let mut min_ax = vec![0.0f64; m];
    let mut max_ax = vec![0.0f64; m];
    let mut open_min = vec![0u32; m];
    let mut open_max = vec![0u32; m];
    for j in 0..n {
        let (rows, vals) = cols.col(j);
        for (k, &i) in rows.iter().enumerate() {
            let a = vals[k];
            if a == 0.0 {
                continue;
            }
            // min side: a>0 → a·l_j (open if l_j = −∞); a<0 → a·u_j (open if u_j=+∞)
            if a > 0.0 {
                if l[j] <= -INF {
                    open_min[i] += 1;
                } else {
                    min_ax[i] += a * l[j];
                }
                if u[j] >= INF {
                    open_max[i] += 1;
                } else {
                    max_ax[i] += a * u[j];
                }
            } else {
                if u[j] >= INF {
                    open_min[i] += 1;
                } else {
                    min_ax[i] += a * u[j];
                }
                if l[j] <= -INF {
                    open_max[i] += 1;
                } else {
                    max_ax[i] += a * l[j];
                }
            }
        }
    }
    let mut sub = vec![INF; n];
    for j in 0..n {
        // A slack is a single-entry column. Anything else keeps `+∞` (the check then
        // conservatively bails that sign), so a caller whose matrix is not this
        // standard form can never get an unsound bound.
        let (rows, vals) = cols.col(j);
        if rows.len() != 1 {
            continue;
        }
        let i = rows[0];
        let a = vals[0];
        if a.abs() < 1e-30 {
            continue;
        }
        // `s_j = (b_i − Σ_{k≠j} A_ik x_k)/a`. Maximizing s_j needs min (a>0) / max
        // (a<0) of the OTHER columns' row sum. Remove j's own contribution to the
        // row min/max (`j` is finite on the min side iff a>0? — compute exactly),
        // and require every remaining column on that side be finite.
        if a > 0.0 {
            // min side: j contributes a·l_j (finite, since a>0 and l_j finite makes
            // it counted in min_ax; if l_j = −∞ it was counted in open_min). Other
            // columns finite iff removing j clears the min side: open_min[i] must be
            // 0 (j finite here) or exactly 1 with that one being j (j open here).
            let (j_open, j_term) = if l[j] <= -INF {
                (true, 0.0)
            } else {
                (false, a * l[j])
            };
            let others_finite = if j_open {
                open_min[i] == 1
            } else {
                open_min[i] == 0
            };
            if others_finite {
                let min_other = min_ax[i] - j_term;
                sub[j] = (b[i] - min_other) / a;
            }
        } else {
            // a<0. max side: j contributes a·l_j (a<0 → j finite iff l_j finite).
            let (j_open, j_term) = if l[j] <= -INF {
                (true, 0.0)
            } else {
                (false, a * l[j])
            };
            let others_finite = if j_open {
                open_max[i] == 1
            } else {
                open_max[i] == 0
            };
            if others_finite {
                let max_other = max_ax[i] - j_term;
                sub[j] = (b[i] - max_other) / a;
            }
        }
    }
    sub
}

/// Whether the candidate ray `y` is a *verified* Farkas certificate that
/// `A x = b, l ≤ x ≤ u` (standard form over the first `n` columns) is empty
/// (discopt C-39). The system is infeasible iff some free-sign `y` has `bᵀy`
/// strictly above the box-maximum of `(Aᵀy)ᵀx` — the `c = 0` Neumaier–Shcherbina
/// safe bound `g₀(y) > 0`. Both `±y` are tried (the ray is known up to sign) with a
/// magnitude-scaled margin keeping the strict inequality rigorous under rounding.
/// A **slack** column `[0, ∞)` whose open upper side is selected would collapse `g₀`
/// to `−∞`; an *exact* upper bound recovered from its defining row
/// ([`slack_upper_bounds`], covering both `+1` and `−1` surplus slacks) makes the
/// term finite, so a genuine infeasibility whose ray touches either a `≤`-slack or a
/// `≥`-surplus-slack still certifies (mirrors the Python boundary's
/// `_farkas_certified_std`; keeps the engine from returning `Numerical` where the
/// cold/Python path rigorously fathoms). Sound: the slack bound is
/// superset-preserving, so it can never make a *feasible* LP certify. A genuinely
/// open *structural* side (none on these lifted relaxations) bails the sign.
pub(super) fn farkas_ray_certifies_cols(
    y: &[f64],
    cols: &SparseCols,
    n: usize,
    m: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> bool {
    if y.len() != m || !y.iter().all(|v| v.is_finite()) {
        return false;
    }
    // Exact, superset-preserving upper bounds for the slack columns (the only open
    // columns a `≤`-Farkas ray selects on these LPs), so a genuine infeasibility
    // whose ray touches a slack still certifies. Built lazily on first need.
    let mut slack_ub: Option<Vec<f64>> = None;
    for sgn in [1.0f64, -1.0] {
        let ys: Vec<f64> = y.iter().map(|v| sgn * v).collect();
        let by: f64 = b.iter().zip(&ys).map(|(bi, yi)| bi * yi).sum();
        let mut contrib = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut open = false;
        for j in 0..n {
            let aty = cols.dot(j, &ys);
            // Box-max of `aty · x` over `[l_j, u_j]`: `aty·u_j` (aty>0) / `aty·l_j`
            // (aty<0). An open selected side collapses `g₀` to `−∞` — recover a finite
            // upper bound for a slack column from its defining row; a genuinely open
            // structural side bails this sign (conservative).
            let boxmax = if aty > 0.0 {
                let uj = if u[j] >= INF {
                    let sub =
                        slack_ub.get_or_insert_with(|| slack_upper_bounds(cols, m, n, b, l, u));
                    if sub[j] >= INF {
                        open = true;
                        break;
                    }
                    sub[j]
                } else {
                    u[j]
                };
                aty * uj
            } else if aty < 0.0 {
                if l[j] <= -INF {
                    open = true;
                    break;
                }
                aty * l[j]
            } else {
                0.0
            };
            contrib += boxmax;
            abs_sum += boxmax.abs();
        }
        if open {
            continue;
        }
        // Neumaier–Shcherbina relative margin so the strict `> 0` is rigorous.
        let margin = 1e-9 * (1.0 + by.abs() + abs_sum);
        if by - contrib > margin {
            return true;
        }
    }
    false
}

/// Bool convenience over [`audit_feasibility`] used by the unit tests (production
/// code switches on the [`Feasibility`] reason to decide whether to refine).
#[cfg(test)]
fn solution_within_tolerance(
    cols: &SparseCols,
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    x: &[f64],
) -> bool {
    audit_feasibility(cols, m, n, b, l, u, x) == Feasibility::Ok
}

struct Simplex<'a> {
    cols: SparseCols, // CSC of the constraint matrix; the sole matrix view (pricing,
    // residual, audit all go through it — no dense `m×n` copy is kept)
    b: &'a [f64], // length m
    c: &'a [f64], // length n
    m: usize,
    n: usize,  // real columns
    na: usize, // n + m (with artificials appended)
    tol: f64,
    max_iter: usize,
    /// Absolute wall-clock deadline (polled inside the iteration loop); `None`
    /// disables the check. See [`SimplexOptions::deadline`].
    deadline: Option<std::time::Instant>,
    lu: FeralLU,
    basis: Vec<usize>, // slot -> column (len m)
    slot_of: Vec<i64>, // column -> slot, or -1
    stat: Vec<i8>,     // column -> BASIC/AT_LOWER/AT_UPPER (len na)
    lb: Vec<f64>,      // len na
    ub: Vec<f64>,      // len na
    /// Primal unbounded ray (length `n`), captured when [`Self::simplex_loop`]
    /// detects unboundedness so [`Self::assemble`] can export it; empty until then.
    unbounded_ray: Vec<f64>,
}

impl<'a> Simplex<'a> {
    fn new(lp: &'a LpView<'a>, b: &'a [f64], opts: &SimplexOptions) -> Self {
        Self::new_from_cols(
            SparseCols::from_dense(lp.a, lp.m, lp.n),
            lp.m,
            lp.n,
            lp.c,
            lp.l,
            lp.u,
            b,
            opts,
        )
    }

    /// Build the cold primal solver from an owned CSC matrix instead of a dense
    /// [`LpView`] — the sparse-native ingestion path (the CSC warm entry and its
    /// cold fallback), which never materializes the dense `m×n` matrix.
    #[allow(clippy::too_many_arguments)]
    fn new_from_cols(
        cols: SparseCols,
        m: usize,
        n: usize,
        c: &'a [f64],
        l: &'a [f64],
        u: &'a [f64],
        b: &'a [f64],
        opts: &SimplexOptions,
    ) -> Self {
        let na = n + m;
        let mut lb = vec![0.0; na];
        let mut ub = vec![0.0; na];
        lb[..n].copy_from_slice(l);
        ub[..n].copy_from_slice(u);
        // Artificials: bounds set per phase (Phase 1 [0,inf], Phase 2 [0,0]).
        Self {
            cols,
            b,
            c,
            m,
            n,
            na,
            tol: opts.tol,
            max_iter: opts.max_iter,
            deadline: opts.deadline,
            lu: node_feral_lu(),
            basis: Vec::new(),
            slot_of: vec![-1; na],
            stat: vec![AT_LOWER; na],
            lb,
            ub,
            unbounded_ray: Vec::new(),
        }
    }

    /// Dense column `A_j` (length m). Real columns from `a`; artificial `n+i` is
    /// `sign_i · e_i`, sign chosen at init so the starting residual is ≥ 0.
    fn column(&self, j: usize, art_sign: &[f64]) -> Vec<f64> {
        let mut col = vec![0.0; self.m];
        if j < self.n {
            self.cols.scatter(j, &mut col);
        } else {
            let i = j - self.n;
            col[i] = art_sign[i];
        }
        col
    }

    /// Sparse `yᵀ A_j` for column `j` (real column via CSC, artificial via its
    /// single signed entry). The pricing/Devex hot path uses this instead of
    /// materializing a dense column.
    #[inline]
    fn col_dot(&self, j: usize, y: &[f64], art_sign: &[f64]) -> f64 {
        if j < self.n {
            self.cols.dot(j, y)
        } else {
            let i = j - self.n;
            y[i] * art_sign[i]
        }
    }

    /// Bound value a nonbasic column sits at (lower or upper).
    fn nb_value(&self, j: usize) -> f64 {
        match self.stat[j] {
            AT_UPPER => self.ub[j],
            _ => {
                // at lower; if lower is -inf (free), sit at 0
                if self.lb[j] <= -INF {
                    0.0
                } else {
                    self.lb[j]
                }
            }
        }
    }

    /// The right-hand side the basic-variable solve runs against:
    /// `b − Σ_{nonbasic} A_j x_j` (so that `B x_B = rhs`).
    fn reduced_rhs(&self, art_sign: &[f64]) -> Vec<f64> {
        let mut rhs = self.b.to_vec();
        for j in 0..self.na {
            if self.stat[j] != BASIC {
                let v = self.nb_value(j);
                if v != 0.0 {
                    if j < self.n {
                        let (rows, vals) = self.cols.col(j);
                        for (k, &r) in rows.iter().enumerate() {
                            rhs[r] -= vals[k] * v;
                        }
                    } else {
                        rhs[j - self.n] -= art_sign[j - self.n] * v;
                    }
                }
            }
        }
        rhs
    }

    /// Recompute basic-variable values: x_B = B⁻¹ (b − Σ_{nonbasic} A_j x_j).
    fn basic_values(&mut self, art_sign: &[f64]) -> Result<Vec<f64>, ()> {
        let mut rhs = self.reduced_rhs(art_sign);
        self.lu.ftran(&mut rhs).map_err(|_| ())?;
        Ok(rhs)
    }

    /// Recover `x_B` for the final basis via a **fresh** numeric-focus
    /// factorization with iterative refinement (discopt#364). Used only when the
    /// incremental factor's `x_B` fails the feasibility audit: a fresh factor
    /// carries none of the accumulated Forrest–Tomlin update error, and
    /// refinement then polishes the residual `b − B·x`, so an ill-conditioned or
    /// update-drifted basis is recovered *inside the engine* rather than by
    /// falling back to another solver. Returns `None` if the fresh factorization
    /// or refined solve fails (the caller keeps the Numerical verdict).
    fn refined_basic_values(&self, art_sign: &[f64]) -> Option<Vec<f64>> {
        let cols: Vec<Vec<f64>> = self
            .basis
            .iter()
            .map(|&j| self.column(j, art_sign))
            .collect();
        let mut lu = node_feral_lu().with_numeric_focus();
        lu.factorize(self.m, &cols).ok()?;
        let mut rhs = self.reduced_rhs(art_sign);
        lu.ftran_refined(&mut rhs).ok()?;
        Some(rhs)
    }

    /// Scatter a basic-values vector `x_B` into the full primal point `x`
    /// (basic vars take their `x_B` slot, nonbasic vars sit at their bound).
    fn assemble_x(&self, xb: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.n];
        for j in 0..self.n {
            x[j] = if self.stat[j] == BASIC {
                xb[self.slot_of[j] as usize]
            } else {
                self.nb_value(j)
            };
        }
        x
    }

    fn refactorize(&mut self, art_sign: &[f64]) -> Result<(), ()> {
        // Build the basis as *sparse* columns (slot -> (row, value) nonzeros) so
        // the factorization never materializes the dense m×m basis: structural
        // columns come straight from the CSC `cols`, artificials are a single
        // signed unit entry. Bit-identical to the dense `column()` build, O(nnz)
        // instead of O(m²) (discopt#268 / feral#87).
        let cols: Vec<Vec<(usize, f64)>> = self
            .basis
            .iter()
            .map(|&j| {
                if j < self.n {
                    let (rows, vals) = self.cols.col(j);
                    rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
                } else {
                    vec![(j - self.n, art_sign[j - self.n])]
                }
            })
            .collect();
        self.lu.factorize_sparse(self.m, &cols).map_err(|_| ())
    }

    fn run(mut self) -> LpSolve {
        let m = self.m;
        // --- initialize: all real vars nonbasic at a finite bound (or 0 if free)
        for j in 0..self.n {
            self.stat[j] = if self.lb[j] > -INF {
                AT_LOWER
            } else if self.ub[j] < INF {
                AT_UPPER
            } else {
                AT_LOWER // free → treated as at 0
            };
        }
        // residual r = b − Σ A_j x_j over real nonbasic vars, accumulated through
        // the sparse columns (O(nnz of the nonbasic-at-nonzero columns)) instead
        // of a dense O(m·n) sweep — on the ~0.3%-dense lifted relaxations almost
        // every nonbasic var sits at 0 (lower bound), so this touches almost
        // nothing rather than scanning the whole matrix.
        let mut r = self.b.to_vec();
        for j in 0..self.n {
            let v = self.nb_value(j);
            if v != 0.0 {
                let (rows, vals) = self.cols.col(j);
                for (k, &i) in rows.iter().enumerate() {
                    r[i] -= vals[k] * v;
                }
            }
        }
        let art_sign: Vec<f64> = (0..m)
            .map(|i| if r[i] >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        // Crash basis: in each row prefer a basic-feasible structural *singleton*
        // (a slack column is one nonzero) over the artificial, so rows already
        // satisfied at the bound point start feasible and skip phase-1 pivots.
        // Each crashed column is the unique nonzero in its row, so the basis stays
        // a permuted (sign-scaled) identity — nonsingular — and any row left with
        // an artificial is still driven feasible by phase-1; the crash only
        // warm-starts and can never change the result. It is what keeps the
        // heavily degenerate lifted relaxations of issue #175 (RLT, affine-power)
        // from stalling phase-1 with thousands of zero-step pivots.
        self.basis = (0..m).map(|i| self.n + i).collect();
        let mut crashed = vec![false; m];
        for j in 0..self.n {
            let (rows, vals) = self.cols.col(j);
            if rows.len() != 1 {
                continue;
            }
            let i = rows[0];
            let a = vals[0];
            if crashed[i] || a == 0.0 {
                continue;
            }
            let xj = self.nb_value(j) + r[i] / a;
            if xj >= self.lb[j] - 1e-9 && xj <= self.ub[j] + 1e-9 {
                crashed[i] = true;
                self.basis[i] = j;
            }
        }
        for i in 0..m {
            let art = self.n + i;
            self.lb[art] = 0.0;
            self.ub[art] = INF; // phase 1 allows artificials to be positive
            let bcol = self.basis[i];
            self.stat[bcol] = BASIC;
            self.slot_of[bcol] = i as i64;
            if bcol != art {
                // a structural singleton was crashed in; its artificial stays
                // nonbasic at zero.
                self.stat[art] = AT_LOWER;
                self.slot_of[art] = -1;
            }
        }
        if self.refactorize(&art_sign).is_err() {
            return self.failed();
        }

        // --- Phase 1: minimize sum of artificials ---
        let mut cost1 = vec![0.0; self.na];
        for i in 0..m {
            cost1[self.n + i] = 1.0;
        }
        match self.simplex_loop(&cost1, &art_sign, true) {
            Ok(_) => {}
            Err(st) => return self.assemble(st, &art_sign),
        }
        // Phase-1 reduced-cost cleanup (discopt C-39). The reduced-cost optimality
        // test above can terminate with an *artificial still basic at a positive
        // value* on a **feasible** LP: on the heavily-redundant lifted McCormick
        // relaxations, every structural column that would expel such an artificial
        // has reduced cost ≈ 0 (it is already spanned), so the strict `dⱼ < −tol`
        // entering rule never selects it and phase-1 stalls above zero — a
        // *numerical false infeasible* (the kall_circles_c8a / nvs06-cut class:
        // HiGHS solves the identical LP feasible). The pricing loop optimizes
        // `Σ artificials` but has no incentive to break a *degenerate* (zero-cost)
        // tie that removes a basic artificial. This pass performs exactly those
        // degenerate pivots: drive every basic artificial out of the basis via a
        // nonbasic **structural** column with a nonzero pivot in its row. An
        // artificial whose row has no such column is genuinely non-removable — that
        // row's residual is a real infeasibility. Sound: expelling an artificial by
        // a structural pivot preserves `A x = b` (it only re-expresses the same
        // point over a different basis) and can never *raise* the artificial sum, so
        // it never turns a feasible LP infeasible or vice-versa; it only completes
        // the phase-1 the reduced-cost test left unfinished.
        //
        // Measure the phase-1 residual. A positive artificial sum is *either* a
        // genuine infeasibility *or* a numerical false-infeasible stall. Distinguish
        // them by the sole rigorous test — a **verified Farkas ray** — computed on
        // the phase-1-optimal basis (which carries the genuine ray) BEFORE any
        // cleanup perturbs it.
        let xb0 = match self.basic_values(&art_sign) {
            Ok(v) => v,
            Err(()) => return self.failed(),
        };
        let infeas0: f64 = self
            .basis
            .iter()
            .enumerate()
            .filter(|(_, &j)| j >= self.n)
            .map(|(slot, _)| xb0[slot].abs())
            .sum();
        if infeas0 > 1e-6 {
            // (1) Genuine infeasibility: the phase-1 duals form a Farkas ray that
            // *verifies* the feasible set is empty. This is the rigorous fathoming
            // proof — return `Infeasible` directly (matching the previous behaviour
            // on truly-infeasible LPs, so no fathom is lost).
            let ray = self.phase1_farkas_ray(&art_sign);
            if ray.as_ref().is_some_and(|y| self.farkas_ray_certifies(y)) {
                return self.assemble(LpStatus::Infeasible, &art_sign);
            }
            // (2) The ray does NOT certify: the `infeasible` verdict is unproven and
            // may be a numerical false-infeasible (discopt C-39 — the kall_circles /
            // nvs06-cut class: HiGHS solves the identical LP feasible). Complete
            // phase-1 by expelling every removable artificial with
            // feasibility-preserving ratio-test pivots; only after that is a residual
            // a genuine (redundant-row) infeasibility.
            if self.drive_out_basic_artificials(&art_sign).is_err() {
                return self.failed();
            }
            let xb = match self.basic_values(&art_sign) {
                Ok(v) => v,
                Err(()) => return self.failed(),
            };
            let infeas: f64 = self
                .basis
                .iter()
                .enumerate()
                .filter(|(_, &j)| j >= self.n)
                .map(|(slot, _)| xb[slot].abs())
                .sum();
            if infeas > 1e-6 {
                // A residual survives the full cleanup. Re-check the ray on the cleaned
                // basis: if it now certifies, the LP is genuinely infeasible; otherwise
                // the verdict remains unproven — return the honest `Numerical`
                // (non-fathoming), NEVER a false `Infeasible`.
                let ray = self.phase1_farkas_ray(&art_sign);
                if ray.as_ref().is_some_and(|y| self.farkas_ray_certifies(y)) {
                    return self.assemble(LpStatus::Infeasible, &art_sign);
                }
                return self.assemble(LpStatus::Numerical, &art_sign);
            }
            // Artificials cleared — the LP is FEASIBLE and, because the cleanup used
            // only phase-1-feasible ratio-test pivots, the basis is primal-feasible.
            // Phase 2 (below) optimizes the real objective from it and yields a bound.
        }

        // --- Phase 2: pin artificials to 0, optimize real objective ---
        for i in 0..m {
            self.ub[self.n + i] = 0.0; // fix artificials at 0
        }
        let mut cost2 = vec![0.0; self.na];
        cost2[..self.n].copy_from_slice(self.c);
        let st = match self.simplex_loop(&cost2, &art_sign, false) {
            Ok(()) => LpStatus::Optimal,
            Err(s) => s,
        };
        self.assemble(st, &art_sign)
    }

    /// Complete phase-1 by driving every basic **artificial** out of the basis with
    /// feasibility-preserving ratio-test pivots (discopt C-39). Phase-1's
    /// reduced-cost optimality test can terminate with an artificial still basic at a
    /// positive value on a *feasible* LP: on the heavily-redundant lifted McCormick
    /// relaxations every structural column that would expel such an artificial is
    /// priced at reduced cost ≈ 0 (already spanned by the redundant rows), so the
    /// strict `dⱼ < −tol` entering rule never selects it and the sum-of-artificials
    /// stalls above zero — the numerical false infeasible (kall_circles_c8a /
    /// nvs06-cut: HiGHS solves the identical LP feasible). This pass performs exactly
    /// the degenerate (zero-reduced-cost) pivots the reduced-cost test skips.
    ///
    /// Every pivot is a **standard bounded-variable ratio-test step in the phase-1
    /// feasible region** (all variables — artificials in `[0, ∞)`, structurals at a
    /// bound — stay feasible throughout), so no separate feasibility-restore is
    /// needed: the basis handed to phase 2 is primal-feasible. For a chosen basic
    /// artificial, an entering structural column with a nonzero pivot in its row is
    /// moved in the direction that *reduces* the artificial; the ratio test lets the
    /// true blocker leave. If the artificial is the blocker it is expelled; otherwise
    /// its value drops and we retry. An artificial that admits no reducing pivot, or
    /// that cannot be reduced within a per-row degenerate-pivot budget, is marked as
    /// a genuinely non-removable (redundant-row) infeasibility and left basic — its
    /// residual then flows to the caller's Farkas-verified infeasibility test.
    /// Returns `Err(())` only on a factorization breakdown.
    ///
    /// Soundness: ratio-test pivots preserve `A x = b` and phase-1 feasibility and
    /// never *raise* the artificial sum, so a feasible LP is never turned infeasible
    /// nor the reverse — the pass only completes the phase-1 the reduced-cost loop
    /// left short.
    fn drive_out_basic_artificials(&mut self, art_sign: &[f64]) -> Result<(), ()> {
        let mut xb = self.basic_values(art_sign)?;
        // Cheap early-out: the common healthy phase-1 leaves no basic artificial, so
        // that path is untouched (zero extra work).
        let any_basic_art = self
            .basis
            .iter()
            .enumerate()
            .any(|(slot, &j)| j >= self.n && xb[slot].abs() > 1e-9);
        if !any_basic_art {
            return Ok(());
        }

        // A usable pivot must be non-tiny (basis conditioning).
        const PIVOT_MIN: f64 = 1e-7;
        const ART_TOL: f64 = 1e-9;
        let mut updates_since_refac = 0usize;
        // Rows whose basic artificial cannot be reduced (no reducing pivot, or only
        // degenerate zero-step pivots that never lower its value): a genuine
        // redundant-row infeasibility. Skipped so the pass always makes progress.
        let mut stuck = vec![false; self.m];
        // Per-target degenerate-pivot budget: a run of zero-step pivots that fails to
        // reduce the current artificial marks its row stuck (anti-cycling — the
        // degenerate ties are broken by leaving-row index, so no basis repeats).
        let degenerate_cap = 2 * self.m + 20;
        let max_pivots = (8 * (self.m + self.n) + 200).min(self.max_iter);
        let mut pivots = 0usize;
        let mut cur_target: Option<usize> = None;
        let mut degenerate_run = 0usize;
        while pivots < max_pivots {
            // (Re)select a basic artificial still positive on a not-yet-stuck row.
            let slot = match cur_target {
                Some(s) if self.basis[s] >= self.n && !stuck[s] && xb[s].abs() > ART_TOL => s,
                _ => {
                    degenerate_run = 0;
                    match self
                        .basis
                        .iter()
                        .enumerate()
                        .find(|(s, &j)| j >= self.n && !stuck[*s] && xb[*s].abs() > ART_TOL)
                        .map(|(s, _)| s)
                    {
                        Some(s) => {
                            cur_target = Some(s);
                            s
                        }
                        None => return Ok(()), // every artificial expelled, ~0, or stuck
                    }
                }
            };

            // Pivot row ρ = e_slotᵀ B⁻¹; pick the largest-|pivot| nonbasic structural
            // column in this row whose entry lets `q` move to *reduce* the artificial.
            let mut rho = vec![0.0; self.m];
            rho[slot] = 1.0;
            if self.lu.btran(&mut rho).is_err() {
                return Err(());
            }
            let mut q: Option<usize> = None;
            let mut q_dir = 1.0f64;
            let mut best = PIVOT_MIN;
            for j in 0..self.n {
                if self.stat[j] == BASIC {
                    continue;
                }
                let arj = self.col_dot(j, &rho, art_sign);
                if arj.abs() <= best {
                    continue;
                }
                // Entering `q` by `+dir·t` changes the artificial by `−dir·arj·t`. The
                // artificial is ≥ 0, so to reduce it we need that change < 0, i.e.
                // `dir·arj > 0`. Choose `dir` off `q`'s current bound accordingly and
                // require the move to be available in that direction.
                let dir_by_sign = if arj > 0.0 { 1.0 } else { -1.0 };
                let at_lower = self.stat[j] == AT_LOWER;
                // A column at its lower bound can only increase (dir +1); at upper only
                // decrease (dir −1). Skip if the reducing direction is not available.
                if (dir_by_sign > 0.0 && at_lower) || (dir_by_sign < 0.0 && !at_lower) {
                    best = arj.abs();
                    q = Some(j);
                    q_dir = dir_by_sign;
                }
            }
            let q = match q {
                Some(q) => q,
                None => {
                    stuck[slot] = true;
                    cur_target = None;
                    continue;
                }
            };

            // Entering column α = B⁻¹A_q; bounded ratio test on the phase-1 feasible
            // region. `q` steps by `dir·t ≥ 0`; basic `i` moves by `−dir·α[i]·t`.
            let mut alpha = self.column(q, art_sign);
            if self.lu.ftran(&mut alpha).is_err() {
                return Err(());
            }
            let dir = q_dir;
            let q_gap = if self.ub[q] < INF && self.lb[q] > -INF {
                self.ub[q] - self.lb[q]
            } else {
                INF
            };
            // Pass 1: the smallest ratio `t` (the tightest bound the step can reach).
            let mut t = q_gap;
            for i in 0..self.m {
                let change = -dir * alpha[i];
                let bi = self.basis[i];
                if change < -self.tol {
                    let lo = self.lb[bi];
                    if lo > -INF {
                        let ti = ((xb[i] - lo) / (-change)).max(0.0);
                        if ti < t {
                            t = ti;
                        }
                    }
                } else if change > self.tol {
                    let hi = self.ub[bi];
                    if hi < INF {
                        let ti = ((hi - xb[i]) / change).max(0.0);
                        if ti < t {
                            t = ti;
                        }
                    }
                }
            }
            // Pass 2: among rows that block within `t` (Harris), take the LARGEST pivot
            // magnitude — a near-singular (tiny-pivot) leaving choice corrupts the LU
            // update, so the stable choice keeps the basis well-conditioned.
            let mut leave: Option<usize> = None;
            let mut leave_to_upper = false;
            let mut best_pivot = PIVOT_MIN;
            for i in 0..self.m {
                let change = -dir * alpha[i];
                let bi = self.basis[i];
                let (ti, to_upper) = if change < -self.tol {
                    let lo = self.lb[bi];
                    if lo <= -INF {
                        continue;
                    }
                    (((xb[i] - lo) / (-change)).max(0.0), false)
                } else if change > self.tol {
                    let hi = self.ub[bi];
                    if hi >= INF {
                        continue;
                    }
                    (((hi - xb[i]) / change).max(0.0), true)
                } else {
                    continue;
                };
                if ti <= t + self.tol && alpha[i].abs() > best_pivot {
                    best_pivot = alpha[i].abs();
                    leave = Some(i);
                    leave_to_upper = to_upper;
                }
            }

            // Track degenerate (zero-step) progress on the current target.
            if t <= self.tol {
                degenerate_run += 1;
                if degenerate_run > degenerate_cap {
                    stuck[slot] = true;
                    cur_target = None;
                    continue;
                }
            } else {
                degenerate_run = 0;
            }

            // Advance basic values.
            for (i, xbi) in xb.iter_mut().enumerate() {
                *xbi -= dir * t * alpha[i];
            }
            pivots += 1;
            match leave {
                None => {
                    if t >= q_gap && q_gap < INF {
                        // No basic blocks before `q` reaches its opposite bound: bound
                        // flip (no basis change), reducing the artificial by the full
                        // gap step. Advance and retry.
                        self.stat[q] = if self.stat[q] == AT_LOWER {
                            AT_UPPER
                        } else {
                            AT_LOWER
                        };
                    } else {
                        // A blocker exists at ratio `t` but every candidate row has a
                        // tiny (unstable) pivot — pivoting would corrupt the basis.
                        // Leave this artificial for the caller's Farkas test.
                        stuck[slot] = true;
                        cur_target = None;
                    }
                }
                Some(li) => {
                    let leaving = self.basis[li];
                    self.stat[leaving] = if leave_to_upper { AT_UPPER } else { AT_LOWER };
                    self.slot_of[leaving] = -1;
                    self.basis[li] = q;
                    self.slot_of[q] = li as i64;
                    self.stat[q] = BASIC;
                    xb[li] = self.nb_value(q) + dir * t;
                    if leaving >= self.n {
                        // The target artificial (or another) left the basis — pick a
                        // fresh target next iteration.
                        cur_target = None;
                    }
                    let col = self.column(q, art_sign);
                    let need_refac = self.lu.update(li, &col).is_err();
                    updates_since_refac += 1;
                    if need_refac || updates_since_refac >= 48 {
                        self.refactorize(art_sign)?;
                        updates_since_refac = 0;
                        xb = self.basic_values(art_sign)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The candidate Farkas dual ray `y = B⁻ᵀ c₁` from the current phase-1 basis
    /// (`c₁` = 1 on basic artificials, 0 elsewhere), the multipliers that make the
    /// artificial sum the dual objective (discopt C-39). At a genuine phase-1
    /// infeasible optimum this `y` proves `A x = b, l ≤ x ≤ u` empty; at a numerical
    /// stall it does not, which is exactly what [`Self::farkas_ray_certifies`]
    /// checks. `None` if the btran fails.
    fn phase1_farkas_ray(&mut self, _art_sign: &[f64]) -> Option<Vec<f64>> {
        let mut y: Vec<f64> = self
            .basis
            .iter()
            .map(|&j| if j >= self.n { 1.0 } else { 0.0 })
            .collect();
        self.lu.btran(&mut y).ok()?;
        Some(y)
    }

    /// Whether the candidate ray `y` is a *verified* Farkas certificate that
    /// `A x = b, l ≤ x ≤ u` is empty (discopt C-39). Delegates to the shared
    /// [`farkas_ray_certifies_cols`], considering only the `n` real columns (the
    /// artificials are not part of the LP whose emptiness is claimed).
    fn farkas_ray_certifies(&self, y: &[f64]) -> bool {
        farkas_ray_certifies_cols(y, &self.cols, self.n, self.m, self.b, &self.lb, &self.ub)
    }

    /// Warm-started phase-2-only run from an inherited `start` basis (cert:T1.4).
    ///
    /// Returns `Ok(solution)` iff the warm basis is nonsingular AND primal-feasible
    /// (then a primal phase-2 optimizes the real objective from it). Otherwise
    /// returns `Err(self.cols)` — handing the owned matrix back so the caller can
    /// run the trusted cold two-phase solve on a fresh solver (no clone, no stale
    /// state). This never establishes feasibility from an infeasible warm basis;
    /// phase 1 is left entirely to the cold fallback, so the result is always the
    /// true LP optimum (or the cold path's verdict).
    fn run_warm(mut self, start: &Basis) -> Result<LpSolve, SparseCols> {
        let (m, n) = (self.m, self.n);
        // Reject a malformed or non-structural warm basis → cold. A warm basic
        // column must be a real column (an artificial can never be inherited).
        if start.basic_vars.len() != m
            || start.col_status.len() != n
            || start.basic_vars.iter().any(|&j| j >= n)
        {
            return Err(self.cols);
        }
        // Ingest nonbasic status for the real columns; pin every artificial to 0
        // (phase-2 config) and nonbasic.
        self.stat[..n].copy_from_slice(&start.col_status);
        for i in 0..m {
            let art = n + i;
            self.stat[art] = AT_LOWER;
            self.slot_of[art] = -1;
            self.lb[art] = 0.0;
            self.ub[art] = 0.0;
        }
        // Basis slots: reset column→slot map, then install the inherited basis.
        for s in self.slot_of.iter_mut() {
            *s = -1;
        }
        for (slot, &j) in start.basic_vars.iter().enumerate() {
            self.slot_of[j] = slot as i64;
            self.stat[j] = BASIC; // keep stat consistent with basic_vars
        }
        self.basis = start.basic_vars.clone();
        // Artificials pinned to 0, so their sign never enters a basic column.
        let art_sign = vec![1.0; m];
        // Nonsingular? A singular inherited basis → cold.
        if self.refactorize(&art_sign).is_err() {
            return Err(self.cols);
        }
        // Primal-feasible? Compute x_B and check every basic value lies within its
        // column's bounds. If not, phase 1 is required — hand back to the cold path.
        let xb = match self.basic_values(&art_sign) {
            Ok(v) => v,
            Err(()) => return Err(self.cols),
        };
        let ftol = 1e-7;
        let feasible = self.basis.iter().enumerate().all(|(slot, &j)| {
            let v = xb[slot];
            v >= self.lb[j] - ftol && v <= self.ub[j] + ftol
        });
        if !feasible {
            if std::env::var("DISCOPT_T14_DBG").is_ok() {
                eprintln!("T14 REJECT primal-infeasible");
            }
            return Err(self.cols);
        }
        if std::env::var("DISCOPT_T14_DBG").is_ok() {
            eprintln!("T14 ACCEPT");
        }
        // Warm basis is nonsingular + primal-feasible → primal phase 2 only.
        let mut cost2 = vec![0.0; self.na];
        cost2[..n].copy_from_slice(self.c);
        let st = match self.simplex_loop(&cost2, &art_sign, false) {
            Ok(()) => LpStatus::Optimal,
            Err(s) => s,
        };
        Ok(self.assemble(st, &art_sign))
    }

    /// Primal simplex iterations for the given `cost`. Returns Ok(()) at
    /// optimality, Err(Unbounded/IterLimit/Numerical) otherwise.
    fn simplex_loop(
        &mut self,
        cost: &[f64],
        art_sign: &[f64],
        is_phase1: bool,
    ) -> Result<(), LpStatus> {
        let m = self.m;
        let mut updates = 0usize;
        let mut stall = 0usize;
        // Adaptive refactorization budget (discopt#268 / feral#87): the FT basis
        // update is O(bump²) on non-localized spikes — on a wide lifted-McCormick
        // basis (dense structural columns) a single `update` can cost as much as a
        // full refactorization, and the eta-replay in every subsequent solve grows
        // with the chain. Refactorize once the accumulated update work
        // (`ft_update_work`) exceeds the factor's own nnz, so wide-bump bases
        // refactor often (now O(nnz), not O(m²)) instead of compounding the FT
        // blowup, while narrow-bump bases — whose eta work stays tiny — keep all 48
        // cheap updates between refactorizations. Sound: a refactorization yields
        // the same basis inverse (only fresher), so pivots and the optimum are
        // unchanged. `0` (dense backend, or feral not reporting) disables the gate.
        let mut refac_work_budget = self.lu.factor_nnz();
        // Devex reference weights γⱼ ≥ 1 for nonbasic pricing, reset per loop
        // (the basis composition differs between Phase 1 and Phase 2). Devex
        // selects the entering column maximizing dⱼ²/γⱼ — a cheap steepest-edge
        // approximation that only changes *which* improving column enters, so it
        // never affects correctness (Bland's rule still guards against cycling).
        let mut gamma = vec![1.0f64; self.na];
        // Maintain x_B incrementally across pivots (rank-1 update per step)
        // instead of recomputing it by a full ftran each iteration; refreshed
        // exactly on every refactorization to bound floating-point drift.
        let mut xb = self
            .basic_values(art_sign)
            .map_err(|_| LpStatus::Numerical)?;
        // EXPAND anti-degeneracy (Gill et al.): the Harris ratio-test feasibility
        // tolerance grows slowly from δ_min to δ_max, and a guaranteed minimum step
        // keeps every pivot strictly progressing — breaking the degenerate zero-step
        // stalls that dominate the lifted relaxations, more cheaply than falling into
        // Bland. δ resets to δ_min every EXPAND_RESET iterations (the incremental x_B
        // is refreshed exactly on the ≤48-update refactorizations, so the sub-δ
        // excursions never accumulate). δ_max stays at the pre-existing 1e-7 ceiling
        // — 10× under the 1e-6 feasibility audit — so soundness is unchanged.
        const EXPAND_MIN: f64 = 1e-8;
        const EXPAND_MAX: f64 = 1e-7;
        const EXPAND_RESET: usize = 10_000;
        let expand_incr = (EXPAND_MAX - EXPAND_MIN) / EXPAND_RESET as f64;
        let mut expand_tol = EXPAND_MIN;
        for _iter in 0..self.max_iter {
            // Poll the wall-clock deadline every 256 pivots (cheap relative to a
            // pricing+ftran iteration). A dense, degenerate lifted-McCormick LP
            // can otherwise grind toward `max_iter` and run uninterruptibly for
            // minutes; bailing as IterLimit keeps the enclosing B&B inside its
            // time budget, and the caller treats the loose bound soundly (no
            // prune, gap left uncertified). Checks at _iter == 0 too, so a cold
            // fallback entered already past the deadline returns immediately.
            if (_iter & 255) == 0
                && self
                    .deadline
                    .is_some_and(|d| std::time::Instant::now() >= d)
            {
                return Err(LpStatus::IterLimit);
            }
            // price: y = B⁻ᵀ c_B ; reduced cost d_j = c_j − yᵀA_j
            let mut y: Vec<f64> = self.basis.iter().map(|&j| cost[j]).collect();
            {
                let _t = crate::profile::Timer::new(crate::profile::Phase::PriceBtran);
                if self.lu.btran(&mut y).is_err() {
                    return Err(LpStatus::Numerical);
                }
            }
            let bland = stall > 2 * (self.na + 1);
            // choose entering: Devex (max dⱼ²/γⱼ over improving cols), with a
            // Bland's-rule fallback (first improving) once a stall is detected.
            let mut enter: Option<usize> = None;
            let mut best_score = 0.0f64;
            let _t_sweep = crate::profile::Timer::new(crate::profile::Phase::PriceSweep);
            for j in 0..self.na {
                if self.stat[j] == BASIC {
                    continue;
                }
                let dj = cost[j] - self.col_dot(j, &y, art_sign);
                let improving = (self.stat[j] == AT_LOWER && dj < -self.tol)
                    || (self.stat[j] == AT_UPPER && dj > self.tol);
                if improving {
                    if bland {
                        enter = Some(j);
                        break;
                    }
                    let score = dj * dj / gamma[j];
                    if score > best_score {
                        best_score = score;
                        enter = Some(j);
                    }
                }
            }
            drop(_t_sweep);
            let q = match enter {
                Some(q) => q,
                None => return Ok(()), // optimal
            };
            let dir = if self.stat[q] == AT_LOWER { 1.0 } else { -1.0 };

            // direction α = B⁻¹ A_q
            let mut alpha = self.column(q, art_sign);
            {
                let _t = crate::profile::Timer::new(crate::profile::Phase::AlphaFtran);
                if self.lu.ftran(&mut alpha).is_err() {
                    return Err(LpStatus::Numerical);
                }
            }

            // Harris two-pass bounded ratio test. Entering moves by t≥0; basic
            // i has value v_i(t) = xb[i] − dir·t·α[i]. Pass 1 finds the largest
            // step keeping every basic within a small feasibility expansion δ of
            // its bound; pass 2 picks, among columns that truly block within that
            // step, the one with the largest pivot |α_i| (numerical stability),
            // which may push others up to δ past a bound — the accepted Harris
            // trade, with δ ≪ the 1e-6 feasibility tolerance used elsewhere. δ is
            // the EXPAND tolerance for this iteration (grows toward EXPAND_MAX).
            let delta_tol = expand_tol;
            // entering's own bound-flip cap (the step at which q hits its far bound)
            let cap = if self.ub[q] < INF && self.lb[q] > -INF {
                self.ub[q] - self.lb[q]
            } else {
                INF
            };
            // pass 1: largest step under δ-expanded bounds
            let mut t_pass1 = cap;
            for i in 0..m {
                let delta = -dir * alpha[i];
                let bi = self.basis[i];
                if delta < -self.tol {
                    let lb = self.lb[bi];
                    if lb > -INF {
                        let t = (xb[i] - lb + delta_tol) / (-delta);
                        if t < t_pass1 {
                            t_pass1 = t;
                        }
                    }
                } else if delta > self.tol {
                    let ub = self.ub[bi];
                    if ub < INF {
                        let t = (ub - xb[i] + delta_tol) / delta;
                        if t < t_pass1 {
                            t_pass1 = t;
                        }
                    }
                }
            }
            let t_pass1 = t_pass1.max(0.0);
            // pass 2: among true blockers with ratio ≤ t_pass1, take the max pivot
            let mut t_max = cap;
            let mut leave_slot: Option<usize> = None;
            let mut leave_to_upper = false;
            let mut best_pivot = 0.0f64;
            for i in 0..m {
                let delta = -dir * alpha[i];
                let bi = self.basis[i];
                let (t_true, to_upper) = if delta < -self.tol {
                    let lb = self.lb[bi];
                    if lb <= -INF {
                        continue;
                    }
                    (((xb[i] - lb) / (-delta)).max(0.0), false)
                } else if delta > self.tol {
                    let ub = self.ub[bi];
                    if ub >= INF {
                        continue;
                    }
                    (((ub - xb[i]) / delta).max(0.0), true)
                } else {
                    continue;
                };
                if t_true <= t_pass1 + self.tol && alpha[i].abs() > best_pivot {
                    best_pivot = alpha[i].abs();
                    leave_slot = Some(i);
                    leave_to_upper = to_upper;
                    t_max = t_true;
                }
            }
            if leave_slot.is_none() {
                t_max = cap; // no basic blocks → pure bound flip (or unbounded)
            }

            // EXPAND minimum step: on a degenerate (≈zero) blocking step, advance by
            // a guaranteed positive amount so the pivot strictly improves the
            // objective and the search cannot stall. The excursion introduced is
            // ≤ expand_tol (≤ 1e-7); the leaving variable is pinned exactly at its
            // bound as it exits, so nothing persists past the next exact refresh.
            // Bounded by the entering variable's own bound-flip distance `cap`.
            if leave_slot.is_some() {
                let t_min = (expand_tol / best_pivot.max(self.tol)).min(cap);
                if t_max < t_min {
                    t_max = t_min;
                    crate::profile::incr(crate::profile::Ctr::ExpandMinSteps);
                }
            }

            if t_max >= INF {
                // Capture the primal unbounded ray over the real columns (length
                // `n`): entering `q` moves by `dir`, each basic variable follows
                // `x_B -= dir·α`, and `A d = 0` by construction (α = B⁻¹A_q ⇒
                // A_q − B·α = 0). A caller verifies `A d = 0`, box-recession, and
                // `cᵀd < 0` before trusting it, so an artificial leaking in (it is
                // pinned to [0,0] in phase 2, so its ray entry is 0 anyway) cannot
                // make the check pass spuriously.
                let mut ray = vec![0.0; self.n];
                if q < self.n {
                    ray[q] = dir;
                }
                for i in 0..m {
                    let bi = self.basis[i];
                    if bi < self.n {
                        ray[bi] = -dir * alpha[i];
                    }
                }
                self.unbounded_ray = ray;
                return Err(LpStatus::Unbounded);
            }
            if t_max <= self.tol {
                stall += 1;
                crate::profile::incr(crate::profile::Ctr::DegeneratePivots);
            } else {
                stall = 0;
            }
            // Grow the EXPAND tolerance for the next iteration; reset periodically so
            // it never exceeds EXPAND_MAX (the excursions are cleared by the exact
            // x_B refresh on each refactorization, so the reset never loses ground).
            expand_tol += expand_incr;
            if expand_tol >= EXPAND_MAX {
                expand_tol = EXPAND_MIN;
            }
            crate::profile::incr(if is_phase1 {
                crate::profile::Ctr::Phase1Pivots
            } else {
                crate::profile::Ctr::Phase2Pivots
            });
            if bland {
                crate::profile::incr(crate::profile::Ctr::BlandActivations);
            }

            // Incremental x_B step: basic values move along −dir·α by t_max.
            let q_val = self.nb_value(q); // entering's value before the move
            for (i, xbi) in xb.iter_mut().enumerate() {
                *xbi -= dir * t_max * alpha[i];
            }

            match leave_slot {
                None => {
                    crate::profile::incr(crate::profile::Ctr::BoundFlips);
                    // bound flip: entering goes to its other bound, no basis change
                    self.stat[q] = if self.stat[q] == AT_LOWER {
                        AT_UPPER
                    } else {
                        AT_LOWER
                    };
                }
                Some(slot) => {
                    // Devex reference-weight update — uses the OLD basis (the LU
                    // still factorizes it until `update` below). Pivot element is
                    // α at the leaving row; it is nonzero by the ratio test.
                    let pivot = alpha[slot];
                    if !bland && pivot.abs() > self.tol {
                        let gamma_q = gamma[q];
                        let mut rho = vec![0.0; m];
                        rho[slot] = 1.0;
                        if self.lu.btran(&mut rho).is_ok() {
                            for j in 0..self.na {
                                if self.stat[j] != BASIC && j != q {
                                    let arj = self.col_dot(j, &rho, art_sign);
                                    let cand = (arj / pivot) * (arj / pivot) * gamma_q;
                                    if cand > gamma[j] {
                                        gamma[j] = cand;
                                    }
                                }
                            }
                            // The leaving variable becomes nonbasic with a fresh weight.
                            let leaving0 = self.basis[slot];
                            gamma[leaving0] = (gamma_q / (pivot * pivot)).max(1.0);
                            // Reframe (reset weights) if drift makes them blow up.
                            if gamma[leaving0] > 1e10 {
                                for g in gamma.iter_mut() {
                                    *g = 1.0;
                                }
                            }
                        }
                    }
                    let leaving = self.basis[slot];
                    self.stat[leaving] = if leave_to_upper { AT_UPPER } else { AT_LOWER };
                    self.slot_of[leaving] = -1;
                    self.basis[slot] = q;
                    self.slot_of[q] = slot as i64;
                    self.stat[q] = BASIC;
                    // the entering variable now occupies this slot
                    xb[slot] = q_val + dir * t_max;
                    // factorization update with the entering column
                    let col = self.column(q, art_sign);
                    let need_refac = {
                        let _t = crate::profile::Timer::new(crate::profile::Phase::FtUpdate);
                        self.lu.update(slot, &col).is_err()
                    };
                    updates += 1;
                    // Refactorize on: a failed FT update, the hard 48-update cap, or
                    // the adaptive work gate (accumulated bump-update work exceeded
                    // the factor's nnz — the wide-McCormick-bump regime).
                    let work_gate =
                        refac_work_budget > 0 && self.lu.ft_update_work() > refac_work_budget;
                    if need_refac || updates >= 48 || work_gate {
                        // THRU-5: attribute the trigger (priority: FT-fail > cap > gate).
                        crate::profile::incr(if need_refac {
                            // ENGINE-1 (#557): split the FT-fail by feral's own
                            // RefactorCause so the "accuracy-necessary vs
                            // over-conservative" fork is measured. No-op without
                            // DISCOPT_PROFILE (last_refactor() is a cheap field read).
                            if crate::profile::enabled() {
                                use feral::RefactorCause::*;
                                match self.lu.last_refactor().map(|(c, _)| c) {
                                    Some(Growth) => {
                                        crate::profile::incr(crate::profile::Ctr::RefacFtGrowth)
                                    }
                                    Some(TinyPivot) => {
                                        crate::profile::incr(crate::profile::Ctr::RefacFtTinyPivot)
                                    }
                                    Some(Singular) => {
                                        crate::profile::incr(crate::profile::Ctr::RefacFtSingular)
                                    }
                                    // UpdateBudget can't reach here (discopt's 48-cap
                                    // trips first) and None means the dense path;
                                    // leave the sub-split silent, the aggregate still
                                    // counts it.
                                    _ => {}
                                }
                            }
                            crate::profile::Ctr::RefacFtFail
                        } else if updates >= 48 {
                            crate::profile::Ctr::RefacCap
                        } else {
                            crate::profile::Ctr::RefacWorkGate
                        });
                        crate::profile::incr(crate::profile::Ctr::Refactorizations);
                        let _t = crate::profile::Timer::new(crate::profile::Phase::Refactorize);
                        if self.refactorize(art_sign).is_err() {
                            return Err(LpStatus::Numerical);
                        }
                        xb = self
                            .basic_values(art_sign)
                            .map_err(|_| LpStatus::Numerical)?;
                        updates = 0;
                        refac_work_budget = self.lu.factor_nnz();
                    }
                }
            }
        }
        Err(LpStatus::IterLimit)
    }

    fn assemble(mut self, status: LpStatus, art_sign: &[f64]) -> LpSolve {
        let xb = self
            .basic_values(art_sign)
            .unwrap_or_else(|_| vec![0.0; self.m]);
        let mut x = self.assemble_x(&xb);
        let mut obj: f64 = (0..self.n).map(|j| self.c[j] * x[j]).sum();

        // Final feasibility audit before certifying Optimal. x_B is maintained
        // incrementally between the ~48-pivot refactorizations and the Harris
        // ratio test permits small bound excursions, so the returned point can
        // drift. Verify it actually satisfies its bounds and Ax=b.
        //
        // On a *row-residual* failure (Ax=b drift — accumulated update error /
        // ill-conditioning), attempt an in-engine refined recovery: recompute x_B
        // from a fresh, refinement-polished factorization of the same basis and
        // re-audit; only downgrade to Numerical if it still fails. The re-audit is
        // the same soundness gate, so recovery can only rescue a solve that would
        // otherwise be Numerical — never certify a wrong "Optimal".
        //
        // A *bounds* excursion is NOT retried: it is a Harris ratio-test artefact,
        // not a solve-accuracy problem, so a sharper x_B cannot pull the variable
        // back inside its bound (measured: on the lifted corpus every audit failure
        // was a bounds excursion, so refining them was pure wasted work — discopt#364).
        // Either way a genuine Numerical flows to the caller as before (the warm
        // path's cold fallback, or the MILP driver decertifying the gap and branching).
        let audit = |slf: &Simplex, x: &[f64]| {
            audit_feasibility(&slf.cols, slf.m, slf.n, slf.b, &slf.lb, &slf.ub, x)
        };
        let status = if status == LpStatus::Optimal {
            match audit(&self, &x) {
                Feasibility::Ok => LpStatus::Optimal,
                // Bound excursion: refinement can't help — downgrade directly.
                Feasibility::Bounds => LpStatus::Numerical,
                // Row residual: the failure mode iterative refinement can recover.
                Feasibility::Rows => {
                    crate::profile::incr(crate::profile::Ctr::RefinedRecoveryAttemptsPrimal);
                    match self.refined_basic_values(art_sign) {
                        Some(xb_r) => {
                            let x_r = self.assemble_x(&xb_r);
                            if audit(&self, &x_r) == Feasibility::Ok {
                                crate::profile::incr(
                                    crate::profile::Ctr::RefinedRecoveryRescuesPrimal,
                                );
                                x = x_r;
                                obj = (0..self.n).map(|j| self.c[j] * x[j]).sum();
                                LpStatus::Optimal
                            } else {
                                LpStatus::Numerical
                            }
                        }
                        None => LpStatus::Numerical,
                    }
                }
            }
        } else {
            status
        };

        // Certificate vector `y = B⁻ᵀ c_B` (length m). On `Optimal` use the real
        // objective costs — these are the row duals feeding a safe (never-too-high)
        // dual bound. On `Infeasible` use the phase-1 costs (1 on artificials, 0
        // elsewhere): the phase-1 multipliers form a Farkas infeasibility ray. The
        // btran is a read-only solve against the final factorization, so it neither
        // perturbs the basis nor the returned `x`/`obj`. Empty on any other status
        // (no meaningful certificate) — and empty too if the btran fails, so the
        // caller simply falls back rather than trusting an unsound vector.
        let dual: Vec<f64> = match status {
            LpStatus::Optimal | LpStatus::Infeasible => {
                let mut y: Vec<f64> = self
                    .basis
                    .iter()
                    .map(|&j| {
                        if status == LpStatus::Optimal {
                            if j < self.n {
                                self.c[j]
                            } else {
                                0.0
                            }
                        } else if j >= self.n {
                            1.0 // basic artificial: phase-1 cost
                        } else {
                            0.0
                        }
                    })
                    .collect();
                if self.lu.btran(&mut y).is_ok() {
                    y
                } else {
                    Vec::new()
                }
            }
            // #517: on a phase-2 `Numerical` breakdown also export the Optimal-style
            // row-dual candidate `y = B⁻ᵀc_B` from the final basis. It is a *candidate*
            // the caller verifies (a Neumaier–Shcherbina safe bound is valid for ANY
            // `y`, so a drifted-basis dual only loosens it, never lifts it above the
            // optimum) — it lets the hda-class no-bound nodes still contribute a sound
            // lower bound. `IterLimit` and the `failed()` path keep the empty dual
            // (no usable factorization to btran against).
            LpStatus::Numerical => {
                let mut y: Vec<f64> = self
                    .basis
                    .iter()
                    .map(|&j| if j < self.n { self.c[j] } else { 0.0 })
                    .collect();
                if self.lu.btran(&mut y).is_ok() {
                    y
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        };
        let ray = std::mem::take(&mut self.unbounded_ray);

        // Export a *complete*, row-ordered basis of real columns (length m).
        //
        // Phase 2 can leave an artificial (column ≥ self.n) basic at value 0 on a
        // degenerate/redundant row — common on the heavily-redundant lifted
        // McCormick relaxations. Naively dropping those slots (the old behaviour)
        // returned fewer than m basic vars, which the warm-start consumer and the
        // dual simplex both reject (they require exactly m), so every
        // cutting-plane re-solve silently cold-started. Substitute that row's own
        // zero-valued structural singleton — its slack column in the [A_ub|I]
        // standard form the warm path uses. Swapping the basic artificial (±eᵢ at
        // 0) for a nonbasic singleton that sits in the same row at value 0 leaves
        // B's support and x_B bit-identical (the entering column was contributing
        // 0 to the RHS and its new basic value is 0), so this is a pure
        // representation fix — the optimum and bound are unchanged.
        let mut slack_for_row: Vec<i64> = vec![-1; self.m];
        for j in 0..self.n {
            if self.stat[j] == BASIC || self.nb_value(j) != 0.0 {
                continue;
            }
            let (rows, _) = self.cols.col(j);
            if rows.len() == 1 && slack_for_row[rows[0]] < 0 {
                slack_for_row[rows[0]] = j as i64;
            }
        }
        let mut col_status: Vec<i8> = (0..self.n).map(|j| self.stat[j]).collect();
        let mut basic_vars: Vec<usize> = Vec::with_capacity(self.m);
        for i in 0..self.m {
            let bcol = self.basis[i];
            if bcol < self.n {
                basic_vars.push(bcol);
                continue;
            }
            // Basic artificial in this slot: it covers constraint row
            // `r = bcol - self.n` (column n+r is ±eᵣ), which need not equal the
            // slot index `i` once pivots have permuted the basis. Substitute that
            // row's zero-valued singleton (its slack).
            let r = bcol - self.n;
            if slack_for_row[r] >= 0 {
                let s = slack_for_row[r] as usize;
                col_status[s] = BASIC;
                basic_vars.push(s);
            }
            // else: no real substitute available (non-[A_ub|I] caller); the basis
            // stays short and the warm path declines it, exactly as before.
        }
        LpSolve {
            status,
            x,
            obj,
            basis: Basis {
                col_status,
                basic_vars,
            },
            iters: 0,
            dual,
            ray,
        }
    }

    fn failed(self) -> LpSolve {
        let n = self.n;
        LpSolve {
            status: LpStatus::Numerical,
            x: vec![0.0; n],
            obj: 0.0,
            basis: Basis {
                col_status: vec![AT_LOWER; n],
                basic_vars: Vec::new(),
            },
            iters: 0,
            dual: Vec::new(),
            ray: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(a: &[f64], m: usize, n: usize, b: &[f64], c: &[f64], l: &[f64], u: &[f64]) -> LpSolve {
        let lp = LpView { a, m, n, c, l, u };
        solve_lp(&lp, b, &SimplexOptions::default())
    }

    #[test]
    fn knapsack_lp_relaxation() {
        // max 16(x0+x1+x2+x3) s.t. 5Σx + s = 9, x∈[0,1], s∈[0,inf]
        // min -16Σx. Optimum: Σx = 9/5 = 1.8 → obj -28.8.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [-16.0, -16.0, -16.0, -16.0, 0.0];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let r = solve(&a, 1, 5, &[9.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - (-28.8)).abs() < 1e-6, "obj {}", r.obj);
        assert!((5.0 * r.x[..4].iter().sum::<f64>() + r.x[4] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn two_constraint_lp() {
        // min -x0 - 2 x1 s.t. x0+x1+s0=4, x0+3x1+s1=6, x∈[0,inf], s∈[0,inf]
        // Optimum at x0=3,x1=1 → -5? check: x0+x1=4, x0+3x1=6 → x1=1,x0=3, obj -5.
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [INF; 4];
        let r = solve(&a, 2, 4, &[4.0, 6.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - (-5.0)).abs() < 1e-6, "obj {}", r.obj);
        assert!((r.x[0] - 3.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equality_constraint_feasible() {
        // min x0+x1 s.t. x0+x1=2 (equality, no slack), x∈[0,inf]. obj 2.
        let a = [1.0, 1.0];
        let c = [1.0, 1.0];
        let l = [0.0, 0.0];
        let u = [INF, INF];
        let r = solve(&a, 1, 2, &[2.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - 2.0).abs() < 1e-6, "obj {}", r.obj);
    }

    #[test]
    fn infeasible_detected() {
        // x0 + s = 1, s∈[0,inf], x0∈[2,inf]: x0≥2 but x0≤1 → infeasible.
        let a = [1.0, 1.0];
        let c = [1.0, 0.0];
        let l = [2.0, 0.0];
        let u = [INF, INF];
        let r = solve(&a, 1, 2, &[1.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Infeasible);
    }

    /// A tiny redundant-row feasible LP that used to false-infeasible (discopt
    /// C-39). Two identical rows `x + s = 1` (a degenerate/redundant pair) with a
    /// third `x + s3 = 1`: phase-1 can leave an artificial basic on the redundant
    /// row. The LP is trivially feasible (`x = 0, s = s3 = 1`); the fix must return
    /// Optimal (or at worst a non-`Infeasible` Numerical), never `Infeasible`.
    #[test]
    fn c39_redundant_row_not_false_infeasible() {
        // Columns: x, s1, s2, s3 (all ≥ 0). Rows:
        //   x + s1      = 1
        //   x      + s2 = 1   (redundant with row 0 up to the slack identity)
        //   x           + s3 = 1
        let a = [
            1.0, 1.0, 0.0, 0.0, // row 0
            1.0, 0.0, 1.0, 0.0, // row 1
            1.0, 0.0, 0.0, 1.0, // row 2
        ];
        let c = [1.0, 0.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [INF; 4];
        let r = solve(&a, 3, 4, &[1.0, 1.0, 1.0], &c, &l, &u);
        assert_ne!(
            r.status,
            LpStatus::Infeasible,
            "feasible redundant-row LP must not be reported Infeasible"
        );
    }

    /// A genuinely-infeasible node LP whose Farkas ray touches `−1` **surplus** slacks
    /// (`≥`-constraints in `[A | −I]` standard form), the class that regressed the AMP
    /// piecewise/trig-square relaxation solves after C-39 (main #554). The LP is
    /// infeasible (HiGHS agrees, supplying a dual ray with `g₀ > 0`), but the C-39
    /// slack recovery only knew `+1` slacks, so the ray's open surplus-slack term
    /// collapsed `g₀` to `−∞`, the certification bailed, and the engine returned the
    /// honest-but-non-fathoming `Numerical` instead of `Infeasible`. The MILP B&B then
    /// could not prune the infeasible node, grinding to the node cap → the solve
    /// reported `iteration_limit` where `optimal` was expected.
    ///
    /// Fixture: a captured node LP (`m=159, n=186`, 12 `−1` surplus slacks on the
    /// `≥`-rows) from the trig-square piecewise relaxation of
    /// `test_continuous_trig_square_uses_direct_piecewise_relaxation`. The `−1`
    /// surplus-slack recovery (`s_i ≤ max_x (A x)_i − b_i`) makes the ray certify.
    /// FAILS pre-fix (returns `Numerical`); passes after (`Infeasible`).
    /// #649 regression: the bchoco06 root uniform relaxation (equilibrated +
    /// flushed, standard form 833×1182). The shipped `solve_lp_warm_csc` path
    /// applies its own pow2 `Scaling::from_sparse` on top; on this already-
    /// equilibrated LP that scaled solve unscales just past the feasibility audit
    /// and returned `Numerical` (no bound) — the bchoco06 "no finite dual bound"
    /// defect (issue #649). The unscaled cold solve of the identical LP is clean
    /// (`solve_lp_cols` → Optimal, obj ≈ −1.0, matching HiGHS's −0.99999). The
    /// unscaled fallback in `solve_lp_warm_csc` must recover that bound.
    ///
    /// Fails pre-fix (`solve_lp_warm_csc` → Numerical); passes after.
    #[test]
    fn bchoco06_illcond_scaled_path_recovers_bound_649() {
        let json = include_str!("testdata/bchoco06_illcond_lp.json");
        let (m, n, col_ptr, row_idx, vals, c, l, u, b) = parse_lp_fixture(json);

        // Ground truth: the UNSCALED cold solve (no Rust scaling) is clean.
        let sp_cold = SparseCols::from_csc(col_ptr.clone(), row_idx.clone(), vals.clone());
        let cold = solve_lp_cols(sp_cold, m, n, &c, &l, &u, &b, &SimplexOptions::default());
        assert_eq!(
            cold.status,
            LpStatus::Optimal,
            "unscaled cold solve of the bchoco06 root LP must be Optimal"
        );
        assert!(
            (cold.obj - (-0.9999893)).abs() < 1e-4,
            "unscaled obj {} must match the HiGHS bound ≈ -0.99999",
            cold.obj
        );

        // The shipped scaled path (Scaling::from_sparse ON) must ALSO reach the
        // bound now, via the #649 unscaled fallback — not Numerical/IterLimit.
        let sp = SparseCols::from_csc(col_ptr, row_idx, vals);
        let scaled = crate::lp::simplex::dual::solve_lp_warm_csc(
            sp,
            m,
            n,
            &c,
            &l,
            &u,
            &b,
            None,
            &SimplexOptions::default(),
        );
        assert_eq!(
            scaled.status,
            LpStatus::Optimal,
            "scaled solve_lp_warm_csc must recover Optimal via the #649 unscaled \
             fallback (was Numerical), got {:?}",
            scaled.status
        );
        assert!(
            (scaled.obj - cold.obj).abs() <= 1e-6 * (1.0 + cold.obj.abs()),
            "scaled-path obj {} must equal the unscaled cold obj {}",
            scaled.obj,
            cold.obj
        );
    }

    #[test]
    fn c39_surplus_slack_infeasible_certifies() {
        let json = include_str!("testdata/c39_surplus_slack_infeasible_lp.json");
        let (m, n, col_ptr, row_idx, vals, c, l, u, b) = parse_lp_fixture(json);
        let sp = SparseCols::from_csc(col_ptr, row_idx, vals);
        let r = solve_lp_cols(sp, m, n, &c, &l, &u, &b, &SimplexOptions::default());
        assert_eq!(
            r.status,
            LpStatus::Infeasible,
            "genuinely-infeasible node LP with −1 surplus slacks (HiGHS-infeasible) \
             must certify Infeasible via surplus-slack recovery, got {:?}",
            r.status
        );
    }

    /// A genuinely-infeasible node LP whose Farkas ray touches **scaled** surplus
    /// slacks with coefficient `−0.5` (not just `±1`) — the perf regression the
    /// `±1`-only recovery caused (main #558 follow-up). Geometric-mean equilibration
    /// scales each `≥`-row slack by its row/col factor, so the surplus slacks appear
    /// with coefficients like `−0.5`; the `±1`-only recovery left those columns open,
    /// so `g₀` collapsed, the first-ray certification failed, and every such node LP
    /// paid the full `drive_out_basic_artificials` loop (~34 ms) before returning
    /// `Numerical` anyway — ~1500 wasted drive-outs made
    /// `test_gas_square_difference_tightening_strengthens_root_relaxation` ~20×
    /// slower (69 s vs 3.4 s single-threaded). The **general-coefficient** slack
    /// recovery (`s_j ≤ (b_i − min/max_x Σ_{k≠j} A_ik x_k)/a`) certifies on the first
    /// ray, so drive_out is never reached.
    ///
    /// Fixture: a captured node LP (`m=531, n=685`, mixed `−1` and `−0.5` scaled
    /// surplus slacks) from the gas-square relaxation; HiGHS reports it infeasible
    /// with `g₀ > 0`. FAILS pre-generalization (returns `Numerical` after a wasted
    /// drive-out); passes after (`Infeasible`, first ray).
    #[test]
    fn c39_scaled_surplus_slack_infeasible_certifies() {
        let json = include_str!("testdata/c39_scaled_surplus_slack_infeasible_lp.json");
        let (m, n, col_ptr, row_idx, vals, c, l, u, b) = parse_lp_fixture(json);
        let sp = SparseCols::from_csc(col_ptr, row_idx, vals);
        let r = solve_lp_cols(sp, m, n, &c, &l, &u, &b, &SimplexOptions::default());
        assert_eq!(
            r.status,
            LpStatus::Infeasible,
            "genuinely-infeasible node LP with scaled (−0.5) surplus slacks \
             (HiGHS-infeasible) must certify Infeasible via general-coefficient \
             slack recovery, got {:?}",
            r.status
        );
    }

    /// The captured McCormick node LP from `kall_circles_c8a` on which the in-house
    /// two-phase simplex used to return a **numerical false `Infeasible`** (discopt
    /// C-39): HiGHS solves the identical LP feasible, but phase-1's reduced-cost
    /// optimality test terminated with basic artificials at a positive value and no
    /// improving (strictly-negative-reduced-cost) column, so a feasible node was
    /// falsely fathomed (the C-38 false-optimal, the C-42 cut-inherit truncation).
    /// After the fix the cold solve must NOT return `Infeasible` — either it solves
    /// (Optimal) or, on this heavily-degenerate LP, returns the honest, non-fathoming
    /// `Numerical`; the one thing it may never do is emit a false `Infeasible` (which
    /// would require a verified Farkas ray this feasible LP cannot supply).
    ///
    /// FAILS on the pre-fix engine (returns `Infeasible`); passes after.
    #[test]
    fn c39_kall_circles_captured_lp_not_false_infeasible() {
        let json = include_str!("testdata/c39_kall_false_infeasible_lp.json");
        let (m, n, col_ptr, row_idx, vals, c, l, u, b) = parse_lp_fixture(json);
        let sp = SparseCols::from_csc(col_ptr, row_idx, vals);
        let r = solve_lp_cols(sp, m, n, &c, &l, &u, &b, &SimplexOptions::default());
        assert_ne!(
            r.status,
            LpStatus::Infeasible,
            "captured feasible McCormick LP must never be reported Infeasible \
             (HiGHS solves it feasible); got {:?}",
            r.status
        );
    }

    /// Minimal JSON parser for the standard-form LP fixture (flat numeric arrays;
    /// no serde dependency). Fields: `m`, `n`, `col_ptr`, `row_idx`, `vals`, `c`,
    /// `l`, `u`, `b`.
    #[allow(clippy::type_complexity)]
    fn parse_lp_fixture(
        json: &str,
    ) -> (
        usize,
        usize,
        Vec<usize>,
        Vec<usize>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        fn scalar(json: &str, key: &str) -> f64 {
            let pat = format!("\"{key}\":");
            let start = json.find(&pat).unwrap() + pat.len();
            let rest = &json[start..];
            let end = rest.find([',', '}']).unwrap();
            rest[..end].trim().parse().unwrap()
        }
        fn arr_f64(json: &str, key: &str) -> Vec<f64> {
            let pat = format!("\"{key}\":");
            let at = json.find(&pat).unwrap() + pat.len();
            let rest = &json[at..];
            let lb = rest.find('[').unwrap() + 1;
            let rest = &rest[lb..];
            let end = rest.find(']').unwrap();
            rest[..end]
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().unwrap())
                .collect()
        }
        fn arr_usize(json: &str, key: &str) -> Vec<usize> {
            arr_f64(json, key).into_iter().map(|v| v as usize).collect()
        }
        (
            scalar(json, "m") as usize,
            scalar(json, "n") as usize,
            arr_usize(json, "col_ptr"),
            arr_usize(json, "row_idx"),
            arr_f64(json, "vals"),
            arr_f64(json, "c"),
            arr_f64(json, "l"),
            arr_f64(json, "u"),
            arr_f64(json, "b"),
        )
    }

    #[test]
    fn unbounded_detected() {
        // min -x0 s.t. x0 - s = 0, x0∈[0,inf], s∈[0,inf]  → x0 can grow → unbounded.
        let a = [1.0, -1.0];
        let c = [-1.0, 0.0];
        let l = [0.0, 0.0];
        let u = [INF, INF];
        let r = solve(&a, 1, 2, &[0.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Unbounded);
    }

    #[test]
    fn upper_bound_active() {
        // min -x0 s.t. x0 + s = 10, x0∈[0,3], s∈[0,inf]. Optimum x0=3, obj -3.
        let a = [1.0, 1.0];
        let c = [-1.0, 0.0];
        let l = [0.0, 0.0];
        let u = [3.0, INF];
        let r = solve(&a, 1, 2, &[10.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.obj - (-3.0)).abs() < 1e-6, "obj {}", r.obj);
        assert!((r.x[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn feasibility_audit_accepts_and_rejects() {
        // Row: x0 + x1 = 4, with x0,x1 ∈ [0, 5]. (n real cols = 2)
        let a = [1.0, 1.0];
        let cols = SparseCols::from_dense(&a, 1, 2);
        let b = [4.0];
        let l = [0.0, 0.0];
        let u = [5.0, 5.0];
        // Exact feasible point passes.
        assert!(solution_within_tolerance(
            &cols,
            1,
            2,
            &b,
            &l,
            &u,
            &[1.0, 3.0]
        ));
        // Ax=b drift beyond tolerance is rejected (would be a false "Optimal").
        assert!(!solution_within_tolerance(
            &cols,
            1,
            2,
            &b,
            &l,
            &u,
            &[1.0, 3.1]
        ));
        // A bound excursion beyond tolerance is rejected.
        assert!(!solution_within_tolerance(
            &cols,
            1,
            2,
            &b,
            &l,
            &u,
            &[-1.0, 5.0]
        ));
        // Tiny within-tolerance noise still passes.
        assert!(solution_within_tolerance(
            &cols,
            1,
            2,
            &b,
            &l,
            &u,
            &[1.0, 3.0 + 1e-9]
        ));
    }

    // --- certificate vectors (issue #356) ------------------------------------

    const CERT_INF: f64 = 1e20;

    /// Safe lower bound `g(y) = bᵀy + Σ_k min_{z_k∈[l,u]} (c−Aᵀy)_k z_k` from
    /// free-sign multipliers `y`. `<= true optimum` for any `y` (weak duality).
    fn safe_bound(
        y: &[f64],
        c: &[f64],
        a: &[f64],
        m: usize,
        n: usize,
        b: &[f64],
        l: &[f64],
        u: &[f64],
    ) -> f64 {
        let mut g = b.iter().zip(y).map(|(bi, yi)| bi * yi).sum::<f64>();
        for j in 0..n {
            let aty: f64 = (0..m).map(|i| a[i * n + j] * y[i]).sum();
            let rc = c[j] - aty;
            if rc > 0.0 {
                g += if l[j] <= -CERT_INF {
                    f64::NEG_INFINITY
                } else {
                    rc * l[j]
                };
            } else if rc < 0.0 {
                g += if u[j] >= CERT_INF {
                    f64::NEG_INFINITY
                } else {
                    rc * u[j]
                };
            }
        }
        g
    }

    #[test]
    fn optimal_duals_reproduce_objective_via_safe_bound() {
        // min -x0 - 2x1 s.t. x0+x1+s=4, x∈[0,5], s∈[0,inf]. Optimum -8.
        let a = [1.0, 1.0, 1.0];
        let c = [-1.0, -2.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [5.0, 5.0, CERT_INF];
        let r = solve(&a, 1, 3, &[4.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert_eq!(r.dual.len(), 1);
        let g = safe_bound(&r.dual, &c, &a, 1, 3, &[4.0], &l, &u);
        // Safe bound from the simplex's own duals reproduces the optimum and is
        // never above it (the soundness property the spatial-B&B relies on).
        assert!((g - r.obj).abs() < 1e-9, "safe bound {g} vs obj {}", r.obj);
        assert!(g <= r.obj + 1e-9);
    }

    #[test]
    fn infeasible_emits_a_verifiable_farkas_ray() {
        // x0 + s = 1, s∈[0,inf], x0∈[2,inf): x0≥2 but x0≤1 → infeasible.
        let a = [1.0, 1.0];
        let c = [1.0, 0.0];
        let l = [2.0, 0.0];
        let u = [CERT_INF, CERT_INF];
        let r = solve(&a, 1, 2, &[1.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Infeasible);
        assert_eq!(r.dual.len(), 1);
        // Farkas: for some sign, the c=0 safe bound g0(±y) > 0 proves emptiness.
        let zeros = [0.0, 0.0];
        let pos = safe_bound(&r.dual, &zeros, &a, 1, 2, &[1.0], &l, &u);
        let neg_y: Vec<f64> = r.dual.iter().map(|v| -v).collect();
        let neg = safe_bound(&neg_y, &zeros, &a, 1, 2, &[1.0], &l, &u);
        assert!(
            pos > 0.0 || neg > 0.0,
            "neither sign certifies: +{pos} -{neg}"
        );
    }

    #[test]
    fn unbounded_emits_a_valid_primal_ray() {
        // min -x0 s.t. x0 - s = 0, x0,s∈[0,inf) → unbounded along x0=s growing.
        let a = [1.0, -1.0];
        let c = [-1.0, 0.0];
        let l = [0.0, 0.0];
        let u = [CERT_INF, CERT_INF];
        let r = solve(&a, 1, 2, &[0.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Unbounded);
        assert_eq!(r.ray.len(), 2);
        // A d = 0 (stays on the constraint), and cᵀd < 0 (objective decreases).
        let ad: f64 = (0..2).map(|j| a[j] * r.ray[j]).sum();
        let cd: f64 = (0..2).map(|j| c[j] * r.ray[j]).sum();
        assert!(ad.abs() < 1e-9, "A·d = {ad} (should be 0)");
        assert!(cd < -1e-9, "c·d = {cd} (should be < 0)");
    }

    #[test]
    fn ill_scaled_optimum_safe_bound_not_above_truth() {
        // Wide-coefficient LP (range 1e9 → equilibration fires). The duals are
        // unscaled, so the safe bound stays <= the true optimum.
        let a = [1e9, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let c = [-1.0, -1.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0, 0.0];
        let u = [1.0, 10.0, CERT_INF, CERT_INF];
        let b = [1e9, 5.0];
        let r = solve(&a, 2, 4, &b, &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        assert_eq!(r.dual.len(), 2);
        let g = safe_bound(&r.dual, &c, &a, 2, 4, &b, &l, &u);
        assert!(g <= r.obj + 1e-6, "safe bound {g} above obj {}", r.obj);
        assert!(
            (g - r.obj).abs() < 1e-3,
            "safe bound {g} too loose vs {}",
            r.obj
        );
    }

    #[test]
    fn audit_downgrades_drifted_optimal_to_numerical() {
        // A clean solve still certifies Optimal (audit must not false-trip).
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [INF; 4];
        let r = solve(&a, 2, 4, &[4.0, 6.0], &c, &l, &u);
        assert_eq!(r.status, LpStatus::Optimal);
        // The returned optimum genuinely satisfies the audit predicate.
        let cols = SparseCols::from_dense(&a, 2, 4);
        assert!(solution_within_tolerance(
            &cols,
            2,
            4,
            &[4.0, 6.0],
            &l,
            &u,
            &r.x
        ));
    }
}
