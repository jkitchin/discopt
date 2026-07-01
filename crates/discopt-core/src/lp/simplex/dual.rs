//! Bounded-variable **dual** simplex — the warm-start engine.
//!
//! After a branch-and-bound node tightens one variable bound, the parent's
//! optimal basis is still **dual-feasible** (reduced costs unchanged) but may be
//! **primal-infeasible** (a basic variable now violates the tightened bound).
//! The dual simplex repairs this in a few pivots — the cheap re-optimization
//! that makes node throughput competitive, versus a cold solve from scratch.
//!
//! [`solve_lp_warm`] factorizes the given basis, checks dual feasibility, and
//! runs dual pivots to primal feasibility (= optimality, since dual feasibility
//! is maintained). On *any* difficulty — a dual-infeasible starting basis, a
//! failed dual ratio test, a singular/over-updated factorization, or the
//! iteration cap — it falls back to a cold [`super::primal::solve_lp`], so the
//! result is always correct; warm-start only ever saves time, never risks it.

#![allow(clippy::needless_range_loop)]

use super::linsolve::{FeralLU, LinearSolver};
use super::primal::{solve_lp_cols, solve_lp_scaled};
use super::scaling::{ScaledLp, Scaling};
use super::sparse::SparseCols;
use super::{LpSolve, LpStatus, SimplexOptions};
use crate::lp::basis::{Basis, AT_LOWER, AT_UPPER, BASIC};
use crate::lp::crossover::LpView;

const INF: f64 = 1e20;

/// Re-optimize `min cᵀx s.t. A x = b, l ≤ x ≤ u` from a warm `start` basis via
/// the dual simplex, falling back to a cold solve on any difficulty.
///
/// Like [`solve_lp`](super::solve_lp), an ill-scaled matrix is equilibrated
/// first so the basis factorization stays well-conditioned, and the scaled
/// solution is mapped back. The warm-start basis is scaling-invariant (a column
/// is basic or not regardless of its scale) and the factors come from `A` alone,
/// so the basis a child inherits from its parent stays valid across the tree.
pub fn solve_lp_warm(lp: &LpView<'_>, b: &[f64], start: &Basis, opts: &SimplexOptions) -> LpSolve {
    match ScaledLp::maybe_new(lp, b) {
        Some(scaled) => {
            let view = scaled.view();
            let mut sol = solve_lp_warm_scaled(&view, scaled.b(), start, opts);
            scaled.unscale_x(&mut sol.x);
            // Map the certificate vectors back to the original space too (see the
            // matching note in `primal::solve_lp`): the duals/Farkas ray are in
            // scaled coordinates and must align with the unscaled A/b the caller
            // checks against.
            scaled.unscale_dual(&mut sol.dual);
            scaled.unscale_ray(&mut sol.ray);
            sol
        }
        None => solve_lp_warm_scaled(lp, b, start, opts),
    }
}

/// Sparse-native warm LP solve from a CSC matrix: equilibrate (sparse factors +
/// in-place scaling, `O(nnz)`), warm dual-simplex re-optimize from `start` (or
/// cold-solve when absent / the basis is unusable), then map the solution and
/// certificates back through the scaling. The whole path avoids materializing the
/// dense `m×n` matrix that [`solve_lp_warm`] builds via `from_dense` — the lifted
/// relaxations are ~0.3% dense, so this is the difference between scanning 54M
/// entries and touching ~19k per solve. `start` is `(col_status, basic_vars)`.
#[allow(clippy::too_many_arguments)]
pub fn solve_lp_warm_csc(
    mut sp: SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    start: Option<&Basis>,
    opts: &SimplexOptions,
) -> LpSolve {
    match Scaling::from_sparse(&sp, m, n) {
        Some(scaling) => {
            scaling.scale_cols(&mut sp);
            let c_s = scaling.scale_c(c);
            let b_s = scaling.scale_b(b);
            let l_s = scaling.scale_lower(l);
            let u_s = scaling.scale_upper(u);
            let mut sol = solve_csc_core(&sp, m, n, &c_s, &l_s, &u_s, &b_s, start, opts);
            // Map x and the dual/Farkas/primal-ray certificates back to the
            // original space (see the matching note in `solve_lp_warm`).
            scaling.unscale_x(&mut sol.x);
            scaling.unscale_dual(&mut sol.dual);
            scaling.unscale_ray(&mut sol.ray);
            sol
        }
        None => solve_csc_core(&sp, m, n, c, l, u, b, start, opts),
    }
}

/// Warm dual (or cold) solve of an already-scaled CSC LP. Shared by the scaled and
/// unscaled branches of [`solve_lp_warm_csc`].
#[allow(clippy::too_many_arguments)]
fn solve_csc_core(
    sp: &SparseCols,
    m: usize,
    n: usize,
    c: &[f64],
    l: &[f64],
    u: &[f64],
    b: &[f64],
    start: Option<&Basis>,
    opts: &SimplexOptions,
) -> LpSolve {
    match start {
        Some(basis) => match PreparedDual::prepare_cols(sp, m, n, c, l, u, basis, opts) {
            Some(p) => p.reoptimize(l, u, b, opts),
            // Unusable warm basis (wrong size / singular / dual-infeasible): cold.
            None => solve_lp_cols(sp.clone(), m, n, c, l, u, b, opts),
        },
        None => solve_lp_cols(sp.clone(), m, n, c, l, u, b, opts),
    }
}

/// Warm dual re-optimization on an already-equilibrated (or known well-scaled)
/// LP, with the cold primal fallback (also on the scaled matrix, so it is never
/// scaled twice). Like [`solve_lp_scaled`], the B&B driver calls this directly
/// when it has equilibrated the working matrix once and shares it across nodes;
/// the caller owns the [`scaling::Scaling`] and unscales the returned `x`.
pub fn solve_lp_warm_scaled(
    lp: &LpView<'_>,
    b: &[f64],
    start: &Basis,
    opts: &SimplexOptions,
) -> LpSolve {
    let sp = SparseCols::from_dense(lp.a, lp.m, lp.n);
    solve_lp_warm_scaled_csc(lp, b, start, opts, &sp)
}

/// As [`solve_lp_warm_scaled`], but reusing a caller-built CSC view of `lp.a`
/// instead of rebuilding it. The B&B driver solves a whole batch of nodes
/// against one constant working matrix (cuts are folded in only between batches),
/// so it builds the CSC **once per batch** and shares it across every node solve
/// here — eliminating the per-node `SparseCols::from_dense` O(m·n) rebuild, which
/// is otherwise pure repeated work (it scans the same dense matrix each node).
/// `sp` must be the CSC of exactly `lp.a` (same `m`, `n`, row-major); a mismatch
/// would silently solve the wrong system, so the entry point is internal.
pub fn solve_lp_warm_scaled_csc(
    lp: &LpView<'_>,
    b: &[f64],
    start: &Basis,
    opts: &SimplexOptions,
    sp: &SparseCols,
) -> LpSolve {
    match PreparedDual::prepare(lp, start, opts, sp) {
        Some(p) => p.reoptimize(lp.l, lp.u, b, opts),
        None => solve_lp_scaled(lp, b, opts), // safe fallback — always correct
    }
}


/// A basis factorization prepared once for repeated dual re-optimizations that
/// differ only in their bounds and right-hand side.
///
/// `prepare` builds the sparse matrix, factorizes the start basis, and verifies
/// dual feasibility (the precondition the dual simplex maintains but does not
/// establish). Each [`reoptimize`](Self::reoptimize) then **clones** the pristine
/// factorization and runs dual pivots from it, so the expensive factorize and the
/// sparse-matrix build are paid once rather than per solve. This is what lets the
/// B&B probe a node's many strong-branching children (one bound each) from the
/// single node-optimal factorization instead of refactorizing it every probe.
///
/// The dual-feasibility check is bound-*value* independent (it depends on the
/// objective, the basis, and which bound each nonbasic sits at), and the
/// re-optimizations only perturb bounds of basic variables (a branch tightens the
/// fractional, hence basic, variable), so the precondition verified at `prepare`
/// holds for every `reoptimize` from the same basis.
pub struct PreparedDual<'a> {
    m: usize,
    n: usize,
    c: &'a [f64],
    sp: &'a SparseCols,
    lu: FeralLU, // pristine factorization of `basis`
    basis: Vec<usize>,
    slot_of: Vec<i64>,
    stat: Vec<i8>,
}

impl<'a> PreparedDual<'a> {
    /// Factorize `start` for the LP `lp` and verify dual feasibility, or `None`
    /// if the basis is unusable (wrong size, singular, or dual-infeasible) and the
    /// caller should cold-solve instead. `lp.l`/`lp.u` are the bounds at which the
    /// basis is dual-feasible (the reference for the precondition check).
    pub fn prepare(
        lp: &LpView<'a>,
        start: &Basis,
        opts: &SimplexOptions,
        sp: &'a SparseCols,
    ) -> Option<Self> {
        let (m, n, l, u, c) = (lp.m, lp.n, lp.l, lp.u, lp.c);
        if start.basic_vars.len() != m {
            return None;
        }
        let tol = opts.tol;
        let basis = start.basic_vars.clone();
        let mut slot_of = vec![-1i64; n];
        for (slot, &j) in basis.iter().enumerate() {
            slot_of[j] = slot as i64;
        }
        let stat = start.col_status.clone();
        if stat.len() != n {
            return None;
        }

        let mut lu = FeralLU::new();
        // Sparse basis columns straight from the CSC view — O(nnz) factorize, no
        // dense m×m basis (discopt#268 / feral#87). Bit-identical to `col()`.
        let cols: Vec<Vec<(usize, f64)>> = basis
            .iter()
            .map(|&j| {
                let (rows, vals) = sp.col(j);
                rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
            })
            .collect();
        if lu.factorize_sparse(m, &cols).is_err() {
            return None; // singular warm basis → fall back to cold
        }

        // Verify the starting basis is actually dual-feasible — the precondition
        // the dual simplex *maintains* but does not *establish*. With y = B⁻ᵀc_B
        // the reduced cost is d_j = c_j − yᵀA_j; a nonbasic-at-lower needs
        // d_j ≥ −tol, a nonbasic-at-upper needs d_j ≤ tol, and a nonbasic *free*
        // variable (both bounds infinite) needs |d_j| ≤ tol. A dual-infeasible
        // start would silently converge to a wrong Optimal/Infeasible.
        {
            let mut y: Vec<f64> = basis.iter().map(|&j| c[j]).collect();
            if lu.btran(&mut y).is_err() {
                return None;
            }
            for j in 0..n {
                if stat[j] == BASIC || u[j] - l[j] <= tol {
                    continue; // basic or fixed: dual feasibility is unconstrained
                }
                let dj = c[j] - sp.dot(j, &y);
                let free = l[j] <= -INF && u[j] >= INF;
                let ok = if free {
                    dj.abs() <= tol
                } else if stat[j] == AT_UPPER {
                    dj <= tol
                } else {
                    dj >= -tol
                };
                if !ok {
                    return None; // dual-infeasible warm start → cold fallback
                }
            }
        }

        Some(PreparedDual {
            m,
            n,
            c,
            sp,
            lu,
            basis,
            slot_of,
            stat,
        })
    }

    /// As [`Self::prepare`] but from a borrowed CSC matrix + bound/cost slices
    /// instead of a dense [`LpView`] — the sparse-native warm path. Identical
    /// preconditions and factorization (both already build the basis from `sp`).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_cols(
        sp: &'a SparseCols,
        m: usize,
        n: usize,
        c: &'a [f64],
        l: &'a [f64],
        u: &'a [f64],
        start: &Basis,
        opts: &SimplexOptions,
    ) -> Option<Self> {
        let lp = LpView {
            a: &[],
            m,
            n,
            c,
            l,
            u,
        };
        Self::prepare(&lp, start, opts, sp)
    }

    /// Dual re-optimization from the prepared basis for bounds `l`/`u` and rhs
    /// `b`, returning a valid solution. On any numerical difficulty it cold-solves
    /// the same LP, so the result is always correct.
    pub fn reoptimize(&self, l: &[f64], u: &[f64], b: &[f64], opts: &SimplexOptions) -> LpSolve {
        crate::profile::incr(crate::profile::Ctr::DualWarmSolves);
        match self.run_dual(l, u, b, opts) {
            Some(sol) => sol,
            // Cold fallback from the prepared CSC matrix (clone is O(nnz), paid only
            // on the rare numerical-breakdown path) — no dense matrix is kept.
            None => {
                crate::profile::incr(crate::profile::Ctr::DualColdFallbacks);
                solve_lp_cols(self.sp.clone(), self.m, self.n, self.c, l, u, b, opts)
            }
        }
    }

    /// The dual pivots, on a fresh clone of the prepared factorization/basis.
    /// `None` requests the cold fallback (numerical breakdown or iteration cap).
    fn run_dual(&self, l: &[f64], u: &[f64], b: &[f64], opts: &SimplexOptions) -> Option<LpSolve> {
        let (m, n, c, sp) = (self.m, self.n, self.c, &self.sp);
        let tol = opts.tol;
        // Clone the pristine prepared state; the loop mutates these in place.
        let mut lu = self.lu.clone();
        let mut basis = self.basis.clone();
        let mut slot_of = self.slot_of.clone();
        let mut stat = self.stat.clone();

        let nb_value = |stat: &[i8], j: usize| -> f64 {
            if stat[j] == AT_UPPER {
                u[j]
            } else if l[j] <= -INF {
                0.0
            } else {
                l[j]
            }
        };

        // Exact basic values and reduced costs — the soundness anchor. Maintained
        // incrementally through the loop and refreshed from these (see the doc).
        let mut xb = recompute_basic_values(&mut lu, sp, b, n, l, u, &stat)?;
        let mut dvec = recompute_reduced_costs(&mut lu, sp, c, n, &basis)?;
        // Iterations since the last exact refresh; bound roundoff in the rank-1
        // updates by refreshing at least this often (still far cheaper than the
        // per-iteration full recompute it replaces).
        const REFRESH: usize = 32;
        let mut since_refresh = 0usize;

        let mut updates = 0usize;
        // Dual Devex reference weights γ_i ≥ 1, one per basic slot. The leaving
        // variable maximizes (bound violation)²/γ_i — a cheap steepest-edge
        // approximation that cuts dual iterations versus plain largest-violation
        // pricing. Pricing only chooses *which* primal-infeasible row leaves, never
        // affects correctness (any infeasible basic var is a valid leaving choice).
        let mut gamma = vec![1.0f64; m];
        // Anti-cycling: consecutive degenerate dual pivots (entering reduced cost
        // ≈ 0 → no dual-objective progress) accumulate `stall`; once it crosses the
        // threshold the pivot rules switch to Bland's smallest-index selection
        // (both leaving row and entering column), which is guaranteed cycle-free.
        // A productive pivot resets the count. Without this the only cycle escape
        // was the iteration cap → cold fallback; Bland resolves it in place.
        let mut stall = 0usize;
        let bland_threshold = 2 * (n + 1);
        // The loop index is the pivot count: each completed iteration performs one
        // pivot, and every early return happens at the loop top (before this
        // iteration's pivot), so `pivots` is exactly the number performed so far.
        for pivots in 0..opts.max_iter {
            let bland = stall > bland_threshold;
            // Poll the wall-clock deadline (see SimplexOptions::deadline). On a
            // timeout we abandon the warm dual re-solve and request the cold
            // fallback by returning None; the cold primal solve re-checks the
            // same deadline on its first iteration and returns IterLimit, which
            // the B&B treats soundly. Polled every 256 pivots to keep the now()
            // cost negligible against a full ftran iteration.
            if (pivots & 255) == 0
                && opts
                    .deadline
                    .is_some_and(|d| std::time::Instant::now() >= d)
            {
                return None;
            }
            // Periodic exact refresh to bound increment roundoff.
            if since_refresh >= REFRESH {
                xb = recompute_basic_values(&mut lu, sp, b, n, l, u, &stat)?;
                dvec = recompute_reduced_costs(&mut lu, sp, c, n, &basis)?;
                since_refresh = 0;
            }

            // Leaving variable from the *maintained* `xb`. If it shows primal
            // feasibility, confirm with an *exact* recompute before declaring
            // optimality, and verify dual feasibility on *exact* reduced costs —
            // so a drifted increment can never return a wrong optimum.
            let (r, to_lower, delta) =
                match select_leaving(m, &basis, l, u, &gamma, &xb, tol, bland) {
                    Some(sel) => sel,
                    None => {
                        xb = recompute_basic_values(&mut lu, sp, b, n, l, u, &stat)?;
                        since_refresh = 0;
                        match select_leaving(m, &basis, l, u, &gamma, &xb, tol, bland) {
                            Some(sel) => sel,
                            None => {
                                dvec = recompute_reduced_costs(&mut lu, sp, c, n, &basis)?;
                                if dual_feasible(n, &stat, l, u, &dvec, tol) {
                                    // In-engine refined recovery (discopt#364): if the
                                    // working factor's growth signal flags possible digit
                                    // loss, recompute x_B from a fresh, refinement-polished
                                    // factorization and re-verify primal feasibility on the
                                    // sharper values. If the sharper x_B reveals an
                                    // infeasibility the drifted one hid, hand to the robust
                                    // cold solve rather than certify a wrong optimum;
                                    // otherwise certify with the sharper x_B. Sound either
                                    // way — the decision is only ever made *more* accurate,
                                    // and the common (benign-growth) path is untouched.
                                    let refined_xb: Option<Vec<f64>> =
                                        if lu.growth().is_some_and(|g| g > GROWTH_REFINE_TRIGGER) {
                                            crate::profile::incr(
                                                crate::profile::Ctr::RefinedRecoveryAttemptsDual,
                                            );
                                            match refined_basic_values(
                                                sp, b, n, m, l, u, &basis, &stat,
                                            ) {
                                                Some(xb_r) => {
                                                    if select_leaving(
                                                        m, &basis, l, u, &gamma, &xb_r, tol, bland,
                                                    )
                                                    .is_some()
                                                    {
                                                        crate::profile::incr(
                                                    crate::profile::Ctr::RefinedRecoveryRescuesDual,
                                                );
                                                        return None; // sharper x_B infeasible → cold fallback
                                                    }
                                                    Some(xb_r)
                                                }
                                                None => None, // fresh factor failed; keep the exact x_B
                                            }
                                        } else {
                                            None
                                        };
                                    let xb_final: &[f64] = refined_xb.as_deref().unwrap_or(&xb);
                                    // Row duals `y = B⁻ᵀ c_B` for the safe dual bound;
                                    // empty (caller falls back) if the btran fails.
                                    let mut y: Vec<f64> = basis.iter().map(|&j| c[j]).collect();
                                    let dual = if lu.btran(&mut y).is_ok() {
                                        y
                                    } else {
                                        Vec::new()
                                    };
                                    return Some(assemble(
                                        n,
                                        m,
                                        &basis,
                                        &slot_of,
                                        &stat,
                                        xb_final,
                                        c,
                                        l,
                                        u,
                                        LpStatus::Optimal,
                                        pivots,
                                        dual,
                                        Vec::new(),
                                    ));
                                }
                                // Exact reduced costs reveal dual infeasibility (a drifted
                                // pivot): hand off to the robust cold solve.
                                return None;
                            }
                        }
                    }
                };

            // Pivot row ρ = e_rᵀ B⁻¹, then α_rj = ρ·A_j for every nonbasic j (kept
            // for both the ratio test and the incremental reduced-cost update). No
            // per-iteration `y = B⁻ᵀc_B` btran: the reduced costs are maintained.
            let mut rho = vec![0.0; m];
            rho[r] = 1.0;
            if lu.btran(&mut rho).is_err() {
                return None;
            }
            let mut alpha_r = vec![0.0; n];
            for (j, arj) in alpha_r.iter_mut().enumerate() {
                if stat[j] != BASIC {
                    *arj = sp.dot(j, &rho);
                }
            }

            // Bound-flipping (long-step) dual ratio test on the *maintained*
            // reduced costs: walk eligible breakpoints in increasing |d_j/α_rj|,
            // flipping each finite-bounded one to its opposite bound until the
            // infeasibility `delta` is closed; the last breakpoint enters. If no
            // column is eligible the LP is primal-infeasible — but confirm on
            // *exact* reduced costs first, since a drifted `dvec` could spuriously
            // empty the set. Flips only change which bound a nonbasic sits at.
            let mut cand = build_candidates(n, &stat, l, u, &alpha_r, &dvec, to_lower, tol);
            if cand.is_empty() {
                dvec = recompute_reduced_costs(&mut lu, sp, c, n, &basis)?;
                cand = build_candidates(n, &stat, l, u, &alpha_r, &dvec, to_lower, tol);
                if cand.is_empty() {
                    xb = recompute_basic_values(&mut lu, sp, b, n, l, u, &stat)?;
                    // Farkas dual ray: the leaving row's `ρ = eᵣᵀ B⁻¹`. With the
                    // ratio test finding no entering column, `ρ` (or `−ρ`) is a
                    // direction along which `bᵀy` beats the box-max of `(Aᵀy)ᵀz`,
                    // certifying primal infeasibility — the caller verifies it.
                    return Some(assemble(
                        n,
                        m,
                        &basis,
                        &slot_of,
                        &stat,
                        &xb,
                        c,
                        l,
                        u,
                        LpStatus::Infeasible,
                        pivots,
                        rho.clone(),
                        Vec::new(),
                    ));
                }
            }
            cand.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
            let (q, flips): (usize, Vec<usize>) = if bland {
                // Bland (anti-cycling): the plain min-ratio dual pivot with the
                // smallest-index tie-break and NO bound flips (a short step). The
                // long-step bound-flipping below can revisit a basis under
                // degeneracy; the smallest-index rule provably cannot cycle.
                let min_ratio = cand[0].1;
                let q = cand
                    .iter()
                    .filter(|&&(_, ratio, _)| ratio <= min_ratio + tol)
                    .map(|&(j, _, _)| j)
                    .min()
                    .expect("cand is non-empty");
                (q, Vec::new())
            } else {
                // Walk breakpoints, flipping finite-bounded ones until `delta` is
                // closed. The last candidate is always taken as the entering column
                // (never flipped), so even if the bound gaps can't fully close
                // `delta` the step is a valid dual pivot and the search progresses.
                let mut slope = 0.0f64;
                let mut q = cand[cand.len() - 1].0;
                let mut flips: Vec<usize> = Vec::new();
                let last = cand.len() - 1;
                for (idx, &(j, _ratio, aj)) in cand.iter().enumerate() {
                    let gap = u[j] - l[j];
                    slope += aj * gap;
                    if idx == last || gap >= INF || slope >= delta - tol {
                        q = j; // this breakpoint enters the basis; stop flipping
                        break;
                    }
                    flips.push(j); // fully traversed → flip to the opposite bound
                }
                (q, flips)
            };
            // Apply the bound flips (no basis change); their effect on x_B and the
            // reduced costs is realized by the next iteration's exact recompute.
            for &j in &flips {
                stat[j] = if stat[j] == AT_LOWER {
                    AT_UPPER
                } else {
                    AT_LOWER
                };
            }

            // Entering column α = B⁻¹A_q for the Devex weight update (and reused as
            // the raw column for the product-form basis update). Scattered from the
            // CSC view into a dense buffer (O(nnz_q)), not read from a dense matrix.
            let mut raw_q = vec![0.0; m];
            sp.scatter(q, &mut raw_q);
            let mut alpha = raw_q.clone();
            if lu.ftran(&mut alpha).is_err() {
                return None;
            }
            let piv = alpha[r];
            if piv.abs() <= tol {
                return None; // unstable (near-zero) pivot → robust cold fallback
            }
            {
                // Goldfarb–Reid dual Devex update (uses the still-current basis).
                let gamma_r = gamma[r];
                for i in 0..m {
                    if i != r {
                        let cand = (alpha[i] / piv).powi(2) * gamma_r;
                        if cand > gamma[i] {
                            gamma[i] = cand;
                        }
                    }
                }
                // Slot r now holds the entering variable; give it a fresh weight.
                gamma[r] = (gamma_r / (piv * piv)).max(1.0);
                if gamma[r] > 1e10 {
                    for g in gamma.iter_mut() {
                        *g = 1.0; // reframe when weights blow up
                    }
                }
            }

            // Pivot: q enters at slot r; leaving var pinned at the violated bound.
            let leaving = basis[r];

            // Degeneracy for anti-cycling: a dual pivot makes no dual-objective
            // progress when the entering variable's reduced cost is ≈ 0 (a
            // zero-length dual step). Consecutive such pivots are what cycle;
            // captured here (before `dvec[q]` is zeroed) and fed to `stall` below.
            let degenerate = dvec[q].abs() <= tol;

            // Incremental reduced-cost update (rank-1): d_j −= (d_q/piv)·α_rj for
            // every nonbasic j; the entering q becomes basic (0) and the leaving
            // column takes −d_q/piv. Bound flips don't change reduced costs.
            let theta_d = dvec[q] / piv;
            for j in 0..n {
                if stat[j] != BASIC && j != q {
                    dvec[j] -= theta_d * alpha_r[j];
                }
            }
            dvec[leaving] = -theta_d;
            dvec[q] = 0.0;

            // Incremental basic-value update. First the bound flips' RHS effect:
            // x_B −= B⁻¹(Σ_flipped A_j·Δx_j). Then the primal step driving the
            // leaving variable to its bound: x_B −= t·α_q, with the entering column
            // taking value v_q + t at slot r. (`stat` already reflects the flips;
            // q/leaving statuses are set just below, so `nb_value(q)` is its bound.)
            if !flips.is_empty() {
                let mut rd = vec![0.0; m];
                for &j in &flips {
                    let gap = u[j] - l[j];
                    let dxj = if stat[j] == AT_UPPER { gap } else { -gap };
                    let (rows, vals) = sp.col(j);
                    for (k, &rr) in rows.iter().enumerate() {
                        rd[rr] += vals[k] * dxj;
                    }
                }
                if lu.ftran(&mut rd).is_err() {
                    return None;
                }
                for (xi, ri) in xb.iter_mut().zip(rd.iter()) {
                    *xi -= *ri;
                }
            }
            let bound_p = if to_lower { l[leaving] } else { u[leaving] };
            let t = (xb[r] - bound_p) / piv;
            let v_q = nb_value(&stat, q);
            for (xi, ai) in xb.iter_mut().zip(alpha.iter()) {
                *xi -= t * *ai;
            }
            xb[r] = v_q + t;

            stat[leaving] = if to_lower { AT_LOWER } else { AT_UPPER };
            slot_of[leaving] = -1;
            basis[r] = q;
            slot_of[q] = r as i64;
            stat[q] = BASIC;
            let need_refac = lu.update(r, &raw_q).is_err();
            updates += 1;
            if need_refac || updates >= 48 {
                let cols: Vec<Vec<(usize, f64)>> = basis
                    .iter()
                    .map(|&j| {
                        let (rows, vals) = sp.col(j);
                        rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
                    })
                    .collect();
                if lu.factorize_sparse(m, &cols).is_err() {
                    return None;
                }
                updates = 0;
                // Fresh factorization → reseed exact values, reset the drift clock.
                xb = recompute_basic_values(&mut lu, sp, b, n, l, u, &stat)?;
                dvec = recompute_reduced_costs(&mut lu, sp, c, n, &basis)?;
                since_refresh = 0;
            } else {
                since_refresh += 1;
            }

            // Anti-cycling bookkeeping: a degenerate pivot advances the stall count
            // (and, once over the threshold, keeps Bland engaged); any productive
            // pivot resets it. Count Bland-mode pivots for observability.
            if bland {
                crate::profile::incr(crate::profile::Ctr::DualBlandActivations);
            }
            if degenerate {
                stall += 1;
                crate::profile::incr(crate::profile::Ctr::DualDegeneratePivots);
            } else {
                stall = 0;
            }
        }
        None // iteration cap → cold fallback
    }
}

/// Exact basic values `x_B = B⁻¹(b − Σ_nonbasic A_j x_j)` for the current basis.
/// The soundness anchor for the incremental dual loop: the maintained `xb` is
/// refreshed from this on a cadence and whenever optimality/infeasibility is
/// claimed, so a drifted increment never decides the returned result.
/// The right-hand side the basic-variable solve runs against:
/// `b − Σ_{nonbasic} A_j x_j` (so that `B x_B = rhs`).
fn reduced_rhs(
    sp: &SparseCols,
    b: &[f64],
    n: usize,
    l: &[f64],
    u: &[f64],
    stat: &[i8],
) -> Vec<f64> {
    let mut xb = b.to_vec();
    for j in 0..n {
        if stat[j] != BASIC {
            let v = if stat[j] == AT_UPPER {
                u[j]
            } else if l[j] <= -INF {
                0.0
            } else {
                l[j]
            };
            if v != 0.0 {
                let (rows, vals) = sp.col(j);
                for (k, &rr) in rows.iter().enumerate() {
                    xb[rr] -= vals[k] * v;
                }
            }
        }
    }
    xb
}

fn recompute_basic_values(
    lu: &mut FeralLU,
    sp: &SparseCols,
    b: &[f64],
    n: usize,
    l: &[f64],
    u: &[f64],
    stat: &[i8],
) -> Option<Vec<f64>> {
    let mut xb = reduced_rhs(sp, b, n, l, u, stat);
    if lu.ftran(&mut xb).is_err() {
        return None;
    }
    Some(xb)
}

/// Growth high-water ratio (feral#93) above which the optimality gate re-solves
/// `x_B` with a fresh refinement-polished factorization before certifying. A
/// benign basis has growth ≈ 1; only a factor that lost digits (≫ 1) pays for the
/// refined recompute, so well-conditioned nodes — the overwhelming majority — are
/// untouched.
const GROWTH_REFINE_TRIGGER: f64 = 1e4;

/// Recompute `x_B` for the final basis via a **fresh** numeric-focus factorization
/// with iterative refinement (discopt#364). The dual's working factor accumulates
/// Forrest–Tomlin update error across pivots; a fresh factor of the same basis
/// carries none of it, and refinement then polishes the residual `b − B·x`, so the
/// optimality decision is made on the sharpest `x_B` available — recovered *inside*
/// the engine rather than by handing off to another solver. Returns `None` if the
/// fresh factorization or refined solve fails (the caller keeps its existing `x_B`).
#[allow(clippy::too_many_arguments)]
fn refined_basic_values(
    sp: &SparseCols,
    b: &[f64],
    n: usize,
    m: usize,
    l: &[f64],
    u: &[f64],
    basis: &[usize],
    stat: &[i8],
) -> Option<Vec<f64>> {
    let cols: Vec<Vec<f64>> = basis
        .iter()
        .map(|&j| {
            let mut col = vec![0.0; m];
            sp.scatter(j, &mut col);
            col
        })
        .collect();
    let mut lu = FeralLU::new().with_numeric_focus();
    lu.factorize(m, &cols).ok()?;
    let mut xb = reduced_rhs(sp, b, n, l, u, stat);
    lu.ftran_refined(&mut xb).ok()?;
    Some(xb)
}

/// Exact reduced costs `d_j = c_j − yᵀA_j` (with `y = B⁻ᵀ c_B`) for every column.
/// Basic columns get ≈0. Used to seed and periodically refresh the maintained
/// `dvec`, and to verify dual feasibility before `Optimal` is returned.
fn recompute_reduced_costs(
    lu: &mut FeralLU,
    sp: &SparseCols,
    c: &[f64],
    n: usize,
    basis: &[usize],
) -> Option<Vec<f64>> {
    let mut y: Vec<f64> = basis.iter().map(|&j| c[j]).collect();
    if lu.btran(&mut y).is_err() {
        return None;
    }
    let mut d = vec![0.0; n];
    for (j, dj) in d.iter_mut().enumerate() {
        *dj = c[j] - sp.dot(j, &y);
    }
    Some(d)
}

/// Most primal-infeasible basic variable by dual-Devex score `violation²/γ`, or
/// `None` if every basic variable is within its bounds (primal feasible). Returns
/// `(slot, to_lower, violation)` where `to_lower` is the bound the leaving
/// variable is pinned at. Pricing only — choosing any infeasible row is valid.
#[allow(clippy::too_many_arguments)]
fn select_leaving(
    m: usize,
    basis: &[usize],
    l: &[f64],
    u: &[f64],
    gamma: &[f64],
    xb: &[f64],
    tol: f64,
    bland: bool,
) -> Option<(usize, bool, f64)> {
    let mut r = None;
    let mut best_score = 0.0f64;
    // Bland's rule tracks the smallest *variable* index among infeasible rows.
    let mut best_var = usize::MAX;
    let mut to_lower = true;
    let mut delta = 0.0f64;
    for i in 0..m {
        let bi = basis[i];
        let viol = if xb[i] < l[bi] - tol {
            Some((l[bi] - xb[i], true))
        } else if xb[i] > u[bi] + tol {
            Some((xb[i] - u[bi], false))
        } else {
            None
        };
        if let Some((v, lo)) = viol {
            // Bland (anti-cycling): smallest variable index. Devex (default):
            // largest bound-violation steepest-edge score. Either choice is a
            // *valid* leaving row — any primal-infeasible basic var may leave — so
            // this only affects the pivot path, never correctness.
            let take = if bland {
                bi < best_var
            } else {
                v * v / gamma[i] > best_score
            };
            if take {
                best_score = v * v / gamma[i];
                best_var = bi;
                r = Some(i);
                to_lower = lo;
                delta = v;
            }
        }
    }
    r.map(|r| (r, to_lower, delta))
}

/// Eligible entering breakpoints for the bound-flipping dual ratio test, each
/// `(j, |d_j/α_rj|, |α_rj|)`. `alpha_r[j] = ρ·A_j` (the pivot row) is precomputed
/// for every nonbasic `j`; `dvec` supplies the reduced costs (maintained, not
/// recomputed). Eligibility is the usual dual sign test against the leaving bound.
#[allow(clippy::too_many_arguments)]
fn build_candidates(
    n: usize,
    stat: &[i8],
    l: &[f64],
    u: &[f64],
    alpha_r: &[f64],
    dvec: &[f64],
    to_lower: bool,
    tol: f64,
) -> Vec<(usize, f64, f64)> {
    let mut cand = Vec::new();
    for j in 0..n {
        if stat[j] == BASIC || u[j] - l[j] <= tol {
            continue;
        }
        let arj = alpha_r[j];
        let eligible = if stat[j] == AT_LOWER {
            if to_lower {
                arj < -tol
            } else {
                arj > tol
            }
        } else if to_lower {
            arj > tol
        } else {
            arj < -tol
        };
        if eligible {
            cand.push((j, (dvec[j] / arj).abs(), arj.abs()));
        }
    }
    cand
}

/// Whether every non-fixed nonbasic column is dual-feasible at `dvec`: a column at
/// its lower bound needs `d ≥ −tol`, at its upper bound `d ≤ tol`, and a free
/// column `|d| ≤ tol`. Checked (with **exact** `dvec`) before `Optimal` is
/// returned, so incremental reduced-cost drift can never yield a wrong optimum.
fn dual_feasible(n: usize, stat: &[i8], l: &[f64], u: &[f64], dvec: &[f64], tol: f64) -> bool {
    for j in 0..n {
        if stat[j] == BASIC || u[j] - l[j] <= tol {
            continue;
        }
        let free = l[j] <= -INF && u[j] >= INF;
        let ok = if free {
            dvec[j].abs() <= tol
        } else if stat[j] == AT_UPPER {
            dvec[j] <= tol
        } else {
            dvec[j] >= -tol
        };
        if !ok {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    n: usize,
    _m: usize,
    basis: &[usize],
    slot_of: &[i64],
    stat: &[i8],
    xb: &[f64],
    c: &[f64],
    l: &[f64],
    u: &[f64],
    status: LpStatus,
    pivots: usize,
    dual: Vec<f64>,
    ray: Vec<f64>,
) -> LpSolve {
    let mut x = vec![0.0; n];
    for j in 0..n {
        x[j] = if stat[j] == BASIC {
            xb[slot_of[j] as usize]
        } else if stat[j] == AT_UPPER {
            u[j]
        } else if l[j] <= -INF {
            0.0
        } else {
            l[j]
        };
    }
    let obj: f64 = (0..n).map(|j| c[j] * x[j]).sum();
    let basic_vars = basis.to_vec();
    let col_status = stat.to_vec();
    LpSolve {
        status,
        x,
        obj,
        basis: Basis {
            col_status,
            basic_vars,
        },
        iters: pivots,
        dual,
        ray,
    }
}

#[cfg(test)]
mod tests {
    use super::super::primal::solve_lp;
    use super::*;

    fn opts() -> SimplexOptions {
        SimplexOptions::default()
    }

    #[test]
    fn bland_leaving_selects_smallest_variable_index() {
        // Two primal-infeasible basic rows with equal Devex scores: slot 0 holds
        // variable 5, slot 1 holds variable 2, both 1 below their lower bound.
        let m = 2;
        let basis = [5usize, 2usize];
        let l = vec![0.0; 6];
        let u = vec![10.0; 6];
        let gamma = [1.0, 1.0];
        let xb = [-1.0, -1.0];

        // Devex: equal scores, so the first infeasible slot (0) wins.
        let dev = select_leaving(m, &basis, &l, &u, &gamma, &xb, 1e-7, false).unwrap();
        assert_eq!(dev.0, 0, "Devex should take the first equal-score slot");
        assert!(dev.1, "both violate the lower bound → leave toward lower");

        // Bland: the smallest *variable* index wins — variable 2 sits in slot 1.
        let bl = select_leaving(m, &basis, &l, &u, &gamma, &xb, 1e-7, true).unwrap();
        assert_eq!(bl.0, 1, "Bland must pick slot 1 (variable 2 < variable 5)");

        // Both modes agree there is no leaving row once feasible.
        let feasible = [1.0, 1.0];
        assert!(select_leaving(m, &basis, &l, &u, &gamma, &feasible, 1e-7, false).is_none());
        assert!(select_leaving(m, &basis, &l, &u, &gamma, &feasible, 1e-7, true).is_none());
    }

    // Solve cold, then perturb one bound and re-solve warm from the cold basis;
    // assert same objective as a fresh cold solve of the perturbed LP.
    #[test]
    fn warm_matches_cold_after_bound_change() {
        // 2 constraints with slacks: x0+x1+s0=4, x0+3x1+s1=6, x∈[0,5], s∈[0,inf].
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let b = [4.0, 6.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [5.0, 5.0, INF, INF];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let cold = solve_lp(&lp, &b, &opts());
        assert_eq!(cold.status, LpStatus::Optimal);

        // Tighten x1's upper bound to 0.5 (a B&B "branch x1 ≤ 0").
        let u2 = [5.0, 0.5, INF, INF];
        let lp2 = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u2,
        };
        let warm = solve_lp_warm(&lp2, &b, &cold.basis, &opts());
        let cold2 = solve_lp(&lp2, &b, &opts());
        assert_eq!(cold2.status, LpStatus::Optimal);
        assert_eq!(warm.status, LpStatus::Optimal);
        assert!(
            (warm.obj - cold2.obj).abs() < 1e-7,
            "warm {} vs cold {}",
            warm.obj,
            cold2.obj
        );
        // The tightened bound made the parent basis primal-infeasible, so the
        // dual path must have taken ≥1 pivot (i.e. it ran, not silently fell
        // back to a cold solve).
        assert!(
            warm.iters >= 1,
            "dual warm-start did not run (fell back to cold)"
        );
    }

    // After a warm dual re-optimization the exported row duals must still satisfy
    // the safe-bound identity `bᵀy + Σ min_box((c−Aᵀy)z) ≈ obj` — i.e. the dual
    // path populates `LpSolve::dual` correctly, not just the primal cold path.
    #[test]
    fn warm_optimal_exports_duals_matching_objective() {
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let b = [4.0, 6.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [5.0, 5.0, INF, INF];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let cold = solve_lp(&lp, &b, &opts());
        assert_eq!(cold.status, LpStatus::Optimal);

        // Tighten x1 ≤ 0.5 and re-solve warm from the cold basis (dual pivots run).
        let u2 = [5.0, 0.5, INF, INF];
        let lp2 = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u2,
        };
        let warm = solve_lp_warm(&lp2, &b, &cold.basis, &opts());
        assert_eq!(warm.status, LpStatus::Optimal);
        assert_eq!(warm.dual.len(), m);
        // Safe bound from the warm-path duals reproduces the warm objective.
        let mut g = b
            .iter()
            .zip(&warm.dual)
            .map(|(bi, yi)| bi * yi)
            .sum::<f64>();
        for j in 0..n {
            let aty: f64 = (0..m).map(|i| a[i * n + j] * warm.dual[i]).sum();
            let rc = c[j] - aty;
            if rc > 0.0 {
                g += rc * l[j];
            } else if rc < 0.0 && u2[j] < INF {
                g += rc * u2[j];
            }
        }
        assert!(
            (g - warm.obj).abs() < 1e-7,
            "safe bound {g} vs warm obj {}",
            warm.obj
        );
    }

    // The batch-cached-CSC entry point must be bit-identical to rebuilding the CSC
    // per solve: it is the same math, only sharing the constant matrix's CSC across
    // a B&B batch. A mismatch would mean the cache desynced from `lp.a`.
    #[test]
    fn csc_path_matches_rebuild_path() {
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let b = [4.0, 6.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [5.0, 5.0, INF, INF];
        let lp0 = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let cold = solve_lp(&lp0, &b, &opts());
        assert_eq!(cold.status, LpStatus::Optimal);

        // A handful of distinct bound perturbations (like sibling B&B nodes sharing
        // one batch matrix); each must agree between the two warm entry points.
        let sp = SparseCols::from_dense(&a, m, n);
        for ub1 in [0.0, 0.5, 1.0, 2.0, 5.0] {
            let u2 = [5.0, ub1, INF, INF];
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u2,
            };
            let rebuilt = solve_lp_warm_scaled(&lp, &b, &cold.basis, &opts());
            let cached = solve_lp_warm_scaled_csc(&lp, &b, &cold.basis, &opts(), &sp);
            assert_eq!(rebuilt.status, cached.status);
            assert_eq!(rebuilt.iters, cached.iters);
            assert!(
                (rebuilt.obj - cached.obj).abs() < 1e-12,
                "ub1={ub1}: rebuilt {} vs cached {}",
                rebuilt.obj,
                cached.obj
            );
        }
    }

    #[test]
    fn warm_random_lps_match_cold() {
        // Deterministic LCG; perturb one upper bound and compare.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for _ in 0..40 {
            let (m, n) = (3, 6);
            let a: Vec<f64> = (0..m * n).map(|_| (next() * 6.0 - 3.0).round()).collect();
            let x0: Vec<f64> = (0..n).map(|_| 0.5 + 4.0 * next()).collect();
            let b: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| a[i * n + j] * x0[j]).sum())
                .collect();
            let c: Vec<f64> = (0..n).map(|_| (next() * 10.0 - 5.0).round()).collect();
            let l = vec![0.0; n];
            let u = vec![5.0; n];
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u,
            };
            let cold = solve_lp(&lp, &b, &opts());
            if cold.status != LpStatus::Optimal {
                continue;
            }
            // tighten one variable's upper bound below its current value
            let k = (next() * n as f64) as usize % n;
            let mut u2 = u.clone();
            u2[k] = (cold.x[k] - 1.0).max(0.0);
            let lp2 = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u2,
            };
            let warm = solve_lp_warm(&lp2, &b, &cold.basis, &opts());
            let cold2 = solve_lp(&lp2, &b, &opts());
            if cold2.status == LpStatus::Optimal {
                assert_eq!(warm.status, LpStatus::Optimal);
                assert!(
                    (warm.obj - cold2.obj).abs() < 1e-6,
                    "warm {} vs cold {}",
                    warm.obj,
                    cold2.obj
                );
            }
        }
    }

    // Stress the incremental dual loop: larger LPs with finite bounds (so the
    // bound-flipping ratio test fires) and perturbations heavy enough to drive
    // many pivots — crossing the exact-refresh cadence and the LU refactor — so
    // the maintained `xb`/`dvec` are genuinely exercised, not just seeded. Every
    // warm re-solve must still reach the cold optimum, and at least some must
    // actually run the warm dual path (iters > 0), proving it isn't silently
    // cold-falling-back and hiding increment bugs behind the safe fallback.
    #[test]
    fn warm_incremental_matches_cold_under_many_pivots() {
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut warm_ran = 0;
        for _ in 0..60 {
            let (m, n) = (8, 22);
            let a: Vec<f64> = (0..m * n).map(|_| (next() * 6.0 - 3.0).round()).collect();
            let x0: Vec<f64> = (0..n).map(|_| 0.3 + 3.4 * next()).collect();
            let b: Vec<f64> = (0..m)
                .map(|i| (0..n).map(|j| a[i * n + j] * x0[j]).sum())
                .collect();
            let c: Vec<f64> = (0..n).map(|_| (next() * 10.0 - 5.0).round()).collect();
            let l = vec![0.0; n];
            let u = vec![4.0; n];
            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u,
            };
            let cold = solve_lp(&lp, &b, &opts());
            if cold.status != LpStatus::Optimal {
                continue;
            }
            // Tighten MANY upper bounds well below their current values at once, so
            // the warm re-solve takes a long sequence of dual pivots + bound flips.
            let mut u2 = u.clone();
            for j in 0..n {
                if next() < 0.6 {
                    u2[j] = (cold.x[j] - 1.5).max(0.0);
                }
            }
            let lp2 = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u2,
            };
            let warm = solve_lp_warm(&lp2, &b, &cold.basis, &opts());
            let cold2 = solve_lp(&lp2, &b, &opts());
            assert_eq!(
                warm.status, cold2.status,
                "status mismatch warm {:?} vs cold {:?}",
                warm.status, cold2.status
            );
            if cold2.status == LpStatus::Optimal {
                assert!(
                    (warm.obj - cold2.obj).abs() <= 1e-6 * (1.0 + cold2.obj.abs()),
                    "warm {} vs cold {}",
                    warm.obj,
                    cold2.obj
                );
                if warm.iters > 0 {
                    warm_ran += 1;
                }
            }
        }
        assert!(
            warm_ran >= 5,
            "warm dual path almost never ran ({warm_ran}) — incremental path not exercised"
        );
    }

    // A dual-INFEASIBLE warm start must be rejected by the precondition check
    // and fall back to a cold solve — not silently declared "Optimal" at the
    // wrong point. The all-slack basis for `min -x0 -2x1` has reduced costs
    // d_x0=-1, d_x1=-2 < 0 at their lower bounds (dual-infeasible) yet is
    // primal-feasible, so without the check the dual simplex would see no primal
    // infeasibility and return the origin (obj 0) as optimal.
    #[test]
    fn dual_infeasible_warm_start_falls_back_to_cold() {
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0];
        let (m, n) = (2, 4);
        let b = [4.0, 6.0];
        let c = [-1.0, -2.0, 0.0, 0.0];
        let l = [0.0; 4];
        let u = [5.0, 5.0, INF, INF];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };

        // All-slack basis: slacks (cols 2,3) basic; structurals at their lower
        // bound. Primal-feasible (x=0, s=b) but dual-infeasible.
        let bad = Basis {
            basic_vars: vec![2, 3],
            col_status: vec![AT_LOWER, AT_LOWER, BASIC, BASIC],
        };

        let warm = solve_lp_warm(&lp, &b, &bad, &opts());
        let cold = solve_lp(&lp, &b, &opts());
        assert_eq!(cold.status, LpStatus::Optimal);
        assert_eq!(warm.status, LpStatus::Optimal);
        // Correct optimum, not the origin the dual-infeasible basis sits at.
        assert!(cold.obj < -1.0, "sanity: true optimum is negative");
        assert!(
            (warm.obj - cold.obj).abs() < 1e-7,
            "warm {} vs cold {} (should match the true optimum, not 0)",
            warm.obj,
            cold.obj
        );
        // iters==0 confirms the cold fallback ran (the dual path sets iters>0).
        assert_eq!(
            warm.iters, 0,
            "precondition should have forced cold fallback"
        );
    }
}
