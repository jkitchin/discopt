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
use super::primal::solve_lp_scaled;
use super::scaling::ScaledLp;
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
            sol
        }
        None => solve_lp_warm_scaled(lp, b, start, opts),
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
    match PreparedDual::prepare(lp, start, opts) {
        Some(p) => p.reoptimize(lp.l, lp.u, b, opts),
        None => solve_lp_scaled(lp, b, opts), // safe fallback — always correct
    }
}

fn col(a: &[f64], m: usize, n: usize, j: usize) -> Vec<f64> {
    (0..m).map(|i| a[i * n + j]).collect()
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
    a: &'a [f64],
    m: usize,
    n: usize,
    c: &'a [f64],
    sp: SparseCols,
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
    pub fn prepare(lp: &LpView<'a>, start: &Basis, opts: &SimplexOptions) -> Option<Self> {
        let (a, m, n, l, u, c) = (lp.a, lp.m, lp.n, lp.l, lp.u, lp.c);
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

        let sp = SparseCols::from_dense(a, m, n);
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
            a,
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

    /// Dual re-optimization from the prepared basis for bounds `l`/`u` and rhs
    /// `b`, returning a valid solution. On any numerical difficulty it cold-solves
    /// the same LP, so the result is always correct.
    pub fn reoptimize(&self, l: &[f64], u: &[f64], b: &[f64], opts: &SimplexOptions) -> LpSolve {
        match self.run_dual(l, u, b, opts) {
            Some(sol) => sol,
            None => {
                let lp = LpView {
                    a: self.a,
                    m: self.m,
                    n: self.n,
                    c: self.c,
                    l,
                    u,
                };
                solve_lp_scaled(&lp, b, opts)
            }
        }
    }

    /// The dual pivots, on a fresh clone of the prepared factorization/basis.
    /// `None` requests the cold fallback (numerical breakdown or iteration cap).
    fn run_dual(&self, l: &[f64], u: &[f64], b: &[f64], opts: &SimplexOptions) -> Option<LpSolve> {
        let (a, m, n, c, sp) = (self.a, self.m, self.n, self.c, &self.sp);
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

        let mut updates = 0usize;
        let mut pivots = 0usize;
        // Dual Devex reference weights γ_i ≥ 1, one per basic slot. The leaving
        // variable maximizes (bound violation)²/γ_i — a cheap steepest-edge
        // approximation that cuts dual iterations versus plain largest-violation
        // pricing. Pricing only chooses *which* primal-infeasible row leaves, never
        // affects correctness (any infeasible basic var is a valid leaving choice).
        let mut gamma = vec![1.0f64; m];
        for _iter in 0..opts.max_iter {
            // Poll the wall-clock deadline (see SimplexOptions::deadline). On a
            // timeout we abandon the warm dual re-solve and request the cold
            // fallback by returning None; the cold primal solve re-checks the
            // same deadline on its first iteration and returns IterLimit, which
            // the B&B treats soundly. Polled every 256 pivots to keep the now()
            // cost negligible against a full ftran iteration.
            if (_iter & 255) == 0
                && opts
                    .deadline
                    .is_some_and(|d| std::time::Instant::now() >= d)
            {
                return None;
            }
            // Basic values x_B = B⁻¹(b − Σ_nonbasic A_j x_j).
            let mut xb = b.to_vec();
            for j in 0..n {
                if stat[j] != BASIC {
                    let v = nb_value(&stat, j);
                    if v != 0.0 {
                        let (rows, vals) = sp.col(j);
                        for (k, &rr) in rows.iter().enumerate() {
                            xb[rr] -= vals[k] * v;
                        }
                    }
                }
            }
            if lu.ftran(&mut xb).is_err() {
                return None;
            }

            // Leaving variable: most primal-infeasible basic var by Devex score
            // (violation²/γ_i). `to_lower` = the bound the leaving var is pinned at.
            let mut r = None;
            let mut best_score = 0.0f64;
            let mut to_lower = true;
            let mut delta = 0.0f64; // bound-violation magnitude of the chosen row
            for i in 0..m {
                let bi = basis[i];
                let viol = if xb[i] < l[bi] - tol {
                    Some((l[bi] - xb[i], true)) // below lower → pin at lower
                } else if xb[i] > u[bi] + tol {
                    Some((xb[i] - u[bi], false)) // above upper → pin at upper
                } else {
                    None
                };
                if let Some((v, lo)) = viol {
                    let score = v * v / gamma[i];
                    if score > best_score {
                        best_score = score;
                        r = Some(i);
                        to_lower = lo;
                        delta = v;
                    }
                }
            }
            let r = match r {
                Some(r) => r,
                None => {
                    // primal feasible + dual feasible (maintained) ⇒ optimal
                    return Some(assemble(
                        n, m, &basis, &slot_of, &stat, &xb, c, l, u, LpStatus::Optimal, pivots,
                    ));
                }
            };

            // Pivot row ρ = e_rᵀ B⁻¹ ; alpha_rj = ρ·A_j for nonbasic j.
            let mut rho = vec![0.0; m];
            rho[r] = 1.0;
            if lu.btran(&mut rho).is_err() {
                return None;
            }

            // Reduced costs d_j = c_j − yᵀA_j with y = B⁻ᵀ c_B.
            let mut y: Vec<f64> = basis.iter().map(|&j| c[j]).collect();
            if lu.btran(&mut y).is_err() {
                return None;
            }

            // Bound-flipping (long-step) dual ratio test. Eligible nonbasic columns
            // are the breakpoints where some d_j would cross zero as the dual step
            // grows; walking them in increasing |d_j/α_rj| and *flipping* each
            // finite-bounded one to its opposite bound lets the leaving variable
            // travel further in a single pivot (each flip closes |α_rj|·(u−l) of the
            // infeasibility `delta`). The breakpoint that closes the rest — or the
            // first with an infinite bound gap (can't flip) — is the entering column.
            // Flips only change which bound a nonbasic sits at (`stat`); the next
            // iteration's exact x_B/y recompute re-establishes primal values and dual
            // feasibility, and any difficulty still falls back to the cold solve.
            let mut cand: Vec<(usize, f64, f64)> = Vec::new(); // (j, ratio, |α_rj|)
            for j in 0..n {
                if stat[j] == BASIC || u[j] - l[j] <= tol {
                    continue; // basic or fixed
                }
                let arj = sp.dot(j, &rho);
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
                    let dj = c[j] - sp.dot(j, &y);
                    cand.push((j, (dj / arj).abs(), arj.abs()));
                }
            }
            if cand.is_empty() {
                // No eligible entering column: the LP is primal-infeasible.
                // Trustworthy because the start basis was verified dual-feasible.
                return Some(assemble(
                    n, m, &basis, &slot_of, &stat, &xb, c, l, u, LpStatus::Infeasible, pivots,
                ));
            }
            cand.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
            // Walk breakpoints, flipping finite-bounded ones until `delta` is closed.
            // The last candidate is always taken as the entering column (never
            // flipped), so even if the bound gaps can't fully close `delta` the step
            // is a valid dual pivot and the search makes progress.
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
            // Apply the bound flips (no basis change); their effect on x_B and the
            // reduced costs is realized by the next iteration's exact recompute.
            for &j in &flips {
                stat[j] = if stat[j] == AT_LOWER { AT_UPPER } else { AT_LOWER };
            }

            // Entering column α = B⁻¹A_q for the Devex weight update (and reused as
            // the raw column for the product-form basis update).
            let raw_q = col(a, m, n, q);
            let mut alpha = raw_q.clone();
            if lu.ftran(&mut alpha).is_err() {
                return None;
            }
            let piv = alpha[r];
            if piv.abs() > tol {
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
            stat[leaving] = if to_lower { AT_LOWER } else { AT_UPPER };
            slot_of[leaving] = -1;
            basis[r] = q;
            slot_of[q] = r as i64;
            stat[q] = BASIC;
            pivots += 1;
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
            }
        }
        None // iteration cap → cold fallback
    }
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
    }
}

#[cfg(test)]
mod tests {
    use super::super::primal::solve_lp;
    use super::*;

    fn opts() -> SimplexOptions {
        SimplexOptions::default()
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
        assert_eq!(warm.iters, 0, "precondition should have forced cold fallback");
    }
}
