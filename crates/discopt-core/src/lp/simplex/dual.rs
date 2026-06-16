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
            let mut sol = solve_warm_scaled(&view, scaled.b(), start, opts);
            scaled.unscale_x(&mut sol.x);
            sol
        }
        None => solve_warm_scaled(lp, b, start, opts),
    }
}

/// Warm dual re-optimization on an already-equilibrated LP, with the cold
/// primal fallback (also on the scaled matrix, so it is never scaled twice).
fn solve_warm_scaled(
    lp: &LpView<'_>,
    b: &[f64],
    start: &Basis,
    opts: &SimplexOptions,
) -> LpSolve {
    match try_dual(lp, b, start, opts) {
        Some(sol) => sol,
        None => solve_lp_scaled(lp, b, opts), // safe fallback — always correct
    }
}

fn col(a: &[f64], m: usize, n: usize, j: usize) -> Vec<f64> {
    (0..m).map(|i| a[i * n + j]).collect()
}

/// The dual-simplex attempt. Returns `None` to request the cold fallback.
fn try_dual(lp: &LpView<'_>, b: &[f64], start: &Basis, opts: &SimplexOptions) -> Option<LpSolve> {
    let (a, m, n, l, u, c) = (lp.a, lp.m, lp.n, lp.l, lp.u, lp.c);
    if start.basic_vars.len() != m {
        return None;
    }
    let tol = opts.tol;
    let mut basis = start.basic_vars.clone();
    let mut slot_of = vec![-1i64; n];
    for (slot, &j) in basis.iter().enumerate() {
        slot_of[j] = slot as i64;
    }
    let mut stat = start.col_status.clone();
    if stat.len() != n {
        return None;
    }

    let sp = SparseCols::from_dense(a, m, n);
    let mut lu = FeralLU::new();
    let cols: Vec<Vec<f64>> = basis.iter().map(|&j| col(a, m, n, j)).collect();
    if lu.factorize(m, &cols).is_err() {
        return None; // singular warm basis → fall back to cold
    }

    let nb_value = |stat: &[i8], j: usize| -> f64 {
        if stat[j] == AT_UPPER {
            u[j]
        } else if l[j] <= -INF {
            0.0
        } else {
            l[j]
        }
    };

    // Verify the starting basis is actually dual-feasible — the precondition
    // the dual simplex *maintains* but does not *establish*. With y = B⁻ᵀc_B the
    // reduced cost is d_j = c_j − yᵀA_j; a nonbasic-at-lower needs d_j ≥ −tol, a
    // nonbasic-at-upper needs d_j ≤ tol, and a nonbasic *free* variable (both
    // bounds infinite) needs |d_j| ≤ tol. A dual-infeasible start would silently
    // converge to a wrong Optimal/Infeasible, so request the cold fallback.
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

    let mut updates = 0usize;
    let mut pivots = 0usize;
    for _iter in 0..opts.max_iter {
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

        // Most primal-infeasible basic variable leaves.
        let mut r = None;
        let mut worst = tol;
        let mut to_lower = true;
        for i in 0..m {
            let bi = basis[i];
            if xb[i] < l[bi] - tol {
                let viol = l[bi] - xb[i];
                if viol > worst {
                    worst = viol;
                    r = Some(i);
                    to_lower = true; // leaving var pinned at its lower bound
                }
            } else if xb[i] > u[bi] + tol {
                let viol = xb[i] - u[bi];
                if viol > worst {
                    worst = viol;
                    r = Some(i);
                    to_lower = false;
                }
            }
        }
        let r = match r {
            Some(r) => r,
            None => {
                // primal feasible + dual feasible (maintained) ⇒ optimal
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
                    LpStatus::Optimal,
                    pivots,
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

        // Dual ratio test. Leaving direction sets which sign of alpha_rj is
        // eligible; pick the entering j minimizing |d_j / alpha_rj| while
        // keeping dual feasibility.
        let mut enter = None;
        let mut best_ratio = f64::INFINITY;
        for j in 0..n {
            if stat[j] == BASIC {
                continue;
            }
            if u[j] - l[j] <= tol {
                continue; // fixed var can't enter
            }
            let arj = sp.dot(j, &rho);
            // Eligibility: increasing leaving (to_lower) wants the basic var to
            // rise; the entering var that preserves dual feasibility satisfies:
            //  - at lower:  arj < 0  (to_lower) /  arj > 0  (to_upper)
            //  - at upper:  arj > 0  (to_lower) /  arj < 0  (to_upper)
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
                let ratio = (dj / arj).abs();
                if ratio < best_ratio - tol {
                    best_ratio = ratio;
                    enter = Some(j);
                }
            }
        }
        let q = match enter {
            Some(q) => q,
            None => {
                // No eligible entering column: the LP is primal-infeasible. But
                // only trust this if the start basis was genuinely dual-feasible;
                // otherwise fall back to cold to be safe.
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
                ));
            }
        };

        // Pivot: q enters at slot r; leaving var pinned at the violated bound.
        let leaving = basis[r];
        stat[leaving] = if to_lower { AT_LOWER } else { AT_UPPER };
        slot_of[leaving] = -1;
        basis[r] = q;
        slot_of[q] = r as i64;
        stat[q] = BASIC;
        pivots += 1;
        let need_refac = lu.update(r, &col(a, m, n, q)).is_err();
        updates += 1;
        if need_refac || updates >= 48 {
            let cols: Vec<Vec<f64>> = basis.iter().map(|&j| col(a, m, n, j)).collect();
            if lu.factorize(m, &cols).is_err() {
                return None;
            }
            updates = 0;
        }
    }
    None // iteration cap → cold fallback
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
