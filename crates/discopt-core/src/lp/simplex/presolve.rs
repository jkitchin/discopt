//! Dimension-preserving LP **presolve**: feasibility-based bound tightening.
//!
//! Given standard form `A x = b, l ≤ x ≤ u`, interval (FBBT) propagation tightens
//! each variable's bounds from every row: for row `i` and column `k`,
//! `a_ik x_k = b_i − Σ_{j≠k} a_ij x_j`, and the residual range over the current
//! box bounds `x_k`. Integer columns are rounded inward. The pass iterates to a
//! (capped) fixpoint and reports infeasibility if any `l_k > u_k`.
//!
//! This is a **sound contraction** — it only ever removes points that violate a
//! constraint within the current box, so it never cuts a feasible (let alone
//! optimal) solution. It changes no dimensions, so there is no postsolve: the
//! tightened bounds are used directly by the B&B tree and the node LPs. Used at
//! the MILP root to shrink the tree (fixing/tightening integer variables);
//! harmless (a no-op) on problems with no propagation, e.g. a lone knapsack row.

use crate::lp::crossover::LpView;

const INF: f64 = 1e20;

/// Tightened bounds plus an infeasibility flag.
pub struct PresolveResult {
    /// Tightened lower bounds (length `n`).
    pub l: Vec<f64>,
    /// Tightened upper bounds (length `n`).
    pub u: Vec<f64>,
    /// True if propagation proved the box empty (`l_k > u_k` for some `k`).
    pub infeasible: bool,
}

/// Tighten the bounds of `lp` (standard form `A x = b`) by interval bound
/// propagation to a fixpoint.
///
/// `is_int[k]` marks integer columns (rounded inward). `tol` is the comparison
/// tolerance. Returns the tightened bounds; on a proven empty box, `infeasible`
/// is set (and the bounds are left as last computed). The objective `lp.c` is
/// not used.
pub fn tighten_bounds(lp: &LpView<'_>, b: &[f64], is_int: &[bool], tol: f64) -> PresolveResult {
    let (a, m, n) = (lp.a, lp.m, lp.n);
    let mut lo = lp.l.to_vec();
    let mut hi = lp.u.to_vec();
    let max_rounds = 8;
    let round_tol = 1e-6;

    for _round in 0..max_rounds {
        let mut changed = false;
        for i in 0..m {
            let row = &a[i * n..(i + 1) * n];
            // Row activity range with infinity bookkeeping. For column j the
            // contribution interval is [cmin_j, cmax_j]; track the finite sums
            // and how many terms are ±infinite so a single-column residual can
            // tell whether *its* residual bound is finite.
            let mut sum_min_finite = 0.0;
            let mut sum_max_finite = 0.0;
            let mut n_min_inf = 0usize; // terms contributing −∞ to the lower activity
            let mut n_max_inf = 0usize; // terms contributing +∞ to the upper activity
            for j in 0..n {
                let aij = row[j];
                if aij == 0.0 {
                    continue;
                }
                let (cmin, cmax) = if aij > 0.0 {
                    (aij * lo[j], aij * hi[j])
                } else {
                    (aij * hi[j], aij * lo[j])
                };
                if cmin <= -INF {
                    n_min_inf += 1;
                } else {
                    sum_min_finite += cmin;
                }
                if cmax >= INF {
                    n_max_inf += 1;
                } else {
                    sum_max_finite += cmax;
                }
            }

            for k in 0..n {
                let aik = row[k];
                if aik == 0.0 {
                    continue;
                }
                let (ck_min, ck_max) = if aik > 0.0 {
                    (aik * lo[k], aik * hi[k])
                } else {
                    (aik * hi[k], aik * lo[k])
                };
                let k_min_inf = ck_min <= -INF;
                let k_max_inf = ck_max >= INF;

                // Residual (over j≠k) bounds, finite only when no *other* term is
                // infinite on that side.
                let res_min_finite = n_min_inf - (k_min_inf as usize) == 0;
                let res_max_finite = n_max_inf - (k_max_inf as usize) == 0;

                // a_ik x_k ≤ b_i − res_min  (upper on the term)
                let mut term_ub = INF;
                if res_min_finite {
                    let res_min = sum_min_finite - if k_min_inf { 0.0 } else { ck_min };
                    term_ub = b[i] - res_min;
                }
                // a_ik x_k ≥ b_i − res_max  (lower on the term)
                let mut term_lb = -INF;
                if res_max_finite {
                    let res_max = sum_max_finite - if k_max_inf { 0.0 } else { ck_max };
                    term_lb = b[i] - res_max;
                }

                // Translate term bounds to x_k bounds (flip if a_ik < 0).
                let (mut new_lo, mut new_hi) = if aik > 0.0 {
                    (
                        if term_lb <= -INF { -INF } else { term_lb / aik },
                        if term_ub >= INF { INF } else { term_ub / aik },
                    )
                } else {
                    (
                        if term_ub >= INF { -INF } else { term_ub / aik },
                        if term_lb <= -INF { INF } else { term_lb / aik },
                    )
                };
                if is_int[k] {
                    if new_lo > -INF {
                        new_lo = (new_lo - round_tol).ceil();
                    }
                    if new_hi < INF {
                        new_hi = (new_hi + round_tol).floor();
                    }
                }
                if new_lo > lo[k] + tol {
                    lo[k] = new_lo;
                    changed = true;
                }
                if new_hi < hi[k] - tol {
                    hi[k] = new_hi;
                    changed = true;
                }
                if lo[k] > hi[k] + tol {
                    return PresolveResult {
                        l: lo,
                        u: hi,
                        infeasible: true,
                    };
                }
            }
        }
        if !changed {
            break;
        }
    }

    PresolveResult {
        l: lo,
        u: hi,
        infeasible: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(a: &'a [f64], m: usize, n: usize, c: &'a [f64], l: &'a [f64], u: &'a [f64]) -> LpView<'a> {
        LpView { a, m, n, c, l, u }
    }

    #[test]
    fn lone_knapsack_structural_unchanged() {
        // 5Σx + s = 9, x∈[0,1], s∈[0,inf]. The integer x bounds cannot tighten
        // (the slack is unbounded above), but the slack itself legitimately
        // tightens: s = 9 − 5Σx ≤ 9 since Σx ≥ 0.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let c = [0.0; 5];
        let l = [0.0; 5];
        let u = [1.0, 1.0, 1.0, 1.0, INF];
        let is_int = [true, true, true, true, false];
        let r = tighten_bounds(&view(&a, 1, 5, &c, &l, &u), &[9.0], &is_int, 1e-9);
        assert!(!r.infeasible);
        assert_eq!(r.l, l.to_vec());
        assert_eq!(&r.u[..4], &[1.0, 1.0, 1.0, 1.0]); // structural bounds intact
        assert!((r.u[4] - 9.0).abs() < 1e-9, "slack upper {}", r.u[4]); // s ≤ 9
    }

    #[test]
    fn equality_tightens_and_fixes() {
        // Row 0: x0 + x1 = 1; row 1: x0 = 1 (singleton) → fixes x0=1, then x1=0.
        let a = [1.0, 1.0, 1.0, 0.0];
        let c = [0.0, 0.0];
        let l = [0.0, 0.0];
        let u = [1.0, 1.0];
        let is_int = [true, true];
        let r = tighten_bounds(&view(&a, 2, 2, &c, &l, &u), &[1.0, 1.0], &is_int, 1e-9);
        assert!(!r.infeasible);
        assert!((r.l[0] - 1.0).abs() < 1e-9 && (r.u[0] - 1.0).abs() < 1e-9, "x0 fixed to 1");
        assert!((r.l[1] - 0.0).abs() < 1e-9 && (r.u[1] - 0.0).abs() < 1e-9, "x1 fixed to 0");
    }

    #[test]
    fn detects_infeasible_box() {
        // x0 + x1 = 5 with x∈[0,1]: max activity 2 < 5 → infeasible.
        let a = [1.0, 1.0];
        let c = [0.0, 0.0];
        let l = [0.0, 0.0];
        let u = [1.0, 1.0];
        let is_int = [false, false];
        let r = tighten_bounds(&view(&a, 1, 2, &c, &l, &u), &[5.0], &is_int, 1e-9);
        assert!(r.infeasible);
    }

    #[test]
    fn tightens_continuous_upper() {
        // 2 x0 + x1 = 4, x0∈[0,10], x1∈[0,2]. From x1≥0: 2x0 ≤ 4 → x0 ≤ 2.
        // From x1≤2: 2x0 ≥ 2 → x0 ≥ 1.
        let a = [2.0, 1.0];
        let c = [0.0, 0.0];
        let l = [0.0, 0.0];
        let u = [10.0, 2.0];
        let is_int = [false, false];
        let r = tighten_bounds(&view(&a, 1, 2, &c, &l, &u), &[4.0], &is_int, 1e-9);
        assert!(!r.infeasible);
        assert!((r.u[0] - 2.0).abs() < 1e-9, "x0 upper {}", r.u[0]);
        assert!((r.l[0] - 1.0).abs() < 1e-9, "x0 lower {}", r.l[0]);
    }
}
