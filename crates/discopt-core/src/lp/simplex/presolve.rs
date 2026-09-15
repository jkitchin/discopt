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
use crate::lp::simplex::sparse::SparseCols;
use crate::presolve::fbbt::FEAS_TOL;

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

    let mut row_nz: Vec<(usize, f64)> = Vec::new();
    for _round in 0..max_rounds {
        let mut changed = false;
        for i in 0..m {
            let row = &a[i * n..(i + 1) * n];
            row_nz.clear();
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    row_nz.push((j, v));
                }
            }
            match fbbt_row(&row_nz, b[i], &mut lo, &mut hi, is_int, tol, round_tol) {
                None => {
                    return PresolveResult {
                        l: lo,
                        u: hi,
                        infeasible: true,
                    }
                }
                Some(c) => changed |= c,
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

/// CSC port of [`tighten_bounds`] (docs/dev/sparse-milp-plan.md T3b4). Bit-identical:
/// per-row nonzeros are gathered from the CSC once (a single `O(nnz)` column sweep,
/// column-ascending per row exactly as the dense row scan produces them, reused
/// across rounds), and the shared [`fbbt_row`] runs the same code. FBBT's row loops
/// already skip structural zeros and the per-column tightening is independent of
/// column order within a row (the activity sums are fixed from the first loop), so
/// the CSC order yields the identical result. Never materializes the dense matrix.
#[allow(clippy::too_many_arguments)]
pub fn tighten_bounds_csc(
    csc: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    is_int: &[bool],
    tol: f64,
) -> PresolveResult {
    let mut lo = l.to_vec();
    let mut hi = u.to_vec();
    let max_rounds = 8;
    let round_tol = 1e-6;

    // Per-row nonzeros (CSR), built once and reused each round (the matrix is
    // constant; only lo/hi change).
    let mut row_nz: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    let (col_ptr, row_idx, vals) = csc.raw();
    for j in 0..n {
        for idx in col_ptr[j]..col_ptr[j + 1] {
            row_nz[row_idx[idx]].push((j, vals[idx]));
        }
    }

    for _round in 0..max_rounds {
        let mut changed = false;
        for i in 0..m {
            match fbbt_row(&row_nz[i], b[i], &mut lo, &mut hi, is_int, tol, round_tol) {
                None => {
                    return PresolveResult {
                        l: lo,
                        u: hi,
                        infeasible: true,
                    }
                }
                Some(c) => changed |= c,
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

/// A column's contribution `a_ij * x_j` over `[lo_j, hi_j]`, as
/// `(cmin, cmin_is_infinite, cmax, cmax_is_infinite)`.
///
/// The infinity flags are derived from the **bounds**, never from the products.
/// `INF` is a sentinel (1e20), not a true infinity, so `a_ij * hi_j` for an
/// unbounded column is an ordinary finite number whenever `|a_ij| < 1` — e.g.
/// `-0.5 * 1e20 = -5e19`, which fails a `<= -INF` test and is therefore booked as
/// a *finite* activity. That is unsound twice over: the row's activity range is no
/// longer recognised as unbounded, and the huge magnitude then annihilates every
/// smaller term in the running sum (`ulp(5e19) = 8192`), so the residual
/// `sum - term` comes back as `0.0` instead of its true value. On gear4's node LP
/// that fabricated the tightening `x >= 35.2244` on a column whose optimal value is
/// `0`, cutting the optimum out of the box and turning a `0.0` LP optimum into a
/// certified-`optimal` `184279.32`.
#[inline]
fn contrib(aij: f64, lo_j: f64, hi_j: f64) -> (f64, bool, f64, bool) {
    let lo_inf = lo_j <= -INF;
    let hi_inf = hi_j >= INF;
    if aij > 0.0 {
        (aij * lo_j, lo_inf, aij * hi_j, hi_inf)
    } else {
        (aij * hi_j, hi_inf, aij * lo_j, lo_inf)
    }
}

/// One FBBT round over a single row, given its nonzeros `(col, coeff)`. Mutates
/// `lo`/`hi` in place; returns `Some(changed)` or `None` if the box was proven empty.
/// Matrix-representation-independent — the dense and CSC entries differ ONLY in how
/// they gather `row_nz`, so both produce byte-identical tightenings.
fn fbbt_row(
    row_nz: &[(usize, f64)],
    b_i: f64,
    lo: &mut [f64],
    hi: &mut [f64],
    is_int: &[bool],
    tol: f64,
    round_tol: f64,
) -> Option<bool> {
    let mut changed = false;
    // Row activity range with infinity bookkeeping.
    let mut sum_min_finite = 0.0;
    let mut sum_max_finite = 0.0;
    let mut n_min_inf = 0usize;
    let mut n_max_inf = 0usize;
    // Largest magnitude entering the running sums. Floating-point accumulation
    // error is ~eps * max_term, and each residual below is formed by SUBTRACTING
    // one term from that sum, so this sets the absolute error scale of every
    // derived bound. See `res_err`.
    let mut max_abs_term = b_i.abs();
    for &(j, aij) in row_nz {
        if aij == 0.0 {
            continue;
        }
        let (cmin, cmin_inf, cmax, cmax_inf) = contrib(aij, lo[j], hi[j]);
        if cmin_inf {
            n_min_inf += 1;
        } else {
            sum_min_finite += cmin;
            max_abs_term = max_abs_term.max(cmin.abs());
        }
        if cmax_inf {
            n_max_inf += 1;
        } else {
            sum_max_finite += cmax;
            max_abs_term = max_abs_term.max(cmax.abs());
        }
    }
    // Error budget for `sum - term`. A residual is only meaningful to within the
    // rounding error already baked into the sum; widening each derived bound by
    // this keeps the contraction sound when a legitimately large (but finite)
    // bound swamps the smaller terms. Negligible in the normal case: for terms of
    // order 1e3 this is ~2e-13.
    let res_err = 8.0 * f64::EPSILON * max_abs_term;

    for &(k, aik) in row_nz {
        if aik == 0.0 {
            continue;
        }
        let (ck_min, k_min_inf, ck_max, k_max_inf) = contrib(aik, lo[k], hi[k]);

        let res_min_finite = n_min_inf - (k_min_inf as usize) == 0;
        let res_max_finite = n_max_inf - (k_max_inf as usize) == 0;

        let mut term_ub = INF;
        if res_min_finite {
            let res_min = sum_min_finite - if k_min_inf { 0.0 } else { ck_min };
            term_ub = b_i - res_min + res_err;
        }
        let mut term_lb = -INF;
        if res_max_finite {
            let res_max = sum_max_finite - if k_max_inf { 0.0 } else { ck_max };
            term_lb = b_i - res_max - res_err;
        }

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
        // Emptiness needs a crossing beyond the feasibility tolerance, not the
        // simplex pivot `tol`: the row data are rounded decimals, so a row can
        // imply `lo` a few 1e-9 above an `hi` another row implied. On HiGHS
        // check instance 2122 a 3.2e-9 crossing (row 307, col 51) was declared
        // empty and the driver returned a certified `infeasible` for a MILP
        // whose optimum is -187612.94. A sub-FEAS_TOL crossing is widened to
        // the pair of endpoints (the #907 idiom in `presolve::fbbt`), so neither
        // derivation's sound endpoint is lost and no inverted box escapes.
        let cross = lo[k] - hi[k];
        if cross > FEAS_TOL {
            return None;
        }
        if cross > 0.0 {
            let (a, b) = (lo[k].min(hi[k]), lo[k].max(hi[k]));
            lo[k] = a;
            hi[k] = b;
        }
    }
    Some(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(
        a: &'a [f64],
        m: usize,
        n: usize,
        c: &'a [f64],
        l: &'a [f64],
        u: &'a [f64],
    ) -> LpView<'a> {
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
        assert!(
            (r.l[0] - 1.0).abs() < 1e-9 && (r.u[0] - 1.0).abs() < 1e-9,
            "x0 fixed to 1"
        );
        assert!(
            (r.l[1] - 0.0).abs() < 1e-9 && (r.u[1] - 0.0).abs() < 1e-9,
            "x1 fixed to 0"
        );
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

    // #1229 panel, `2122.lp`: the file's rounded coefficients leave the exact box
    // empty by ~3e-9 on a continuous column (row `x51 − x258 + x259 − x525 = 0`), and
    // the absolute `1e-9` test certified the MILP `infeasible` at 0 nodes. HiGHS
    // (−187616.11) and SCIP (−187612.94) both solve it: a point that violates by less
    // than the feasibility tolerance is feasible by the same definition the driver's
    // own point gate uses, so an emptiness claim must exceed that tolerance.
    //
    // Minimal shape: `3 x0 = 1` puts x0 at 1/3; `x0 − x1 = 0` with `x1 ≤ 0.3333333`
    // (1/3 rounded to 7 digits) crosses by 3.3e-8. `x0 = x1 = 0.3333333` satisfies
    // both rows within 1e-7.
    #[test]
    fn crossing_within_feasibility_tolerance_is_not_an_empty_box() {
        let a = [3.0, 0.0, 1.0, -1.0];
        let c = [0.0, 0.0];
        let l = [0.0, 0.0];
        let u = [1.0, 0.3333333];
        let is_int = [false, false];
        let r = tighten_bounds(&view(&a, 2, 2, &c, &l, &u), &[1.0, 0.0], &is_int, 1e-9);
        assert!(
            !r.infeasible,
            "a 3.3e-8 crossing is data rounding, not an empty box (l={:?} u={:?})",
            r.l, r.u
        );
        let p = 0.3333333;
        for k in 0..2 {
            assert!(r.l[k] <= r.u[k], "col {k}: returned bounds must be ordered");
            assert!(
                r.l[k] - 1e-6 <= p && p <= r.u[k] + 1e-6,
                "col {k}: the tolerance-feasible point must stay in the box: [{}, {}]",
                r.l[k],
                r.u[k]
            );
        }
        let csc = SparseCols::from_dense(&a, 2, 2);
        let rc = tighten_bounds_csc(&csc, 2, 2, &l, &u, &[1.0, 0.0], &is_int, 1e-9);
        assert_eq!((rc.infeasible, &rc.l, &rc.u), (r.infeasible, &r.l, &r.u));
    }

    // The tolerance must not swallow a real empty box: the same shape with
    // `x1 ≤ 0.3333` crosses by 3.3e-5, well past any feasibility tolerance.
    #[test]
    fn crossing_beyond_feasibility_tolerance_is_still_infeasible() {
        let a = [3.0, 0.0, 1.0, -1.0];
        let c = [0.0, 0.0];
        let l = [0.0, 0.0];
        let u = [1.0, 0.3333];
        let is_int = [false, false];
        let r = tighten_bounds(&view(&a, 2, 2, &c, &l, &u), &[1.0, 0.0], &is_int, 1e-9);
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

    /// T3b4 gate: `tighten_bounds_csc` produces byte-identical tightened bounds to
    /// the dense `tighten_bounds` on a multi-row system that fires activity-range
    /// propagation and integer rounding (u[0]→3, u[1]→2), and on an infeasible box.
    #[test]
    fn csc_matches_dense_fbbt() {
        // 2x0 + 3x1 + s0 = 6 ; x0 − x1 + s1 = 1 ; x0,x1 binary→general int ≥ 0.
        let a = [2.0, 3.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0]; // 2×4
        let (m, n) = (2usize, 4usize);
        let c = [0.0; 4];
        let l = [0.0, 0.0, 0.0, 0.0];
        let u = [1e20, 1e20, 1e20, 1e20];
        let b = [6.0, 1.0];
        let is_int = [true, true, false, false];
        let dense = tighten_bounds(&view(&a, m, n, &c, &l, &u), &b, &is_int, 1e-9);
        let csc = tighten_bounds_csc(
            &SparseCols::from_dense(&a, m, n),
            m,
            n,
            &l,
            &u,
            &b,
            &is_int,
            1e-9,
        );
        assert_eq!(dense.infeasible, csc.infeasible, "infeasible verdict drift");
        assert_eq!(dense.l, csc.l, "lower bounds drift");
        assert_eq!(dense.u, csc.u, "upper bounds drift");
        assert!(
            dense.u[0] < 1e20 && dense.u[1] < 1e20,
            "sanity: expected tightening"
        );

        // Infeasible box: x0 ≥ 5 but the same 2x0 ≤ 6 forces x0 ≤ 3.
        let l2 = [5.0, 0.0, 0.0, 0.0];
        let d2 = tighten_bounds(&view(&a, m, n, &c, &l2, &u), &b, &is_int, 1e-9);
        let c2 = tighten_bounds_csc(
            &SparseCols::from_dense(&a, m, n),
            m,
            n,
            &l2,
            &u,
            &b,
            &is_int,
            1e-9,
        );
        assert!(
            d2.infeasible && c2.infeasible,
            "expected infeasible both paths"
        );
        assert_eq!(d2.l, c2.l);
        assert_eq!(d2.u, c2.u);
    }
}
