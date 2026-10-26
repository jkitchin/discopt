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
use crate::presolve::directed::{add_up, div_down, div_up};
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
    tighten_bounds_csc_with_row(csc, m, n, l, u, b, is_int, tol, None)
}

/// [`tighten_bounds_csc`] plus one extra row `Σ_j w_j x_j <= rhs`, given as its
/// dense coefficients `w` (length `n`) and `rhs`, propagated alongside `A x = b`.
///
/// The extra row is an inequality, so it is carried as an equality with a virtual
/// slack column `s ∈ [0, INF]` (index `n`, dropped from the result). The MILP
/// driver passes the objective cutoff `c x <= U` here (#1953): an epigraph column
/// that no row bounds above is bounded by it.
#[allow(clippy::too_many_arguments)]
pub fn tighten_bounds_csc_with_row(
    csc: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    is_int: &[bool],
    tol: f64,
    extra: Option<(&[f64], f64)>,
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
    let mut rhs = b.to_vec();
    let int_ext;
    let mut is_int = is_int;
    if let Some((w, r)) = extra {
        let mut nz: Vec<(usize, f64)> =
            (0..n).filter(|&j| w[j] != 0.0).map(|j| (j, w[j])).collect();
        nz.push((n, 1.0));
        row_nz.push(nz);
        rhs.push(r);
        lo.push(0.0);
        hi.push(INF);
        int_ext = is_int
            .iter()
            .copied()
            .chain(std::iter::once(false))
            .collect::<Vec<_>>();
        is_int = &int_ext;
    }

    let mut infeasible = false;
    'rounds: for _round in 0..max_rounds {
        let mut changed = false;
        for (i, nz) in row_nz.iter().enumerate() {
            match fbbt_row(nz, rhs[i], &mut lo, &mut hi, is_int, tol, round_tol) {
                None => {
                    infeasible = true;
                    break 'rounds;
                }
                Some(c) => changed |= c,
            }
        }
        if !changed {
            break;
        }
    }
    lo.truncate(n);
    hi.truncate(n);
    PresolveResult {
        l: lo,
        u: hi,
        infeasible,
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

/// Knuth's error-free transform: `a + b = s + e` exactly, with `s = fl(a + b)`.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `|a * x - p|` for `p = fl(a * x)`, bounded above: the FMA remainder is exact
/// unless the product is near underflow, where a tiny absolute slack covers it.
#[inline]
fn prod_err(a: f64, p: f64, x: f64) -> f64 {
    if a != 0.0 && x != 0.0 && p.abs() < 1.0e-290 {
        return add_up(a.mul_add(x, -p).abs(), 4.0 * f64::MIN_POSITIVE);
    }
    a.mul_add(x, -p).abs()
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
    // Largest magnitude entering the running sums; sets the legacy error floor
    // `res_err` below.
    let mut max_abs_term = b_i.abs();
    // Exact rounding errors of the running sums, accumulated upward: every product's
    // FMA remainder and every addition's `two_sum` remainder. The float sums alone
    // are not enough -- `x0 + z + sum_{2000} y_i == 2^40` with `y_i <= 2^-14` loses
    // every `y_i` (each below half an ulp of 2^40), and the old `8 eps max|term|`
    // budget covered about one such rounding: presolve cut off the optimum and the
    // native MILP route certified 0.998 for a model whose optimum is 0.8779 (#1955
    // sweep). The tracked error is exactly what was lost (here 2000 * 2^-14), so a
    // row whose sums are exact pays nothing.
    let mut err_min = 0.0f64;
    let mut err_max = 0.0f64;
    for &(j, aij) in row_nz {
        if aij == 0.0 {
            continue;
        }
        let (cmin, cmin_inf, cmax, cmax_inf) = contrib(aij, lo[j], hi[j]);
        if cmin_inf {
            n_min_inf += 1;
        } else {
            let pe = prod_err(aij, cmin, if aij > 0.0 { lo[j] } else { hi[j] });
            let (sum, e) = two_sum(sum_min_finite, cmin);
            sum_min_finite = sum;
            err_min = add_up(add_up(err_min, e.abs()), pe);
            max_abs_term = max_abs_term.max(cmin.abs());
        }
        if cmax_inf {
            n_max_inf += 1;
        } else {
            let pe = prod_err(aij, cmax, if aij > 0.0 { hi[j] } else { lo[j] });
            let (sum, e) = two_sum(sum_max_finite, cmax);
            sum_max_finite = sum;
            err_max = add_up(add_up(err_max, e.abs()), pe);
            max_abs_term = max_abs_term.max(cmax.abs());
        }
    }
    // Legacy budget for `sum - term`, kept as a floor so a row whose tracked error is
    // smaller tightens exactly as before; the tracked error only ever widens.
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
            let (res_min, e1) = two_sum(sum_min_finite, -if k_min_inf { 0.0 } else { ck_min });
            let (t, e2) = two_sum(b_i, -res_min);
            let err = add_up(add_up(err_min, e1.abs()), e2.abs());
            term_ub = add_up(t, err.max(res_err));
        }
        let mut term_lb = -INF;
        if res_max_finite {
            let (res_max, e1) = two_sum(sum_max_finite, -if k_max_inf { 0.0 } else { ck_max });
            let (t, e2) = two_sum(b_i, -res_max);
            let err = add_up(add_up(err_max, e1.abs()), e2.abs());
            term_lb = -add_up(-t, err.max(res_err));
        }

        // Quotients rounded outward.
        let (mut new_lo, mut new_hi) = if aik > 0.0 {
            (
                if term_lb <= -INF {
                    -INF
                } else {
                    div_down(term_lb, aik)
                },
                if term_ub >= INF {
                    INF
                } else {
                    div_up(term_ub, aik)
                },
            )
        } else {
            (
                if term_ub >= INF {
                    -INF
                } else {
                    div_down(term_ub, aik)
                },
                if term_lb <= -INF {
                    INF
                } else {
                    div_up(term_lb, aik)
                },
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

    /// #1953: the extra `<=` row bounds a column no row of `A` touches (an OA
    /// epigraph column under the incumbent cutoff), through a virtual slack that
    /// is not returned. `A`: `x1 = 0.5`; extra row `2 x0 + x1 <= 3.5` -> `x0 <= 1.5`.
    #[test]
    fn extra_inequality_row_bounds_an_untouched_column() {
        use super::super::sparse::SparseCols;
        let sp = SparseCols::from_dense(&[0.0, 1.0], 1, 2);
        let (l, u) = (vec![0.0, 0.0], vec![INF, 1.0]);
        let w = [2.0, 1.0];
        let r = tighten_bounds_csc_with_row(
            &sp,
            1,
            2,
            &l,
            &u,
            &[0.5],
            &[false, false],
            1e-9,
            Some((&w, 3.5)),
        );
        assert!(!r.infeasible && r.l.len() == 2 && r.u.len() == 2);
        assert!(r.u[0] >= 1.5 && r.u[0] <= 1.5 + 1e-12, "{}", r.u[0]);
        let plain = tighten_bounds_csc(&sp, 1, 2, &l, &u, &[0.5], &[false, false], 1e-9);
        assert_eq!(plain.u[0], INF);
        // Under a cutoff nothing in the box satisfies, the box is proved empty.
        let r = tighten_bounds_csc_with_row(
            &sp,
            1,
            2,
            &l,
            &u,
            &[0.5],
            &[false, false],
            1e-9,
            Some((&w, 0.1)),
        );
        assert!(r.infeasible);
    }
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
    fn many_small_terms_next_to_a_huge_one_stay_in_the_bound() {
        // x0 + z + sum_{2000} y_i == 2^40, z <= 2^40, y_i <= 2^-14. Every y_i is
        // below half an ulp of 2^40, so the float activity sum loses all of them:
        // the old `8 eps max|term|` budget derived x0 >= -0.00195, cutting off the
        // feasible x0 = -2000 * 2^-14 = -0.1220703125 (z = 2^40, every y_i at its
        // upper bound). The bound must enclose it.
        let n_y = 2000;
        let n = n_y + 2;
        let mut a = vec![1.0; n];
        a[0] = 1.0;
        let c = vec![0.0; n];
        let mut l = vec![0.0; n];
        let mut u = vec![2.0f64.powi(-14); n];
        l[0] = -1e6;
        u[0] = 1e6;
        u[1] = 2.0f64.powi(40);
        let is_int = vec![false; n];
        let r = tighten_bounds(
            &view(&a, 1, n, &c, &l, &u),
            &[2.0f64.powi(40)],
            &is_int,
            1e-9,
        );
        assert!(!r.infeasible);
        let x0_min = -(n_y as f64) * 2.0f64.powi(-14);
        assert!(r.l[0] <= x0_min, "x0 lower {} cuts off {}", r.l[0], x0_min);
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
