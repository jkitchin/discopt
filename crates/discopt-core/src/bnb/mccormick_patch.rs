//! Closed-form McCormick envelope patching for the native spatial B&B node kernel
//! (issue #764, C1 build-order item 1).
//!
//! Across a spatial-B&B tree the lifted-relaxation LP *structure* is identical for
//! every node box — only the McCormick envelope rows and the auxiliary-variable
//! bounds depend on the box. The Python `incremental_mccormick.py` engine exploits
//! this: build the structure once, then per node recompute only the box-dependent
//! rows in closed form (~0.1 ms) instead of re-walking the expression DAG. This
//! module is the Rust port of that closed-form math, so the per-node envelope patch
//! can run inside the native node loop with no Python boundary crossing.
//!
//! Scope of this first increment: the three term families the Python `_patch`
//! actually dispatches on —
//!   * bilinear products `w = x_i * x_j` (`_bilinear_rows`, 4 McCormick rows);
//!   * integer powers `s = x_i^p` on a sign-definite box (`_monomial_rows`, the
//!     secant + tangents at `li`, the box midpoint, and `ui` — 4 rows; `p = 2` is
//!     the plain square);
//!   * affine squares `w = (a*x_j + c)^2` (`_affine_square_rows`, 4 rows).
//! Subsequent increments extend it to `sqrt` / general fractional-power / trilinear
//! terms (the coverage `tanksize` needs).
//!
//! CRITICAL — bound-neutrality: the formulas here mirror the functions
//! `IncrementalMcCormickLP._patch` calls (`_bilinear_rows`, `_monomial_rows`,
//! `_affine_square_rows` and their aux-bound helpers) *byte-for-byte*, so the ported
//! patcher reproduces the trusted cold build row-for-row. In particular the square
//! path is the **4-row** monomial form (with the box-midpoint tangent that the
//! uniform engine's `_emit_1d` emits), NOT the textbook 3-row square envelope — the
//! cold build uses the tighter midpoint-tangent hull, so a 3-row port would produce
//! a weaker bound and fail the neutrality gate. The `#[cfg(test)]` fixtures below
//! pin exact numeric equality against the Python reference.
//!
//! Row convention: every row is an inequality `sum(coeffs[k] * x[cols[k]]) <= rhs`,
//! i.e. the `A x <= b` form the LP solver consumes.

/// A single envelope inequality `sum(coeffs[k] * x[cols[k]]) <= rhs`.
///
/// Bilinear rows touch 3 columns `(x_i, x_j, w)`; monomial/affine-square rows touch
/// 2 `(x_i, s)`. Fixed small inline storage keeps the hot per-node patch
/// allocation-free.
#[derive(Clone, Copy, Debug)]
pub struct EnvRow {
    /// Column indices touched by this row (first `nnz` entries are meaningful).
    pub cols: [usize; 3],
    /// Row coefficients aligned with `cols` (first `nnz` entries are meaningful).
    pub coeffs: [f64; 3],
    /// Number of used entries in `cols`/`coeffs` (2 for 1-D terms, 3 for bilinear).
    pub nnz: usize,
    /// Right-hand side `b` of the inequality `sum(coeffs*x) <= rhs`.
    pub rhs: f64,
}

impl EnvRow {
    /// Evaluate `sum(coeffs*x) - rhs` (the row slack; `<= 0` iff satisfied).
    pub fn residual(&self, x: &[f64]) -> f64 {
        let mut s = -self.rhs;
        for k in 0..self.nnz {
            s += self.coeffs[k] * x[self.cols[k]];
        }
        s
    }
}

/// The 4 McCormick inequalities for `w = x_i * x_j` over `[li,ui] x [lj,uj]`.
///
/// Mirrors `incremental_mccormick._bilinear_rows`, each tuple
/// `(coeff_on_i, coeff_on_j, coeff_on_w, rhs)` of a `... <= rhs` row:
/// ```text
///   ( lj,  li, -1, li*lj)   # w >= lj*xi + li*xj - li*lj
///   ( uj,  ui, -1, ui*uj)   # w >= uj*xi + ui*xj - ui*uj
///   (-uj, -li,  1, -li*uj)  # w <= uj*xi + li*xj - li*uj
///   (-lj, -ui,  1, -ui*lj)  # w <= lj*xi + ui*xj - ui*lj
/// ```
#[inline]
pub fn bilinear_rows(
    i: usize,
    j: usize,
    w: usize,
    li: f64,
    ui: f64,
    lj: f64,
    uj: f64,
) -> [EnvRow; 4] {
    let row = |ci: f64, cj: f64, cw: f64, rhs: f64| EnvRow {
        cols: [i, j, w],
        coeffs: [ci, cj, cw],
        nnz: 3,
        rhs,
    };
    [
        row(lj, li, -1.0, li * lj),
        row(uj, ui, -1.0, ui * uj),
        row(-uj, -li, 1.0, -li * uj),
        row(-lj, -ui, 1.0, -ui * lj),
    ]
}

/// The 4 envelope rows for `s = x_i^p` over a **sign-definite** box `[li,ui]`
/// (secant + tangents at `li`, the box midpoint, and `ui`). Mirrors
/// `incremental_mccormick._monomial_rows`; `p = 2` is the plain square.
///
/// On a sign-definite box `x^p` is monotone and single-convexity: convex when `p`
/// is even or `li >= 0`; concave when `p` is odd and `ui <= 0`. Convex → the three
/// tangents underestimate and the secant overestimates; concave → the roles flip.
/// Matches the uniform engine's `_emit_1d` (3-tangent hull incl. the midpoint) so
/// the patch reproduces the cold build row-for-row.
///
/// Each row is `(coeff_on_x, coeff_on_s, rhs)`, columns `(i, s)`.
#[inline]
pub fn monomial_rows(i: usize, s: usize, li: f64, ui: f64, p: i32) -> [EnvRow; 4] {
    let mid = 0.5 * (li + ui);
    let fl = li.powi(p);
    let fm = mid.powi(p);
    let fu = ui.powi(p);
    let pf = p as f64;
    let dfl = pf * li.powi(p - 1);
    let dfm = pf * mid.powi(p - 1);
    let dfu = pf * ui.powi(p - 1);
    // Degenerate box (variable pinned by integer branching, li == ui): the secant
    // slope is 0/0; fall back to the endpoint derivative so the "secant" collapses
    // to the tangent at the pinned point. Guarded on EXACT zero width only — for any
    // positive width the true secant is the sound convex overestimator.
    let slope = if ui <= li {
        dfl
    } else {
        (fu - fl) / (ui - li)
    };
    let convex = (p % 2 == 0) || (li >= 0.0);
    let row = |cx: f64, cs: f64, rhs: f64| EnvRow {
        cols: [i, s, 0],
        coeffs: [cx, cs, 0.0],
        nnz: 2,
        rhs,
    };
    if convex {
        [
            row(dfl, -1.0, dfl * li - fl),        // tangent at li:  s >= f'(li)(x-li)+f(li)
            row(dfm, -1.0, dfm * mid - fm),       // tangent at midpoint
            row(dfu, -1.0, dfu * ui - fu),        // tangent at ui
            row(-slope, 1.0, fl - slope * li),    // secant (overestimator): s <= ...
        ]
    } else {
        [
            row(-dfl, 1.0, fl - dfl * li),        // tangent at li (overestimator): s <= ...
            row(-dfm, 1.0, fm - dfm * mid),       // tangent at midpoint
            row(-dfu, 1.0, fu - dfu * ui),        // tangent at ui
            row(slope, -1.0, slope * li - fl),    // secant (underestimator): s >= ...
        ]
    }
}

/// The 4 envelope rows for `w = (coeff*x + const)^2` over `x in [li,ui]` (secant +
/// tangents at `t_lo`, the midpoint, `t_hi`, where `t = coeff*x + const`). `t^2` is
/// convex for every `t` (no sign gating). Mirrors `_affine_square_rows`; each row is
/// `(coeff_on_x, coeff_on_w, rhs)`, columns `(j, w)`.
#[inline]
pub fn affine_square_rows(j: usize, w: usize, coeff: f64, cst: f64, li: f64, ui: f64) -> [EnvRow; 4] {
    let tl = coeff * li + cst;
    let tu = coeff * ui + cst;
    let (t_lo, t_hi) = if tl <= tu { (tl, tu) } else { (tu, tl) };
    let mid = 0.5 * (t_lo + t_hi);
    // t_hi + t_lo == (t_hi^2 - t_lo^2)/(t_hi - t_lo); at a degenerate base box it
    // already equals 2*t_lo == f'(t_lo), so no divide-by-zero guard is needed.
    let slope = t_hi + t_lo;
    let a = t_lo * t_lo - slope * t_lo;
    let row = |cx: f64, cw: f64, rhs: f64| EnvRow {
        cols: [j, w, 0],
        coeffs: [cx, cw, 0.0],
        nnz: 2,
        rhs,
    };
    [
        row(-slope * coeff, 1.0, a + slope * cst),                    // secant (overestimator)
        row(2.0 * t_lo * coeff, -1.0, t_lo * t_lo - 2.0 * t_lo * cst), // tangent at t_lo
        row(2.0 * mid * coeff, -1.0, mid * mid - 2.0 * mid * cst),     // tangent at midpoint
        row(2.0 * t_hi * coeff, -1.0, t_hi * t_hi - 2.0 * t_hi * cst), // tangent at t_hi
    ]
}

/// Auxiliary-variable bounds for `w = x_i * x_j` — the min/max over the box corners.
/// Mirrors `_bilinear_aux_bounds`.
#[inline]
pub fn bilinear_aux_bounds(li: f64, ui: f64, lj: f64, uj: f64) -> (f64, f64) {
    let c = [li * lj, li * uj, ui * lj, ui * uj];
    let mut lo = c[0];
    let mut hi = c[0];
    for &v in &c[1..] {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

/// Auxiliary-variable bounds for `s = x_i^p` over a sign-definite `[li,ui]` (monotone
/// there). Mirrors `_monomial_aux_bounds`.
#[inline]
pub fn monomial_aux_bounds(li: f64, ui: f64, p: i32) -> (f64, f64) {
    let a = li.powi(p);
    let b = ui.powi(p);
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Auxiliary-variable bounds for the squared base `t^2` over `t in [t_lo,t_hi]`
/// (0 if the base straddles zero). Mirrors `_square_aux_bounds`.
#[inline]
pub fn square_aux_bounds(t_lo: f64, t_hi: f64) -> (f64, f64) {
    if t_lo >= 0.0 {
        (t_lo * t_lo, t_hi * t_hi)
    } else if t_hi <= 0.0 {
        (t_hi * t_hi, t_lo * t_lo)
    } else {
        (0.0, (t_lo * t_lo).max(t_hi * t_hi))
    }
}

/// Auxiliary-variable bounds for `w = (coeff*x + const)^2` over `x in [li,ui]`.
/// Mirrors `_affine_square_aux_bounds`.
#[inline]
pub fn affine_square_aux_bounds(coeff: f64, cst: f64, li: f64, ui: f64) -> (f64, f64) {
    let tl = coeff * li + cst;
    let tu = coeff * ui + cst;
    let (t_lo, t_hi) = if tl <= tu { (tl, tu) } else { (tu, tl) };
    square_aux_bounds(t_lo, t_hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sat(row: &EnvRow, x: &[f64], tol: f64) -> bool {
        row.residual(x) <= tol
    }

    // Compare an EnvRow's (coeffs..., rhs) against a Python-reference tuple.
    fn assert_row_eq(r: &EnvRow, expect_coeffs: &[f64], expect_rhs: f64) {
        assert_eq!(r.nnz, expect_coeffs.len(), "nnz mismatch");
        for k in 0..r.nnz {
            assert!(
                (r.coeffs[k] - expect_coeffs[k]).abs() < 1e-12,
                "coeff[{k}] {} != {}",
                r.coeffs[k],
                expect_coeffs[k]
            );
        }
        assert!(
            (r.rhs - expect_rhs).abs() < 1e-12,
            "rhs {} != {}",
            r.rhs,
            expect_rhs
        );
    }

    // --- Differential fixtures: EXACT numeric equality vs the Python reference
    //     (generated from incremental_mccormick's _bilinear_rows / _monomial_rows /
    //     _affine_square_rows). This is the bound-neutrality gate at the formula
    //     level: the ported patcher must reproduce the cold build row-for-row. ---

    #[test]
    fn bilinear_matches_python_reference() {
        // _bilinear_rows(0,1,2,-1,3,2,5)
        let rows = bilinear_rows(0, 1, 2, -1.0, 3.0, 2.0, 5.0);
        assert_row_eq(&rows[0], &[2.0, -1.0, -1.0], -2.0);
        assert_row_eq(&rows[1], &[5.0, 3.0, -1.0], 15.0);
        assert_row_eq(&rows[2], &[-5.0, 1.0, 1.0], 5.0);
        assert_row_eq(&rows[3], &[-2.0, -3.0, 1.0], -6.0);
        assert_eq!(bilinear_aux_bounds(-1.0, 3.0, 2.0, 5.0), (-5.0, 15.0));
    }

    #[test]
    fn monomial_p2_matches_python_reference() {
        // _monomial_rows(-2,3,2)  (the plain square, 4-row midpoint-tangent form)
        let rows = monomial_rows(0, 1, -2.0, 3.0, 2);
        assert_row_eq(&rows[0], &[-4.0, -1.0], 4.0);
        assert_row_eq(&rows[1], &[1.0, -1.0], 0.25);
        assert_row_eq(&rows[2], &[6.0, -1.0], 9.0);
        assert_row_eq(&rows[3], &[-1.0, 1.0], 6.0);
        assert_eq!(monomial_aux_bounds(-2.0, 3.0, 2), (4.0, 9.0));
    }

    #[test]
    fn monomial_p3_matches_python_reference() {
        // _monomial_rows(1,4,3) — convex (li>=0)
        let rows = monomial_rows(0, 1, 1.0, 4.0, 3);
        assert_row_eq(&rows[0], &[3.0, -1.0], 2.0);
        assert_row_eq(&rows[1], &[18.75, -1.0], 31.25);
        assert_row_eq(&rows[2], &[48.0, -1.0], 128.0);
        assert_row_eq(&rows[3], &[-21.0, 1.0], -20.0);
        assert_eq!(monomial_aux_bounds(1.0, 4.0, 3), (1.0, 64.0));
    }

    #[test]
    fn monomial_p3_negative_box_matches_python_reference() {
        // _monomial_rows(-4,-1,3) — concave (p odd, ui<=0): roles flip
        let rows = monomial_rows(0, 1, -4.0, -1.0, 3);
        assert_row_eq(&rows[0], &[-48.0, 1.0], 128.0);
        assert_row_eq(&rows[1], &[-18.75, 1.0], 31.25);
        assert_row_eq(&rows[2], &[-3.0, 1.0], 2.0);
        assert_row_eq(&rows[3], &[21.0, -1.0], -20.0);
    }

    #[test]
    fn monomial_degenerate_box_matches_python_reference() {
        // _monomial_rows(2,2,2) — pinned variable: secant collapses to endpoint tangent
        let rows = monomial_rows(0, 1, 2.0, 2.0, 2);
        assert_row_eq(&rows[0], &[4.0, -1.0], 4.0);
        assert_row_eq(&rows[1], &[4.0, -1.0], 4.0);
        assert_row_eq(&rows[2], &[4.0, -1.0], 4.0);
        assert_row_eq(&rows[3], &[-4.0, 1.0], -4.0);
    }

    #[test]
    fn affine_square_matches_python_reference() {
        // _affine_square_rows(2,-1,-2,3)
        let rows = affine_square_rows(0, 1, 2.0, -1.0, -2.0, 3.0);
        assert_row_eq(&rows[0], &[0.0, 1.0], 25.0);
        assert_row_eq(&rows[1], &[-20.0, -1.0], 15.0);
        assert_row_eq(&rows[2], &[0.0, -1.0], 0.0);
        assert_row_eq(&rows[3], &[20.0, -1.0], 35.0);
        assert_eq!(affine_square_aux_bounds(2.0, -1.0, -2.0, 3.0), (0.0, 25.0));
    }

    // --- Envelope validity / geometry (belt-and-braces over the fixture equality). ---

    /// McCormick is valid at the four box corners: w = x_i*x_j satisfies every row.
    #[test]
    fn bilinear_exact_at_corners() {
        let (li, ui, lj, uj) = (-1.0, 3.0, 2.0, 5.0);
        let rows = bilinear_rows(0, 1, 2, li, ui, lj, uj);
        for &(xi, xj) in &[(li, lj), (li, uj), (ui, lj), (ui, uj)] {
            let x = [xi, xj, xi * xj];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "corner ({xi},{xj}) residual {}", r.residual(&x));
            }
        }
    }

    /// The true bilinear surface is inside the hull: over a grid, w=x_i*x_j is never
    /// cut by any of the 4 rows.
    #[test]
    fn bilinear_hull_contains_true_product() {
        let (li, ui, lj, uj) = (-2.0, 4.0, -3.0, 1.5);
        let rows = bilinear_rows(0, 1, 2, li, ui, lj, uj);
        let n = 11;
        for a in 0..=n {
            for b in 0..=n {
                let xi = li + (ui - li) * (a as f64) / (n as f64);
                let xj = lj + (uj - lj) * (b as f64) / (n as f64);
                let x = [xi, xj, xi * xj];
                for r in &rows {
                    assert!(sat(r, &x, 1e-9), "true product cut: residual {}", r.residual(&x));
                }
            }
        }
    }

    /// The relaxation is a strict OUTER approximation at the box centre (nonzero gap).
    #[test]
    fn bilinear_has_gap_at_center() {
        let (li, ui, lj, uj) = (0.0, 2.0, 0.0, 2.0);
        let rows = bilinear_rows(0, 1, 2, li, ui, lj, uj);
        let (xi, xj) = (1.0, 1.0); // true product = 1, admissible w spans [0,2]
        let x_low = [xi, xj, 0.0];
        let x_high = [xi, xj, 2.0];
        for r in &rows {
            assert!(sat(r, &x_low, 1e-9) && sat(r, &x_high, 1e-9));
        }
    }

    /// The 4-row monomial hull (p=2) validly contains s=x^2 over a straddling box and
    /// is exact at li/mid/ui.
    #[test]
    fn monomial_p2_valid_and_exact_at_tangent_points() {
        let (li, ui) = (-2.0, 3.0);
        let rows = monomial_rows(0, 1, li, ui, 2);
        let mid = 0.5 * (li + ui);
        for &xi in &[li, mid, ui] {
            let x = [xi, xi * xi];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "tangent point {xi} violated");
            }
        }
        let n = 20;
        for a in 0..=n {
            let xi = li + (ui - li) * (a as f64) / (n as f64);
            let x = [xi, xi * xi];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "square true point {xi} cut");
            }
        }
    }

    /// Affine-square hull validly contains (coeff*x+const)^2 over the box.
    #[test]
    fn affine_square_valid_on_grid() {
        let (coeff, cst, li, ui) = (2.0, -1.0, -2.0, 3.0);
        let rows = affine_square_rows(0, 1, coeff, cst, li, ui);
        let n = 20;
        for a in 0..=n {
            let xi = li + (ui - li) * (a as f64) / (n as f64);
            let t = coeff * xi + cst;
            let x = [xi, t * t];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "affine-square true point {xi} cut");
            }
        }
    }

    #[test]
    fn aux_bounds_match_corners() {
        assert_eq!(bilinear_aux_bounds(-1.0, 3.0, 2.0, 5.0), (-5.0, 15.0));
        assert_eq!(bilinear_aux_bounds(-2.0, 4.0, -3.0, 1.5), (-12.0, 6.0));
        assert_eq!(square_aux_bounds(-2.0, 3.0), (0.0, 9.0));
        assert_eq!(square_aux_bounds(1.0, 4.0), (1.0, 16.0));
        assert_eq!(square_aux_bounds(-4.0, -1.0), (1.0, 16.0));
    }
}
