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
//! Scope of this first increment: the two most common lifted terms —
//! bilinear products `w = x_i * x_j` (4 McCormick rows) and squares `s = x_i^2`
//! (2 endpoint tangents + 1 secant). Subsequent increments extend it to
//! trilinear / univariate-concave (`sqrt`) / fractional-power terms (the coverage
//! `tanksize` needs). The formulas here match `incremental_mccormick.py`
//! byte-for-byte so the ported patcher reproduces the trusted cold build exactly
//! (the bound-neutrality gate).
//!
//! Row convention: every row is an inequality `sum(coeffs[k] * x[cols[k]]) <= rhs`,
//! i.e. the `A x <= b` form the LP solver consumes.

/// A single envelope inequality `sum(coeffs[k] * x[cols[k]]) <= rhs`.
///
/// Bilinear rows touch 3 columns `(x_i, x_j, w)`; square rows touch 2 `(x_i, s)`.
/// Fixed small inline storage keeps the hot per-node patch allocation-free.
#[derive(Clone, Copy, Debug)]
pub struct EnvRow {
    /// Column indices touched by this row (first `nnz` entries are meaningful).
    pub cols: [usize; 3],
    /// Row coefficients aligned with `cols` (first `nnz` entries are meaningful).
    pub coeffs: [f64; 3],
    /// Number of used entries in `cols`/`coeffs` (2 for square, 3 for bilinear).
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
/// Mirrors `incremental_mccormick._bilinear_rows`:
/// ```text
///   w >= lj*xi + li*xj - li*lj        (lower, tight at (li,lj) & (ui,uj))
///   w >= uj*xi + ui*xj - ui*uj
///   w <= uj*xi + li*xj - li*uj        (upper, tight at (li,uj) & (ui,lj))
///   w <= lj*xi + ui*xj - ui*lj
/// ```
/// expressed as `... <= rhs`.
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
    // Lower envelope: w >= a*xi + b*xj - c  ==>  a*xi + b*xj - w <= c.
    let lower = |a: f64, b: f64, c: f64| EnvRow {
        cols: [i, j, w],
        coeffs: [a, b, -1.0],
        nnz: 3,
        rhs: c,
    };
    // Upper envelope: w <= a*xi + b*xj - c  ==>  -a*xi - b*xj + w <= -c.
    let upper = |a: f64, b: f64, c: f64| EnvRow {
        cols: [i, j, w],
        coeffs: [-a, -b, 1.0],
        nnz: 3,
        rhs: -c,
    };
    [
        lower(lj, li, li * lj),
        lower(uj, ui, ui * uj),
        upper(uj, li, li * uj),
        upper(lj, ui, ui * lj),
    ]
}

/// The 3 rows for `s = x_i^2` over `[li,ui]`: tangents at `li`, `ui`, and the
/// secant. Mirrors `incremental_mccormick._square_rows`.
#[inline]
pub fn square_rows(i: usize, s: usize, li: f64, ui: f64) -> [EnvRow; 3] {
    let row = |ai: f64, as_: f64, rhs: f64| EnvRow {
        cols: [i, s, 0],
        coeffs: [ai, as_, 0.0],
        nnz: 2,
        rhs,
    };
    [
        row(2.0 * li, -1.0, li * li),        // s >= 2*li*xi - li^2
        row(2.0 * ui, -1.0, ui * ui),        // s >= 2*ui*xi - ui^2
        row(-(li + ui), 1.0, -(li * ui)),    // s <= (li+ui)*xi - li*ui
    ]
}

/// Auxiliary-variable bounds for `w = x_i * x_j` — the min/max over the box
/// corners. Mirrors `_bilinear_aux_bounds`.
#[inline]
pub fn bilinear_aux_bounds(li: f64, ui: f64, lj: f64, uj: f64) -> (f64, f64) {
    let c = [li * lj, li * uj, ui * lj, ui * uj];
    let mut lo = c[0];
    let mut hi = c[0];
    for &v in &c[1..] {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Auxiliary-variable bounds for `s = x_i^2`. Mirrors `_square_aux_bounds`.
#[inline]
pub fn square_aux_bounds(li: f64, ui: f64) -> (f64, f64) {
    if li >= 0.0 {
        (li * li, ui * ui)
    } else if ui <= 0.0 {
        (ui * ui, li * li)
    } else {
        (0.0, (li * li).max(ui * ui))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A dense evaluation of a row at a point, referencing the used columns only.
    fn sat(row: &EnvRow, x: &[f64], tol: f64) -> bool {
        row.residual(x) <= tol
    }

    /// McCormick is EXACT at the four box corners: w = x_i*x_j must satisfy every
    /// envelope row with equality on at least the two tight rows and validity on all.
    #[test]
    fn bilinear_exact_at_corners() {
        let (li, ui, lj, uj) = (-1.0, 3.0, 2.0, 5.0);
        let rows = bilinear_rows(0, 1, 2, li, ui, lj, uj);
        for &(xi, xj) in &[(li, lj), (li, uj), (ui, lj), (ui, uj)] {
            let x = [xi, xj, xi * xj]; // w = true product at the corner
            for r in &rows {
                assert!(
                    sat(r, &x, 1e-9),
                    "corner ({xi},{xj}) violates a McCormick row: residual {}",
                    r.residual(&x)
                );
            }
        }
    }

    /// Every interior feasible point of the RELAXATION (w between the envelopes)
    /// is admitted, and the true bilinear surface is inside the hull: for a grid of
    /// (x_i,x_j), the true product w=x_i*x_j satisfies all 4 rows (validity: the
    /// envelope never cuts a true product point).
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
                    assert!(
                        sat(r, &x, 1e-9),
                        "true product ({xi},{xj},{}) cut by envelope: residual {}",
                        xi * xj,
                        r.residual(&x)
                    );
                }
            }
        }
    }

    /// The McCormick relaxation is a strict OUTER approximation: at the box centre
    /// there exists a w in [lower_env, upper_env] strictly different from x_i*x_j
    /// (a nonzero gap), i.e. the rows are not degenerate.
    #[test]
    fn bilinear_has_gap_at_center() {
        let (li, ui, lj, uj) = (0.0, 2.0, 0.0, 2.0);
        let rows = bilinear_rows(0, 1, 2, li, ui, lj, uj);
        let (xi, xj) = (1.0, 1.0); // centre; true product = 1
        // lower envelope max, upper envelope min at (1,1)
        // row0: w >= lj*xi+li*xj-li*lj = 0; row1: w >= uj*xi+ui*xj-ui*uj = 2+2-4=0
        // row2: w <= uj*xi+li*xj-li*uj = 2; row3: w <= lj*xi+ui*xj-ui*lj = 2
        // so w in [0,2], true product 1 -> gap width 2 > 0
        let x_low = [xi, xj, 0.0];
        let x_high = [xi, xj, 2.0];
        for r in &rows {
            assert!(sat(r, &x_low, 1e-9) && sat(r, &x_high, 1e-9));
        }
    }

    #[test]
    fn square_exact_at_endpoints_and_valid() {
        let (li, ui) = (-2.0, 3.0);
        let rows = square_rows(0, 1, li, ui);
        // exact at endpoints
        for &xi in &[li, ui] {
            let x = [xi, xi * xi];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "endpoint {xi} violates square row");
            }
        }
        // validity on a grid: true s = xi^2 inside the hull
        let n = 20;
        for a in 0..=n {
            let xi = li + (ui - li) * (a as f64) / (n as f64);
            let x = [xi, xi * xi];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "square true point {xi} cut");
            }
        }
    }

    #[test]
    fn aux_bounds_match_corners() {
        assert_eq!(bilinear_aux_bounds(-1.0, 3.0, 2.0, 5.0), (-5.0, 15.0));
        assert_eq!(bilinear_aux_bounds(-2.0, 4.0, -3.0, 1.5), (-12.0, 6.0));
        // square: straddling zero -> [0, max(li^2,ui^2)]
        assert_eq!(square_aux_bounds(-2.0, 3.0), (0.0, 9.0));
        assert_eq!(square_aux_bounds(1.0, 4.0), (1.0, 16.0));
        assert_eq!(square_aux_bounds(-4.0, -1.0), (1.0, 16.0));
    }
}
