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
//! Current coverage — the term families the Python `_patch` dispatches on, plus the
//! univariate `sqrt` atom `tanksize` needs:
//!
//!   * bilinear products `w = x_i * x_j` (`_bilinear_rows`, 4 McCormick rows);
//!   * integer powers `s = x_i^p` on a sign-definite box (`_monomial_rows`, the
//!     secant + tangents at `li`, the box midpoint, and `ui` — 4 rows; `p = 2` is
//!     the plain square);
//!   * affine squares `w = (a*x_j + c)^2` (`_affine_square_rows`, 4 rows);
//!   * univariate `sqrt` (`_emit_1d`, secant + 3 tangents; the concave case).
//!
//! Subsequent increments extend the univariate path to general fractional-power /
//! log / exp atoms and to trilinear products.
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

/// Minimum base-box width below which the 1-D envelope collapses to the aux
/// interval floor (matches `uniform_relax._MIN_WIDTH`).
pub const MIN_WIDTH: f64 = 1e-12;

/// A univariate atom `w = f(t)` with `t = coeff*x + const`, relaxed by its exact
/// two-sided 1-D envelope (secant + tangents). The variants carry their own
/// curvature and closed-form `f`/`f'`; extend this enum to widen coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Univariate {
    /// `f(t) = sqrt(t)` — concave on `t >= 0` (the coverage `tanksize` needs).
    Sqrt,
}

/// Curvature of a univariate atom over its (sign-definite) box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Curv {
    /// Convex: secant overestimates, tangents underestimate.
    Convex,
    /// Concave: secant underestimates, tangents overestimate.
    Concave,
}

impl Univariate {
    /// The atom's curvature over its valid box.
    pub fn curvature(&self) -> Curv {
        match self {
            Univariate::Sqrt => Curv::Concave,
        }
    }

    /// Evaluate `f(t)`.
    pub fn f(&self, t: f64) -> f64 {
        match self {
            Univariate::Sqrt => t.sqrt(),
        }
    }

    /// Evaluate `f'(t)`.
    pub fn fp(&self, t: f64) -> f64 {
        match self {
            Univariate::Sqrt => 0.5 / t.sqrt(),
        }
    }
}

/// The exact two-sided 1-D envelope of `w = atom(coeff*x + const)` over
/// `x in [x_lo, x_hi]` — secant + tangents at `t_lo`, the box midpoint, and `t_hi`
/// (`t = coeff*x + const`). Mirrors `uniform_relax._emit_1d` row-for-row (same
/// secant/tangent construction, same sign convention per curvature, same
/// `(secant, tan@lo, tan@mid, tan@hi)` order).
///
/// Returns `None` — matching `_emit_1d`'s `tight = False` — when the base box is
/// degenerate/unbounded (`width < MIN_WIDTH`) or `f` is non-finite on the box
/// (e.g. `sqrt` of a negative endpoint); the caller then relies on the aux
/// interval bound alone. Each row is `(coeff_on_x, coeff_on_w, rhs)`, cols `(x, w)`.
#[inline]
pub fn univariate_rows(
    x: usize,
    w: usize,
    coeff: f64,
    cst: f64,
    x_lo: f64,
    x_hi: f64,
    atom: Univariate,
) -> Option<[EnvRow; 4]> {
    let ta = coeff * x_lo + cst;
    let tb = coeff * x_hi + cst;
    let (t_lo, t_hi) = if ta <= tb { (ta, tb) } else { (tb, ta) };
    let width = t_hi - t_lo;
    if !t_lo.is_finite() || !t_hi.is_finite() || width < MIN_WIDTH {
        return None;
    }
    let flo = atom.f(t_lo);
    let fhi = atom.f(t_hi);
    if !flo.is_finite() || !fhi.is_finite() {
        return None;
    }
    let slope = (fhi - flo) / width;
    let mid = 0.5 * (t_lo + t_hi);
    // sign = +1 for convex, -1 for concave (mirrors _emit_1d's ±1.0 dispatch).
    let s = match atom.curvature() {
        Curv::Convex => 1.0,
        Curv::Concave => -1.0,
    };
    let row = |cx: f64, cw: f64, rhs: f64| EnvRow {
        cols: [x, w, 0],
        coeffs: [cx, cw, 0.0],
        nnz: 2,
        rhs,
    };
    // secant: sign*w <= sign*(flo + slope*(t - t_lo)); intercept a = flo - slope*t_lo.
    let a = flo - slope * t_lo;
    let secant = row(-s * slope * coeff, s, s * (a + slope * cst));
    // tangent at t0: sign*w >= sign*(f(t0) + f'(t0)*(t - t0)).
    let tangent = |t0: f64| {
        let g = atom.f(t0);
        let gp = atom.fp(t0);
        let intercept = g - gp * t0;
        row(s * gp * coeff, -s, -s * intercept - s * gp * cst)
    };
    Some([secant, tangent(t_lo), tangent(mid), tangent(t_hi)])
}

/// Auxiliary-variable bounds for `w = sqrt(coeff*x + const)` over `x in [x_lo,x_hi]`
/// (sqrt is monotone increasing on `t >= 0`, so the aux range is the image of the
/// base-box endpoints). Returns `None` if the base box dips below 0 (sqrt undefined).
#[inline]
pub fn sqrt_aux_bounds(coeff: f64, cst: f64, x_lo: f64, x_hi: f64) -> Option<(f64, f64)> {
    let ta = coeff * x_lo + cst;
    let tb = coeff * x_hi + cst;
    let (t_lo, t_hi) = if ta <= tb { (ta, tb) } else { (tb, ta) };
    if t_lo < 0.0 {
        return None;
    }
    Some((t_lo.sqrt(), t_hi.sqrt()))
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

/// Interval `[lo, hi]` of the linear form `const + sum(coeffs[k]*x[cols[k]])` over the
/// node box `(box_lo, box_hi)` — the standard interval enclosure (a positive coeff
/// takes the low/high endpoint, a negative coeff flips them).
#[inline]
pub fn linform_interval(
    cols: &[usize],
    coeffs: &[f64],
    cst: f64,
    box_lo: &[f64],
    box_hi: &[f64],
) -> (f64, f64) {
    let mut lo = cst;
    let mut hi = cst;
    for (k, &c) in coeffs.iter().enumerate() {
        let j = cols[k];
        if c >= 0.0 {
            lo += c * box_lo[j];
            hi += c * box_hi[j];
        } else {
            lo += c * box_hi[j];
            hi += c * box_lo[j];
        }
    }
    (lo, hi)
}

/// The (up to) 4 McCormick rows for `w = A * B`, where `A` and `B` are affine forms
/// `A = a_const + sum a_coeffs[k]*x[a_cols[k]]` and likewise `B`, over the node box.
/// Mirrors `uniform_relax._emit_mccormick` byte-for-byte: with `A`/`B` interval
/// enclosures `[aL,aH]`/`[bL,bH]` the rows are
/// ```text
///   w >= bL*A + aL*B - aL*bL ,  w >= bH*A + aH*B - aH*bH
///   w <= bL*A + aH*B - aH*bL ,  w <= bH*A + aL*B - aL*bH
/// ```
/// (a row is skipped when an endpoint product is non-finite — the aux interval floor
/// stands). Each emitted row is pushed to `out` as `(cols, coeffs, rhs)` of a
/// `sum(coeffs*x) <= rhs` inequality, with coefficients on shared columns merged (so
/// `x_i*(x_i+…)` folds its two `x_i` contributions). This generalizes
/// [`bilinear_rows`] (both forms single bare columns) to the variable × linear-form
/// and linear-form × linear-form products the factorable engine emits.
#[allow(clippy::too_many_arguments)]
pub fn bilinear_linform_rows(
    a_cols: &[usize],
    a_coeffs: &[f64],
    a_const: f64,
    b_cols: &[usize],
    b_coeffs: &[f64],
    b_const: f64,
    w: usize,
    box_lo: &[f64],
    box_hi: &[f64],
    out: &mut Vec<(Vec<usize>, Vec<f64>, f64)>,
) {
    let (a_lo, a_hi) = linform_interval(a_cols, a_coeffs, a_const, box_lo, box_hi);
    let (b_lo, b_hi) = linform_interval(b_cols, b_coeffs, b_const, box_lo, box_hi);
    // (coef_a, coef_b, cc, sign) — identical order/values to `_emit_mccormick`.
    let specs = [
        (b_lo, a_lo, -a_lo * b_lo, 1.0f64),
        (b_hi, a_hi, -a_hi * b_hi, 1.0),
        (b_lo, a_hi, -a_hi * b_lo, -1.0),
        (b_hi, a_lo, -a_lo * b_hi, -1.0),
    ];
    for (coef_a, coef_b, cc, sign) in specs {
        if !(coef_a.is_finite() && coef_b.is_finite() && cc.is_finite()) {
            continue;
        }
        // Accumulate coefficients per column (a and b may share columns), plus w.
        // Column order in the emitted row: w, then a's columns, then any new b cols.
        let mut cols: Vec<usize> = Vec::with_capacity(1 + a_cols.len() + b_cols.len());
        let mut coeffs: Vec<f64> = Vec::with_capacity(cols.capacity());
        let idx_of = |cols: &mut Vec<usize>, coeffs: &mut Vec<f64>, col: usize| -> usize {
            if let Some(p) = cols.iter().position(|&c| c == col) {
                p
            } else {
                cols.push(col);
                coeffs.push(0.0);
                cols.len() - 1
            }
        };
        let pw = idx_of(&mut cols, &mut coeffs, w);
        coeffs[pw] += -sign;
        for (k, &ac) in a_coeffs.iter().enumerate() {
            let p = idx_of(&mut cols, &mut coeffs, a_cols[k]);
            coeffs[p] += sign * coef_a * ac;
        }
        for (k, &bc) in b_coeffs.iter().enumerate() {
            let p = idx_of(&mut cols, &mut coeffs, b_cols[k]);
            coeffs[p] += sign * coef_b * bc;
        }
        let rhs = -sign * (cc + coef_a * a_const + coef_b * b_const);
        out.push((cols, coeffs, rhs));
    }
}

/// Auxiliary bounds for `w = A * B` — the interval product of the two forms'
/// enclosures over the box. Mirrors `_interval_mul(interval(A), interval(B))`.
#[allow(clippy::too_many_arguments)]
pub fn bilinear_linform_aux_bounds(
    a_cols: &[usize],
    a_coeffs: &[f64],
    a_const: f64,
    b_cols: &[usize],
    b_coeffs: &[f64],
    b_const: f64,
    box_lo: &[f64],
    box_hi: &[f64],
) -> (f64, f64) {
    let (a_lo, a_hi) = linform_interval(a_cols, a_coeffs, a_const, box_lo, box_hi);
    let (b_lo, b_hi) = linform_interval(b_cols, b_coeffs, b_const, box_lo, box_hi);
    let p = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi];
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in p {
        if v.is_nan() {
            continue;
        }
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi {
        (f64::NEG_INFINITY, f64::INFINITY)
    } else {
        (lo, hi)
    }
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

    #[test]
    fn sqrt_bare_matches_emit_1d_reference() {
        // _emit_1d(sqrt, t=x, [1,4], concave): bare sqrt(x), col x=0, col w=1.
        let rows = univariate_rows(0, 1, 1.0, 0.0, 1.0, 4.0, Univariate::Sqrt).unwrap();
        assert_row_eq(&rows[0], &[0.333333333333, -1.0], -0.666666666667); // secant
        assert_row_eq(&rows[1], &[-0.5, 1.0], 0.5); // tangent @ t_lo=1
        assert_row_eq(&rows[2], &[-0.316227766017, 1.0], 0.790569415042); // tangent @ mid=2.5
        assert_row_eq(&rows[3], &[-0.25, 1.0], 1.0); // tangent @ t_hi=4
        assert_eq!(sqrt_aux_bounds(1.0, 0.0, 1.0, 4.0), Some((1.0, 2.0)));
    }

    #[test]
    fn sqrt_affine_matches_emit_1d_reference() {
        // _emit_1d(sqrt, t=2x+1, x in [1,4] -> t in [3,9], concave).
        let rows = univariate_rows(0, 1, 2.0, 1.0, 1.0, 4.0, Univariate::Sqrt).unwrap();
        assert_row_eq(&rows[0], &[0.422649730810, -1.0], -1.309401076759); // secant
        assert_row_eq(&rows[1], &[-0.577350269190, 1.0], 1.154700538379); // tangent @ t_lo=3
        assert_row_eq(&rows[2], &[-0.408248290464, 1.0], 1.428869016624); // tangent @ mid=6
        assert_row_eq(&rows[3], &[-0.333333333333, 1.0], 1.666666666667); // tangent @ t_hi=9
    }

    /// Degenerate/undefined boxes yield no tight rows (aux floor only), matching
    /// `_emit_1d`'s `tight = False` return.
    #[test]
    fn sqrt_degenerate_or_undefined_yields_none() {
        // pinned base box (width 0)
        assert!(univariate_rows(0, 1, 1.0, 0.0, 2.0, 2.0, Univariate::Sqrt).is_none());
        // base dips below zero -> sqrt undefined at the low endpoint
        assert!(univariate_rows(0, 1, 1.0, 0.0, -1.0, 4.0, Univariate::Sqrt).is_none());
        assert!(sqrt_aux_bounds(1.0, 0.0, -1.0, 4.0).is_none());
    }

    /// Concave sqrt hull validly contains w = sqrt(t) over the box, and the secant
    /// underestimates while the tangents overestimate.
    #[test]
    fn sqrt_hull_valid_and_two_sided() {
        let (coeff, cst, x_lo, x_hi) = (2.0, 1.0, 1.0, 4.0);
        let rows = univariate_rows(0, 1, coeff, cst, x_lo, x_hi, Univariate::Sqrt).unwrap();
        let n = 20;
        for a in 0..=n {
            let xi = x_lo + (x_hi - x_lo) * (a as f64) / (n as f64);
            let w = (coeff * xi + cst).sqrt();
            let x = [xi, w];
            for r in &rows {
                assert!(sat(r, &x, 1e-9), "sqrt true point {xi} cut: residual {}", r.residual(&x));
            }
        }
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

    // Compare a variable-width row (cols, coeffs, rhs) against a Python-reference
    // map {col: coeff} + rhs (order-independent).
    fn assert_wide_row(
        row: &(Vec<usize>, Vec<f64>, f64),
        expect: &[(usize, f64)],
        expect_rhs: f64,
    ) {
        let (cols, coeffs, rhs) = row;
        let mut got: Vec<(usize, f64)> = cols
            .iter()
            .zip(coeffs.iter())
            .filter(|(_, &c)| c.abs() > 1e-12)
            .map(|(&j, &c)| (j, c))
            .collect();
        got.sort_by_key(|(j, _)| *j);
        let mut exp: Vec<(usize, f64)> = expect.to_vec();
        exp.sort_by_key(|(j, _)| *j);
        assert_eq!(got.len(), exp.len(), "col count: {got:?} vs {exp:?}");
        for ((gj, gc), (ej, ec)) in got.iter().zip(exp.iter()) {
            assert_eq!(gj, ej, "col mismatch");
            assert!((gc - ec).abs() < 1e-9, "coeff {gc} != {ec} at col {gj}");
        }
        assert!((rhs - expect_rhs).abs() < 1e-9, "rhs {rhs} != {expect_rhs}");
    }

    /// EXACT match to `_emit_mccormick` for a variable × linear-form product
    /// `w = x0 * (1.7 x1 + 0.4 x2 + 0.5)` over x0∈[1,3], x1∈[2,5], x2∈[-1,4].
    #[test]
    fn bilinear_linform_matches_emit_mccormick_var_times_form() {
        // box (only the referenced columns matter; w=9 needs the arrays long enough).
        let lo = vec![1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hi = vec![3.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = Vec::new();
        bilinear_linform_rows(&[0], &[1.0], 0.0, &[1, 2], &[1.7, 0.4], 0.5, 9, &lo, &hi, &mut out);
        assert_eq!(out.len(), 4);
        assert_wide_row(&out[0], &[(0, 3.5), (1, 1.7), (2, 0.4), (9, -1.0)], 3.0);
        assert_wide_row(&out[1], &[(0, 10.6), (1, 5.1), (2, 1.2), (9, -1.0)], 30.3);
        assert_wide_row(&out[2], &[(0, -3.5), (1, -5.1), (2, -1.2), (9, 1.0)], -9.0);
        assert_wide_row(&out[3], &[(0, -10.6), (1, -1.7), (2, -0.4), (9, 1.0)], -10.1);
        // aux bounds = interval(A)*interval(B) = [1,3]*[3.5,10.6] = [3.5, 31.8].
        let (alo, ahi) = bilinear_linform_aux_bounds(&[0], &[1.0], 0.0, &[1, 2], &[1.7, 0.4], 0.5, &lo, &hi);
        assert!((alo - 3.5).abs() < 1e-9 && (ahi - 31.8).abs() < 1e-9);
    }

    /// Shared-column merge: `w = x0 * (x0 + x1)` — the two x0 contributions fold into
    /// one coefficient. Matches `_emit_mccormick`.
    #[test]
    fn bilinear_linform_merges_shared_columns() {
        let lo = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hi = vec![2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = Vec::new();
        bilinear_linform_rows(&[0], &[1.0], 0.0, &[0, 1], &[1.0, 1.0], 0.0, 5, &lo, &hi, &mut out);
        assert_eq!(out.len(), 4);
        assert_wide_row(&out[0], &[(0, 2.0), (1, 1.0), (5, -1.0)], 1.0);
        assert_wide_row(&out[1], &[(0, 7.0), (1, 2.0), (5, -1.0)], 10.0);
        assert_wide_row(&out[2], &[(0, -3.0), (1, -2.0), (5, 1.0)], -2.0);
        assert_wide_row(&out[3], &[(0, -6.0), (1, -1.0), (5, 1.0)], -5.0);
    }

    /// A single-column × single-column linform product reproduces `bilinear_rows`
    /// (the general path subsumes the special one).
    #[test]
    fn bilinear_linform_reduces_to_bilinear_rows() {
        let lo = vec![-1.0, 2.0, 0.0];
        let hi = vec![3.0, 5.0, 0.0];
        let special = bilinear_rows(0, 1, 2, -1.0, 3.0, 2.0, 5.0);
        let mut general = Vec::new();
        bilinear_linform_rows(&[0], &[1.0], 0.0, &[1], &[1.0], 0.0, 2, &lo, &hi, &mut general);
        assert_eq!(general.len(), 4);
        // Same SET of rows (the McCormick hull) — `_emit_mccormick` emits the two
        // over-rows in the opposite order to `_bilinear_rows`, so match set-wise: each
        // special row is reproduced by some general row.
        let norm = |cols: &[usize], coeffs: &[f64], rhs: f64| -> Vec<(usize, i64)> {
            let mut v: Vec<(usize, i64)> = cols
                .iter()
                .zip(coeffs.iter())
                .filter(|(_, &c)| c.abs() > 1e-12)
                .map(|(&j, &c)| (j, (c * 1e6).round() as i64))
                .collect();
            v.push((usize::MAX, (rhs * 1e6).round() as i64));
            v.sort();
            v
        };
        let gset: Vec<_> = general.iter().map(|(c, k, r)| norm(c, k, *r)).collect();
        for s in &special {
            let key = norm(&s.cols[..s.nnz], &s.coeffs[..s.nnz], s.rhs);
            assert!(gset.contains(&key), "special row {key:?} not in general set");
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
