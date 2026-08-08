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
//!
//! OUTWARD ROUNDING (issue #956) — the invariant every generator here must hold is
//!
//! > for every `x` in the box, the point `(x, f(x))` satisfies every emitted row
//! > and lies inside the auxiliary column's own bounds.
//!
//! In exact arithmetic the closed forms below hold it by construction. In f64 they
//! do not: an envelope rhs is a *cancelling* combination of box-endpoint quantities
//! (`slope*x0 - f(x0)`), and the auxiliary bound is an independent rounding of the
//! same quantity, so at the box corner the two disagree by ~1 ulp of the row's
//! magnitude. That is an absolute residual no LP feasibility tolerance absorbs once
//! the magnitude is large. Measured worst violations of an *exactly feasible* point,
//! by a random sweep over these generators: `1.3e-1` (affine square), `2.0e-3`
//! (affine-form product), `1.2e-4` (square), `9.5e-7` (bilinear), `6.4e1` (cubic) —
//! and a cubic lift needs a box of only `~5e3` to cut its own graph by `6.1e-5`. The
//! node LP then has no feasible point at all and the simplex returns `Numerical` —
//! 49 % of `nvs20`'s node LPs, and over half of `ex1252`'s under the #707 reform
//! flags.
//!
//! So every rhs is relaxed **outward** by an ulp-scaled guard computed from the
//! magnitudes of the terms that formed it, and every auxiliary bound is widened
//! outward by the matching guard. A flat epsilon cannot do this job — the
//! magnitudes here span `1e-12` to `1e18`. This mirrors what `spatial_propagate`'s
//! header already documents for every bound it writes; the row generators simply
//! never had the equivalent.
//!
//! Direction matters for soundness: the guard only ever *loosens* the relaxation
//! (a larger rhs admits more points, a wider aux box admits more points), so it can
//! never cut a feasible point and never produce a bound above the true optimum. Its
//! cost is `~3.6e-15` relative on every bound — below every tolerance the solver
//! reasons with (abs `1e-6`, rel `1e-4`, integrality `1e-5`).

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

/// The LP layer's infinity sentinel (`lp/simplex/primal.rs::INF`). A bound at or
/// beyond it means "unbounded", not "this large" — it must never scale a guard
/// (`1e20 * ULP_GUARD` would be a `3.6e5` relaxation, not a rounding repair).
const INF_SENTINEL: f64 = 1e20;

/// Size of the outward-rounding guard, in `f64::EPSILON` units of the row's term
/// magnitude (issue #956).
///
/// Sized from measurement, not taste: sweeping all four generators over ten orders
/// of magnitude, the worst violation of an exactly feasible point `(x, f(x))` was
/// **1.9 ulp** of the row's term magnitude. 16 ulp is that worst case with an ~8x
/// margin. The price is a `3.6e-15` relative loosening of every envelope.
const ULP_GUARD: f64 = 16.0 * f64::EPSILON;

/// Whether the outward rounding is applied (issue #956).
///
/// Default ON — the guarded rows are the ones that hold the relaxation's defining
/// invariant. `DISCOPT_ENVELOPE_OUTWARD_ROUND=0` restores the legacy unguarded
/// generators bit-for-bit, which is what the differential panel A/Bs. Read once.
fn outward_rounding_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DISCOPT_ENVELOPE_OUTWARD_ROUND")
            .ok()
            .map(|v| !matches!(v.trim(), "0" | "false" | "False"))
            .unwrap_or(true)
    })
}

/// Magnitude contribution of a box-derived quantity to a guard.
///
/// Non-finite values and anything at/past the [`INF_SENTINEL`] contribute **zero**:
/// on an unbounded box a *relative* guard has no meaning, and scaling by `1e20`
/// would swamp the relaxation. Under-guarding there is exactly the pre-#956
/// behaviour, so this is never a regression.
#[inline]
pub fn bounded_mag(v: f64) -> f64 {
    if v.is_finite() && v.abs() < INF_SENTINEL {
        v.abs()
    } else {
        0.0
    }
}

/// The outward slack for a row rhs / bound whose terms have total magnitude
/// `mag_sum` (issue #956). Always `>= 0`, so adding it to a `<=` rhs (or to the
/// high side of a bound, negated on the low side) can only ever *relax*.
///
/// `max(1.0)` floors the guard at [`ULP_GUARD`] absolute so a row whose magnitudes
/// are all sub-unit still gets its own last-bit repaired.
#[inline]
pub fn outward_slack(mag_sum: f64) -> f64 {
    if !outward_rounding_enabled() {
        return 0.0;
    }
    ULP_GUARD * mag_sum.max(1.0)
}

/// Widen `[lo, hi]` outward by the guard for magnitude `mag_sum` (issue #956), so
/// a value the box maps to can never fall outside its own auxiliary bounds by a
/// rounding.
#[inline]
fn widen(lo: f64, hi: f64, mag_sum: f64) -> (f64, f64) {
    let g = outward_slack(mag_sum);
    (lo - g, hi + g)
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
    // #956: the point (x_i, x_j, x_i*x_j) must satisfy every row it is on. Guard by
    // the row's own term magnitudes over the box (|w| <= imax*jmax at a corner).
    let imax = bounded_mag(li).max(bounded_mag(ui));
    let jmax = bounded_mag(lj).max(bounded_mag(uj));
    let wmax = imax * jmax;
    let row = |ci: f64, cj: f64, cw: f64, rhs: f64| EnvRow {
        cols: [i, j, w],
        coeffs: [ci, cj, cw],
        nnz: 3,
        rhs: rhs
            + outward_slack(
                bounded_mag(ci) * imax
                    + bounded_mag(cj) * jmax
                    + bounded_mag(cw) * wmax
                    + bounded_mag(rhs),
            ),
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
    let slope = if ui <= li { dfl } else { (fu - fl) / (ui - li) };
    let convex = (p % 2 == 0) || (li >= 0.0);
    // #956: the tangent-at-`ui` rhs is `f'(ui)*ui - f(ui)`, a cancelling subtraction
    // of two same-signed quantities; at `x = ui` it disagrees with `ui^p` by ~1 ulp
    // of `f'(ui)*ui`. Guard by the magnitudes that form the row.
    let xmax = bounded_mag(li).max(bounded_mag(ui));
    let fmax = bounded_mag(fl).max(bounded_mag(fm)).max(bounded_mag(fu));
    let row = |cx: f64, cs: f64, rhs: f64| EnvRow {
        cols: [i, s, 0],
        coeffs: [cx, cs, 0.0],
        nnz: 2,
        rhs: rhs
            + outward_slack(bounded_mag(cx) * xmax + bounded_mag(cs) * fmax + bounded_mag(rhs)),
    };
    if convex {
        [
            row(dfl, -1.0, dfl * li - fl), // tangent at li:  s >= f'(li)(x-li)+f(li)
            row(dfm, -1.0, dfm * mid - fm), // tangent at midpoint
            row(dfu, -1.0, dfu * ui - fu), // tangent at ui
            row(-slope, 1.0, fl - slope * li), // secant (overestimator): s <= ...
        ]
    } else {
        [
            row(-dfl, 1.0, fl - dfl * li), // tangent at li (overestimator): s <= ...
            row(-dfm, 1.0, fm - dfm * mid), // tangent at midpoint
            row(-dfu, 1.0, fu - dfu * ui), // tangent at ui
            row(slope, -1.0, slope * li - fl), // secant (underestimator): s >= ...
        ]
    }
}

/// The 4 envelope rows for `w = (coeff*x + const)^2` over `x in [li,ui]` (secant +
/// tangents at `t_lo`, the midpoint, `t_hi`, where `t = coeff*x + const`). `t^2` is
/// convex for every `t` (no sign gating). Mirrors `_affine_square_rows`; each row is
/// `(coeff_on_x, coeff_on_w, rhs)`, columns `(j, w)`.
#[inline]
pub fn affine_square_rows(
    j: usize,
    w: usize,
    coeff: f64,
    cst: f64,
    li: f64,
    ui: f64,
) -> [EnvRow; 4] {
    let tl = coeff * li + cst;
    let tu = coeff * ui + cst;
    let (t_lo, t_hi) = if tl <= tu { (tl, tu) } else { (tu, tl) };
    let mid = 0.5 * (t_lo + t_hi);
    // t_hi + t_lo == (t_hi^2 - t_lo^2)/(t_hi - t_lo); at a degenerate base box it
    // already equals 2*t_lo == f'(t_lo), so no divide-by-zero guard is needed.
    let slope = t_hi + t_lo;
    let a = t_lo * t_lo - slope * t_lo;
    // #956: `t` is itself a rounded affine image of `x`, so the squares that form
    // these rhs values carry the base box's magnitude twice over. Guard by them.
    let xmax = bounded_mag(li).max(bounded_mag(ui));
    let wmax = bounded_mag(t_lo * t_lo).max(bounded_mag(t_hi * t_hi));
    let row = |cx: f64, cw: f64, rhs: f64| EnvRow {
        cols: [j, w, 0],
        coeffs: [cx, cw, 0.0],
        nnz: 2,
        rhs: rhs
            + outward_slack(bounded_mag(cx) * xmax + bounded_mag(cw) * wmax + bounded_mag(rhs)),
    };
    [
        row(-slope * coeff, 1.0, a + slope * cst), // secant (overestimator)
        row(2.0 * t_lo * coeff, -1.0, t_lo * t_lo - 2.0 * t_lo * cst), // tangent at t_lo
        row(2.0 * mid * coeff, -1.0, mid * mid - 2.0 * mid * cst), // tangent at midpoint
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
    // #956: same outward guard as the other generators — `f` is monotone on the box
    // (all covered atoms are), so `|w| <= max(|f(t_lo)|, |f(t_hi)|)`.
    let xmax = bounded_mag(x_lo).max(bounded_mag(x_hi));
    let wmax = bounded_mag(flo).max(bounded_mag(fhi));
    let row = |cx: f64, cw: f64, rhs: f64| EnvRow {
        cols: [x, w, 0],
        coeffs: [cx, cw, 0.0],
        nnz: 2,
        rhs: rhs
            + outward_slack(bounded_mag(cx) * xmax + bounded_mag(cw) * wmax + bounded_mag(rhs)),
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
    let (lo, hi) = (t_lo.sqrt(), t_hi.sqrt());
    Some(widen(lo, hi, bounded_mag(lo).max(bounded_mag(hi))))
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
    // #956: widened outward so the corner product can never fall outside the
    // auxiliary column's own bounds by a rounding — the disagreement between a row
    // and this bound is precisely what made the node LP unsolvable.
    widen(lo, hi, bounded_mag(lo).max(bounded_mag(hi)))
}

/// Auxiliary-variable bounds for `s = x_i^p` over a sign-definite `[li,ui]` (monotone
/// there). Mirrors `_monomial_aux_bounds`.
#[inline]
pub fn monomial_aux_bounds(li: f64, ui: f64, p: i32) -> (f64, f64) {
    let a = li.powi(p);
    let b = ui.powi(p);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    // #956: widened outward (see `bilinear_aux_bounds`).
    widen(lo, hi, bounded_mag(lo).max(bounded_mag(hi)))
}

/// Auxiliary-variable bounds for the squared base `t^2` over `t in [t_lo,t_hi]`
/// (0 if the base straddles zero). Mirrors `_square_aux_bounds`.
#[inline]
pub fn square_aux_bounds(t_lo: f64, t_hi: f64) -> (f64, f64) {
    let (lo, hi) = if t_lo >= 0.0 {
        (t_lo * t_lo, t_hi * t_hi)
    } else if t_hi <= 0.0 {
        (t_hi * t_hi, t_lo * t_lo)
    } else {
        (0.0, (t_lo * t_lo).max(t_hi * t_hi))
    };
    // #956: widened outward (see `bilinear_aux_bounds`). `t` reaching an endpoint is
    // itself a rounded affine image of `x`, so even the monotone side can miss.
    widen(lo, hi, bounded_mag(lo).max(bounded_mag(hi)))
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
    // #956: magnitude of the `w` column over this box — the interval product's
    // widest corner. The a/b columns are guarded by their own box bounds below.
    let wmax = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
        .iter()
        .fold(0.0f64, |m, &v| m.max(bounded_mag(v)));
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
        // #956: guard by this row's own term magnitudes over the box. `A`/`B` are
        // sums, so their evaluation can cancel; scaling by each column's own bound
        // (rather than by the form's interval) covers that.
        let mut mag_sum = bounded_mag(rhs);
        for (k, &c) in coeffs.iter().enumerate() {
            let col_mag = if cols[k] == w {
                wmax
            } else {
                bounded_mag(box_lo[cols[k]]).max(bounded_mag(box_hi[cols[k]]))
            };
            mag_sum += bounded_mag(c) * col_mag;
        }
        out.push((cols, coeffs, rhs + outward_slack(mag_sum)));
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
        // #956: widened outward (see `bilinear_aux_bounds`).
        widen(lo, hi, bounded_mag(lo).max(bounded_mag(hi)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sat(row: &EnvRow, x: &[f64], tol: f64) -> bool {
        row.residual(x) <= tol
    }

    /// Assert an auxiliary bound pair reproduces the Python reference, allowing the
    /// #956 outward guard: the widened box must CONTAIN the reference and differ
    /// from it by at most the ulp-scaled guard, on the outward side only.
    fn assert_aux_widened(got: (f64, f64), expect: (f64, f64)) {
        let g = outward_slack(expect.0.abs().max(expect.1.abs()));
        assert!(
            got.0 <= expect.0 && got.0 >= expect.0 - g - 1e-15,
            "aux lo {} not an outward rounding of {} (guard {g:e})",
            got.0,
            expect.0
        );
        assert!(
            got.1 >= expect.1 && got.1 <= expect.1 + g + 1e-15,
            "aux hi {} not an outward rounding of {} (guard {g:e})",
            got.1,
            expect.1
        );
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
        // Coefficients are still exact; the rhs carries the #956 outward guard, so it
        // must reproduce the reference from the OUTWARD side only, by at most an
        // ulp-scaled amount (`1e-13` relative dominates the guard on these fixtures).
        let tol = 1e-12 + 1e-13 * expect_rhs.abs();
        assert!(
            r.rhs >= expect_rhs - 1e-12,
            "rhs {} moved INWARD of the reference {}",
            r.rhs,
            expect_rhs
        );
        assert!(
            r.rhs <= expect_rhs + tol,
            "rhs {} != {} (beyond the outward guard {tol:e})",
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
        assert_aux_widened(bilinear_aux_bounds(-1.0, 3.0, 2.0, 5.0), (-5.0, 15.0));
    }

    #[test]
    fn monomial_p2_matches_python_reference() {
        // _monomial_rows(-2,3,2)  (the plain square, 4-row midpoint-tangent form)
        let rows = monomial_rows(0, 1, -2.0, 3.0, 2);
        assert_row_eq(&rows[0], &[-4.0, -1.0], 4.0);
        assert_row_eq(&rows[1], &[1.0, -1.0], 0.25);
        assert_row_eq(&rows[2], &[6.0, -1.0], 9.0);
        assert_row_eq(&rows[3], &[-1.0, 1.0], 6.0);
        assert_aux_widened(monomial_aux_bounds(-2.0, 3.0, 2), (4.0, 9.0));
    }

    #[test]
    fn monomial_p3_matches_python_reference() {
        // _monomial_rows(1,4,3) — convex (li>=0)
        let rows = monomial_rows(0, 1, 1.0, 4.0, 3);
        assert_row_eq(&rows[0], &[3.0, -1.0], 2.0);
        assert_row_eq(&rows[1], &[18.75, -1.0], 31.25);
        assert_row_eq(&rows[2], &[48.0, -1.0], 128.0);
        assert_row_eq(&rows[3], &[-21.0, 1.0], -20.0);
        assert_aux_widened(monomial_aux_bounds(1.0, 4.0, 3), (1.0, 64.0));
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
        assert_aux_widened(affine_square_aux_bounds(2.0, -1.0, -2.0, 3.0), (0.0, 25.0));
    }

    #[test]
    fn sqrt_bare_matches_emit_1d_reference() {
        // _emit_1d(sqrt, t=x, [1,4], concave): bare sqrt(x), col x=0, col w=1.
        let rows = univariate_rows(0, 1, 1.0, 0.0, 1.0, 4.0, Univariate::Sqrt).unwrap();
        assert_row_eq(&rows[0], &[0.333333333333, -1.0], -0.666666666667); // secant
        assert_row_eq(&rows[1], &[-0.5, 1.0], 0.5); // tangent @ t_lo=1
        assert_row_eq(&rows[2], &[-0.316227766017, 1.0], 0.790569415042); // tangent @ mid=2.5
        assert_row_eq(&rows[3], &[-0.25, 1.0], 1.0); // tangent @ t_hi=4
        assert_aux_widened(sqrt_aux_bounds(1.0, 0.0, 1.0, 4.0).unwrap(), (1.0, 2.0));
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
                assert!(
                    sat(r, &x, 1e-9),
                    "sqrt true point {xi} cut: residual {}",
                    r.residual(&x)
                );
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
                assert!(
                    sat(r, &x, 1e-9),
                    "corner ({xi},{xj}) residual {}",
                    r.residual(&x)
                );
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
                    assert!(
                        sat(r, &x, 1e-9),
                        "true product cut: residual {}",
                        r.residual(&x)
                    );
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
        bilinear_linform_rows(
            &[0],
            &[1.0],
            0.0,
            &[1, 2],
            &[1.7, 0.4],
            0.5,
            9,
            &lo,
            &hi,
            &mut out,
        );
        assert_eq!(out.len(), 4);
        assert_wide_row(&out[0], &[(0, 3.5), (1, 1.7), (2, 0.4), (9, -1.0)], 3.0);
        assert_wide_row(&out[1], &[(0, 10.6), (1, 5.1), (2, 1.2), (9, -1.0)], 30.3);
        assert_wide_row(&out[2], &[(0, -3.5), (1, -5.1), (2, -1.2), (9, 1.0)], -9.0);
        assert_wide_row(
            &out[3],
            &[(0, -10.6), (1, -1.7), (2, -0.4), (9, 1.0)],
            -10.1,
        );
        // aux bounds = interval(A)*interval(B) = [1,3]*[3.5,10.6] = [3.5, 31.8].
        let (alo, ahi) =
            bilinear_linform_aux_bounds(&[0], &[1.0], 0.0, &[1, 2], &[1.7, 0.4], 0.5, &lo, &hi);
        assert!((alo - 3.5).abs() < 1e-9 && (ahi - 31.8).abs() < 1e-9);
    }

    /// Shared-column merge: `w = x0 * (x0 + x1)` — the two x0 contributions fold into
    /// one coefficient. Matches `_emit_mccormick`.
    #[test]
    fn bilinear_linform_merges_shared_columns() {
        let lo = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hi = vec![2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = Vec::new();
        bilinear_linform_rows(
            &[0],
            &[1.0],
            0.0,
            &[0, 1],
            &[1.0, 1.0],
            0.0,
            5,
            &lo,
            &hi,
            &mut out,
        );
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
        bilinear_linform_rows(
            &[0],
            &[1.0],
            0.0,
            &[1],
            &[1.0],
            0.0,
            2,
            &lo,
            &hi,
            &mut general,
        );
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
            assert!(
                gset.contains(&key),
                "special row {key:?} not in general set"
            );
        }
    }

    // --- #956: outward rounding. The invariant is that a point ON the graph of the
    //     relaxed function is in the relaxation. The witnesses below are boxes found
    //     by a magnitude sweep (`probe956_witness.py`) where the UNGUARDED
    //     generators cut that point by far more than any LP feasibility tolerance;
    //     each test names the residual it used to produce.
    //
    //     These tests assert the shipped default. Running the suite with
    //     `DISCOPT_ENVELOPE_OUTWARD_ROUND=0` selects the legacy unguarded generators
    //     and they FAIL — that opt-out is the before/after demonstration, not a
    //     supported configuration to be green in.

    /// A point on the graph of the relaxed function is never cut, at magnitudes
    /// where the unguarded rows cut it by 1.2e-4 (square) / 6.4e1 (cubic).
    #[test]
    fn monomial_rows_never_cut_the_true_point_at_large_magnitude() {
        let cases = [
            (66784.5398643343f64, 770432.1452594515f64, 2i32),
            (766768.5864777878, 825273.9822479703, 3),
            (0.0, 2950.7, 3),
        ];
        let mut checked = 0usize;
        for &(li, ui, p) in &cases {
            let rows = monomial_rows(0, 1, li, ui, p);
            let (alo, ahi) = monomial_aux_bounds(li, ui, p);
            let n = 64;
            for k in 0..=n {
                let x = li + (ui - li) * (k as f64) / (n as f64);
                let s = x.powi(p);
                for r in &rows {
                    let res = r.residual(&[x, s]);
                    assert!(
                        res <= 0.0,
                        "p={p} box [{li},{ui}]: true point x={x} cut by {res:e}"
                    );
                    checked += 1;
                }
                assert!(
                    s >= alo && s <= ahi,
                    "p={p} box [{li},{ui}]: f({x}) = {s} outside aux bounds [{alo},{ahi}]"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            cases.len() * 65 * 5,
            "probe did not fire as expected"
        );
    }

    /// Same invariant for the bilinear hull (unguarded residual 9.5e-7 at a corner of
    /// this box) — checked at the corners, where the rows are tight, and on a grid.
    #[test]
    fn bilinear_rows_never_cut_the_true_product_at_large_magnitude() {
        let (li, ui) = (-72367.88703192568f64, 89055.82553869025f64);
        let (lj, uj) = (-72619.82005656551f64, 98196.62997065719f64);
        let rows = bilinear_rows(0, 1, 2, li, ui, lj, uj);
        let (alo, ahi) = bilinear_aux_bounds(li, ui, lj, uj);
        let mut checked = 0usize;
        let n = 24;
        for a in 0..=n {
            for b in 0..=n {
                let xi = li + (ui - li) * (a as f64) / (n as f64);
                let xj = lj + (uj - lj) * (b as f64) / (n as f64);
                let w = xi * xj;
                for r in &rows {
                    let res = r.residual(&[xi, xj, w]);
                    assert!(res <= 0.0, "true product ({xi},{xj}) cut by {res:e}");
                    checked += 1;
                }
                assert!(
                    w >= alo && w <= ahi,
                    "product {w} outside aux [{alo},{ahi}]"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 25 * 25 * 5, "probe did not fire as expected");
    }

    /// Same invariant for the affine-square hull (unguarded residual 9.0e-5 here) and
    /// for the concave `sqrt` hull (unguarded 9.1e-13 — the smallest of the families,
    /// since `sqrt` compresses its argument's magnitude rather than expanding it).
    #[test]
    fn affine_square_and_sqrt_rows_never_cut_the_true_point_at_large_magnitude() {
        let mut checked = 0usize;

        let (coeff, cst) = (-78584.27778155846f64, 79795.31365347138f64);
        let (li, ui) = (-0.37087115568115436f64, 16.77735521273895f64);
        let rows = affine_square_rows(0, 1, coeff, cst, li, ui);
        let (alo, ahi) = affine_square_aux_bounds(coeff, cst, li, ui);
        let n = 64;
        for k in 0..=n {
            let x = li + (ui - li) * (k as f64) / (n as f64);
            let t = coeff * x + cst;
            let w = t * t;
            for r in &rows {
                let res = r.residual(&[x, w]);
                assert!(res <= 0.0, "affine-square true point x={x} cut by {res:e}");
                checked += 1;
            }
            assert!(w >= alo && w <= ahi, "square {w} outside aux [{alo},{ahi}]");
            checked += 1;
        }

        let (coeff, cst) = (720802.2240007615f64, 359958.2913327332f64);
        let (li, ui) = (9.070106771272291f64, 23.092166581992302f64);
        let rows = univariate_rows(0, 1, coeff, cst, li, ui, Univariate::Sqrt).unwrap();
        let (alo, ahi) = sqrt_aux_bounds(coeff, cst, li, ui).unwrap();
        for k in 0..=n {
            let x = li + (ui - li) * (k as f64) / (n as f64);
            let w = (coeff * x + cst).sqrt();
            for r in &rows {
                let res = r.residual(&[x, w]);
                assert!(res <= 0.0, "sqrt true point x={x} cut by {res:e}");
                checked += 1;
            }
            assert!(w >= alo && w <= ahi, "sqrt {w} outside aux [{alo},{ahi}]");
            checked += 1;
        }

        assert_eq!(checked, 2 * 65 * 5, "probe did not fire as expected");
    }

    /// The affine-form product path holds the same invariant: `w = A*B` evaluated at
    /// a box corner satisfies every emitted row and sits inside the aux interval
    /// (unguarded residual 2.0e-3 on this box).
    #[test]
    fn bilinear_linform_rows_never_cut_the_true_product_at_large_magnitude() {
        // A = 1.7 x0 - 930.5, B = 4.3e4 x1 + 2.9e4 x2 - 1.1e5, over a large box.
        let lo = vec![-53219.7031f64, -8123.1177, 4471.3399];
        let hi = vec![91733.5297f64, 66021.8813, 77913.2231];
        let (a_cols, a_coeffs, a_const) = (vec![0usize], vec![1.7f64], -930.5f64);
        let (b_cols, b_coeffs, b_const) = (vec![1usize, 2], vec![4.3e4f64, 2.9e4], -1.1e5f64);
        // Column 3 is `w`; give it the sentinel box the kernel starts it with.
        let mut box_lo = lo.clone();
        let mut box_hi = hi.clone();
        box_lo.push(-1e20);
        box_hi.push(1e20);
        let mut out = Vec::new();
        bilinear_linform_rows(
            &a_cols, &a_coeffs, a_const, &b_cols, &b_coeffs, b_const, 3, &box_lo, &box_hi, &mut out,
        );
        let (alo, ahi) = bilinear_linform_aux_bounds(
            &a_cols, &a_coeffs, a_const, &b_cols, &b_coeffs, b_const, &box_lo, &box_hi,
        );
        let mut checked = 0usize;
        let n = 4;
        for i0 in 0..=n {
            for i1 in 0..=n {
                for i2 in 0..=n {
                    let x0 = lo[0] + (hi[0] - lo[0]) * (i0 as f64) / (n as f64);
                    let x1 = lo[1] + (hi[1] - lo[1]) * (i1 as f64) / (n as f64);
                    let x2 = lo[2] + (hi[2] - lo[2]) * (i2 as f64) / (n as f64);
                    let av = a_const + 1.7 * x0;
                    let bv = b_const + 4.3e4 * x1 + 2.9e4 * x2;
                    let w = av * bv;
                    let pt = [x0, x1, x2, w];
                    for (cols, coeffs, rhs) in &out {
                        let mut act = -rhs;
                        for (k, &c) in coeffs.iter().enumerate() {
                            act += c * pt[cols[k]];
                        }
                        assert!(act <= 0.0, "A*B true point cut by {act:e}");
                        checked += 1;
                    }
                    assert!(w >= alo && w <= ahi, "A*B {w} outside aux [{alo},{ahi}]");
                    checked += 1;
                }
            }
        }
        assert_eq!(out.len(), 4);
        assert_eq!(checked, 125 * 5, "probe did not fire as expected");
    }

    /// The guard is a bounded OUTWARD relaxation, never an inward one: each emitted
    /// rhs is `>=` the exact closed form and exceeds it by at most the ulp-scaled
    /// guard. This is the differential bound test at the formula level — the
    /// relaxation can only get looser, and only by `~1e-14` relative.
    #[test]
    fn guard_is_outward_and_bounded() {
        let (li, ui, p) = (766768.5864777878f64, 825273.9822479703f64, 3i32);
        let (fl, fu) = (li.powi(p), ui.powi(p));
        let mid = 0.5 * (li + ui);
        let (fm, pf) = (mid.powi(p), p as f64);
        let (dfl, dfm, dfu) = (
            pf * li.powi(p - 1),
            pf * mid.powi(p - 1),
            pf * ui.powi(p - 1),
        );
        let slope = (fu - fl) / (ui - li);
        // The exact (unguarded) closed forms, in the same operation order.
        let exact = [
            dfl * li - fl,
            dfm * mid - fm,
            dfu * ui - fu,
            fl - slope * li,
        ];
        let rows = monomial_rows(0, 1, li, ui, p);
        let xmax = ui.abs();
        let fmax = fu.abs();
        let mut checked = 0usize;
        for (r, &e) in rows.iter().zip(exact.iter()) {
            let delta = r.rhs - e;
            // `rhs + guard` is itself rounded, so the realized delta can differ from
            // the computed guard by up to an ulp of `rhs` (here 256).
            let allowed =
                outward_slack(r.coeffs[0].abs() * xmax + fmax + e.abs()) + f64::EPSILON * e.abs();
            assert!(
                delta >= 0.0,
                "rhs moved INWARD by {delta:e} — cuts the hull"
            );
            assert!(
                delta <= allowed,
                "rhs relaxed by {delta:e}, more than the guard {allowed:e}"
            );
            // Relative cost of the guard on a 1e18-magnitude row stays at ulp scale.
            assert!(
                delta <= 1e-13 * e.abs(),
                "guard is not ulp-scaled: {delta:e}"
            );
            checked += 1;
        }
        assert_eq!(checked, 4, "probe did not fire as expected");
    }

    #[test]
    fn aux_bounds_match_corners() {
        assert_aux_widened(bilinear_aux_bounds(-1.0, 3.0, 2.0, 5.0), (-5.0, 15.0));
        assert_aux_widened(bilinear_aux_bounds(-2.0, 4.0, -3.0, 1.5), (-12.0, 6.0));
        assert_aux_widened(square_aux_bounds(-2.0, 3.0), (0.0, 9.0));
        assert_aux_widened(square_aux_bounds(1.0, 4.0), (1.0, 16.0));
        assert_aux_widened(square_aux_bounds(-4.0, -1.0), (1.0, 16.0));
    }
}
