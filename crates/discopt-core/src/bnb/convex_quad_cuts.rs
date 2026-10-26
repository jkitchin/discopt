//! Outer-approximation cuts for convex quadratic forms over lifted product columns
//! (issue #1962, family A).
//!
//! The native spatial kernel relaxes every product separately: a row
//! `Σ a_k w_k + … <= rhs` whose lifted columns are `w_k = x_i x_j` (or `x_i^2`, or an
//! affine product) sees each `w_k` only through its own McCormick envelope, so a
//! *convex* quadratic form `F(w) = Σ a_k w_k = xᵀQx + lᵀx + c₀` with `Q ⪰ 0` loses its
//! convexity. Every point that is feasible for the original problem has
//! `w_k = q_k(x)` exactly, and for `Q ⪰ 0`
//!
//! ```text
//! F(w) = xᵀQx + lᵀx + c₀ >= 2x̂ᵀQx − x̂ᵀQx̂ + lᵀx + c₀        for every x̂,
//! ```
//!
//! because the difference is `(x − x̂)ᵀQ(x − x̂) >= 0`. Written as a `<=` row over
//! the LP columns this is
//!
//! ```text
//! −Σ a_k w_k + (2Qx̂ + l)ᵀx <= x̂ᵀQx̂ − c₀,
//! ```
//!
//! a cut that is *globally* valid (it depends on no box), so a node may hand it to
//! its children. At the LP point `(x̂, ŵ)` it is violated by exactly
//! `q(x̂) − F(ŵ)`, i.e. by how far the LP has pushed the form below its true value.
//!
//! Where the form comes from does not matter for validity, only for usefulness: a
//! form helps when the LP pushes it DOWN, which is the objective (minimised) and the
//! left side of a `<=` row. Both are scanned; an `==` row reaches the kernel as two
//! `<=` rows, one of which carries `+Q`.
//!
//! Soundness (CLAUDE.md §1) rests on three things, each handled here explicitly:
//!
//! 1. **`Q ⪰ 0` is proven, not estimated.** [`certify_pd`] runs a floating-point
//!    Cholesky on `Q − cI` with `c` at least Rump's bound on the rounding error of
//!    the factorisation ("Verification of positive definiteness", BIT 46, 2006), so a
//!    success proves `Q` positive definite. A form that is only semidefinite
//!    (`(x − y)²`) is certified as `Q + τI ≻ 0` instead, which proves
//!    `λ_min(Q) > −τ`; the cut then subtracts `τ‖x − x̂‖²`, bounded over the global
//!    box, from its right side.
//! 2. **Rounding goes outward.** `Q`, `l`, `c₀`, the gradient and the right side are
//!    computed in floating point; [`ConvexQuadForm::separate`] adds a slack that
//!    bounds every one of those errors over the global box (`γ_n · S` with `S` the
//!    form's absolute magnitude there), so the emitted row is implied by the exact
//!    tangent.
//! 3. **Only finite boxes.** Both the rounding slack and the semidefinite correction
//!    need a finite bound on every column the form reads; a form that touches an
//!    unbounded column is not registered.

use crate::bnb::spatial_kernel::{EnvTerm, FixedRow, SpatialKernelSpec};
use std::collections::HashMap;

/// Largest form (distinct variables) registered. Certification is a dense
/// `O(d³)` Cholesky and each separation `O(d²)`; a dense QP row over thousands of
/// variables (QPLIB) would spend the budget here, so it is left to the envelopes.
const MAX_FORM_DIM: usize = 300;

/// `f64` unit roundoff.
const U: f64 = f64::EPSILON * 0.5;
/// The kernel's infinite-bound sentinel (`INF` in the LP layer is `1e20`).
const INF_SENTINEL: f64 = 1e20;

/// The exact quadratic a lifted column equals at every feasible point:
/// `w = (a₀ + aᵀx)(b₀ + bᵀx)`. Every quadratic family the kernel lifts is this
/// shape (a square is `a = b`).
#[derive(Clone, Debug)]
struct QuadDef {
    a: Vec<(usize, f64)>,
    a0: f64,
    b: Vec<(usize, f64)>,
    b0: f64,
}

/// The quadratic definition of every lifted column the spec carries, keyed by the
/// aux column. Monomials of degree other than 2 and `sqrt` are not quadratic and
/// are absent.
fn quad_defs(spec: &SpatialKernelSpec) -> HashMap<usize, QuadDef> {
    let mut defs = HashMap::new();
    for t in &spec.terms {
        let d = match *t {
            EnvTerm::Bilinear { i, j, w } => (
                w,
                QuadDef {
                    a: vec![(i, 1.0)],
                    a0: 0.0,
                    b: vec![(j, 1.0)],
                    b0: 0.0,
                },
            ),
            EnvTerm::Monomial { i, s, p: 2 } => (
                s,
                QuadDef {
                    a: vec![(i, 1.0)],
                    a0: 0.0,
                    b: vec![(i, 1.0)],
                    b0: 0.0,
                },
            ),
            EnvTerm::AffineSquare { j, w, coeff, cst } => (
                w,
                QuadDef {
                    a: vec![(j, coeff)],
                    a0: cst,
                    b: vec![(j, coeff)],
                    b0: cst,
                },
            ),
            _ => continue,
        };
        defs.insert(d.0, d.1);
    }
    for t in &spec.blf_terms {
        defs.insert(
            t.w,
            QuadDef {
                a: t.a_cols
                    .iter()
                    .copied()
                    .zip(t.a_coeffs.iter().copied())
                    .collect(),
                a0: t.a_const,
                b: t.b_cols
                    .iter()
                    .copied()
                    .zip(t.b_coeffs.iter().copied())
                    .collect(),
                b0: t.b_const,
            },
        );
    }
    defs
}

/// A certified convex quadratic form over lifted columns (see the module docs).
#[derive(Clone, Debug)]
pub struct ConvexQuadForm {
    /// The lifted columns `w_k` and their coefficients `a_k`: `F(w) = Σ a_k w_k`.
    pub aux: Vec<(usize, f64)>,
    /// The LP columns `Q` and `l` are indexed over (local index -> column).
    pub vars: Vec<usize>,
    /// Dense row-major `Q` over `vars` (`d × d`, symmetric).
    q: Vec<f64>,
    /// Linear part `l` over `vars`.
    lin: Vec<f64>,
    /// Constant `c₀`.
    c0: f64,
    /// Global bounds of `vars`, the box every feasible point lies in.
    vlo: Vec<f64>,
    vhi: Vec<f64>,
    /// `τ` with `λ_min(Q) > −τ` proven (0 when `Q` is proven positive definite).
    tau: f64,
    /// Absolute magnitude of the form over the global box, `Σ |a_k| max|q_k|`: the
    /// scale every rounding error below is measured against.
    mag: f64,
    /// Number of floating-point operations any one computed quantity accumulates,
    /// for the `γ_n` rounding bound.
    n_ops: usize,
}

/// Rump's bound on the error of a floating-point Cholesky of a `d × d` symmetric
/// matrix with trace `tr`: a factorisation of `A − cI` that succeeds with
/// `c >= rump_shift` proves `A` positive definite. Doubled for margin (the shifted
/// diagonal is itself rounded).
fn rump_shift(d: usize, tr: f64) -> f64 {
    let n1 = (d + 1) as f64;
    let gamma = n1 * U / (1.0 - n1 * U);
    2.0 * (gamma / (1.0 - gamma) * tr + 4.0 * d as f64 * (2.0 * n1 + tr) * f64::MIN_POSITIVE)
}

/// Whether `a + shift·I` (dense row-major, `d × d`) is PROVEN positive definite:
/// the floating-point Cholesky of `a + (shift − c)I` succeeds with `c` from
/// [`rump_shift`].
fn certify_pd(a: &[f64], d: usize, shift: f64) -> bool {
    let tr: f64 = (0..d).map(|i| (a[i * d + i] + shift).abs()).sum();
    if !tr.is_finite() {
        return false;
    }
    let c = rump_shift(d, tr);
    let mut l = vec![0.0f64; d * d];
    for j in 0..d {
        let mut s = a[j * d + j] + shift - c;
        for k in 0..j {
            s -= l[j * d + k] * l[j * d + k];
        }
        if !(s > 0.0) || !s.is_finite() {
            return false;
        }
        let ljj = s.sqrt();
        l[j * d + j] = ljj;
        for i in (j + 1)..d {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            l[i * d + j] = s / ljj;
        }
    }
    true
}

impl ConvexQuadForm {
    /// Build and certify the form `Σ coeffs[k] · w_{cols[k]}` restricted to the
    /// quadratic lifted columns among `cols`. `None` when the restriction is not a
    /// form worth separating (fewer than two variables), reads an unbounded column,
    /// or is not certified convex.
    fn build(
        spec: &SpatialKernelSpec,
        defs: &HashMap<usize, QuadDef>,
        cols: &[usize],
        coeffs: &[f64],
    ) -> Option<Self> {
        let mut aux: Vec<(usize, f64)> = Vec::new();
        for (&c, &v) in cols.iter().zip(coeffs.iter()) {
            if v != 0.0 && defs.contains_key(&c) {
                aux.push((c, v));
            }
        }
        if aux.is_empty() {
            return None;
        }
        aux.sort_by_key(|&(c, _)| c);
        // Variable index map over every operand column.
        let mut vars: Vec<usize> = Vec::new();
        for &(c, _) in &aux {
            let d = &defs[&c];
            for &(j, _) in d.a.iter().chain(d.b.iter()) {
                vars.push(j);
            }
        }
        vars.sort_unstable();
        vars.dedup();
        let d = vars.len();
        if !(2..=MAX_FORM_DIM).contains(&d) {
            return None;
        }
        let pos: HashMap<usize, usize> = vars.iter().enumerate().map(|(k, &j)| (j, k)).collect();
        let vlo: Vec<f64> = vars.iter().map(|&j| spec.global_lo[j]).collect();
        let vhi: Vec<f64> = vars.iter().map(|&j| spec.global_hi[j]).collect();
        let bounded = |v: f64| v.is_finite() && v.abs() < INF_SENTINEL;
        if !vlo.iter().chain(vhi.iter()).all(|&v| bounded(v)) {
            return None;
        }
        let mmax: Vec<f64> = vlo
            .iter()
            .zip(vhi.iter())
            .map(|(l, h)| l.abs().max(h.abs()))
            .collect();

        let mut q = vec![0.0f64; d * d];
        let mut lin = vec![0.0f64; d];
        let mut c0 = 0.0f64;
        let mut mag = 0.0f64;
        let mut n_terms = 0usize;
        for &(c, kappa) in &aux {
            let def = &defs[&c];
            let mag_a = def.a0.abs()
                + def
                    .a
                    .iter()
                    .map(|&(j, v)| v.abs() * mmax[pos[&j]])
                    .sum::<f64>();
            let mag_b = def.b0.abs()
                + def
                    .b
                    .iter()
                    .map(|&(j, v)| v.abs() * mmax[pos[&j]])
                    .sum::<f64>();
            mag += kappa.abs() * mag_a * mag_b;
            // Q += κ/2 (a bᵀ + b aᵀ); l += κ (a₀ b + b₀ a); c₀ += κ a₀ b₀.
            for &(i, ai) in &def.a {
                for &(j, bj) in &def.b {
                    let (pi, pj) = (pos[&i], pos[&j]);
                    let h = 0.5 * kappa * ai * bj;
                    q[pi * d + pj] += h;
                    q[pj * d + pi] += h;
                    n_terms += 1;
                }
            }
            for &(j, bj) in &def.b {
                lin[pos[&j]] += kappa * def.a0 * bj;
            }
            for &(i, ai) in &def.a {
                lin[pos[&i]] += kappa * def.b0 * ai;
            }
            c0 += kappa * def.a0 * def.b0;
        }
        if !mag.is_finite() || !q.iter().all(|v| v.is_finite()) {
            return None;
        }
        if q.iter().all(|&v| v == 0.0) {
            return None;
        }
        // Positive definite outright, else positive definite after a tiny shift
        // (a proven `λ_min > −τ`), else not convex.
        let max_diag = (0..d).map(|i| q[i * d + i].abs()).fold(0.0f64, f64::max);
        let tau = if certify_pd(&q, d, 0.0) {
            0.0
        } else {
            let tau = 1e-9 * max_diag.max(1e-12);
            if !certify_pd(&q, d, tau) {
                return None;
            }
            // The shifted diagonal `q_ii + τ` was rounded: widen τ by that error.
            tau * (1.0 + 4.0 * U) + 4.0 * U * (max_diag + tau)
        };
        Some(ConvexQuadForm {
            aux,
            vars,
            q,
            lin,
            c0,
            vlo,
            vhi,
            tau,
            mag,
            n_ops: n_terms + d * d + 2 * d + 8,
        })
    }

    /// The `<=` cut at the LP point `x` (all `n_cols` structural columns) and its
    /// violation there, or `None` when the point does not violate it by more than
    /// `viol_tol` (relative to `max(1, |rhs|)`).
    pub fn separate(&self, x: &[f64], viol_tol: f64) -> Option<(FixedRow, f64)> {
        let d = self.vars.len();
        // The tangent point: the LP values, clamped into the global box (any point
        // gives a valid tangent; the clamp keeps the rounding bound below honest).
        let xh: Vec<f64> = (0..d)
            .map(|k| x[self.vars[k]].clamp(self.vlo[k], self.vhi[k]))
            .collect();
        let qx: Vec<f64> = (0..d)
            .map(|i| (0..d).map(|j| self.q[i * d + j] * xh[j]).sum::<f64>())
            .collect();
        let xqx: f64 = (0..d).map(|i| xh[i] * qx[i]).sum();
        // Every computed quantity is within `γ_n · mag` of its exact value over the
        // global box, the left side's evaluation included; 4x for margin.
        let n = self.n_ops as f64;
        let mut slack = 4.0 * (n * U / (1.0 - n * U)) * self.mag.max(1.0);
        if self.tau > 0.0 {
            // `(x − x̂)ᵀQ(x − x̂) >= −τ‖x − x̂‖²`, both points in the global box.
            let diam2: f64 = (0..d).map(|k| (self.vhi[k] - self.vlo[k]).powi(2)).sum();
            slack += self.tau * diam2 * (1.0 + 8.0 * U);
        }
        let rhs = (xqx - self.c0) + slack;
        if !rhs.is_finite() {
            return None;
        }
        let mut row: HashMap<usize, f64> = HashMap::with_capacity(self.aux.len() + d);
        for &(c, a) in &self.aux {
            *row.entry(c).or_insert(0.0) -= a;
        }
        for k in 0..d {
            *row.entry(self.vars[k]).or_insert(0.0) += 2.0 * qx[k] + self.lin[k];
        }
        let mut entries: Vec<(usize, f64)> = row.into_iter().filter(|&(_, v)| v != 0.0).collect();
        entries.sort_by_key(|&(c, _)| c);
        if !entries.iter().all(|&(_, v)| v.is_finite()) {
            return None;
        }
        let lhs: f64 = entries.iter().map(|&(c, v)| v * x[c]).sum();
        let viol = lhs - rhs;
        if !(viol > viol_tol * rhs.abs().max(1.0)) {
            return None;
        }
        let (cols, coeffs) = entries.into_iter().unzip();
        Some((FixedRow { cols, coeffs, rhs }, viol))
    }
}

/// Every certified convex quadratic form in `spec`: the objective's and each fixed
/// row's restriction to its quadratic lifted columns (duplicates dropped).
pub fn detect_convex_quad_forms(spec: &SpatialKernelSpec) -> Vec<ConvexQuadForm> {
    let defs = quad_defs(spec);
    if defs.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<ConvexQuadForm> = Vec::new();
    let push = |f: Option<ConvexQuadForm>, out: &mut Vec<ConvexQuadForm>| {
        if let Some(f) = f {
            if !out.iter().any(|g| g.aux == f.aux) {
                out.push(f);
            }
        }
    };
    let obj_cols: Vec<usize> = (0..spec.n_cols).filter(|&k| spec.c[k] != 0.0).collect();
    let obj_coeffs: Vec<f64> = obj_cols.iter().map(|&k| spec.c[k]).collect();
    push(
        ConvexQuadForm::build(spec, &defs, &obj_cols, &obj_coeffs),
        &mut out,
    );
    for fr in &spec.fixed_rows {
        push(
            ConvexQuadForm::build(spec, &defs, &fr.cols, &fr.coeffs),
            &mut out,
        );
    }
    out
}

/// One separation round: the violated cut of every form at `x`, most violated
/// first, at most `max_cuts`.
pub fn separate_round(
    forms: &[ConvexQuadForm],
    x: &[f64],
    viol_tol: f64,
    max_cuts: usize,
) -> Vec<FixedRow> {
    let mut cuts: Vec<(FixedRow, f64)> = forms
        .iter()
        .filter_map(|f| f.separate(x, viol_tol))
        .collect();
    cuts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cuts.truncate(max_cuts);
    cuts.into_iter().map(|(r, _)| r).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bnb::spatial_kernel::BlfTerm;

    fn blf(i: usize, j: usize, w: usize) -> BlfTerm {
        BlfTerm {
            a_cols: vec![i],
            a_coeffs: vec![1.0],
            a_const: 0.0,
            b_cols: vec![j],
            b_coeffs: vec![1.0],
            b_const: 0.0,
            w,
        }
    }

    /// `min x² + y² + xy` style spec over `[lo, hi]²` with columns
    /// `x=0, y=1, w_xx=2, w_yy=3, w_xy=4`, objective `c` over the aux columns.
    fn spec(c_aux: [f64; 3], lo: f64, hi: f64) -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols: 5,
            n_orig: 2,
            c: vec![0.0, 0.0, c_aux[0], c_aux[1], c_aux[2]],
            integrality: vec![false; 5],
            global_lo: vec![lo, lo, 0.0, 0.0, -hi * hi],
            global_hi: vec![hi, hi, hi * hi, hi * hi, hi * hi],
            fixed_rows: vec![],
            terms: vec![
                EnvTerm::Monomial { i: 0, s: 2, p: 2 },
                EnvTerm::Monomial { i: 1, s: 3, p: 2 },
            ],
            blf_terms: vec![blf(0, 1, 4)],
            obbt_candidates: vec![],
        }
    }

    #[test]
    fn pd_form_is_detected_and_indefinite_is_not() {
        // x² + y² + xy: Q = [[1, .5], [.5, 1]] ≻ 0.
        assert_eq!(
            detect_convex_quad_forms(&spec([1.0, 1.0, 1.0], -2.0, 3.0)).len(),
            1
        );
        // x² + y² + 3xy: eigenvalues 2.5, −0.5.
        assert!(detect_convex_quad_forms(&spec([1.0, 1.0, 3.0], -2.0, 3.0)).is_empty());
        // (x − y)² = x² + y² − 2xy: semidefinite, certified with a shift.
        let f = detect_convex_quad_forms(&spec([1.0, 1.0, -2.0], -2.0, 3.0));
        assert_eq!(f.len(), 1);
        assert!(f[0].tau > 0.0);
        // Unbounded operand: never registered.
        assert!(detect_convex_quad_forms(&spec([1.0, 1.0, 1.0], -2.0, f64::INFINITY)).is_empty());
    }

    /// The cut is violated at an LP point that underestimates the form, and no
    /// feasible point (`w = q(x)`) on a grid over the box violates it.
    #[test]
    fn cut_separates_and_never_cuts_a_feasible_point() {
        let mut executed = 0usize;
        for c_aux in [[1.0, 1.0, 1.0], [1.0, 1.0, -2.0], [2.0, 0.5, 1.9]] {
            let s = spec(c_aux, -2.0, 3.0);
            let forms = detect_convex_quad_forms(&s);
            assert_eq!(forms.len(), 1, "{c_aux:?}");
            for &(px, py) in &[(0.3, -1.1), (2.9, -2.0), (-1.7, 2.2), (0.0, 0.0)] {
                // LP point with the products pushed below their true values.
                let lp = [px, py, px * px - 1.0, py * py - 1.0, px * py];
                let Some((cut, viol)) = forms[0].separate(&lp, 1e-9) else {
                    panic!("no cut at an underestimating point {c_aux:?}");
                };
                assert!(viol > 0.0);
                for a in 0..=50 {
                    for b in 0..=50 {
                        let x = -2.0 + 5.0 * a as f64 / 50.0;
                        let y = -2.0 + 5.0 * b as f64 / 50.0;
                        let z = [x, y, x * x, y * y, x * y];
                        let lhs: f64 = cut
                            .cols
                            .iter()
                            .zip(&cut.coeffs)
                            .map(|(&c, &v)| v * z[c])
                            .sum();
                        assert!(lhs <= cut.rhs, "{c_aux:?} cuts feasible ({x},{y})");
                        executed += 1;
                    }
                }
                // At a feasible point the form is exact: nothing to separate.
                let exact = [px, py, px * px, py * py, px * py];
                assert!(forms[0].separate(&exact, 1e-9).is_none());
            }
        }
        assert!(executed > 0);
    }

    #[test]
    fn certify_pd_rejects_negative_and_singular() {
        assert!(certify_pd(&[2.0, 1.0, 1.0, 2.0], 2, 0.0));
        assert!(!certify_pd(&[1.0, 1.0, 1.0, 1.0], 2, 0.0)); // singular
        assert!(!certify_pd(&[1.0, 2.0, 2.0, 1.0], 2, 0.0)); // indefinite
        assert!(certify_pd(&[1.0, 1.0, 1.0, 1.0], 2, 1e-6)); // shifted
    }
}
