//! Power-of-two geometric-mean equilibration scaling for the simplex.
//!
//! The revised simplex factorizes its basis with [`feral`]'s LU, which declares
//! a basis singular when its pivots are tiny relative to the matrix entries. On
//! lifted McCormick LPs whose coefficients span many orders of magnitude — FBBT
//! pushes product-aux bounds to ~1e9, so the secant-envelope slopes do too — the
//! *raw* constraint matrix is so ill-scaled that the LU breaks down after a few
//! dozen pivots and the solve returns [`LpStatus::Numerical`](super::LpStatus).
//! The LP is benign once rescaled (HiGHS solves the same system in ~0.01 s),
//! and a `Numerical` relaxation bound is uncertified, so a MILP B&B that relies
//! on it can never fathom and degenerates into full enumeration (issue #170).
//!
//! This module equilibrates the matrix so every nonzero is brought near
//! magnitude 1 before the simplex ever factorizes it, matching what HiGHS does.
//! Scaling replaces `A` with `R A C` (`R`, `C` positive diagonal), and `x` with
//! `C x̂`, giving the equivalent LP
//!
//! ```text
//!   min ĉᵀx̂  s.t.  (R A C) x̂ = R b,   C⁻¹l ≤ x̂ ≤ C⁻¹u,   ĉ = C c.
//! ```
//!
//! The objective is invariant (`ĉᵀx̂ = cᵀx`) and the optimal basis is unchanged
//! (scaling permutes nothing — a column is basic or not regardless of its
//! scale), so a warm-start basis stays valid across the B&B tree as long as the
//! factors come from `A` alone — they do; only bounds change between nodes.
//!
//! Factors are snapped to powers of two, so every scaled quantity is an exact
//! float (only the exponent shifts) and unscaling `x_j = c_j x̂_j` is exact.
//! Scaling is applied only when the matrix's dynamic range warrants it
//! ([`SCALE_TRIGGER`]); a well-conditioned LP is left untouched and bit-identical
//! to the unscaled solve, so existing behavior is preserved exactly.
//
// The equilibration sweeps index the row-major matrix and the row/col factor
// vectors by the same index, so range loops read clearer than zipping slices.
#![allow(clippy::needless_range_loop)]

use crate::lp::crossover::LpView;

const INF: f64 = 1e20;

/// Equilibrate only when the matrix's dynamic range (largest over smallest
/// nonzero magnitude) exceeds this. The fast simplex is reliable on raw matrices
/// up to a range of ~1e7; triggering at 1e6 leaves a margin while keeping every
/// well-conditioned LP an exact no-op.
const SCALE_TRIGGER: f64 = 1e6;

/// Alternating column/row sweeps. Power-of-two geometric-mean equilibration
/// converges within a few passes; the loop also stops early once no factor
/// changes (and guards against the occasional pow2 limit-cycle that never
/// fully settles).
const MAX_PASSES: usize = 4;

/// Round `v > 0` to the nearest power of two (an exact float scale factor).
#[inline]
fn nearest_pow2(v: f64) -> f64 {
    if !v.is_finite() || v <= 0.0 {
        return 1.0;
    }
    2f64.powi(v.log2().round() as i32)
}

/// Diagonal row/column equilibration factors for a fixed constraint matrix `A`,
/// derived from `A` alone. Because the factors depend only on the matrix — not on
/// the rhs or bounds — one `Scaling` can be reused to scale a whole family of LPs
/// that share `A` (the B&B tree, a multi-rhs or batched solve): compute it once,
/// then apply the cheap per-vector transforms below.
pub struct Scaling {
    row: Vec<f64>, // r_i, length m
    col: Vec<f64>, // c_j, length n  (also the x unscale factors)
    m: usize,
    n: usize,
}

impl Scaling {
    /// Equilibration factors for the dense row-major `m × n` matrix `a`, or
    /// `None` when its dynamic range is below [`SCALE_TRIGGER`] (well-conditioned
    /// — the caller should solve unscaled, which is then bit-identical to before).
    pub fn from_matrix(a: &[f64], m: usize, n: usize) -> Option<Scaling> {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &v in a.iter() {
            let av = v.abs();
            if av > 0.0 {
                lo = lo.min(av);
                hi = hi.max(av);
            }
        }
        if hi == 0.0 || hi / lo <= SCALE_TRIGGER {
            return None;
        }
        let (row, col) = equilibrate(a, m, n);
        Some(Scaling { row, col, m, n })
    }

    /// Scaled matrix `Â = R A C` (owned, row-major `m × n`).
    pub fn scale_matrix(&self, a: &[f64]) -> Vec<f64> {
        let (m, n) = (self.m, self.n);
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            let ri = self.row[i];
            for j in 0..n {
                out[i * n + j] = ri * a[i * n + j] * self.col[j];
            }
        }
        out
    }

    /// Scaled objective `ĉ = C c` (objective value is then invariant: `ĉᵀx̂ = cᵀx`).
    pub fn scale_c(&self, c: &[f64]) -> Vec<f64> {
        (0..self.n).map(|j| self.col[j] * c[j]).collect()
    }

    /// Scaled rhs `b̂ = R b`.
    pub fn scale_b(&self, b: &[f64]) -> Vec<f64> {
        (0..self.m).map(|i| self.row[i] * b[i]).collect()
    }

    /// Scaled lower bounds `l̂ = C⁻¹ l` (infinite bounds preserved).
    pub fn scale_lower(&self, l: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|j| if l[j] <= -INF { l[j] } else { l[j] / self.col[j] })
            .collect()
    }

    /// Scaled upper bounds `û = C⁻¹ u` (infinite bounds preserved).
    pub fn scale_upper(&self, u: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|j| if u[j] >= INF { u[j] } else { u[j] / self.col[j] })
            .collect()
    }

    /// Map a scaled primal point back to the original space in place: the first
    /// `n` entries are scaled `x_j ← c_j x̂_j` (exact — `c_j` is a power of two).
    /// Any trailing entries are left untouched.
    pub fn unscale_x(&self, x: &mut [f64]) {
        for (j, c) in self.col.iter().enumerate() {
            if j >= x.len() {
                break;
            }
            x[j] *= *c;
        }
    }
}

/// An owned, equilibrated copy of an [`LpView`] together with its RHS, for a
/// single solve. Thin convenience wrapper over [`Scaling`].
pub struct ScaledLp {
    scaling: Scaling,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
}

impl ScaledLp {
    /// Equilibrate `lp` (with rhs `b`) if its dynamic range exceeds
    /// [`SCALE_TRIGGER`]; otherwise return `None` so the caller solves the
    /// original system unchanged (zero copy, bit-identical).
    pub fn maybe_new(lp: &LpView<'_>, b: &[f64]) -> Option<ScaledLp> {
        let scaling = Scaling::from_matrix(lp.a, lp.m, lp.n)?;
        Some(ScaledLp {
            a: scaling.scale_matrix(lp.a),
            b: scaling.scale_b(b),
            c: scaling.scale_c(lp.c),
            l: scaling.scale_lower(lp.l),
            u: scaling.scale_upper(lp.u),
            scaling,
        })
    }

    /// Borrow the scaled LP as an [`LpView`].
    pub fn view(&self) -> LpView<'_> {
        LpView {
            a: &self.a,
            m: self.scaling.m,
            n: self.scaling.n,
            c: &self.c,
            l: &self.l,
            u: &self.u,
        }
    }

    /// The scaled right-hand side.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Map a scaled primal point back to the original space in place.
    pub fn unscale_x(&self, x: &mut [f64]) {
        self.scaling.unscale_x(x);
    }
}

/// Geometric-mean equilibration of the dense row-major `m × n` matrix `a`:
/// alternating sweeps set each column/row factor to `1/sqrt(min·max)` of the
/// current scaled magnitudes in that line, snapped to a power of two. Returns
/// `(row, col)` factor vectors.
fn equilibrate(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut row = vec![1.0f64; m];
    let mut col = vec![1.0f64; n];
    for _ in 0..MAX_PASSES {
        let mut changed = false;
        // Column sweep.
        for j in 0..n {
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for i in 0..m {
                let v = (row[i] * a[i * n + j] * col[j]).abs();
                if v > 0.0 {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if hi > 0.0 {
                let f = nearest_pow2(1.0 / (lo * hi).sqrt());
                if f != 1.0 {
                    col[j] *= f;
                    changed = true;
                }
            }
        }
        // Row sweep.
        for i in 0..m {
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            let base = i * n;
            for j in 0..n {
                let v = (row[i] * a[base + j] * col[j]).abs();
                if v > 0.0 {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if hi > 0.0 {
                let f = nearest_pow2(1.0 / (lo * hi).sqrt());
                if f != 1.0 {
                    row[i] *= f;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_rounding() {
        assert_eq!(nearest_pow2(1.0), 1.0);
        assert_eq!(nearest_pow2(3.0), 4.0); // log2(3)=1.58 → 2^2
        assert_eq!(nearest_pow2(1.4), 1.0); // log2(1.4)=0.49 → 2^0
        assert_eq!(nearest_pow2(0.2), 0.25); // log2(0.2)=-2.32 → 2^-2
        assert_eq!(nearest_pow2(0.0), 1.0);
        assert_eq!(nearest_pow2(f64::INFINITY), 1.0);
    }

    #[test]
    fn well_conditioned_is_no_op() {
        // Entries in [1,5]: dynamic range 5 ≪ trigger → no scaling.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 5,
            c: &[0.0; 5],
            l: &[0.0; 5],
            u: &[1.0; 5],
        };
        assert!(ScaledLp::maybe_new(&lp, &[9.0]).is_none());
    }

    #[test]
    fn wide_range_triggers_and_equilibrates() {
        // One huge and one tiny coefficient: range 1e9 ≫ trigger → scaled, and
        // the equilibrated matrix has its nonzero magnitudes brought together.
        let a = [1e9, 1.0, 1.0, 1e-3];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 2,
            c: &[1.0, 1.0],
            l: &[0.0, 0.0],
            u: &[1e12, 1e12],
        };
        let scaled = ScaledLp::maybe_new(&lp, &[1.0, 1.0]).expect("should scale");
        let sv = scaled.view();
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &v in sv.a.iter() {
            let av = v.abs();
            if av > 0.0 {
                lo = lo.min(av);
                hi = hi.max(av);
            }
        }
        // Raw range was 1e9/1e-3 = 1e12; equilibration must shrink it sharply.
        assert!(hi / lo < 1e4, "post-scale range {} too wide", hi / lo);
    }

    #[test]
    fn unscale_recovers_original_scale() {
        // Scaling derived from a wide-range matrix; x̂ maps back exactly by the
        // (power-of-two) column factors, and scale_lower/upper invert it.
        let a = [1e9, 0.0, 0.0, 1.0];
        let scaling = Scaling::from_matrix(&a, 2, 2).expect("should scale");
        let l = scaling.scale_lower(&[4.0, 8.0]);
        let mut x = l.clone();
        scaling.unscale_x(&mut x);
        // unscale ∘ scale_lower is the identity (exact for power-of-two factors).
        assert_eq!(x, vec![4.0, 8.0]);
        // Infinite bounds are preserved, not divided.
        assert_eq!(scaling.scale_upper(&[INF, 1e12])[0], INF);
    }

    #[test]
    fn shared_scaling_matches_single_solve_transform() {
        // The reusable Scaling applied piecewise must reproduce ScaledLp's combined
        // transform (the batch path and the single-solve path agree).
        let a = [1e8, 2.0, 3.0, 1e-2];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 2,
            c: &[5.0, 7.0],
            l: &[0.0, 0.0],
            u: &[1e10, 1e10],
        };
        let b = [1.0, 2.0];
        let single = ScaledLp::maybe_new(&lp, &b).expect("should scale");
        let sc = Scaling::from_matrix(lp.a, lp.m, lp.n).expect("should scale");
        assert_eq!(single.a, sc.scale_matrix(lp.a));
        assert_eq!(single.b, sc.scale_b(&b));
        assert_eq!(single.c, sc.scale_c(lp.c));
        assert_eq!(single.l, sc.scale_lower(lp.l));
        assert_eq!(single.u, sc.scale_upper(lp.u));
    }
}
