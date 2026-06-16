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

/// An owned, equilibrated copy of an [`LpView`] together with its RHS and the
/// column factors needed to map a scaled solution back to the original space.
pub struct ScaledLp {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
    m: usize,
    n: usize,
    col: Vec<f64>, // column scale factors c_j (for unscaling x)
}

impl ScaledLp {
    /// Equilibrate `lp` (with rhs `b`) if its dynamic range exceeds
    /// [`SCALE_TRIGGER`]; otherwise return `None` so the caller solves the
    /// original system unchanged (zero copy, bit-identical).
    pub fn maybe_new(lp: &LpView<'_>, b: &[f64]) -> Option<ScaledLp> {
        let (m, n) = (lp.m, lp.n);
        // Dynamic range of the matrix nonzeros.
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &v in lp.a.iter() {
            let av = v.abs();
            if av > 0.0 {
                lo = lo.min(av);
                hi = hi.max(av);
            }
        }
        if hi == 0.0 || hi / lo <= SCALE_TRIGGER {
            return None; // empty or well-conditioned: no scaling
        }

        let (row, col) = equilibrate(lp.a, m, n);
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            let ri = row[i];
            for j in 0..n {
                a[i * n + j] = ri * lp.a[i * n + j] * col[j];
            }
        }
        let b = (0..m).map(|i| row[i] * b[i]).collect();
        let c = (0..n).map(|j| col[j] * lp.c[j]).collect();
        // Bounds: l̂ = l / c_j (c_j > 0, so order is preserved). Infinite bounds
        // stay infinite so free / one-sided variables keep their character.
        let l = (0..n)
            .map(|j| if lp.l[j] <= -INF { lp.l[j] } else { lp.l[j] / col[j] })
            .collect();
        let u = (0..n)
            .map(|j| if lp.u[j] >= INF { lp.u[j] } else { lp.u[j] / col[j] })
            .collect();
        Some(ScaledLp {
            a,
            b,
            c,
            l,
            u,
            m,
            n,
            col,
        })
    }

    /// Borrow the scaled LP as an [`LpView`].
    pub fn view(&self) -> LpView<'_> {
        LpView {
            a: &self.a,
            m: self.m,
            n: self.n,
            c: &self.c,
            l: &self.l,
            u: &self.u,
        }
    }

    /// The scaled right-hand side.
    pub fn b(&self) -> &[f64] {
        &self.b
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
        let a = [1e9, 0.0, 0.0, 1.0];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 2,
            c: &[1.0, 1.0],
            l: &[0.0, 0.0],
            u: &[1e12, 1e12],
        };
        let scaled = ScaledLp::maybe_new(&lp, &[1.0, 1.0]).expect("should scale");
        // x̂ in scaled space maps back by the (power-of-two) column factors.
        let mut x = vec![2.0, 3.0];
        let xhat = x.clone();
        scaled.unscale_x(&mut x);
        assert_eq!(x[0], xhat[0] * scaled.col[0]);
        assert_eq!(x[1], xhat[1] * scaled.col[1]);
    }
}
