//! Floating-point error bounds shared across the solver.
//!
//! These are the *yardsticks* the #1397 audit is about: a comparison or a division
//! whose operand is a cancelling sum cannot be guarded by an absolute constant,
//! because the operand's round-off is governed by the magnitudes of the terms it is
//! a difference of, not by the magnitude of the result. Every consumer that needs
//! such a bound uses the one definition here rather than re-deriving a factor at the
//! call site — the divergence between [`crate::bnb::milp_driver`]'s reduced-cost
//! fixing and [`crate::presolve::duality`]'s is how the latter ended up strictly
//! weaker than the former (#1409).

/// Bound magnitude at or above which a column side is UNBOUNDED for any activity
/// or quotient computed from it — Python's `_relax/_numeric.py` `EFFECTIVE_INF`.
///
/// It sits below the LP layer's `INF = 1e20` sentinel on purpose: a Python column
/// declared without bounds carries `DEFAULT_VARIABLE_BOUND = ±9.999e19`
/// (`python/discopt/constants.py`), an ordinary float that a `>= 1e20` test reads as
/// finite. A consumer that multiplies or sums such a bound must test it against this
/// threshold instead (CLAUDE.md, INF note: test the bound, never a product).
pub const EFFECTIVE_INF: f64 = 1e19;

/// The classic dot-product error factor `γ_k = k·u/(1 − k·u)` (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, §3.1): a sum of `k` floating-point terms
/// evaluated in any order differs from the exact sum by at most `γ_k · Σ|terms|`.
///
/// `u` is taken as `f64::EPSILON` — **twice** the true unit roundoff `2⁻⁵³` — so the
/// first-order bound carries a factor-2 headroom over the `O(u²)` terms it drops and
/// over the float64 evaluation of the margin expression itself. (The Python
/// boundary's sharp NS margin, `_safe_lp_lower_bound_sharp`, buys the same headroom
/// with `u = 2⁻⁵³` and an explicit ×1.0625; see
/// `docs/dev/ns-sharp-margin-2026-07-16.md` §2.) Returns `+∞` — no usable bound, the
/// caller must bail — once `k·u ≥ 1`.
#[inline]
pub fn gamma(k: usize) -> f64 {
    let ku = k as f64 * f64::EPSILON;
    if ku >= 1.0 {
        f64::INFINITY
    } else {
        ku / (1.0 - ku)
    }
}

/// A rigorous lower bound on `|d|` where `d` was computed as a cancelling sum of `k`
/// terms whose magnitudes sum to `abs_sum`.
///
/// Reduced-cost fixing divides by a reduced cost `d_j = c_j − A_jᵀy`. **Over**-stating
/// `|d_j|` shrinks `⌊gap/|d_j|⌋` and can land it a whole integer low, fixing an
/// improving point out of the box — a false tightening on a certifying path. So the
/// divisor must be deflated *toward zero* by its own error before the division, which
/// is what this returns. `None` means the sign of `d` is not certain at this scale and
/// the caller must not tighten at all (#1409).
///
/// Note the two errors are not in the same currency, which is why a relative slack on
/// the *gap* does not cover this: that slack is relative to `gap`, this error is
/// relative to the *divisor*. Measured on the #1409 witness — a column with
/// `Σ|a_ij y_i| = 2e14` and `d_j = 1e7` — the computed `|d_j|` came out
/// `1e7 + 6.25e-3`, which turned a valid `⌊gap/d_j⌋ = 3` into `2` and excluded a point
/// beating the incumbent by 9.37e-3, i.e. 9374× the solver's absolute tolerance,
/// while the gap slack supplied a relative widening of only 3.3e-14.
#[inline]
pub fn deflated_magnitude(d: f64, abs_sum: f64, k: usize) -> Option<f64> {
    if !d.is_finite() || !abs_sum.is_finite() {
        return None;
    }
    // `k + 2` rather than `k`: two additions beyond the dot itself — the `c_j − dot`
    // subtraction and the evaluation of this bound — are also rounded.
    let err = gamma(k + 2) * abs_sum;
    if !err.is_finite() {
        return None;
    }
    let safe = d.abs() - err;
    if safe > 0.0 {
        Some(safe)
    } else {
        None
    }
}

/// `v` clamped into the box edge pair `(a, b)` taken in either order, so a box
/// crossed by round-off never panics `f64::clamp` (#1917). On an ordered box this
/// is exactly `v.clamp(a, b)`.
#[inline]
pub fn clamp_ordered(v: f64, a: f64, b: f64) -> f64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    v.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #1917: an ulp-crossed edge pair clamps instead of panicking, and an ordered
    /// pair is exactly `f64::clamp`.
    #[test]
    fn clamp_ordered_accepts_a_crossed_pair_1917() {
        assert_eq!(clamp_ordered(1.0, 1.0, 0.9999999999999996), 1.0);
        assert_eq!(
            clamp_ordered(0.0, 1.0, 0.9999999999999996),
            0.9999999999999996
        );
        for &(v, a, b) in &[(0.5, 0.0, 1.0), (-3.0, -1.0, 2.0), (9.0, -1.0, 2.0)] {
            assert_eq!(clamp_ordered(v, a, b), f64::clamp(v, a, b));
        }
    }

    #[test]
    fn gamma_is_monotone_and_bails_at_the_limit() {
        assert!(gamma(1) > 0.0);
        assert!(gamma(2) > gamma(1));
        assert!(
            gamma(usize::MAX).is_infinite(),
            "k*u >= 1 must give no bound"
        );
    }

    #[test]
    fn deflation_moves_toward_zero_and_refuses_an_uncertain_sign() {
        // A well-resolved reduced cost keeps essentially all of its magnitude.
        let safe = deflated_magnitude(1.0, 4.0, 4).expect("1.0 vs a 4.0 term sum is resolved");
        assert!(safe < 1.0, "deflation must move the divisor toward zero");
        assert!(safe > 0.999, "a resolved value must not be gutted");
        // The #1409 regime: the result is dwarfed by the magnitudes it is a
        // difference of, so its sign is not certain and the caller must not divide.
        assert!(
            deflated_magnitude(1e-9, 2e14, 4).is_none(),
            "|d| far below its own error bound must refuse, not return a tiny divisor"
        );
        // Non-finite input is a refusal, never a silent pass-through.
        assert!(deflated_magnitude(f64::NAN, 1.0, 1).is_none());
        assert!(deflated_magnitude(1.0, f64::INFINITY, 1).is_none());
    }
}
