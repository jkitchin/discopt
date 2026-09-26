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

// ─────────────────────────────────────────────────────────────
// Directed rounding (rigorous interval endpoints)
// ─────────────────────────────────────────────────────────────
//
// IEEE-754 `+ - * /` are correctly rounded: the computed result is the exact result
// rounded to nearest. An interval endpoint computed that way can therefore land on
// the WRONG side of the exact value by up to half an ulp, and a backward FBBT chain
// can amplify that: an exact-zero preimage computed as `-9e-14`, pushed through an
// odd root, becomes a `3e-6` lower bound on a binary, which integer rounding then
// snaps to `1` — cutting off the optimum and certifying a false optimum.
//
// The functions below return a value on the requested side of the exact result.
// They use error-free transformations (TwoSum; an FMA residual for `*` and `/`) to
// learn whether — and in which direction — the rounding happened, and step one ulp
// only when it went the wrong way. An exact result is returned bit-for-bit, so
// integer-valued arithmetic (most of a MILP's) is unchanged.
//
// Non-finite operands pass through with IEEE semantics (`±∞` is already the loosest
// possible endpoint). A finite operation that overflows to `±∞` on the wrong side
// is clamped to `∓f64::MAX`, the representable value just inside it.

/// The next representable `f64` above `x`. `+∞` and NaN are returned unchanged.
///
/// `f64::next_up` would do, but it is stable only from Rust 1.86 and the MSRV is
/// 1.84; this is the same bit-step.
#[inline]
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        // Both +0.0 and -0.0 step to the smallest positive subnormal.
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// The next representable `f64` below `x`. `-∞` and NaN are returned unchanged.
#[inline]
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// Magnitude below which an FMA residual may be inexact (gradual underflow), so the
/// directed operations below stop trusting it and step unconditionally.
const RESIDUAL_TRUST_FLOOR: f64 = 1e-290;

#[inline]
fn two_sum_err(a: f64, b: f64, s: f64) -> f64 {
    // Knuth's TwoSum: `a + b == s + err` exactly, for finite a, b, s.
    let bp = s - a;
    let ap = s - bp;
    (a - ap) + (b - bp)
}

/// A lower bound on the exact `a + b`.
#[inline]
pub fn add_down(a: f64, b: f64) -> f64 {
    let s = a + b;
    if !(a.is_finite() && b.is_finite()) {
        return s;
    }
    if s == f64::INFINITY {
        return f64::MAX;
    }
    if !s.is_finite() {
        return s;
    }
    if two_sum_err(a, b, s) < 0.0 {
        next_down(s)
    } else {
        s
    }
}

/// An upper bound on the exact `a + b`.
#[inline]
pub fn add_up(a: f64, b: f64) -> f64 {
    let s = a + b;
    if !(a.is_finite() && b.is_finite()) {
        return s;
    }
    if s == f64::NEG_INFINITY {
        return f64::MIN;
    }
    if !s.is_finite() {
        return s;
    }
    if two_sum_err(a, b, s) > 0.0 {
        next_up(s)
    } else {
        s
    }
}

/// A lower bound on the exact `a - b`.
#[inline]
pub fn sub_down(a: f64, b: f64) -> f64 {
    add_down(a, -b)
}

/// An upper bound on the exact `a - b`.
#[inline]
pub fn sub_up(a: f64, b: f64) -> f64 {
    add_up(a, -b)
}

/// Sign of `exact - computed` for `p = a * b`: `-1`, `0`, `+1`, or `None` when the
/// residual cannot be trusted (underflow territory).
#[inline]
fn mul_err_sign(a: f64, b: f64, p: f64) -> Option<i8> {
    if p.abs() < RESIDUAL_TRUST_FLOOR && a != 0.0 && b != 0.0 {
        return None;
    }
    let e = a.mul_add(b, -p);
    Some(if e > 0.0 {
        1
    } else if e < 0.0 {
        -1
    } else {
        0
    })
}

/// A lower bound on the exact `a * b`.
#[inline]
pub fn mul_down(a: f64, b: f64) -> f64 {
    let p = a * b;
    if !(a.is_finite() && b.is_finite()) {
        return p;
    }
    if p == f64::INFINITY {
        return f64::MAX;
    }
    if !p.is_finite() {
        return p;
    }
    match mul_err_sign(a, b, p) {
        Some(s) if s >= 0 => p,
        _ => next_down(p),
    }
}

/// An upper bound on the exact `a * b`.
#[inline]
pub fn mul_up(a: f64, b: f64) -> f64 {
    let p = a * b;
    if !(a.is_finite() && b.is_finite()) {
        return p;
    }
    if p == f64::NEG_INFINITY {
        return f64::MIN;
    }
    if !p.is_finite() {
        return p;
    }
    match mul_err_sign(a, b, p) {
        Some(s) if s <= 0 => p,
        _ => next_up(p),
    }
}

/// Sign of `exact - computed` for `q = a / b` (b finite, nonzero), or `None`.
#[inline]
fn div_err_sign(a: f64, b: f64, q: f64) -> Option<i8> {
    if q.abs() < RESIDUAL_TRUST_FLOOR && a != 0.0 {
        return None;
    }
    // r = a - q*b exactly (for a correctly rounded q), and a/b = q + r/b.
    let r = (-q).mul_add(b, a);
    if !r.is_finite() {
        return None;
    }
    let s = if r == 0.0 {
        0
    } else if (r > 0.0) == (b > 0.0) {
        1
    } else {
        -1
    };
    Some(s)
}

/// A lower bound on the exact `a / b`.
#[inline]
pub fn div_down(a: f64, b: f64) -> f64 {
    let q = a / b;
    if !(a.is_finite() && b.is_finite()) || b == 0.0 {
        return q;
    }
    if q == f64::INFINITY {
        return f64::MAX;
    }
    if !q.is_finite() {
        return q;
    }
    match div_err_sign(a, b, q) {
        Some(s) if s >= 0 => q,
        _ => next_down(q),
    }
}

/// An upper bound on the exact `a / b`.
#[inline]
pub fn div_up(a: f64, b: f64) -> f64 {
    let q = a / b;
    if !(a.is_finite() && b.is_finite()) || b == 0.0 {
        return q;
    }
    if q == f64::NEG_INFINITY {
        return f64::MIN;
    }
    if !q.is_finite() {
        return q;
    }
    match div_err_sign(a, b, q) {
        Some(s) if s <= 0 => q,
        _ => next_up(q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn next_up_down_step_one_ulp_and_pass_infinities() {
        assert_eq!(next_up(1.0), 1.0 + f64::EPSILON);
        assert_eq!(next_down(1.0), 1.0 - f64::EPSILON / 2.0);
        assert!(next_up(0.0) > 0.0 && next_up(-0.0) > 0.0);
        assert!(next_down(0.0) < 0.0);
        assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
        assert_eq!(next_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(next_up(f64::NEG_INFINITY), -f64::MAX);
        assert!(next_up(f64::NAN).is_nan());
    }

    #[test]
    fn directed_ops_are_exact_when_exact() {
        // Bit-identical on exact arithmetic: the property that keeps an
        // integer-coefficient model's FBBT unchanged.
        assert_eq!(add_down(2.0, 3.0), 5.0);
        assert_eq!(add_up(2.0, 3.0), 5.0);
        assert_eq!(sub_down(3461.89, 3461.89), 0.0);
        assert_eq!(mul_down(3.0, -4.0), -12.0);
        assert_eq!(mul_up(3.0, -4.0), -12.0);
        assert_eq!(div_down(1.0, 4.0), 0.25);
        assert_eq!(div_up(-1.0, 4.0), -0.25);
    }

    /// The directed pair must bracket the round-to-nearest result, on the side the
    /// exact residual (TwoSum / FMA) says the rounding went.
    #[test]
    fn directed_ops_bracket_the_exact_result() {
        // The #9219 chain: fl(-1.15 + 3461.89) is not the exact sum of the two
        // doubles; the directed pair must bracket it (TwoSum error is exact).
        let (a, b) = (-1.15_f64, 3461.89_f64);
        let s = a + b;
        let err = two_sum_err(a, b, s);
        assert!(err != 0.0, "fixture must actually round");
        let (lo, hi) = (add_down(a, b), add_up(a, b));
        assert!(lo <= s && s <= hi && lo < hi);
        // exact = s + err lies inside [lo, hi]
        if err > 0.0 {
            assert_eq!(lo, s);
            assert!(hi > s);
        } else {
            assert_eq!(hi, s);
            assert!(lo < s);
        }
        // Sweep: directed results always straddle the round-to-nearest result, on
        // the side the exact residual says.
        let vals: [f64; 10] = [
            0.1,
            -0.1,
            1.0 / 3.0,
            -2.0 / 3.0,
            3460.74,
            -3461.89,
            1e-300,
            7e300,
            -1.15,
            1e16,
        ];
        let mut n = 0;
        for &x in &vals {
            for &y in &vals {
                let p = x * y;
                if p.is_finite() && p.abs() > 1e-280 {
                    let e = x.mul_add(y, -p);
                    let (d, u) = (mul_down(x, y), mul_up(x, y));
                    assert!(d <= p && p <= u);
                    assert!(if e > 0.0 {
                        u > p
                    } else if e < 0.0 {
                        d < p
                    } else {
                        d == p && u == p
                    });
                    n += 1;
                }
                if y != 0.0 {
                    let q = x / y;
                    if q.is_finite() && q.abs() > 1e-280 {
                        let (d, u) = (div_down(x, y), div_up(x, y));
                        assert!(d <= q && q <= u);
                        n += 1;
                    }
                }
            }
        }
        assert!(n > 100, "sweep executed {n} checks");
    }

    #[test]
    fn directed_ops_clamp_overflow_on_the_inner_side() {
        assert_eq!(add_down(f64::MAX, f64::MAX), f64::MAX);
        assert_eq!(add_up(f64::MAX, f64::MAX), f64::INFINITY);
        assert_eq!(mul_up(-f64::MAX, 2.0), f64::MIN);
        assert_eq!(mul_down(-f64::MAX, 2.0), f64::NEG_INFINITY);
    }
}
