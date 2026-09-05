//! Objective-integrality detection: the *granularity* of a MILP objective.
//!
//! When every variable carrying a nonzero objective coefficient is
//! integer-constrained and the coefficients share a rational scale, every
//! attainable objective value lies on a lattice `k·g + obj_const`. The spacing
//! `g` is the **granularity**, and it is worth a surprising amount:
//!
//! * A node whose lower bound `lb` satisfies `lb > U - g` (for incumbent `U`)
//!   cannot contain a *strictly better* solution — the next attainable value
//!   below `U` is `U - g`. Pruning on `U - g` rather than `U` is what lets a
//!   fractional dual bound close an integral gap. On `mk_binpack` the root LP
//!   bound is 7.72 against an optimum of 8 and **no cut moves it** (measured:
//!   SCIP applies 140 cuts and stays at 7.72); the entire gap is closed by this
//!   rule, and without it the search runs 551,849 nodes with the bound frozen.
//! * Symmetrically the reported dual bound may be rounded *up* onto the lattice.
//!
//! Prior art: SCIP computes the ceiling once, onto the primal cutoff
//! (`scip/src/scip/prob.c` `SCIPprobCheckObjIntegral`, applied at
//! `scip/src/scip/primal.c:428`), so a single value feeds node fathoming, the
//! leaf queue, the LP objective limit and conflict analysis. HiGHS gates on an
//! integral *scale* rather than integral coefficients
//! (`highs/mip/HighsObjectiveFunction.h` `isIntegral()`, detector
//! `highs/mip/HighsMipSolverData.cpp` `checkObjIntegrality`), which is strictly
//! more general — a `0.5`-spaced objective still qualifies. We follow HiGHS.
//!
//! # Soundness
//! The detector is the load-bearing part: a granularity that is too LARGE prunes
//! nodes that could hold the optimum, i.e. produces a false bound — the one
//! failure mode this whole file must not have. So it **refuses** (returns
//! `None`) on anything it cannot prove: a nonzero cost on a continuous column, a
//! coefficient that is not a clean rational at the working precision, or a
//! denominator beyond `MAX_DENOM`. Refusing costs a missed speedup; guessing
//! costs a wrong answer.

/// Largest denominator admitted when rationalizing a coefficient. Beyond this
/// the coefficient is treated as irrational and the objective as non-integral.
///
/// Deliberately small. The cap is not a performance guard, it is the soundness
/// guard: continued fractions will rationalize *anything* if allowed a big
/// enough denominator — π is 103993/33102 to well inside `RATIONAL_TOL` — and a
/// lattice inferred that way is an artifact of floating point, not a property of
/// the model. Capping at 1000 rejects π (the best convergent under the cap,
/// 355/113, is off by 2.7e-7) while still admitting every spacing a real
/// objective uses (halves, thirds, 0.01, 0.001). It also floors the granularity
/// at 1e-3, below which the cutoff `U - g` is indistinguishable from `U` and the
/// rule buys nothing anyway.
const MAX_DENOM: i64 = 1_000;

/// Relative tolerance for accepting `d·c` as an integer.
const RATIONAL_TOL: f64 = 1e-9;

/// Coefficients below this magnitude are treated as exactly zero (they carry no
/// objective contribution, so they impose no lattice constraint).
const ZERO_TOL: f64 = 1e-12;

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn lcm_i64(a: i64, b: i64) -> Option<i64> {
    if a == 0 || b == 0 {
        return None;
    }
    let g = gcd_i64(a, b);
    a.checked_div(g).and_then(|q| q.checked_mul(b))
}

/// Smallest `d in 1..=MAX_DENOM` with `d·v` integral to `RATIONAL_TOL`, else
/// `None`. Continued-fraction expansion, so a value like `1/3` is found at
/// `d = 3` rather than by scanning.
fn denominator_of(v: f64) -> Option<i64> {
    if !v.is_finite() {
        return None;
    }
    // Continued-fraction convergents p/q -> v.
    let (mut p0, mut q0, mut p1, mut q1) = (0i64, 1i64, 1i64, 0i64);
    let mut x = v;
    for _ in 0..64 {
        let a = x.floor();
        if !a.is_finite() || a.abs() > MAX_DENOM as f64 {
            return None;
        }
        let ai = a as i64;
        let p2 = ai.checked_mul(p1)?.checked_add(p0)?;
        let q2 = ai.checked_mul(q1)?.checked_add(q0)?;
        if q2.abs() > MAX_DENOM {
            return None;
        }
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        if q1 != 0 {
            let approx = p1 as f64 / q1 as f64;
            if (approx - v).abs() <= RATIONAL_TOL * v.abs().max(1.0) {
                let q = q1.abs();
                return if q == 0 { None } else { Some(q) };
            }
        }
        let frac = x - a;
        if frac.abs() <= 1e-15 {
            return None; // exhausted without meeting the tolerance
        }
        x = 1.0 / frac;
    }
    None
}

/// The objective granularity `g > 0` when every attainable objective value lies
/// on the lattice `k·g + obj_const`, or `None` when that cannot be proven.
///
/// `c` and `is_int` are indexed alike over the structural columns. `obj_const`
/// is deliberately **not** a parameter: it shifts the lattice but not its
/// spacing, and every use here is a *difference* of objective values.
pub fn objective_granularity(c: &[f64], is_int: &[bool]) -> Option<f64> {
    debug_assert_eq!(c.len(), is_int.len());
    let mut den: i64 = 1;
    let mut any = false;
    for (j, &cj) in c.iter().enumerate() {
        if cj.abs() <= ZERO_TOL {
            continue;
        }
        // A nonzero cost on a continuous column destroys the lattice outright.
        if !is_int.get(j).copied().unwrap_or(false) {
            return None;
        }
        any = true;
        den = lcm_i64(den, denominator_of(cj)?)?;
        if den > MAX_DENOM {
            return None;
        }
    }
    if !any {
        // A constant objective: every value is `obj_const`. Sound to call this
        // granular, but there is nothing to prune, so decline and keep the
        // downstream arithmetic free of a degenerate case.
        return None;
    }
    // With `d·c_j` all integral, the objective is `(1/d)·Σ (d·c_j) x_j` over
    // integer `x_j`, hence a multiple of `gcd_j(d·c_j) / d`.
    let mut g_num: i64 = 0;
    for &cj in c.iter() {
        if cj.abs() <= ZERO_TOL {
            continue;
        }
        let scaled = cj * den as f64;
        let r = scaled.round();
        if (scaled - r).abs() > RATIONAL_TOL * scaled.abs().max(1.0) {
            return None;
        }
        if r.abs() > i64::MAX as f64 {
            return None;
        }
        g_num = gcd_i64(g_num, r as i64);
    }
    if g_num == 0 {
        return None;
    }
    let g = g_num as f64 / den as f64;
    if !g.is_finite() || g <= 0.0 {
        return None;
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_binary_objective_has_granularity_one() {
        let c = vec![1.0, 1.0, 1.0];
        let is_int = vec![true; 3];
        assert_eq!(objective_granularity(&c, &is_int), Some(1.0));
    }

    #[test]
    fn integer_coefficients_use_their_gcd() {
        // 4x + 6y over integers hits only multiples of 2.
        let c = vec![4.0, 6.0];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), Some(2.0));
    }

    #[test]
    fn half_integral_objective_is_detected() {
        let c = vec![0.5, 1.5];
        let is_int = vec![true, true];
        let g = objective_granularity(&c, &is_int).expect("0.5-spaced lattice");
        assert!((g - 0.5).abs() < 1e-12, "got {g}");
    }

    #[test]
    fn zero_cost_columns_are_ignored_even_when_continuous() {
        let c = vec![1.0, 0.0, 2.0];
        let is_int = vec![true, false, true];
        assert_eq!(objective_granularity(&c, &is_int), Some(1.0));
    }

    #[test]
    fn refuses_when_a_continuous_column_carries_cost() {
        let c = vec![1.0, 3.0];
        let is_int = vec![true, false];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    #[test]
    fn refuses_an_irrational_coefficient() {
        // Left uncapped, continued fractions would hand back 1/33102 here.
        let c = vec![1.0, std::f64::consts::PI];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    #[test]
    fn handles_negative_and_mixed_sign_coefficients() {
        let c = vec![-2.0, 4.0, -6.0];
        let is_int = vec![true; 3];
        assert_eq!(objective_granularity(&c, &is_int), Some(2.0));
    }

    #[test]
    fn refuses_a_denominator_past_the_cap() {
        let c = vec![1.0, 1.0 / 4001.0];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    #[test]
    fn refuses_a_constant_objective() {
        let c = vec![0.0, 0.0];
        let is_int = vec![true, true];
        assert_eq!(objective_granularity(&c, &is_int), None);
    }

    /// The soundness property, stated as a test: for random integer points the
    /// objective must actually land on the detected lattice. A granularity that
    /// is too large would produce a false bound, so this is the guard that
    /// matters.
    #[test]
    fn detected_granularity_never_overstates_the_lattice() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut checked = 0usize;
        for _ in 0..400 {
            let n = 1 + (next() % 6) as usize;
            let c: Vec<f64> = (0..n)
                .map(|_| {
                    let num = (next() % 21) as i64 - 10;
                    let den = 1 + (next() % 4) as i64;
                    num as f64 / den as f64
                })
                .collect();
            let is_int = vec![true; n];
            let Some(g) = objective_granularity(&c, &is_int) else {
                continue;
            };
            for _ in 0..20 {
                let obj: f64 = c
                    .iter()
                    .map(|&cj| cj * (((next() % 41) as i64) - 20) as f64)
                    .sum();
                let k = obj / g;
                assert!(
                    (k - k.round()).abs() < 1e-6,
                    "objective {obj} is not a multiple of granularity {g} (c={c:?})"
                );
                checked += 1;
            }
        }
        // CLAUDE.md §6: a probe that asserted nothing must fail, not pass.
        assert!(
            checked > 500,
            "granularity property never exercised ({checked})"
        );
    }
}
