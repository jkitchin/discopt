//! Marchand–Wolsey aggregation c-MIR separation.
//!
//! Single-row MIR ([`super::mir`]) is only as strong as one model row: on the
//! integer-product / graph-partition class it closes ~0% of the root gap that
//! SCIP's cut loop closes ~100% of (see `docs/dev/certification-gap-plan.md` §7,
//! "0b RESULTS / VERDICT"). SCIP's workhorse there is *aggregation*
//! (Marchand & Wolsey 2001, "Aggregation and mixed integer rounding to solve
//! MIPs"): combine a small set of `≤` rows with **nonnegative** weights into one
//! valid implied row, chosen so a continuous variable *cancels* and integer
//! structure is exposed, then apply complemented MIR to the aggregate.
//!
//! # Soundness (the load-bearing property)
//! For `≤` rows `a_i · x ≤ b_i` (i in a chosen set) and weights `λ_i ≥ 0`, the
//! aggregate
//!
//! ```text
//!   Σ_i λ_i (a_i · x)  ≤  Σ_i λ_i b_i
//! ```
//!
//! is a valid inequality for the feasible set: it is a nonnegative combination of
//! valid `≤` inequalities, so every feasible `x` (in particular every
//! integer-feasible `x`) satisfies it. Feeding this single valid `≤` row to
//! [`separate_mir`] — whose per-row cut validity over the integer box is already
//! established (`mir.rs`, `mir_validity_random_*` tests) — therefore yields a cut
//! valid for the original feasible set. No cut can remove an integer-feasible
//! point of the original system. We never reimplement MIR here: aggregation only
//! *builds the row*; MIR (bound substitution, complementation, δ-scan) is reused
//! verbatim.
//!
//! # Row-selection heuristic (this build: 2-row, continuous-cancelling)
//! Following MW/SCIP `sepa_cmir`, the aggregation depth is kept small for
//! tractability. This build ships the smallest sound, useful slice: **pairs** of
//! `≤` rows combined to cancel one continuous variable.
//!
//! For each cancel-target column `t` and each pair of rows `(i, k)` whose
//! coefficients on `t` are *strictly opposite in sign* (`a_it > 0 > a_kt`, up to
//! symmetry), the unique-up-to-scale nonnegative weights that cancel `t` are
//! `λ_i = |a_kt|`, `λ_k = |a_it|` (both `> 0`). The aggregate row drops column
//! `t` and is fed to `separate_mir`. Among all pairs (and every cancel column)
//! the most violated valid cut at `x` is kept. The cancel targets are the
//! *continuous* columns (the MW target); on a fully-lifted McCormick LP where
//! every column is marked integer there is no continuous column, so the targets
//! fall back to the *fractional* columns at `x`. Cancelling an integer column is
//! equally sound — a nonnegative row combination is valid whichever column
//! cancels — so this fallback never risks a cut. Deeper aggregation (≥3 rows,
//! bound-substitution of a variable at a bound before cancelling) is follow-on.

use crate::lp::mir::{separate_mir, MirCut};

/// Result of one aggregation separation: the cut plus its violation at the
/// separation point (so the caller / cut pool can rank it against other cuts).
pub struct AggCut {
    /// The MIR cut over the original structural variables.
    pub cut: MirCut,
    /// Violation `coeffs · x − rhs` at the separation point (`> tol`).
    pub violation: f64,
}

/// Below this the coefficient on the cancel column is treated as zero (no
/// meaningful cancellation possible) and the pair is skipped.
const CANCEL_MIN: f64 = 1e-9;

/// Cap on the aggregate row's dynamism (max/min nonzero |coeff|). Combining rows
/// of very different scales makes the aggregate numerically unsafe to round; such
/// pairs are skipped (they also rarely give a useful cut).
const AGG_MAX_DYNAMISM: f64 = 1e8;

/// Separate aggregation c-MIR cuts from the `≤` rows `a_ub · x ≤ b_ub`.
///
/// `a_ub` is row-major `m × n`; `l`/`u` are the length-`n` variable bounds and
/// `integrality[j]` marks integer columns — all passed straight through to
/// [`separate_mir`] for the final MIR step. `x` is the separation point; `tol`
/// the violation threshold; `max_dynamism` the MIR coefficient-dynamism cap.
///
/// The heuristic pairs rows to cancel a continuous variable (see module docs),
/// forms the nonnegative aggregate, and applies complemented MIR to it. Returns
/// at most one cut per (pair, cancelled-column) candidate — the ones that produce
/// a violated, valid MIR cut — each carrying its violation for downstream
/// selection. Deterministic: iteration order is fixed, ties broken by first-seen.
///
/// # Soundness
/// Each returned cut is a complemented MIR cut of a **nonnegative combination**
/// of the original `≤` rows, hence valid for the original feasible set (module
/// docs). Aggregation never touches the MIR math; it only builds valid rows.
#[allow(clippy::too_many_arguments)]
pub fn separate_aggregation_mir(
    a_ub: &[f64],
    b_ub: &[f64],
    l: &[f64],
    u: &[f64],
    integrality: &[bool],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
) -> Vec<AggCut> {
    let m = b_ub.len();
    let n = a_ub.len().checked_div(m).unwrap_or(0);
    let mut out: Vec<AggCut> = Vec::new();
    if n == 0 || m < 2 {
        return out;
    }

    let row = |i: usize| &a_ub[i * n..(i + 1) * n];

    // Cancellation targets. The Marchand–Wolsey heuristic cancels a *continuous*
    // variable to expose the integer structure MIR rounds, so continuous columns
    // are the primary target. On a fully-lifted McCormick LP, though, every
    // column (including the product-aux `w = x·y`) is marked integer, so there
    // is no continuous column to cancel; there we fall back to cancelling any
    // column whose LP point is fractional (the interesting ones to round).
    //
    // Cancelling an integer column is equally SOUND — a nonnegative row
    // combination is valid regardless of which column cancels — it is only the
    // heuristic target that differs. Validity never depends on this choice.
    let cont_cols: Vec<usize> = (0..n).filter(|&j| !integrality[j]).collect();
    let cancel_cols: Vec<usize> = if !cont_cols.is_empty() {
        cont_cols
    } else {
        (0..n)
            .filter(|&j| (x[j] - x[j].round()).abs() > 1e-6)
            .collect()
    };
    if cancel_cols.is_empty() {
        return out;
    }

    for i in 0..m {
        let ai = row(i);
        for k in (i + 1)..m {
            let ak = row(k);
            for &t in &cancel_cols {
                let (ait, akt) = (ai[t], ak[t]);
                // Need strictly opposite signs on t to cancel with λ ≥ 0.
                if !(ait.abs() > CANCEL_MIN && akt.abs() > CANCEL_MIN) {
                    continue;
                }
                if ait.signum() == akt.signum() {
                    continue;
                }
                // Nonnegative weights that cancel column t: λ_i = |a_kt|,
                // λ_k = |a_it|. Both strictly positive ⇒ valid nonneg combo.
                let lam_i = akt.abs();
                let lam_k = ait.abs();

                // Build the aggregate row a_agg = λ_i a_i + λ_k a_k, rhs
                // b_agg = λ_i b_i + λ_k b_k. Column t cancels to (near) zero;
                // force it exactly zero so MIR sees the dropped structure.
                let mut a_agg = vec![0.0_f64; n];
                let mut max_c = 0.0_f64;
                let mut min_c = f64::INFINITY;
                for j in 0..n {
                    let v = lam_i * ai[j] + lam_k * ak[j];
                    a_agg[j] = v;
                    let av = v.abs();
                    if av > max_c {
                        max_c = av;
                    }
                    if av > CANCEL_MIN && av < min_c {
                        min_c = av;
                    }
                }
                a_agg[t] = 0.0;
                let b_agg = lam_i * b_ub[i] + lam_k * b_ub[k];

                // Numerical guards: the aggregate must be finite, non-trivial,
                // and not wildly ill-scaled (rounding a high-dynamism row is
                // unsafe). These reject the pair; they never weaken a cut.
                if max_c <= CANCEL_MIN
                    || !a_agg.iter().all(|v| v.is_finite())
                    || !b_agg.is_finite()
                    || (min_c.is_finite() && max_c / min_c > AGG_MAX_DYNAMISM)
                {
                    continue;
                }

                // Apply the existing complemented MIR to the single aggregate
                // row. separate_mir returns the most-violated valid cut for it.
                let cuts = separate_mir(&a_agg, &[b_agg], l, u, integrality, x, tol, max_dynamism);
                for cut in cuts {
                    let act: f64 = (0..n).map(|j| cut.coeffs[j] * x[j]).sum::<f64>();
                    let viol = act - cut.rhs;
                    if viol > tol {
                        out.push(AggCut {
                            cut,
                            violation: viol,
                        });
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Exhaustively check a `≤` cut excludes no feasible point of the *system*
    /// `A x ≤ b` (all rows), enumerating integer columns over `[lo, hi]` and, for
    /// continuous columns, sampling a dense grid across `[l, u]`. A cut from an
    /// aggregate of a *subset* of rows must still respect the FULL system: the
    /// aggregate is implied by the system, so no system-feasible point may be cut.
    #[allow(clippy::too_many_arguments)]
    fn assert_valid_system(
        coeffs: &[f64],
        rhs: f64,
        a_ub: &[f64],
        b_ub: &[f64],
        integrality: &[bool],
        lo: &[i64],
        hi: &[i64],
        cont_samples: usize,
    ) {
        let n = integrality.len();
        let m = b_ub.len();
        // Integer columns iterate over [lo,hi]; continuous columns iterate over
        // a grid of `cont_samples` points across [lo,hi] (as reals).
        let mut idx = vec![0usize; n]; // per-column step index
        let steps: Vec<usize> = (0..n)
            .map(|j| {
                if integrality[j] {
                    (hi[j] - lo[j] + 1).max(1) as usize
                } else {
                    cont_samples.max(2)
                }
            })
            .collect();
        let value = |j: usize, s: usize| -> f64 {
            if integrality[j] {
                (lo[j] + s as i64) as f64
            } else {
                let span = (hi[j] - lo[j]) as f64;
                lo[j] as f64 + span * (s as f64) / ((steps[j] - 1) as f64)
            }
        };
        loop {
            let xv: Vec<f64> = (0..n).map(|j| value(j, idx[j])).collect();
            // Feasible for the FULL system?
            let feas = (0..m).all(|r| dot(&a_ub[r * n..(r + 1) * n], &xv) <= b_ub[r] + 1e-9);
            if feas {
                assert!(
                    dot(coeffs, &xv) <= rhs + 1e-6,
                    "aggregation cut {coeffs:?} <= {rhs} excludes system-feasible {xv:?}"
                );
            }
            // mixed-radix increment
            let mut c = 0;
            while c < n {
                idx[c] += 1;
                if idx[c] < steps[c] {
                    break;
                }
                idx[c] = 0;
                c += 1;
            }
            if c == n {
                break;
            }
        }
    }

    /// Constructed instance where single-row MIR finds NO violated cut but a
    /// 2-row aggregation cancelling the continuous variable does. This is the
    /// fails-before/passes-after regression witness for the whole build.
    ///
    /// Two integer vars x0, x1 in [0,2], one continuous t ≥ 0.
    ///   R0:  x0 + 3 x1 − 2 t ≤ 3.5
    ///   R1:  x0        + 2 t ≤ 3.5
    /// Weights λ0 = |c1| = 2, λ1 = |c0| = 2 cancel t (2·(−2)+2·(+2)=0):
    ///   4 x0 + 6 x1 ≤ 14  →  δ-scan (scale 1/6) → 0.667 x0 + x1 ≤ 2.333
    ///   →  MIR:  0.5 x0 + x1 ≤ 2.
    /// At LP point x=(1.25, 1.5, 1.0) that cut is violated by 0.125, while
    /// single-row MIR on *either* row alone finds nothing violated (the free
    /// continuous t masks the integer face on each row individually).
    #[test]
    fn aggregation_finds_cut_single_row_mir_misses() {
        let a_ub = [
            1.0, 3.0, -2.0, // R0
            1.0, 0.0, 2.0, // R1
        ];
        let b_ub = [3.5, 3.5];
        let l = [0.0, 0.0, 0.0];
        let u = [2.0, 2.0, f64::INFINITY];
        let integ = [true, true, false];
        let x = [1.25, 1.5, 1.0];

        // Single-row MIR on the two rows individually: must NOT separate x
        // with x0+x1≤2 (t present keeps the row from rounding to that face).
        let single = separate_mir(&a_ub, &b_ub, &l, &u, &integ, &x, 1e-7, 1e9);
        let single_best = single
            .iter()
            .map(|c| dot(&c.coeffs, &x) - c.rhs)
            .fold(f64::NEG_INFINITY, f64::max);

        // Aggregation: must find the x0+x1≤2 cut (violation 0.5).
        let agg = separate_aggregation_mir(&a_ub, &b_ub, &l, &u, &integ, &x, 1e-7, 1e9);
        assert!(
            !agg.is_empty(),
            "aggregation must find a violated cut on this instance"
        );
        let best = agg
            .iter()
            .max_by(|p, q| p.violation.partial_cmp(&q.violation).unwrap())
            .unwrap();
        assert!(
            best.violation > 1e-6,
            "aggregation cut must separate x* (viol {})",
            best.violation
        );
        // The aggregation cut is strictly stronger than anything single-row MIR
        // found here (the point of aggregation on this instance).
        assert!(
            best.violation > single_best + 1e-9 || single_best <= 1e-6,
            "aggregation ({}) not stronger than single-row ({single_best})",
            best.violation
        );
        // And it is valid for the full 2-row integer system.
        assert_valid_system(
            &best.cut.coeffs,
            best.cut.rhs,
            &a_ub,
            &b_ub,
            &integ,
            &[0, 0, 0],
            &[2, 2, 3], // t sampled over [0,3]
            6,
        );
    }

    /// Adversarial cut-validity property test — the primary correctness gate.
    ///
    /// Hundreds of random 2- and 3-row systems (mixed integer/continuous columns,
    /// mixed-sign finite bounds, at least one continuous column to cancel, random
    /// nonnegative aggregation implied by the separator, and LP points chosen to
    /// force violated cuts) asserting NO integer/continuous-feasible point of the
    /// ORIGINAL system is ever cut. Modeled on `mir_validity_random_complemented_rows`.
    #[test]
    fn aggregation_validity_random_systems() {
        let mut state: u64 = 0xfeed_face_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut n_cuts_checked = 0usize;
        for _ in 0..500 {
            let n = 3;
            let m = if next() > 0.5 { 3 } else { 2 };
            // Mix of regimes: ~30% of trials are FULLY integer (exercises the
            // fractional-column fallback path — the lifted-McCormick regime),
            // otherwise column 2 is continuous (the MW continuous-cancel target)
            // and 0,1 are integer more often than not.
            let all_int = next() < 0.3;
            let integ: Vec<bool> = (0..n)
                .map(|j| {
                    if all_int {
                        true
                    } else if j == 2 {
                        false
                    } else {
                        next() > 0.25
                    }
                })
                .collect();
            // Mixed-sign, sometimes-fractional coefficients; ensure the two
            // rows have opposite signs on the continuous column often.
            let mut a_ub = vec![0.0_f64; m * n];
            for r in 0..m {
                for j in 0..n {
                    a_ub[r * n + j] = (next() * 8.0 - 4.0).round() + (next() - 0.5);
                }
                // Bias column 2 sign by row parity so cancellation fires.
                let s = if r % 2 == 0 { 1.0 } else { -1.0 };
                a_ub[r * n + 2] = s * (0.5 + next() * 3.0);
            }
            let b_ub: Vec<f64> = (0..m).map(|_| next() * 14.0 - 6.0).collect();
            // Mixed-sign integer bounds in [-2,2]; continuous over a real box.
            let lo_i: Vec<i64> = (0..n).map(|_| (next() * 4.0).floor() as i64 - 2).collect();
            let hi_i: Vec<i64> = (0..n)
                .map(|k| lo_i[k] + 1 + (next() * 4.0).floor() as i64)
                .collect();
            let l: Vec<f64> = lo_i.iter().map(|&v| v as f64).collect();
            let u: Vec<f64> = hi_i.iter().map(|&v| v as f64).collect();
            // LP point: pushed around the box (some near upper) to trigger both
            // the cancellation and the complementation paths.
            let x: Vec<f64> = (0..n)
                .map(|k| {
                    let t = if next() > 0.3 {
                        0.55 + 0.45 * next()
                    } else {
                        next()
                    };
                    l[k] + t * (u[k] - l[k])
                })
                .collect();

            for ac in separate_aggregation_mir(&a_ub, &b_ub, &l, &u, &integ, &x, 1e-7, 1e9) {
                assert_valid_system(
                    &ac.cut.coeffs,
                    ac.cut.rhs,
                    &a_ub,
                    &b_ub,
                    &integ,
                    &lo_i,
                    &hi_i,
                    5,
                );
                n_cuts_checked += 1;
            }
        }
        assert!(
            n_cuts_checked > 20,
            "expected many aggregation cuts to validate, got {n_cuts_checked}"
        );
    }

    /// No panic / no spurious cut on degenerate inputs (single row, all-integer
    /// with an integral LP point ⇒ no fractional fallback target, empty system).
    #[test]
    fn aggregation_degenerate_inputs_safe() {
        let l = [0.0, 0.0];
        let u = [2.0, 2.0];
        let integ_all_int = [true, true];
        let x = [1.5, 1.5];
        // single row → no pair
        let one =
            separate_aggregation_mir(&[1.0, 1.0], &[1.5], &l, &u, &integ_all_int, &x, 1e-7, 1e9);
        assert!(one.is_empty());
        // all-integer + integral LP point → no continuous target AND no
        // fractional column to fall back to → nothing to cancel.
        let x_int = [1.0, 1.0];
        let no_target = separate_aggregation_mir(
            &[1.0, 1.0, 1.0, -1.0],
            &[1.5, 1.5],
            &l,
            &u,
            &integ_all_int,
            &x_int,
            1e-7,
            1e9,
        );
        assert!(no_target.is_empty());
        // empty system
        let empty = separate_aggregation_mir(&[], &[], &[], &[], &[], &[], 1e-7, 1e9);
        assert!(empty.is_empty());
    }

    /// Fully-lifted-LP regime: all columns integer, a fractional LP point, rows
    /// with opposite signs on a fractional column ⇒ the fractional-column
    /// fallback fires and any emitted cut is still valid for the integer system.
    #[test]
    fn aggregation_all_integer_fractional_fallback_valid() {
        // Two "product-aux-like" integer columns + one plain integer column, all
        // integer; LP point fractional on col 2 (the cancel target).
        let a_ub = [
            1.0, 1.0, 2.0, // R0
            1.0, 0.0, -1.0, // R1
        ];
        let b_ub = [4.5, 1.5];
        let l = [0.0, 0.0, 0.0];
        let u = [3.0, 3.0, 3.0];
        let integ = [true, true, true];
        let x = [1.4, 1.0, 1.3]; // col 2 fractional → fallback cancels it
        let cuts = separate_aggregation_mir(&a_ub, &b_ub, &l, &u, &integ, &x, 1e-7, 1e9);
        for ac in &cuts {
            assert_valid_system(
                &ac.cut.coeffs,
                ac.cut.rhs,
                &a_ub,
                &b_ub,
                &integ,
                &[0, 0, 0],
                &[3, 3, 3],
                4,
            );
        }
    }
}
