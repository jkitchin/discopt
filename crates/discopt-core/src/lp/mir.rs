//! Mixed-integer rounding (MIR) cuts from original constraint rows.
//!
//! These complement the Gomory cuts in [`super::gomory`]: GMI applies the MIR
//! function to *tableau* rows (needs the basis), whereas this separates MIR
//! directly from the model's `≤` rows — basis-free, so it fires on constraint
//! structure the tableau does not expose.
//!
//! For a single row `Σ_j a_j x_j ≤ b` we first reformulate every variable to a
//! nonnegative variable `y_j ≥ 0` by *bound substitution*: per column we choose
//! either the **lower** bound (`y_j = x_j − l_j`, coefficient `a_j`) or, when the
//! variable is pressed near its **upper** bound, the **upper** complement
//! (`y_j = u_j − x_j`, coefficient `−a_j`, with `a_j u_j` folded into the rhs).
//! Both keep the row in the form `Σ ã_j y_j ≤ b̃` with `y_j ≥ 0`, and an integer
//! `x_j` stays integer under complementation only when `u_j` is integral — so we
//! complement an integer column only then. Complementation is the standard
//! Marchand–Wolsey strengthening: it changes *which* facet the MIR function
//! produces, and on rows whose LP point sits at an upper bound the complemented
//! cut is the violated one.
//!
//! With `f = b̃' − ⌊b̃'⌋ ∈ (0,1)` the MIR inequality on the reformulated row is
//! `Σ_j γ_j y_j ≤ ⌊b̃'⌋` with, for `f_j = ã_j − ⌊ã_j⌋`:
//!
//! - integer `j`: `γ_j = ⌊ã_j⌋ + max(0, f_j − f)/(1 − f)`;
//! - continuous `j`: `γ_j = min(ã_j, 0)/(1 − f)`.
//!
//! The integer branch is valid only when the shifted `y_j` is integer-valued,
//! which requires the *active substitution bound* to be integral (`l_j` for a
//! lower shift, `u_j` for an upper complement). Presolve can leave an integer
//! column with a fractional bound; there the integer branch would cut feasible
//! points, so such a column falls back to the continuous coefficient (C-4) —
//! the same premise the GMI separator guards in `super::gomory`.
//!
//! Single-row MIR is only as strong as the row's scaling, so each row is tried
//! at several scalings `δ` (1, and `1/|a_j|` for each integer column) and — for
//! both the all-lower substitution and the near-bound (per-column upper-vs-lower)
//! substitution — the most-violated valid cut is kept. The result is mapped back
//! to the original `x`. Every cut is valid for the row's integer hull regardless
//! of the scaling, the complementation choice, or the (interior) point it was
//! separated at, so soundness is independent of the relaxation solver.

/// A separated MIR cut `coeffs · x ≤ rhs` over the structural variables.
pub struct MirCut {
    /// Dense length-`n` coefficient vector.
    pub coeffs: Vec<f64>,
    /// Right-hand side (the cut is `coeffs · x ≤ rhs`).
    pub rhs: f64,
}

/// Minimum fractionality of the scaled rhs to cut on (avoids tiny `f`/`1−f`
/// denominators blowing the coefficients up).
const FRAC_MIN: f64 = 1e-3;
/// Absolute cap on a cut coefficient; larger ones are numerically unsafe.
const MAX_ABS_COEFF: f64 = 1e7;

/// Apply the MIR function to one (scaled, bound-shifted) row, returning the
/// cut over the shifted variables `y` as `(coeffs', rhs')` or `None`.
///
/// The integer-MIR rounding is only valid when the shifted variable `y_j` is a
/// nonnegative *integer*, which requires the substitution bound to be integral.
/// `int_shift[j]` records that premise (integer column AND integral active
/// bound); when it is false the column takes the always-valid continuous
/// coefficient instead — mirroring the `use_integer` guard in `gomory.rs`.
fn mir_row(a: &[f64], b: f64, int_shift: &[bool]) -> Option<(Vec<f64>, f64)> {
    let f = b - b.floor();
    if !(FRAC_MIN..=1.0 - FRAC_MIN).contains(&f) {
        return None;
    }
    let n = a.len();
    let mut g = vec![0.0_f64; n];
    for j in 0..n {
        if int_shift[j] {
            let fj = a[j] - a[j].floor();
            g[j] = a[j].floor() + (fj - f).max(0.0) / (1.0 - f);
        } else {
            g[j] = a[j].min(0.0) / (1.0 - f);
        }
    }
    Some((g, b.floor()))
}

/// Tolerance within which the active substitution bound of an integer column
/// must sit to an integer for the shifted variable to remain integer-valued:
/// the lower bound `l_j` for `y_j = x_j − l_j`, or the upper bound `u_j` for the
/// complement `y_j = u_j − x_j`. When the bound is fractional the integer-MIR
/// rounding is unsound and the column falls back to the continuous coefficient.
const INT_BOUND_TOL: f64 = 1e-6;

/// Whether the LP point is within `frac` of the way up the `[l, u]` box, i.e.
/// close enough to the upper bound that complementing there is worth trying.
fn near_upper(xj: f64, lj: f64, uj: f64) -> bool {
    if !uj.is_finite() || uj <= lj {
        return false;
    }
    // Strictly above the box midpoint ⇒ nearer the upper bound.
    xj - lj > 0.5 * (uj - lj)
}

/// Apply single-row MIR to `row` under a fixed per-column bound-substitution
/// `comp` (`comp[j]` = complement column `j` at its upper bound), at scaling
/// `d`, and return the resulting cut over the *original* `x` together with its
/// violation at `x`, or `None` if no valid, sufficiently violated cut results.
///
/// # Soundness
/// Under `y_j = u_j − x_j` (for complemented `j`) and `y_j = x_j − l_j`
/// (otherwise), every `y_j ≥ 0`, and the row becomes `Σ ã_j y_j ≤ b̃` with
/// `ã_j = −a_j` (complemented) or `a_j`, and `b̃ = b − Σ_{comp} a_j u_j −
/// Σ_{not comp} a_j l_j`. MIR is valid for this nonnegative row, and mapping the
/// cut `Σ γ_j y_j ≤ r` back through the (affine, invertible) substitution yields
/// a valid inequality in `x`. Any integer-feasible `x` maps to a feasible `y`,
/// but `y_j` is *integer-valued* only when the active substitution bound is
/// integral (`l_j` for a lower shift, `u_j` for a complement). The integer-MIR
/// rounding is therefore requested (`int_shift[j]`) only for integer columns
/// whose active bound is integral; a fractional bound falls back to the
/// continuous coefficient so the cut removes no integer-feasible point (C-4).
#[allow(clippy::too_many_arguments)]
fn mir_under_substitution(
    row: &[f64],
    b: f64,
    l: &[f64],
    u: &[f64],
    integrality: &[bool],
    comp: &[bool],
    x: &[f64],
    d: f64,
    tol: f64,
    max_dynamism: f64,
) -> Option<(Vec<f64>, f64, f64)> {
    let n = row.len();
    // Reformulated coefficients ã_j and the shift folded into the rhs. Also record
    // per column whether the integer-MIR branch is admissible: the shifted
    // `y_j = x_j − l_j` (or `u_j − x_j` when complemented) stays integer-valued
    // only when the *active* substitution bound is itself integral. Presolve
    // (coefficient strengthening / implied bounds) can leave an integer column
    // with a fractional bound; applying the integer rounding there yields a cut
    // that can exclude a feasible integer point (C-4). Mirrors the `use_integer`
    // guard in `gomory.rs`.
    let mut a_ref = vec![0.0_f64; n];
    let mut int_shift = vec![false; n];
    let mut b_ref = b;
    for j in 0..n {
        let pinned = if comp[j] {
            a_ref[j] = -row[j];
            b_ref -= row[j] * u[j];
            u[j]
        } else {
            a_ref[j] = row[j];
            b_ref -= row[j] * l[j];
            l[j]
        };
        int_shift[j] = integrality[j] && (pinned - pinned.round()).abs() <= INT_BOUND_TOL;
    }
    // Scale, then apply the MIR function in the nonnegative y-space.
    let a_scaled: Vec<f64> = a_ref.iter().map(|&aj| aj * d).collect();
    let (g, r) = mir_row(&a_scaled, b_ref * d, &int_shift)?;

    // Point value of y_j and violation of `Σ g_j y_j ≤ r` at x.
    let yj = |j: usize| if comp[j] { u[j] - x[j] } else { x[j] - l[j] };
    let viol: f64 = (0..n).map(|j| g[j] * yj(j)).sum::<f64>() - r;
    if viol <= tol {
        return None;
    }

    // Map the cut `Σ g_j y_j ≤ r` back to the original x. For complemented j,
    // g_j y_j = g_j(u_j − x_j) = −g_j x_j + g_j u_j, so the coefficient on x_j is
    // −g_j and the constant g_j u_j moves to the rhs as −g_j u_j; for a lower
    // column g_j y_j = g_j x_j − g_j l_j, so coefficient g_j and rhs += g_j l_j.
    let mut coeffs = vec![0.0_f64; n];
    let mut rhs = r;
    for j in 0..n {
        if comp[j] {
            coeffs[j] = -g[j];
            rhs -= g[j] * u[j];
        } else {
            coeffs[j] = g[j];
            rhs += g[j] * l[j];
        }
    }

    let max_c = coeffs.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let min_c = coeffs
        .iter()
        .filter(|&&v| v.abs() > tol)
        .fold(f64::INFINITY, |a, &v| a.min(v.abs()));
    if max_c == 0.0
        || max_c > MAX_ABS_COEFF
        || rhs.abs() > MAX_ABS_COEFF
        || (min_c.is_finite() && max_c / min_c > max_dynamism)
        || !coeffs.iter().all(|v| v.is_finite())
        || !rhs.is_finite()
    {
        return None;
    }
    Some((coeffs, rhs, viol))
}

/// Separate MIR cuts from the `≤` rows `a_ub · x ≤ b_ub` at point `x`.
///
/// `a_ub` is row-major `m × n`; `l`/`u` are the variable lower/upper bounds
/// (used to reformulate each column to a nonnegative variable via lower shift or
/// upper complement — see the module docs); `integrality[j]` marks integer
/// columns. Returns the most violated valid MIR cut per row (over the original
/// `x`), subject to a violation threshold and numerical guards. Both the
/// all-lower substitution and the per-column near-bound substitution are tried at
/// every candidate scaling, and the strongest valid cut is kept.
#[allow(clippy::too_many_arguments)]
pub fn separate_mir(
    a_ub: &[f64],
    b_ub: &[f64],
    l: &[f64],
    u: &[f64],
    integrality: &[bool],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
) -> Vec<MirCut> {
    let m = b_ub.len();
    let n = a_ub.len().checked_div(m).unwrap_or(0);
    let mut cuts = Vec::new();
    if n == 0 {
        return cuts;
    }

    for i in 0..m {
        let row = &a_ub[i * n..(i + 1) * n];
        let b = b_ub[i];

        // Candidate scalings: 1, then 1/|a_j| for each integer column.
        let mut deltas = vec![1.0_f64];
        for (j, &aj) in row.iter().enumerate() {
            if integrality[j] && aj.abs() > tol {
                deltas.push(1.0 / aj.abs());
            }
        }

        // All-lower substitution (the original behaviour).
        let comp_lower = vec![false; n];
        // Near-bound substitution: complement columns whose LP point sits near a
        // finite upper bound. Integer columns are complemented only when their
        // upper bound is integral, so the complemented variable stays integer.
        let mut comp_near = vec![false; n];
        for j in 0..n {
            if row[j].abs() <= tol {
                continue;
            }
            if !near_upper(x[j], l[j], u[j]) {
                continue;
            }
            if integrality[j] && (u[j] - u[j].round()).abs() > INT_BOUND_TOL {
                continue;
            }
            comp_near[j] = true;
        }
        let mut patterns: Vec<&[bool]> = vec![&comp_lower];
        if comp_near.iter().any(|&c| c) {
            patterns.push(&comp_near);
        }

        let mut best: Option<(Vec<f64>, f64, f64)> = None; // (coeffs, rhs, violation)
        for &comp in &patterns {
            for &d in &deltas {
                if let Some((coeffs, rhs, viol)) =
                    mir_under_substitution(row, b, l, u, integrality, comp, x, d, tol, max_dynamism)
                {
                    if best.as_ref().map(|bst| viol > bst.2).unwrap_or(true) {
                        best = Some((coeffs, rhs, viol));
                    }
                }
            }
        }
        if let Some((coeffs, rhs, _)) = best {
            cuts.push(MirCut { coeffs, rhs });
        }
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Exhaustively check a `≤` cut excludes no integer-feasible point of the
    /// row `Σ a x ≤ b` over integer x in `[lo, hi]`.
    fn assert_valid_le_box(coeffs: &[f64], rhs: f64, a: &[f64], b: f64, lo: &[i64], hi: &[i64]) {
        let n = a.len();
        let mut idx = lo.to_vec();
        loop {
            let xv: Vec<f64> = idx.iter().map(|&v| v as f64).collect();
            if dot(a, &xv) <= b + 1e-9 {
                assert!(
                    dot(coeffs, &xv) <= rhs + 1e-6,
                    "cut {coeffs:?} <= {rhs} excludes feasible {xv:?}"
                );
            }
            // increment mixed-radix over [lo, hi]
            let mut k = 0;
            while k < n {
                idx[k] += 1;
                if idx[k] <= hi[k] {
                    break;
                }
                idx[k] = lo[k];
                k += 1;
            }
            if k == n {
                break;
            }
        }
    }

    /// `assert_valid_le_box` over the `[0, hi]` box (lower bound zero).
    fn assert_valid_le(coeffs: &[f64], rhs: f64, a: &[f64], b: f64, hi: &[i64]) {
        let lo = vec![0i64; a.len()];
        assert_valid_le_box(coeffs, rhs, a, b, &lo, hi);
    }

    #[test]
    fn mir_cut_rounds_a_simple_row() {
        // x0 + x1 <= 1.5, x0,x1 in {0,1} -> MIR gives x0 + x1 <= 1.
        let a = [1.0, 1.0];
        let b = 1.5;
        let l = [0.0, 0.0];
        let integ = [true, true];
        let u = [1.0, 1.0];
        let x = [0.75, 0.75]; // fractional relaxation point, violates x0+x1<=1
        let cuts = separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        np_close(&cuts[0].coeffs, &[1.0, 1.0]);
        assert!((cuts[0].rhs - 1.0).abs() < 1e-9);
        assert!(dot(&cuts[0].coeffs, &x) > cuts[0].rhs + 1e-6); // separates x*
        assert_valid_le(&cuts[0].coeffs, cuts[0].rhs, &a, b, &[1, 1]);
    }

    #[test]
    fn mir_cut_uses_scaling_on_fractional_coeffs() {
        // 2 x0 + 2 x1 <= 3, x0,x1 in {0,1,2}. Scaling by 1/2 gives
        // x0 + x1 <= 1.5 -> MIR x0 + x1 <= 1, which the row alone (delta=1,
        // b=3 integral) would miss.
        let a = [2.0, 2.0];
        let b = 3.0;
        let l = [0.0, 0.0];
        let integ = [true, true];
        let u = [2.0, 2.0];
        let x = [0.75, 0.75];
        let cuts = separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        assert!(dot(&cuts[0].coeffs, &x) > cuts[0].rhs + 1e-6);
        assert_valid_le(&cuts[0].coeffs, cuts[0].rhs, &a, b, &[2, 2]);
    }

    #[test]
    fn mir_validity_random_rows() {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for _ in 0..200 {
            let n = 3;
            let a: Vec<f64> = (0..n)
                .map(|_| (next() * 6.0 - 3.0).round() + (next() - 0.5))
                .collect();
            let hi = [2i64, 2, 2];
            // rhs that is usually fractional
            let b = next() * 10.0 - 2.0;
            let l = [0.0; 3];
            let u = [2.0; 3];
            let integ = [true, true, true];
            let x: Vec<f64> = (0..n).map(|_| next() * 2.0).collect();
            for cut in separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9) {
                assert_valid_le(&cut.coeffs, cut.rhs, &a, b, &hi);
            }
        }
    }

    /// Strong random-validity property test that specifically exercises the
    /// upper-bound complementation path: mixed-sign nonzero *lower* bounds,
    /// per-column *finite upper* bounds, LP points pushed near the upper bound
    /// (so complementation actually fires), and a mix of integer/continuous
    /// columns. Every emitted cut must exclude no integer-feasible point of the
    /// row over the full `[l, u]` integer box.
    #[test]
    fn mir_validity_random_complemented_rows() {
        let mut state: u64 = 0x0bad_c0de_dead_beef;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut n_cuts_checked = 0usize;
        for _ in 0..600 {
            let n = 3;
            // Mixed-sign, sometimes-fractional coefficients.
            let a: Vec<f64> = (0..n)
                .map(|_| (next() * 8.0 - 4.0).round() + (next() - 0.5))
                .collect();
            // Mixed-sign integer lower bounds in [-2, 1]; width 1..=4 above.
            let lo_i: Vec<i64> = (0..n).map(|_| (next() * 4.0).floor() as i64 - 2).collect();
            let hi_i: Vec<i64> = (0..n)
                .map(|k| lo_i[k] + 1 + (next() * 4.0).floor() as i64)
                .collect();
            let l: Vec<f64> = lo_i.iter().map(|&v| v as f64).collect();
            let u: Vec<f64> = hi_i.iter().map(|&v| v as f64).collect();
            // Some columns continuous (integrality off).
            let integ: Vec<bool> = (0..n).map(|_| next() > 0.35).collect();
            let b = next() * 14.0 - 6.0;
            // LP point pushed toward the upper bound for most columns, so the
            // near-bound complementation pattern is selected.
            let x: Vec<f64> = (0..n)
                .map(|k| {
                    let t = if next() > 0.25 {
                        0.6 + 0.4 * next() // near upper
                    } else {
                        next()
                    };
                    l[k] + t * (u[k] - l[k])
                })
                .collect();
            for cut in separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9) {
                assert_valid_le_box(&cut.coeffs, cut.rhs, &a, b, &lo_i, &hi_i);
                n_cuts_checked += 1;
            }
        }
        // Ensure the path is actually exercised (not vacuously passing).
        assert!(
            n_cuts_checked > 20,
            "expected many cuts to validate, got {n_cuts_checked}"
        );
    }

    /// The complemented cut must be at least as strong as (dominate or tie) the
    /// non-complemented cut on a row whose LP point sits at an upper bound — the
    /// exact case complementation is FOR. Constructed so all-lower substitution
    /// yields no violated cut while upper complementation does. This fails if the
    /// complementation path is removed (only the weaker/empty cut would remain).
    #[test]
    fn complemented_cut_dominates_at_upper_bound() {
        // Row: -3 x0 + x1 <= -4.5, x0,x1 integer in [0,3], LP point near the
        // upper bound at x = (2.7, 2.7). The all-lower MIR substitution finds no
        // violated cut here (δ-scan exhausted), but complementing both columns
        // at their upper bound yields the valid cut -x0 + (2/3) x1 <= -1, which
        // the LP point violates by 0.1. This is exactly the regime upper-bound
        // complementation is FOR; it fails if the complementation path is removed
        // (only the empty/weaker all-lower result would remain).
        let a = [-3.0, 1.0];
        let b = -4.5;
        let l = [0.0, 0.0];
        let u = [3.0, 3.0];
        let integ = [true, true];
        let x = [2.7, 2.7];

        // With complementation enabled (the shipped behaviour).
        let cuts = separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1, "complementation should yield a cut");
        let c = &cuts[0];
        assert_valid_le(&c.coeffs, c.rhs, &a, b, &[3, 3]);
        let comp_viol = dot(&c.coeffs, &x) - c.rhs;
        assert!(
            comp_viol > 1e-6,
            "complemented cut {:?} <= {} must separate x* {x:?} (viol {comp_viol})",
            c.coeffs,
            c.rhs
        );

        // Baseline: force the all-lower substitution only (no complementation),
        // by claiming there is no finite upper bound. This is the pre-change
        // behaviour; it must NOT separate x* here — so the complemented cut is
        // strictly stronger (dominates) on this row.
        let no_ub = [f64::INFINITY, f64::INFINITY];
        let base = separate_mir(&a, &[b], &l, &no_ub, &integ, &x, 1e-7, 1e9);
        let base_viol = base
            .iter()
            .map(|k| dot(&k.coeffs, &x) - k.rhs)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            base.is_empty() || base_viol <= 1e-6,
            "non-complemented MIR unexpectedly separated x* (viol {base_viol}); \
             test no longer isolates complementation"
        );
        // The complemented cut is at least as violated as anything all-lower found.
        assert!(
            comp_viol >= base_viol - 1e-9,
            "complemented cut ({comp_viol}) weaker than all-lower ({base_viol})"
        );
    }

    /// C-4 property test: random rows with integer columns carrying *fractional*
    /// lower and/or upper bounds (the state presolve coefficient-strengthening can
    /// produce). Every emitted cut must exclude no integer-feasible point of the
    /// row over the integer box. This locks the class: if the fractional-bound
    /// guard is removed, the buggy integer rounding cuts a feasible integer point
    /// on some draw and the assertion trips. Fails before the fix.
    #[test]
    fn c4_mir_validity_random_fractional_int_bounds() {
        let mut state: u64 = 0xc4c4_dead_beef_0004;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut n_cuts_checked = 0usize;
        for _ in 0..1000 {
            let n = 3;
            let a: Vec<f64> = (0..n)
                .map(|_| (next() * 8.0 - 4.0).round() + (next() - 0.5))
                .collect();
            // Fractional offsets in (0,1) added to integer endpoints, so the
            // active substitution bounds are often non-integral for integer cols.
            let frac_l: Vec<f64> = (0..n)
                .map(|_| if next() > 0.4 { next() } else { 0.0 })
                .collect();
            let frac_u: Vec<f64> = (0..n)
                .map(|_| if next() > 0.4 { next() } else { 0.0 })
                .collect();
            let lo_i: Vec<i64> = (0..n).map(|_| (next() * 4.0).floor() as i64 - 2).collect();
            let hi_i: Vec<i64> = (0..n)
                .map(|k| lo_i[k] + 1 + (next() * 4.0).floor() as i64)
                .collect();
            let l: Vec<f64> = (0..n).map(|k| lo_i[k] as f64 + frac_l[k]).collect();
            let u: Vec<f64> = (0..n).map(|k| hi_i[k] as f64 + frac_u[k]).collect();
            let integ: Vec<bool> = (0..n).map(|_| next() > 0.3).collect();
            let b = next() * 14.0 - 6.0;
            let x: Vec<f64> = (0..n)
                .map(|k| {
                    let t = if next() > 0.25 {
                        0.6 + 0.4 * next()
                    } else {
                        next()
                    };
                    l[k] + t * (u[k] - l[k]).max(0.0)
                })
                .collect();
            // Integer-feasible box: integers within [l, u] → [ceil(l), floor(u)].
            let lo_feas: Vec<i64> = (0..n).map(|k| l[k].ceil() as i64).collect();
            let hi_feas: Vec<i64> = (0..n).map(|k| u[k].floor() as i64).collect();
            if (0..n).any(|k| lo_feas[k] > hi_feas[k]) {
                continue;
            }
            for cut in separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9) {
                assert_valid_le_box(&cut.coeffs, cut.rhs, &a, b, &lo_feas, &hi_feas);
                n_cuts_checked += 1;
            }
        }
        assert!(
            n_cuts_checked > 20,
            "expected many cuts to validate, got {n_cuts_checked}"
        );
    }

    /// C-4 regression: an integer column with a *fractional* lower bound (as
    /// presolve coefficient-strengthening / implied bounds can produce) must NOT
    /// receive the integer-MIR rounding, because the bound substitution
    /// `y_j = x_j − l_j` is non-integer there and the integer γ can cut a feasible
    /// integer point. Mirrors `gomory.rs`'s
    /// `gmi_cut_valid_when_integer_var_has_fractional_bound`.
    ///
    /// Row `−x0 − 2·x1 ≤ −2`, x0 integer in [0.5, 4] (fractional lower bound),
    /// x1 integer in [0, 2]. Feasible integer points include (2, 0)
    /// (`−2 − 0 = −2 ≤ −2`). At the LP point (0.8, 0.1) the *buggy* integer-MIR
    /// cut is `−x0 − 2·x1 ≤ −2.5`, which excludes (2, 0) (`−2 > −2.5`). With the
    /// fractional-lower-bound guard, column 0 falls back to the continuous
    /// coefficient and every emitted cut stays valid for the whole integer box.
    #[test]
    fn c4_mir_valid_when_integer_var_has_fractional_lower_bound() {
        let a = [-1.0, -2.0];
        let b = -2.0;
        let l = [0.5, 0.0]; // x0 has a fractional lower bound
        let u = [4.0, 2.0];
        let integ = [true, true];
        let x = [0.8, 0.1]; // LP point that selects the (buggy) integer-MIR cut

        let cuts = separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9);
        // Every emitted cut must exclude no integer-feasible point of the row over
        // the integer box x0 ∈ {1,2,3,4} (integer ≥ 0.5), x1 ∈ {0,1,2}.
        for cut in &cuts {
            assert_valid_le_box(&cut.coeffs, cut.rhs, &a, b, &[1, 0], &[4, 2]);
        }
    }

    #[test]
    fn mir_handles_continuous_columns() {
        // x0 (int) + y (cont >=0) <= 1.5 -> MIR drops the positive-coeff
        // continuous y: x0 <= 1.
        let a = [1.0, 1.0];
        let b = 1.5;
        let l = [0.0, 0.0];
        let integ = [true, false];
        let u = [2.0, f64::INFINITY]; // y continuous with no finite upper bound
        let x = [1.4, 0.0];
        let cuts = separate_mir(&a, &[b], &l, &u, &integ, &x, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        assert!((cuts[0].coeffs[1]).abs() < 1e-9); // continuous y dropped
        assert!((cuts[0].coeffs[0] - 1.0).abs() < 1e-9 && (cuts[0].rhs - 1.0).abs() < 1e-9);
    }

    fn np_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-9, "{a:?} != {b:?}");
        }
    }
}
