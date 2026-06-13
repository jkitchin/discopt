//! Mixed-integer rounding (MIR) cuts from original constraint rows.
//!
//! These complement the Gomory cuts in [`super::gomory`]: GMI applies the MIR
//! function to *tableau* rows (needs the basis), whereas this separates MIR
//! directly from the model's `≤` rows — basis-free, so it fires on constraint
//! structure the tableau does not expose.
//!
//! For a single row `Σ_j a_j x_j ≤ b` (shifted to `x'_j = x_j − l_j ≥ 0`) with
//! `f = b' − ⌊b'⌋ ∈ (0,1)`, the MIR inequality is
//! `Σ_j γ_j x'_j ≤ ⌊b'⌋` with, for `f_j = a_j − ⌊a_j⌋`:
//!
//! - integer `j`: `γ_j = ⌊a_j⌋ + max(0, f_j − f)/(1 − f)`;
//! - continuous `j`: `γ_j = min(a_j, 0)/(1 − f)`.
//!
//! Single-row MIR is only as strong as the row's scaling, so each row is tried
//! at several scalings `δ` (1, and `1/|a_j|` for each integer column) and the
//! most-violated valid cut is kept. The result is mapped back to the original
//! `x` (`x' = x − l`). Every cut is valid for the row's integer hull regardless
//! of the scaling or the (interior) point it was separated at, so soundness is
//! independent of the relaxation solver.

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

/// Apply the MIR function to one (scaled, lower-shifted) row, returning the
/// cut over the shifted variables `x'` as `(coeffs', rhs')` or `None`.
fn mir_row(a: &[f64], b: f64, integrality: &[bool]) -> Option<(Vec<f64>, f64)> {
    let f = b - b.floor();
    if !(FRAC_MIN..=1.0 - FRAC_MIN).contains(&f) {
        return None;
    }
    let n = a.len();
    let mut g = vec![0.0_f64; n];
    for j in 0..n {
        if integrality[j] {
            let fj = a[j] - a[j].floor();
            g[j] = a[j].floor() + (fj - f).max(0.0) / (1.0 - f);
        } else {
            g[j] = a[j].min(0.0) / (1.0 - f);
        }
    }
    Some((g, b.floor()))
}

/// Separate MIR cuts from the `≤` rows `a_ub · x ≤ b_ub` at point `x`.
///
/// `a_ub` is row-major `m × n`; `l` are the variable lower bounds (used to
/// shift to `x' ≥ 0`; upper-bound complementation is a future strengthening);
/// `integrality[j]` marks integer columns. Returns the most violated valid MIR
/// cut per row (over the original `x`), subject to a violation threshold and
/// numerical guards.
pub fn separate_mir(
    a_ub: &[f64],
    b_ub: &[f64],
    l: &[f64],
    integrality: &[bool],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
) -> Vec<MirCut> {
    let m = b_ub.len();
    let n = if m > 0 { a_ub.len() / m } else { 0 };
    let mut cuts = Vec::new();
    if n == 0 {
        return cuts;
    }

    for i in 0..m {
        let row = &a_ub[i * n..(i + 1) * n];
        let b = b_ub[i];
        // Shifted rhs for x' = x - l:  Σ a_j (l_j + x'_j) ≤ b.
        let b_shift = b - (0..n).map(|j| row[j] * l[j]).sum::<f64>();

        // Candidate scalings: 1, then 1/|a_j| for each integer column.
        let mut deltas = vec![1.0_f64];
        for (j, &aj) in row.iter().enumerate() {
            if integrality[j] && aj.abs() > tol {
                deltas.push(1.0 / aj.abs());
            }
        }

        let mut best: Option<(Vec<f64>, f64, f64)> = None; // (coeffs, rhs, violation)
        for &d in &deltas {
            let a_scaled: Vec<f64> = row.iter().map(|&aj| aj * d).collect();
            let (g, r) = match mir_row(&a_scaled, b_shift * d, integrality) {
                Some(v) => v,
                None => continue,
            };
            // Violation on x':  Σ g_j (x_j - l_j) - r  (cut is ≤, so >0 ⇒ cut).
            let viol: f64 = (0..n).map(|j| g[j] * (x[j] - l[j])).sum::<f64>() - r;
            if viol <= tol {
                continue;
            }
            // Map back to original x:  Σ g_j x_j ≤ r + Σ g_j l_j.
            let rhs = r + (0..n).map(|j| g[j] * l[j]).sum::<f64>();
            let max_c = g.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            let min_c = g
                .iter()
                .filter(|&&v| v.abs() > tol)
                .fold(f64::INFINITY, |a, &v| a.min(v.abs()));
            if max_c == 0.0
                || max_c > MAX_ABS_COEFF
                || rhs.abs() > MAX_ABS_COEFF
                || (min_c.is_finite() && max_c / min_c > max_dynamism)
                || !g.iter().all(|v| v.is_finite())
                || !rhs.is_finite()
            {
                continue;
            }
            if best.as_ref().map(|bst| viol > bst.2).unwrap_or(true) {
                best = Some((g, rhs, viol));
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
    /// row `Σ a x ≤ b` over integer x in `[0, hi]`.
    fn assert_valid_le(coeffs: &[f64], rhs: f64, a: &[f64], b: f64, hi: &[i64]) {
        let n = a.len();
        let mut idx = vec![0i64; n];
        loop {
            let xv: Vec<f64> = idx.iter().map(|&v| v as f64).collect();
            if dot(a, &xv) <= b + 1e-9 {
                assert!(
                    dot(coeffs, &xv) <= rhs + 1e-6,
                    "cut {coeffs:?} <= {rhs} excludes feasible {xv:?}"
                );
            }
            // increment mixed-radix
            let mut k = 0;
            while k < n {
                idx[k] += 1;
                if idx[k] <= hi[k] {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
            if k == n {
                break;
            }
        }
    }

    #[test]
    fn mir_cut_rounds_a_simple_row() {
        // x0 + x1 <= 1.5, x0,x1 in {0,1} -> MIR gives x0 + x1 <= 1.
        let a = [1.0, 1.0];
        let b = 1.5;
        let l = [0.0, 0.0];
        let integ = [true, true];
        let x = [0.75, 0.75]; // fractional relaxation point, violates x0+x1<=1
        let cuts = separate_mir(&a, &[b], &l, &integ, &x, 1e-7, 1e9);
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
        let x = [0.75, 0.75];
        let cuts = separate_mir(&a, &[b], &l, &integ, &x, 1e-7, 1e9);
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
            let integ = [true, true, true];
            let x: Vec<f64> = (0..n).map(|_| next() * 2.0).collect();
            for cut in separate_mir(&a, &[b], &l, &integ, &x, 1e-7, 1e9) {
                assert_valid_le(&cut.coeffs, cut.rhs, &a, b, &hi);
            }
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
        let x = [1.4, 0.0];
        let cuts = separate_mir(&a, &[b], &l, &integ, &x, 1e-7, 1e9);
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
