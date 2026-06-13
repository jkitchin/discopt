//! Knapsack **cover cut** separation.
//!
//! For a `≤` row `Σ wⱼ xⱼ ≤ cap` over binary `x` with `wⱼ ≥ 0`, a *cover* `C`
//! (a set with `Σ_{C} wⱼ > cap`) cannot be fully selected, so the cover
//! inequality `Σ_{C} xⱼ ≤ |C| − 1` is valid. At a fractional point `x*` we use
//! the Crowder–Johnson–Padberg greedy heuristic: build a cover from the
//! most-"in" variables (largest `x*ⱼ` first), reduce it to a *minimal* cover
//! (removing any element only raises the violation `Σ_C x* − (|C|−1)`), and emit
//! it when violated. Cover cuts are sparse and combinatorially strong on
//! knapsack structure — where Gomory cuts are weak — and add little to the LP.
//!
//! Cuts are returned in the `coeffs · x ≥ rhs` form (`coeffs = −1` on `C`,
//! `rhs = −(|C|−1)`) so the driver's existing surplus-row augmentation
//! (`coeffs·x − s = rhs`, `s ≥ 0`) installs `Σ_C xⱼ ≤ |C|−1` directly.

// The slack-detection loop records the slack column index `j`, so a range loop
// reads more clearly than an enumerate over the row slice.
#![allow(clippy::needless_range_loop)]

use crate::lp::crossover::LpView;
use crate::lp::gomory::GomoryCut;

const INF: f64 = 1e19;

/// Separate violated knapsack cover cuts from the first `n_orig_rows` rows of
/// the standard-form LP (the original `≤` constraints; later rows are cuts).
///
/// `x` is the current fractional point (length `lp.n`), `ns` the structural
/// column count, `is_int[j]` the integer mask. A row qualifies as a knapsack
/// when it has a single unit slack `s ∈ [0, ∞)` and its structural entries are
/// binary with nonnegative weights.
pub fn separate_cover(
    lp: &LpView<'_>,
    b: &[f64],
    x: &[f64],
    ns: usize,
    is_int: &[bool],
    n_orig_rows: usize,
    tol: f64,
) -> Vec<GomoryCut> {
    let (a, n) = (lp.a, lp.n);
    let mut cuts = Vec::new();

    for i in 0..n_orig_rows.min(lp.m) {
        let row = &a[i * n..(i + 1) * n];

        // Require exactly one slack column (≥ ns) with coefficient +1 and
        // bounds [0, ∞): i.e. a genuine `≤` row.
        let mut slack: Option<usize> = None;
        let mut ok = true;
        for j in ns..n {
            if row[j].abs() > tol {
                if slack.is_some() || (row[j] - 1.0).abs() > tol {
                    ok = false;
                    break;
                }
                slack = Some(j);
            }
        }
        let s = match (ok, slack) {
            (true, Some(s)) => s,
            _ => continue,
        };
        if lp.l[s].abs() > tol || lp.u[s] < INF {
            continue;
        }

        // Structural items must be binary with nonnegative weights.
        let cap = b[i];
        let mut items: Vec<(usize, f64, f64)> = Vec::new(); // (col, weight, x*)
        let mut knap = true;
        for j in 0..ns {
            let w = row[j];
            if w.abs() <= tol {
                continue;
            }
            if w < -tol || !is_int[j] || lp.l[j] < -tol || lp.u[j] > 1.0 + tol {
                knap = false;
                break;
            }
            items.push((j, w, x[j]));
        }
        if !knap || items.is_empty() {
            continue;
        }

        // Greedy cover: add items by x* descending until the weight exceeds cap.
        items.sort_by(|p, q| q.2.partial_cmp(&p.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut cover: Vec<(usize, f64, f64)> = Vec::new();
        let mut wsum = 0.0;
        let mut xsum = 0.0;
        for &it in &items {
            cover.push(it);
            wsum += it.1;
            xsum += it.2;
            if wsum > cap + tol {
                break;
            }
        }
        if wsum <= cap + tol {
            continue; // never exceeded cap → not a cover
        }

        // Reduce to a minimal cover: drop the largest-weight element that keeps
        // it a cover, repeatedly. Each removal raises the violation (by 1 − x*).
        loop {
            cover.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut removed = false;
            for idx in 0..cover.len() {
                if wsum - cover[idx].1 > cap + tol {
                    wsum -= cover[idx].1;
                    xsum -= cover[idx].2;
                    cover.remove(idx);
                    removed = true;
                    break;
                }
            }
            if !removed {
                break;
            }
        }

        // Violated when Σ_C x* > |C| − 1.
        let rhs_le = cover.len() as f64 - 1.0;
        if xsum > rhs_le + 1e-4 && cover.len() >= 2 {
            let mut coeffs = vec![0.0; n];
            for &(j, _, _) in &cover {
                coeffs[j] = -1.0;
            }
            cuts.push(GomoryCut {
                coeffs,
                rhs: -rhs_le,
            });
        }
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_violated_cover() {
        // 5x0 + 5x1 + 5x2 + s = 9, x binary. LP root x* = (0.6,0.6,0.6) (1.8 units
        // of 0.45 each scaled)… use x* with each 0.6: any two items (5+5=10>9) is a
        // cover, so x0+x1 ≤ 1 etc. The full {0,1,2}: Σw=15>9 cover, minimal cover
        // is any pair. Σx* over a pair = 1.2 > 1 → violated.
        let a = [5.0, 5.0, 5.0, 1.0];
        let c = [0.0; 4];
        let l = [0.0; 4];
        let u = [1.0, 1.0, 1.0, INF + 1.0];
        let lp = LpView { a: &a, m: 1, n: 4, c: &c, l: &l, u: &u };
        let x = [0.6, 0.6, 0.6, 0.0];
        let is_int = [true, true, true, false];
        let cuts = separate_cover(&lp, &[9.0], &x, 3, &is_int, 1, 1e-9);
        assert!(!cuts.is_empty(), "expected a violated cover cut");
        // ≥ form: coeffs = −1 on the cover, rhs = −(|C|−1). A minimal cover here
        // has size 2 → rhs = −1, two −1 coefficients.
        let cut = &cuts[0];
        let nnz: Vec<usize> = (0..3).filter(|&j| cut.coeffs[j].abs() > 1e-9).collect();
        assert_eq!(nnz.len(), 2, "minimal cover should have 2 items");
        assert!((cut.rhs - (-1.0)).abs() < 1e-9, "rhs {}", cut.rhs);
    }

    #[test]
    fn no_cut_when_not_knapsack() {
        // Negative weight → not a cover-cut candidate.
        let a = [-5.0, 5.0, 1.0];
        let c = [0.0; 3];
        let l = [0.0; 3];
        let u = [1.0, 1.0, INF + 1.0];
        let lp = LpView { a: &a, m: 1, n: 3, c: &c, l: &l, u: &u };
        let x = [0.6, 0.6, 0.0];
        let is_int = [true, true, false];
        let cuts = separate_cover(&lp, &[9.0], &x, 2, &is_int, 1, 1e-9);
        assert!(cuts.is_empty());
    }
}
