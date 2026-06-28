//! Cut selection / management: efficacy + orthogonality filtering.
//!
//! On sparse-row MILPs, separating cuts closes the integrality gap to
//! SCIP-competitive node counts — but *carrying* every separated cut at every
//! B&B node makes each node's LP re-solve expensive and erases the win in wall
//! time (lifted cover/GMI cuts are dense). SCIP keeps a small, diverse, high-impact
//! active set instead. This module selects that set from the candidate cuts:
//!
//! * **efficacy** — the Euclidean distance the cut moves the current fractional
//!   point, `violation / ‖coeffs‖₂`. A cut that barely separates `x*` (or does
//!   not separate it) is dropped.
//! * **orthogonality** — greedily keep the most-efficacious cuts but skip any
//!   whose direction is nearly parallel (`|cos| > max_parallel`) to an already
//!   kept cut, so the set spans diverse faces rather than piling near-duplicates.
//!
//! Selection only *chooses among* already-valid cuts and never modifies one, so
//! it cannot affect soundness — only which valid inequalities enter the LP, and
//! thus the bound-per-row and the per-node cost.

use crate::lp::gomory::GomoryCut;

fn norm2(c: &[f64]) -> f64 {
    c.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Efficacy of `coeffs · x ≥ rhs` at `x`: violation `rhs − coeffs·x` normalized
/// by `‖coeffs‖₂`. Positive ⇔ the cut separates `x`.
fn efficacy(cut: &GomoryCut, x: &[f64]) -> f64 {
    let nrm = norm2(&cut.coeffs);
    if nrm <= 0.0 {
        return 0.0;
    }
    let n = cut.coeffs.len().min(x.len());
    let act: f64 = (0..n).map(|j| cut.coeffs[j] * x[j]).sum();
    (cut.rhs - act) / nrm
}

/// Absolute cosine between two coefficient vectors (1 = parallel, 0 = orthogonal).
fn parallelism(a: &[f64], b: &[f64]) -> f64 {
    let (na, nb) = (norm2(a), norm2(b));
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    let n = a.len().min(b.len());
    let dot: f64 = (0..n).map(|j| a[j] * b[j]).sum();
    (dot / (na * nb)).abs()
}

/// Greedily select up to `max_keep` cuts from `cuts` evaluated at `x`: discard
/// cuts with efficacy below `min_efficacy`, then take most-efficacious-first,
/// skipping any more than `max_parallel` parallel to an already-selected cut.
/// Deterministic (stable sort), so the result is independent of thread order.
pub fn select_cuts(
    cuts: Vec<GomoryCut>,
    x: &[f64],
    max_keep: usize,
    min_efficacy: f64,
    max_parallel: f64,
) -> Vec<GomoryCut> {
    if max_keep == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(f64, GomoryCut)> = cuts
        .into_iter()
        .map(|c| (efficacy(&c, x), c))
        .filter(|(e, _)| *e >= min_efficacy)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<GomoryCut> = Vec::new();
    for (_, c) in scored {
        if kept.len() >= max_keep {
            break;
        }
        if kept
            .iter()
            .any(|k| parallelism(&k.coeffs, &c.coeffs) > max_parallel)
        {
            continue;
        }
        kept.push(c);
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(coeffs: Vec<f64>, rhs: f64) -> GomoryCut {
        GomoryCut { coeffs, rhs }
    }

    #[test]
    fn drops_non_separating_cuts() {
        let x = [0.5, 0.5];
        // coeffs·x = 1.0, rhs 1.0 ⇒ not violated → dropped.
        let kept = select_cuts(vec![cut(vec![1.0, 1.0], 1.0)], &x, 10, 1e-6, 0.99);
        assert!(kept.is_empty());
    }

    #[test]
    fn keeps_most_efficacious_first() {
        let x = [0.6, 0.6];
        let weak = cut(vec![1.0, 1.0], 1.3); // viol 0.1, norm √2
        let strong = cut(vec![1.0, 0.0], 0.9); // viol 0.3, norm 1
        let kept = select_cuts(vec![weak, strong], &x, 1, 1e-6, 0.99);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].coeffs[1]).abs() < 1e-12); // the [1,0] cut
    }

    #[test]
    fn orthogonality_filter_skips_parallel_duplicates() {
        let x = [0.6, 0.6];
        let a = cut(vec![1.0, 1.0], 1.5);
        let a2 = cut(vec![2.0, 2.0], 3.0); // parallel to a
        let b = cut(vec![1.0, -1.0], 0.1); // orthogonal
        let kept = select_cuts(vec![a, a2, b], &x, 10, 1e-9, 0.99);
        assert_eq!(kept.len(), 2);
        for i in 0..kept.len() {
            for j in (i + 1)..kept.len() {
                assert!(parallelism(&kept[i].coeffs, &kept[j].coeffs) <= 0.99);
            }
        }
    }

    #[test]
    fn respects_max_keep() {
        let x = [0.9, 0.9, 0.9];
        let cuts = vec![
            cut(vec![1.0, 0.0, 0.0], 1.0),
            cut(vec![0.0, 1.0, 0.0], 1.0),
            cut(vec![0.0, 0.0, 1.0], 1.0),
        ];
        assert_eq!(select_cuts(cuts, &x, 2, 1e-9, 0.99).len(), 2);
    }
}
