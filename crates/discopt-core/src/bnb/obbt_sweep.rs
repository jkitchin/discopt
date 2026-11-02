//! Native in-kernel OBBT (optimization-based bound tightening) probe sweep
//! (issue #764, C1 build-order item 3, MVP).
//!
//! OBBT tightens a variable's bound by *optimizing* it over the node relaxation:
//! `min xₖ` gives a valid tightened lower bound, `max xₖ` a tightened upper bound,
//! subject to the same polytope `A z <= b, l <= z <= u`. On the slow spatial class
//! (e.g. `tanksize`) this dominates the per-node wall — ~110 probes/node — and today
//! it runs as a **Python loop** around the Rust LP binding: measured at 4.25 ms/probe
//! in-loop vs 1.28 ms/probe for the raw Rust solve on the same node LP, i.e. **~70 %
//! of each probe is Python marshaling** (`np.concatenate`×3 + `ascontiguousarray` +
//! the PyO3 crossing, rebuilt every probe). This module runs the whole sweep inside
//! `discopt-core` with no Python between probes, deleting that overhead outright.
//!
//! Scope of this MVP: the loop and the tighten-only bound update, built on the
//! existing warm primal simplex (`solve_lp_cols_warm`), threading the optimal basis
//! probe-to-probe. Every probe is the same LP the Python path solves, via the same
//! trusted solver. Since #1917 each tightened side is that probe's Neumaier--
//! Shcherbina *safe* bound rather than its vertex objective, which can drift past the
//! true optimum and cut feasible points off (see [`probe_safe_bound`]); the safe
//! bound is never tighter than the exact LP optimum, so the sweep stays sound.
//!
//! A later optimization (`PreparedPrimal`, item-3 tail) reuses one scaled
//! matrix + LU factorization across all probes to push below the 1.28 ms stateless
//! per-probe floor; this MVP still clones the CSC per probe (O(nnz), cheap on the
//! profiled node LP) but already captures the dominant marshaling win.

use crate::lp::basis::Basis;
use crate::lp::simplex::refine::ns_safe_bound_csc;
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{primal::solve_lp_cols_warm, LpSolve, LpStatus, SimplexOptions};

/// A rigorous lower bound on `min cᵀz` from one probe's row duals, or `None` when
/// the probe did not solve or its duals do not certify a finite bound (#1917).
///
/// The probe's vertex objective `sol.obj` is **not** a bound: on an ill-conditioned
/// basis it drifts past the true optimum, and as a tightened box edge that drift cuts
/// feasible points off -- measured on GDPlib `cstr` big-M, a binary whose LP maximum
/// is exactly 1 came back as `0.9999999999999996`, crossed the node's `lo = 1`, and
/// panicked the tree in `f64::clamp`. The Neumaier--Shcherbina evaluation of the
/// same duals is the bound the node LP itself certifies with.
#[allow(clippy::too_many_arguments)]
fn probe_safe_bound(
    sol: &LpSolve,
    sp: &SparseCols,
    c: &[f64],
    m: usize,
    n_total: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
) -> Option<f64> {
    if sol.status != LpStatus::Optimal {
        return None;
    }
    let (col_ptr, row_idx, vals) = sp.raw();
    ns_safe_bound_csc(&sol.dual, c, col_ptr, row_idx, vals, m, n_total, b, l, u)
        .filter(|v| v.is_finite())
}

/// Outcome of an OBBT probe sweep.
#[derive(Clone, Debug)]
pub struct ObbtSweepResult {
    /// Tightened `(lo, hi)` per candidate, aligned with the `candidates` input.
    /// Each entry is tighten-only: `lo >= l[k]`, `hi <= u[k]`. Each side is the
    /// probe's *safe* (Neumaier--Shcherbina) bound, never its vertex objective; a
    /// probe that does not solve to optimality, or whose duals do not certify a
    /// finite bound, leaves that side at the incoming bound.
    pub bounds: Vec<(f64, f64)>,
    /// Total LP solves performed (`<= 2 * candidates.len()`).
    pub n_solves: usize,
}

/// Run `min xₖ` / `max xₖ` over the node LP for each candidate structural column,
/// warm-starting every probe from the running optimal basis, and return the
/// tighten-only bounds.
///
/// The LP is the standard form the OBBT path assembles: `A z <= b` encoded as
/// `[A_ub | I] z = b` with one slack per row, so `sp`/`m`/`n_total`/`b`/`l`/`u`
/// describe the `m`-row, `n_total`-column (structural + slack) system, and
/// `candidates` indexes structural columns (`< n_total - m`).
///
/// `start` is the node's optimal basis (the natural warm start; every OBBT
/// objective shares this polytope, so it is primal-feasible for all of them).
/// Bound-neutral: each probe is `solve_lp_cols_warm` on the same LP the Python
/// sweep solves.
#[allow(clippy::too_many_arguments)]
pub fn obbt_probe_sweep(
    sp: &SparseCols,
    m: usize,
    n_total: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    candidates: &[usize],
    start: &Basis,
    opts: &SimplexOptions,
) -> ObbtSweepResult {
    let mut bounds = Vec::with_capacity(candidates.len());
    let mut n_solves = 0usize;
    // Thread the optimal basis probe-to-probe: consecutive unit objectives differ
    // by one column, so the previous optimum is a near-optimal warm start.
    let mut warm = start.clone();
    let mut c = vec![0.0f64; n_total];

    for &k in candidates {
        debug_assert!(k < n_total, "candidate column out of range");
        let (lo_in, hi_in) = (l[k], u[k]);

        // min xₖ : minimize eₖᵀz -> obj is the tightened lower bound.
        c[k] = 1.0;
        let lo = {
            let sol = solve_lp_cols_warm(sp.clone(), m, n_total, &c, l, u, b, &warm, opts);
            n_solves += 1;
            let safe = probe_safe_bound(&sol, sp, &c, m, n_total, b, l, u);
            if sol.status == LpStatus::Optimal {
                warm = sol.basis;
            }
            // tighten-only: never loosen below the incoming bound.
            safe.map_or(lo_in, |v| lo_in.max(v))
        };

        // max xₖ : minimize -eₖᵀz -> max value is -obj.
        c[k] = -1.0;
        let hi = {
            let sol = solve_lp_cols_warm(sp.clone(), m, n_total, &c, l, u, b, &warm, opts);
            n_solves += 1;
            // `min -xₖ >= safe`, so `xₖ <= -safe`.
            let safe = probe_safe_bound(&sol, sp, &c, m, n_total, b, l, u);
            if sol.status == LpStatus::Optimal {
                warm = sol.basis;
            }
            safe.map_or(hi_in, |v| hi_in.min(-v))
        };
        c[k] = 0.0; // reset for the next candidate

        // Guard against a crossed interval from tiny numerical drift: never emit
        // lo > hi (would be an empty box); clamp to the incoming interval.
        let (lo, hi) = if lo > hi { (lo_in, hi_in) } else { (lo, hi) };
        bounds.push((lo, hi));
    }

    ObbtSweepResult { bounds, n_solves }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::simplex::primal::solve_lp_cols;

    // Build a small standard-form LP [A_ub | I] z = b for a 2-var box problem and
    // return (sp, m, n_total, b, l, u). Constraints:
    //   x0 + x1 <= 3     (row 0)
    //   x0      <= 2     (row 1)
    // box 0 <= x0 <= 5, 0 <= x1 <= 5. Slacks s0, s1 >= 0.
    // Columns (CSC): x0, x1, s0, s1.  m = 2, n_total = 4.
    fn small_lp() -> (SparseCols, usize, usize, Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = 2;
        let n_total = 4;
        // CSC by column: col_ptr length n_total+1.
        // x0: rows {0:1, 1:1}; x1: rows {0:1}; s0: rows {0:1}; s1: rows {1:1}.
        let col_ptr = vec![0usize, 2, 3, 4, 5];
        let row_idx = vec![0usize, 1, 0, 0, 1];
        let vals = vec![1.0f64, 1.0, 1.0, 1.0, 1.0];
        let sp = SparseCols::from_csc(col_ptr, row_idx, vals);
        let b = vec![3.0, 2.0];
        let l = vec![0.0, 0.0, 0.0, 0.0];
        let u = vec![5.0, 5.0, 1e20, 1e20];
        (sp, m, n_total, b, l, u)
    }

    /// The sweep's tightened bounds must equal independent cold min/max solves of
    /// the same LP (bound-neutrality of the loop), and match the analytic truth:
    /// x0 in [0,2] (from x0<=2), x1 in [0,3] (from x0+x1<=3, x0>=0).
    #[test]
    fn sweep_matches_cold_solves_and_truth() {
        let (sp, m, n, b, l, u) = small_lp();
        let opts = SimplexOptions::default();
        // A feasible starting basis: both slacks basic.
        let start = Basis::from_basic(n, vec![2, 3]);
        let candidates = [0usize, 1];
        let res = obbt_probe_sweep(&sp, m, n, &b, &l, &u, &candidates, &start, &opts);
        assert_eq!(res.n_solves, 4);

        // Independent cold reference for each probe.
        for (idx, &k) in candidates.iter().enumerate() {
            let mut c = vec![0.0; n];
            c[k] = 1.0;
            let lo = solve_lp_cols(sp.clone(), m, n, &c, &l, &u, &b, &opts);
            c[k] = -1.0;
            let hi = solve_lp_cols(sp.clone(), m, n, &c, &l, &u, &b, &opts);
            assert_eq!(lo.status, LpStatus::Optimal);
            assert_eq!(hi.status, LpStatus::Optimal);
            let (ref_lo, ref_hi) = (l[k].max(lo.obj), u[k].min(-hi.obj));
            let (got_lo, got_hi) = res.bounds[idx];
            assert!(
                (got_lo - ref_lo).abs() < 1e-7 && (got_hi - ref_hi).abs() < 1e-7,
                "candidate {k}: sweep ({got_lo},{got_hi}) != cold ({ref_lo},{ref_hi})"
            );
        }
        // Analytic truth.
        assert!((res.bounds[0].0 - 0.0).abs() < 1e-7 && (res.bounds[0].1 - 2.0).abs() < 1e-7);
        assert!((res.bounds[1].0 - 0.0).abs() < 1e-7 && (res.bounds[1].1 - 3.0).abs() < 1e-7);
    }

    /// Tighten-only: an already-tight incoming bound is never loosened even though
    /// the LP would admit a wider range.
    #[test]
    fn sweep_is_tighten_only() {
        let (sp, m, n, b, mut l, mut u) = small_lp();
        // Pre-tighten x0 to [0.5, 1.5] — tighter than the LP's [0,2].
        l[0] = 0.5;
        u[0] = 1.5;
        let opts = SimplexOptions::default();
        let start = Basis::from_basic(n, vec![2, 3]);
        let res = obbt_probe_sweep(&sp, m, n, &b, &l, &u, &[0usize], &start, &opts);
        // OBBT over [0.5,1.5] recovers min 0.5 / max 1.5 (the box binds, not the LP),
        // so the bound is unchanged — never loosened toward the LP's [0,2].
        assert!(
            res.bounds[0].0 >= 0.5 - 1e-9,
            "lower loosened: {}",
            res.bounds[0].0
        );
        assert!(
            res.bounds[0].1 <= 1.5 + 1e-9,
            "upper loosened: {}",
            res.bounds[0].1
        );
    }

    /// #1917: every side is a *safe* bound, so it contains the exact LP range
    /// (x0 in [0, 2], x1 in [0, 3]) -- never tighter, even by an ulp. The vertex
    /// objective it replaced could land inside it and cut a feasible point off.
    #[test]
    fn sweep_sides_contain_the_exact_lp_range_1917() {
        let (sp, m, n, b, l, u) = small_lp();
        let opts = SimplexOptions::default();
        let start = Basis::from_basic(n, vec![2, 3]);
        let res = obbt_probe_sweep(&sp, m, n, &b, &l, &u, &[0usize, 1], &start, &opts);
        let exact = [(0.0, 2.0), (0.0, 3.0)];
        for (k, &(lo, hi)) in res.bounds.iter().enumerate() {
            assert!(
                lo <= exact[k].0 && hi >= exact[k].1,
                "{k}: ({lo},{hi}) cuts {:?}",
                exact[k]
            );
            assert!(
                hi <= exact[k].1 + 1e-7,
                "{k}: hi {hi} not tightened from the box"
            );
        }
    }
}
