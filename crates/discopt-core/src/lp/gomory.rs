//! Gomory mixed-integer (GMI) cuts from a recovered LP basis.
//!
//! Given a basis (from [`super::basis::recover_basis`]) the simplex tableau row
//! for a basic variable `x_{B_i}` is `x_{B_i} + Σ_{j∈N} ā_j x_j = b̄_i`, where
//! `ā = B⁻¹A` over the nonbasic columns `N`. When `x_{B_i}` is an
//! integer-constrained variable with a fractional value, this row yields a
//! **Gomory mixed-integer cut** — a valid inequality every integer-feasible
//! point satisfies but the current fractional vertex violates.
//!
//! The derivation works in the *shifted nonbasic space* `x̃_j ≥ 0`
//! (`x̃_j = x_j − l_j` at a lower bound, `u_j − x_j` at an upper bound). With the
//! row as `x_{B_i} + Σ_j ᾱ_j x̃_j = β` (`f₀ = β − ⌊β⌋`), the GMI cut is
//! `Σ_j ψ_j x̃_j ≥ 1` with, for `f_j = ᾱ_j − ⌊ᾱ_j⌋`:
//!
//! - integer nonbasic `j`: `ψ_j = f_j/f₀` if `f_j ≤ f₀`, else `(1−f_j)/(1−f₀)`;
//! - continuous nonbasic `j`: `ψ_j = ᾱ_j/f₀` if `ᾱ_j ≥ 0`, else `−ᾱ_j/(1−f₀)`.
//!
//! Substituting `x̃` back gives a cut `Σ_j γ_j x_j ≥ δ` over the original
//! standard-form variables (basic variables do not appear).
//!
//! **Numerical safety.** GMI is notoriously sensitive: an inaccurate tableau
//! coefficient `ā_j` near an integer flips `f_j` between ≈0 and ≈1, an O(1)
//! error that can make the cut *invalid* and cut off the true integer optimum.
//! Because the basis comes from an interior-point solve + crossover (accurate
//! only to ~1e-7), we do **not** trust the input vertex: the basic primal
//! values `x_B = B⁻¹(b − A_N x_N)` and the tableau rows `B⁻ᵀ e_i` are both
//! recomputed from the *exact* basis and bounds with **iterative refinement**
//! (driving the residual to ~machine precision).
//!
//! The refined `ā_j` are **not** snapped to the nearest integer. This module used
//! to snap them within [`SNAP_TOL`], justified as collapsing the flip error above.
//! That justification does not hold: the flip is in `f_j`, and `ψ(f)` is
//! *continuous* with `ψ → 0` at both `f → 0` and `f → 1`, so a coefficient
//! straddling an integer produces a near-zero `ψ` either way and there is nothing
//! to collapse. What snapping actually did was zero the `ψ` of a **continuous**
//! column whose `ā_j` was merely small, deleting `ψ_j·x̃_j` from the LHS of a `≥`
//! cut — a *strengthening*, and the one way this routine can emit an invalid cut.
//! See #1236: on the `fac2` OA master that deleted two terms worth 0.47 and 0.025
//! against a rhs of 1 (continuous slacks, `u = 1e20`, `x̃ ≈ 7e7` and `2.3e8`), and
//! the driver certified a false `optimal` 7839 above the true optimum.
//!
//! A tiny `ψ_j` may still leave the LHS, but only as a **relaxation**: its maximum
//! over the box, `ψ_j·(u_j − l_j)`, is charged to the rhs, and only when that
//! product is itself negligible. When the range is unbounded the term cannot move
//! and is kept exactly. Bounds are tested against the `1e20` sentinel directly and
//! never through a product — for a small `ψ` the product of an unbounded bound is
//! an ordinary finite number, which is precisely how this layer has produced a
//! false certified `optimal` before.

use super::basis::{Basis, AT_UPPER, BASIC};
use super::crossover::LpView;
use super::simplex::linsolve::{FeralLU, LinearSolver};
use super::simplex::sparse::SparseCols;

/// Tolerance for snapping a refined tableau coefficient to the nearest integer.
const SNAP_TOL: f64 = 1e-9;

/// Minimum fractionality of a basic variable to cut on. A value nearer than
/// this to an integer would divide by a tiny `f₀` (or `1−f₀`) and blow the cut
/// coefficients up to numerically unsafe magnitudes, so we skip it.
const FRAC_MIN: f64 = 1e-3;

/// Absolute cap on a cut coefficient. A cut with a larger coefficient comes
/// from an ill-conditioned basis (or a near-integral pivot) and is dropped: it
/// would dominate the relaxation numerically and is unsafe even when formally
/// valid.
const MAX_ABS_COEFF: f64 = 1e7;

/// A generated cut `coeffs · x ≥ rhs` over the standard-form variables.
#[derive(Debug, Clone)]
pub struct GomoryCut {
    /// Dense length-`n` coefficient vector.
    pub coeffs: Vec<f64>,
    /// Right-hand side (the cut is `coeffs · x ≥ rhs`).
    pub rhs: f64,
}

/// Solve the dense `n × n` system `mat · x = rhs` (row-major) by Gaussian
/// elimination with partial pivoting. Returns `None` if singular to `tol`.
pub(crate) fn solve_dense(mat: &[f64], n: usize, rhs: &[f64], tol: f64) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut x = rhs.to_vec();
    for col in 0..n {
        let mut p = col;
        for i in (col + 1)..n {
            if a[i * n + col].abs() > a[p * n + col].abs() {
                p = i;
            }
        }
        if a[p * n + col].abs() <= tol {
            return None;
        }
        if p != col {
            for j in 0..n {
                a.swap(col * n + j, p * n + j);
            }
            x.swap(col, p);
        }
        let piv = a[col * n + col];
        for j in 0..n {
            a[col * n + j] /= piv;
        }
        x[col] /= piv;
        for i in 0..n {
            if i != col {
                let f = a[i * n + col];
                if f != 0.0 {
                    for j in 0..n {
                        a[i * n + j] -= f * a[col * n + j];
                    }
                    x[i] -= f * x[col];
                }
            }
        }
    }
    Some(x)
}

/// Solve `B x = rhs` with iterative refinement, where `B` is the basis whose
/// columns are `A[:, basis[k]]` (factorized into `lu`) and `sp` is the CSC view
/// of `A`. The feral factor gives the initial solve; each refinement round
/// computes the residual `rhs − B x` by a sparse matvec against the basis
/// columns and corrects with another ftran — the same machine-precision drive
/// as the dense [`solve_refined`], at O(nnz) per round instead of O(m²). Returns
/// `None` if the factor solve fails.
fn ftran_refined(
    lu: &mut FeralLU,
    sp: &SparseCols,
    basis: &[usize],
    rhs: &[f64],
    _tol: f64,
) -> Option<Vec<f64>> {
    let mut x = rhs.to_vec();
    lu.ftran(&mut x).ok()?;
    for _ in 0..3 {
        // r = rhs − B x  (B column k contributes x[k]·A[:, basis[k]]).
        let mut r = rhs.to_vec();
        for (k, &bv) in basis.iter().enumerate() {
            let xk = x[k];
            if xk != 0.0 {
                let (rows, vals) = sp.col(bv);
                for (t, &i) in rows.iter().enumerate() {
                    r[i] -= vals[t] * xk;
                }
            }
        }
        let mut dx = r;
        lu.ftran(&mut dx).ok()?;
        let mut maxdx = 0.0_f64;
        for (xi, dxi) in x.iter_mut().zip(&dx) {
            *xi += dxi;
            maxdx = maxdx.max(dxi.abs());
        }
        if maxdx <= 1e-15 {
            break;
        }
    }
    Some(x)
}

/// Solve `Bᵀ w = rhs` with iterative refinement (row of `B⁻¹` when `rhs = e_i`).
/// The residual `rhs − Bᵀ w` is `rhs[k] − w·A[:, basis[k]]`, a sparse column dot
/// per basis slot. O(nnz) per refinement round; mirrors [`ftran_refined`].
fn btran_refined(
    lu: &mut FeralLU,
    sp: &SparseCols,
    basis: &[usize],
    rhs: &[f64],
    _tol: f64,
) -> Option<Vec<f64>> {
    let mut w = rhs.to_vec();
    lu.btran(&mut w).ok()?;
    for _ in 0..3 {
        let mut r = rhs.to_vec();
        for (k, &bv) in basis.iter().enumerate() {
            r[k] -= sp.dot(bv, &w);
        }
        let mut dw = r;
        lu.btran(&mut dw).ok()?;
        let mut maxd = 0.0_f64;
        for (wi, dwi) in w.iter_mut().zip(&dw) {
            *wi += dwi;
            maxd = maxd.max(dwi.abs());
        }
        if maxd <= 1e-15 {
            break;
        }
    }
    Some(w)
}

/// Separate Gomory mixed-integer cuts from the basis of `lp` (`b` is the
/// length-`m` right-hand side of `A x = b`).
///
/// The vertex is reconstructed from the exact basis and bounds (not from any
/// approximate input point): nonbasic variables sit at the bound named by
/// `basis.col_status`, and the basic values come from a refined solve of
/// `B x_B = b − A_N x_N`. `integrality[j]` marks integer-constrained variables,
/// `tol` is the fractionality/zero tolerance, and `max_dynamism` caps a cut's
/// `max|coeff| / min nonzero |coeff|` ratio. Returns one cut per fractional
/// integer basic variable that yields a numerically sound inequality.
pub fn separate_gomory(
    lp: &LpView<'_>,
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    // Dense-entry wrapper (used by tests): build the CSC once and delegate. The
    // driver calls `separate_gomory_cols` directly with its working CSC (no dense
    // matrix) — see docs/dev/sparse-milp-plan.md T3b5.
    let sp = SparseCols::from_dense(lp.a, lp.m, lp.n);
    separate_gomory_cols(
        &sp,
        lp.m,
        lp.n,
        lp.l,
        lp.u,
        b,
        basis,
        integrality,
        tol,
        max_dynamism,
    )
}

/// CSC-input GMI separation (the body of [`separate_gomory`]). Bit-identical: the
/// function already worked entirely through a `SparseCols` (`sp.col(j)` + a sparse
/// LU factorization); it just took the dense matrix and rebuilt the CSC internally.
#[allow(clippy::too_many_arguments)]
pub fn separate_gomory_cols(
    sp: &SparseCols,
    m: usize,
    n: usize,
    l: &[f64],
    u: &[f64],
    b: &[f64],
    basis: &Basis,
    integrality: &[bool],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    let mut cuts = Vec::new();
    crate::profile::incr(crate::profile::Ctr::SepGomoryCalls);
    if m == 0 {
        return cuts;
    }
    // A complete row-ordered basis is required to factorize B; a short basis
    // (degenerate phase-2 artifact the caller could not complete) is unusable —
    // decline rather than index past it. Matches the old dense path's `None`.
    if basis.basic_vars.len() != m {
        crate::profile::incr(crate::profile::Ctr::SepGomoryShortBasis);
        return cuts;
    }

    let mut lu = FeralLU::new();
    let bcols: Vec<Vec<(usize, f64)>> = basis
        .basic_vars
        .iter()
        .map(|&bv| {
            let (rows, vals) = sp.col(bv);
            rows.iter().zip(vals).map(|(&r, &v)| (r, v)).collect()
        })
        .collect();
    if lu.factorize_sparse(m, &bcols).is_err() {
        crate::profile::incr(crate::profile::Ctr::SepGomorySingular);
        return cuts; // singular basis → no cuts (as the dense solve's None did)
    }

    // Reconstruct the vertex exactly: nonbasic at bounds, x_B = B⁻¹(b − A_N x_N).
    let mut rhs_b = b.to_vec();
    for j in 0..n {
        if basis.col_status[j] == BASIC {
            continue;
        }
        let val = if basis.col_status[j] == AT_UPPER {
            u[j]
        } else {
            l[j]
        };
        if val != 0.0 {
            let (rows, vals) = sp.col(j);
            for (k, &i) in rows.iter().enumerate() {
                rhs_b[i] -= vals[k] * val;
            }
        }
    }
    let xb = match ftran_refined(&mut lu, sp, &basis.basic_vars, &rhs_b, tol) {
        Some(xb) => xb,
        None => {
            crate::profile::incr(crate::profile::Ctr::SepGomoryFtranFail);
            return cuts;
        }
    };

    for (i, &bi) in basis.basic_vars.iter().enumerate() {
        if !integrality[bi] {
            continue; // only integer basic variables yield cuts
        }
        let f0 = xb[i] - xb[i].floor();
        if !(FRAC_MIN..=1.0 - FRAC_MIN).contains(&f0) {
            continue; // integral, or too close to integral for a safe cut
        }

        // Row i of B⁻¹: refined solve of Bᵀ w = e_i (one btran + refinement).
        let mut e_i = vec![0.0_f64; m];
        e_i[i] = 1.0;
        let w = match btran_refined(&mut lu, sp, &basis.basic_vars, &e_i, tol) {
            Some(w) => w,
            None => {
                crate::profile::incr(crate::profile::Ctr::SepGomoryBtranFail);
                continue;
            }
        };

        // GMI cut Σ ψ_j x̃_j ≥ 1, accumulated into original-variable space.
        let mut coeffs = vec![0.0_f64; n];
        let mut rhs = 1.0_f64;
        let mut max_c = 0.0_f64;
        let mut min_c = f64::INFINITY;
        let mut ok = true;
        for j in 0..n {
            if basis.col_status[j] == BASIC {
                continue;
            }
            // ā_j = w · A[:,j], snapped to the nearest integer when very close
            // (the refined value is accurate, so this only removes ulp noise).
            // Sparse dot over column j's nonzeros (was an O(m) dense scan per j).
            let abar: f64 = sp.dot(j, &w);
            // #1236: `abar` is NOT snapped to the nearest integer. The header used
            // to justify snapping as collapsing a "flip error", but psi(f) is
            // continuous and zero at every integer, so there is no flip to
            // collapse -- and snapping a CONTINUOUS column's abar to 0 zeroes its
            // psi, which silently deletes `psi * xtilde_j` from the LHS of a `>=`
            // cut. Deleting a nonnegative term from the LHS of `sum psi xtilde >= 1`
            // STRENGTHENS it, so the cut can exclude feasible integer points.
            // Measured on the fac2 OA master: two such terms sat on continuous
            // slacks with u = 1e20 and xtilde ~ 7e7 / 2.3e8 at the optimum, so
            // coefficients of ~1e-10 were worth 0.47 and 0.025 against a rhs of 1.
            // The resulting cut cut off the optimum and the driver certified a
            // false `optimal` 7839 above it. `snap_eligible` survives only as the
            // "this integer column's abar is at an integer" flag used below.
            let snap_eligible = (abar - abar.round()).abs() < SNAP_TOL;
            // Nonbasic at its upper bound uses x̃_j = u_j − x_j (sign flip).
            let at_upper = basis.col_status[j] == AT_UPPER;
            let alpha = if at_upper { -abar } else { abar };

            // The integer GMI strengthening is valid only when the nonbasic
            // integer variable is pinned at an INTEGER bound, so the shifted
            // x̃_j = x_j − l_j (or u_j − x_j) takes integer values. If presolve
            // (coefficient strengthening / implied bounds) handed it a
            // fractional bound, that premise fails and the integer ψ can cut the
            // true optimum — fall back to the continuous formula, which is
            // always valid (a weaker but sound cut).
            let pinned = if at_upper { u[j] } else { l[j] };
            let use_integer = integrality[j] && (pinned - pinned.round()).abs() <= tol;
            let psi = if use_integer {
                let fj = alpha - alpha.floor();
                if fj <= f0 {
                    fj / f0
                } else {
                    (1.0 - fj) / (1.0 - f0)
                }
            } else if alpha >= 0.0 {
                alpha / f0
            } else {
                -alpha / (1.0 - f0)
            };
            if !psi.is_finite() {
                ok = false;
                break;
            }
            if psi == 0.0 {
                continue;
            }
            // #1236: a tiny term `psi * xtilde_j` (psi >= 0, xtilde >= 0) may only
            // leave the LHS of `sum psi xtilde >= 1` if its MAXIMUM over the box,
            // `psi * (u_j - l_j)`, is charged to the rhs -- that is a relaxation,
            // which is always sound. Dropping it outright is a strengthening and is
            // not. With an infinite range the term cannot move at all, so it is kept
            // exactly. The bound is tested against the 1e20 sentinel directly, never
            // via a product: for a small `psi` the product of an unbounded bound is
            // an ordinary finite number, which is exactly how this layer has
            // produced a false certified `optimal` before (CLAUDE.md, INF note).
            //
            // `substitute_slacks_to_structural` in the driver already applies this
            // rule correctly (it moves a small term to the rhs at its maximising
            // bound and refuses the cut when that pin is infinite); the separator
            // was the inconsistent one.
            // #1236 (review finding 3): gate `snap_eligible` on `use_integer`, not on
            // `integrality[j]`. When an integer column is pinned at a FRACTIONAL
            // bound the integer premise fails and the block above falls back to the
            // CONTINUOUS formula, where an integral `abar` gives psi = |alpha| / f0
            // -- which is >= 1/f0 and can be in the thousands, the opposite of tiny.
            // Calling that "tiny" excluded a large coefficient from the dynamism
            // gate below, so the gate silently stopped seeing the worst term in the
            // cut. Under `use_integer` an integral `abar` really does give psi = 0
            // (caught above) and a near-integral one gives ulp noise, which is the
            // only case the flag was ever meant to name.
            let tiny = psi.abs() <= tol || (use_integer && snap_eligible);
            if tiny {
                let range = u[j] - l[j];
                // Charge the term to the rhs only when its MAXIMUM contribution
                // over the box is itself negligible (<= tol): a bounded
                // weakening. Otherwise keep the exact coefficient.
                if u[j] < 1e20 && l[j] > -1e20 && range.is_finite() && psi * range <= tol {
                    rhs -= psi * range;
                    continue;
                }
            }
            if at_upper {
                coeffs[j] = -psi;
                rhs -= psi * u[j];
            } else {
                coeffs[j] = psi;
                rhs += psi * l[j];
            }
            max_c = max_c.max(psi.abs());
            // #1236: a tiny coefficient kept EXACTLY -- because its range was
            // unbounded and it therefore could not be charged to the rhs -- does not
            // count toward the dynamism gate. Counting it refuses nearly every cut
            // on a big-M master, which is sound but measurably harmful: on the
            // rsyn0830m master the refusals cost 6408 -> 17804 nodes. The cut itself
            // is exact either way; where a bound-based cleanup exists
            // (`substitute_slacks_to_structural`) it weakens these terms soundly or
            // refuses the cut, and where it does not (the convex kernel's
            // `substitute_slacks`) the row is still exact -- strictly better than the
            // deletion this replaces, which was unsound.
            if !tiny {
                min_c = min_c.min(psi.abs());
            }
        }

        // #1236 (review finding 2): `min_c` stays INFINITY when EVERY kept
        // coefficient was tiny, and `max_c / INFINITY == 0` slips past the dynamism
        // gate while `max_c != 0.0` slips past the emptiness gate -- so a cut whose
        // coefficients are all ~1e-12 would be emitted. Before the drop-term fix
        // above, such terms were skipped outright, `max_c` stayed 0.0 and the cut
        // was refused here; excluding tiny terms from `min_c` removed that refusal
        // as a side effect. `min_c.is_finite()` is exactly "at least one kept
        // coefficient was above tolerance", which restores it.
        //
        // It matters because the cut is not just weak, it is a FALSE PRUNE waiting
        // to happen: `sum 1e-12 xtilde >= 1` reads as `0 >= 1` at a 1e-9 LP
        // feasibility tolerance, so the node is fathomed as infeasible although it
        // contains feasible points, and that feeds a certified bound. The driver
        // path is shielded by `substitute_slacks_to_structural`, but the convex
        // kernel's `substitute_slacks` drops only exact zeros and would pass it
        // straight to the node LP. Refusing a cut is always sound.
        let all_kept_coefficients_are_tiny = !min_c.is_finite();
        if !ok
            || max_c == 0.0
            || all_kept_coefficients_are_tiny
            || max_c > MAX_ABS_COEFF
            || rhs.abs() > MAX_ABS_COEFF
            || (min_c > 0.0 && max_c / min_c > max_dynamism)
        {
            continue;
        }
        // #1236 review finding 4: record what was emitted, not just that it was.
        let nnz = coeffs.iter().filter(|v| **v != 0.0).count();
        crate::profile::incr(crate::profile::Ctr::SepGomoryCutsEmitted);
        crate::profile::incr_by(crate::profile::Ctr::SepGomoryCutNnz, nnz as u64);
        cuts.push(GomoryCut { coeffs, rhs });
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::basis::{recover_basis, AT_LOWER};

    /// Every refusal inside `separate_gomory_cols` must leave a counter behind.
    ///
    /// This function had five uncounted early returns, so "no cuts separated" and
    /// "never looked at a single tableau row" were the same observation. That is
    /// the CLAUDE.md §6 failure mode, and it is what hid the short-basis export
    /// defect (A0'.1): on `neos-2624317-amur` the cold root LP exported 338 basic
    /// variables for m = 342, every separation call bailed at the length guard,
    /// and the panel reported `0/0` cuts — read as "nothing to cut".
    ///
    /// Asserted as *deltas* around each call, and with the call counter as the
    /// denominator, so the test itself cannot pass vacuously.
    #[test]
    fn a_short_basis_refusal_is_counted_not_silent() {
        let _guard = crate::profile::test_guard();
        crate::profile::set_enabled(true);

        // x0 + x1 + s = 1.5, x0,x1 ∈ {0,1} relaxed, s ≥ 0 — the LP of the first
        // test, whose complete basis is known to separate exactly one GMI cut.
        let a = [1.0, 1.0, 1.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, f64::INFINITY];
        let b = [1.5];
        let integrality = [true, true, false];
        let sp = crate::lp::simplex::sparse::SparseCols::from_dense(&a, 1, 3);

        let calls = |()| crate::profile::counter(crate::profile::Ctr::SepGomoryCalls);
        let short = |()| crate::profile::counter(crate::profile::Ctr::SepGomoryShortBasis);

        // Arm 1: a basis one column short of `m`. It must refuse AND say so.
        let c0 = calls(());
        let s0 = short(());
        let stub = Basis {
            col_status: vec![AT_LOWER, AT_LOWER, AT_LOWER],
            basic_vars: vec![], // 0 != m = 1
        };
        let cuts = separate_gomory_cols(&sp, 1, 3, &l, &u, &b, &stub, &integrality, 1e-7, 1e9);
        assert!(cuts.is_empty(), "a short basis cannot yield cuts");
        assert_eq!(calls(()) - c0, 1, "the call counter is the denominator");
        assert_eq!(short(()) - s0, 1, "the short-basis refusal must be counted");

        // Arm 2: the complete basis. The refusal counter must NOT move — without
        // this arm the test would pass on a counter that fires unconditionally.
        let x = [1.0, 0.5, 0.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &[0.0, 0.0, 0.0],
            l: &l,
            u: &u,
        };
        let good = recover_basis(&x, &lp, 1e-7).expect("basis");
        let c1 = calls(());
        let s1 = short(());
        let cuts = separate_gomory_cols(&sp, 1, 3, &l, &u, &b, &good, &integrality, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1, "the complete basis still separates its cut");
        assert_eq!(calls(()) - c1, 1);
        assert_eq!(
            short(()) - s1,
            0,
            "a complete basis must not count a refusal"
        );

        crate::profile::set_enabled(false);
    }

    /// #1236 review finding 2: a cut whose every coefficient is tiny must be REFUSED.
    ///
    /// Such a cut is mathematically valid and numerically a trap: at a 1e-9 LP
    /// feasibility tolerance `2e-12 y + 2e-12 s >= 1` reads as `0 >= 1`, so the node
    /// is fathomed as infeasible although it contains feasible points -- a FALSE
    /// PRUNE feeding a certified bound.
    ///
    /// Before the drop-term fix these terms were skipped outright, `max_c` stayed
    /// 0.0 and the `max_c == 0.0` arm refused the cut. Keeping them (which
    /// soundness requires) while excluding them from `min_c` left `min_c` at
    /// INFINITY, and `max_c / INFINITY == 0` passes the dynamism gate.
    #[test]
    fn a_cut_whose_every_coefficient_is_tiny_is_refused() {
        // 1e12·x0 + y + s = 5e11, x0 ∈ {0,1} relaxed, y,s continuous on [0, 1e20].
        // At x0 = 0.5 the basic row scales by w = 1e-12, so BOTH nonbasic columns
        // get abar = 1e-12 -> psi = 2e-12, and neither can be charged to the rhs
        // because its range is unbounded. Every kept coefficient is tiny.
        let a = [1e12, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1e20, 1e20];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [5e11];
        let x = [0.5, 0.0, 0.0];
        let integrality = [true, false, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(
            basis.basic_vars,
            vec![0],
            "x0 must be the basic fractional var"
        );

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);

        // The assertion is the SPEC, not the count: whatever is emitted must be
        // enforceable at LP tolerance. A cut whose largest coefficient is below the
        // feasibility tolerance cannot be, whatever its rhs says.
        let mut checked = 0usize;
        for cut in &cuts {
            let max_c = cut.coeffs.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            checked += 1;
            assert!(
                max_c > 1e-7,
                "emitted a cut no LP can enforce: max|coeff| = {max_c:e} vs rhs {} \
                 -- at a 1e-9 feasibility tolerance this reads as `0 >= 1` and \
                 fathoms a node that contains feasible points",
                cut.rhs
            );
        }
        assert!(
            cuts.is_empty(),
            "all {} coefficients are ~2e-12; the cut must be refused outright \
             (checked {checked} emitted cuts)",
            cuts.len()
        );
    }

    /// #1236 review finding 3: `snap_eligible` may only be consulted on the
    /// `use_integer` branch.
    ///
    /// An integer column pinned at a FRACTIONAL bound falls back to the continuous
    /// formula, where an integral `abar` yields psi = |alpha| / f0 >= 1 -- a LARGE
    /// coefficient. Flagging it `tiny` (because the column is integral and its abar
    /// sits on an integer) excluded it from `min_c`, so the dynamism gate stopped
    /// seeing the smallest coefficient in the cut and waved through a cut with a
    /// 1e6 spread.
    #[test]
    fn an_integer_column_pinned_at_a_fractional_bound_is_not_tiny() {
        // x0 + x1 + 1e6·s = 1; x0 ∈ {0,1} basic at 0.5; x1 INTEGER pinned at the
        // fractional bound l = 0.5; s continuous. w = 1, so abar_x1 = 1 (integral,
        // hence `snap_eligible`) while psi_x1 = 1 / 0.5 = 2, and psi_s = 2e6.
        let a = [1.0, 1.0, 1e6];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.5, 0.0];
        let u = [1.0, 2.5, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [1.0];
        let x = [0.5, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]);

        // Arm 1 (loose gate): the cut IS emitted, and x1's coefficient is the large
        // continuous-formula psi = 2 -- not zero, and not tiny.
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert_eq!(
            cuts.len(),
            1,
            "a loose dynamism gate must still emit the cut"
        );
        let cut = &cuts[0];
        assert!(
            (cut.coeffs[1] - 2.0).abs() < 1e-9,
            "x1's coefficient must be the continuous-formula psi = 2, got {}",
            cut.coeffs[1]
        );
        assert!(
            dot(&cut.coeffs, &x) < cut.rhs - 1e-6,
            "must cut off the vertex"
        );
        // The only integer-feasible point: x1 ∈ {1,2}, and x1 = 2 forces s < 0.
        let pt = [0.0, 1.0, 0.0];
        assert!(
            dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
            "cut excludes feasible point {pt:?}"
        );

        // Arm 2 (tight gate): the true spread is max_c / min_c = 2e6 / 2 = 1e6, so a
        // gate of 1e3 must refuse it. Treating x1 as `tiny` hid it from `min_c`,
        // leaving min_c = max_c = 2e6, a reported spread of 1.0, and the cut passed.
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e3);
        assert!(
            cuts.is_empty(),
            "dynamism 1e6 must be refused by a 1e3 gate; the gate saw {:?}",
            cuts.iter().map(|c| c.coeffs.clone()).collect::<Vec<_>>()
        );
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn gmi_cut_separates_fractional_vertex_and_is_valid() {
        // x0 + x1 + s = 1.5, x0,x1 ∈ {0,1} (relaxed [0,1]), s >= 0.
        // Vertex (1, 0.5, 0): x1 basic & fractional → GMI cut 2 s ≥ 1
        // (equivalently x0 + x1 ≤ 1).
        let a = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [1.5];
        let x = [1.0, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![1]); // x1 is the basic (free) var

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6); // cuts off the vertex
        for b0 in 0..=1 {
            for b1 in 0..=1 {
                let s = 1.5 - b0 as f64 - b1 as f64;
                if s < -1e-9 {
                    continue;
                }
                let pt = [b0 as f64, b1 as f64, s];
                assert!(
                    dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                    "cut excludes feasible point {pt:?}"
                );
            }
        }
    }

    #[test]
    fn gmi_cut_valid_for_general_integer_at_upper_bound() {
        // 2x0 + 2x1 + s = 5, x0,x1 integer in [0,2], s >= 0. Vertex
        // (0.5, 2, 0): x0 basic & fractional, x1 nonbasic at its UPPER bound 2
        // (the general-integer case that broke a naive, unrefined GMI). The cut
        // is s ≥ 1, i.e. x0 + x1 ≤ 2.
        let a = [2.0, 2.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [2.0, 2.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [5.0];
        let x = [0.5, 2.0, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]);

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6); // separates (0.5, 2, 0)
        for i0 in 0..=2 {
            for i1 in 0..=2 {
                let s = 5.0 - 2.0 * i0 as f64 - 2.0 * i1 as f64;
                if s < -1e-9 {
                    continue;
                }
                let pt = [i0 as f64, i1 as f64, s];
                assert!(
                    dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                    "cut excludes feasible point {pt:?}"
                );
            }
        }
    }

    #[test]
    fn no_cut_when_vertex_is_integral() {
        // Same first system, integral vertex (1, 0, 0.5): the only basic var is
        // the continuous slack, so no GMI cut.
        let a = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [1.5];
        let x = [1.0, 0.0, 0.5]; // s basic = 0.5, but s is continuous
        let integrality = [true, true, false];
        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert!(cuts.is_empty());
    }

    #[test]
    fn gmi_cut_two_constraints_is_valid() {
        // 2x0 + x1 + s0 = 3,  x0 + 2x1 + s1 = 3,  x0,x1 integer >= 0, s >= 0.
        // Vertex x0 = 1.5, x1 = 0, s0 = 0, s1 = 1.5 (x0 basic & fractional).
        let n = 4;
        let m = 2;
        let a = [
            2.0, 1.0, 1.0, 0.0, // 2x0 + x1 + s0 = 3
            1.0, 2.0, 0.0, 1.0, // x0 + 2x1 + s1 = 3
        ];
        let c = [0.0; 4];
        let l = [0.0; 4];
        let u = [f64::INFINITY; 4];
        let lp = LpView {
            a: &a,
            m,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [3.0, 3.0];
        let x = [1.5, 0.0, 0.0, 1.5];
        let integrality = [true, true, false, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        assert!(!cuts.is_empty(), "expected a cut from fractional x0");

        for cut in &cuts {
            assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6);
            for i0 in 0..=3 {
                for i1 in 0..=3 {
                    let (x0, x1) = (i0 as f64, i1 as f64);
                    let s0 = 3.0 - 2.0 * x0 - x1;
                    let s1 = 3.0 - x0 - 2.0 * x1;
                    if s0 < -1e-9 || s1 < -1e-9 {
                        continue;
                    }
                    let pt = [x0, x1, s0, s1];
                    assert!(
                        dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                        "cut excludes feasible point {pt:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn gmi_cut_valid_when_integer_var_has_fractional_bound() {
        // x0 + x1 + s = 2, x0 integer in [0,2], x1 integer with a *fractional*
        // lower bound 0.5 (as presolve coefficient-strengthening / implied bounds
        // could produce), s >= 0. Vertex (1.5, 0.5, 0): x0 basic & fractional,
        // x1 nonbasic at its fractional lower bound. The integer GMI ψ assumes
        // the nonbasic integer sits at an integer bound, which is FALSE here; the
        // guard must fall back to the continuous ψ so the cut stays valid for
        // every feasible integer point (x1 ∈ {1,2}, since x1 is integer ≥ 0.5).
        let a = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.5, 0.0];
        let u = [2.0, 2.0, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [2.0];
        let x = [1.5, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]); // x0 is the fractional basic var

        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-7, 1e9);
        for cut in &cuts {
            assert!(
                dot(&cut.coeffs, &x) < cut.rhs - 1e-6,
                "cut must separate vertex"
            );
            // Feasible integer points: x1 integer ≥ 0.5 → x1 ∈ {1, 2}.
            for x0i in 0..=2 {
                for x1i in 1..=2 {
                    let s = 2.0 - x0i as f64 - x1i as f64;
                    if s < -1e-9 {
                        continue;
                    }
                    let pt = [x0i as f64, x1i as f64, s];
                    assert!(
                        dot(&cut.coeffs, &pt) >= cut.rhs - 1e-6,
                        "cut excludes feasible integer point {pt:?}"
                    );
                }
            }
        }
    }

    /// #1236 regression: a continuous column with a TINY tableau coefficient and a
    /// huge range must not be silently dropped from a GMI cut. Row
    /// `x0 + 1e-10*y + s = 0.5`, `x0 ∈ {0,1}`, `y ∈ [0, 1e20]`, `s >= 0`; vertex
    /// (0.5, 0, 0). Snapping `abar_y = 1e-10` to 0 (or skipping `psi_y = 2e-10`)
    /// yields `2 s >= 1`, which cuts off the feasible integer point
    /// `(0, 5e9, 0)`. The exact cut `2e-10 y + 2 s >= 1` holds there with equality.
    #[test]
    fn gmi_tiny_coefficient_on_wide_continuous_column_is_not_dropped() {
        let a = [1.0, 1e-10, 1.0];
        let c = [0.0, 0.0, 0.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1e20, f64::INFINITY];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let b = [0.5];
        let x = [0.5, 0.0, 0.0];
        let integrality = [true, false, false];
        let basis = recover_basis(&x, &lp, 1e-9).expect("basis");
        assert_eq!(basis.basic_vars, vec![0]);
        let cuts = separate_gomory(&lp, &b, &basis, &integrality, 1e-9, 1e12);
        assert_eq!(cuts.len(), 1, "one GMI cut off the fractional basic x0");
        let cut = &cuts[0];
        assert!(
            dot(&cut.coeffs, &x) < cut.rhs - 1e-6,
            "cut must separate the vertex"
        );
        let pt = [0.0, 5e9, 0.0]; // feasible: 0 + 1e-10*5e9 + 0 = 0.5
        assert!(
            dot(&cut.coeffs, &pt) >= cut.rhs - 1e-9,
            "cut excludes feasible integer point {pt:?}: lhs={} rhs={}",
            dot(&cut.coeffs, &pt),
            cut.rhs
        );
    }
}
