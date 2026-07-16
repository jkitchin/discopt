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
//! (driving the residual to ~machine precision), and the refined `ā_j` are then
//! snapped to the nearest integer within [`SNAP_TOL`]. This collapses the flip
//! error so the cut is valid up to machine precision; a small caller-side rhs
//! margin then absorbs the remainder.

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
    if m == 0 {
        return cuts;
    }
    // A complete row-ordered basis is required to factorize B; a short basis
    // (degenerate phase-2 artifact the caller could not complete) is unusable —
    // decline rather than index past it. Matches the old dense path's `None`.
    if basis.basic_vars.len() != m {
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
    let xb = match ftran_refined(&mut lu, &sp, &basis.basic_vars, &rhs_b, tol) {
        Some(xb) => xb,
        None => return cuts,
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
        let w = match btran_refined(&mut lu, &sp, &basis.basic_vars, &e_i, tol) {
            Some(w) => w,
            None => continue,
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
            let mut abar: f64 = sp.dot(j, &w);
            if (abar - abar.round()).abs() < SNAP_TOL {
                abar = abar.round();
            }
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
            if psi.abs() <= tol {
                continue;
            }
            if !psi.is_finite() {
                ok = false;
                break;
            }
            if at_upper {
                coeffs[j] = -psi;
                rhs -= psi * u[j];
            } else {
                coeffs[j] = psi;
                rhs += psi * l[j];
            }
            max_c = max_c.max(psi.abs());
            min_c = min_c.min(psi.abs());
        }

        if !ok
            || max_c == 0.0
            || max_c > MAX_ABS_COEFF
            || rhs.abs() > MAX_ABS_COEFF
            || (min_c > 0.0 && max_c / min_c > max_dynamism)
        {
            continue;
        }
        cuts.push(GomoryCut { coeffs, rhs });
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::basis::recover_basis;

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
}
