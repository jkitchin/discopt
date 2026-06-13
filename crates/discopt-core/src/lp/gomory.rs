//! Gomory mixed-integer (GMI) cuts from a recovered LP basis.
//!
//! Given an optimal basis (from [`super::basis::recover_basis`]) at a vertex
//! `x*`, the simplex tableau row for a basic variable `x_{B_i}` is
//! `x_{B_i} + Σ_{j∈N} ā_j x_j = b̄_i`, where `ā = B⁻¹A` restricted to the
//! nonbasic columns `N` and `b̄_i = x*_{B_i}`. When `x_{B_i}` is an
//! integer-constrained variable with a fractional value, this row yields a
//! **Gomory mixed-integer cut** — a valid inequality that every integer-feasible
//! point satisfies but `x*` violates.
//!
//! The derivation works in the *shifted nonbasic space* `x̃_j ≥ 0`
//! (`x̃_j = x_j − l_j` if `x_j` is nonbasic at its lower bound,
//! `x̃_j = u_j − x_j` if at its upper bound). With the row written as
//! `x_{B_i} + Σ_j ᾱ_j x̃_j = β` (so `f₀ = β − ⌊β⌋ = frac(x*_{B_i})`), the GMI
//! cut is `Σ_j ψ_j x̃_j ≥ 1` with, for `f_j = ᾱ_j − ⌊ᾱ_j⌋`:
//!
//! - integer nonbasic `j`: `ψ_j = f_j/f₀` if `f_j ≤ f₀`, else `(1−f_j)/(1−f₀)`;
//! - continuous nonbasic `j`: `ψ_j = ᾱ_j/f₀` if `ᾱ_j ≥ 0`, else `−ᾱ_j/(1−f₀)`.
//!
//! Substituting `x̃` back gives a cut `Σ_j γ_j x_j ≥ δ` in the original
//! standard-form variables (basic variables do not appear). The cut is valid
//! for any basis row with an integer basic variable — optimality is not
//! required — so soundness does not depend on crossover/IPM numerics.

use super::basis::{Basis, AT_UPPER, BASIC};
use super::crossover::LpView;

/// A generated cut `coeffs · x ≥ rhs` over the standard-form variables.
pub struct GomoryCut {
    /// Dense length-`n` coefficient vector.
    pub coeffs: Vec<f64>,
    /// Right-hand side (the cut is `coeffs · x ≥ rhs`).
    pub rhs: f64,
}

/// Solve the dense `n × n` system `mat · w = rhs` (row-major `mat`) by Gaussian
/// elimination with partial pivoting. Returns `None` if singular to `tol`.
fn solve_dense(mat: &[f64], n: usize, rhs: &[f64], tol: f64) -> Option<Vec<f64>> {
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

/// Separate Gomory mixed-integer cuts at the basic vertex `x` of `lp`.
///
/// `basis` is the recovered basis, `integrality[j]` whether variable `j` is
/// integer-constrained, `tol` the fractionality/zero tolerance, and
/// `max_dynamism` the largest allowed `max|coeff| / min nonzero |coeff|` ratio
/// (cuts exceeding it are dropped as numerically unsafe). Returns one cut per
/// fractional integer basic variable that produces a numerically sound
/// inequality.
///
/// The right-hand side `b` of `A x = b` is not needed: each basic variable's
/// value is read directly from `x` (which `recover_basis` guarantees is
/// consistent with `A_B x_B = b − A_N x_N`).
pub fn separate_gomory(
    lp: &LpView<'_>,
    basis: &Basis,
    integrality: &[bool],
    x: &[f64],
    tol: f64,
    max_dynamism: f64,
) -> Vec<GomoryCut> {
    let (a, m, n, l, u) = (lp.a, lp.m, lp.n, lp.l, lp.u);
    let mut cuts = Vec::new();
    if m == 0 {
        return cuts;
    }

    // B^T (row-major m×m): row r is basis column basic_vars[r].
    // bt[r*m + c] = B[c][r] = a[c*n + basic_vars[r]].
    let bt: Vec<f64> = {
        let mut v = vec![0.0_f64; m * m];
        for (r, &bv) in basis.basic_vars.iter().enumerate() {
            for c in 0..m {
                v[r * m + c] = a[c * n + bv];
            }
        }
        v
    };

    for (i, &bi) in basis.basic_vars.iter().enumerate() {
        if !integrality[bi] {
            continue; // only integer basic variables yield cuts
        }
        let f0 = x[bi] - x[bi].floor();
        if f0 < tol || f0 > 1.0 - tol {
            continue; // already integral
        }

        // Row i of B⁻¹: solve B^T w = e_i.
        let mut e_i = vec![0.0_f64; m];
        e_i[i] = 1.0;
        let w = match solve_dense(&bt, m, &e_i, tol) {
            Some(w) => w,
            None => continue,
        };

        // Build ψ over nonbasic vars and accumulate the cut in original space:
        //   Σ_j ψ_j x̃_j ≥ 1, with x̃_j = x_j − l_j (lower) or u_j − x_j (upper).
        let mut coeffs = vec![0.0_f64; n];
        let mut rhs = 1.0_f64;
        let mut max_c = 0.0_f64;
        let mut min_c = f64::INFINITY;
        let mut ok = true;
        for j in 0..n {
            if basis.col_status[j] == BASIC {
                continue;
            }
            // ā_j = w · A[:,j].
            let abar: f64 = (0..m).map(|r| w[r] * a[r * n + j]).sum();
            // Nonbasic at its upper bound uses x̃_j = u_j − x_j (sign flip).
            let at_upper = basis.col_status[j] == AT_UPPER;
            // ᾱ_j: coefficient of x̃_j in the row (sign flips at the upper bound).
            let alpha = if at_upper { -abar } else { abar };

            let psi = if integrality[j] {
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
            // Σ ψ_j x̃_j ≥ 1  →  original-variable coefficients.
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

        if !ok || max_c == 0.0 || (min_c > 0.0 && max_c / min_c > max_dynamism) {
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
        let x = [1.0, 0.5, 0.0];
        let integrality = [true, true, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        assert_eq!(basis.basic_vars, vec![1]); // x1 is the basic (free) var

        let cuts = separate_gomory(&lp, &basis, &integrality, &x, 1e-7, 1e9);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];

        // Violated by x*: coeffs · x* < rhs.
        assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6);

        // Valid for every integer-feasible standard-form point.
        for b0 in 0..=1 {
            for b1 in 0..=1 {
                let s = 1.5 - b0 as f64 - b1 as f64;
                if s < -1e-9 {
                    continue; // infeasible (x0=x1=1)
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
    fn no_cut_when_vertex_is_integral() {
        // Same system, integral vertex (1, 0, 0.5): no fractional integer basic
        // variable (s is continuous), so no GMI cut.
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
        let x = [1.0, 0.0, 0.5]; // s basic = 0.5, but s is continuous
        let integrality = [true, true, false];
        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        let cuts = separate_gomory(&lp, &basis, &integrality, &x, 1e-7, 1e9);
        assert!(cuts.is_empty());
    }

    #[test]
    fn gmi_cut_two_constraints_is_valid() {
        // 2x0 + x1 + s0 = 3,  x0 + 2x1 + s1 = 3,  x0,x1 integer >= 0, s >= 0.
        // LP relaxation optimum (max x0+x1) is (1,1)... pick a fractional
        // vertex by construction: x0 = x1 = 1.0 is integral; instead use the
        // vertex where only the first row binds. Take x0 = 1.5, x1 = 0,
        // s0 = 0, s1 = 1.5 → 2(1.5)+0+0 = 3 ✓, 1.5+0+1.5 = 3 ✓. x0 fractional.
        let n = 4; // x0, x1, s0, s1
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
        let x = [1.5, 0.0, 0.0, 1.5];
        let integrality = [true, true, false, false];

        let basis = recover_basis(&x, &lp, 1e-7).expect("basis");
        let cuts = separate_gomory(&lp, &basis, &integrality, &x, 1e-7, 1e9);
        assert!(!cuts.is_empty(), "expected a cut from fractional x0");

        for cut in &cuts {
            // Violated by x*.
            assert!(dot(&cut.coeffs, &x) < cut.rhs - 1e-6);
            // Valid for integer-feasible points (enumerate small x0,x1).
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
}
