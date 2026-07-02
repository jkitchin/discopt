//! Interior-point → vertex crossover.
//!
//! Given a feasible LP `min cᵀx  s.t.  A x = b,  l ≤ x ≤ u` and a point `x`
//! that is optimal (value `z* = cᵀx`), the *optimal face* is
//! `{x : A x = b, cᵀx = z*, l ≤ x ≤ u}`. An interior-point solve lands at the
//! analytic center of that face; cover/clique cuts and branching separate a
//! *vertex* of it far more sharply (a symmetric interior point violates no
//! cover, but a vertex does).
//!
//! [`crossover_to_vertex`] walks the interior optimum to a vertex: repeatedly
//! move along a direction `d` supported on the *free* variables (those strictly
//! inside their bounds) with `A d = 0` **and** `cᵀd = 0` — so the move keeps
//! both feasibility and the objective — until a free variable hits a bound
//! (ratio test). Fixing that variable removes a degree of freedom; when the
//! free columns of `[A; cᵀ]` are independent (no such direction remains) the
//! point is a vertex of the optimal face. This terminates in at most `n` steps.
//!
//! The direction is a null-space vector of the `(m+1) × |free|` matrix
//! `[A_free; c_freeᵀ]`, found by a rank-revealing reduced-row-echelon
//! elimination (no external linear-algebra dependency).
//!
//! This is used to *locate* sharp cut/branching structure; cuts are validated
//! by their own (cover/clique) structure, so crossover numerics never affect
//! soundness.

/// Skip the push on very wide problems; interior-point separation is the
/// fallback and cut soundness does not depend on the crossover.
pub const MAX_CROSSOVER_VARS: usize = 400;

/// Borrowed view of a standard-form LP `min cᵀx s.t. A x = b, l ≤ x ≤ u`.
///
/// `a` is the `m × n` constraint matrix in row-major order. The right-hand
/// side `b` is intentionally absent: the crossover preserves `A x = b` by
/// moving only in the null space of `A`, so it never needs it. The basis and
/// cut routines that will share this view add `b` when they need it.
pub struct LpView<'a> {
    /// Row-major `m × n` equality-constraint matrix.
    pub a: &'a [f64],
    /// Number of constraint rows.
    pub m: usize,
    /// Number of variables.
    pub n: usize,
    /// Objective coefficients, length `n`.
    pub c: &'a [f64],
    /// Lower bounds, length `n`.
    pub l: &'a [f64],
    /// Upper bounds, length `n`.
    pub u: &'a [f64],
}

/// A unit null-space direction of the `rows × cols` row-major matrix `mat`,
/// or `None` when its columns are linearly independent (rank `== cols`).
///
/// Uses Gaussian elimination to reduced row-echelon form with partial
/// pivoting. `tol` scales the pivot-zero threshold by the largest pivot seen.
fn null_direction(mat: &[f64], rows: usize, cols: usize, tol: f64) -> Option<Vec<f64>> {
    if cols == 0 {
        return None;
    }
    // Working copy in row-major order; reduced in place.
    let mut r: Vec<f64> = mat.to_vec();
    let at = |data: &[f64], i: usize, j: usize| data[i * cols + j];

    // Pivot column for each pivot row, and whether each column is a pivot.
    let mut pivot_col_of_row: Vec<usize> = Vec::with_capacity(rows.min(cols));
    let mut is_pivot_col = vec![false; cols];

    // Largest pivot magnitude, for a relative zero threshold.
    let mut max_pivot = 0.0_f64;
    for j in 0..cols {
        for i in 0..rows {
            let v = at(&r, i, j).abs();
            if v > max_pivot {
                max_pivot = v;
            }
        }
    }
    let zero = tol * max_pivot.max(1.0);

    let mut pr = 0usize; // current pivot row
    for col in 0..cols {
        if pr >= rows {
            break;
        }
        // Partial pivot: largest-magnitude entry in this column at/below pr.
        let mut best = pr;
        let mut best_val = at(&r, pr, col).abs();
        for i in (pr + 1)..rows {
            let v = at(&r, i, col).abs();
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        if best_val <= zero {
            continue; // free (non-pivot) column
        }
        // Swap row `best` into pivot position `pr`.
        if best != pr {
            for j in 0..cols {
                r.swap(best * cols + j, pr * cols + j);
            }
        }
        // Scale pivot row so the pivot is 1.
        let piv = r[pr * cols + col];
        for j in 0..cols {
            r[pr * cols + j] /= piv;
        }
        // Eliminate this column from every other row (reduced echelon form).
        for i in 0..rows {
            if i == pr {
                continue;
            }
            let factor = r[i * cols + col];
            if factor != 0.0 {
                for j in 0..cols {
                    r[i * cols + j] -= factor * r[pr * cols + j];
                }
            }
        }
        is_pivot_col[col] = true;
        pivot_col_of_row.push(col);
        pr += 1;
    }

    let rank = pr;
    if rank >= cols {
        return None; // full column rank → trivial null space
    }

    // Build a null vector: set the first free column to 1, solve pivots from
    // the reduced rows (pivot var = −coefficient of the free column).
    let free_col = (0..cols).find(|&c| !is_pivot_col[c]).unwrap();
    let mut d = vec![0.0_f64; cols];
    d[free_col] = 1.0;
    for (row_idx, &pcol) in pivot_col_of_row.iter().enumerate() {
        d[pcol] = -at(&r, row_idx, free_col);
    }

    // Normalize to unit length.
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return None;
    }
    for v in d.iter_mut() {
        *v /= norm;
    }
    Some(d)
}

/// Largest signed step `t` keeping `x + t·d` within `[l, u]` on the `free`
/// variables. Tries `+d` then `−d` (both preserve feasibility and objective);
/// returns the signed step that drives some free variable to a bound, or
/// `None` if the face is unbounded along `d`.
fn max_step(x: &[f64], d: &[f64], l: &[f64], u: &[f64], free: &[usize], tol: f64) -> Option<f64> {
    let step = |sign: f64| -> f64 {
        let mut t = f64::INFINITY;
        for &j in free {
            let dj = sign * d[j];
            if dj > tol {
                t = t.min((u[j] - x[j]) / dj);
            } else if dj < -tol {
                t = t.min((l[j] - x[j]) / dj);
            }
        }
        t
    };
    let tp = step(1.0);
    if tp.is_finite() && tp > tol {
        return Some(tp);
    }
    let tn = step(-1.0);
    if tn.is_finite() && tn > tol {
        return Some(-tn);
    }
    None
}

/// Push the interior optimum `x` to a vertex of the LP optimal face.
///
/// Returns a point with the same objective and (to tolerance) the same
/// feasibility as `x` but at a vertex of the optimal face. Returns `x`
/// unchanged when the problem is too wide (`lp.n > MAX_CROSSOVER_VARS`) or the
/// face is unbounded along a push direction.
///
/// `tol` is the bound/pivot tolerance (e.g. `1e-7`); `max_iter` caps the number
/// of variable-fixing steps (`n + 1` is always sufficient — pass `0` for the
/// default).
pub fn crossover_to_vertex(x: &[f64], lp: &LpView<'_>, tol: f64, max_iter: usize) -> Vec<f64> {
    let (a, m, n, c, l, u) = (lp.a, lp.m, lp.n, lp.c, lp.l, lp.u);
    let mut x = x.to_vec();
    if n == 0 || n > MAX_CROSSOVER_VARS {
        return x;
    }
    let iters = if max_iter == 0 { n + 1 } else { max_iter };

    for _ in 0..iters {
        // Free variables: strictly inside their bounds.
        let free: Vec<usize> = (0..n)
            .filter(|&j| x[j] > l[j] + tol && x[j] < u[j] - tol)
            .collect();
        if free.is_empty() {
            break;
        }
        let k = free.len();

        // Build M = [A_free ; c_freeᵀ], an (m+1) × k row-major matrix.
        let rows = m + 1;
        let mut mmat = vec![0.0_f64; rows * k];
        for (col, &j) in free.iter().enumerate() {
            for i in 0..m {
                mmat[i * k + col] = a[i * n + j];
            }
            mmat[m * k + col] = c[j];
        }

        let d_free = match null_direction(&mmat, rows, k, tol) {
            Some(d) => d,
            None => break, // free columns independent → vertex of optimal face
        };

        // Scatter the free-space direction back to a full-length direction.
        let mut d = vec![0.0_f64; n];
        for (col, &j) in free.iter().enumerate() {
            d[j] = d_free[col];
        }

        let t = match max_step(&x, &d, l, u, &free, tol) {
            Some(t) if t.abs() >= tol => t,
            _ => break,
        };

        for j in 0..n {
            let xj = x[j] + t * d[j];
            // Guard against rounding-induced bound inversion (l[j] a few ULP above
            // u[j] on a near-fixed variable): f64::clamp panics when min > max.
            // Clamp into the well-ordered interval instead — identical to the
            // direct clamp when bounds are ordered, and collapses to the degenerate
            // (ULP-wide) box when they cross, which is the correct projection.
            let (lo, hi) = if l[j] <= u[j] {
                (l[j], u[j])
            } else {
                (u[j], l[j])
            };
            x[j] = xj.clamp(lo, hi);
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `true` if `x` is a vertex of the optimal face: the free columns of
    /// `[A; cᵀ]` are independent (no objective/feasibility-preserving move).
    fn is_vertex(
        x: &[f64],
        a: &[f64],
        m: usize,
        n: usize,
        c: &[f64],
        l: &[f64],
        u: &[f64],
    ) -> bool {
        let free: Vec<usize> = (0..n)
            .filter(|&j| x[j] > l[j] + 1e-6 && x[j] < u[j] - 1e-6)
            .collect();
        if free.is_empty() {
            return true;
        }
        let k = free.len();
        let rows = m + 1;
        let mut mmat = vec![0.0_f64; rows * k];
        for (col, &j) in free.iter().enumerate() {
            for i in 0..m {
                mmat[i * k + col] = a[i * n + j];
            }
            mmat[m * k + col] = c[j];
        }
        null_direction(&mmat, rows, k, 1e-7).is_none()
    }

    fn matvec(a: &[f64], m: usize, n: usize, x: &[f64]) -> Vec<f64> {
        (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    }

    fn dot(c: &[f64], x: &[f64]) -> f64 {
        c.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn already_at_vertex_is_stable() {
        // sum = 2 over [0,1]^3; (1,1,0) is a vertex.
        let a = [1.0, 1.0, 1.0];
        let c = [-1.0, -1.0, -1.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, 1.0];
        let x = [1.0, 1.0, 0.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let xv = crossover_to_vertex(&x, &lp, 1e-7, 0);
        for j in 0..3 {
            assert!((xv[j] - x[j]).abs() < 1e-6);
        }
    }

    #[test]
    fn pushes_interior_center_to_vertex() {
        // Optimal face is the whole polygon {sum = 2} ∩ [0,1]^3 (objective is
        // constant on it); crossover must reach a 0/1 vertex.
        let a = [1.0, 1.0, 1.0];
        let c = [-1.0, -1.0, -1.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, 1.0];
        let x = [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let xv = crossover_to_vertex(&x, &lp, 1e-7, 0);

        assert!((matvec(&a, 1, 3, &xv)[0] - 2.0).abs() < 1e-6); // feasibility
        assert!((dot(&c, &xv) - dot(&c, &x)).abs() < 1e-6); // objective
        assert!(is_vertex(&xv, &a, 1, 3, &c, &l, &u));
        for &xj in &xv {
            assert!(xj < 1e-6 || xj > 1.0 - 1e-6); // every coord at a bound
        }
    }

    #[test]
    fn unique_optimum_is_unchanged() {
        // min −2x0 − x1 − x2 s.t. sum = 2, [0,1]^3. The unique optimum puts
        // weight on x0 first: (1, 1, 0). It is already a vertex AND uniquely
        // optimal, so the free set is empty and crossover is a no-op.
        let a = [1.0, 1.0, 1.0];
        let c = [-2.0, -1.0, -1.0];
        let l = [0.0, 0.0, 0.0];
        let u = [1.0, 1.0, 1.0];
        let x = [1.0, 1.0, 0.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 3,
            c: &c,
            l: &l,
            u: &u,
        };
        let xv = crossover_to_vertex(&x, &lp, 1e-7, 0);
        for j in 0..3 {
            assert!((xv[j] - x[j]).abs() < 1e-6);
        }
    }

    #[test]
    fn size_guard_returns_unchanged() {
        let n = MAX_CROSSOVER_VARS + 5;
        let x = vec![0.5_f64; n];
        let l = vec![0.0_f64; n];
        let u = vec![1.0_f64; n];
        let c = vec![1.0_f64; n];
        let lp = LpView {
            a: &[],
            m: 0,
            n,
            c: &c,
            l: &l,
            u: &u,
        };
        let xv = crossover_to_vertex(&x, &lp, 1e-7, 0);
        assert_eq!(xv, x);
    }

    #[test]
    fn random_lps_preserve_objective_and_reach_vertex() {
        // Deterministic LCG; no external RNG dependency.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        for _ in 0..50 {
            let (m, n) = (2usize, 5usize);
            let a: Vec<f64> = (0..m * n).map(|_| next() * 2.0 - 1.0).collect();
            // Interior feasible start in (0.2, 0.8); b := A x_feas.
            let x_feas: Vec<f64> = (0..n).map(|_| 0.2 + 0.6 * next()).collect();
            let c: Vec<f64> = (0..n).map(|_| next() * 2.0 - 1.0).collect();
            let l = vec![0.0_f64; n];
            let u = vec![1.0_f64; n];

            let lp = LpView {
                a: &a,
                m,
                n,
                c: &c,
                l: &l,
                u: &u,
            };
            let xv = crossover_to_vertex(&x_feas, &lp, 1e-7, 0);

            assert!((dot(&c, &xv) - dot(&c, &x_feas)).abs() < 1e-5);
            let ax = matvec(&a, m, n, &xv);
            let b = matvec(&a, m, n, &x_feas);
            for i in 0..m {
                assert!((ax[i] - b[i]).abs() < 1e-5);
            }
            for &xj in &xv {
                assert!(xj >= -1e-6 && xj <= 1.0 + 1e-6);
            }
            assert!(is_vertex(&xv, &a, m, n, &c, &l, &u));
        }
    }
}
