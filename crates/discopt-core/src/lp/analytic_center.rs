//! Analytic centre of a standard-form polytope `{x : A x = b, l <= x <= u}`.
//!
//! The point maximizing `Σ log(x_j - l_j) + Σ log(u_j - x_j)` over the finite bounds,
//! subject to `A x = b`: the "middle" of the relaxation. HiGHS computes it with its
//! interior-point solver (IPX, zero objective, centring on, no crossover, at most 200
//! iterations; `HighsMipSolverData::startAnalyticCenterComputation`) and its central
//! rounding heuristic rounds along the segment from the root LP solution to it
//! (`HighsPrimalHeuristics::centralRounding` / `linesearchRounding`).
//!
//! This is not a translation of IPX. It is an infeasible-start Newton method on the
//! barrier (Boyd & Vandenberghe, *Convex Optimization*, §10.3), with the normal
//! equations `A D⁻¹ Aᵀ Δy = …` solved by feral's sparse symmetric `LDLᵀ`
//! (`feral::Solver`). That is all a centre needs: there is no objective to trade
//! against centrality, so no barrier parameter or path following.
//!
//! Only a heuristic consumes the result: every point built from it is re-verified
//! against the original rows before it can become an incumbent, so an inaccurate
//! centre costs solution quality, never soundness. `None` means "no centre": the
//! polytope looked empty or unbounded in a direction the bounds do not close, the
//! normal matrix was too dense to form, or the iteration or time budget ran out
//! before the residual was small.

use super::simplex::sparse::SparseCols;

/// The LP layer's infinity sentinel.
const INF: f64 = 1e20;

/// Newton iterations, as HiGHS's `ipm_iteration_limit = 200` for this solve.
pub const MAX_ITER: usize = 200;

/// Upper limit on the lower-triangle entries of `A D⁻¹ Aᵀ` the solve will form.
/// A column with `k` nonzeros contributes `k(k+1)/2`; a few dense columns make the
/// normal matrix dense, which the solve then declines rather than forming it.
const MAX_NORMAL_NNZ: usize = 20_000_000;

/// Weight given to a free column in `D` (it has no barrier term): keeps the
/// Newton system definite without pulling the column anywhere.
const FREE_CURVATURE: f64 = 1e-8;

/// Outcome of [`analytic_center`].
#[derive(Debug, Clone)]
pub struct Center {
    /// The centre, all `n` columns.
    pub x: Vec<f64>,
    /// Newton iterations taken.
    pub iterations: usize,
    /// Final `‖A x − b‖_∞`.
    pub primal_residual: f64,
}

fn fin(v: f64) -> f64 {
    if v <= -INF {
        f64::NEG_INFINITY
    } else if v >= INF {
        f64::INFINITY
    } else {
        v
    }
}

/// The analytic centre of `{A x = b, l <= x <= u}` (`A` is `m x n`, column-wise).
/// Fixed columns (`l = u`) stay at their value. See the module docs.
#[allow(clippy::too_many_arguments)]
pub fn analytic_center(
    a: &SparseCols,
    m: usize,
    n: usize,
    b: &[f64],
    l: &[f64],
    u: &[f64],
    max_iter: usize,
    deadline: Option<std::time::Instant>,
) -> Option<Center> {
    let lo: Vec<f64> = l[..n].iter().map(|&v| fin(v)).collect();
    let hi: Vec<f64> = u[..n].iter().map(|&v| fin(v)).collect();
    if (0..n).any(|j| lo[j] > hi[j]) {
        return None;
    }
    let fixed: Vec<bool> = (0..n).map(|j| lo[j] == hi[j]).collect();
    // Normal-matrix size guard.
    let mut tri = 0usize;
    for (j, &fx) in fixed.iter().enumerate() {
        if !fx {
            let k = a.col(j).0.len();
            tri = tri.saturating_add(k * (k + 1) / 2);
        }
    }
    if tri > MAX_NORMAL_NNZ {
        return None;
    }
    // Start: the box midpoint, a unit inside a one-sided bound, 0 when free.
    let mut x: Vec<f64> = (0..n)
        .map(|j| match (lo[j].is_finite(), hi[j].is_finite()) {
            _ if fixed[j] => lo[j],
            (true, true) => 0.5 * (lo[j] + hi[j]),
            (true, false) => lo[j] + 1.0_f64.max(lo[j].abs() * 1e-3),
            (false, true) => hi[j] - 1.0_f64.max(hi[j].abs() * 1e-3),
            (false, false) => 0.0,
        })
        .collect();
    let mut y = vec![0.0; m];
    let bnorm = b.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let tol = 1e-7 * (1.0 + bnorm);
    let mut solver = feral::Solver::new();
    let (mut rp, mut rd, mut d) = (vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    let residuals = |x: &[f64], y: &[f64], rp: &mut [f64], rd: &mut [f64], d: &mut [f64]| {
        rp.iter_mut().zip(b).for_each(|(r, &bi)| *r = -bi);
        for j in 0..n {
            let (rows, vals) = a.col(j);
            for (&i, &v) in rows.iter().zip(vals) {
                rp[i] += v * x[j];
            }
            if fixed[j] {
                rd[j] = 0.0;
                d[j] = 1.0;
                continue;
            }
            // f = -Σ log(x - l) - Σ log(u - x): gradient and curvature.
            let (mut g, mut h) = (0.0, 0.0);
            if lo[j].is_finite() {
                let s = x[j] - lo[j];
                g -= 1.0 / s;
                h += 1.0 / (s * s);
            }
            if hi[j].is_finite() {
                let s = hi[j] - x[j];
                g += 1.0 / s;
                h += 1.0 / (s * s);
            }
            if h == 0.0 {
                h = FREE_CURVATURE;
            }
            let aty: f64 = rows.iter().zip(vals).map(|(&i, &v)| v * y[i]).sum();
            rd[j] = g + aty;
            d[j] = h;
        }
    };
    let norm = |v: &[f64]| v.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    let mut iterations = 0usize;
    residuals(&x, &y, &mut rp, &mut rd, &mut d);
    while iterations < max_iter {
        if deadline.is_some_and(|t| std::time::Instant::now() >= t) {
            return None;
        }
        let (pr, dr) = (norm(&rp), norm(&rd));
        if pr <= tol && dr <= 1e-7 {
            break;
        }
        iterations += 1;
        // M = A D⁻¹ Aᵀ (lower triangle) plus a small regularization for rank
        // deficiency; rhs = rp - A D⁻¹ rd.
        let mut rows_t = Vec::with_capacity(tri + m);
        let mut cols_t = Vec::with_capacity(tri + m);
        let mut vals_t = Vec::with_capacity(tri + m);
        let mut rhs = rp.clone();
        let mut diag_max = 0.0f64;
        for j in 0..n {
            if fixed[j] {
                continue;
            }
            let w = 1.0 / d[j];
            let (rows, vals) = a.col(j);
            for (p, (&ip, &vp)) in rows.iter().zip(vals).enumerate() {
                rhs[ip] -= vp * w * rd[j];
                for (&iq, &vq) in rows[..=p].iter().zip(&vals[..=p]) {
                    let (r, c) = if ip >= iq { (ip, iq) } else { (iq, ip) };
                    rows_t.push(r);
                    cols_t.push(c);
                    let v = w * vp * vq;
                    vals_t.push(v);
                    if r == c {
                        diag_max = diag_max.max(v);
                    }
                }
            }
        }
        let reg = 1e-10 * (1.0 + diag_max);
        for i in 0..m {
            rows_t.push(i);
            cols_t.push(i);
            vals_t.push(reg);
        }
        let mat = feral::CscMatrix::from_triplets(m, &rows_t, &cols_t, &vals_t).ok()?;
        match solver.factor(&mat, None) {
            feral::FactorStatus::Success | feral::FactorStatus::WrongInertia { .. } => {}
            _ => return None,
        }
        let dy = solver.solve(&rhs).ok()?;
        // dx = -D⁻¹ (rd + Aᵀ dy).
        let mut dx = vec![0.0; n];
        for j in 0..n {
            if fixed[j] {
                continue;
            }
            let (rows, vals) = a.col(j);
            let atdy: f64 = rows.iter().zip(vals).map(|(&i, &v)| v * dy[i]).sum();
            dx[j] = -(rd[j] + atdy) / d[j];
        }
        // Fraction to the boundary, then backtrack on the residual norm.
        let mut t = 1.0f64;
        for j in 0..n {
            if dx[j] < 0.0 && lo[j].is_finite() {
                t = t.min(0.99 * (x[j] - lo[j]) / -dx[j]);
            }
            if dx[j] > 0.0 && hi[j].is_finite() {
                t = t.min(0.99 * (hi[j] - x[j]) / dx[j]);
            }
        }
        if t.is_nan() || t <= 0.0 {
            return None;
        }
        let r0 = pr.max(dr);
        let (mut rp1, mut rd1, mut d1) = (vec![0.0; m], vec![0.0; n], vec![0.0; n]);
        let mut accepted = false;
        for _ in 0..40 {
            let xt: Vec<f64> = x.iter().zip(&dx).map(|(xi, di)| xi + t * di).collect();
            let yt: Vec<f64> = y.iter().zip(&dy).map(|(yi, di)| yi + t * di).collect();
            residuals(&xt, &yt, &mut rp1, &mut rd1, &mut d1);
            if norm(&rp1).max(norm(&rd1)) <= (1.0 - 0.01 * t) * r0 {
                x = xt;
                y = yt;
                std::mem::swap(&mut rp, &mut rp1);
                std::mem::swap(&mut rd, &mut rd1);
                std::mem::swap(&mut d, &mut d1);
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let primal_residual = norm(&rp);
    (primal_residual <= tol.max(1e-6)).then_some(Center {
        x,
        iterations,
        primal_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The centre of a box with no rows is its midpoint; with `x0 + x1 = 1` on
    /// `[0, 1]²` it is `(1/2, 1/2)` by symmetry; a fixed column stays fixed.
    #[test]
    fn centre_of_symmetric_polytopes() {
        // x0 + x1 = 1, x2 fixed at 3 (no row), x3 in [0, 4].
        let sp = SparseCols::from_csc(vec![0, 1, 2, 2, 2], vec![0, 0], vec![1.0, 1.0]);
        let c = analytic_center(
            &sp,
            1,
            4,
            &[1.0],
            &[0.0, 0.0, 3.0, 0.0],
            &[1.0, 1.0, 3.0, 4.0],
            MAX_ITER,
            None,
        )
        .expect("a centre exists");
        assert!(
            (c.x[0] - 0.5).abs() < 1e-6 && (c.x[1] - 0.5).abs() < 1e-6,
            "{:?}",
            c.x
        );
        assert_eq!(c.x[2], 3.0);
        assert!((c.x[3] - 2.0).abs() < 1e-6);
        assert!(c.primal_residual < 1e-7);
    }

    /// An asymmetric triangle: `x0 + x1 + s = 1`, all three `>= 0` (`s` the slack).
    /// The centre of the simplex is its barycentre `(1/3, 1/3, 1/3)`.
    #[test]
    fn centre_of_the_simplex_is_its_barycentre() {
        let sp = SparseCols::from_csc(vec![0, 1, 2, 3], vec![0, 0, 0], vec![1.0, 1.0, 1.0]);
        let c = analytic_center(&sp, 1, 3, &[1.0], &[0.0; 3], &[INF; 3], MAX_ITER, None)
            .expect("a centre exists");
        for v in &c.x {
            assert!((v - 1.0 / 3.0).abs() < 1e-6, "{:?}", c.x);
        }
    }
}
