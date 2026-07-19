//! Convex LP-OA branch-and-cut kernel (issue #798).
//!
//! The SCIP/BARON-parity path for the convex MINLP family (`rsyn*`/`syn*`): an LP
//! relaxation at every node, cut into natively (Quesada–Grossmann LP/NLP-based
//! branch-and-cut), rather than an NLP per node. This module owns the node
//! relaxation: the outer-approximation (OA) LP over a box.
//!
//! ## Composite-of-affine convex rows (K1 architecture, entry-experiment-verified)
//!
//! Each convex nonlinear constraint is represented as
//! `g_i(x) = a_i·x + b_i + Σ_t coeff_t·func_t(p_t·x + q_t) ≤ rhs_i`, so the node
//! loop can evaluate `g_i(x)` and `∇g_i(x)` at LP vertices with only closed-form
//! univariate `f`/`f'` — no autodiff engine. The Python producer
//! (`issue798_convex_decompose_probe.py::decompose`) emits exactly this form; the
//! entry experiment verified the reconstruction reproduces the JAX evaluator's
//! value AND gradient to machine precision on all 4 convex panel instances.
//!
//! ## Soundness
//!
//! For a convex `g` and a `≤` row, the first-order linearization at any `x̄`,
//! `g(x̄) + ∇g(x̄)·(x − x̄) ≤ rhs`, is a valid relaxation cut (the tangent of a
//! convex function underestimates it, so every feasible point satisfies it). The
//! per-column convexity of each `coeff_t·func_t(affine)` term (convex func with
//! `coeff ≥ 0`, concave func with `coeff ≤ 0`) is certified UPSTREAM before an
//! instance is routed here — this module assumes the convex certificate, exactly
//! as the Python `_RootLP` root-cut path does. K1 is byte-checked against that
//! reference so any divergence is caught.

use crate::lp::simplex::refine::ns_safe_bound_csc;
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{primal::solve_lp_cols_scaled, LpStatus, SimplexOptions};

/// A univariate function admissible in a convex composite row, carrying its own
/// closed-form value and derivative. Extend as new convex-certifiable functions
/// appear; an unknown function makes the Python producer decline the instance
/// (→ NLP-BB fallback), so this enum need only cover what is actually routed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvexFunc {
    /// Natural logarithm `ln(t)` (concave; appears as `−c·ln(1+x)`, c>0 → convex).
    Log,
    /// Exponential `e^t` (convex).
    Exp,
    /// Square root `sqrt(t)` (concave).
    Sqrt,
    /// `ln(1 + t)` (concave).
    Log1p,
}

impl ConvexFunc {
    /// Evaluate `f(t)`.
    #[inline]
    pub fn eval(self, t: f64) -> f64 {
        match self {
            ConvexFunc::Log => t.ln(),
            ConvexFunc::Exp => t.exp(),
            ConvexFunc::Sqrt => t.sqrt(),
            ConvexFunc::Log1p => t.ln_1p(),
        }
    }

    /// Evaluate `(f(t), f'(t))` together — the OA-tangent primitive.
    #[inline]
    pub fn eval_and_deriv(self, t: f64) -> (f64, f64) {
        match self {
            ConvexFunc::Log => (t.ln(), 1.0 / t),
            ConvexFunc::Exp => {
                let e = t.exp();
                (e, e)
            }
            ConvexFunc::Sqrt => {
                let s = t.sqrt();
                (s, 0.5 / s)
            }
            ConvexFunc::Log1p => (t.ln_1p(), 1.0 / (1.0 + t)),
        }
    }
}

/// A sparse affine form `Σ coeffs[k]·x[cols[k]] + cst`.
#[derive(Clone, Debug, Default)]
pub struct Affine {
    /// Column indices of the affine form (need not be sorted or unique on input,
    /// but the producer emits them merged).
    pub cols: Vec<usize>,
    /// Coefficients aligned with `cols`.
    pub coeffs: Vec<f64>,
    /// Constant term.
    pub cst: f64,
}

impl Affine {
    /// Evaluate the affine form at `x`.
    #[inline]
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut v = self.cst;
        for (c, a) in self.cols.iter().zip(self.coeffs.iter()) {
            v += a * x[*c];
        }
        v
    }
}

/// One composite term `coeff · func(arg)` of a convex row.
#[derive(Clone, Debug)]
pub struct CompositeTerm {
    /// Outer coefficient.
    pub coeff: f64,
    /// The univariate function.
    pub func: ConvexFunc,
    /// The affine argument `p·x + q`.
    pub arg: Affine,
}

/// A convex nonlinear constraint `g(x) = lin·x + b + Σ_t coeff_t·func_t(arg_t) ≤ rhs`.
#[derive(Clone, Debug)]
pub struct ConvexRow {
    /// The affine part `lin·x + b`.
    pub lin: Affine,
    /// The composite univariate terms.
    pub terms: Vec<CompositeTerm>,
    /// Right-hand side of the `≤` constraint.
    pub rhs: f64,
}

/// A sparse linear inequality `Σ coeffs[k]·x[cols[k]] ≤ rhs` — the OA cut form.
#[derive(Clone, Debug)]
pub struct LinCut {
    /// Sorted, merged column indices.
    pub cols: Vec<usize>,
    /// Coefficients aligned with `cols`.
    pub coeffs: Vec<f64>,
    /// Right-hand side.
    pub rhs: f64,
}

impl ConvexRow {
    /// Evaluate `g(x)` (the constraint is `g(x) ≤ rhs`).
    pub fn value(&self, x: &[f64]) -> f64 {
        let mut v = self.lin.eval(x);
        for t in &self.terms {
            v += t.coeff * t.func.eval(t.arg.eval(x));
        }
        v
    }

    /// The constraint residual `g(x) − rhs` (`≤ 0` iff satisfied).
    #[inline]
    pub fn residual(&self, x: &[f64]) -> f64 {
        self.value(x) - self.rhs
    }

    /// Accumulate `∇g(x)` into a column→coefficient map.
    fn accumulate_gradient(&self, x: &[f64], grad: &mut std::collections::BTreeMap<usize, f64>) {
        for (c, a) in self.lin.cols.iter().zip(self.lin.coeffs.iter()) {
            *grad.entry(*c).or_insert(0.0) += a;
        }
        for t in &self.terms {
            let arg = t.arg.eval(x);
            let (_f, fp) = t.func.eval_and_deriv(arg);
            let outer = t.coeff * fp;
            for (c, a) in t.arg.cols.iter().zip(t.arg.coeffs.iter()) {
                *grad.entry(*c).or_insert(0.0) += outer * a;
            }
        }
    }

    /// `∇g(x)` as a dense vector of length `n`.
    pub fn gradient_dense(&self, x: &[f64], n: usize) -> Vec<f64> {
        let mut map = std::collections::BTreeMap::new();
        self.accumulate_gradient(x, &mut map);
        let mut g = vec![0.0; n];
        for (c, v) in map {
            g[c] = v;
        }
        g
    }

    /// The OA (first-order) tangent cut at `x̄`: the valid linear inequality
    /// `∇g(x̄)·x ≤ rhs − g(x̄) + ∇g(x̄)·x̄`. Returns `None` when the gradient or
    /// value is non-finite at `x̄` (e.g. `ln` of a non-positive argument), so the
    /// caller skips an ill-defined tangent rather than emit a NaN row.
    pub fn oa_tangent(&self, x: &[f64]) -> Option<LinCut> {
        let mut map = std::collections::BTreeMap::new();
        self.accumulate_gradient(x, &mut map);
        let g_val = self.value(x);
        if !g_val.is_finite() {
            return None;
        }
        let mut cols = Vec::with_capacity(map.len());
        let mut coeffs = Vec::with_capacity(map.len());
        let mut grad_dot_x = 0.0;
        for (c, v) in map {
            if !v.is_finite() {
                return None;
            }
            if v == 0.0 {
                continue;
            }
            grad_dot_x += v * x[c];
            cols.push(c);
            coeffs.push(v);
        }
        let rhs = self.rhs - g_val + grad_dot_x;
        if !rhs.is_finite() {
            return None;
        }
        Some(LinCut { cols, coeffs, rhs })
    }
}

/// A sparse linear row `Σ coeffs·x[cols]  {≤,=}  rhs`.
#[derive(Clone, Debug)]
pub struct LinRow {
    /// Column indices.
    pub cols: Vec<usize>,
    /// Coefficients aligned with `cols`.
    pub coeffs: Vec<f64>,
    /// Right-hand side.
    pub rhs: f64,
}

/// The analyze-once convex-kernel problem (marshaled from Python once per solve).
///
/// Objective sense is carried explicitly: the in-house simplex MINIMIZES, so a
/// `sense_max` model is solved by minimizing `−c·x` and the rigorous safe bound is
/// negated back to a valid UPPER bound on the maximization.
#[derive(Clone, Debug)]
pub struct ConvexKernelSpec {
    /// Structural column count.
    pub n: usize,
    /// Objective coefficients (declared model sense).
    pub c: Vec<f64>,
    /// `true` → maximize `c·x`; `false` → minimize.
    pub sense_max: bool,
    /// Per-column integrality.
    pub integrality: Vec<bool>,
    /// Global column lower bounds.
    pub lb: Vec<f64>,
    /// Global column upper bounds.
    pub ub: Vec<f64>,
    /// Linear `≤` rows.
    pub le_rows: Vec<LinRow>,
    /// Linear `=` rows.
    pub eq_rows: Vec<LinRow>,
    /// Convex nonlinear `≤` rows (outer-approximated by tangents).
    pub nl_rows: Vec<ConvexRow>,
}

/// Result of one convex node's LP-OA relaxation.
#[derive(Clone, Debug)]
pub struct ConvexNodeResult {
    /// LP status of the final OA relaxation solve.
    pub status: LpStatus,
    /// Rigorous dual bound in the MODEL's sense (a valid upper bound for a
    /// maximization, lower bound for a minimization) — the Neumaier–Shcherbina
    /// safe bound, negated for sense. `±inf` when uncertifiable (e.g. a nonzero
    /// reduced cost meets an infinite structural bound) or the relaxation is
    /// infeasible — sound (never fathoms). This is what the tree fathoms on.
    pub bound: f64,
    /// The RAW simplex LP optimum in the model's sense (not directed-rounded).
    /// Diagnostic / byte-check use only — NEVER fathom on this (it can drift above
    /// the true optimum on an ill-conditioned basis). Equals `bound` up to the NS
    /// safety margin when the bound certifies.
    pub raw_bound: f64,
    /// Structural primal point at the final OA vertex (length `n`).
    pub x: Vec<f64>,
    /// Number of OA rounds (LP solves) performed.
    pub oa_rounds: usize,
    /// Number of OA tangent cuts accumulated.
    pub n_tangents: usize,
}

/// One row of the standard-form assembly, tagged by sense.
struct AsmRow {
    cols: Vec<usize>,
    coeffs: Vec<f64>,
    rhs: f64,
    is_eq: bool,
}

/// Assemble the standard-form node LP `[A | I] z = b` over the box `(lo, hi)`.
///
/// Every row gets an explicit slack: `≤` rows a slack in `[0, cap]` (cap from row
/// min-activity, keeping the Neumaier–Shcherbina term finite — the tanksize
/// certification lesson), `=` rows a slack fixed at `0`. Returns
/// `(sp, m, n_total, b, l, u)`.
fn assemble(
    n_cols: usize,
    lo: &[f64],
    hi: &[f64],
    rows: &[AsmRow],
) -> (SparseCols, usize, usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = rows.len();
    let n_total = n_cols + m;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_total];
    let mut b = vec![0.0f64; m];
    for (r, row) in rows.iter().enumerate() {
        b[r] = row.rhs;
        for (c, v) in row.cols.iter().zip(row.coeffs.iter()) {
            debug_assert!(*c < n_cols, "row references column out of range");
            cols[*c].push((r, *v));
        }
        cols[n_cols + r].push((r, 1.0)); // slack
    }
    let mut col_ptr = Vec::with_capacity(n_total + 1);
    let mut row_idx = Vec::new();
    let mut vals = Vec::new();
    col_ptr.push(0usize);
    for col in &cols {
        for (r, v) in col {
            row_idx.push(*r);
            vals.push(*v);
        }
        col_ptr.push(row_idx.len());
    }
    let sp = SparseCols::from_csc(col_ptr, row_idx, vals);

    let mut l = lo.to_vec();
    let mut u = hi.to_vec();
    l.extend(std::iter::repeat(0.0).take(m));
    for row in rows.iter() {
        if row.is_eq {
            u.push(0.0); // slack fixed at 0 → equality
            continue;
        }
        // ≤ row: finite slack cap from min-activity (keeps NS bound finite).
        let mut min_act = 0.0f64;
        for (c, v) in row.cols.iter().zip(row.coeffs.iter()) {
            min_act += if *v > 0.0 { v * l[*c] } else { v * u[*c] };
        }
        let cap = if min_act.is_finite() {
            (row.rhs - min_act).max(0.0)
        } else {
            1e20
        };
        u.push(cap.min(1e20));
    }
    (sp, m, n_total, b, l, u)
}

impl ConvexKernelSpec {
    /// Solve the LP-OA node relaxation over the box `(lo, hi)` (length `n`):
    /// linear rows + OA tangents refined to OA convergence, solved by the warm
    /// in-house simplex. Returns the rigorous dual bound in the model's sense.
    ///
    /// `oa_tol`: nonlinear-residual tolerance to add a tangent. `max_oa_rounds`:
    /// safety cap on the OA loop.
    pub fn solve_node(
        &self,
        lo: &[f64],
        hi: &[f64],
        oa_tol: f64,
        max_oa_rounds: usize,
        opts: &SimplexOptions,
    ) -> ConvexNodeResult {
        debug_assert_eq!(lo.len(), self.n);
        debug_assert_eq!(hi.len(), self.n);
        // Objective: minimize c·x (min sense) or −c·x (max sense).
        let sign = if self.sense_max { -1.0 } else { 1.0 };

        let mut tangents: Vec<LinRow> = Vec::new();
        let mut last_x = vec![0.0f64; self.n];
        let mut status = LpStatus::Optimal;
        let mut bound = f64::NEG_INFINITY;
        let mut raw_bound = f64::NEG_INFINITY;
        let mut rounds = 0usize;

        for _ in 0..max_oa_rounds.max(1) {
            rounds += 1;
            // Build the row list: ≤ (linear + tangents), then = rows.
            let mut rows: Vec<AsmRow> =
                Vec::with_capacity(self.le_rows.len() + tangents.len() + self.eq_rows.len());
            for r in self.le_rows.iter().chain(tangents.iter()) {
                rows.push(AsmRow {
                    cols: r.cols.clone(),
                    coeffs: r.coeffs.clone(),
                    rhs: r.rhs,
                    is_eq: false,
                });
            }
            for r in &self.eq_rows {
                rows.push(AsmRow {
                    cols: r.cols.clone(),
                    coeffs: r.coeffs.clone(),
                    rhs: r.rhs,
                    is_eq: true,
                });
            }
            let (sp, m, n_total, b, l, u) = assemble(self.n, lo, hi, &rows);
            let mut c = vec![0.0f64; n_total];
            for (cj, sc) in c.iter_mut().zip(self.c.iter()) {
                *cj = sign * sc;
            }
            let sol = solve_lp_cols_scaled(sp.clone(), m, n_total, &c, &l, &u, &b, opts);
            status = sol.status;
            if status != LpStatus::Optimal {
                // Non-Optimal relaxation → return a sound model-sense sentinel that
                // never falsely survives fathoming. Callers key off `status` and
                // treat any non-Optimal node as "skip region".
                //   Infeasible: the box is empty → prune. Report the WORST bound
                //     (−inf for max / +inf for min) so it contributes nothing.
                //   Unbounded/Numerical: no certified bound → report the trivial
                //     bound (+inf for max / −inf for min) so it never fathoms.
                let sentinel = match (status, self.sense_max) {
                    (LpStatus::Infeasible, true) => f64::NEG_INFINITY,
                    (LpStatus::Infeasible, false) => f64::INFINITY,
                    (_, true) => f64::INFINITY,
                    (_, false) => f64::NEG_INFINITY,
                };
                last_x.iter_mut().for_each(|v| *v = 0.0);
                return ConvexNodeResult {
                    status,
                    bound: sentinel,
                    raw_bound: sentinel,
                    x: last_x,
                    oa_rounds: rounds,
                    n_tangents: tangents.len(),
                };
            }
            last_x.copy_from_slice(&sol.x[..self.n]);
            // Raw LP optimum in the model sense (diagnostic; never fathom on it).
            raw_bound = sign * sol.obj;

            // Rigorous safe bound on min(c·x) from the row duals; negate for sense.
            let (col_ptr, row_idx, vals) = sp.raw();
            let safe_min = ns_safe_bound_csc(
                &sol.dual, &c, col_ptr, row_idx, vals, m, n_total, &b, &l, &u,
            );
            bound = match safe_min {
                Some(v) => sign * v, // max: −(safe lower on −c·x) = safe upper on c·x
                None => {
                    if self.sense_max {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            };

            // Separate OA tangents for violated convex rows at this vertex.
            let mut added = false;
            for row in &self.nl_rows {
                if row.residual(&last_x) > oa_tol {
                    if let Some(cut) = row.oa_tangent(&last_x) {
                        tangents.push(LinRow {
                            cols: cut.cols,
                            coeffs: cut.coeffs,
                            rhs: cut.rhs,
                        });
                        added = true;
                    }
                }
            }
            if !added {
                break;
            }
        }

        ConvexNodeResult {
            status,
            bound,
            raw_bound,
            x: last_x,
            oa_rounds: rounds,
            n_tangents: tangents.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::simplex::SimplexOptions;

    const N: usize = 4;

    fn fd_deriv(f: impl Fn(f64) -> f64, t: f64) -> f64 {
        let h = 1e-6 * t.abs().max(1.0);
        (f(t + h) - f(t - h)) / (2.0 * h)
    }

    #[test]
    fn func_values_and_derivs_match_finite_difference() {
        let cases = [
            (ConvexFunc::Log, 2.3),
            (ConvexFunc::Exp, 0.7),
            (ConvexFunc::Sqrt, 3.1),
            (ConvexFunc::Log1p, 1.4),
        ];
        for (func, t) in cases {
            let (f, fp) = func.eval_and_deriv(t);
            assert!((f - func.eval(t)).abs() < 1e-15, "{func:?} eval mismatch");
            let fp_fd = fd_deriv(|z| func.eval(z), t);
            assert!(
                (fp - fp_fd).abs() < 1e-5,
                "{func:?} deriv {fp} vs fd {fp_fd}"
            );
        }
    }

    #[test]
    fn known_values() {
        assert!((ConvexFunc::Log.eval(1.0)).abs() < 1e-15);
        assert!((ConvexFunc::Exp.eval(0.0) - 1.0).abs() < 1e-15);
        assert!((ConvexFunc::Sqrt.eval(4.0) - 2.0).abs() < 1e-15);
        assert!((ConvexFunc::Log1p.eval(0.0)).abs() < 1e-15);
    }

    /// The panel's row shape: `−c·ln(1 + x0) + x1 + x2 − 1 ≤ 0`.
    fn panel_row() -> ConvexRow {
        ConvexRow {
            lin: Affine {
                cols: vec![1, 2],
                coeffs: vec![1.0, 1.0],
                cst: -1.0,
            },
            terms: vec![CompositeTerm {
                coeff: -1.2,
                func: ConvexFunc::Log,
                arg: Affine {
                    cols: vec![0],
                    coeffs: vec![1.0],
                    cst: 1.0,
                },
            }],
            rhs: 0.0,
        }
    }

    #[test]
    fn value_and_gradient_are_correct() {
        let row = panel_row();
        let x = [3.0, 0.5, 0.25, 0.0];
        // g = -1.2*ln(1+3) + 0.5 + 0.25 - 1
        let expect = -1.2 * (4.0_f64).ln() + 0.5 + 0.25 - 1.0;
        assert!((row.value(&x) - expect).abs() < 1e-12);
        // ∇g: x0 = -1.2 * 1/(1+x0) = -1.2/4 = -0.3 ; x1 = 1 ; x2 = 1
        let g = row.gradient_dense(&x, N);
        assert!((g[0] - (-0.3)).abs() < 1e-12, "g0={}", g[0]);
        assert!((g[1] - 1.0).abs() < 1e-12);
        assert!((g[2] - 1.0).abs() < 1e-12);
        assert!(g[3].abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_finite_difference_multivar() {
        let row = panel_row();
        let x = [3.0, 0.5, 0.25, 0.0];
        let g = row.gradient_dense(&x, N);
        for j in 0..N {
            let h = 1e-6;
            let mut xp = x;
            let mut xm = x;
            xp[j] += h;
            xm[j] -= h;
            let fd = (row.value(&xp) - row.value(&xm)) / (2.0 * h);
            assert!((g[j] - fd).abs() < 1e-5, "col {j}: {} vs fd {fd}", g[j]);
        }
    }

    #[test]
    fn oa_tangent_is_valid_and_tight_at_the_point() {
        let row = panel_row();
        let xbar = [3.0, 0.5, 0.25, 0.0];
        let cut = row.oa_tangent(&xbar).expect("finite tangent");
        // At x̄ the tangent is exact: a·x̄ == rhs_cut − (g(x̄) − rhs_row).
        let a_dot_xbar: f64 = cut
            .cols
            .iter()
            .zip(cut.coeffs.iter())
            .map(|(c, a)| a * xbar[*c])
            .sum();
        // residual of the cut at x̄ equals the constraint residual g(x̄)-rhs.
        let cut_resid = a_dot_xbar - cut.rhs;
        assert!(
            (cut_resid - row.residual(&xbar)).abs() < 1e-12,
            "cut not tight at x̄"
        );
        // Convex underestimator: the tangent must lie BELOW g everywhere, i.e.
        // for any x, a·x - rhs_cut <= g(x) - rhs_row. Check at random points.
        for k in 0..20 {
            let d = (k as f64) * 0.1;
            let x = [3.0 + d, 0.5 - 0.01 * d, 0.25 + 0.02 * d, d];
            let a_dot_x: f64 = cut
                .cols
                .iter()
                .zip(cut.coeffs.iter())
                .map(|(c, a)| a * x[*c])
                .sum();
            let lhs = a_dot_x - cut.rhs; // tangent residual
            let rhs = row.residual(&x); // true residual
            assert!(
                lhs <= rhs + 1e-9,
                "tangent overestimates g at x={x:?}: {lhs} > {rhs}"
            );
        }
    }

    fn opts() -> SimplexOptions {
        SimplexOptions {
            expel_zero_artificials: true,
            ..Default::default()
        }
    }

    /// K1c: the OA loop converges to the true convex bound. Model:
    /// `max t  s.t.  exp(t) ≤ 5,  t ∈ [0, 10]` → optimum `t* = ln 5 ≈ 1.6094`.
    #[test]
    fn node_oa_loop_converges_to_convex_optimum() {
        let spec = ConvexKernelSpec {
            n: 1,
            c: vec![1.0],
            sense_max: true,
            integrality: vec![false],
            lb: vec![0.0],
            ub: vec![10.0],
            le_rows: vec![],
            eq_rows: vec![],
            nl_rows: vec![ConvexRow {
                lin: Affine::default(),
                terms: vec![CompositeTerm {
                    coeff: 1.0,
                    func: ConvexFunc::Exp,
                    arg: Affine {
                        cols: vec![0],
                        coeffs: vec![1.0],
                        cst: 0.0,
                    },
                }],
                rhs: 5.0,
            }],
        };
        let r = spec.solve_node(&[0.0], &[10.0], 1e-9, 60, &opts());
        assert_eq!(r.status, LpStatus::Optimal);
        let truth = 5.0_f64.ln();
        // OA outer-approximates from OUTSIDE → bound is a valid UPPER bound on the
        // max, converging DOWN to truth. Must be ≥ truth (sound) and close.
        assert!(
            r.bound >= truth - 1e-6,
            "bound {} below truth {}",
            r.bound,
            truth
        );
        assert!(
            r.bound <= truth + 1e-4,
            "bound {} not converged to truth {}",
            r.bound,
            truth
        );
        assert!(r.oa_rounds > 1, "OA should take several rounds");
    }

    /// K1c: a linear-only node reproduces the LP optimum exactly (no OA rounds
    /// beyond the first), and the safe bound is a valid upper bound.
    #[test]
    fn node_linear_only_matches_lp() {
        // max 2·x0 + 3·x1  s.t.  x0 + x1 ≤ 4,  x0,x1 ∈ [0,3]  → x0=1,x1=3, obj=11.
        let spec = ConvexKernelSpec {
            n: 2,
            c: vec![2.0, 3.0],
            sense_max: true,
            integrality: vec![false, false],
            lb: vec![0.0, 0.0],
            ub: vec![3.0, 3.0],
            le_rows: vec![LinRow {
                cols: vec![0, 1],
                coeffs: vec![1.0, 1.0],
                rhs: 4.0,
            }],
            eq_rows: vec![],
            nl_rows: vec![],
        };
        let r = spec.solve_node(&[0.0, 0.0], &[3.0, 3.0], 1e-9, 10, &opts());
        assert_eq!(r.status, LpStatus::Optimal);
        assert!((r.bound - 11.0).abs() < 1e-6, "bound {} != 11", r.bound);
        assert_eq!(r.oa_rounds, 1, "linear node needs one solve");
    }
}
