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

#[cfg(test)]
mod tests {
    use super::*;

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
}
