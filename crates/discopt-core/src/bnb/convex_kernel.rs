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

use crate::bnb::branching::Pseudocosts;
use crate::lp::basis::{Basis, BASIC};
use crate::lp::cover::separate_cover_csc;
use crate::lp::crossover::LpView;
use crate::lp::cut_select::select_cuts;
use crate::lp::gomory::{separate_gomory_cols, GomoryCut};
use crate::lp::mir::separate_mir;
use crate::lp::simplex::refine::ns_safe_bound_csc;
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{
    primal::solve_lp_cols_scaled, solve_lp_warm_scaled_csc, LpStatus, SimplexOptions,
};

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
    /// safety cap on the OA loop. This is the K1 (OA-only) node relaxation — no
    /// integrality separation. See [`solve_node_cut`](Self::solve_node_cut) for K2.
    pub fn solve_node(
        &self,
        lo: &[f64],
        hi: &[f64],
        oa_tol: f64,
        max_oa_rounds: usize,
        opts: &SimplexOptions,
    ) -> ConvexNodeResult {
        self.solve_node_cut(lo, hi, oa_tol, max_oa_rounds, 0, opts)
    }

    /// K2: the LP-OA node relaxation WITH in-tree integrality separation. Runs
    /// OA to convergence, then separates GMI + knapsack-cover cuts into the node
    /// LP under efficacy×orthogonality selection, re-converging OA, up to
    /// `max_sep_rounds`. Cuts are node-local (never shared across siblings — the
    /// C-43 lesson) and rewritten over structural columns by substituting the
    /// slacks `s_r = b_r − A_r·x`, so each is a valid inequality for the node's
    /// integer-feasible set.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_node_cut(
        &self,
        lo: &[f64],
        hi: &[f64],
        oa_tol: f64,
        max_oa_rounds: usize,
        max_sep_rounds: usize,
        opts: &SimplexOptions,
    ) -> ConvexNodeResult {
        debug_assert_eq!(lo.len(), self.n);
        debug_assert_eq!(hi.len(), self.n);
        let sign = if self.sense_max { -1.0 } else { 1.0 };
        let trivial = if self.sense_max {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };

        // Base rows: `le` (≤) then `eq` (=). OA tangents and cuts are APPENDED
        // after the base, so structural and base-slack columns never move and the
        // optimal basis carries (extended) across every re-solve — the wall lever.
        let mut rows: Vec<AsmRow> = Vec::with_capacity(self.le_rows.len() + self.eq_rows.len());
        for r in &self.le_rows {
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
        let n_base = rows.len();

        let mut basis: Option<Basis> = None;
        let mut last_x = vec![0.0f64; self.n];
        let mut last_bound = trivial;
        let mut last_raw = trivial;
        let mut oa_rounds = 0usize;
        let mut sep_used = 0usize;
        // Generous safety cap on total (OA + separation) re-solves; normal
        // convergence exits far sooner.
        let iter_cap = max_oa_rounds.max(1) * (max_sep_rounds + 1) + max_sep_rounds + 4;

        for _ in 0..iter_cap {
            oa_rounds += 1;
            let (sp, m, n_total, b, l, u) = assemble(self.n, lo, hi, &rows);
            let mut c = vec![0.0f64; n_total];
            for (cj, sc) in c.iter_mut().zip(self.c.iter()) {
                *cj = sign * sc;
            }
            // Warm-start from the carried basis (extended so the appended rows'
            // slacks are basic); cold+scaled on the first solve. The warm path
            // equilibrates and falls back to a cold solve when the basis is
            // unusable, so the result is always correct — only faster.
            let sol = match &basis {
                Some(bs) => {
                    let ext = extend_basis(bs.clone(), n_total);
                    let lp = LpView {
                        a: &[],
                        m,
                        n: n_total,
                        c: &c,
                        l: &l,
                        u: &u,
                    };
                    solve_lp_warm_scaled_csc(&lp, &b, &ext, opts, &sp)
                }
                None => solve_lp_cols_scaled(sp.clone(), m, n_total, &c, &l, &u, &b, opts),
            };
            if sol.status != LpStatus::Optimal {
                // Sound model-sense sentinel — never falsely survives fathoming.
                let sentinel = match (sol.status, self.sense_max) {
                    (LpStatus::Infeasible, true) => f64::NEG_INFINITY,
                    (LpStatus::Infeasible, false) => f64::INFINITY,
                    (_, true) => f64::INFINITY,
                    (_, false) => f64::NEG_INFINITY,
                };
                return ConvexNodeResult {
                    status: sol.status,
                    bound: sentinel,
                    raw_bound: sentinel,
                    x: vec![0.0; self.n],
                    oa_rounds,
                    n_tangents: rows.len() - n_base,
                };
            }
            basis = Some(sol.basis.clone());
            last_x.copy_from_slice(&sol.x[..self.n]);
            last_raw = sign * sol.obj;
            let (col_ptr, row_idx, vals) = sp.raw();
            let safe_min = ns_safe_bound_csc(
                &sol.dual, &c, col_ptr, row_idx, vals, m, n_total, &b, &l, &u,
            );
            last_bound = match safe_min {
                Some(v) => sign * v,
                None if self.sense_max => f64::INFINITY,
                None => f64::NEG_INFINITY,
            };

            // 1. OA tangents for violated convex rows at this vertex (append).
            let mut added = false;
            for row in &self.nl_rows {
                if row.residual(&last_x) > oa_tol {
                    if let Some(cut) = row.oa_tangent(&last_x) {
                        rows.push(AsmRow {
                            cols: cut.cols,
                            coeffs: cut.coeffs,
                            rhs: cut.rhs,
                            is_eq: false,
                        });
                        added = true;
                    }
                }
            }
            if added {
                continue; // OA not yet converged
            }
            // 2. OA converged. Separate integrality cuts if budget remains.
            if sep_used >= max_sep_rounds {
                break;
            }
            sep_used += 1;
            let mut is_int_full = vec![false; n_total];
            is_int_full[..self.n].copy_from_slice(&self.integrality);
            let mut raw = separate_cover_csc(
                &sp,
                n_total,
                m,
                &l,
                &u,
                &b,
                &sol.x,
                self.n,
                &is_int_full,
                self.le_rows.len(),
                opts.tol,
            );
            raw.extend(separate_gomory_cols(
                &sp,
                m,
                n_total,
                &l,
                &u,
                &b,
                &sol.basis,
                &is_int_full,
                opts.tol,
                1e7,
            ));
            // MIR (c-MIR family) over the structural ≤ rows — the lever GMI+cover
            // leave open (measured: closes ~2× more of syn40m's root gap). MIR cuts
            // are structural; express in the standard-form ≥ convention so they
            // flow through select_cuts + substitute_slacks with GMI/cover.
            {
                let mut a_ub: Vec<f64> = Vec::new();
                let mut b_ub: Vec<f64> = Vec::new();
                for row in rows.iter().filter(|r| !r.is_eq) {
                    let mut dense = vec![0.0f64; self.n];
                    for (cc, vv) in row.cols.iter().zip(row.coeffs.iter()) {
                        dense[*cc] = *vv;
                    }
                    a_ub.extend_from_slice(&dense);
                    b_ub.push(row.rhs);
                }
                if !b_ub.is_empty() {
                    for mc in separate_mir(
                        &a_ub,
                        &b_ub,
                        &l[..self.n],
                        &u[..self.n],
                        &self.integrality,
                        &sol.x[..self.n],
                        opts.tol,
                        1e7,
                    ) {
                        let mut coeffs = vec![0.0f64; n_total];
                        for (j, &v) in mc.coeffs.iter().enumerate() {
                            coeffs[j] = -v;
                        }
                        raw.push(GomoryCut {
                            coeffs,
                            rhs: -mc.rhs,
                        });
                    }
                }
            }
            if raw.is_empty() {
                break;
            }
            let selected = select_cuts(raw, &sol.x, 8, 1e-4, 0.90);
            if selected.is_empty() {
                break;
            }
            let mut added_cuts = 0usize;
            for gc in &selected {
                if let Some(lin) = substitute_slacks(gc, &rows, self.n) {
                    rows.push(AsmRow {
                        cols: lin.cols,
                        coeffs: lin.coeffs,
                        rhs: lin.rhs,
                        is_eq: false,
                    });
                    added_cuts += 1;
                }
            }
            if added_cuts == 0 {
                break;
            }
        }

        ConvexNodeResult {
            status: LpStatus::Optimal,
            bound: last_bound,
            raw_bound: last_raw,
            x: last_x,
            oa_rounds,
            n_tangents: rows.len() - n_base,
        }
    }
}

/// Extend a stored basis to `n_total` columns by making the appended slack
/// columns basic — a valid, dual-repairable warm start after rows/cols were
/// appended (each new column is its row's slack). No-op when already spanning.
fn extend_basis(mut basis: Basis, n_total: usize) -> Basis {
    for j in basis.col_status.len()..n_total {
        basis.col_status.push(BASIC);
        basis.basic_vars.push(j);
    }
    basis
}

/// Rewrite a standard-form cut `Σ coeffs·z ≥ rhs` (z = structural ‖ slacks) over
/// structural columns only, by substituting `s_r = b_r − A_r·x` for each row's
/// slack, then flip to `≤` form for a [`LinRow`]. Returns `None` if the resulting
/// row is empty or non-finite. Sound: `s_r = b_r − A_r·x` is an identity on the
/// feasible region, so the substituted structural cut holds for every
/// integer-feasible `x` the original cut was valid for.
fn substitute_slacks(gc: &GomoryCut, rows: &[AsmRow], n_struct: usize) -> Option<LinRow> {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
    // Structural part (≥ form).
    for (j, &v) in gc.coeffs.iter().take(n_struct).enumerate() {
        if v != 0.0 {
            *acc.entry(j).or_insert(0.0) += v;
        }
    }
    let mut rhs_ge = gc.rhs;
    // Slack part: column n_struct + r corresponds to rows[r].
    for (r, row) in rows.iter().enumerate() {
        let cs = gc.coeffs.get(n_struct + r).copied().unwrap_or(0.0);
        if cs == 0.0 {
            continue;
        }
        rhs_ge -= cs * row.rhs;
        for (c, v) in row.cols.iter().zip(row.coeffs.iter()) {
            *acc.entry(*c).or_insert(0.0) -= cs * v;
        }
    }
    // acc·x ≥ rhs_ge  →  (−acc)·x ≤ −rhs_ge.
    let mut cols = Vec::with_capacity(acc.len());
    let mut coeffs = Vec::with_capacity(acc.len());
    for (c, v) in acc {
        if v == 0.0 || !v.is_finite() {
            if !v.is_finite() {
                return None;
            }
            continue;
        }
        cols.push(c);
        coeffs.push(-v);
    }
    if cols.is_empty() || !rhs_ge.is_finite() {
        return None;
    }
    Some(LinRow {
        cols,
        coeffs,
        rhs: -rhs_ge,
    })
}

// ── K2c: best-bound branch-and-cut tree ──────────────────────────────────────

/// Terminal state of the convex kernel tree. `Optimal` is emitted ONLY when the
/// dual bound has closed onto the incumbent within `gap_tol` — a residual gap
/// yields `Exhausted`/`NodeLimit`/`TimeLimit`, never a falsely-upgraded optimum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvexTreeStatus {
    /// Certified optimal (dual bound closed onto the incumbent).
    Optimal,
    /// Node budget exhausted with a residual gap.
    NodeLimit,
    /// Wall-clock deadline hit with a residual gap.
    TimeLimit,
    /// Tree fully explored but the residual gap did not close to `gap_tol`
    /// (numerical), or no incumbent was found.
    Exhausted,
    /// The root relaxation is infeasible.
    Infeasible,
}

/// Configuration for [`ConvexKernelSpec::solve_tree`].
#[derive(Clone, Debug)]
pub struct ConvexTreeConfig {
    /// Node budget.
    pub max_nodes: usize,
    /// Relative optimality gap tolerance.
    pub gap_tol: f64,
    /// Integrality tolerance.
    pub int_tol: f64,
    /// OA nonlinear-residual tolerance.
    pub oa_tol: f64,
    /// Per-node OA round cap.
    pub max_oa_rounds: usize,
    /// Per-node separation round cap (0 = no in-tree cutting).
    pub max_sep_rounds: usize,
    /// FBBT propagation round cap per node.
    pub fbbt_rounds: usize,
    /// Optional wall-clock deadline.
    pub deadline: Option<std::time::Instant>,
    /// Optional known-feasible incumbent objective (model sense) to seed pruning.
    pub initial_incumbent: Option<f64>,
}

impl Default for ConvexTreeConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            gap_tol: 1e-4,
            int_tol: 1e-5,
            oa_tol: 1e-6,
            max_oa_rounds: 60,
            max_sep_rounds: 12,
            fbbt_rounds: 20,
            deadline: None,
            initial_incumbent: None,
        }
    }
}

/// Result of a convex kernel tree solve.
#[derive(Clone, Debug)]
pub struct ConvexTreeResult {
    /// Terminal status.
    pub status: ConvexTreeStatus,
    /// Best feasible objective (model sense), or `None` if none was found.
    pub incumbent: Option<f64>,
    /// Structural point of the incumbent (empty if none).
    pub incumbent_x: Vec<f64>,
    /// Best dual bound in the model sense (upper bound for max, lower for min).
    pub bound: f64,
    /// Nodes processed.
    pub node_count: usize,
}

/// A worklist node: the box `(lo, hi)`, the parent's rigorous dual bound, and the
/// branching that created it — `(var, frac_at_parent, is_down)` — used to update
/// pseudocosts once this node's own bound is known.
struct TreeNode {
    key: f64, // best-bound priority: pop the most promising node first (max-heap).
    parent_bound: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    branch: Option<(usize, f64, bool)>,
}
impl PartialEq for TreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for TreeNode {}
impl PartialOrd for TreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TreeNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.total_cmp(&other.key)
    }
}

impl ConvexKernelSpec {
    /// Interval FBBT over the linear rows (`≤` and `=`, both directions) plus
    /// integer rounding, tightening `(lo, hi)` in place to a fixpoint. Returns
    /// `false` if the box is proven empty. REQUIRED before the node LP: it makes
    /// the structural bounds finite so the Neumaier–Shcherbina safe bound
    /// certifies (an infinite structural bound meeting a roundoff reduced cost
    /// makes NS decline — the K1d finding).
    pub fn propagate_fbbt(&self, lo: &mut [f64], hi: &mut [f64], max_rounds: usize) -> bool {
        // Each pass tightens `a·x ≤ rhs` rows; equalities contribute both `a·x ≤ b`
        // and `−a·x ≤ −b`.
        let apply = |lo: &mut [f64],
                     hi: &mut [f64],
                     cols: &[usize],
                     coeffs: &[f64],
                     rhs: f64|
         -> Option<bool> {
            // Returns Some(changed) or None if the box is proven empty.
            let mut changed = false;
            for (jpos, &j) in cols.iter().enumerate() {
                let aj = coeffs[jpos];
                if aj == 0.0 {
                    continue;
                }
                // rest = Σ_{k≠j} min contribution (a_k>0 → lo, else hi).
                let mut rest = 0.0;
                let mut ok = true;
                for (kpos, &k) in cols.iter().enumerate() {
                    if kpos == jpos {
                        continue;
                    }
                    let ak = coeffs[kpos];
                    let v = if ak > 0.0 { lo[k] } else { hi[k] };
                    if !v.is_finite() {
                        ok = false;
                        break;
                    }
                    rest += ak * v;
                }
                if !ok {
                    continue;
                }
                let bnd = (rhs - rest) / aj;
                if aj > 0.0 {
                    let nb = if self.integrality[j] {
                        (bnd + 1e-9).floor()
                    } else {
                        bnd
                    };
                    if nb < hi[j] - 1e-9 {
                        hi[j] = nb;
                        changed = true;
                    }
                } else {
                    let nb = if self.integrality[j] {
                        (bnd - 1e-9).ceil()
                    } else {
                        bnd
                    };
                    if nb > lo[j] + 1e-9 {
                        lo[j] = nb;
                        changed = true;
                    }
                }
                if lo[j] > hi[j] + 1e-7 {
                    return None;
                }
            }
            Some(changed)
        };

        for _ in 0..max_rounds.max(1) {
            let mut changed = false;
            for row in &self.le_rows {
                match apply(lo, hi, &row.cols, &row.coeffs, row.rhs) {
                    None => return false,
                    Some(c) => changed |= c,
                }
            }
            for row in &self.eq_rows {
                match apply(lo, hi, &row.cols, &row.coeffs, row.rhs) {
                    None => return false,
                    Some(c) => changed |= c,
                }
                let neg: Vec<f64> = row.coeffs.iter().map(|v| -v).collect();
                match apply(lo, hi, &row.cols, &neg, -row.rhs) {
                    None => return false,
                    Some(c) => changed |= c,
                }
            }
            if !changed {
                break;
            }
        }
        // Final integer rounding on the box.
        for ((lj, hj), &is_int) in lo
            .iter_mut()
            .zip(hi.iter_mut())
            .zip(self.integrality.iter())
        {
            if is_int {
                *lj = (*lj - 1e-9).ceil();
                *hj = (*hj + 1e-9).floor();
                if *lj > *hj + 1e-7 {
                    return false;
                }
            }
        }
        true
    }

    /// Is `x` integer-integral on all integer columns AND OA-tight (every convex
    /// row satisfied to `oa_tol`)? Such an LP vertex is genuinely feasible → a
    /// valid incumbent (the minimal LP-NLP-BB primal; K3 adds NLP/rounding).
    fn is_integer_feasible(&self, x: &[f64], int_tol: f64, oa_tol: f64) -> bool {
        for (&is_int, &xj) in self.integrality.iter().zip(x.iter()) {
            if is_int && (xj - xj.round()).abs() > int_tol {
                return false;
            }
        }
        self.nl_rows.iter().all(|r| r.residual(x) <= oa_tol)
    }

    /// The fractional integer column with the highest pseudocost score (SCIP
    /// product rule); falls back toward most-fractional for unobserved variables
    /// (their default pseudocost makes the score track fractionality). `None` if
    /// all integers are integral.
    fn select_branch(&self, x: &[f64], pcosts: &Pseudocosts, int_tol: f64) -> Option<usize> {
        let mut best = f64::NEG_INFINITY;
        let mut bj = None;
        for (j, (&is_int, &xj)) in self.integrality.iter().zip(x.iter()).enumerate() {
            if is_int {
                let f = xj - xj.floor();
                if f.min(1.0 - f) > int_tol {
                    let s = pcosts.score(j, f);
                    if s > best {
                        best = s;
                        bj = Some(j);
                    }
                }
            }
        }
        bj
    }

    /// The objective `c·x` in the model sense.
    fn objective(&self, x: &[f64]) -> f64 {
        self.c.iter().zip(x.iter()).map(|(c, x)| c * x).sum()
    }

    /// Best-bound LP-OA branch-and-cut over the global box `(lb, hi)`. Best-bound
    /// worklist, per-node FBBT + `solve_node_cut`, sound fathoming on the NS safe
    /// bound, branch on the most-fractional integer. Never reports `Optimal`
    /// unless the dual bound closes onto the incumbent within `gap_tol`.
    pub fn solve_tree(&self, config: &ConvexTreeConfig, opts: &SimplexOptions) -> ConvexTreeResult {
        let sense = if self.sense_max { 1.0 } else { -1.0 };
        // Work in a "maximize sense·bound" convention: priority = sense·bound so
        // the heap always pops the most promising node; incumbent improves when
        // sense·obj increases; fathom when sense·dual ≤ sense·incumbent + gap.
        let worse = f64::NEG_INFINITY; // sense·(worst possible objective)
        let mut inc_sense = config.initial_incumbent.map(|v| sense * v).unwrap_or(worse);
        let mut incumbent_x: Vec<f64> = Vec::new();
        // Pseudocost branching (SCIP product rule) — far fewer nodes than
        // most-fractional. Costs are learned in the tree's own "maximize sense·bound"
        // convention negated to a minimize (lower-bound-rising) convention.
        let mut pcosts = Pseudocosts::new(self.n);

        // Root node over the FBBT-propagated global box.
        let mut root_lo = self.lb.clone();
        let mut root_hi = self.ub.clone();
        if !self.propagate_fbbt(&mut root_lo, &mut root_hi, config.fbbt_rounds) {
            return ConvexTreeResult {
                status: ConvexTreeStatus::Infeasible,
                incumbent: None,
                incumbent_x: Vec::new(),
                bound: worse,
                node_count: 0,
            };
        }

        let mut heap = std::collections::BinaryHeap::new();
        heap.push(TreeNode {
            key: f64::INFINITY,
            parent_bound: sense * f64::INFINITY, // sense·(trivial dual bound)
            lo: root_lo,
            hi: root_hi,
            branch: None,
        });

        let mut node_count = 0usize;
        let mut status = ConvexTreeStatus::Exhausted;
        // Global dual bound (sense convention): max over the frontier of node duals.
        let mut global_dual_sense = f64::INFINITY;

        while let Some(node) = heap.pop() {
            // Global dual bound = best (largest sense·bound) still on the frontier,
            // including this node.
            global_dual_sense = node.parent_bound.max(
                heap.iter()
                    .map(|n| n.parent_bound)
                    .fold(f64::NEG_INFINITY, f64::max),
            );

            // Gap check: if the best remaining dual has closed onto the incumbent,
            // the whole tree is certified.
            if inc_sense > worse
                && global_dual_sense <= inc_sense + config.gap_tol * inc_sense.abs().max(1.0)
            {
                status = ConvexTreeStatus::Optimal;
                break;
            }
            if config
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                status = ConvexTreeStatus::TimeLimit;
                break;
            }
            if node_count >= config.max_nodes {
                status = ConvexTreeStatus::NodeLimit;
                break;
            }
            // Fathom by parent bound vs incumbent.
            if inc_sense > worse
                && node.parent_bound <= inc_sense + config.gap_tol * inc_sense.abs().max(1.0)
            {
                continue;
            }
            node_count += 1;

            // FBBT the child box (root already propagated, but branching added
            // bounds that propagate further).
            let mut lo = node.lo;
            let mut hi = node.hi;
            if !self.propagate_fbbt(&mut lo, &mut hi, config.fbbt_rounds) {
                continue; // empty box → fathom
            }

            let r = self.solve_node_cut(
                &lo,
                &hi,
                config.oa_tol,
                config.max_oa_rounds,
                config.max_sep_rounds,
                opts,
            );
            if r.status != LpStatus::Optimal {
                continue; // infeasible/unbounded region → skip
            }
            // Node dual in the sense convention, floored by the parent (rigorous:
            // a child's bound can only be ≤ the parent's in sense convention).
            let node_dual_sense = (sense * r.bound).min(node.parent_bound);
            if !node_dual_sense.is_finite() {
                // Uncertified node bound → cannot fathom on it, but we can still
                // branch to make progress; treat its dual as the parent's.
            }
            // Learn a pseudocost from the branch that created this node. In the
            // minimize (lower-bound-rising) convention the bound is −sense·dual, so
            // the gain of tightening = node.parent_bound − node_dual_sense ≥ 0.
            if let Some((var, frac, is_down)) = node.branch {
                if node_dual_sense.is_finite() && node.parent_bound.is_finite() {
                    pcosts.update(var, -node.parent_bound, -node_dual_sense, frac, is_down);
                }
            }
            // Fathom by node bound.
            if inc_sense > worse
                && node_dual_sense <= inc_sense + config.gap_tol * inc_sense.abs().max(1.0)
            {
                continue;
            }

            // Incumbent from an integer-feasible OA-tight vertex.
            if self.is_integer_feasible(&r.x, config.int_tol, config.oa_tol * 10.0) {
                let obj_sense = sense * self.objective(&r.x);
                if obj_sense > inc_sense {
                    inc_sense = obj_sense;
                    incumbent_x = r.x.clone();
                }
                continue; // integral node → nothing to branch
            }

            // Branch on the highest pseudocost-score fractional integer.
            let Some(j) = self.select_branch(&r.x, &pcosts, config.int_tol) else {
                continue;
            };
            let xj = r.x[j];
            let frac = xj - xj.floor();
            let branch_key = node_dual_sense;
            // Down child: x_j ≤ floor(x_j).
            {
                let clo = lo.clone();
                let mut chi = hi.clone();
                chi[j] = xj.floor();
                if clo[j] <= chi[j] + 1e-9 {
                    heap.push(TreeNode {
                        key: branch_key,
                        parent_bound: node_dual_sense,
                        lo: clo,
                        hi: chi,
                        branch: Some((j, frac, true)),
                    });
                }
            }
            // Up child: x_j ≥ ceil(x_j).
            {
                let mut clo = lo;
                let chi = hi;
                clo[j] = xj.ceil();
                if clo[j] <= chi[j] + 1e-9 {
                    heap.push(TreeNode {
                        key: branch_key,
                        parent_bound: node_dual_sense,
                        lo: clo,
                        hi: chi,
                        branch: Some((j, frac, false)),
                    });
                }
            }
        }

        if heap.is_empty() && status == ConvexTreeStatus::Exhausted {
            // Frontier drained: the global dual collapses to the incumbent.
            global_dual_sense = inc_sense;
            if inc_sense > worse {
                status = ConvexTreeStatus::Optimal;
            }
        }

        let incumbent = if inc_sense > worse {
            Some(sense * inc_sense)
        } else {
            None
        };
        let bound = if global_dual_sense.is_finite() {
            sense * global_dual_sense
        } else if inc_sense > worse {
            sense * inc_sense
        } else {
            sense * f64::INFINITY
        };
        ConvexTreeResult {
            status: if incumbent.is_none() && status == ConvexTreeStatus::Optimal {
                ConvexTreeStatus::Exhausted
            } else {
                status
            },
            incumbent,
            incumbent_x,
            bound,
            node_count,
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

    /// K2: in-node separation tightens the LP bound toward the integer hull
    /// WITHOUT cutting any integer-feasible point.
    ///
    /// `max x0+x1  s.t.  x0+x1 ≤ 1.5,  x0,x1 ∈ {0,1}`. LP relaxation optimum 1.5;
    /// integer optimum 1. `{x0,x1}` is a knapsack cover (1+1 > 1.5) → the cover
    /// cut `x0+x1 ≤ 1` closes the LP to exactly the integer hull.
    #[test]
    fn node_separation_tightens_to_integer_hull_soundly() {
        let spec = ConvexKernelSpec {
            n: 2,
            c: vec![1.0, 1.0],
            sense_max: true,
            integrality: vec![true, true],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
            le_rows: vec![LinRow {
                cols: vec![0, 1],
                coeffs: vec![1.0, 1.0],
                rhs: 1.5,
            }],
            eq_rows: vec![],
            nl_rows: vec![],
        };
        // K1 (no separation): the LP relaxation bound is 1.5.
        let r0 = spec.solve_node(&[0.0, 0.0], &[1.0, 1.0], 1e-9, 10, &opts());
        assert_eq!(r0.status, LpStatus::Optimal);
        assert!(
            (r0.bound - 1.5).abs() < 1e-6,
            "no-sep bound {} != 1.5",
            r0.bound
        );
        // K2 (with separation): the cover cut closes it to the integer hull, 1.0.
        let r1 = spec.solve_node_cut(&[0.0, 0.0], &[1.0, 1.0], 1e-9, 10, 8, &opts());
        assert_eq!(r1.status, LpStatus::Optimal);
        // Strictly tighter than the un-cut relaxation ...
        assert!(
            r1.bound < 1.5 - 1e-6,
            "separation did not tighten: {}",
            r1.bound
        );
        // ... reaches the integer hull ...
        assert!(
            r1.bound <= 1.0 + 1e-6,
            "bound {} did not reach hull 1.0",
            r1.bound
        );
        // ... and is SOUND: never below the true integer optimum (a cut that
        // removed an integer-feasible point would push the bound under 1.0).
        assert!(
            r1.bound >= 1.0 - 1e-6,
            "UNSOUND: bound {} < integer optimum 1.0",
            r1.bound
        );
    }

    /// K2c: the tree certifies a small MILP. `max x0+x1 s.t. x0+x1≤1.5,
    /// x∈{0,1}²` → integer optimum 1 (branching + fathoming, no nl rows).
    #[test]
    fn tree_certifies_milp() {
        let spec = ConvexKernelSpec {
            n: 2,
            c: vec![1.0, 1.0],
            sense_max: true,
            integrality: vec![true, true],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
            le_rows: vec![LinRow {
                cols: vec![0, 1],
                coeffs: vec![1.0, 1.0],
                rhs: 1.5,
            }],
            eq_rows: vec![],
            nl_rows: vec![],
        };
        let cfg = ConvexTreeConfig::default();
        let r = spec.solve_tree(&cfg, &opts());
        assert_eq!(r.status, ConvexTreeStatus::Optimal, "status {:?}", r.status);
        let inc = r.incumbent.expect("incumbent");
        assert!((inc - 1.0).abs() < 1e-6, "incumbent {inc} != 1.0");
        // Dual bound is a valid upper bound that closed onto the incumbent.
        assert!(
            r.bound >= inc - 1e-6 && r.bound <= inc + 1e-4,
            "bound {} vs inc {inc}",
            r.bound
        );
    }

    /// K2c: the tree certifies a small convex MINLP.
    /// `max x + k  s.t.  k ≤ x,  exp(x) ≤ 5,  x∈[0,10] cont, k∈{0..3} int`.
    /// exp(x)≤5 → x ≤ ln5 ≈ 1.6094; k ≤ x → k=1. Optimum ≈ ln5 + 1 ≈ 2.6094.
    #[test]
    fn tree_certifies_convex_minlp() {
        let spec = ConvexKernelSpec {
            n: 2, // x0 = x (cont), x1 = k (int)
            c: vec![1.0, 1.0],
            sense_max: true,
            integrality: vec![false, true],
            lb: vec![0.0, 0.0],
            ub: vec![10.0, 3.0],
            le_rows: vec![LinRow {
                cols: vec![1, 0],
                coeffs: vec![1.0, -1.0], // k − x ≤ 0
                rhs: 0.0,
            }],
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
        let cfg = ConvexTreeConfig::default();
        let r = spec.solve_tree(&cfg, &opts());
        assert_eq!(r.status, ConvexTreeStatus::Optimal, "status {:?}", r.status);
        let inc = r.incumbent.expect("incumbent");
        let truth = 5.0_f64.ln() + 1.0;
        assert!((inc - truth).abs() < 1e-3, "incumbent {inc} != {truth}");
        // Sound: the dual bound never below the true optimum (it's an UPPER bound).
        assert!(
            r.bound >= truth - 1e-3,
            "UNSOUND dual bound {} < truth {truth}",
            r.bound
        );
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
