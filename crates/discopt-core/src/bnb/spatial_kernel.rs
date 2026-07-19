//! Native spatial-B&B node kernel (issue #764, C1 build-order item 4, core).
//!
//! The item-4 centerpiece: run a spatial-B&B *node* end-to-end inside
//! `discopt-core` — patch the box-dependent McCormick envelopes ([`mccormick_patch`]),
//! assemble the node LP, warm-solve it, and run the in-kernel OBBT sweep
//! ([`obbt_sweep`]) — with no Python boundary crossing. The measured payoff
//! (`issue-764-native-node-kernel-scope.md`, entry experiment step 2): the current
//! Python/JAX node is ~1352 ms on `tanksize` and ~99.99 % of that is orchestration a
//! native loop removes, for a conservative ~9×.
//!
//! Python hands the *box-independent* structure over once as a [`SpatialKernelSpec`]
//! — the fixed linear rows, the objective, the column count / integrality / global
//! bounds, and per-lifted-term envelope descriptors ([`EnvTerm`]). Rust regenerates
//! every box-dependent envelope row + auxiliary bound in closed form per node. This
//! module is the Rust side of that contract; the PyO3 surface and the tree loop are
//! follow-on increments that compose [`solve_spatial_node`].
//!
//! Column layout: `0..n_orig` are the original (branchable) variables the node box
//! applies to; `n_orig..n_cols` are the lifted auxiliaries (`w = xᵢxⱼ`, `s = xᵢ^p`,
//! …) whose bounds are derived per box. Standard form assembled for the simplex is
//! `[A | I] z = b` with one slack per `<=` row.

use crate::bnb::mccormick_patch as mc;
use crate::bnb::obbt_sweep::{obbt_probe_sweep, ObbtSweepResult};
use crate::lp::simplex::refine::ns_safe_bound_csc;
use crate::lp::simplex::sparse::SparseCols;
use crate::lp::simplex::{primal::solve_lp_cols, LpStatus, SimplexOptions};

/// A box-independent linear constraint row `sum(coeffs[k] * x[cols[k]]) <= rhs`
/// (senses are normalized to `<=` by the Python producer; an `==` row is two `<=`).
#[derive(Clone, Debug)]
pub struct FixedRow {
    /// Structural column indices touched by this row.
    pub cols: Vec<usize>,
    /// Coefficients aligned with `cols`.
    pub coeffs: Vec<f64>,
    /// Right-hand side of the `<= rhs` inequality.
    pub rhs: f64,
}

/// One lifted term whose McCormick envelope rows + auxiliary-variable bounds are
/// regenerated per node box. Each variant names its operand column(s) and its
/// output (auxiliary) column, and delegates to [`mccormick_patch`] for the math.
#[derive(Clone, Copy, Debug)]
pub enum EnvTerm {
    /// `w = x_i * x_j` — 4 McCormick rows.
    Bilinear {
        /// First operand column.
        i: usize,
        /// Second operand column.
        j: usize,
        /// Output (auxiliary) column `w`.
        w: usize,
    },
    /// `s = x_i^p` on a sign-definite box — 4 rows (secant + 3 tangents).
    Monomial {
        /// Operand column.
        i: usize,
        /// Output column `s`.
        s: usize,
        /// Integer power `p >= 2`.
        p: i32,
    },
    /// `w = (coeff*x_j + cst)^2` — 4 rows.
    AffineSquare {
        /// Operand column.
        j: usize,
        /// Output column `w`.
        w: usize,
        /// Affine coefficient.
        coeff: f64,
        /// Affine constant.
        cst: f64,
    },
    /// `w = sqrt(coeff*x + cst)` — the concave univariate envelope (4 rows, or the
    /// aux-floor only when degenerate/undefined).
    Sqrt {
        /// Operand column.
        x: usize,
        /// Output column `w`.
        w: usize,
        /// Affine coefficient inside the sqrt.
        coeff: f64,
        /// Affine constant inside the sqrt.
        cst: f64,
    },
}

impl EnvTerm {
    /// The auxiliary (output) column this term defines.
    pub fn aux_col(&self) -> usize {
        match *self {
            EnvTerm::Bilinear { w, .. } => w,
            EnvTerm::Monomial { s, .. } => s,
            EnvTerm::AffineSquare { w, .. } => w,
            EnvTerm::Sqrt { w, .. } => w,
        }
    }

    /// Closed-form auxiliary-variable bounds over the node box `(lo, hi)` (indexed
    /// by structural column). `None` for `Sqrt` on a base box that dips below 0.
    fn aux_bounds(&self, lo: &[f64], hi: &[f64]) -> Option<(f64, f64)> {
        Some(match *self {
            EnvTerm::Bilinear { i, j, .. } => {
                mc::bilinear_aux_bounds(lo[i], hi[i], lo[j], hi[j])
            }
            EnvTerm::Monomial { i, p, .. } => mc::monomial_aux_bounds(lo[i], hi[i], p),
            EnvTerm::AffineSquare { j, coeff, cst, .. } => {
                mc::affine_square_aux_bounds(coeff, cst, lo[j], hi[j])
            }
            EnvTerm::Sqrt { x, coeff, cst, .. } => {
                return mc::sqrt_aux_bounds(coeff, cst, lo[x], hi[x])
            }
        })
    }

    /// Generate this term's envelope rows over the node box, pushing each as
    /// `(cols, coeffs, rhs)` (a `<= rhs` row) onto `out`.
    fn push_rows(&self, lo: &[f64], hi: &[f64], out: &mut Vec<(Vec<usize>, Vec<f64>, f64)>) {
        let emit = |rows: &[mc::EnvRow], out: &mut Vec<(Vec<usize>, Vec<f64>, f64)>| {
            for r in rows {
                out.push((
                    r.cols[..r.nnz].to_vec(),
                    r.coeffs[..r.nnz].to_vec(),
                    r.rhs,
                ));
            }
        };
        match *self {
            EnvTerm::Bilinear { i, j, w } => {
                emit(&mc::bilinear_rows(i, j, w, lo[i], hi[i], lo[j], hi[j]), out)
            }
            EnvTerm::Monomial { i, s, p } => emit(&mc::monomial_rows(i, s, lo[i], hi[i], p), out),
            EnvTerm::AffineSquare { j, w, coeff, cst } => {
                emit(&mc::affine_square_rows(j, w, coeff, cst, lo[j], hi[j]), out)
            }
            EnvTerm::Sqrt { x, w, coeff, cst } => {
                if let Some(rows) = mc::univariate_rows(x, w, coeff, cst, lo[x], hi[x], mc::Univariate::Sqrt)
                {
                    emit(&rows, out);
                }
                // else: degenerate/undefined base box -> aux-floor only, no rows.
            }
        }
    }
}

/// A product of two affine forms `w = A * B`, `A = a_const + Σ a_coeffs·x[a_cols]`
/// and likewise `B`, relaxed by the general McCormick envelope
/// ([`mccormick_patch::bilinear_linform_rows`]). This is the variable × linear-form /
/// linear-form × linear-form product the factorable engine emits when a bilinear
/// factor is a linear combination it did not lift to its own column
/// (`_emit_mccormick`); its envelope rows span all of `a_cols ∪ b_cols ∪ {w}`, so —
/// unlike the fixed-width [`EnvTerm`] variants — it carries variable-length forms.
#[derive(Clone, Debug)]
pub struct BlfTerm {
    /// Columns of the first affine form `A`.
    pub a_cols: Vec<usize>,
    /// Coefficients of `A`, aligned with `a_cols`.
    pub a_coeffs: Vec<f64>,
    /// Constant term of `A`.
    pub a_const: f64,
    /// Columns of the second affine form `B`.
    pub b_cols: Vec<usize>,
    /// Coefficients of `B`, aligned with `b_cols`.
    pub b_coeffs: Vec<f64>,
    /// Constant term of `B`.
    pub b_const: f64,
    /// Output (auxiliary) column `w = A * B`.
    pub w: usize,
}

/// The box-independent structure Python hands to the native kernel once per solve.
#[derive(Clone, Debug)]
pub struct SpatialKernelSpec {
    /// Total structural columns (original + lifted auxiliaries).
    pub n_cols: usize,
    /// Number of original (branchable) variables; the node box applies to `0..n_orig`.
    pub n_orig: usize,
    /// Objective `min cᵀx` over the structural columns, length `n_cols`.
    pub c: Vec<f64>,
    /// Integrality flag per structural column, length `n_cols`.
    pub integrality: Vec<bool>,
    /// Global lower/upper bounds per structural column, length `n_cols`.
    pub global_lo: Vec<f64>,
    /// Global upper bounds per structural column, length `n_cols`.
    pub global_hi: Vec<f64>,
    /// Box-independent linear constraint rows (`<= rhs`).
    pub fixed_rows: Vec<FixedRow>,
    /// Fixed-width lifted-term envelope descriptors.
    pub terms: Vec<EnvTerm>,
    /// Affine-form product terms `w = A*B` (variable-width envelopes).
    pub blf_terms: Vec<BlfTerm>,
    /// Structural columns to probe with OBBT (empty disables the sweep).
    pub obbt_candidates: Vec<usize>,
}

/// The assembled standard-form node LP `[A | I] z = b`, ready for the simplex.
#[derive(Clone)]
pub struct AssembledLp {
    /// Constraint matrix in CSC (structural columns then `m` slack columns).
    pub sp: SparseCols,
    /// Row count.
    pub m: usize,
    /// Total columns `n_cols + m`.
    pub n_total: usize,
    /// Right-hand side, length `m`.
    pub b: Vec<f64>,
    /// Column lower bounds, length `n_total`.
    pub l: Vec<f64>,
    /// Column upper bounds, length `n_total`.
    pub u: Vec<f64>,
}

/// Result of one native spatial node.
#[derive(Clone, Debug)]
pub struct SpatialNodeResult {
    /// LP status of the node relaxation solve.
    pub status: LpStatus,
    /// Rigorous Neumaier–Shcherbina safe lower bound for the node relaxation
    /// (`<=` the true LP optimum at any conditioning), computed from the row duals
    /// via [`ns_safe_bound_csc`]. `-inf` when the bound cannot be certified (a
    /// nonzero reduced cost meets an infinite bound) — sound: a `-inf` bound never
    /// fathoms. **Never** the raw simplex objective, which can drift *above* the
    /// true optimum on an ill-conditioned basis and cause an unsound fathom.
    pub bound: f64,
    /// Primal relaxation point, length `n_total` (structural + slacks).
    pub x: Vec<f64>,
    /// Row duals at the optimum (for the safe-bound evaluation), length `m`.
    pub dual: Vec<f64>,
    /// OBBT-tightened `(lo, hi)` per candidate (aligned with `spec.obbt_candidates`);
    /// empty when the sweep did not run.
    pub tightened: Vec<(f64, f64)>,
    /// Total LP solves performed (1 node solve + OBBT probes).
    pub n_lp_solves: usize,
}

/// Assemble the node LP over the box `(lo, hi)` (both length `n_cols`; `lo/hi` on the
/// original columns are the node bounds, on the aux columns the incoming bounds).
///
/// Auxiliary bounds are set to the closed-form box-derived range intersected with the
/// incoming aux bounds (tighten-only — always sound). Envelope rows are regenerated
/// per box; fixed rows are copied. Panics only on a malformed spec (column out of
/// range), which is a producer bug.
pub fn assemble_node_lp(spec: &SpatialKernelSpec, lo: &[f64], hi: &[f64]) -> AssembledLp {
    let n_cols = spec.n_cols;
    assert_eq!(lo.len(), n_cols);
    assert_eq!(hi.len(), n_cols);

    // Column bounds: start from the incoming box, then tighten aux columns to the
    // box-derived envelope range (intersection = tighten-only, sound).
    let mut l = lo.to_vec();
    let mut u = hi.to_vec();
    for t in &spec.terms {
        if let Some((alo, ahi)) = t.aux_bounds(lo, hi) {
            let a = t.aux_col();
            l[a] = l[a].max(alo);
            u[a] = u[a].min(ahi);
        }
    }
    for t in &spec.blf_terms {
        let (alo, ahi) = mc::bilinear_linform_aux_bounds(
            &t.a_cols, &t.a_coeffs, t.a_const, &t.b_cols, &t.b_coeffs, t.b_const, lo, hi,
        );
        if alo.is_finite() {
            l[t.w] = l[t.w].max(alo);
        }
        if ahi.is_finite() {
            u[t.w] = u[t.w].min(ahi);
        }
    }

    // Rows: fixed rows, then per-term envelope rows (fixed-width, then affine-form).
    let mut rows: Vec<(Vec<usize>, Vec<f64>, f64)> = Vec::with_capacity(
        spec.fixed_rows.len() + spec.terms.len() * 4 + spec.blf_terms.len() * 4,
    );
    for fr in &spec.fixed_rows {
        rows.push((fr.cols.clone(), fr.coeffs.clone(), fr.rhs));
    }
    for t in &spec.terms {
        t.push_rows(lo, hi, &mut rows);
    }
    for t in &spec.blf_terms {
        mc::bilinear_linform_rows(
            &t.a_cols, &t.a_coeffs, t.a_const, &t.b_cols, &t.b_coeffs, t.b_const, t.w, lo, hi,
            &mut rows,
        );
    }

    let m = rows.len();
    let n_total = n_cols + m;

    // Build CSC column-by-column: structural columns accumulate their row entries,
    // then `m` slack columns form an identity block.
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_total];
    let mut b = vec![0.0f64; m];
    for (r, (rc, rcoef, rhs)) in rows.iter().enumerate() {
        b[r] = *rhs;
        for (c, v) in rc.iter().zip(rcoef.iter()) {
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

    // Full column bounds: structural (l/u) then slacks [0, +inf).
    let mut lfull = l;
    let mut ufull = u;
    lfull.extend(std::iter::repeat(0.0).take(m));
    ufull.extend(std::iter::repeat(1e20).take(m));

    AssembledLp {
        sp,
        m,
        n_total,
        b,
        l: lfull,
        u: ufull,
    }
}

/// Run one native spatial node over the box `(lo, hi)` (length `n_cols`): assemble,
/// solve the relaxation, and — if `run_obbt` and candidates exist — run the in-kernel
/// OBBT sweep warm-started from the node's optimal basis.
pub fn solve_spatial_node(
    spec: &SpatialKernelSpec,
    lo: &[f64],
    hi: &[f64],
    run_obbt: bool,
    opts: &SimplexOptions,
) -> SpatialNodeResult {
    let lp = assemble_node_lp(spec, lo, hi);
    // Objective padded with zeros over the slack columns.
    let mut c = spec.c.clone();
    c.resize(lp.n_total, 0.0);

    let sol = solve_lp_cols(lp.sp.clone(), lp.m, lp.n_total, &c, &lp.l, &lp.u, &lp.b, opts);
    let mut n_lp_solves = 1usize;

    // Rigorous safe lower bound from the row duals — NEVER the raw simplex objective
    // (which can drift above the true optimum and cause an unsound fathom). `-inf`
    // when uncertifiable, which never prunes.
    let bound = if sol.status == LpStatus::Optimal {
        let (col_ptr, row_idx, vals) = lp.sp.raw();
        ns_safe_bound_csc(
            &sol.dual, &c, col_ptr, row_idx, vals, lp.m, lp.n_total, &lp.b, &lp.l, &lp.u,
        )
        .unwrap_or(f64::NEG_INFINITY)
    } else {
        f64::NEG_INFINITY
    };

    let mut tightened = Vec::new();
    if run_obbt && sol.status == LpStatus::Optimal && !spec.obbt_candidates.is_empty() {
        let sweep: ObbtSweepResult = obbt_probe_sweep(
            &lp.sp,
            lp.m,
            lp.n_total,
            &lp.b,
            &lp.l,
            &lp.u,
            &spec.obbt_candidates,
            &sol.basis,
            opts,
        );
        n_lp_solves += sweep.n_solves;
        tightened = sweep.bounds;
    }

    SpatialNodeResult {
        status: sol.status,
        bound,
        x: sol.x,
        dual: sol.dual,
        tightened,
        n_lp_solves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny bilinear model: minimize w = x*y over x,y in [0,2], with the 4
    /// McCormick rows. At the LP relaxation the minimum of w over the McCormick
    /// hull on [0,2]^2 is 0 (achieved at any corner where the underestimators
    /// bind), so the node bound is 0 — and the assembled LP must be feasible and
    /// optimal. Columns: x=0, y=1, w=2. n_orig=2 (x,y branchable), n_cols=3.
    fn bilinear_spec() -> SpatialKernelSpec {
        SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0, 0.0, 1.0], // minimize w
            integrality: vec![false, false, false],
            global_lo: vec![0.0, 0.0, -1e20],
            global_hi: vec![2.0, 2.0, 1e20],
            fixed_rows: vec![],
            terms: vec![EnvTerm::Bilinear { i: 0, j: 1, w: 2 }],
            blf_terms: vec![],
            obbt_candidates: vec![0, 1],
        }
    }

    #[test]
    fn assembles_and_solves_bilinear_node() {
        let spec = bilinear_spec();
        let lo = vec![0.0, 0.0, -1e20];
        let hi = vec![2.0, 2.0, 1e20];
        let opts = SimplexOptions::default();
        let res = solve_spatial_node(&spec, &lo, &hi, false, &opts);
        assert_eq!(res.status, LpStatus::Optimal);
        // min w over the McCormick hull on [0,2]^2 is 0; the SAFE bound reproduces
        // it (well-conditioned) and is never above the true optimum (soundness).
        assert!(res.bound <= 0.0 + 1e-9, "safe bound {} above true optimum 0", res.bound);
        assert!(res.bound.abs() < 1e-7, "bound {} != 0", res.bound);
        // aux w column bound derived from the box: [0*0, 2*2] = [0,4].
        let lp = assemble_node_lp(&spec, &lo, &hi);
        assert!((lp.l[2] - 0.0).abs() < 1e-9 && (lp.u[2] - 4.0).abs() < 1e-9);
        // 4 envelope rows -> m=4, n_total = 3 + 4.
        assert_eq!(lp.m, 4);
        assert_eq!(lp.n_total, 7);
    }

    #[test]
    fn blf_term_single_columns_matches_bilinear() {
        // w = A*B with A={x0}, B={x1} (single-column forms) must give the SAME node
        // bound as the EnvTerm::Bilinear spec — the general path subsumes the special.
        let lo = vec![1.0, 1.0, -1e20];
        let hi = vec![2.0, 2.0, 1e20];
        let opts = SimplexOptions::default();
        let bilinear = solve_spatial_node(&bilinear_spec(), &lo, &hi, false, &opts);
        let blf_spec = SpatialKernelSpec {
            n_cols: 3,
            n_orig: 2,
            c: vec![0.0, 0.0, 1.0],
            integrality: vec![false, false, false],
            global_lo: vec![1.0, 1.0, -1e20],
            global_hi: vec![2.0, 2.0, 1e20],
            fixed_rows: vec![],
            terms: vec![],
            blf_terms: vec![BlfTerm {
                a_cols: vec![0],
                a_coeffs: vec![1.0],
                a_const: 0.0,
                b_cols: vec![1],
                b_coeffs: vec![1.0],
                b_const: 0.0,
                w: 2,
            }],
            obbt_candidates: vec![0, 1],
        };
        let blf = solve_spatial_node(&blf_spec, &lo, &hi, false, &opts);
        assert_eq!(blf.status, LpStatus::Optimal);
        assert!(
            (blf.bound - bilinear.bound).abs() < 1e-7,
            "blf bound {} != bilinear bound {}",
            blf.bound,
            bilinear.bound
        );
        assert!((blf.bound - 1.0).abs() < 1e-6, "blf bound {} != 1", blf.bound);
    }

    #[test]
    fn tighter_box_raises_the_min_product() {
        // On [1,2]x[1,2] the McCormick underestimator w >= 1*x + 1*y - 1 gives
        // min w = 1 at (1,1); the node bound must climb from 0 to 1.
        let spec = bilinear_spec();
        let lo = vec![1.0, 1.0, -1e20];
        let hi = vec![2.0, 2.0, 1e20];
        let opts = SimplexOptions::default();
        let res = solve_spatial_node(&spec, &lo, &hi, false, &opts);
        assert_eq!(res.status, LpStatus::Optimal);
        // Safe bound reproduces the tightened optimum and never exceeds it.
        assert!(res.bound <= 1.0 + 1e-9, "safe bound {} above true optimum 1", res.bound);
        assert!((res.bound - 1.0).abs() < 1e-6, "bound {} != 1", res.bound);
    }

    #[test]
    fn obbt_sweep_runs_and_is_tighten_only() {
        // Add a fixed row x + y <= 2 so OBBT can tighten: over [0,2]^2 with
        // x+y<=2, max x = 2 (unchanged) but the box stays valid; the sweep must
        // return bounds within the incoming box and run 2 probes/candidate.
        let mut spec = bilinear_spec();
        spec.fixed_rows = vec![FixedRow {
            cols: vec![0, 1],
            coeffs: vec![1.0, 1.0],
            rhs: 2.0,
        }];
        let lo = vec![0.0, 0.0, -1e20];
        let hi = vec![2.0, 2.0, 1e20];
        let opts = SimplexOptions::default();
        let res = solve_spatial_node(&spec, &lo, &hi, true, &opts);
        assert_eq!(res.status, LpStatus::Optimal);
        assert_eq!(res.tightened.len(), 2);
        assert_eq!(res.n_lp_solves, 1 + 4); // node + 2 candidates * 2 probes
        for (idx, &(glo, ghi)) in res.tightened.iter().enumerate() {
            assert!(glo >= lo[idx] - 1e-9 && ghi <= hi[idx] + 1e-9, "loosened candidate {idx}");
            // x+y<=2 with the other var >=0 gives max x = 2, min x = 0.
            assert!((glo - 0.0).abs() < 1e-6 && (ghi - 2.0).abs() < 1e-6);
        }
    }
}
