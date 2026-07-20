//! PyO3 bindings for the convex LP-OA branch-and-cut kernel
//! (`discopt_core::bnb::convex_kernel`, issue #798). The Python producer
//! decomposes each convex nonlinear row into composite-of-affine form once and
//! hands the flat arrays over; Rust builds a `ConvexKernelSpec` and either solves
//! one node relaxation (`solve_convex_node_py`, the K1 byte-check) or runs the
//! whole best-bound branch-and-cut tree (`solve_convex_tree_py`, K2+).
//!
//! ## Marshaling schema (all int arrays are i64, float arrays f64)
//!
//! Structural: `n`, `c` (len n), `integrality` (len n), `lo`, `hi` (len n),
//! `sense_max` (bool).
//!
//! Linear `≤` rows, CSR: `le_row_ptr` (len n_le+1) into `le_cols`/`le_coeffs`,
//! `le_rhs` (len n_le). Linear `=` rows likewise via `eq_*`.
//!
//! Convex nonlinear `≤` rows (`g = lin + Σ coeff·func(arg) ≤ rhs`):
//! * per row (len n_nl): `nl_rhs`, `nl_lin_const`, `nl_lin_ptr` (len n_nl+1) into
//!   `nl_lin_cols`/`nl_lin_coeffs`, and `nl_term_ptr` (len n_nl+1) into the term
//!   arrays.
//! * per term (len n_terms): `term_coeff`, `term_func` (0=Log,1=Exp,2=Sqrt,
//!   3=Log1p), `term_arg_const`, `term_arg_ptr` (len n_terms+1) into
//!   `term_arg_cols`/`term_arg_coeffs`.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use discopt_core::bnb::convex_kernel::{
    Affine, CompositeTerm, ConvexFunc, ConvexKernelSpec, ConvexRow, ConvexTreeConfig,
    ConvexTreeStatus, LinRow,
};
use discopt_core::lp::simplex::{LpStatus, SimplexOptions};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{Duration, Instant};

fn func_from_code(code: i64) -> PyResult<ConvexFunc> {
    Ok(match code {
        0 => ConvexFunc::Log,
        1 => ConvexFunc::Exp,
        2 => ConvexFunc::Sqrt,
        3 => ConvexFunc::Log1p,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown term_func code {other} (0=Log,1=Exp,2=Sqrt,3=Log1p)"
            )))
        }
    })
}

/// Build a CSR row's `(cols, coeffs)` slice, validating pointer ranges.
fn csr_row(
    ptr: &[i64],
    cols: &[i64],
    coeffs: &[f64],
    r: usize,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let s = ptr[r] as usize;
    let e = ptr[r + 1] as usize;
    if s > e || e > cols.len() || e > coeffs.len() {
        return Err(PyValueError::new_err("CSR pointer out of range"));
    }
    Ok((
        cols[s..e].iter().map(|&v| v as usize).collect(),
        coeffs[s..e].to_vec(),
    ))
}

fn build_lin_rows(ptr: &[i64], cols: &[i64], coeffs: &[f64], rhs: &[f64]) -> PyResult<Vec<LinRow>> {
    if ptr.len() != rhs.len() + 1 {
        return Err(PyValueError::new_err("row_ptr must have length n_rows+1"));
    }
    let mut out = Vec::with_capacity(rhs.len());
    for r in 0..rhs.len() {
        let (c, k) = csr_row(ptr, cols, coeffs, r)?;
        out.push(LinRow {
            cols: c,
            coeffs: k,
            rhs: rhs[r],
        });
    }
    Ok(out)
}

/// The full flat-array bundle (all the marshaled arrays) shared by both bindings.
struct SpecArrays<'py> {
    n: usize,
    sense_max: bool,
    c: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, i64>,
    lo: PyReadonlyArray1<'py, f64>,
    hi: PyReadonlyArray1<'py, f64>,
    le_row_ptr: PyReadonlyArray1<'py, i64>,
    le_cols: PyReadonlyArray1<'py, i64>,
    le_coeffs: PyReadonlyArray1<'py, f64>,
    le_rhs: PyReadonlyArray1<'py, f64>,
    eq_row_ptr: PyReadonlyArray1<'py, i64>,
    eq_cols: PyReadonlyArray1<'py, i64>,
    eq_coeffs: PyReadonlyArray1<'py, f64>,
    eq_rhs: PyReadonlyArray1<'py, f64>,
    nl_rhs: PyReadonlyArray1<'py, f64>,
    nl_lin_const: PyReadonlyArray1<'py, f64>,
    nl_lin_ptr: PyReadonlyArray1<'py, i64>,
    nl_lin_cols: PyReadonlyArray1<'py, i64>,
    nl_lin_coeffs: PyReadonlyArray1<'py, f64>,
    nl_term_ptr: PyReadonlyArray1<'py, i64>,
    term_coeff: PyReadonlyArray1<'py, f64>,
    term_func: PyReadonlyArray1<'py, i64>,
    term_arg_const: PyReadonlyArray1<'py, f64>,
    term_arg_ptr: PyReadonlyArray1<'py, i64>,
    term_arg_cols: PyReadonlyArray1<'py, i64>,
    term_arg_coeffs: PyReadonlyArray1<'py, f64>,
}

impl SpecArrays<'_> {
    /// Validate and build the `ConvexKernelSpec` (`lb/ub` from `lo/hi`).
    fn build(&self) -> PyResult<ConvexKernelSpec> {
        let c = self.c.as_slice()?;
        let integrality = self.integrality.as_slice()?;
        let lo = self.lo.as_slice()?;
        let hi = self.hi.as_slice()?;
        let n = self.n;
        if c.len() != n || integrality.len() != n || lo.len() != n || hi.len() != n {
            return Err(PyValueError::new_err(
                "c / integrality / lo / hi must all have length n",
            ));
        }
        let le_rows = build_lin_rows(
            self.le_row_ptr.as_slice()?,
            self.le_cols.as_slice()?,
            self.le_coeffs.as_slice()?,
            self.le_rhs.as_slice()?,
        )?;
        let eq_rows = build_lin_rows(
            self.eq_row_ptr.as_slice()?,
            self.eq_cols.as_slice()?,
            self.eq_coeffs.as_slice()?,
            self.eq_rhs.as_slice()?,
        )?;

        let nl_rhs = self.nl_rhs.as_slice()?;
        let nl_lin_const = self.nl_lin_const.as_slice()?;
        let nl_lin_ptr = self.nl_lin_ptr.as_slice()?;
        let nl_lin_cols = self.nl_lin_cols.as_slice()?;
        let nl_lin_coeffs = self.nl_lin_coeffs.as_slice()?;
        let nl_term_ptr = self.nl_term_ptr.as_slice()?;
        let term_coeff = self.term_coeff.as_slice()?;
        let term_func = self.term_func.as_slice()?;
        let term_arg_const = self.term_arg_const.as_slice()?;
        let term_arg_ptr = self.term_arg_ptr.as_slice()?;
        let term_arg_cols = self.term_arg_cols.as_slice()?;
        let term_arg_coeffs = self.term_arg_coeffs.as_slice()?;
        let n_nl = nl_rhs.len();
        if nl_lin_const.len() != n_nl
            || nl_lin_ptr.len() != n_nl + 1
            || nl_term_ptr.len() != n_nl + 1
        {
            return Err(PyValueError::new_err(
                "nl per-row arrays must have length n_nl (ptrs n_nl+1)",
            ));
        }
        let mut nl_rows = Vec::with_capacity(n_nl);
        for i in 0..n_nl {
            let (lc, lk) = csr_row(nl_lin_ptr, nl_lin_cols, nl_lin_coeffs, i)?;
            let lin = Affine {
                cols: lc,
                coeffs: lk,
                cst: nl_lin_const[i],
            };
            let (ts, te) = (nl_term_ptr[i] as usize, nl_term_ptr[i + 1] as usize);
            if ts > te || te > term_coeff.len() {
                return Err(PyValueError::new_err("nl_term_ptr out of range"));
            }
            let mut terms = Vec::with_capacity(te - ts);
            for t in ts..te {
                let (ac, ak) = csr_row(term_arg_ptr, term_arg_cols, term_arg_coeffs, t)?;
                terms.push(CompositeTerm {
                    coeff: term_coeff[t],
                    func: func_from_code(term_func[t])?,
                    arg: Affine {
                        cols: ac,
                        coeffs: ak,
                        cst: term_arg_const[t],
                    },
                });
            }
            nl_rows.push(ConvexRow {
                lin,
                terms,
                rhs: nl_rhs[i],
            });
        }
        Ok(ConvexKernelSpec {
            n,
            c: c.to_vec(),
            sense_max: self.sense_max,
            integrality: integrality.iter().map(|&v| v != 0).collect(),
            lb: lo.to_vec(),
            ub: hi.to_vec(),
            le_rows,
            eq_rows,
            nl_rows,
        })
    }
}

fn lp_status_str(s: LpStatus) -> &'static str {
    match s {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
        LpStatus::IterLimit => "iter_limit",
        LpStatus::Numerical => "numerical",
    }
}

/// Solve one convex LP-OA node relaxation over the box `(lo, hi)` (K1 byte-check).
///
/// Returns `{status, bound (NS safe), raw_bound (LP optimum), x, oa_rounds,
/// n_tangents}`.
#[pyfunction]
#[pyo3(signature = (
    n, c, integrality, lo, hi, sense_max,
    le_row_ptr, le_cols, le_coeffs, le_rhs,
    eq_row_ptr, eq_cols, eq_coeffs, eq_rhs,
    nl_rhs, nl_lin_const, nl_lin_ptr, nl_lin_cols, nl_lin_coeffs, nl_term_ptr,
    term_coeff, term_func, term_arg_const, term_arg_ptr, term_arg_cols, term_arg_coeffs,
    oa_tol=1e-6, max_oa_rounds=60, max_sep_rounds=0, expel_zero_artificials=true,
))]
pub fn solve_convex_node_py<'py>(
    py: Python<'py>,
    n: usize,
    c: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, i64>,
    lo: PyReadonlyArray1<'py, f64>,
    hi: PyReadonlyArray1<'py, f64>,
    sense_max: bool,
    le_row_ptr: PyReadonlyArray1<'py, i64>,
    le_cols: PyReadonlyArray1<'py, i64>,
    le_coeffs: PyReadonlyArray1<'py, f64>,
    le_rhs: PyReadonlyArray1<'py, f64>,
    eq_row_ptr: PyReadonlyArray1<'py, i64>,
    eq_cols: PyReadonlyArray1<'py, i64>,
    eq_coeffs: PyReadonlyArray1<'py, f64>,
    eq_rhs: PyReadonlyArray1<'py, f64>,
    nl_rhs: PyReadonlyArray1<'py, f64>,
    nl_lin_const: PyReadonlyArray1<'py, f64>,
    nl_lin_ptr: PyReadonlyArray1<'py, i64>,
    nl_lin_cols: PyReadonlyArray1<'py, i64>,
    nl_lin_coeffs: PyReadonlyArray1<'py, f64>,
    nl_term_ptr: PyReadonlyArray1<'py, i64>,
    term_coeff: PyReadonlyArray1<'py, f64>,
    term_func: PyReadonlyArray1<'py, i64>,
    term_arg_const: PyReadonlyArray1<'py, f64>,
    term_arg_ptr: PyReadonlyArray1<'py, i64>,
    term_arg_cols: PyReadonlyArray1<'py, i64>,
    term_arg_coeffs: PyReadonlyArray1<'py, f64>,
    oa_tol: f64,
    max_oa_rounds: usize,
    max_sep_rounds: usize,
    expel_zero_artificials: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let arrays = SpecArrays {
        n,
        sense_max,
        c,
        integrality,
        lo,
        hi,
        le_row_ptr,
        le_cols,
        le_coeffs,
        le_rhs,
        eq_row_ptr,
        eq_cols,
        eq_coeffs,
        eq_rhs,
        nl_rhs,
        nl_lin_const,
        nl_lin_ptr,
        nl_lin_cols,
        nl_lin_coeffs,
        nl_term_ptr,
        term_coeff,
        term_func,
        term_arg_const,
        term_arg_ptr,
        term_arg_cols,
        term_arg_coeffs,
    };
    let spec = arrays.build()?;
    let lo = arrays.lo.as_slice()?;
    let hi = arrays.hi.as_slice()?;
    let opts = SimplexOptions {
        expel_zero_artificials,
        ..Default::default()
    };
    let res = if max_sep_rounds > 0 {
        spec.solve_node_cut(lo, hi, oa_tol, max_oa_rounds, max_sep_rounds, &opts)
    } else {
        spec.solve_node(lo, hi, oa_tol, max_oa_rounds, &opts)
    };
    let out = PyDict::new(py);
    out.set_item("status", lp_status_str(res.status))?;
    out.set_item("bound", res.bound)?;
    out.set_item("raw_bound", res.raw_bound)?;
    out.set_item("x", PyArray1::from_slice(py, &res.x))?;
    out.set_item("oa_rounds", res.oa_rounds)?;
    out.set_item("n_tangents", res.n_tangents)?;
    Ok(out)
}

/// Run the full convex LP-OA branch-and-cut tree (K2). Returns
/// `{status, incumbent, incumbent_x, bound, node_count}`.
#[pyfunction]
#[pyo3(signature = (
    n, c, integrality, lo, hi, sense_max,
    le_row_ptr, le_cols, le_coeffs, le_rhs,
    eq_row_ptr, eq_cols, eq_coeffs, eq_rhs,
    nl_rhs, nl_lin_const, nl_lin_ptr, nl_lin_cols, nl_lin_coeffs, nl_term_ptr,
    term_coeff, term_func, term_arg_const, term_arg_ptr, term_arg_cols, term_arg_coeffs,
    max_nodes=100_000, gap_tol=1e-4, int_tol=1e-5, oa_tol=1e-6,
    max_oa_rounds=60, max_sep_rounds=12, fbbt_rounds=20,
    initial_incumbent=None, time_limit_s=None,
))]
pub fn solve_convex_tree_py<'py>(
    py: Python<'py>,
    n: usize,
    c: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, i64>,
    lo: PyReadonlyArray1<'py, f64>,
    hi: PyReadonlyArray1<'py, f64>,
    sense_max: bool,
    le_row_ptr: PyReadonlyArray1<'py, i64>,
    le_cols: PyReadonlyArray1<'py, i64>,
    le_coeffs: PyReadonlyArray1<'py, f64>,
    le_rhs: PyReadonlyArray1<'py, f64>,
    eq_row_ptr: PyReadonlyArray1<'py, i64>,
    eq_cols: PyReadonlyArray1<'py, i64>,
    eq_coeffs: PyReadonlyArray1<'py, f64>,
    eq_rhs: PyReadonlyArray1<'py, f64>,
    nl_rhs: PyReadonlyArray1<'py, f64>,
    nl_lin_const: PyReadonlyArray1<'py, f64>,
    nl_lin_ptr: PyReadonlyArray1<'py, i64>,
    nl_lin_cols: PyReadonlyArray1<'py, i64>,
    nl_lin_coeffs: PyReadonlyArray1<'py, f64>,
    nl_term_ptr: PyReadonlyArray1<'py, i64>,
    term_coeff: PyReadonlyArray1<'py, f64>,
    term_func: PyReadonlyArray1<'py, i64>,
    term_arg_const: PyReadonlyArray1<'py, f64>,
    term_arg_ptr: PyReadonlyArray1<'py, i64>,
    term_arg_cols: PyReadonlyArray1<'py, i64>,
    term_arg_coeffs: PyReadonlyArray1<'py, f64>,
    max_nodes: usize,
    gap_tol: f64,
    int_tol: f64,
    oa_tol: f64,
    max_oa_rounds: usize,
    max_sep_rounds: usize,
    fbbt_rounds: usize,
    initial_incumbent: Option<f64>,
    time_limit_s: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let arrays = SpecArrays {
        n,
        sense_max,
        c,
        integrality,
        lo,
        hi,
        le_row_ptr,
        le_cols,
        le_coeffs,
        le_rhs,
        eq_row_ptr,
        eq_cols,
        eq_coeffs,
        eq_rhs,
        nl_rhs,
        nl_lin_const,
        nl_lin_ptr,
        nl_lin_cols,
        nl_lin_coeffs,
        nl_term_ptr,
        term_coeff,
        term_func,
        term_arg_const,
        term_arg_ptr,
        term_arg_cols,
        term_arg_coeffs,
    };
    let spec = arrays.build()?;
    let config = ConvexTreeConfig {
        max_nodes,
        gap_tol,
        int_tol,
        oa_tol,
        max_oa_rounds,
        max_sep_rounds,
        fbbt_rounds,
        deadline: time_limit_s.map(|s| Instant::now() + Duration::from_secs_f64(s.max(0.0))),
        initial_incumbent,
    };
    let opts = SimplexOptions {
        expel_zero_artificials: true,
        ..Default::default()
    };
    let res = spec.solve_tree(&config, &opts);
    let status = match res.status {
        ConvexTreeStatus::Optimal => "optimal",
        ConvexTreeStatus::NodeLimit => "node_limit",
        ConvexTreeStatus::TimeLimit => "time_limit",
        ConvexTreeStatus::Exhausted => "exhausted",
        ConvexTreeStatus::Infeasible => "infeasible",
    };
    let out = PyDict::new(py);
    out.set_item("status", status)?;
    out.set_item("incumbent", res.incumbent)?;
    out.set_item("incumbent_x", PyArray1::from_slice(py, &res.incumbent_x))?;
    out.set_item("bound", res.bound)?;
    out.set_item("node_count", res.node_count)?;
    Ok(out)
}

/// W0 entry-experiment probe (#807, TEMPORARY). Runs the seeded best-bound
/// OA-only mini-tree and returns per-node cold-vs-warm measurements as parallel
/// arrays: `{cold_us, warm_us, warm_pivots, bound_diff, ns_ok, new_tangents,
/// pool_before, is_jump}`. Reverted after W0 records its GO/KILL.
#[pyfunction]
#[pyo3(signature = (
    n, c, integrality, lo, hi, sense_max,
    le_row_ptr, le_cols, le_coeffs, le_rhs,
    eq_row_ptr, eq_cols, eq_coeffs, eq_rhs,
    nl_rhs, nl_lin_const, nl_lin_ptr, nl_lin_cols, nl_lin_coeffs, nl_term_ptr,
    term_coeff, term_func, term_arg_const, term_arg_ptr, term_arg_cols, term_arg_coeffs,
    max_stats=25, gap_tol=1e-4, int_tol=1e-5, oa_tol=1e-6,
    max_oa_rounds=60, fbbt_rounds=20, initial_incumbent=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn convex_warmlp_probe_py<'py>(
    py: Python<'py>,
    n: usize,
    c: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, i64>,
    lo: PyReadonlyArray1<'py, f64>,
    hi: PyReadonlyArray1<'py, f64>,
    sense_max: bool,
    le_row_ptr: PyReadonlyArray1<'py, i64>,
    le_cols: PyReadonlyArray1<'py, i64>,
    le_coeffs: PyReadonlyArray1<'py, f64>,
    le_rhs: PyReadonlyArray1<'py, f64>,
    eq_row_ptr: PyReadonlyArray1<'py, i64>,
    eq_cols: PyReadonlyArray1<'py, i64>,
    eq_coeffs: PyReadonlyArray1<'py, f64>,
    eq_rhs: PyReadonlyArray1<'py, f64>,
    nl_rhs: PyReadonlyArray1<'py, f64>,
    nl_lin_const: PyReadonlyArray1<'py, f64>,
    nl_lin_ptr: PyReadonlyArray1<'py, i64>,
    nl_lin_cols: PyReadonlyArray1<'py, i64>,
    nl_lin_coeffs: PyReadonlyArray1<'py, f64>,
    nl_term_ptr: PyReadonlyArray1<'py, i64>,
    term_coeff: PyReadonlyArray1<'py, f64>,
    term_func: PyReadonlyArray1<'py, i64>,
    term_arg_const: PyReadonlyArray1<'py, f64>,
    term_arg_ptr: PyReadonlyArray1<'py, i64>,
    term_arg_cols: PyReadonlyArray1<'py, i64>,
    term_arg_coeffs: PyReadonlyArray1<'py, f64>,
    max_stats: usize,
    gap_tol: f64,
    int_tol: f64,
    oa_tol: f64,
    max_oa_rounds: usize,
    fbbt_rounds: usize,
    initial_incumbent: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let arrays = SpecArrays {
        n,
        sense_max,
        c,
        integrality,
        lo,
        hi,
        le_row_ptr,
        le_cols,
        le_coeffs,
        le_rhs,
        eq_row_ptr,
        eq_cols,
        eq_coeffs,
        eq_rhs,
        nl_rhs,
        nl_lin_const,
        nl_lin_ptr,
        nl_lin_cols,
        nl_lin_coeffs,
        nl_term_ptr,
        term_coeff,
        term_func,
        term_arg_const,
        term_arg_ptr,
        term_arg_cols,
        term_arg_coeffs,
    };
    let spec = arrays.build()?;
    let config = ConvexTreeConfig {
        max_nodes: 100_000,
        gap_tol,
        int_tol,
        oa_tol,
        max_oa_rounds,
        max_sep_rounds: 0,
        fbbt_rounds,
        deadline: None,
        initial_incumbent,
    };
    let opts = SimplexOptions {
        expel_zero_artificials: true,
        ..Default::default()
    };
    let stats = spec.warmlp_w0_probe(&config, &opts, max_stats);
    let cold: Vec<f64> = stats.iter().map(|s| s.cold_us).collect();
    let warm: Vec<f64> = stats.iter().map(|s| s.warm_us).collect();
    let pivots: Vec<f64> = stats.iter().map(|s| s.warm_pivots as f64).collect();
    let bound_diff: Vec<f64> = stats.iter().map(|s| s.bound_diff).collect();
    let ns_ok: Vec<f64> = stats
        .iter()
        .map(|s| if s.ns_ok { 1.0 } else { 0.0 })
        .collect();
    let new_tan: Vec<f64> = stats.iter().map(|s| s.new_tangents as f64).collect();
    let pool: Vec<f64> = stats.iter().map(|s| s.pool_before as f64).collect();
    let jump: Vec<f64> = stats
        .iter()
        .map(|s| if s.is_jump { 1.0 } else { 0.0 })
        .collect();
    let cold_bound: Vec<f64> = stats.iter().map(|s| s.cold_bound).collect();
    let warm_bound: Vec<f64> = stats.iter().map(|s| s.warm_bound).collect();
    let out = PyDict::new(py);
    out.set_item("cold_bound", PyArray1::from_slice(py, &cold_bound))?;
    out.set_item("warm_bound", PyArray1::from_slice(py, &warm_bound))?;
    out.set_item("cold_us", PyArray1::from_slice(py, &cold))?;
    out.set_item("warm_us", PyArray1::from_slice(py, &warm))?;
    out.set_item("warm_pivots", PyArray1::from_slice(py, &pivots))?;
    out.set_item("bound_diff", PyArray1::from_slice(py, &bound_diff))?;
    out.set_item("ns_ok", PyArray1::from_slice(py, &ns_ok))?;
    out.set_item("new_tangents", PyArray1::from_slice(py, &new_tan))?;
    out.set_item("pool_before", PyArray1::from_slice(py, &pool))?;
    out.set_item("is_jump", PyArray1::from_slice(py, &jump))?;
    Ok(out)
}
