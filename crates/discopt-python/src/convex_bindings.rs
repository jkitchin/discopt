//! PyO3 binding for the convex LP-OA branch-and-cut kernel node relaxation
//! (`discopt_core::bnb::convex_kernel`, issue #798 / K1). The Python producer
//! decomposes each convex nonlinear row into composite-of-affine form once and
//! hands the flat arrays over; Rust builds a `ConvexKernelSpec` and solves the
//! LP-OA node relaxation over a box (K1d byte-check gate against `_RootLP`).
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
    Affine, CompositeTerm, ConvexFunc, ConvexKernelSpec, ConvexRow, LinRow,
};
use discopt_core::lp::simplex::{LpStatus, SimplexOptions};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

/// Solve one convex LP-OA node relaxation over the box `(lo, hi)`.
///
/// Returns a dict: `status` ("optimal"|"infeasible"|"unbounded"|"iter_limit"|
/// "numerical"), `bound` (rigorous dual bound in the model's sense), `x`
/// (structural primal, length `n`), `oa_rounds`, `n_tangents`.
#[pyfunction]
#[pyo3(signature = (
    n, c, integrality, lo, hi, sense_max,
    le_row_ptr, le_cols, le_coeffs, le_rhs,
    eq_row_ptr, eq_cols, eq_coeffs, eq_rhs,
    nl_rhs, nl_lin_const, nl_lin_ptr, nl_lin_cols, nl_lin_coeffs, nl_term_ptr,
    term_coeff, term_func, term_arg_const, term_arg_ptr, term_arg_cols, term_arg_coeffs,
    oa_tol=1e-6, max_oa_rounds=60, expel_zero_artificials=true,
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
    expel_zero_artificials: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let c = c.as_slice()?;
    let integrality = integrality.as_slice()?;
    let lo = lo.as_slice()?;
    let hi = hi.as_slice()?;
    if c.len() != n || integrality.len() != n || lo.len() != n || hi.len() != n {
        return Err(PyValueError::new_err(
            "c / integrality / lo / hi must all have length n",
        ));
    }

    let le_rows = build_lin_rows(
        le_row_ptr.as_slice()?,
        le_cols.as_slice()?,
        le_coeffs.as_slice()?,
        le_rhs.as_slice()?,
    )?;
    let eq_rows = build_lin_rows(
        eq_row_ptr.as_slice()?,
        eq_cols.as_slice()?,
        eq_coeffs.as_slice()?,
        eq_rhs.as_slice()?,
    )?;

    // Nonlinear rows.
    let nl_rhs = nl_rhs.as_slice()?;
    let nl_lin_const = nl_lin_const.as_slice()?;
    let nl_lin_ptr = nl_lin_ptr.as_slice()?;
    let nl_lin_cols = nl_lin_cols.as_slice()?;
    let nl_lin_coeffs = nl_lin_coeffs.as_slice()?;
    let nl_term_ptr = nl_term_ptr.as_slice()?;
    let term_coeff = term_coeff.as_slice()?;
    let term_func = term_func.as_slice()?;
    let term_arg_const = term_arg_const.as_slice()?;
    let term_arg_ptr = term_arg_ptr.as_slice()?;
    let term_arg_cols = term_arg_cols.as_slice()?;
    let term_arg_coeffs = term_arg_coeffs.as_slice()?;
    let n_nl = nl_rhs.len();
    if nl_lin_const.len() != n_nl || nl_lin_ptr.len() != n_nl + 1 || nl_term_ptr.len() != n_nl + 1 {
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

    let spec = ConvexKernelSpec {
        n,
        c: c.to_vec(),
        sense_max,
        integrality: integrality.iter().map(|&v| v != 0).collect(),
        lb: lo.to_vec(),
        ub: hi.to_vec(),
        le_rows,
        eq_rows,
        nl_rows,
    };

    let opts = SimplexOptions {
        expel_zero_artificials,
        ..Default::default()
    };
    let res = spec.solve_node(lo, hi, oa_tol, max_oa_rounds, &opts);

    let status = match res.status {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
        LpStatus::IterLimit => "iter_limit",
        LpStatus::Numerical => "numerical",
    };
    let out = PyDict::new(py);
    out.set_item("status", status)?;
    out.set_item("bound", res.bound)?;
    out.set_item("raw_bound", res.raw_bound)?;
    out.set_item("x", PyArray1::from_slice(py, &res.x))?;
    out.set_item("oa_rounds", res.oa_rounds)?;
    out.set_item("n_tangents", res.n_tangents)?;
    Ok(out)
}
