//! PyO3 bindings for the native spatial B&B kernel (`discopt_core::bnb::spatial_*`,
//! issue #764). The Python producer extracts the box-independent relaxation
//! structure from the JAX McCormick compiler once and hands it over as flat arrays;
//! Rust regenerates every box-dependent envelope row per node and runs the whole
//! spatial B&B with no Python boundary crossing.
//!
//! Term encoding (parallel arrays, one entry per lifted term):
//! * `term_kind`: 0=Bilinear, 1=Monomial, 2=AffineSquare, 3=Sqrt
//! * `term_i`  : operand column (`i` / `x` / `j`)
//! * `term_j`  : second operand for Bilinear, else ignored (pass -1)
//! * `term_out`: auxiliary output column (`w` / `s`)
//! * `term_p`  : integer power for Monomial, else ignored (pass 0)
//! * `term_coeff`, `term_cst`: affine `coeff*x+cst` for AffineSquare/Sqrt, else 0
//!
//! Fixed linear rows are passed CSR-style: `fixed_row_ptr` (len `n_fixed+1`) into
//! `fixed_cols`/`fixed_coeffs`, plus `fixed_rhs` (each row is `<= rhs`).
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use discopt_core::bnb::spatial_kernel::{BlfTerm, EnvTerm, FixedRow, SpatialKernelSpec};
use discopt_core::bnb::spatial_tree::{solve_spatial_tree, SpatialTreeConfig, TreeStatus};
use discopt_core::lp::simplex::SimplexOptions;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Solve a spatial-B&B problem with the native Rust kernel.
///
/// Returns a dict: `status` ("optimal"|"node_limit"|"infeasible"), `incumbent`
/// (float or None), `incumbent_x` (length-`n_cols` array or empty), `bound`
/// (global lower bound), `node_count`, `n_lp_solves`.
#[pyfunction]
#[pyo3(signature = (
    n_cols, n_orig, c, integrality, global_lo, global_hi,
    fixed_row_ptr, fixed_cols, fixed_coeffs, fixed_rhs,
    term_kind, term_i, term_j, term_out, term_p, term_coeff, term_cst,
    blf_w, blf_a_ptr, blf_a_cols, blf_a_coeffs, blf_a_const,
    blf_b_ptr, blf_b_cols, blf_b_coeffs, blf_b_const,
    obbt_candidates,
    max_nodes=100_000, gap_tol=1e-6, int_tol=1e-5, mccormick_tol=1e-6,
    min_box_width=1e-9, run_obbt=false, run_propagation=true,
    propagation_rounds=15, initial_incumbent=None,
))]
pub fn solve_spatial_tree_py<'py>(
    py: Python<'py>,
    n_cols: usize,
    n_orig: usize,
    c: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, i64>,
    global_lo: PyReadonlyArray1<'py, f64>,
    global_hi: PyReadonlyArray1<'py, f64>,
    fixed_row_ptr: PyReadonlyArray1<'py, i64>,
    fixed_cols: PyReadonlyArray1<'py, i64>,
    fixed_coeffs: PyReadonlyArray1<'py, f64>,
    fixed_rhs: PyReadonlyArray1<'py, f64>,
    term_kind: PyReadonlyArray1<'py, i64>,
    term_i: PyReadonlyArray1<'py, i64>,
    term_j: PyReadonlyArray1<'py, i64>,
    term_out: PyReadonlyArray1<'py, i64>,
    term_p: PyReadonlyArray1<'py, i64>,
    term_coeff: PyReadonlyArray1<'py, f64>,
    term_cst: PyReadonlyArray1<'py, f64>,
    blf_w: PyReadonlyArray1<'py, i64>,
    blf_a_ptr: PyReadonlyArray1<'py, i64>,
    blf_a_cols: PyReadonlyArray1<'py, i64>,
    blf_a_coeffs: PyReadonlyArray1<'py, f64>,
    blf_a_const: PyReadonlyArray1<'py, f64>,
    blf_b_ptr: PyReadonlyArray1<'py, i64>,
    blf_b_cols: PyReadonlyArray1<'py, i64>,
    blf_b_coeffs: PyReadonlyArray1<'py, f64>,
    blf_b_const: PyReadonlyArray1<'py, f64>,
    obbt_candidates: PyReadonlyArray1<'py, i64>,
    max_nodes: usize,
    gap_tol: f64,
    int_tol: f64,
    mccormick_tol: f64,
    min_box_width: f64,
    run_obbt: bool,
    run_propagation: bool,
    propagation_rounds: usize,
    initial_incumbent: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let c = c.as_slice()?;
    let integrality = integrality.as_slice()?;
    let global_lo = global_lo.as_slice()?;
    let global_hi = global_hi.as_slice()?;
    if c.len() != n_cols
        || integrality.len() != n_cols
        || global_lo.len() != n_cols
        || global_hi.len() != n_cols
    {
        return Err(PyValueError::new_err(
            "c / integrality / global_lo / global_hi must all have length n_cols",
        ));
    }

    // Fixed rows (CSR-style).
    let row_ptr = fixed_row_ptr.as_slice()?;
    let fcols = fixed_cols.as_slice()?;
    let fcoef = fixed_coeffs.as_slice()?;
    let frhs = fixed_rhs.as_slice()?;
    if row_ptr.is_empty() || row_ptr.len() != frhs.len() + 1 {
        return Err(PyValueError::new_err(
            "fixed_row_ptr must have length n_fixed+1",
        ));
    }
    let mut fixed_rows = Vec::with_capacity(frhs.len());
    for r in 0..frhs.len() {
        let (s, e) = (row_ptr[r] as usize, row_ptr[r + 1] as usize);
        if e > fcols.len() || e > fcoef.len() || s > e {
            return Err(PyValueError::new_err("fixed_row_ptr out of range"));
        }
        fixed_rows.push(FixedRow {
            cols: fcols[s..e].iter().map(|&v| v as usize).collect(),
            coeffs: fcoef[s..e].to_vec(),
            rhs: frhs[r],
        });
    }

    // Terms.
    let kind = term_kind.as_slice()?;
    let ti = term_i.as_slice()?;
    let tj = term_j.as_slice()?;
    let tout = term_out.as_slice()?;
    let tp = term_p.as_slice()?;
    let tcoeff = term_coeff.as_slice()?;
    let tcst = term_cst.as_slice()?;
    let nt = kind.len();
    if [ti.len(), tj.len(), tout.len(), tp.len(), tcoeff.len(), tcst.len()]
        .iter()
        .any(|&l| l != nt)
    {
        return Err(PyValueError::new_err(
            "all term_* arrays must have the same length",
        ));
    }
    let mut terms = Vec::with_capacity(nt);
    for k in 0..nt {
        let term = match kind[k] {
            0 => EnvTerm::Bilinear {
                i: ti[k] as usize,
                j: tj[k] as usize,
                w: tout[k] as usize,
            },
            1 => EnvTerm::Monomial {
                i: ti[k] as usize,
                s: tout[k] as usize,
                p: tp[k] as i32,
            },
            2 => EnvTerm::AffineSquare {
                j: ti[k] as usize,
                w: tout[k] as usize,
                coeff: tcoeff[k],
                cst: tcst[k],
            },
            3 => EnvTerm::Sqrt {
                x: ti[k] as usize,
                w: tout[k] as usize,
                coeff: tcoeff[k],
                cst: tcst[k],
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown term_kind {other} at index {k}"
                )))
            }
        };
        terms.push(term);
    }

    // Affine-form product terms (CSR-encoded forms A, B).
    let blf_w = blf_w.as_slice()?;
    let a_ptr = blf_a_ptr.as_slice()?;
    let a_cols = blf_a_cols.as_slice()?;
    let a_coeffs = blf_a_coeffs.as_slice()?;
    let a_const = blf_a_const.as_slice()?;
    let b_ptr = blf_b_ptr.as_slice()?;
    let b_cols = blf_b_cols.as_slice()?;
    let b_coeffs = blf_b_coeffs.as_slice()?;
    let b_const = blf_b_const.as_slice()?;
    let n_blf = blf_w.len();
    if a_ptr.len() != n_blf + 1
        || b_ptr.len() != n_blf + 1
        || a_const.len() != n_blf
        || b_const.len() != n_blf
    {
        return Err(PyValueError::new_err(
            "blf_*_ptr must have length n_blf+1 and blf_*_const length n_blf",
        ));
    }
    let mut blf_terms = Vec::with_capacity(n_blf);
    for t in 0..n_blf {
        let (as_, ae) = (a_ptr[t] as usize, a_ptr[t + 1] as usize);
        let (bs, be) = (b_ptr[t] as usize, b_ptr[t + 1] as usize);
        if ae > a_cols.len() || ae > a_coeffs.len() || be > b_cols.len() || be > b_coeffs.len() {
            return Err(PyValueError::new_err("blf form pointer out of range"));
        }
        blf_terms.push(BlfTerm {
            a_cols: a_cols[as_..ae].iter().map(|&v| v as usize).collect(),
            a_coeffs: a_coeffs[as_..ae].to_vec(),
            a_const: a_const[t],
            b_cols: b_cols[bs..be].iter().map(|&v| v as usize).collect(),
            b_coeffs: b_coeffs[bs..be].to_vec(),
            b_const: b_const[t],
            w: blf_w[t] as usize,
        });
    }

    let spec = SpatialKernelSpec {
        n_cols,
        n_orig,
        c: c.to_vec(),
        integrality: integrality.iter().map(|&v| v != 0).collect(),
        global_lo: global_lo.to_vec(),
        global_hi: global_hi.to_vec(),
        fixed_rows,
        terms,
        blf_terms,
        obbt_candidates: obbt_candidates.as_slice()?.iter().map(|&v| v as usize).collect(),
    };

    let cfg = SpatialTreeConfig {
        max_nodes,
        gap_tol,
        int_tol,
        mccormick_tol,
        min_box_width,
        run_obbt,
        run_propagation,
        propagation_rounds,
        initial_incumbent,
    };
    let opts = SimplexOptions::default();

    // Release the GIL for the (potentially long) solve.
    let res = py.allow_threads(|| solve_spatial_tree(&spec, &cfg, &opts));

    let status = match res.status {
        TreeStatus::Optimal => "optimal",
        TreeStatus::NodeLimit => "node_limit",
        TreeStatus::Exhausted => "exhausted",
        TreeStatus::Infeasible => "infeasible",
    };
    let out = PyDict::new(py);
    out.set_item("status", status)?;
    out.set_item("incumbent", res.incumbent)?;
    out.set_item("incumbent_x", PyArray1::from_vec(py, res.incumbent_x))?;
    out.set_item("bound", res.bound)?;
    out.set_item("node_count", res.node_count)?;
    out.set_item("n_lp_solves", res.n_lp_solves)?;
    out.set_item("n_uncertified", res.n_uncertified)?;
    Ok(out)
}
