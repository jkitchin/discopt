//! PyO3 bindings for the LP crossover + basis recovery (`discopt_core::lp`).
//!
//! The LP is passed in standard form `min cᵀx s.t. A x = b, l ≤ x ≤ u` with `A`
//! a C-contiguous `m × n` array (the Python side builds this via
//! `_decompose_eq_slack_form`). These let the Python B&B cut loop push an
//! interior POUNCE optimum to a vertex and recover a simplex basis for
//! basis-derived cuts.
//!
// PyO3 entry points are necessarily flat (one parameter per Python argument)
// and return Python-owned array tuples, so the argument-count and
// type-complexity lints don't meaningfully apply to these binding shims.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use discopt_core::lp::basis::recover_basis;
use discopt_core::lp::crossover::{crossover_to_vertex, LpView};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Push an interior LP optimum `x` to a vertex of the optimal face.
///
/// `a` is the C-contiguous `m × n` equality-constraint matrix; `c`, `lb`, `ub`
/// are length `n`. Returns the vertex as a new length-`n` array (same objective
/// and feasibility as `x`). `max_iter = 0` selects the `n + 1` default.
#[pyfunction]
#[pyo3(signature = (x, a, c, lb, ub, tol=1e-7, max_iter=0))]
pub fn crossover_to_vertex_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dims = a.shape();
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    let xv = crossover_to_vertex(x.as_slice()?, &lp, tol, max_iter);
    Ok(PyArray1::from_vec(py, xv))
}

/// Recover a simplex basis at the vertex `x` of the standard-form LP.
///
/// Returns `(col_status, basic_vars)` — `col_status` is a length-`n` `int8`
/// array of HiGHS `HighsBasisStatus` codes (`0`=AtLower, `1`=Basic,
/// `2`=AtUpper) and `basic_vars` the `m` basic column indices — or `None` when
/// `x` is not a basic feasible solution (see `recover_basis`).
#[pyfunction]
#[pyo3(signature = (x, a, c, lb, ub, tol=1e-7))]
pub fn recover_basis_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    tol: f64,
) -> PyResult<Option<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>)>> {
    let dims = a.shape();
    let (m, n) = (dims[0], dims[1]);
    let a_flat = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?;
    let lp = LpView {
        a: a_flat,
        m,
        n,
        c: c.as_slice()?,
        l: lb.as_slice()?,
        u: ub.as_slice()?,
    };
    match recover_basis(x.as_slice()?, &lp, tol) {
        Some(b) => {
            let status = PyArray1::from_vec(py, b.col_status);
            let basic: Vec<i64> = b.basic_vars.iter().map(|&v| v as i64).collect();
            Ok(Some((status, PyArray1::from_vec(py, basic))))
        }
        None => Ok(None),
    }
}
