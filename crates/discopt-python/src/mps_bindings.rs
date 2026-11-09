//! PyO3 binding for the in-house MPS reader (`discopt_core::mps`, highs-parity P6).

use discopt_core::mps::{read_mps_str, MpsError};
use numpy::PyArray1;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Read an MPS model from its text (decompress `.gz` first) into the arrays HiGHS's
/// `readModel` -> `getLp` would return: a dict with `num_row`, `num_col`, the CSC
/// matrix (`a_start`, `a_index`, `a_value`), `col_cost`, `offset`, `maximize`,
/// `col_lower`, `col_upper`, `row_lower`, `row_upper` (infinite bounds as `inf`),
/// `integrality` (empty when all continuous), `col_names`, `row_names` (empty when
/// names repeat) and `has_hessian`.
///
/// Raises `ValueError` where HiGHS's reader fails, and `NotImplementedError` for a
/// fixed-format file (HiGHS's legacy fixed reader is not emulated).
#[pyfunction]
pub fn read_mps_text<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    let m = py
        .allow_threads(|| read_mps_str(text))
        .map_err(|e| match e {
            MpsError::Parse(_) => PyValueError::new_err(e.to_string()),
            MpsError::FixedFormat(_) => PyNotImplementedError::new_err(e.to_string()),
        })?;
    let d = PyDict::new(py);
    d.set_item("num_row", m.num_row)?;
    d.set_item("num_col", m.num_col)?;
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    d.set_item("a_start", PyArray1::from_vec(py, to_i64(m.a_start)))?;
    d.set_item("a_index", PyArray1::from_vec(py, to_i64(m.a_index)))?;
    d.set_item("a_value", PyArray1::from_vec(py, m.a_value))?;
    d.set_item("col_cost", PyArray1::from_vec(py, m.col_cost))?;
    d.set_item("offset", m.offset)?;
    d.set_item("maximize", m.maximize)?;
    d.set_item("col_lower", PyArray1::from_vec(py, m.col_lower))?;
    d.set_item("col_upper", PyArray1::from_vec(py, m.col_upper))?;
    d.set_item("row_lower", PyArray1::from_vec(py, m.row_lower))?;
    d.set_item("row_upper", PyArray1::from_vec(py, m.row_upper))?;
    let integ: Vec<i64> = m.integrality.into_iter().map(i64::from).collect();
    d.set_item("integrality", PyArray1::from_vec(py, integ))?;
    d.set_item("col_names", m.col_names)?;
    d.set_item("row_names", m.row_names)?;
    d.set_item("has_hessian", m.has_hessian)?;
    Ok(d)
}
