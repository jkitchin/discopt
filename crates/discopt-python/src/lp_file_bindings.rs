//! PyO3 binding for the in-house LP-format reader (`discopt_core::lp_file`,
//! highs-parity P6).

use discopt_core::lp_file::read_lp_bytes;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Read an LP-format model from its bytes (decompress `.gz` first) into the arrays
/// HiGHS's `readModel` -> `getLp` would return: a dict with `num_row`, `num_col`,
/// the CSC matrix (`a_start`, `a_index`, `a_value`), `col_cost`, `offset`,
/// `maximize`, `col_lower`, `col_upper`, `row_lower`, `row_upper` (infinite bounds
/// as `inf`), `integrality` (empty when all continuous), `col_names`, `row_names`
/// (empty when HiGHS clears them), `objective_name`, and `hessian`: `None`, or a
/// dict with `dim`, `start`, `index`, `value` (HiGHS's normalised lower triangle,
/// entry order included).
///
/// Raises `ValueError` wherever HiGHS's reader returns `kError`.
#[pyfunction]
pub fn read_lp_text<'py>(py: Python<'py>, text: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let m = py
        .allow_threads(|| read_lp_bytes(text))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let to_i64 = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    let d = PyDict::new(py);
    d.set_item("num_row", m.num_row)?;
    d.set_item("num_col", m.num_col)?;
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
    d.set_item("objective_name", m.objective_name)?;
    match m.hessian {
        None => d.set_item("hessian", py.None())?,
        Some(h) => {
            let hd = PyDict::new(py);
            hd.set_item("dim", h.dim)?;
            hd.set_item("start", PyArray1::from_vec(py, to_i64(h.start)))?;
            hd.set_item("index", PyArray1::from_vec(py, to_i64(h.index)))?;
            hd.set_item("value", PyArray1::from_vec(py, h.value))?;
            d.set_item("hessian", hd)?;
        }
    }
    Ok(d)
}
