//! PyO3 bindings for the .nl file parser.
//!
//! Exposes `parse_nl_file` to Python, returning a `PyModelRepr`.

use pyo3::prelude::*;

use discopt_core::nl_parser;

use crate::expr_bindings::PyModelRepr;

/// Parse a .nl file and return a PyModelRepr.
///
/// Arguments:
///     path: Path to the .nl file (text format).
///
/// Returns:
///     PyModelRepr wrapping the parsed model.
#[pyfunction]
pub fn parse_nl_file(path: &str) -> PyResult<PyModelRepr> {
    let (model, compl) = nl_parser::parse_nl_file_with_complementarity(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
    Ok(PyModelRepr::from_model_repr_with_complementarity(
        model, compl,
    ))
}
