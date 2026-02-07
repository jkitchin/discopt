use pyo3::prelude::*;

/// Returns the discopt version.
#[pyfunction]
fn version() -> &'static str {
    discopt_core::version()
}

/// The discopt Rust extension module.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
