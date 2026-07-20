//! PyO3 bindings for the discopt MINLP solver.
//!
//! Provides Python-accessible wrappers for the Rust B&B tree manager,
//! expression IR, batch dispatch, and .nl parser. The IPM NLP solver
//! is POUNCE (https://github.com/jkitchin/pounce), used from Python.

#![deny(missing_docs)]

use pyo3::prelude::*;

mod batch;
mod bnb_bindings;
mod convex_bindings;
mod decomp_bindings;
mod expr_bindings;
mod lp_bindings;
mod nl_bindings;
mod spatial_bindings;

/// Returns the discopt version.
#[pyfunction]
fn version() -> &'static str {
    discopt_core::version()
}

/// The discopt Rust extension module.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<batch::PyBatchDispatcher>()?;
    m.add_class::<bnb_bindings::PyTreeManager>()?;
    m.add_class::<expr_bindings::PyModelRepr>()?;
    m.add_class::<expr_bindings::PyModelBuilder>()?;
    m.add_function(wrap_pyfunction!(expr_bindings::model_to_repr, m)?)?;
    m.add_function(wrap_pyfunction!(nl_bindings::parse_nl_file, m)?)?;
    m.add_function(wrap_pyfunction!(nl_bindings::parse_nl_string, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::crossover_to_vertex_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::recover_basis_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::gomory_cuts_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::mir_cuts_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::aggregation_mir_cuts_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::solve_lp_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::solve_lp_warm_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::solve_lp_warm_csc_py, m)?)?;
    m.add_function(wrap_pyfunction!(convex_bindings::solve_convex_node_py, m)?)?;
    m.add_function(wrap_pyfunction!(convex_bindings::solve_convex_tree_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        convex_bindings::convex_warmlp_probe_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_bindings::solve_spatial_tree_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::solve_lp_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::solve_milp_py, m)?)?;
    m.add_function(wrap_pyfunction!(lp_bindings::solve_milp_csc_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        decomp_bindings::decomp_connected_components,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decomp_bindings::decomp_strongly_connected_components,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        decomp_bindings::decomp_articulation_and_bridges,
        m
    )?)?;
    Ok(())
}
