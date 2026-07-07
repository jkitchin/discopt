//! PyO3 bindings for the decomposition graph kernels.
//!
//! Thin wrappers over `discopt_core::decomp` (see
//! `crates/discopt-core/src/decomp/`) exposing the connected-components,
//! strongly-connected-components, and articulation/bridge kernels to Python.
//! Inputs are plain edge lists (`(u, v)` pairs); pyo3 converts the `Vec`
//! results to Python lists/tuples. The Python graph layer
//! (`discopt.decomposition.graph.kernels`) calls these when the extension is
//! available and falls back to its pure-Python reference otherwise.

use discopt_core::decomp::{
    articulation_and_bridges, connected_components, strongly_connected_components, CsrGraph,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Reject any edge whose endpoints fall outside `0..n`. The CSR builders index a
/// `vec![0; n]` degree array directly, so an out-of-range vertex would panic and
/// unwind into Python as a `PanicException`; surface a clean `ValueError` instead.
fn validate_edges(n: usize, edges: &[(u32, u32)]) -> PyResult<()> {
    for &(a, b) in edges {
        if a as usize >= n || b as usize >= n {
            return Err(PyValueError::new_err(format!(
                "edge vertex out of range: ({a}, {b}) with n={n}"
            )));
        }
    }
    Ok(())
}

/// Connected-component label per vertex of an undirected graph on `n` vertices
/// with the given `edges`. Labels are assigned in ascending first-seen vertex
/// order (matching the pure-Python reference and the block-ordering convention).
#[pyfunction]
pub fn decomp_connected_components(n: usize, edges: Vec<(u32, u32)>) -> PyResult<Vec<u32>> {
    validate_edges(n, &edges)?;
    let g = CsrGraph::from_edges_undirected(n, &edges);
    Ok(connected_components(&g).0)
}

/// Strongly-connected-component id per vertex of a directed graph on `n`
/// vertices with the given `arcs` (`from -> to`).
#[pyfunction]
pub fn decomp_strongly_connected_components(n: usize, arcs: Vec<(u32, u32)>) -> PyResult<Vec<u32>> {
    validate_edges(n, &arcs)?;
    let g = CsrGraph::from_edges_directed(n, &arcs);
    Ok(strongly_connected_components(&g).0)
}

/// Articulation points and bridges of an undirected simple graph on `n`
/// vertices with the given `edges`. Returns `(articulation_points, bridges)`
/// with articulation points ascending and each bridge as a `(min, max)` pair.
#[pyfunction]
pub fn decomp_articulation_and_bridges(
    n: usize,
    edges: Vec<(u32, u32)>,
) -> PyResult<(Vec<u32>, Vec<(u32, u32)>)> {
    validate_edges(n, &edges)?;
    let g = CsrGraph::from_edges_undirected(n, &edges);
    let r = articulation_and_bridges(&g);
    Ok((r.articulation_points(), r.bridges))
}
