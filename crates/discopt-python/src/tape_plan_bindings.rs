//! Python binding for [`discopt_core::tape_plan::plan_chunks`] (#2324).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// `(starts, first_row, last_row)` for `discopt._arena_tape.plan_chunks`.
///
/// See `discopt_core::tape_plan::plan_chunks`; `starts` is empty when no exact
/// cut into two or more chunks exists. Runs without the GIL.
#[pyfunction]
#[pyo3(signature = (ptr, idx, poly, leaf, row_root, obj_root, target_rows))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn tape_chunk_plan<'py>(
    py: Python<'py>,
    ptr: PyReadonlyArray1<'py, i64>,
    idx: PyReadonlyArray1<'py, i64>,
    poly: PyReadonlyArray1<'py, bool>,
    leaf: PyReadonlyArray1<'py, bool>,
    row_root: PyReadonlyArray1<'py, i64>,
    obj_root: i64,
    target_rows: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let (ptr, idx, poly, leaf, row_root) = (
        ptr.as_slice()?,
        idx.as_slice()?,
        poly.as_slice()?,
        leaf.as_slice()?,
        row_root.as_slice()?,
    );
    if poly.len() + 1 != ptr.len() || leaf.len() != poly.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tape_chunk_plan: ptr must have one entry per node plus one, leaf one per node",
        ));
    }
    if obj_root < 0 || obj_root as usize >= poly.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tape_chunk_plan: objective root out of range",
        ));
    }
    let plan = py.allow_threads(|| {
        discopt_core::tape_plan::plan_chunks(ptr, idx, poly, leaf, row_root, obj_root, target_rows)
    });
    Ok((
        plan.starts.into_pyarray(py),
        plan.first_row.into_pyarray(py),
        plan.last_row.into_pyarray(py),
    ))
}
