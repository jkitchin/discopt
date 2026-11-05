//! `fj_search_py`: the Feasibility Jump primal heuristic (highs-parity P4.6), exposed
//! for its entry experiment. The logic is `discopt_core::bnb::primal_fj`; this file
//! only marshals.

use std::time::{Duration, Instant};

use discopt_core::bnb::primal_fj::fj_search;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `(x or None, moves, weight_bumps)` for `row_lo <= A x <= row_hi`, `lb <= x <= ub`,
/// integer where `is_int`; `A` in CSR (`m` x `n`). Infinite sides may be `±inf`.
#[pyfunction]
#[pyo3(signature = (m, n, indptr, indices, data, row_lo, row_hi, lb, ub, is_int, seed=0, max_moves=1_000_000, time_limit_s=None))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn fj_search_py<'py>(
    py: Python<'py>,
    m: usize,
    n: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f64>,
    row_lo: PyReadonlyArray1<'py, f64>,
    row_hi: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    is_int: PyReadonlyArray1<'py, bool>,
    seed: u64,
    max_moves: u64,
    time_limit_s: Option<f64>,
) -> PyResult<(Option<Bound<'py, PyArray1<f64>>>, u64, u64)> {
    let ip: Vec<usize> = indptr.as_slice()?.iter().map(|&v| v as usize).collect();
    let ix: Vec<usize> = indices.as_slice()?.iter().map(|&v| v as usize).collect();
    let (d, rl, rh) = (data.as_slice()?, row_lo.as_slice()?, row_hi.as_slice()?);
    let (l, u, ii) = (lb.as_slice()?, ub.as_slice()?, is_int.as_slice()?);
    if ip.len() != m + 1 || rl.len() != m || rh.len() != m {
        return Err(PyValueError::new_err(
            "fj_search_py: row arrays do not match m",
        ));
    }
    if l.len() != n || u.len() != n || ii.len() != n {
        return Err(PyValueError::new_err(
            "fj_search_py: column arrays do not match n",
        ));
    }
    if ix.len() != d.len() || ix.iter().any(|&j| j >= n) {
        return Err(PyValueError::new_err("fj_search_py: bad CSR indices"));
    }
    let deadline = time_limit_s.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
    let r = py.allow_threads(|| {
        fj_search(
            m, n, &ip, &ix, d, rl, rh, l, u, ii, seed, max_moves, deadline,
        )
    });
    Ok((
        r.x.map(|x| PyArray1::from_vec(py, x)),
        r.moves,
        r.weight_bumps,
    ))
}
