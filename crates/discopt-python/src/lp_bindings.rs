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

use discopt_core::bnb::milp_driver::{solve_milp as core_solve_milp, MilpOptions, MilpStatus};
use discopt_core::lp::basis::recover_basis;
use discopt_core::lp::crossover::{crossover_to_vertex, LpView};
use discopt_core::lp::gomory::separate_gomory;
use discopt_core::lp::mir::separate_mir;
use discopt_core::lp::simplex::{solve_lp as simplex_solve_lp, LpStatus, SimplexOptions};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
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

/// Separate Gomory mixed-integer cuts at the vertex `x` of the standard-form LP.
///
/// Recovers a basis at `x`, reconstructs the vertex from the exact basis +
/// bounds + `b` (the length-`m` rhs of `A x = b`), and derives one GMI cut per
/// fractional integer basic variable. `integrality` is a length-`n` bool array.
/// Returns `(coeffs, rhs)` — `coeffs` is a `k × n` array and `rhs` length `k`,
/// the cuts `coeffs[i] · x ≥ rhs[i]` over the standard-form variables — or
/// `None` when `x` is not a basic feasible solution (basis recovery declined).
#[pyfunction]
#[pyo3(signature = (x, a, b, c, lb, ub, integrality, tol=1e-7, max_dynamism=1e7))]
pub fn gomory_cuts_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    c: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, bool>,
    tol: f64,
    max_dynamism: f64,
) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>> {
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
    let xs = x.as_slice()?;
    let basis = match recover_basis(xs, &lp, tol) {
        Some(bs) => bs,
        None => return Ok(None),
    };
    let cuts = separate_gomory(
        &lp,
        b.as_slice()?,
        &basis,
        integrality.as_slice()?,
        tol,
        max_dynamism,
    );

    let k = cuts.len();
    let mut flat = Vec::with_capacity(k * n);
    let mut rhs = Vec::with_capacity(k);
    for cut in &cuts {
        flat.extend_from_slice(&cut.coeffs);
        rhs.push(cut.rhs);
    }
    let coeffs = PyArray1::from_vec(py, flat).reshape([k, n])?;
    Ok(Some((coeffs, PyArray1::from_vec(py, rhs))))
}

/// Separate MIR cuts from the `≤` rows `a_ub · x ≤ b_ub` at point `x`.
///
/// `a_ub` is C-contiguous `m × n`; `lb` the length-`n` lower bounds;
/// `integrality` a length-`n` bool array. Returns `(coeffs, rhs)` — a `k × n`
/// array and length-`k` rhs, the cuts `coeffs[i] · x ≤ rhs[i]` over the
/// structural variables — or `None` when no cut is produced.
#[pyfunction]
#[pyo3(signature = (a_ub, b_ub, lb, integrality, x, tol=1e-7, max_dynamism=1e7))]
pub fn mir_cuts_py<'py>(
    py: Python<'py>,
    a_ub: PyReadonlyArray2<'py, f64>,
    b_ub: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    integrality: PyReadonlyArray1<'py, bool>,
    x: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_dynamism: f64,
) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>> {
    let dims = a_ub.shape();
    let n = dims[1];
    let a_flat = a_ub
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a_ub` must be C-contiguous"))?;
    let cuts = separate_mir(
        a_flat,
        b_ub.as_slice()?,
        lb.as_slice()?,
        integrality.as_slice()?,
        x.as_slice()?,
        tol,
        max_dynamism,
    );
    let k = cuts.len();
    if k == 0 {
        return Ok(None);
    }
    let mut flat = Vec::with_capacity(k * n);
    let mut rhs = Vec::with_capacity(k);
    for cut in &cuts {
        flat.extend_from_slice(&cut.coeffs);
        rhs.push(cut.rhs);
    }
    let coeffs = PyArray1::from_vec(py, flat).reshape([k, n])?;
    Ok(Some((coeffs, PyArray1::from_vec(py, rhs))))
}

/// Solve a standard-form LP `min cᵀx s.t. A x = b, lb ≤ x ≤ ub` with the
/// warm-startable revised simplex (cold start). `a` is C-contiguous `m × n`.
/// Returns `(status, x, obj, iters)` where status is one of `optimal`,
/// `infeasible`, `unbounded`, `iter_limit`, `numerical`. For validation against
/// HiGHS / Netlib.
#[pyfunction]
#[pyo3(signature = (c, a, b, lb, ub, tol=1e-9, max_iter=100_000))]
pub fn solve_lp_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(String, Bound<'py, PyArray1<f64>>, f64, usize)> {
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
    let opts = SimplexOptions { tol, max_iter };
    let sol = simplex_solve_lp(&lp, b.as_slice()?, &opts);
    let status = match sol.status {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
        LpStatus::IterLimit => "iter_limit",
        LpStatus::Numerical => "numerical",
    };
    Ok((
        status.to_string(),
        PyArray1::from_vec(py, sol.x),
        sol.obj,
        sol.iters,
    ))
}

/// Solve a pure MILP `min cᵀx + obj_const s.t. A x = b, lb ≤ x ≤ ub`, with the
/// columns in `integer_cols` integer-constrained, by the Rust-internal
/// warm-started-simplex branch and bound. `a` is C-contiguous `m × n` standard
/// form (structural columns `[0, n_struct)`, slacks after). Returns
/// `(status, x[n_struct], obj, bound, nodes, lp_iters)` where status is one of
/// `optimal`, `feasible`, `infeasible`, `unbounded`, `node_limit`.
#[pyfunction]
#[pyo3(signature = (c, a, b, lb, ub, integer_cols, n_struct, obj_const=0.0,
                    max_nodes=1_000_000, gap_tol=1e-6, tol=1e-9, root_cuts=16,
                    cut_rounds=1, node_cuts=false, max_pool_cuts=128, heuristics=true,
                    presolve=true, strong_branch=true, sb_max_cands=8, sb_node_budget=1024))]
pub fn solve_milp_py<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    lb: PyReadonlyArray1<'py, f64>,
    ub: PyReadonlyArray1<'py, f64>,
    integer_cols: PyReadonlyArray1<'py, i64>,
    n_struct: usize,
    obj_const: f64,
    max_nodes: usize,
    gap_tol: f64,
    tol: f64,
    root_cuts: usize,
    cut_rounds: usize,
    node_cuts: bool,
    max_pool_cuts: usize,
    heuristics: bool,
    presolve: bool,
    strong_branch: bool,
    sb_max_cands: usize,
    sb_node_budget: usize,
) -> PyResult<(String, Bound<'py, PyArray1<f64>>, f64, f64, usize, usize)> {
    let dims = a.shape();
    let (m, n) = (dims[0], dims[1]);
    // Materialize owned copies of the borrowed numpy inputs. `PyReadonlyArray`
    // borrows are only valid while the GIL is held, so we copy before releasing
    // it; the copies (not the numpy buffers) feed the solve. The solve is long
    // and touches no Python objects, so it runs under `py.allow_threads` — this
    // unblocks the interpreter and lets the core's rayon workers (when built with
    // `parallel`) run without contending on the GIL.
    let a_owned: Vec<f64> = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("`a` must be C-contiguous"))?
        .to_vec();
    let c_owned: Vec<f64> = c.as_slice()?.to_vec();
    let b_owned: Vec<f64> = b.as_slice()?.to_vec();
    let l_owned: Vec<f64> = lb.as_slice()?.to_vec();
    let u_owned: Vec<f64> = ub.as_slice()?.to_vec();
    let int_cols: Vec<usize> = integer_cols
        .as_slice()?
        .iter()
        .map(|&v| v as usize)
        .collect();
    let opts = MilpOptions {
        n_struct,
        integer_cols: int_cols,
        max_nodes,
        gap_tol,
        root_cuts,
        cut_rounds,
        node_cuts,
        max_pool_cuts,
        heuristics,
        presolve,
        strong_branch,
        sb_max_cands,
        sb_node_budget,
        simplex: SimplexOptions {
            tol,
            max_iter: 100_000,
        },
    };
    let res = py.allow_threads(|| {
        let lp = LpView {
            a: &a_owned,
            m,
            n,
            c: &c_owned,
            l: &l_owned,
            u: &u_owned,
        };
        core_solve_milp(&lp, &b_owned, obj_const, &opts)
    });
    let status = match res.status {
        MilpStatus::Optimal => "optimal",
        MilpStatus::Feasible => "feasible",
        MilpStatus::Infeasible => "infeasible",
        MilpStatus::Unbounded => "unbounded",
        MilpStatus::NodeLimit => "node_limit",
    };
    Ok((
        status.to_string(),
        PyArray1::from_vec(py, res.x),
        res.obj,
        res.bound,
        res.nodes,
        res.lp_iters,
    ))
}
