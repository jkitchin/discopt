//! `MilpHandle` pyclass: a persistent in-house MILP, edited in place and re-solved
//! (highs-parity P3.1). The logic is `discopt_core::bnb::milp_handle::MilpHandle`;
//! this file only marshals. A solve takes the same keyword arguments, defaults and
//! [`milp_options`] construction as `solve_milp_csc_py`, and returns the same
//! 6-tuple, so a handle solve and a fresh `solve_milp_csc_py` call on the same arrays
//! are the same driver run (checked bit for bit in `discopt-core` and in
//! `python/tests/test_highs_parity_p31_milp_handle.py`).

use discopt_core::bnb::milp_driver::MilpStatus;
use discopt_core::bnb::milp_handle::MilpHandle as CoreHandle;
use discopt_core::lp::simplex::sparse::SparseCols;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::lp_bindings::milp_options;

/// `solve_milp_csc_py`'s return: `(status, x, obj, bound, nodes, lp_iters)`.
type SolveTuple<'py> = (String, Bound<'py, PyArray1<f64>>, f64, f64, usize, usize);

#[pyclass(name = "MilpHandle", module = "discopt._rust")]
pub struct PyMilpHandle {
    inner: CoreHandle,
}

fn status_str(s: MilpStatus) -> &'static str {
    match s {
        MilpStatus::Optimal => "optimal",
        MilpStatus::Feasible => "feasible",
        MilpStatus::Infeasible => "infeasible",
        MilpStatus::Unbounded => "unbounded",
        MilpStatus::NodeLimit => "node_limit",
    }
}

#[pymethods]
impl PyMilpHandle {
    /// Same arrays as `solve_milp_csc_py`: `A x = b` in CSC (`m` x `n`), column box
    /// `lb`/`ub`, structural columns `[0, n_struct)`, `integer_cols` among them.
    #[new]
    #[pyo3(signature = (c, m, n, col_ptr, row_idx, vals, b, lb, ub, integer_cols, n_struct,
                        obj_const=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        c: PyReadonlyArray1<'_, f64>,
        m: usize,
        n: usize,
        col_ptr: PyReadonlyArray1<'_, i64>,
        row_idx: PyReadonlyArray1<'_, i64>,
        vals: PyReadonlyArray1<'_, f64>,
        b: PyReadonlyArray1<'_, f64>,
        lb: PyReadonlyArray1<'_, f64>,
        ub: PyReadonlyArray1<'_, f64>,
        integer_cols: PyReadonlyArray1<'_, i64>,
        n_struct: usize,
        obj_const: f64,
    ) -> PyResult<Self> {
        let cp_raw = col_ptr.as_slice()?;
        let ri_raw = row_idx.as_slice()?;
        let vals_v = vals.as_slice()?.to_vec();
        if cp_raw.len() != n + 1 || cp_raw[0] != 0 || ri_raw.len() != vals_v.len() {
            return Err(PyValueError::new_err(
                "MilpHandle: col_ptr must have length n+1 starting at 0, and row_idx/vals \
                 must have equal length",
            ));
        }
        if cp_raw.windows(2).any(|w| w[1] < w[0]) || cp_raw[n] as usize != vals_v.len() {
            return Err(PyValueError::new_err(
                "MilpHandle: col_ptr must be non-decreasing and end at nnz",
            ));
        }
        if ri_raw.iter().any(|&r| r < 0) {
            return Err(PyValueError::new_err("MilpHandle: negative row index"));
        }
        let ints: Vec<usize> = integer_cols
            .as_slice()?
            .iter()
            .map(|&v| {
                usize::try_from(v)
                    .map_err(|_| PyValueError::new_err("MilpHandle: negative integer column"))
            })
            .collect::<PyResult<_>>()?;
        let csc = SparseCols::from_csc(
            cp_raw.iter().map(|&x| x as usize).collect(),
            ri_raw.iter().map(|&x| x as usize).collect(),
            vals_v,
        );
        let inner = CoreHandle::new(
            csc,
            m,
            n,
            c.as_slice()?.to_vec(),
            b.as_slice()?.to_vec(),
            lb.as_slice()?.to_vec(),
            ub.as_slice()?.to_vec(),
            n_struct,
            &ints,
            obj_const,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Set column `idx`'s box (structural or logical).
    fn change_col_bounds(&mut self, idx: usize, lo: f64, hi: f64) -> PyResult<()> {
        self.inner
            .change_col_bounds(idx, lo, hi)
            .map_err(PyValueError::new_err)
    }

    /// Make structural column `idx` integer (`True`) or continuous.
    fn change_integrality(&mut self, idx: usize, is_int: bool) -> PyResult<()> {
        self.inner
            .change_integrality(idx, is_int)
            .map_err(PyValueError::new_err)
    }

    /// Replace every column's box (structural and logical) at once.
    fn set_col_bounds_all(
        &mut self,
        lb: PyReadonlyArray1<'_, f64>,
        ub: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        self.inner
            .set_col_bounds_all(lb.as_slice()?, ub.as_slice()?)
            .map_err(PyValueError::new_err)
    }

    /// Replace the structural columns' integrality at once.
    fn set_integrality_all(&mut self, is_int: PyReadonlyArray1<'_, bool>) -> PyResult<()> {
        self.inner
            .set_integrality_all(is_int.as_slice()?)
            .map_err(PyValueError::new_err)
    }

    /// Seed (`x` over the structural columns) or clear (`None`) the incumbent the
    /// next solves start from. Validated by the driver, never trusted.
    #[pyo3(signature = (x=None))]
    fn set_solution(&mut self, x: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<()> {
        let v = x.map(|a| a.as_slice().map(|s| s.to_vec())).transpose()?;
        self.inner.set_solution(v).map_err(PyValueError::new_err)
    }

    /// `(lb, ub)` of every column, as the handle currently holds them.
    fn col_bounds<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (l, u) = self.inner.col_bounds();
        (
            PyArray1::from_vec(py, l.to_vec()),
            PyArray1::from_vec(py, u.to_vec()),
        )
    }

    /// The integer columns, ascending.
    fn integer_cols(&self) -> Vec<usize> {
        self.inner.integer_cols()
    }

    /// Solve the current problem. Keyword arguments and defaults are
    /// `solve_milp_csc_py`'s; returns its `(status, x, obj, bound, nodes, lp_iters)`.
    #[pyo3(signature = (max_nodes=1_000_000, gap_tol=1e-6, tol=1e-9, root_cuts=16,
                        cut_rounds=1, gmi_cuts=true, cmir_cuts=false, cut_select=false,
                        node_cuts=false, max_pool_cuts=128, heuristics=true, presolve=true,
                        strong_branch=true, node_propagation=true, reduced_cost_fixing=true,
                        sb_max_cands=6, sb_node_budget=48, time_limit_s=None,
                        root_cut_time_s=None, root_cut_prune=true, abs_gap_tol=None,
                        max_open_nodes=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        max_nodes: usize,
        gap_tol: f64,
        tol: f64,
        root_cuts: usize,
        cut_rounds: usize,
        gmi_cuts: bool,
        cmir_cuts: bool,
        cut_select: bool,
        node_cuts: bool,
        max_pool_cuts: usize,
        heuristics: bool,
        presolve: bool,
        strong_branch: bool,
        node_propagation: bool,
        reduced_cost_fixing: bool,
        sb_max_cands: usize,
        sb_node_budget: usize,
        time_limit_s: Option<f64>,
        root_cut_time_s: Option<f64>,
        root_cut_prune: bool,
        abs_gap_tol: Option<f64>,
        max_open_nodes: Option<usize>,
    ) -> PyResult<SolveTuple<'py>> {
        // n_struct, integer columns and the seed are the handle's; `options` fills
        // them in over these.
        let base = milp_options(
            0,
            Vec::new(),
            max_nodes,
            max_open_nodes,
            time_limit_s,
            gap_tol,
            abs_gap_tol,
            tol,
            root_cuts,
            cut_rounds,
            gmi_cuts,
            cmir_cuts,
            cut_select,
            node_cuts,
            max_pool_cuts,
            heuristics,
            presolve,
            strong_branch,
            node_propagation,
            reduced_cost_fixing,
            sb_max_cands,
            sb_node_budget,
            0,
            0,
            None,
            root_cut_time_s,
            root_cut_prune,
        )?;
        let inner = &self.inner;
        let res = py.allow_threads(|| {
            let r = inner.solve(&base);
            discopt_core::profile::dump();
            r
        });
        Ok((
            status_str(res.status).to_string(),
            PyArray1::from_vec(py, res.x),
            res.obj,
            res.bound,
            res.nodes,
            res.lp_iters,
        ))
    }

    /// Whether the driver's root preparation and presolve prove the current box
    /// empty: the verdict a `presolve=True` solve returns as `infeasible` with zero
    /// nodes. `False` decides nothing.
    #[pyo3(signature = (tol=1e-9))]
    fn presolve_only(&self, py: Python<'_>, tol: f64) -> PyResult<bool> {
        let base = milp_options(
            0,
            Vec::new(),
            1_000_000,
            None,
            None,
            1e-6,
            None,
            tol,
            16,
            1,
            true,
            false,
            false,
            false,
            128,
            true,
            true,
            true,
            true,
            true,
            6,
            48,
            0,
            0,
            None,
            None,
            true,
        )?;
        let inner = &self.inner;
        Ok(py.allow_threads(|| inner.presolve_only(&base)))
    }
}
