//! PyO3 bindings for the ripopt Rust interior-point solver.
//!
//! Implements `ripopt::NlpProblem` using Python callbacks (via the evaluator
//! object), then exposes a `solve_ripopt` function to Python.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Wraps Python evaluator callbacks as a `ripopt::NlpProblem`.
///
/// Each callback acquires the GIL and calls back into the Python evaluator.
/// Bounds, structure, and initial point are cached as Rust-owned data.
/// When the evaluator provides sparse COO structure (via `has_sparse_structure`),
/// uses `evaluate_jacobian_values` / `evaluate_hessian_values` for efficient
/// value extraction. Otherwise falls back to dense evaluation.
struct PyNlpProblem {
    evaluator: PyObject,
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    /// Cached Jacobian sparsity: COO (rows, cols)
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    /// Cached Hessian sparsity: COO (rows, cols), lower triangle
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    /// Whether to use sparse value callbacks
    use_sparse_jac: bool,
    use_sparse_hess: bool,
}

impl ripopt::NlpProblem for PyNlpProblem {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        self.m
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.copy_from_slice(&self.x_l);
        x_u.copy_from_slice(&self.x_u);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l.copy_from_slice(&self.g_l);
        g_u.copy_from_slice(&self.g_u);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&self.x0);
    }

    fn objective(&self, x: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let x_py = PyArray1::from_slice(py, x);
            let result = self
                .evaluator
                .call_method1(py, "evaluate_objective", (x_py,))
                .expect("evaluate_objective failed");
            result.extract::<f64>(py).expect("objective not f64")
        })
    }

    fn gradient(&self, x: &[f64], grad: &mut [f64]) {
        Python::with_gil(|py| {
            let x_py = PyArray1::from_slice(py, x);
            let result = self
                .evaluator
                .call_method1(py, "evaluate_gradient", (x_py,))
                .expect("evaluate_gradient failed");
            let arr: PyReadonlyArray1<f64> = result
                .bind(py)
                .extract()
                .expect("gradient not ndarray");
            let slice = arr.as_slice().expect("gradient not contiguous");
            grad.copy_from_slice(slice);
        })
    }

    fn constraints(&self, x: &[f64], g: &mut [f64]) {
        if self.m == 0 {
            return;
        }
        Python::with_gil(|py| {
            let x_py = PyArray1::from_slice(py, x);
            let result = self
                .evaluator
                .call_method1(py, "evaluate_constraints", (x_py,))
                .expect("evaluate_constraints failed");
            let arr: PyReadonlyArray1<f64> = result
                .bind(py)
                .extract()
                .expect("constraints not ndarray");
            let slice = arr.as_slice().expect("constraints not contiguous");
            g.copy_from_slice(slice);
        })
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.jac_rows.clone(), self.jac_cols.clone())
    }

    fn jacobian_values(&self, x: &[f64], values: &mut [f64]) {
        if self.m == 0 {
            return;
        }
        Python::with_gil(|py| {
            let x_py = PyArray1::from_slice(py, x);
            if self.use_sparse_jac {
                // Sparse path: evaluator returns 1-D values in COO order
                let result = self
                    .evaluator
                    .call_method1(py, "evaluate_jacobian_values", (x_py,))
                    .expect("evaluate_jacobian_values failed");
                let arr: PyReadonlyArray1<f64> = result
                    .bind(py)
                    .extract()
                    .expect("jacobian_values not ndarray");
                let slice = arr.as_slice().expect("jacobian_values not contiguous");
                values.copy_from_slice(slice);
            } else {
                // Dense path: flatten (m, n) Jacobian row-major
                let result = self
                    .evaluator
                    .call_method1(py, "evaluate_jacobian", (x_py,))
                    .expect("evaluate_jacobian failed");
                let arr: PyReadonlyArray1<f64> = result
                    .call_method0(py, "flatten")
                    .expect("flatten failed")
                    .bind(py)
                    .extract()
                    .expect("jacobian not ndarray");
                let slice = arr.as_slice().expect("jacobian not contiguous");
                values.copy_from_slice(slice);
            }
        })
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], values: &mut [f64]) {
        Python::with_gil(|py| {
            let x_py = PyArray1::from_slice(py, x);
            let lambda_py = PyArray1::from_slice(py, lambda);

            if self.use_sparse_hess {
                // Sparse path: evaluator returns 1-D values in COO order
                let result = self
                    .evaluator
                    .call_method1(
                        py,
                        "evaluate_hessian_values",
                        (x_py, obj_factor, lambda_py),
                    )
                    .expect("evaluate_hessian_values failed");
                let arr: PyReadonlyArray1<f64> = result
                    .bind(py)
                    .extract()
                    .expect("hessian_values not ndarray");
                let slice = arr.as_slice().expect("hessian_values not contiguous");
                values.copy_from_slice(slice);
            } else {
                // Dense path: extract lower triangle from (n, n) array
                let result = self
                    .evaluator
                    .call_method1(
                        py,
                        "evaluate_lagrangian_hessian",
                        (x_py, obj_factor, lambda_py),
                    )
                    .expect("evaluate_lagrangian_hessian failed");

                let flat: PyReadonlyArray1<f64> = result
                    .call_method0(py, "flatten")
                    .expect("flatten failed")
                    .bind(py)
                    .extract()
                    .expect("hessian not ndarray");
                let data = flat.as_slice().expect("hessian not contiguous");
                let n = self.n;
                let mut idx = 0;
                for i in 0..n {
                    for j in 0..=i {
                        values[idx] = data[i * n + j];
                        idx += 1;
                    }
                }
            }
        })
    }
}

/// Map ripopt::SolveStatus to a Python-friendly string.
fn status_to_string(status: ripopt::SolveStatus) -> &'static str {
    match status {
        ripopt::SolveStatus::Optimal => "optimal",
        ripopt::SolveStatus::Infeasible => "infeasible",
        ripopt::SolveStatus::LocalInfeasibility => "local_infeasibility",
        ripopt::SolveStatus::MaxIterations => "max_iterations",
        ripopt::SolveStatus::NumericalError => "numerical_error",
        ripopt::SolveStatus::Unbounded => "unbounded",
        ripopt::SolveStatus::RestorationFailed => "restoration_failed",
        ripopt::SolveStatus::InternalError => "internal_error",
    }
}

/// Solve an NLP using ripopt.
///
/// Arguments:
///     evaluator: Python evaluator object with evaluate_* methods
///     x0: Initial point (n,) numpy array
///     x_l: Variable lower bounds (n,) numpy array
///     x_u: Variable upper bounds (n,) numpy array
///     g_l: Constraint lower bounds (m,) numpy array
///     g_u: Constraint upper bounds (m,) numpy array
///     options: Dict with solver options (max_iter, tol, print_level, etc.)
///
/// Returns:
///     Dict with keys: x, objective, status, iterations, constraint_multipliers
#[pyfunction]
pub fn solve_ripopt(
    py: Python<'_>,
    evaluator: PyObject,
    x0: PyReadonlyArray1<f64>,
    x_l: PyReadonlyArray1<f64>,
    x_u: PyReadonlyArray1<f64>,
    g_l: PyReadonlyArray1<f64>,
    g_u: PyReadonlyArray1<f64>,
    options: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let x0_vec: Vec<f64> = x0.as_slice()?.to_vec();
    let x_l_vec: Vec<f64> = x_l.as_slice()?.to_vec();
    let x_u_vec: Vec<f64> = x_u.as_slice()?.to_vec();
    let g_l_vec: Vec<f64> = g_l.as_slice()?.to_vec();
    let g_u_vec: Vec<f64> = g_u.as_slice()?.to_vec();

    let n = x0_vec.len();
    let m = g_l_vec.len();

    // Check if evaluator provides sparse COO structure
    let use_sparse = evaluator
        .call_method0(py, "has_sparse_structure")
        .ok()
        .and_then(|r| r.extract::<bool>(py).ok())
        .unwrap_or(false);

    let (jac_rows, jac_cols, use_sparse_jac) = if use_sparse && m > 0 {
        let structure = evaluator
            .call_method0(py, "jacobian_structure")
            .expect("jacobian_structure failed");
        let tuple = structure.bind(py).downcast::<pyo3::types::PyTuple>()
            .expect("jacobian_structure must return tuple");
        let rows: Vec<usize> = tuple
            .get_item(0).expect("missing rows")
            .extract().expect("rows not array of int");
        let cols: Vec<usize> = tuple
            .get_item(1).expect("missing cols")
            .extract().expect("cols not array of int");
        (rows, cols, true)
    } else {
        // Dense fallback: all m*n entries, row-major
        let mut rows = Vec::with_capacity(m * n);
        let mut cols = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols, false)
    };

    let (hess_rows, hess_cols, use_sparse_hess) = if use_sparse {
        let structure = evaluator
            .call_method0(py, "hessian_structure")
            .expect("hessian_structure failed");
        let tuple = structure.bind(py).downcast::<pyo3::types::PyTuple>()
            .expect("hessian_structure must return tuple");
        let rows: Vec<usize> = tuple
            .get_item(0).expect("missing rows")
            .extract().expect("rows not array of int");
        let cols: Vec<usize> = tuple
            .get_item(1).expect("missing cols")
            .extract().expect("cols not array of int");
        (rows, cols, true)
    } else {
        // Dense fallback: full lower triangle
        let hess_nnz = n * (n + 1) / 2;
        let mut rows = Vec::with_capacity(hess_nnz);
        let mut cols = Vec::with_capacity(hess_nnz);
        for i in 0..n {
            for j in 0..=i {
                rows.push(i);
                cols.push(j);
            }
        }
        (rows, cols, false)
    };

    let problem = PyNlpProblem {
        evaluator,
        n,
        m,
        x_l: x_l_vec,
        x_u: x_u_vec,
        g_l: g_l_vec,
        g_u: g_u_vec,
        x0: x0_vec,
        jac_rows,
        jac_cols,
        hess_rows,
        hess_cols,
        use_sparse_jac,
        use_sparse_hess,
    };

    // Build SolverOptions from the Python dict
    let mut opts = ripopt::SolverOptions::default();
    if let Some(val) = options.get_item("max_iter")? {
        opts.max_iter = val.extract::<usize>()?;
    }
    if let Some(val) = options.get_item("tol")? {
        opts.tol = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("print_level")? {
        opts.print_level = val.extract::<u8>()?;
    }
    if let Some(val) = options.get_item("mu_init")? {
        opts.mu_init = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("constr_viol_tol")? {
        opts.constr_viol_tol = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("max_wall_time")? {
        opts.max_wall_time = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("warm_start")? {
        opts.warm_start = val.extract::<bool>()?;
    }
    if let Some(val) = options.get_item("bound_push")? {
        opts.bound_push = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("kappa")? {
        opts.kappa = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("dual_inf_tol")? {
        opts.dual_inf_tol = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("compl_inf_tol")? {
        opts.compl_inf_tol = val.extract::<f64>()?;
    }
    // Release the GIL for the main solve loop — callbacks re-acquire it
    let result = py.allow_threads(|| ripopt::solve(&problem, &opts));

    // Build result dict
    let dict = PyDict::new(py);
    let x_out = PyArray1::from_vec(py, result.x);
    dict.set_item("x", x_out)?;
    dict.set_item("objective", result.objective)?;
    dict.set_item("status", status_to_string(result.status))?;
    dict.set_item("iterations", result.iterations)?;
    let mults = PyArray1::from_vec(py, result.constraint_multipliers);
    dict.set_item("constraint_multipliers", mults)?;

    Ok(dict.into_any().unbind())
}
