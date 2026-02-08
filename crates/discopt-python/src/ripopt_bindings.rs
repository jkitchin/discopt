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
struct PyNlpProblem {
    evaluator: PyObject,
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    /// Cached Jacobian sparsity: (rows, cols) — dense pattern
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    /// Cached Hessian sparsity: (rows, cols) — lower triangle dense pattern
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
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
            // Dense Jacobian flattened row-major: values in same order as structure
            values.copy_from_slice(slice);
        })
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], values: &mut [f64]) {
        Python::with_gil(|py| {
            let x_py = PyArray1::from_slice(py, x);
            let lambda_py = PyArray1::from_slice(py, lambda);

            let result = self
                .evaluator
                .call_method1(
                    py,
                    "evaluate_lagrangian_hessian",
                    (x_py, obj_factor, lambda_py),
                )
                .expect("evaluate_lagrangian_hessian failed");

            // Result is (n, n) dense array. Extract lower triangle values
            // matching hessian_structure order (row-major lower triangle).
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
        })
    }
}

/// Map ripopt::SolveStatus to a Python-friendly string.
fn status_to_string(status: ripopt::SolveStatus) -> &'static str {
    match status {
        ripopt::SolveStatus::Optimal => "optimal",
        ripopt::SolveStatus::Acceptable => "acceptable",
        ripopt::SolveStatus::Infeasible => "infeasible",
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

    // Build dense Jacobian structure (m x n, row-major)
    let mut jac_rows = Vec::with_capacity(m * n);
    let mut jac_cols = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            jac_rows.push(i);
            jac_cols.push(j);
        }
    }

    // Build dense lower-triangle Hessian structure
    let hess_nnz = n * (n + 1) / 2;
    let mut hess_rows = Vec::with_capacity(hess_nnz);
    let mut hess_cols = Vec::with_capacity(hess_nnz);
    for i in 0..n {
        for j in 0..=i {
            hess_rows.push(i);
            hess_cols.push(j);
        }
    }

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
    if let Some(val) = options.get_item("acceptable_tol")? {
        opts.acceptable_tol = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("acceptable_iter")? {
        opts.acceptable_iter = val.extract::<usize>()?;
    }
    if let Some(val) = options.get_item("mu_init")? {
        opts.mu_init = val.extract::<f64>()?;
    }
    if let Some(val) = options.get_item("constr_viol_tol")? {
        opts.constr_viol_tol = val.extract::<f64>()?;
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
