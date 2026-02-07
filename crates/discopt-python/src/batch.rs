//! Batch dispatch interface for Rust <-> Python/JAX zero-copy array transfer.
//!
//! Provides `PyBatchDispatcher` which manages pending B&B nodes and exports
//! their bounds as numpy arrays with zero-copy semantics. Results (relaxation
//! bounds, solutions, feasibility) are imported back via zero-copy readonly arrays.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Result data for a solved node.
#[derive(Debug, Clone)]
struct NodeResult {
    lower_bound: f64,
    solution: Vec<f64>,
    feasible: bool,
}

/// A batch dispatcher exposed to Python for zero-copy array exchange.
///
/// This is a self-contained implementation that demonstrates the zero-copy
/// interface. It will later wrap the real `TreeManager` from discopt-core
/// once T11 is integrated.
#[pyclass]
pub struct PyBatchDispatcher {
    n_vars: usize,
    pending_lb: Vec<Vec<f64>>,
    pending_ub: Vec<Vec<f64>>,
    node_ids: Vec<usize>,
    next_id: usize,
    results: Vec<(usize, NodeResult)>,
    last_export_ptr: usize,
}

#[pymethods]
impl PyBatchDispatcher {
    #[new]
    fn new(n_vars: usize) -> PyResult<Self> {
        if n_vars == 0 {
            return Err(PyValueError::new_err("n_vars must be positive"));
        }
        Ok(Self {
            n_vars,
            pending_lb: Vec::new(),
            pending_ub: Vec::new(),
            node_ids: Vec::new(),
            next_id: 0,
            results: Vec::new(),
            last_export_ptr: 0,
        })
    }

    /// Add a node with given lower and upper bounds.
    /// Returns the node ID assigned.
    fn add_node(&mut self, lb: Vec<f64>, ub: Vec<f64>) -> PyResult<usize> {
        if lb.len() != self.n_vars {
            return Err(PyValueError::new_err(format!(
                "lb length {} != n_vars {}",
                lb.len(),
                self.n_vars
            )));
        }
        if ub.len() != self.n_vars {
            return Err(PyValueError::new_err(format!(
                "ub length {} != n_vars {}",
                ub.len(),
                self.n_vars
            )));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.pending_lb.push(lb);
        self.pending_ub.push(ub);
        self.node_ids.push(id);
        Ok(id)
    }

    /// Export a batch of up to `batch_size` pending nodes as numpy arrays.
    ///
    /// Returns `(lb_array[N, n_vars], ub_array[N, n_vars], node_ids[N])`.
    /// The numpy arrays own the Rust-allocated memory (zero-copy transfer).
    #[allow(clippy::type_complexity)]
    fn export_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let n = batch_size.min(self.pending_lb.len());

        if n == 0 {
            // Return empty arrays with correct shape
            let lb = numpy::PyArray2::zeros(py, [0, self.n_vars], false);
            let ub = numpy::PyArray2::zeros(py, [0, self.n_vars], false);
            let ids = numpy::PyArray1::zeros(py, [0], false);
            self.last_export_ptr = 0;
            return Ok((lb, ub, ids));
        }

        // Drain the first `n` pending nodes
        let drain_lb: Vec<Vec<f64>> = self.pending_lb.drain(..n).collect();
        let drain_ub: Vec<Vec<f64>> = self.pending_ub.drain(..n).collect();
        let drain_ids: Vec<usize> = self.node_ids.drain(..n).collect();

        // Flatten into contiguous arrays for zero-copy transfer
        let n_vars = self.n_vars;
        let mut flat_lb: Vec<f64> = Vec::with_capacity(n * n_vars);
        let mut flat_ub: Vec<f64> = Vec::with_capacity(n * n_vars);
        for row in &drain_lb {
            flat_lb.extend_from_slice(row);
        }
        for row in &drain_ub {
            flat_ub.extend_from_slice(row);
        }

        let ids_i64: Vec<i64> = drain_ids.iter().map(|&id| id as i64).collect();

        // Zero-copy: into_pyarray transfers ownership of the Vec's buffer to numpy.
        let lb_array = PyArray1::from_vec(py, flat_lb).reshape([n, n_vars])?;
        let ub_array = PyArray1::from_vec(py, flat_ub).reshape([n, n_vars])?;
        let ids_array = PyArray1::from_vec(py, ids_i64);

        // Store the data pointer for zero-copy verification
        self.last_export_ptr = lb_array.as_raw_array().as_ptr() as usize;

        Ok((lb_array, ub_array, ids_array))
    }

    /// Import results for a batch of solved nodes.
    ///
    /// - `node_ids`: `[N]` array of node IDs
    /// - `lower_bounds`: `[N]` array of relaxation lower bounds
    /// - `solutions`: `[N, n_vars]` array of relaxation solutions
    /// - `feasible`: `[N]` boolean array (is solution integer-feasible?)
    fn import_results(
        &mut self,
        node_ids: PyReadonlyArray1<i64>,
        lower_bounds: PyReadonlyArray1<f64>,
        solutions: PyReadonlyArray2<f64>,
        feasible: PyReadonlyArray1<bool>,
    ) -> PyResult<()> {
        let ids = node_ids.as_array();
        let lbs = lower_bounds.as_array();
        let sols = solutions.as_array();
        let feas = feasible.as_array();

        let n = ids.len();
        if lbs.len() != n || feas.len() != n {
            return Err(PyValueError::new_err(
                "All input arrays must have the same first dimension",
            ));
        }
        if sols.shape()[0] != n {
            return Err(PyValueError::new_err(format!(
                "solutions first dimension {} != {}",
                sols.shape()[0],
                n
            )));
        }
        if n > 0 && sols.shape()[1] != self.n_vars {
            return Err(PyValueError::new_err(format!(
                "solutions second dimension {} != n_vars {}",
                sols.shape()[1],
                self.n_vars
            )));
        }

        for i in 0..n {
            let nid = ids[i] as usize;
            let sol_row: Vec<f64> = sols.row(i).to_vec();
            self.results.push((
                nid,
                NodeResult {
                    lower_bound: lbs[i],
                    solution: sol_row,
                    feasible: feas[i],
                },
            ));
        }

        Ok(())
    }

    /// Get the data pointer of the last exported lb array (for zero-copy verification).
    fn last_export_ptr(&self) -> usize {
        self.last_export_ptr
    }

    /// Number of pending (un-exported) nodes.
    fn pending_count(&self) -> usize {
        self.pending_lb.len()
    }

    /// Number of results received.
    fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Number of variables per node.
    fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Get the result for a specific node ID, if available.
    /// Returns `(lower_bound, solution, feasible)` or None.
    fn get_result(&self, py: Python<'_>, node_id: usize) -> Option<(f64, PyObject, bool)> {
        for (nid, res) in &self.results {
            if *nid == node_id {
                let sol = PyArray1::from_vec(py, res.solution.clone()).into_any().unbind();
                return Some((res.lower_bound, sol, res.feasible));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        pyo3::prepare_freethreaded_python();
        let mut dispatcher = PyBatchDispatcher::new(3).unwrap();
        let id = dispatcher.add_node(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(id, 0);
        assert_eq!(dispatcher.pending_count(), 1);
    }

    #[test]
    fn test_add_node_wrong_size() {
        pyo3::prepare_freethreaded_python();
        let mut dispatcher = PyBatchDispatcher::new(3).unwrap();
        let result = dispatcher.add_node(vec![0.0, 0.0], vec![1.0, 1.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_nvars() {
        pyo3::prepare_freethreaded_python();
        let result = PyBatchDispatcher::new(0);
        assert!(result.is_err());
    }
}
