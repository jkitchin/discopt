//! PyO3 bindings for the Branch-and-Bound TreeManager.
//!
//! Wraps `discopt_core::bnb::TreeManager` with zero-copy numpy array interface
//! for Python orchestration of the B&B solve loop.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use discopt_core::bnb::{NodeId, NodeResult, SelectionStrategy, TreeManager, VarBranchInfo};

/// Python wrapper around the Rust TreeManager for B&B search.
#[pyclass]
pub struct PyTreeManager {
    inner: TreeManager,
    n_vars: usize,
}

#[pymethods]
impl PyTreeManager {
    /// Create a new TreeManager.
    ///
    /// Args:
    ///     n_vars: Number of decision variables.
    ///     lb: Global lower bounds for each variable.
    ///     ub: Global upper bounds for each variable.
    ///     int_var_offsets: Flat offsets of integer variable groups.
    ///     int_var_sizes: Sizes of integer variable groups.
    ///     strategy: Node selection strategy ("best_first" or "depth_first").
    #[new]
    fn new(
        n_vars: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
        int_var_offsets: Vec<usize>,
        int_var_sizes: Vec<usize>,
        strategy: &str,
    ) -> PyResult<Self> {
        if lb.len() != n_vars {
            return Err(PyValueError::new_err(format!(
                "lb length {} != n_vars {}",
                lb.len(),
                n_vars
            )));
        }
        if ub.len() != n_vars {
            return Err(PyValueError::new_err(format!(
                "ub length {} != n_vars {}",
                ub.len(),
                n_vars
            )));
        }
        if int_var_offsets.len() != int_var_sizes.len() {
            return Err(PyValueError::new_err(
                "int_var_offsets and int_var_sizes must have the same length",
            ));
        }

        let sel_strategy = match strategy {
            "best_first" => SelectionStrategy::BestFirst,
            "depth_first" => SelectionStrategy::DepthFirst,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown strategy: '{}'. Use 'best_first' or 'depth_first'.",
                    strategy
                )));
            }
        };

        let integer_vars: Vec<VarBranchInfo> = int_var_offsets
            .iter()
            .zip(int_var_sizes.iter())
            .map(|(&offset, &size)| VarBranchInfo {
                offset,
                size,
                is_integer: true,
            })
            .collect();

        let tm = TreeManager::new(n_vars, lb, ub, integer_vars, sel_strategy);
        Ok(Self { inner: tm, n_vars })
    }

    /// Initialize the tree with the root node.
    fn initialize(&mut self) {
        self.inner.initialize();
    }

    /// Export a batch of up to `batch_size` pending nodes as numpy arrays.
    ///
    /// Returns (lb_array[N, n_vars], ub_array[N, n_vars], node_ids[N],
    ///          parent_solutions[N, n_vars]).
    /// Parent solutions are filled with NaN where no parent solution exists
    /// (root node).
    #[allow(clippy::type_complexity)]
    fn export_batch<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let batch = self.inner.export_batch(batch_size);
        let n = batch.node_ids.len();

        if n == 0 {
            let lb = numpy::PyArray2::zeros(py, [0, self.n_vars], false);
            let ub = numpy::PyArray2::zeros(py, [0, self.n_vars], false);
            let ids = numpy::PyArray1::zeros(py, [0], false);
            let psols = numpy::PyArray2::zeros(py, [0, self.n_vars], false);
            return Ok((lb, ub, ids, psols));
        }

        let n_vars = self.n_vars;
        let mut flat_lb: Vec<f64> = Vec::with_capacity(n * n_vars);
        let mut flat_ub: Vec<f64> = Vec::with_capacity(n * n_vars);
        let mut flat_psols: Vec<f64> = Vec::with_capacity(n * n_vars);
        for row in &batch.lb {
            flat_lb.extend_from_slice(row);
        }
        for row in &batch.ub {
            flat_ub.extend_from_slice(row);
        }
        for psol in &batch.parent_solutions {
            match psol {
                Some(sol) => flat_psols.extend_from_slice(sol),
                None => flat_psols.extend(std::iter::repeat(f64::NAN).take(n_vars)),
            }
        }

        let ids_i64: Vec<i64> = batch.node_ids.iter().map(|nid| nid.0 as i64).collect();

        let lb_array = PyArray1::from_vec(py, flat_lb).reshape([n, n_vars])?;
        let ub_array = PyArray1::from_vec(py, flat_ub).reshape([n, n_vars])?;
        let ids_array = PyArray1::from_vec(py, ids_i64);
        let psols_array = PyArray1::from_vec(py, flat_psols).reshape([n, n_vars])?;

        Ok((lb_array, ub_array, ids_array, psols_array))
    }

    /// Import relaxation results for a batch of solved nodes.
    ///
    /// Args:
    ///     node_ids: [N] array of node IDs.
    ///     lower_bounds: [N] array of relaxation lower bounds.
    ///     solutions: [N, n_vars] array of relaxation solutions.
    ///     feasible: [N] boolean array (is solution integer-feasible?).
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

        let results: Vec<NodeResult> = (0..n)
            .map(|i| {
                let nid = discopt_core::bnb::NodeId(ids[i] as usize);
                NodeResult {
                    node_id: nid,
                    lower_bound: lbs[i],
                    solution: sols.row(i).to_vec(),
                    is_feasible: feas[i],
                }
            })
            .collect();

        self.inner.import_results(&results);
        Ok(())
    }

    /// Process all evaluated nodes: prune, check integrality, branch.
    ///
    /// Returns a dict with {pruned, fathomed, branched, incumbent_updates,
    /// unreliable_candidates}.
    fn process_evaluated<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.process_evaluated();
        let dict = PyDict::new(py);
        dict.set_item("pruned", stats.pruned)?;
        dict.set_item("fathomed", stats.fathomed)?;
        dict.set_item("branched", stats.branched)?;
        dict.set_item("incumbent_updates", stats.incumbent_updates)?;
        let unreliable: Vec<i64> = stats
            .unreliable_candidates
            .iter()
            .map(|&x| x as i64)
            .collect();
        dict.set_item("unreliable_candidates", PyArray1::from_vec(py, unreliable))?;
        Ok(dict)
    }

    /// Check if the search is complete.
    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Current relative optimality gap.
    fn gap(&self) -> f64 {
        self.inner.gap()
    }

    /// Get aggregate tree statistics as a dict.
    ///
    /// Returns {total_nodes, open_nodes, incumbent_value, global_lower_bound, gap}.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("total_nodes", s.total_nodes)?;
        dict.set_item("open_nodes", s.open_nodes)?;
        dict.set_item("incumbent_value", s.incumbent_value)?;
        dict.set_item("global_lower_bound", s.global_lower_bound)?;
        dict.set_item("gap", s.gap)?;
        Ok(dict)
    }

    /// Get the current incumbent solution, if any.
    ///
    /// Returns (solution_array, objective_value) or None.
    fn incumbent<'py>(&self, py: Python<'py>) -> Option<(Bound<'py, PyArray1<f64>>, f64)> {
        self.inner.incumbent().map(|(sol, val)| {
            let arr = PyArray1::from_vec(py, sol.to_vec());
            (arr, val)
        })
    }

    /// Inject an externally-found incumbent (e.g. from a primal heuristic).
    ///
    /// Updates the incumbent only if obj_val improves on the current best.
    /// Returns True if the incumbent was updated.
    fn inject_incumbent(&mut self, solution: PyReadonlyArray1<f64>, obj_val: f64) -> bool {
        let sol_vec = solution.as_slice().unwrap().to_vec();
        self.inner.inject_incumbent(sol_vec, obj_val)
    }

    /// Enable nonconvex spatial branching mode.
    ///
    /// When enabled, integer-feasible nodes are NOT fathomed.  Instead,
    /// spatial branching on continuous variables continues to close the gap
    /// between the convex relaxation bound and the NLP incumbent.
    fn set_nonconvex(&mut self, nonconvex: bool) {
        self.inner.set_nonconvex(nonconvex);
    }

    /// Score branching candidates for a solution vector.
    ///
    /// Returns (var_indices, frac_parts, obs_counts, scores) as four arrays.
    /// Only fractional integer variables are included.
    #[allow(clippy::type_complexity)]
    fn score_candidates<'py>(
        &self,
        py: Python<'py>,
        solution: PyReadonlyArray1<f64>,
    ) -> (
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let sol = solution.as_slice().unwrap();
        let candidates = self.inner.score_candidates(sol);
        let n = candidates.len();
        let mut var_indices = Vec::with_capacity(n);
        let mut frac_parts = Vec::with_capacity(n);
        let mut obs_counts = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        for (idx, frac, obs, score) in candidates {
            var_indices.push(idx as i64);
            frac_parts.push(frac);
            obs_counts.push(obs as i64);
            scores.push(score);
        }
        (
            PyArray1::from_vec(py, var_indices),
            PyArray1::from_vec(py, frac_parts),
            PyArray1::from_vec(py, obs_counts),
            PyArray1::from_vec(py, scores),
        )
    }

    /// Get the reliability threshold for pseudocost branching.
    fn reliability_threshold(&self) -> u32 {
        self.inner.get_reliability_threshold()
    }

    /// Set a branch hint for the next branching decision.
    ///
    /// When set, process_evaluated() will branch on the hinted variable
    /// index instead of using most-fractional branching. The hint is
    /// consumed after use (one-shot).
    fn set_branch_hints(
        &mut self,
        node_ids: PyReadonlyArray1<i64>,
        var_indices: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let ids = node_ids.as_slice().unwrap();
        let vars = var_indices.as_slice().unwrap();
        for (&nid, &vid) in ids.iter().zip(vars.iter()) {
            self.inner
                .set_branch_hint(NodeId(nid as usize), vid as usize);
        }
        Ok(())
    }
}
