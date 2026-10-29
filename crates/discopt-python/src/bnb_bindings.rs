//! PyO3 bindings for the Branch-and-Bound TreeManager.
//!
//! Wraps `discopt_core::bnb::TreeManager` with zero-copy numpy array interface
//! for Python orchestration of the B&B solve loop.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use discopt_core::bnb::{
    NodeId, NodeLineage, NodeResult, SelectionStrategy, TreeManager, VarBranchInfo,
};

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
            "best_estimate" => SelectionStrategy::BestEstimate,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown strategy: '{}'. Use 'best_first', 'depth_first', or 'best_estimate'.",
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
    ///     sentinel_is_exclusion: [N] optional boolean array. True where the
    ///         node carries the 1e30 sentinel as a JUSTIFIED exclusion of its
    ///         region (a user-callback veto, #1038/#748), False where it carries
    ///         it because the relaxation FAILED and nothing about the region was
    ///         proved. Omitted means "none are justified" — the conservative arm,
    ///         which treats a sentinel as unbounded rather than as grounds to
    ///         prune.
    #[pyo3(signature = (node_ids, lower_bounds, solutions, feasible, certified_infeasible=None, sentinel_is_exclusion=None))]
    fn import_results(
        &mut self,
        node_ids: PyReadonlyArray1<i64>,
        lower_bounds: PyReadonlyArray1<f64>,
        solutions: PyReadonlyArray2<f64>,
        feasible: PyReadonlyArray1<bool>,
        // #956 T3': per-node RIGOROUS emptiness certificate (Farkas-verified
        // infeasible relaxation / empty box). Optional so existing callers are
        // unchanged; omitted means "no certificate", the pre-fix behaviour.
        certified_infeasible: Option<PyReadonlyArray1<bool>>,
        // Which of the sentinelled nodes had their region JUSTIFIABLY excluded
        // (a callback veto, #1038/#748) rather than merely failing to bound.
        // Optional for the same reason; omitted means "none were", which is the
        // arm that treats a sentinel as unbounded rather than as grounds to
        // prune.
        sentinel_is_exclusion: Option<PyReadonlyArray1<bool>>,
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

        let cinf = certified_infeasible.as_ref().map(|a| a.as_array());
        if let Some(ci) = cinf.as_ref() {
            if ci.len() != n {
                return Err(PyValueError::new_err(
                    "certified_infeasible must have the same length as node_ids",
                ));
            }
        }
        let excl = sentinel_is_exclusion.as_ref().map(|a| a.as_array());
        if let Some(ex) = excl.as_ref() {
            if ex.len() != n {
                return Err(PyValueError::new_err(
                    "sentinel_is_exclusion must have the same length as node_ids",
                ));
            }
        }
        let results: Vec<NodeResult> = (0..n)
            .map(|i| {
                let nid = discopt_core::bnb::NodeId(ids[i] as usize);
                NodeResult {
                    node_id: nid,
                    lower_bound: lbs[i],
                    solution: sols.row(i).to_vec(),
                    is_feasible: feas[i],
                    certified_infeasible: cinf.as_ref().map(|c| c[i]).unwrap_or(false),
                    sentinel_is_exclusion: excl.as_ref().map(|e| e[i]).unwrap_or(false),
                }
            })
            .collect();

        self.inner.import_results(&results);
        Ok(())
    }

    /// Return exported nodes to the open frontier without importing a result.
    ///
    /// The lazy-constraint path (#1365) calls this for every node whose integer
    /// point a user separator vetoed *with a cut*: the veto says "this point is
    /// not acceptable", never "this box is empty", so the box must be re-solved
    /// against the cut-augmented relaxation rather than fathomed. Sentinelling
    /// it instead is the documented non-rigorous fathom of #748 -- it removes a
    /// region that was never proven free of acceptable points, which is how a
    /// subtour-elimination loop lost the corrected tour and returned
    /// `status="unknown"` with no solution at all.
    ///
    /// The caller MUST NOT also import a `NodeResult` for a requeued node:
    /// `import_results` asserts `Evaluated` status and would queue a pending
    /// result for a node that is open again.
    fn requeue_nodes(&mut self, node_ids: PyReadonlyArray1<i64>) -> PyResult<()> {
        let ids = node_ids.as_array();
        for &i in ids.iter() {
            if i < 0 {
                return Err(PyValueError::new_err(format!("negative node id {i}")));
            }
            self.inner.requeue_node(NodeId(i as usize));
        }
        Ok(())
    }

    /// Pop-time `local_lower_bound` for each of `node_ids` (task #89 / B2-FIX).
    ///
    /// Read-only. Called between `export_batch` and `import_results`, each
    /// value is the bound the node carried when it was popped — inherited from
    /// its parent's rigorous relaxation solve — which remains a valid lower
    /// bound for that node's whole subtree even if the node is subsequently
    /// fathomed non-rigorously (failure sentinel without an infeasibility
    /// proof). Unknown / negative ids map to `-inf` (conservative: a `-inf`
    /// floor makes the caller discard the tree bound, the pre-fix behavior).
    fn node_lower_bounds<'py>(
        &self,
        py: Python<'py>,
        node_ids: PyReadonlyArray1<i64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let ids = node_ids.as_array();
        let out: Vec<f64> = ids
            .iter()
            .map(|&i| {
                if i < 0 {
                    f64::NEG_INFINITY
                } else {
                    self.inner
                        .node_lower_bound(NodeId(i as usize))
                        .unwrap_or(f64::NEG_INFINITY)
                }
            })
            .collect();
        PyArray1::from_vec(py, out)
    }

    /// Tree depth for each of `node_ids` (root = 0), for the in-tree presolve
    /// depth-stride gate (issue #632 / PF1). Called between `export_batch` and
    /// `import_results`. Depth is immutable per node, so this is valid for any
    /// exported node. Unknown / negative ids map to `0` (conservative: the pass
    /// then fires at that node under any stride, never wrongly skipped).
    fn node_depths<'py>(
        &self,
        py: Python<'py>,
        node_ids: PyReadonlyArray1<i64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let ids = node_ids.as_array();
        let out: Vec<i64> = ids
            .iter()
            .map(|&i| {
                if i < 0 {
                    0
                } else {
                    self.inner.node_depth(NodeId(i as usize)).unwrap_or(0) as i64
                }
            })
            .collect();
        PyArray1::from_vec(py, out)
    }

    /// Lineage of `node_ids` for the debugger (#2043): a dict of arrays aligned
    /// with the ids -- `id`, `parent` (-1 for the root), `depth`, `branch_var`
    /// (-1 for the root), `branch_side` ("down"/"up", "" for the root),
    /// `branch_value` (NaN for the root) and `status`. Raises `ValueError` on an
    /// id this tree never allocated.
    fn node_lineage<'py>(
        &self,
        py: Python<'py>,
        node_ids: PyReadonlyArray1<i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut ids = Vec::new();
        for &i in node_ids.as_array().iter() {
            if i < 0 {
                return Err(PyValueError::new_err(format!(
                    "node_lineage: negative id {i}"
                )));
            }
            ids.push(NodeId(i as usize));
        }
        lineage_dict(py, &self.inner, &ids)
    }

    /// Ids (int64) of the nodes created by the latest `process_evaluated`, in
    /// creation order (#2043), including children pruned before selection.
    fn last_created<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.inner.last_created().map(|i| i as i64).collect())
    }

    /// Replace a node's structural variable bounds with reduction-tightened ones
    /// (cert:T2.4c), so its children inherit the contracted box.
    ///
    /// A sound contraction only — the caller must pass a box that is a SUBSET of
    /// the node's current box (per-node `reduce_node` guarantees this: it only
    /// intersects). Called on the spatial path AFTER the node LP + `reduce_node`,
    /// BEFORE `process_evaluated` branches the node. `lb`/`ub` are flat per-variable
    /// arrays of length `n_vars`; a length mismatch is a no-op inside Rust.
    fn set_node_bounds(
        &mut self,
        node_id: i64,
        lb: PyReadonlyArray1<f64>,
        ub: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let nid = NodeId(node_id as usize);
        let lb_v: Vec<f64> = lb.as_array().to_vec();
        let ub_v: Vec<f64> = ub.as_array().to_vec();
        self.inner.set_node_bounds(nid, lb_v, ub_v);
        Ok(())
    }

    /// Ids (int64) of the nodes fathomed as TIGHT boxes since the last call
    /// (#1510), drained. Each still floors the global dual bound at its own
    /// bound; see `raise_tight_fathom_floor`.
    fn take_tight_fathoms<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let ids: Vec<i64> = self
            .inner
            .take_tight_fathoms()
            .into_iter()
            .map(|n| n.0 as i64)
            .collect();
        PyArray1::from_vec(py, ids)
    }

    /// The box `(lb, ub)` a node carries in the tree (the one its branching and
    /// tight-box test read). Raises `ValueError` for an unknown id.
    fn node_box<'py>(
        &self,
        py: Python<'py>,
        node_id: i64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        match self.inner.node_box(NodeId(node_id.max(0) as usize)) {
            Some((lb, ub)) if node_id >= 0 => {
                Ok((PyArray1::from_vec(py, lb), PyArray1::from_vec(py, ub)))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node_box: unknown node id {node_id}"
            ))),
        }
    }

    /// Raise the floor a tight-fathomed node holds on the global dual bound
    /// (#1510) to `bound`, a valid lower bound of the internally-minimized
    /// objective over that node's box. Monotone; a non-finite bound is ignored.
    /// Raises `ValueError` if the node was not tight-fathomed.
    fn raise_tight_fathom_floor(&mut self, node_id: i64, bound: f64) -> PyResult<()> {
        if node_id < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "raise_tight_fathom_floor: negative node id {node_id}"
            )));
        }
        self.inner
            .raise_tight_fathom_floor(NodeId(node_id as usize), bound)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Install an externally-proved rigorous lower bound for the root box
    /// (#933 part (a)): the caller asserts `bound` is a valid lower bound of
    /// the internally-minimized objective over the box the tree was created
    /// with (e.g. the root LP/McCormick relaxation bound proved during setup).
    /// The seed floors every open node's `local_lower_bound` (children inherit
    /// it) and permanently floors the reported `global_lower_bound`, so the
    /// tree reports a finite anytime dual bound from the first proved root
    /// relaxation onward instead of `-inf` until the first processed batch.
    /// Non-finite values are ignored.
    fn seed_root_bound(&mut self, bound: f64) {
        self.inner.seed_root_bound(bound);
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
    /// Returns {total_nodes, open_nodes, incumbent_value, global_lower_bound,
    /// gap, bound_unresolved, unresolved_floor}.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("total_nodes", s.total_nodes)?;
        dict.set_item("open_nodes", s.open_nodes)?;
        dict.set_item("incumbent_value", s.incumbent_value)?;
        dict.set_item("global_lower_bound", s.global_lower_bound)?;
        dict.set_item("gap", s.gap)?;
        dict.set_item("bound_unresolved", s.bound_unresolved)?;
        dict.set_item("unresolved_floor", s.unresolved_floor)?;
        Ok(dict)
    }

    /// R3a measurement accessor (temporary, behavior-neutral): per-variable
    /// branch frequency as an int64 array of length n_vars. Element `i` is the
    /// number of branching events (integer or spatial) that split flat column
    /// `i`. Used by the responsiveness-fingerprint experiment; does not affect
    /// any solver decision.
    fn branch_var_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let counts: Vec<i64> = self
            .inner
            .branch_var_counts()
            .iter()
            .map(|&c| c as i64)
            .collect();
        PyArray1::from_vec(py, counts)
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

    /// #1869: split an unbounded dimension at a finite point instead of fathoming
    /// the node unresolved (`TreeManager::set_unbounded_branching`).
    fn set_unbounded_branching(&mut self, on: bool) {
        self.inner.set_unbounded_branching(on);
    }

    /// Register integer columns (in nonlinear terms) eligible for spatial
    /// domain-partition branching when no continuous variable remains to bisect
    /// (issue #194).
    fn set_spatial_integer_cols(&mut self, cols: PyReadonlyArray1<i64>) -> PyResult<()> {
        let idx: Vec<usize> = cols
            .as_slice()
            .unwrap()
            .iter()
            .map(|&c| c as usize)
            .collect();
        self.inner.set_spatial_integer_cols(&idx);
        Ok(())
    }

    /// Register functionally-dependent continuous columns (each pinned by a
    /// single equality `x_i = f(others)`) that spatial branching should
    /// deprioritize. Branching falls back to them only when no independent
    /// continuous variable still qualifies, preserving completeness.
    fn set_branch_deprioritized(&mut self, cols: PyReadonlyArray1<i64>) -> PyResult<()> {
        let idx: Vec<usize> = cols
            .as_slice()
            .unwrap()
            .iter()
            .map(|&c| c as usize)
            .collect();
        self.inner.set_branch_deprioritized(&idx);
        Ok(())
    }

    /// #1978: register the nonlinear terms the spatial branch scores its continuous
    /// column on (`DISCOPT_PY_TREE_SPATIAL_SCORE`). `bilinear` is `(k, 2)` original
    /// column pairs (`i == j` a square); `power_cols` / `power_exps` are the powers
    /// `x_i ^ p`. Empty arrays restore the legacy longest-edge rule.
    fn set_spatial_score_terms(
        &mut self,
        bilinear: PyReadonlyArray2<i64>,
        power_cols: PyReadonlyArray1<i64>,
        power_exps: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        use discopt_core::bnb::branching::ScoreTerm;
        let b = bilinear.as_array();
        if b.ncols() != 2 && b.nrows() > 0 {
            return Err(PyValueError::new_err("bilinear must have shape (k, 2)"));
        }
        let pc = power_cols.as_slice()?;
        let pe = power_exps.as_slice()?;
        if pc.len() != pe.len() {
            return Err(PyValueError::new_err(
                "power_cols and power_exps differ in length",
            ));
        }
        let mut terms = Vec::with_capacity(b.nrows() + pc.len());
        for r in 0..b.nrows() {
            let (i, j) = (b[[r, 0]], b[[r, 1]]);
            if i < 0 || j < 0 {
                return Err(PyValueError::new_err("negative column index"));
            }
            terms.push(ScoreTerm::Bilinear(i as usize, j as usize));
        }
        for (&c, &e) in pc.iter().zip(pe.iter()) {
            if c < 0 {
                return Err(PyValueError::new_err("negative column index"));
            }
            terms.push(ScoreTerm::Power(c as usize, e));
        }
        self.inner.set_spatial_score_terms(terms);
        Ok(())
    }

    /// #1978: the points (`(len(node_ids), n_vars)`) the score rule measures each
    /// node's term gaps at -- the node relaxation LP points.
    fn set_spatial_score_points(
        &mut self,
        node_ids: PyReadonlyArray1<i64>,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let ids = node_ids.as_slice()?;
        let x = points.as_array();
        if x.nrows() != ids.len() {
            return Err(PyValueError::new_err("one point per node id"));
        }
        for (r, &nid) in ids.iter().enumerate() {
            self.inner
                .set_spatial_score_point(NodeId(nid as usize), x.row(r).to_vec());
        }
        Ok(())
    }

    /// #1978 probe: spatial branches chosen by the score rule.
    fn spatial_score_picks(&self) -> u64 {
        self.inner.spatial_score_picks()
    }

    /// Register continuous SOS1-selector columns for spatial branch precedence
    /// (issue #196, `DISCOPT_SOS1_SELECTOR_BRANCH`). Empty (the default) keeps the
    /// legacy branch path byte-identical. Out-of-range indices are ignored.
    fn set_sos1_selector_cols(&mut self, cols: PyReadonlyArray1<i64>) -> PyResult<()> {
        let idx: Vec<usize> = cols
            .as_slice()
            .unwrap()
            .iter()
            .map(|&c| c as usize)
            .collect();
        self.inner.set_sos1_selector_cols(&idx);
        Ok(())
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

/// Lineage of `ids` in `tm` as a debugger dict (#2043); see `node_lineage`.
fn lineage_dict<'py>(
    py: Python<'py>,
    tm: &TreeManager,
    ids: &[NodeId],
) -> PyResult<Bound<'py, PyDict>> {
    let mut rows = Vec::with_capacity(ids.len());
    for &id in ids {
        rows.push(tm.node_lineage(id).ok_or_else(|| {
            PyValueError::new_err(format!("node_lineage: unknown node id {}", id.0))
        })?);
    }
    lineage_to_dict(py, &rows)
}

/// The debugger's lineage dict (#2043), shared by the Python-tree binding and the
/// Rust MILP hook so both routes report the same fields: arrays `id`, `parent`
/// (-1 for the root), `depth`, `branch_var` (-1 for the root), `branch_value`
/// (NaN for the root), and string lists `branch_side` ("down"/"up"/"") and
/// `status`.
pub(crate) fn lineage_to_dict<'py>(
    py: Python<'py>,
    rows: &[NodeLineage],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let ids: Vec<i64> = rows.iter().map(|r| r.id.0 as i64).collect();
    let parent: Vec<i64> = rows
        .iter()
        .map(|r| r.parent.map_or(-1, |p| p.0 as i64))
        .collect();
    let depth: Vec<i64> = rows.iter().map(|r| r.depth as i64).collect();
    let var: Vec<i64> = rows
        .iter()
        .map(|r| r.branch.map_or(-1, |b| b.var as i64))
        .collect();
    let value: Vec<f64> = rows
        .iter()
        .map(|r| r.branch.map_or(f64::NAN, |b| b.value))
        .collect();
    let side: Vec<&str> = rows
        .iter()
        .map(|r| match r.branch {
            None => "",
            Some(b) if b.up => "up",
            Some(_) => "down",
        })
        .collect();
    let status: Vec<&str> = rows.iter().map(|r| r.status.as_str()).collect();
    d.set_item("id", PyArray1::from_vec(py, ids))?;
    d.set_item("parent", PyArray1::from_vec(py, parent))?;
    d.set_item("depth", PyArray1::from_vec(py, depth))?;
    d.set_item("branch_var", PyArray1::from_vec(py, var))?;
    d.set_item("branch_value", PyArray1::from_vec(py, value))?;
    d.set_item("branch_side", side)?;
    d.set_item("status", status)?;
    Ok(d)
}
