//! PyO3 bindings for the Expression IR.
//!
//! Converts Python `jaxminlp_api.Model` objects to Rust `ModelRepr`,
//! and exposes evaluation functions for round-trip verification.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use discopt_core::expr::{
    BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, IndexElem, IndexSpec,
    MathFunc, ModelBuilder, ModelRepr, ObjectiveSense, UnOp, VarInfo, VarType,
};

// ─────────────────────────────────────────────────────────────
// PyModelRepr — opaque wrapper around ModelRepr
// ─────────────────────────────────────────────────────────────

#[pyclass]
pub struct PyModelRepr {
    inner: ModelRepr,
}

impl PyModelRepr {
    /// Create a PyModelRepr from a Rust ModelRepr (crate-internal).
    pub(crate) fn from_model_repr(model: ModelRepr) -> Self {
        Self { inner: model }
    }
}

#[pymethods]
impl PyModelRepr {
    /// Number of expression nodes in the arena.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.arena.len()
    }

    /// Total number of scalar variables.
    #[getter]
    fn n_vars(&self) -> usize {
        self.inner.n_vars
    }

    /// Number of constraints.
    #[getter]
    fn n_constraints(&self) -> usize {
        self.inner.constraints.len()
    }

    /// Number of variable blocks.
    #[getter]
    fn n_var_blocks(&self) -> usize {
        self.inner.variables.len()
    }

    /// Objective sense as string.
    #[getter]
    fn objective_sense(&self) -> &str {
        match self.inner.objective_sense {
            ObjectiveSense::Minimize => "minimize",
            ObjectiveSense::Maximize => "maximize",
        }
    }

    /// Variable names.
    fn var_names(&self) -> Vec<String> {
        self.inner
            .variables
            .iter()
            .map(|v| v.name.clone())
            .collect()
    }

    /// Variable types as strings.
    fn var_types(&self) -> Vec<String> {
        self.inner
            .variables
            .iter()
            .map(|v| match v.var_type {
                VarType::Continuous => "continuous".to_string(),
                VarType::Binary => "binary".to_string(),
                VarType::Integer => "integer".to_string(),
            })
            .collect()
    }

    /// Variable shapes.
    fn var_shapes(&self) -> Vec<Vec<usize>> {
        self.inner
            .variables
            .iter()
            .map(|v| v.shape.clone())
            .collect()
    }

    /// Variable lower bounds (flat).
    fn var_lb(&self, index: usize) -> Vec<f64> {
        self.inner.variables[index].lb.clone()
    }

    /// Variable upper bounds (flat).
    fn var_ub(&self, index: usize) -> Vec<f64> {
        self.inner.variables[index].ub.clone()
    }

    /// Tighten variable block `block_idx`'s element-wise bounds by
    /// intersection with the supplied `lb`/`ub` vectors. Element `k`
    /// of the stored block satisfies
    ///     `new_lb[k] = max(stored_lb[k], lb[k])`
    ///     `new_ub[k] = min(stored_ub[k], ub[k])`.
    /// Returns the number of endpoint updates that strictly tightened.
    /// Used by the A3 Rust↔Python presolve handshake.
    fn tighten_var_bounds(
        &mut self,
        block_idx: usize,
        lb: Vec<f64>,
        ub: Vec<f64>,
    ) -> PyResult<usize> {
        if block_idx >= self.inner.variables.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "block_idx {} out of range (n_blocks = {})",
                block_idx,
                self.inner.variables.len()
            )));
        }
        let v = &mut self.inner.variables[block_idx];
        if lb.len() != v.lb.len() || ub.len() != v.ub.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bound length mismatch: block has {} elements, got lb={} ub={}",
                v.lb.len(),
                lb.len(),
                ub.len()
            )));
        }
        let mut count: usize = 0;
        for i in 0..v.lb.len() {
            if lb[i] > v.lb[i] + 1e-12 {
                v.lb[i] = lb[i];
                count += 1;
            }
            if ub[i] < v.ub[i] - 1e-12 {
                v.ub[i] = ub[i];
                count += 1;
            }
        }
        Ok(count)
    }

    /// Is the objective linear?
    fn is_objective_linear(&self) -> bool {
        self.inner.arena.is_linear(self.inner.objective)
    }

    /// Is the objective quadratic?
    fn is_objective_quadratic(&self) -> bool {
        self.inner.arena.is_quadratic(self.inner.objective)
    }

    /// Is the objective bilinear?
    fn is_objective_bilinear(&self) -> bool {
        self.inner.arena.is_bilinear(self.inner.objective)
    }

    /// Is constraint i linear?
    fn is_constraint_linear(&self, i: usize) -> bool {
        self.inner.arena.is_linear(self.inner.constraints[i].body)
    }

    /// Is constraint i quadratic?
    fn is_constraint_quadratic(&self, i: usize) -> bool {
        self.inner
            .arena
            .is_quadratic(self.inner.constraints[i].body)
    }

    /// Constraint name (or None).
    fn constraint_name(&self, i: usize) -> Option<String> {
        self.inner.constraints[i].name.clone()
    }

    /// Constraint sense as string ("<=", "==", or ">=").
    fn constraint_sense(&self, i: usize) -> &str {
        match self.inner.constraints[i].sense {
            ConstraintSense::Le => "<=",
            ConstraintSense::Eq => "==",
            ConstraintSense::Ge => ">=",
        }
    }

    /// Constraint right-hand side.
    fn constraint_rhs(&self, i: usize) -> f64 {
        self.inner.constraints[i].rhs
    }

    /// Number of nodes in the expression arena (alias for arena access).
    fn arena_len(&self) -> usize {
        self.inner.arena.len()
    }

    /// Return a Python dict describing arena node at `idx`.
    fn get_node(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        if idx >= self.inner.arena.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "node index {idx} out of range (arena has {} nodes)",
                self.inner.arena.len()
            )));
        }
        let node = self.inner.arena.get(ExprId(idx));
        let dict = PyDict::new(py);
        match node {
            ExprNode::Constant(v) => {
                dict.set_item("type", "constant")?;
                dict.set_item("value", *v)?;
            }
            ExprNode::ConstantArray(data, shape) => {
                dict.set_item("type", "constant_array")?;
                dict.set_item("value", data.clone())?;
                dict.set_item("shape", shape.clone())?;
            }
            ExprNode::Variable {
                name,
                index,
                size,
                shape,
            } => {
                dict.set_item("type", "variable")?;
                dict.set_item("name", name.clone())?;
                dict.set_item("index", *index)?;
                dict.set_item("size", *size)?;
                dict.set_item("shape", shape.clone())?;
            }
            ExprNode::Parameter { name, value, shape } => {
                dict.set_item("type", "parameter")?;
                dict.set_item("name", name.clone())?;
                dict.set_item("value", value.clone())?;
                dict.set_item("shape", shape.clone())?;
            }
            ExprNode::BinaryOp { op, left, right } => {
                dict.set_item("type", "binary_op")?;
                dict.set_item(
                    "op",
                    match op {
                        BinOp::Add => "+",
                        BinOp::Sub => "-",
                        BinOp::Mul => "*",
                        BinOp::Div => "/",
                        BinOp::Pow => "**",
                    },
                )?;
                dict.set_item("left", left.0)?;
                dict.set_item("right", right.0)?;
            }
            ExprNode::UnaryOp { op, operand } => {
                dict.set_item("type", "unary_op")?;
                dict.set_item(
                    "op",
                    match op {
                        UnOp::Neg => "neg",
                        UnOp::Abs => "abs",
                    },
                )?;
                dict.set_item("arg", operand.0)?;
            }
            ExprNode::FunctionCall { func, args } => {
                dict.set_item("type", "function_call")?;
                dict.set_item(
                    "func",
                    match func {
                        MathFunc::Exp => "exp",
                        MathFunc::Log => "log",
                        MathFunc::Log2 => "log2",
                        MathFunc::Log10 => "log10",
                        MathFunc::Sqrt => "sqrt",
                        MathFunc::Sin => "sin",
                        MathFunc::Cos => "cos",
                        MathFunc::Tan => "tan",
                        MathFunc::Atan => "atan",
                        MathFunc::Sinh => "sinh",
                        MathFunc::Cosh => "cosh",
                        MathFunc::Asin => "asin",
                        MathFunc::Acos => "acos",
                        MathFunc::Tanh => "tanh",
                        MathFunc::Abs => "abs",
                        MathFunc::Sign => "sign",
                        MathFunc::Min => "min",
                        MathFunc::Max => "max",
                        MathFunc::Prod => "prod",
                        MathFunc::Norm2 => "norm2",
                    },
                )?;
                let arg_indices: Vec<usize> = args.iter().map(|a| a.0).collect();
                dict.set_item("args", arg_indices)?;
            }
            ExprNode::Index { base, index } => {
                dict.set_item("type", "index")?;
                dict.set_item("base", base.0)?;
                match index {
                    IndexSpec::Scalar(i) => dict.set_item("index_spec", *i)?,
                    IndexSpec::Tuple(indices) => dict.set_item("index_spec", indices.clone())?,
                    IndexSpec::Multi(elems) => {
                        // Encode each axis as either an int (scalar) or a
                        // string slice spec like ":" / "1:3" / "::2".
                        let parts = pyo3::types::PyList::empty(py);
                        for e in elems {
                            match e {
                                IndexElem::Scalar(i) => parts.append(*i)?,
                                IndexElem::Slice { start, stop, step } => {
                                    parts.append(format_slice(*start, *stop, *step))?
                                }
                            }
                        }
                        dict.set_item("index_spec", parts)?;
                    }
                }
            }
            ExprNode::MatMul { left, right } => {
                dict.set_item("type", "matmul")?;
                dict.set_item("left", left.0)?;
                dict.set_item("right", right.0)?;
            }
            ExprNode::Sum { operand, axis } => {
                dict.set_item("type", "sum")?;
                dict.set_item("operand", operand.0)?;
                dict.set_item("axis", *axis)?;
            }
            ExprNode::SumOver { terms } => {
                dict.set_item("type", "sum_over")?;
                let term_indices: Vec<usize> = terms.iter().map(|t| t.0).collect();
                dict.set_item("terms", term_indices)?;
            }
        }
        Ok(dict.into())
    }

    /// ExprId (index) of the objective expression root.
    fn objective_id(&self) -> usize {
        self.inner.objective.0
    }

    /// ExprId (index) of each constraint expression root.
    fn constraint_ids(&self) -> Vec<usize> {
        self.inner.constraints.iter().map(|c| c.body.0).collect()
    }

    /// (expr_id, sense, rhs) for constraint i.
    fn constraint_info(&self, i: usize) -> (usize, String, f64) {
        let c = &self.inner.constraints[i];
        let sense = match c.sense {
            ConstraintSense::Le => "<=".to_string(),
            ConstraintSense::Eq => "==".to_string(),
            ConstraintSense::Ge => ">=".to_string(),
        };
        (c.body.0, sense, c.rhs)
    }

    /// Evaluate the objective at a given point x.
    fn evaluate_objective(&self, x: numpy::PyReadonlyArray1<f64>) -> f64 {
        let x_arr = x.as_array();
        self.inner.evaluate_objective(x_arr.as_slice().unwrap())
    }

    /// Evaluate constraint body at a given point x.
    fn evaluate_constraint(&self, i: usize, x: numpy::PyReadonlyArray1<f64>) -> f64 {
        let x_arr = x.as_array();
        self.inner
            .evaluate_expr(self.inner.constraints[i].body, x_arr.as_slice().unwrap())
    }

    /// Run FBBT (Feasibility-Based Bound Tightening) on the model.
    ///
    /// Returns (lower_bounds, upper_bounds) as numpy arrays, one element per
    /// variable block (not per scalar variable).
    fn fbbt(&self, py: Python<'_>, max_iter: usize, tol: f64) -> PyResult<(PyObject, PyObject)> {
        use discopt_core::presolve::fbbt::fbbt;
        let bounds = fbbt(&self.inner, max_iter, tol);
        let lbs: Vec<f64> = bounds.iter().map(|b| b.lo).collect();
        let ubs: Vec<f64> = bounds.iter().map(|b| b.hi).collect();
        let lb_arr = numpy::PyArray1::from_vec(py, lbs);
        let ub_arr = numpy::PyArray1::from_vec(py, ubs);
        Ok((lb_arr.into_any().unbind(), ub_arr.into_any().unbind()))
    }

    /// Eliminate continuous scalar variables uniquely determined by a
    /// singleton equality constraint (M10 of #51).
    ///
    /// Returns a new `PyModelRepr` plus a stats dict with keys
    /// `variables_fixed`, `constraints_removed`, `candidates_examined`.
    fn eliminate_variables(&self, py: Python<'_>) -> PyResult<(PyModelRepr, PyObject)> {
        use discopt_core::presolve::eliminate::eliminate_variables;
        let (new_model, stats) = eliminate_variables(&self.inner);
        let dict = PyDict::new(py);
        dict.set_item("variables_fixed", stats.variables_fixed)?;
        dict.set_item("constraints_removed", stats.constraints_removed)?;
        dict.set_item("candidates_examined", stats.candidates_examined)?;
        Ok((PyModelRepr { inner: new_model }, dict.into()))
    }

    /// Reformulate polynomial monomials of degree > 2 into bilinear
    /// auxiliary products (M4 of #51), and derive McCormick-style aux
    /// variable bounds from forward-interval propagation (M5).
    ///
    /// Returns a new `PyModelRepr` plus a stats dict with keys
    /// `constraints_rewritten`, `constraints_skipped`,
    /// `aux_variables_introduced`, `aux_constraints_introduced`,
    /// `aux_bounds_derived`.
    fn reformulate_polynomial(&self, py: Python<'_>) -> PyResult<(PyModelRepr, PyObject)> {
        use discopt_core::presolve::polynomial::reformulate_polynomial;
        let (new_model, stats) = reformulate_polynomial(&self.inner);
        let dict = PyDict::new(py);
        dict.set_item("constraints_rewritten", stats.constraints_rewritten)?;
        dict.set_item("constraints_skipped", stats.constraints_skipped)?;
        dict.set_item("aux_variables_introduced", stats.aux_variables_introduced)?;
        dict.set_item(
            "aux_constraints_introduced",
            stats.aux_constraints_introduced,
        )?;
        dict.set_item("aux_bounds_derived", stats.aux_bounds_derived)?;
        Ok((PyModelRepr { inner: new_model }, dict.into()))
    }

    /// Run FBBT with an optional incumbent cutoff bound.
    ///
    /// When `incumbent_bound` is provided, an additional synthetic constraint
    /// is injected: `objective <= bound` (minimize) or `objective >= bound`
    /// (maximize), allowing tighter bounds without LP solves.
    ///
    /// Returns (lower_bounds, upper_bounds) as numpy arrays, one element per
    /// variable block.
    #[pyo3(signature = (max_iter, tol, incumbent_bound=None))]
    fn fbbt_with_cutoff(
        &self,
        py: Python<'_>,
        max_iter: usize,
        tol: f64,
        incumbent_bound: Option<f64>,
    ) -> PyResult<(PyObject, PyObject)> {
        use discopt_core::presolve::fbbt::fbbt_with_cutoff;
        let bounds = fbbt_with_cutoff(&self.inner, max_iter, tol, incumbent_bound);
        let lbs: Vec<f64> = bounds.iter().map(|b| b.lo).collect();
        let ubs: Vec<f64> = bounds.iter().map(|b| b.hi).collect();
        let lb_arr = numpy::PyArray1::from_vec(py, lbs);
        let ub_arr = numpy::PyArray1::from_vec(py, ubs);
        Ok((lb_arr.into_any().unbind(), ub_arr.into_any().unbind()))
    }

    /// Classify AMP nonlinear product terms using the Rust expression arena.
    fn classify_nonlinear_terms(&self, py: Python<'_>) -> PyResult<PyObject> {
        let terms = discopt_core::amp::classify_nonlinear_terms(&self.inner);
        let dict = PyDict::new(py);

        dict.set_item("bilinear", terms.bilinear)?;
        dict.set_item("trilinear", terms.trilinear)?;
        dict.set_item("multilinear", terms.multilinear)?;
        dict.set_item("monomial", terms.monomial)?;
        dict.set_item("general_nl_count", terms.general_nl_count)?;
        dict.set_item("partition_candidates", terms.partition_candidates)?;

        let incidence = PyDict::new(py);
        for (var_idx, term_ids) in terms.term_incidence {
            let ids: Vec<usize> = term_ids.into_iter().collect();
            incidence.set_item(var_idx, ids)?;
        }
        dict.set_item("term_incidence", incidence)?;

        Ok(dict.into())
    }

    /// Detect candidate variable permutation orbits (item D4 of issue #51).
    ///
    /// Returns a list of dicts, each `{"vars": [int, ...]}` listing
    /// variable-block indices that are exchangeable under the model's
    /// constraint and objective structure. Sound for fully-linear
    /// models; reported as candidates for nonlinear models.
    ///
    /// Returns an empty list if the objective or any constraint is
    /// not expressible as a polynomial (the kernel abstains rather
    /// than risk an unsound orbit emission).
    fn detect_symmetries(&self, py: Python<'_>) -> PyResult<PyObject> {
        use discopt_core::presolve::detect_symmetries;
        let (orbits, stats) = detect_symmetries(&self.inner);
        let result = PyDict::new(py);
        let orbit_list = pyo3::types::PyList::empty(py);
        for orb in &orbits {
            let d = PyDict::new(py);
            d.set_item("vars", orb.vars.clone())?;
            orbit_list.append(d)?;
        }
        result.set_item("orbits", orbit_list)?;
        result.set_item("variables_examined", stats.variables_examined)?;
        result.set_item("orbits_found", stats.orbits_found)?;
        result.set_item("total_orbit_members", stats.total_orbit_members)?;
        Ok(result.into_any().unbind())
    }

    /// Run persistent in-tree FBBT at a B&B node (item B3 of issue #51).
    ///
    /// `node_lb` / `node_ub` override the model's declared variable
    /// bounds with the node's branched-on bounds. The pass is gated by
    /// `depth_stride`: it runs only when `node_depth % depth_stride == 0`,
    /// or skips and echoes the input bounds back unchanged. Setting
    /// `depth_stride = 0` disables the pass entirely.
    ///
    /// Returns a dict with keys:
    /// - `lb`, `ub`: numpy arrays of post-tightening per-variable bounds.
    /// - `bounds_tightened`: int — number of half-bounds that tightened.
    /// - `infeasible`: bool — `True` if the kernel detected emptiness.
    /// - `ran`: bool — `True` if the schedule actually fired at this depth.
    #[pyo3(signature = (
        node_lb,
        node_ub,
        node_depth=0,
        depth_stride=4,
        max_iter=8,
        tol=1e-6,
        incumbent=None,
    ))]
    fn in_tree_presolve(
        &self,
        py: Python<'_>,
        node_lb: Vec<f64>,
        node_ub: Vec<f64>,
        node_depth: usize,
        depth_stride: u32,
        max_iter: usize,
        tol: f64,
        incumbent: Option<f64>,
    ) -> PyResult<PyObject> {
        use discopt_core::bnb::{run_in_tree_presolve, InTreePresolveOptions};
        if node_lb.len() != self.inner.variables.len()
            || node_ub.len() != self.inner.variables.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "node_lb/node_ub length must match number of variable blocks",
            ));
        }
        let opts = InTreePresolveOptions {
            depth_stride,
            max_iter,
            tol,
        };
        let delta = run_in_tree_presolve(
            &self.inner,
            &node_lb,
            &node_ub,
            node_depth,
            incumbent,
            &opts,
        );
        let out = PyDict::new(py);
        out.set_item(
            "lb",
            numpy::PyArray1::from_vec(py, delta.lb).into_any().unbind(),
        )?;
        out.set_item(
            "ub",
            numpy::PyArray1::from_vec(py, delta.ub).into_any().unbind(),
        )?;
        out.set_item("bounds_tightened", delta.bounds_tightened)?;
        out.set_item("infeasible", delta.infeasible)?;
        out.set_item("ran", delta.ran)?;
        Ok(out.into_any().unbind())
    }

    /// Run the presolve orchestrator (item A1 of issue #53) to a fixed
    /// point and return the tightened model plus a structured stats dict.
    ///
    /// `passes`, when given, is a list of pass identifiers controlling
    /// which kernels run and in what order. Recognised identifiers:
    /// `"eliminate"`, `"aggregate"`, `"redundancy"`,
    /// `"polynomial_reform"`, `"simplify"`, `"implied_bounds"`,
    /// `"fbbt"`, `"fbbt_fixed_point"`, `"probing"`, `"scaling"`,
    /// `"cliques"`, `"reduced_cost_fixing"`.
    ///
    /// `reduced_cost_info`, when given, is a dict with keys
    /// `lp_value: float`, `cutoff: float`, `reduced_costs: list[float]`
    /// (one per variable block). Required for the
    /// `"reduced_cost_fixing"` pass to do anything; otherwise that pass
    /// is a no-op.
    /// Default order matches the historical
    /// `_jax/presolve_pipeline.py:run_root_presolve` behaviour:
    /// `["eliminate", "simplify", "fbbt", "probing"]`. Polynomial
    /// reformulation is opt-in (it changes variable indexing).
    ///
    /// `max_iterations` caps the number of full sweeps over the pass
    /// list. `time_limit_ms` and `work_unit_budget` cap wall time and
    /// pass-defined work units; 0 disables a budget.
    ///
    /// Returns `(new_repr, stats)` where `stats` carries:
    /// - `terminated_by`: one of `"NoProgress"`, `"IterationCap"`,
    ///   `"TimeBudget"`, `"WorkBudget"`, `"Infeasible"`.
    /// - `iterations`: number of full sweeps run.
    /// - `bounds_lo`, `bounds_hi`: numpy arrays of final tightened
    ///   per-block variable bounds.
    /// - `deltas`: list of per-pass dicts (chronological).
    #[pyo3(signature = (
        passes=None,
        max_iterations=16,
        time_limit_ms=0,
        work_unit_budget=0,
        fbbt_max_iter=20,
        fbbt_tol=1e-8,
        reduced_cost_info=None,
    ))]
    fn presolve(
        &self,
        py: Python<'_>,
        passes: Option<Vec<String>>,
        max_iterations: u32,
        time_limit_ms: u64,
        work_unit_budget: u64,
        fbbt_max_iter: usize,
        fbbt_tol: f64,
        reduced_cost_info: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(PyModelRepr, PyObject)> {
        use discopt_core::presolve::{
            run_orchestrator, AggregatePass, CliquePass, CoefficientStrengtheningPass,
            EliminatePass, FactorableElimPass, FbbtFixedPointPass, FbbtPass, ImpliedBoundsPass,
            OrchestratorOptions, PolynomialReformPass, PresolvePass, ProbingPass,
            ReducedCostFixingPass, ReducedCostInfo, ReductionConstraintsPass, RedundancyPass,
            ScalingPass, SimplifyPass,
        };

        // Parse the optional reduced-cost-fixing info dict once, up
        // front, so that every "reduced_cost_fixing" pass instance
        // shares the same input.
        let rc_info: Option<ReducedCostInfo> = match reduced_cost_info {
            Some(d) => {
                let lp_value: f64 = d
                    .get_item("lp_value")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(
                            "reduced_cost_info missing 'lp_value'",
                        )
                    })?
                    .extract()?;
                let cutoff: f64 = d
                    .get_item("cutoff")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err("reduced_cost_info missing 'cutoff'")
                    })?
                    .extract()?;
                let reduced_costs: Vec<f64> = d
                    .get_item("reduced_costs")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(
                            "reduced_cost_info missing 'reduced_costs'",
                        )
                    })?
                    .extract()?;
                Some(ReducedCostInfo {
                    lp_value,
                    cutoff,
                    reduced_costs,
                })
            }
            None => None,
        };

        let pass_names: Vec<String> = passes.unwrap_or_else(|| {
            vec![
                "eliminate".to_string(),
                "factorable_elim".to_string(),
                "simplify".to_string(),
                "coefficient_strengthening".to_string(),
                "fbbt".to_string(),
                "probing".to_string(),
            ]
        });

        let mut pass_objs: Vec<Box<dyn PresolvePass>> = Vec::with_capacity(pass_names.len());
        for name in &pass_names {
            let p: Box<dyn PresolvePass> = match name.as_str() {
                "eliminate" => Box::new(EliminatePass::default()),
                "aggregate" => Box::new(AggregatePass),
                "redundancy" => Box::new(RedundancyPass),
                "implied_bounds" => Box::new(ImpliedBoundsPass),
                "polynomial_reform" => Box::new(PolynomialReformPass::default()),
                "simplify" => Box::new(SimplifyPass::default()),
                "fbbt" => Box::new(FbbtPass {
                    max_iter: fbbt_max_iter,
                    tol: fbbt_tol,
                    incumbent_bound: None,
                }),
                "fbbt_fixed_point" => Box::new(FbbtFixedPointPass {
                    tol: fbbt_tol,
                    max_visits: 0,
                }),
                "scaling" => Box::new(ScalingPass),
                "cliques" => Box::new(CliquePass),
                "reduced_cost_fixing" => Box::new(ReducedCostFixingPass {
                    info: rc_info.clone(),
                }),
                "reduction_constraints" => Box::new(ReductionConstraintsPass),
                "coefficient_strengthening" => Box::new(CoefficientStrengtheningPass),
                "factorable_elim" => Box::new(FactorableElimPass),
                "probing" => Box::new(ProbingPass::default()),
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown presolve pass: {}",
                        other
                    )));
                }
            };
            pass_objs.push(p);
        }

        let opts = OrchestratorOptions {
            max_iterations,
            time_limit_ms,
            work_unit_budget,
            pass_order: pass_objs,
        };
        let result = run_orchestrator(self.inner.clone(), opts);

        // Build stats dict.
        let stats = PyDict::new(py);
        stats.set_item("terminated_by", format!("{:?}", result.terminated_by))?;
        stats.set_item("iterations", result.iterations)?;
        let lbs: Vec<f64> = result.bounds.iter().map(|b| b.lo).collect();
        let ubs: Vec<f64> = result.bounds.iter().map(|b| b.hi).collect();
        let lb_arr = numpy::PyArray1::from_vec(py, lbs);
        let ub_arr = numpy::PyArray1::from_vec(py, ubs);
        stats.set_item("bounds_lo", lb_arr.into_any().unbind())?;
        stats.set_item("bounds_hi", ub_arr.into_any().unbind())?;

        let deltas_list = pyo3::types::PyList::empty(py);
        for d in &result.deltas {
            let dd = PyDict::new(py);
            dd.set_item("pass_name", d.pass_name)?;
            dd.set_item("pass_iter", d.pass_iter)?;
            dd.set_item("bounds_tightened", d.bounds_tightened)?;
            dd.set_item("aux_vars_introduced", d.aux_vars_introduced)?;
            dd.set_item("aux_constraints_introduced", d.aux_constraints_introduced)?;
            dd.set_item("constraints_removed", d.constraints_removed.clone())?;
            dd.set_item("constraints_rewritten", d.constraints_rewritten.clone())?;
            dd.set_item("vars_fixed", d.vars_fixed.clone())?;
            let aggs = pyo3::types::PyList::empty(py);
            for a in &d.vars_aggregated {
                let ad = PyDict::new(py);
                ad.set_item("target", a.target)?;
                ad.set_item("sources", a.sources.clone())?;
                ad.set_item("coeffs", a.coeffs.clone())?;
                ad.set_item("constant", a.constant)?;
                aggs.append(ad)?;
            }
            dd.set_item("vars_aggregated", aggs)?;
            if let Some(rs) = &d.row_scales {
                dd.set_item("row_scales", rs.clone())?;
            }
            if let Some(cs) = &d.col_scales {
                dd.set_item("col_scales", cs.clone())?;
            }
            if !d.structure.cliques.is_empty() {
                let edges: Vec<(usize, usize)> = d.structure.cliques.clone();
                dd.set_item("cliques", edges)?;
            }
            if !d.structure.convex_constraints.is_empty() {
                dd.set_item("convex_constraints", d.structure.convex_constraints.clone())?;
            }
            dd.set_item("work_units", d.work_units)?;
            dd.set_item("wall_time_ms", d.wall_time_ms)?;
            deltas_list.append(dd)?;
        }
        stats.set_item("deltas", deltas_list)?;

        Ok((
            PyModelRepr {
                inner: result.model,
            },
            stats.into(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────
// model_to_repr: Convert Python Model -> Rust ModelRepr
// ─────────────────────────────────────────────────────────────

/// Convert a Python Model object to a Rust ModelRepr.
///
/// When `builder` is provided, starts from the builder's pre-populated arena
/// (fast-API constraints) and merges in any expression-based constraints.
#[pyfunction]
#[pyo3(signature = (model, builder=None))]
pub fn model_to_repr(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    builder: Option<&mut PyModelBuilder>,
) -> PyResult<PyModelRepr> {
    // Determine whether we're in hybrid mode (builder provided) or pure-expression mode.
    // Clone (not take) from builder so it can be reused across multiple calls.
    let (
        mut arena,
        mut variables,
        mut constraints,
        builder_objective,
        builder_sense,
        n_vars_init,
        builder_var_expr_ids,
    ) = if let Some(b) = builder {
        let a = b.inner.arena.clone();
        let v = b.inner.variables.clone();
        let c = b.inner.constraints.clone();
        let obj = b.inner.objective;
        let sense = b.inner.objective_sense;
        let nv = b.inner.n_vars;
        let ve = b.inner.var_expr_ids.clone();
        (a, v, c, obj, Some(sense), nv, ve)
    } else {
        (
            ExprArena::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            0usize,
            Vec::new(),
        )
    };

    // Build variable info from model._variables for the expression path.
    let py_vars = model.getattr("_variables")?;
    let py_vars_list: Vec<Bound<'_, PyAny>> = py_vars.extract()?;

    // Map from Python Variable object id -> ExprId.
    let mut var_expr_ids: std::collections::HashMap<isize, ExprId> =
        std::collections::HashMap::new();

    if !builder_var_expr_ids.is_empty() {
        // Builder mode: variables already registered in arena. Map Python objects
        // to the builder's ExprIds using _builder_idx.
        for py_var in &py_vars_list {
            let py_id = py_var.as_ptr() as isize;
            // Try to get _builder_idx — if the variable was registered via the builder
            if let Ok(idx) = py_var.getattr("_builder_idx") {
                if let Ok(bidx) = idx.extract::<usize>() {
                    if bidx < builder_var_expr_ids.len() {
                        var_expr_ids.insert(py_id, builder_var_expr_ids[bidx]);
                    }
                }
            }
        }
    } else {
        // Pure-expression mode: build VarInfo and arena nodes from scratch.
        let mut offset = 0usize;
        for py_var in &py_vars_list {
            let name: String = py_var.getattr("name")?.extract()?;
            let shape_tuple: Vec<usize> = py_var.getattr("shape")?.extract()?;
            let size: usize = py_var.getattr("size")?.extract()?;

            let var_type_obj = py_var.getattr("var_type")?;
            let var_type_str: String = var_type_obj.getattr("value")?.extract()?;
            let var_type = match var_type_str.as_str() {
                "continuous" => VarType::Continuous,
                "binary" => VarType::Binary,
                "integer" => VarType::Integer,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown VarType: {var_type_str}"
                    )));
                }
            };

            let lb_obj = py_var.getattr("lb")?;
            let lb = extract_flat_f64(&lb_obj)?;
            let ub_obj = py_var.getattr("ub")?;
            let ub = extract_flat_f64(&ub_obj)?;

            let vi_idx = variables.len();
            variables.push(VarInfo {
                name: name.clone(),
                var_type,
                offset,
                size,
                shape: shape_tuple.clone(),
                lb,
                ub,
            });

            let expr_id = arena.add(ExprNode::Variable {
                name,
                index: vi_idx,
                size,
                shape: shape_tuple,
            });
            let py_id = py_var.as_ptr() as isize;
            var_expr_ids.insert(py_id, expr_id);

            offset += size;
        }
    }

    let n_vars = if n_vars_init > 0 {
        n_vars_init
    } else {
        variables.iter().map(|v| v.size).sum()
    };

    // Register parameters.
    let py_params = model.getattr("_parameters")?;
    let py_params_list: Vec<Bound<'_, PyAny>> = py_params.extract()?;
    let mut param_expr_ids: std::collections::HashMap<isize, ExprId> =
        std::collections::HashMap::new();
    for py_param in &py_params_list {
        let name: String = py_param.getattr("name")?.extract()?;
        let value_obj = py_param.getattr("value")?;
        let value = extract_flat_f64(&value_obj)?;
        let shape: Vec<usize> = py_param.getattr("shape")?.extract()?;

        let expr_id = arena.add(ExprNode::Parameter { name, value, shape });
        let py_id = py_param.as_ptr() as isize;
        param_expr_ids.insert(py_id, expr_id);
    }

    // Determine objective: builder's objective takes priority, fall back to model._objective.
    let py_objective = model.getattr("_objective")?;
    let (objective, objective_sense) = if let Some(obj_id) = builder_objective {
        // Builder has an objective. Check if model also has one (expression-based override).
        if py_objective.is_none() {
            // No Python objective — use builder's
            (obj_id, builder_sense.unwrap_or(ObjectiveSense::Minimize))
        } else {
            // Python objective exists — check if it's a placeholder (from fast API)
            let is_placeholder: bool = py_objective
                .getattr("_is_placeholder")
                .and_then(|v| v.extract())
                .unwrap_or(false);
            if is_placeholder {
                (obj_id, builder_sense.unwrap_or(ObjectiveSense::Minimize))
            } else {
                // Expression-based objective overrides builder
                let py_obj_expr = py_objective.getattr("expression")?;
                let obj =
                    convert_expr(py, &py_obj_expr, &mut arena, &var_expr_ids, &param_expr_ids)?;
                let sense_obj = py_objective.getattr("sense")?;
                let sense_str: String = sense_obj.getattr("value")?.extract()?;
                let os = match sense_str.as_str() {
                    "minimize" => ObjectiveSense::Minimize,
                    "maximize" => ObjectiveSense::Maximize,
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown ObjectiveSense: {sense_str}"
                        )));
                    }
                };
                (obj, os)
            }
        }
    } else {
        // No builder objective — must come from Python model.
        let py_obj_expr = py_objective.getattr("expression")?;
        let obj = convert_expr(py, &py_obj_expr, &mut arena, &var_expr_ids, &param_expr_ids)?;
        let sense_obj = py_objective.getattr("sense")?;
        let sense_str: String = sense_obj.getattr("value")?.extract()?;
        let os = match sense_str.as_str() {
            "minimize" => ObjectiveSense::Minimize,
            "maximize" => ObjectiveSense::Maximize,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown ObjectiveSense: {sense_str}"
                )));
            }
        };
        (obj, os)
    };

    // Convert expression-based constraints (only standard Constraint objects)
    let py_constraints = model.getattr("_constraints")?;
    let py_constraints_list: Vec<Bound<'_, PyAny>> = py_constraints.extract()?;
    for py_con in &py_constraints_list {
        let class_name = get_class_name(py_con)?;
        if class_name != "Constraint" {
            continue;
        }
        let body_expr = py_con.getattr("body")?;
        let body = convert_expr(py, &body_expr, &mut arena, &var_expr_ids, &param_expr_ids)?;
        let sense_str: String = py_con.getattr("sense")?.extract()?;
        let sense = match sense_str.as_str() {
            "<=" => ConstraintSense::Le,
            "==" => ConstraintSense::Eq,
            ">=" => ConstraintSense::Ge,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown constraint sense: {sense_str}"
                )));
            }
        };
        let rhs: f64 = py_con.getattr("rhs")?.extract()?;
        let name: Option<String> = py_con.getattr("name")?.extract()?;
        constraints.push(ConstraintRepr {
            body,
            sense,
            rhs,
            name,
        });
    }

    let inner = ModelRepr {
        arena,
        objective,
        objective_sense,
        constraints,
        variables,
        n_vars,
    };

    Ok(PyModelRepr { inner })
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/// Extract a numpy array (any dimension) into a flat Vec<f64>.
fn extract_flat_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Try extracting as a numpy array first, then fall back to list/scalar.
    let arr: numpy::PyReadonlyArrayDyn<f64> = obj.extract()?;
    let view = arr.as_array();
    Ok(view.iter().copied().collect())
}

/// Get the Python class name of an object.
fn get_class_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.getattr("__class__")?.getattr("__name__")?.extract()
}

/// Recursively convert a Python Expression to ExprNode(s) in the arena.
fn convert_expr(
    _py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    arena: &mut ExprArena,
    var_ids: &std::collections::HashMap<isize, ExprId>,
    param_ids: &std::collections::HashMap<isize, ExprId>,
) -> PyResult<ExprId> {
    let class_name = get_class_name(obj)?;

    match class_name.as_str() {
        "Variable" => {
            let py_id = obj.as_ptr() as isize;
            if let Some(&expr_id) = var_ids.get(&py_id) {
                Ok(expr_id)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Variable not found in model",
                ))
            }
        }
        "Parameter" => {
            let py_id = obj.as_ptr() as isize;
            if let Some(&expr_id) = param_ids.get(&py_id) {
                Ok(expr_id)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Parameter not found in model",
                ))
            }
        }
        "Constant" => {
            let value_obj = obj.getattr("value")?;
            let data = extract_flat_f64(&value_obj)?;
            let arr: numpy::PyReadonlyArrayDyn<f64> = value_obj.extract()?;
            let shape: Vec<usize> = arr.as_array().shape().to_vec();
            if data.len() == 1 {
                Ok(arena.add(ExprNode::Constant(data[0])))
            } else {
                Ok(arena.add(ExprNode::ConstantArray(data, shape)))
            }
        }
        "BinaryOp" => {
            let op_str: String = obj.getattr("op")?.extract()?;
            let op = match op_str.as_str() {
                "+" => BinOp::Add,
                "-" => BinOp::Sub,
                "*" => BinOp::Mul,
                "/" => BinOp::Div,
                "**" => BinOp::Pow,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown BinOp: {op_str}"
                    )));
                }
            };
            let left = obj.getattr("left")?;
            let right = obj.getattr("right")?;
            let left_id = convert_expr(_py, &left, arena, var_ids, param_ids)?;
            let right_id = convert_expr(_py, &right, arena, var_ids, param_ids)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op,
                left: left_id,
                right: right_id,
            }))
        }
        "UnaryOp" => {
            let op_str: String = obj.getattr("op")?.extract()?;
            let op = match op_str.as_str() {
                "neg" => UnOp::Neg,
                "abs" => UnOp::Abs,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown UnOp: {op_str}"
                    )));
                }
            };
            let operand = obj.getattr("operand")?;
            let operand_id = convert_expr(_py, &operand, arena, var_ids, param_ids)?;
            Ok(arena.add(ExprNode::UnaryOp {
                op,
                operand: operand_id,
            }))
        }
        "FunctionCall" => {
            let func_str: String = obj.getattr("func_name")?.extract()?;
            let func = match func_str.as_str() {
                "exp" => MathFunc::Exp,
                "log" => MathFunc::Log,
                "log2" => MathFunc::Log2,
                "log10" => MathFunc::Log10,
                "sqrt" => MathFunc::Sqrt,
                "sin" => MathFunc::Sin,
                "cos" => MathFunc::Cos,
                "tan" => MathFunc::Tan,
                "atan" => MathFunc::Atan,
                "sinh" => MathFunc::Sinh,
                "cosh" => MathFunc::Cosh,
                "asin" => MathFunc::Asin,
                "acos" => MathFunc::Acos,
                "tanh" => MathFunc::Tanh,
                "abs" => MathFunc::Abs,
                "sign" => MathFunc::Sign,
                "min" => MathFunc::Min,
                "max" => MathFunc::Max,
                "prod" => MathFunc::Prod,
                "norm2" => MathFunc::Norm2,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown MathFunc: {func_str}"
                    )));
                }
            };
            let py_args = obj.getattr("args")?;
            let py_args_tuple: Vec<Bound<'_, PyAny>> = py_args.extract()?;
            let mut args = Vec::new();
            for a in &py_args_tuple {
                args.push(convert_expr(_py, a, arena, var_ids, param_ids)?);
            }
            Ok(arena.add(ExprNode::FunctionCall { func, args }))
        }
        "IndexExpression" => {
            let base = obj.getattr("base")?;
            let base_id = convert_expr(_py, &base, arena, var_ids, param_ids)?;
            let py_index = obj.getattr("index")?;
            let index_spec = convert_index_spec(&py_index)?;
            Ok(arena.add(ExprNode::Index {
                base: base_id,
                index: index_spec,
            }))
        }
        "MatMulExpression" => {
            let left = obj.getattr("left")?;
            let right = obj.getattr("right")?;
            let left_id = convert_expr(_py, &left, arena, var_ids, param_ids)?;
            let right_id = convert_expr(_py, &right, arena, var_ids, param_ids)?;
            Ok(arena.add(ExprNode::MatMul {
                left: left_id,
                right: right_id,
            }))
        }
        "SumExpression" => {
            let operand = obj.getattr("operand")?;
            let operand_id = convert_expr(_py, &operand, arena, var_ids, param_ids)?;
            let axis: Option<usize> = obj.getattr("axis")?.extract()?;
            Ok(arena.add(ExprNode::Sum {
                operand: operand_id,
                axis,
            }))
        }
        "SumOverExpression" => {
            let py_terms = obj.getattr("terms")?;
            let py_terms_list: Vec<Bound<'_, PyAny>> = py_terms.extract()?;
            let mut terms = Vec::new();
            for t in &py_terms_list {
                terms.push(convert_expr(_py, t, arena, var_ids, param_ids)?);
            }
            Ok(arena.add(ExprNode::SumOver { terms }))
        }
        other => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Unknown expression type: {other}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────
// PyModelBuilder — fast construction without Python expression objects
// ─────────────────────────────────────────────────────────────

/// Fast model builder that constructs the Rust ExprArena directly,
/// bypassing Python expression objects for linear/quadratic models.
#[pyclass]
pub struct PyModelBuilder {
    inner: ModelBuilder,
}

#[pymethods]
impl PyModelBuilder {
    /// Create a new empty model builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: ModelBuilder::new(),
        }
    }

    /// Register a variable block. Returns the block index.
    fn add_variable(
        &mut self,
        name: String,
        var_type: &str,
        shape: Vec<usize>,
        lb: numpy::PyReadonlyArray1<f64>,
        ub: numpy::PyReadonlyArray1<f64>,
    ) -> PyResult<usize> {
        let vt = match var_type {
            "continuous" => VarType::Continuous,
            "binary" => VarType::Binary,
            "integer" => VarType::Integer,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown VarType: {var_type}"
                )));
            }
        };
        let lb_vec: Vec<f64> = lb.as_array().iter().copied().collect();
        let ub_vec: Vec<f64> = ub.as_array().iter().copied().collect();
        Ok(self.inner.add_variable(name, vt, shape, lb_vec, ub_vec))
    }

    /// Add linear constraints from CSR sparse data.
    ///
    /// Parameters:
    ///   indptr — CSR row pointers (int64)
    ///   indices — CSR column indices (int64)
    ///   data — CSR nonzero values (float64)
    ///   var_idx — Variable block index
    ///   sense — "<=" / "==" / ">="
    ///   rhs — Right-hand side vector (float64)
    ///   name_prefix — Optional name prefix
    #[pyo3(signature = (indptr, indices, data, var_idx, sense, rhs, name_prefix=None))]
    fn add_linear_constraints(
        &mut self,
        indptr: numpy::PyReadonlyArray1<i64>,
        indices: numpy::PyReadonlyArray1<i64>,
        data: numpy::PyReadonlyArray1<f64>,
        var_idx: usize,
        sense: &str,
        rhs: numpy::PyReadonlyArray1<f64>,
        name_prefix: Option<String>,
    ) -> PyResult<()> {
        let cs = parse_constraint_sense(sense)?;

        let indptr_usize: Vec<usize> = indptr.as_array().iter().map(|&v| v as usize).collect();
        let indices_usize: Vec<usize> = indices.as_array().iter().map(|&v| v as usize).collect();
        let data_slice: Vec<f64> = data.as_array().iter().copied().collect();
        let rhs_slice: Vec<f64> = rhs.as_array().iter().copied().collect();

        self.inner.add_linear_constraints_csr(
            &indptr_usize,
            &indices_usize,
            &data_slice,
            var_idx,
            cs,
            &rhs_slice,
            name_prefix.as_deref(),
        );
        Ok(())
    }

    /// Set a linear objective: c'x + constant.
    #[pyo3(signature = (c, var_idx, constant=0.0, sense="minimize"))]
    fn set_linear_objective(
        &mut self,
        c: numpy::PyReadonlyArray1<f64>,
        var_idx: usize,
        constant: f64,
        sense: &str,
    ) -> PyResult<()> {
        let os = parse_objective_sense(sense)?;
        let c_vec: Vec<f64> = c.as_array().iter().copied().collect();
        self.inner
            .set_linear_objective(&c_vec, var_idx, constant, os);
        Ok(())
    }

    /// Set a quadratic objective: 0.5 x'Qx + c'x + constant.
    ///
    /// Q is provided in CSR format.
    #[pyo3(signature = (q_indptr, q_indices, q_data, c, var_idx, constant=0.0, sense="minimize"))]
    fn set_quadratic_objective(
        &mut self,
        q_indptr: numpy::PyReadonlyArray1<i64>,
        q_indices: numpy::PyReadonlyArray1<i64>,
        q_data: numpy::PyReadonlyArray1<f64>,
        c: numpy::PyReadonlyArray1<f64>,
        var_idx: usize,
        constant: f64,
        sense: &str,
    ) -> PyResult<()> {
        let os = parse_objective_sense(sense)?;
        let qi: Vec<usize> = q_indptr.as_array().iter().map(|&v| v as usize).collect();
        let qj: Vec<usize> = q_indices.as_array().iter().map(|&v| v as usize).collect();
        let qd: Vec<f64> = q_data.as_array().iter().copied().collect();
        let cv: Vec<f64> = c.as_array().iter().copied().collect();
        self.inner
            .set_quadratic_objective(&qi, &qj, &qd, &cv, var_idx, constant, os);
        Ok(())
    }

    /// Number of constraints built so far.
    #[getter]
    fn n_constraints(&self) -> usize {
        self.inner.constraints.len()
    }

    /// Number of nodes in the arena.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.arena.len()
    }
}

/// Parse a constraint sense string.
fn parse_constraint_sense(s: &str) -> PyResult<ConstraintSense> {
    match s {
        "<=" => Ok(ConstraintSense::Le),
        "==" => Ok(ConstraintSense::Eq),
        ">=" => Ok(ConstraintSense::Ge),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown constraint sense: '{s}'. Expected '<=', '==', or '>='."
        ))),
    }
}

/// Parse an objective sense string.
fn parse_objective_sense(s: &str) -> PyResult<ObjectiveSense> {
    match s {
        "minimize" => Ok(ObjectiveSense::Minimize),
        "maximize" => Ok(ObjectiveSense::Maximize),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown objective sense: '{s}'. Expected 'minimize' or 'maximize'."
        ))),
    }
}

/// Convert a Python index (int, slice, or tuple of ints/slices) to an
/// IndexSpec.
fn convert_index_spec(obj: &Bound<'_, PyAny>) -> PyResult<IndexSpec> {
    if obj.is_instance_of::<PyTuple>() {
        let tuple: &Bound<'_, PyTuple> = obj.downcast()?;
        // If every element is a plain int, keep the simple Tuple form.
        // Otherwise (any slice present), build a Multi spec.
        let mut all_scalar = true;
        for item in tuple.iter() {
            if item.is_instance_of::<PySlice>() {
                all_scalar = false;
                break;
            }
        }
        if all_scalar {
            let indices: Vec<usize> = tuple
                .iter()
                .map(|item| item.extract::<usize>())
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(IndexSpec::Tuple(indices));
        }
        let mut elems: Vec<IndexElem> = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            if item.is_instance_of::<PySlice>() {
                elems.push(slice_to_index_elem(&item)?);
            } else {
                let i: usize = item.extract()?;
                elems.push(IndexElem::Scalar(i));
            }
        }
        Ok(IndexSpec::Multi(elems))
    } else if obj.is_instance_of::<PySlice>() {
        Ok(IndexSpec::Multi(vec![slice_to_index_elem(obj)?]))
    } else {
        let idx: usize = obj.extract()?;
        Ok(IndexSpec::Scalar(idx))
    }
}

/// Convert a Python slice to an IndexElem. Supports the full Python slice
/// protocol: `None`/missing fields become defaults, negative indices are
/// allowed, and `step` may be negative. A step of zero is rejected with the
/// same error Python raises (`ValueError`).
fn slice_to_index_elem(obj: &Bound<'_, PyAny>) -> PyResult<IndexElem> {
    let start = optional_isize(&obj.getattr("start")?)?;
    let stop = optional_isize(&obj.getattr("stop")?)?;
    let step = optional_isize(&obj.getattr("step")?)?;
    if step == Some(0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "slice step cannot be zero",
        ));
    }
    Ok(IndexElem::Slice { start, stop, step })
}

fn optional_isize(obj: &Bound<'_, PyAny>) -> PyResult<Option<isize>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.extract::<isize>()?))
    }
}

/// Format a slice as a compact `start:stop:step` string with `None` rendered
/// as the empty string (matching Python's repr of `slice(None, None, None)`
/// when stringified to `:`).
fn format_slice(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> String {
    let s = |v: Option<isize>| match v {
        Some(i) => i.to_string(),
        None => String::new(),
    };
    if step.is_none() {
        format!("{}:{}", s(start), s(stop))
    } else {
        format!("{}:{}:{}", s(start), s(stop), s(step))
    }
}
