//! PyO3 bindings for the Expression IR.
//!
//! Converts Python `jaxminlp_api.Model` objects to Rust `ModelRepr`,
//! and exposes evaluation functions for round-trip verification.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use discopt_core::expr::{
    BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, IndexSpec, MathFunc,
    ModelRepr, ObjectiveSense, UnOp, VarInfo, VarType,
};

// ─────────────────────────────────────────────────────────────
// PyModelRepr — opaque wrapper around ModelRepr
// ─────────────────────────────────────────────────────────────

#[pyclass]
pub struct PyModelRepr {
    inner: ModelRepr,
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
        self.inner.variables.iter().map(|v| v.name.clone()).collect()
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

    /// Evaluate the objective at a given point x.
    fn evaluate_objective(&self, x: numpy::PyReadonlyArray1<f64>) -> f64 {
        let x_arr = x.as_array();
        self.inner
            .evaluate_objective(x_arr.as_slice().unwrap())
    }

    /// Evaluate constraint body at a given point x.
    fn evaluate_constraint(&self, i: usize, x: numpy::PyReadonlyArray1<f64>) -> f64 {
        let x_arr = x.as_array();
        self.inner
            .evaluate_expr(self.inner.constraints[i].body, x_arr.as_slice().unwrap())
    }
}

// ─────────────────────────────────────────────────────────────
// model_to_repr: Convert Python Model -> Rust ModelRepr
// ─────────────────────────────────────────────────────────────

/// Convert a Python Model object to a Rust ModelRepr.
#[pyfunction]
pub fn model_to_repr(py: Python<'_>, model: &Bound<'_, PyAny>) -> PyResult<PyModelRepr> {
    let mut arena = ExprArena::new();

    // Build variable info from model._variables
    let py_vars = model.getattr("_variables")?;
    let py_vars_list: Vec<Bound<'_, PyAny>> = py_vars.extract()?;
    let mut variables = Vec::new();
    let mut offset = 0;
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

        // Extract bounds as flat arrays
        let lb_obj = py_var.getattr("lb")?;
        let lb = extract_flat_f64(&lb_obj)?;
        let ub_obj = py_var.getattr("ub")?;
        let ub = extract_flat_f64(&ub_obj)?;

        variables.push(VarInfo {
            name,
            var_type,
            offset,
            size,
            shape: shape_tuple,
            lb,
            ub,
        });
        offset += size;
    }
    let n_vars = offset;

    // Register all variables in the arena upfront.
    // Map from Python Variable object id -> ExprId.
    let mut var_expr_ids: std::collections::HashMap<isize, ExprId> =
        std::collections::HashMap::new();
    for (i, py_var) in py_vars_list.iter().enumerate() {
        let vi = &variables[i];
        let expr_id = arena.add(ExprNode::Variable {
            name: vi.name.clone(),
            index: i,
            size: vi.size,
            shape: vi.shape.clone(),
        });
        let py_id = py_var.as_ptr() as isize;
        var_expr_ids.insert(py_id, expr_id);
    }

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

        let expr_id = arena.add(ExprNode::Parameter {
            name,
            value,
            shape,
        });
        let py_id = py_param.as_ptr() as isize;
        param_expr_ids.insert(py_id, expr_id);
    }

    // Convert the objective expression
    let py_objective = model.getattr("_objective")?;
    let py_obj_expr = py_objective.getattr("expression")?;
    let objective = convert_expr(py, &py_obj_expr, &mut arena, &var_expr_ids, &param_expr_ids)?;

    let sense_obj = py_objective.getattr("sense")?;
    let sense_str: String = sense_obj.getattr("value")?.extract()?;
    let objective_sense = match sense_str.as_str() {
        "minimize" => ObjectiveSense::Minimize,
        "maximize" => ObjectiveSense::Maximize,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown ObjectiveSense: {sense_str}"
            )));
        }
    };

    // Convert constraints (only standard Constraint objects)
    let py_constraints = model.getattr("_constraints")?;
    let py_constraints_list: Vec<Bound<'_, PyAny>> = py_constraints.extract()?;
    let mut constraints = Vec::new();
    for py_con in &py_constraints_list {
        let class_name = get_class_name(py_con)?;
        if class_name != "Constraint" {
            // Skip _IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint
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

/// Convert a Python index (int, tuple of ints) to an IndexSpec.
fn convert_index_spec(obj: &Bound<'_, PyAny>) -> PyResult<IndexSpec> {
    if obj.is_instance_of::<PyTuple>() {
        let tuple: &Bound<'_, PyTuple> = obj.downcast()?;
        let indices: Vec<usize> = tuple
            .iter()
            .map(|item| item.extract::<usize>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(IndexSpec::Tuple(indices))
    } else {
        let idx: usize = obj.extract()?;
        Ok(IndexSpec::Scalar(idx))
    }
}
