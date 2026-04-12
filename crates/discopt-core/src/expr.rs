//! Expression IR — arena-allocated DAG for MINLP expressions.
//!
//! Mirrors the Python expression hierarchy in `jaxminlp_api/core.py`.
//! Each expression node is stored in an [`ExprArena`] and referenced by
//! an [`ExprId`] (lightweight index). The arena provides O(1) lookup
//! and guaranteed memory locality for tree-walking passes.

use std::fmt;

// ─────────────────────────────────────────────────────────────
// Expression identifiers and node types
// ─────────────────────────────────────────────────────────────

/// Index into the [`ExprArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub usize);

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// Binary arithmetic operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Addition (`left + right`).
    Add,
    /// Subtraction (`left - right`).
    Sub,
    /// Multiplication (`left * right`).
    Mul,
    /// Division (`left / right`).
    Div,
    /// Exponentiation (`left ^ right`).
    Pow,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// Arithmetic negation (`-x`).
    Neg,
    /// Absolute value (`|x|`).
    Abs,
}

/// Named mathematical functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFunc {
    /// Exponential function (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Log,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    /// Square root.
    Sqrt,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Arctangent.
    Atan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Inverse sine (arcsine).
    Asin,
    /// Inverse cosine (arccosine).
    Acos,
    /// Hyperbolic tangent.
    Tanh,
    /// Absolute value.
    Abs,
    /// Sign function (-1, 0, or 1).
    Sign,
    /// Minimum of arguments.
    Min,
    /// Maximum of arguments.
    Max,
    /// Product of arguments.
    Prod,
    /// L2 norm.
    Norm2,
}

/// Indexing specification for array access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexSpec {
    /// Single scalar index: `x[i]`
    Scalar(usize),
    /// Tuple index: `x[i, j]`
    Tuple(Vec<usize>),
}

/// A single node in the expression DAG.
#[derive(Debug, Clone)]
pub enum ExprNode {
    /// Scalar constant.
    Constant(f64),
    /// Dense constant array (flat data + shape).
    ConstantArray(Vec<f64>, Vec<usize>),
    /// Decision variable.
    Variable {
        /// Variable name.
        name: String,
        /// Index in the variables list.
        index: usize,
        /// Total number of scalar elements.
        size: usize,
        /// Shape of the variable (empty for scalars).
        shape: Vec<usize>,
    },
    /// Parameter (fixed per solve, differentiable).
    Parameter {
        /// Parameter name.
        name: String,
        /// Flat parameter values.
        value: Vec<f64>,
        /// Shape of the parameter (empty for scalars).
        shape: Vec<usize>,
    },
    /// Binary arithmetic: left op right.
    BinaryOp {
        /// The binary operator.
        op: BinOp,
        /// Left operand.
        left: ExprId,
        /// Right operand.
        right: ExprId,
    },
    /// Unary operation.
    UnaryOp {
        /// The unary operator.
        op: UnOp,
        /// The operand expression.
        operand: ExprId,
    },
    /// Named function call (exp, log, sin, ...).
    FunctionCall {
        /// The mathematical function.
        func: MathFunc,
        /// Function arguments.
        args: Vec<ExprId>,
    },
    /// Indexing into an array expression.
    Index {
        /// The base array expression.
        base: ExprId,
        /// The index specification.
        index: IndexSpec,
    },
    /// Matrix multiply: left @ right.
    MatMul {
        /// Left matrix operand.
        left: ExprId,
        /// Right matrix operand.
        right: ExprId,
    },
    /// Sum over an expression (optionally along an axis).
    Sum {
        /// The expression to sum.
        operand: ExprId,
        /// Optional axis to sum along (`None` for full reduction).
        axis: Option<usize>,
    },
    /// Sum of a list of terms.
    SumOver {
        /// The terms to sum.
        terms: Vec<ExprId>,
    },
}

// ─────────────────────────────────────────────────────────────
// Arena
// ─────────────────────────────────────────────────────────────

/// Arena allocator for expression nodes.
///
/// All nodes live here; everything else holds [`ExprId`] handles.
#[derive(Debug, Clone)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Create an arena with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
        }
    }

    /// Insert a node and return its id.
    pub fn add(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Retrieve a node by id.
    ///
    /// # Panics
    /// Panics if the id is out of bounds.
    pub fn get(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0]
    }

    /// Number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────
// Model representation
// ─────────────────────────────────────────────────────────────

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSense {
    /// Minimize the objective function.
    Minimize,
    /// Maximize the objective function.
    Maximize,
}

/// Constraint comparison sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintSense {
    /// Less-than-or-equal (`<=`).
    Le,
    /// Equality (`==`).
    Eq,
    /// Greater-than-or-equal (`>=`).
    Ge,
}

/// Variable domain type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// Real-valued continuous variable.
    Continuous,
    /// Binary variable (0 or 1).
    Binary,
    /// General integer variable.
    Integer,
}

/// Metadata for one decision variable block.
#[derive(Debug, Clone)]
pub struct VarInfo {
    /// Variable name.
    pub name: String,
    /// Domain type (continuous, binary, or integer).
    pub var_type: VarType,
    /// Position in the flat variable vector.
    pub offset: usize,
    /// Total number of scalar elements.
    pub size: usize,
    /// Shape of the variable (empty for scalars).
    pub shape: Vec<usize>,
    /// Element-wise lower bounds.
    pub lb: Vec<f64>,
    /// Element-wise upper bounds.
    pub ub: Vec<f64>,
}

/// A single constraint: body sense rhs.
#[derive(Debug, Clone)]
pub struct ConstraintRepr {
    /// Expression for the constraint left-hand side.
    pub body: ExprId,
    /// Comparison sense (<=, ==, >=).
    pub sense: ConstraintSense,
    /// Right-hand side constant.
    pub rhs: f64,
    /// Optional constraint name.
    pub name: Option<String>,
}

/// Complete model representation in Rust.
#[derive(Debug, Clone)]
pub struct ModelRepr {
    /// Expression arena holding all nodes.
    pub arena: ExprArena,
    /// Root expression id for the objective function.
    pub objective: ExprId,
    /// Minimize or maximize.
    pub objective_sense: ObjectiveSense,
    /// List of constraints.
    pub constraints: Vec<ConstraintRepr>,
    /// Variable metadata blocks.
    pub variables: Vec<VarInfo>,
    /// Total number of scalar variables (sum of all var sizes).
    pub n_vars: usize,
}

// ─────────────────────────────────────────────────────────────
// ModelBuilder — fast construction without Python expression objects
// ─────────────────────────────────────────────────────────────

/// Incremental model builder for fast construction of linear/quadratic
/// models without Python expression objects. Builds directly into the
/// Rust ExprArena.
pub struct ModelBuilder {
    /// Expression arena holding all nodes.
    pub arena: ExprArena,
    /// Variable metadata blocks.
    pub variables: Vec<VarInfo>,
    /// Constraints built so far.
    pub constraints: Vec<ConstraintRepr>,
    /// Objective expression (if set).
    pub objective: Option<ExprId>,
    /// Optimization direction.
    pub objective_sense: ObjectiveSense,
    /// Total number of scalar variables.
    pub n_vars: usize,
    /// Block index → ExprId of the Variable node in the arena.
    pub var_expr_ids: Vec<ExprId>,
    /// Cache of Index nodes: (var_block_idx, column) → ExprId.
    index_cache: std::collections::HashMap<(usize, usize), ExprId>,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            arena: ExprArena::new(),
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: None,
            objective_sense: ObjectiveSense::Minimize,
            n_vars: 0,
            var_expr_ids: Vec::new(),
            index_cache: std::collections::HashMap::new(),
        }
    }

    /// Register a variable block. Returns the block index.
    pub fn add_variable(
        &mut self,
        name: String,
        var_type: VarType,
        shape: Vec<usize>,
        lb: Vec<f64>,
        ub: Vec<f64>,
    ) -> usize {
        let size: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let offset = self.n_vars;
        let block_idx = self.variables.len();

        self.variables.push(VarInfo {
            name: name.clone(),
            var_type,
            offset,
            size,
            shape: shape.clone(),
            lb,
            ub,
        });

        let expr_id = self.arena.add(ExprNode::Variable {
            name,
            index: block_idx,
            size,
            shape,
        });
        self.var_expr_ids.push(expr_id);

        self.n_vars += size;
        block_idx
    }

    /// Get or create an Index(var, Scalar(col)) node, caching to avoid duplicates.
    fn get_index_node(&mut self, var_block_idx: usize, col: usize) -> ExprId {
        let key = (var_block_idx, col);
        if let Some(&id) = self.index_cache.get(&key) {
            return id;
        }
        let var_id = self.var_expr_ids[var_block_idx];
        let id = self.arena.add(ExprNode::Index {
            base: var_id,
            index: IndexSpec::Scalar(col),
        });
        self.index_cache.insert(key, id);
        id
    }

    /// Add linear constraints from CSR sparse data: A[row] @ x[var_idx] sense rhs[row].
    ///
    /// # Arguments
    /// * `indptr` — CSR row pointer array, length m+1
    /// * `indices` — CSR column indices
    /// * `data` — CSR nonzero values
    /// * `var_idx` — Variable block index
    /// * `sense` — Constraint sense (Le, Eq, Ge)
    /// * `rhs` — Right-hand side vector, length m
    /// * `name_prefix` — Optional name prefix for constraints
    #[allow(clippy::too_many_arguments)]
    pub fn add_linear_constraints_csr(
        &mut self,
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        var_idx: usize,
        sense: ConstraintSense,
        rhs: &[f64],
        name_prefix: Option<&str>,
    ) {
        let m = indptr.len() - 1; // number of rows

        // Pre-allocate arena capacity: each nonzero needs a Constant + Mul node,
        // each row needs a SumOver node.
        let nnz = data.len();
        self.arena.reserve(nnz * 2 + m);

        for row in 0..m {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];

            let body = if row_start == row_end {
                // Empty row → constant 0
                self.arena.add(ExprNode::Constant(0.0))
            } else {
                let mut terms = Vec::with_capacity(row_end - row_start);
                for k in row_start..row_end {
                    let col = indices[k];
                    let val = data[k];

                    let idx_node = self.get_index_node(var_idx, col);

                    if (val - 1.0).abs() < 1e-15 {
                        // Coefficient is 1.0, skip multiplication
                        terms.push(idx_node);
                    } else {
                        let const_node = self.arena.add(ExprNode::Constant(val));
                        let mul_node = self.arena.add(ExprNode::BinaryOp {
                            op: BinOp::Mul,
                            left: const_node,
                            right: idx_node,
                        });
                        terms.push(mul_node);
                    }
                }

                if terms.len() == 1 {
                    terms[0]
                } else {
                    self.arena.add(ExprNode::SumOver { terms })
                }
            };

            let name = name_prefix.map(|p| format!("{}_{}", p, row));
            self.constraints.push(ConstraintRepr {
                body,
                sense,
                rhs: rhs[row],
                name,
            });
        }
    }

    /// Set a linear objective: c'x + constant.
    pub fn set_linear_objective(
        &mut self,
        c: &[f64],
        var_idx: usize,
        constant: f64,
        sense: ObjectiveSense,
    ) {
        let mut terms = Vec::new();

        for (j, &cj) in c.iter().enumerate() {
            if cj.abs() < 1e-15 {
                continue;
            }
            let idx_node = self.get_index_node(var_idx, j);
            if (cj - 1.0).abs() < 1e-15 {
                terms.push(idx_node);
            } else {
                let const_node = self.arena.add(ExprNode::Constant(cj));
                let mul_node = self.arena.add(ExprNode::BinaryOp {
                    op: BinOp::Mul,
                    left: const_node,
                    right: idx_node,
                });
                terms.push(mul_node);
            }
        }

        let expr = if terms.is_empty() {
            self.arena.add(ExprNode::Constant(constant))
        } else {
            let lin = if terms.len() == 1 {
                terms[0]
            } else {
                self.arena.add(ExprNode::SumOver { terms })
            };
            if constant.abs() < 1e-15 {
                lin
            } else {
                let c_node = self.arena.add(ExprNode::Constant(constant));
                self.arena.add(ExprNode::BinaryOp {
                    op: BinOp::Add,
                    left: lin,
                    right: c_node,
                })
            }
        };

        self.objective = Some(expr);
        self.objective_sense = sense;
    }

    /// Set a quadratic objective: 0.5 x'Qx + c'x + constant.
    ///
    /// Q is provided in CSR format.
    #[allow(clippy::too_many_arguments)]
    pub fn set_quadratic_objective(
        &mut self,
        q_indptr: &[usize],
        q_indices: &[usize],
        q_data: &[f64],
        c: &[f64],
        var_idx: usize,
        constant: f64,
        sense: ObjectiveSense,
    ) {
        let mut terms = Vec::new();

        // Quadratic terms: 0.5 * sum_{i,j} Q[i,j] * x[i] * x[j]
        let n_rows = q_indptr.len() - 1;
        for i in 0..n_rows {
            let row_start = q_indptr[i];
            let row_end = q_indptr[i + 1];
            for k in row_start..row_end {
                let j = q_indices[k];
                let qij = q_data[k];
                if qij.abs() < 1e-15 {
                    continue;
                }
                // Only process upper triangle (i <= j) to avoid double-counting
                if i > j {
                    continue;
                }

                let xi = self.get_index_node(var_idx, i);
                let xj = self.get_index_node(var_idx, j);

                let coeff = if i == j { 0.5 * qij } else { qij };
                let prod = self.arena.add(ExprNode::BinaryOp {
                    op: BinOp::Mul,
                    left: xi,
                    right: xj,
                });
                if (coeff - 1.0).abs() < 1e-15 {
                    terms.push(prod);
                } else {
                    let c_node = self.arena.add(ExprNode::Constant(coeff));
                    terms.push(self.arena.add(ExprNode::BinaryOp {
                        op: BinOp::Mul,
                        left: c_node,
                        right: prod,
                    }));
                }
            }
        }

        // Linear terms: c'x
        for (j, &cj) in c.iter().enumerate() {
            if cj.abs() < 1e-15 {
                continue;
            }
            let idx_node = self.get_index_node(var_idx, j);
            if (cj - 1.0).abs() < 1e-15 {
                terms.push(idx_node);
            } else {
                let const_node = self.arena.add(ExprNode::Constant(cj));
                terms.push(self.arena.add(ExprNode::BinaryOp {
                    op: BinOp::Mul,
                    left: const_node,
                    right: idx_node,
                }));
            }
        }

        let expr = if terms.is_empty() {
            self.arena.add(ExprNode::Constant(constant))
        } else {
            let body = if terms.len() == 1 {
                terms[0]
            } else {
                self.arena.add(ExprNode::SumOver { terms })
            };
            if constant.abs() < 1e-15 {
                body
            } else {
                let c_node = self.arena.add(ExprNode::Constant(constant));
                self.arena.add(ExprNode::BinaryOp {
                    op: BinOp::Add,
                    left: body,
                    right: c_node,
                })
            }
        };

        self.objective = Some(expr);
        self.objective_sense = sense;
    }

    /// Consume the builder into a ModelRepr.
    ///
    /// Panics if no objective has been set.
    pub fn build(self) -> ModelRepr {
        let objective = self
            .objective
            .expect("ModelBuilder: no objective set. Call set_linear_objective() or set_quadratic_objective().");
        ModelRepr {
            arena: self.arena,
            objective,
            objective_sense: self.objective_sense,
            constraints: self.constraints,
            variables: self.variables,
            n_vars: self.n_vars,
        }
    }
}

impl ExprArena {
    /// Reserve additional capacity in the arena.
    pub fn reserve(&mut self, additional: usize) {
        self.nodes.reserve(additional);
    }
}

// ─────────────────────────────────────────────────────────────
// Structure detection
// ─────────────────────────────────────────────────────────────

impl ExprArena {
    /// Returns `true` if the expression is linear in the variables.
    ///
    /// An expression is linear if it is a sum of (constant * variable)
    /// terms plus a constant offset, with no variable-variable products.
    pub fn is_linear(&self, id: ExprId) -> bool {
        self.max_degree(id) <= 1
    }

    /// Returns `true` if the expression is at most quadratic.
    pub fn is_quadratic(&self, id: ExprId) -> bool {
        self.max_degree(id) <= 2
    }

    /// Returns `true` if the expression is a bilinear product of two
    /// different variables (exactly degree 2 with two distinct variable
    /// factors).
    pub fn is_bilinear(&self, id: ExprId) -> bool {
        match self.get(id) {
            ExprNode::BinaryOp {
                op: BinOp::Mul,
                left,
                right,
            } => {
                let ld = self.max_degree(*left);
                let rd = self.max_degree(*right);
                if ld == 1 && rd == 1 {
                    // Both sides must depend on variables, check they
                    // involve distinct variables.
                    let lv = self.collect_var_indices(*left);
                    let rv = self.collect_var_indices(*right);
                    // Bilinear means they touch at least one variable
                    // each, and at least some variables are different.
                    !lv.is_empty() && !rv.is_empty() && lv != rv
                } else {
                    false
                }
            }
            // A SumOver of bilinear terms is also bilinear-structured.
            ExprNode::SumOver { terms } => terms.iter().all(|t| self.is_bilinear(*t)),
            _ => false,
        }
    }

    /// Compute the maximum polynomial degree of an expression.
    ///
    /// Returns `usize::MAX` for transcendental functions (exp, log, sin, ...).
    fn max_degree(&self, id: ExprId) -> usize {
        match self.get(id) {
            ExprNode::Constant(_) | ExprNode::ConstantArray(_, _) | ExprNode::Parameter { .. } => 0,
            ExprNode::Variable { .. } => 1,
            ExprNode::BinaryOp { op, left, right } => {
                let ld = self.max_degree(*left);
                let rd = self.max_degree(*right);
                match op {
                    BinOp::Add | BinOp::Sub => ld.max(rd),
                    BinOp::Mul => ld.saturating_add(rd),
                    BinOp::Div => {
                        // If denominator involves variables, not polynomial.
                        if rd > 0 {
                            usize::MAX
                        } else {
                            ld
                        }
                    }
                    BinOp::Pow => {
                        // x^c where c is constant integer
                        if rd == 0 {
                            // Try to extract the constant exponent.
                            if let Some(exp) = self.try_constant_value(*right) {
                                let exp_int = exp as usize;
                                if (exp - exp_int as f64).abs() < 1e-12 {
                                    ld.saturating_mul(exp_int)
                                } else {
                                    usize::MAX
                                }
                            } else {
                                usize::MAX
                            }
                        } else {
                            usize::MAX
                        }
                    }
                }
            }
            ExprNode::UnaryOp { op, operand } => match op {
                UnOp::Neg => self.max_degree(*operand),
                UnOp::Abs => {
                    // abs is not polynomial in general, but for structure
                    // detection we treat abs(linear) as degree 1.
                    let d = self.max_degree(*operand);
                    if d <= 1 {
                        d
                    } else {
                        usize::MAX
                    }
                }
            },
            ExprNode::FunctionCall { func, .. } => {
                match func {
                    // All transcendental functions are non-polynomial.
                    MathFunc::Exp
                    | MathFunc::Log
                    | MathFunc::Log2
                    | MathFunc::Log10
                    | MathFunc::Sqrt
                    | MathFunc::Sin
                    | MathFunc::Cos
                    | MathFunc::Tan
                    | MathFunc::Atan
                    | MathFunc::Sinh
                    | MathFunc::Cosh
                    | MathFunc::Asin
                    | MathFunc::Acos
                    | MathFunc::Tanh
                    | MathFunc::Norm2 => usize::MAX,
                    // abs, sign: not strictly polynomial but handled specially
                    MathFunc::Abs | MathFunc::Sign => usize::MAX,
                    // min/max: non-smooth
                    MathFunc::Min | MathFunc::Max => usize::MAX,
                    // prod: depends on arguments
                    MathFunc::Prod => usize::MAX,
                }
            }
            ExprNode::Index { base, .. } => self.max_degree(*base),
            ExprNode::MatMul { left, right } => {
                let ld = self.max_degree(*left);
                let rd = self.max_degree(*right);
                ld.saturating_add(rd)
            }
            ExprNode::Sum { operand, .. } => self.max_degree(*operand),
            ExprNode::SumOver { terms } => {
                terms.iter().map(|t| self.max_degree(*t)).max().unwrap_or(0)
            }
        }
    }

    /// Try to extract a scalar constant value from a node.
    fn try_constant_value(&self, id: ExprId) -> Option<f64> {
        match self.get(id) {
            ExprNode::Constant(v) => Some(*v),
            ExprNode::Parameter { value, shape, .. } => {
                if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
                    value.first().copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Collect all variable indices referenced by an expression.
    fn collect_var_indices(&self, id: ExprId) -> Vec<usize> {
        let mut indices = Vec::new();
        self.collect_var_indices_inner(id, &mut indices);
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    fn collect_var_indices_inner(&self, id: ExprId, out: &mut Vec<usize>) {
        match self.get(id) {
            ExprNode::Variable { index, .. } => out.push(*index),
            ExprNode::BinaryOp { left, right, .. } | ExprNode::MatMul { left, right } => {
                self.collect_var_indices_inner(*left, out);
                self.collect_var_indices_inner(*right, out);
            }
            ExprNode::UnaryOp { operand, .. } | ExprNode::Sum { operand, .. } => {
                self.collect_var_indices_inner(*operand, out);
            }
            ExprNode::FunctionCall { args, .. } => {
                for a in args {
                    self.collect_var_indices_inner(*a, out);
                }
            }
            ExprNode::Index { base, .. } => self.collect_var_indices_inner(*base, out),
            ExprNode::SumOver { terms } => {
                for t in terms {
                    self.collect_var_indices_inner(*t, out);
                }
            }
            ExprNode::Constant(_) | ExprNode::ConstantArray(_, _) | ExprNode::Parameter { .. } => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Expression evaluation
// ─────────────────────────────────────────────────────────────

impl ExprArena {
    /// Evaluate an expression at a given point.
    ///
    /// `x` is the flat variable vector (length = total scalar variables).
    /// For scalar expressions this returns the scalar value. For array
    /// variables, indexing is required first.
    pub fn evaluate(&self, id: ExprId, x: &[f64]) -> f64 {
        match self.get(id) {
            ExprNode::Constant(v) => *v,
            ExprNode::ConstantArray(data, _shape) => {
                // If used as a scalar, return the single element or sum.
                if data.len() == 1 {
                    data[0]
                } else {
                    // Array used as scalar -- shouldn't happen in well-formed
                    // expressions, but return NaN to signal misuse.
                    f64::NAN
                }
            }
            ExprNode::Variable { size, shape, .. } => {
                if *size == 1 {
                    // Scalar variable — compute the flat offset.
                    let offset = self.var_offset(id);
                    x[offset]
                } else {
                    // Array variable evaluated as scalar (need index node above).
                    // Return NaN to signal the user needs to index first.
                    // However, for single-element shapes like (1,), return the element.
                    if shape.iter().product::<usize>() == 1 {
                        let offset = self.var_offset(id);
                        x[offset]
                    } else {
                        f64::NAN
                    }
                }
            }
            ExprNode::Parameter { value, shape, .. } => {
                if value.len() == 1 || shape.is_empty() {
                    value[0]
                } else {
                    f64::NAN
                }
            }
            ExprNode::BinaryOp { op, left, right } => {
                let lv = self.evaluate(*left, x);
                let rv = self.evaluate(*right, x);
                match op {
                    BinOp::Add => lv + rv,
                    BinOp::Sub => lv - rv,
                    BinOp::Mul => lv * rv,
                    BinOp::Div => lv / rv,
                    BinOp::Pow => lv.powf(rv),
                }
            }
            ExprNode::UnaryOp { op, operand } => {
                let v = self.evaluate(*operand, x);
                match op {
                    UnOp::Neg => -v,
                    UnOp::Abs => v.abs(),
                }
            }
            ExprNode::FunctionCall { func, args } => {
                let a0 = if args.is_empty() {
                    f64::NAN
                } else {
                    self.evaluate(args[0], x)
                };
                match func {
                    MathFunc::Exp => a0.exp(),
                    MathFunc::Log => a0.ln(),
                    MathFunc::Log2 => a0.log2(),
                    MathFunc::Log10 => a0.log10(),
                    MathFunc::Sqrt => a0.sqrt(),
                    MathFunc::Sin => a0.sin(),
                    MathFunc::Cos => a0.cos(),
                    MathFunc::Tan => a0.tan(),
                    MathFunc::Atan => a0.atan(),
                    MathFunc::Sinh => a0.sinh(),
                    MathFunc::Cosh => a0.cosh(),
                    MathFunc::Asin => a0.asin(),
                    MathFunc::Acos => a0.acos(),
                    MathFunc::Tanh => a0.tanh(),
                    MathFunc::Abs => a0.abs(),
                    MathFunc::Sign => {
                        if a0 > 0.0 {
                            1.0
                        } else if a0 < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                    MathFunc::Min => {
                        let a1 = if args.len() > 1 {
                            self.evaluate(args[1], x)
                        } else {
                            f64::NAN
                        };
                        a0.min(a1)
                    }
                    MathFunc::Max => {
                        let a1 = if args.len() > 1 {
                            self.evaluate(args[1], x)
                        } else {
                            f64::NAN
                        };
                        a0.max(a1)
                    }
                    MathFunc::Prod => a0,  // single-arg prod is identity
                    MathFunc::Norm2 => a0, // single-arg norm2 is abs
                }
            }
            ExprNode::Index { base, index } => {
                // Evaluate the indexed element from a variable or constant array.
                match self.get(*base) {
                    ExprNode::Variable { shape, .. } => {
                        let offset = self.var_offset(*base);
                        let flat = index_spec_to_flat(index, shape);
                        x[offset + flat]
                    }
                    ExprNode::ConstantArray(data, shape) => {
                        let flat = index_spec_to_flat(index, shape);
                        data[flat]
                    }
                    ExprNode::Parameter { value, shape, .. } => {
                        let flat = index_spec_to_flat(index, shape);
                        value[flat]
                    }
                    _ => {
                        // Indexing a compound expression — not yet supported
                        // in scalar evaluation. Would need array evaluation.
                        f64::NAN
                    }
                }
            }
            ExprNode::MatMul { left, right } => {
                // MatMul in scalar evaluation context: treat as dot product
                // of the two flat vectors.
                self.evaluate_matmul(*left, *right, x)
            }
            ExprNode::Sum { operand, .. } => {
                // Sum all elements of the operand.
                self.evaluate_sum_all(*operand, x)
            }
            ExprNode::SumOver { terms } => terms.iter().map(|t| self.evaluate(*t, x)).sum(),
        }
    }

    /// Compute the flat offset for a variable node by scanning all
    /// Variable nodes with lower indices.
    fn var_offset(&self, id: ExprId) -> usize {
        if let ExprNode::Variable { index, .. } = self.get(id) {
            let target_idx = *index;
            // Compute offset by summing sizes of all vars with
            // index < target_idx. But we only have one node per variable
            // identity, so we need the VarInfo. Instead, use the simpler
            // approach: scan for all distinct Variable nodes, sort by index,
            // sum sizes up to target.
            let mut vars: Vec<(usize, usize)> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for node in &self.nodes {
                if let ExprNode::Variable {
                    index: idx, size, ..
                } = node
                {
                    if seen.insert(*idx) {
                        vars.push((*idx, *size));
                    }
                }
            }
            vars.sort_by_key(|(idx, _)| *idx);
            let mut off = 0;
            for (idx, sz) in &vars {
                if *idx == target_idx {
                    return off;
                }
                off += sz;
            }
            off
        } else {
            0
        }
    }

    /// Evaluate matrix-multiply as a dot product for 1-D vectors,
    /// or sum(a_i * b_i) for arrays.
    fn evaluate_matmul(&self, left: ExprId, right: ExprId, x: &[f64]) -> f64 {
        let lv = self.collect_array_values(left, x);
        let rv = self.collect_array_values(right, x);
        // Dot product of the collected values.
        lv.iter().zip(rv.iter()).map(|(a, b)| a * b).sum()
    }

    /// Collect all scalar values from an expression that represents
    /// an array (variable or constant array).
    fn collect_array_values(&self, id: ExprId, x: &[f64]) -> Vec<f64> {
        match self.get(id) {
            ExprNode::Variable { size, .. } => {
                let offset = self.var_offset(id);
                x[offset..offset + size].to_vec()
            }
            ExprNode::ConstantArray(data, _) => data.clone(),
            ExprNode::Constant(v) => vec![*v],
            ExprNode::Parameter { value, .. } => value.clone(),
            _ => vec![self.evaluate(id, x)],
        }
    }

    /// Sum all elements of an array-valued expression.
    fn evaluate_sum_all(&self, operand: ExprId, x: &[f64]) -> f64 {
        let vals = self.collect_array_values(operand, x);
        vals.iter().sum()
    }
}

/// Convert an IndexSpec to a flat index given a shape (row-major / C-order).
fn index_spec_to_flat(spec: &IndexSpec, shape: &[usize]) -> usize {
    match spec {
        IndexSpec::Scalar(i) => *i,
        IndexSpec::Tuple(indices) => {
            let mut flat = 0;
            let mut stride = 1;
            // Walk dimensions right-to-left, accumulating strides.
            for (&idx, &dim) in indices.iter().rev().zip(shape.iter().rev()) {
                flat += idx * stride;
                stride *= dim;
            }
            flat
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Evaluate with ModelRepr (uses VarInfo for offsets)
// ─────────────────────────────────────────────────────────────

impl ModelRepr {
    /// Evaluate the objective at a given point.
    pub fn evaluate_objective(&self, x: &[f64]) -> f64 {
        self.evaluate_expr(self.objective, x)
    }

    /// Evaluate an expression using the model's variable info for offsets.
    pub fn evaluate_expr(&self, id: ExprId, x: &[f64]) -> f64 {
        self.evaluate_node(id, x)
    }

    fn evaluate_node(&self, id: ExprId, x: &[f64]) -> f64 {
        match self.arena.get(id) {
            ExprNode::Constant(v) => *v,
            ExprNode::ConstantArray(data, _) => {
                if data.len() == 1 {
                    data[0]
                } else {
                    f64::NAN
                }
            }
            ExprNode::Variable { index, size, .. } => {
                if *size == 1 {
                    let offset = self.variables[*index].offset;
                    x[offset]
                } else {
                    f64::NAN
                }
            }
            ExprNode::Parameter { value, .. } => {
                if value.len() == 1 {
                    value[0]
                } else {
                    f64::NAN
                }
            }
            ExprNode::BinaryOp { op, left, right } => {
                let lv = self.evaluate_node(*left, x);
                let rv = self.evaluate_node(*right, x);
                match op {
                    BinOp::Add => lv + rv,
                    BinOp::Sub => lv - rv,
                    BinOp::Mul => lv * rv,
                    BinOp::Div => lv / rv,
                    BinOp::Pow => lv.powf(rv),
                }
            }
            ExprNode::UnaryOp { op, operand } => {
                let v = self.evaluate_node(*operand, x);
                match op {
                    UnOp::Neg => -v,
                    UnOp::Abs => v.abs(),
                }
            }
            ExprNode::FunctionCall { func, args } => {
                let a0 = if args.is_empty() {
                    f64::NAN
                } else {
                    self.evaluate_node(args[0], x)
                };
                match func {
                    MathFunc::Exp => a0.exp(),
                    MathFunc::Log => a0.ln(),
                    MathFunc::Log2 => a0.log2(),
                    MathFunc::Log10 => a0.log10(),
                    MathFunc::Sqrt => a0.sqrt(),
                    MathFunc::Sin => a0.sin(),
                    MathFunc::Cos => a0.cos(),
                    MathFunc::Tan => a0.tan(),
                    MathFunc::Atan => a0.atan(),
                    MathFunc::Sinh => a0.sinh(),
                    MathFunc::Cosh => a0.cosh(),
                    MathFunc::Asin => a0.asin(),
                    MathFunc::Acos => a0.acos(),
                    MathFunc::Tanh => a0.tanh(),
                    MathFunc::Abs => a0.abs(),
                    MathFunc::Sign => {
                        if a0 > 0.0 {
                            1.0
                        } else if a0 < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                    MathFunc::Min => {
                        let a1 = if args.len() > 1 {
                            self.evaluate_node(args[1], x)
                        } else {
                            f64::NAN
                        };
                        a0.min(a1)
                    }
                    MathFunc::Max => {
                        let a1 = if args.len() > 1 {
                            self.evaluate_node(args[1], x)
                        } else {
                            f64::NAN
                        };
                        a0.max(a1)
                    }
                    MathFunc::Prod => a0,
                    MathFunc::Norm2 => a0,
                }
            }
            ExprNode::Index { base, index } => match self.arena.get(*base) {
                ExprNode::Variable {
                    index: var_idx,
                    shape,
                    ..
                } => {
                    let offset = self.variables[*var_idx].offset;
                    let flat = index_spec_to_flat(index, shape);
                    x[offset + flat]
                }
                ExprNode::ConstantArray(data, shape) => {
                    let flat = index_spec_to_flat(index, shape);
                    data[flat]
                }
                ExprNode::Parameter { value, shape, .. } => {
                    let flat = index_spec_to_flat(index, shape);
                    value[flat]
                }
                _ => f64::NAN,
            },
            ExprNode::MatMul { left, right } => self.evaluate_matmul(*left, *right, x),
            ExprNode::Sum { operand, .. } => self.evaluate_sum_all(*operand, x),
            ExprNode::SumOver { terms } => terms.iter().map(|t| self.evaluate_node(*t, x)).sum(),
        }
    }

    fn collect_array_values(&self, id: ExprId, x: &[f64]) -> Vec<f64> {
        match self.arena.get(id) {
            ExprNode::Variable { index, size, .. } => {
                let offset = self.variables[*index].offset;
                x[offset..offset + size].to_vec()
            }
            ExprNode::ConstantArray(data, _) => data.clone(),
            ExprNode::Constant(v) => vec![*v],
            ExprNode::Parameter { value, .. } => value.clone(),
            _ => vec![self.evaluate_node(id, x)],
        }
    }

    fn evaluate_matmul(&self, left: ExprId, right: ExprId, x: &[f64]) -> f64 {
        let lv = self.collect_array_values(left, x);
        let rv = self.collect_array_values(right, x);
        lv.iter().zip(rv.iter()).map(|(a, b)| a * b).sum()
    }

    fn evaluate_sum_all(&self, operand: ExprId, x: &[f64]) -> f64 {
        let vals = self.collect_array_values(operand, x);
        vals.iter().sum()
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_add_get() {
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(3.14));
        assert_eq!(c, ExprId(0));
        assert_eq!(arena.len(), 1);
        match arena.get(c) {
            ExprNode::Constant(v) => assert!((v - 3.14).abs() < 1e-15),
            _ => panic!("expected Constant"),
        }
    }

    #[test]
    fn test_linear_detection() {
        let mut arena = ExprArena::new();
        // x0 + 3.0
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let c3 = arena.add(ExprNode::Constant(3.0));
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x0,
            right: c3,
        });
        assert!(arena.is_linear(sum));
        assert!(arena.is_quadratic(sum));
    }

    #[test]
    fn test_quadratic_detection() {
        let mut arena = ExprArena::new();
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let c2 = arena.add(ExprNode::Constant(2.0));
        let sq = arena.add(ExprNode::BinaryOp {
            op: BinOp::Pow,
            left: x0,
            right: c2,
        });
        assert!(!arena.is_linear(sq));
        assert!(arena.is_quadratic(sq));
    }

    #[test]
    fn test_nonlinear_detection() {
        let mut arena = ExprArena::new();
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let exp_x = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![x0],
        });
        assert!(!arena.is_linear(exp_x));
        assert!(!arena.is_quadratic(exp_x));
    }

    #[test]
    fn test_bilinear_detection() {
        let mut arena = ExprArena::new();
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let x1 = arena.add(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x1,
        });
        assert!(arena.is_bilinear(prod));
        assert!(arena.is_quadratic(prod));
    }

    #[test]
    fn test_not_bilinear_same_var() {
        let mut arena = ExprArena::new();
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let x0b = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x0b,
        });
        // x * x is quadratic but NOT bilinear (same variable)
        assert!(!arena.is_bilinear(prod));
    }

    #[test]
    fn test_evaluate_constant() {
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(42.0));
        assert!((arena.evaluate(c, &[]) - 42.0).abs() < 1e-15);
    }

    #[test]
    fn test_evaluate_linear() {
        // 2*x + 3
        let mut arena = ExprArena::new();
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let c2 = arena.add(ExprNode::Constant(2.0));
        let c3 = arena.add(ExprNode::Constant(3.0));
        let mul = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c2,
            right: x0,
        });
        let add = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: mul,
            right: c3,
        });
        let val = arena.evaluate(add, &[5.0]);
        assert!((val - 13.0).abs() < 1e-15);
    }

    #[test]
    fn test_evaluate_quadratic() {
        // x^2 + y^2
        let model = ModelRepr {
            arena: {
                let mut a = ExprArena::new();
                let x = a.add(ExprNode::Variable {
                    name: "x".into(),
                    index: 0,
                    size: 1,
                    shape: vec![],
                });
                let y = a.add(ExprNode::Variable {
                    name: "y".into(),
                    index: 1,
                    size: 1,
                    shape: vec![],
                });
                let c2 = a.add(ExprNode::Constant(2.0));
                let c2b = a.add(ExprNode::Constant(2.0));
                let xsq = a.add(ExprNode::BinaryOp {
                    op: BinOp::Pow,
                    left: x,
                    right: c2,
                });
                let ysq = a.add(ExprNode::BinaryOp {
                    op: BinOp::Pow,
                    left: y,
                    right: c2b,
                });
                let _sum = a.add(ExprNode::BinaryOp {
                    op: BinOp::Add,
                    left: xsq,
                    right: ysq,
                });
                a
            },
            objective: ExprId(6), // the sum node
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![
                VarInfo {
                    name: "x".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 1,
                    shape: vec![],
                    lb: vec![-1e20],
                    ub: vec![1e20],
                },
                VarInfo {
                    name: "y".into(),
                    var_type: VarType::Continuous,
                    offset: 1,
                    size: 1,
                    shape: vec![],
                    lb: vec![-1e20],
                    ub: vec![1e20],
                },
            ],
            n_vars: 2,
        };
        let val = model.evaluate_objective(&[3.0, 4.0]);
        assert!((val - 25.0).abs() < 1e-15); // 9 + 16 = 25
    }

    #[test]
    fn test_evaluate_exp() {
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let exp_x = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![x],
        });
        let val = arena.evaluate(exp_x, &[1.0]);
        assert!((val - 1.0_f64.exp()).abs() < 1e-14);
    }

    #[test]
    fn test_index_spec_to_flat() {
        assert_eq!(index_spec_to_flat(&IndexSpec::Scalar(3), &[5]), 3);
        // 2D: shape (3, 4), index (1, 2) => 1*4 + 2 = 6
        assert_eq!(
            index_spec_to_flat(&IndexSpec::Tuple(vec![1, 2]), &[3, 4]),
            6
        );
    }

    #[test]
    fn test_evaluate_sum_over() {
        // sum of [c1, c2, c3] = 6.0
        let mut arena = ExprArena::new();
        let c1 = arena.add(ExprNode::Constant(1.0));
        let c2 = arena.add(ExprNode::Constant(2.0));
        let c3 = arena.add(ExprNode::Constant(3.0));
        let s = arena.add(ExprNode::SumOver {
            terms: vec![c1, c2, c3],
        });
        assert!((arena.evaluate(s, &[]) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_unary_neg() {
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(5.0));
        let neg = arena.add(ExprNode::UnaryOp {
            op: UnOp::Neg,
            operand: c,
        });
        assert!((arena.evaluate(neg, &[]) - (-5.0)).abs() < 1e-15);
    }

    #[test]
    fn test_default_arena() {
        let arena = ExprArena::default();
        assert!(arena.is_empty());
    }
}
