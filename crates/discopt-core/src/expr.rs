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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// Arithmetic negation (`-x`).
    Neg,
    /// Absolute value (`|x|`).
    Abs,
}

/// Named mathematical functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Inverse hyperbolic sine (`asinh(x) = ln(x + sqrt(x^2 + 1))`).
    Asinh,
    /// Inverse hyperbolic cosine (`acosh(x) = ln(x + sqrt(x^2 - 1))`, x >= 1).
    Acosh,
    /// Inverse hyperbolic tangent (`atanh(x) = 0.5 ln((1+x)/(1-x))`, |x| < 1).
    Atanh,
    /// Gauss error function.
    Erf,
    /// Natural log of one plus the argument (`log1p(x) = ln(1 + x)`).
    Log1p,
    /// Logistic sigmoid (`sigmoid(x) = 1 / (1 + e^{-x})`).
    Sigmoid,
    /// Softplus (`softplus(x) = ln(1 + e^x)`).
    Softplus,
    /// L1 norm (sum of absolute values).
    Norm1,
    /// L-infinity norm (maximum absolute value).
    NormInf,
    /// General p-norm `(Σ |x_i|^p)^{1/p}` for an integer order `p >= 1`
    /// (orders 1 and 2 use the dedicated `Norm1` / `Norm2` variants).
    NormP(u32),
}

/// One axis of a generalized index: either a scalar position or a slice.
///
/// Slices use Python semantics: `start`, `stop`, `step` may be `None` to mean
/// "default for the direction of `step`", and negative values are interpreted
/// relative to the axis length. A step of zero is rejected at construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexElem {
    /// A scalar index along this axis (e.g. `i` in `x[i, :]`).
    Scalar(usize),
    /// A slice along this axis. `Slice { start: None, stop: None, step: None }`
    /// represents the full slice (`:`).
    Slice {
        /// Inclusive start (Python-relative; negatives count from the end).
        start: Option<isize>,
        /// Exclusive stop.
        stop: Option<isize>,
        /// Stride. `None` means 1; must be non-zero.
        step: Option<isize>,
    },
}

impl IndexElem {
    /// The full-axis slice (`:`).
    pub const FULL: IndexElem = IndexElem::Slice {
        start: None,
        stop: None,
        step: None,
    };
}

/// Indexing specification for array access.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexSpec {
    /// Single scalar index: `x[i]`
    Scalar(usize),
    /// Tuple of scalar indices: `x[i, j]`
    Tuple(Vec<usize>),
    /// Mixed scalar/slice indexing: `x[i, :]`, `x[:, j]`, etc. The result is
    /// array-valued unless every element is `Scalar`.
    Multi(Vec<IndexElem>),
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

/// Content-addressed structural key for hash-consing (CSE) of arena nodes.
///
/// Two nodes with equal [`StructuralKey`]s are *structurally identical*: same
/// operator, same operand [`ExprId`]s (in order — commutative operands are **not**
/// reordered, so `a+b` and `b+a` intern separately; correctness over completeness),
/// and same literal payload. Interning a node whose key already exists returns the
/// existing id, which is sound because arena evaluation is a pure function of node
/// structure + operand ids (see [`ExprArena::evaluate`]). Scalar constants are keyed
/// by their exact IEEE-754 bit pattern (`f64::to_bits`) so `0.0`/`-0.0` and any NaN
/// bit patterns never falsely merge. Different shapes/types never share a key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StructuralKey {
    Constant(u64),
    ConstantArray(Vec<u64>, Vec<usize>),
    /// Variable identity is its block index (name is metadata, not identity).
    Variable(usize),
    /// Parameter identity is (name, value bits, shape).
    Parameter(String, Vec<u64>, Vec<usize>),
    BinaryOp(BinOp, ExprId, ExprId),
    UnaryOp(UnOp, ExprId),
    FunctionCall(MathFunc, Vec<ExprId>),
    Index(ExprId, IndexSpec),
    MatMul(ExprId, ExprId),
    Sum(ExprId, Option<usize>),
    SumOver(Vec<ExprId>),
}

impl StructuralKey {
    /// Derive the structural key for a node. All operand ids referenced are the
    /// (already-interned) child ids, so equal keys imply semantic equivalence.
    fn of(node: &ExprNode) -> Self {
        match node {
            ExprNode::Constant(v) => StructuralKey::Constant(v.to_bits()),
            ExprNode::ConstantArray(data, shape) => StructuralKey::ConstantArray(
                data.iter().map(|v| v.to_bits()).collect(),
                shape.clone(),
            ),
            ExprNode::Variable { index, .. } => StructuralKey::Variable(*index),
            ExprNode::Parameter { name, value, shape } => StructuralKey::Parameter(
                name.clone(),
                value.iter().map(|v| v.to_bits()).collect(),
                shape.clone(),
            ),
            ExprNode::BinaryOp { op, left, right } => StructuralKey::BinaryOp(*op, *left, *right),
            ExprNode::UnaryOp { op, operand } => StructuralKey::UnaryOp(*op, *operand),
            ExprNode::FunctionCall { func, args } => {
                StructuralKey::FunctionCall(*func, args.clone())
            }
            ExprNode::Index { base, index } => StructuralKey::Index(*base, index.clone()),
            ExprNode::MatMul { left, right } => StructuralKey::MatMul(*left, *right),
            ExprNode::Sum { operand, axis } => StructuralKey::Sum(*operand, *axis),
            ExprNode::SumOver { terms } => StructuralKey::SumOver(terms.clone()),
        }
    }
}

/// Arena allocator for expression nodes.
///
/// All nodes live here; everything else holds [`ExprId`] handles.
///
/// Hash-consing (CSE): when interning is enabled via [`ExprArena::enable_interning`],
/// [`ExprArena::intern`] deduplicates structurally-identical nodes so that building the
/// same subexpression twice returns the same [`ExprId`]. The raw [`ExprArena::add`] is
/// **unchanged** — it always appends a fresh node — so post-construction passes
/// (presolve, reformulation) that rely on fresh-node semantics are unaffected.
#[derive(Debug, Clone)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
    /// Content-address → existing id, populated only while interning is enabled.
    /// Keyed lookup only (never iterated on an ordering-sensitive path), so
    /// node-id assignment stays deterministic and byte-reproducible.
    intern: Option<std::collections::HashMap<StructuralKey, ExprId>>,
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            intern: None,
        }
    }

    /// Create an arena with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            intern: None,
        }
    }

    /// Enable content-addressed hash-consing for subsequent [`ExprArena::intern`]
    /// calls. Seeds the intern table from any nodes already present so that
    /// interning a node equal to a pre-existing one returns that existing id.
    /// Idempotent. Interning is a *construction-time* optimization; disable it
    /// (or simply stop calling `intern`) before mutating passes that append nodes
    /// they intend to keep distinct.
    pub fn enable_interning(&mut self) {
        if self.intern.is_some() {
            return;
        }
        let mut map = std::collections::HashMap::with_capacity(self.nodes.len());
        // First occurrence of each structural key wins (lowest id), matching the
        // append order and keeping the mapping deterministic.
        for (i, node) in self.nodes.iter().enumerate() {
            map.entry(StructuralKey::of(node)).or_insert(ExprId(i));
        }
        self.intern = Some(map);
    }

    /// Disable hash-consing. Subsequent [`ExprArena::intern`] calls append like
    /// [`ExprArena::add`]. Existing node ids are untouched.
    pub fn disable_interning(&mut self) {
        self.intern = None;
    }

    /// Whether hash-consing is currently enabled.
    pub fn interning_enabled(&self) -> bool {
        self.intern.is_some()
    }

    /// Insert a node and return its id.
    pub fn add(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Content-addressed insert: if interning is enabled and a structurally
    /// identical node already exists, returns its existing id without appending;
    /// otherwise appends (via [`ExprArena::add`]) and records the mapping. With
    /// interning disabled this is exactly [`ExprArena::add`].
    ///
    /// Semantic-preserving by construction: two nodes share an id only when their
    /// [`StructuralKey`]s are equal, i.e. same op, same operand ids, and same literal
    /// payload/shape — so the deduped node evaluates identically to the duplicate it
    /// replaces (both are pure functions of the same structure). Operands are assumed
    /// to already be interned; the caller (a bottom-up build) guarantees this.
    pub fn intern(&mut self, node: ExprNode) -> ExprId {
        if self.intern.is_some() {
            let key = StructuralKey::of(&node);
            if let Some(&id) = self.intern.as_ref().unwrap().get(&key) {
                return id;
            }
            let id = self.add(node);
            self.intern.as_mut().unwrap().insert(key, id);
            id
        } else {
            self.add(node)
        }
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

/// A complementarity relation recovered from an AMPL `.nl` file.
///
/// Represents `body ⊥ x[var_index]`: the constraint body expression is
/// complementary to the variable at `var_index` (0-based). This is *not* a field
/// of [`ModelRepr`] — the parser threads it out separately (see
/// [`crate::nl_parser::parse_nl_with_complementarity`]) so the ubiquitous
/// `ModelRepr` struct literal stays untouched — and it references arena nodes
/// owned by the accompanying `ModelRepr`.
///
/// `flag` is the raw type-5 bound flag from the `.nl` `r` segment, with AMPL MP
/// `ComplInfo` semantics: bit 0 set ⇒ the body's lower bound is `-inf` (else 0),
/// bit 1 set ⇒ the body's upper bound is `+inf` (else 0). The standard MPEC form
/// `0 <= body ⊥ x >= 0` is `flag == 2`.
#[derive(Debug, Clone)]
pub struct ComplementarityRepr {
    /// Arena id of the constraint body `f` complementary to the variable.
    pub body: ExprId,
    /// 0-based index of the complementary variable.
    pub var_index: usize,
    /// Raw type-5 bound flag (AMPL MP `ComplInfo` bit semantics; see struct docs).
    pub flag: usize,
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

    // ── Symbolic quadratic-form extraction ────────────────────────────────
    //
    // `is_quadratic` above walks the entire DAG to decide a *degree*, then
    // throws away everything it learned and returns a bool. The Python QP
    // extractor, left with only that bool, recovered the coefficients by
    // finite-difference PROBING -- one full model evaluation per variable
    // *pair*, O(|support|^2) (`_relax/problem_classifier.py`,
    // `_extract_qp_data_from_repr`). Measured over the 150-instance MINLPLib
    // MIQP family (BQP/IQP/MBQP/MIQP): 29 instances need more than 60 s of
    // probing before the search can start, 71,330 s in total, worst case
    // `unitcommit_200_100_1_mod_8` at 28,288 s. That instance's objective DAG
    // has 56,299 nodes and 330,262,150 variable pairs -- the structure is
    // ~5,900x smaller than the space the probe searches.
    //
    // Worse, the probe is not merely slow. Its off-diagonal identity
    // `f(e_i + e_j) - f(e_i) - f(e_j) + f(0)` is a difference of nearly-equal
    // floats, so it loses precision to cancellation: on `chimera_mis-01` the
    // sweep costs 275 s and then fails its own #866 verification (recovers
    // -171.42 against a true -174.09), and all 275 s is discarded.
    //
    // This walk emits the coefficients the degree walk already sees. Every
    // output coefficient is a sum of products of literals that appear in the
    // DAG, so there is no subtractive cancellation and no step size. It is
    // O(nodes), it is iterative (a `.nl` objective is a left-nested chain --
    // 18,499 `+` nodes deep on `unitcommit_200_100_1_mod_8` -- which would
    // blow a recursive walk's stack), and it either succeeds exactly or
    // DECLINES. It never approximates: any form it cannot represent returns
    // `None` and the caller falls back to the existing ladder.
    //
    // The output is the same sparse COO triplet that `set_quadratic_objective`
    // (crates/discopt-python/src/expr_bindings.rs) already accepts in the
    // other direction; this is that function's missing inverse.

    /// Build the flat-offset map for every variable in the arena, in one pass.
    ///
    /// Reproduces [`Self::var_offset`] exactly (distinct `index`, sorted,
    /// prefix-summed `size`), but computes it once instead of re-scanning
    /// every node per variable occurrence -- which would itself be quadratic
    /// on a model with many variables.
    fn var_offset_map(&self) -> std::collections::HashMap<usize, usize> {
        let mut vars: Vec<(usize, usize)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for node in &self.nodes {
            if let ExprNode::Variable { index, size, .. } = node {
                if seen.insert(*index) {
                    vars.push((*index, *size));
                }
            }
        }
        vars.sort_by_key(|(idx, _)| *idx);
        let mut map = std::collections::HashMap::with_capacity(vars.len());
        let mut off = 0usize;
        for (idx, sz) in vars {
            map.insert(idx, off);
            off += sz;
        }
        map
    }

    /// The direct children of a node, in no particular order.
    fn quad_children(&self, id: ExprId) -> Vec<ExprId> {
        match self.get(id) {
            ExprNode::BinaryOp { left, right, .. } | ExprNode::MatMul { left, right } => {
                vec![*left, *right]
            }
            ExprNode::UnaryOp { operand, .. } | ExprNode::Sum { operand, .. } => vec![*operand],
            ExprNode::FunctionCall { args, .. } => args.clone(),
            ExprNode::SumOver { terms } => terms.clone(),
            // `Index` deliberately does NOT descend: the base is consumed
            // structurally below (only a Variable / ConstantArray / Parameter
            // base is representable), never as a polynomial value.
            _ => Vec::new(),
        }
    }

    /// Extract the objective/constraint body `id` as an exact sparse quadratic
    /// form over the arena's flat variable space.
    ///
    /// Returns `None` -- decline, never an approximation -- when the
    /// expression is not a quadratic form, when it uses a construct this walk
    /// does not represent, or when the accumulated coefficient count would
    /// exceed `max_terms` *live* coefficients (a dense `Q` on a large model
    /// is exactly the
    /// materialization this is meant to avoid; the caller keeps its fallback).
    ///
    /// The semantics mirror [`Self::evaluate`] / [`Self::collect_array_values`]
    /// node for node, including the `Sum { axis: Some(_) }` refusal (#1160) and
    /// the `abs`-of-a-variable refusal (#739). A caller that wants belt and
    /// braces can evaluate the returned form against `evaluate` at a random
    /// point; they agree to machine precision by construction.
    pub fn quadratic_form(&self, id: ExprId, max_terms: usize) -> Option<QuadForm> {
        let offsets = self.var_offset_map();
        let mut memo: std::collections::HashMap<ExprId, Option<std::rc::Rc<Vec<QuadForm>>>> =
            std::collections::HashMap::new();

        // In-degree over the reachable sub-DAG, counting multiplicity (`x * x`
        // consumes `x` twice). A memo entry is dropped the moment its last
        // parent has consumed it, which is what keeps peak memory proportional
        // to the *live* frontier rather than to the whole DAG: a left-nested
        // `+` chain accumulating a dense form would otherwise memoize every
        // prefix, at O(terms^2).
        let mut indeg: std::collections::HashMap<ExprId, usize> = std::collections::HashMap::new();
        let mut seen = std::collections::HashSet::new();
        seen.insert(id);
        let mut walk = vec![id];
        while let Some(cur) = walk.pop() {
            for c in self.quad_children(cur) {
                *indeg.entry(c).or_insert(0) += 1;
                if seen.insert(c) {
                    walk.push(c);
                }
            }
        }

        let mut live = 0usize;

        // Iterative post-order: push (node, children_done). A `.nl` objective
        // is a deeply left-nested chain, so this must not recurse.
        let mut stack: Vec<(ExprId, bool)> = vec![(id, false)];
        while let Some((cur, done)) = stack.pop() {
            if memo.contains_key(&cur) {
                continue;
            }
            if !done {
                stack.push((cur, true));
                // Push children in REVERSE so they pop left-to-right. A `.nl`
                // objective is a left-nested `+` chain, and popping the right
                // operand first would compute and hold every term on the way
                // down -- peak live memory O(chain length) instead of O(1).
                for c in self.quad_children(cur).into_iter().rev() {
                    if !memo.contains_key(&c) {
                        stack.push((c, false));
                    }
                }
            } else {
                let v = self.quad_combine(cur, &offsets, &memo);
                if let Some(rc) = v.as_ref() {
                    live += rc.iter().map(|q| q.n_terms()).sum::<usize>();
                    // Declining here keeps a would-be dense `Q` from ever being
                    // materialized; the caller keeps its fallback.
                    if live > max_terms {
                        return None;
                    }
                }
                memo.insert(cur, v);

                // Release every child whose last parent was this node. The
                // entry is tombstoned rather than removed so the
                // `contains_key` guards above stay valid; a stale read would
                // see `None` and decline, never a wrong answer.
                for c in self.quad_children(cur) {
                    if c == id {
                        continue;
                    }
                    if let Some(rem) = indeg.get_mut(&c) {
                        *rem -= 1;
                        if *rem == 0 {
                            if let Some(Some(rc)) = memo.get(&c) {
                                live -= rc.iter().map(|q| q.n_terms()).sum::<usize>();
                            }
                            memo.insert(c, None);
                        }
                    }
                }
            }
        }

        let vals = memo.get(&id)?.clone()?;
        if vals.len() != 1 {
            // An array-valued objective is not a scalar quadratic form.
            return None;
        }
        let mut out = vals[0].clone();
        // Drop exact zeros once, at the end. Pruning inside the accumulation
        // loop would re-scan the whole accumulator per term and reintroduce
        // the quadratic behaviour this function exists to remove.
        out.linear.retain(|_, v| *v != 0.0);
        out.quadratic.retain(|_, v| *v != 0.0);

        // Refuse a non-finite coefficient rather than emit one. `scaled(0.0)`
        // collapses `0 * x` to the constant 0, which differs from `evaluate`'s
        // `0 * inf = NaN`; division by an infinite constant is the reachable
        // route to that divergence. Checking the output once (O(nnz)) is
        // cheaper and more general than case-analysing every operator, and a
        // decline costs only the caller's fallback -- never a wrong model.
        if !out.is_finite() {
            return None;
        }
        Some(out)
    }

    /// Combine a node's already-computed children into its polynomial value.
    fn quad_combine(
        &self,
        id: ExprId,
        offsets: &std::collections::HashMap<usize, usize>,
        memo: &std::collections::HashMap<ExprId, Option<std::rc::Rc<Vec<QuadForm>>>>,
    ) -> Option<std::rc::Rc<Vec<QuadForm>>> {
        let child = |c: &ExprId| -> Option<std::rc::Rc<Vec<QuadForm>>> { memo.get(c)?.clone() };

        let vals: Vec<QuadForm> = match self.get(id) {
            ExprNode::Constant(v) => vec![QuadForm::constant(*v)],
            ExprNode::ConstantArray(data, _) => {
                data.iter().map(|v| QuadForm::constant(*v)).collect()
            }
            ExprNode::Parameter { value, .. } => {
                value.iter().map(|v| QuadForm::constant(*v)).collect()
            }
            ExprNode::Variable { index, size, .. } => {
                let off = *offsets.get(index)?;
                (0..*size).map(|k| QuadForm::variable(off + k)).collect()
            }
            ExprNode::BinaryOp { op, left, right } => {
                let l = child(left)?;
                let r = child(right)?;
                match op {
                    BinOp::Add => broadcast(&l, &r, |a, b| {
                        let mut o = a.clone();
                        o.add_assign_scaled(b, 1.0);
                        Some(o)
                    })?,
                    BinOp::Sub => broadcast(&l, &r, |a, b| {
                        let mut o = a.clone();
                        o.add_assign_scaled(b, -1.0);
                        Some(o)
                    })?,
                    BinOp::Mul => broadcast(&l, &r, |a, b| a.mul(b))?,
                    BinOp::Div => broadcast(&l, &r, |a, b| {
                        // A variable in the denominator is not polynomial;
                        // a zero denominator is not representable either.
                        //
                        // A non-finite denominator is refused for a subtler
                        // reason: `1.0 / inf` is `0.0`, and scaling by zero
                        // would report `x / inf` as the exact constant 0 while
                        // `evaluate` yields 0 only for finite `x` and NaN for
                        // `inf / inf`. Declining costs the caller its probe
                        // fallback; collapsing would hand back a coefficient
                        // the walk cannot stand behind.
                        if b.degree() != 0 || !b.constant.is_finite() || b.constant == 0.0 {
                            None
                        } else {
                            Some(a.scaled(1.0 / b.constant))
                        }
                    })?,
                    BinOp::Pow => broadcast(&l, &r, |a, e| {
                        if e.degree() != 0 {
                            return None; // variable exponent
                        }
                        let p = e.constant;
                        if a.degree() == 0 {
                            return Some(QuadForm::constant(a.constant.powf(p)));
                        }
                        // Non-constant base: only integer powers 0, 1, 2, and
                        // only when the result stays within degree 2.
                        let n = p as i64;
                        if (p - n as f64).abs() != 0.0 || n < 0 {
                            return None;
                        }
                        match n {
                            0 => Some(QuadForm::constant(1.0)),
                            1 => Some(a.clone()),
                            2 => a.mul(a),
                            _ => None,
                        }
                    })?,
                }
            }
            ExprNode::UnaryOp { op, operand } => {
                let v = child(operand)?;
                match op {
                    UnOp::Neg => v.iter().map(|a| a.scaled(-1.0)).collect(),
                    // `abs` of anything variable-dependent is piecewise, not
                    // polynomial. Treating it as degree 1 is what certified a
                    // false optimum in #739; `max_degree` refuses it too.
                    UnOp::Abs => {
                        let mut out = Vec::with_capacity(v.len());
                        for a in v.iter() {
                            if a.degree() != 0 {
                                return None;
                            }
                            out.push(QuadForm::constant(a.constant.abs()));
                        }
                        out
                    }
                }
            }
            // Every named function is transcendental or piecewise; none is a
            // quadratic form of its argument.
            ExprNode::FunctionCall { .. } => return None,
            ExprNode::Index { base, index } => match self.get(*base) {
                ExprNode::Variable {
                    shape,
                    index: vi,
                    size,
                    ..
                } => {
                    let off = *offsets.get(vi)?;
                    let flat = index_spec_collect_flat(index, shape);
                    let mut out = Vec::with_capacity(flat.len());
                    for f in flat {
                        if f >= *size {
                            return None;
                        }
                        out.push(QuadForm::variable(off + f));
                    }
                    out
                }
                ExprNode::ConstantArray(data, shape) => {
                    let mut out = Vec::new();
                    for f in index_spec_collect_flat(index, shape) {
                        out.push(QuadForm::constant(*data.get(f)?));
                    }
                    out
                }
                ExprNode::Parameter { value, shape, .. } => {
                    let mut out = Vec::new();
                    for f in index_spec_collect_flat(index, shape) {
                        out.push(QuadForm::constant(*value.get(f)?));
                    }
                    out
                }
                // Indexing a compound expression needs shape inference the
                // arena does not carry; `evaluate` returns NaN here.
                _ => return None,
            },
            ExprNode::MatMul { left, right } => {
                let l = child(left)?;
                let r = child(right)?;
                // `evaluate_matmul` contracts the two flat vectors. Equal
                // lengths only: zipping mismatched lengths would silently
                // answer a different model.
                if l.len() != r.len() {
                    return None;
                }
                let mut acc = QuadForm::default();
                for (a, b) in l.iter().zip(r.iter()) {
                    acc.add_assign_scaled(&a.mul(b)?, 1.0);
                }
                vec![acc]
            }
            ExprNode::Sum { operand, axis } => {
                if axis.is_some() {
                    // An axis reduction is array-valued; collapsing it to a
                    // full sum answers a DIFFERENT model (#1160).
                    return None;
                }
                let v = child(operand)?;
                let mut acc = QuadForm::default();
                for a in v.iter() {
                    acc.add_assign_scaled(a, 1.0);
                }
                vec![acc]
            }
            ExprNode::SumOver { terms } => {
                let mut acc = QuadForm::default();
                for t in terms {
                    let v = child(t)?;
                    if v.len() != 1 {
                        return None; // `evaluate` treats each term as scalar
                    }
                    acc.add_assign_scaled(&v[0], 1.0);
                }
                vec![acc]
            }
        };

        Some(std::rc::Rc::new(vals))
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
                    // abs(constant) is a constant, but abs of anything
                    // variable-dependent is piecewise and NOT polynomial:
                    // classifying abs(linear) as degree 1 routed |x| models
                    // onto the LP fast path, whose extractors bake in one
                    // side's slope and certify a false optimum / drop the
                    // constraint (issue #739). FunctionCall(Abs) below already
                    // returns usize::MAX for the same reason.
                    if self.max_degree(*operand) == 0 {
                        0
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
                    | MathFunc::Asinh
                    | MathFunc::Acosh
                    | MathFunc::Atanh
                    | MathFunc::Erf
                    | MathFunc::Log1p
                    | MathFunc::Sigmoid
                    | MathFunc::Softplus
                    | MathFunc::Norm1
                    | MathFunc::NormInf
                    | MathFunc::NormP(_)
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
                    MathFunc::Asinh => a0.asinh(),
                    MathFunc::Acosh => a0.acosh(),
                    MathFunc::Atanh => a0.atanh(),
                    MathFunc::Erf => libm::erf(a0),
                    MathFunc::Log1p => a0.ln_1p(),
                    // Numerically stable logistic: 0.5 + 0.5*tanh(x/2).
                    MathFunc::Sigmoid => 0.5 + 0.5 * (0.5 * a0).tanh(),
                    // Numerically stable softplus: max(x,0) + ln(1 + e^{-|x|}).
                    MathFunc::Softplus => a0.max(0.0) + (-a0.abs()).exp().ln_1p(),
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
                    MathFunc::Prod => self.reduction_values(args, x).iter().product(),
                    MathFunc::Norm1 => self.reduction_values(args, x).iter().map(|t| t.abs()).sum(),
                    MathFunc::Norm2 => self
                        .reduction_values(args, x)
                        .iter()
                        .map(|t| t * t)
                        .sum::<f64>()
                        .sqrt(),
                    MathFunc::NormInf => self
                        .reduction_values(args, x)
                        .iter()
                        .fold(0.0_f64, |m, t| m.max(t.abs())),
                    MathFunc::NormP(p) => {
                        let pf = *p as f64;
                        self.reduction_values(args, x)
                            .iter()
                            .map(|t| t.abs().powf(pf))
                            .sum::<f64>()
                            .powf(1.0 / pf)
                    }
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
            ExprNode::Sum { operand, axis } => {
                if axis.is_some() {
                    // An axis reduction is ARRAY-valued: `sum(A, axis=1)` on a
                    // (2, 3) operand is two row sums, not one sum of six terms.
                    // This evaluator is scalar-valued, so there is no honest
                    // answer -- returning the full sum silently answers a
                    // DIFFERENT model, and the Python LP/QP repr extractors then
                    // emit one collapsed row and certify its optimum (#1160).
                    // NaN is this arena's established "not scalar-representable"
                    // signal (see the Index fall-through above); every extractor
                    // that probes the repr checks for it and falls back to a
                    // per-component path. The arena carries no shape inference,
                    // so a 1-D `axis=0` operand -- which IS a full reduction --
                    // is refused here too; `_relax.scalarize.sum_is_full_reduction`
                    // is the precise predicate, on the Python side where shapes
                    // are known.
                    return f64::NAN;
                }
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
            ExprNode::Index { base, index } => match self.get(*base) {
                ExprNode::Variable { shape, .. } => {
                    let offset = self.var_offset(*base);
                    index_spec_collect_flat(index, shape)
                        .into_iter()
                        .map(|f| x[offset + f])
                        .collect()
                }
                ExprNode::ConstantArray(data, shape) => index_spec_collect_flat(index, shape)
                    .into_iter()
                    .map(|f| data[f])
                    .collect(),
                ExprNode::Parameter { value, shape, .. } => index_spec_collect_flat(index, shape)
                    .into_iter()
                    .map(|f| value[f])
                    .collect(),
                _ => vec![self.evaluate(id, x)],
            },
            _ => vec![self.evaluate(id, x)],
        }
    }

    /// Collect the scalar operands of a reduction function (prod / norm).
    ///
    /// A single array-valued argument (`norm(x)` with vector `x`) is expanded
    /// to its components; multiple scalar arguments are each evaluated.
    fn reduction_values(&self, args: &[ExprId], x: &[f64]) -> Vec<f64> {
        if args.len() == 1 {
            self.collect_array_values(args[0], x)
        } else {
            args.iter().map(|a| self.evaluate(*a, x)).collect()
        }
    }

    /// Sum all elements of an array-valued expression.
    fn evaluate_sum_all(&self, operand: ExprId, x: &[f64]) -> f64 {
        let vals = self.collect_array_values(operand, x);
        vals.iter().sum()
    }
}

/// Convert an IndexSpec to a flat index given a shape (row-major / C-order).
///
/// Returns 0 for index specs that contain a slice — callers that need to
/// enumerate the selected elements should use [`index_spec_collect_flat`]
/// instead.
fn index_spec_to_flat(spec: &IndexSpec, shape: &[usize]) -> usize {
    match spec {
        IndexSpec::Scalar(i) => *i,
        IndexSpec::Tuple(indices) => tuple_to_flat(indices, shape),
        IndexSpec::Multi(elems) => {
            let mut indices: Vec<usize> = Vec::with_capacity(elems.len());
            for elem in elems {
                match elem {
                    IndexElem::Scalar(i) => indices.push(*i),
                    IndexElem::Slice { .. } => return 0,
                }
            }
            tuple_to_flat(&indices, shape)
        }
    }
}

fn tuple_to_flat(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for (&idx, &dim) in indices.iter().rev().zip(shape.iter().rev()) {
        flat += idx * stride;
        stride *= dim;
    }
    flat
}

/// Enumerate the flat indices selected by an [`IndexSpec`] on an array of the
/// given shape (row-major / C-order). Always returns at least one index;
/// scalar specs collapse to a single-element vector. May return an empty
/// vector for slices that select no elements (e.g. `x[5:5]`).
pub fn index_spec_collect_flat(spec: &IndexSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        IndexSpec::Scalar(i) => vec![*i],
        IndexSpec::Tuple(indices) => vec![tuple_to_flat(indices, shape)],
        IndexSpec::Multi(elems) => {
            let mut out: Vec<Vec<usize>> = vec![Vec::new()];
            for (axis, elem) in elems.iter().enumerate() {
                let dim = shape.get(axis).copied().unwrap_or(1);
                let positions: Vec<usize> = match elem {
                    IndexElem::Scalar(i) => vec![*i],
                    IndexElem::Slice { start, stop, step } => {
                        resolve_slice(*start, *stop, *step, dim)
                    }
                };
                let mut next: Vec<Vec<usize>> = Vec::with_capacity(out.len() * positions.len());
                for prefix in &out {
                    for p in &positions {
                        let mut v = prefix.clone();
                        v.push(*p);
                        next.push(v);
                    }
                }
                out = next;
            }
            out.iter().map(|ix| tuple_to_flat(ix, shape)).collect()
        }
    }
}

/// Resolve a Python-style slice `(start, stop, step)` against an axis of the
/// given length and return the selected positions.
///
/// Mirrors the semantics of `slice(start, stop, step).indices(length)` in
/// CPython, including handling of `None`, negative indices, and reversed
/// iteration when `step < 0`.
fn resolve_slice(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    length: usize,
) -> Vec<usize> {
    let length = length as isize;
    let step = step.unwrap_or(1);
    if step == 0 {
        // Rejected at construction time; treat as empty here.
        return Vec::new();
    }

    // Default bounds depend on the direction of iteration.
    let (lower, upper) = if step < 0 {
        (-1_isize, length - 1)
    } else {
        (0_isize, length)
    };

    let mut start = match start {
        None => {
            if step < 0 {
                upper
            } else {
                lower
            }
        }
        Some(mut s) => {
            if s < 0 {
                s += length;
            }
            s.clamp(lower, upper)
        }
    };
    let stop = match stop {
        None => {
            if step < 0 {
                lower
            } else {
                upper
            }
        }
        Some(mut s) => {
            if s < 0 {
                s += length;
            }
            s.clamp(lower, upper)
        }
    };

    let mut out = Vec::new();
    if step > 0 {
        while start < stop {
            out.push(start as usize);
            start += step;
        }
    } else {
        while start > stop {
            out.push(start as usize);
            start += step;
        }
    }
    out
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
                    MathFunc::Asinh => a0.asinh(),
                    MathFunc::Acosh => a0.acosh(),
                    MathFunc::Atanh => a0.atanh(),
                    MathFunc::Erf => libm::erf(a0),
                    MathFunc::Log1p => a0.ln_1p(),
                    // Numerically stable logistic: 0.5 + 0.5*tanh(x/2).
                    MathFunc::Sigmoid => 0.5 + 0.5 * (0.5 * a0).tanh(),
                    // Numerically stable softplus: max(x,0) + ln(1 + e^{-|x|}).
                    MathFunc::Softplus => a0.max(0.0) + (-a0.abs()).exp().ln_1p(),
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
                    MathFunc::Prod => self.reduction_values(args, x).iter().product(),
                    MathFunc::Norm1 => self.reduction_values(args, x).iter().map(|t| t.abs()).sum(),
                    MathFunc::Norm2 => self
                        .reduction_values(args, x)
                        .iter()
                        .map(|t| t * t)
                        .sum::<f64>()
                        .sqrt(),
                    MathFunc::NormInf => self
                        .reduction_values(args, x)
                        .iter()
                        .fold(0.0_f64, |m, t| m.max(t.abs())),
                    MathFunc::NormP(p) => {
                        let pf = *p as f64;
                        self.reduction_values(args, x)
                            .iter()
                            .map(|t| t.abs().powf(pf))
                            .sum::<f64>()
                            .powf(1.0 / pf)
                    }
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
            ExprNode::Sum { operand, axis } => {
                // Array-valued unless it is a full reduction -- see the matching
                // guard in `ExprArena::evaluate` (#1160).
                if axis.is_some() {
                    f64::NAN
                } else {
                    self.evaluate_sum_all(*operand, x)
                }
            }
            ExprNode::SumOver { terms } => terms.iter().map(|t| self.evaluate_node(*t, x)).sum(),
        }
    }

    /// Collect the scalar operands of a reduction (prod / norm): a single
    /// array argument is expanded to its components, multiple scalar arguments
    /// are each evaluated.
    fn reduction_values(&self, args: &[ExprId], x: &[f64]) -> Vec<f64> {
        if args.len() == 1 {
            self.collect_array_values(args[0], x)
        } else {
            args.iter().map(|a| self.evaluate_node(*a, x)).collect()
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
            ExprNode::Index { base, index } => match self.arena.get(*base) {
                ExprNode::Variable {
                    index: var_idx,
                    shape,
                    ..
                } => {
                    let offset = self.variables[*var_idx].offset;
                    index_spec_collect_flat(index, shape)
                        .into_iter()
                        .map(|f| x[offset + f])
                        .collect()
                }
                ExprNode::ConstantArray(data, shape) => index_spec_collect_flat(index, shape)
                    .into_iter()
                    .map(|f| data[f])
                    .collect(),
                ExprNode::Parameter { value, shape, .. } => index_spec_collect_flat(index, shape)
                    .into_iter()
                    .map(|f| value[f])
                    .collect(),
                _ => vec![self.evaluate_node(id, x)],
            },
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

    // ── Symbolic quadratic-form extraction ────────────────────────────────

    /// Evaluate a [`QuadForm`] at `x`, for cross-checking against
    /// [`ExprArena::evaluate`].
    fn qf_eval(q: &QuadForm, x: &[f64]) -> f64 {
        let mut v = q.constant;
        for (i, c) in &q.linear {
            v += c * x[*i];
        }
        for ((i, j), c) in &q.quadratic {
            v += c * x[*i] * x[*j];
        }
        v
    }

    fn var(arena: &mut ExprArena, name: &str, index: usize, size: usize) -> ExprId {
        let shape = if size == 1 { vec![] } else { vec![size] };
        arena.add(ExprNode::Variable {
            name: name.into(),
            index,
            size,
            shape,
        })
    }

    fn bin(arena: &mut ExprArena, op: BinOp, l: ExprId, r: ExprId) -> ExprId {
        arena.add(ExprNode::BinaryOp {
            op,
            left: l,
            right: r,
        })
    }

    #[test]
    fn test_quadratic_form_declines_non_finite_coefficients() {
        // x / inf: `scaled` would collapse this to the constant 0, but the
        // constant term inf/inf is NaN under `evaluate`. The walk must decline
        // rather than hand a coefficient it cannot stand behind to the caller.
        let mut a = ExprArena::new();
        let x = var(&mut a, "x0", 0, 1);
        let inf = a.add(ExprNode::Constant(f64::INFINITY));
        let q = bin(&mut a, BinOp::Div, inf, inf);
        let e = bin(&mut a, BinOp::Add, x, q);
        assert!(
            a.quadratic_form(e, 1_000).is_none(),
            "a non-finite coefficient must be declined, not emitted"
        );

        // The finite twin still extracts, so the guard is not a blanket refusal.
        let mut b = ExprArena::new();
        let x2 = var(&mut b, "x0", 0, 1);
        let two = b.add(ExprNode::Constant(2.0));
        let ok = bin(&mut b, BinOp::Div, x2, two);
        let f = b
            .quadratic_form(ok, 1_000)
            .expect("x/2 is a quadratic form");
        assert_eq!(f.linear.get(&0).copied(), Some(0.5));
    }

    #[test]
    fn test_quadratic_form_coefficients() {
        // 2*x0*x0 + 3*x0*x1 - x1 + 5
        let mut a = ExprArena::new();
        let x0 = var(&mut a, "x0", 0, 1);
        let x1 = var(&mut a, "x1", 1, 1);
        let c2 = a.add(ExprNode::Constant(2.0));
        let c3 = a.add(ExprNode::Constant(3.0));
        let c5 = a.add(ExprNode::Constant(5.0));
        let sq = bin(&mut a, BinOp::Mul, x0, x0);
        let t1 = bin(&mut a, BinOp::Mul, c2, sq);
        let x0x1 = bin(&mut a, BinOp::Mul, x0, x1);
        let t2 = bin(&mut a, BinOp::Mul, c3, x0x1);
        let s1 = bin(&mut a, BinOp::Add, t1, t2);
        let s2 = bin(&mut a, BinOp::Sub, s1, x1);
        let root = bin(&mut a, BinOp::Add, s2, c5);

        let q = a.quadratic_form(root, 1_000).expect("should extract");
        assert_eq!(q.constant, 5.0);
        assert_eq!(q.linear.len(), 1);
        assert_eq!(q.linear[&1], -1.0);
        assert_eq!(q.quadratic.len(), 2);
        assert_eq!(q.quadratic[&(0, 0)], 2.0);
        assert_eq!(q.quadratic[&(0, 1)], 3.0);
        // The (i, j) key carries the FULL cross coefficient, not a half.
        assert!(!q.quadratic.contains_key(&(1, 0)));
    }

    #[test]
    fn test_quadratic_form_matches_evaluate() {
        // Cross-check the symbolic walk against the arena's own evaluator on a
        // mix of forms: powers, negation, division by a constant, nested sums.
        let mut a = ExprArena::new();
        let x0 = var(&mut a, "x0", 0, 1);
        let x1 = var(&mut a, "x1", 1, 1);
        let x2 = var(&mut a, "x2", 2, 1);
        let two = a.add(ExprNode::Constant(2.0));
        let seven = a.add(ExprNode::Constant(7.0));
        let four = a.add(ExprNode::Constant(4.0));

        let p = bin(&mut a, BinOp::Pow, x0, two); // x0^2
        let neg = a.add(ExprNode::UnaryOp {
            op: UnOp::Neg,
            operand: x1,
        });
        let prod = bin(&mut a, BinOp::Mul, x1, x2);
        let div = bin(&mut a, BinOp::Div, prod, four); // x1*x2/4
        let s = bin(&mut a, BinOp::Add, p, neg);
        let s = bin(&mut a, BinOp::Add, s, div);
        let root = bin(&mut a, BinOp::Sub, s, seven);

        let q = a.quadratic_form(root, 1_000).expect("should extract");
        let mut checked = 0;
        for pt in [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.5, 0.25, 4.0],
            [1e3, -1e3, 1e-3],
        ] {
            let want = a.evaluate(root, &pt);
            let got = qf_eval(&q, &pt);
            assert!(
                (want - got).abs() <= 1e-9 * want.abs().max(1.0),
                "point {pt:?}: evaluate={want} quadratic_form={got}"
            );
            checked += 1;
        }
        assert_eq!(checked, 4);
    }

    #[test]
    fn test_quadratic_form_declines_non_quadratic() {
        // Every arm here must DECLINE, not approximate. `is_quadratic` agrees.
        let mut a = ExprArena::new();
        let x0 = var(&mut a, "x0", 0, 1);
        let x1 = var(&mut a, "x1", 1, 1);
        let three = a.add(ExprNode::Constant(3.0));

        let exp = a.add(ExprNode::FunctionCall {
            func: MathFunc::Exp,
            args: vec![x0],
        });
        assert!(
            a.quadratic_form(exp, 1_000).is_none(),
            "exp(x) is not quadratic"
        );

        let ratio = bin(&mut a, BinOp::Div, x0, x1);
        assert!(
            a.quadratic_form(ratio, 1_000).is_none(),
            "x0/x1 is not polynomial"
        );

        // |x| is piecewise, not degree 1 -- treating it as linear certified a
        // false optimum in #739.
        let abs = a.add(ExprNode::UnaryOp {
            op: UnOp::Abs,
            operand: x0,
        });
        assert!(
            a.quadratic_form(abs, 1_000).is_none(),
            "abs(x) is not quadratic"
        );
        assert!(!a.is_quadratic(abs));

        let cube = bin(&mut a, BinOp::Pow, x0, three);
        assert!(
            a.quadratic_form(cube, 1_000).is_none(),
            "x^3 exceeds degree 2"
        );

        // Degree 3 as a product of a square and a variable.
        let sq = bin(&mut a, BinOp::Mul, x0, x0);
        let cubed = bin(&mut a, BinOp::Mul, sq, x1);
        assert!(
            a.quadratic_form(cubed, 1_000).is_none(),
            "x0^2*x1 exceeds degree 2"
        );

        // An axis reduction is array-valued; collapsing it answers a different
        // model (#1160), so it is refused here as it is in `evaluate`.
        let axis_sum = a.add(ExprNode::Sum {
            operand: x0,
            axis: Some(0),
        });
        assert!(
            a.quadratic_form(axis_sum, 1_000).is_none(),
            "axis sum is refused"
        );
    }

    #[test]
    fn test_quadratic_form_deep_chain_does_not_overflow() {
        // A `.nl` objective is a left-nested chain -- 18,499 `+` nodes deep on
        // `unitcommit_200_100_1_mod_8`. A recursive walk blows the stack here.
        let mut a = ExprArena::new();
        let n = 100_000usize;
        let x = var(&mut a, "x", 0, 1);
        let mut acc = a.add(ExprNode::Constant(0.0));
        for k in 0..n {
            let c = a.add(ExprNode::Constant((k % 7) as f64));
            let t = bin(&mut a, BinOp::Mul, c, x);
            acc = bin(&mut a, BinOp::Add, acc, t);
        }
        // A budget of 16 live coefficients proves the walk holds only the
        // frontier: the chain has 100,000 terms but never more than a handful
        // alive at once.
        let q = a
            .quadratic_form(acc, 16)
            .expect("deep chain should extract");
        let want: f64 = (0..n).map(|k| (k % 7) as f64).sum();
        assert_eq!(q.linear[&0], want);
        assert_eq!(q.constant, 0.0);
    }

    #[test]
    fn test_quadratic_form_array_matmul() {
        // x' x for a 3-vector, via MatMul -- the array path.
        let mut a = ExprArena::new();
        let x = var(&mut a, "x", 0, 3);
        let root = a.add(ExprNode::MatMul { left: x, right: x });
        let q = a.quadratic_form(root, 1_000).expect("should extract");
        assert_eq!(q.quadratic.len(), 3);
        for i in 0..3 {
            assert_eq!(q.quadratic[&(i, i)], 1.0);
        }
        let pt = [1.0, -2.0, 3.0];
        assert_eq!(qf_eval(&q, &pt), a.evaluate(root, &pt));
    }

    #[test]
    fn test_quadratic_form_respects_term_budget() {
        // The budget exists so a would-be dense `Q` is never materialized.
        // A 200-variable dense form is 20,100 coefficients.
        let mut a = ExprArena::new();
        let x = var(&mut a, "x", 0, 200);
        let root = a.add(ExprNode::MatMul { left: x, right: x });
        assert!(
            a.quadratic_form(root, 10).is_none(),
            "should decline under budget"
        );
        assert!(
            a.quadratic_form(root, 1_000_000).is_some(),
            "should extract with budget"
        );
    }

    #[test]
    fn test_quadratic_form_coo_is_sorted_and_deterministic() {
        // COO output must be byte-reproducible: iterating the hash maps
        // directly would not be.
        let mut a = ExprArena::new();
        let x = var(&mut a, "x", 0, 40);
        let root = a.add(ExprNode::MatMul { left: x, right: x });
        let q = a.quadratic_form(root, 1_000_000).unwrap();
        let first = q.to_coo();
        for _ in 0..5 {
            let again = a.quadratic_form(root, 1_000_000).unwrap().to_coo();
            assert_eq!(first, again);
        }
        let (qi, qj, _, _, _, _) = &first;
        let mut keys: Vec<(usize, usize)> = qi.iter().copied().zip(qj.iter().copied()).collect();
        let sorted = {
            let mut k = keys.clone();
            k.sort();
            k
        };
        assert_eq!(keys, sorted, "COO entries must be in ascending key order");
        keys.dedup();
        assert_eq!(keys.len(), qi.len(), "no duplicate COO keys");
        // i <= j for every entry.
        assert!(qi.iter().zip(qj.iter()).all(|(i, j)| i <= j));
    }

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
    fn test_abs_of_variable_is_not_linear_or_quadratic() {
        // Issue #739: abs(linear) is piecewise linear, NOT linear. Classifying
        // it as degree 1 sent |x| models down the LP fast path, which baked in
        // one side's slope (false optimal certificate / dropped constraint).
        let mut arena = ExprArena::new();
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let abs_x = arena.add(ExprNode::UnaryOp {
            op: UnOp::Abs,
            operand: x0,
        });
        assert!(!arena.is_linear(abs_x));
        assert!(!arena.is_quadratic(abs_x));

        // abs buried in an otherwise-linear sum still poisons the degree.
        let c3 = arena.add(ExprNode::Constant(3.0));
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: abs_x,
            right: c3,
        });
        assert!(!arena.is_linear(sum));
        assert!(!arena.is_quadratic(sum));
    }

    #[test]
    fn test_abs_of_constant_is_constant_degree() {
        // abs of a genuine constant folds to degree 0: x + |c| must stay linear.
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(-3.0));
        let abs_c = arena.add(ExprNode::UnaryOp {
            op: UnOp::Abs,
            operand: c,
        });
        let x0 = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let sum = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x0,
            right: abs_c,
        });
        assert!(arena.is_linear(sum));
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
    fn test_evaluate_vector_norm_and_prod() {
        // Regression: norm/prod over an array argument must reduce over the
        // vector components, not return NaN (the old single-scalar stub).
        let mut arena = ExprArena::new();
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 2,
            shape: vec![2],
        });
        let pt = [3.0_f64, -4.0_f64];
        let n1 = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Norm1,
            args: vec![x],
        });
        let n2 = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Norm2,
            args: vec![x],
        });
        let ninf = arena.add(ExprNode::FunctionCall {
            func: MathFunc::NormInf,
            args: vec![x],
        });
        let pr = arena.add(ExprNode::FunctionCall {
            func: MathFunc::Prod,
            args: vec![x],
        });
        assert!((arena.evaluate(n1, &pt) - 7.0).abs() < 1e-12);
        assert!((arena.evaluate(n2, &pt) - 5.0).abs() < 1e-12);
        assert!((arena.evaluate(ninf, &pt) - 4.0).abs() < 1e-12);
        assert!((arena.evaluate(pr, &pt) - (-12.0)).abs() < 1e-12);
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
    fn test_axis_reduced_sum_is_not_scalar_representable() {
        // #1160: `sum(A, axis=k)` is ARRAY-valued -- one row per surviving
        // element. This evaluator returns one f64, so the only honest answer is
        // NaN (the arena's "not scalar-representable" signal, which every
        // repr-based extractor checks for). Returning the full sum answered a
        // DIFFERENT model: `sum(A, axis=1) <= b` was extracted as the single row
        // `sum(A) <= b` and its optimum certified.
        let mut arena = ExprArena::new();
        let a = arena.add(ExprNode::Variable {
            name: "A".into(),
            index: 0,
            size: 6,
            shape: vec![2, 3],
        });
        let pt = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        let full = arena.add(ExprNode::Sum {
            operand: a,
            axis: None,
        });
        assert!((arena.evaluate(full, &pt) - 21.0).abs() < 1e-12);

        for axis in [0_usize, 1] {
            let reduced = arena.add(ExprNode::Sum {
                operand: a,
                axis: Some(axis),
            });
            let v = arena.evaluate(reduced, &pt);
            assert!(v.is_nan(), "axis={axis} evaluated to {v}, expected NaN");
        }
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

    // ─────────────────────────────────────────────────────────────
    // Phase 4 CSE / hash-consing (structural interning)
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn test_intern_same_subexpr_returns_same_id() {
        // Building the identical subexpression twice returns the SAME id and
        // does not append a duplicate node.
        let mut arena = ExprArena::new();
        arena.enable_interning();
        let x = arena.intern(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let c2 = arena.intern(ExprNode::Constant(2.0));
        let c2_again = arena.intern(ExprNode::Constant(2.0));
        assert_eq!(c2, c2_again, "identical constants must intern to one id");

        let prod1 = arena.intern(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c2,
            right: x,
        });
        let prod2 = arena.intern(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: c2_again,
            right: x,
        });
        assert_eq!(prod1, prod2, "identical products must intern to one id");
        // Nodes present: x, c2, prod  → exactly 3 (no duplicate constant/product).
        assert_eq!(arena.len(), 3);
    }

    #[test]
    fn test_intern_structurally_different_get_different_ids() {
        let mut arena = ExprArena::new();
        arena.enable_interning();
        let x = arena.intern(ExprNode::Variable {
            name: "x".into(),
            index: 0,
            size: 1,
            shape: vec![],
        });
        let y = arena.intern(ExprNode::Variable {
            name: "y".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        // Different variable indices → different ids.
        assert_ne!(x, y);
        // Different operator over the same operands → different ids.
        let add = arena.intern(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        let mul = arena.intern(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x,
            right: y,
        });
        assert_ne!(add, mul);
        // Different operand ORDER is NOT merged (commutative operands not reordered).
        let add_rev = arena.intern(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: y,
            right: x,
        });
        assert_ne!(add, add_rev, "operand order is not canonicalized");
        // Constants distinguished bit-exactly: 0.0 vs -0.0 must not merge.
        let cp = arena.intern(ExprNode::Constant(0.0));
        let cn = arena.intern(ExprNode::Constant(-0.0));
        assert_ne!(cp, cn, "+0.0 and -0.0 must not intern together");
    }

    #[test]
    fn test_intern_disabled_appends_like_add() {
        let mut arena = ExprArena::new();
        // No enable_interning → intern must behave exactly like add.
        let a = arena.intern(ExprNode::Constant(1.0));
        let b = arena.intern(ExprNode::Constant(1.0));
        assert_ne!(a, b, "with interning off, duplicates are distinct nodes");
        assert_eq!(arena.len(), 2);
        assert!(!arena.interning_enabled());
    }

    #[test]
    fn test_intern_seeds_from_existing_nodes() {
        // A node added BEFORE enabling interning is found by a later intern of an
        // equal node (enable seeds the table from pre-existing nodes).
        let mut arena = ExprArena::new();
        let c = arena.add(ExprNode::Constant(7.0));
        arena.enable_interning();
        let c2 = arena.intern(ExprNode::Constant(7.0));
        assert_eq!(c, c2);
        assert_eq!(arena.len(), 1);
    }

    /// Build a moderately deep expression with heavy structural sharing, once
    /// with interning ON and once with interning OFF, then assert both arenas
    /// evaluate the same objective id to within 1e-12 on random points — the
    /// deduped arena must be evaluation-equivalent to the naive one. Returns the
    /// (deduped_len, naive_len) node counts so the test can also assert dedup
    /// actually removed nodes.
    fn build_shared_expr(intern: bool) -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        // Three variables, added first (identity by index).
        let vars: Vec<ExprId> = (0..3)
            .map(|i| {
                a.add(ExprNode::Variable {
                    name: format!("x{i}"),
                    index: i,
                    size: 1,
                    shape: vec![],
                })
            })
            .collect();
        if intern {
            a.enable_interning();
        }
        // Helper builders that go through intern (a no-op append when disabled).
        let mk = |a: &mut ExprArena, n: ExprNode| a.intern(n);

        // Common subexpression t = x0*x1 + x2, built repeatedly.
        let build_t = |a: &mut ExprArena| -> ExprId {
            let p = mk(
                a,
                ExprNode::BinaryOp {
                    op: BinOp::Mul,
                    left: vars[0],
                    right: vars[1],
                },
            );
            mk(
                a,
                ExprNode::BinaryOp {
                    op: BinOp::Add,
                    left: p,
                    right: vars[2],
                },
            )
        };
        // Build t four separate times; with interning they collapse to one.
        let t1 = build_t(&mut a);
        let t2 = build_t(&mut a);
        let t3 = build_t(&mut a);
        let t4 = build_t(&mut a);

        // f = exp(t1) + exp(t2) + t3*t4  (exp(t1)==exp(t2) structurally)
        let e1 = mk(
            &mut a,
            ExprNode::FunctionCall {
                func: MathFunc::Exp,
                args: vec![t1],
            },
        );
        let e2 = mk(
            &mut a,
            ExprNode::FunctionCall {
                func: MathFunc::Exp,
                args: vec![t2],
            },
        );
        let sum_e = mk(
            &mut a,
            ExprNode::BinaryOp {
                op: BinOp::Add,
                left: e1,
                right: e2,
            },
        );
        let prod_t = mk(
            &mut a,
            ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: t3,
                right: t4,
            },
        );
        let f = mk(
            &mut a,
            ExprNode::BinaryOp {
                op: BinOp::Add,
                left: sum_e,
                right: prod_t,
            },
        );
        (a, f)
    }

    #[test]
    fn test_intern_evaluation_equivalence_random_points() {
        let (deduped, f_dedup) = build_shared_expr(true);
        let (naive, f_naive) = build_shared_expr(false);

        // Dedup must have strictly fewer nodes (structural sharing realized).
        assert!(
            deduped.len() < naive.len(),
            "hash-consing should remove duplicate nodes: deduped={} naive={}",
            deduped.len(),
            naive.len()
        );

        // Deterministic LCG so the test is reproducible (no rand dependency).
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-2, 2].
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
        };
        for _ in 0..200 {
            let x = [next(), next(), next()];
            let vd = deduped.evaluate(f_dedup, &x);
            let vn = naive.evaluate(f_naive, &x);
            assert!(
                (vd - vn).abs() <= 1e-12 * (1.0 + vn.abs()),
                "deduped {vd} != naive {vn} at x={x:?}"
            );
        }
    }
}

/// An exact sparse quadratic form over an [`ExprArena`]'s flat variable space:
///
/// ```text
/// constant + sum_i linear[i] * x_i + sum_{i <= j} quadratic[(i, j)] * x_i * x_j
/// ```
///
/// `quadratic` is keyed by an ordered pair, so `(i, i)` carries the full
/// coefficient of `x_i^2` and `(i, j)` with `i < j` the full coefficient of the
/// cross term -- NOT the symmetric-matrix halves. A caller building a `Q` with
/// the `0.5 x' Q x` convention must double the off-diagonals; one building the
/// COO triplet that `set_quadratic_objective` consumes can use these directly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuadForm {
    /// The constant term.
    pub constant: f64,
    /// Linear coefficients, keyed by flat variable index.
    pub linear: std::collections::HashMap<usize, f64>,
    /// Quadratic coefficients, keyed by `(i, j)` with `i <= j`.
    pub quadratic: std::collections::HashMap<(usize, usize), f64>,
    /// Structural degree (0, 1 or 2), carried explicitly.
    ///
    /// It is tracked incrementally rather than read off the maps for two
    /// reasons. It keeps [`Self::degree`] O(1): deriving it from map emptiness
    /// would require pruning cancelled coefficients after every operation,
    /// which re-scans the whole accumulator per term and restores the
    /// quadratic cost this module exists to remove. And it keeps the answer
    /// consistent with [`ExprArena::max_degree`], which likewise counts a term
    /// whose coefficient happens to cancel to zero.
    deg: usize,
}

impl QuadForm {
    /// A degree-0 form holding a single constant.
    pub fn constant(v: f64) -> Self {
        Self {
            constant: v,
            ..Default::default()
        }
    }

    /// The degree-1 form `x_i`.
    pub fn variable(i: usize) -> Self {
        let mut linear = std::collections::HashMap::with_capacity(1);
        linear.insert(i, 1.0);
        Self {
            constant: 0.0,
            linear,
            quadratic: std::collections::HashMap::new(),
            deg: 1,
        }
    }

    /// Structural polynomial degree: 0, 1 or 2.
    pub fn degree(&self) -> usize {
        self.deg
    }

    /// Number of stored linear + quadratic coefficients.
    pub fn n_terms(&self) -> usize {
        self.linear.len() + self.quadratic.len()
    }

    /// True when every coefficient of this form is finite.
    pub fn is_finite(&self) -> bool {
        self.constant.is_finite()
            && self.linear.values().all(|v| v.is_finite())
            && self.quadratic.values().all(|v| v.is_finite())
    }

    /// This form scaled by a constant.
    ///
    /// Scaling by exactly zero collapses the degree: `0 * x` is the constant 0,
    /// and keeping it at degree 1 would make an otherwise-representable
    /// product (`(0 * x) * y`) decline for no reason.
    pub fn scaled(&self, s: f64) -> Self {
        // `0 * x` is the constant 0 -- but only when the coefficients of `x`
        // are finite, since `evaluate` gives `0 * inf = NaN`. Short-circuiting
        // unconditionally would erase a non-finite before the walk's output
        // check can decline it, so a non-finite operand falls through to the
        // ordinary multiply and propagates the NaN the caller must see.
        if s == 0.0 && self.is_finite() {
            return Self::constant(0.0);
        }
        Self {
            constant: self.constant * s,
            linear: self.linear.iter().map(|(k, v)| (*k, v * s)).collect(),
            quadratic: self.quadratic.iter().map(|(k, v)| (*k, v * s)).collect(),
            deg: self.deg,
        }
    }

    /// `self += other * s`, in place.
    pub fn add_assign_scaled(&mut self, other: &Self, s: f64) {
        self.constant += other.constant * s;
        if s == 0.0 {
            return;
        }
        for (k, v) in &other.linear {
            *self.linear.entry(*k).or_insert(0.0) += v * s;
        }
        for (k, v) in &other.quadratic {
            *self.quadratic.entry(*k).or_insert(0.0) += v * s;
        }
        self.deg = self.deg.max(other.deg);
    }

    /// The product `self * other`, or `None` when it would exceed degree 2.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.deg + other.deg > 2 {
            return None;
        }
        let mut out = Self::constant(self.constant * other.constant);
        out.deg = self.deg + other.deg;
        // constant x (linear, quadratic), both directions.
        for (i, a) in &self.linear {
            *out.linear.entry(*i).or_insert(0.0) += a * other.constant;
        }
        for (i, b) in &other.linear {
            *out.linear.entry(*i).or_insert(0.0) += b * self.constant;
        }
        for (k, a) in &self.quadratic {
            *out.quadratic.entry(*k).or_insert(0.0) += a * other.constant;
        }
        for (k, b) in &other.quadratic {
            *out.quadratic.entry(*k).or_insert(0.0) += b * self.constant;
        }
        // linear x linear -> quadratic. Degree 3 and 4 products are excluded
        // by the guard above, so there is nothing else to form.
        for (i, a) in &self.linear {
            for (j, b) in &other.linear {
                let key = if i <= j { (*i, *j) } else { (*j, *i) };
                *out.quadratic.entry(key).or_insert(0.0) += a * b;
            }
        }
        Some(out)
    }

    /// The form as sparse COO plus linear and constant parts, ready for the
    /// Python boundary: `(qi, qj, qd, ci, cd, constant)`.
    ///
    /// Entries are emitted in ascending key order so the output is
    /// deterministic and byte-reproducible across runs, which iterating the
    /// hash maps directly would not be.
    pub fn to_coo(&self) -> QuadFormCoo {
        let mut q: Vec<((usize, usize), f64)> =
            self.quadratic.iter().map(|(k, v)| (*k, *v)).collect();
        q.sort_by_key(|(k, _)| *k);
        let mut l: Vec<(usize, f64)> = self.linear.iter().map(|(k, v)| (*k, *v)).collect();
        l.sort_by_key(|(k, _)| *k);
        (
            q.iter().map(|((i, _), _)| *i).collect(),
            q.iter().map(|((_, j), _)| *j).collect(),
            q.iter().map(|(_, v)| *v).collect(),
            l.iter().map(|(i, _)| *i).collect(),
            l.iter().map(|(_, v)| *v).collect(),
            self.constant,
        )
    }
}

/// The COO payload of [`QuadForm::to_coo`]: `(qi, qj, qd, ci, cd, constant)`,
/// i.e. the quadratic row/col/value triple, the linear index/value pair, and
/// the constant term. Named so the tuple stays one thing at every call site
/// rather than six positional vectors clippy has to read as a type.
pub type QuadFormCoo = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>, Vec<f64>, f64);

/// Apply a binary operation elementwise over two flat polynomial arrays,
/// broadcasting a length-1 operand against a longer one.
///
/// Any other length pairing is refused rather than zipped: truncating to the
/// shorter operand would silently answer a different model.
fn broadcast<F>(l: &[QuadForm], r: &[QuadForm], f: F) -> Option<Vec<QuadForm>>
where
    F: Fn(&QuadForm, &QuadForm) -> Option<QuadForm>,
{
    let n = match (l.len(), r.len()) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        _ => return None,
    };
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let a = if l.len() == 1 { &l[0] } else { &l[k] };
        let b = if r.len() == 1 { &r[0] } else { &r[k] };
        out.push(f(a, b)?);
    }
    Some(out)
}
