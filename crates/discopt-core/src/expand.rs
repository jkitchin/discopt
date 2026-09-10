//! Scalar expansion of array-valued arena nodes.
//!
//! A discopt model may hold **array-structured** expressions: `A @ x <= b` is
//! one [`ConstraintRepr`] and many rows; `exp(x) + y <= b` on a shaped variable
//! is one node and many rows. Measured, that vectorised form reaches solve-ready
//! 36x faster and 28x lighter than the equivalent per-element model
//! (`docs/dev/performance-plan.md` §41), so it is the shape a 100k-row model has
//! to be written in.
//!
//! Both consumers of a model are nevertheless **scalar**: POUNCE's `NlExpr` has
//! no vector node, and `.nl` is a row-oriented file format. Each therefore needs
//! the same fan-out — one array node to N scalar rows — and it was being done
//! twice in Python (`_nl_expr_compiler` for the tape, `export/_arrays.py` for
//! the writers) while `tape_program` simply refused array bodies outright. This
//! module does it once, in Rust, over the arena that `model_to_repr` has already
//! built.
//!
//! # What comes out
//!
//! A [`ScalarProgram`]: a flat instruction list in the same opcode encoding
//! `tape_program` uses, plus one root per output row. A caller lowers it with a
//! single forward scan — every instruction's operands have lower indices than
//! itself, because instructions are only ever appended after their operands.
//!
//! # Refusal, not approximation
//!
//! Every path that cannot be expanded exactly returns [`ExpandError`] naming the
//! node. Nothing is guessed and nothing is skipped: a wrong fan-out is a wrong
//! model — the wrong number of rows, or the right number carrying the wrong
//! variables — and unlike a refusal it would not announce itself. Callers fall
//! back to the existing Python path, which is unchanged.

use crate::expr::{
    BinOp, ExprArena, ExprId, ExprNode, IndexElem, IndexSpec, MathFunc, ModelRepr, UnOp,
};

// Opcodes. These MUST track `crates/discopt-python/src/expr_bindings.rs`
// (`tape_program`, `math_func_code`) and `python/discopt/_arena_tape.py`, which
// share this encoding.
/// Node the encoding does not cover.
pub const OP_UNSUPPORTED: i32 = 0;
/// Literal constant; the value is in `k`.
pub const OP_CONST: i32 = 1;
/// Variable reference; the **flat slot** is in `k`.
pub const OP_VAR: i32 = 2;
/// `a + b`.
pub const OP_ADD: i32 = 3;
/// `a - b`.
pub const OP_SUB: i32 = 4;
/// `a * b`.
pub const OP_MUL: i32 = 5;
/// `a / b`.
pub const OP_DIV: i32 = 6;
/// `a ** b`.
pub const OP_POW: i32 = 7;
/// `-a`.
pub const OP_NEG: i32 = 8;
/// `|a|`.
pub const OP_ABS: i32 = 9;
/// n-ary sum over the instruction's `args` slice.
pub const OP_SUMOVER: i32 = 10;
/// Base for [`MathFunc`] opcodes; the code is added to this.
pub const OP_FUNC_BASE: i32 = 20;

/// A node or shape this expansion does not cover. Never silently skipped.
#[derive(Debug, Clone)]
pub struct ExpandError(pub String);

impl std::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ExpandError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ExpandError> {
    Err(ExpandError(msg.into()))
}

/// A flat, fully scalar instruction list plus one root per output row.
#[derive(Debug, Clone, Default)]
pub struct ScalarProgram {
    /// Opcode per instruction.
    pub op: Vec<i32>,
    /// First operand instruction index, or -1.
    pub a: Vec<i64>,
    /// Second operand instruction index, or -1.
    pub b: Vec<i64>,
    /// Literal payload: value for `OP_CONST`, flat variable slot for `OP_VAR`.
    pub k: Vec<f64>,
    /// Operands of n-ary instructions, sliced by `args_ptr`.
    pub args_flat: Vec<i64>,
    /// `args_flat[args_ptr[i]..args_ptr[i + 1]]` are instruction `i`'s operands.
    /// Length is `op.len() + 1`.
    pub args_ptr: Vec<i64>,
    /// Instruction index of the objective's root.
    pub objective_root: i64,
    /// Instruction index of each constraint ROW's root, constraints in order.
    pub row_roots: Vec<i64>,
    /// How many rows each source constraint expanded to, same order as
    /// [`ModelRepr::constraints`]. Sums to `row_roots.len()`.
    pub rows_per_constraint: Vec<usize>,
}

/// Numpy broadcast of two shapes, or `None` when they do not broadcast.
fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for axis in 0..rank {
        // Align right, as numpy does: a missing leading axis is an implicit 1.
        let da = if axis + a.len() >= rank {
            a[axis + a.len() - rank]
        } else {
            1
        };
        let db = if axis + b.len() >= rank {
            b[axis + b.len() - rank]
        } else {
            1
        };
        out[axis] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

fn numel(shape: &[usize]) -> usize {
    shape
        .iter()
        .product::<usize>()
        .max(if shape.is_empty() { 1 } else { 0 })
}

/// Row-major flat index of `idx` within `shape`.
fn flat_of(idx: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    for (axis, &i) in idx.iter().enumerate() {
        let mut stride = 1usize;
        for &d in &shape[axis + 1..] {
            stride *= d;
        }
        flat += i * stride;
    }
    flat
}

/// Row-major multi-index of flat position `pos` within `shape`.
fn idx_of(mut pos: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let d = shape[axis].max(1);
        out[axis] = pos % d;
        pos /= d;
    }
    out
}

/// Flat index into `operand_shape` for output multi-index `out_idx`, applying
/// numpy broadcasting: a size-1 operand axis is read at 0 whatever the output
/// index is, and missing leading axes are ignored.
fn broadcast_flat(out_idx: &[usize], operand_shape: &[usize]) -> usize {
    let r = operand_shape.len();
    let lead = out_idx.len() - r;
    let mut idx = vec![0usize; r];
    for axis in 0..r {
        idx[axis] = if operand_shape[axis] == 1 {
            0
        } else {
            out_idx[lead + axis]
        };
    }
    flat_of(&idx, operand_shape)
}

/// Resolve a slice element against an axis length, matching Python semantics.
fn slice_indices(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    len: usize,
) -> Result<Vec<usize>, ExpandError> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return err("slice step must be non-zero");
    }
    if step < 0 {
        // Negative strides are legal Python but the modelling layer never emits
        // them; refusing beats an off-by-one nobody would notice.
        return err("negative slice step is not expanded");
    }
    let n = len as isize;
    let norm = |v: isize| -> isize {
        if v < 0 {
            (v + n).max(0)
        } else {
            v.min(n)
        }
    };
    let s = start.map(norm).unwrap_or(0);
    let e = stop.map(norm).unwrap_or(n);
    let mut out = Vec::new();
    let mut i = s;
    while i < e {
        out.push(i as usize);
        i += step;
    }
    Ok(out)
}

/// The per-axis index lists an [`IndexSpec`] selects from `base_shape`, and
/// whether each axis is dropped (a scalar index) or kept (a slice).
fn index_axes(
    spec: &IndexSpec,
    base_shape: &[usize],
) -> Result<Vec<(Vec<usize>, bool)>, ExpandError> {
    let elems: Vec<IndexElem> = match spec {
        IndexSpec::Scalar(i) => vec![IndexElem::Scalar(*i)],
        IndexSpec::Tuple(v) => v.iter().map(|i| IndexElem::Scalar(*i)).collect(),
        IndexSpec::Multi(v) => v.clone(),
    };
    if elems.len() > base_shape.len() {
        return err(format!(
            "index of arity {} into a shape of rank {}",
            elems.len(),
            base_shape.len()
        ));
    }
    let mut axes: Vec<(Vec<usize>, bool)> = Vec::with_capacity(base_shape.len());
    for (axis, dim) in base_shape.iter().enumerate() {
        match elems.get(axis) {
            Some(IndexElem::Scalar(i)) => {
                if *i >= *dim {
                    return err(format!(
                        "index {i} out of range for axis {axis} of size {dim}"
                    ));
                }
                axes.push((vec![*i], true));
            }
            Some(IndexElem::Slice { start, stop, step }) => {
                axes.push((slice_indices(*start, *stop, *step, *dim)?, false));
            }
            // Unindexed trailing axes are kept whole, as numpy does.
            None => axes.push(((0..*dim).collect(), false)),
        }
    }
    Ok(axes)
}

/// Static shape of every arena node, computed in one forward pass.
fn shapes_of(arena: &ExprArena) -> Result<Vec<Vec<usize>>, ExpandError> {
    let n = arena.len();
    let mut shapes: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let s: Vec<usize> = match arena.get(ExprId(i)) {
            ExprNode::Constant(_) => Vec::new(),
            ExprNode::ConstantArray(_, shape) => shape.clone(),
            ExprNode::Variable { shape, .. } => shape.clone(),
            ExprNode::Parameter { shape, .. } => shape.clone(),
            ExprNode::BinaryOp { left, right, .. } => broadcast(&shapes[left.0], &shapes[right.0])
                .ok_or_else(|| {
                    ExpandError(format!(
                        "operand shapes {:?} and {:?} do not broadcast",
                        shapes[left.0], shapes[right.0]
                    ))
                })?,
            ExprNode::UnaryOp { operand, .. } => shapes[operand.0].clone(),
            ExprNode::FunctionCall { func, args } => match func {
                // Reductions, NOT element-wise: folding them element-wise would
                // turn `norm2(x)` on a 3-vector into three rows of `norm2(x[i])`,
                // which is a different model (see `export/_arrays.py`).
                MathFunc::Prod | MathFunc::Norm2 => Vec::new(),
                _ => {
                    let mut acc: Vec<usize> = Vec::new();
                    for arg in args {
                        acc = broadcast(&acc, &shapes[arg.0]).ok_or_else(|| {
                            ExpandError("function argument shapes do not broadcast".into())
                        })?;
                    }
                    acc
                }
            },
            ExprNode::Index { base, index } => {
                let axes = index_axes(index, &shapes[base.0])?;
                axes.iter()
                    .filter(|(_, dropped)| !dropped)
                    .map(|(sel, _)| sel.len())
                    .collect()
            }
            ExprNode::MatMul { left, right } => matmul_shape(&shapes[left.0], &shapes[right.0])?,
            ExprNode::Sum { operand, axis } => match axis {
                None => Vec::new(),
                Some(ax) => {
                    let s = &shapes[operand.0];
                    if *ax >= s.len() {
                        return err(format!("sum axis {ax} out of range for shape {s:?}"));
                    }
                    s.iter()
                        .enumerate()
                        .filter(|(j, _)| j != ax)
                        .map(|(_, d)| *d)
                        .collect()
                }
            },
            ExprNode::SumOver { terms } => {
                let mut acc: Vec<usize> = Vec::new();
                for t in terms {
                    acc = broadcast(&acc, &shapes[t.0])
                        .ok_or_else(|| ExpandError("sum term shapes do not broadcast".into()))?;
                }
                acc
            }
        };
        shapes[i] = s;
    }
    Ok(shapes)
}

/// Numpy `@` result shape for the four rank combinations discopt emits.
fn matmul_shape(l: &[usize], r: &[usize]) -> Result<Vec<usize>, ExpandError> {
    match (l.len(), r.len()) {
        (1, 1) if l[0] == r[0] => Ok(Vec::new()),
        (2, 1) if l[1] == r[0] => Ok(vec![l[0]]),
        (1, 2) if l[0] == r[0] => Ok(vec![r[1]]),
        (2, 2) if l[1] == r[0] => Ok(vec![l[0], r[1]]),
        _ => err(format!("matmul of shapes {l:?} and {r:?}")),
    }
}

/// [`MathFunc`] opcode offset. Must match `expr_bindings.rs::math_func_code`.
fn func_code(f: MathFunc) -> Option<i32> {
    Some(match f {
        MathFunc::Exp => 0,
        MathFunc::Log => 1,
        MathFunc::Log2 => 2,
        MathFunc::Log10 => 3,
        MathFunc::Sqrt => 4,
        MathFunc::Sin => 5,
        MathFunc::Cos => 6,
        MathFunc::Tan => 7,
        MathFunc::Atan => 8,
        MathFunc::Sinh => 9,
        MathFunc::Cosh => 10,
        MathFunc::Asin => 11,
        _ => return None,
    })
}

/// Append-only instruction buffer. Operands always have lower indices than the
/// instruction referencing them, because nothing is emitted before its operands.
struct Emitter {
    op: Vec<i32>,
    a: Vec<i64>,
    b: Vec<i64>,
    k: Vec<f64>,
    args_flat: Vec<i64>,
    args_ptr: Vec<i64>,
}

impl Emitter {
    fn new() -> Self {
        Emitter {
            op: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            k: Vec::new(),
            args_flat: Vec::new(),
            args_ptr: Vec::new(),
        }
    }

    fn push(&mut self, op: i32, a: i64, b: i64, k: f64) -> i64 {
        self.args_ptr.push(self.args_flat.len() as i64);
        self.op.push(op);
        self.a.push(a);
        self.b.push(b);
        self.k.push(k);
        (self.op.len() - 1) as i64
    }

    fn push_nary(&mut self, op: i32, args: &[i64]) -> i64 {
        self.args_ptr.push(self.args_flat.len() as i64);
        self.args_flat.extend_from_slice(args);
        self.op.push(op);
        self.a.push(-1);
        self.b.push(-1);
        self.k.push(0.0);
        (self.op.len() - 1) as i64
    }

    /// A sum of `parts`, matching `_nl_expr_compiler`'s degenerate-arity rules:
    /// no terms is the constant 0, one term is that term.
    fn sum(&mut self, parts: &[i64]) -> i64 {
        match parts.len() {
            0 => self.push(OP_CONST, -1, -1, 0.0),
            1 => parts[0],
            _ => self.push_nary(OP_SUMOVER, parts),
        }
    }
}

/// Expand `repr`'s objective and constraints into a fully scalar program.
///
/// Returns one root per constraint ROW (an array-valued body contributes
/// several), in constraint order, alongside the per-constraint row counts a
/// caller needs to map rows back to the `Constraint` they came from — duals,
/// row maps and feasibility reports all depend on that attribution.
pub fn expand(repr: &ModelRepr) -> Result<ScalarProgram, ExpandError> {
    let arena = &repr.arena;
    let shapes = shapes_of(arena)?;

    // Flat variable offsets, matching `ExprArena::add_variable`.
    let offsets: Vec<usize> = repr.variables.iter().map(|v| v.offset).collect();

    let mut em = Emitter::new();
    // `slots[i]` holds one instruction index per element of node `i`, filled in
    // arena order. Empty until the node is expanded; a node is expanded at most
    // once however many parents use it, which is what keeps a shared
    // subexpression shared in the emitted program too.
    let mut slots: Vec<Vec<i64>> = vec![Vec::new(); arena.len()];
    let mut done = vec![false; arena.len()];

    for i in 0..arena.len() {
        expand_node(arena, i, &shapes, &offsets, &mut em, &mut slots, &mut done)?;
    }

    let obj_slots = &slots[repr.objective.0];
    if obj_slots.len() != 1 {
        return err(format!(
            "objective expands to {} scalar expressions; an objective must be scalar",
            obj_slots.len()
        ));
    }
    let objective_root = obj_slots[0];

    let mut row_roots = Vec::new();
    let mut rows_per_constraint = Vec::with_capacity(repr.constraints.len());
    for c in &repr.constraints {
        let rows = &slots[c.body.0];
        rows_per_constraint.push(rows.len());
        row_roots.extend_from_slice(rows);
    }

    let mut args_ptr = em.args_ptr;
    args_ptr.push(em.args_flat.len() as i64);

    Ok(ScalarProgram {
        op: em.op,
        a: em.a,
        b: em.b,
        k: em.k,
        args_flat: em.args_flat,
        args_ptr,
        objective_root,
        row_roots,
        rows_per_constraint,
    })
}

#[allow(clippy::too_many_arguments)]
fn expand_node(
    arena: &ExprArena,
    i: usize,
    shapes: &[Vec<usize>],
    offsets: &[usize],
    em: &mut Emitter,
    slots: &mut Vec<Vec<i64>>,
    done: &mut Vec<bool>,
) -> Result<(), ExpandError> {
    if done[i] {
        return Ok(());
    }
    done[i] = true;
    let shape = &shapes[i];
    let count = numel(shape);
    let mut out: Vec<i64> = Vec::with_capacity(count);

    match arena.get(ExprId(i)) {
        ExprNode::Constant(v) => out.push(em.push(OP_CONST, -1, -1, *v)),
        ExprNode::ConstantArray(data, _) => {
            for v in data.iter().take(count) {
                out.push(em.push(OP_CONST, -1, -1, *v));
            }
        }
        ExprNode::Parameter { value, .. } => {
            // A parameter is fixed for the solve, so it lowers to a constant --
            // the same choice the tape and every writer make.
            for v in value.iter().take(count) {
                out.push(em.push(OP_CONST, -1, -1, *v));
            }
        }
        ExprNode::Variable { index, size, .. } => {
            let base = offsets[*index];
            for e in 0..(*size).min(count.max(1)) {
                out.push(em.push(OP_VAR, -1, -1, (base + e) as f64));
            }
            if out.len() != count {
                return err(format!(
                    "variable block {index} has {} elements but shape {shape:?}",
                    out.len()
                ));
            }
        }
        ExprNode::Index { base, index } => {
            let bshape = &shapes[base.0];
            let axes = index_axes(index, bshape)?;
            // Cartesian product of the per-axis selections, row-major over the
            // axes the index KEEPS -- exactly numpy's result ordering.
            let kept: Vec<usize> = axes
                .iter()
                .enumerate()
                .filter(|(_, (_, d))| !d)
                .map(|(j, _)| j)
                .collect();
            let kept_shape: Vec<usize> = kept.iter().map(|&j| axes[j].0.len()).collect();
            for pos in 0..numel(&kept_shape) {
                let kidx = idx_of(pos, &kept_shape);
                let mut full = vec![0usize; bshape.len()];
                for (axis, (sel, dropped)) in axes.iter().enumerate() {
                    full[axis] = if *dropped {
                        sel[0]
                    } else {
                        let which = kept.iter().position(|&j| j == axis).unwrap();
                        sel[kidx[which]]
                    };
                }
                out.push(slots[base.0][flat_of(&full, bshape)]);
            }
        }
        ExprNode::BinaryOp { op, left, right } => {
            let code = match op {
                BinOp::Add => OP_ADD,
                BinOp::Sub => OP_SUB,
                BinOp::Mul => OP_MUL,
                BinOp::Div => OP_DIV,
                BinOp::Pow => OP_POW,
            };
            for pos in 0..count {
                let idx = idx_of(pos, shape);
                let la = slots[left.0][broadcast_flat(&idx, &shapes[left.0])];
                let rb = slots[right.0][broadcast_flat(&idx, &shapes[right.0])];
                out.push(em.push(code, la, rb, 0.0));
            }
        }
        ExprNode::UnaryOp { op, operand } => {
            let code = match op {
                UnOp::Neg => OP_NEG,
                UnOp::Abs => OP_ABS,
            };
            for pos in 0..count {
                out.push(em.push(code, slots[operand.0][pos], -1, 0.0));
            }
        }
        ExprNode::FunctionCall { func, args } => match func {
            MathFunc::Prod => {
                let terms = &slots[args[0].0];
                if terms.is_empty() {
                    return err("prod over an empty array has no scalar form");
                }
                let mut acc = terms[0];
                for t in &terms[1..] {
                    acc = em.push(OP_MUL, acc, *t, 0.0);
                }
                out.push(acc);
            }
            MathFunc::Norm2 => {
                let terms: Vec<i64> = slots[args[0].0].clone();
                if terms.is_empty() {
                    return err("norm2 over an empty array has no scalar form");
                }
                let squares: Vec<i64> =
                    terms.iter().map(|t| em.push(OP_MUL, *t, *t, 0.0)).collect();
                let total = em.sum(&squares);
                let sqrt = OP_FUNC_BASE + func_code(MathFunc::Sqrt).unwrap();
                out.push(em.push(sqrt, total, -1, 0.0));
            }
            f => {
                if args.len() != 1 {
                    return err(format!("{f:?} with {} arguments", args.len()));
                }
                let code = match func_code(*f) {
                    Some(c) => OP_FUNC_BASE + c,
                    None => return err(format!("no opcode for {f:?}")),
                };
                for pos in 0..count {
                    let idx = idx_of(pos, shape);
                    let a = slots[args[0].0][broadcast_flat(&idx, &shapes[args[0].0])];
                    out.push(em.push(code, a, -1, 0.0));
                }
            }
        },
        ExprNode::SumOver { terms } => {
            for pos in 0..count {
                let idx = idx_of(pos, shape);
                let parts: Vec<i64> = terms
                    .iter()
                    .map(|t| slots[t.0][broadcast_flat(&idx, &shapes[t.0])])
                    .collect();
                out.push(em.sum(&parts));
            }
        }
        ExprNode::Sum { operand, axis } => {
            let os = &shapes[operand.0];
            match axis {
                None => {
                    let parts: Vec<i64> = slots[operand.0].clone();
                    out.push(em.sum(&parts));
                }
                Some(ax) => {
                    for pos in 0..count {
                        let oidx = idx_of(pos, shape);
                        let mut parts = Vec::with_capacity(os[*ax]);
                        for j in 0..os[*ax] {
                            let mut full = Vec::with_capacity(os.len());
                            let mut it = oidx.iter();
                            for (axis_i, _) in os.iter().enumerate() {
                                full.push(if axis_i == *ax {
                                    j
                                } else {
                                    *it.next().unwrap()
                                });
                            }
                            parts.push(slots[operand.0][flat_of(&full, os)]);
                        }
                        out.push(em.sum(&parts));
                    }
                }
            }
        }
        ExprNode::MatMul { left, right } => {
            let (ls, rs) = (&shapes[left.0], &shapes[right.0]);
            let inner = if ls.len() == 1 { ls[0] } else { ls[1] };
            let (rows, cols) = match (ls.len(), rs.len()) {
                (1, 1) => (1usize, 1usize),
                (2, 1) => (ls[0], 1),
                (1, 2) => (1, rs[1]),
                (2, 2) => (ls[0], rs[1]),
                _ => return err(format!("matmul of shapes {ls:?} and {rs:?}")),
            };
            for r in 0..rows {
                for c in 0..cols {
                    let mut parts = Vec::with_capacity(inner);
                    for p in 0..inner {
                        let lf = if ls.len() == 1 {
                            p
                        } else {
                            flat_of(&[r, p], ls)
                        };
                        let rf = if rs.len() == 1 {
                            p
                        } else {
                            flat_of(&[p, c], rs)
                        };
                        parts.push(em.push(OP_MUL, slots[left.0][lf], slots[right.0][rf], 0.0));
                    }
                    out.push(em.sum(&parts));
                }
            }
        }
    }

    if out.len() != count {
        return err(format!(
            "node {i} has shape {shape:?} ({count} elements) but expanded to {}",
            out.len()
        ));
    }
    slots[i] = out;
    Ok(())
}
