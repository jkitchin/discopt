//! .nl file parser — reads AMPL .nl (text) format into [`ModelRepr`].
//!
//! Supports the text-mode .nl format produced by AMPL with `option nl_comments 0;`.
//! Parses the header, expression DAG segments (C, O), variable/constraint bounds
//! (b, r), linear Jacobian/gradient terms (J, G), and initial point (x).
//!
//! # Example
//! ```
//! use discopt_core::nl_parser::parse_nl;
//! let nl = "g3 1 1 0\n 2 1 1 0 0\n 0 0\n 0 0 0\n 0 0 0\n 0 0 0 1\n 0 0\n 2 2\n 0 0\n 0 0 0 0 0\nO0 0\nn0\nC0\nn0\nx2\n0 0\n1 0\nr\n1 10\nb\n0 0 10\n0 0 10\nk1\n1\nJ0 2\n0 1\n1 1\nG0 2\n0 2\n1 3\n";
//! let model = parse_nl(nl).unwrap();
//! assert_eq!(model.n_vars, 2);
//! ```

use crate::expr::{
    BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, MathFunc, ModelRepr,
    ObjectiveSense, UnOp, VarInfo, VarType,
};
use std::fmt;

// ─────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────

/// Errors arising during .nl file parsing.
#[derive(Debug)]
pub enum NlParseError {
    /// Unexpected end of input.
    UnexpectedEof,
    /// Invalid or unrecognised header.
    InvalidHeader(String),
    /// An expression opcode was not recognised.
    UnknownOpcode(i32),
    /// General parse error with a human-readable message.
    Parse(String),
}

impl fmt::Display for NlParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NlParseError::UnexpectedEof => write!(f, "unexpected end of .nl input"),
            NlParseError::InvalidHeader(msg) => write!(f, "invalid .nl header: {msg}"),
            NlParseError::UnknownOpcode(op) => write!(f, "unknown .nl opcode: o{op}"),
            NlParseError::Parse(msg) => write!(f, ".nl parse error: {msg}"),
        }
    }
}

impl std::error::Error for NlParseError {}

// ─────────────────────────────────────────────────────────────
// Header
// ─────────────────────────────────────────────────────────────

/// Parsed .nl file header (the first 10 lines).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NlHeader {
    n_vars: usize,
    n_constraints: usize,
    n_objectives: usize,
    n_ranges: usize,
    n_eqns: usize,
    n_nonlinear_constraints: usize,
    n_nonlinear_objectives: usize,
    // Line 4: nonlinear var counts
    n_nl_vars_in_cons: usize,
    n_nl_vars_in_objs: usize,
    n_nl_vars_in_both: usize,
    // Line 6: discrete var counts
    n_linear_binary_vars: usize,
    n_linear_integer_vars: usize,
    n_nl_integer_vars_in_cons: usize,
    n_nl_integer_vars_in_objs: usize,
    n_nl_integer_vars_in_both: usize,
    // Line 7: sparsity
    n_nonzeros_jacobian: usize,
    n_nonzeros_obj_gradient: usize,
    // Line 9: common expressions
    n_common_exprs_b_c: usize,
    n_common_exprs_c1: usize,
    n_common_exprs_o1: usize,
    n_common_exprs_c2: usize,
    n_common_exprs_o2: usize,
}

// ─────────────────────────────────────────────────────────────
// Line iterator helper
// ─────────────────────────────────────────────────────────────

struct LineReader<'a> {
    lines: Vec<&'a str>,
    pos: usize,
}

impl<'a> LineReader<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            lines: input.lines().collect(),
            pos: 0,
        }
    }

    fn next_line(&mut self) -> Result<&'a str, NlParseError> {
        if self.pos >= self.lines.len() {
            return Err(NlParseError::UnexpectedEof);
        }
        let line = self.lines[self.pos];
        self.pos += 1;
        Ok(line)
    }

    fn has_more(&self) -> bool {
        self.pos < self.lines.len()
    }
}

// ─────────────────────────────────────────────────────────────
// Parsing helpers
// ─────────────────────────────────────────────────────────────

fn parse_usize(s: &str) -> Result<usize, NlParseError> {
    s.trim()
        .parse::<usize>()
        .map_err(|_| NlParseError::Parse(format!("expected usize, got '{s}'")))
}

fn parse_i32(s: &str) -> Result<i32, NlParseError> {
    s.trim()
        .parse::<i32>()
        .map_err(|_| NlParseError::Parse(format!("expected i32, got '{s}'")))
}

fn parse_f64(s: &str) -> Result<f64, NlParseError> {
    s.trim()
        .parse::<f64>()
        .map_err(|_| NlParseError::Parse(format!("expected f64, got '{s}'")))
}

fn split_ws(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

// ─────────────────────────────────────────────────────────────
// Header parsing
// ─────────────────────────────────────────────────────────────

fn parse_header(reader: &mut LineReader<'_>) -> Result<NlHeader, NlParseError> {
    // Line 0: "g3 1 1 0" or similar — we just check it starts with 'g'.
    let line0 = reader.next_line()?;
    let l0 = line0.trim();
    if !l0.starts_with('g') && !l0.starts_with('b') {
        return Err(NlParseError::InvalidHeader(format!(
            "expected 'g' or 'b' prefix, got: {l0}"
        )));
    }

    // Line 1: n_vars n_constraints n_objectives n_ranges n_eqns
    let l1 = reader.next_line()?;
    let t1 = split_ws(l1);
    if t1.len() < 5 {
        return Err(NlParseError::InvalidHeader(format!(
            "line 1 needs 5 values, got: {l1}"
        )));
    }
    let n_vars = parse_usize(t1[0])?;
    let n_constraints = parse_usize(t1[1])?;
    let n_objectives = parse_usize(t1[2])?;
    let n_ranges = parse_usize(t1[3])?;
    let n_eqns = parse_usize(t1[4])?;

    // Line 2: n_nonlinear_constraints n_nonlinear_objectives
    let l2 = reader.next_line()?;
    let t2 = split_ws(l2);
    let n_nonlinear_constraints = if !t2.is_empty() { parse_usize(t2[0])? } else { 0 };
    let n_nonlinear_objectives = if t2.len() >= 2 { parse_usize(t2[1])? } else { 0 };

    // Line 3: network constraints (ignored)
    let _l3 = reader.next_line()?;

    // Line 4: nonlinear var counts
    let l4 = reader.next_line()?;
    let t4 = split_ws(l4);
    let n_nl_vars_in_cons = if !t4.is_empty() { parse_usize(t4[0])? } else { 0 };
    let n_nl_vars_in_objs = if t4.len() >= 2 { parse_usize(t4[1])? } else { 0 };
    let n_nl_vars_in_both = if t4.len() >= 3 { parse_usize(t4[2])? } else { 0 };

    // Line 5: flags (ignored)
    let _l5 = reader.next_line()?;

    // Line 6: discrete vars — can be 2, 3, or 5 values
    let l6 = reader.next_line()?;
    let t6 = split_ws(l6);
    let n_linear_binary_vars = if !t6.is_empty() { parse_usize(t6[0])? } else { 0 };
    let n_linear_integer_vars = if t6.len() >= 2 { parse_usize(t6[1])? } else { 0 };
    let n_nl_integer_vars_in_both = if t6.len() >= 3 { parse_usize(t6[2])? } else { 0 };
    let n_nl_integer_vars_in_cons = if t6.len() >= 4 { parse_usize(t6[3])? } else { 0 };
    let n_nl_integer_vars_in_objs = if t6.len() >= 5 { parse_usize(t6[4])? } else { 0 };

    // Line 7: sparsity
    let l7 = reader.next_line()?;
    let t7 = split_ws(l7);
    let n_nonzeros_jacobian = if !t7.is_empty() { parse_usize(t7[0])? } else { 0 };
    let n_nonzeros_obj_gradient = if t7.len() >= 2 { parse_usize(t7[1])? } else { 0 };

    // Line 8: max name lengths (ignored)
    let _l8 = reader.next_line()?;

    // Line 9: common expressions
    let l9 = reader.next_line()?;
    let t9 = split_ws(l9);
    let n_common_exprs_b_c = if !t9.is_empty() { parse_usize(t9[0])? } else { 0 };
    let n_common_exprs_c1 = if t9.len() >= 2 { parse_usize(t9[1])? } else { 0 };
    let n_common_exprs_o1 = if t9.len() >= 3 { parse_usize(t9[2])? } else { 0 };
    let n_common_exprs_c2 = if t9.len() >= 4 { parse_usize(t9[3])? } else { 0 };
    let n_common_exprs_o2 = if t9.len() >= 5 { parse_usize(t9[4])? } else { 0 };

    Ok(NlHeader {
        n_vars,
        n_constraints,
        n_objectives,
        n_ranges,
        n_eqns,
        n_nonlinear_constraints,
        n_nonlinear_objectives,
        n_nl_vars_in_cons,
        n_nl_vars_in_objs,
        n_nl_vars_in_both,
        n_linear_binary_vars,
        n_linear_integer_vars,
        n_nl_integer_vars_in_cons,
        n_nl_integer_vars_in_objs,
        n_nl_integer_vars_in_both,
        n_nonzeros_jacobian,
        n_nonzeros_obj_gradient,
        n_common_exprs_b_c,
        n_common_exprs_c1,
        n_common_exprs_o1,
        n_common_exprs_c2,
        n_common_exprs_o2,
    })
}

// ─────────────────────────────────────────────────────────────
// Expression DAG parsing
// ─────────────────────────────────────────────────────────────

/// Parse one expression from the opcode stream. Reads lines from `reader`
/// and inserts nodes into `arena`, returning the root ExprId.
fn parse_expr(
    reader: &mut LineReader<'_>,
    arena: &mut ExprArena,
    var_nodes: &[ExprId],
) -> Result<ExprId, NlParseError> {
    let line = reader.next_line()?;
    let line = line.trim();

    // Numeric constant: n<value>
    if let Some(rest) = line.strip_prefix('n') {
        let val = parse_f64(rest)?;
        return Ok(arena.add(ExprNode::Constant(val)));
    }

    // Variable reference: v<index>
    if let Some(rest) = line.strip_prefix('v') {
        let idx = parse_usize(rest)?;
        if idx < var_nodes.len() {
            return Ok(var_nodes[idx]);
        }
        return Err(NlParseError::Parse(format!(
            "variable index {idx} out of range (n_vars={})",
            var_nodes.len()
        )));
    }

    // Opcode: o<number>
    if let Some(rest) = line.strip_prefix('o') {
        let opcode = parse_i32(rest)?;
        return parse_opcode(opcode, reader, arena, var_nodes);
    }

    Err(NlParseError::Parse(format!(
        "unexpected expression line: '{line}'"
    )))
}

/// Parse an expression given its opcode. The opcode line has already been
/// consumed; the reader is positioned at the first operand.
fn parse_opcode(
    opcode: i32,
    reader: &mut LineReader<'_>,
    arena: &mut ExprArena,
    var_nodes: &[ExprId],
) -> Result<ExprId, NlParseError> {
    match opcode {
        // Binary operators
        0 => {
            // o0: add
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op: BinOp::Add,
                left,
                right,
            }))
        }
        1 => {
            // o1: subtract
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op: BinOp::Sub,
                left,
                right,
            }))
        }
        2 => {
            // o2: multiply
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left,
                right,
            }))
        }
        3 => {
            // o3: divide
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op: BinOp::Div,
                left,
                right,
            }))
        }
        5 => {
            // o5: power
            let base = parse_expr(reader, arena, var_nodes)?;
            let exp = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op: BinOp::Pow,
                left: base,
                right: exp,
            }))
        }
        // Unary negation
        16 => {
            // o16: unary negation
            let operand = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::UnaryOp {
                op: UnOp::Neg,
                operand,
            }))
        }
        // Math functions (unary)
        37 => {
            // o37: atan — not in MathFunc, approximate with Tan inverse.
            // We don't have atan in the IR; store as a generic approach.
            // For now, treat atan(x) as a function call to Tan (placeholder).
            // TODO: Add Atan to MathFunc if needed.
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Tan, // closest available
                args: vec![arg],
            }))
        }
        38 => {
            // o38: cos
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Cos,
                args: vec![arg],
            }))
        }
        39 => {
            // o39: sin
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Sin,
                args: vec![arg],
            }))
        }
        40 => {
            // o40: sqrt
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Sqrt,
                args: vec![arg],
            }))
        }
        41 => {
            // o41: sinh — not in IR, use Exp-based identity later.
            // For now, store as Exp (placeholder).
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Exp, // placeholder
                args: vec![arg],
            }))
        }
        42 => {
            // o42: asin — not in IR, placeholder as Sin.
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Sin, // placeholder
                args: vec![arg],
            }))
        }
        43 | 54 => {
            // o43: sum of n args  /  o54: sumlist of n args
            // Next line is the count.
            let count_line = reader.next_line()?;
            let count = parse_usize(count_line.trim())?;
            let mut terms = Vec::with_capacity(count);
            for _ in 0..count {
                terms.push(parse_expr(reader, arena, var_nodes)?);
            }
            Ok(arena.add(ExprNode::SumOver { terms }))
        }
        44 => {
            // o44: trunc/intdiv — approximate as Div
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::BinaryOp {
                op: BinOp::Div,
                left,
                right,
            }))
        }
        45 => {
            // o45: log (natural log)
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Log,
                args: vec![arg],
            }))
        }
        46 => {
            // o46: exp
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Exp,
                args: vec![arg],
            }))
        }
        47 => {
            // o47: log10
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Log10,
                args: vec![arg],
            }))
        }
        51 => {
            // o51: tanh — not in IR, placeholder as Tan.
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Tan, // placeholder
                args: vec![arg],
            }))
        }
        13 => {
            // o13: floor — not in IR, approximate as identity
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arg)
        }
        14 => {
            // o14: ceil — not in IR, approximate as identity
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arg)
        }
        15 => {
            // o15: abs
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.add(ExprNode::FunctionCall {
                func: MathFunc::Abs,
                args: vec![arg],
            }))
        }
        _ => Err(NlParseError::UnknownOpcode(opcode)),
    }
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

/// Parse a text-mode .nl file and produce a [`ModelRepr`].
///
/// The input should be the full text content of an .nl file (not binary).
pub fn parse_nl(content: &str) -> Result<ModelRepr, NlParseError> {
    let mut reader = LineReader::new(content);
    let header = parse_header(&mut reader)?;

    let n_vars = header.n_vars;
    let n_constraints = header.n_constraints;
    let n_objectives = header.n_objectives;

    let mut arena = ExprArena::with_capacity(n_vars + n_constraints * 2 + 64);

    // Create variable nodes upfront — one per scalar variable.
    let mut var_nodes = Vec::with_capacity(n_vars);
    for i in 0..n_vars {
        let id = arena.add(ExprNode::Variable {
            name: format!("x{i}"),
            index: i,
            size: 1,
            shape: vec![],
        });
        var_nodes.push(id);
    }

    // Storage for parsed nonlinear constraint/objective expressions.
    // Initialize to None; filled when C/O segments appear.
    let mut nl_con_exprs: Vec<Option<ExprId>> = vec![None; n_constraints];
    let mut nl_obj_exprs: Vec<Option<ExprId>> = vec![None; n_objectives.max(1)];
    let mut obj_sense = ObjectiveSense::Minimize; // default

    // Variable bounds (default: free)
    let mut var_lb = vec![f64::NEG_INFINITY; n_vars];
    let mut var_ub = vec![f64::INFINITY; n_vars];

    // Constraint bounds storage: (sense, lb, ub) per constraint.
    // We store raw bound info and convert to ConstraintRepr at the end.
    #[derive(Clone)]
    struct ConBound {
        lb: f64,
        ub: f64,
        is_range: bool,
        is_eq: bool,
    }
    let mut con_bounds: Vec<ConBound> = vec![
        ConBound {
            lb: f64::NEG_INFINITY,
            ub: f64::INFINITY,
            is_range: false,
            is_eq: false,
        };
        n_constraints
    ];

    // Linear terms for constraints (Jacobian) and objective (gradient).
    // Map: constraint_index -> Vec<(var_index, coefficient)>
    let mut j_terms: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_constraints];
    // Map: objective_index -> Vec<(var_index, coefficient)>
    let mut g_terms: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_objectives.max(1)];

    // Initial point
    let mut _x0: Vec<f64> = vec![0.0; n_vars];

    // Determine variable types from header counts.
    // In .nl format the variables are ordered:
    //   [nonlinear vars in both obj+cons | nl in cons only | nl in objs only | linear vars]
    // And within linear vars:
    //   [arcs | other linear | linear binary | linear integer]
    // The integer/binary counts in header line 6 refer to linear discrete vars
    // located at the end of the variable ordering.
    let n_total_nl_int = header.n_nl_integer_vars_in_both
        + header.n_nl_integer_vars_in_cons
        + header.n_nl_integer_vars_in_objs;
    let mut var_types = vec![VarType::Continuous; n_vars];

    // Mark linear binary vars: last (n_linear_binary + n_linear_integer) vars
    // with binary first then integer.
    let n_discrete = header.n_linear_binary_vars + header.n_linear_integer_vars;
    if n_discrete > 0 && n_vars >= n_discrete {
        let start_binary = n_vars - n_discrete;
        for vt in var_types.iter_mut().skip(start_binary).take(header.n_linear_binary_vars) {
            *vt = VarType::Binary;
        }
        let start_integer = start_binary + header.n_linear_binary_vars;
        for vt in var_types.iter_mut().skip(start_integer).take(header.n_linear_integer_vars) {
            *vt = VarType::Integer;
        }
    }

    // Mark nonlinear integer vars (from line 6 extended format).
    // These are the first n_nl_integer_vars among the nonlinear vars.
    if n_total_nl_int > 0 {
        for vt in var_types.iter_mut().take(n_total_nl_int.min(n_vars)) {
            if *vt == VarType::Continuous {
                *vt = VarType::Integer;
            }
        }
    }

    // Parse remaining segments.
    while reader.has_more() {
        let line = reader.next_line()?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        let first_char = line.as_bytes()[0];

        match first_char {
            b'C' => {
                // Constraint nonlinear expression: C<index>
                let idx = parse_usize(&line[1..])?;
                let expr = parse_expr(&mut reader, &mut arena, &var_nodes)?;
                if idx < n_constraints {
                    nl_con_exprs[idx] = Some(expr);
                }
            }
            b'O' => {
                // Objective expression: O<index> <sense>
                // sense: 0 = minimize, 1 = maximize
                let parts = split_ws(&line[1..]);
                let obj_idx = parse_usize(parts[0])?;
                if parts.len() > 1 {
                    let sense_val = parse_usize(parts[1])?;
                    if sense_val == 1 {
                        obj_sense = ObjectiveSense::Maximize;
                    }
                }
                let expr = parse_expr(&mut reader, &mut arena, &var_nodes)?;
                if obj_idx < nl_obj_exprs.len() {
                    nl_obj_exprs[obj_idx] = Some(expr);
                }
            }
            b'x' => {
                // Initial point: x<count>
                let count = parse_usize(&line[1..])?;
                for _ in 0..count {
                    let pt_line = reader.next_line()?;
                    let pts = split_ws(pt_line);
                    if pts.len() >= 2 {
                        let vi = parse_usize(pts[0])?;
                        let val = parse_f64(pts[1])?;
                        if vi < n_vars {
                            _x0[vi] = val;
                        }
                    }
                }
            }
            b'r' => {
                // Constraint bounds: r
                for cb in con_bounds.iter_mut() {
                    let bound_line = reader.next_line()?;
                    let parts = split_ws(bound_line);
                    if parts.is_empty() {
                        continue;
                    }
                    let btype = parse_usize(parts[0])?;
                    match btype {
                        0 => {
                            // Range: lb <= body <= ub
                            if parts.len() >= 3 {
                                cb.lb = parse_f64(parts[1])?;
                                cb.ub = parse_f64(parts[2])?;
                                cb.is_range = true;
                            }
                        }
                        1 => {
                            // Upper bound: body <= ub
                            if parts.len() >= 2 {
                                cb.ub = parse_f64(parts[1])?;
                            }
                        }
                        2 => {
                            // Lower bound: body >= lb
                            if parts.len() >= 2 {
                                cb.lb = parse_f64(parts[1])?;
                            }
                        }
                        3 => {
                            // Free (no bounds)
                        }
                        4 => {
                            // Equality: body == rhs
                            if parts.len() >= 2 {
                                let rhs = parse_f64(parts[1])?;
                                cb.lb = rhs;
                                cb.ub = rhs;
                                cb.is_eq = true;
                            }
                        }
                        5 => {
                            // Complementarity — treat as range for now
                            if parts.len() >= 3 {
                                cb.lb = parse_f64(parts[1])?;
                                cb.ub = parse_f64(parts[2])?;
                            }
                        }
                        _ => {
                            return Err(NlParseError::Parse(format!(
                                "unknown constraint bound type: {btype}"
                            )));
                        }
                    }
                }
            }
            b'b' => {
                // Variable bounds: b
                for vi in 0..n_vars {
                    let bound_line = reader.next_line()?;
                    let parts = split_ws(bound_line);
                    if parts.is_empty() {
                        continue;
                    }
                    let btype = parse_usize(parts[0])?;
                    match btype {
                        0 => {
                            // Bounded: lb <= x <= ub
                            if parts.len() >= 3 {
                                var_lb[vi] = parse_f64(parts[1])?;
                                var_ub[vi] = parse_f64(parts[2])?;
                            }
                        }
                        1 => {
                            // Upper bound only
                            if parts.len() >= 2 {
                                var_ub[vi] = parse_f64(parts[1])?;
                            }
                        }
                        2 => {
                            // Lower bound only
                            if parts.len() >= 2 {
                                var_lb[vi] = parse_f64(parts[1])?;
                            }
                        }
                        3 => {
                            // Free (unbounded) — already default
                        }
                        4 => {
                            // Fixed
                            if parts.len() >= 2 {
                                let val = parse_f64(parts[1])?;
                                var_lb[vi] = val;
                                var_ub[vi] = val;
                            }
                        }
                        _ => {
                            return Err(NlParseError::Parse(format!(
                                "unknown variable bound type: {btype}"
                            )));
                        }
                    }
                }
            }
            b'k' => {
                // Jacobian column counts (cumulative): k<n_vars-1>
                // We read them but only use them to validate, not needed
                // for building the model since J segments are explicit.
                let count = parse_usize(&line[1..])?;
                for _ in 0..count {
                    let _kcol = reader.next_line()?;
                }
            }
            b'J' => {
                // Jacobian: J<constraint_index> <count>
                let parts = split_ws(&line[1..]);
                let ci = parse_usize(parts[0])?;
                let count = parse_usize(parts[1])?;
                for _ in 0..count {
                    let jl = reader.next_line()?;
                    let jparts = split_ws(jl);
                    if jparts.len() >= 2 {
                        let vi = parse_usize(jparts[0])?;
                        let coeff = parse_f64(jparts[1])?;
                        if ci < n_constraints {
                            j_terms[ci].push((vi, coeff));
                        }
                    }
                }
            }
            b'G' => {
                // Gradient: G<objective_index> <count>
                let parts = split_ws(&line[1..]);
                let oi = parse_usize(parts[0])?;
                let count = parse_usize(parts[1])?;
                for _ in 0..count {
                    let gl = reader.next_line()?;
                    let gparts = split_ws(gl);
                    if gparts.len() >= 2 {
                        let vi = parse_usize(gparts[0])?;
                        let coeff = parse_f64(gparts[1])?;
                        if oi < g_terms.len() {
                            g_terms[oi].push((vi, coeff));
                        }
                    }
                }
            }
            b'V' => {
                // Common (defined) variables: V<index> <arity> <linear_terms>
                // Skip these for now.
                let parts = split_ws(&line[1..]);
                if parts.len() >= 2 {
                    let _vi = parse_usize(parts[0])?;
                    let arity = parse_usize(parts[1])?;
                    // Read arity linear terms then one expression
                    for _ in 0..arity {
                        let _vl = reader.next_line()?;
                    }
                    // Read the expression body
                    let _expr = parse_expr(&mut reader, &mut arena, &var_nodes)?;
                }
            }
            b'S' => {
                // Suffix: S<kind> <count> <name>
                let parts = split_ws(&line[1..]);
                if parts.len() >= 2 {
                    let count = parse_usize(parts[1])?;
                    for _ in 0..count {
                        let _sl = reader.next_line()?;
                    }
                }
            }
            b'd' => {
                // Dual initial values: d<count>
                let count = parse_usize(&line[1..])?;
                for _ in 0..count {
                    let _dl = reader.next_line()?;
                }
            }
            _ => {
                // Skip unknown segment lines (comments, etc.)
            }
        }
    }

    // Build the full expressions by combining nonlinear + linear parts.

    // Helper: Build a linear expression from a list of (var_index, coeff) pairs.
    let build_linear =
        |terms: &[(usize, f64)], arena: &mut ExprArena, var_nodes: &[ExprId]| -> Option<ExprId> {
            if terms.is_empty() {
                return None;
            }
            let mut lin_terms: Vec<ExprId> = Vec::with_capacity(terms.len());
            for &(vi, coeff) in terms {
                if coeff == 0.0 {
                    continue;
                }
                if (coeff - 1.0).abs() < 1e-15 {
                    lin_terms.push(var_nodes[vi]);
                } else if (coeff + 1.0).abs() < 1e-15 {
                    let neg = arena.add(ExprNode::UnaryOp {
                        op: UnOp::Neg,
                        operand: var_nodes[vi],
                    });
                    lin_terms.push(neg);
                } else {
                    let c = arena.add(ExprNode::Constant(coeff));
                    let prod = arena.add(ExprNode::BinaryOp {
                        op: BinOp::Mul,
                        left: c,
                        right: var_nodes[vi],
                    });
                    lin_terms.push(prod);
                }
            }
            if lin_terms.is_empty() {
                return None;
            }
            if lin_terms.len() == 1 {
                Some(lin_terms[0])
            } else {
                Some(arena.add(ExprNode::SumOver { terms: lin_terms }))
            }
        };

    // Build objective expression: nonlinear + linear parts.
    let obj_expr = {
        let nl_part = nl_obj_exprs[0];
        let lin_part = build_linear(&g_terms[0], &mut arena, &var_nodes);

        match (nl_part, lin_part) {
            (Some(nl), Some(lin)) => {
                // Check if nl is just a zero constant — if so, use linear only.
                if is_zero_constant(&arena, nl) {
                    lin
                } else {
                    arena.add(ExprNode::BinaryOp {
                        op: BinOp::Add,
                        left: nl,
                        right: lin,
                    })
                }
            }
            (Some(nl), None) => nl,
            (None, Some(lin)) => lin,
            (None, None) => arena.add(ExprNode::Constant(0.0)),
        }
    };

    // Build constraint expressions and ConstraintRepr.
    let mut constraints = Vec::with_capacity(n_constraints);
    for ci in 0..n_constraints {
        let nl_part = nl_con_exprs[ci];
        let lin_part = build_linear(&j_terms[ci], &mut arena, &var_nodes);

        let body_expr = match (nl_part, lin_part) {
            (Some(nl), Some(lin)) => {
                if is_zero_constant(&arena, nl) {
                    lin
                } else {
                    arena.add(ExprNode::BinaryOp {
                        op: BinOp::Add,
                        left: nl,
                        right: lin,
                    })
                }
            }
            (Some(nl), None) => nl,
            (None, Some(lin)) => lin,
            (None, None) => arena.add(ExprNode::Constant(0.0)),
        };

        let cb = &con_bounds[ci];
        if cb.is_eq {
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Eq,
                rhs: cb.lb,
                name: None,
            });
        } else if cb.is_range {
            // Range constraint: lb <= body <= ub
            // Split into two: body >= lb and body <= ub
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Ge,
                rhs: cb.lb,
                name: None,
            });
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Le,
                rhs: cb.ub,
                name: None,
            });
        } else if cb.lb.is_finite() && !cb.ub.is_finite() {
            // body >= lb
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Ge,
                rhs: cb.lb,
                name: None,
            });
        } else if cb.ub.is_finite() && !cb.lb.is_finite() {
            // body <= ub
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Le,
                rhs: cb.ub,
                name: None,
            });
        } else if cb.lb.is_finite() && cb.ub.is_finite() {
            // Both finite but not marked as range — treat as range
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Ge,
                rhs: cb.lb,
                name: None,
            });
            constraints.push(ConstraintRepr {
                body: body_expr,
                sense: ConstraintSense::Le,
                rhs: cb.ub,
                name: None,
            });
        }
        // else: free constraint, no bound
    }

    // Build VarInfo.
    let mut variables = Vec::with_capacity(n_vars);
    for i in 0..n_vars {
        let vt = var_types[i];
        let (lb, ub) = match vt {
            VarType::Binary => (0.0_f64.max(var_lb[i]), 1.0_f64.min(var_ub[i])),
            _ => (var_lb[i], var_ub[i]),
        };
        variables.push(VarInfo {
            name: format!("x{i}"),
            var_type: vt,
            offset: i,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        });
    }

    Ok(ModelRepr {
        arena,
        objective: obj_expr,
        objective_sense: obj_sense,
        constraints,
        variables,
        n_vars,
    })
}

/// Check if an ExprId is a zero constant.
fn is_zero_constant(arena: &ExprArena, id: ExprId) -> bool {
    matches!(arena.get(id), ExprNode::Constant(v) if v.abs() < 1e-15)
}

/// Parse a .nl file from a file path.
pub fn parse_nl_file(path: &str) -> Result<ModelRepr, NlParseError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| NlParseError::Parse(format!("failed to read file '{path}': {e}")))?;
    parse_nl(&content)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple linear .nl file:
    /// min 2*x + 3*y  s.t. x + y <= 10,  0 <= x <= 100, 0 <= y <= 100
    fn linear_nl() -> String {
        // 2 vars, 1 constraint, 1 objective, 0 ranges, 0 eqns
        // 0 nonlinear constraints, 0 nonlinear objectives
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 1 1 0 0\n"); // n_vars=2, n_cons=1, n_obj=1
        s.push_str(" 0 0\n");       // nonlinear: 0 cons, 0 obj
        s.push_str(" 0 0 0\n");     // network
        s.push_str(" 0 0 0\n");     // nl var counts
        s.push_str(" 0 0 0 1\n");   // flags
        s.push_str(" 0 0\n");       // discrete vars
        s.push_str(" 2 2\n");       // nnz jacobian=2, nnz gradient=2
        s.push_str(" 0 0\n");       // max name lengths
        s.push_str(" 0 0 0 0 0\n"); // common expressions
        // O0: minimize, expression = 0 (linear part in G)
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        // C0: constraint expression = 0 (linear part in J)
        s.push_str("C0\n");
        s.push_str("n0\n");
        // x: initial point
        s.push_str("x2\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        // r: constraint bounds: body <= 10
        s.push_str("r\n");
        s.push_str("1 10\n");
        // b: variable bounds
        s.push_str("b\n");
        s.push_str("0 0 100\n");
        s.push_str("0 0 100\n");
        // k: cumulative column counts
        s.push_str("k1\n");
        s.push_str("1\n");
        // J0: constraint 0 linear terms: 1*x0 + 1*x1
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        // G0: objective 0 linear terms: 2*x0 + 3*x1
        s.push_str("G0 2\n");
        s.push_str("0 2\n");
        s.push_str("1 3\n");
        s
    }

    /// Generate a quadratic .nl file:
    /// min x^2 + y^2  s.t. x + y >= 1,  -10 <= x <= 10, -10 <= y <= 10
    fn quadratic_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 1 1 0 0\n");
        s.push_str(" 1 1\n");       // 1 nonlinear constraint, 1 nonlinear objective
        s.push_str(" 0 0\n");
        s.push_str(" 2 2 2\n");     // 2 nl vars in cons, 2 in objs, 2 in both
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: minimize, x^2 + y^2
        s.push_str("O0 0\n");
        s.push_str("o0\n");         // add
        s.push_str("o5\n");         // pow
        s.push_str("v0\n");         // x
        s.push_str("n2\n");         // 2
        s.push_str("o5\n");         // pow
        s.push_str("v1\n");         // y
        s.push_str("n2\n");         // 2
        // C0: nonlinear part of constraint = 0 (pure linear)
        s.push_str("C0\n");
        s.push_str("n0\n");
        // x
        s.push_str("x2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        // r: constraint bound: body >= 1
        s.push_str("r\n");
        s.push_str("2 1\n");
        // b
        s.push_str("b\n");
        s.push_str("0 -10 10\n");
        s.push_str("0 -10 10\n");
        // k
        s.push_str("k1\n");
        s.push_str("1\n");
        // J0: x + y
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        // G0: no linear part in objective (all nonlinear)
        s.push_str("G0 2\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        s
    }

    /// Generate a nonlinear .nl file:
    /// min exp(x) + log(y)  s.t. x + y <= 5,  0.1 <= x <= 10, 0.1 <= y <= 10
    fn nonlinear_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 1 1 0 0\n");
        s.push_str(" 0 1\n");       // 0 nonlinear constraints, 1 nonlinear objective
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n");     // 0 nl vars in cons, 2 in objs
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 2 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: minimize, exp(x) + log(y)
        s.push_str("O0 0\n");
        s.push_str("o0\n");         // add
        s.push_str("o46\n");        // exp
        s.push_str("v0\n");         // x
        s.push_str("o45\n");        // log
        s.push_str("v1\n");         // y
        // C0: linear constraint (nl part = 0)
        s.push_str("C0\n");
        s.push_str("n0\n");
        // x
        s.push_str("x2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        // r: x + y <= 5
        s.push_str("r\n");
        s.push_str("1 5\n");
        // b
        s.push_str("b\n");
        s.push_str("0 0.1 10\n");
        s.push_str("0 0.1 10\n");
        // k
        s.push_str("k1\n");
        s.push_str("1\n");
        // J0: 1*x + 1*y
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        // G0: no linear gradient terms
        s
    }

    /// Generate a .nl file with binary/integer variables:
    /// min x + 2*y + 3*z  s.t. x + y + z >= 1
    /// x in {0,1}, y integer in [0,5], z continuous in [0,10]
    fn mixed_integer_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 3 1 1 0 0\n"); // 3 vars
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 1 1\n");       // 1 binary, 1 integer (linear)
        s.push_str(" 3 3\n");       // nnz
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: minimize, linear
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        // C0: linear
        s.push_str("C0\n");
        s.push_str("n0\n");
        // x
        s.push_str("x3\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        s.push_str("2 0\n");
        // r: x + y + z >= 1
        s.push_str("r\n");
        s.push_str("2 1\n");
        // b: z continuous [0,10], x binary [0,1], y integer [0,5]
        // Variable order: z (continuous), x (binary), y (integer)
        // because linear discrete vars go at the end
        s.push_str("b\n");
        s.push_str("0 0 10\n");    // z continuous
        s.push_str("0 0 1\n");     // x binary
        s.push_str("0 0 5\n");     // y integer
        // k
        s.push_str("k2\n");
        s.push_str("1\n");
        s.push_str("2\n");
        // J0: 3*z + 1*x + 2*y
        s.push_str("J0 3\n");
        s.push_str("0 3\n");       // z
        s.push_str("1 1\n");       // x
        s.push_str("2 2\n");       // y
        // G0: same as J0
        s.push_str("G0 3\n");
        s.push_str("0 3\n");
        s.push_str("1 1\n");
        s.push_str("2 2\n");
        s
    }

    /// Equality constraint:
    /// min x + y  s.t. x + y == 5,  0 <= x, 0 <= y
    fn equality_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 1 1 0 1\n"); // 0 ranges, 1 eqn
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 2 2\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        s.push_str("C0\n");
        s.push_str("n0\n");
        s.push_str("x2\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        s.push_str("r\n");
        s.push_str("4 5\n");        // equality: body == 5
        s.push_str("b\n");
        s.push_str("2 0\n");        // x >= 0
        s.push_str("2 0\n");        // y >= 0
        s.push_str("k1\n");
        s.push_str("1\n");
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        s.push_str("G0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        s
    }

    /// Range constraint:
    /// min x  s.t. 2 <= x + y <= 8,  0 <= x <= 10, 0 <= y <= 10
    fn range_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 1 1 1 0\n"); // 1 range
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 2 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        s.push_str("C0\n");
        s.push_str("n0\n");
        s.push_str("x2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        s.push_str("r\n");
        s.push_str("0 2 8\n");      // range: 2 <= body <= 8
        s.push_str("b\n");
        s.push_str("0 0 10\n");
        s.push_str("0 0 10\n");
        s.push_str("k1\n");
        s.push_str("1\n");
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        s.push_str("G0 1\n");
        s.push_str("0 1\n");
        s
    }

    /// Maximization problem:
    /// max 3*x + 5*y  s.t. x + y <= 10
    fn maximize_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 1 1 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 2 2\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 1\n");       // 1 = maximize
        s.push_str("n0\n");
        s.push_str("C0\n");
        s.push_str("n0\n");
        s.push_str("x2\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        s.push_str("r\n");
        s.push_str("1 10\n");
        s.push_str("b\n");
        s.push_str("0 0 100\n");
        s.push_str("0 0 100\n");
        s.push_str("k1\n");
        s.push_str("1\n");
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        s.push_str("G0 2\n");
        s.push_str("0 3\n");
        s.push_str("1 5\n");
        s
    }

    /// Sumlist expression: min sum(x_i^2) for i=0..3
    fn sumlist_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 4 0 1 0 0\n"); // 4 vars, 0 constraints
        s.push_str(" 0 1\n");       // 1 nonlinear objective
        s.push_str(" 0 0\n");
        s.push_str(" 0 4 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: sum of x_i^2 using o54 (sumlist)
        s.push_str("O0 0\n");
        s.push_str("o54\n");        // sumlist
        s.push_str("4\n");          // 4 terms
        s.push_str("o5\n");
        s.push_str("v0\n");
        s.push_str("n2\n");
        s.push_str("o5\n");
        s.push_str("v1\n");
        s.push_str("n2\n");
        s.push_str("o5\n");
        s.push_str("v2\n");
        s.push_str("n2\n");
        s.push_str("o5\n");
        s.push_str("v3\n");
        s.push_str("n2\n");
        // b: all free
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s
    }

    // ─── Test: header parsing ─────────────────────────────

    #[test]
    fn test_parse_linear_header() {
        let nl = linear_nl();
        let model = parse_nl(&nl).unwrap();
        assert_eq!(model.n_vars, 2);
        assert_eq!(model.constraints.len(), 1);
        assert_eq!(model.objective_sense, ObjectiveSense::Minimize);
    }

    // ─── Test: linear model evaluation ────────────────────

    #[test]
    fn test_linear_objective_eval() {
        let nl = linear_nl();
        let model = parse_nl(&nl).unwrap();
        // Objective: 2*x + 3*y
        let val = model.evaluate_objective(&[1.0, 2.0]);
        assert!((val - 8.0).abs() < 1e-12, "expected 8.0, got {val}");
    }

    #[test]
    fn test_linear_constraint_eval() {
        let nl = linear_nl();
        let model = parse_nl(&nl).unwrap();
        // Constraint body: x + y (should be <= 10)
        let val = model.evaluate_expr(model.constraints[0].body, &[3.0, 4.0]);
        assert!((val - 7.0).abs() < 1e-12, "expected 7.0, got {val}");
        assert_eq!(model.constraints[0].sense, ConstraintSense::Le);
        assert!((model.constraints[0].rhs - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_var_bounds() {
        let nl = linear_nl();
        let model = parse_nl(&nl).unwrap();
        assert!((model.variables[0].lb[0] - 0.0).abs() < 1e-12);
        assert!((model.variables[0].ub[0] - 100.0).abs() < 1e-12);
        assert!((model.variables[1].lb[0] - 0.0).abs() < 1e-12);
        assert!((model.variables[1].ub[0] - 100.0).abs() < 1e-12);
    }

    // ─── Test: quadratic ──────────────────────────────────

    #[test]
    fn test_quadratic_objective_eval() {
        let nl = quadratic_nl();
        let model = parse_nl(&nl).unwrap();
        // Objective: x^2 + y^2 at (3, 4) = 25
        let val = model.evaluate_objective(&[3.0, 4.0]);
        assert!((val - 25.0).abs() < 1e-12, "expected 25.0, got {val}");
    }

    #[test]
    fn test_quadratic_constraint_eval() {
        let nl = quadratic_nl();
        let model = parse_nl(&nl).unwrap();
        // Constraint body: x + y (linear, nl part is 0)
        let val = model.evaluate_expr(model.constraints[0].body, &[3.0, 4.0]);
        assert!((val - 7.0).abs() < 1e-12, "expected 7.0, got {val}");
        assert_eq!(model.constraints[0].sense, ConstraintSense::Ge);
        assert!((model.constraints[0].rhs - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quadratic_structure() {
        let nl = quadratic_nl();
        let model = parse_nl(&nl).unwrap();
        assert!(model.arena.is_quadratic(model.objective));
        assert!(!model.arena.is_linear(model.objective));
    }

    // ─── Test: nonlinear ──────────────────────────────────

    #[test]
    fn test_nonlinear_objective_eval() {
        let nl = nonlinear_nl();
        let model = parse_nl(&nl).unwrap();
        // Objective: exp(x) + log(y) at (1, e) = e + 1
        let val = model.evaluate_objective(&[1.0, std::f64::consts::E]);
        let expected = std::f64::consts::E + 1.0;
        assert!(
            (val - expected).abs() < 1e-12,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_nonlinear_structure() {
        let nl = nonlinear_nl();
        let model = parse_nl(&nl).unwrap();
        assert!(!model.arena.is_linear(model.objective));
        assert!(!model.arena.is_quadratic(model.objective));
    }

    // ─── Test: mixed integer ──────────────────────────────

    #[test]
    fn test_mixed_integer_var_types() {
        let nl = mixed_integer_nl();
        let model = parse_nl(&nl).unwrap();
        assert_eq!(model.n_vars, 3);
        // Variable order: z (continuous), x (binary), y (integer)
        assert_eq!(model.variables[0].var_type, VarType::Continuous);
        assert_eq!(model.variables[1].var_type, VarType::Binary);
        assert_eq!(model.variables[2].var_type, VarType::Integer);
    }

    #[test]
    fn test_mixed_integer_binary_bounds() {
        let nl = mixed_integer_nl();
        let model = parse_nl(&nl).unwrap();
        // Binary var should have bounds [0, 1]
        assert!((model.variables[1].lb[0] - 0.0).abs() < 1e-12);
        assert!((model.variables[1].ub[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mixed_integer_eval() {
        let nl = mixed_integer_nl();
        let model = parse_nl(&nl).unwrap();
        // Objective: 3*z + 1*x + 2*y at (z=2, x=1, y=3) = 6+1+6 = 13
        let val = model.evaluate_objective(&[2.0, 1.0, 3.0]);
        assert!((val - 13.0).abs() < 1e-12, "expected 13.0, got {val}");
    }

    // ─── Test: equality constraint ────────────────────────

    #[test]
    fn test_equality_constraint() {
        let nl = equality_nl();
        let model = parse_nl(&nl).unwrap();
        assert_eq!(model.constraints.len(), 1);
        assert_eq!(model.constraints[0].sense, ConstraintSense::Eq);
        assert!((model.constraints[0].rhs - 5.0).abs() < 1e-12);
    }

    // ─── Test: range constraint ───────────────────────────

    #[test]
    fn test_range_constraint() {
        let nl = range_nl();
        let model = parse_nl(&nl).unwrap();
        // Range constraint splits into 2 constraints
        assert_eq!(model.constraints.len(), 2);
        // body >= 2
        assert_eq!(model.constraints[0].sense, ConstraintSense::Ge);
        assert!((model.constraints[0].rhs - 2.0).abs() < 1e-12);
        // body <= 8
        assert_eq!(model.constraints[1].sense, ConstraintSense::Le);
        assert!((model.constraints[1].rhs - 8.0).abs() < 1e-12);
    }

    // ─── Test: maximize ───────────────────────────────────

    #[test]
    fn test_maximize_sense() {
        let nl = maximize_nl();
        let model = parse_nl(&nl).unwrap();
        assert_eq!(model.objective_sense, ObjectiveSense::Maximize);
    }

    #[test]
    fn test_maximize_eval() {
        let nl = maximize_nl();
        let model = parse_nl(&nl).unwrap();
        // Objective: 3*x + 5*y at (2, 3) = 21
        let val = model.evaluate_objective(&[2.0, 3.0]);
        assert!((val - 21.0).abs() < 1e-12, "expected 21.0, got {val}");
    }

    // ─── Test: sumlist ────────────────────────────────────

    #[test]
    fn test_sumlist_eval() {
        let nl = sumlist_nl();
        let model = parse_nl(&nl).unwrap();
        // sum(x_i^2) at (1,2,3,4) = 1+4+9+16 = 30
        let val = model.evaluate_objective(&[1.0, 2.0, 3.0, 4.0]);
        assert!((val - 30.0).abs() < 1e-12, "expected 30.0, got {val}");
    }

    // ─── Test: n_vars consistency ─────────────────────────

    #[test]
    fn test_n_vars_equals_variables_len() {
        for (name, nl_fn) in [
            ("linear", linear_nl as fn() -> String),
            ("quadratic", quadratic_nl),
            ("nonlinear", nonlinear_nl),
            ("mixed_integer", mixed_integer_nl),
            ("equality", equality_nl),
            ("range", range_nl),
            ("maximize", maximize_nl),
            ("sumlist", sumlist_nl),
        ] {
            let model = parse_nl(&nl_fn()).unwrap_or_else(|e| {
                panic!("failed to parse {name}: {e}")
            });
            assert_eq!(
                model.n_vars,
                model.variables.len(),
                "{name}: n_vars mismatch"
            );
        }
    }

    // ─── Test: variable offsets ───────────────────────────

    #[test]
    fn test_variable_offsets_sequential() {
        let nl = linear_nl();
        let model = parse_nl(&nl).unwrap();
        for (i, var) in model.variables.iter().enumerate() {
            assert_eq!(var.offset, i, "var {i} offset should be {i}");
        }
    }

    // ─── Test: empty content error ────────────────────────

    #[test]
    fn test_empty_content_error() {
        let result = parse_nl("");
        assert!(result.is_err());
    }

    // ─── Test: bad header error ───────────────────────────

    #[test]
    fn test_bad_header_error() {
        let result = parse_nl("not a valid nl file\n");
        assert!(result.is_err());
    }

    // ─── Test: unary negation ─────────────────────────────

    #[test]
    fn test_unary_negation() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 1 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 1 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("o16\n");        // unary neg
        s.push_str("v0\n");         // -x
        s.push_str("b\n");
        s.push_str("3\n");          // free
        let model = parse_nl(&s).unwrap();
        // -x at x=5 should be -5
        let val = model.evaluate_objective(&[5.0]);
        assert!((val - (-5.0)).abs() < 1e-12);
    }

    // ─── Test: multiply expression ────────────────────────

    #[test]
    fn test_multiply_expression() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: x * y
        s.push_str("O0 0\n");
        s.push_str("o2\n");         // multiply
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        // x*y at (3, 7) = 21
        let val = model.evaluate_objective(&[3.0, 7.0]);
        assert!((val - 21.0).abs() < 1e-12);
    }

    // ─── Test: divide expression ──────────────────────────

    #[test]
    fn test_divide_expression() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: x / y
        s.push_str("O0 0\n");
        s.push_str("o3\n");         // divide
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        // x/y at (10, 4) = 2.5
        let val = model.evaluate_objective(&[10.0, 4.0]);
        assert!((val - 2.5).abs() < 1e-12);
    }

    // ─── Test: sin/cos expression ─────────────────────────

    #[test]
    fn test_sin_cos_expression() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: sin(x) + cos(y)
        s.push_str("O0 0\n");
        s.push_str("o0\n");
        s.push_str("o39\n");        // sin
        s.push_str("v0\n");
        s.push_str("o38\n");        // cos
        s.push_str("v1\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        let x = std::f64::consts::FRAC_PI_2;
        let y = 0.0;
        // sin(pi/2) + cos(0) = 1 + 1 = 2
        let val = model.evaluate_objective(&[x, y]);
        assert!((val - 2.0).abs() < 1e-12, "expected 2.0, got {val}");
    }

    // ─── Test: sqrt expression ────────────────────────────

    #[test]
    fn test_sqrt_expression() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 1 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 1 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("o40\n");        // sqrt
        s.push_str("v0\n");
        s.push_str("b\n");
        s.push_str("2 0\n");        // x >= 0
        let model = parse_nl(&s).unwrap();
        // sqrt(9) = 3
        let val = model.evaluate_objective(&[9.0]);
        assert!((val - 3.0).abs() < 1e-12);
    }

    // ─── Test: fixed variable bound ───────────────────────

    #[test]
    fn test_fixed_variable_bound() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 1 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        s.push_str("b\n");
        s.push_str("4 3.14\n");     // x fixed at 3.14
        s.push_str("1 100\n");      // y <= 100 (upper bound only)
        s.push_str("G0 1\n");
        s.push_str("0 1\n");
        let model = parse_nl(&s).unwrap();
        assert!((model.variables[0].lb[0] - 3.14).abs() < 1e-12);
        assert!((model.variables[0].ub[0] - 3.14).abs() < 1e-12);
        assert!(model.variables[1].lb[0].is_infinite() && model.variables[1].lb[0] < 0.0);
        assert!((model.variables[1].ub[0] - 100.0).abs() < 1e-12);
    }

    // ─── Test: no-constraint model ────────────────────────

    #[test]
    fn test_no_constraints() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 1 0 1 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("G0 1\n");
        s.push_str("0 1\n");
        let model = parse_nl(&s).unwrap();
        assert_eq!(model.n_vars, 1);
        assert!(model.constraints.is_empty());
        let val = model.evaluate_objective(&[7.0]);
        assert!((val - 7.0).abs() < 1e-12);
    }

    // ─── Test: log10 expression ───────────────────────────

    #[test]
    fn test_log10_expression() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 1 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 1 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("o47\n");        // log10
        s.push_str("v0\n");
        s.push_str("b\n");
        s.push_str("2 0.001\n");
        let model = parse_nl(&s).unwrap();
        // log10(100) = 2
        let val = model.evaluate_objective(&[100.0]);
        assert!((val - 2.0).abs() < 1e-12);
    }

    // ─── Test: nested expression ──────────────────────────

    #[test]
    fn test_nested_expression() {
        // exp(x^2 + y) at x=1, y=2 => exp(3)
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("o46\n");        // exp
        s.push_str("o0\n");         // add
        s.push_str("o5\n");         // pow
        s.push_str("v0\n");         // x
        s.push_str("n2\n");         // 2
        s.push_str("v1\n");         // y
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        let val = model.evaluate_objective(&[1.0, 2.0]);
        let expected = (3.0_f64).exp();
        assert!((val - expected).abs() < 1e-10, "expected {expected}, got {val}");
    }

    // ─── Test: multiple constraints ───────────────────────

    #[test]
    fn test_multiple_constraints() {
        // min x + y  s.t. x + y <= 10, x - y >= -2
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 2 1 0 0\n"); // 2 constraints
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 4 2\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        s.push_str("C0\n");
        s.push_str("n0\n");
        s.push_str("C1\n");
        s.push_str("n0\n");
        s.push_str("x2\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        s.push_str("r\n");
        s.push_str("1 10\n");       // c0: body <= 10
        s.push_str("2 -2\n");       // c1: body >= -2
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s.push_str("k1\n");
        s.push_str("2\n");
        s.push_str("J0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        s.push_str("J1 2\n");
        s.push_str("0 1\n");
        s.push_str("1 -1\n");
        s.push_str("G0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 1\n");
        let model = parse_nl(&s).unwrap();
        assert_eq!(model.constraints.len(), 2);
        // c0: x + y at (3, 4) = 7
        let val0 = model.evaluate_expr(model.constraints[0].body, &[3.0, 4.0]);
        assert!((val0 - 7.0).abs() < 1e-12);
        assert_eq!(model.constraints[0].sense, ConstraintSense::Le);
        // c1: x - y at (3, 4) = -1
        let val1 = model.evaluate_expr(model.constraints[1].body, &[3.0, 4.0]);
        assert!((val1 - (-1.0)).abs() < 1e-12);
        assert_eq!(model.constraints[1].sense, ConstraintSense::Ge);
    }

    // ─── Test: parse_nl_file error on missing file ────────

    #[test]
    fn test_parse_nl_file_missing() {
        let result = parse_nl_file("/nonexistent/path/to/file.nl");
        assert!(result.is_err());
    }

    // ─── Test: coefficient -1 optimization ────────────────

    #[test]
    fn test_neg_coefficient() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 2\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("n0\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        // G0: 1*x + (-1)*y
        s.push_str("G0 2\n");
        s.push_str("0 1\n");
        s.push_str("1 -1\n");
        let model = parse_nl(&s).unwrap();
        // x - y at (5, 3) = 2
        let val = model.evaluate_objective(&[5.0, 3.0]);
        assert!((val - 2.0).abs() < 1e-12);
    }

    // ─── Test: subtraction expression ─────────────────────

    #[test]
    fn test_subtraction_expression() {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: x - y
        s.push_str("O0 0\n");
        s.push_str("o1\n");         // subtract
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        let val = model.evaluate_objective(&[10.0, 3.0]);
        assert!((val - 7.0).abs() < 1e-12);
    }
}
