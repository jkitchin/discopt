//! .nl file parser — reads AMPL .nl (text) format into [`ModelRepr`].
//!
//! Supports the text-mode .nl format produced by AMPL with `option nl_comments 0;`.
//! Parses the header, expression DAG segments (C, O), variable/constraint bounds
//! (b, r), linear Jacobian/gradient terms (J, G), and initial point (x).
//!
//! # Constraint numbering (C-12)
//!
//! A range constraint `l <= body <= u` is split into two [`ConstraintRepr`] rows
//! (`body >= l` and `body <= u`), and any constraint with two finite bounds not
//! flagged as an explicit range is likewise split. As a result, the index of a
//! constraint in [`ModelRepr::constraints`] does **not** in general equal its
//! original row index in the source `.nl` file: every split shifts all subsequent
//! constraints by one. The model is internally self-consistent — bodies, senses,
//! and RHS values are correct — so nothing in the current solver path depends on
//! source-index alignment. This note exists because any *future* feature that maps
//! results back to source rows by index (e.g. AMPL constraint duals, `.sol`
//! suffixes, or a round-trip writer) must first thread a `source_row` field through
//! `ConstraintRepr` at each split site here, rather than assuming a 1:1 mapping.
//! Tracked as correctness-issue C-12 (P3, hygiene — no current mis-certification).
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
    /// A recognised opcode maps to an operator that has no sound representation
    /// in the IR (e.g. floor/ceil/round/trunc/intdiv). Carries the operator name
    /// and the numeric opcode. We refuse loudly rather than silently substituting
    /// a different operator (identity / real division), which would solve a
    /// different problem than the user posed. See correctness issue C-5.
    UnsupportedOpcode {
        /// Human-readable operator name (e.g. "floor").
        name: &'static str,
        /// The numeric .nl opcode (e.g. 13 for `o13`).
        opcode: i32,
    },
    /// The file uses the binary (`b`) .nl encoding, which is not supported.
    BinaryUnsupported,
    /// General parse error with a human-readable message.
    Parse(String),
}

impl fmt::Display for NlParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NlParseError::UnexpectedEof => write!(f, "unexpected end of .nl input"),
            NlParseError::InvalidHeader(msg) => write!(f, "invalid .nl header: {msg}"),
            NlParseError::UnknownOpcode(op) => write!(f, "unknown .nl opcode: o{op}"),
            NlParseError::UnsupportedOpcode { name, opcode } => write!(
                f,
                "unsupported .nl operator '{name}' (opcode o{opcode}): this operator has no \
                 sound representation in the solver IR. Parsing is refused rather than \
                 silently substituting a different operator (e.g. identity or real division), \
                 which would solve a different problem than the one posed. Reformulate the \
                 model without '{name}', or file a feature request for real IR + relaxation \
                 support for it."
            ),
            NlParseError::BinaryUnsupported => write!(
                f,
                "binary .nl format not supported: file uses the binary ('b') encoding, \
                 but only the text/ASCII ('g') format is supported. Re-export the model \
                 in text .nl format (e.g. AMPL `write g<stub>;` rather than `write b<stub>;`)"
            ),
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
    // Line 0: "g3 1 1 0" or similar. The leading char selects the encoding of
    // the file body: 'g' = text/ASCII, 'b' = binary. Only text is supported.
    let line0 = reader.next_line()?;
    let l0 = line0.trim();
    if l0.starts_with('b') {
        return Err(NlParseError::BinaryUnsupported);
    }
    if !l0.starts_with('g') {
        return Err(NlParseError::InvalidHeader(format!(
            "expected 'g' (text) prefix, got: {l0}"
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
    let n_nonlinear_constraints = if !t2.is_empty() {
        parse_usize(t2[0])?
    } else {
        0
    };
    let n_nonlinear_objectives = if t2.len() >= 2 {
        parse_usize(t2[1])?
    } else {
        0
    };

    // Line 3: network constraints (ignored)
    let _l3 = reader.next_line()?;

    // Line 4: nonlinear var counts
    let l4 = reader.next_line()?;
    let t4 = split_ws(l4);
    let n_nl_vars_in_cons = if !t4.is_empty() {
        parse_usize(t4[0])?
    } else {
        0
    };
    let n_nl_vars_in_objs = if t4.len() >= 2 {
        parse_usize(t4[1])?
    } else {
        0
    };
    let n_nl_vars_in_both = if t4.len() >= 3 {
        parse_usize(t4[2])?
    } else {
        0
    };

    // Line 5: flags (ignored)
    let _l5 = reader.next_line()?;

    // Line 6: discrete vars — can be 2, 3, or 5 values
    let l6 = reader.next_line()?;
    let t6 = split_ws(l6);
    let n_linear_binary_vars = if !t6.is_empty() {
        parse_usize(t6[0])?
    } else {
        0
    };
    let n_linear_integer_vars = if t6.len() >= 2 {
        parse_usize(t6[1])?
    } else {
        0
    };
    let n_nl_integer_vars_in_both = if t6.len() >= 3 {
        parse_usize(t6[2])?
    } else {
        0
    };
    let n_nl_integer_vars_in_cons = if t6.len() >= 4 {
        parse_usize(t6[3])?
    } else {
        0
    };
    let n_nl_integer_vars_in_objs = if t6.len() >= 5 {
        parse_usize(t6[4])?
    } else {
        0
    };

    // Line 7: sparsity
    let l7 = reader.next_line()?;
    let t7 = split_ws(l7);
    let n_nonzeros_jacobian = if !t7.is_empty() {
        parse_usize(t7[0])?
    } else {
        0
    };
    let n_nonzeros_obj_gradient = if t7.len() >= 2 {
        parse_usize(t7[1])?
    } else {
        0
    };

    // Line 8: max name lengths (ignored)
    let _l8 = reader.next_line()?;

    // Line 9: common expressions
    let l9 = reader.next_line()?;
    let t9 = split_ws(l9);
    let n_common_exprs_b_c = if !t9.is_empty() {
        parse_usize(t9[0])?
    } else {
        0
    };
    let n_common_exprs_c1 = if t9.len() >= 2 {
        parse_usize(t9[1])?
    } else {
        0
    };
    let n_common_exprs_o1 = if t9.len() >= 3 {
        parse_usize(t9[2])?
    } else {
        0
    };
    let n_common_exprs_c2 = if t9.len() >= 4 {
        parse_usize(t9[3])?
    } else {
        0
    };
    let n_common_exprs_o2 = if t9.len() >= 5 {
        parse_usize(t9[4])?
    } else {
        0
    };

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
        return Ok(arena.intern(ExprNode::Constant(val)));
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
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Add,
                left,
                right,
            }))
        }
        1 => {
            // o1: subtract
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Sub,
                left,
                right,
            }))
        }
        2 => {
            // o2: multiply
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left,
                right,
            }))
        }
        3 => {
            // o3: divide
            let left = parse_expr(reader, arena, var_nodes)?;
            let right = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Div,
                left,
                right,
            }))
        }
        5 => {
            // o5: power
            let base = parse_expr(reader, arena, var_nodes)?;
            let exp = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Pow,
                left: base,
                right: exp,
            }))
        }
        // Unary negation
        16 => {
            // o16: unary negation
            let operand = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::UnaryOp {
                op: UnOp::Neg,
                operand,
            }))
        }
        // ── Unary operators (class 1) ──
        // Opcode table from Gay, "Writing .nl Files", 2005, Table 4.
        37 => {
            // o37: tanh
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Tanh,
                args: vec![arg],
            }))
        }
        38 => {
            // o38: tan
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Tan,
                args: vec![arg],
            }))
        }
        39 => {
            // o39: sqrt
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Sqrt,
                args: vec![arg],
            }))
        }
        40 => {
            // o40: sinh
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Sinh,
                args: vec![arg],
            }))
        }
        41 => {
            // o41: sin
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Sin,
                args: vec![arg],
            }))
        }
        42 => {
            // o42: log10
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Log10,
                args: vec![arg],
            }))
        }
        43 => {
            // o43: log (natural)
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Log,
                args: vec![arg],
            }))
        }
        44 => {
            // o44: exp
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Exp,
                args: vec![arg],
            }))
        }
        45 => {
            // o45: cosh
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Cosh,
                args: vec![arg],
            }))
        }
        46 => {
            // o46: cos
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Cos,
                args: vec![arg],
            }))
        }
        47 => {
            // o47: atanh
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Atanh,
                args: vec![arg],
            }))
        }
        49 => {
            // o49: atan
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Atan,
                args: vec![arg],
            }))
        }
        50 => {
            // o50: asinh
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Asinh,
                args: vec![arg],
            }))
        }
        51 => {
            // o51: asin
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Asin,
                args: vec![arg],
            }))
        }
        52 => {
            // o52: acosh
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Acosh,
                args: vec![arg],
            }))
        }
        53 => {
            // o53: acos
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
                func: MathFunc::Acos,
                args: vec![arg],
            }))
        }
        // ── N-ary operators (class 3) ──
        54 => {
            // o54: sum of n args
            let count_line = reader.next_line()?;
            let count = parse_usize(count_line.trim())?;
            let mut terms = Vec::with_capacity(count);
            for _ in 0..count {
                terms.push(parse_expr(reader, arena, var_nodes)?);
            }
            Ok(arena.intern(ExprNode::SumOver { terms }))
        }
        // o11: minlist / o12: maxlist — n-ary min/max. Same wire format as o54
        // (a count line, then that many operand expressions). The IR's Min/Max
        // functions are evaluated as strictly BINARY (args[0], args[1]) in every
        // consumer (evaluate/evaluate_node/FBBT), so an n-ary FunctionCall would
        // silently drop args[2..] — a silent misparse. We therefore fold the list
        // into a left-nested tree of binary Min/Max, which is exactly equivalent
        // and uses only the sound binary semantics. (C-8)
        11 | 12 => {
            let count_line = reader.next_line()?;
            let count = parse_usize(count_line.trim())?;
            if count == 0 {
                return Err(NlParseError::Parse(format!(
                    "min/max list (o{opcode}) with zero arguments"
                )));
            }
            let func = if opcode == 11 {
                MathFunc::Min
            } else {
                MathFunc::Max
            };
            let mut acc = parse_expr(reader, arena, var_nodes)?;
            for _ in 1..count {
                let next = parse_expr(reader, arena, var_nodes)?;
                acc = arena.intern(ExprNode::FunctionCall {
                    func,
                    args: vec![acc, next],
                });
            }
            Ok(acc)
        }
        // ── Power forms (class 2/1) — all map to BinOp::Pow (C-8). ──
        // The .nl operand order (Gay, "Writing .nl Files"; cf. Couenne nl2e.cpp
        // OP1POW/OP2POW/OPCPOW): o76 = base^const (base expr then constant power),
        // o77 = base^2 (unary), o78 = const^exp (constant base then exponent expr).
        // Constants arrive as ordinary `n<value>` leaves via parse_expr.
        76 => {
            // o76 OP1POW: base ^ constant-power
            let base = parse_expr(reader, arena, var_nodes)?;
            let exp = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Pow,
                left: base,
                right: exp,
            }))
        }
        77 => {
            // o77 OP2POW: base ^ 2 (unary)
            let base = parse_expr(reader, arena, var_nodes)?;
            let two = arena.intern(ExprNode::Constant(2.0));
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Pow,
                left: base,
                right: two,
            }))
        }
        78 => {
            // o78 OPCPOW: constant-base ^ exponent
            let base = parse_expr(reader, arena, var_nodes)?;
            let exp = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::BinaryOp {
                op: BinOp::Pow,
                left: base,
                right: exp,
            }))
        }
        // o48 atan2 (binary) and o35 if-then-else (ternary) have no sound IR
        // representation — refuse loudly rather than drop operands or misuse the
        // single-argument atan. (C-8; mirrors the C-5 refusal policy.) The error
        // aborts the parse, so we deliberately do NOT consume operands.
        48 => Err(NlParseError::UnsupportedOpcode {
            name: "atan2",
            opcode: 48,
        }),
        35 => Err(NlParseError::UnsupportedOpcode {
            name: "if",
            opcode: 35,
        }),
        // ── Binary operators (class 2) ──
        //
        // C-5: floor/ceil/round/trunc/intdiv have NO sound representation in the IR.
        // Previously each of these was silently rewritten to a *different* operator
        // (identity for floor/ceil/round/trunc; real Div for intdiv), which makes the
        // solver certify the optimum of a different problem than the user posed. We now
        // refuse loudly (UnsupportedOpcode) so the solve errors instead of returning a
        // false certificate. Real support would require IR + relaxation work (out of
        // scope). The error aborts the parse, so we deliberately do NOT consume operands.
        55 => Err(NlParseError::UnsupportedOpcode {
            name: "intdiv",
            opcode: 55,
        }),
        57 => Err(NlParseError::UnsupportedOpcode {
            name: "round",
            opcode: 57,
        }),
        58 => Err(NlParseError::UnsupportedOpcode {
            name: "trunc",
            opcode: 58,
        }),
        13 => Err(NlParseError::UnsupportedOpcode {
            name: "floor",
            opcode: 13,
        }),
        14 => Err(NlParseError::UnsupportedOpcode {
            name: "ceil",
            opcode: 14,
        }),
        15 => {
            // o15: abs
            let arg = parse_expr(reader, arena, var_nodes)?;
            Ok(arena.intern(ExprNode::FunctionCall {
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

    // Phase 4 CSE (hash-consing): intern expression nodes as they are built so
    // that structurally-identical subexpressions (AMPL emits many — bounds
    // constants, repeated products, shared nonlinear terms) collapse to a single
    // arena node. Seeded from the variable nodes above (interning trivially by
    // block index). Semantic-preserving by construction (see `ExprArena::intern`).
    // Disabled again before the model leaves the parser so downstream mutating
    // passes (presolve/reformulation) keep plain append semantics.
    //
    // `DISCOPT_DISABLE_CSE` is a measurement/escape hatch: when set, the arena is
    // built with plain append (pre-CSE behavior). Because interning only ever
    // merges structurally-identical nodes, the two builds are evaluation-identical;
    // the flag exists to quantify the node-count lever and as a safety fallback.
    let cse_enabled = std::env::var_os("DISCOPT_DISABLE_CSE").is_none();
    if cse_enabled {
        arena.enable_interning();
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
    let _n_total_nl_int = header.n_nl_integer_vars_in_both
        + header.n_nl_integer_vars_in_cons
        + header.n_nl_integer_vars_in_objs;
    let mut var_types = vec![VarType::Continuous; n_vars];

    // Mark linear binary vars: last (n_linear_binary + n_linear_integer) vars
    // with binary first then integer.
    let n_discrete = header.n_linear_binary_vars + header.n_linear_integer_vars;
    if n_discrete > 0 && n_vars >= n_discrete {
        let start_binary = n_vars - n_discrete;
        for vt in var_types
            .iter_mut()
            .skip(start_binary)
            .take(header.n_linear_binary_vars)
        {
            *vt = VarType::Binary;
        }
        let start_integer = start_binary + header.n_linear_binary_vars;
        for vt in var_types
            .iter_mut()
            .skip(start_integer)
            .take(header.n_linear_integer_vars)
        {
            *vt = VarType::Integer;
        }
    }

    // Mark nonlinear integer vars (from line 6 extended format).
    // Per the AMPL .nl spec, nonlinear integer variables are the LAST
    // ones in each variable group:
    //   - Last nlvbi of the nlvb group (vars in both cons + objs)
    //   - Last nlvci of the (nlvc - nlvb) group (vars in cons only)
    //   - Last nlvoi of the (nlvo - nlvb) group (vars in objs only)
    let nlvb = header.n_nl_vars_in_both;
    let nlvc = header.n_nl_vars_in_cons;
    let nlvo = header.n_nl_vars_in_objs;
    let nlvbi = header.n_nl_integer_vars_in_both;
    let nlvci = header.n_nl_integer_vars_in_cons;
    let nlvoi = header.n_nl_integer_vars_in_objs;

    // Last nlvbi of the nlvb group
    if nlvbi > 0 && nlvb >= nlvbi {
        let start = nlvb - nlvbi;
        for vt in var_types.iter_mut().skip(start).take(nlvbi) {
            *vt = VarType::Integer;
        }
    }

    // Last nlvci of the (nlvc - nlvb) group (starts at nlvb)
    if nlvci > 0 && nlvc > nlvb {
        let group_size = nlvc - nlvb;
        if group_size >= nlvci {
            let start = nlvb + group_size - nlvci;
            for vt in var_types.iter_mut().skip(start).take(nlvci) {
                *vt = VarType::Integer;
            }
        }
    }

    // Last nlvoi of the objective-only nonlinear group. The group's bounds depend
    // on whether more nonlinear variables live in objectives than in constraints:
    //   * nlvc >= nlvo (common): block = [max(nlvc,nlvb), max(nlvc,nlvb)+(nlvo-nlvb))
    //   * nlvo  >  nlvc (AMPL edge case): the whole nonlinear block is [0, nlvo) and
    //     the objective-only group is [nlvc, nlvo); its integer tail is
    //     [nlvo-nlvoi, nlvo).  The old `(nlvo - nlvb)`-sized formula placed that tail
    //     out of range here, silently leaving the variables continuous — which on
    //     ex1252a relaxed its 3 binaries (b22..b24) to [0,1], enlarging the feasible
    //     set and yielding a *false-feasible* incumbent below the true optimum.
    if nlvoi > 0 && nlvo > nlvb {
        let (obj_only_start, obj_only_end) = if nlvo > nlvc {
            (nlvc, nlvo)
        } else {
            let s = std::cmp::max(nlvc, nlvb);
            (s, s + (nlvo - nlvb))
        };
        let group_size = obj_only_end.saturating_sub(obj_only_start);
        if group_size >= nlvoi && obj_only_end <= n_vars {
            let start = obj_only_end - nlvoi;
            for vt in var_types.iter_mut().skip(start).take(nlvoi) {
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
                // Common (defined) variables: V<index> <n_linear> <k>
                //   <var_index> <coeff>   (n_linear lines: the linear part)
                //   <nonlinear expression tree>
                // The defined variable's value is `linear_part + nonlinear_body`.
                // Its index is `n_vars + position` and references to it (`v<idx>`
                // with idx >= n_vars) appear in later C/O/V segments. (C-7)
                //
                // We INLINE it: build the defining expression and push it onto
                // `var_nodes` at its index, so a later `v<idx>` resolves to the
                // subexpression. Defined vars only ever reference *earlier* vars /
                // defined vars, so parsing the body before the push is sound.
                // (CSE/sharing is a separate performance upgrade; inlining is the
                // correctness fix — it makes AMPL/Pyomo common-subexpression .nl
                // parse instead of erroring on the reference.)
                let parts = split_ws(&line[1..]);
                if parts.len() >= 2 {
                    let vi = parse_usize(parts[0])?;
                    let n_linear = parse_usize(parts[1])?;
                    // Read the n_linear (var_index, coeff) pairs of the linear part.
                    let mut lin_pairs: Vec<(usize, f64)> = Vec::with_capacity(n_linear);
                    for _ in 0..n_linear {
                        let vl = reader.next_line()?;
                        let lp = split_ws(vl);
                        if lp.len() >= 2 {
                            lin_pairs.push((parse_usize(lp[0])?, parse_f64(lp[1])?));
                        }
                    }
                    // Read the nonlinear body (may reference earlier defined vars).
                    let body = parse_expr(&mut reader, &mut arena, &var_nodes)?;

                    // Build linear part: Σ coeff·v_j (reusing +1/-1 shortcuts).
                    let mut lin_terms: Vec<ExprId> = Vec::with_capacity(lin_pairs.len());
                    for (lvi, coeff) in lin_pairs {
                        if coeff == 0.0 {
                            continue;
                        }
                        let operand = if lvi < var_nodes.len() {
                            var_nodes[lvi]
                        } else {
                            return Err(NlParseError::Parse(format!(
                                "defined variable V{vi} references variable index {lvi} \
                                 which is not yet defined (n_vars+defined so far = {})",
                                var_nodes.len()
                            )));
                        };
                        if (coeff - 1.0).abs() < 1e-15 {
                            lin_terms.push(operand);
                        } else if (coeff + 1.0).abs() < 1e-15 {
                            lin_terms.push(arena.intern(ExprNode::UnaryOp {
                                op: UnOp::Neg,
                                operand,
                            }));
                        } else {
                            let c = arena.intern(ExprNode::Constant(coeff));
                            lin_terms.push(arena.intern(ExprNode::BinaryOp {
                                op: BinOp::Mul,
                                left: c,
                                right: operand,
                            }));
                        }
                    }
                    let lin_part: Option<ExprId> = if lin_terms.is_empty() {
                        None
                    } else if lin_terms.len() == 1 {
                        Some(lin_terms[0])
                    } else {
                        Some(arena.intern(ExprNode::SumOver { terms: lin_terms }))
                    };

                    // Combine: linear + nonlinear body (dropping a zero-constant
                    // body so a purely-linear defined var stays clean).
                    let defined_expr = match lin_part {
                        Some(lin) => {
                            if is_zero_constant(&arena, body) {
                                lin
                            } else {
                                arena.intern(ExprNode::BinaryOp {
                                    op: BinOp::Add,
                                    left: body,
                                    right: lin,
                                })
                            }
                        }
                        None => body,
                    };

                    // Push at the defined var's index. .nl numbers defined vars
                    // sequentially from n_vars, matching var_nodes' next slot.
                    if vi == var_nodes.len() {
                        var_nodes.push(defined_expr);
                    } else if vi < var_nodes.len() {
                        // Out-of-order redefinition would silently mis-map later
                        // references — refuse rather than guess.
                        return Err(NlParseError::Parse(format!(
                            "defined variable V{vi} redefines an existing index \
                             (expected the next sequential index {})",
                            var_nodes.len()
                        )));
                    } else {
                        return Err(NlParseError::Parse(format!(
                            "defined variable V{vi} is out of sequence (expected index {})",
                            var_nodes.len()
                        )));
                    }
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
                    let neg = arena.intern(ExprNode::UnaryOp {
                        op: UnOp::Neg,
                        operand: var_nodes[vi],
                    });
                    lin_terms.push(neg);
                } else {
                    let c = arena.intern(ExprNode::Constant(coeff));
                    let prod = arena.intern(ExprNode::BinaryOp {
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
                Some(arena.intern(ExprNode::SumOver { terms: lin_terms }))
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
                    arena.intern(ExprNode::BinaryOp {
                        op: BinOp::Add,
                        left: nl,
                        right: lin,
                    })
                }
            }
            (Some(nl), None) => nl,
            (None, Some(lin)) => lin,
            (None, None) => arena.intern(ExprNode::Constant(0.0)),
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
                    arena.intern(ExprNode::BinaryOp {
                        op: BinOp::Add,
                        left: nl,
                        right: lin,
                    })
                }
            }
            (Some(nl), None) => nl,
            (None, Some(lin)) => lin,
            (None, None) => arena.intern(ExprNode::Constant(0.0)),
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

    // Construction is complete: drop the intern table so downstream passes that
    // append nodes via `arena.add` (presolve, reformulation) see plain, unshared
    // append semantics — no lingering hash-consing behavior.
    arena.disable_interning();

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
///
/// Reads the file as raw bytes and inspects the leading magic character to
/// select the encoding: `g` = text/ASCII (supported), `b` = binary (not
/// supported — yields an explicit [`NlParseError::BinaryUnsupported`] rather
/// than a confusing UTF-8 decode error, since binary bodies embed raw IEEE-754
/// doubles that are not valid UTF-8).
pub fn parse_nl_file(path: &str) -> Result<ModelRepr, NlParseError> {
    let bytes = std::fs::read(path)
        .map_err(|e| NlParseError::Parse(format!("failed to read file '{path}': {e}")))?;

    // Detect the format from the first non-whitespace byte.
    if let Some(b'b') = bytes.iter().find(|b| !b.is_ascii_whitespace()) {
        return Err(NlParseError::BinaryUnsupported);
    }

    let content = std::str::from_utf8(&bytes).map_err(|e| {
        NlParseError::Parse(format!(
            "failed to read file '{path}': not valid UTF-8 text .nl ({e})"
        ))
    })?;
    parse_nl(content)
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
        s.push_str(" 0 0\n"); // nonlinear: 0 cons, 0 obj
        s.push_str(" 0 0 0\n"); // network
        s.push_str(" 0 0 0\n"); // nl var counts
        s.push_str(" 0 0 0 1\n"); // flags
        s.push_str(" 0 0\n"); // discrete vars
        s.push_str(" 2 2\n"); // nnz jacobian=2, nnz gradient=2
        s.push_str(" 0 0\n"); // max name lengths
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
        s.push_str(" 1 1\n"); // 1 nonlinear constraint, 1 nonlinear objective
        s.push_str(" 0 0\n");
        s.push_str(" 2 2 2\n"); // 2 nl vars in cons, 2 in objs, 2 in both
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: minimize, x^2 + y^2
        s.push_str("O0 0\n");
        s.push_str("o0\n"); // add
        s.push_str("o5\n"); // pow
        s.push_str("v0\n"); // x
        s.push_str("n2\n"); // 2
        s.push_str("o5\n"); // pow
        s.push_str("v1\n"); // y
        s.push_str("n2\n"); // 2
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
        s.push_str(" 0 1\n"); // 0 nonlinear constraints, 1 nonlinear objective
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n"); // 0 nl vars in cons, 2 in objs
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 2 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: minimize, exp(x) + log(y)
        s.push_str("O0 0\n");
        s.push_str("o0\n"); // add
        s.push_str("o44\n"); // exp
        s.push_str("v0\n"); // x
        s.push_str("o43\n"); // log
        s.push_str("v1\n"); // y
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
        s.push_str(" 1 1\n"); // 1 binary, 1 integer (linear)
        s.push_str(" 3 3\n"); // nnz
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
        s.push_str("0 0 10\n"); // z continuous
        s.push_str("0 0 1\n"); // x binary
        s.push_str("0 0 5\n"); // y integer
                               // k
        s.push_str("k2\n");
        s.push_str("1\n");
        s.push_str("2\n");
        // J0: 3*z + 1*x + 2*y
        s.push_str("J0 3\n");
        s.push_str("0 3\n"); // z
        s.push_str("1 1\n"); // x
        s.push_str("2 2\n"); // y
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
        s.push_str("4 5\n"); // equality: body == 5
        s.push_str("b\n");
        s.push_str("2 0\n"); // x >= 0
        s.push_str("2 0\n"); // y >= 0
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
        s.push_str("0 2 8\n"); // range: 2 <= body <= 8
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
        s.push_str("O0 1\n"); // 1 = maximize
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
        s.push_str(" 0 1\n"); // 1 nonlinear objective
        s.push_str(" 0 0\n");
        s.push_str(" 0 4 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: sum of x_i^2 using o54 (sumlist)
        s.push_str("O0 0\n");
        s.push_str("o54\n"); // sumlist
        s.push_str("4\n"); // 4 terms
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
            let model =
                parse_nl(&nl_fn()).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"));
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
        s.push_str("o16\n"); // unary neg
        s.push_str("v0\n"); // -x
        s.push_str("b\n");
        s.push_str("3\n"); // free
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
        s.push_str("o2\n"); // multiply
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
        s.push_str("o3\n"); // divide
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
        s.push_str("o41\n"); // sin
        s.push_str("v0\n");
        s.push_str("o46\n"); // cos
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
        s.push_str("o39\n"); // sqrt
        s.push_str("v0\n");
        s.push_str("b\n");
        s.push_str("2 0\n"); // x >= 0
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
        s.push_str("4 3.14\n"); // x fixed at 3.14
        s.push_str("1 100\n"); // y <= 100 (upper bound only)
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
        s.push_str("o42\n"); // log10
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
        s.push_str("o44\n"); // exp
        s.push_str("o0\n"); // add
        s.push_str("o5\n"); // pow
        s.push_str("v0\n"); // x
        s.push_str("n2\n"); // 2
        s.push_str("v1\n"); // y
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        let val = model.evaluate_objective(&[1.0, 2.0]);
        let expected = (3.0_f64).exp();
        assert!(
            (val - expected).abs() < 1e-10,
            "expected {expected}, got {val}"
        );
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
        s.push_str("1 10\n"); // c0: body <= 10
        s.push_str("2 -2\n"); // c1: body >= -2
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

    // ─── Test: binary .nl format yields explicit diagnostic ───

    #[test]
    fn test_binary_header_rejected() {
        // A 'b'-prefixed header signals the binary encoding.
        let err = parse_nl("b3 1 1 0\n").unwrap_err();
        assert!(matches!(err, NlParseError::BinaryUnsupported));
        let msg = err.to_string();
        assert!(msg.contains("binary .nl format not supported"), "{msg}");
    }

    #[test]
    fn test_parse_nl_file_binary_detected() {
        // Write a fake binary .nl: text header line plus a non-UTF-8 body byte.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("discopt_binary_test_{}.nl", std::process::id()));
        std::fs::write(&path, b"b3 0 1 0\t# problem\n\x00\xf0\x3f").unwrap();
        let result = parse_nl_file(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(NlParseError::BinaryUnsupported)));
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
        s.push_str("o1\n"); // subtract
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        let model = parse_nl(&s).unwrap();
        let val = model.evaluate_objective(&[10.0, 3.0]);
        assert!((val - 7.0).abs() < 1e-12);
    }

    // ─── C-5: floor/ceil/round/trunc/intdiv must REFUSE, not silently misparse ───
    //
    // Before this fix each of these opcodes was silently rewritten to a different
    // operator: floor/ceil/round/trunc → identity, intdiv → real division. That made
    // the parser accept a *different* problem than the file encoded, and the solver
    // then certified the wrong optimum. The sound behavior (no IR/relaxation support
    // exists) is a loud refusal. These tests fail on the pre-fix code (parse_nl
    // returns Ok with a wrong expression) and pass after (parse_nl returns the
    // UnsupportedOpcode error naming the operator).

    /// Build a single-objective .nl whose objective is `<unary-op>(v0)`.
    fn nl_with_unary_obj(opcode_line: &str) -> String {
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
        s.push_str(opcode_line);
        s.push_str("v0\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s
    }

    /// Build a single-objective .nl whose objective is `<binary-op>(v0, v1)`.
    fn nl_with_binary_obj(opcode_line: &str) -> String {
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
        s.push_str(opcode_line);
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s
    }

    fn assert_unsupported(content: &str, expect_name: &str, expect_opcode: i32) {
        let err = parse_nl(content)
            .err()
            .unwrap_or_else(|| panic!("expected UnsupportedOpcode for o{expect_opcode}, got Ok"));
        match err {
            NlParseError::UnsupportedOpcode { name, opcode } => {
                assert_eq!(name, expect_name, "operator name");
                assert_eq!(opcode, expect_opcode, "opcode number");
            }
            other => panic!("expected UnsupportedOpcode({expect_name}), got {other:?}"),
        }
        // The Display message must name the operator so the failure is actionable.
        let msg = parse_nl(content).unwrap_err().to_string();
        assert!(
            msg.contains(expect_name) && msg.contains("unsupported"),
            "message should name '{expect_name}' and say unsupported, got: {msg}"
        );
    }

    #[test]
    fn test_floor_opcode_refused_not_identity() {
        // o13 floor: pre-fix parsed to identity (floor(v0) == v0 == 1.7 at v0=1.7).
        assert_unsupported(&nl_with_unary_obj("o13\n"), "floor", 13);
    }

    #[test]
    fn test_ceil_opcode_refused_not_identity() {
        // o14 ceil: pre-fix parsed to identity.
        assert_unsupported(&nl_with_unary_obj("o14\n"), "ceil", 14);
    }

    #[test]
    fn test_round_opcode_refused_not_identity() {
        // o57 round (binary): pre-fix returned the left operand (identity).
        assert_unsupported(&nl_with_binary_obj("o57\n"), "round", 57);
    }

    #[test]
    fn test_trunc_opcode_refused_not_identity() {
        // o58 trunc (binary): pre-fix returned the left operand (identity).
        assert_unsupported(&nl_with_binary_obj("o58\n"), "trunc", 58);
    }

    #[test]
    fn test_intdiv_opcode_refused_not_real_div() {
        // o55 intdiv: pre-fix rewrote to real Div (7 intdiv 2 -> 3.5 instead of 3).
        assert_unsupported(&nl_with_binary_obj("o55\n"), "intdiv", 55);
    }

    // ─── C-8: common opcodes o76/o77/o78 (power forms), o11/o12 (min/max),
    //          o48 (atan2 — refuse), o35 (if — refuse) ─────────────────────
    //
    // Pre-fix, every one of these hit the `_ => UnknownOpcode` fallthrough and the
    // parse aborted with a bare "unknown opcode" error, rejecting a large slice of
    // AMPL/Pyomo-exported .nl (AMPL emits o77 for `x**2` routinely, o76 for `x**n`).
    // Fixed opcodes now parse to the correct IR; genuinely unrepresentable ones
    // (atan2, if) refuse loudly via UnsupportedOpcode (never silent misparse).

    /// Build a single-objective .nl whose objective is an n-ary op over v0..v{n-1}.
    /// `opcode` is e.g. "o11" (minlist) or "o12" (maxlist).
    fn nl_with_nary_obj(opcode: &str, n: usize) -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(&format!(" {n} 0 1 0 0\n"));
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(&format!(" 0 {n} 0\n"));
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str(opcode);
        s.push('\n');
        s.push_str(&format!("{n}\n")); // n-ary count line
        for i in 0..n {
            s.push_str(&format!("v{i}\n"));
        }
        s.push_str("b\n");
        for _ in 0..n {
            s.push_str("3\n");
        }
        s
    }

    #[test]
    fn test_o77_2pow_is_square_not_unknown() {
        // o77 (OP2POW): unary `x^2`. Pre-fix: UnknownOpcode(77) -> parse fails.
        // Post-fix: parses to Pow(v0, 2); (3)^2 = 9.
        let model = parse_nl(&nl_with_unary_obj("o77\n")).expect("o77 must parse");
        let val = model.evaluate_objective(&[3.0]);
        assert!(
            (val - 9.0).abs() < 1e-12,
            "o77 x^2 at 3 must be 9, got {val}"
        );
    }

    #[test]
    fn test_o76_1pow_is_power_with_constant_exponent() {
        // o76 (OP1POW): `base ^ const`. Stream: o76, <base expr>, n<const>.
        // Build min x^3 as objective: o76 / v0 / n3. (2)^3 = 8.
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
        s.push_str("o76\n");
        s.push_str("v0\n");
        s.push_str("n3\n");
        s.push_str("b\n");
        s.push_str("3\n");
        let model = parse_nl(&s).expect("o76 must parse");
        let val = model.evaluate_objective(&[2.0]);
        assert!(
            (val - 8.0).abs() < 1e-12,
            "o76 x^3 at 2 must be 8, got {val}"
        );
    }

    #[test]
    fn test_o78_cpow_is_constant_base_to_var_power() {
        // o78 (OPCPOW): `const ^ exp`. Stream: o78, n<const>, <exp expr>.
        // Build min 2^x: o78 / n2 / v0. 2^3 = 8.
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
        s.push_str("o78\n");
        s.push_str("n2\n");
        s.push_str("v0\n");
        s.push_str("b\n");
        s.push_str("3\n");
        let model = parse_nl(&s).expect("o78 must parse");
        let val = model.evaluate_objective(&[3.0]);
        assert!(
            (val - 8.0).abs() < 1e-12,
            "o78 2^x at 3 must be 8, got {val}"
        );
    }

    #[test]
    fn test_o11_minlist_is_minimum() {
        // o11 (MINLIST): n-ary min. min over {v0,v1,v2} at (5,2,8) = 2.
        let model = parse_nl(&nl_with_nary_obj("o11", 3)).expect("o11 must parse");
        let val = model.evaluate_objective(&[5.0, 2.0, 8.0]);
        assert!((val - 2.0).abs() < 1e-12, "o11 min must be 2, got {val}");
    }

    #[test]
    fn test_o12_maxlist_is_maximum() {
        // o12 (MAXLIST): n-ary max. max over {v0,v1,v2} at (5,2,8) = 8.
        let model = parse_nl(&nl_with_nary_obj("o12", 3)).expect("o12 must parse");
        let val = model.evaluate_objective(&[5.0, 2.0, 8.0]);
        assert!((val - 8.0).abs() < 1e-12, "o12 max must be 8, got {val}");
    }

    #[test]
    fn test_o48_atan2_refused_not_misparsed() {
        // o48 (atan2): binary; no Atan2 in the IR. Must refuse loudly rather than
        // silently drop an operand or misuse single-arg atan.
        assert_unsupported(&nl_with_binary_obj("o48\n"), "atan2", 48);
    }

    #[test]
    fn test_o35_if_refused_not_misparsed() {
        // o35 (OPIFnl): if-then-else; not representable. Refuse loudly.
        // 3 operands (cond, then, else); the error aborts before they are consumed.
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 3 0 1 0 0\n");
        s.push_str(" 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 3 0\n");
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        s.push_str("O0 0\n");
        s.push_str("o35\n");
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("v2\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s.push_str("3\n");
        assert_unsupported(&s, "if", 35);
    }

    // ─── C-7: defined variables (V segments) are inlined, not discarded ─────
    //
    // Pre-fix, a V segment was parsed then dropped (never pushed to `var_nodes`),
    // so any later `v<idx>` reference to a defined var (idx >= n_vars) hit the
    // "variable index out of range" error and the whole instance failed to parse.
    // Post-fix, the defined var's expression (linear part + nonlinear body) is
    // inlined into `var_nodes` so references resolve to the subexpression.

    /// A .nl with one defined variable `V2 = 2*v0 + v1^2` (n_vars=2, so the
    /// defined var occupies index 2), referenced by the objective as `min v2`.
    ///   objective(x, y) = 2x + y^2
    fn defined_var_nl() -> String {
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 2 0 1 0 0\n"); // 2 real vars, 0 cons, 1 obj
        s.push_str(" 0 1\n"); // 0 nl cons, 1 nl obj
        s.push_str(" 0 0\n");
        s.push_str(" 0 2 0\n"); // 2 nl vars in objs
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 1 0 0 0\n"); // 1 common expr (defined var referenced in objs)
                                    // V2: defined var, 1 linear term (2*v0), then nonlinear body v1^2.
        s.push_str("V2 1 0\n");
        s.push_str("0 2\n"); // linear part: coeff 2 on v0
        s.push_str("o5\n"); // nonlinear body: pow
        s.push_str("v1\n");
        s.push_str("n2\n");
        // O0: minimize, objective = v2 (the defined variable)
        s.push_str("O0 0\n");
        s.push_str("v2\n");
        s.push_str("b\n");
        s.push_str("3\n");
        s.push_str("3\n");
        s
    }

    #[test]
    fn test_defined_variable_inlined_and_evaluated() {
        // Pre-fix: parse_nl errors ("variable index 2 out of range").
        // Post-fix: v2 inlines to 2*v0 + v1^2; at (x=3, y=4): 6 + 16 = 22.
        let model = parse_nl(&defined_var_nl()).expect("defined-var .nl must parse");
        let val = model.evaluate_objective(&[3.0, 4.0]);
        assert!(
            (val - 22.0).abs() < 1e-12,
            "defined var 2*x + y^2 at (3,4) must be 22, got {val}"
        );
        // Independent-evaluation cross-check at two more points.
        let v2 = model.evaluate_objective(&[0.0, 5.0]); // 0 + 25
        assert!((v2 - 25.0).abs() < 1e-12, "at (0,5) expected 25, got {v2}");
        let v3 = model.evaluate_objective(&[-1.0, 2.0]); // -2 + 4
        assert!((v3 - 2.0).abs() < 1e-12, "at (-1,2) expected 2, got {v3}");
    }

    #[test]
    fn test_defined_variable_chained_reference() {
        // A defined var referencing an EARLIER defined var must resolve.
        // n_vars=1 (v0). V1 = v0 + 1 (index 1). V2 = v1 * v1 (index 2, refs v1).
        // objective = v2 = (v0 + 1)^2. At v0=3: 16.
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
        s.push_str(" 0 2 0 0 0\n"); // 2 common exprs
                                    // V1 = v0 + 1  (0 linear terms; nonlinear body o0 v0 n1)
        s.push_str("V1 0 0\n");
        s.push_str("o0\n");
        s.push_str("v0\n");
        s.push_str("n1\n");
        // V2 = v1 * v1  (references the earlier defined var v1)
        s.push_str("V2 0 0\n");
        s.push_str("o2\n");
        s.push_str("v1\n");
        s.push_str("v1\n");
        s.push_str("O0 0\n");
        s.push_str("v2\n");
        s.push_str("b\n");
        s.push_str("3\n");
        let model = parse_nl(&s).expect("chained defined-var .nl must parse");
        let val = model.evaluate_objective(&[3.0]);
        assert!(
            (val - 16.0).abs() < 1e-12,
            "(v0+1)^2 at 3 must be 16, got {val}"
        );
    }

    // ─── C-9: objective-only nonlinear integer block when nlvo > nlvc ───────
    //
    // AMPL edge case (ex1252a): when more nonlinear vars live in objectives than
    // constraints (nlvb < nlvc < nlvo), the objective-only nonlinear group is
    // [nlvc, nlvo) and its integer tail is the LAST nlvoi of that group. The old
    // `(nlvo - nlvb)`-sized formula placed the tail out of range, silently leaving
    // those variables Continuous — relaxing integrality, enlarging the feasible set,
    // and admitting a false-feasible incumbent below the true optimum. Locked here.
    #[test]
    fn test_nlvo_gt_nlvc_objective_only_integer_block_typed() {
        // Header shape: nlvb=1 < nlvc=2 < nlvo=4, nlvoi=2 (last 2 of the obj-only
        // group [2,4) are integer). n_vars=4. Expect vars 2,3 Integer; 0,1 Continuous.
        let mut s = String::new();
        s.push_str("g3 1 1 0\n");
        s.push_str(" 4 1 1 0 0\n"); // 4 vars, 1 con, 1 obj
        s.push_str(" 1 1\n"); // 1 nl con, 1 nl obj
        s.push_str(" 0 0\n");
        s.push_str(" 2 4 1\n"); // nlvc=2, nlvo=4, nlvb=1
        s.push_str(" 0 0 0 1\n");
        s.push_str(" 0 0 0 0 2\n"); // line 6: ... nlvoi=2 (obj-only nl integers)
        s.push_str(" 4 4\n");
        s.push_str(" 0 0\n");
        s.push_str(" 0 0 0 0 0\n");
        // O0: minimize v0 + v1 + v2 + v3 (nl so the vars register as nonlinear)
        s.push_str("O0 0\n");
        s.push_str("o54\n");
        s.push_str("4\n");
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("v2\n");
        s.push_str("v3\n");
        // C0: nl part references v0,v1 (nlvc=2)
        s.push_str("C0\n");
        s.push_str("o0\n");
        s.push_str("v0\n");
        s.push_str("v1\n");
        s.push_str("x4\n");
        s.push_str("0 0\n");
        s.push_str("1 0\n");
        s.push_str("2 0\n");
        s.push_str("3 0\n");
        s.push_str("r\n");
        s.push_str("1 10\n");
        s.push_str("b\n");
        s.push_str("0 0 5\n");
        s.push_str("0 0 5\n");
        s.push_str("0 0 5\n");
        s.push_str("0 0 5\n");
        let model = parse_nl(&s).expect("nlvo>nlvc .nl must parse");
        assert_eq!(model.variables[0].var_type, VarType::Continuous, "v0");
        assert_eq!(model.variables[1].var_type, VarType::Continuous, "v1");
        assert_eq!(
            model.variables[2].var_type,
            VarType::Integer,
            "v2 must be integer (obj-only nl integer tail); regression would leave it Continuous"
        );
        assert_eq!(model.variables[3].var_type, VarType::Integer, "v3");
    }

    // ─── Corpus regression: all in-repo MINLPLib .nl still parse ────────────
    #[test]
    fn test_minlplib_corpus_all_parse() {
        // Zero-regression guard for the parser changes (C-7/C-8): every .nl in the
        // 61-file corpus must still parse. Path is resolved relative to the crate.
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../python/tests/data/minlplib_nl"
        );
        let entries =
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read corpus dir {dir}: {e}"));
        let mut n_parsed = 0usize;
        for entry in entries {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("nl") {
                continue;
            }
            let p = path.to_string_lossy().to_string();
            parse_nl_file(&p).unwrap_or_else(|e| panic!("corpus file {p} failed to parse: {e}"));
            n_parsed += 1;
        }
        assert!(
            n_parsed >= 61,
            "expected >= 61 corpus .nl, parsed {n_parsed}"
        );
    }
}
