//! AMPL `.nl` writer: [`ModelRepr`] to file text.
//!
//! The inverse of [`crate::nl_parser`], sharing its opcode table. It exists
//! because writing `.nl` was the whole remaining performance gap for external
//! solvers: measured, discopt's Python writer costs **16.17 µs/row of a 16.21
//! µs/row** model-to-file pipeline, against oximo's 0.86 (`.nl` is a Rust
//! writer there) — 18.8x, and 99.8% of discopt's total. See
//! `docs/dev/performance-plan.md` §43.
//!
//! # Why [`ModelRepr`] and not POUNCE's `NlProblem`
//!
//! `NlProblem` looks like the natural substrate — discopt already builds one on
//! every solve — but it carries **no integrality**: no `var_type`, no integer
//! flag, and its header census counts only nonlinear variables, not the
//! integer blocks. That is correct for an NLP solver, which cannot branch, and
//! fatal for a writer: a MINLP written through it becomes a *silent continuous
//! relaxation of the user's model*. `ModelRepr` carries `VarInfo.var_type`,
//! names, bounds and the full arena, so nothing is lost (§44).
//!
//! # Canonical variable order
//!
//! `.nl` requires nonlinear variables at the lowest indices, grouped
//! `[nl in both | nl in cons only | nl in objs only | linear]`, with the
//! discrete members **last within each group**, and the linear tail ordered
//! `[continuous | binary | integer]`. This is not cosmetic: under any other
//! order a discrete variable appearing nonlinearly is declared through
//! `nbv`/`niv`, and AMPL-compatible solvers then read it as **continuous** and
//! silently solve a relaxation (discopt issue #210).

use std::collections::{BTreeMap, BTreeSet};

use crate::expand::{
    expand, ExpandError, ScalarProgram, OP_ABS, OP_ADD, OP_CONST, OP_DIV, OP_FUNC_BASE, OP_MUL,
    OP_NEG, OP_POW, OP_SUB, OP_SUMOVER, OP_VAR,
};
use crate::expr::{ConstraintSense, ModelRepr, ObjectiveSense, VarType};

// `.nl` opcodes (Gay, "Writing .nl Files", 2005, Tables 3-4). Kept in step with
// `python/discopt/export/nl.py`, which this mirrors.
const NL_ADD: i32 = 0;
const NL_SUB: i32 = 1;
const NL_MUL: i32 = 2;
const NL_DIV: i32 = 3;
const NL_POW: i32 = 5;
const NL_ABS: i32 = 15;
const NL_NEG: i32 = 16;
const NL_SUMLIST: i32 = 54;

/// `.nl` opcode for a [`crate::expand`] `OP_FUNC_BASE + code` function.
/// `expand.rs::func_code`'s value for `MathFunc::Log2`.
///
/// `.nl` has no base-2 log opcode, so this one is not in the table below: it is
/// rewritten at the emit site as `log(x) / ln 2`, exactly as `export/nl.py`
/// writes it. Named rather than spelled `2` inline so the two places that care
/// are greppable together.
const FUNC_LOG2: i32 = 2;

/// `expand.rs::func_code`'s value for `MathFunc::Log`, used by the `log2` rewrite.
const FUNC_LOG: i32 = 1;

/// `expand.rs::func_code`'s value for `MathFunc::Tan`.
///
/// `.nl` HAS a tan opcode (`o38`), but `export/nl.py` has always written
/// `sin(x) / cos(x)` instead -- one of its deliberate rewrites, alongside
/// `log2`, `log1p`, `sigmoid` and `softplus`. This writer's whole safety
/// argument is that it is byte-identical to that one, so it rewrites too rather
/// than emitting the native opcode. Changing what discopt writes for `tan` is a
/// user-visible output change and belongs to whoever measures it, not here.
const FUNC_TAN: i32 = 7;

/// `expand.rs::func_code` values for the functions the rewrites above call.
const FUNC_SIN: i32 = 5;
const FUNC_COS: i32 = 6;

/// `expand.rs::func_code` value -> `.nl` opcode.
///
/// MUST stay in step with `expand.rs::func_code`, which decides what reaches
/// this writer at all: a code that `func_code` admits and this table omits used
/// to hit `.expect("mapped function")` and **panic**, and a `PanicException`
/// derives from `BaseException`, so `export/nl.py`'s `except Exception` fallback
/// could not catch it and the panic reached the user. That is how `log2` shipped
/// broken. Returning `None` now refuses instead, which the Python writer picks
/// up. `nl_func_opcode_covers_every_expanded_func` asserts the two agree.
fn nl_func_opcode(code: i32) -> Option<i32> {
    Some(match code {
        0 => 44,  // exp
        1 => 43,  // log
        3 => 42,  // log10
        4 => 39,  // sqrt
        5 => 41,  // sin
        6 => 46,  // cos
        8 => 49,  // atan
        9 => 40,  // sinh
        10 => 45, // cosh
        11 => 51, // asin
        _ => return None,
    })
}

/// Format a float the way Python's `repr` does, because this writer's output is
/// diffed byte-for-byte against the Python writer it replaces. Rust's `{}`
/// agrees on shortest-round-trip digits but not on presentation: it renders an
/// integral value as `4` where Python gives `4.0`, and `1e20` as
/// `100000000000000000000` where Python gives `1e+20`.
fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = v.abs();
    // Python switches to exponent form below 1e-4 and at or above 1e16.
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let s = format!("{v:e}"); // e.g. "1e20", "1.5e-7"
        let (mant, exp) = s.split_once('e').expect("exponent form");
        let (sign, digits) = match exp.strip_prefix('-') {
            Some(d) => ('-', d),
            None => ('+', exp),
        };
        return format!("{mant}e{sign}{digits:0>2}");
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// One flat scalar variable.
struct FlatVar {
    var_type: VarType,
    lb: f64,
    ub: f64,
}

/// A row split into `linear + nonlinear + constant`, mirroring
/// `export/nl.py::_split_expr`.
///
/// The constant is peeled out rather than folded into the nonlinear body so the
/// caller can carry it in the `r`-section bound. That is what lets `x + y <= 3`
/// — which discopt normalises to `x + y - 3 <= 0` — be written as a genuinely
/// *linear* row with an `n0` body, which the ASL convention requires (nonlinear
/// rows must be the first `n_nl_cons` and carry the only non-`n0` bodies).
#[derive(Clone)]
struct Split {
    linear: BTreeMap<usize, f64>,
    nonlinear: Option<i64>,
    constant: f64,
}

impl ScalarProgram {
    fn unseal(&mut self) {
        self.args_ptr.pop();
    }

    fn seal(&mut self) {
        self.args_ptr.push(self.args_flat.len() as i64);
    }

    fn emit(&mut self, op: i32, a: i64, b: i64, k: f64) -> i64 {
        self.args_ptr.push(self.args_flat.len() as i64);
        self.op.push(op);
        self.a.push(a);
        self.b.push(b);
        self.k.push(k);
        (self.op.len() - 1) as i64
    }

    /// Operands of n-ary instruction `i`.
    ///
    /// The end sentinel is popped while the writer appends instructions of its
    /// own, so the last instruction has no `args_ptr[i + 1]` to read -- its
    /// operands run to the end of `args_flat`. Indexing blindly panicked here.
    fn args_of(&self, i: usize) -> &[i64] {
        let lo = self.args_ptr[i] as usize;
        let hi = match self.args_ptr.get(i + 1) {
            Some(v) => *v as usize,
            None => self.args_flat.len(),
        };
        &self.args_flat[lo..hi]
    }
}

/// Separate an instruction tree into linear terms, a nonlinear remainder and a
/// constant, exactly as `export/nl.py::_collect_linear` does.
///
/// Driven by an explicit `(instruction, coefficient)` stack: a body built from
/// `sum()` of many terms is a deep chain, and the nonlinear output is
/// order-sensitive because it is folded into a sum afterwards, so children are
/// pushed reversed to preserve left-to-right pre-order.
fn split(prog: &mut ScalarProgram, root: i64) -> Split {
    let mut linear: BTreeMap<usize, f64> = BTreeMap::new();
    let mut nonlinear: Vec<(f64, i64)> = Vec::new();
    let mut constant = 0.0f64;
    let mut stack: Vec<(i64, f64)> = vec![(root, 1.0)];

    while let Some((node, c)) = stack.pop() {
        let i = node as usize;
        let op = prog.op[i];
        match op {
            OP_CONST => {
                let v = prog.k[i];
                if v != 0.0 {
                    constant += v * c;
                }
            }
            OP_VAR => {
                let slot = prog.k[i] as usize;
                *linear.entry(slot).or_insert(0.0) += c;
            }
            OP_ADD => {
                stack.push((prog.b[i], c));
                stack.push((prog.a[i], c));
            }
            OP_SUB => {
                stack.push((prog.b[i], -c));
                stack.push((prog.a[i], c));
            }
            OP_MUL => {
                let (l, r) = (prog.a[i] as usize, prog.b[i] as usize);
                if prog.op[l] == OP_CONST {
                    stack.push((prog.b[i], c * prog.k[l]));
                } else if prog.op[r] == OP_CONST {
                    stack.push((prog.a[i], c * prog.k[r]));
                } else {
                    nonlinear.push((c, node));
                }
            }
            OP_NEG => stack.push((prog.a[i], -c)),
            OP_SUMOVER => {
                let args: Vec<i64> = prog.args_of(i).to_vec();
                for t in args.into_iter().rev() {
                    stack.push((t, c));
                }
            }
            _ => nonlinear.push((c, node)),
        }
    }

    let nl = if nonlinear.is_empty() {
        None
    } else {
        let mut terms: Vec<i64> = Vec::with_capacity(nonlinear.len());
        for (c, node) in nonlinear {
            if c == 1.0 {
                terms.push(node);
            } else {
                let k = prog.emit(OP_CONST, -1, -1, c);
                terms.push(prog.emit(OP_MUL, k, node, 0.0));
            }
        }
        let mut acc = terms[0];
        for t in &terms[1..] {
            acc = prog.emit(OP_ADD, acc, *t, 0.0);
        }
        Some(acc)
    };
    Split {
        linear,
        nonlinear: nl,
        constant,
    }
}

/// Flat variable slots referenced anywhere under `root`.
fn vars_under(prog: &ScalarProgram, root: i64, out: &mut BTreeSet<usize>) {
    let mut stack = vec![root];
    let mut seen: BTreeSet<i64> = BTreeSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        let i = node as usize;
        if prog.op[i] == OP_VAR {
            out.insert(prog.k[i] as usize);
            continue;
        }
        if prog.a[i] >= 0 {
            stack.push(prog.a[i]);
        }
        if prog.b[i] >= 0 {
            stack.push(prog.b[i]);
        }
        for t in prog.args_of(i) {
            stack.push(*t);
        }
    }
}

/// One pending item in [`write_expr`]'s stack: a subtree still to emit, or a
/// ready-made token to emit once whatever was pushed above it has drained.
enum Emit {
    Node(i64),
    Text(String),
}

/// Emit one expression tree in `.nl` prefix notation.
///
/// Fallible: a function code this writer has no opcode for is a **refusal**, so
/// `export/nl.py` falls back to the Python writer, which handles it.
fn write_expr(
    prog: &ScalarProgram,
    root: i64,
    remap: &[usize],
    out: &mut String,
) -> Result<(), ExpandError> {
    // Explicit stack, children pushed reversed so they pop in prefix order:
    // a `sum()` body is deep enough that recursion is not safe here.
    let mut stack = vec![Emit::Node(root)];
    while let Some(item) = stack.pop() {
        let node = match item {
            Emit::Text(t) => {
                out.push_str(&t);
                continue;
            }
            Emit::Node(n) => n,
        };
        let i = node as usize;
        let op = prog.op[i];
        match op {
            OP_CONST => out.push_str(&format!("n{}\n", py_float(prog.k[i]))),
            OP_VAR => out.push_str(&format!("v{}\n", remap[prog.k[i] as usize])),
            OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_POW => {
                let code = match op {
                    OP_ADD => NL_ADD,
                    OP_SUB => NL_SUB,
                    OP_MUL => NL_MUL,
                    OP_DIV => NL_DIV,
                    _ => NL_POW,
                };
                out.push_str(&format!("o{code}\n"));
                stack.push(Emit::Node(prog.b[i]));
                stack.push(Emit::Node(prog.a[i]));
            }
            OP_NEG | OP_ABS => {
                let code = if op == OP_NEG { NL_NEG } else { NL_ABS };
                out.push_str(&format!("o{code}\n"));
                stack.push(Emit::Node(prog.a[i]));
            }
            OP_SUMOVER => {
                let args = prog.args_of(i);
                match args.len() {
                    0 => out.push_str("n0\n"),
                    1 => stack.push(Emit::Node(args[0])),
                    n => {
                        out.push_str(&format!("o{NL_SUMLIST}\n{n}\n"));
                        for t in args.iter().rev() {
                            stack.push(Emit::Node(*t));
                        }
                    }
                }
            }
            c if c >= OP_FUNC_BASE && c - OP_FUNC_BASE == FUNC_TAN => {
                // `export/nl.py`: `o3 o41 <arg> o46 <arg>` -- sin(arg)/cos(arg),
                // with the argument subtree emitted twice, exactly as it does.
                out.push_str(&format!("o{NL_DIV}\n"));
                out.push_str(&format!("o{}\n", nl_func_opcode(FUNC_SIN).expect("sin")));
                stack.push(Emit::Node(prog.a[i]));
                stack.push(Emit::Text(format!(
                    "o{}\n",
                    nl_func_opcode(FUNC_COS).expect("cos")
                )));
                stack.push(Emit::Node(prog.a[i]));
            }
            c if c >= OP_FUNC_BASE && c - OP_FUNC_BASE == FUNC_LOG2 => {
                // `.nl` has no base-2 log. `export/nl.py` writes
                // `o3 o43 <arg> n0.6931471805599453` -- log(arg) / ln 2 -- and
                // this must match it byte for byte, so the divisor is emitted
                // AFTER the argument subtree drains, via a text item.
                out.push_str(&format!("o{NL_DIV}\n"));
                out.push_str(&format!("o{}\n", nl_func_opcode(FUNC_LOG).expect("log")));
                stack.push(Emit::Text(format!(
                    "n{}\n",
                    py_float(std::f64::consts::LN_2)
                )));
                stack.push(Emit::Node(prog.a[i]));
            }
            c if c >= OP_FUNC_BASE => {
                let fc = c - OP_FUNC_BASE;
                // Refusal, not a panic: `export/nl.py` falls back to the Python
                // writer, which covers everything this table does not.
                let code = nl_func_opcode(fc).ok_or_else(|| {
                    ExpandError(format!(
                        "nl_writer has no .nl opcode for expanded function code {fc}; \
                         falling back to the Python writer"
                    ))
                })?;
                out.push_str(&format!("o{code}\n"));
                stack.push(Emit::Node(prog.a[i]));
            }
            _ => unreachable!("unsupported opcode {op} reached the writer"),
        }
    }
    Ok(())
}

/// Write `repr` as AMPL `.nl` text.
///
/// `n_builder_constraints` is how many of the LEADING entries of
/// `repr.constraints` came from the Rust model builder (`add_linear_constraints`
/// and the `Model.constraint` linear fast path). `model_to_repr` clones the
/// builder's constraints first and appends the expression rows after, so those
/// rows lead the arena — while every Python writer (`export/nl.py`, `lp.py`,
/// `mps.py`, `gams.py`) emits `model._constraints` first and the builder rows
/// after. Left alone the two would write the same model with its rows
/// **permuted**, and row order is not cosmetic: it is how a solver's `.sol`
/// duals map back to constraints.
///
/// So the rows are reordered here to the Python writers' order rather than the
/// Python writers being changed to the arena's. Four writers share that order
/// and every `.nl`/`.lp`/`.mps`/`.gms` discopt has already emitted uses it;
/// changing it would silently redefine what a stored file means. Reordering
/// here costs one `O(rows)` index permutation over data already materialised.
///
/// Pass `0` for a model with no builder rows (every arena constraint is then an
/// expression row and the permutation is the identity).
pub fn write_nl(
    repr: &ModelRepr,
    model_name: &str,
    n_builder_constraints: usize,
) -> Result<String, ExpandError> {
    let mut prog = expand(repr)?;
    prog.unseal();

    // ── flat variables, in declaration order (the order `expand` emits slots in)
    let n_total: usize = repr.n_vars;
    let mut flat: Vec<FlatVar> = Vec::with_capacity(n_total);
    for v in &repr.variables {
        for e in 0..v.size {
            let pick = |xs: &Vec<f64>| -> f64 {
                if e < xs.len() {
                    xs[e]
                } else if xs.is_empty() {
                    0.0
                } else {
                    xs[0]
                }
            };
            flat.push(FlatVar {
                var_type: v.var_type,
                lb: pick(&v.lb),
                ub: pick(&v.ub),
            });
        }
    }
    if flat.len() != n_total {
        return Err(ExpandError(format!(
            "variable blocks total {} scalars but n_vars is {n_total}",
            flat.len()
        )));
    }

    // ── split every row and the objective
    let obj_root = prog.objective_root;
    let mut obj_split = split(&mut prog, obj_root);
    // A constraint carries its split-out constant in the `r`-section bound, but
    // an objective has no bound to carry it -- so it goes back into the body, as
    // `export/nl.py::_attach_const` does. Dropping it silently shifts the
    // exported objective by that constant, and an objective that is ONLY a
    // constant must still be written (and counted as a nonlinear objective,
    // since its body is not `n0`).
    if obj_split.constant != 0.0 {
        let c = prog.emit(OP_CONST, -1, -1, obj_split.constant);
        obj_split.nonlinear = Some(match obj_split.nonlinear {
            Some(nl) => prog.emit(OP_ADD, nl, c, 0.0),
            None => c,
        });
        obj_split.constant = 0.0;
    }
    let row_roots = prog.row_roots.clone();
    let mut rows: Vec<Split> = Vec::with_capacity(row_roots.len());
    for r in &row_roots {
        rows.push(split(&mut prog, *r));
    }

    // `row_source[r]` is the index in `repr.constraints` that scalar row `r` came
    // from. An array-valued body is ONE constraint and many rows, so this is not
    // a one-to-one zip -- `rows_per_constraint` carries the fan-out.
    let mut row_source: Vec<usize> = Vec::with_capacity(rows.len());
    for (ci, n) in prog.rows_per_constraint.iter().enumerate() {
        row_source.extend(std::iter::repeat(ci).take(*n));
    }
    if row_source.len() != rows.len() {
        return Err(ExpandError(format!(
            "row map covers {} rows but {} were expanded",
            row_source.len(),
            rows.len()
        )));
    }

    // A builder row's EXPLICIT ZERO coefficients are dropped, because the Python
    // writer's `_decompose_builder_blocks` drops them (`if coeff == 0.0:
    // continue`) before they ever reach its linear map. A stored zero therefore
    // produces no `J` entry there and, if it was the row's only entry, an empty
    // row -- while the arena path faithfully builds a `Constant(0.0) * Var` node
    // and reports a zero coefficient.
    //
    // This is a real asymmetry in the Python writer, not a tidy rule: its
    // EXPRESSION path keeps a zero coefficient (an explicit `0.0 * x` term
    // survives `_collect_linear`). So the drop is scoped to builder rows, which
    // is exactly what `row_source` identifies. Matching the existing output is
    // the requirement; regularising the asymmetry would change bytes discopt has
    // already written.
    //
    // Missing this cost a regression: 129bf02 routed builder rows through this
    // writer and `test_builder_linear_block_skips_zero_coeff_and_empty_row`
    // started failing, because no shape in the byte-diff suite had a stored zero.
    if n_builder_constraints > 0 {
        for (r, &ci) in row_source.iter().enumerate() {
            if ci < n_builder_constraints {
                rows[r].linear.retain(|_, coeff| *coeff != 0.0);
            }
        }
    }

    // Builder rows lead the arena and trail every Python writer's output; put
    // them back where the Python writers put them (see this function's docs).
    if n_builder_constraints > 0 {
        if n_builder_constraints > repr.constraints.len() {
            return Err(ExpandError(format!(
                "n_builder_constraints is {} but the model has {} constraints",
                n_builder_constraints,
                repr.constraints.len()
            )));
        }
        // Stable within each group, so the relative order of the expression rows
        // and of the builder rows is untouched.
        let mut order: Vec<usize> = Vec::with_capacity(rows.len());
        for (r, &ci) in row_source.iter().enumerate() {
            if ci >= n_builder_constraints {
                order.push(r);
            }
        }
        for (r, &ci) in row_source.iter().enumerate() {
            if ci < n_builder_constraints {
                order.push(r);
            }
        }
        debug_assert_eq!(order.len(), rows.len());
        let mut permuted_rows: Vec<Split> = Vec::with_capacity(rows.len());
        let mut permuted_source: Vec<usize> = Vec::with_capacity(rows.len());
        for &r in &order {
            permuted_rows.push(rows[r].clone());
            permuted_source.push(row_source[r]);
        }
        rows = permuted_rows;
        row_source = permuted_source;
    }

    // ── canonical variable order (see the module docs; issue #210)
    let mut nl_cons: BTreeSet<usize> = BTreeSet::new();
    for row in &rows {
        if let Some(nl) = row.nonlinear {
            vars_under(&prog, nl, &mut nl_cons);
        }
    }
    let mut nl_objs: BTreeSet<usize> = BTreeSet::new();
    if let Some(nl) = obj_split.nonlinear {
        vars_under(&prog, nl, &mut nl_objs);
    }
    let nl_both: BTreeSet<usize> = nl_cons.intersection(&nl_objs).copied().collect();
    let cons_only: Vec<usize> = nl_cons.difference(&nl_both).copied().collect();
    let objs_only: Vec<usize> = nl_objs.difference(&nl_both).copied().collect();

    let is_discrete = |i: usize| matches!(flat[i].var_type, VarType::Binary | VarType::Integer);
    let split_group = |idxs: &[usize]| -> (Vec<usize>, Vec<usize>) {
        let mut cont = Vec::new();
        let mut disc = Vec::new();
        for &i in idxs {
            if is_discrete(i) {
                disc.push(i)
            } else {
                cont.push(i)
            }
        }
        (cont, disc)
    };
    let both_vec: Vec<usize> = nl_both.iter().copied().collect();
    let (both_cont, both_disc) = split_group(&both_vec);
    let (cons_cont, cons_disc) = split_group(&cons_only);
    let (objs_cont, objs_disc) = split_group(&objs_only);

    let in_nl: BTreeSet<usize> = nl_cons.union(&nl_objs).copied().collect();
    let (mut lin_cont, mut lin_bin, mut lin_int) = (Vec::new(), Vec::new(), Vec::new());
    for (i, v) in flat.iter().enumerate() {
        if in_nl.contains(&i) {
            continue;
        }
        match v.var_type {
            VarType::Binary => lin_bin.push(i),
            VarType::Integer => lin_int.push(i),
            _ => lin_cont.push(i),
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n_total);
    for g in [
        &both_cont, &both_disc, &cons_cont, &cons_disc, &objs_cont, &objs_disc, &lin_cont,
        &lin_bin, &lin_int,
    ] {
        order.extend_from_slice(g);
    }
    // Old flat slot -> canonical index.
    let mut remap = vec![usize::MAX; n_total];
    for (new_idx, &old) in order.iter().enumerate() {
        remap[old] = new_idx;
    }

    // ── per-row sparsity: the ASL union of linear and nonlinear variables, so
    //    the header count, the `k` section and the `J` blocks cannot disagree
    let jac_cols: Vec<BTreeMap<usize, f64>> = rows
        .iter()
        .map(|row| {
            let mut cols: BTreeMap<usize, f64> =
                row.linear.iter().map(|(k, v)| (remap[*k], *v)).collect();
            if let Some(nl) = row.nonlinear {
                let mut vs = BTreeSet::new();
                vars_under(&prog, nl, &mut vs);
                for v in vs {
                    cols.entry(remap[v]).or_insert(0.0);
                }
            }
            cols
        })
        .collect();
    let mut grad_cols: BTreeMap<usize, f64> = obj_split
        .linear
        .iter()
        .map(|(k, v)| (remap[*k], *v))
        .collect();
    if let Some(nl) = obj_split.nonlinear {
        let mut vs = BTreeSet::new();
        vars_under(&prog, nl, &mut vs);
        for v in vs {
            grad_cols.entry(remap[v]).or_insert(0.0);
        }
    }

    // ── header
    let n_cons = rows.len();
    let n_objs = 1;
    let n_nl_cons = jac_cols
        .iter()
        .zip(&rows)
        .filter(|(_, r)| r.nonlinear.is_some())
        .count();
    let n_nl_objs = usize::from(obj_split.nonlinear.is_some());
    let n_jac_nz: usize = jac_cols.iter().map(|c| c.len()).sum();
    let n_grad_nz = grad_cols.len();

    let mut out = String::with_capacity(64 * (n_cons + n_total) + 256);
    out.push_str(&format!("g3 1 1 0\t# problem {model_name}\n"));
    out.push_str(&format!(
        " {n_total} {n_cons} {n_objs} 0 0\t# vars, constraints, objectives\n"
    ));
    out.push_str(&format!(
        " {n_nl_cons} {n_nl_objs}\t# nonlinear constraints, objectives\n"
    ));
    out.push_str(" 0 0\t# network constraints\n");
    // `nlvo` is ASL's PREFIX BOUND, not the raw objective count. ASL sizes its
    // nonlinear-column prefix as `max(nlvc, nlvo)` and reads the objective-only
    // nonlinear block as the columns in `[nlvc, nlvo)`. The canonical order
    // emitted above is `[both | cons-only | objs-only | linear]`, so when an
    // objective-only nonlinear variable exists that block ENDS at
    // `nlvc + |objs-only|`, which is what must be declared. The raw count
    // under-declares the prefix whenever there are also cons-only nonlinear
    // variables; ASL truncates, every column past the truncation is
    // mis-assigned, and the reader silently solves a different problem --
    // issue #1222, where Ipopt reported "Optimal Solution Found." with 7269.45
    // instead of 8457.69 on `fuel`.
    //
    // Conditional, NOT the unconditional `nlvc + nlvo - nlvb`: with no
    // objective-only nonlinear variable the two readings coincide at
    // `len(nl_objs)` (0 for a linear objective) and `nlvo` must not be inflated
    // to `nlvc`. Mirrors `export/nl.py::_NLWriter._nlvo_prefix_bound` exactly --
    // the two writers are diffed byte for byte, so a divergence here is a
    // silent wrong header on whichever path the model happens to take.
    let nlvo = if objs_only.is_empty() {
        nl_objs.len()
    } else {
        nl_cons.len() + objs_only.len()
    };
    out.push_str(&format!(
        " {} {} {}\t# nonlinear vars in cons, objs, both\n",
        nl_cons.len(),
        nlvo,
        nl_both.len()
    ));
    out.push_str(" 0 0 0 1 0\t# flags\n");
    out.push_str(&format!(
        " {} {} {} {} {}\t# nbv niv nlvbi nlvci nlvoi\n",
        lin_bin.len(),
        lin_int.len(),
        both_disc.len(),
        cons_disc.len(),
        objs_disc.len()
    ));
    out.push_str(&format!(
        " {n_jac_nz} {n_grad_nz}\t# Jacobian, gradient nonzeros\n"
    ));
    out.push_str(" 0 0\t# max name lengths\n");
    out.push_str(" 0 0 0 0 0\t# common expressions\n");

    // ── C: constraint nonlinear bodies
    for (i, row) in rows.iter().enumerate() {
        out.push_str(&format!("C{i}\n"));
        match row.nonlinear {
            Some(nl) => write_expr(&prog, nl, &remap, &mut out)?,
            None => out.push_str("n0\n"),
        }
    }

    // ── O: objective
    let sense = match repr.objective_sense {
        ObjectiveSense::Minimize => 0,
        ObjectiveSense::Maximize => 1,
    };
    out.push_str(&format!("O0 {sense}\n"));
    match obj_split.nonlinear {
        Some(nl) => write_expr(&prog, nl, &remap, &mut out)?,
        None => out.push_str("n0\n"),
    }

    // ── r: constraint bounds, with the split-out constant folded in
    if !rows.is_empty() {
        // One line per ROW, looked up through `row_source` (built above, and
        // permuted with `rows`) rather than zipped one-to-one against
        // `repr.constraints` -- an array-valued body is one constraint and many
        // rows, and zipping silently truncated this section to the constraint
        // count.
        out.push_str("r\n");
        for (row, &ci) in rows.iter().zip(&row_source) {
            let c = &repr.constraints[ci];
            let rhs = c.rhs - row.constant;
            match c.sense {
                ConstraintSense::Le => out.push_str(&format!("1 {}\n", py_float(rhs))),
                ConstraintSense::Ge => out.push_str(&format!("2 {}\n", py_float(rhs))),
                ConstraintSense::Eq => out.push_str(&format!("4 {}\n", py_float(rhs))),
            }
        }
    }

    // ── b: variable bounds, in canonical order
    out.push_str("b\n");
    for &old in &order {
        let v = &flat[old];
        let (has_lb, has_ub) = (v.lb > -1e18, v.ub < 1e18);
        if has_lb && has_ub {
            if (v.lb - v.ub).abs() < 1e-15 {
                out.push_str(&format!("4 {}\n", py_float(v.lb)));
            } else {
                out.push_str(&format!("0 {} {}\n", py_float(v.lb), py_float(v.ub)));
            }
        } else if has_lb {
            out.push_str(&format!("2 {}\n", py_float(v.lb)));
        } else if has_ub {
            out.push_str(&format!("1 {}\n", py_float(v.ub)));
        } else {
            out.push_str("3\n");
        }
    }

    // ── k: cumulative per-column Jacobian counts for columns 0..n-2
    if n_total <= 1 {
        out.push_str("k0\n");
    } else {
        let mut per_col = vec![0usize; n_total];
        for cols in &jac_cols {
            for col in cols.keys() {
                per_col[*col] += 1;
            }
        }
        out.push_str(&format!("k{}\n", n_total - 1));
        let mut cumulative = 0usize;
        for col in per_col.iter().take(n_total - 1) {
            cumulative += col;
            out.push_str(&format!("{cumulative}\n"));
        }
    }

    // ── J: per-constraint Jacobian blocks
    for (i, cols) in jac_cols.iter().enumerate() {
        if cols.is_empty() {
            continue;
        }
        out.push_str(&format!("J{i} {}\n", cols.len()));
        for (col, coeff) in cols {
            out.push_str(&format!("{col} {}\n", py_float(*coeff)));
        }
    }

    // ── G: objective gradient
    if !grad_cols.is_empty() {
        out.push_str(&format!("G0 {}\n", grad_cols.len()));
        for (col, coeff) in &grad_cols {
            out.push_str(&format!("{col} {}\n", py_float(*coeff)));
        }
    }

    prog.seal();
    Ok(out)
}

#[cfg(test)]
mod func_table_tests {
    use super::*;
    use crate::expand::func_code;
    use crate::expr::MathFunc;

    /// Whether this writer can emit the function at all -- by opcode, or by the
    /// `log2` rewrite.
    ///
    /// No wildcard arm on purpose: adding a `MathFunc` variant fails to compile
    /// here until it is classified. That is the guard that was missing when
    /// `expand.rs::func_code` admitted `Log2` and `nl_func_opcode` did not map
    /// it, which made `to_nl()` PANIC -- and a `PanicException` derives from
    /// `BaseException`, so `export/nl.py`'s fallback could not catch it.
    fn writer_handles(f: MathFunc) -> bool {
        match f {
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
            | MathFunc::Asin => true,
            MathFunc::Acos
            | MathFunc::Tanh
            | MathFunc::Abs
            | MathFunc::Sign
            | MathFunc::Min
            | MathFunc::Max
            | MathFunc::Prod
            | MathFunc::Norm2
            | MathFunc::Asinh
            | MathFunc::Acosh
            | MathFunc::Atanh
            | MathFunc::Erf
            | MathFunc::Log1p
            | MathFunc::Sigmoid
            | MathFunc::Softplus
            | MathFunc::Norm1
            | MathFunc::NormInf
            | MathFunc::NormP(_) => false,
        }
    }

    const ALL: &[MathFunc] = &[
        MathFunc::Exp,
        MathFunc::Log,
        MathFunc::Log2,
        MathFunc::Log10,
        MathFunc::Sqrt,
        MathFunc::Sin,
        MathFunc::Cos,
        MathFunc::Tan,
        MathFunc::Atan,
        MathFunc::Sinh,
        MathFunc::Cosh,
        MathFunc::Asin,
        MathFunc::Acos,
        MathFunc::Tanh,
        MathFunc::Abs,
        MathFunc::Sign,
        MathFunc::Min,
        MathFunc::Max,
        MathFunc::Prod,
        MathFunc::Norm2,
        MathFunc::Asinh,
        MathFunc::Acosh,
        MathFunc::Atanh,
        MathFunc::Erf,
        MathFunc::Log1p,
        MathFunc::Sigmoid,
        MathFunc::Softplus,
        MathFunc::Norm1,
        MathFunc::NormInf,
        MathFunc::NormP(3),
    ];

    #[test]
    fn nl_func_opcode_covers_every_expanded_func() {
        let mut admitted = 0;
        for &f in ALL {
            match func_code(f) {
                // Not expanded -> never reaches this writer. Nothing to check
                // beyond the classification staying honest.
                None => assert!(
                    !writer_handles(f),
                    "{f:?} is classified writable but func_code refuses it"
                ),
                Some(c) => {
                    admitted += 1;
                    assert!(
                        writer_handles(f),
                        "{f:?} is expanded but classified unwritable"
                    );
                    assert!(
                        nl_func_opcode(c).is_some() || c == FUNC_LOG2 || c == FUNC_TAN,
                        "{f:?} (code {c}) reaches the writer with no .nl opcode \
                         and no rewrite -- this is the log2 panic all over again"
                    );
                }
            }
        }
        // Prove the loop ran: an empty ALL would pass every assertion above.
        assert_eq!(admitted, 12, "expected 12 expanded functions, saw {admitted}");
    }

    #[test]
    fn log2_is_rewritten_not_mapped() {
        assert!(
            nl_func_opcode(FUNC_LOG2).is_none(),
            "log2 must have no direct opcode -- .nl has no base-2 log"
        );
        assert_eq!(nl_func_opcode(FUNC_LOG), Some(43));
        // tan is rewritten as sin/cos too -- see FUNC_TAN's comment.
        assert!(nl_func_opcode(FUNC_TAN).is_none());
        assert_eq!(nl_func_opcode(FUNC_SIN), Some(41));
        assert_eq!(nl_func_opcode(FUNC_COS), Some(46));
        // The divisor the rewrite emits must render exactly as Python's
        // `repr(math.log(2))`, or the two writers stop being byte-identical.
        assert_eq!(py_float(std::f64::consts::LN_2), "0.6931471805599453");
    }
}
