"""
AMPL .nl format writer for discopt models.

Produces text-mode .nl files compatible with AMPL-compatible solvers
(Ipopt, BARON, Couenne, SCIP, etc.) and the discopt Rust .nl parser.

The .nl format uses a prefix-notation expression encoding with numeric
opcodes. Key sections:

  Header (10 lines):  problem dimensions
  C sections:         nonlinear constraint bodies (expression DAG)
  O sections:         nonlinear objective (expression DAG)
  r section:          constraint bounds/senses
  b section:          variable bounds
  k section:          Jacobian column counts (cumulative)
  J sections:         linear terms in constraints (Jacobian)
  G sections:         linear terms in objective (gradient)
  x section:          initial point (optional)

Reference: Gay, D.M. "Hooking Your Solver to AMPL" (2003).
Inspired by Pyomo's NLv2Writer (pyomo/repn/plugins/nl_writer.py).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Union

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    ObjectiveSense,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)


def to_nl(
    model: Model,
    path: Union[str, Path, None] = None,
) -> Union[str, None]:
    """Export a discopt Model to AMPL .nl text format.

    Parameters
    ----------
    model : Model
        A discopt optimization model.
    path : str or Path, optional
        If provided, write the .nl string to this file and return ``None``.
        Otherwise return the .nl string.

    Returns
    -------
    str or None
        The .nl text if *path* is ``None``, otherwise ``None``.
    """
    model.validate()
    writer = _NLWriter(model)
    text = writer.write()
    if path is not None:
        Path(path).write_text(text)
        return None
    return text


# ── Expression opcodes (AMPL .nl format) ────────────────────────
# Binary operators
_OP_ADD = 0
_OP_SUB = 1
_OP_MUL = 2
_OP_DIV = 3
_OP_POW = 5

# Unary operators
_OP_FLOOR = 13
_OP_CEIL = 14
_OP_ABS = 15
_OP_NEG = 16

# Math functions
_OP_ATAN = 37
_OP_COS = 38
_OP_SIN = 39
_OP_SQRT = 40
_OP_SINH = 41
_OP_ASIN = 42
_OP_SUMLIST = 43
_OP_LOG = 45
_OP_EXP = 46
_OP_LOG10 = 47
_OP_COSH = 49
_OP_TANH = 51
_OP_ACOS = 53

# Function name -> opcode mapping
_FUNC_OPCODES: dict[str, int] = {
    "exp": _OP_EXP,
    "log": _OP_LOG,
    "log10": _OP_LOG10,
    "sqrt": _OP_SQRT,
    "sin": _OP_SIN,
    "cos": _OP_COS,
    "asin": _OP_ASIN,
    "acos": _OP_ACOS,
    "atan": _OP_ATAN,
    "sinh": _OP_SINH,
    "cosh": _OP_COSH,
    "tanh": _OP_TANH,
    "abs": _OP_ABS,
}


class _NLWriter:
    def __init__(self, model: Model):
        self.model = model
        # Flatten all variables to a single indexed list
        # .nl format: continuous first, then binary, then integer (at end)
        self._flat_vars: list[tuple[Variable, int]] = []  # (var, element_idx)
        self._var_index: dict[tuple[str, int], int] = {}  # (name, elem) -> flat_idx
        self._n_total = 0
        # Separate linear and nonlinear parts
        self._obj_linear: dict[int, float] = {}  # var_idx -> coeff
        self._obj_nonlinear: Expression | None = None
        self._con_linear: list[dict[int, float]] = []  # per constraint
        self._con_nonlinear: list[Expression | None] = []
        self._con_bounds: list[tuple[int, float, float]] = []  # (type, lb, ub)

    def write(self) -> str:
        self._build_var_map()
        self._decompose_expressions()
        buf = io.StringIO()
        self._write_header(buf)
        self._write_C_sections(buf)
        self._write_O_section(buf)
        self._write_r_section(buf)
        self._write_b_section(buf)
        self._write_k_section(buf)
        self._write_J_sections(buf)
        self._write_G_section(buf)
        return buf.getvalue()

    # ── Build flat variable map ──

    def _build_var_map(self):
        """Build flat variable index map. Continuous vars first, then discrete."""
        continuous: list[tuple[Variable, int]] = []
        binary: list[tuple[Variable, int]] = []
        integer: list[tuple[Variable, int]] = []

        for var in self.model._variables:
            size = max(1, int(np.prod(var.shape)))
            for elem in range(size):
                entry = (var, elem)
                if var.var_type == VarType.BINARY:
                    binary.append(entry)
                elif var.var_type == VarType.INTEGER:
                    integer.append(entry)
                else:
                    continuous.append(entry)

        self._flat_vars = continuous + binary + integer
        self._n_total = len(self._flat_vars)
        self._n_binary = len(binary)
        self._n_integer = len(integer)

        for idx, (var, elem) in enumerate(self._flat_vars):
            self._var_index[(var.name, elem)] = idx

    # ── Decompose expressions into linear + nonlinear parts ──

    def _decompose_expressions(self):
        """Split objective and constraints into linear and nonlinear parts."""
        obj = self.model._objective
        if obj is not None:
            linear, nonlinear = self._split_expr(obj.expression)
            self._obj_linear = linear
            self._obj_nonlinear = nonlinear

        for con in self.model._constraints:
            linear, nonlinear = self._split_expr(con.body)
            self._con_linear.append(linear)
            self._con_nonlinear.append(nonlinear)
            # Determine constraint bound type
            rhs = con.rhs
            if con.sense == "<=":
                self._con_bounds.append((1, 0.0, float(rhs)))  # body <= rhs
            elif con.sense == ">=":
                self._con_bounds.append((2, float(rhs), 0.0))  # body >= rhs (stored as <=)
            elif con.sense == "==":
                self._con_bounds.append((4, float(rhs), float(rhs)))

    def _split_expr(self, expr: Expression) -> tuple[dict[int, float], Expression | None]:
        """Split an expression into linear terms {var_idx: coeff} and nonlinear remainder."""
        linear: dict[int, float] = {}
        nonlinear_terms: list[Expression] = []
        self._collect_linear(expr, 1.0, linear, nonlinear_terms)
        if nonlinear_terms:
            if len(nonlinear_terms) == 1:
                nl_expr = nonlinear_terms[0]
            else:
                nl_expr = nonlinear_terms[0]
                for t in nonlinear_terms[1:]:
                    nl_expr = BinaryOp("+", nl_expr, t)
            return linear, nl_expr
        return linear, None

    def _collect_linear(
        self,
        expr: Expression,
        coeff: float,
        linear: dict[int, float],
        nonlinear: list[Expression],
    ):
        """Recursively collect linear terms and segregate nonlinear ones."""
        if isinstance(expr, Constant):
            val = float(expr.value)
            if val != 0.0:
                # Constant contributes to the nonlinear part (or rather, a constant offset)
                nonlinear.append(Constant(val * coeff))
            return

        if isinstance(expr, Variable):
            if expr.shape == () or expr.shape == (1,):
                idx = self._var_index.get((expr.name, 0))
                if idx is not None:
                    linear[idx] = linear.get(idx, 0.0) + coeff
                    return
            # Array variable without indexing - treat as nonlinear
            nonlinear.append(expr if coeff == 1.0 else BinaryOp("*", Constant(coeff), expr))
            return

        if isinstance(expr, IndexExpression):
            var_idx = self._resolve_var_index(expr)
            if var_idx is not None:
                linear[var_idx] = linear.get(var_idx, 0.0) + coeff
                return
            nonlinear.append(expr if coeff == 1.0 else BinaryOp("*", Constant(coeff), expr))
            return

        if isinstance(expr, BinaryOp):
            if expr.op == "+":
                self._collect_linear(expr.left, coeff, linear, nonlinear)
                self._collect_linear(expr.right, coeff, linear, nonlinear)
                return
            if expr.op == "-":
                self._collect_linear(expr.left, coeff, linear, nonlinear)
                self._collect_linear(expr.right, -coeff, linear, nonlinear)
                return
            if expr.op == "*":
                # Check if one side is a constant
                lc = self._get_const(expr.left)
                rc = self._get_const(expr.right)
                if lc is not None:
                    self._collect_linear(expr.right, coeff * lc, linear, nonlinear)
                    return
                if rc is not None:
                    self._collect_linear(expr.left, coeff * rc, linear, nonlinear)
                    return
            # Not linear
            if coeff == 1.0:
                nonlinear.append(expr)
            else:
                nonlinear.append(BinaryOp("*", Constant(coeff), expr))
            return

        if isinstance(expr, UnaryOp) and expr.op == "neg":
            self._collect_linear(expr.operand, -coeff, linear, nonlinear)
            return

        if isinstance(expr, SumOverExpression):
            for term in expr.terms:
                self._collect_linear(term, coeff, linear, nonlinear)
            return

        # Everything else is nonlinear
        if coeff == 1.0:
            nonlinear.append(expr)
        else:
            nonlinear.append(BinaryOp("*", Constant(coeff), expr))

    def _get_const(self, expr: Expression) -> float | None:
        if isinstance(expr, Constant):
            return float(expr.value)
        return None

    def _resolve_var_index(self, expr: IndexExpression) -> int | None:
        """Resolve x[i] or x[i,j] to a flat variable index."""
        if not isinstance(expr.base, Variable):
            return None
        var = expr.base
        idx = expr.index
        if isinstance(idx, tuple):
            # Multi-dimensional: flatten to row-major
            flat = 0
            for dim, i in enumerate(idx):
                if not isinstance(i, int):
                    return None
                stride = 1
                for d2 in range(dim + 1, len(var.shape)):
                    stride *= var.shape[d2]
                flat += i * stride
            return self._var_index.get((var.name, flat))
        if isinstance(idx, int):
            return self._var_index.get((var.name, idx))
        return None

    # ── Header (10 mandatory lines) ──

    def _write_header(self, buf: io.StringIO):
        n_vars = self._n_total
        n_cons = len(self.model._constraints)
        n_objs = 1 if self.model._objective else 0

        # Count nonlinear constraints and objectives
        n_nl_cons = sum(1 for nl in self._con_nonlinear if nl is not None)
        n_nl_objs = 1 if self._obj_nonlinear is not None else 0

        # Count variables appearing in nonlinear expressions
        nl_var_set_cons: set[int] = set()
        nl_var_set_objs: set[int] = set()
        for nl in self._con_nonlinear:
            if nl is not None:
                self._collect_var_indices(nl, nl_var_set_cons)
        if self._obj_nonlinear is not None:
            self._collect_var_indices(self._obj_nonlinear, nl_var_set_objs)
        nl_both = nl_var_set_cons & nl_var_set_objs
        nl_cons_only = nl_var_set_cons - nl_both
        nl_objs_only = nl_var_set_objs - nl_both

        # Count Jacobian nonzeros (linear + nonlinear vars in each constraint)
        n_jac_nz = sum(len(lin) for lin in self._con_linear)
        # Add nonlinear variable references in constraints
        for nl in self._con_nonlinear:
            if nl is not None:
                vs: set[int] = set()
                self._collect_var_indices(nl, vs)
                n_jac_nz += len(vs)

        # Count objective gradient nonzeros
        n_grad_nz = len(self._obj_linear)
        if self._obj_nonlinear is not None:
            vs2: set[int] = set()
            self._collect_var_indices(self._obj_nonlinear, vs2)
            n_grad_nz += len(vs2)

        # Line 0: format marker
        buf.write(f"g3 1 1 0\t# problem {self.model.name}\n")
        # Line 1: core dimensions
        buf.write(f" {n_vars} {n_cons} {n_objs} 0 0\t# vars, constraints, objectives\n")
        # Line 2: nonlinearity counts
        buf.write(f" {n_nl_cons} {n_nl_objs}\t# nonlinear constraints, objectives\n")
        # Line 3: network (unused)
        buf.write(" 0 0\t# network constraints\n")
        # Line 4: nonlinear variable distribution
        buf.write(
            f" {len(nl_cons_only)} {len(nl_objs_only)} {len(nl_both)}"
            "\t# nonlinear vars in cons, objs, both\n"
        )
        # Line 5: flags
        buf.write(" 0 0 0 1 0\t# flags\n")
        # Line 6: discrete variable counts
        buf.write(f" {self._n_binary} {self._n_integer} 0 0 0\t# binary, integer vars\n")
        # Line 7: sparsity
        buf.write(f" {n_jac_nz} {n_grad_nz}\t# Jacobian, gradient nonzeros\n")
        # Line 8: name lengths
        buf.write(" 0 0\t# max name lengths\n")
        # Line 9: common expressions
        buf.write(" 0 0 0 0 0\t# common expressions\n")

    def _collect_var_indices(self, expr: Expression, result: set[int]):
        """Collect all variable indices referenced in an expression."""
        if isinstance(expr, Variable):
            if expr.shape == () or expr.shape == (1,):
                idx = self._var_index.get((expr.name, 0))
                if idx is not None:
                    result.add(idx)
        elif isinstance(expr, IndexExpression):
            vi = self._resolve_var_index(expr)
            if vi is not None:
                result.add(vi)
        elif isinstance(expr, BinaryOp):
            self._collect_var_indices(expr.left, result)
            self._collect_var_indices(expr.right, result)
        elif isinstance(expr, UnaryOp):
            self._collect_var_indices(expr.operand, result)
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._collect_var_indices(arg, result)
        elif isinstance(expr, SumExpression):
            self._collect_var_indices(expr.operand, result)
        elif isinstance(expr, SumOverExpression):
            for term in expr.terms:
                self._collect_var_indices(term, result)

    # ── C sections (nonlinear constraint bodies) ──

    def _write_C_sections(self, buf: io.StringIO):
        for i, nl in enumerate(self._con_nonlinear):
            buf.write(f"C{i}\n")
            if nl is not None:
                self._write_expr(nl, buf)
            else:
                buf.write("n0\n")

    # ── O section (nonlinear objective) ──

    def _write_O_section(self, buf: io.StringIO):
        if self.model._objective is None:
            return
        sense = 0 if self.model._objective.sense == ObjectiveSense.MINIMIZE else 1
        buf.write(f"O0 {sense}\n")
        if self._obj_nonlinear is not None:
            self._write_expr(self._obj_nonlinear, buf)
        else:
            buf.write("n0\n")

    # ── r section (constraint bounds) ──

    def _write_r_section(self, buf: io.StringIO):
        if not self._con_bounds:
            return
        buf.write("r\n")
        for btype, lb, ub in self._con_bounds:
            if btype == 1:  # <= ub
                buf.write(f"1 {ub}\n")
            elif btype == 2:  # >= lb
                buf.write(f"2 {lb}\n")
            elif btype == 4:  # == rhs
                buf.write(f"4 {lb}\n")
            elif btype == 0:  # range
                buf.write(f"0 {lb} {ub}\n")
            else:
                buf.write("3\n")  # free

    # ── b section (variable bounds) ──

    def _write_b_section(self, buf: io.StringIO):
        buf.write("b\n")
        for var, elem in self._flat_vars:
            lb_arr = np.asarray(var.lb).flat
            ub_arr = np.asarray(var.ub).flat
            lb = float(lb_arr[elem]) if elem < len(lb_arr) else float(lb_arr[0])
            ub = float(ub_arr[elem]) if elem < len(ub_arr) else float(ub_arr[0])

            has_lb = lb > -1e18
            has_ub = ub < 1e18

            if has_lb and has_ub:
                if abs(lb - ub) < 1e-15:
                    buf.write(f"4 {lb}\n")  # fixed
                else:
                    buf.write(f"0 {lb} {ub}\n")  # range
            elif has_lb:
                buf.write(f"2 {lb}\n")  # lb only
            elif has_ub:
                buf.write(f"1 {ub}\n")  # ub only
            else:
                buf.write("3\n")  # free

    # ── k section (Jacobian column counts) ──

    def _write_k_section(self, buf: io.StringIO):
        if self._n_total <= 1:
            buf.write("k0\n")
            return
        buf.write(f"k{self._n_total - 1}\n")
        cumulative = 0
        for col in range(self._n_total - 1):
            count = 0
            for lin in self._con_linear:
                if col in lin:
                    count += 1
            cumulative += count
            buf.write(f"{cumulative}\n")

    # ── J sections (linear Jacobian terms) ──

    def _write_J_sections(self, buf: io.StringIO):
        for i, lin in enumerate(self._con_linear):
            if not lin:
                continue
            buf.write(f"J{i} {len(lin)}\n")
            for var_idx in sorted(lin.keys()):
                buf.write(f"{var_idx} {lin[var_idx]}\n")

    # ── G section (linear objective gradient) ──

    def _write_G_section(self, buf: io.StringIO):
        if not self._obj_linear:
            return
        buf.write(f"G0 {len(self._obj_linear)}\n")
        for var_idx in sorted(self._obj_linear.keys()):
            buf.write(f"{var_idx} {self._obj_linear[var_idx]}\n")

    # ── Expression → .nl opcode encoding ──

    def _write_expr(self, expr: Expression, buf: io.StringIO):
        """Recursively emit an expression in .nl prefix notation."""
        if isinstance(expr, Constant):
            val = float(expr.value)
            buf.write(f"n{val}\n")
            return

        if isinstance(expr, Variable):
            if expr.shape == () or expr.shape == (1,):
                idx = self._var_index.get((expr.name, 0))
                if idx is not None:
                    buf.write(f"v{idx}\n")
                    return
            raise ValueError(f"Cannot write array variable {expr.name} without indexing")

        if isinstance(expr, IndexExpression):
            vi = self._resolve_var_index(expr)
            if vi is not None:
                buf.write(f"v{vi}\n")
                return
            raise ValueError(f"Cannot resolve indexed expression: {expr}")

        if isinstance(expr, BinaryOp):
            op_map = {
                "+": _OP_ADD,
                "-": _OP_SUB,
                "*": _OP_MUL,
                "/": _OP_DIV,
                "**": _OP_POW,
            }
            opcode = op_map.get(expr.op)
            if opcode is None:
                raise ValueError(f"Unknown binary operator: {expr.op}")
            buf.write(f"o{opcode}\n")
            self._write_expr(expr.left, buf)
            self._write_expr(expr.right, buf)
            return

        if isinstance(expr, UnaryOp):
            if expr.op == "neg":
                buf.write(f"o{_OP_NEG}\n")
                self._write_expr(expr.operand, buf)
                return
            if expr.op == "abs":
                buf.write(f"o{_OP_ABS}\n")
                self._write_expr(expr.operand, buf)
                return
            raise ValueError(f"Unknown unary operator: {expr.op}")

        if isinstance(expr, FunctionCall):
            fname = expr.func_name.lower()
            # Special cases
            if fname == "tan":
                # tan(x) = sin(x) / cos(x)
                buf.write(f"o{_OP_DIV}\n")
                buf.write(f"o{_OP_SIN}\n")
                self._write_expr(expr.args[0], buf)
                buf.write(f"o{_OP_COS}\n")
                self._write_expr(expr.args[0], buf)
                return
            if fname == "log2":
                # log2(x) = log(x) / log(2)
                buf.write(f"o{_OP_DIV}\n")
                buf.write(f"o{_OP_LOG}\n")
                self._write_expr(expr.args[0], buf)
                buf.write(f"n{math.log(2)}\n")
                return
            if fname == "log1p":
                # log1p(x) = log(1 + x)
                buf.write(f"o{_OP_LOG}\n")
                buf.write(f"o{_OP_ADD}\n")
                buf.write("n1\n")
                self._write_expr(expr.args[0], buf)
                return
            if fname == "sigmoid":
                # sigmoid(x) = 1 / (1 + exp(-x))
                buf.write(f"o{_OP_DIV}\n")
                buf.write("n1\n")
                buf.write(f"o{_OP_ADD}\n")
                buf.write("n1\n")
                buf.write(f"o{_OP_EXP}\n")
                buf.write(f"o{_OP_NEG}\n")
                self._write_expr(expr.args[0], buf)
                return
            if fname == "softplus":
                # softplus(x) = log(1 + exp(x))
                buf.write(f"o{_OP_LOG}\n")
                buf.write(f"o{_OP_ADD}\n")
                buf.write("n1\n")
                buf.write(f"o{_OP_EXP}\n")
                self._write_expr(expr.args[0], buf)
                return
            if fname == "erf":
                # No direct .nl opcode — approximate or raise
                raise ValueError("erf() has no .nl opcode; reformulate without erf")
            if fname == "sign":
                raise ValueError("sign() has no .nl opcode; reformulate without sign")
            if fname in ("min", "max"):
                # min/max of two args: use ternary or raise
                raise ValueError(f"{fname}() requires DNLP model type; not supported in .nl export")
            # Standard single-arg functions
            opcode = _FUNC_OPCODES.get(fname)
            if opcode is not None:
                buf.write(f"o{opcode}\n")
                self._write_expr(expr.args[0], buf)
                return
            raise ValueError(f"Unknown function in .nl export: {fname}")

        if isinstance(expr, SumExpression):
            self._write_expr(expr.operand, buf)
            return

        if isinstance(expr, SumOverExpression):
            if len(expr.terms) == 0:
                buf.write("n0\n")
                return
            if len(expr.terms) == 1:
                self._write_expr(expr.terms[0], buf)
                return
            # Use sumlist opcode
            buf.write(f"o{_OP_SUMLIST}\n")
            buf.write(f"{len(expr.terms)}\n")
            for term in expr.terms:
                self._write_expr(term, buf)
            return

        if isinstance(expr, MatMulExpression):
            raise ValueError(
                "MatMul expressions must be expanded before .nl export. "
                "Use explicit indexing instead of @."
            )

        raise ValueError(f"Cannot write expression type to .nl: {type(expr).__name__}")
