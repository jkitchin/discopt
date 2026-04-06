"""
GAMS (.gms) export for discopt models.

Walks the discopt expression DAG and emits valid GAMS source text.
Supports MINLP models with nonlinear expressions, binary/integer variables,
and all standard math functions.
"""

from __future__ import annotations

from pathlib import Path

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


def to_gams(
    model: Model,
    path: str | Path | None = None,
    model_type: str | None = None,
) -> str | None:
    """Export a discopt Model to GAMS (.gms) format.

    Parameters
    ----------
    model : Model
        A discopt optimization model.
    path : str or Path, optional
        If provided, write the .gms string to this file and return ``None``.
        Otherwise return the .gms string.
    model_type : str, optional
        GAMS model type (LP, MIP, NLP, MINLP, etc.).  Auto-detected if not given.

    Returns
    -------
    str or None
        The GAMS source if *path* is ``None``, otherwise ``None``.
    """
    model.validate()
    writer = _GamsWriter(model, model_type)
    text = writer.write()
    if path is not None:
        Path(path).write_text(text)
        return None
    return text


class _GamsWriter:
    def __init__(self, model: Model, model_type: str | None):
        self.model = model
        self._model_type = model_type
        self._set_counter = 0
        # Map Variable -> (set_names, set_elements) for indexed variables
        self._var_sets: dict[str, list[tuple[str, list[str]]]] = {}

    def write(self) -> str:
        lines: list[str] = []
        lines.append(f"* GAMS export of discopt model: {self.model.name}")
        lines.append("")

        # Generate sets from variable shapes
        self._generate_sets(lines)

        # Variables (including synthetic obj_var)
        self._write_variables(lines)

        # Equations declarations + definitions
        self._write_equations(lines)

        # Model and Solve
        self._write_model_solve(lines)

        lines.append("")
        return "\n".join(lines)

    def _generate_sets(self, lines: list[str]):
        """Generate GAMS Set declarations from variable shapes."""
        # Track unique dimension sizes to avoid duplicate sets
        dim_sets: dict[int, str] = {}  # size -> set_name

        for var in self.model._variables:
            if var.shape == () or var.shape == (1,):
                continue
            var_set_info: list[tuple[str, list[str]]] = []
            for dim_idx, dim_size in enumerate(var.shape):
                if dim_size in dim_sets:
                    set_name = dim_sets[dim_size]
                    elems = [str(k + 1) for k in range(dim_size)]
                else:
                    self._set_counter += 1
                    set_name = f"s{self._set_counter}"
                    elems = [str(k + 1) for k in range(dim_size)]
                    dim_sets[dim_size] = set_name
                    elem_str = ", ".join(elems)
                    lines.append(f"Set {set_name} / {elem_str} /;")
                var_set_info.append((set_name, elems))
            self._var_sets[var.name] = var_set_info

        if dim_sets:
            lines.append("")

    def _write_variables(self, lines: list[str]):
        """Write variable declarations and bounds."""
        # Group by type
        groups: dict[str, list[Variable]] = {
            "free": [],
            "positive": [],
            "binary": [],
            "integer": [],
        }
        for var in self.model._variables:
            if var.var_type == VarType.BINARY:
                groups["binary"].append(var)
            elif var.var_type == VarType.INTEGER:
                groups["integer"].append(var)
            elif np.all(np.asarray(var.lb) >= 0) and np.any(np.asarray(var.lb) < 1e18):
                groups["positive"].append(var)
            else:
                groups["free"].append(var)

        # Add synthetic obj_var only if model has an objective
        if self.model._objective is not None:
            groups["free"].insert(0, None)  # placeholder for obj_var

        type_kw = {
            "free": "Free Variables",
            "positive": "Positive Variables",
            "binary": "Binary Variables",
            "integer": "Integer Variables",
        }

        for gtype, vars_list in groups.items():
            if not vars_list:
                continue
            names = []
            for var in vars_list:
                if var is None:
                    names.append("obj_var")
                    continue
                if var.name in self._var_sets:
                    dom = ", ".join(s[0] for s in self._var_sets[var.name])
                    names.append(f"{var.name}({dom})")
                else:
                    names.append(var.name)
            lines.append(f"{type_kw[gtype]} {', '.join(names)};")

        lines.append("")

        # Write bounds for non-default bounds
        for var in self.model._variables:
            if var.var_type == VarType.BINARY:
                continue  # 0-1 is implicit
            lb_arr = np.asarray(var.lb)
            ub_arr = np.asarray(var.ub)
            if var.shape == () or var.shape == (1,):
                lb_val = float(lb_arr)
                ub_val = float(ub_arr)
                if lb_val > -1e18:
                    lines.append(f"{var.name}.lo = {lb_val};")
                if ub_val < 1e18:
                    lines.append(f"{var.name}.up = {ub_val};")
            else:
                # Check if bounds are uniform
                if lb_arr.size > 0 and np.all(lb_arr == lb_arr.flat[0]):
                    lb_val = float(lb_arr.flat[0])
                    if lb_val > -1e18:
                        if var.name in self._var_sets:
                            dom = ", ".join(s[0] for s in self._var_sets[var.name])
                            lines.append(f"{var.name}.lo({dom}) = {lb_val};")
                        else:
                            lines.append(f"{var.name}.lo = {lb_val};")
                if ub_arr.size > 0 and np.all(ub_arr == ub_arr.flat[0]):
                    ub_val = float(ub_arr.flat[0])
                    if ub_val < 1e18:
                        if var.name in self._var_sets:
                            dom = ", ".join(s[0] for s in self._var_sets[var.name])
                            lines.append(f"{var.name}.up({dom}) = {ub_val};")
                        else:
                            lines.append(f"{var.name}.up = {ub_val};")

        lines.append("")

    def _write_equations(self, lines: list[str]):
        """Write equation declarations and definitions."""
        obj = self.model._objective
        constraints = self.model._constraints

        # Declare equations
        eq_names = ["obj_eq"]
        for i, c in enumerate(constraints):
            name = c.name or f"c{i + 1}"
            eq_names.append(name)
        lines.append(f"Equations {', '.join(eq_names)};")
        lines.append("")

        # Objective equation: obj_var =e= expr
        if obj is not None:
            obj_expr_str = self._expr_to_gams(obj.expression)
            lines.append(f"obj_eq.. obj_var =e= {obj_expr_str};")
            lines.append("")

        # Constraint equations
        for i, c in enumerate(constraints):
            name = c.name or f"c{i + 1}"
            body_str = self._expr_to_gams(c.body)
            if c.sense == "<=":
                lines.append(f"{name}.. {body_str} =l= {c.rhs};")
            elif c.sense == ">=":
                lines.append(f"{name}.. {body_str} =g= {c.rhs};")
            elif c.sense == "==":
                lines.append(f"{name}.. {body_str} =e= {c.rhs};")

        lines.append("")

    def _write_model_solve(self, lines: list[str]):
        """Write Model and Solve statements."""
        mtype = self._model_type or self._detect_model_type()
        sense = (
            "minimizing"
            if (self.model._objective and self.model._objective.sense == ObjectiveSense.MINIMIZE)
            else "maximizing"
        )

        lines.append(f"Model {self.model.name} / all /;")
        lines.append(f"Solve {self.model.name} using {mtype} {sense} obj_var;")

    def _detect_model_type(self) -> str:
        """Auto-detect GAMS model type from variable types and expression structure."""
        has_integer = any(
            v.var_type in (VarType.BINARY, VarType.INTEGER) for v in self.model._variables
        )
        has_nonlinear = self._has_nonlinear()
        if has_integer and has_nonlinear:
            return "MINLP"
        if has_integer:
            return "MIP"
        if has_nonlinear:
            return "NLP"
        return "LP"

    def _has_nonlinear(self) -> bool:
        """Check if any expression in the model is nonlinear."""
        exprs = []
        if self.model._objective:
            exprs.append(self.model._objective.expression)
        for c in self.model._constraints:
            exprs.append(c.body)
        return any(self._expr_is_nonlinear(e) for e in exprs)

    def _expr_is_nonlinear(self, expr: Expression) -> bool:
        if isinstance(expr, FunctionCall):
            return True
        if isinstance(expr, BinaryOp):
            if expr.op == "**":
                return True
            if expr.op == "*":
                # bilinear if both sides contain variables
                lv = self._contains_var(expr.left)
                rv = self._contains_var(expr.right)
                if lv and rv:
                    return True
            return self._expr_is_nonlinear(expr.left) or self._expr_is_nonlinear(expr.right)
        if isinstance(expr, UnaryOp):
            return self._expr_is_nonlinear(expr.operand)
        if isinstance(expr, (SumExpression, SumOverExpression)):
            if isinstance(expr, SumExpression):
                return self._expr_is_nonlinear(expr.operand)
            return any(self._expr_is_nonlinear(t) for t in expr.terms)
        return False

    def _contains_var(self, expr: Expression) -> bool:
        if isinstance(expr, Variable):
            return True
        if isinstance(expr, IndexExpression):
            return self._contains_var(expr.base)
        if isinstance(expr, BinaryOp):
            return self._contains_var(expr.left) or self._contains_var(expr.right)
        if isinstance(expr, UnaryOp):
            return self._contains_var(expr.operand)
        if isinstance(expr, FunctionCall):
            return any(self._contains_var(a) for a in expr.args)
        return False

    # ── Expression to GAMS string ──────────────────────────────

    _FUNC_MAP = {
        "exp": "exp",
        "log": "log",
        "log2": "log2",
        "log10": "log10",
        "sqrt": "sqrt",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "abs": "abs",
        "sign": "sign",
        "erf": "errorf",
        "min": "min",
        "max": "max",
        "sigmoid": "sigmoid",
    }

    def _expr_to_gams(self, expr: Expression) -> str:
        if isinstance(expr, Constant):
            val = float(expr.value)
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return f"{val}"

        if isinstance(expr, Variable):
            return expr.name

        if isinstance(expr, IndexExpression):
            base = self._expr_to_gams(expr.base)
            if isinstance(expr.index, tuple):
                idx_str = ", ".join(
                    str(i + 1) if isinstance(i, int) else self._expr_to_gams(i) for i in expr.index
                )
            elif isinstance(expr.index, int):
                idx_str = str(expr.index + 1)  # GAMS is 1-based
            else:
                idx_str = self._expr_to_gams(expr.index)
            return f"{base}({idx_str})"

        if isinstance(expr, BinaryOp):
            left = self._expr_to_gams(expr.left)
            right = self._expr_to_gams(expr.right)
            if expr.op == "**":
                return f"power({left}, {right})"
            return f"({left} {expr.op} {right})"

        if isinstance(expr, UnaryOp):
            operand = self._expr_to_gams(expr.operand)
            if expr.op == "neg":
                return f"(-{operand})"
            if expr.op == "abs":
                return f"abs({operand})"
            return f"{expr.op}({operand})"

        if isinstance(expr, FunctionCall):
            fn = expr.func_name.lower()
            # Decompose functions GAMS doesn't have natively
            if fn == "log1p":
                inner = self._expr_to_gams(expr.args[0])
                return f"log(1 + {inner})"
            if fn == "softplus":
                inner = self._expr_to_gams(expr.args[0])
                return f"log(1 + exp({inner}))"
            if fn == "log2":
                inner = self._expr_to_gams(expr.args[0])
                return f"(log({inner}) / log(2))"
            gams_fn = self._FUNC_MAP.get(expr.func_name, expr.func_name)
            args_str = ", ".join(self._expr_to_gams(a) for a in expr.args)
            return f"{gams_fn}({args_str})"

        if isinstance(expr, SumExpression):
            return f"({self._expr_to_gams(expr.operand)})"

        if isinstance(expr, SumOverExpression):
            terms = " + ".join(self._expr_to_gams(t) for t in expr.terms)
            return f"({terms})"

        if isinstance(expr, MatMulExpression):
            # flatten matmul to explicit sum of products
            return f"({self._expr_to_gams(expr.left)} * {self._expr_to_gams(expr.right)})"

        return f"<unsupported:{type(expr).__name__}>"
