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

        has_obj = self.model._objective is not None

        type_kw = {
            "free": "Free Variables",
            "positive": "Positive Variables",
            "binary": "Binary Variables",
            "integer": "Integer Variables",
        }

        for gtype, vars_list in groups.items():
            names: list[str] = []
            # Prepend synthetic obj_var to free group
            if gtype == "free" and has_obj:
                names.append("obj_var")
            for var in vars_list:
                if var.name in self._var_sets:
                    dom = ", ".join(s[0] for s in self._var_sets[var.name])
                    names.append(f"{var.name}({dom})")
                else:
                    names.append(var.name)
            if names:
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
                # X-2 (#413) / EX-4: an array-variable block is NOT a scalar.
                # The previous code only emitted a bound when it was UNIFORM
                # across every element (`np.all(arr == arr.flat[0])`); on
                # heterogeneous per-element bounds it silently dropped the bound
                # *entirely*, exporting e.g. a DAE model with pinned initial
                # conditions as an unbounded (or default-Positive) variable —
                # GAMS then solves a different model than discopt. Emit the
                # uniform case compactly over the whole domain and, when
                # heterogeneous, emit one line per element at its 1-based label.
                self._write_array_bound(lines, var, lb_arr, "lo", -1e18, np.greater)
                self._write_array_bound(lines, var, ub_arr, "up", 1e18, np.less)

        lines.append("")

    def _element_label(self, var: Variable, k: int) -> str:
        """1-based, quoted GAMS label(s) for flat element ``k`` of ``var``.

        Multi-dimensional variables produce a comma-separated tuple of labels
        matching the declared set domains (e.g. ``'1', '2'``); flat/1-D
        variables produce a single ``'k+1'``.
        """
        idx = np.unravel_index(k, var.shape)
        return ", ".join(f"'{i + 1}'" for i in np.atleast_1d(idx))

    def _write_array_bound(self, lines, var, arr, kind, inf_sentinel, cmp):
        """Emit ``.lo``/``.up`` bounds for an array variable, per element.

        ``kind`` is ``"lo"`` or ``"up"``; ``cmp(value, inf_sentinel)`` decides
        whether the bound is finite enough to emit (``np.greater`` for lower,
        ``np.less`` for upper). Uniform bounds are compacted to one domain-wide
        assignment; heterogeneous bounds are written one element at a time so no
        per-element bound is ever dropped (X-2 / EX-4).
        """
        flat = np.asarray(arr, dtype=np.float64).ravel()
        if flat.size == 0:
            return
        has_dom = var.name in self._var_sets
        dom = ", ".join(s[0] for s in self._var_sets[var.name]) if has_dom else None

        if np.all(flat == flat[0]):
            val = float(flat[0])
            if bool(cmp(val, inf_sentinel)):
                target = f"{var.name}.{kind}({dom})" if has_dom else f"{var.name}.{kind}"
                lines.append(f"{target} = {val};")
            return

        # Heterogeneous: one assignment per element at its 1-based label. When
        # the variable was declared over a set domain we address elements by
        # label; otherwise (no set info) fall back to a plain scalar name, which
        # only occurs for the degenerate size-1 arrays handled above.
        for k in range(flat.size):
            val = float(flat[k])
            if not bool(cmp(val, inf_sentinel)):
                continue
            if has_dom:
                lines.append(f"{var.name}.{kind}({self._element_label(var, k)}) = {val};")
            else:
                lines.append(f"{var.name}.{kind} = {val};")

    def _write_equations(self, lines: list[str]):
        """Write equation declarations and definitions."""
        from discopt.export._common import iter_builder_linear_rows

        obj = self.model._objective
        constraints = self.model._constraints
        # X-1: fast-API / builder-resident linear rows (`add_linear_constraints` /
        # the `Model.constraint` fast path) live only in the Rust builder. Without
        # them the GAMS export would emit an empty model.
        builder_rows = iter_builder_linear_rows(self.model)

        # Declare equations
        eq_names = ["obj_eq"]
        for i, c in enumerate(constraints):
            name = c.name or f"c{i + 1}"
            eq_names.append(name)
        for j, brow in enumerate(builder_rows):
            eq_names.append(self._sanitize_eq_name(brow.name) or f"blk{j + 1}")
        lines.append(f"Equations {', '.join(eq_names)};")
        lines.append("")

        # Objective equation: obj_var =e= expr. A builder-resident objective
        # (`add_linear_objective` / `add_quadratic_objective`, X-1) leaves a zero
        # placeholder in `model._objective` — recover its real terms rather than
        # emit `obj_var =e= 0`.
        from discopt.export._common import builder_objective

        builder_obj = builder_objective(self.model)
        if builder_obj is not None:
            obj_str = self._builder_objective_expr(builder_obj)
            lines.append(f"obj_eq.. obj_var =e= {obj_str};")
            lines.append("")
        elif obj is not None:
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

        # Builder-resident linear equations, emitted with variable names + 1-based
        # element references (matching the GAMS index convention above).
        gams_op = {"<=": "=l=", ">=": "=g=", "==": "=e="}
        for j, brow in enumerate(builder_rows):
            name = self._sanitize_eq_name(brow.name) or f"blk{j + 1}"
            lhs = self._builder_row_lhs(brow.terms)
            lines.append(f"{name}.. {lhs} {gams_op[brow.sense]} {self._fmt_num(brow.rhs)};")

        lines.append("")

    def _flat_index_ref(self, gidx: int) -> str:
        """Render flat variable index ``gidx`` as a GAMS variable/element reference.

        Walks ``model._variables`` in the same order that
        ``export._common.variable_flat_offsets`` assigns flat indices.
        """
        offset = 0
        for var in self.model._variables:
            n = max(var.size, 1) if var.shape != () else 1
            if offset <= gidx < offset + n:
                local = gidx - offset
                if var.shape == () or var.shape == (1,):
                    return var.name
                idx = np.unravel_index(local, var.shape)
                idx_str = ", ".join(f"'{i + 1}'" for i in idx)
                return f"{var.name}({idx_str})"
            offset += n
        raise ValueError(f"Flat variable index {gidx} out of range for GAMS export.")

    def _builder_objective_expr(self, builder_obj) -> str:
        """Render a recovered builder objective (linear + optional quadratic) for GAMS."""
        linear, quad, constant, _sense = builder_obj
        parts: list[str] = []

        def _emit(coeff: float, ref: str) -> None:
            sign = "+" if coeff >= 0 else "-"
            mag = abs(coeff)
            term = ref if mag == 1.0 else f"{self._fmt_num(mag)}*{ref}"
            parts.append(f"{sign} {term}" if parts else (f"-{term}" if coeff < 0 else term))

        for gidx in sorted(linear):
            _emit(linear[gidx], self._flat_index_ref(gidx))
        if quad:
            for i, j in sorted(quad):
                # extract_quadratic convention: coefficient of x_i x_j. GAMS gets the
                # raw product term; the diagonal (i == j) becomes x_i * x_i.
                ri, rj = self._flat_index_ref(i), self._flat_index_ref(j)
                _emit(quad[(i, j)], f"{ri}*{rj}")
        if constant != 0.0:
            _emit(constant, "1")
        return " ".join(parts) if parts else "0"

    def _builder_row_lhs(self, terms: list) -> str:
        """Render a builder row's ``sum(coeff * var[idx])`` as a GAMS expression."""
        parts: list[str] = []
        for var, local, coeff in terms:
            if var.shape == () or var.shape == (1,):
                ref = var.name
            else:
                idx = np.unravel_index(local, var.shape)
                idx_str = ", ".join(f"'{i + 1}'" for i in idx)
                ref = f"{var.name}({idx_str})"
            sign = "+" if coeff >= 0 else "-"
            mag = abs(coeff)
            term = ref if mag == 1.0 else f"{self._fmt_num(mag)}*{ref}"
            parts.append(f"{sign} {term}" if parts else (f"-{term}" if coeff < 0 else term))
        return " ".join(parts) if parts else "0"

    @staticmethod
    def _sanitize_eq_name(name: str | None) -> str | None:
        if not name:
            return None
        return name.replace(" ", "_").replace("-", "_").replace("[", "_").replace("]", "")

    @staticmethod
    def _fmt_num(value: float) -> str:
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.15g}"

    def _write_model_solve(self, lines: list[str]):
        """Write Model and Solve statements."""
        from discopt.export._common import builder_objective

        mtype = self._model_type or self._detect_model_type()
        builder_obj = builder_objective(self.model)
        if builder_obj is not None:
            is_min = not str(builder_obj[3]).lower().startswith("max")
        else:
            is_min = bool(
                self.model._objective and self.model._objective.sense == ObjectiveSense.MINIMIZE
            )
        sense = "minimizing" if is_min else "maximizing"

        lines.append(f"Model {self.model.name} / all /;")
        lines.append(f"Solve {self.model.name} using {mtype} {sense} obj_var;")

    def _detect_model_type(self) -> str:
        """Auto-detect GAMS model type from variable types and expression structure."""
        from discopt.export._common import builder_objective

        has_integer = any(
            v.var_type in (VarType.BINARY, VarType.INTEGER) for v in self.model._variables
        )
        # A builder-resident quadratic objective is nonlinear even though the
        # placeholder expression is not (X-1).
        builder_obj = builder_objective(self.model)
        builder_quad = builder_obj is not None and builder_obj[1]
        has_nonlinear = self._has_nonlinear() or bool(builder_quad)
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
            # GAMS is 1-based and requires quoted set labels for concrete element
            # references (f_pipe('1'), not f_pipe(1)). Symbolic indices (set/alias
            # names from a loop) are emitted bare via _expr_to_gams.
            if isinstance(expr.index, tuple):
                idx_str = ", ".join(
                    f"'{i + 1}'" if isinstance(i, int) else self._expr_to_gams(i)
                    for i in expr.index
                )
            elif isinstance(expr.index, int):
                idx_str = f"'{expr.index + 1}'"
            else:
                idx_str = self._expr_to_gams(expr.index)
            return f"{base}({idx_str})"

        if isinstance(expr, BinaryOp):
            left = self._expr_to_gams(expr.left)
            right = self._expr_to_gams(expr.right)
            if expr.op == "**":
                # GAMS power(x, n) requires an INTEGER exponent; a fractional or
                # symbolic exponent must use rPower(x, r) (== x ** r, base >= 0).
                if isinstance(expr.right, Constant) and float(expr.right.value) == int(
                    float(expr.right.value)
                ):
                    return f"power({left}, {right})"
                return f"rPower({left}, {right})"
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
