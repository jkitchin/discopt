"""Rich rendering of :class:`~discopt.modeling.core.Model` objects.

Produces LaTeX (and MathJax-backed HTML) in **standard PSE problem form**::

    minimize/maximize   f(x)
    subject to          g_i(x)  <=/>=/=  c_i
                        bounds and integrality on the variables

A small recursive visitor maps the expression DAG to LaTeX with minimal
parenthesisation (operator precedence), proper superscripts/fractions/function
names, and variable subscripts. Large models are summarised so an auto-rendered
``_repr_`` cannot flood a notebook; the public ``model_to_latex`` / ``model_to_html``
take ``max_rows=None`` for the full, untruncated form.

Imported lazily by ``core.Model`` (the ``to_latex``/``_repr_latex_`` methods) to
avoid a circular import.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

# Default number of constraint / variable rows shown in an auto ``_repr_``.
_DEFAULT_MAX_ROWS = 24

# LaTeX renderings for known unary/elementary functions; ``None`` => special-cased.
_FUNC_LATEX = {
    "exp": r"\exp",
    "log": r"\ln",
    "log10": r"\log_{10}",
    "sqrt": None,  # \sqrt{...}
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "abs": None,  # \left|...\right|
}

# Binary-operator precedence (higher binds tighter) for minimal parenthesisation.
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "@": 2, "**": 4}

# Precedence of the non-BinaryOp nodes that also produce operator-like output, so
# they parenthesise when a tighter-binding parent requires it (L4). Unary minus and
# an unbounded ``\sum`` bind as loosely as additive ops (a following ``^2`` or a
# ``\cdot`` must wrap them); a matrix product binds like other multiplicative ops.
_NEG_PREC = 1
_SUM_PREC = 1
_MATMUL_PREC = 2


def _fmt_num(v: float) -> str:
    f = float(v)
    if not np.isfinite(f):  # int(inf) raises OverflowError (L6)
        if np.isnan(f):
            return r"\mathrm{nan}"
        return r"\infty" if f > 0 else r"-\infty"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.6g}"


def _const_to_latex(value: Any) -> str:
    """Render a constant: scalars as numbers, small vectors/matrices inline, and
    larger arrays as a shape-annotated bold placeholder (the model does not name
    numpy coefficients, so a shaped placeholder is the most honest compact form)."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _fmt_num(float(arr))
    if arr.size <= 6 and arr.ndim == 1:
        return r"\begin{bmatrix}" + r" \\ ".join(_fmt_num(x) for x in arr) + r"\end{bmatrix}"
    if arr.size <= 6 and arr.ndim == 2:
        return (
            r"\begin{bmatrix}"
            + r" \\ ".join(" & ".join(_fmt_num(x) for x in row) for row in arr)
            + r"\end{bmatrix}"
        )
    shape = r"{\times}".join(str(d) for d in arr.shape)
    return rf"\mathbf{{C}}_{{[{shape}]}}"


def _sym(name: str) -> str:
    """Render a variable name: ``x3`` -> ``x_{3}``; escape underscores otherwise."""
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    return name.replace("_", r"\_")


def expr_to_latex(e: Any, parent_prec: int = 0) -> str:
    """Render an expression DAG node to LaTeX. Never raises — unknown nodes fall
    back to an escaped string form so a display can't crash a model build."""
    try:
        if isinstance(e, Constant):
            return _const_to_latex(e.value)
        if isinstance(e, Variable):
            return _sym(e.name)
        if isinstance(e, IndexExpression):
            idx = e.index
            idx_s = ",".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
            base = e.base
            # Merge a trailing-digit variable subscript with the index so `y1[0]`
            # renders as a single subscript `y_{1,0}` rather than the invalid
            # double subscript `y_{1}_{0}` (a MathJax error) (L5).
            if isinstance(base, Variable):
                mm = re.fullmatch(r"([A-Za-z]+)(\d+)", base.name)
                if mm:
                    return f"{mm.group(1)}_{{{mm.group(2)},{idx_s}}}"
                return f"{_sym(base.name)}_{{{idx_s}}}"
            # Brace-wrap any other base so its own subscripts nest validly.
            return f"{{{expr_to_latex(base, 5)}}}_{{{idx_s}}}"
        if isinstance(e, BinaryOp):
            return _binop_to_latex(e, parent_prec)
        if isinstance(e, Parameter):
            return _sym(e.name)
        if isinstance(e, UnaryOp):
            if e.op == "neg":
                # Unary minus binds loosely: parenthesise under a tighter parent so
                # `(-x)**2` renders `\left(-x\right)^{2}`, not the wrong `-x^{2}` (L4).
                inner = f"-{expr_to_latex(e.operand, 3)}"
                return _paren(inner, _NEG_PREC, parent_prec)
            if e.op == "abs":
                return rf"\left|{expr_to_latex(e.operand)}\right|"
            return rf"\mathrm{{{e.op}}}\!\left({expr_to_latex(e.operand)}\right)"
        if isinstance(e, FunctionCall):
            args = ", ".join(expr_to_latex(a) for a in e.args)
            if e.func_name == "sqrt":
                return rf"\sqrt{{{args}}}"
            name = _FUNC_LATEX.get(e.func_name) or rf"\mathrm{{{_sym(str(e.func_name))}}}"
            return rf"{name}\!\left({args}\right)"
        if isinstance(e, CustomCall):
            args = ", ".join(expr_to_latex(a) for a in e.args)
            return rf"\mathrm{{{_sym(str(e.name))}}}\!\left({args}\right)"
        if isinstance(e, MatMulExpression):
            mm_left = expr_to_latex(e.left, _MATMUL_PREC)
            mm_right = expr_to_latex(e.right, _MATMUL_PREC)
            return _paren(rf"{mm_left}\,{mm_right}", _MATMUL_PREC, parent_prec)
        if isinstance(e, SumExpression):
            inner = rf"\textstyle\sum {expr_to_latex(e.operand, 3)}"
            return _paren(inner, _SUM_PREC, parent_prec)
        if isinstance(e, SumOverExpression):
            return _sum_over_to_latex(e, parent_prec)
        if isinstance(e, Expression):
            return _latex_text(repr(e))
        return _fmt_num(e) if _is_number(e) else _latex_text(str(e))
    except Exception:
        return _latex_text(repr(e))


def _paren(inner: str, own_prec: int, parent_prec: int) -> str:
    """Wrap *inner* in ``\\left(...\\right)`` when a tighter-binding parent needs it."""
    if own_prec < parent_prec:
        return rf"\left({inner}\right)"
    return inner


# Cap on the number of explicit terms shown in a fully-expanded indexed summation
# before an ellipsis; keeps a large `dm.sum(..., over=...)` from flooding the render.
_SUM_OVER_MAX_TERMS = 8


def _sum_over_to_latex(e: Any, parent_prec: int) -> str:
    """Render a ``SumOverExpression`` (``dm.sum(expr(i) for i in set)``) as a proper
    ``\\sum`` of its expanded terms, rather than the raw ``Σ[N terms]`` repr (L3).

    ``SumOverExpression`` stores only the already-expanded term list (no symbolic
    index set survives expansion), so the faithful render is the explicit sum of
    those terms; long sums are truncated with ``+ \\cdots`` so the display stays
    compact. The whole group parenthesises under a tighter parent (e.g. ``(...)^2``).
    """
    terms = list(getattr(e, "terms", []) or [])
    if not terms:
        return r"\textstyle\sum \left(\ \right)"
    shown = terms[:_SUM_OVER_MAX_TERMS]
    parts = [expr_to_latex(t, 2) for t in shown]
    joined = " + ".join(parts)
    if len(terms) > _SUM_OVER_MAX_TERMS:
        joined += r" + \cdots"
    inner = rf"\textstyle\sum \left( {joined} \right)"
    return _paren(inner, _SUM_PREC, parent_prec)


def _binop_to_latex(e: Any, parent_prec: int) -> str:
    op = e.op
    if op == "/":
        return rf"\frac{{{expr_to_latex(e.left)}}}{{{expr_to_latex(e.right)}}}"
    if op == "**":
        return f"{expr_to_latex(e.left, 5)}^{{{expr_to_latex(e.right)}}}"
    prec = _PREC.get(op, 1)
    left = expr_to_latex(e.left, prec)
    # right operand of a non-associative op needs one extra precedence level
    right = expr_to_latex(e.right, prec + (1 if op in ("-", "/") else 0))
    sym = {"+": "+", "-": "-", "*": r"\cdot", "@": r"\,"}[op]
    s = f"{left} {sym} {right}"
    if prec < parent_prec:
        return rf"\left({s}\right)"
    return s


def _constraint_to_latex(c: Any) -> str:
    """Render a constraint. Constraints are normalised to ``body sense 0``; when the
    body is a top-level subtraction ``L - R`` we un-normalise to ``L sense R`` for a
    natural reading (e.g. ``x + 5y >= 4`` instead of ``4 - (x + 5y) <= 0``)."""
    # Non-arithmetic constraint types (indicator / disjunctive / SOS / logical,
    # from if_then/either_or/m.logical) carry no .body/.sense — render a readable
    # placeholder instead of raising AttributeError and crashing the whole model
    # display / Jupyter _repr_ (L1).
    if not (hasattr(c, "sense") and hasattr(c, "body")):
        kind = type(c).__name__.lstrip("_")
        label = getattr(c, "name", None)
        text = f"{kind}: {label}" if label else kind
        escaped = text.replace("_", r"\_")
        return rf"\text{{[{escaped}]}}"
    sense = {"<=": r"\le", ">=": r"\ge", "==": "="}.get(c.sense, c.sense)
    body = c.body
    if isinstance(body, BinaryOp) and body.op == "-":
        return f"{expr_to_latex(body.left)} {sense} {expr_to_latex(body.right)}"
    rhs = _fmt_num(c.rhs) if getattr(c, "rhs", 0.0) else "0"
    return f"{expr_to_latex(body)} {sense} {rhs}"


def _linear_row_to_latex(row: Any) -> str:
    """Render a builder-resident linear row (``export._common.LinearRow``) as
    ``sum(coeff * var[idx]) sense rhs`` (L2).

    These rows come from the fast-construction API (``Model.constraint`` fast path /
    ``add_linear_constraints``); they live only in ``model._builder_linear_blocks``
    and never reach ``model._constraints``. Rendering them keeps the display honest.
    """
    sense = {"<=": r"\le", ">=": r"\ge", "==": "="}.get(row.sense, row.sense)
    terms = getattr(row, "terms", None)
    if terms:
        pieces: list[str] = []
        for var, local, coeff in terms:
            # A vector variable's element uses its flat local index as a subscript;
            # a scalar carries no index.
            base = _sym(var.name)
            sym = base if getattr(var, "size", 1) <= 1 else f"{base}_{{{local}}}"
            if coeff == 1.0:
                term = sym
            elif coeff == -1.0:
                term = f"-{sym}"
            else:
                term = rf"{_fmt_num(coeff)}\,{sym}"
            if pieces and not term.startswith("-"):
                pieces.append(f"+ {term}")
            elif pieces:
                pieces.append(f"- {term[1:]}")
            else:
                pieces.append(term)
        body = " ".join(pieces) if pieces else "0"
    else:
        body = "0"
    return f"{body} {sense} {_fmt_num(row.rhs)}"


def _var_domain_to_latex(v: Any) -> str:
    vtype = v.var_type.value if hasattr(v.var_type, "value") else str(v.var_type)
    lb = float(np.min(v.lb))
    ub = float(np.max(v.ub))
    big = 1e15
    name = _sym(v.name) + (r"_{i}" if getattr(v, "size", 1) > 1 else "")
    if vtype == "binary":
        return rf"{name} \in \{{0, 1\}}"
    if lb > -big and ub < big:
        rng = rf"{_fmt_num(lb)} \le {name} \le {_fmt_num(ub)}"
    elif lb > -big:
        rng = rf"{name} \ge {_fmt_num(lb)}"
    elif ub < big:
        rng = rf"{name} \le {_fmt_num(ub)}"
    else:
        rng = ""
    domain = {"integer": r"\mathbb{Z}", "continuous": r"\mathbb{R}"}.get(vtype, "")
    if vtype == "integer":
        return rf"{rng},\ {name} \in \mathbb{{Z}}" if rng else rf"{name} \in \mathbb{{Z}}"
    return rng or rf"{name} \in {domain}"


def _all_constraint_rows(model: Any) -> tuple[list, list]:
    """Return ``(expression_constraint_rows, builder_linear_rows)`` for *model*.

    The expression rows are the *full* ``model._constraints`` list — including the
    non-arithmetic GDP/indicator/SOS/logical constraint objects that render as an
    L1 placeholder (``iter_all_rows`` filters those out, but the display must keep
    showing them). The builder rows come from the shared X-1 primitive
    ``export._common.iter_all_rows`` so builder-resident fast-path rows appear too
    (L2). The import is local to avoid a heavyweight import at module load and to
    keep the display resilient: if the export helper is unavailable or the model is
    malformed, fall back to the expression-path constraints alone rather than
    crashing a Jupyter ``_repr_``.
    """
    expr_rows = list(getattr(model, "_constraints", []) or [])
    try:
        from discopt.export._common import iter_builder_linear_rows

        builder_rows = list(iter_builder_linear_rows(model))
    except Exception:
        builder_rows = []
    return expr_rows, builder_rows


def model_to_latex(model: Any, max_rows: int | None = None, env: str = "aligned") -> str:
    """Render *model* to a LaTeX ``aligned`` block in standard PSE form.

    ``max_rows`` caps the number of constraint and variable rows shown (``None`` =
    no limit); excess is replaced with a ``\\vdots`` summary row.
    """
    rows: list[str] = []
    obj = getattr(model, "_objective", None)
    if obj is not None:
        sense = "minimize" if obj.sense.value == "minimize" else "maximize"
        rows.append(rf"& \text{{{sense}}} \quad && {expr_to_latex(obj.expression)} \\")
    else:
        rows.append(r"& \text{find} \quad && x \\")

    # Route through the shared X-1 primitive so builder-resident fast-path rows
    # (Model.constraint fast path / add_linear_constraints) render too — reading only
    # `_constraints` would silently hide them and misrepresent the model (L2).
    expr_rows, builder_rows = _all_constraint_rows(model)
    renderers = [(_constraint_to_latex, c) for c in expr_rows] + [
        (_linear_row_to_latex, r) for r in builder_rows
    ]
    n_cons = len(renderers)
    shown = renderers if max_rows is None else renderers[:max_rows]
    for i, (render, c) in enumerate(shown):
        lead = r"\text{subject to}" if i == 0 else ""
        rows.append(rf"& {lead} \quad && {render(c)} \\")
    if max_rows is not None and n_cons > max_rows:
        lead = r"\text{subject to}" if not shown else ""
        rows.append(rf"& {lead} \quad && \vdots \quad (\text{{{n_cons} constraints}}) \\")

    variables = list(getattr(model, "_variables", []) or [])
    rows.extend(_variable_rows(variables, max_rows))
    body = "\n".join(rows)
    return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"


def _variable_rows(variables: list, max_rows: int | None) -> list[str]:
    if not variables:
        return []
    if max_rows is not None and len(variables) > max_rows:
        # Summarise by type rather than listing hundreds of declarations.
        counts: dict[str, int] = {}
        for v in variables:
            t = v.var_type.value if hasattr(v.var_type, "value") else str(v.var_type)
            counts[t] = counts.get(t, 0) + 1
        parts = ", ".join(f"{n}\\ \\text{{{t}}}" for t, n in counts.items())
        return [rf"& \text{{with}} \quad && {parts} \text{{ variables}} \\"]
    out = []
    for j, v in enumerate(variables):
        lead = r"\text{with}" if j == 0 else ""
        out.append(rf"& {lead} \quad && {_var_domain_to_latex(v)} \\")
    return out


def model_to_html(model: Any, max_rows: int | None = None) -> str:
    """Standalone HTML rendering: a titled block with the PSE LaTeX rendered via the
    notebook's MathJax. Falls back gracefully to showing the LaTeX source."""
    name = getattr(model, "name", "model")
    n_v = len(getattr(model, "_variables", []) or [])
    # Count builder-resident fast-path rows too, so the header count matches the
    # rendered body and does not misreport a fast-API model as "0 constraints" (L2).
    expr_rows, builder_rows = _all_constraint_rows(model)
    n_c = len(expr_rows) + len(builder_rows)
    latex = model_to_latex(model, max_rows=max_rows)
    return (
        f'<div class="discopt-model">'
        f'<div style="font-weight:600">Model <code>{_escape_html(str(name))}</code>'
        f' <span style="color:#888;font-weight:400">'
        f"({n_v} variable{'s' if n_v != 1 else ''}, "
        f"{n_c} constraint{'s' if n_c != 1 else ''})</span></div>"
        f"$$\n{latex}\n$$"
        f"</div>"
    )


# LaTeX special characters that must be escaped when arbitrary text appears in a
# math environment (inside `\text{...}` or a fallback). Ordered so the backslash
# substitution runs first and does not re-escape the escapes it introduces.
_LATEX_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def _latex_text(s: str) -> str:
    """Render arbitrary prose safely inside a math environment as ``\\text{...}``.

    Escapes the LaTeX specials (``% _ # & $ { } \\ ~ ^``) so an unknown-node repr or
    a stray string cannot break math mode, and — unlike the old shared escaper —
    never injects HTML entities (``&amp;``) into a LaTeX ``aligned`` block (L7).
    """
    out = s
    for ch, rep in _LATEX_ESCAPES:
        out = out.replace(ch, rep)
    return rf"\text{{{out}}}"


def _escape_html(s: str) -> str:
    """HTML-escape ``< > &`` for text placed in the HTML chrome (not math) (L7)."""
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "<" in s or ">" in s or "&" in s
        else s
    )


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))
