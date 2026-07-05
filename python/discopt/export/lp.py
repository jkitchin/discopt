"""
CPLEX LP format export for discopt models.

Produces a human-readable LP file with sections: Minimize/Maximize,
Subject To, Bounds, Generals, Binaries, End.

Only linear and quadratic models are supported. Nonlinear expressions
raise ``ValueError``.
"""

from __future__ import annotations

from pathlib import Path

from discopt.export._common import builder_objective, iter_builder_linear_rows
from discopt.export._extract import (
    extract_linear_terms,
    extract_quadratic_terms,
    flatten_variables,
)
from discopt.modeling.core import (
    Model,
    ObjectiveSense,
    VarType,
)


def to_lp(model: Model, path: str | Path | None = None) -> str | None:
    """Export a discopt Model to CPLEX LP format.

    Parameters
    ----------
    model : Model
        A discopt optimization model. Must be linear or quadratic;
        nonlinear expressions raise ``ValueError``.
    path : str or Path, optional
        If provided, write the LP string to this file path and
        return ``None``. Otherwise return the LP string.

    Returns
    -------
    str or None
        The LP string if *path* is ``None``, otherwise ``None``.

    Raises
    ------
    ValueError
        If the model contains nonlinear (non-quadratic) expressions.
    """
    model.validate()
    flat_vars = flatten_variables(model)
    var_names = [name for name, _, _, _, _ in flat_vars]
    mvars = model._variables

    lines: list[str] = []

    # Comment header
    lines.append(f"\\ Problem: {model.name}")
    lines.append("")

    # Objective section. A builder-resident objective (`add_linear_objective` /
    # `add_quadratic_objective`, X-1) leaves a zero placeholder in `model._objective`
    # — recover its real coefficients and sense rather than emit `obj: 0`.
    assert model._objective is not None
    builder_obj = builder_objective(model)
    if builder_obj is not None:
        obj_linear, obj_quad_map, obj_const, builder_sense = builder_obj
        obj_quad = obj_quad_map or {}
        is_min = not str(builder_sense).lower().startswith("max")
    else:
        obj_expr = model._objective.expression
        obj_quad, obj_linear, obj_const = extract_quadratic_terms(
            obj_expr, flat_vars, model_vars=mvars
        )
        is_min = model._objective.sense == ObjectiveSense.MINIMIZE

    if is_min:
        lines.append("Minimize")
    else:
        lines.append("Maximize")

    obj_str = _format_linear_expr(obj_linear, var_names, obj_const)
    if not obj_str:
        obj_str = "0"
    lines.append(f"  obj: {obj_str}")

    # Quadratic objective
    if obj_quad:
        lines.append("  + [ ")
        q_terms: list[str] = []
        for (i, j), coeff in sorted(obj_quad.items()):
            if coeff == 0.0:
                continue
            # LP format quadratic section: coefficients are doubled
            # because the form is 0.5 * x'Qx, so we write 2*coeff
            c = 2.0 * coeff
            vi = var_names[i]
            vj = var_names[j]
            if i == j:
                q_terms.append(_format_coeff(c, len(q_terms) == 0) + f" {vi} ^ 2")
            else:
                q_terms.append(_format_coeff(c, len(q_terms) == 0) + f" {vi} * {vj}")
        lines.append("    " + " ".join(q_terms))
        lines.append("  ] / 2")

    lines.append("")

    # Subject To section. Emit both expression-path rows and fast-API /
    # builder-resident rows (X-1: rows from `add_linear_constraints` / the
    # `Model.constraint` fast path live only in the Rust builder; without them the
    # LP would have an empty `Subject To`).
    lines.append("Subject To")
    rows: list[tuple[str, dict[int, float], str, float]] = []
    for i, con in enumerate(model._constraints):
        con_name = (con.name if con.name else f"c{i}").replace(" ", "_").replace("-", "_")
        lin, const = extract_linear_terms(con.body, flat_vars, model_vars=mvars)
        rows.append((con_name, lin, con.sense, -const))
    for j, brow in enumerate(iter_builder_linear_rows(model)):
        con_name = (brow.name if brow.name else f"b{j}").replace(" ", "_").replace("-", "_")
        rows.append((con_name, dict(brow.coeffs), brow.sense, brow.rhs))

    for con_name, lin, sense_str, rhs in rows:
        expr_str = _format_linear_expr(lin, var_names, 0.0)
        if not expr_str:
            expr_str = "0"
        if sense_str == "<=":
            lp_sense = "<="
        elif sense_str == ">=":
            lp_sense = ">="
        elif sense_str == "==":
            lp_sense = "="
        else:
            raise ValueError(f"Unknown constraint sense: {sense_str}")
        lines.append(f"  {con_name}: {expr_str} {lp_sense} {_fmt_val(rhs)}")

    lines.append("")

    # Bounds section
    lines.append("Bounds")
    for vname, vtype, _shape, lb, ub in flat_vars:
        if vtype == VarType.BINARY:
            # Binary bounds are implicit (0 <= x <= 1)
            continue
        if lb <= -1e19 and ub >= 1e19:
            lines.append(f"  {vname} Free")
        elif lb <= -1e19:
            lines.append(f"  -Inf <= {vname} <= {_fmt_val(ub)}")
        elif ub >= 1e19:
            lines.append(f"  {_fmt_val(lb)} <= {vname} <= +Inf")
        else:
            lines.append(f"  {_fmt_val(lb)} <= {vname} <= {_fmt_val(ub)}")

    lines.append("")

    # Generals (integer variables)
    int_vars = [name for name, vt, _, _, _ in flat_vars if vt == VarType.INTEGER]
    if int_vars:
        lines.append("Generals")
        lines.append("  " + " ".join(int_vars))
        lines.append("")

    # Binaries
    bin_vars = [name for name, vt, _, _, _ in flat_vars if vt == VarType.BINARY]
    if bin_vars:
        lines.append("Binaries")
        lines.append("  " + " ".join(bin_vars))
        lines.append("")

    lines.append("End")

    lp_str = "\n".join(lines) + "\n"

    if path is not None:
        Path(path).write_text(lp_str)
        return None
    return lp_str


def _format_linear_expr(
    coeffs: dict[int, float],
    var_names: list[str],
    constant: float,
) -> str:
    """Format a linear expression as an LP-format string."""
    parts: list[str] = []

    # Constant first if nonzero
    if constant != 0.0:
        parts.append(_fmt_val(constant))

    for idx in sorted(coeffs.keys()):
        c = coeffs[idx]
        if c == 0.0:
            continue
        vname = var_names[idx]
        parts.append(_format_coeff(c, len(parts) == 0) + f" {vname}")

    return " ".join(parts)


def _format_coeff(c: float, is_first: bool) -> str:
    """Format a coefficient with appropriate sign."""
    if is_first:
        if c == 1.0:
            return ""
        if c == -1.0:
            return "-"
        return _fmt_val(c)
    else:
        if c == 1.0:
            return "+"
        if c == -1.0:
            return "-"
        if c > 0:
            return f"+ {_fmt_val(c)}"
        return f"- {_fmt_val(abs(c))}"


def _fmt_val(value: float) -> str:
    """Format a numeric value for LP output."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.15g}"
