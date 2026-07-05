"""
MPS (Mathematical Programming System) file export for discopt models.

Supports LP, MILP, and QP problems. For QP objectives, the QUADOBJ
section is emitted. Nonlinear models raise ``ValueError``.

The output uses free MPS format for readability and to avoid the
fixed-width column limitations of the original MPS specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def to_mps(model: Model, path: str | Path | None = None) -> str | None:
    """Export a discopt Model to free MPS format.

    Parameters
    ----------
    model : Model
        A discopt optimization model. Must be linear or quadratic;
        nonlinear expressions raise ``ValueError``.
    path : str or Path, optional
        If provided, write the MPS string to this file path and
        return ``None``. Otherwise return the MPS string.

    Returns
    -------
    str or None
        The MPS string if *path* is ``None``, otherwise ``None``.

    Raises
    ------
    ValueError
        If the model contains nonlinear (non-quadratic) expressions.
    """
    model.validate()
    flat_vars = flatten_variables(model)
    var_names = [name for name, _, _, _, _ in flat_vars]

    lines: list[str] = []

    # NAME
    lines.append(f"NAME          {model.name}")

    # ROWS
    lines.append("ROWS")
    # Objective row
    obj_row_name = "OBJ"
    lines.append(f" N  {obj_row_name}")

    # Constraints — both expression-path rows and fast-API / builder-resident rows
    # (X-1: rows from `add_linear_constraints` / the `Model.constraint` fast path
    # live only in the Rust builder; without them the export is an empty model).
    mvars = model._variables
    row_senses: list[str] = []
    con_data: list[tuple[dict[int, float], float]] = []
    constraint_row_names = []
    for i, con in enumerate(model._constraints):
        row_name = _sanitize_name(con.name if con.name else f"C{i}")
        lin, const = extract_linear_terms(con.body, flat_vars, model_vars=mvars)
        constraint_row_names.append(row_name)
        row_senses.append(con.sense)
        con_data.append((lin, -const))
    for j, brow in enumerate(iter_builder_linear_rows(model)):
        row_name = _sanitize_name(brow.name if brow.name else f"B{j}")
        constraint_row_names.append(row_name)
        row_senses.append(brow.sense)
        con_data.append((dict(brow.coeffs), brow.rhs))

    for row_name, sense in zip(constraint_row_names, row_senses):
        if sense == "<=":
            lines.append(f" L  {row_name}")
        elif sense == ">=":
            lines.append(f" G  {row_name}")
        elif sense == "==":
            lines.append(f" E  {row_name}")
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")

    # COLUMNS
    lines.append("COLUMNS")

    # Build objective linear/quadratic coefficients. A builder-resident objective
    # (`add_linear_objective` / `add_quadratic_objective`, X-1) leaves a zero
    # placeholder in `model._objective` — recover its real coefficients rather than
    # emit `obj: 0`.
    assert model._objective is not None
    builder_obj = builder_objective(model)
    if builder_obj is not None:
        obj_linear, obj_quad_map, obj_const, _obj_sense = builder_obj
        obj_quad = obj_quad_map or {}
    else:
        obj_expr = model._objective.expression
        obj_quad, obj_linear, obj_const = extract_quadratic_terms(
            obj_expr, flat_vars, model_vars=mvars
        )

    has_quad = len(obj_quad) > 0

    # Constraint coefficients (`con_data`) were assembled above from both
    # expression-path and builder-resident rows.

    # Track which variables are integer/binary for MARKER sections
    int_marker_open = False

    for j, (vname, vtype, _, _, _) in enumerate(flat_vars):
        is_int = vtype in (VarType.BINARY, VarType.INTEGER)

        # Open integer marker if needed
        if is_int and not int_marker_open:
            lines.append("    MARKER        'MARKER'      'INTORG'")
            int_marker_open = True
        elif not is_int and int_marker_open:
            lines.append("    MARKER        'MARKER'      'INTEND'")
            int_marker_open = False

        # Objective coefficient
        if j in obj_linear and obj_linear[j] != 0.0:
            lines.append(f"    {vname}  {obj_row_name}  {_fmt(obj_linear[j])}")

        # Constraint coefficients
        for i, (lin, _rhs) in enumerate(con_data):
            if j in lin and lin[j] != 0.0:
                lines.append(f"    {vname}  {constraint_row_names[i]}  {_fmt(lin[j])}")

    # Close integer marker if still open
    if int_marker_open:
        lines.append("    MARKER        'MARKER'      'INTEND'")

    # RHS
    lines.append("RHS")
    for i, (_lin, rhs) in enumerate(con_data):
        if rhs != 0.0:
            lines.append(f"    RHS  {constraint_row_names[i]}  {_fmt(rhs)}")

    # Objective constant as RHS of OBJ row (negated, since MPS convention
    # treats it as obj_row - RHS_obj = 0, i.e. offset)
    if obj_const != 0.0:
        lines.append(f"    RHS  {obj_row_name}  {_fmt(-obj_const)}")

    # RANGES (not needed for our models)

    # BOUNDS
    lines.append("BOUNDS")
    for vname, vtype, _shape, lb, ub in flat_vars:
        lb_val = float(lb)
        ub_val = float(ub)

        if vtype == VarType.BINARY:
            lines.append(f" BV BND  {vname}")
        elif lb_val <= -1e19 and ub_val >= 1e19:
            # Free variable
            lines.append(f" FR BND  {vname}")
        elif lb_val <= -1e19:
            # Only upper bound
            lines.append(f" MI BND  {vname}")
            lines.append(f" UP BND  {vname}  {_fmt(ub_val)}")
        elif ub_val >= 1e19:
            # Only lower bound
            lines.append(f" LO BND  {vname}  {_fmt(lb_val)}")
        elif np.isclose(lb_val, ub_val, atol=1e-12):
            # Fixed variable
            lines.append(f" FX BND  {vname}  {_fmt(lb_val)}")
        else:
            # Both bounds
            lines.append(f" LO BND  {vname}  {_fmt(lb_val)}")
            lines.append(f" UP BND  {vname}  {_fmt(ub_val)}")

    # QUADOBJ (for QP problems)
    if has_quad:
        lines.append("QUADOBJ")
        # MPS QUADOBJ stores the Q matrix (the full quadratic form is
        # 0.5 * x' Q x, but discopt extract_quadratic_terms returns
        # coefficients of x_i * x_j directly). The QUADOBJ section
        # expects the upper-triangular entries of Q where the objective
        # quadratic contribution is 0.5 * sum Q_{ij} x_i x_j.
        # So we need to multiply by 2 for off-diagonal and keep diagonal as-is?
        # Actually, the convention varies. The standard is:
        #   QUADOBJ stores Q_{ij} for i <= j.
        #   The objective contribution is 0.5 * x' Q x.
        #   So if we have coeff * x_i * x_j, then Q_{ij} = coeff (for i==j)
        #   or Q_{ij} = coeff (and Q_{ji} = coeff) for i!=j.
        # Since extract_quadratic_terms returns the coefficient of x_i * x_j
        # (not 0.5 * Q), we need Q_{ij} = 2 * coeff for i==j? No.
        # Let's think carefully:
        #   0.5 * Q_{ii} * x_i^2 = coeff * x_i^2 => Q_{ii} = 2*coeff
        #   0.5 * (Q_{ij} + Q_{ji}) * x_i * x_j = coeff * x_i * x_j
        #     => Q_{ij} = coeff (upper triangle only, since we store each pair once)
        # Wait, for upper triangular storage: the contribution of (i,j) with i<j
        # is 0.5 * Q_{ij} * x_i * x_j + 0.5 * Q_{ji} * x_j * x_i.
        # If Q is symmetric, that's Q_{ij} * x_i * x_j.
        # So Q_{ij} = coeff for i != j.
        # For diagonal: 0.5 * Q_{ii} * x_i^2 = coeff => Q_{ii} = 2*coeff
        for (i, j), coeff in sorted(obj_quad.items()):
            if coeff == 0.0:
                continue
            vi = var_names[i]
            vj = var_names[j]
            if i == j:
                lines.append(f"    {vi}  {vj}  {_fmt(2.0 * coeff)}")
            else:
                # Only emit upper triangle (i <= j)
                if i <= j:
                    lines.append(f"    {vi}  {vj}  {_fmt(coeff)}")
                else:
                    lines.append(f"    {vj}  {vi}  {_fmt(coeff)}")

    lines.append("ENDATA")

    # Handle maximization: MPS is always minimization.
    # For maximize, we negate all objective coefficients.
    # But we already built the coefficients from the expression as-is.
    # We need to handle this at extraction time or post-process.
    # Actually, the standard approach: if maximizing, negate obj coefficients
    # and negate the result. But MPS doesn't have a native maximize.
    # The OBJSENSE section (extension) or RANGES trick is used.
    # Let's use the OBJSENSE extension supported by most solvers.
    assert model._objective is not None
    if builder_obj is not None:
        is_max = str(builder_obj[3]).lower().startswith("max")
    else:
        is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
    if is_max:
        # Insert OBJSENSE MAX after NAME
        lines.insert(1, "OBJSENSE MAX")

    mps_str = "\n".join(lines) + "\n"

    if path is not None:
        Path(path).write_text(mps_str)
        return None
    return mps_str


def _sanitize_name(name: str) -> str:
    """Sanitize a name for MPS format (replace spaces/special chars)."""
    return name.replace(" ", "_").replace("-", "_")


def _fmt(value: float) -> str:
    """Format a float for MPS output."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.15g}"
