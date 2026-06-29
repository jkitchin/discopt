"""Write a Pyomo model to a temporary AMPL ``.nl`` file for discopt.

The bridge round-trips through ``.nl`` and maps the solution back by **column
order**, not by name (Pyomo and discopt name variables differently). To make that
mapping a direct index identity we disable two NL-writer transforms:

* ``linear_presolve=False`` — keep every model variable as a column, so the
  solution covers all of them and there is no eliminated-variable back-substitution.
* ``scale_model=False`` — emit the ``.nl`` in the original variable space, so the
  values discopt returns need no un-scaling.

The writer's :class:`NLWriterInfo` reports the exact variables (column order) and
constraints (row order) it wrote; ``discopt.from_nl`` reads those same rows/columns
in the same order, so ``info.variables[i]`` aligns with discopt's ``i``-th column
and ``info.constraints[i]`` with discopt's ``i``-th constraint row.
"""

from __future__ import annotations

from typing import Any


def write_nl(model: Any, nl_path: str) -> tuple[list, list, list]:
    """Write *model* to *nl_path* as an AMPL ``.nl`` file.

    Returns ``(vars_in_col_order, cons_in_row_order, eliminated_vars)`` — the first
    two are lists of Pyomo ``VarData`` / ``ConstraintData`` in the exact column/row
    order the ``.nl`` encodes (for index-aligned solution and dual mapping); the
    third is the (normally empty) list of presolve-eliminated ``(var, expr)`` pairs
    to back-substitute defensively.
    """
    from pyomo.repn.plugins.nl_writer import NLWriter

    with open(nl_path, "w") as stream:
        info = NLWriter().write(
            model,
            stream,
            linear_presolve=False,
            scale_model=False,
            skip_trivial_constraints=False,
        )

    # With presolve disabled this is normally empty; keep a defensive recovery so a
    # future writer-default change can't silently drop a variable's value.
    eliminated = list(getattr(info, "eliminated_vars", []) or [])
    return list(info.variables), list(info.constraints), eliminated
