"""Conflict analysis and no-good cuts.

When a partial assignment of binary variables is *jointly* infeasible — no
feasible solution sets those binaries that way — the assignment can be excluded
tree-wide with a **no-good cut**

    sum_{j: a_j = 1} (1 - x_j) + sum_{j: a_j = 0} x_j  >= 1,

which removes exactly the assignment ``a`` (on the involved binaries) and nothing
else. Adding such cuts prunes dead subtrees without weakening the global
optimality certificate.

Soundness rests on **feasibility-based bound tightening (FBBT)** as the
infeasibility oracle: FBBT only ever derives valid bounds, so if it produces an
empty interval for a subproblem, that subproblem is genuinely infeasible (FBBT
never reports a feasible problem as infeasible). A no-good cut emitted from an
FBBT-proven conflict therefore never excludes a feasible point — in particular
it never cuts off the global optimum.

The detector probes small subsets of binaries (up to ``max_order``) and keeps
only *minimal* conflicts (it skips assignments already excluded by a shorter
conflict). This mirrors the conflict analysis in SOTA MINLP solvers, realised
here entirely through the existing, validated FBBT engine.

Example
-------
>>> import discopt.modeling.core as dm
>>> from discopt.conflict import add_conflict_cuts
>>> m = dm.Model("c")
>>> x = [m.binary(f"x{i}") for i in range(3)]
>>> m.maximize(x[0] + 2 * x[1] + 3 * x[2])
>>> m.subject_to(x[0] + x[1] + x[2] <= 1)      # any two ones is infeasible
>>> add_conflict_cuts(m, max_order=2) > 0
True
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from discopt.modeling.core import Constraint, Expression, Model, VarType

__all__ = [
    "no_good_cut",
    "find_conflict_cuts",
    "add_conflict_cuts",
]


def _scalar_binaries(model: Model) -> list:
    """Return the model's scalar binary variables (size 1, type BINARY)."""
    return [v for v in model._variables if v.var_type == VarType.BINARY and v.size == 1]


def no_good_cut(assignment: list[tuple]) -> Constraint:
    """Build a no-good constraint excluding ``assignment``.

    ``assignment`` is a list of ``(binary_variable, value)`` pairs with
    ``value`` in ``{0, 1}``. The returned constraint is

        sum_{value=1} (1 - var) + sum_{value=0} var >= 1,

    which is violated only by the exact assignment and is satisfied by every
    other 0/1 combination of the involved variables.
    """
    if not assignment:
        raise ValueError("no_good_cut requires a non-empty assignment")
    expr: Optional[Expression] = None
    for var, val in assignment:
        term: Expression = (1 - var) if round(val) == 1 else var
        expr = term if expr is None else expr + term
    assert expr is not None
    result = expr >= 1
    assert isinstance(result, Constraint)
    return result


def _fbbt_infeasible(model: Model, tol: float = 1e-9, max_iter: int = 20) -> bool:
    """True if FBBT proves the current model (with its present bounds) infeasible."""
    from discopt.tightening import fbbt_box

    return fbbt_box(model, max_iter=max_iter, tol=tol).infeasible


def find_conflict_cuts(
    model: Model,
    *,
    max_order: int = 2,
    max_binaries: int = 24,
    tol: float = 1e-9,
) -> list[Constraint]:
    """Detect jointly-infeasible binary assignments and return no-good cuts.

    Probes subsets of up to ``max_order`` scalar binaries; for each assignment
    not already excluded by a shorter (minimal) conflict, fixes those binaries
    and runs FBBT. An FBBT-proven infeasible assignment yields a no-good cut.

    Returns the list of no-good :class:`Constraint` objects (not yet added to the
    model). Returns an empty list when the model has no scalar binaries or the
    binary count exceeds ``max_binaries``.
    """
    binaries = _scalar_binaries(model)
    if not binaries or len(binaries) > max_binaries:
        return []

    # Save/restore variable bounds around each probe.
    saved = [(np.array(v.lb), np.array(v.ub)) for v in model._variables]
    var_index = {id(v): i for i, v in enumerate(model._variables)}

    # Found minimal conflicts as frozensets of (var_id, value); a candidate
    # assignment is skipped if it is a superset of a known conflict.
    found: list[frozenset] = []
    cuts: list[Constraint] = []

    def _covered(assign_set: frozenset) -> bool:
        return any(c <= assign_set for c in found)

    try:
        for order in range(1, max_order + 1):
            for combo in itertools.combinations(binaries, order):
                for values in itertools.product((0, 1), repeat=order):
                    assign_set = frozenset((id(v), val) for v, val in zip(combo, values))
                    if _covered(assign_set):
                        continue
                    # Fix this assignment.
                    for v, val in zip(combo, values):
                        v.lb = np.array([float(val)])
                        v.ub = np.array([float(val)])
                    infeasible = _fbbt_infeasible(model, tol=tol)
                    # Restore the probed binaries before the next candidate.
                    for v, val in zip(combo, values):
                        i = var_index[id(v)]
                        v.lb = np.array(saved[i][0])
                        v.ub = np.array(saved[i][1])
                    if infeasible:
                        found.append(assign_set)
                        cuts.append(no_good_cut(list(zip(combo, values))))
    finally:
        for v, (lb, ub) in zip(model._variables, saved):
            v.lb = np.array(lb)
            v.ub = np.array(ub)

    return cuts


def add_conflict_cuts(
    model: Model,
    *,
    max_order: int = 2,
    max_binaries: int = 24,
    tol: float = 1e-9,
    name: Optional[str] = "no_good",
) -> int:
    """Detect conflicts and add the resulting no-good cuts to ``model``.

    Returns the number of cuts added. Every cut excludes only an FBBT-proven
    infeasible assignment, so the model's feasible region — and its optimum — is
    unchanged; only dead assignments are pruned.
    """
    cuts = find_conflict_cuts(model, max_order=max_order, max_binaries=max_binaries, tol=tol)
    for k, cut in enumerate(cuts):
        model.subject_to(cut, name=f"{name}_{k}" if name else None)
    return len(cuts)
