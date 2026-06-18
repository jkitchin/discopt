"""Irreducible Infeasible Subsystem (IIS) computation.

When a model is infeasible, the full constraint set does not explain *why* — most
constraints are irrelevant to the conflict. An **IIS** is a minimal subset of
constraints (and variable bounds) that is itself infeasible but becomes feasible
if *any* single member is removed. It is the smallest self-contained explanation
of the infeasibility, the analogue of Gurobi's ``computeIIS`` / CPLEX's conflict
refiner.

The algorithm is **deletion filtering** (Chinneck & Dravnieks 1991): start from
the full (infeasible) model and tentatively drop one candidate at a time; if the
remainder is still *provably* infeasible the candidate is redundant and is
dropped permanently, otherwise it is essential and kept. What remains is an IIS.

Soundness with an incomplete solver: a candidate is dropped only when the reduced
model is **proven** infeasible (``status == "infeasible"``). A solve that is
merely inconclusive (e.g. a nonconvex MINLP that hits the time limit without a
proof) is treated as "keep the candidate". The returned set is therefore always a
genuine infeasible subsystem; it is *irreducible* exactly when every filtering
solve was conclusive (always the case for LP/MILP/convex models). See
:attr:`IISResult.proven_irreducible`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from discopt.modeling.core import (
    Constant,
    Constraint,
    Objective,
    ObjectiveSense,
    VarType,
)

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable

logger = logging.getLogger(__name__)

# discopt treats |bound| >= this as unbounded; relaxing a bound sets it here.
_INF = 1e20
_FINITE = 1e19


@dataclass
class IISResult:
    """An Irreducible Infeasible Subsystem.

    Attributes
    ----------
    constraints : list[Constraint]
        Constraints belonging to the IIS.
    variable_bounds : list[tuple[Variable, str]]
        Variable bounds in the IIS, each as ``(variable, "lower"|"upper")``.
    n_solves : int
        Number of feasibility solves performed (deletion-filter cost).
    proven_irreducible : bool
        ``True`` when every filtering solve was conclusive, so the result is a
        true (minimal) IIS. ``False`` means some solves were inconclusive and the
        set is a valid — but possibly non-minimal — infeasible subsystem.
    """

    constraints: list[Constraint] = field(default_factory=list)
    variable_bounds: list[tuple["Variable", str]] = field(default_factory=list)
    n_solves: int = 0
    proven_irreducible: bool = True

    def __len__(self) -> int:
        return len(self.constraints) + len(self.variable_bounds)

    def __bool__(self) -> bool:
        return len(self) > 0

    def summary(self) -> str:
        """A human-readable multi-line explanation of the infeasibility."""
        lines = [
            f"Irreducible Infeasible Subsystem ({len(self)} members"
            + ("" if self.proven_irreducible else ", not proven minimal")
            + "):"
        ]
        if self.constraints:
            lines.append("  Constraints:")
            for c in self.constraints:
                label = c.name if c.name else repr(c)
                lines.append(f"    - {label}:  {c.body} {c.sense} {c.rhs:g}")
        if self.variable_bounds:
            lines.append("  Variable bounds:")
            for var, which in self.variable_bounds:
                bound = var.lb if which == "lower" else var.ub
                op = ">=" if which == "lower" else "<="
                val = float(np.min(bound)) if which == "lower" else float(np.max(bound))
                lines.append(f"    - {var.name} {op} {val:g}")
        if not self:
            lines.append("  (empty)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _bound_candidates(model: "Model") -> list[tuple[int, str]]:
    """Finite lower/upper bounds of non-binary variables, as (var_index, side).

    Binary variables are skipped: their ``{0, 1}`` domain is intrinsic, not a
    removable bound. Infinite bounds cannot cause infeasibility and are excluded.
    """
    items: list[tuple[int, str]] = []
    for vi, v in enumerate(model._variables):
        if v.var_type == VarType.BINARY:
            continue
        if float(np.min(np.asarray(v.lb))) > -_FINITE:
            items.append((vi, "lower"))
        if float(np.max(np.asarray(v.ub))) < _FINITE:
            items.append((vi, "upper"))
    return items


def _solve_is_infeasible(
    model: "Model",
    active_constraints: list[Constraint],
    relaxed_bounds: set[tuple[int, str]],
    saved_bounds: list[tuple[np.ndarray, np.ndarray]],
    time_limit: float,
) -> tuple[bool, bool]:
    """Solve the model restricted to ``active_constraints`` with ``relaxed_bounds``
    dropped, under a feasibility (constant) objective.

    Returns ``(is_infeasible, conclusive)``: ``is_infeasible`` is True only on a
    proven ``"infeasible"`` status; ``conclusive`` is False when the solver could
    neither prove infeasibility nor find a feasible point.
    """
    saved_cons = model._constraints
    saved_obj = model._objective
    model._constraints = active_constraints
    model._objective = Objective(Constant(0.0), ObjectiveSense.MINIMIZE)
    for vi, side in relaxed_bounds:
        v = model._variables[vi]
        if side == "lower":
            v.lb = np.full_like(np.asarray(v.lb, dtype=np.float64), -_INF)
        else:
            v.ub = np.full_like(np.asarray(v.ub, dtype=np.float64), _INF)
    try:
        # ``solve`` is typed ``SolveResult | Iterator`` (streaming overload); the
        # non-streaming call returns a SolveResult with ``.status``.
        status = getattr(model.solve(time_limit=time_limit), "status", "error")
    except Exception as exc:  # noqa: BLE001 - a crashed feasibility probe is inconclusive
        logger.debug("IIS feasibility probe raised: %s", exc)
        status = "error"
    finally:
        model._constraints = saved_cons
        model._objective = saved_obj
        for vi, (lb, ub) in zip(range(len(model._variables)), saved_bounds):
            model._variables[vi].lb = lb
            model._variables[vi].ub = ub

    if status == "infeasible":
        return True, True
    if status in ("optimal", "feasible"):
        return False, True
    return False, False  # inconclusive (time_limit / error / unbounded)


def compute_iis(
    model: "Model",
    *,
    include_bounds: bool = True,
    time_limit: float = 30.0,
) -> IISResult:
    """Compute an Irreducible Infeasible Subsystem of an infeasible ``model``.

    Parameters
    ----------
    model : Model
        A model whose constraints (and optionally variable bounds) are
        collectively infeasible.
    include_bounds : bool, default True
        Also consider finite variable bounds as removable members of the IIS.
        Required to explain infeasibilities driven by bounds (e.g. ``x >= 5`` and
        ``x <= 2`` declared as bounds rather than constraints).
    time_limit : float, default 30.0
        Per-solve wall-clock limit for each feasibility probe.

    Returns
    -------
    IISResult
        The constraints and variable bounds forming the IIS.

    Raises
    ------
    ValueError
        If ``model`` is not (provably) infeasible to begin with — there is no
        infeasibility to explain.
    """
    all_constraints = list(model._constraints)
    bound_items = _bound_candidates(model) if include_bounds else []

    # Snapshot every variable's bounds once; each probe restores from this.
    saved_bounds = [(np.asarray(v.lb).copy(), np.asarray(v.ub).copy()) for v in model._variables]

    # Active set: indices into all_constraints, plus active bound items.
    active_con_idx = set(range(len(all_constraints)))
    active_bounds = set(bound_items)
    n_solves = 0
    proven = True

    def probe() -> tuple[bool, bool]:
        nonlocal n_solves
        n_solves += 1
        cons = [all_constraints[i] for i in sorted(active_con_idx)]
        relaxed = set(bound_items) - active_bounds
        return _solve_is_infeasible(model, cons, relaxed, saved_bounds, time_limit)

    # Baseline: the full system must be provably infeasible.
    base_infeasible, base_conclusive = probe()
    if not base_infeasible:
        if base_conclusive:
            raise ValueError(
                "compute_iis: the model is feasible — there is no infeasibility to explain."
            )
        raise ValueError(
            "compute_iis: could not prove the model infeasible within the time limit; "
            "an IIS is only defined for an infeasible model."
        )

    # Deletion filter over constraints, then bounds. Drop a member only when the
    # remainder is *proven* infeasible; an inconclusive probe keeps the member
    # (and flags the result as not-proven-minimal).
    for i in list(active_con_idx):
        active_con_idx.discard(i)
        infeasible, conclusive = probe()
        if infeasible:
            continue  # constraint i is redundant to the conflict
        active_con_idx.add(i)  # essential — keep it
        if not conclusive:
            proven = False

    for item in list(active_bounds):
        active_bounds.discard(item)
        infeasible, conclusive = probe()
        if infeasible:
            continue
        active_bounds.add(item)
        if not conclusive:
            proven = False

    return IISResult(
        constraints=[all_constraints[i] for i in sorted(active_con_idx)],
        variable_bounds=[(model._variables[vi], side) for (vi, side) in sorted(active_bounds)],
        n_solves=n_solves,
        proven_irreducible=proven,
    )
