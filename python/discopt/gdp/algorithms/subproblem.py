"""The fixed-configuration subproblem of the native GDP algorithms (#1775 G3.2).

For a full selection of disjuncts, :func:`solve_subproblem` builds the reduced
model of spec §5.2 (:func:`~discopt.gdp.state.reduced_subproblem`: global rows,
the selected disjuncts' rows, pinned indicators; the inactive rows are gone) and
solves it within the caller's :class:`~discopt.gdp.algorithms.base.Deadline`:

* ``mode="local"`` -- one local NLP (POUNCE or Ipopt), as LOA's subproblem. A
  local solve proves nothing: a failure is ``"unresolved"``, never
  ``"infeasible"``, and a point is kept only when it verifies on the reduced
  model (rows, bounds and integrality -- a discrete column the NLP relaxed and
  left fractional makes the point unresolved, not an incumbent).
* ``mode="global"`` -- discopt itself (``Model.solve``) on the reduced model,
  with the deadline remainder and the caller's ``gap_tolerance``. Its
  certificate is read from ``gap_certified``: ``"optimal"`` with a valid
  ``bound`` for the configuration, a certified ``"infeasible"``, or an
  uncertified ``"feasible"`` / ``"unresolved"``.

The reduced model shares the original's variables, so ``x`` is laid out like the
original model's flat vector. The caller still verifies an incumbent on the
original model (:meth:`GDPAlgorithm.offer_incumbent`).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.gdp.algorithms.base import Deadline
from discopt.gdp.state import reduced_subproblem
from discopt.gdp.tree import DisjunctNode
from discopt.modeling.core import Model

SUBPROBLEM_MODES = ("local", "global")


@dataclass
class SubproblemResult:
    """``status`` is ``"optimal"`` (global mode, certified), ``"feasible"`` (a
    verified point, not proved optimal for the configuration), ``"infeasible"``
    (certified: the configuration can be excluded) or ``"unresolved"`` (nothing
    proved; the configuration must stay open, C-35). ``objective`` and ``bound``
    are in model units; ``bound`` is a valid lower bound (upper, for MAXIMIZE) on
    the configuration, or ``None``."""

    status: str
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    bound: Optional[float] = None
    reason: str = ""

    @property
    def certified(self) -> bool:
        return self.status in ("optimal", "infeasible")


def _flat(model: Model, x_dict: dict) -> Optional[np.ndarray]:
    parts = []
    for v in model._variables:
        if v.name not in x_dict:
            return None
        parts.append(np.asarray(x_dict[v.name], dtype=np.float64).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0)


def _verified(sub: Model, x: Optional[np.ndarray]) -> tuple[Optional[float], str]:
    """The point's objective when it verifies on ``sub``, else ``(None, reason)``."""
    from discopt.validation.feasibility import verify_point

    if x is None:
        return None, "no point"
    vr = verify_point(sub, x, with_objective=True)
    if not vr.ok or vr.objective is None or not math.isfinite(vr.objective):
        return None, vr.reason or "objective not finite"
    return float(vr.objective), ""


def _local(sub: Model, limit: float, nlp_solver: str) -> SubproblemResult:
    from discopt.solvers import SolveStatus, pounce_incumbent_options, pounce_option_defaults

    if nlp_solver == "ipopt":
        from discopt.solvers.nlp_ipopt import solve_nlp_from_model
    else:
        from discopt.solvers.nlp_pounce import solve_nlp_from_model
    # The point is the product (#945): request the incumbent options.
    opts = pounce_option_defaults()
    opts.update(pounce_incumbent_options())
    opts["max_wall_time"] = max(limit, 0.05)
    r = solve_nlp_from_model(sub, options=opts)
    x = None if r.x is None else np.asarray(r.x, dtype=np.float64)
    if r.status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT, SolveStatus.TIME_LIMIT):
        return SubproblemResult("unresolved", reason=f"local NLP status {r.status}")
    obj, why = _verified(sub, x)
    if obj is None:
        return SubproblemResult("unresolved", reason=f"local point does not verify: {why}")
    return SubproblemResult("feasible", x=x, objective=obj)


def _global(sub: Model, limit: float, gap_tolerance: float) -> SubproblemResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = sub.solve(time_limit=limit, gap_tolerance=gap_tolerance)
    if r.status == "infeasible":
        return SubproblemResult("infeasible", reason="certified by discopt")
    bound = r.bound if (r.bound_valid and r.bound is not None and math.isfinite(r.bound)) else None
    x = None if r.x is None else _flat(sub, r.x)
    obj, why = _verified(sub, x)
    if obj is None:
        return SubproblemResult(
            "unresolved", bound=bound, reason=f"status {r.status}; point: {why}"
        )
    if r.status == "optimal" and r.gap_certified and bound is not None:
        return SubproblemResult("optimal", x=x, objective=obj, bound=bound)
    return SubproblemResult("feasible", x=x, objective=obj, bound=bound, reason=r.status)


def solve_subproblem(
    model: Model,
    selected: Iterable[DisjunctNode],
    deadline: Deadline,
    *,
    mode: str = "local",
    gap_tolerance: float = 1e-4,
    nlp_solver: str = "pounce",
    time_cap: Optional[float] = None,
) -> SubproblemResult:
    """Solve ``model`` with the disjuncts ``selected`` fixed; see the module docstring.

    The time limit is ``deadline.sub_limit(time_cap)``; with nothing left the
    result is ``"unresolved"`` and nothing is solved.
    """
    if mode not in SUBPROBLEM_MODES:
        raise ValueError(f"mode must be one of {SUBPROBLEM_MODES}, got {mode!r}")
    if nlp_solver not in ("pounce", "ipopt"):
        raise ValueError(f"nlp_solver must be 'pounce' or 'ipopt', got {nlp_solver!r}")
    sub = reduced_subproblem(model, selected)
    limit = deadline.sub_limit(time_cap)
    if not limit > 0.0:
        return SubproblemResult("unresolved", reason="deadline expired")
    if mode == "local":
        return _local(sub, limit, nlp_solver)
    return _global(sub, limit, gap_tolerance)


__all__ = ["SUBPROBLEM_MODES", "SubproblemResult", "solve_subproblem"]
