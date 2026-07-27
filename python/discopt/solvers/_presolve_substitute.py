"""Solve-path entry for the batch substitution aggregator (issue #844, P2(a′)).

The Rust pass ``ModelRepr.substitute`` rewrites a variable determined by a
linear equality out of *every* expression it appears in and drops the defining
row (see ``crates/discopt-core/src/presolve/substitute.rs``). Until this module
existed the reduced model was computed and then discarded — the solve path
consumed only the tightened *bounds*
(``presolve_pipeline.propagate_bounds_to_model``).

This module closes that loop:

1. build the pristine ``ModelRepr``;
2. substitute to a fixed point;
3. reconstruct a Python ``Model`` from the reduced repr and solve **that**;
4. lift the incumbent back through the postsolve chain and report it in the
   ORIGINAL variables.

Soundness rules, in force regardless of what the reduced solve reports:

- an incumbent that cannot be inverted, or whose lifted point is not feasible
  for the **pristine** model (the #779 guard), is *discarded* — this function
  returns ``None`` and the caller runs the ordinary path. A point that cannot
  be verified is never reported;
- the reported objective is the one **recomputed on the pristine model** at the
  lifted point, not the reduced model's number, and the two must agree;
- the dual bound is carried over unchanged, which is valid because the
  substitution is an exact reformulation (equal feasible sets under the affine
  bijection, equal objective), not a relaxation.

Gated by ``DISCOPT_PRESOLVE_SUBSTITUTE``, default **OFF** (CLAUDE.md §5:
bound-changing work ships behind a flag until a differential panel passes).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import SolveResult

logger = logging.getLogger(__name__)

#: Objective agreement tolerance between the reduced and pristine evaluations.
_OBJ_TOL = 1e-6
#: Feasibility tolerance for the pristine-model check (matches the #779 guard).
_FEAS_TOL = 1e-5

#: Re-entrancy guard: the reduced solve must not substitute again.
_state = threading.local()


def substitution_enabled() -> bool:
    """True when ``DISCOPT_PRESOLVE_SUBSTITUTE`` is set to a truthy value."""
    return os.environ.get("DISCOPT_PRESOLVE_SUBSTITUTE", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


def _in_reduced_solve() -> bool:
    return bool(getattr(_state, "active", False))


def build_reduced(model) -> Optional[tuple[Any, Any, Any]]:
    """``(reduced_model, chain, pristine_repr)`` or ``None`` if not applicable.

    ``None`` means "nothing to gain / not safe here": the flag is off, the Rust
    repr could not be built, the model carries complementarity relations, or no
    variable was eliminated.
    """
    if not substitution_enabled() or _in_reduced_solve():
        return None
    try:
        from discopt._rust import model_to_repr
        from discopt.modeling.core import model_from_repr
    except Exception as exc:  # pragma: no cover - import-time capability probe
        logger.debug("substitution presolve unavailable: %s", exc)
        return None

    pristine = model_to_repr(model, getattr(model, "_builder", None))
    reduced_repr, chain = pristine.substitute(4)
    if chain.refused is not None:
        logger.info("substitution presolve declined: %s", chain.refused)
        return None
    if chain.variables_eliminated == 0:
        return None
    reduced_model = model_from_repr(reduced_repr, f"{model.name}_substituted")
    logger.info(
        "substitution presolve: %d -> %d vars, %d -> %d constraints (%d sweeps)",
        pristine.n_vars,
        reduced_repr.n_vars,
        pristine.n_constraints,
        reduced_repr.n_constraints,
        chain.n_sweeps,
    )
    return reduced_model, chain, pristine


def _flat_point(model, x_dict) -> Optional[np.ndarray]:
    """Flatten a ``SolveResult.x`` dict into the model's variable order."""
    if not isinstance(x_dict, dict):
        return None
    flat: list[float] = []
    for v in model._variables:
        if v.name not in x_dict:
            return None
        flat.extend(np.asarray(x_dict[v.name], dtype=float).reshape(-1).tolist())
    return np.asarray(flat, dtype=float)


def lift_result(
    model, reduced_model, chain, pristine_repr, result: Optional[SolveResult]
) -> Optional[SolveResult]:
    """Map a reduced-space :class:`SolveResult` back to the original variables.

    Returns ``None`` when the result cannot be lifted **and verified**, in which
    case the caller must fall back to the ordinary solve path.
    """
    from discopt.solvers._convex_kernel import _incumbent_is_feasible, _unflatten

    if result is None:
        return None

    # A status with no incumbent still carries a valid dual bound for the
    # ORIGINAL problem (the transform is an equivalence), so it can be reported
    # as-is once we know there is nothing to invert.
    if result.objective is None or result.x is None:
        return result

    x_red = _flat_point(reduced_model, result.x)
    if x_red is None or x_red.size != reduced_model_n_vars(reduced_model):
        logger.warning("substitution postsolve: reduced point unusable; discarding incumbent")
        return None
    try:
        x_full = np.asarray(chain.postsolve(x_red.tolist()), dtype=float)
    except Exception as exc:
        logger.warning("substitution postsolve failed (%s); discarding incumbent", exc)
        return None
    if x_full.size != pristine_repr.n_vars or not np.all(np.isfinite(x_full)):
        logger.warning("substitution postsolve produced a bad point; discarding incumbent")
        return None

    # #779 guard: the lifted point must be feasible for the PRISTINE model.
    if not _incumbent_is_feasible(model, x_full, tol=_FEAS_TOL):
        logger.warning(
            "substitution postsolve: lifted incumbent is infeasible for the "
            "pristine model; discarding it and falling back"
        )
        return None

    obj_pristine, con_viol, bnd_viol = pristine_repr.evaluate_point(x_full.tolist())
    if not np.isfinite(obj_pristine):
        logger.warning("substitution postsolve: non-finite pristine objective; discarding")
        return None
    if abs(obj_pristine - float(result.objective)) > _OBJ_TOL * (1.0 + abs(obj_pristine)):
        logger.warning(
            "substitution postsolve: objective mismatch (pristine %.12g vs reduced "
            "%.12g); discarding incumbent",
            obj_pristine,
            result.objective,
        )
        return None
    if max(con_viol, bnd_viol) > _FEAS_TOL:
        logger.warning(
            "substitution postsolve: pristine violation %.3g exceeds tolerance; discarding",
            max(con_viol, bnd_viol),
        )
        return None

    x_dict, _ = _unflatten(model, x_full)
    result.x = x_dict
    result.objective = float(obj_pristine)
    result._model = model
    return result


def reduced_model_n_vars(reduced_model) -> int:
    """Total scalar variable count of a Python model."""
    return int(sum(int(v.size) for v in reduced_model._variables))


class reduced_solve_scope:  # noqa: N801 - context manager, lowercase by convention here
    """Marks the nested solve so it does not substitute recursively."""

    def __enter__(self):
        self._prev = getattr(_state, "active", False)
        _state.active = True
        return self

    def __exit__(self, *exc):
        _state.active = self._prev
        return False
