"""The single incumbent-feasibility verifier (#908).

Every solver path that promotes a point to an incumbent — the native kernel's
cutoff seed, the convex kernel's incumbent guard — must agree on what "feasible"
means, and must be right. Before this module there were two independent
implementations and both were wrong in the *wrongly-accept* direction.

What went wrong, and why the fix is shaped this way
---------------------------------------------------

Both verifiers advanced **one row index per** :class:`~discopt.modeling.core.Constraint`
**object** while :class:`~discopt._jax.nlp_evaluator.NLPEvaluator` emits **one row
per flat element**. A constraint body may be array-valued — ``x <= 1`` on a
3-vector is one ``Constraint`` and three rows — so on any model with a vector
constraint the two streams desynchronise and every check from that point on reads
the wrong row. Measured on a purpose-built corpus: a point violating row 2 of a
size-3 vector constraint **by 5.0** was reported feasible by both verifiers, and
4 of 5 constructed-infeasible points were wrongly accepted.

The fix is not to patch the arithmetic. Rows are enumerated from
:meth:`NLPEvaluator.constraint_row_map`, the evaluator's own map, so the
misalignment class is **structurally impossible**: the same
``_source_constraints`` / ``_constraint_flat_sizes`` that build the compiled
concatenation also drive the verification, and they cannot drift.

That row map also closes a second hole for free. The evaluator's row set is
``model._constraints`` **plus** ``model._builder_linear_constraints()`` (#840);
a verifier reading only ``_constraints`` never examined the builder-resident
linear rows at all — so a model built through ``add_linear_constraints`` or the
``constraint(fast=True)`` path had those rows silently unchecked.

Tolerance
---------

The old form was ``abs_tol + rel_tol * |residual|``, which is self-referential:
it scales with the *residual* rather than with the *row*. On an equality row it
collapses to a flat ``1e-6`` no matter whether the row's natural magnitude is 1
or 10,782, so the ``rel_tol`` term is arithmetically dead and the check is
scale-blind. Measured consequence in the other direction: ``nvs22``'s incumbent
was rejected at a *relative* residual of 8.1e-9 (absolute 1.71e-5 against a row
value of 2121.64).

This module keys the tolerance on the **row's** scale::

    violation_i <= abs_tol * max(1, |rhs_i|, max_j |J_ij| * max(1, |x_j|))

Two properties of that form are load-bearing and easy to get wrong:

* The relative coefficient is ``abs_tol`` (1e-6), **not** the repo's ``rel_tol``
  (1e-4). Using ``rel_tol`` would loosen every unit-scale row 100x.
* The Jacobian term is a *row-scale estimate*, not a slack. Dropping it makes the
  test **stricter**, never looser — which is why the fallback below is sound.

``test_vector_constraint_corpus.py`` carries a control for each naive widening of
this form, showing that each one accepts a bad point this form rejects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Absolute feasibility tolerance, and the *relative* coefficient in the
#: scale-keyed form above. Deliberately not ``rel_tol`` — see the module note.
ABS_TOL = 1e-6
#: Integrality tolerance for INTEGER/BINARY variables.
INT_TOL = 1e-5
#: Legacy relative coefficient, used ONLY for the variable-bound test, where it
#: is keyed on the bound (a genuine scale) rather than on the residual.
BOUND_REL_TOL = 1e-4


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a feasibility verification.

    ``objective`` is the point's TRUE objective in model units (MAXIMIZE
    un-negated), or ``None`` when it was not requested or could not be computed.
    ``reason`` names the first failing check — for logs, never for control flow.
    """

    ok: bool
    objective: Optional[float] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _sense_str(con) -> Optional[str]:
    """``Constraint.sense`` is a ``str`` on some paths and an enum on others."""
    s = con.sense
    if not isinstance(s, str):
        s = getattr(s, "value", None)
    return s if s in ("<=", ">=", "==") else None


def _row_violation(val: float, sense: str) -> float:
    """Signed violation of ``body <sense> 0``, ``<= 0`` meaning satisfied."""
    if sense == "<=":
        return val
    if sense == ">=":
        return -val
    return abs(val)


def check_variable_bounds(model, x_flat: np.ndarray) -> VerifyResult:
    """Variable bounds + integrality against the ORIGINAL declared model.

    Bounds use ``abs_tol + rel_tol * |bound|``. Unlike the old *constraint*
    tolerance this is keyed on the bound — a real scale — not on the residual, so
    it is not self-referential: a local NLP returns a bound-active variable a few
    ULPs off its bound, and on a large-magnitude bound (``tanksize`` x41 lb=536) a
    4e-6 absolute slack is 8e-9 relative, inside the regime the whole solver
    operates in.
    """
    from discopt.modeling.core import VarType

    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        vals = x_flat[off : off + size]
        if vals.shape[0] != size:
            return VerifyResult(False, None, f"length mismatch at variable {v.name!r}")
        lb_flat = np.asarray(v.lb, dtype=np.float64).flatten()
        ub_flat = np.asarray(v.ub, dtype=np.float64).flatten()
        lb_tol = ABS_TOL + BOUND_REL_TOL * np.abs(lb_flat)
        ub_tol = ABS_TOL + BOUND_REL_TOL * np.abs(ub_flat)
        if np.any(vals < lb_flat - lb_tol) or np.any(vals > ub_flat + ub_tol):
            return VerifyResult(False, None, f"variable {v.name!r} out of bounds")
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            if np.any(np.abs(vals - np.round(vals)) > INT_TOL):
                return VerifyResult(False, None, f"variable {v.name!r} not integral")
        off += size
    return VerifyResult(True)


def _row_scales(evaluator, x_flat: np.ndarray, rows: np.ndarray) -> Optional[np.ndarray]:
    """``max_j |J_ij| * max(1, |x_j|)`` for the given row indices, or ``None``.

    ``None`` means the Jacobian was unavailable, and the caller must then fall
    back to the Jacobian-free bound — which is STRICTER, so the fallback can only
    reject a point the full form would have accepted. It can never accept one the
    full form would reject.
    """
    try:
        J = np.asarray(evaluator.evaluate_jacobian(x_flat), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        logger.debug("feasibility: Jacobian unavailable, using the stricter bound: %s", exc)
        return None
    if J.ndim != 2 or J.shape[0] <= int(rows.max()):
        logger.debug("feasibility: Jacobian shape %s cannot cover rows; stricter bound", J.shape)
        return None
    xw = np.maximum(1.0, np.abs(np.asarray(x_flat, dtype=np.float64)))
    sub = np.abs(J[rows, :]) * xw[None, :]
    if not np.all(np.isfinite(sub)):
        logger.debug("feasibility: non-finite Jacobian entry; stricter bound")
        return None
    return np.asarray(sub.max(axis=1), dtype=np.float64)


def check_constraints(model, x_flat: np.ndarray, evaluator=None) -> VerifyResult:
    """Every constraint row, enumerated from the evaluator's own row map."""
    if evaluator is None:
        from discopt._jax.nlp_evaluator import cached_evaluator

        evaluator = cached_evaluator(model)

    if evaluator.n_constraints <= 0:
        return VerifyResult(True)

    g = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64)
    row_map = evaluator.constraint_row_map()
    n_rows = row_map[-1][1] if row_map else 0
    if g.shape[0] < n_rows:
        # The evaluator produced fewer rows than its own map claims. Refuse to
        # vouch rather than check a prefix.
        return VerifyResult(
            False, None, f"evaluator produced {g.shape[0]} rows, map wants {n_rows}"
        )

    # Pass 1 — residuals under the Jacobian-FREE (stricter) bound. Rows that
    # clear this also clear the scale-aware bound, which is never smaller, so the
    # Jacobian is computed only when some row is actually near or over the line.
    viol = np.zeros(n_rows, dtype=np.float64)
    anchor = np.ones(n_rows, dtype=np.float64)
    for start, stop, con in row_map:
        sense = _sense_str(con)
        if sense is None:
            return VerifyResult(False, None, f"unknown constraint sense {con.sense!r}")
        # Honour Constraint.rhs. Bodies built through the operator API are
        # normalised to rhs == 0, but the field is settable and the evaluator
        # compiles the BODY ONLY, so `body <sense> rhs` must be re-centred here.
        rhs = float(getattr(con, "rhs", 0.0) or 0.0)
        for i in range(start, stop):
            val = float(g[i]) - rhs
            if not math.isfinite(val):
                return VerifyResult(False, None, f"non-finite residual in row {i}")
            viol[i] = _row_violation(val, sense)
            anchor[i] = max(1.0, abs(rhs))

    suspect = np.nonzero(viol > ABS_TOL * anchor)[0]
    if suspect.size == 0:
        return VerifyResult(True)

    # Pass 2 — only the suspect rows get the full scale-keyed bound.
    scales = _row_scales(evaluator, x_flat, suspect)
    if scales is None:
        worst = int(suspect[int(np.argmax(viol[suspect]))])
        return VerifyResult(False, None, f"row {worst} violated by {viol[worst]:.3e}")
    allowed = ABS_TOL * np.maximum(anchor[suspect], scales)
    over = viol[suspect] > allowed
    if np.any(over):
        k = int(np.argmax(np.where(over, viol[suspect] - allowed, -np.inf)))
        w = int(suspect[k])
        return VerifyResult(
            False, None, f"row {w} violated by {viol[w]:.3e} (allowed {allowed[k]:.3e})"
        )
    return VerifyResult(True)


def verify_point(
    model,
    x_flat,
    *,
    with_objective: bool = False,
) -> VerifyResult:
    """Verify ``x_flat`` is feasible for ``model``; optionally return its objective.

    The contract is strict, because callers use this to decide whether a value may
    seed an incumbent cutoff and an unverified seed poisons every downstream
    certificate: this returns ``ok=True`` ONLY when the evaluator successfully
    evaluated every constraint row and every residual, bound and integrality
    condition is within tolerance. Any evaluator failure, shape mismatch or
    non-finite value yields ``ok=False`` — never an optimistic pass.
    """
    from discopt.modeling.core import ObjectiveSense

    x_flat = np.asarray(x_flat, dtype=np.float64)
    if x_flat.ndim != 1 or not np.all(np.isfinite(x_flat)):
        return VerifyResult(False, None, "point is not a finite 1-D vector")

    res = check_variable_bounds(model, x_flat)
    if not res.ok:
        return res

    try:
        from discopt._jax.nlp_evaluator import cached_evaluator

        evaluator = cached_evaluator(model)
        res = check_constraints(model, x_flat, evaluator=evaluator)
        if not res.ok:
            return res
        if not with_objective:
            return VerifyResult(True)
        obj_min = float(evaluator.evaluate_objective(x_flat))
    except Exception as exc:  # noqa: BLE001 - the evaluator could not vouch
        logger.debug("feasibility verification declined (evaluator error): %s", exc)
        return VerifyResult(False, None, f"evaluator error: {exc}")

    if not math.isfinite(obj_min):
        return VerifyResult(False, None, "non-finite objective")
    # ``evaluate_objective`` minimises the negation for a MAXIMIZE model; undo
    # that so the returned value is the objective in model units.
    model_obj = -obj_min if model._objective.sense == ObjectiveSense.MAXIMIZE else obj_min
    return VerifyResult(True, float(model_obj))
