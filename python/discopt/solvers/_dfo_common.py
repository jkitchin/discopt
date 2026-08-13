"""Machinery shared by the derivative-free backends (``direct``, ``surrogate``).

Both backends are black-box searches over a finite box: they see the model only
through "evaluate at a point, get an objective and a total constraint violation",
and they rank the points they have seen with the DIRECT-GLce auxiliary function
(Stripinis, Paulavičius & Žilinskas 2018). Those two pieces are the whole of
their shared surface, and they are the two pieces that *must* agree exactly:

* The oracle decides what a point is *worth*. If the two backends disagree there,
  a comparison between them measures the plumbing rather than the algorithms.
* The merit rule decides which point is *better*. A drift between two copies
  would have them rank the same pair of points differently while both cite the
  same source.

They lived as two copies until #1010, and the duplication had already cost
something: the guard rejecting a ``cl``/``cu`` length mismatch landed in
``direct`` first and had to be applied to ``surrogate`` separately. That guard
protects a *feasibility verdict*, so the copy without it was silently wrong
rather than loudly broken — which is the failure mode this module exists to
remove.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)

Oracle = Callable[[np.ndarray], tuple[float, float]]


def build_oracle(model: Model, *, log_prefix: str) -> tuple[Oracle, int, np.ndarray]:
    """``(evaluate, n_vars, integer_mask)`` for ``model``.

    ``evaluate`` maps a model point to ``(objective, total_violation)``. Both come
    from the one evaluator funnel the rest of the solver uses, so an opaque
    ``dm.custom`` body is evaluated exactly as the local NLP path would.

    ``log_prefix`` names the calling backend in the "evaluator backend is X" line
    and changes nothing else — the values a backend sees must not depend on which
    backend asked for them.
    """
    from discopt.solver import _extract_variable_info, _infer_constraint_bounds, _make_evaluator

    evaluator = _make_evaluator(model)
    logger.info("%s: evaluator backend is %s", log_prefix, type(evaluator).__name__)

    n_vars, _lb, _ub, int_offsets, int_sizes = _extract_variable_info(model)
    integer_mask = np.zeros(n_vars, dtype=bool)
    for off, size in zip(int_offsets, int_sizes):
        integer_mask[off : off + size] = True

    n_cons = int(getattr(evaluator, "n_constraints", 0) or 0)
    # One binding rather than two: cl and cu are set together and cleared
    # together, so a single Optional makes that inseparable. Two independent
    # Optionals are exactly the shape of the bug fixed in 067abc6, where a
    # ``cl``-keyed guard left ``cu`` at a different length and the violation sum
    # broadcast to zero.
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
    if n_cons:
        cl_raw, cu_raw = _infer_constraint_bounds(model, evaluator)
        cl = np.asarray(cl_raw, dtype=np.float64)
        cu = np.asarray(cu_raw, dtype=np.float64)
        # The violation sum below indexes cl and cu against the same g vector, so
        # a length mismatch would broadcast-or-truncate into a silently wrong
        # violation — i.e. a wrong feasibility verdict, not a crash. Refuse loudly.
        if cl.shape != (n_cons,) or cu.shape != (n_cons,):
            raise ValueError(
                f"constraint bounds do not match the evaluator: n_constraints={n_cons} "
                f"but cl has shape {cl.shape} and cu has shape {cu.shape}"
            )
        bounds = (cl, cu)

    def evaluate(x: np.ndarray) -> tuple[float, float]:
        fval = float(evaluator.evaluate_objective(x))
        if not np.isfinite(fval):
            # A black box may be undefined here. Treat it as an unusable point
            # rather than letting a NaN poison the ordering: +inf loses every
            # comparison, which is the honest ranking for "no value". A backend
            # that has to *fit* the merit rather than only compare it asks
            # ``glce_merit`` for a finite stand-in (see ``finite_fill``).
            fval = np.inf
        viol = 0.0
        if bounds is not None:
            lo, hi = bounds
            g = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
            g = np.where(np.isfinite(g), g, np.inf)
            viol = float(np.sum(np.maximum(0.0, g - hi)) + np.sum(np.maximum(0.0, lo - g)))
        return fval, viol

    return evaluate, n_vars, integer_mask


def glce_merit(
    f: np.ndarray,
    v: np.ndarray,
    best_feasible_value: Optional[float],
    eps_cons: float,
    *,
    finite_fill: bool,
) -> np.ndarray:
    """The DIRECT-GLce auxiliary value, one per evaluated point. Lower is better.

    * **Phase A** (``best_feasible_value is None``, nothing feasible seen yet):
      rank by total violation, so the search first hunts for feasibility.
    * **Phase B** (a feasible point exists): a feasible point ranks by its
      objective; an infeasible one is penalized by its violation *plus*
      ``|f - f_min|``. That last term is what stops an infeasible point from
      earning credit for an objective below the incumbent, and it needs no
      penalty weight to be tuned.
    * The ``ce`` refinement: a violation within ``eps_cons`` counts as feasible,
      so the ranking is not discontinuous exactly at the boundary where optima
      usually sit (DIRECT has no convergence guarantee across a discontinuity).

    ``finite_fill`` decides what happens to a non-finite merit — which arises
    when the black box was undefined at a point and the oracle returned ``+inf``.
    It is an explicit argument because the two backends genuinely want different
    answers, not because they drifted:

    * ``False`` (DIRECT): the merit is only ever *compared*, and ``+inf`` is the
      honest ranking for "no value" — it loses every comparison and the point is
      never selected.
    * ``True`` (surrogate): the merit is *fitted*, and an infinite right-hand
      side makes the interpolation system meaningless. The non-finite entries are
      mapped to the worst finite merit plus the observed spread, which keeps the
      one thing the point does tell us — that the region is bad — without
      discarding it or poisoning the fit. With no finite merit at all there is no
      spread to speak of, so every point is given the same value: an
      uninformative but well-posed system.
    """
    f = np.asarray(f, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if best_feasible_value is None:
        merit = v.copy()
    else:
        near_feasible = v <= eps_cons
        merit = np.where(near_feasible, f, f + v + np.abs(f - best_feasible_value))
    if not finite_fill:
        compared: np.ndarray = merit
        return compared
    bad = ~np.isfinite(merit)
    if bad.any():
        finite = merit[~bad]
        if finite.size == 0:
            merit = np.zeros_like(merit)
        else:
            spread = float(finite.max() - finite.min())
            merit = np.where(bad, finite.max() + spread + 1.0, merit)
    out: np.ndarray = merit
    return out


def glce_merit_scalar(
    fval: float,
    viol: float,
    best_feasible_value: Optional[float],
    eps_cons: float,
) -> float:
    """:func:`glce_merit` for a single point, for callers that rank one at a time.

    Delegates rather than restating the formula: a second scalar implementation
    of the rule is exactly the drift this module exists to prevent. ``finite_fill``
    is not offered because a single point has no other point to take a finite
    stand-in from.
    """
    merit = glce_merit(
        np.array([fval], dtype=np.float64),
        np.array([viol], dtype=np.float64),
        best_feasible_value,
        eps_cons,
        finite_fill=False,
    )
    return float(merit[0])
