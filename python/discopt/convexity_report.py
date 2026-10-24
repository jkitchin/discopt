"""Public convexity verdict: :meth:`discopt.modeling.core.Model.convexity`.

The solver decides whether to take its convex fast path from a classification
that runs *after* two exact rewrites (:func:`discopt.solver._convexity_rewrites`).
The bare classifier in ``discopt._relax.convexity`` sees the model as written, so
it could call a model nonconvex that the solve then treats as convex -- the #1616
A-01 report: ``minimize x*log(x)`` classified ``False`` and solved on the convex
fast path, because the solve rewrites ``x*log(x)`` to ``entropy(x)`` first.

:func:`convexity` runs the same rewrites and the same classifier walk the solver
dispatches on, and reports which parts were proven.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discopt.modeling.core import Model


@dataclass(frozen=True)
class ConvexityReport:
    """Result of :meth:`Model.convexity() <discopt.modeling.core.Model.convexity>`.

    Attributes:
        is_convex: ``True`` when the objective and every constraint were proven
            convex (for ``maximize``, a concave or affine objective). ``False``
            means *not proven*, not *proven nonconvex*: the classifier is sound
            but incomplete. ``None`` when classification ran out of its time
            budget before reaching a verdict.
        objective_convex: The objective's verdict (``None`` on budget exhaustion).
        constraints: ``(name, proven_convex)`` per constraint, in model order.
            Indicator, disjunctive, SOS and logical constraints are never
            reported convex. Empty on budget exhaustion.
        rewrites: The exact rewrites applied before classifying, in order
            (``"entropy"``, ``"squared_soc"``,
            ``"objective_epigraph"``). These are what let a model
            be proven convex here that the bare classifier would not prove.

    For a model with integer or binary variables the verdict is about the
    continuous relaxation, i.e. whether the problem is a *convex* MINLP.

    The verdict is the one the solve dispatches on before presolve. The solve
    tightens variable bounds before some later classifications, and tightening
    can only add proofs (a smaller box never makes a curvature certificate
    fail). So ``is_convex=True`` here means the solve also proves the model
    convex; ``False`` here can still become a proof after bound tightening.
    """

    is_convex: bool | None
    objective_convex: bool | None
    constraints: tuple[tuple[str | None, bool], ...]
    rewrites: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.is_convex)

    def nonconvex_constraints(self) -> list[str | None]:
        """Names of the constraints that were not proven convex."""
        return [name for name, ok in self.constraints if not ok]


def convexity(model: Model, *, time_limit: float | None = 15.0) -> ConvexityReport:
    """Classify *model* the way :meth:`Model.solve` does. See :class:`ConvexityReport`."""
    from discopt._relax.convexity.rules import ConvexityBudgetExceeded, classify_model_parts
    from discopt.solver import _convexity_rewrites

    rewritten, applied = _convexity_rewrites(model)
    if len(rewritten._constraints) != len(model._constraints):
        # Both rewrites replace constraints one for one; the names below are read
        # from the caller's model, so a count change would misattribute verdicts.
        raise RuntimeError(
            "convexity rewrites changed the constraint count "
            f"({len(model._constraints)} -> {len(rewritten._constraints)})"
        )
    deadline = None if time_limit is None else time.perf_counter() + float(time_limit)
    try:
        obj_convex, mask = classify_model_parts(rewritten, use_certificate=True, deadline=deadline)
    except ConvexityBudgetExceeded:
        return ConvexityReport(None, None, (), applied)
    names = tuple(getattr(c, "name", None) for c in model._constraints)
    return ConvexityReport(
        is_convex=bool(obj_convex and all(mask)),
        objective_convex=bool(obj_convex),
        constraints=tuple(zip(names, (bool(v) for v in mask))),
        rewrites=applied,
    )
