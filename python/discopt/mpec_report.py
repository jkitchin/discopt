r"""Source-level residuals for complementarity relations, and the result schema.

Slice 2 of the MCP/MPEC RFC (`#1148 <https://github.com/jkitchin/discopt/issues/1148>`_,
background `#1123 <https://github.com/jkitchin/discopt/issues/1123>`_). It builds
directly on the durable source provenance of
`#1147 <https://github.com/jkitchin/discopt/issues/1147>`_: a residual computed
here reads the relation's **declared operands**, which survive every rebuilding
pass, so it measures the *source* condition rather than whatever relaxed row the
condition was lowered into.

Why the two numbers must be reported separately
-----------------------------------------------

A Scholtes homotopy stopped at ``t = 1e-8`` produces a point where the generated
row ``f·g <= t`` is **satisfied** — its violation is exactly 0.0 — while the
source condition ``0 <= f ⊥ g >= 0`` is violated by ``min(f, g) ≈ 1e-4``. The
lowered-NLP residual and the source residual differ by four orders of magnitude
at the same point, and only the second one is a statement about the user's model.
Conflating them (or reporting only the first, which any generic NLP result does)
is the specific error the CCOpt discussion in #1123 warns about, and the reason
this module exists. POUNCE Gate 0 (`jkitchin/pounce#794
<https://github.com/jkitchin/pounce/issues/794>`_) documented that
``sqrt(t)`` scale for the ``min`` residual; :func:`admitted_residual_scale` makes
it visible in the report — per residual definition, since the scale differs by
formula — rather than letting the result imply exact orthogonality.

Every residual carries its **definition** as a string, per #1148's acceptance
criteria: a number whose formula is not recorded cannot be compared across
solvers, and the shared benchmark schema of `jkitchin/pounce#780
<https://github.com/jkitchin/pounce/issues/780>`_ is exactly this field set.

Evaluation, and why it is interval-based
----------------------------------------

Operands are evaluated with
:func:`~discopt._relax.convexity.interval_eval.evaluate_interval` over a
**degenerate** box (``[x, x]`` per variable, keyed by *object identity*, which is
what #1147 guarantees). Two properties buy a lot:

* no flat-index arithmetic is involved, so the relation's operands cannot be
  read off the wrong columns — the failure mode #1147's ``_target_flat_offsets``
  had to fix, and the one a second hand-rolled evaluator would reintroduce;
* the interval evaluator returns ``[-inf, +inf]`` for an atom it does not
  support. On a degenerate input box a *correct* evaluation stays degenerate, so
  a widened result is a positive signal that the walk did not actually evaluate
  the expression — :func:`evaluate_at_point` refuses instead of returning a
  midpoint that looks like a measurement (CLAUDE.md §6/§7).

It also keeps this path off JAX: ``_relax.dag_compiler`` imports ``jax.numpy`` at
module scope, and a residual report is not a reason to load it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from discopt.status import (
    CERTIFYING_STATUSES,
    LOCAL_STATUSES,
    is_certified_status,
    is_local_status,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Constraint, Expression, Model
    from discopt.mpec import Complementarity

__all__ = [
    "CERTIFYING_STATUSES",
    "LOCAL_STATUSES",
    "ComplementarityKind",
    "ContinuationStage",
    "ContinuationTrace",
    "RelationResidual",
    "Residual",
    "SourceResidualReport",
    "accept_local_incumbent",
    "admitted_residual_scale",
    "admitted_residual_scale_definition",
    "evaluate_at_point",
    "is_certified_status",
    "is_local_status",
    "max_source_complementarity",
    "point_from_flat",
    "relation_residuals",
    "source_residual_report",
]

#: Relative width above which a degenerate-box interval evaluation is refused.
#: Interval arithmetic on a point box stays degenerate up to directed-rounding
#: slop (measured: 7.1e-14 absolute on a value of 32.1, i.e. 2.2e-15 relative,
#: for ``exp(x·y) + log(x+1) - sqrt(y+2)``). Anything wider means some atom in
#: the graph was not actually evaluated.
_POINT_EVAL_REL_WIDTH = 1e-6

_INF = float("inf")

logger = logging.getLogger(__name__)


class ComplementarityKind:
    """The complementarity-residual definitions this module can compute.

    Not an :class:`enum.Enum`: these values are written into a report field that
    is serialized to JSON and compared across repositories (the shared schema of
    `jkitchin/pounce#780 <https://github.com/jkitchin/pounce/issues/780>`_), and
    a bare string round-trips without every consumer needing the enum class.
    """

    #: ``min(f, g)`` — the natural residual of the nonnegative pair. Zero exactly
    #: when the pair is complementary *given* nonnegativity, which is measured
    #: separately, so the two numbers stay independent.
    MIN = "min"
    #: ``f·g`` — the product form. Scale-sensitive (it is quadratic in the
    #: operand magnitudes), which is why it is not the default.
    PRODUCT = "product"
    #: Fischer-Burmeister ``f + g - sqrt(f² + g²)``: zero iff ``f, g >= 0`` and
    #: ``f·g = 0``, so it folds nonnegativity in rather than assuming it.
    FISCHER_BURMEISTER = "fischer_burmeister"
    #: The MCP normal map ``z - mid(l, u, z - F(z))``, which is the box form's
    #: natural residual and reduces to :data:`MIN` at ``l = 0, u = +inf``.
    NATURAL_MAP = "natural_map"

    #: Chosen from the relation's declared bounds: the symmetric nonnegative
    #: pair uses :data:`MIN`, a general box uses :data:`NATURAL_MAP`.
    AUTO = "auto"

    _FORMULA = {
        MIN: "r_i = min(f_i(x), g_i(x)); reported as max_i r_i",
        PRODUCT: "r_i = f_i(x) * g_i(x); reported as max_i r_i",
        FISCHER_BURMEISTER: (
            "r_i = |f_i(x) + g_i(x) - sqrt(f_i(x)^2 + g_i(x)^2)|; reported as max_i r_i"
        ),
        NATURAL_MAP: (
            "r_i = |z_i - mid(l_i, u_i, z_i - F_i(x))| with z = g, F = f; reported as max_i r_i"
        ),
    }

    @classmethod
    def formula(cls, kind: str) -> str:
        """The recorded definition string for ``kind``."""
        try:
            return cls._FORMULA[kind]
        except KeyError:
            raise ValueError(
                f"unknown complementarity residual kind {kind!r}; use one of "
                f"{sorted(cls._FORMULA)} or {cls.AUTO!r}"
            ) from None


# ───────────────────────────── the schema ─────────────────────────────


@dataclass(frozen=True)
class Residual:
    """One reported residual, with the definition that produced it.

    ``value`` is in the units the definition states. ``scale`` is the
    characteristic magnitude it was measured against — the relation's declared
    :attr:`~discopt.mpec.Complementarity.scale` (or the magnitude derived from
    its bounds) for a complementarity residual, ``1.0`` for a quantity that has
    no meaningful scale. A uniform tolerance is not meaningful when predicates
    carry unrelated physical magnitudes (a multiplier in 1e-6 against a flow in
    1e3), so :attr:`scaled_value` is the number a tolerance should be compared
    against, and both are reported.

    ``admitted_scale`` is the **worst-case residual the relaxation admits** at the
    parameter it was stopped at — the largest value of *this residual definition*
    consistent with the relaxed rows (see :func:`admitted_residual_scale`).
    ``None`` means no relaxation-derived scale applies.

    It is **not** a limit on attainable accuracy, and an earlier draft of this
    field said it was (#1158 review 3, nonblocking 1). ``f·g <= t`` admits
    ``min(f, g)`` as large as ``sqrt(t)``, but it also admits ``(f, g) = (0, 1)``,
    whose residual is exactly ``0`` at every positive ``t`` — a solver can and
    routinely does land far below the scale. What the number buys is the other
    direction: a residual of 1e-4 from a ``t = 1e-8`` continuation is *within what
    the relaxation allows*, so it is not by itself evidence of a convergence
    failure. Whether the subsolver could have done better is an empirical question
    about the subsolver, which this number says nothing about; the continuation
    trace's per-stage statuses are where that evidence lives.

    ``admitted_scale_definition`` records the formula, because the scale depends on
    which residual definition was selected — ``sqrt(t)`` for ``min``, ``t`` for the
    product form, ``(2-sqrt(2))*sqrt(t)`` for Fischer-Burmeister — and a number
    whose formula is not recorded cannot be compared across solvers. Copying one
    definition's scale into another's report was the second half of that finding.
    """

    name: str
    value: float
    definition: str
    scale: float = 1.0
    admitted_scale: Optional[float] = None
    admitted_scale_definition: Optional[str] = None
    where: Optional[str] = None

    @property
    def scaled_value(self) -> float:
        """``value / scale`` — the number a tolerance should be compared against."""
        s = float(self.scale)
        return float(self.value) / (s if s > 0.0 else 1.0)

    @property
    def within_admitted_scale(self) -> bool:
        """True when the residual is no larger than the scale the relaxation admits.

        A comparison against :attr:`admitted_scale` and nothing else. **It does
        not establish feasibility of the relaxed rows**, and only one direction is
        implied: for a nonnegative pair, a ``min`` residual *above* ``sqrt(t)``
        forces both operands above ``sqrt(t)`` and hence ``f*g > t``, so ``False``
        does mean the relaxed row is violated. The converse fails — at
        ``t = 1e-8``, ``(f, g) = (1e-4, 1)`` gives ``min(f, g) = 1e-4 = sqrt(t)``
        and this returns ``True`` while ``f*g = 1e-4`` exceeds ``t`` by four orders
        of magnitude (#1158 review 4). Whether the generated rows actually hold is
        :attr:`SourceResidualReport.lowered_row_residual`; how close the solver got
        to the true condition is not this number either.
        """
        return self.admitted_scale is not None and float(self.value) <= float(self.admitted_scale)

    def as_dict(self) -> dict:
        """JSON-ready mapping, definition included (never dropped)."""
        return {
            "name": self.name,
            "value": float(self.value),
            "definition": self.definition,
            "scale": float(self.scale),
            "scaled_value": self.scaled_value,
            "admitted_scale": (None if self.admitted_scale is None else float(self.admitted_scale)),
            "admitted_scale_definition": self.admitted_scale_definition,
            "within_admitted_scale": self.within_admitted_scale,
            "where": self.where,
        }


@dataclass(frozen=True)
class RelationResidual:
    """The source residuals of one **scalar** complementarity relation.

    ``relation`` is the scalar element (see
    :meth:`~discopt.mpec.Complementarity.elements`); ``source_name`` names the
    declared relation it came from, so an elementwise number stays attributable
    to the vector relation the user wrote (#1147).
    """

    relation: "Complementarity"
    name: str
    source_name: Optional[str]
    role: str
    index: Optional[tuple[int, ...]]
    parent: Optional[str]
    f_value: float
    g_value: float
    complementarity: Residual
    f_bound: Residual
    g_bound: Residual

    @property
    def max_bound_violation(self) -> float:
        """The larger of the two operands' declared-bound violations."""
        return max(float(self.f_bound.value), float(self.g_bound.value))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "role": self.role,
            "index": None if self.index is None else list(self.index),
            "parent": self.parent,
            "f_value": float(self.f_value),
            "g_value": float(self.g_value),
            "complementarity": self.complementarity.as_dict(),
            "f_bound": self.f_bound.as_dict(),
            "g_bound": self.g_bound.as_dict(),
        }


@dataclass(frozen=True)
class ContinuationStage:
    """One stage of a regularization homotopy.

    ``accepted`` records whether the stage's point was taken forward;
    ``reason`` says why, in words, for the stage that ended the continuation.
    Kept rather than discarded (#1148 §D): the sequence of ``t`` values and the
    residual achieved at each is the only evidence of *how* the final point was
    reached, and a result that reports the final ``t`` alone cannot distinguish
    "converged at 1e-8" from "stalled at 1e-2 and stopped".
    """

    iteration: int
    t: float
    status: str
    accepted: bool
    reason: str
    objective: Optional[float] = None
    source_complementarity: Optional[float] = None
    #: Whether the subsolver *converged* at this stage, as opposed to merely
    #: returning a point. ``accepted`` says the point was taken forward; a
    #: stalled stage is accepted and not certified, and collapsing the two is
    #: what let a zero-iteration subsolver report ``local_optimal`` at its own
    #: starting point (#1158 review 3, blocking 3).
    certified: bool = False

    def as_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "t": float(self.t),
            "status": self.status,
            "accepted": self.accepted,
            "certified": self.certified,
            "reason": self.reason,
            "objective": None if self.objective is None else float(self.objective),
            "source_complementarity": (
                None if self.source_complementarity is None else float(self.source_complementarity)
            ),
        }


@dataclass(frozen=True)
class ContinuationTrace:
    """The continuation parameter, its schedule, and why the homotopy stopped."""

    parameter: str
    stages: tuple[ContinuationStage, ...]
    final_t: Optional[float]
    termination_reason: str
    #: The worst-case ``min(f, g)`` residual the regularization at :attr:`final_t`
    #: admits (see :func:`admitted_residual_scale`). Reported for the ``min``
    #: definition because that is the trace's own summary number; the per-relation
    #: residuals in the report each carry the scale of *their* definition, which is
    #: not the same number.
    admitted_residual_scale: Optional[float]
    #: Whether any stage's subsolver actually converged. Reaching the end of the
    #: schedule is not the same thing: the loop can walk t down to ``t_min`` with
    #: every stage failing, and reporting that as converged would call a run that
    #: produced no point a success (#1158 review, LOW 9).
    any_stage_accepted: bool = True
    #: Whether the stage that produced the **reported** point converged. This is
    #: what :attr:`converged` turns on: accepting an iterate and converging are
    #: different events, and a trace that reports the first as the second lets a
    #: run that did no optimization at all read as a solved one (#1158 review 3,
    #: blocking 3). Defaults to ``False`` — absent evidence of convergence is
    #: read as not converged, the same closed-set rule as
    #: :func:`~discopt.status.is_certified_status`.
    reported_point_certified: bool = False
    admitted_residual_scale_definition: str = (
        "sqrt(t_final): the largest min(f, g) residual a Scholtes regularization "
        "f*g <= t admits, per POUNCE Gate 0 (jkitchin/pounce#794). An upper bound "
        "on the admitted residual, NOT a lower limit on attainable accuracy: "
        "(f, g) = (0, 1) satisfies the relaxed row with residual 0 at every t."
    )

    @property
    def any_stage_certified(self) -> bool:
        """True when some stage's subsolver converged (not merely returned a point)."""
        return any(stage.certified for stage in self.stages)

    @property
    def converged(self) -> bool:
        """True when the schedule reached its target **and** the reported point converged.

        Not ``any_stage_accepted``: a stage is accepted when the subsolver handed
        back a point it stopped at, which a subsolver allowed zero iterations
        also does — at the starting point. Convergence is a property of the
        stage whose point is being reported, so that is what this reads
        (#1158 review 3, blocking 3).
        """
        return self.termination_reason == "t_min_reached" and self.reported_point_certified

    def as_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "stages": [s.as_dict() for s in self.stages],
            "final_t": None if self.final_t is None else float(self.final_t),
            "termination_reason": self.termination_reason,
            "converged": self.converged,
            "any_stage_accepted": self.any_stage_accepted,
            "any_stage_certified": self.any_stage_certified,
            "reported_point_certified": self.reported_point_certified,
            "admitted_residual_scale": (
                None
                if self.admitted_residual_scale is None
                else float(self.admitted_residual_scale)
            ),
            "admitted_residual_scale_definition": self.admitted_residual_scale_definition,
        }


@dataclass(frozen=True)
class SourceResidualReport:
    """Every residual of a point, measured against the **source** model.

    Aggregates are infinity norms over the scalar relations. ``complementarity``
    and ``bound_violation`` are properties of the *declared* relations;
    ``lowered_row_residual`` is the maximum violation of the rows a lowering
    generated, and is reported alongside precisely so the two can be compared
    (see the module docstring for why they differ, and by how much).

    ``n_scalar_relations`` is the executed-measurement count. It is not
    decoration: a report over zero relations would otherwise print residuals of
    ``0.0`` and read as a clean pass — the probe-that-measured-nothing failure
    CLAUDE.md §6 exists to prevent — so :func:`source_residual_report` refuses to
    build one when it was handed relations and measured none.
    """

    relations: tuple[RelationResidual, ...]
    complementarity: Residual
    bound_violation: Residual
    primal_feasibility: Residual
    n_scalar_relations: int
    kind: str
    integrality: Optional[Residual] = None
    lowered_row_residual: Optional[Residual] = None
    continuation: Optional[ContinuationTrace] = None
    stationarity: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def residuals(self) -> tuple[Residual, ...]:
        """Every top-level residual in the report, each carrying its definition."""
        out = [self.complementarity, self.bound_violation, self.primal_feasibility]
        if self.integrality is not None:
            out.append(self.integrality)
        if self.lowered_row_residual is not None:
            out.append(self.lowered_row_residual)
        return tuple(out)

    @property
    def definitions(self) -> dict[str, str]:
        """``{residual name: formula}`` for every residual reported."""
        return {r.name: r.definition for r in self.residuals}

    @property
    def source_satisfied(self) -> bool:
        """Whether **every** source residual is within the scaled tolerance.

        Quantified over **every relation**, not over the aggregate alone. The
        aggregate is one relation's value against that relation's own scale, and
        even ranked by scaled value it is a single row; testing each row directly
        is what makes this a statement about the whole model rather than about
        whichever relation happened to rank worst (#1158 review 2, HIGH 1).

        Deliberately not a certificate and never used as one: it is the
        source-side reading a caller compares against its own tolerance.
        """
        if any(r.complementarity.scaled_value > _SOURCE_TOL for r in self.relations):
            return False
        if any(r.max_bound_violation > _SOURCE_TOL for r in self.relations):
            return False
        return (
            self.complementarity.scaled_value <= _SOURCE_TOL
            and self.bound_violation.value <= _SOURCE_TOL
            and self.primal_feasibility.value <= _SOURCE_TOL
            and (self.integrality is None or self.integrality.value <= _INTEGRALITY_TOL)
        )

    def as_dict(self) -> dict:
        """JSON-ready mapping — the shared benchmark schema of #1148."""
        return {
            "kind": self.kind,
            "n_scalar_relations": self.n_scalar_relations,
            "complementarity": self.complementarity.as_dict(),
            "bound_violation": self.bound_violation.as_dict(),
            "primal_feasibility": self.primal_feasibility.as_dict(),
            "integrality": None if self.integrality is None else self.integrality.as_dict(),
            "lowered_row_residual": (
                None if self.lowered_row_residual is None else self.lowered_row_residual.as_dict()
            ),
            "continuation": None if self.continuation is None else self.continuation.as_dict(),
            "stationarity": self.stationarity,
            "source_satisfied": self.source_satisfied,
            "relations": [r.as_dict() for r in self.relations],
            "definitions": self.definitions,
            "notes": list(self.notes),
        }


#: Source-side tolerances. Deliberately the repo's published solver tolerances
#: (``CLAUDE.md`` "Key Constraints"), so a source reading is judged by the same
#: numbers as everything else rather than by a private, looser copy.
_SOURCE_TOL = 1e-6
_INTEGRALITY_TOL = 1e-5


# ───────────────────────────── evaluation ─────────────────────────────


def point_from_flat(model: "Model", x_flat) -> dict[int, np.ndarray]:
    """``{id(variable): values}`` for ``model``'s variables at the flat point.

    The layout is ``model._variables`` order — the model's **true** layout —
    rather than ``Model._flat_var_offset``, which indexes by ``Variable._index``
    (the variable's position in the model it was *declared* on). The two differ
    exactly when a rebuilding pass hands a target model shared variable objects
    in a new order, and that is the case a source residual must survive
    (#1147, review of #1149 blocking 1).
    """
    x = np.asarray(x_flat, dtype=np.float64).ravel()
    out: dict[int, np.ndarray] = {}
    off = 0
    for v in model._variables:
        size = int(v.size)
        if id(v) not in out:
            if off + size > x.shape[0]:
                raise ValueError(
                    f"point_from_flat: flat point has {x.shape[0]} entries but "
                    f"{model.name!r} needs at least {off + size} by variable {v.name!r}"
                )
            out[id(v)] = x[off : off + size]
        off += size
    return out


def evaluate_at_point(
    model: "Model", expr: "Expression", point: dict[int, np.ndarray]
) -> np.ndarray:
    """Evaluate ``expr`` at ``point``, refusing anything but an exact evaluation.

    ``point`` maps ``id(variable)`` to that variable's values (see
    :func:`point_from_flat`); lookup is by **object identity**, which is what
    makes this usable on a relation's source operands after any number of
    rebuilding passes (#1147).

    Raises
    ------
    ValueError
        When the interval evaluation does not stay degenerate — i.e. when some
        atom of the expression graph was not actually evaluated (the interval
        evaluator widens to ``[-inf, +inf]`` for an atom it does not support).
        Returning the midpoint of a widened enclosure would produce a number
        that looks like a measurement and is not one; the failure is loud
        instead (CLAUDE.md §6/§7).
    """
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval

    box = {}
    for v in model._variables:
        vals = point.get(id(v))
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        shaped = arr.reshape(v.shape) if v.shape else arr.reshape(())
        box[v] = Interval(shaped.copy(), shaped.copy())

    enclosure = evaluate_interval(expr, model, box)
    lo = np.asarray(enclosure.lo, dtype=np.float64)
    hi = np.asarray(enclosure.hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(
            f"point evaluation of {expr!r} returned a ragged enclosure {lo.shape} vs {hi.shape}"
        )
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError(
            f"point evaluation of {expr!r} is not finite ({lo} .. {hi}). The "
            "interval evaluator widens to [-inf, +inf] for an atom it does not "
            "support, so this is a graph it did not actually evaluate — not a "
            "measurement. Add the atom to the interval evaluator rather than "
            "reading the midpoint."
        )
    mid = 0.5 * (lo + hi)
    width = hi - lo
    slack = _POINT_EVAL_REL_WIDTH * np.maximum(1.0, np.abs(mid))
    if np.any(width > slack):
        worst = float(np.max(width - slack))
        raise ValueError(
            f"point evaluation of {expr!r} did not stay degenerate on a point box "
            f"(worst excess width {worst:.3e}). Some atom of the graph was not "
            "evaluated exactly; refusing to report its midpoint as a residual."
        )
    return np.atleast_1d(mid)


def _scalar(model: "Model", expr: "Expression", point: dict[int, np.ndarray]) -> float:
    vals = evaluate_at_point(model, expr, point)
    if vals.size != 1:
        raise ValueError(
            f"expected a scalar operand, got {vals.size} elements from {expr!r}. "
            "Complementarity relations are scalarized before residuals are computed."
        )
    return float(vals.reshape(-1)[0])


# ─────────────────────────── residual formulas ───────────────────────────


#: ``2 - sqrt(2)``: the Fischer-Burmeister value at ``f = g``, which is where the
#: function is largest on ``{f, g >= 0, f*g <= t}``.
_FB_ADMITTED_COEFF = 2.0 - math.sqrt(2.0)

_ADMITTED_SCALE_DEFINITION = {
    ComplementarityKind.MIN: (
        "sqrt(t): the largest min(f, g) admitted by the Scholtes row f*g <= t "
        "with f, g >= 0, attained at f = g = sqrt(t)"
    ),
    ComplementarityKind.NATURAL_MAP: (
        "sqrt(t): on the nonnegative pair (l=0, u=+inf) the normal map reduces to "
        "min(f, g), so the Scholtes row f*g <= t admits the same worst case"
    ),
    ComplementarityKind.PRODUCT: (
        "t: the product residual IS the quantity the Scholtes row bounds, so the "
        "row admits exactly f*g = t"
    ),
    ComplementarityKind.FISCHER_BURMEISTER: (
        "(2 - sqrt(2))*sqrt(t): phi(f, g) = f + g - hypot(f, g) is largest at "
        "f = g on {f, g >= 0, f*g <= t}, where f = g = sqrt(t)"
    ),
}


def admitted_residual_scale(
    t: Optional[float], kind: str = ComplementarityKind.MIN
) -> Optional[float]:
    r"""The worst-case residual a Scholtes regularization at ``t`` **admits**.

    The regularized feasible set replaces ``0 <= f ⊥ g >= 0`` with ``f, g >= 0``
    and ``f·g <= t``. That relaxation admits points whose residual is nonzero, and
    this is how large the residual of ``kind`` can get on it:

    ============================ ====================
    ``kind``                     admitted scale
    ============================ ====================
    ``min`` / ``natural_map``    ``sqrt(t)``
    ``product``                  ``t``
    ``fischer_burmeister``       ``(2-sqrt(2))*sqrt(t)``
    ============================ ====================

    Each is a **maximum over the relaxed set**, not a lower limit on what a
    solver can reach: ``(f, g) = (0, 1)`` satisfies ``f·g <= t`` for every ``t``
    with a residual of exactly ``0``. Calling ``sqrt(t)`` an accuracy *floor* —
    as this function's earlier name and docstring did — asserts the opposite and
    is wrong (#1158 review 3, nonblocking 1); POUNCE Gate 0
    (`jkitchin/pounce#794 <https://github.com/jkitchin/pounce/issues/794>`_)
    documents it as the scale the regularization admits.

    The number is reported so a residual can be read against what the algorithm
    that produced it was actually asking for: at ``t = 1e-8``, a ``min`` residual
    of ``1e-4`` sits at the edge of the admitted set rather than outside it, which
    is a different statement from "the subsolver failed to converge". It is
    **per definition**, which is why ``kind`` is required rather than assumed:
    copying the ``min`` number into a product report compares a residual against
    the scale of a different formula (``1e-4`` against ``1e-8``) and reads as a
    gross violation when it is the same point.

    Returns ``None`` for ``t`` that is missing or non-positive, and for
    ``kind = "auto"``, which names no formula to derive a scale from.
    """
    if t is None:
        return None
    tv = float(t)
    if not math.isfinite(tv) or tv <= 0.0:
        return None
    if kind == ComplementarityKind.PRODUCT:
        return tv
    if kind == ComplementarityKind.FISCHER_BURMEISTER:
        return _FB_ADMITTED_COEFF * math.sqrt(tv)
    if kind in (ComplementarityKind.MIN, ComplementarityKind.NATURAL_MAP):
        return math.sqrt(tv)
    if kind == ComplementarityKind.AUTO:
        # Mixed or unresolved definitions: there is no single formula, so there is
        # no scale. Reporting one of them would be the copied-number bug again.
        return None
    raise ValueError(f"unknown complementarity residual kind {kind!r}")


def admitted_residual_scale_definition(kind: str) -> Optional[str]:
    """The recorded formula behind :func:`admitted_residual_scale` for ``kind``."""
    if kind == ComplementarityKind.AUTO:
        return None
    try:
        return _ADMITTED_SCALE_DEFINITION[kind]
    except KeyError:
        raise ValueError(f"unknown complementarity residual kind {kind!r}") from None


def _bound_violation(value: float, bounds: tuple[float, float]) -> float:
    """How far ``value`` lies outside ``bounds`` (0.0 when inside)."""
    lo, hi = float(bounds[0]), float(bounds[1])
    return float(max(0.0, lo - value, value - hi))


def _mid(lo: float, x: float, hi: float) -> float:
    """The median of ``lo``, ``x``, ``hi`` — the projection of ``x`` onto ``[lo, hi]``."""
    return float(min(max(x, lo), hi))


def _complementarity_residual(kind: str, f: float, g: float, pair: "Complementarity") -> float:
    if kind == ComplementarityKind.MIN:
        return float(min(f, g))
    if kind == ComplementarityKind.PRODUCT:
        return float(f * g)
    if kind == ComplementarityKind.FISCHER_BURMEISTER:
        return float(abs(f + g - math.hypot(f, g)))
    if kind == ComplementarityKind.NATURAL_MAP:
        lo, hi = pair.g_bounds
        return float(abs(g - _mid(lo, g - f, hi)))
    raise ValueError(f"unknown complementarity residual kind {kind!r}")


def _resolve_kind(kind: str, pair: "Complementarity") -> str:
    """Pick the residual definition for one relation.

    ``auto`` reads the relation's **declared bounds**, never its ``role``: role
    is provenance, and a relation built directly with ``g_bounds=(-1, 1)`` and
    the default ``NCP_PAIR`` role reaches here (the HIGH-2 lesson from the review
    of #1149). ``min(f, g)`` is meaningless on such a relation; the normal map
    is its natural residual.
    """
    if kind != ComplementarityKind.AUTO:
        return kind
    return (
        ComplementarityKind.MIN
        if pair.is_symmetric_nonnegative
        else ComplementarityKind.NATURAL_MAP
    )


# ───────────────────────────── the report ─────────────────────────────


def relation_residuals(
    model: "Model",
    pairs: Sequence["Complementarity"],
    point: dict[int, np.ndarray],
    *,
    kind: str = ComplementarityKind.AUTO,
    continuation_t: Optional[float] = None,
) -> tuple[RelationResidual, ...]:
    """Source residuals of every **scalar** relation ``pairs`` stands for.

    Vector relations are expanded through
    :meth:`~discopt.mpec.Complementarity.elements`, so a per-index residual stays
    attributable to the declared relation (``source_name``/``index``).

    Raises
    ------
    ValueError
        When ``pairs`` is non-empty and nothing was measured. A residual list
        that is empty because the expansion produced nothing would let every
        aggregate above it read ``0.0`` and pass — the probe-that-measured-
        nothing failure of CLAUDE.md §6, caught at the one place it can start.
    """
    rows: list[RelationResidual] = []
    for parent in pairs:
        for elem in parent.elements(model):
            elem_kind = _resolve_kind(kind, elem)
            fv = _scalar(model, elem.f, point)
            gv = _scalar(model, elem.g, point)
            scale = float(elem.effective_scale)
            rows.append(
                RelationResidual(
                    relation=elem,
                    name=elem.name or "<unnamed>",
                    source_name=(elem.source or elem).name,
                    role=elem.role.value,
                    index=elem.index,
                    parent=elem.parent,
                    f_value=fv,
                    g_value=gv,
                    complementarity=Residual(
                        name="source_complementarity",
                        value=_complementarity_residual(elem_kind, fv, gv, elem),
                        definition=ComplementarityKind.formula(elem_kind),
                        scale=scale,
                        # The admitted scale is derived from THIS row's resolved
                        # definition, never copied from another's: sqrt(t) against a
                        # product residual compares 1e-4 to 1e-8 and reads as a gross
                        # violation of the same point (#1158 review 3, nonblocking 1).
                        admitted_scale=admitted_residual_scale(continuation_t, elem_kind),
                        admitted_scale_definition=(
                            None
                            if continuation_t is None
                            else admitted_residual_scale_definition(elem_kind)
                        ),
                        where=elem.describe(),
                    ),
                    f_bound=Residual(
                        name="source_f_bound",
                        value=_bound_violation(fv, elem.f_bounds),
                        definition=(
                            "max(0, lb_f - f(x), f(x) - ub_f) against the bounds the "
                            "RELATION declares on f (0 <= f for a nonnegative pair)"
                        ),
                        scale=scale,
                        where=elem.describe(),
                    ),
                    g_bound=Residual(
                        name="source_g_bound",
                        value=_bound_violation(gv, elem.g_bounds),
                        definition=(
                            "max(0, lb_g - g(x), g(x) - ub_g) against the bounds the "
                            "RELATION declares on g (the box [l, u] for an MCP)"
                        ),
                        scale=scale,
                        where=elem.describe(),
                    ),
                )
            )
    if pairs and not rows:
        raise ValueError(
            f"relation_residuals: {len(list(pairs))} relation(s) were handed in and none "
            "was measured. Residuals over zero relations read as a clean pass; refusing "
            "instead (CLAUDE.md §6)."
        )
    return tuple(rows)


def max_source_complementarity(
    model: "Model",
    pairs: Sequence["Complementarity"],
    point: dict[int, np.ndarray],
    *,
    kind: str = ComplementarityKind.AUTO,
) -> Optional[float]:
    """The infinity-norm source complementarity residual, or ``None`` with no relations.

    The cheap per-stage reading a continuation records at each homotopy step —
    the full :func:`source_residual_report` also walks every constraint row and
    is not worth paying per stage.
    """
    rows = relation_residuals(model, pairs, point, kind=kind)
    if not rows:
        return None
    return max(float(r.complementarity.value) for r in rows)


def source_residual_report(
    model: "Model",
    pairs: Optional[Sequence["Complementarity"]] = None,
    *,
    x_flat=None,
    point: Optional[dict[int, np.ndarray]] = None,
    kind: str = ComplementarityKind.AUTO,
    source_constraints: Optional[Sequence["Constraint"]] = None,
    source_bounds: Optional[dict[int, tuple[np.ndarray, np.ndarray]]] = None,
    continuation: Optional[ContinuationTrace] = None,
    lowered_rows: Optional[Sequence["Constraint"]] = None,
    notes: Sequence[str] = (),
) -> SourceResidualReport:
    """Measure a point against the **source** complementarity relations of ``model``.

    Parameters
    ----------
    model
        The model carrying the relations. Its variables define the flat layout
        of ``x_flat``.
    pairs
        Relations to measure. Defaults to ``model._complementarities`` — the
        durable record #1147 keeps across every rebuilding pass.
    x_flat, point
        The point, as a flat vector in ``model._variables`` order, or already
        resolved by :func:`point_from_flat`. Exactly one is required.
    kind
        Which complementarity definition to use; see :class:`ComplementarityKind`.
        ``"auto"`` picks per relation from its declared bounds.
    source_constraints
        The rows that define **source** primal feasibility. Pass the model's
        constraint list *as it was before any lowering added rows* — otherwise
        the "source" feasibility number includes the regularized row ``f·g <= t``
        and stops being a statement about the user's model. Defaults to the
        model's current rows, with that caveat recorded in ``notes``.
    source_bounds
        ``{id(variable): (lb, ub)}`` snapshot of the **original** declared
        bounds, for the same reason: a presolve or a complementarity
        bound-tightening moves them. Defaults to the live bounds.
    continuation
        The homotopy trace, when one ran.
    lowered_rows
        The rows a lowering generated from these relations, so the lowered-NLP
        residual can be reported beside the source one. ``None`` when the
        generated rows are not tracked for this model.

    Raises
    ------
    ValueError
        When ``pairs`` is non-empty and no scalar relation was measured. A report
        over zero relations prints residuals of ``0.0`` and reads as a clean pass;
        refusing is the executed-assertion discipline of CLAUDE.md §6 applied to
        the instrument itself.
    """
    from discopt.modeling.core import VarType

    if (x_flat is None) == (point is None):
        raise ValueError("source_residual_report: pass exactly one of x_flat or point")
    if point is None:
        point = point_from_flat(model, x_flat)

    relations = list(model._complementarities if pairs is None else pairs)
    note_list = list(notes)

    # Pass the continuation PARAMETER, not a scale derived from it: each row
    # resolves its own residual definition and the admitted scale follows from
    # that definition, so the derivation has to happen where the kind is known.
    continuation_t = None if continuation is None else continuation.final_t
    rows = relation_residuals(model, relations, point, kind=kind, continuation_t=continuation_t)

    # ── aggregates ──
    resolved_kind = kind
    if kind == ComplementarityKind.AUTO:
        kinds = {_resolve_kind(kind, r.relation) for r in rows}
        resolved_kind = kinds.pop() if len(kinds) == 1 else ComplementarityKind.AUTO

    # Worst by SCALED value, not raw. ``effective_scale`` is genuinely per-relation
    # (a declared ``scale=``, else the largest finite declared-bound magnitude), so
    # picking the worst raw residual and then reporting *that relation's* scale
    # mixes one relation's numerator with its own denominator and says nothing
    # about the others. Measured: a box MCP on ``z in [0, 1e3]`` with residual 1e-3
    # (scaled 1e-6, fine) outranks an NCP pair with residual 1e-4 (scaled 1e-4,
    # 100x over tolerance), and the aggregate reported 1e-6 -> ``source_satisfied``
    # True with a badly violated relation in the report (#1158 review 2, HIGH 1).
    # The scaled value is the number a tolerance is compared against -- this
    # module's own docstring says so -- so it is the number the max ranks by.
    worst_c = max(rows, key=lambda r: r.complementarity.scaled_value, default=None)
    worst_b = max(rows, key=lambda r: r.max_bound_violation, default=None)
    complementarity = Residual(
        name="source_complementarity",
        value=0.0 if worst_c is None else float(worst_c.complementarity.value),
        definition=(
            "max over scalar relations of "
            + (
                ComplementarityKind.formula(resolved_kind)
                if resolved_kind != ComplementarityKind.AUTO
                else "each relation's own definition (see relations[].complementarity)"
            )
            + "; computed on the SOURCE operands (#1147 provenance), not on the lowered rows"
        ),
        scale=1.0 if worst_c is None else float(worst_c.complementarity.scale),
        # The aggregate reports the worst row's value, so it reports that row's
        # admitted scale too — the two must come from the same definition or the
        # comparison between them is meaningless.
        admitted_scale=(None if worst_c is None else worst_c.complementarity.admitted_scale),
        admitted_scale_definition=(
            None if worst_c is None else worst_c.complementarity.admitted_scale_definition
        ),
        where=None if worst_c is None else worst_c.name,
    )
    bound_violation = Residual(
        name="source_operand_bounds",
        value=0.0 if worst_b is None else float(worst_b.max_bound_violation),
        definition=(
            "max over scalar relations and both operands of the violation of the "
            "bounds the RELATION declares (nonnegativity for an NCP pair)"
        ),
        where=None if worst_b is None else worst_b.name,
    )

    # ── source primal feasibility ──
    if source_constraints is None:
        source_constraints = _model_rows(model)
        note_list.append(
            "primal feasibility was measured against the model's CURRENT rows: no "
            "pre-lowering snapshot was supplied, so generated rows are included"
        )
    primal = _primal_feasibility(model, source_constraints, source_bounds, point)

    # ── Boolean/selector integrality ──
    integrality = _integrality_residual(model, point, VarType)

    # ── the lowered rows, for comparison ──
    lowered = None
    if lowered_rows is not None:
        lowered = _row_residual(
            model,
            lowered_rows,
            point,
            name="lowered_row_residual",
            definition=(
                "max violation of the rows the lowering GENERATED from these relations "
                "(e.g. f*g <= t for Scholtes). A property of the generated NLP, not of "
                "the source condition — reported so the two can be compared"
            ),
        )

    return SourceResidualReport(
        relations=tuple(rows),
        complementarity=complementarity,
        bound_violation=bound_violation,
        primal_feasibility=primal,
        n_scalar_relations=len(rows),
        kind=resolved_kind,
        integrality=integrality,
        lowered_row_residual=lowered,
        continuation=continuation,
        stationarity=None,  # never claimed: discopt does not check C-/M-/S- conditions
        notes=tuple(note_list),
    )


def _model_rows(model: "Model") -> list["Constraint"]:
    """The model's constraint rows, including the builder-resident linear ones.

    Reading only ``_constraints`` silently skips every row added through
    ``add_linear_constraints`` / the ``constraint(fast=True)`` path — the second
    hole the #908 verifier work found.
    """
    rows = list(model._constraints)
    try:
        rows.extend(model._builder_linear_constraints())
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        raise ValueError(
            f"could not materialize {model.name!r}'s builder-resident linear rows for a "
            f"source-feasibility measurement: {type(exc).__name__}: {exc}"
        ) from exc
    return rows


def _row_residual(
    model: "Model",
    rows: Sequence["Constraint"],
    point: dict[int, np.ndarray],
    *,
    name: str,
    definition: str,
) -> Residual:
    """Max violation over ``rows``, evaluated elementwise on each row's body.

    No row-index arithmetic: each :class:`~discopt.modeling.core.Constraint` body
    is evaluated as an array and reduced elementwise, so the
    one-index-per-constraint-versus-one-row-per-element desynchronisation that
    made both pre-#908 verifiers wrongly accept infeasible points cannot occur
    here by construction.
    """
    worst = 0.0
    where = None
    for con in rows:
        v = _one_row_violation(model, con, point)
        if v > worst:
            worst, where = v, getattr(con, "name", None)
    return Residual(name=name, value=worst, definition=definition, where=where)


def _one_row_violation(model: "Model", con, point: dict[int, np.ndarray]) -> float:
    """Violation of one row of a model's constraint list, whatever kind it is.

    ``Model._constraints`` holds three kinds of object, and a residual walker
    that knows only the first silently reports 0.0 for the other two — which is
    exactly the rows a complementarity lowering emits (``either_or`` appends a
    ``_DisjunctiveConstraint``, ``sos1`` a ``_SOSConstraint``). An unknown kind
    **raises** rather than contributing nothing, for the same reason the #1147
    operand walker does (CLAUDE.md §6).

    * ``Constraint`` — the elementwise violation of ``body <sense> rhs``.
    * ``_DisjunctiveConstraint`` — ``min`` over disjuncts of the ``max``
      violation inside that disjunct: zero exactly when *some* disjunct holds,
      which is what the disjunction asserts.
    * ``_SOSConstraint`` (type 1) — the second-largest member magnitude: zero
      exactly when at most one member is nonzero. SOS2 is not encoded by any
      complementarity lowering, and guessing its residual would be a silent
      approximation, so it is refused.
    """
    from discopt.modeling.core import Constraint, _DisjunctiveConstraint, _SOSConstraint

    if isinstance(con, Constraint):
        vals = evaluate_at_point(model, con.body, point)
        sense = getattr(con.sense, "value", con.sense)
        rhs = float(getattr(con, "rhs", 0.0) or 0.0)
        shifted = vals - rhs
        if sense == "<=":
            viol = np.maximum(shifted, 0.0)
        elif sense == ">=":
            viol = np.maximum(-shifted, 0.0)
        elif sense == "==":
            viol = np.abs(shifted)
        else:
            raise ValueError(f"unknown constraint sense {con.sense!r} on row {con.name!r}")
        return float(np.max(viol)) if viol.size else 0.0

    if isinstance(con, _DisjunctiveConstraint):
        if not con.disjuncts:
            return 0.0
        return float(
            min(
                max((_one_row_violation(model, row, point) for row in disjunct), default=0.0)
                for disjunct in con.disjuncts
            )
        )

    if isinstance(con, _SOSConstraint):
        if con.sos_type != 1:
            raise ValueError(
                f"SOS{con.sos_type} row {con.name!r} has no residual definition here. No "
                "complementarity lowering emits one, and inventing a formula for it "
                "would be a silent approximation."
            )
        mags: list[float] = []
        for var in con.variables:
            mags.extend(float(abs(a)) for a in evaluate_at_point(model, var, point).ravel())
        if len(mags) < 2:
            return 0.0
        mags.sort(reverse=True)
        return mags[1]

    raise ValueError(
        f"cannot compute a residual for constraint-list entry of type "
        f"{type(con).__name__!r} (name={getattr(con, 'name', None)!r}). Add it to this "
        "dispatch — a row silently skipped here would make the residual report a "
        "maximum over the rows it happened to understand and read as complete."
    )


def _primal_feasibility(
    model: "Model",
    rows: Sequence["Constraint"],
    source_bounds: Optional[dict[int, tuple[np.ndarray, np.ndarray]]],
    point: dict[int, np.ndarray],
) -> Residual:
    """Max violation of the source rows **and** the source variable bounds."""
    res = _row_residual(
        model,
        rows,
        point,
        name="source_primal_feasibility",
        definition=(
            "max over source constraint rows of the violation of body <sense> rhs, "
            "and over variables of max(0, lb - x, x - ub) against the ORIGINAL "
            "declared bounds; evaluated on the source expression tree"
        ),
    )
    worst, where = float(res.value), res.where
    for v in model._variables:
        vals = point.get(id(v))
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=np.float64).ravel()
        if source_bounds is not None and id(v) in source_bounds:
            lb, ub = source_bounds[id(v)]
        else:
            lb, ub = v.lb, v.ub
        lb_arr = np.broadcast_to(np.asarray(lb, dtype=np.float64).ravel(), arr.shape)
        ub_arr = np.broadcast_to(np.asarray(ub, dtype=np.float64).ravel(), arr.shape)
        viol = float(np.max(np.maximum(np.maximum(lb_arr - arr, arr - ub_arr), 0.0)))
        if viol > worst:
            worst, where = viol, v.name
    return Residual(name=res.name, value=worst, definition=res.definition, where=where)


def _integrality_residual(
    model: "Model", point: dict[int, np.ndarray], VarType
) -> Optional[Residual]:
    r"""``max_i y_i(1 - y_i)`` over binary selectors, or ``None`` when there are none.

    The value is only meaningful for ``y in [0, 1]``: outside that interval
    ``y(1-y)`` is negative and *decreases* with the violation, so it would read
    as a better result the further out the point lies. The check is therefore
    made first, and a selector outside its box is reported as an out-of-box
    violation rather than folded into a product that hides it (#1148 §A).
    """
    ys: list[float] = []
    out_of_box = 0.0
    where = None
    for v in model._variables:
        if getattr(v, "var_type", None) is not VarType.BINARY:
            continue
        vals = point.get(id(v))
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=np.float64).ravel()
        excess = float(np.max(np.maximum(np.maximum(-arr, arr - 1.0), 0.0))) if arr.size else 0.0
        if excess > out_of_box:
            out_of_box, where = excess, v.name
        ys.extend(float(a) for a in arr)
    if not ys:
        return None
    # ``> _INTEGRALITY_TOL``, NOT ``> 0.0``. A binary comes back from an LP/MILP
    # as -1e-15 as a matter of routine roundoff; an exact-zero threshold called
    # that "outside the box" and reported the residual as +inf, so
    # ``source_satisfied`` went False on a perfectly good solution (#1158 review,
    # MEDIUM 5). The module's own integrality tolerance is the right scale.
    if out_of_box > _INTEGRALITY_TOL:
        return Residual(
            name="selector_integrality",
            value=float("inf"),
            definition=(
                "max_i y_i(1 - y_i) over binary selectors, AFTER checking 0 <= y_i <= 1; "
                "reported as +inf because a selector left the [0, 1] box by "
                f"{out_of_box:.3e}, where the product is negative and would read as a "
                "better result the further out the point lies"
            ),
            where=where,
        )
    return Residual(
        name="selector_integrality",
        value=float(max(y * (1.0 - y) for y in ys)),
        definition="max_i y_i(1 - y_i) over binary selectors, with 0 <= y_i <= 1 checked first",
    )


# ───────────────────── the local/certified result contract ─────────────────────


def accept_local_incumbent(
    model: "Model",
    result,
    *,
    x_flat=None,
):
    """Independently verify a **local** result before it may become an incumbent.

    A feasible point is a valid upper bound on a minimum, so a local solve may
    contribute an incumbent — but only after the point has been verified against
    the model by machinery that did not produce it. It may **never** contribute a
    dual bound: the dual bound is the certificate, and a local solve proves
    nothing about it. That asymmetry is what lets a local mode feed the global
    solver without contaminating it (#1148 §C).

    **The source relations are checked, and they are checked first.** Running
    ``verify_point`` alone against ``model`` is not verification of an MPEC point
    and was the bug this signature exists to prevent: on the Scholtes arm the
    model no longer holds the complementarity condition — the lowering replaced
    it with ``f >= 0``, ``g >= 0`` and the **relaxed** ``f·g <= t`` — so a point
    with a source residual of 1.4e-4 passed, and the function vouched for an
    objective strictly better than the true global optimum. Fed to a global solve
    as a cutoff, that fathoms the optimum away (#1158 review, HIGH 2). A relaxed
    row is weaker than the condition it replaced, so a check against the lowered
    model can never stand in for a check against the source relation.

    **The report is recomputed here, never accepted from the caller.** An earlier
    draft gated on ``result.mpec_report`` when one was carried, which authorized
    a *different* point: a report is a measurement of one ``(model, x)`` pair,
    and nothing tied it to the ``x_flat`` actually being vouched for. Passing an
    explicit ``x_flat`` alongside a result whose report was computed at the
    solver's own iterate made the gate vouch for an unmeasured point — verified
    ``-2e-4`` against a true optimum of ``0`` on a two-variable NCP whose carried
    report was clean at ``(0, 0)`` (#1158 review 3, blocking 1). The same hazard
    applies to the *model*: relations may have been added, rebuilt or re-scaled
    since the report was taken. Recomputing costs one evaluation of the source
    relations and is the only form of the check that is a statement about the
    point and the relations this call is actually asked to accept.

    Returns
    -------
    float or None
        The verified objective in model units, or ``None`` when the point could
        not be verified — in which case the caller must not use it. Never raises
        on an unverifiable point: refusing to vouch is the answer.
    """
    from discopt.validation.feasibility import verify_point

    if x_flat is None:
        x_flat = _flat_from_result(model, result)
    if x_flat is None:
        return None
    x_flat = np.asarray(x_flat, dtype=np.float64)

    # Recompute against the point and the relations in front of us. Absent
    # certification is interpreted as NOT certified (#1148 §C), so a measurement
    # that cannot be taken is a refusal, never a pass.
    if getattr(model, "_complementarities", None):
        try:
            report = source_residual_report(model, x_flat=x_flat)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            logger.warning(
                "accept_local_incumbent: refusing to vouch for the point because its "
                "source residuals could not be measured (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return None
        if not report.source_satisfied:
            logger.info(
                "accept_local_incumbent: refusing the point — source complementarity "
                "%.3e (scaled %.3e), operand bounds %.3e, source primal %.3e",
                report.complementarity.value,
                report.complementarity.scaled_value,
                report.bound_violation.value,
                report.primal_feasibility.value,
            )
            return None

    verdict = verify_point(model, x_flat, with_objective=True)
    return verdict.objective if verdict.ok else None


def _flat_from_result(model: "Model", result):
    """Flatten ``result.x`` into ``model._variables`` order, or ``None``."""
    x = getattr(result, "x", None)
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.ravel()
    if not isinstance(x, dict) or not x:
        return None
    parts = []
    for v in model._variables:
        if v.name not in x:
            return None
        parts.append(np.atleast_1d(np.asarray(x[v.name], dtype=np.float64)).ravel())
    return np.concatenate(parts) if parts else None
