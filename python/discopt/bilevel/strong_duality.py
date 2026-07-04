"""Strong-duality (aggregated-complementarity) reformulation of a convex lower level.

Phase 2 of the bilevel module (``docs/dev/bilevel-module-plan.md`` §4). An
alternative to the disjunctive KKT–MPEC encoding: keep the KKT **stationarity**,
**primal feasibility**, and **dual feasibility** (``μ_i >= 0``), but replace the
per-row complementarity conditions with the *single* aggregate equality

    Σ_i μ_i · g_i(x, y)  +  Σ_j ν_j · h_j(x, y)  ==  0.

For inequality rows ``μ_i >= 0`` and ``g_i <= 0`` (primal feasibility), so every
term ``μ_i g_i <= 0``; their sum vanishing forces each ``μ_i g_i = 0`` — exact
complementarity. Equality rows contribute ``ν_j h_j`` with ``h_j = 0`` at
feasibility. For an LP lower level this is precisely the **strong-duality**
condition ``primal_obj == dual_obj`` (Σ μ_i g_i = 0 ⇔ ``d·y = -h·μ``).

The trade vs. the GDP/SOS1 KKT path: one **bilinear equality** (handled by the
spatial/McCormick global path) instead of a disjunction / SOS1 set — no discrete
complementarity branching, at the cost of a nonconvex equality. Both are exact and
must agree on the optimum (the ``kkt ≡ strong-duality`` cross-check).

Reuses :func:`discopt.bilevel.kkt.build_kkt` for stationarity/primal/multipliers,
so it inherits the symbolic-differentiator stationarity (certified global path).
"""

from __future__ import annotations

from dataclasses import dataclass

from discopt.bilevel import kkt as _kkt
from discopt.modeling.core import Constraint, Expression, Model, Variable


@dataclass
class StrongDualitySystem:
    """Pieces emitted by :func:`build_strong_duality` (for inspection/testing)."""

    multipliers: list[Variable]
    stationarity: list[Constraint]
    primal: list[Constraint]
    strong_duality: Constraint
    lagrangian: Expression


def build_strong_duality(
    model: Model,
    *,
    lower_vars: list[Variable],
    lower_objective: Expression,
    lower_constraints: list[Constraint],
    lower_sense: str = "min",
    prefix: str = "bl",
) -> StrongDualitySystem:
    """Emit the strong-duality single-level conditions onto ``model`` (in place).

    Adds the KKT stationarity + primal feasibility + multipliers (via
    :func:`~discopt.bilevel.kkt.build_kkt`), then the single aggregate equality
    ``Σ_i μ_i g_i + Σ_j ν_j h_j == 0`` in place of per-row complementarity. The
    leader's objective is not touched.
    """
    sys = _kkt.build_kkt(
        model,
        lower_vars=lower_vars,
        lower_objective=lower_objective,
        lower_constraints=lower_constraints,
        lower_sense=lower_sense,
        prefix=prefix,
    )

    # Aggregate complementarity / strong-duality: Σ μ_i g_i == 0.
    # (lagrangian = f_signed + Σ μ_i g_i, so the aggregate is lagrangian - f_signed;
    # build it directly from the multiplier·body terms to keep the node small.)
    agg: Expression | None = None
    for mult, con in zip(sys.multipliers, lower_constraints):
        term = mult * con.body
        agg = term if agg is None else agg + term
    if agg is None:
        raise ValueError("strong-duality reformulation needs at least one lower constraint")

    sd = Constraint(body=agg, sense="==", rhs=0.0, name=f"{prefix}_strong_duality")
    model.subject_to(sd, name=sd.name)

    return StrongDualitySystem(
        multipliers=sys.multipliers,
        stationarity=sys.stationarity,
        primal=sys.primal,
        strong_duality=sd,
        lagrangian=sys.lagrangian,
    )
