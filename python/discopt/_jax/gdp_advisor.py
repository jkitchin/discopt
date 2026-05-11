"""GDP big-M vs. hull reformulation advisor (F1 of issue #53).

discopt-distinctive. ``_jax/gdp_reformulate.py`` already supports three
reformulation methods (``big-m``, ``mbigm``, ``hull``) but the choice is
fixed by the caller. This module makes the choice a *presolve decision*
driven by structural properties of each disjunction:

- **Hull** is the tightest LP relaxation but pays for it with
  disaggregated copies of every common variable per disjunct. It is
  exact for linear constraints and is asymptotically dominant when
  disjunctions are small and linear.
- **Big-M** introduces only one indicator per disjunct and at most a
  constant number of constraints per original constraint. It is
  *structurally* the cheapest option and the LP relaxation is
  competitive when M is naturally tight.
- **MBigM** keeps the big-M structure but solves an LP to tighten each
  M, recouping a fraction of the LP-tightness gap at one LP solve per
  active constraint.

The advisor inspects each ``_DisjunctiveConstraint`` and emits a
recommendation per disjunction, plus a diagnostic record explaining
which structural feature drove the choice.

References
----------
- Grossmann, Trespalacios (2013), *Systematic modeling of
  discrete-continuous optimization models through generalized
  disjunctive programming*, AIChE J. 59.
- Trespalacios, Grossmann (2014), *Review of mixed-integer nonlinear
  and generalized disjunctive programming methods*, Chem. Ing. Tech.
  86.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    Constraint,
    Model,
    _DisjunctiveConstraint,
)

# Heuristic thresholds. Treat these as defaults; callers may override.
DEFAULT_HULL_AUX_BUDGET = 50  # Max hull aux vars before big-M wins on cost
DEFAULT_BIG_M_TIGHT_THRESHOLD = 100.0  # If max-M ≤ this, big-M is "tight"


@dataclass
class GdpAdvice:
    """Per-disjunction reformulation recommendation.

    Attributes
    ----------
    disjunction_index : int
        Position of the disjunction in ``model._constraints``.
    name : Optional[str]
        Disjunction name, copied from the source constraint.
    recommendation : str
        One of ``"big-m"``, ``"hull"``, ``"mbigm"``.
    rationale : str
        Short human-readable explanation of which heuristic fired.
    n_disjuncts : int
        Number of disjuncts (branches) in the disjunction.
    n_common_vars : int
        Distinct continuous/integer variables appearing across disjuncts
        (the hull-disaggregation cost is driven by this).
    hull_aux_cost : int
        Number of disaggregated continuous variables hull would
        introduce: ``n_disjuncts * n_common_vars``.
    has_nonlinear : bool
        True iff any constraint body is non-linear (perspective hull
        is numerically tricky for these — big-M is usually preferred).
    max_big_m : float
        Largest big-M value over all disjunct constraints, computed via
        interval arithmetic. ``inf`` if any bound is unbounded.
    """

    disjunction_index: int
    name: Optional[str]
    recommendation: str
    rationale: str
    n_disjuncts: int
    n_common_vars: int
    hull_aux_cost: int
    has_nonlinear: bool
    max_big_m: float


def recommend_method(
    dc: _DisjunctiveConstraint,
    model: Model,
    *,
    hull_aux_budget: int = DEFAULT_HULL_AUX_BUDGET,
    big_m_tight_threshold: float = DEFAULT_BIG_M_TIGHT_THRESHOLD,
) -> GdpAdvice:
    """Recommend a reformulation method for one disjunction.

    Heuristic order, mirroring the trade-offs in Grossmann & Trespalacios
    (2013) §4:

    1. If any disjunct contains a non-linear constraint, prefer
       ``big-m`` — perspective hull on non-linear bodies needs ε-clamping
       and is numerically delicate. Promote to ``mbigm`` if the body is
       bounded so the LP-tightening pays off.
    2. If hull's aux-variable cost (``n_disjuncts * n_common_vars``)
       exceeds ``hull_aux_budget``, fall back to ``big-m``/``mbigm`` even
       for linear disjunctions.
    3. Otherwise, with all-linear constraints and a manageable
       disaggregation cost, pick ``hull``: its LP relaxation strictly
       dominates and the cost is bounded.
    4. Within the big-M branch, choose ``mbigm`` over ``big-m`` when
       interval-arithmetic M is loose (max_M > threshold) — LP-based M
       tightening recovers most of the relaxation gap.
    """
    from discopt._jax.gdp_reformulate import _bound_expression, _collect_variables, _is_linear

    name = getattr(dc, "name", None)
    n_disjuncts = len(dc.disjuncts)

    # Distinct variables appearing anywhere in the disjunction.
    common_vars: dict[str, object] = {}
    for disjunct in dc.disjuncts:
        for c in disjunct:
            if isinstance(c, Constraint):
                common_vars.update(_collect_variables(c.body))
    n_common_vars = len(common_vars)
    hull_aux_cost = n_disjuncts * n_common_vars

    # Linearity + max big-M scan.
    has_nonlinear = False
    max_big_m = 0.0
    any_unbounded_M = False
    for disjunct in dc.disjuncts:
        for c in disjunct:
            if not isinstance(c, Constraint):
                # Nested disjunctions etc. are out of scope for the
                # advisor — fall back to big-M because nested hull
                # blows up combinatorially.
                has_nonlinear = True  # use as a "force big-M" flag
                continue
            if not _is_linear(c.body):
                has_nonlinear = True
            try:
                lo, hi = _bound_expression(c.body, model)
            except Exception:
                any_unbounded_M = True
                continue
            for v in (lo, hi):
                fv = float(np.asarray(v).max() if np.ndim(v) > 0 else v)
                if not np.isfinite(fv):
                    any_unbounded_M = True
                    continue
                if abs(fv) > max_big_m:
                    max_big_m = abs(fv)
    effective_M = float("inf") if any_unbounded_M else max_big_m

    # --- Decision tree ---
    if has_nonlinear:
        if any_unbounded_M:
            rec = "big-m"
            why = "nonlinear+unbounded body ⇒ big-m with default M"
        elif effective_M > big_m_tight_threshold:
            rec = "mbigm"
            why = f"nonlinear, big-M ≈ {effective_M:.3g} > {big_m_tight_threshold} ⇒ tighten via LP"
        else:
            rec = "big-m"
            why = f"nonlinear, big-M ≈ {effective_M:.3g} already tight"
        return GdpAdvice(
            disjunction_index=-1,
            name=name,
            recommendation=rec,
            rationale=why,
            n_disjuncts=n_disjuncts,
            n_common_vars=n_common_vars,
            hull_aux_cost=hull_aux_cost,
            has_nonlinear=True,
            max_big_m=effective_M,
        )

    # Linear from here on.
    if hull_aux_cost > hull_aux_budget:
        # Too many disaggregated variables — fall back to MBM.
        if any_unbounded_M:
            rec = "big-m"
            why = (
                f"linear but hull cost {hull_aux_cost} > {hull_aux_budget}, body unbounded ⇒ big-m"
            )
        else:
            rec = "mbigm"
            why = f"linear but hull cost {hull_aux_cost} > {hull_aux_budget} ⇒ tighten big-m via LP"
        return GdpAdvice(
            disjunction_index=-1,
            name=name,
            recommendation=rec,
            rationale=why,
            n_disjuncts=n_disjuncts,
            n_common_vars=n_common_vars,
            hull_aux_cost=hull_aux_cost,
            has_nonlinear=False,
            max_big_m=effective_M,
        )

    # Linear and small enough — hull wins on tightness, cost is bounded.
    return GdpAdvice(
        disjunction_index=-1,
        name=name,
        recommendation="hull",
        rationale=(
            f"linear, hull cost {hull_aux_cost} ≤ {hull_aux_budget} ⇒ hull (tightest LP relaxation)"
        ),
        n_disjuncts=n_disjuncts,
        n_common_vars=n_common_vars,
        hull_aux_cost=hull_aux_cost,
        has_nonlinear=False,
        max_big_m=effective_M,
    )


def recommend_methods(
    model: Model,
    *,
    hull_aux_budget: int = DEFAULT_HULL_AUX_BUDGET,
    big_m_tight_threshold: float = DEFAULT_BIG_M_TIGHT_THRESHOLD,
) -> list[GdpAdvice]:
    """Run the F1 advisor over every disjunction in ``model``.

    Returns an advice list parallel to the disjunction order; each
    record carries its source ``disjunction_index`` so the caller can
    align with ``model._constraints``.
    """
    out: list[GdpAdvice] = []
    for ci, c in enumerate(model._constraints):
        if not isinstance(c, _DisjunctiveConstraint):
            continue
        adv = recommend_method(
            c,
            model,
            hull_aux_budget=hull_aux_budget,
            big_m_tight_threshold=big_m_tight_threshold,
        )
        adv.disjunction_index = ci
        out.append(adv)
    return out


__all__ = [
    "DEFAULT_BIG_M_TIGHT_THRESHOLD",
    "DEFAULT_HULL_AUX_BUDGET",
    "GdpAdvice",
    "recommend_method",
    "recommend_methods",
]
