"""Robust optimization for discopt models.

This module provides tools to reformulate a nominal MINLP model with uncertain
parameters into a deterministic *robust counterpart* that is feasible for every
realization of the uncertainty within a prescribed set.

Supported uncertainty sets
--------------------------
:class:`BoxUncertaintySet`
    Each uncertain parameter component varies independently within a symmetric
    interval [p̄ - δ, p̄ + δ].  The worst-case is separable and leads to a
    1-norm (box) penalty.

:class:`EllipsoidalUncertaintySet`
    The uncertain vector lies within an ellipsoid parameterized by a shape
    matrix Σ and radius ρ.  Leads to a 2-norm (SOCP) penalty.

:class:`PolyhedralUncertaintySet`
    The perturbation vector satisfies a set of linear inequalities Aξ ≤ b.
    Includes the budget-of-uncertainty (Bertsimas & Sim) as a special case.
    Reformulated via LP duality.

Main entry point
----------------
:class:`RobustCounterpart`
    Wraps a model and an uncertainty set; calling :meth:`~RobustCounterpart.formulate`
    rewrites the model in-place.

Example
-------
>>> import discopt.modeling as dm
>>> from discopt.ro import BoxUncertaintySet, RobustCounterpart
>>>
>>> m = dm.Model("robust_production")
>>> x = m.continuous("x", shape=(3,), lb=0)
>>> c = m.parameter("c", value=[10.0, 15.0, 8.0])
>>> d = m.parameter("d", value=100.0)
>>>
>>> m.minimize(dm.sum(c * x))
>>> m.subject_to(dm.sum(x) >= d, name="demand")
>>>
>>> rc = RobustCounterpart(m, [BoxUncertaintySet(c, delta=1.0),
...                             BoxUncertaintySet(d, delta=5.0)])
>>> rc.formulate()
>>> result = m.solve()

References
----------
Ben-Tal, A., Nemirovski, A. (1999). Robust solutions of uncertain linear
programs. *Operations Research Letters*, 25(1), 1–13.

Bertsimas, D., Sim, M. (2004). The price of robustness. *Operations Research*,
52(1), 35–53.

Ben-Tal, A., El Ghaoui, L., Nemirovski, A. (2009). *Robust Optimization*.
Princeton University Press.
"""

from discopt.ro.affine_policy import AffineDecisionRule
from discopt.ro.counterpart import RobustCounterpart
from discopt.ro.uncertainty import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    UncertaintySet,
    budget_uncertainty_set,
)

__all__ = [
    "AffineDecisionRule",
    "BoxUncertaintySet",
    "EllipsoidalUncertaintySet",
    "PolyhedralUncertaintySet",
    "RobustCounterpart",
    "UncertaintySet",
    "budget_uncertainty_set",
]
