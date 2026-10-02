"""#1569: the convex-quadratic objective node bound on ARRAY-variable models.

``_objective_is_convex_quadratic`` built its sample box as
``[v.lb for v in model._variables for _ in range(v.size)]``, which repeats a
``shape=(n,)`` variable's whole bound array ``n`` times. Every array-variable
model therefore raised inside the test (``ValueError: inhomogeneous shape`` or
``hessian: x: expected length 3, got 9``), the ``except`` turned it into a
one-time warning, and the bound never engaged.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
from discopt.solver import (
    _convex_objective_lower_bound,
    _make_evaluator,
    _objective_is_convex_quadratic,
)


def _array_quadratic(objective: str = "convex") -> dm.Model:
    m = dm.Model(f"arr_{objective}")
    x = m.continuous("x", shape=(3,), lb=np.array([-2.0, -1.0, 0.0]), ub=np.array([2.0, 3.0, 1.5]))
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.subject_to(x[0] * x[1] >= 0.5)  # nonconvex: keeps the spatial B&B path
    if objective == "convex":
        m.minimize(sum((x[i] - 0.3 * i) ** 2 for i in range(3)) + x[0] * x[1] + y**2)
    elif objective == "concave":
        m.minimize(-sum(x[i] ** 2 for i in range(3)) + y)
    else:  # cubic: not a quadratic at all
        m.minimize(sum(x[i] ** 3 for i in range(3)) + y**2)
    return m


def _n(m: dm.Model) -> int:
    return sum(int(v.size) for v in m._variables)


def test_convex_quadratic_detected_on_array_model(caplog):
    m = _array_quadratic("convex")
    with caplog.at_level(logging.WARNING, logger="discopt"):
        ok = _objective_is_convex_quadratic(m, _make_evaluator(m), _n(m))
    assert ok, "a PSD quadratic objective over an array variable must engage"
    assert not [r for r in caplog.records if "convex-quadratic objective test" in r.getMessage()]


def test_array_model_does_not_over_claim():
    for kind in ("concave", "cubic"):
        m = _array_quadratic(kind)
        assert not _objective_is_convex_quadratic(m, _make_evaluator(m), _n(m)), kind


def test_bound_never_exceeds_the_objective_inside_the_box():
    """Feasible-point sampling: for random node boxes, every sampled point's
    objective is at least the convex bound for that box."""
    m = _array_quadratic("convex")
    ev = _make_evaluator(m)
    assert _objective_is_convex_quadratic(m, ev, _n(m))
    lo = np.array([-2.0, -1.0, 0.0, -1.0])
    hi = np.array([2.0, 3.0, 1.5, 1.0])
    rng = np.random.default_rng(1569)
    checked = 0
    for _ in range(40):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        nlb, nub = np.minimum(a, b), np.maximum(a, b)
        lb = _convex_objective_lower_bound(ev, nlb, nub)
        assert np.isfinite(lb)
        for _ in range(25):
            pt = rng.uniform(nlb, nub)
            assert float(ev.evaluate_objective(pt)) >= lb - 1e-9
            checked += 1
    assert checked == 1000


def test_solve_engages_the_bound_without_the_warning(caplog):
    m = _array_quadratic("convex")
    with caplog.at_level(logging.DEBUG, logger="discopt"):
        r = m.solve(time_limit=20, deterministic=True)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert not [s for s in msgs if "convex-quadratic objective test" in s]
    assert [s for s in msgs if "convex-objective node bound enabled" in s]
    assert r.gap_certified
