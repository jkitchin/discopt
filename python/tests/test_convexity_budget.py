"""Convexity-classification cost controls (time-limit overshoot fix).

Convexity classification is eigenvalue-heavy and was called at several solver
dispatch sites — each running the full ~O(constraints) walk — and never checked
the time budget. On a large quadratic model this could spend tens of seconds
before search even started, blowing a tight ``time_limit``. Two controls:

* the result is memoized per solve (``classify_model`` runs at most once), and
* the walk honors a ``deadline`` and aborts to convexity-unknown (sound: routes
  to the spatial Branch and Bound) rather than overrunning the budget.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import discopt.solver as S
import pytest
from discopt._jax.convexity.rules import ConvexityBudgetExceeded, classify_model


def _qp(n: int = 4) -> dm.Model:
    m = dm.Model("qp")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.minimize(dm.sum([x[i] * x[i] for i in range(n)]))
    for i in range(n - 1):
        m.subject_to(x[i] + x[i + 1] <= 1.5, name=f"c{i}")
    return m


def test_classify_model_honors_deadline():
    """A deadline already in the past aborts the per-constraint walk."""
    m = _qp(6)
    with pytest.raises(ConvexityBudgetExceeded):
        classify_model(m, use_certificate=True, deadline=time.perf_counter() - 1.0)


def test_classify_model_no_deadline_completes():
    m = _qp(4)
    is_convex, mask = classify_model(m, use_certificate=True, deadline=None)
    assert is_convex is True  # sum of squares with linear constraints is convex
    assert len(mask) == 3


def test_classification_is_memoized_per_solve():
    """``_classify_model_convexity`` runs the underlying classifier at most once."""
    m = _qp(4)
    m._convexity_classification_cache = None
    m._convexity_time_budget = 5.0

    import discopt._jax.convexity as cvx

    calls = {"n": 0}
    orig = cvx.classify_model

    def _counting(model, **kw):
        calls["n"] += 1
        return orig(model, **kw)

    cvx.classify_model = _counting
    try:
        r1 = S._classify_model_convexity(m)
        r2 = S._classify_model_convexity(m)
        r3 = S._classify_model_convexity(m, log_nonconvex_continuous=True)
    finally:
        cvx.classify_model = orig

    assert r1 == r2 == r3
    assert calls["n"] == 1, f"classifier ran {calls['n']} times, expected 1 (memoized)"


def test_budget_exceeded_reports_unknown_not_crash():
    """When classification times out, the solver treats the model as unknown."""
    m = _qp(4)
    m._convexity_classification_cache = None
    m._convexity_time_budget = 1e-9  # effectively zero -> immediate timeout
    ok, is_convex, mask = S._classify_model_convexity(m)
    assert ok is False and is_convex is False and mask is None


@pytest.mark.parametrize("n", [3, 5])
def test_tight_time_limit_still_solves_correctly(n):
    """A tight limit caps classification but never changes the answer."""
    m = _qp(n)
    res = m.solve(time_limit=20)
    assert res.status == "optimal"
    # min sum x_i^2 over x>=0 with x_i+x_{i+1}<=1.5 -> all x=0, obj 0.
    assert abs(float(res.objective)) < 1e-4
