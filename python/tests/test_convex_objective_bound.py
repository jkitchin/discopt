"""Convex-objective node bound for nonconvex-constrained problems.

A large class of MINLPLib integer programs (nvs17/19/23/24) have a **convex
quadratic objective** but **nonconvex constraints**. The spatial B&B's McCormick
LP relaxation linearizes the convex objective with two tangents per square — a gap
of ``width**2/4`` at the box midpoint, ~10^4 per term over the instances' [0,200]
integer ranges — so the dual bound is hopelessly loose (nvs17 root −2522 vs the
optimum −1100) and never certifies in a reasonable node budget.

The fix keeps the convex objective *exact*: when the minimized objective is a
provable convex quadratic (structurally at most bilinear, with a PSD Hessian — a
quadratic's Hessian is constant, so one PSD check holds globally), each node adds
the **supporting-hyperplane** lower bound ``min_box [f(x0) + grad(x0)·(x − x0)]``,
which is a rigorous lower bound (exact arithmetic, valid for any ``x0`` by
convexity) and almost tight (≈ the box minimum near ``x0``). nvs17/19/23 flip from
"feasible, ~100% gap" to certified optimal in seconds.

Soundness hinges on the convexity gate: a nonconvex objective must NOT be flagged,
or the hyperplane would be an over-estimator and could fathom the optimum.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver import (
    _convex_objective_lower_bound,
    _make_evaluator,
    _objective_is_convex_quadratic,
)

_NVS17 = "python/tests/data/minlplib/nvs17.nl"


def _flat_bounds(m):
    lb = np.array([v.lb for v in m._variables for _ in range(v.size)], dtype=float)
    ub = np.array([v.ub for v in m._variables for _ in range(v.size)], dtype=float)
    return lb, ub


def test_convex_quadratic_objective_is_detected():
    """A PSD-Hessian quadratic objective is flagged; the bound is a valid LB."""
    m = dm.Model("cvxq")
    x = m.integer("x", lb=0, ub=20)
    y = m.integer("y", lb=0, ub=20)
    # 3x^2 + 2y^2 + xy is convex (Hessian [[6,1],[1,4]] is PD), minus a linear term.
    m.minimize(3 * x**2 + 2 * y**2 + x * y - 40 * x - 30 * y)
    m.subject_to(x + y >= 1)
    n = 2
    ev = _make_evaluator(m)
    assert _objective_is_convex_quadratic(m, ev, n) is True
    lb, ub = _flat_bounds(m)
    bound = _convex_objective_lower_bound(ev, lb, ub)
    # The continuous box minimum is a valid lower bound on the integer optimum.
    assert np.isfinite(bound)
    # Brute-force the true integer optimum and confirm the bound never exceeds it.
    best = min(
        3 * i * i + 2 * j * j + i * j - 40 * i - 30 * j
        for i in range(21)
        for j in range(21)
        if i + j >= 1
    )
    assert bound <= best + 1e-6


def test_nonconvex_objective_is_not_flagged():
    """Soundness gate: an indefinite-Hessian objective must NOT be treated as
    convex (its supporting hyperplane would be an unsound over-estimator)."""
    m = dm.Model("noncvx")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(x * y)  # Hessian [[0,1],[1,0]] is indefinite (eigs +-1)
    m.subject_to(x + y >= 3)
    assert _objective_is_convex_quadratic(m, _make_evaluator(m), 2) is False


def test_non_quadratic_objective_is_not_flagged():
    """A higher-than-quadratic objective is not a convex quadratic."""
    m = dm.Model("cubic")
    x = m.continuous("x", lb=1, ub=5)
    m.minimize(x**2 / (x + 1))  # division -> not a polynomial quadratic
    assert _objective_is_convex_quadratic(m, _make_evaluator(m), 1) is False


def test_supporting_hyperplane_is_a_valid_lower_bound():
    """The hyperplane bound is sound for ANY box (not just near the minimizer)."""
    m = dm.Model("q")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(x**2 + y**2 + 0.5 * x * y)
    ev = _make_evaluator(m)
    rng = np.random.default_rng(0)

    def f(v):
        return float(ev.evaluate_objective(np.asarray(v, float)))

    for _ in range(20):
        lo = rng.uniform(-5, 4, size=2)
        hi = lo + rng.uniform(0.5, 5, size=2)
        hi = np.minimum(hi, 5.0)
        bound = _convex_objective_lower_bound(ev, lo, hi)
        # Sample the box densely; the bound must underestimate every sample.
        for _ in range(50):
            xs = lo + rng.random(2) * (hi - lo)
            assert bound <= f(xs) + 1e-6


@pytest.mark.skipif(not os.path.exists(_NVS17), reason="nvs17.nl not vendored")
def test_nvs17_certifies_via_convex_objective_bound():
    """End-to-end: nvs17 (convex quadratic objective, nonconvex constraints,
    [0,200] integer ranges) certifies to its known optimum −1100.4."""
    r = from_nl(_NVS17).solve(time_limit=40, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective == pytest.approx(-1100.4, abs=1e-2)
    # The certified bound is a valid lower bound that met the incumbent.
    assert r.bound is not None and r.bound <= r.objective + 1e-4 * max(1.0, abs(r.objective))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
