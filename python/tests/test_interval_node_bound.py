"""Soundness tests for the per-node interval-arithmetic lower bound.

``_compute_interval_bound`` fills the lower bound of nonconvex spatial-B&B
nodes with a cheap, always-valid interval enclosure so a node never sits at
``-inf`` between the periodic (and tighter) McCormick-NLP solves. The bound is
allowed to be loose, but it must NEVER exceed the true minimum of the
objective over the node box — otherwise the B&B could prune the global optimum
and report a wrong "optimal". These tests pin that invariant, and the
end-to-end guarantee that a certified result is sound.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest


def _sampled_min(eval_obj, lb, ub, n=20000, seed=0):
    """Monte-Carlo estimate of min f over the box (an upper bound on the true
    min, hence a conservative target the interval bound must stay below)."""
    rng = np.random.default_rng(seed)
    pts = lb + (ub - lb) * rng.random((n, len(lb)))
    return min(float(eval_obj(p)) for p in pts)


@pytest.mark.parametrize(
    "build,lb,ub",
    [
        # Bilinear (nonconvex)
        (lambda m: m.minimize(-m._v["x"] * m._v["y"] + m._v["x"]), [0.0, 0.0], [3.0, 2.0]),
        # Concave quadratic
        (
            lambda m: m.minimize(-(m._v["x"] * m._v["x"]) - (m._v["y"] * m._v["y"])),
            [-3.0, -3.0],
            [3.0, 3.0],
        ),
        # Mixed sign products + linear
        (
            lambda m: m.minimize(m._v["x"] * m._v["y"] - 4 * m._v["x"] - 3 * m._v["y"]),
            [0.0, 0.0],
            [10.0, 10.0],
        ),
    ],
)
def test_interval_bound_is_valid_underestimator(build, lb, ub):
    m = dm.Model("t")
    x = m.continuous("x", lb=lb[0], ub=ub[0])
    y = m.continuous("y", lb=lb[1], ub=ub[1])
    m._v = {"x": x, "y": y}
    build(m)

    evaluator = S._make_evaluator(m)
    lb_a = np.array(lb)
    ub_a = np.array(ub)

    bound = S._compute_interval_bound(m, lb_a, ub_a, negate=False)
    target = _sampled_min(evaluator.evaluate_objective, lb_a, ub_a)

    assert np.isfinite(bound)
    # The interval bound underestimates the true min, which is <= the sampled
    # min. A tiny tolerance absorbs outward-rounding / sampling noise.
    assert bound <= target + 1e-6


def test_interval_bound_respects_maximize_negation():
    """For a maximization model the internal B&B minimizes ``-f``; the bound
    must be a valid lower bound on ``-f`` (i.e. ``<= -max f``)."""
    m = dm.Model("t")
    a = m.continuous("a", lb=0.0, ub=3.0)
    b = m.continuous("b", lb=0.0, ub=2.0)
    m.maximize(a * b)  # max f = 6 at (3, 2) -> internal min of -f is -6

    bound = S._compute_interval_bound(m, np.array([0.0, 0.0]), np.array([3.0, 2.0]), negate=True)
    assert np.isfinite(bound)
    assert bound <= -6.0 + 1e-6


def test_interval_bound_never_invalid_on_unsupported_ops():
    """Unsupported operators yield an unbounded enclosure; the helper must
    degrade to ``-inf`` (a no-op) rather than emit an invalid finite bound."""
    m = dm.Model("t")
    x = m.continuous("x", lb=0.1, ub=2.0)
    # erf is not modelled by the interval evaluator -> unbounded -> -inf.
    try:
        m.minimize(dm.erf(x))
    except AttributeError:
        pytest.skip("dm.erf not available in this build")
    bound = S._compute_interval_bound(m, np.array([0.1]), np.array([2.0]), negate=False)
    assert bound == -np.inf


def test_certified_optimal_is_sound():
    """End-to-end: when the default spatial B&B reports a certified optimum on
    a nonconvex model, the returned bound must not exceed the objective."""
    m = dm.Model("concave1d")
    x = m.continuous("x", lb=0.0, ub=3.0)
    m.minimize(-(x - 1) * (x - 1))  # global min -4 at x=3
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-4
    assert abs(r.objective - (-4.0)) <= 1e-3
