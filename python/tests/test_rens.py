"""Tests for RENS (Relaxation Enforced Neighborhood Search), the convex-MINLP
root primal heuristic (issue #281).

RENS fixes the integers that are already integral in the continuous relaxation,
restricts each fractional integer to its ``{floor, ceil}`` unit box, and solves
the resulting small sub-MINLP exactly. On a near-integral convex relaxation this
lands the *optimal* integer assignment at the root — where the feasibility pump's
all-at-once rounding settles for a feasible-but-suboptimal one (or none).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.primal_heuristics import rens  # noqa: E402


def _miqp():
    """Convex integer QP. Continuous relaxation projects onto ``sum == 5`` at the
    fractional point ~``[1.7, 2.5, 0.8]``; the in-box integer optimum is
    ``[2, 2, 1]`` (obj 0.41), which all-at-once rounding of the relaxation misses
    (e.g. it violates the budget)."""
    m = dm.Model("miqp")
    x = m.integer("x", shape=3, lb=0, ub=4)
    m.minimize((x[0] - 1.6) ** 2 + (x[1] - 2.4) ** 2 + (x[2] - 0.7) ** 2)
    m.subject_to(x[0] + x[1] + x[2] == 5)
    return m, x


def test_rens_neighborhood_restricts_and_restores():
    """Integral relaxation integers are fixed to their rounding; fractional ones
    are restricted to ``{floor, ceil}``; the model's bounds are restored after."""
    m, _ = _miqp()
    # x0 integral (2.0), x1 fractional (2.4), x2 integral (1.0).
    x_relax = np.array([2.0, 2.4, 1.0])
    saved = [(np.array(v.lb).copy(), np.array(v.ub).copy()) for v in m._variables]

    seen: dict = {}

    def sub_solver(model):
        v = model._variables[0]
        seen["lb"] = np.array(v.lb, dtype=float).copy()
        seen["ub"] = np.array(v.ub, dtype=float).copy()
        return np.array([2.0, 2.0, 1.0]), 0.41

    res = rens(m, x_relax, sub_solver=sub_solver)
    assert res is not None
    assert seen["lb"].tolist() == [2.0, 2.0, 1.0]
    assert seen["ub"].tolist() == [2.0, 3.0, 1.0]  # x1 free over {2, 3}
    # Bounds restored to the originals.
    for v, (lb, ub) in zip(m._variables, saved):
        assert np.array_equal(np.array(v.lb), lb)
        assert np.array_equal(np.array(v.ub), ub)


def test_rens_returns_none_when_too_many_fractional():
    """RENS bails (without invoking the sub-solver) when the neighbourhood is too
    large — so the pump/diving fallback still covers many-fractional relaxations."""
    m, _ = _miqp()
    x_relax = np.array([1.5, 2.5, 0.5])  # all three fractional
    called = {"n": 0}

    def sub_solver(model):
        called["n"] += 1
        return None

    assert rens(m, x_relax, sub_solver=sub_solver, max_free=2) is None
    assert called["n"] == 0


@pytest.mark.requires_pounce
def test_rens_solve_reaches_optimum():
    """End-to-end: the convex MIQP solves to its known optimum with RENS on."""
    m, _ = _miqp()
    r = m.solve(time_limit=30, gap_tolerance=1e-4, rens=True)
    assert r.objective == pytest.approx(0.41, abs=1e-3)
    assert np.round(np.asarray(r.x["x"])).astype(int).tolist() == [2, 2, 1]


@pytest.mark.requires_pounce
def test_rens_disabled_is_still_correct():
    """Disabling RENS must not change correctness (the optimum is still reached)."""
    m, _ = _miqp()
    r = m.solve(time_limit=30, gap_tolerance=1e-4, rens=False)
    assert r.objective == pytest.approx(0.41, abs=1e-3)
