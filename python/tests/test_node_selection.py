"""Tests for B&B node-selection strategies, including best-estimate search.

Best-estimate ordering is a *heuristic* over the open node set; it must never
change the proven optimum, only the order in which nodes are explored. We verify
that ``best_estimate`` returns the same global optimum as ``best_first`` and that
an unknown strategy is rejected.

The model is a small nonconvex MINLP (integer ``x``, continuous ``y``, a bilinear
constraint ``x*y <= 3``) so that the solve routes through the spatial
branch-and-bound engine (``PyTreeManager``) where the selection strategy actually
takes effect — pure-linear MILPs use the simplex driver, which is best-first.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import pytest


def _minlp() -> dm.Model:
    m = dm.Model("minlp_select")
    x = m.integer("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize((x - 1.7) ** 2 + (y - 2.3) ** 2)
    m.subject_to(x * y <= 3.0)  # nonconvex bilinear -> spatial B&B path
    return m


# Optimum: x=2, y=1.5 (x*y=3 binding) gives 0.09 + 0.64 = 0.73? — let the solver
# decide; the assertion below pins both strategies to the *same* value instead.
def test_best_estimate_matches_best_first():
    r1 = _minlp().solve(strategy="best_first")
    r2 = _minlp().solve(strategy="best_estimate")
    assert r1.status == "optimal" and r2.status == "optimal"
    assert abs(float(r1.objective) - float(r2.objective)) < 1e-4


@pytest.mark.parametrize("strategy", ["best_first", "depth_first", "best_estimate"])
def test_strategy_solves_to_optimal(strategy):
    res = _minlp().solve(strategy=strategy)
    assert res.status == "optimal"


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        _minlp().solve(strategy="bogus")
