"""#1496: ``nonlinear_to_pwl`` must publish incumbents inside the declared box.

``verify_point`` accepts a point up to tolerance outside the bounds. The polished
(or mapped) point used to be kept raw, so its objective could beat every in-box
point: ``min (x-0.3)**2`` on ``[1e4, 1e4+10]`` returned x = 9999.9999 with an
objective below the certified bound, and ``max exp(x)`` on ``[0, 18]`` evaluated
past e^18 and tripped the soundness tripwire on a perfectly valid bound.
The candidate is now clipped into the box (integers rounded) and re-verified.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _check_result(r, lb, ub, maximize):
    xv = float(np.asarray(r.x["x"]).ravel()[0])
    assert lb <= xv <= ub, f"incumbent x = {xv!r} outside [{lb}, {ub}]"
    assert r.bound is not None and r.objective is not None
    slack = 1e-9 * max(1.0, abs(r.objective))
    if maximize:
        assert r.objective <= r.bound + slack, (r.objective, r.bound)
    else:
        assert r.objective >= r.bound - slack, (r.objective, r.bound)
    return 2


def test_issue_repro_max_exp_does_not_trip():
    m = dm.Model("a")
    x = m.continuous("x", lb=0, ub=18)
    m.maximize(dm.exp(x))
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert _check_result(r, 0.0, 18.0, True) == 2
    assert r.objective <= np.exp(18.0) * (1 + 1e-12)


def test_issue_repro_min_square_objective_not_below_bound():
    m = dm.Model("b")
    x = m.continuous("x", lb=1e4, ub=1e4 + 10)
    m.minimize((x - 0.3) ** 2)
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert _check_result(r, 1e4, 1e4 + 10, False) == 2
    assert r.objective >= (1e4 - 0.3) ** 2 * (1 - 1e-12)


@pytest.mark.parametrize("ub", [5.0, 12.0, 18.0])
def test_max_exp_panel(ub):
    m = dm.Model("e")
    x = m.continuous("x", lb=0.0, ub=ub)
    m.maximize(dm.exp(x))
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert _check_result(r, 0.0, ub, True) == 2


@pytest.mark.parametrize("off", [1e2, 1e3, 1e4])
def test_min_square_at_the_lower_bound_panel(off):
    """Optimum at the lower bound: the polish NLP tends to step just past it."""
    m = dm.Model("s")
    x = m.continuous("x", lb=off, ub=off + 10)
    m.minimize((x - 0.3) ** 2)
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    assert _check_result(r, off, off + 10, False) == 2
