"""A certified result's reported pair must itself close the gap (#1285).

A heuristic incumbent that beats the true optimum by using up its feasibility
tolerance closed the tree; the terminal KKT polish then swapped in a feasible
point 5.1e-5 worse and the result stayed ``gap_certified=True`` with a pair
failing both the absolute and the relative test (and no ``gap_criterion``).
"""

import math

import discopt.modeling as dm
import pytest
from discopt.solver import _DEFAULT_ABS_GAP_TOL, _gap_values_converged

F_STAR = 2.4 - (math.sqrt(3.0) - math.sqrt(0.13)) ** 2


def _model(square):
    m = dm.Model("t")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    k = m.integer("k", lb=0, ub=3)
    sq = (lambda v: v * v) if square == "mul" else (lambda v: v**2)
    m.subject_to(sq(x) + sq(y) >= 1.5 + 0.5 * k)
    m.subject_to(x - y >= 0.3 * k)
    m.subject_to(x * y <= 0.4)
    m.maximize(-((x - 0.3) ** 2) - (y + 0.2) ** 2 + 0.8 * k)
    return m


@pytest.mark.parametrize("square", ["mul", "pow"])
def test_certified_pair_closes_the_gap(square):
    r = _model(square).solve(time_limit=30)
    assert r.objective == pytest.approx(F_STAR, abs=1e-6)
    assert r.bound >= F_STAR - 1e-9
    if r.status == "optimal" or r.gap_certified:
        assert r.status == "optimal" and r.gap_certified
        # MAXIMIZE: the bound is the upper end of the pair.
        assert _gap_values_converged(r.bound, r.objective, 1e-4, _DEFAULT_ABS_GAP_TOL)
        assert (r.solver_stats or {}).get("gap_criterion") in ("absolute", "relative")
