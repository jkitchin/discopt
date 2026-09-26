"""#1495: ``nonlinear_to_pwl`` (outer) at a large input offset.

The disaggregated ``_n2p*_x`` row was written uncentred,
``x == sum_i (b_i wl_i + b_{i+1} wr_i)`` with ``b_i ~ 1e5``. HiGHS meets
``sum(wl + wr) = sum(z) = 1`` only to ~1e-6, so ``x`` could drift by ~0.1 and
the relaxation "certified" 2.5e-5 on ``min (x - (1e5 + 0.37))**2`` over
``[1e5, 1e5 + 1]`` (true minimum 0). With ``polish=False`` that false bound was
published as ``gap_certified=True``; by default the tripwire raised. The rows
are now centred on the first breakpoint and its sample.

The invariant tested is the certificate one: for a minimisation the reported
bound never exceeds the true minimum, and a certified result's objective meets it.
"""

from __future__ import annotations

import re

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model, SolveResult

OFFSETS = [1e3, 1e4, 1e5, 1e6, 1e7]


def _shifted_square(off, width, frac=0.37):
    c0 = off + frac * width
    m = dm.Model("t")
    x = m.continuous("x", lb=off, ub=off + width)
    m.minimize((x - c0) ** 2)
    return m


def test_issue_repro_surrogate_solve():
    """The transformed MILP itself: its optimum is 0 (a feasible point with w = 0
    exists), not 2.5e-5."""
    t = dm.nonlinear_to_pwl(_shifted_square(1e5, 1.0))
    r = t.model.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective <= 1e-6, r.objective


@pytest.mark.parametrize("polish", [False, True])
@pytest.mark.parametrize("width", [1.0, 10.0])
@pytest.mark.parametrize("off", OFFSETS)
def test_offset_sweep_bound_is_valid(off, width, polish):
    """Bound never above the true minimum 0; a certified result meets it."""
    m = _shifted_square(off, width)
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30, polish=polish)
    n = 0
    assert r.bound is not None
    assert r.bound <= 1e-9, f"bound {r.bound!r} above the true minimum 0"
    n += 1
    if r.gap_certified:
        assert r.objective is not None and r.objective <= 1e-5, r.objective
        n += 1
    assert r.status in ("optimal", "feasible")
    assert n >= 1


def test_lowered_x_row_is_centred():
    """Structural: the ``_n2p*_x`` row's weight coefficients are breakpoint
    differences (<= box width), not breakpoint values (~1e5)."""
    t = dm.nonlinear_to_pwl(_shifted_square(1e5, 1.0))
    n = 0
    for con in t.model._constraints:
        name = str(getattr(con, "name", ""))
        if not re.fullmatch(r"_n2p\d+_x", name):
            continue
        s = str(con.body)
        coeffs = [abs(float(c)) for c in re.findall(r"(-?[0-9.]+(?:e[+-]?[0-9]+)?) \* ", s)]
        assert coeffs and max(coeffs) <= 1.0 + 1e-12, s
        n += 1
    assert n == 1


def test_infeasible_after_a_verified_point_trips(monkeypatch):
    """A relaxation that turns "infeasible" after an earlier round produced a
    point verified on the original cannot be published as a certified
    infeasibility (the tripwire's extreme case)."""
    m = dm.Model("q")
    x = m.continuous("x", lb=0.0, ub=3.0)
    m.minimize(dm.exp(x) - 3 * x)
    t = dm.nonlinear_to_pwl(m, segments=2)

    real_solve = Model.solve
    calls = {"n": 0}

    def solve(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_solve(self, *a, **k)
        return SolveResult(status="infeasible", gap_certified=True, wall_time=0.0)

    monkeypatch.setattr(Model, "solve", solve)
    with pytest.raises(AssertionError, match="reported infeasible in round 2"):
        # A gap tolerance nothing meets in one round forces a second round.
        t.solve(time_limit=30, max_rounds=3, gap_tolerance=1e-12, abs_gap_tolerance=1e-12)
    assert calls["n"] == 2


def test_value_offset_samples_are_centred_too():
    """A term whose *values* are large (x**2 at x ~ 1e5, values ~1e10): the ``w``
    row is centred on the first sample, and the bound stays valid."""
    m = dm.Model("sq")
    x = m.continuous("x", lb=1e5, ub=1e5 + 1)
    m.minimize(x**2 - 2e5 * x)  # = (x - 1e5)**2 - 1e10, min -1e10 at x = 1e5
    r = dm.nonlinear_to_pwl(m).solve(time_limit=30)
    true_min = -1e10
    assert r.bound is not None
    assert r.bound <= true_min + 1e-6 * abs(true_min)
    assert r.objective is not None and np.isfinite(r.objective)
