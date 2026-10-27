"""#2012: an affine denominator is bounded by the range its linear rows imply
(``DISCOPT_IMPLIED_DENOMINATOR``), not by its box interval alone.

``x / (a - b)`` with ``a, b`` in ``[0, 5]`` has a pole inside the box, so the
relaxation had no objective bound and the tree stopped at the root; a linear row
``a - b >= 1`` keeps every feasible point away from it. These tests pin:

* the regression: with the row the solve certifies the optimum; without the flag it
  ends with no bound, as before;
* the pole refusal still fires when the rows do NOT exclude zero;
* ``_LinearRows.safe_range`` encloses the true LP range on random LPs.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.poles import _LinearRows
from scipy.optimize import linprog

OPT = 2.0 * math.sqrt(0.1)  # 1/a + 0.1 a at a = sqrt(10), b = 0, x = 1


def _model(row_lo: float):
    """min x/(a - b) + 0.1 a  s.t.  a - b >= row_lo,  a, b in [0, 5], x in [1, 2]."""
    m = dm.Model("implied_denominator")
    a = m.continuous("a", lb=0.0, ub=5.0)
    b = m.continuous("b", lb=0.0, ub=5.0)
    x = m.continuous("x", lb=1.0, ub=2.0)
    m.subject_to(a - b >= row_lo)
    m.minimize(x / (a - b) + 0.1 * a)
    return m


@pytest.mark.smoke
def test_a_row_that_excludes_the_pole_gives_a_bound(monkeypatch):
    monkeypatch.setenv("DISCOPT_IMPLIED_DENOMINATOR", "0")
    off = _model(1.0).solve(time_limit=20)
    assert off.bound is None or not np.isfinite(off.bound), off.bound
    monkeypatch.setenv("DISCOPT_IMPLIED_DENOMINATOR", "1")
    on = _model(1.0).solve(time_limit=20)
    assert on.status == "optimal" and on.gap_certified, on
    assert on.objective == pytest.approx(OPT, abs=1e-4)
    assert on.bound <= OPT + 1e-6


@pytest.mark.smoke
def test_default_on_with_an_opt_out(monkeypatch):
    """Graduated on its section 5 panel (performance-plan section 101)."""
    from discopt._relax.uniform_relax import _implied_denominator_enabled

    monkeypatch.delenv("DISCOPT_IMPLIED_DENOMINATOR", raising=False)
    assert _implied_denominator_enabled()
    monkeypatch.setenv("DISCOPT_IMPLIED_DENOMINATOR", "0")
    assert not _implied_denominator_enabled()


def test_the_refusal_still_fires_when_the_rows_admit_the_pole(monkeypatch):
    """``a - b >= -1`` leaves 0 in the denominator's feasible range: the objective
    is unbounded below near it, so no bound may be reported."""
    monkeypatch.setenv("DISCOPT_IMPLIED_DENOMINATOR", "1")
    r = _model(-1.0).solve(time_limit=10)
    assert not r.gap_certified
    assert r.bound is None or not np.isfinite(r.bound), r.bound


def test_safe_range_encloses_the_lp_range():
    rng = np.random.default_rng(2012)
    executed = 0
    for _ in range(40):
        n = int(rng.integers(2, 6))
        m = dm.Model("lp")
        xs = [m.continuous(f"x{i}", lb=-3.0, ub=float(rng.uniform(1, 4))) for i in range(n)]
        for _r in range(int(rng.integers(1, 4))):
            coef = rng.normal(size=n)
            m.subject_to(sum(float(c) * v for c, v in zip(coef, xs)) <= float(rng.uniform(0, 2)))
        if rng.random() < 0.5:
            coef = rng.normal(size=n)
            m.subject_to(sum(float(c) * v for c, v in zip(coef, xs)) == 0.0)
        m.minimize(xs[0])
        rows = _LinearRows(m)
        terms = {i: float(c) for i, c in enumerate(rng.normal(size=n))}
        lo, hi = rows.safe_range(terms, 0.5)
        c = np.array([terms[i] for i in range(n)])
        bnds = list(zip(rows.lb, rows.ub))
        kw = dict(A_ub=rows.A_ub, b_ub=rows.b_ub, A_eq=rows.A_eq, b_eq=rows.b_eq)
        kw = {k: v for k, v in kw.items() if v is not None and getattr(v, "size", 1)}
        mn = linprog(c, bounds=bnds, method="highs", **kw)
        mx = linprog(-c, bounds=bnds, method="highs", **kw)
        if mn.status != 0:
            continue
        assert lo <= mn.fun + 0.5 + 1e-7, (lo, mn.fun + 0.5)
        assert hi >= -mx.fun + 0.5 - 1e-7, (hi, -mx.fun + 0.5)
        assert lo >= mn.fun + 0.5 - 1e-5 * (1 + abs(mn.fun)), "dual bound far from LP value"
        executed += 1
    assert executed >= 20, executed
