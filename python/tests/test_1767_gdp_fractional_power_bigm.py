"""#1767: GDP big-M over a disjunct body with a non-integer power.

``_bound_expression`` handled only positive *integer* constant exponents, so any
``x**p`` with a fractional ``p`` -- ``(x + y)**1.5``, a six-tenths cost law
``a + b*V**0.6`` -- enclosed to ``(-inf, inf)`` and the big-M reformulation
refused the disjunct even on a finite nonnegative box.

The fix: for a constant non-integer ``p`` and a base enclosure ``[lo, hi]`` with
``lo >= 0``, ``x**p`` is monotone, so the enclosure is the endpoint powers
(``p < 0`` with ``lo == 0`` stays unbounded above). A base that can be negative,
or a non-constant exponent, still declines and the big-M still refuses loudly.
``bound_expression_error`` (#1397) gains the matching endpoint-error rule.

Soundness is checked the GDP-1 way: sampled points of the box never exceed the
enclosure / big-M, and a full solve reaches the hand-derived optimum.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import (
    _bound_expression,
    _compute_big_m,
    _fractional_power_slope,
    bound_expression_error,
)

pytestmark = pytest.mark.smoke


def _box():
    m = dm.Model("p1767")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.continuous("z", lb=0.0, ub=4.0)
    w = m.continuous("w", lb=-1.0, ub=4.0)
    return m, x, y, z, w


def test_fractional_power_enclosures_match_endpoint_powers():
    m, x, y, z, _w = _box()
    cases = [
        ((x + y) ** 1.5, (0.2**1.5, 8.0**1.5)),
        (x**1.5, (0.1**1.5, 4.0**1.5)),
        ((2 * x) ** 1.5, (0.2**1.5, 8.0**1.5)),
        ((x + y) ** 0.5, (0.2**0.5, 8.0**0.5)),
        (z**0.6, (0.0, 4.0**0.6)),
        (x**-0.5, (4.0**-0.5, 0.1**-0.5)),
    ]
    for expr, (lo, hi) in cases:
        got = _bound_expression(expr, m)
        assert got == pytest.approx((lo, hi), rel=1e-15), expr


def test_negative_exponent_at_zero_is_unbounded_above():
    m, _x, _y, z, _w = _box()
    lo, hi = _bound_expression(z**-0.5, m)
    assert lo == pytest.approx(4.0**-0.5)
    assert hi == np.inf


def test_unrecognised_powers_still_decline():
    m, x, y, _z, w = _box()
    # A base that can be negative: the real power is undefined there.
    for expr in (w**0.5, (x - y) ** 1.5, w**-0.5):
        assert _bound_expression(expr, m) == (-np.inf, np.inf), expr
        assert bound_expression_error(expr, m) == (np.inf, np.inf), expr
    # A non-constant exponent.
    assert _bound_expression(x**y, m) == (-np.inf, np.inf)


def test_enclosure_contains_sampled_values():
    """Differential sampling: every box point's value lies inside the enclosure."""
    m, x, y, z, _w = _box()
    exprs = [
        ((x + y) ** 1.5, lambda a, b, c: (a + b) ** 1.5),
        (x**0.6 + 3.0 * y**0.6, lambda a, b, c: a**0.6 + 3.0 * b**0.6),
        (z**0.25 * x, lambda a, b, c: c**0.25 * a),
        (x**-1.5, lambda a, b, c: a**-1.5),
        (2.0 + 5.0 * (x * y) ** 0.6, lambda a, b, c: 2.0 + 5.0 * (a * b) ** 0.6),
    ]
    rng = np.random.default_rng(1767)
    pts = rng.uniform([0.1, 0.1, 0.0], [4.0, 4.0, 4.0], size=(2000, 3))
    # Box corners too: monotone enclosures are attained there.
    corners = np.array([[a, b, c] for a in (0.1, 4.0) for b in (0.1, 4.0) for c in (0.0, 4.0)])
    pts = np.vstack([pts, corners])
    checked = 0
    for expr, f in exprs:
        lo, hi = _bound_expression(expr, m)
        assert np.isfinite(lo) and np.isfinite(hi), expr
        for a, b, c in pts:
            v = f(a, b, c)
            tol = 1e-12 * max(1.0, abs(v))
            assert lo - tol <= v <= hi + tol, (expr, a, b, c, v, lo, hi)
            checked += 1
    assert checked == len(exprs) * len(pts)


def test_bound_error_finite_and_propagates():
    m, x, y, z, _w = _box()
    # Exact base endpoints: only local round-off, so a few ulps of each endpoint.
    e_lo, e_hi = bound_expression_error(x**1.5, m)
    assert 0.0 < e_lo < 1e-14 and 0.0 < e_hi < 1e-13
    # A sum base carries error into the power, scaled by the slope.
    e_lo, e_hi = bound_expression_error((x + y) ** 0.5, m)
    assert np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo > 0.0
    # Base lower endpoint exactly 0 with no error: still finite.
    assert np.all(np.isfinite(bound_expression_error(z**0.6, m)))
    # A base error ball reaching 0 needs an unbounded slope (p < 1): refused.
    assert _fractional_power_slope(1e-3, 2e-3, 0.6) == np.inf
    assert _fractional_power_slope(0.0, 0.0, 0.6) == 0.0
    # Mean value theorem: the slope is the ball's worst end, not the endpoint's.
    assert _fractional_power_slope(1.0, 0.5, 0.5) == pytest.approx(0.5 * 0.5**-0.5)
    assert _fractional_power_slope(1.0, 0.5, 1.5) == pytest.approx(1.5 * 1.5**0.5)
    # Unbounded upper endpoint from 0**(-p): error is inf there, finite below.
    e_lo, e_hi = bound_expression_error(z**-0.5, m)
    assert np.isfinite(e_lo) and e_hi == np.inf


def test_big_m_over_estimates_sampled_bodies():
    """A big-M built from the new enclosure never cuts a box point."""
    m, x, y, _z, _w = _box()
    V = m.continuous("V", lb=0.0, ub=30.0)
    cons = [
        (V >= (x + y) ** 1.5, lambda a, b, v: (a + b) ** 1.5 - v),
        (V <= 1.0 + 2.0 * x**0.6, lambda a, b, v: v - (1.0 + 2.0 * a**0.6)),
        (y**0.5 + x**1.5 <= 2.0, lambda a, b, v: b**0.5 + a**1.5 - 2.0),
    ]
    rng = np.random.default_rng(17670)
    pts = rng.uniform([0.1, 0.1, 0.0], [4.0, 4.0, 30.0], size=(3000, 3))
    checked = 0
    for con, viol in cons:
        M = _compute_big_m(con, m)
        assert np.isfinite(M)
        for a, b, v in pts:
            # Relaxed row is ``viol <= M``: it must hold everywhere in the box.
            assert viol(a, b, v) <= M, (con, a, b, v, M)
            checked += 1
    assert checked == 3 * len(pts)


def test_either_or_with_fractional_power_solves_to_hand_optimum():
    """The issue's repro: ``[V >= (x + y)**1.5, ...]`` no longer raises.

    Disjunct A forces ``x + y >= 4``, so ``V >= 4**1.5 = 8``; disjunct B forces
    ``V >= 9``. The optimum is 8 via A.
    """
    m = dm.Model("p1767_solve")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    V = m.continuous("V", lb=0.0, ub=30.0)
    m.minimize(V)
    m.either_or([[V >= (x + y) ** 1.5, x + y >= 4.0], [V >= 9.0]])
    r = m.solve(gdp_method="big-m")
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(8.0, abs=1e-4)


def test_six_tenths_cost_law_inside_disjunct():
    """``a + b*V**0.6`` costs inside the disjuncts (process-synthesis idiom).

    Unit A: cost 10 + 2*V**0.6, unit B: cost 4 + 5*V**0.6, demand V >= 8.
    At V = 8, 8**0.6 = 3.482: A costs 16.96, B costs 21.41, so A wins.
    """
    m = dm.Model("p1767_cost")
    V = m.continuous("V", lb=0.0, ub=20.0)
    C = m.continuous("C", lb=0.0, ub=200.0)
    m.minimize(C)
    m.subject_to(V >= 8.0)
    m.either_or([[C >= 10.0 + 2.0 * V**0.6], [C >= 4.0 + 5.0 * V**0.6]])
    r = m.solve(gdp_method="big-m")
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(10.0 + 2.0 * 8.0**0.6, abs=1e-4)
