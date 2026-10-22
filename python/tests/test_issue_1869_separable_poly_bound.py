"""Issue #1869 (c): a rigorous bound for a polynomial objective over an unbounded box.

``ex4_1_5`` (``2x^2 - 1.05x^4 + x^6/6 - xy + y^2`` over ``x >= -5, y <= 5``, optimum 0):
interval arithmetic on ``x^6/6 - 1.05x^4`` over ``[P, inf)`` is ``inf - inf``, so the
outer pieces of an unbounded split carried no bound and the tree never closed. The
separable minorant (AM-GM on ``-xy``, each univariate part minimised rigorously inside
its derivative's Cauchy radius) bounds them.

Every claimed bound is checked against dense sampling of the same box.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import separable_poly_bound as S
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_SEPARABLE_POLY_BOUND"
DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
INF = math.inf


def _sample_box(lo, hi, span=100.0):
    if math.isfinite(lo) and math.isfinite(hi):
        return lo, hi
    if math.isfinite(lo):
        return lo, lo + span
    if math.isfinite(hi):
        return hi - span, hi
    return -span / 2, span / 2


def test_univariate_min_is_a_lower_bound_on_random_polynomials():
    rng = np.random.default_rng(1869)
    checked = 0
    for _ in range(400):
        deg = int(rng.integers(1, 7))
        c = list(rng.normal(size=deg + 1))
        if deg % 2 == 0:
            c[-1] = abs(c[-1]) + 0.1  # coercive on both sides
        lo = float(rng.choice([-INF, rng.uniform(-5, 2)]))
        hi = float(rng.choice([INF, rng.uniform(2, 6)]))
        got = S._univariate_min(list(c), lo, hi)
        a, b = _sample_box(lo, hi, span=40.0)
        xs = np.linspace(a, b, 20001)
        vals = np.polyval(c[::-1], xs)
        if not math.isfinite(got):
            continue  # abstaining is always sound
        assert got <= vals.min() + 1e-9 * (1 + abs(vals.min())), (c, lo, hi, got, vals.min())
        checked += 1
    assert checked > 150, checked


def test_unbounded_side_toward_minus_infinity_abstains():
    assert S._univariate_min([0.0, 0.0, 0.0, 1.0], -INF, 0.0) == -INF  # x^3 on (-inf, 0]
    assert S._univariate_min([0.0, 0.0, -1.0], 0.0, INF) == -INF  # -x^2 on [0, inf)
    assert S._univariate_min([0.0, 0.0, 0.0, 1.0], 0.0, INF) >= -1e-12  # x^3 on [0, inf)


@pytest.mark.parametrize(
    "lb,ub",
    [
        ([-5, -INF], [INF, 5]),
        ([1e4, -INF], [INF, 5]),
        ([-5, -INF], [INF, -1e4]),
        ([1.5, -1], [2, 1]),
        ([-INF, -INF], [INF, INF]),
    ],
)
def test_ex4_1_5_bound_never_exceeds_the_sampled_minimum(lb, ub):
    m = dm.from_nl(os.path.join(DATA, "ex4_1_5.nl"))
    b = S.separable_poly_lower_bound(m, np.array(lb, float), np.array(ub, float), False)
    (x0, x1), (y0, y1) = _sample_box(lb[0], ub[0]), _sample_box(lb[1], ub[1])
    X, Y = np.meshgrid(np.linspace(x0, x1, 601), np.linspace(y0, y1, 601))
    F = 2 * X**2 - 1.05 * X**4 + X**6 / 6 - X * Y + Y**2
    assert b <= F.min() + 1e-9 * (1 + abs(F.min())), (b, F.min())


def test_outer_pieces_get_a_pruning_bound():
    m = dm.from_nl(os.path.join(DATA, "ex4_1_5.nl"))
    far = S.separable_poly_lower_bound(m, np.array([1e4, -INF]), np.array([INF, 5.0]), False)
    assert far > 1e20  # x^6/6 dominates; prunes against the incumbent 0


def test_maximisation_bounds_the_negated_objective():
    """For ``max f`` the internal objective is ``-f``; its bound must be <= -f."""
    m = Model("maxpoly")
    x = m.continuous("x", lb=-3, ub=None)
    y = m.continuous("y", lb=None, ub=2)
    m.maximize(-(x**4) + 2 * x * y - y**2)
    b = S.separable_poly_lower_bound(m, np.array([-3.0, -INF]), np.array([INF, 2.0]), True)
    X, Y = np.meshgrid(np.linspace(-3, 50, 801), np.linspace(-50, 2, 801))
    negf = X**4 - 2 * X * Y + Y**2
    assert math.isfinite(b) and b <= negf.min() + 1e-9


def test_a_non_polynomial_or_higher_cross_term_abstains():
    m = Model("cubiccross")
    x = m.continuous("x", lb=0, ub=None)
    y = m.continuous("y", lb=0, ub=None)
    m.minimize(x**2 * y + y**2)  # x^2 y: multivariate beyond bilinear
    assert S.separable_poly_lower_bound(m, np.zeros(2), np.full(2, INF), False) == -INF


def test_flag_on_certifies_ex4_1_5(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    r = dm.from_nl(os.path.join(DATA, "ex4_1_5.nl")).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound, r.node_count)
    assert abs(r.objective) <= 1e-6 and r.bound <= 1e-6


def test_flag_off_leaves_ex4_1_5_without_a_bound(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    r = dm.from_nl(os.path.join(DATA, "ex4_1_5.nl")).solve(time_limit=5)
    assert not r.gap_certified and r.bound is None


def test_flag_default_is_off_until_its_panel(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not S.separable_poly_bound_enabled()
