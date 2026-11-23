"""#2313: rigorous lower-bound helpers must not step an exact 0 to a negative subnormal.

``_rowless_lp_safe_bound`` and ``separable_poly_bound``'s interval arithmetic
moved every intermediate one float toward ``-inf`` with ``nextafter``, also when
the result was exact, so an exact optimum of 0 came back as ``-2e-323`` and
reached ``SolveResult.bound`` (``-3e-323`` on MINLPLib ex8_1_4; the #957 pattern).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.milp_relaxation import MilpRelaxationModel, _rowless_lp_safe_bound
from discopt._relax.separable_poly_bound import _iadd, _imul, _univariate_min


@pytest.mark.unit
def test_rowless_bound_of_an_exact_zero_is_zero():
    box = [(0.0, 1.0), (0.0, 1.0)]
    assert _rowless_lp_safe_bound(np.array([1.0, 2.0]), box, 0.0) == 0.0
    r = MilpRelaxationModel(np.array([1.0, 2.0]), None, None, box).solve(time_limit=10)
    assert r.objective == 0.0
    assert r.safe_bound == 0.0


@pytest.mark.unit
def test_rowless_bound_is_the_largest_float_not_above_the_exact_minimum():
    rng = random.Random(2313)
    checked = 0
    for _ in range(2000):
        n = rng.randint(1, 6)
        c = [rng.uniform(-10, 10) * 10 ** rng.randint(-5, 5) for _ in range(n)]
        box = []
        for _ in range(n):
            a, b = sorted(rng.uniform(-10, 10) * 10 ** rng.randint(-5, 5) for _ in range(2))
            box.append((a, b))
        off = rng.uniform(-1, 1)
        got = _rowless_lp_safe_bound(np.array(c), box, off)
        exact = Fraction(off) + sum(
            Fraction(cj) * Fraction(lo if cj > 0 else hi) for cj, (lo, hi) in zip(c, box)
        )
        assert Fraction(got) <= exact
        assert Fraction(math.nextafter(got, math.inf)) > exact
        checked += 1
    assert checked == 2000


@pytest.mark.unit
def test_rowless_bound_keeps_the_sign_of_an_underflowing_negative_product():
    got = _rowless_lp_safe_bound(np.array([-(2.0**-800)]), [(0.0, 2.0**-300)], 0.0)
    assert got is not None and got < 0.0


@pytest.mark.unit
def test_rowless_bound_refuses_non_finite_inputs():
    assert _rowless_lp_safe_bound(np.array([1.0]), [(-math.inf, 1.0)], 0.0) is None
    assert _rowless_lp_safe_bound(np.array([1.0]), [(0.0, 1.0)], math.inf) is None


@pytest.mark.unit
def test_interval_ops_leave_exact_results_alone_and_round_inexact_ones_outward():
    assert _univariate_min([0.0, 0.0, 1.0], 0.0, 1.0) == 0.0
    assert _imul((0.0, 1.0), (0.0, 1.0)) == (0.0, 1.0)
    assert _iadd((0.0, 0.0), (0.0, 0.0)) == (0.0, 0.0)
    lo, hi = _imul((0.1, 0.1), (0.1, 0.1))
    exact = Fraction(0.1) * Fraction(0.1)
    assert Fraction(lo) <= exact <= Fraction(hi) and lo < hi
    lo, hi = _iadd((0.1, 0.1), (0.2, 0.2))
    exact = Fraction(0.1) + Fraction(0.2)
    assert Fraction(lo) <= exact <= Fraction(hi) and lo < hi
    assert _imul((0.0, 1.0), (1.0, math.inf)) == (-math.inf, math.inf)


@pytest.mark.correctness
def test_ex8_1_4_reports_a_zero_bound():
    """MINLPLib ex8_1_4 (optimum 0): 12x^2 - 6.3x^4 + x^6 - 6xy + 6y^2."""
    m = dm.Model("ex8_1_4")
    # Unbounded, as in the instance: that is the path that reaches the
    # separable-polynomial bound.
    x = m.continuous("x0", lb=-math.inf, ub=math.inf)
    y = m.continuous("x1", lb=-math.inf, ub=math.inf)
    m.minimize(12 * x**2 - 6.3 * x**4 + x**6 - 6 * x * y + 6 * y**2)
    with pytest.warns(UserWarning, match="large or infinite declared bounds"):
        r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal"
    assert r.bound is not None and r.bound <= r.objective + 1e-9
    # No negative subnormal: the bound is 0 or an ordinary-magnitude value.
    assert r.bound == 0.0 or abs(r.bound) > 1e-300
