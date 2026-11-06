"""#2190: interval ``sin``/``cos`` must contain the true value after argument reduction.

``_sinusoid`` used to reduce the argument modulo the double ``2π`` (below the
real one by ~2.4e-16) and evaluate ``np.sin`` on the *reduced* argument; ``cos``
also added the rounded ``π/2``. Neither shift was enclosed, so the enclosure
missed the true value by ~``|x| * 1e-16`` -- ``sin(1e15)`` came back as the point
``0.850`` (true ``0.858``), and a solve whose ``sin`` constraint relied on it
certified objective 1 where 0 is optimal (and called the ``y``-free variant
infeasible).

Reference values were computed with mpmath at 50 digits. The randomized check
uses an exact-in-double reference: on an interval that holds no critical point
the image is ``[min, max]`` of the endpoint values, which ``math.sin`` computes
correctly rounded-or-faithful, so the enclosure's ends must bracket them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval, cos, sin

# (function, point, true value from mpmath at 50 digits)
_POINTS = [
    (sin, 6.283185307179586, -2.4492935982947064e-16),
    (cos, 1.5707963267948966, 6.123233995736766e-17),
    (sin, 1e15, 0.8582727931702359),
    (sin, 3.141592653589793, 1.2246467991473532e-16),
    (cos, 4.71238898038469, -1.8369701987210297e-16),
    (sin, -3138.4510609362032, -1.969492024896311e-13),
]


@pytest.mark.smoke
@pytest.mark.parametrize("fn,x,true", _POINTS)
def test_point_enclosure_contains_true_value(fn, x, true):
    out = fn(Interval.point(x))
    assert float(out.lo) <= true <= float(out.hi), (fn.__name__, x, out, true)
    # The fix must not degrade a small-argument point to [-1, 1].
    if abs(x) < 1e4:
        assert float(out.hi) - float(out.lo) < 1e-15


@pytest.mark.smoke
def test_random_intervals_bracket_endpoint_values_and_extrema():
    """Vectorized: every enclosure contains f(lo), f(hi) and every interior extremum."""
    rng = np.random.default_rng(2190)
    n = 4000
    mag = 10.0 ** rng.uniform(-3, 15, n)
    lo = rng.uniform(-1.0, 1.0, n) * mag
    hi = lo + 10.0 ** rng.uniform(-8, 0.8, n)
    executed = 0
    for fn, ref, c0 in ((sin, math.sin, math.pi / 2), (cos, math.cos, 0.0)):
        out = fn(Interval(lo, hi))
        assert np.all(out.lo >= -1.0) and np.all(out.hi <= 1.0)
        for i in range(n):
            a, b = float(lo[i]), float(hi[i])
            assert out.lo[i] <= min(ref(a), ref(b)) and max(ref(a), ref(b)) <= out.hi[i]
            # An extremum strictly inside (by a clear margin) must be reported.
            m = math.ceil((a - c0) / math.pi + 1e-6)
            p = c0 + m * math.pi
            if abs(p) < 1e6 and a + 1e-6 < p < b - 1e-6:
                want = 1.0 if m % 2 == 0 else -1.0
                assert (out.hi[i] if want > 0 else out.lo[i]) == want
            executed += 1
    assert executed == 2 * n


@pytest.mark.smoke
def test_one_ulp_intervals_at_multiples_of_half_pi():
    """Point and one-ulp boxes at every rounded ``k π/2``: the sign must be enclosed.

    ``sin(fl(kπ))`` is a tiny number whose sign the pre-fix code got wrong for many
    ``k``; ``math.sin`` of the double is faithful, so it is the reference here.
    """
    k = np.arange(-2000, 2001)
    c = k * (np.pi / 2)
    for lo, hi in ((c, c), (c, np.nextafter(c, np.inf)), (np.nextafter(c, -np.inf), c)):
        for fn, ref in ((sin, math.sin), (cos, math.cos)):
            out = fn(Interval(lo, hi))
            for i in range(len(k)):
                v = (ref(float(lo[i])), ref(float(hi[i])))
                assert out.lo[i] <= min(v) and max(v) <= out.hi[i], (fn.__name__, k[i])


def test_nonfinite_and_full_period_give_unit_box():
    for fn in (sin, cos):
        for lo, hi in ((-np.inf, 0.0), (0.0, np.inf), (0.0, 2 * np.pi), (1e17, 1e17)):
            out = fn(Interval(np.asarray(lo), np.asarray(hi)))
            assert float(out.lo) == -1.0 and float(out.hi) == 1.0, (fn.__name__, lo, hi)


def test_large_argument_solve_finds_true_optimum():
    """The issue's model: ``x = a, y = 0`` is feasible (slack 5.5e-5), so the optimum is 0."""
    import discopt.modeling as dm

    a = 1038850000000.0
    assert math.sin(a) <= 0.0971
    for with_y in (True, False):
        m = dm.Model("sin_large_arg_2190")
        x = m.continuous("x", lb=a, ub=a + 1.0)
        y = m.binary("y")
        m.minimize(y)
        m.subject_to(dm.sin(x) <= 0.0971 + (y if with_y else 0.0))
        r = m.solve(time_limit=60)
        assert r.status == "optimal", (with_y, r.status)
        assert r.objective == pytest.approx(0.0, abs=1e-6), (with_y, r.objective)
        assert r.bound <= r.objective + 1e-9
