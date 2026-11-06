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


# ── Same bug class at the other trig sites (audit for #2190) ─────────────────
#
# Each site compared against a rounded multiple of pi with no margin. The
# references below need no extended precision: near an inflection or pole
# ``math.tan``/``math.cos`` of a double gets the *sign* right (libm reduces the
# argument accurately), and the period check is exact rational arithmetic.


def test_curv_tan_verdict_matches_sign_of_tan_near_k_pi():
    """``tan''`` has the sign of ``tan``; a verdict must agree with the true sign.

    The old ``_curv_tan`` compared ``hi`` with the rounded ``k*pi``, so a box ending
    an ulp or two past the true inflection was called ``concave`` (e.g.
    ``[-942.477796076938, -942.4777960769379]``, where ``tan(hi) = +5.1e-14``).
    """
    from discopt._relax.uniform_relax import _curv_tan

    executed = 0
    for k in range(-2000, 2001):
        c = k * np.pi
        for lo, hi in (
            (np.nextafter(c, -np.inf), c),
            (c, np.nextafter(c, np.inf)),
            (c, c),
            (c - 1e-3, c),
            (c, c + 1e-3),
        ):
            lo, hi = float(lo), float(hi)
            verdict = _curv_tan(lo, hi)
            # Within one branch tan is increasing, so its range is [tan(lo), tan(hi)].
            if verdict == "convex":
                assert math.tan(lo) >= 0.0, (k, lo, hi)
            elif verdict == "concave":
                assert math.tan(hi) <= 0.0, (k, lo, hi)
            executed += 1
    assert executed == 5 * 4001
    # Verdicts are kept away from the inflection: the fix is not a blanket abstain.
    assert _curv_tan(0.1, 1.0) == "convex" and _curv_tan(-1.0, -0.1) == "concave"


def test_periodic_window_spans_a_true_period():
    """``PeriodicVariableBoundRule`` claims a lossless one-period window.

    ``lo + fl(2*pi)`` is short of ``2*pi`` by 2.4e-16 at ``lo = 0`` and by up to
    ``ulp(lo)/2`` (6e-5 at 1e12) after rounding; ``[-fl(pi), fl(pi)]`` is short too.
    """
    from fractions import Fraction

    import discopt._relax.nonlinear_bound_tightening as nbt
    from discopt._relax.nonlinear_bound_tightening import (
        PeriodicVariableBoundRule,
        build_flat_variable_metadata,
    )

    two_pi_up = Fraction("6.28318530717958647692528676656")  # > 2*pi
    for lo in (0.0, 1.0, -7.5, 1e6, 1e9, 1038850000000.0, -1e12, 1e14):
        hi = nbt._period_end_up(lo)
        assert Fraction(hi) - Fraction(lo) >= two_pi_up, lo
        assert hi - lo < 2 * np.pi + 1e-12 * max(1.0, abs(lo)) + 1e-12  # still ~one period
        start = nbt._period_start_down(lo)
        assert Fraction(lo) - Fraction(start) >= two_pi_up, lo

    rule = PeriodicVariableBoundRule()
    m = dm_model_cos_only()
    md = build_flat_variable_metadata(m)
    for lb, ub in (([-1e20], [1e20]), ([1e12], [1e20]), ([-1e20], [-3.0]), ([0.0], [100.0])):
        nlb, nub = rule.tighten(m, np.array(lb, float), np.array(ub, float), md)
        assert Fraction(float(nub[0])) - Fraction(float(nlb[0])) >= two_pi_up, (lb, ub)
        assert nlb[0] >= lb[0] and nub[0] <= ub[0]  # never leaves the incoming box


def dm_model_cos_only():
    import discopt.modeling as dm

    m = dm.Model("periodic_2190")
    y = m.continuous("y")
    m.minimize(dm.cos(y))
    return m


def test_operator_relaxations_trig_range_is_an_enclosure():
    """AMP's ``trig_range`` returned the point ``(1.0, 1.0)`` for ``cos(fl(-300*pi))``.

    ``fl(-300*pi)`` is not a multiple of ``2*pi`` (pi is irrational), so the true
    ``cos`` is strictly below 1 and a sound enclosure must reach below 1.
    """
    from discopt._relax.operator_relaxations import trig_range

    c = -300 * np.pi
    lo, hi = trig_range("cos", c, c)
    assert lo < 1.0 <= hi
    # Large arguments: contains the endpoint values (libm-accurate).
    a = 1038850000000.0
    lo, hi = trig_range("sin", a, a + 1.0)
    assert lo <= math.sin(a) and max(math.sin(a), math.sin(a + 1.0)) <= hi
    # tan: finite only on one branch, and outward-rounded.
    lo, hi = trig_range("tan", 0.1, 1.0)
    assert lo <= math.tan(0.1) and math.tan(1.0) <= hi


@pytest.mark.parametrize(
    "lo,hi",
    [(-963176972.5272598, -963176972.5272593), (1179630105.120879, 1179630105.12088)],
)
def test_scip_tan_pole_test_sees_pole_near_large_k(lo, hi):
    """``tan`` changes sign from + to - across these boxes, so a pole lies inside.

    The old test compared ``hi`` with the rounded ``pi/2 + k*pi`` (off by ~0.1 ulp
    per unit ``k``) and reported no pole, skipping the singular-site guard.
    """
    import discopt.modeling as dm
    from discopt.solvers.scip import _tan_pole_possible

    assert math.tan(lo) > 0.0 > math.tan(hi) and hi - lo < 1.0  # a pole is inside
    m = dm.Model("tan_pole_2190")
    x = m.continuous("x", lb=lo, ub=hi)
    assert _tan_pole_possible(x)
