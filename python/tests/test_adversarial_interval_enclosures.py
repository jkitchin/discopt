"""Enclosure misses and wrong solves found by mpmath-referenced fuzzing.

Each case was a true miss against a 50-digit reference (not a loose-but-valid
enclosure), and most reached ``Model.solve()`` as a false ``infeasible`` or a
wrong certified objective:

* Rust ``log``/``log2``/``log10`` clamped the lower end to ``f64::MIN_POSITIVE``,
  so ``log(x) <= -709`` (feasible at ``x = e^-709``) was declared infeasible.
* Rust ``f64::atanh`` (FBBT, the expression evaluator) and POUNCE's ``atanh``
  opcode cancel near ``-1``: ``min atanh(x)`` certified ``-15.31267`` against
  the true ``-15.31322``.
* A negative base raised to an integer *variable* exponent had its base clamped
  to ``[0, 0]``: ``x**n >= 3`` with ``x in [-2, -1]``, ``n in {2, 3}`` was
  declared infeasible.
* An exponent ``>= 2^63`` saturated ``as i64`` (segfault on ``-(2^63)``), and the
  Python ``Interval.__pow__`` looped ``n`` times (``x**1e19`` hung past
  ``time_limit``).
* The Python interval atoms widened library results by one ulp; scipy's ``erf``,
  ``np.tan``, ``expit`` and others err by up to ~2.5 ulps, and ``expit``
  underflows in the subnormal range.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import interval as I

mp = pytest.importorskip("mpmath")
mp.mp.dps = 50


def _solve(m):
    return m.solve(time_limit=60)


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("rhs", [-709.0, -720.0])
def test_log_below_min_positive_is_feasible(rhs):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(dm.log(x) <= rhs)
    m.minimize(-x)
    r = _solve(m)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-math.exp(rhs), rel=1e-6, abs=0)


def test_min_atanh_near_minus_one_objective():
    m = dm.Model()
    x = m.continuous("x", lb=-1 + 1e-13, ub=0)
    m.minimize(dm.atanh(x))
    r = _solve(m)
    truth = float(mp.atanh(mp.mpf(-1 + 1e-13)))  # -15.313221246314329
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-9)
    assert r.bound <= truth + 1e-9


def test_negative_base_integer_variable_exponent():
    m = dm.Model()
    x = m.continuous("x", lb=-2, ub=-1)
    k = m.integer("k", lb=2, ub=3)
    m.subject_to(x**k >= 3)
    m.minimize(x)
    r = _solve(m)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-2.0, abs=1e-6)


def test_huge_even_exponent_solves_without_hang_or_overflow():
    m = dm.Model()
    x = m.continuous("x", lb=-1.0, ub=-0.5)
    m.subject_to(x**1e19 >= 0.5)  # (-1)**1e19 = 1; feasible only at x = -1
    m.minimize(x)
    r = _solve(m)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-1.0, abs=1e-6)


def test_tape_atanh_matches_reference():
    from discopt._tape_nlp_evaluator import build_evaluator

    n = 0
    for xv in [-1 + 1e-13, -0.999, -0.75, -0.5, -0.3, -1e-10, 0.0, 1e-10, 0.3, 0.7, 1 - 1e-13]:
        m = dm.Model()
        x = m.continuous("x", lb=-1 + 1e-16, ub=1 - 1e-16)
        m.minimize(dm.atanh(x))
        ev = build_evaluator(m, None)
        x0 = np.array([xv])
        f = ev.evaluate_objective(x0)
        g = ev.evaluate_gradient(x0)[0]
        tf = float(mp.atanh(mp.mpf(xv)))
        tg = float(1 / (1 - mp.mpf(xv) ** 2))
        assert f == pytest.approx(tf, rel=1e-14, abs=1e-300), xv
        assert g == pytest.approx(tg, rel=1e-14), xv
        n += 1
    assert n == 11


# ---------------------------------------------------------- Python intervals


def test_interval_pow_closed_form_is_fast_and_sound():
    x = I.Interval.from_bounds(-1.0, -0.5)
    r = x ** (10**19)  # used to loop 1e19 times
    assert float(r.lo) >= 0.0 and float(r.hi) >= 1.0
    # Past the loop cap: even powers over a box straddling zero are nonnegative
    # and enclose the endpoints' powers; odd powers enclose both ends.
    r = I.Interval.from_bounds(-1.0, 1.1) ** 66
    assert float(r.lo) == 0.0 and mp.mpf(float(r.hi)) >= mp.mpf(1.1) ** 66
    r = I.Interval.from_bounds(-1.1, 1.0) ** 65
    assert mp.mpf(float(r.lo)) <= -(mp.mpf(1.1) ** 65) and float(r.hi) >= 1.0


def test_interval_pow_small_n_is_unchanged_repeated_multiplication():
    # The incremental engine's monomial aux bounds reproduce this loop and its
    # build is gated on parity with it (a tighter closed form broke that parity
    # and silently disabled the engine on Rosenbrock).
    x = I.Interval.from_bounds(-10.0, 5.0)
    loop = x
    for _ in range(3):
        loop = loop * x
    r = x**4
    assert float(r.lo) == float(loop.lo) and float(r.hi) == float(loop.hi)


_LIB_CASES = [
    ("tanh", I.tanh, mp.tanh, -0.5103882830007409),
    ("sigmoid", I.sigmoid, lambda t: 1 / (1 + mp.exp(-t)), -711.0),
    ("tan", I.tan, mp.tan, -709.7),
    ("tan", I.tan, mp.tan, -37.00000000000001),
]


@pytest.mark.parametrize("name,fn,ref,x", _LIB_CASES, ids=[f"{c[0]}({c[3]})" for c in _LIB_CASES])
def test_library_atoms_enclose_reference(name, fn, ref, x):
    r = fn(I.Interval.point(x))
    truth = ref(mp.mpf(x))
    assert mp.mpf(float(r.lo)) <= truth <= mp.mpf(float(r.hi)), (name, x, r, truth)


def test_library_atoms_random_sweep():
    rng = np.random.default_rng(0)
    pairs = [
        (I.tanh, mp.tanh),
        (I.asinh, mp.asinh),
        (I.erf, mp.erf),
        (I.sigmoid, lambda t: 1 / (1 + mp.exp(-t))),
        (I.tan, mp.tan),
        (I.cosh, mp.cosh),
        (I.atan, mp.atan),
    ]
    pts = np.concatenate([rng.uniform(-3, 3, 300), rng.uniform(-40, 40, 100)])
    n = 0
    for fn, ref in pairs:
        for x in pts:
            r = fn(I.Interval.point(float(x)))
            t = ref(mp.mpf(float(x)))
            assert mp.mpf(float(r.lo)) <= t <= mp.mpf(float(r.hi)), (fn.__name__, x)
            n += 1
    assert n == len(pairs) * len(pts)
