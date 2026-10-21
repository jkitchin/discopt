"""#1855: the big-M interval enclosure answered ``(-inf, inf)`` for ``dm.sum`` and
``@`` nodes, so ``dm.sum((x - c)**2) <= 4`` on a bounded vector ``x`` was refused
("interval enclosure is unbounded") while the same row written as a scalar sum was
accepted. The GDP enclosure now bounds both reductions.

Soundness is checked by sampling: every sampled value of a random reduction lies
inside its enclosure. The other callers of ``_bound_expression`` (factorable
reformulation, OBBT) keep the old ``(-inf, inf)`` -- the rules are opt-in.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import (
    _bound_expression,
    _BoundMemo,
    _compute_big_m,
    _gdp_body_bounds,
)
from discopt.modeling.core import Constant

pytestmark = pytest.mark.smoke


def _solve(m, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(gdp_method=method, time_limit=60)


@pytest.mark.parametrize("method", ["big-m", "mbigm"])
@pytest.mark.parametrize(
    "form, expected",
    [
        # Ball at (1, 2), radius 2, or x0 >= 2.5; min x0 + x1 over [-3, 3]^2:
        # the second disjunct wins at (2.5, -3), -0.5.
        ("shifted", -0.5),
        # Ball at the origin, radius 2: -2*sqrt(2) beats -0.5.
        ("origin", -2.0 * np.sqrt(2.0)),
    ],
)
def test_vector_sum_rows_are_lowered(method, form, expected):
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=-3, ub=3)
    if form == "shifted":
        row = dm.sum((x - np.array([1.0, 2.0])) ** 2) <= 4
    else:
        row = dm.sum(x**2) <= 4
    m.either_or([[row], [x[0] >= 2.5]])
    m.minimize(x[0] + x[1])
    r = _solve(m, method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(expected, abs=1e-5)


def test_big_m_value_is_finite_and_valid():
    m = dm.Model()
    x = m.continuous("x", shape=(2,), lb=0, ub=3)
    row = dm.sum((x - np.array([1.0, 2.0])) ** 2) <= 4
    M = _compute_big_m(row, m)
    # max over the box of sum((x-c)^2) - 4 is (3-1)^2 + (0-2)^2 - 4 = 4.
    assert np.isfinite(M) and M >= 4.0


def test_other_callers_unchanged():
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=-1, ub=2)
    assert _bound_expression(dm.sum(x), m) == (-np.inf, np.inf)
    assert _gdp_body_bounds(dm.sum(x), m) == (-3.0, 6.0)


def _enclosure_contains(expr, m, sample):
    lo, hi = _bound_expression(expr, m, _BoundMemo(array_reductions=True))
    vals = np.ravel(sample)
    return lo <= vals.min() + 1e-9 and vals.max() - 1e-9 <= hi


def test_sampled_soundness_of_reduction_enclosures():
    rng = np.random.default_rng(1855)
    checked = 0
    for _ in range(40):
        k, mm, n = (int(v) for v in rng.integers(1, 4, size=3))
        m = dm.Model()
        lb = rng.uniform(-3, 0, size=(k, n))
        ub = lb + rng.uniform(0, 3, size=(k, n))
        X = m.continuous("X", shape=(k, n), lb=lb, ub=ub)
        y = m.continuous("y", shape=(k,), lb=lb[:, 0], ub=ub[:, 0])
        A = rng.normal(size=(mm, k))
        c = rng.normal(size=k)
        exprs = {
            "A@X": (Constant(A) @ X, lambda Xv, yv: A @ Xv),
            "c@X": (Constant(c) @ X, lambda Xv, yv: c @ Xv),
            "y@X": (y @ X, lambda Xv, yv: yv @ Xv),
            "X.T-free y@y": (y @ y, lambda Xv, yv: yv @ yv),
            "sum(A@X)": (dm.sum(Constant(A) @ X), lambda Xv, yv: np.sum(A @ Xv)),
            "sum0(X**2)": (dm.sum(X**2, axis=0), lambda Xv, yv: np.sum(Xv**2, axis=0)),
        }
        for _name, (expr, fn) in exprs.items():
            for _s in range(20):
                Xv = rng.uniform(lb, ub)
                yv = rng.uniform(lb[:, 0], ub[:, 0])
                assert _enclosure_contains(expr, m, fn(Xv, yv)), _name
                checked += 1
    assert checked == 40 * 6 * 20
