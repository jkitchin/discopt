"""#1836: box sign tracking passed a direction through non-monotone nodes.

``sign_tracking_substitute`` propagated the accumulated sign unchanged through
``e / d`` with a non-constant divisor, ``**``, ``abs`` and every
``FunctionCall``, so ``x + 1/p <= 10`` substituted the *upper* bound of ``p``
(the best case of ``1/p``) and certified ``x = 9.33`` where the robust optimum
is ``x = 8``. A direction now passes only through nodes proven monotone; a
variable-free sub-expression below any other node is worst-cased by a sound
interval enclosure over the box, and one that mixes in a decision variable is
refused.

Every optimum is checked against a dense grid oracle over the box, and the
returned point is verified feasible on the grid.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, RobustCounterpart

pytestmark = pytest.mark.smoke

# (name, row(p, x) for ``row <= 10``, p_bar, delta, numeric twin of the p-part)
CASES = [
    ("reciprocal", lambda p, x: x + 1.0 / p, 1.0, 0.5, lambda p: 1.0 / p),
    ("negative_power", lambda p, x: x + p ** (-1.0), 1.0, 0.5, lambda p: p**-1.0),
    ("abs", lambda p, x: x + abs(p), -0.5, 1.5, np.abs),
    ("square", lambda p, x: x + p**2, -0.5, 1.5, lambda p: p**2),
    ("cos", lambda p, x: x + dm.cos(p), 0.0, 2.0, np.cos),
    ("product_pp", lambda p, x: x + p * p, -0.5, 1.5, lambda p: p * p),
    ("neg_reciprocal", lambda p, x: x - 1.0 / p, 1.0, 0.5, lambda p: -1.0 / p),
    ("acos_decreasing", lambda p, x: x + dm.acos(p), 0.0, 0.5, np.arccos),
    # Monotone nodes keep the direction (locked, not changed).
    ("exp", lambda p, x: x + dm.exp(p), 0.0, 1.0, np.exp),
    ("exp_neg", lambda p, x: x + dm.exp(-p), 0.0, 1.0, lambda p: np.exp(-p)),
    ("cube", lambda p, x: x + p**3, -0.5, 1.5, lambda p: p**3),
]


@pytest.mark.parametrize("name,row,pbar,delta,f", CASES, ids=[c[0] for c in CASES])
def test_box_counterpart_matches_grid_oracle(name, row, pbar, delta, f):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=pbar)
    m.subject_to(row(p, x) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    xv = float(r.value(x))

    grid = np.linspace(pbar - delta, pbar + delta, 20001)
    truth = 10.0 - float(np.max(f(grid)))
    assert xv == pytest.approx(truth, abs=1e-4)
    # The returned point is robust: feasible at every grid point.
    assert float(np.max(xv + f(grid))) <= 10.0 + 1e-5


def test_non_monotone_with_variable_solved_exactly():
    """``(x - p)**2`` is even in ``x - p``: worst-cased by its exact ``|x - p|``.

    ``max |x - p|`` over ``p in [0.5, 1.5]`` is ``max(x - 0.5, 1.5 - x)``, so
    the robust row is ``x <= 0.5 + sqrt(10)`` (and a slack lower bound). A
    sign-tracked substitution at either single end of ``p`` would be wrong on
    one side of ``x = 1``.
    """
    m = dm.Model("even_var")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=1.0)
    m.subject_to((x - p) ** 2 <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    r = m.solve()
    assert r.status in ("optimal", "feasible")
    xv = float(r.value(x))
    assert xv == pytest.approx(0.5 + np.sqrt(10.0), abs=1e-5)
    grid = np.linspace(0.5, 1.5, 2001)
    assert float(np.max((xv - grid) ** 2)) <= 10.0 + 1e-6


def test_divisor_through_zero_refused():
    """``1/p`` over a box containing 0 has no finite worst case."""
    m = dm.Model("zero")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=0.5)
    m.subject_to(x + 1.0 / p <= 10)
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="#1836"):
        RobustCounterpart(m, BoxUncertaintySet(p, delta=1.0)).formulate()


def test_variable_divisor_of_known_sign():
    """``p / y`` with ``y >= 1``: monotone in ``p``, worst case at p_max."""
    m = dm.Model("vdiv")
    y = m.continuous("y", lb=1, ub=4)
    p = m.parameter("p", value=2.0)
    m.subject_to(p / y <= 1.0)
    m.minimize(y)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=1.0)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(y)) == pytest.approx(3.0, abs=1e-5)


def test_variable_divisor_of_unknown_sign_refused():
    m = dm.Model("vdiv_bad")
    y = m.continuous("y", lb=-1, ub=4)
    p = m.parameter("p", value=2.0)
    m.subject_to(p / y <= 1.0)
    m.minimize(y)
    with pytest.raises(NotImplementedError, match="#1836"):
        RobustCounterpart(m, BoxUncertaintySet(p, delta=1.0)).formulate()
