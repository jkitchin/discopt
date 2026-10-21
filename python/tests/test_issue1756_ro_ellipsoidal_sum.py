"""#1756: the ellipsoidal counterpart accepts ``dm.sum(a * x)`` as well as ``a @ x``.

Both spell the same affine row; only the matmul form was recognised, so the sum
form left ``a`` in place and the RO-2 guard raised ``NotImplementedError``.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import EllipsoidalUncertaintySet, RobustCounterpart

pytestmark = pytest.mark.smoke

A_HAT = np.array([2.0, 3.0, 4.0])
RHO = 0.5


def _row_model(form):
    m = dm.Model("e")
    x = m.continuous("x", shape=3, lb=0, ub=10)
    a = m.parameter("a", value=A_HAT)
    row = {
        "matmul": lambda: a @ x,
        "sum": lambda: dm.sum(a * x),
        "sum_rev": lambda: dm.sum(x * a),
    }[form]()
    m.subject_to(row >= 10)
    m.minimize(dm.sum(x))
    RobustCounterpart(m, EllipsoidalUncertaintySet(a, rho=RHO)).formulate()
    r = m.solve()
    return r, r.value(x)


@pytest.mark.parametrize("form", ["sum", "sum_rev"])
def test_sum_form_matches_matmul(form):
    ref, _ = _row_model("matmul")
    r, xv = _row_model(form)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, rel=1e-6)
    # The SOC counterpart a_hat @ x - rho * ||x|| >= 10 holds at the solution.
    assert A_HAT @ xv - RHO * np.linalg.norm(xv) >= 10 - 1e-5


def test_sum_form_in_objective():
    m = dm.Model("o")
    x = m.continuous("x", shape=3, lb=0, ub=10)
    a = m.parameter("a", value=A_HAT)
    m.subject_to(dm.sum(x) <= 3)
    m.maximize(dm.sum(a * x))
    RobustCounterpart(m, EllipsoidalUncertaintySet(a, rho=RHO)).formulate()
    r = m.solve()
    # Worst case of a @ x over the ball at x = (0, 0, 3): 12 - 0.5 * 3.
    assert r.objective == pytest.approx(10.5, abs=1e-5)


def test_scalar_times_vector_sum_is_robustified():
    # sum(a * s) for a scalar s is (1^T a) s: coefficient of a is s * 1 (#1835),
    # so the row is 9 s - RHO * sqrt(3) s >= 10. It used to be refused.
    m = dm.Model("s")
    s = m.continuous("s", lb=0, ub=10)
    a = m.parameter("a", value=A_HAT)
    m.subject_to(dm.sum(a * s) >= 10)
    m.minimize(s)
    RobustCounterpart(m, EllipsoidalUncertaintySet(a, rho=RHO)).formulate()
    expected = 10.0 / (A_HAT.sum() - RHO * np.sqrt(3.0))
    assert m.solve().objective == pytest.approx(expected, abs=1e-5)
