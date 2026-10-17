"""#1726: the box robust counterpart of ``a @ x`` with sign-indefinite ``x``.

Sign-tracking replaced ``a`` by one corner of its box -- sound only for ``x >= 0``
-- because the RO-1 routing test lived in the ``*`` branch only, so ``a @ x``
was "robustified" into a non-counterpart (worst case 0 against a required 10).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, RobustCounterpart

A_HAT = np.array([2.0, 3.0])
DELTA = np.array([0.5, 0.5])


def _solve(form: str, lb: float):
    m = dm.Model("r")
    x = m.continuous("x", shape=(2,), lb=lb, ub=10)
    a = m.parameter("a", value=A_HAT)
    lhs = {"matmul": lambda: a @ x, "rmatmul": lambda: x @ a, "sum": lambda: dm.sum(a * x)}[form]()
    m.subject_to(lhs >= 10)
    m.minimize(dm.sum(x))
    RobustCounterpart(m, BoxUncertaintySet(a, delta=DELTA)).formulate()
    r = m.solve()
    return m, r, r.value(x)


@pytest.mark.parametrize("form", ["matmul", "rmatmul", "sum"])
def test_every_box_corner_satisfies_the_row(form):
    _, r, xv = _solve(form, lb=-10.0)
    assert r.status == "optimal"
    corners = 0
    for signs in itertools.product((-1.0, 1.0), repeat=2):
        a = A_HAT + np.asarray(signs) * DELTA
        assert a @ xv >= 10.0 - 1e-6, (form, a, xv)
        corners += 1
    assert corners == 4
    assert r.objective == pytest.approx(4.0, abs=1e-6)


def test_matmul_gets_abs_auxiliaries_like_sum():
    m_mat, _, _ = _solve("matmul", lb=-10.0)
    m_sum, _, _ = _solve("sum", lb=-10.0)
    names = lambda m: sorted(v.name for v in m._variables if v.name.startswith("ro_abs"))  # noqa: E731
    assert names(m_mat), "matmul row was not routed to the |coeff| linearization"
    assert len(names(m_mat)) == len(names(m_sum))


def test_nonnegative_matmul_keeps_sign_tracking():
    m, r, xv = _solve("matmul", lb=0.0)
    assert r.status == "optimal"
    assert not [v for v in m._variables if v.name.startswith("ro_abs")]
    worst = (A_HAT - DELTA) @ xv
    assert worst >= 10.0 - 1e-6
