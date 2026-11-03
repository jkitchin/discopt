"""#1917: mbigm's M-value LPs must read the default box as *no bound*.

A variable declared with no bound carries ``DEFAULT_VARIABLE_BOUND`` (9.999e19),
just below the simplex's 1e20 infinity. ``_lp_box`` passed it through, so the
simplex took it for a finite bound and exited ``numerical`` -- on GDPlib
``batch_processing`` 402 of 450 M-value LPs, every one of which HiGHS solves.
Each failure fell back to interval arithmetic, which cannot bound a free variable
and refuses the model, so an mbigm lowering that the LP proves finite was refused.

The fix hands a default-box side to the LP as ``±inf``. That only enlarges the
LP's region, so the optimum still bounds the body and M stays valid; the tests
check the M against an independent HiGHS optimum (never below it) and that every
point of a sampled feasible set satisfies the relaxed row.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.constants import DEFAULT_VARIABLE_BOUND
from discopt.modeling.core import Constraint

# Four bounded columns x in [0, 10]; one free column w defined by the equality
# w = 1.5 x0 + 0.9 x3; three <= rows. Found by a random sweep (58 of 200 seeds
# trip it): with the default box passed as a finite bound the simplex exits
# ``numerical`` on this LP and the interval fallback raises on the free ``w``.
_EQ = {0: 1.5, 3: 0.9}
_UB = [
    ([0.6, 0.8, 0.2, 0.5], 0.1, 5.0),
    ([0.9, 0.6, -1.0, 0.7], -0.9, 5.0),
    ([0.5, -0.6, 0.7, 0.1], -0.4, 5.0),
]


def _model():
    m = dm.Model("mbigm_default_box")
    x = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(4)]
    w = m.continuous("w")  # no bound declared: the default box
    m.subject_to(w - sum(a * x[i] for i, a in _EQ.items()) == 0)
    for row, aw, rhs in _UB:
        m.subject_to(sum(a * xi for a, xi in zip(row, x)) + aw * w <= rhs)
    m.minimize(x[0])
    return m, x, w


def _highs_max_w() -> float:
    from scipy.optimize import linprog

    c = np.zeros(5)
    c[4] = -1.0
    a_eq = np.zeros((1, 5))
    a_eq[0, 4] = 1.0
    for i, a in _EQ.items():
        a_eq[0, i] = -a
    a_ub = np.array([row + [aw] for row, aw, _ in _UB])
    b_ub = np.array([rhs for *_, rhs in _UB])
    r = linprog(
        c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[0.0], bounds=[(0, 10)] * 4 + [(None, None)]
    )
    assert r.status == 0
    return -r.fun


def test_lp_box_opens_the_default_box():
    from discopt._relax.gdp_reformulate import _lp_box

    m, _, _ = _model()
    box = _lp_box(m, 5)
    assert box[:4] == [(0.0, 10.0)] * 4
    assert box[4] == (-np.inf, np.inf)
    # a declared bound, however large, below the default box is kept finite
    m2 = dm.Model("big")
    m2.continuous("z", lb=-1e19, ub=0.5 * DEFAULT_VARIABLE_BOUND)
    assert _lp_box(m2, 1) == [(-1e19, 0.5 * DEFAULT_VARIABLE_BOUND)]


def test_mbigm_m_value_lp_solves_over_a_free_variable():
    """Before the fix this raised: the LP exited ``numerical`` and the interval
    fallback refused the free ``w``."""
    from discopt._relax.gdp_reformulate import _compute_big_m_lp, _precompute_lp_relaxation

    m, _, w = _model()
    con = Constraint(body=w - 3.0, sense="<=", rhs=0.0)
    M = _compute_big_m_lp(con, m, _precompute_lp_relaxation(m))
    true_max = _highs_max_w() - 3.0
    checks = 0
    assert M >= true_max - 1e-9  # never below the body's true maximum: valid
    checks += 1
    assert M <= max(abs(true_max), 1e-8) * 1.01 + 1e-9  # and it is the LP value
    checks += 1

    # feasible-point sampling: no point of the relaxation's region violates
    # w - 3 <= M (the row with the disjunct inactive).
    rng = np.random.default_rng(0)
    for _ in range(4000):
        x = rng.uniform(0, 10, 4)
        wv = sum(a * x[i] for i, a in _EQ.items())
        if all(np.dot(row, x) + aw * wv <= rhs for row, aw, rhs in _UB):
            assert wv - 3.0 <= M + 1e-9
            checks += 1
    assert checks > 100, checks


def test_mbigm_end_to_end_reformulates_and_solves():
    m, x, w = _model()
    d1 = m.make_disjunct("d1")
    d1.subject_to(w <= 3.0)
    d2 = m.make_disjunct("d2")
    d2.subject_to(x[1] >= 9.0)
    m.add_disjunction([d1, d2], name="dj")
    r = m.solve(gdp_method="mbigm", time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-6)
