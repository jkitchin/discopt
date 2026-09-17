"""#1296: the Rust MILP driver must not certify a bound its LP could not compute.

The instance is #116 of the #1280 adversary generator: a knapsack pair whose overflow
is bought through ``W x - eps*y <= cap`` and ``y - Ybig*t <= 0`` with ``eps = 1e-11``
on an open column. That entry sits under the simplex's equilibration noise floor, so
the node LPs ran as if it were absent. Before the fix ``solve_milp`` returned
``optimal`` with bound -198.07, above the exact witness below (-199.694), and an
objective that was not ``c·x`` of its own point (-199.69).
"""

from fractions import Fraction

import numpy as np
import pytest
from discopt import Model
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import solve_milp

W = [
    [13, 19, 10, 19, 17, 14, 3, 17, 18, 13],
    [3, 10, 18, 12, 3, 7, 19, 19, 10, 20],
]
CAP = [Fraction(157, 2), Fraction(135, 2)]
EPS = [Fraction(1, 10**11), Fraction(1, 10**7)]
YBIG = [10**11, 10**10]
C = [
    -18.888452709465238,
    -21.456179554563477,
    -31.948010253315744,
    -24.61654598023271,
    -24.59685763236504,
    -26.541716782487985,
    -22.42885012693303,
    -43.20121279610278,
    -27.57876012953446,
    -40.70864208846242,
    0.0,
    0.0,
    0.7237376799957796,
    553.8487117864396,
]
BOUNDS = [(0, 1)] * 8 + [(0, 3)] * 2 + [(0, None), (0, 1e12), (0, 1), (0, 1)]
INTEG = np.array([1] * 10 + [0] * 4)
X_WITNESS = [1, 0, 1, 0, 0, 1, 1, 0, 0, 3]


def _rows():
    A = np.zeros((4, 14))  # noqa: N806
    A[:2, :10] = W
    A[0, 10], A[1, 11] = -float(EPS[0]), -float(EPS[1])
    A[2, 10], A[2, 12] = 1.0, -YBIG[0]
    A[3, 11], A[3, 13] = 1.0, -YBIG[1]
    return A, np.array([float(CAP[0]), float(CAP[1]), 0.0, 0.0])


def _witness():
    """Exact point: ``y`` and ``t`` at their smallest values for ``X_WITNESS``."""
    over = [sum(W[k][j] * X_WITNESS[j] for j in range(10)) - CAP[k] for k in range(2)]
    y = [max(Fraction(0), over[k]) / EPS[k] for k in range(2)]
    t = [y[k] / YBIG[k] for k in range(2)]
    return [Fraction(v) for v in X_WITNESS] + y + t


def _witness_objective():
    return float(sum(Fraction(C[j]) * v for j, v in enumerate(_witness())))


def _slack(v):
    return 1e-6 * (1 + abs(v))


def test_witness_is_exactly_feasible():
    w = _witness()
    n = 0
    for k in range(2):
        assert sum(W[k][j] * w[j] for j in range(10)) - EPS[k] * w[10 + k] <= CAP[k]
        assert w[10 + k] - YBIG[k] * w[12 + k] <= 0
        n += 2
    for j, (lo, hi) in enumerate(BOUNDS):
        assert lo <= w[j] and (hi is None or w[j] <= Fraction(hi))
        n += 1
    assert n == 18
    assert w[10] > 0, "the 1e-11 row must be active, or the instance tests nothing"
    assert _witness_objective() == pytest.approx(-199.694063182027, abs=1e-9)


def test_solve_milp_does_not_certify_past_the_witness():
    A, b = _rows()  # noqa: N806
    res = solve_milp(
        np.array(C), A_ub=A, b_ub=b, bounds=BOUNDS, integrality=INTEG,
        time_limit=60, gap_tolerance=1e-9,
    )  # fmt: skip
    w = _witness_objective()
    assert res.x is not None
    cx = float(np.dot(C, res.x))
    assert res.objective == pytest.approx(cx, abs=_slack(cx)), "objective is not c.x"
    assert res.bound is None or res.bound <= w + _slack(w), (res.status, res.bound)
    assert res.status != SolveStatus.OPTIMAL or res.objective <= w + _slack(w)
    # The driver refuses the certificate rather than guessing at the dropped term.
    assert res.status == SolveStatus.ITERATION_LIMIT and res.bound is None


def test_model_solve_on_the_rust_backend_is_uncertified(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    A, b = _rows()  # noqa: N806
    m = Model("i1296")
    v = []
    for j, (lo, hi) in enumerate(BOUNDS):
        if INTEG[j]:
            v.append(m.integer(f"v{j}", lb=lo, ub=hi))
        elif hi is None:
            v.append(m.continuous(f"v{j}", lb=lo))
        else:
            v.append(m.continuous(f"v{j}", lb=lo, ub=hi))
    for i in range(4):
        m.subject_to(sum(float(A[i, j]) * v[j] for j in range(14) if A[i, j]) <= float(b[i]))
    m.minimize(sum(C[j] * v[j] for j in range(14) if C[j]))
    r = m.solve(time_limit=60)
    assert (r.solver_stats or {}).get("route/lp_milp_backend") is None, "HiGHS route taken"
    w = _witness_objective()
    assert r.bound is None or r.bound <= w + _slack(w), (r.status, r.bound)
    if r.gap_certified:
        assert r.objective <= w + _slack(w), (r.status, r.objective)
    assert not r.gap_certified


def test_well_scaled_milp_keeps_its_certificate():
    # Same shape with eps = 1e-2 and Ybig = 1e3: every entry within the LP's range.
    A, b = _rows()  # noqa: N806
    A[0, 10], A[1, 11], A[2, 12], A[3, 13] = -1e-2, -1e-2, -1e3, -1e3
    res = solve_milp(
        np.array(C), A_ub=A, b_ub=b, bounds=BOUNDS, integrality=INTEG,
        time_limit=60, gap_tolerance=1e-9,
    )  # fmt: skip
    assert res.status == SolveStatus.OPTIMAL
    assert res.bound is not None
    assert res.bound <= res.objective + _slack(res.objective)
    assert res.objective == pytest.approx(float(np.dot(C, res.x)), abs=1e-6)
