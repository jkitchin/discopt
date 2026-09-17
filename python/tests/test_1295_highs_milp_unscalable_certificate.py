"""#1295: the pure-MILP HiGHS route must not certify HiGHS's floating-point MIP bound
on a model whose unbounded columns carry entries HiGHS's scaling cannot equilibrate.

The instance is seed-11 #70 of the adversary generator (big-M coupling with
coefficients 1e-5 and 1e-9 on unbounded columns). Before the fix the route returned
``optimal`` with ``gap_certified=True`` and bound -131.6443, above the objective of the
exact witness below (-131.8438).
"""

from fractions import Fraction

import numpy as np
import pytest
from discopt import Model
from discopt.solvers.lp_milp_highs import (
    UNSCALABLE_OPEN_RATIO,
    StdForm,
    open_column_coefficient_ratio,
)

W = [
    [5, 6, 3, 14, 19, 18, 18, 13, 15, 10],
    [11, 4, 20, 17, 10, 7, 3, 6, 10, 17],
]
CAP = [Fraction(131, 2), Fraction(119, 2)]
PROFIT = [
    -1.6618264004042903e01,
    -1.3105367620010762e01,
    -1.5046271282550368e01,
    -3.9782804330010144e01,
    -2.9517336869671588e01,
    -2.9643673996339182e01,
    -1.9550785570172312e01,
    -1.5901507331289586e01,
    -2.1714006772893509e01,
    -2.2129287675547619e01,
]
EPS = [Fraction(1, 10**5), Fraction(1, 10**9)]
BIG = [10**10, 10**8]
COST_Z = [1.6338927530993160e05, 1.9946357485363558e-01]
X_WITNESS = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]


def _build(y1_ub=1e20):
    m = Model("i1295")
    x = [m.integer(f"x{j}", lb=0, ub=1 if j < 8 else 3) for j in range(10)]
    y = [m.continuous("y0", lb=0), m.continuous("y1", lb=0, ub=y1_ub)]
    z = [m.continuous(f"z{i}", lb=0, ub=1) for i in range(2)]
    for i in range(2):
        m.subject_to(sum(W[i][j] * x[j] for j in range(10)) - float(EPS[i]) * y[i] <= float(CAP[i]))
        m.subject_to(y[i] - BIG[i] * z[i] <= 0)
    m.minimize(sum(PROFIT[j] * x[j] for j in range(10)) + sum(COST_Z[i] * z[i] for i in range(2)))
    return m


def _witness():
    """Exact point: y_i and z_i at their smallest values for ``X_WITNESS``."""
    y = [max(Fraction(0), (sum(W[i][j] * X_WITNESS[j] for j in range(10)) - CAP[i]) / EPS[i])
         for i in range(2)]  # fmt: skip
    z = [y[i] / BIG[i] for i in range(2)]
    return y, z


def _witness_objective():
    y, z = _witness()
    return sum(Fraction(PROFIT[j]) * X_WITNESS[j] for j in range(10)) + sum(
        Fraction(COST_Z[i]) * z[i] for i in range(2)
    )


def test_witness_is_exactly_feasible():
    y, z = _witness()
    n = 0
    for i in range(2):
        lhs = sum(W[i][j] * X_WITNESS[j] for j in range(10)) - EPS[i] * y[i]
        assert lhs <= CAP[i]
        assert y[i] - BIG[i] * z[i] <= 0
        assert 0 <= z[i] <= 1 and y[i] >= 0
        n += 4
    assert y[1] <= 10**20
    assert n == 8
    assert float(_witness_objective()) == pytest.approx(-131.8437730540767, abs=1e-9)


def test_route_does_not_certify_a_bound_above_the_witness():
    r = _build().solve(time_limit=30)
    stats = r.solver_stats or {}
    assert stats.get("route/lp_milp_backend") == 1.0, "HiGHS MILP route not taken"
    w = float(_witness_objective())
    assert r.bound is None or r.bound <= w + 1e-6 * (1 + abs(w)), (r.status, r.bound)
    if r.gap_certified:
        assert r.objective <= w + 1e-6 * (1 + abs(w)), (r.status, r.objective)
    assert stats.get("milp/decertified_unscalable") == 1.0
    assert stats["milp/open_column_coef_ratio"] < UNSCALABLE_OPEN_RATIO
    assert r.status == "feasible" and not r.gap_certified
    assert r.x is not None
    if r.bound is not None:
        assert r.bound_source == "root_relaxation"


def test_sentinel_magnitude_bound_counts_as_open():
    # y1 <= 1e12: not the default box but past READBACK_LIMIT, the same class (#45).
    r = _build(y1_ub=1e12).solve(time_limit=30)
    stats = r.solver_stats or {}
    assert stats.get("route/lp_milp_backend") == 1.0
    assert stats.get("milp/decertified_unscalable") == 1.0
    assert not r.gap_certified


def test_well_scaled_milp_keeps_its_certificate():
    m = Model("knap")
    x = [m.integer(f"x{j}", lb=0, ub=3) for j in range(4)]
    s = m.continuous("s", lb=0)
    m.subject_to(3 * x[0] + 5 * x[1] + 2 * x[2] + 7 * x[3] + s <= 17)
    m.subject_to(s - 0.5 * x[0] >= 0)
    m.minimize(-4 * x[0] - 6 * x[1] - 3 * x[2] - 8 * x[3] + 0.1 * s)
    r = m.solve(time_limit=30)
    stats = r.solver_stats or {}
    assert stats.get("route/lp_milp_backend") == 1.0
    assert r.status == "optimal" and r.gap_certified
    assert "milp/decertified_unscalable" not in stats
    assert stats["milp/open_column_coef_ratio"] >= UNSCALABLE_OPEN_RATIO


def _sf(A, xl, xu):
    A = np.asarray(A, float)  # noqa: N806
    return StdForm.from_arrays(
        np.zeros(A.shape[1]), A, np.zeros(A.shape[0]), np.asarray(xl, float), np.asarray(xu, float)
    )


def test_ratio_counts_only_open_columns():
    A = [[1.0, 1e-9, 1e-3], [2.0, 0.0, 4.0]]
    assert open_column_coefficient_ratio(_sf(A, [0, 0, 0], [1, 1, 1])) == 1.0
    assert open_column_coefficient_ratio(_sf(A, [0, 0, 0], [1, 1, np.inf])) == pytest.approx(1e-3)
    assert open_column_coefficient_ratio(_sf(A, [0, -1e15, 0], [1, 1, 1])) == pytest.approx(1e-9)
    assert open_column_coefficient_ratio(_sf(A, [0, 0, 0], [1, 1e20, 1])) == pytest.approx(1e-9)
    assert open_column_coefficient_ratio(_sf([[0.0, 0.0]], [0, 0], [np.inf, 1])) == 1.0
