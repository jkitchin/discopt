"""Both MILP engines round a row over one free integer column exactly (#2252
follow-up, found reviewing #2257).

The engine enforces rows to its 1e-6 tolerance, but ``verify_point`` allows a row
whose other columns are all fixed only float noise (#1254): no fractional move can
absorb the residual. On ``2x + y >= 2.0000004`` (``y`` fixed at 0, integer ``x``)
the engine certified x = 1 (violation 4e-7) whenever its presolve did not run or
declared the problem infeasible -- the latter on two near-inconsistent continuous
equalities ``c == 1``, ``c == 1 + 5e-8``, which also made the ``auto`` engine's
HiGHS route error and fall back here. #1561 then published x = 1 ``feasible``;
the optimum is x = 2. With a 5e-8 gap HiGHS's route kept x = 1 itself: its
1e-9 tight re-solve (#2151) is infeasible on the ``c`` rows, so it never reached
x = 2. Both routes now hand their engine the rounded box (x >= 2).
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solver import _round_single_integer_rows_sparse
from discopt.validation.feasibility import verify_point


def _model(near_inconsistent: bool, row_rhs: float = 2.0000004) -> dm.Model:
    m = dm.Model("int_row_round")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=0)
    c = m.continuous("c", lb=-10, ub=10)
    m.subject_to(2 * x + y >= row_rhs)
    if near_inconsistent:
        m.subject_to(c == 1)
        m.subject_to(c == 1.00000005)
    m.minimize(x)
    return m


@pytest.mark.parametrize("engine", ["rust", "auto"])
@pytest.mark.parametrize("milp_presolve", ["1", "0"])
@pytest.mark.parametrize("near_inconsistent", [False, True])
def test_rust_route_certifies_the_rounded_optimum(
    monkeypatch, engine, milp_presolve, near_inconsistent
):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", engine)
    monkeypatch.setenv("DISCOPT_MILP_PRESOLVE", milp_presolve)
    m = _model(near_inconsistent)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.algorithm_route)
    assert r.gap_certified
    assert float(r.x["x"]) == pytest.approx(2.0, abs=1e-9)
    assert r.objective == pytest.approx(2.0, abs=1e-9)
    xs = np.array([float(r.x["x"]), float(r.x["y"]), float(r.x["c"])])
    assert verify_point(m, xs).ok


def test_rust_route_row_on_an_integer_keeps_it(monkeypatch):
    # A row resting on an integer up to float noise must not be rounded past it.
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    monkeypatch.setenv("DISCOPT_MILP_PRESOLVE", "0")
    r = _model(False, row_rhs=2.0 + 1e-12).solve(time_limit=30)
    assert r.status == "optimal"
    assert float(r.x["x"]) == pytest.approx(1.0, abs=1e-9)


def test_rust_route_integer_equality_without_integer_solution_is_infeasible(monkeypatch):
    # 2x == 3 has no integer point: the rounded box is [2, 1].
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    monkeypatch.setenv("DISCOPT_MILP_PRESOLVE", "0")
    m = dm.Model("int_eq")
    x = m.integer("x", lb=0, ub=10)
    m.continuous("c", lb=0, ub=1)
    m.subject_to(2 * x == 3)
    m.minimize(x)
    assert m.solve(time_limit=30).status == "infeasible"


def test_helper_rounds_only_single_free_integer_rows():
    # cols: x0 int, x1 int, x2 continuous, x3 int fixed at 1; one slack col (index 4).
    xl = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    xu = np.array([10.0, 10.0, 10.0, 1.0, 1e20])
    A_ub = sp.csr_matrix(
        np.array(
            [
                [-2.0, 0.0, 0.0, 1.0],  # -2 x0 + 1 <= -1.0000004 -> x0 >= 2 (rounded)
                [0.0, 3.0, 0.0, 0.0],  # 3 x1 <= 7.5             -> x1 <= 2
                [1.0, 1.0, 0.0, 0.0],  # two free integers        -> untouched
                [0.0, 0.0, 1.0, 0.0],  # continuous               -> untouched
            ]
        )
    )
    b_ub = np.array([-1.0000004, 7.5, 0.5, 0.5])
    A_eq = sp.csr_matrix(np.array([[0.0, 2.0, 0.0, 0.0]]))  # 2 x1 == 4 -> x1 == 2
    b_eq = np.array([4.0 + 1e-12])
    lo, hi, moved = _round_single_integer_rows_sparse(
        xl, xu, 4, [0, 1, 3], ((A_ub, b_ub, "<="), (A_eq, b_eq, "=="))
    )
    assert lo.tolist() == [2.0, 2.0, 0.0, 1.0, 0.0]
    assert hi.tolist() == [10.0, 2.0, 10.0, 1.0, 1e20]
    assert moved == 3
    # Inputs are not mutated.
    assert xl.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
