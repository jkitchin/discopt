"""Rust revised-simplex LP solver vs HiGHS (roadmap P1).

``discopt._rust.solve_lp_py`` solves the standard-form LP
``min cᵀx s.t. A x = b, lb ≤ x ≤ ub`` with the bounded-variable two-phase primal
simplex (feral LU backend). This is the cold-solve correctness gate: on random
feasible/bounded LPs its optimum must match HiGHS within 1e-6, and it must
detect infeasibility/unboundedness on constructed cases.
"""

from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("discopt._rust")
if not hasattr(rust, "solve_lp_py"):
    pytest.skip("simplex LP binding not built", allow_module_level=True)
highspy = pytest.importorskip("highspy")  # noqa: F841

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_highs import solve_lp as highs_lp  # noqa: E402


def _rand_standard_form(seed, m, n):
    """Random feasible+bounded standard-form LP: A x = b with x in [0,5]^n and
    b = A x0 for an interior x0, so a feasible (hence optimal) point exists."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 4, (m, n)).astype(float)
    x0 = rng.uniform(0.5, 4.5, n)
    b = A @ x0
    c = rng.integers(-5, 6, n).astype(float)
    lb = np.zeros(n)
    ub = np.full(n, 5.0)
    return c, A, b, lb, ub


class TestSimplexVsHighs:
    @pytest.mark.parametrize("seed", list(range(40)))
    def test_matches_highs_on_random_lps(self, seed):
        m = 2 + (seed % 4)
        n = m + 2 + (seed % 3)
        c, A, b, lb, ub = _rand_standard_form(seed, m, n)

        status, x, obj, _iters = rust.solve_lp_py(
            np.ascontiguousarray(c),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(lb),
            np.ascontiguousarray(ub),
        )
        hi = highs_lp(c=c, A_eq=A, b_eq=b, bounds=list(zip(lb.tolist(), ub.tolist())))
        if hi.status != SolveStatus.OPTIMAL:
            return  # only compare where HiGHS certified an optimum
        assert status == "optimal", f"seed={seed}: simplex status {status}"
        assert abs(obj - hi.objective) < 1e-6, f"seed={seed}: simplex {obj} vs HiGHS {hi.objective}"
        # returned point is feasible: A x = b, bounds respected.
        x = np.asarray(x)
        assert np.allclose(A @ x, b, atol=1e-6)
        assert np.all(x >= lb - 1e-7) and np.all(x <= ub + 1e-7)

    def test_knapsack_relaxation(self):
        a = np.array([[5.0, 5.0, 5.0, 5.0, 1.0]])
        b = np.array([9.0])
        c = np.array([-16.0, -16.0, -16.0, -16.0, 0.0])
        lb = np.zeros(5)
        ub = np.array([1.0, 1.0, 1.0, 1.0, 1e20])
        status, _x, obj, _ = rust.solve_lp_py(c, a, b, lb, ub)
        assert status == "optimal"
        assert abs(obj - (-28.8)) < 1e-6

    def test_infeasible(self):
        # x0 + s = 1, s>=0, x0 in [2,inf): x0>=2 but x0<=1 → infeasible.
        a = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        c = np.array([1.0, 0.0])
        lb = np.array([2.0, 0.0])
        ub = np.array([1e20, 1e20])
        status, _x, _obj, _ = rust.solve_lp_py(c, a, b, lb, ub)
        assert status == "infeasible"

    def test_unbounded(self):
        a = np.array([[1.0, -1.0]])
        b = np.array([0.0])
        c = np.array([-1.0, 0.0])
        lb = np.array([0.0, 0.0])
        ub = np.array([1e20, 1e20])
        status, _x, _obj, _ = rust.solve_lp_py(c, a, b, lb, ub)
        assert status == "unbounded"
