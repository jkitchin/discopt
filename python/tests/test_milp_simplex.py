"""Rust-internal warm-started-simplex MILP solver vs HiGHS (roadmap P4).

``discopt._rust.solve_milp_py`` runs the whole pure-MILP branch-and-bound in
Rust: the existing tree manager, with each node's LP solved by the bounded
simplex — root cold, children warm-started (dual simplex) from the basis they
inherit from their parent. This is the B&B correctness gate: on random MILPs its
optimum must match HiGHS (``incorrect_count == 0``, within 1e-4), and it must
detect infeasibility.

The solver consumes standard form ``A x = b`` with explicit slack columns; the
helper here slacks a ``≤`` MILP so the same instance can be sent to both
solvers.
"""

from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("discopt._rust")
if not hasattr(rust, "solve_milp_py"):
    pytest.skip("simplex MILP binding not built", allow_module_level=True)

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.milp_highs import solve_milp as highs_milp  # noqa: E402


def _slack_standard_form(c, A_ub, b_ub, lb, ub):
    """Turn `min c x s.t. A_ub x <= b_ub, lb<=x<=ub` into standard form
    `A_eq z = b` with one slack per row. Returns (c_s, A_eq, b, l_s, u_s, n)."""
    m, n = A_ub.shape
    A_eq = np.zeros((m, n + m))
    A_eq[:, :n] = A_ub
    A_eq[:, n:] = np.eye(m)
    c_s = np.concatenate([c, np.zeros(m)])
    l_s = np.concatenate([lb, np.zeros(m)])
    u_s = np.concatenate([ub, np.full(m, 1e20)])
    return c_s, A_eq, b_ub.astype(float), l_s, u_s, n


def _solve_simplex(c, A_ub, b_ub, lb, ub, integer_cols):
    c_s, A_eq, b, l_s, u_s, n = _slack_standard_form(c, A_ub, b_ub, lb, ub)
    return rust.solve_milp_py(
        np.ascontiguousarray(c_s),
        np.ascontiguousarray(A_eq),
        np.ascontiguousarray(b),
        np.ascontiguousarray(l_s),
        np.ascontiguousarray(u_s),
        np.ascontiguousarray(np.asarray(integer_cols, dtype=np.int64)),
        n,
    )


class TestMilpSimplexVsHighs:
    @pytest.mark.parametrize("seed", list(range(30)))
    def test_matches_highs(self, seed):
        rng = np.random.default_rng(seed)
        n, m = 4, 3
        c = rng.integers(-5, 6, n).astype(float)
        A = rng.integers(0, 4, (m, n)).astype(float)
        b = (A @ rng.integers(0, 4, n) + rng.integers(1, 5, m)).astype(float)
        lb = np.zeros(n)
        ub = np.full(n, 5.0)
        integer_cols = list(range(n))

        status, x, obj, _bound, _nodes, _it = _solve_simplex(c, A, b, lb, ub, integer_cols)
        hi = highs_milp(c=c, A_ub=A, b_ub=b, bounds=[(0, 5)] * n, integrality=np.ones(n))
        if hi.status != SolveStatus.OPTIMAL:
            return
        assert status in ("optimal", "feasible"), f"seed={seed}: {status}"
        assert abs(obj - hi.objective) < 1e-4, f"seed={seed}: simplex {obj} vs HiGHS {hi.objective}"

    def test_binary_knapsack(self):
        c = np.array([-10.0, -9.0, -8.0, -1.0])
        A = np.array([[5.0, 5.0, 5.0, 5.0]])
        b = np.array([9.0])
        status, _x, obj, _b, _n, _i = _solve_simplex(
            c, A, b, np.zeros(4), np.ones(4), [0, 1, 2, 3]
        )
        assert status == "optimal"
        assert abs(obj - (-10.0)) < 1e-6

    def test_infeasible(self):
        # x0 <= 1 (slack) but x0 >= 2 → infeasible.
        c = np.array([1.0])
        A = np.array([[1.0]])
        b = np.array([1.0])
        status, _x, _obj, _b, _n, _i = _solve_simplex(
            c, A, b, np.array([2.0]), np.array([5.0]), [0]
        )
        assert status == "infeasible"
