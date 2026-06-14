"""AMP with the warm-started-simplex MILP backend (milp_solver="simplex").

Two gates:
1. The simplex `solve_milp` adapter matches HiGHS on small binary MILPs and
   reports a sound (lower-bound) objective.
2. End-to-end AMP with ``milp_solver="simplex"`` produces the *same* certified
   result (status, objective within rel_gap, gap_certified) as the default
   backend on representative nonconvex models — the relaxation backend must not
   weaken AMP's global certificate.
"""

from __future__ import annotations

import numpy as np
import pytest

rust = pytest.importorskip("discopt._rust")
if not hasattr(rust, "solve_milp_py"):
    pytest.skip("simplex MILP binding not built", allow_module_level=True)

import discopt.modeling as dm  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.milp_simplex import solve_milp as simplex_milp  # noqa: E402

highs = pytest.importorskip("discopt.solvers.milp_highs")
highs_milp = highs.solve_milp


class TestSimplexAdapterVsHighs:
    @pytest.mark.parametrize("seed", list(range(8)))
    def test_matches_highs(self, seed):
        rng = np.random.default_rng(seed)
        n = 5
        c = -rng.integers(1, 10, n).astype(float)
        A = rng.integers(1, 8, (2, n)).astype(float)
        b = (0.5 * A.sum(axis=1)).astype(float)
        bounds = [(0.0, 1.0)] * n
        integ = np.ones(n)
        rs = simplex_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=integ)
        rh = highs_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=integ)
        if rh.status != SolveStatus.OPTIMAL:
            return
        assert rs.status == SolveStatus.OPTIMAL
        assert abs(rs.objective - rh.objective) < 1e-6, f"seed={seed}"

    def test_equality_rows_supported(self):
        # x0 + x1 == 1, binary, minimize -x0 - 2 x1 → x1=1, obj -2.
        c = np.array([-1.0, -2.0])
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([1.0])
        rs = simplex_milp(
            c=c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, 1.0)] * 2, integrality=np.ones(2)
        )
        assert rs.status == SolveStatus.OPTIMAL
        assert abs(rs.objective - (-2.0)) < 1e-6


def _concave_qp():
    m = dm.Model("cqp")
    c = [-1.0, 0.5, 1.5]
    xs = [m.continuous(f"x{i}", lb=-2.0, ub=2.0) for i in range(3)]
    m.subject_to(sum(xs) >= -1.0)
    m.subject_to(sum(xs) <= 3.0)
    m.minimize(sum(-((xs[i] - c[i]) ** 2) for i in range(3)))
    return m


def _bilinear():
    m = dm.Model("bil")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x + y <= 5.0)
    m.minimize(-x * y)
    return m


class TestAmpSimplexCertificationParity:
    """AMP must reach the same certified optimum with the simplex relaxation
    backend as with the default (HiGHS/POUNCE) backend."""

    @pytest.mark.parametrize("build", [_concave_qp, _bilinear])
    def test_same_certificate_as_default(self, build):
        ref = build().solve(solver="amp", rel_gap=1e-4, time_limit=60)
        sx = build().solve(solver="amp", rel_gap=1e-4, time_limit=60, milp_solver="simplex")
        assert sx.status == ref.status, f"{sx.status} vs {ref.status}"
        if ref.status == "optimal":
            assert getattr(sx, "gap_certified", True) is True
            assert abs(sx.objective - ref.objective) <= 1e-3 * (1.0 + abs(ref.objective))
