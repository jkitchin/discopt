"""AMP with the warm-started-simplex MILP backend (milp_solver="simplex").

Two gates:
1. The simplex `solve_milp` adapter matches the POUNCE B&B reference on small
   binary MILPs (HiGHS was removed, issue #356) and reports a sound
   (lower-bound) objective.
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


class TestSimplexAdapterVsPounce:
    @pytest.mark.parametrize("seed", list(range(8)))
    def test_matches_pounce(self, seed):
        pytest.importorskip("pounce")
        from discopt.solvers.milp_pounce import solve_milp as ref_milp

        rng = np.random.default_rng(seed)
        n = 5
        c = -rng.integers(1, 10, n).astype(float)
        A = rng.integers(1, 8, (2, n)).astype(float)
        b = (0.5 * A.sum(axis=1)).astype(float)
        bounds = [(0.0, 1.0)] * n
        integ = np.ones(n)
        rs = simplex_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=integ)
        rh = ref_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=integ)
        if rh.status != SolveStatus.OPTIMAL:
            return
        assert rs.status == SolveStatus.OPTIMAL
        # POUNCE is an interior-point reference, so allow the project rel
        # tolerance rather than the exact-vs-exact 1e-6 the HiGHS oracle gave.
        assert abs(rs.objective - rh.objective) < 1e-4, f"seed={seed}"

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

    # Per-case budget: ``_concave_qp`` *certifies* (its status is time-sensitive),
    # so it needs enough wall-clock that both backends reach ``optimal`` even on a
    # slow CI runner under parallel load — a 5s budget flaked as
    # ``optimal vs feasible`` when only one backend certified in time. ``_bilinear``
    # never certifies (``feasible`` at any budget — a relaxation-tightness property,
    # not a backend property), so it stays short to avoid burning wall-clock. The
    # seam under test is backend *parity*, which holds at any budget.
    @pytest.mark.slow
    @pytest.mark.parametrize("build,time_limit", [(_concave_qp, 60), (_bilinear, 5)])
    def test_same_certificate_as_default(self, build, time_limit):
        ref = build().solve(solver="amp", rel_gap=1e-4, time_limit=time_limit)
        sx = build().solve(solver="amp", rel_gap=1e-4, time_limit=time_limit, milp_solver="simplex")
        assert sx.status == ref.status, f"{sx.status} vs {ref.status}"
        if ref.status == "optimal":
            assert getattr(sx, "gap_certified", True) is True
        # Objective parity must hold whether the run certified (``optimal``) or
        # only reached the shared global incumbent (``feasible``): the simplex
        # relaxation backend must not weaken AMP's result either way.
        if ref.objective is not None and sx.objective is not None:
            assert abs(sx.objective - ref.objective) <= 1e-3 * (1.0 + abs(ref.objective))
