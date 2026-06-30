"""Validation of the POUNCE LP path (roadmap P0.1).

Mirrors the legacy LP-oracle tests and additionally pins the interior-point-specific
behaviours that matter when POUNCE replaces the simplex as the LP engine for OBBT,
McCormick-LP, and OA/GDP masters:

  - Correct objective and status across inequality / equality / mixed /
    bounded / free-variable / empty-constraint LPs.
  - Sound status mapping for the convex-LP corner cases an IPM must get right:
    infeasible -> INFEASIBLE, unbounded -> UNBOUNDED.
  - On degenerate / dual-degenerate LPs the IPM returns an interior point of
    the optimal face (not a simplex vertex): the objective matches but the
    primal does not, so those cases assert objective + feasibility only.
  - The exact Rust simplex is used as a cross-check oracle (HiGHS removed, #356).
  - No simplex basis / warm-start (basis is None; warm_basis is ignored).

Skipped entirely when POUNCE is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pounce")

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_pounce import (  # noqa: E402
    _INF,
    _phase1_min_violation,
    _stack_constraints,
    solve_lp,
)

# Cross-check oracle: the exact Rust simplex (HiGHS was removed, issue #356).
# It exposes objective/duals/reduced-costs in HiGHS's convention.
try:
    from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE
    from discopt.solvers.lp_simplex import solve_lp as ref_lp

    _HIGHS = SIMPLEX_AVAILABLE
except ImportError:  # pragma: no cover
    _HIGHS = False

_OBJ_TOL = 1e-5


def _feasible(x, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, tol=1e-6):
    if A_ub is not None:
        assert np.all(A_ub @ x <= np.asarray(b_ub) + tol)
    if A_eq is not None:
        np.testing.assert_allclose(A_eq @ x, np.asarray(b_eq), atol=1e-5)
    if bounds is not None:
        for xi, (lo, hi) in zip(x, bounds):
            assert lo - tol <= xi <= hi + tol
    else:
        assert np.all(x >= -tol)


# ---------------------------------------------------------------------------
# 1. Basic LP with a unique optimum
# ---------------------------------------------------------------------------
class TestBasicLP:
    """min -x1 - 2*x2  s.t.  x1+x2 <= 10, x1,x2 >= 0  =>  x=[0,10], obj=-20."""

    def test_optimal_status(self):
        r = solve_lp(c=np.array([-1.0, -2.0]), A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([10.0]))
        assert r.status == SolveStatus.OPTIMAL

    def test_optimal_solution(self):
        r = solve_lp(c=np.array([-1.0, -2.0]), A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([10.0]))
        assert r.x is not None
        np.testing.assert_allclose(r.x, [0.0, 10.0], atol=1e-4)
        assert abs(r.objective - (-20.0)) < _OBJ_TOL

    def test_no_basis_returned(self):
        """IPM has no simplex basis; warm_basis is accepted but ignored."""
        r = solve_lp(
            c=np.array([-1.0, -2.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([10.0]),
            warm_basis=object(),
        )
        assert r.status == SolveStatus.OPTIMAL
        assert r.basis is None


# ---------------------------------------------------------------------------
# 2. Inequality constraints only
# ---------------------------------------------------------------------------
class TestInequalityConstraints:
    def test_solution(self):
        c = np.array([-3.0, -5.0])
        A_ub = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 5.0]])
        b_ub = np.array([4.0, 12.0, 25.0])
        r = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub)
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-25.0)) < _OBJ_TOL  # x1=5/3, x2=4


# ---------------------------------------------------------------------------
# 3. Equality constraints only
# ---------------------------------------------------------------------------
class TestEqualityConstraints:
    def test_solution(self):
        r = solve_lp(c=np.array([1.0, 1.0]), A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([5.0]))
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - 5.0) < _OBJ_TOL


# ---------------------------------------------------------------------------
# 4. Mixed inequality and equality (transportation problem)
# ---------------------------------------------------------------------------
class TestMixedConstraints:
    def _data(self):
        c = np.array([2.0, 3.0, 1.0, 4.0])
        A_eq = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        b_eq = np.array([10.0, 15.0])
        A_ub = np.array([[-1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, -1.0]])
        b_ub = np.array([-8.0, -12.0])
        return c, A_ub, b_ub, A_eq, b_eq

    def test_optimal_and_feasible(self):
        c, A_ub, b_ub, A_eq, b_eq = self._data()
        r = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        assert r.status == SolveStatus.OPTIMAL
        _feasible(r.x, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    @pytest.mark.skipif(not _HIGHS, reason="HiGHS oracle unavailable")
    def test_objective_matches_highs(self):
        c, A_ub, b_ub, A_eq, b_eq = self._data()
        r = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        ref = ref_lp(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        assert abs(r.objective - ref.objective) < 1e-4


# ---------------------------------------------------------------------------
# 5. Variable bounds (unique optimum at the corner)
# ---------------------------------------------------------------------------
class TestVariableBounds:
    def test_with_bounds(self):
        r = solve_lp(c=np.array([-1.0, -1.0]), bounds=[(1.0, 3.0), (2.0, 4.0)])
        assert r.status == SolveStatus.OPTIMAL
        np.testing.assert_allclose(r.x, [3.0, 4.0], atol=1e-4)
        assert abs(r.objective - (-7.0)) < _OBJ_TOL


# ---------------------------------------------------------------------------
# 6. Free variables (lb=-inf, ub=+inf)
# ---------------------------------------------------------------------------
class TestFreeVariables:
    def test_free_vars_equality(self):
        """min x1+x2 s.t. x1+x2=5, x free  => obj=5 (x not unique)."""
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([5.0]),
            bounds=[(-np.inf, np.inf), (-np.inf, np.inf)],
        )
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - 5.0) < _OBJ_TOL
        np.testing.assert_allclose(r.x[0] + r.x[1], 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Empty constraint set (bound-constrained LP only)
# ---------------------------------------------------------------------------
class TestEmptyConstraints:
    def test_bounds_only(self):
        r = solve_lp(c=np.array([2.0, -1.0]), bounds=[(0.0, 5.0), (0.0, 5.0)])
        assert r.status == SolveStatus.OPTIMAL
        # min 2*x1 - x2 over [0,5]^2 => x1=0, x2=5, obj=-5
        assert abs(r.objective - (-5.0)) < _OBJ_TOL
        np.testing.assert_allclose(r.x, [0.0, 5.0], atol=1e-4)


# ---------------------------------------------------------------------------
# 8. Degenerate / dual-degenerate: interior point of the optimal face
# ---------------------------------------------------------------------------
class TestDegenerate:
    """min -x1-x2 s.t. x1+x2<=1, x>=0.

    The optimal face is the whole segment x1+x2=1; an IPM returns its analytic
    center (~[0.5, 0.5]) rather than a vertex. Objective and feasibility hold;
    the primal differs from simplex by design.
    """

    def test_objective_and_optimal_face(self):
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([1.0])
        r = solve_lp(c=np.array([-1.0, -1.0]), A_ub=A_ub, b_ub=b_ub)
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-1.0)) < _OBJ_TOL
        np.testing.assert_allclose(r.x[0] + r.x[1], 1.0, atol=1e-5)
        _feasible(r.x, A_ub=A_ub, b_ub=b_ub)

    def test_interior_not_vertex(self):
        """Documents the IPM behaviour: solution is strictly interior."""
        r = solve_lp(c=np.array([-1.0, -1.0]), A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([1.0]))
        assert r.x[0] > 1e-3 and r.x[1] > 1e-3  # not at a vertex


# ---------------------------------------------------------------------------
# 9. Infeasible LP  (convex => local infeasibility is global)
# ---------------------------------------------------------------------------
class TestInfeasible:
    def test_infeasible(self):
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
            b_ub=np.array([1.0, -10.0]),
        )
        assert r.status == SolveStatus.INFEASIBLE
        assert r.x is None

    def test_infeasible_equality(self):
        """Inconsistent equalities (x1+x2 = 1 and = 5).

        The IPM alone can cycle to the iteration limit here rather than
        certify infeasibility (a finding from P0.1). The elastic Phase-1
        certificate (P0.2) resolves it: the minimal total violation is 4 > 0,
        which proves infeasibility, so the status is INFEASIBLE.
        """
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0], [1.0, 1.0]]),
            b_eq=np.array([1.0, 5.0]),
            options={"max_iter": 300},
        )
        assert r.status == SolveStatus.INFEASIBLE
        assert r.x is None


# ---------------------------------------------------------------------------
# 10. Unbounded LP
# ---------------------------------------------------------------------------
class TestUnbounded:
    def test_unbounded_bounds(self):
        r = solve_lp(c=np.array([-1.0]), bounds=[(0.0, float("inf"))])
        assert r.status == SolveStatus.UNBOUNDED
        assert r.x is None

    def test_unbounded_constraint_dir(self):
        # min -x1-x2 s.t. x1-x2<=1, x>=0 : unbounded along x1=x2->inf
        r = solve_lp(
            c=np.array([-1.0, -1.0]),
            A_ub=np.array([[1.0, -1.0]]),
            b_ub=np.array([1.0]),
        )
        assert r.status == SolveStatus.UNBOUNDED


# ---------------------------------------------------------------------------
# 11. Dual values / reduced costs shapes
# ---------------------------------------------------------------------------
class TestDuals:
    def test_dual_shapes(self):
        c = np.array([1.0, 1.0])
        A_ub = np.array([[1.0, 0.0], [0.0, 1.0]])
        b_ub = np.array([3.0, 4.0])
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([5.0])
        r = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        assert r.status == SolveStatus.OPTIMAL
        assert r.dual_values is not None and r.dual_values.shape == (3,)  # 2 ub + 1 eq
        assert r.reduced_costs is not None and r.reduced_costs.shape == (2,)


# ---------------------------------------------------------------------------
# 11b. Dual sign convention parity with HiGHS
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HIGHS, reason="HiGHS oracle unavailable")
class TestDualConvention:
    """LPResult documents one dual convention (HiGHS shadow prices, y=dz/db).

    Ipopt-style multipliers enter the Lagrangian as f + mult_g^T g and are the
    negation of that; lp_pounce flips them. On LPs with a *unique* dual
    solution the two backends must agree exactly.
    """

    def test_equality_dual_matches_highs(self):
        # min x1+x2 s.t. x1+x2 = 5: unique dual y = 1.
        kw = dict(c=np.array([1.0, 1.0]), A_eq=np.array([[1.0, 1.0]]), b_eq=np.array([5.0]))
        rp = solve_lp(**kw)
        rh = ref_lp(**kw)
        np.testing.assert_allclose(rp.dual_values, rh.dual_values, atol=1e-5)

    def test_inequality_duals_match_highs(self):
        # min -3x1-5x2 s.t. x1<=4, 2x2<=12, 3x1+5x2<=25: nondegenerate optimum
        # at x=(5/3, 4) with rows 2,3 active -> unique duals.
        kw = dict(
            c=np.array([-3.0, -5.0]),
            A_ub=np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 5.0]]),
            b_ub=np.array([4.0, 12.0, 25.0]),
        )
        rp = solve_lp(**kw)
        rh = ref_lp(**kw)
        np.testing.assert_allclose(rp.dual_values, rh.dual_values, atol=1e-5)
        np.testing.assert_allclose(rp.reduced_costs, rh.reduced_costs, atol=1e-5)


# ---------------------------------------------------------------------------
# 12. Sparse matrix support
# ---------------------------------------------------------------------------
class TestSparse:
    def test_sparse_matches_dense(self):
        c = np.array([-3.0, -5.0])
        A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 5.0]])
        b = np.array([4.0, 12.0, 25.0])
        dense = solve_lp(c=c, A_ub=A, b_ub=b)
        spr = solve_lp(c=c, A_ub=sp.csr_matrix(A), b_ub=b)
        assert spr.status == SolveStatus.OPTIMAL
        assert abs(spr.objective - dense.objective) < _OBJ_TOL

    def test_sparse_A_eq(self):
        r = solve_lp(c=np.array([1.0, 1.0]), A_eq=sp.csc_matrix([[1.0, 1.0]]), b_eq=np.array([5.0]))
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - 5.0) < _OBJ_TOL


# ---------------------------------------------------------------------------
# 13. Dimension-mismatch errors (parity with lp_simplex)
# ---------------------------------------------------------------------------
class TestDimensionMismatch:
    def test_A_ub_wrong_cols(self):
        with pytest.raises(ValueError, match="columns"):
            solve_lp(c=np.array([1.0, 2.0]), A_ub=np.array([[1.0, 2.0, 3.0]]), b_ub=np.array([1.0]))

    def test_b_ub_wrong_rows(self):
        with pytest.raises(ValueError, match="rows"):
            solve_lp(c=np.array([1.0, 2.0]), A_ub=np.array([[1.0, 2.0]]), b_ub=np.array([1.0, 2.0]))

    def test_A_eq_wrong_cols(self):
        with pytest.raises(ValueError, match="columns"):
            solve_lp(c=np.array([1.0, 2.0]), A_eq=np.array([[1.0]]), b_eq=np.array([1.0]))

    def test_bounds_wrong_length(self):
        with pytest.raises(ValueError, match="bounds"):
            solve_lp(c=np.array([1.0, 2.0]), bounds=[(0.0, 1.0)])

    def test_b_ub_missing(self):
        with pytest.raises(ValueError, match="b_ub"):
            solve_lp(c=np.array([1.0]), A_ub=np.array([[1.0]]))

    def test_b_eq_missing(self):
        with pytest.raises(ValueError, match="b_eq"):
            solve_lp(c=np.array([1.0]), A_eq=np.array([[1.0]]))


# ---------------------------------------------------------------------------
# 13b. Elastic Phase-1 infeasibility certificate (roadmap P0.2)
# ---------------------------------------------------------------------------
class TestInfeasibilityCertificate:
    """The Phase-1 LP minimizes total constraint violation; its optimum is an
    exact infeasibility certificate for an LP (>0 iff infeasible)."""

    _OPTS = {"print_level": 0}

    def _violation(self, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, n=2):
        A, cl, cu = _stack_constraints(A_ub, b_ub, A_eq, b_eq, n)
        if bounds is None:
            lb, ub = np.zeros(n), np.full(n, _INF)
        else:
            lb = np.array([b[0] for b in bounds], dtype=float)
            ub = np.array([b[1] for b in bounds], dtype=float)
        slacks = _phase1_min_violation(A, cl, cu, lb, ub, self._OPTS)
        return None if slacks is None else float(slacks.sum())

    def test_violation_positive_when_infeasible(self):
        # x1+x2 = 1 and = 5  =>  minimal total violation is 4.
        v = self._violation(A_eq=np.array([[1.0, 1.0], [1.0, 1.0]]), b_eq=np.array([1.0, 5.0]))
        assert v is not None and abs(v - 4.0) < 1e-4

    def test_violation_zero_when_feasible(self):
        # A feasible system: minimal violation is 0.
        v = self._violation(A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([10.0]))
        assert v is not None and v < 1e-5

    def test_certificate_attached_on_infeasible(self):
        """An infeasibility found via Phase-1 carries the witness for free."""
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0], [1.0, 1.0]]),
            b_eq=np.array([1.0, 5.0]),
            options={"max_iter": 300},
        )
        assert r.status == SolveStatus.INFEASIBLE
        cert = r.infeasibility_certificate
        assert cert is not None
        assert abs(cert.total_violation - 4.0) < 1e-4
        assert cert.ineq_violations.shape == (0,)
        assert cert.eq_violations.shape == (2,)
        # Both equalities participate in the conflict.
        assert np.all(cert.eq_violations >= -1e-9)
        assert cert.eq_violations.sum() == pytest.approx(4.0, abs=1e-4)

    def test_certificate_identifies_conflicting_inequalities(self):
        """x1+x2 <= 1 and -(x1+x2) <= -10 conflict; both rows show violation."""
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
            b_ub=np.array([1.0, -10.0]),
            certificate=True,
        )
        assert r.status == SolveStatus.INFEASIBLE
        cert = r.infeasibility_certificate
        assert cert is not None
        assert cert.ineq_violations.shape == (2,)
        assert cert.total_violation > 1.0  # gap between <=1 and >=10
        # At least one of the two conflicting rows carries positive violation.
        assert cert.ineq_violations.max() > 1e-3

    def test_no_certificate_without_flag_on_direct_infeasible(self):
        """Directly POUNCE-detected infeasibility skips the extra Phase-1
        unless a certificate is requested."""
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0], [-1.0, -1.0]]),
            b_ub=np.array([1.0, -10.0]),
        )
        assert r.status == SolveStatus.INFEASIBLE
        assert r.infeasibility_certificate is None

    def test_feasible_consistent_redundant_not_flagged(self):
        """Consistent but redundant equalities must NOT be called infeasible."""
        r = solve_lp(
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0], [2.0, 2.0]]),
            b_eq=np.array([5.0, 10.0]),
            options={"max_iter": 300},
        )
        assert r.status != SolveStatus.INFEASIBLE
        if r.status == SolveStatus.OPTIMAL:
            assert abs(r.objective - 5.0) < _OBJ_TOL

    def test_certificate_does_not_alter_optimal(self):
        """A clean optimal solve never triggers the Phase-1 path."""
        r = solve_lp(c=np.array([-1.0, -2.0]), A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([10.0]))
        assert r.status == SolveStatus.OPTIMAL
        assert abs(r.objective - (-20.0)) < _OBJ_TOL


# ---------------------------------------------------------------------------
# 14. HiGHS cross-check oracle over a battery of random feasible LPs
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HIGHS, reason="HiGHS oracle unavailable")
class TestHighsOracle:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_objective_matches_highs(self, seed):
        rng = np.random.default_rng(seed)
        n, m = 5, 7
        c = rng.standard_normal(n)
        A_ub = rng.standard_normal((m, n))
        x_feas = rng.uniform(0, 1, n)
        b_ub = A_ub @ x_feas + rng.uniform(0.5, 1.5, m)  # keep feasible
        bounds = [(0.0, 10.0)] * n  # bounded => finite optimum
        rp = solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        rh = ref_lp(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        assert rp.status == SolveStatus.OPTIMAL
        assert rh.status == SolveStatus.OPTIMAL
        assert abs(rp.objective - rh.objective) < 1e-4, (
            f"seed={seed}: POUNCE {rp.objective} vs HiGHS {rh.objective}"
        )


class TestUnboundedDisambiguation:
    """Ipopt codes 3/4 (too-small direction / diverging iterates) map to
    UNBOUNDED, but they fire the same way on an *infeasible* LP. The Phase-1
    Farkas disambiguation must run before UNBOUNDED is trusted (PR #117
    review #3), so a spurious UNBOUNDED on an infeasible system is corrected
    to INFEASIBLE."""

    def test_infeasible_reported_as_unbounded_is_corrected(self, monkeypatch):
        import discopt.solvers.lp_pounce as M
        from discopt.solvers import LPResult

        # Infeasible LP: x == 0 and x == 1 (two equality rows on one var).
        A = np.array([[1.0], [1.0]])
        b = np.array([0.0, 1.0])

        # Force only the *main* solve to (wrongly) report UNBOUNDED, as codes 3/4
        # would; the real solver still runs the Phase-1 elastic LP underneath.
        orig_core = M._solve_core
        calls = {"n": 0}

        def fake_core(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return LPResult(status=SolveStatus.UNBOUNDED)
            return orig_core(*a, **k)

        monkeypatch.setattr(M, "_solve_core", fake_core)

        res = M.solve_lp(c=np.array([1.0]), A_eq=A, b_eq=b, bounds=[(-1e20, 1e20)])
        # Phase-1 finds positive minimal violation -> genuine INFEASIBLE.
        assert res.status == SolveStatus.INFEASIBLE
        assert res.infeasibility_certificate is not None

    def test_feasible_unbounded_stays_unbounded(self, monkeypatch):
        import discopt.solvers.lp_pounce as M
        from discopt.solvers import LPResult

        # Feasible, genuinely unbounded: minimize -x with x >= 0, no upper bound,
        # one trivial satisfiable equality so m > 0 (Phase-1 path runs).
        A = np.array([[0.0]])
        b = np.array([0.0])

        orig_core = M._solve_core
        calls = {"n": 0}

        def fake_core(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return LPResult(status=SolveStatus.UNBOUNDED)
            return orig_core(*a, **k)

        monkeypatch.setattr(M, "_solve_core", fake_core)

        res = M.solve_lp(c=np.array([-1.0]), A_eq=A, b_eq=b, bounds=[(0.0, 1e20)])
        # Phase-1 violation ~0 (feasible) -> the genuine UNBOUNDED stands.
        assert res.status == SolveStatus.UNBOUNDED
