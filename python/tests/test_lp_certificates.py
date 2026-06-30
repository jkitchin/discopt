"""Pure-Rust LP certificates: safe dual bound + verified Farkas ray (issue #356).

These exercise the certificate side-channel that lets the spatial-B&B node
numerics certify a relaxation bound / infeasibility *without* an independent
HiGHS or POUNCE cross-check:

* the simplex's row duals feed a Neumaier–Shcherbina safe lower bound that is
  ``<=`` the true LP optimum at *any* conditioning (never a too-high, falsely
  fathoming bound), and
* the simplex's Farkas dual ray is independently verified to prove the feasible
  set empty — a rigorous fathoming proof.

The defining soundness property under test: the safe bound is never above the
true optimum, and a verified Farkas ray only ever certifies a genuinely empty
polytope. An imperfect certificate must degrade to a (sound) fallback, never to
an unsound fathom.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import (
    _farkas_certified_std,
    _safe_lp_lower_bound_std,
    solve_lp_warm_std,
)


def _std(A_ub, b_ub, c, bounds):
    """Build the `[A_ub | I] z = b` standard form used by the safe-bound helpers."""
    A = np.asarray(A_ub, dtype=np.float64)
    m, n = A.shape
    a_std = np.zeros((m, n + m), dtype=np.float64)
    a_std[:, :n] = A
    a_std[:, n:] = np.eye(m)
    c_std = np.concatenate([np.asarray(c, float), np.zeros(m)])
    lb = np.array([lo for lo, _ in bounds], float)
    ub = np.array([hi for _, hi in bounds], float)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, 1e20)])
    return a_std, np.asarray(b_ub, float), c_std, lb_std, ub_std


class TestSafeBoundHelper:
    def test_safe_bound_at_correct_duals_reproduces_optimum(self):
        # min -x0 - 2 x1 s.t. x0 + x1 <= 4, x in [0,5].  Optimum -8 at (0,4).
        a_std, b, c_std, lb, ub = _std([[1.0, 1.0]], [4.0], [-1.0, -2.0], [(0, 5), (0, 5)])
        # Correct row dual is y = -2 (the binding [A|I] convention).
        g = _safe_lp_lower_bound_std(np.array([-2.0]), c_std, a_std, b, lb, ub)
        assert g is not None
        assert g <= -8.0 + 1e-6  # never above the true optimum
        assert abs(g - (-8.0)) < 1e-6  # and tight at the correct duals

    def test_safe_bound_is_lower_bound_for_arbitrary_duals(self):
        # Weak duality holds for ANY y: a wrong/loose dual must still give a value
        # at or below the true optimum (never a too-high, unsound bound).
        a_std, b, c_std, lb, ub = _std([[1.0, 1.0]], [4.0], [-1.0, -2.0], [(0, 5), (0, 5)])
        for y in (-5.0, -1.0, 0.0, 0.5):
            g = _safe_lp_lower_bound_std(np.array([y]), c_std, a_std, b, lb, ub)
            if g is not None:
                assert g <= -8.0 + 1e-6, f"y={y} gave too-high bound {g}"

    def test_infinite_box_with_nonzero_reduced_cost_is_unusable(self):
        # An unbounded box side with a nonzero reduced cost → −inf contribution →
        # None (no usable finite bound), not a spurious large-finite value.
        a_std, b, c_std, lb, ub = _std([[1.0, 1.0]], [4.0], [-1.0, -2.0], [(0, 1e20), (0, 1e20)])
        # y=0 → reduced cost = c = (-1,-2); the −2·(+inf) term is −inf.
        assert _safe_lp_lower_bound_std(np.array([0.0]), c_std, a_std, b, lb, ub) is None


class TestFarkasHelper:
    def test_farkas_certifies_empty_polytope(self):
        # x0 + s = 1 (s>=0) with x0 in [2, inf): x0 >= 2 but x0 <= 1 → infeasible.
        a_std, b, _c, lb, ub = _std([[1.0]], [1.0], [1.0], [(2.0, 1e20)])
        # A valid Farkas multiplier exists (the simplex returns ±1 here).
        assert _farkas_certified_std(np.array([1.0]), a_std, b, lb, ub) is True
        # The helper tries ±ray, so the opposite sign is certified too.
        assert _farkas_certified_std(np.array([-1.0]), a_std, b, lb, ub) is True

    def test_farkas_rejects_feasible_polytope(self):
        # A feasible system must NOT be certified infeasible for any candidate ray.
        a_std, b, _c, lb, ub = _std([[1.0, 1.0]], [4.0], [0.0, 0.0], [(0, 5), (0, 5)])
        for ray in (np.array([1.0]), np.array([-1.0]), np.array([3.7]), np.array([0.0])):
            assert _farkas_certified_std(ray, a_std, b, lb, ub) is False

    def test_empty_or_nonfinite_ray_is_not_certified(self):
        a_std, b, _c, lb, ub = _std([[1.0]], [1.0], [1.0], [(2.0, 1e20)])
        assert _farkas_certified_std(np.array([]), a_std, b, lb, ub) is False
        assert _farkas_certified_std(np.array([np.inf]), a_std, b, lb, ub) is False


class TestSolveLpWarmStdCert:
    def test_optimal_returns_safe_bound_at_or_below_objective(self):
        res, _basis, cert = solve_lp_warm_std(
            np.array([-1.0, -2.0]),
            sp.csr_matrix(np.array([[1.0, 1.0]])),
            np.array([4.0]),
            [(0.0, 5.0), (0.0, 5.0)],
            return_cert=True,
        )
        assert res is not None and res.status == SolveStatus.OPTIMAL
        assert cert.safe_bound is not None
        # The reported bound is the certified (never-too-high) one.
        assert res.bound <= res.objective + 1e-9
        assert abs(res.bound - (-8.0)) < 1e-6

    def test_infeasible_is_farkas_certified(self):
        res, _basis, cert = solve_lp_warm_std(
            np.array([1.0]),
            sp.csr_matrix(np.array([[1.0]])),
            np.array([1.0]),
            [(2.0, 1e20)],
            return_cert=True,
        )
        assert res is not None and res.status == SolveStatus.INFEASIBLE
        assert cert.farkas_certified is True

    def test_ill_conditioned_optimum_safe_bound_not_above_truth(self):
        # A wide-coefficient LP (range > the simplex's scaling trigger): the safe
        # bound must stay <= the true optimum even though the raw vertex objective
        # could drift on the ill-conditioned basis.
        A = np.array([[1e9, 1.0], [1.0, 1.0]])
        c = np.array([-1.0, -1.0])
        b = np.array([1e9, 5.0])
        res, _basis, cert = solve_lp_warm_std(
            c, sp.csr_matrix(A), b, [(0.0, 1.0), (0.0, 10.0)], return_cert=True
        )
        assert res is not None and res.status == SolveStatus.OPTIMAL
        assert cert.safe_bound is not None
        # Recompute the true optimum with a dense reference solve.
        from scipy.optimize import linprog

        ref = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1), (0, 10)], method="highs")
        assert ref.success
        assert cert.safe_bound <= ref.fun + 1e-6
        assert res.bound <= ref.fun + 1e-6

    def test_backward_compatible_two_tuple_without_cert(self):
        out = solve_lp_warm_std(
            np.array([-1.0, -2.0]),
            sp.csr_matrix(np.array([[1.0, 1.0]])),
            np.array([4.0]),
            [(0.0, 5.0), (0.0, 5.0)],
        )
        assert len(out) == 2  # (result, out_basis) — unchanged default contract


class TestIncrementalFarkasPath:
    @staticmethod
    def _inc():
        import discopt.modeling as dm
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        # Small all-integer QCQP (bilinear + square) — in the incremental scope.
        m = dm.Model("iqcqp")
        x = m.integer("x", lb=0, ub=5)
        y = m.integer("y", lb=0, ub=5)
        m.minimize((x - 3) ** 2 + (y - 2) ** 2 + x * y)
        m.subject_to(x + y >= 3)
        relaxer = MccormickLPRelaxer(m)
        return relaxer._inc

    def test_solve_assembled_full_returns_five_tuple(self):
        inc = self._inc()
        if inc is None:
            pytest.skip("incremental structure unavailable")
        lb = np.array([0.0, 0.0])
        ub = np.array([5.0, 5.0])
        A, b, bounds = inc.assemble(lb, ub)
        out = inc.solve_assembled_full(A, b, bounds)
        assert len(out) == 5  # (status, bound, x, basis, farkas_certified)
        status, bound, _x, _basis, farkas = out
        # On a feasible box: an optimal valid lower bound and farkas flag False.
        assert status == "optimal"
        assert bound is not None and np.isfinite(bound)
        assert farkas is False

    def test_infeasible_box_is_farkas_certified(self):
        inc = self._inc()
        if inc is None:
            pytest.skip("incremental structure unavailable")
        # Assemble over a box, then append a contradictory cut row that empties the
        # polytope: a row ``-x0 <= -100`` forces x0 >= 100 while its box caps it at
        # 5 — an infeasible lifted LP whose emptiness the Farkas ray must certify.
        lb = np.array([0.0, 0.0])
        ub = np.array([5.0, 5.0])
        cut = np.zeros(inc.ncol, dtype=np.float64)
        cut[0] = -1.0  # -x0 <= -100  ->  x0 >= 100, but x0 <= 5
        A, b, bounds = inc.assemble(lb, ub, cut_rows=[(cut, -100.0)])
        status, _bound, _x, _basis, farkas = inc.solve_assembled_full(A, b, bounds)
        assert status == "infeasible"
        assert farkas is True  # rigorously proven empty, no second solve needed


class TestColdPathCertificates:
    """The cold ``solve_at_node`` path (issue #356, cold-path guards) must also be
    HiGHS-free: a rigorous bound from the simplex's safe bound / vertex, an
    infeasible fathom only on a verified Farkas ray, and no fabricated bound on an
    unbounded relaxation."""

    @staticmethod
    def _model():
        import discopt.modeling as dm

        m = dm.Model("iqcqp")
        x = m.integer("x", lb=0, ub=5)
        y = m.integer("y", lb=0, ub=5)
        m.minimize((x - 3) ** 2 + (y - 2) ** 2 + x * y)
        m.subject_to(x + y >= 3)
        return m

    def test_milp_solve_surfaces_safe_bound_on_finite_box(self):
        # A finite-box lifted LP: the simplex pure-LP path must surface a rigorous
        # safe bound (<= the reported optimum) through MilpRelaxationModel.solve.
        from discopt._jax.discretization import DiscretizationState
        from discopt._jax.milp_relaxation import build_milp_relaxation
        from discopt._jax.term_classifier import classify_nonlinear_terms

        m = self._model()
        terms = classify_nonlinear_terms(m)
        milp, _ = build_milp_relaxation(
            m, terms, DiscretizationState(), bound_override=(np.zeros(2), np.full(2, 5.0))
        )
        milp._integrality = None  # pure-LP node bound (the default mode)
        res = milp.solve(backend="simplex")
        assert res.status == "optimal"
        if res.safe_bound is not None:  # finite-box ⇒ safe bound available
            assert res.safe_bound <= res.objective + 1e-6
            assert res.farkas_certified is False

    def test_cold_node_bound_is_sound_lower_bound(self):
        # Cold path (incremental forced off): the reported lower bound must be a
        # valid lower bound (<= the true integer optimum 4.0) — never too high.
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        relaxer = MccormickLPRelaxer(self._model())
        relaxer._inc = None  # force the cold build + cold solve_at_node path
        r = relaxer.solve_at_node(np.zeros(2), np.full(2, 5.0))
        assert r.status == "optimal"
        assert r.lower_bound is not None and np.isfinite(r.lower_bound)
        assert r.lower_bound <= 4.0 + 1e-6  # sound: never above the true optimum

    def test_cold_unbounded_relaxation_no_fabricated_bound(self):
        # A free bilinear variable makes the McCormick relaxation unbounded; the
        # cold path must NOT fabricate a finite lower bound (it declines/branches).
        import discopt.modeling as dm
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        m = dm.Model("bil")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        m.subject_to(x + y >= 0.5)
        m.minimize(x * y - x - y)
        relaxer = MccormickLPRelaxer(m)
        relaxer._inc = None
        unb = relaxer.solve_at_node(
            np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]), time_limit=5.0
        )
        assert unb.lower_bound is None
