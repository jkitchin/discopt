"""Routing tests for the LP backend seam (roadmap P0.4).

``_solve_lp`` tries matrix-form engines in SIMPLEX-first order, always. HiGHS has
been removed from the LP path entirely (issue #356); the HiGHS-free engines are
the pure-Rust warm-started simplex and POUNCE.

**This changed (#1454).** The order used to be POUNCE-first by default -- the
"POUNCE everywhere" reading of roadmap P0.4 -- inverted by a ``prefer_pounce``
argument computed as ``nlp_solver == "pounce"`` against a parameter whose DEFAULT
is ``"pounce"``. It was therefore true for every caller. POUNCE is an
interior-point method: it converges in VARIABLE space, so on an LP whose
objective coefficients span orders of magnitude a residual well inside any
feasibility tolerance is amplified into a large OBJECTIVE error, which the route
reported with ``gap_certified=True``.

Measured on ``min C*x + (1/C)*y  s.t. x + y >= 1, x,y in [0,1]`` (optimum
``x=0, y=1``, value ``1/C``), 24 comparisons over six ratios::

    C      simplex   POUNCE-first (old default)
    1e4    0.0       7.518e-07
    1e6    0.0       7.518e-05
    1e8    0.0       2.728e-05
    1e12   0.0       1.331e-05   -- and NEGATIVE, for an objective that is
                                    provably non-negative on the feasible box

all certified, crossing the documented ``abs=1e-6`` tolerance between 1e4 and
1e6. CLAUDE.md §1 puts the certificate above a routing preference, so the exact
engine now leads for a pure LP and POUNCE is the fallback. An explicit
``nlp_solver="pounce"`` no longer reorders the LP path -- POUNCE remains the NLP
engine everywhere it is the right one, but it does not certify LPs.

These tests pin:

  - default LP solves route to the exact simplex,
  - ``nlp_solver="ipm"`` is the same route (the back-compat alias still works),
  - when the simplex is unavailable the LP falls back to POUNCE (not the JAX IPM),
  - all routes agree on the optimum, and duals are exposed either way.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402


def _build_lp() -> dm.Model:
    """max 3x + 2y s.t. x+y<=4, x+3y<=6, 0<=x,y<=10 -> optimum (4,0), obj 12."""
    m = dm.Model("seam_lp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.maximize(3 * x + 2 * y)
    m.subject_to(x + y <= 4)
    m.subject_to(x + 3 * y <= 6)
    return m


def _spy(monkeypatch, name):
    calls = []
    orig = getattr(S, name)

    def wrapper(*a, **k):
        r = orig(*a, **k)
        calls.append(r is not None)
        return r

    monkeypatch.setattr(S, name, wrapper)
    return calls


class TestLPBackendSeam:
    @pytest.fixture(autouse=True)
    def _legacy_lp_route(self, monkeypatch):
        # ``_solve_lp``'s engine order is the legacy LP route. A pure LP goes to the
        # #1229 HiGHS route by default (covered in ``test_lp_milp_highs_route.py``),
        # so pin the opt-out that keeps this seam reachable.
        monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")

    def test_default_routes_to_the_exact_simplex(self, monkeypatch):
        # #1454: the EXACT engine leads for a pure LP. This assertion was the
        # reverse until an IPM answering every LP was measured certifying
        # objectives up to 7.5e-5 wrong (see the module docstring).
        simplex_calls = _spy(monkeypatch, "_solve_lp_simplex")
        pounce_calls = _spy(monkeypatch, "_solve_lp_pounce")
        res = _build_lp().solve(time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert simplex_calls == [True]
        assert pounce_calls == []  # simplex succeeded; POUNCE never consulted

    def test_ipm_alias_routes_to_simplex(self, monkeypatch):
        # The "ipm" back-compat alias still names the simplex route; since #1454
        # that is also the default, so this pins the alias rather than a choice.
        simplex_calls = _spy(monkeypatch, "_solve_lp_simplex")
        pounce_calls = _spy(monkeypatch, "_solve_lp_pounce")
        res = _build_lp().solve(nlp_solver="ipm", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert simplex_calls == [True]
        assert pounce_calls == []  # simplex succeeded; POUNCE never consulted

    def test_an_explicit_pounce_request_still_gets_the_exact_engine_for_an_lp(self, monkeypatch):
        """#1454: ``nlp_solver="pounce"`` no longer reorders the LP path.

        POUNCE stays the NLP engine everywhere it is the right one; what it may
        no longer do is CERTIFY a pure LP, because it cannot do so at arbitrary
        objective scale. Honouring the request here was indistinguishable from
        honouring the default anyway -- ``nlp_solver`` reaches ``solve_model``
        through ``**kwargs`` with the same value either way -- so there was no
        way to keep the explicit case without keeping the defect.
        """
        simplex_calls = _spy(monkeypatch, "_solve_lp_simplex")
        res = _build_lp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert simplex_calls == [True]

    def test_fallback_to_pounce_when_simplex_unavailable(self, monkeypatch):
        # In the simplex-first ("ipm") route, a missing simplex falls back to
        # POUNCE (not the JAX IPM).
        monkeypatch.setattr(S, "_solve_lp_simplex", lambda *a, **k: None)
        pounce_calls = _spy(monkeypatch, "_solve_lp_pounce")
        res = _build_lp().solve(nlp_solver="ipm", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert pounce_calls == [True]

    def test_pounce_route_exposes_duals(self):
        res = _build_lp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert res.constraint_duals is not None

    def test_routes_agree_with_each_other(self):
        r_s = _build_lp().solve(nlp_solver="ipm", time_limit=30)  # simplex route
        r_p = _build_lp().solve(nlp_solver="pounce", time_limit=30)
        assert abs(r_s.objective - r_p.objective) < 1e-5
        for name in ("x", "y"):
            np.testing.assert_allclose(r_s.x[name], r_p.x[name], atol=1e-4)


def _build_infeasible_lp() -> dm.Model:
    """x+y <= 1 and x+y >= 10 with x,y >= 0: infeasible."""
    m = dm.Model("seam_infeasible_lp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    m.subject_to(x + y <= 1)
    m.subject_to(x + y >= 10)
    return m


class TestInfeasibilityCertificateExposed:
    """An infeasible LP solved via POUNCE surfaces the certificate on
    SolveResult (roadmap P0.2).

    The engine wrappers are called directly here: at the full ``Model.solve``
    level, simple infeasible LPs are usually proved infeasible by FBBT bound
    tightening *before* any LP engine runs (returning "infeasible" with no
    certificate), so these tests exercise the engine→SolveResult plumbing that
    the internal LP consumers (OBBT, masters) actually hit.
    """

    def test_pounce_engine_attaches_certificate(self):
        import time

        res = S._solve_lp_pounce(_build_infeasible_lp(), time.perf_counter())
        assert res is not None and res.status == "infeasible"
        cert = res.infeasibility_certificate
        assert cert is not None
        # Gap between x+y<=1 and x+y>=10 forces total violation ~9.
        assert cert.total_violation > 1.0

    def test_simplex_engine_has_no_certificate(self):
        import time

        res = S._solve_lp_simplex(_build_infeasible_lp(), time.perf_counter())
        assert res is not None and res.status == "infeasible"
        # The simplex path does not compute the elastic Phase-1 witness.
        assert res.infeasibility_certificate is None
