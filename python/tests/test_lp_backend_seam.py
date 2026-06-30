"""Routing tests for the LP backend seam (roadmap P0.4).

``_solve_lp`` tries matrix-form engines in POUNCE-first order by default
(``nlp_solver`` now defaults to ``"pounce"`` — POUNCE everywhere), and flips
to simplex-first when the user opts into the back-compat ``nlp_solver="ipm"``
alias. HiGHS has been removed from the LP path entirely (issue #356); the
HiGHS-free engines are the pure-Rust warm-started simplex and POUNCE. These
tests pin:

  - default LP solves route to POUNCE (the universal default),
  - ``nlp_solver="ipm"`` opts into the simplex-first route,
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
    def test_default_routes_to_pounce(self, monkeypatch):
        # POUNCE is now the universal default: the default solve must route to
        # the POUNCE engine and must NOT consult the simplex.
        simplex_calls = _spy(monkeypatch, "_solve_lp_simplex")
        pounce_calls = _spy(monkeypatch, "_solve_lp_pounce")
        res = _build_lp().solve(time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert pounce_calls == [True]
        assert simplex_calls == []  # POUNCE is default; simplex never consulted

    def test_ipm_alias_routes_to_simplex(self, monkeypatch):
        # The "ipm" back-compat alias opts into the simplex-first route.
        simplex_calls = _spy(monkeypatch, "_solve_lp_simplex")
        pounce_calls = _spy(monkeypatch, "_solve_lp_pounce")
        res = _build_lp().solve(nlp_solver="ipm", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert simplex_calls == [True]
        assert pounce_calls == []  # simplex succeeded; POUNCE never consulted

    def test_pounce_request_routes_to_pounce(self, monkeypatch):
        pounce_calls = _spy(monkeypatch, "_solve_lp_pounce")
        res = _build_lp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 12.0) < 1e-5
        assert pounce_calls == [True]

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
