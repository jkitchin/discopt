"""Routing tests for the QP backend seam (roadmap P0.4, mirrors the LP seam).

``_solve_qp`` tries matrix-form engines in order HiGHS -> POUNCE -> JAX QP
IPM, flipped to POUNCE-first on ``nlp_solver="pounce"``. The POUNCE engine
handles pure-continuous QPs only; models with integer variables are declined
so MIQPs stay on HiGHS / the B&B path.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402


def _build_qp() -> dm.Model:
    """min (x-1)^2 + (y-2)^2 s.t. x+y <= 2 -> optimum (0.5, 1.5), obj 0.5."""
    m = dm.Model("seam_qp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y <= 2)
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


class TestQPBackendSeam:
    def test_default_routes_to_highs(self, monkeypatch):
        pytest.importorskip("highspy")
        highs_calls = _spy(monkeypatch, "_solve_qp_highs")
        pounce_calls = _spy(monkeypatch, "_solve_qp_pounce")
        res = _build_qp().solve(time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4
        assert highs_calls == [True]
        assert pounce_calls == []  # HiGHS succeeded; POUNCE never consulted

    def test_pounce_request_routes_to_pounce(self, monkeypatch):
        pounce_calls = _spy(monkeypatch, "_solve_qp_pounce")
        res = _build_qp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4
        assert pounce_calls == [True]

    def test_fallback_to_pounce_when_highs_unavailable(self, monkeypatch):
        monkeypatch.setattr(S, "_solve_qp_highs", lambda *a, **k: None)
        pounce_calls = _spy(monkeypatch, "_solve_qp_pounce")
        res = _build_qp().solve(time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4
        assert pounce_calls == [True]

    def test_pounce_route_exposes_duals(self):
        res = _build_qp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert res.constraint_duals is not None

    def test_routes_agree_with_each_other(self):
        pytest.importorskip("highspy")
        r_h = _build_qp().solve(time_limit=30)
        r_p = _build_qp().solve(nlp_solver="pounce", time_limit=30)
        assert abs(r_h.objective - r_p.objective) < 1e-4
        for name in ("x", "y"):
            np.testing.assert_allclose(r_h.x[name], r_p.x[name], atol=1e-3)

    def test_pounce_engine_declines_miqp(self):
        """The POUNCE QP engine returns None for models with integer vars."""
        m = dm.Model("seam_miqp")
        x = m.integer("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize((x - 2.3) ** 2 + (y - 1.7) ** 2)
        assert S._solve_qp_pounce(m, time.perf_counter()) is None
        # The full solve still works (HiGHS MIQP or B&B fallback).
        # y is continuous, so the optimum is x=2, y=1.7 -> (2-2.3)^2 = 0.09.
        res = m.solve(time_limit=60)
        assert res.status in ("optimal", "feasible")
        assert abs(res.objective - 0.09) < 1e-3

    def test_infeasible_optimal_engine_result_falls_through(self, monkeypatch):
        """The feasibility guard rejects an engine result whose 'optimal'
        point violates its own constraints (observed with HiGHS QP), falling
        through to the next engine instead of returning a wrong answer."""
        from discopt.solvers import QPResult, SolveStatus

        def lying_engine(**kwargs):
            n = len(kwargs["c"])
            return QPResult(
                status=SolveStatus.OPTIMAL,
                x=np.full(n, 100.0),  # violates x+y<=2 and the bounds
                objective=-1e6,
            )

        import discopt.solvers.qp_highs as qp_highs

        monkeypatch.setattr(qp_highs, "solve_qp", lying_engine)
        res = _build_qp().solve(time_limit=30)
        # The guard discards the lying HiGHS result; POUNCE answers correctly.
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4

    def test_qp_infeasible_certificate_via_pounce_engine(self):
        m = dm.Model("seam_qp_infeasible")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y <= 1)
        m.subject_to(x + y >= 10)
        res = S._solve_qp_pounce(m, time.perf_counter())
        assert res is not None and res.status == "infeasible"
        cert = res.infeasibility_certificate
        assert cert is not None and cert.total_violation > 1.0
