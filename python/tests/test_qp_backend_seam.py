"""Routing tests for the QP backend seam (roadmap P0.4, mirrors the LP seam).

``_solve_qp`` is HiGHS-free (issue #359 / pure-Rust goal): a continuous QP is
solved by POUNCE, and a POUNCE failure or non-converged solve degrades to
discopt's JAX QP IPM — never to HiGHS. ``nlp_solver`` defaults to ``"pounce"``;
``"ipm"`` is a deprecated alias that routes to the same POUNCE-first path. The
POUNCE engine handles pure-continuous QPs only; models with integer variables
are declined so MIQPs stay on the self-hosted B&B path.
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
    def test_no_highs_qp_engine_exists(self):
        # qp_highs was removed (issue #359): there is no HiGHS QP engine to route
        # to, on any path.
        assert not hasattr(S, "_solve_qp_highs")
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("discopt.solvers.qp_highs")

    def test_default_routes_to_pounce(self, monkeypatch):
        # POUNCE is the universal default QP engine.
        pounce_calls = _spy(monkeypatch, "_solve_qp_pounce")
        res = _build_qp().solve(time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4
        assert pounce_calls == [True]

    def test_ipm_alias_routes_to_pounce(self, monkeypatch):
        # The deprecated "ipm" alias no longer opts into HiGHS — the QP path is
        # HiGHS-free, so it routes to POUNCE like the default.
        pounce_calls = _spy(monkeypatch, "_solve_qp_pounce")
        res = _build_qp().solve(nlp_solver="ipm", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4
        assert pounce_calls == [True]

    def test_pounce_request_routes_to_pounce(self, monkeypatch):
        pounce_calls = _spy(monkeypatch, "_solve_qp_pounce")
        res = _build_qp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert abs(res.objective - 0.5) < 1e-4
        assert pounce_calls == [True]

    def test_pounce_route_exposes_duals(self):
        res = _build_qp().solve(nlp_solver="pounce", time_limit=30)
        assert res.status == "optimal"
        assert res.constraint_duals is not None

    def test_routes_agree_with_each_other(self):
        # Both the default and the deprecated "ipm" alias route HiGHS-free to
        # POUNCE, so they must agree.
        r_i = _build_qp().solve(nlp_solver="ipm", time_limit=30)
        r_p = _build_qp().solve(nlp_solver="pounce", time_limit=30)
        assert abs(r_i.objective - r_p.objective) < 1e-4
        for name in ("x", "y"):
            np.testing.assert_allclose(r_i.x[name], r_p.x[name], atol=1e-3)

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

    def test_feasibility_guard_rejects_infeasible_optimal(self):
        """``_solve_qp_matrix`` rejects an engine result whose 'optimal' point
        violates its own constraints, returning ``None`` so the caller falls
        through instead of trusting a wrong answer."""
        from discopt.solvers import QPResult, SolveStatus

        def lying_engine(*args, **kwargs):
            n = len(kwargs["c"])
            return QPResult(
                status=SolveStatus.OPTIMAL,
                x=np.full(n, 100.0),  # violates x+y<=2 and the bounds
                objective=-1e6,
            )

        out = S._solve_qp_matrix(_build_qp(), time.perf_counter(), None, lying_engine, "POUNCE")
        assert out is None

    def test_convergence_guard_rejects_nonstationary_optimal(self):
        """``_solve_qp_matrix`` rejects a feasible but non-stationary 'optimal'
        (large KKT residual, issue #145 drift): the convergence guard returns
        ``None`` rather than trust the drifted objective."""
        from discopt.solvers import QPResult, SolveStatus

        def drifting_engine(*args, **kwargs):
            return QPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([0.5, 1.5]),  # feasible for x+y<=2, bounds ok
                objective=-1e6,  # drifted; true optimum is 0.5
                kkt_error=1e-2,  # >> _QP_KKT_RESIDUAL_TOL
            )

        out = S._solve_qp_matrix(_build_qp(), time.perf_counter(), None, drifting_engine, "POUNCE")
        assert out is None

    def test_pounce_no_result_logs_tracking_marker(self, monkeypatch, caplog):
        """When POUNCE yields no usable result, the HiGHS-free path logs the
        'qp-pounce-no-result' marker (issue #359 no-rescue tracking) before
        dropping to the JAX last resort."""
        monkeypatch.setattr(S, "_solve_qp_pounce", lambda *a, **k: None)
        with caplog.at_level("WARNING", logger="discopt.solver"):
            S._solve_qp(_build_qp(), time.perf_counter())
        assert any("qp-pounce-no-result" in r.message for r in caplog.records)

    def test_convergence_guard_accepts_converged_optimal(self):
        """A small KKT residual (normal converged POUNCE solve) passes the guard
        and the result is returned."""
        from discopt.solvers import QPResult, SolveStatus

        def good_engine(*args, **kwargs):
            return QPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([0.5, 1.5]),
                objective=0.5,
                kkt_error=1e-9,  # << _QP_KKT_RESIDUAL_TOL
            )

        out = S._solve_qp_matrix(_build_qp(), time.perf_counter(), None, good_engine, "POUNCE")
        assert out is not None  # guard accepted the converged result
        assert out.status == "optimal"

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
