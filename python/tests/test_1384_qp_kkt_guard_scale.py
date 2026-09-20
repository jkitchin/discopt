"""Issue #1384: the QP convergence guard must be scale-relative.

``_QP_KKT_RESIDUAL_TOL`` was compared against ``QPResult.kkt_error`` as an
absolute number. ``kkt_error`` is POUNCE's residual in the *model's own units*,
and the stationarity residual ``Qx + c + A'y - z`` carries the units of the
objective gradient — so writing the same QP in joules instead of kilojoules
multiplies the residual by 1e3 without moving the point. Read absolutely, the
guard refused correct answers to every convex QP whose objective scale was above
~1e4 and the route reported ``status="error"``.

Baseline behaviour (``1225337f``, before this fix): ``min 1e4*(x-3)^2`` over
``[0,10]`` returns ``status="error"``, ``objective=None``, ``bound=None`` — the
first three parametrizations of
``test_convex_box_qp_solves_at_every_objective_scale`` fail there. The guard's
teeth are what #145 bought, so the tests below also pin that a genuinely
non-stationary point is *still* refused at every scale.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402


@pytest.mark.parametrize("scale", [1e4, 1e5, 1e8, 1e3, 1.0])
def test_convex_box_qp_solves_at_every_objective_scale(scale):
    """``min s*(x-3)^2`` over ``[0,10]`` has optimum 0 at x=3 for every ``s``.

    Oracle: closed form. A scale-free solver must return the same point and the
    same (scaled) objective at every ``s``.
    """
    m = dm.Model(f"scaled_qp_{scale:g}")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(scale * (x - 3) ** 2)
    r = m.solve(time_limit=30)

    assert r.status == "optimal", f"scale={scale:g} -> {r.status} (error={r.error})"
    assert r.objective is not None
    # Objective floor is 0; compare relative to the objective scale, since an
    # absolute 1e-6 is itself a scale-dependent yardstick.
    assert abs(r.objective) <= 1e-6 * scale
    assert r.bound is not None and r.bound <= 1e-6 * scale
    assert abs(float(r.x["x"]) - 3.0) < 1e-6


def test_scale_carried_by_the_variable_not_the_coefficient():
    """The same defect reachable through variable scaling: ``min (x-3e6)^2``."""
    m = dm.Model("var_scaled_qp")
    x = m.continuous("x", lb=0, ub=3e7)
    m.minimize((x - 3e6) ** 2)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", f"{r.status} (error={r.error})"
    assert r.objective is not None and abs(r.objective) <= 1e-3
    assert abs(float(r.x["x"]) - 3e6) < 1.0


def _build_qp() -> dm.Model:
    """min (x-1)^2 + (y-2)^2 s.t. x+y <= 2 -> optimum (0.5, 1.5), obj 0.5."""
    m = dm.Model("guard_qp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y <= 2)
    return m


def _scaled_qp(scale: float) -> dm.Model:
    """The same QP with the objective multiplied by ``scale``."""
    m = dm.Model(f"guard_qp_{scale:g}")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(scale * ((x - 1) ** 2 + (y - 2) ** 2))
    m.subject_to(x + y <= 2)
    return m


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
def test_guard_still_refuses_a_drifted_point_at_every_scale(scale):
    """#145's teeth survive the relative yardstick.

    The engine returns a feasible point with a residual 1e4x larger than the
    allowance *for that scale*; it must still be refused, so the route reports
    ``error`` rather than the drifted objective.
    """
    from discopt.solvers import QPResult, SolveStatus

    # Stationarity scale of the scaled QP: ||Qx||_inf = 3*scale, ||c||_inf = 4*scale.
    allowance = S._QP_KKT_RESIDUAL_TOL * 4.0 * scale

    def drifting_engine(*args, **kwargs):
        return QPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.5, 1.5]),  # feasible for x+y<=2 and the bounds
            objective=-1e6 * scale,  # drifted; the true optimum is 0.5*scale
            kkt_error=1e4 * allowance,
        )

    reasons: list[str] = []
    out = S._solve_qp_matrix(
        _scaled_qp(scale),
        time.perf_counter(),
        None,
        drifting_engine,
        "POUNCE",
        reject_reason=reasons,
    )
    assert out is None, "a drifted point was accepted"
    assert reasons and "non-stationary" in reasons[0]


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8])
def test_guard_accepts_a_converged_point_at_every_scale(scale):
    """A residual well inside the scale-relative allowance is accepted."""
    from discopt.solvers import QPResult, SolveStatus

    allowance = S._QP_KKT_RESIDUAL_TOL * 4.0 * scale

    def good_engine(*args, **kwargs):
        return QPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.5, 1.5]),
            objective=0.5 * scale,
            kkt_error=1e-3 * allowance,
        )

    out = S._solve_qp_matrix(_scaled_qp(scale), time.perf_counter(), None, good_engine, "POUNCE")
    assert out is not None and out.status == "optimal"


class TestStationarityScale:
    """Unit tests for the yardstick itself."""

    def test_floors_at_one_so_o1_problems_keep_the_absolute_test(self):
        Q = np.array([[2.0, 0.0], [0.0, 2.0]])
        c = np.array([-2.0, -4.0])
        x = np.array([0.5, 1.5])
        # max(1, ||Qx||=3, ||c||=4) == 4; a tiny-data QP floors at 1.0.
        assert S._qp_stationarity_scale(Q, c, x) == pytest.approx(4.0)
        assert S._qp_stationarity_scale(
            np.array([[1e-9]]), np.array([1e-9]), np.array([1e-9])
        ) == pytest.approx(1.0)

    def test_scales_linearly_with_the_objective(self):
        Q = np.array([[2.0, 0.0], [0.0, 2.0]])
        c = np.array([-2.0, -4.0])
        x = np.array([0.5, 1.5])
        base = S._qp_stationarity_scale(Q, c, x)
        for s in (1e3, 1e6, 1e9):
            assert S._qp_stationarity_scale(s * Q, s * c, x) == pytest.approx(s * base)

    def test_does_not_use_the_gradient_norm(self):
        """At an interior optimum ``Qx + c == 0``; the scale must not collapse.

        This is the trap the yardstick is written around: ``||Qx + c||`` is ~0
        exactly when the point is optimal, so using it would restore the absolute
        test on the very instances #1384 is about.
        """
        s = 1e6
        Q = np.array([[2.0 * s]])
        c = np.array([-6.0 * s])
        x = np.array([3.0])  # the optimum: Qx + c == 0
        assert np.allclose(Q @ x + c, 0.0)
        assert S._qp_stationarity_scale(Q, c, x) >= 1e6

    def test_multipliers_enter_the_scale(self):
        Q = np.zeros((1, 1))
        c = np.zeros(1)
        x = np.zeros(1)
        assert S._qp_stationarity_scale(Q, c, x, row_duals=np.array([5e5])) == pytest.approx(5e5)
        assert S._qp_stationarity_scale(Q, c, x, col_duals=np.array([-7e5])) == pytest.approx(7e5)

    def test_non_finite_scale_falls_back_to_the_absolute_test(self):
        """A NaN/inf yardstick would make the guard vacuous; refuse the allowance."""
        Q = np.array([[np.inf]])
        c = np.array([1.0])
        x = np.array([1.0])
        assert S._qp_stationarity_scale(Q, c, x) == 1.0


def test_terminal_error_carries_the_reason(monkeypatch):
    """``SolveResult.error`` must say why, not only the log (#1384)."""
    from discopt.solvers import QPResult, SolveStatus

    def drifting_engine(*args, **kwargs):
        return QPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([0.5, 1.5]),
            objective=-1e6,
            kkt_error=1e-2,  # >> the allowance at this O(1) scale
        )

    import functools

    monkeypatch.setattr(
        S,
        "_solve_qp_pounce",
        lambda model, t_start, time_limit=None, reject_reason=None: S._solve_qp_matrix(
            model, t_start, time_limit, drifting_engine, "POUNCE", reject_reason=reject_reason
        ),
    )
    del functools
    res = S._solve_qp(_build_qp(), time.perf_counter())
    assert res.status == "error"
    assert res.objective is None and res.bound is None
    assert res.error is not None
    assert "qp-pounce-no-result" in res.error
    assert "non-stationary" in res.error
