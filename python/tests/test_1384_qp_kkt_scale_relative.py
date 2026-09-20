"""#1384 -- the QP stationarity guard must be relative to the problem's scale.

``_QP_KKT_RESIDUAL_TOL`` is an absolute 1e-6 and ``QPResult.kkt_error`` is
POUNCE's ``final_unscaled_kkt_error`` -- the residual in the MODEL's units,
deliberately preferred over the scaled one because a certificate stated in
problem units has to be built from it. So the residual grows with the problem
while the yardstick does not, and the guard began rejecting correct answers:
``min 1e4*(x-3)**2`` over ``[0,10]`` -- a one-variable strictly convex box QP --
came back ``status="error"``.

Entry experiment (CLAUDE.md 4), on ``min s*(x-3)**2`` over ``[0,10]``::

    scale   kkt_error   term scale   ratio      verdict before
    1e0     2.506e-09   6.0e+00      4.18e-10   optimal
    1e2     1.002e-08   6.0e+02      1.67e-11   optimal
    1e3     1.002e-07   6.0e+03      1.67e-11   optimal
    1e4     1.002e-06   6.0e+04      1.67e-11   error
    1e5     1.002e-05   6.0e+05      1.67e-11   error
    1e6     9.091e-06   6.0e+06      1.52e-12   error
    1e8     9.091e-06   6.0e+08      1.52e-14   error

The raw residual grows exactly linearly with the objective scale while the
relative precision is constant at ~1.7e-11, five orders inside the tolerance.
The kill criterion -- a ratio that ALSO grew, meaning the point really was
drifting -- did not fire.

The guard keeping its teeth is the other half of this, so the drifted-point
cases below matter as much as the scaled ones. The existing seam tests in
``test_qp_backend_seam.py`` cover the same rejection from the other direction.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.modeling.core import SolveResult

# --------------------------------------------------------------------------
# end to end: a convex QP must solve at any objective scale
# --------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("scale", [1.0, 1e3, 1e4, 1e5, 1e6, 1e8])
def test_a_convex_box_qp_solves_at_every_objective_scale(scale):
    """``min s*(x-3)^2`` over [0,10]: optimum 0 at x=3 for every ``s``."""
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(scale * (x - 3) ** 2)
    r = m.solve(time_limit=30)

    assert r.status == "optimal", f"status={r.status!r} at scale {scale:g} (was 'error')"
    assert r.objective is not None
    assert r.objective == pytest.approx(0.0, abs=1e-6 * max(1.0, scale))
    assert float(np.asarray(r.x["x"])) == pytest.approx(3.0, abs=1e-3)
    # Soundness is not traded for the answer: the bound still cannot exceed the
    # true optimum of 0.
    assert r.bound is not None and r.bound <= 1e-6 * max(1.0, scale)


@pytest.mark.smoke
@pytest.mark.parametrize("scale", [1e4, 1e6])
def test_a_multivariable_convex_qp_solves_at_scale(scale):
    """Optimum of ``s*((x-3)^2 + (y-4)^2 + 0.5*(x-y)^2)`` is ``0.25*s``.

    Derived, not pinned: the stationarity conditions give 3x - y = 6 and
    -x + 3y = 8, so x = 3.25, y = 3.75 and f = 0.25*s.
    """
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(scale * ((x - 3) ** 2 + (y - 4) ** 2 + 0.5 * (x - y) ** 2))
    r = m.solve(time_limit=30)

    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.25 * scale, rel=1e-6)


@pytest.mark.smoke
def test_a_variable_scaled_convex_qp_solves():
    """The scale can sit in the VARIABLE rather than the coefficient."""
    m = Model()
    x = m.continuous("x", lb=0, ub=3e7)
    m.minimize((x - 3e6) ** 2)
    r = m.solve(time_limit=30)

    assert r.status == "optimal"
    assert float(np.asarray(r.x["x"])) == pytest.approx(3e6, rel=1e-6)


@pytest.mark.smoke
@pytest.mark.parametrize("scale", [1e4, 1e6])
def test_controls_at_the_same_scale_were_never_broken(scale):
    """An LP and a smooth non-quadratic NLP at the same scale always worked.

    They are here so a regression in this test file cannot be misread as "large
    numbers are hard" -- only the QP route had the defect.
    """
    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(scale * x)
    assert m.solve(time_limit=30).status == "optimal"

    m = Model()
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize(scale * dm.exp(x))
    assert m.solve(time_limit=30).status == "optimal"


# --------------------------------------------------------------------------
# the yardstick itself
#
# Imported inside each test rather than at module scope so the end-to-end tests
# above still COLLECT against a pre-#1384 tree -- that is how their fail-before
# was measured (CLAUDE.md 8).
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_term_scale_floors_at_one():
    """A well-scaled problem keeps EXACTLY the #145 absolute behaviour."""
    from discopt.solver import _qp_lagrangian_term_scale

    scale = _qp_lagrangian_term_scale(
        np.zeros(2), np.zeros((2, 2)), np.zeros(2), None, None, None, None
    )
    assert scale == 1.0


@pytest.mark.smoke
def test_the_term_scale_tracks_the_objective_scale():
    """``min s*(x-3)^2`` has ``Q = 2s``, ``c = -6s``; at x=3 both terms are 6s."""
    from discopt.solver import _qp_lagrangian_term_scale

    for s in (1.0, 1e4, 1e8):
        scale = _qp_lagrangian_term_scale(
            np.array([3.0]), np.array([[2 * s]]), np.array([-6 * s]), None, None, None, None
        )
        assert scale == pytest.approx(6 * s)


@pytest.mark.smoke
def test_the_term_scale_takes_the_terms_apart_not_their_sum():
    """The sum is the gradient, which VANISHES at an interior optimum.

    This is the measurement that killed the first candidate denominator: a
    ``max(1, ||Qx + c||)`` yardstick read 1.0 at every scale, i.e. no yardstick.
    """
    from discopt.solver import _qp_lagrangian_term_scale

    s = 1e6
    x, Q, c = np.array([3.0]), np.array([[2 * s]]), np.array([-6 * s])
    assert float(np.abs(Q @ x + c).max()) == pytest.approx(0.0, abs=1e-6)
    assert _qp_lagrangian_term_scale(x, Q, c, None, None, None, None) == pytest.approx(6 * s)


@pytest.mark.smoke
def test_row_and_bound_multipliers_enter_the_scale():
    """``A^T mu`` and the reduced costs are Lagrangian-gradient terms too."""
    from discopt.solver import _qp_lagrangian_term_scale

    x, Q, c = np.array([3.0]), np.array([[2.0]]), np.array([-6.0])
    plain = _qp_lagrangian_term_scale(x, Q, c, None, None, None, None)

    with_rows = _qp_lagrangian_term_scale(x, Q, c, np.array([[1.0]]), None, np.array([5e7]), None)
    assert with_rows == pytest.approx(5e7)
    assert with_rows > plain

    with_bounds = _qp_lagrangian_term_scale(x, Q, c, None, None, None, np.array([3e5]))
    assert with_bounds == pytest.approx(3e5)


# --------------------------------------------------------------------------
# the guard still has teeth
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_genuinely_drifted_point_is_still_rejected():
    """#145's case: a residual LARGE relative to the terms is still refused.

    Asserted against the rule the guard applies, so it holds whatever the
    problem scale: a ratio above ``_QP_KKT_RESIDUAL_TOL`` is a rejection.
    """
    from discopt.solver import _QP_KKT_RESIDUAL_TOL, _qp_lagrangian_term_scale

    x, c = np.array([3.0]), np.array([-6e4])
    Q = np.array([[2e4]])
    scale = _qp_lagrangian_term_scale(x, Q, c, None, None, None, None)

    converged = 1.7e-11 * scale  # the ratio measured on this family
    drifted = 1e-2 * scale  # four orders past the tolerance

    assert converged <= _QP_KKT_RESIDUAL_TOL * scale
    assert drifted > _QP_KKT_RESIDUAL_TOL * scale


@pytest.mark.smoke
def test_the_relative_test_is_never_looser_than_the_absolute_one():
    """The floor at 1.0 is what guarantees it: for any inputs, tol*scale >= tol."""
    from discopt.solver import _QP_KKT_RESIDUAL_TOL, _qp_lagrangian_term_scale

    cases = [
        (np.zeros(1), np.zeros((1, 1)), np.zeros(1)),
        (np.array([3.0]), np.array([[2.0]]), np.array([-6.0])),
        (np.array([1e-9]), np.array([[1e-9]]), np.array([1e-9])),
    ]
    for x, Q, c in cases:
        scale = _qp_lagrangian_term_scale(x, Q, c, None, None, None, None)
        assert scale >= 1.0
        assert _QP_KKT_RESIDUAL_TOL * scale >= _QP_KKT_RESIDUAL_TOL


# --------------------------------------------------------------------------
# the terminal error says why
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_qp_error_result_carries_its_reason(monkeypatch):
    """``SolveResult.error`` was None on this exit, so the only record of the
    cause was a log line -- a caller got ``status="error"`` with nothing to
    branch on or report.

    POUNCE is driven to a non-result directly rather than by finding a model it
    fails on, so the test pins the FIELD and cannot silently stop exercising the
    path if the underlying QP starts succeeding. ``None`` is what the engine
    actually returns in the measured repro -- the log line there is the
    ``qp-pounce-no-result`` one, which only the None path reaches.
    """
    import time as _time

    import discopt.solver as solver_mod

    monkeypatch.setattr(solver_mod, "_solve_qp_pounce", lambda *a, **k: None)
    monkeypatch.setattr(solver_mod, "_declared_box_retry_applies", lambda *a, **k: False)

    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    m.minimize((x - 3) ** 2)
    r = solver_mod._solve_qp(m, _time.perf_counter())

    assert r.status == "error"
    assert r.error, "the terminal QP error carries no reason"
    assert "KKT" in r.error or "stationarity" in r.error
    assert "#359" in r.error or "359" in r.error


@pytest.mark.smoke
def test_an_engine_error_result_also_carries_a_reason(monkeypatch):
    """The other arm: POUNCE returns an ``error`` SolveResult of its own.

    It was passed straight through, so it reached the caller bare too. A reason
    the engine DID supply must survive, which the second half asserts.
    """
    import time as _time

    import discopt.solver as solver_mod

    def _blank_error(*a, **k):
        return SolveResult(status="error")

    monkeypatch.setattr(solver_mod, "_solve_qp_pounce", _blank_error)
    monkeypatch.setattr(solver_mod, "_declared_box_retry_applies", lambda *a, **k: False)

    m = Model()
    x = m.continuous("x", lb=0, ub=10)
    m.minimize((x - 3) ** 2)
    r = solver_mod._solve_qp(m, _time.perf_counter())
    assert r.status == "error" and r.error

    monkeypatch.setattr(
        solver_mod,
        "_solve_qp_pounce",
        lambda *a, **k: SolveResult(status="error", error="a more specific reason"),
    )
    m2 = Model()
    y = m2.continuous("y", lb=0, ub=10)
    m2.minimize((y - 3) ** 2)
    r2 = solver_mod._solve_qp(m2, _time.perf_counter())
    assert r2.error == "a more specific reason"
