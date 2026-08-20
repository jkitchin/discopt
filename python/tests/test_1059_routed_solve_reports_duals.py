"""#1059: the auto-route must not silently drop the duals the caller used to get.

The MIP-NLP family returns a point and a bound and no multipliers -- its master
is a MILP and its subproblems are solved on an integer-fixed model it does not
keep. Every other exit in ``solver.py`` reports duals, so when the route
graduated to default-ON the same ``.solve()`` that used to return
``constraint_duals`` started returning ``None``. Measured on both a convex MIQP
(``ProblemClass.MIQP`` -> QP path) and a convex ``exp()`` MINLP (-> NLP-BB), so
it was not confined to one class.

That is a silent capability loss, which is the defect class CLAUDE.md §3
forbids, and it is the reason ``_recover_nlp_duals_at_incumbent`` exists: pin the
integer columns at the incumbent and re-solve the resulting convex NLP for its
multipliers, exactly as ``_solve_nlp_bb`` already does at its own exit.

These tests fail before that recovery (``constraint_duals is None``) and pass
after. The ROUTE_ENV=0 arm is the control: it pins the value the routed arm has
to reproduce, so a recovery that returned *some* dict rather than the *right*
one would still fail here.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

ROUTE_ENV = "DISCOPT_CONVEX_MINLP_ROUTE"


def _miqp_with_a_known_dual():
    """min (x-0.5)^2 + (y-2)^2 s.t. x + y >= 5.

    With ``y`` at its incumbent 3 the relaxation is min (x-0.5)^2 s.t. x >= 2,
    whose multiplier on the row is 3. Same fixture as
    ``test_solver_duals.test_miqp_returns_relaxation_duals_at_incumbent``; the
    y-centre is offset so the optimum is unique and the dual is therefore not
    tie-dependent.
    """
    m = dm.Model("miqp_duals")
    m.continuous("x", lb=0.0, ub=10.0)
    m.integer("y", lb=0, ub=10)
    x, y = m._variables[0], m._variables[1]
    m.minimize((x - 0.5) ** 2 + (y - 2.0) ** 2)
    m.subject_to(x + y >= 5, name="c1")
    return m


def _convex_exp_minlp():
    m = dm.Model("convex_exp_duals")
    x = m.continuous("x", lb=0, ub=5)
    y = m.binary("y")
    m.minimize(dm.exp(x) + 3 * y)
    m.subject_to(x + y >= 1, name="c1")
    return m


def _scalar(name, duals):
    v = np.asarray(duals[name], dtype=float).ravel()
    assert v.size == 1, f"{name} is not a scalar dual: shape {v.shape}"
    return float(v[0])


@pytest.mark.parametrize("build", [_miqp_with_a_known_dual, _convex_exp_minlp])
def test_the_routed_solve_still_reports_constraint_duals(build, monkeypatch):
    """Route ON must report the same row dual that route OFF reports."""
    monkeypatch.setenv(ROUTE_ENV, "0")
    off = build().solve(time_limit=30)
    monkeypatch.setenv(ROUTE_ENV, "1")
    on = build().solve(time_limit=30)

    assert off.status == "optimal"
    assert on.status == "optimal"
    assert off.constraint_duals is not None, "control arm has no duals; nothing to compare"
    assert on.constraint_duals is not None, (
        "the routed solve dropped constraint_duals that the default path returns"
    )
    assert _scalar("c1", on.constraint_duals) == pytest.approx(
        _scalar("c1", off.constraint_duals), abs=1e-3
    )


def test_the_route_actually_ran_on_these_models():
    """§6: without this the test above passes vacuously if the route declines.

    ``algorithm_route`` is populated only by the auto-route, so its presence is
    proof the routed arm went through the MIP-NLP family rather than quietly
    landing back on the default path that always had duals.
    """
    checked = 0
    for build in (_miqp_with_a_known_dual, _convex_exp_minlp):
        r = build().solve(time_limit=30)
        assert r.algorithm_route is not None, f"{build.__name__} was not routed"
        assert "mip-nlp" in r.algorithm_route
        checked += 1
    assert checked == 2


def test_recovery_refuses_a_point_of_the_wrong_width():
    """A mismatched layout returns no duals rather than duals named after the
    wrong rows -- the #941 failure mode this module's guard exists to prevent."""
    from discopt.solver import _recover_nlp_duals_at_incumbent

    m = _miqp_with_a_known_dual()
    cd, bdl, bdu = _recover_nlp_duals_at_incumbent(m, np.zeros(7), time_budget=1.0)
    assert (cd, bdl, bdu) == (None, None, None)


def test_bound_duals_on_integer_columns_are_zeroed():
    """The fixing bounds price the act of fixing, not bound activity in the
    model the user declared. Same convention as ``_solve_nlp_bb``."""
    from discopt.solver import _recover_nlp_duals_at_incumbent

    m = _miqp_with_a_known_dual()
    x_flat = np.array([2.0, 3.0])
    cd, bdl, bdu = _recover_nlp_duals_at_incumbent(m, x_flat, time_budget=5.0)
    assert cd is not None, "recovery failed on the incumbent; nothing to assert"
    assert _scalar("c1", cd) == pytest.approx(3.0, abs=1e-3)
    for d in (bdl, bdu):
        assert d is not None
        assert np.all(np.asarray(d["y"], dtype=float) == 0.0)
