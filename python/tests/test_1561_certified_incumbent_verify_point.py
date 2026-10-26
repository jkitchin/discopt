"""#1561 -- a certified incumbent must pass ``verify_point`` on the solved model.

The defect
----------

``tls2`` with every row multiplied by 1e6 was returned ``optimal`` with
``gap_certified=True`` on an incumbent that ``verify_point`` rejects: row 18
(``x2 - 3 x31 - 8 x32 - 15 x33 - 1 == 0``, scaled) violated by 6.669 where 1.0 is
allowed.

Root cause: the NLP-BB route's C-3 integer snap is refused when the snapped point
leaves the rows, and the route then reported the UNROUNDED point. Its exit gate
judged that point as computed, integer columns up to 1e-5 off an integer, and the
rows it satisfies are satisfied with slack bought by that fractionality.
``verify_point`` judges the integral realisation (#1380), so the two arbiters
disagreed: x31 off by ~2.2e-6 times the row's coefficient 3 (times 1e6) is the
6.669.

The two layers, and why both
----------------------------

1. **The route.** NLP-BB now judges the incumbent's integral realisation at the
   refine-adoption step (so a refine with integers pinned exactly beats a
   fractional incumbent whose snap fails) and at the exit gate; a point whose
   integral realisation fails is reported uncertified.
2. **The class.** ``Model.solve`` runs ``verify_point`` -- on the pre-solve
   snapshot of the declared rows -- on every certified result before handing it
   back, on the default path and the convex-kernel early return. A route whose
   gate drifts from ``verify_point`` again cannot certify what it rejects.

The fast tests below pin layer 2 deterministically by publishing a certified
point from a stubbed route. The ``slow`` test runs the real tls2 x1e6 solve; that
one is timing-dependent (the bad incumbent appeared on a loaded runner, see the
issue), so it asserts the invariant rather than a particular path.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import rescale_rows
from discopt.modeling.core import SolveResult
from discopt.validation.feasibility import verify_point

DATA = Path(__file__).parent / "data" / "minlplib_nl"

#: ``y`` is integer; ``y = 1 + 3e-6`` is inside the 1e-5 integrality tolerance and
#: satisfies ``x - 3 y == 1`` exactly with ``x = 4.000009``. Its integral
#: realisation ``y = 1`` leaves the row violated by 9e-6, where ``verify_point``
#: allows ``1e-6 * max(1, |J x|) = 4e-6``.
FRAC_Y = 1.0 + 3e-6
FRAC_X = 1.0 + 3.0 * FRAC_Y


def _model(nonlinear: bool):
    m = dm.Model("frac_certificate")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.integer("y", lb=0, ub=3)
    m.subject_to(x - 3 * y == 1, name="link")
    if nonlinear:
        # A nonlinear row puts the model outside the LP/MILP fast family, so
        # ``Model.solve`` takes its pre-solve snapshot evaluator (the arm under
        # test); the linear control below exercises the no-snapshot arm.
        m.subject_to(dm.exp(x) <= 1e3, name="nl")
    m.minimize(x + y)
    return m


def _certified(x, y, *, bound=4.0):
    r = SolveResult(
        status="optimal",
        objective=float(x + y),
        bound=bound,
        gap=0.0,
        x={"x": np.array(x), "y": np.array(y)},
        gap_certified=True,
    )
    r._set_bound(bound, valid=True, source="bnb_tree")
    return r


@pytest.fixture
def no_kernel(monkeypatch):
    """Keep the convex kernel out of the way, so ``solve_model`` is what returns."""
    import discopt.solvers._convex_kernel as ck

    monkeypatch.setattr(ck, "try_convex_solve", lambda *a, **k: None)


def _publish(monkeypatch, result):
    import discopt.solver as solver

    monkeypatch.setattr(solver, "solve_model", lambda model, **kw: result)


def test_the_fixture_point_fails_verify_point():
    """The probe fired: the published point is one ``verify_point`` rejects, its
    integral realisation is what fails, and the integral control passes."""
    m = _model(nonlinear=True)
    bad = verify_point(m, np.array([FRAC_X, FRAC_Y]))
    assert not bad.ok, bad
    assert "row 0" in str(bad.reason), bad
    assert verify_point(m, np.array([4.0, 1.0])).ok


@pytest.mark.parametrize("nonlinear", [True, False], ids=["snapshot", "no-snapshot"])
def test_a_certified_point_that_fails_verify_point_is_decertified(
    monkeypatch, no_kernel, nonlinear
):
    m = _model(nonlinear)
    _publish(monkeypatch, _certified(FRAC_X, FRAC_Y))
    res = m.solve(time_limit=5.0)

    assert res.status == "feasible", res.status
    assert res.gap_certified is False
    assert (res.solver_stats or {}).get("certificate/incumbent_unverified") == 1.0
    # Not a withhold: the loose #772 screen passes this point; only the
    # certificate is withdrawn. The dual bound stands.
    assert res.x is not None and res.objective is not None
    assert res.bound == pytest.approx(4.0)


@pytest.mark.parametrize("nonlinear", [True, False], ids=["snapshot", "no-snapshot"])
def test_an_integral_certified_point_keeps_its_certificate(monkeypatch, no_kernel, nonlinear):
    m = _model(nonlinear)
    _publish(monkeypatch, _certified(4.0, 1.0))
    res = m.solve(time_limit=5.0)

    assert res.status == "optimal" and res.gap_certified is True
    assert "certificate/incumbent_unverified" not in (res.solver_stats or {})


def test_the_convex_kernel_return_is_checked_too(monkeypatch):
    """The kernel returns before ``solve_model`` and the #772 screen; the
    backstop must run on that return as well."""
    import discopt.solvers._convex_kernel as ck

    m = _model(nonlinear=True)
    bad = _certified(FRAC_X, FRAC_Y)
    monkeypatch.setattr(ck, "try_convex_solve", lambda *a, **k: bad)
    res = m.solve(time_limit=5.0)

    assert res is bad, "the stub did not take the kernel's early return"
    assert res.status == "feasible" and res.gap_certified is False
    assert (res.solver_stats or {}).get("certificate/incumbent_unverified") == 1.0


def test_verify_point_judges_the_rows_of_the_evaluator_it_is_given():
    """``verify_point(..., evaluator=)`` is what lets ``Model.solve`` judge the
    pre-solve rows: the rows come from that evaluator, not from ``model``."""
    from discopt._tape_nlp_evaluator import make_evaluator

    looser = dm.Model("looser")
    lx = looser.continuous("x", lb=0.0, ub=10.0)
    ly = looser.integer("y", lb=0, ub=3)
    looser.subject_to(lx - 3 * ly <= 10, name="link")
    looser.subject_to(dm.exp(lx) <= 1e3, name="nl")
    looser.minimize(lx + ly)

    m = _model(nonlinear=True)
    x = np.array([FRAC_X, FRAC_Y])
    assert not verify_point(m, x).ok
    assert verify_point(m, x, evaluator=make_evaluator(looser)).ok


# ── a deterministic witness on the real NLP-BB route ────────────────────────
#
# Found by the full-suite A/B for the PR (the ``mo1442`` lexicographic
# epsilon-constraint sweep, ``test_1442``): an ε subproblem whose bound
# ``v0^2 + v1^2 + s <= 32 - 8e-6`` excludes the integer point (4, 4) by 8e-6.
# NLP-BB returned v = 3.9999995 -- inside INT_TOL of (4, 4), satisfying the
# row as computed -- and ``origin/main`` certified ``optimal`` with objective
# -8 on it, deterministically. The integral realisation (4, 4) violates the row
# by 8.1e-6, and the true optimum is -7 (the point (4, 3)). So this is not only
# an unverifiable certificate; the certified objective is wrong by 1.


def _eps_witness():
    m = dm.Model("eps1561")
    v = m.integer("v", shape=(2,), lb=0, ub=4)
    s = m.continuous("s", lb=0.0, ub=1e6)
    m.subject_to(dm.sum(v) >= 2)
    m.subject_to(v[0] ** 2 + v[1] ** 2 + s <= 32.0 - 8e-6)
    m.minimize(-(v[0] + v[1]) - s / 30000.0)
    return m


def test_nlpbb_does_not_certify_a_point_whose_integral_realisation_fails():
    m = _eps_witness()
    res = m.solve(time_limit=60.0)
    assert res.x, res.status
    x = np.concatenate(
        [np.atleast_1d(np.asarray(res.x[v.name], float)).ravel() for v in m._variables]
    )
    verdict = verify_point(m, x)
    if res.gap_certified or res.status == "optimal":
        # A certificate is only acceptable on a verified point, and then the
        # objective must be the true optimum, -7 (minus the slack bonus).
        assert verdict.ok, (res.status, res.objective, verdict.reason)
        assert res.objective == pytest.approx(-7.0, abs=1e-3)
    else:
        # The probe fired: the route's own gate is what withdrew it.
        assert not verdict.ok
        assert (res.solver_stats or {}).get("nlpbb/integral_incumbent_unverified") == 1.0


# ── a solve that EXTENDS the model (structure cuts) ─────────────────────────
#
# Structure cuts (``cut_recognizer``) append auxiliary columns and rows to the
# caller's model during the solve. The backstop must judge the point on the
# DECLARED columns and rows: flattening ``x`` over the extended column list and
# feeding it to the pre-solve evaluator is a length mismatch, which the first
# version of the backstop turned into a withdrawn certificate on a correct
# solve (``build_gas_network_minlp``: 19 declared columns, 21 after the cuts).


def _extending_route(monkeypatch, result, *, aux_in_x: bool):
    """A route that appends an aux column + a row on it, as structure cuts do.

    Returns a list that receives the extended column count on each call, so a
    test can prove the extension happened -- since #1610 the solve hands the
    caller's model back unextended, so it cannot be read off the model after.
    """
    import discopt.solver as solver

    fired: list[int] = []

    def _solve_model(model, **kw):
        u = model.continuous("u_cut_0", lb=0.0, ub=1e6)
        xv = next(v for v in model._variables if v.name == "x")
        model.subject_to(u == 2.0 * xv, name="cut_def_0")
        if aux_in_x:
            result.x = dict(result.x, u_cut_0=np.array(2.0 * float(result.x["x"])))
        fired.append(len(model._variables))
        return result

    monkeypatch.setattr(solver, "solve_model", _solve_model)
    return fired


@pytest.mark.parametrize("aux_in_x", [True, False], ids=["aux-in-x", "aux-not-in-x"])
def test_a_solve_that_appends_variables_keeps_a_valid_certificate(monkeypatch, no_kernel, aux_in_x):
    m = _model(nonlinear=True)
    n_declared = len(m._variables)
    fired = _extending_route(monkeypatch, _certified(4.0, 1.0), aux_in_x=aux_in_x)
    res = m.solve(time_limit=5.0)

    assert fired == [n_declared + 1], "the stub did not extend the model"
    # #1610: the extension is the solve's working state, not the caller's model.
    assert len(m._variables) == n_declared
    assert res.status == "optimal" and res.gap_certified is True, (
        res.status,
        res.solver_stats,
    )
    assert "certificate/incumbent_unverified" not in (res.solver_stats or {})


def test_an_extended_solve_still_decertifies_a_bad_declared_point(monkeypatch, no_kernel):
    """Judging the declared columns must not become judging nothing."""
    m = _model(nonlinear=True)
    _extending_route(monkeypatch, _certified(FRAC_X, FRAC_Y), aux_in_x=True)
    res = m.solve(time_limit=5.0)

    assert res.status == "feasible" and res.gap_certified is False
    assert (res.solver_stats or {}).get("certificate/incumbent_unverified") == 1.0


def test_verify_point_judges_the_declared_variables_it_is_given():
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.validation.feasibility import declared_variables

    m = _model(nonlinear=True)
    ev = make_evaluator(m)
    declared = declared_variables(m)
    m.continuous("u_cut_0", lb=0.0, ub=1.0)  # appended after the snapshot

    assert verify_point(m, np.array([4.0, 1.0]), evaluator=ev, variables=declared).ok
    assert not verify_point(m, np.array([FRAC_X, FRAC_Y]), evaluator=ev, variables=declared).ok
    wrong_len = verify_point(m, np.array([4.0, 1.0, 0.5]), evaluator=ev, variables=declared)
    assert not wrong_len.ok and "declared 2" in str(wrong_len.reason)
    with pytest.raises(ValueError, match="requires an explicit evaluator"):
        verify_point(m, np.array([4.0, 1.0]), variables=declared)


def test_declared_bounds_are_frozen_at_the_snapshot():
    """A bound tightened in place after the snapshot does not move the verdict."""
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.validation.feasibility import declared_variables

    m = _model(nonlinear=True)
    ev = make_evaluator(m)
    declared = declared_variables(m)
    m._variables[0].ub = np.asarray(3.0)  # x <= 3 now cuts the point x = 4
    assert verify_point(m, np.array([4.0, 1.0]), evaluator=ev, variables=declared).ok


# ── the reported instance ───────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_tls2_certified_incumbent_passes_verify_point(scale):
    m = dm.from_nl(str(DATA / "tls2.nl"))
    t = rescale_rows(m, scale) if scale != 1.0 else m
    res = t.solve(time_limit=float(os.environ.get("DISCOPT_1561_TL", "20")))
    assert res.x, f"no incumbent ({res.status})"
    x = np.concatenate(
        [np.atleast_1d(np.asarray(res.x[v.name], float)).ravel() for v in t._variables]
    )
    verdict = verify_point(t, x)
    if res.gap_certified or res.status == "optimal":
        assert verdict.ok, (
            f"certified {res.status} (objective {res.objective!r}) on a point "
            f"verify_point rejects: {verdict.reason}"
        )
        # tls2's optimum is 5.3 (minlplib.solu); rows x1e6 do not move it.
        assert res.objective == pytest.approx(5.3, rel=1e-4)
