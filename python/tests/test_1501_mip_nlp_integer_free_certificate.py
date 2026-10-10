"""Regression tests for #1501.

``solver="mip-nlp"`` on an integer-free model runs ONE local NLP. Before #1501 it
certified that solve whenever the model was classified convex:

* a point from an ``ITERATION_LIMIT`` exit (``min |x-2|`` stopped at ``x=2.94``)
  was returned as ``optimal`` with ``bound = objective``;
* ANY failed solve (``Error_In_Step_Computation`` on a feasible model,
  ``Diverging_Iterates`` on an unbounded LP) was returned as ``infeasible`` with
  ``gap_certified=True``.

The contract tested here is honesty, not strength: whatever the route returns, it
must not certify a wrong value and must not certify infeasibility of a feasible
model.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _abs1():
    m = dm.Model("abs")
    x = m.continuous("x", lb=-3, ub=3)
    m.minimize(dm.abs(x - 2))
    return m, 0.0


def _abs2():
    m = dm.Model("abs2")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(x + y >= 1)
    m.minimize(dm.abs(x - 2) + dm.abs(y + 1))
    return m, 0.0


def _max1():
    m = dm.Model("mx")
    x = m.continuous("x", lb=-3, ub=3)
    m.minimize(dm.maximum(x - 2, 2 - x))
    return m, 0.0


def _abs3():
    m = dm.Model("abs3")
    x = m.continuous("x", lb=-3, ub=3)
    m.minimize(dm.abs(x - 2) + 0.3 * dm.abs(x))
    return m, 0.6


def _unbounded_min():
    m = dm.Model("u")
    x = m.continuous("x")
    m.minimize(x)
    return m, None


def _unbounded_max():
    m = dm.Model("u2")
    x = m.continuous("x", lb=0)
    m.maximize(x)
    return m, None


def _solve(m, method, profile):
    opts = {"mip_nlp_profile": profile} if profile == "shot" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(time_limit=15, solver="mip-nlp", mip_nlp_method=method, mip_nlp_options=opts)


_METHODS = [
    ("oa", "default"),
    ("ecp", "default"),
    ("goa", "default"),
    ("lp_nlp_bb", "default"),
    ("oa", "shot"),
]


@pytest.mark.parametrize("method, profile", _METHODS)
@pytest.mark.parametrize("build", [_abs1, _abs2, _max1, _abs3], ids=lambda f: f.__name__)
def test_nonsmooth_convex_not_falsely_certified(build, method, profile):
    m, true_opt = build()
    r = _solve(m, method, profile)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    # The models are feasible: an infeasibility certificate is always false.
    assert r.status != "infeasible", info
    if r.status == "optimal" or r.gap_certified:
        assert r.objective is not None, info
        assert r.objective == pytest.approx(true_opt, abs=1e-4), info
    if r.bound is not None:
        assert r.bound <= true_opt + 1e-4, info
    if r.objective is not None:
        # Any reported incumbent value is attained, so it cannot beat the optimum.
        assert r.objective >= true_opt - 1e-4, info


@pytest.mark.parametrize("method, profile", _METHODS)
@pytest.mark.parametrize("build", [_unbounded_min, _unbounded_max], ids=lambda f: f.__name__)
def test_unbounded_lp_not_certified_infeasible(build, method, profile):
    m, _ = build()
    r = _solve(m, method, profile)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    # x = 0 is feasible, so "infeasible" is a false certificate.
    assert r.status != "infeasible", info


# ---------------------------------------------------------------------------
# Answer quality (#1501 completion). Honesty alone left every nonsmooth convex
# model uncertified (or with no point at all) under solver="mip-nlp", and the
# default-box LPs ``no_feasible_point``. The route now lifts monotone-position
# abs/max/min atoms to their exact smooth epigraph, and answers a pure LP with
# the default route's LP engine.
# ---------------------------------------------------------------------------

_ALL_METHODS = _METHODS + [("fp", "default")]


@pytest.mark.parametrize("method, profile", _METHODS)
@pytest.mark.parametrize("build", [_abs1, _abs2, _max1, _abs3], ids=lambda f: f.__name__)
def test_nonsmooth_convex_certified_at_the_true_optimum(build, method, profile):
    m, true_opt = build()
    r = _solve(m, method, profile)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    assert r.status == "optimal" and r.gap_certified, info
    assert r.objective == pytest.approx(true_opt, abs=1e-6), info
    assert r.bound <= r.objective, info
    # Only the declared columns come back -- the lifting's aux columns do not.
    assert set(r.x) == {v.name for v in m._variables}, sorted(r.x)


def _maxmin():
    m = dm.Model("mm")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(x + y <= 1)
    m.maximize(dm.minimum(x, y) - 0.1 * dm.abs(x - y))
    return m, 0.5


def _vec_abs():
    m = dm.Model("va")
    X = m.continuous("X", shape=(4,), lb=-5, ub=5)
    c = np.array([1.0, -2.0, 0.5, 3.0])
    m.subject_to(dm.sum(X) == 1)
    m.minimize(dm.sum(dm.abs(X - c)))
    return m, 1.5


def _nlp_abs():
    # A smooth convex NLP after lifting (not an LP): the NLP certificate path.
    m = dm.Model("na")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(x**2 + y**2 <= 4)
    m.minimize((x - 3) ** 2 + dm.abs(y - 0.5) + dm.maximum(x, y))
    return m, None


@pytest.mark.parametrize("method, profile", _METHODS)
@pytest.mark.parametrize("build", [_maxmin, _vec_abs, _nlp_abs], ids=lambda f: f.__name__)
def test_lifted_models_certified_and_consistent(build, method, profile):
    m, true_opt = build()
    r = _solve(m, method, profile)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    assert r.status == "optimal" and r.gap_certified, info
    maximize = m._objective.sense.value == "maximize"
    # The certificate invariant, with no slack: the published bound never
    # passes the published (re-verified) objective.
    assert (r.bound >= r.objective) if maximize else (r.bound <= r.objective), info
    if true_opt is not None:
        assert r.objective == pytest.approx(true_opt, abs=1e-6), info
    if true_opt is not None:
        return
    # No reference optimum: no sampled feasible point of the declared model may
    # beat the certified value (the models without one have no equality rows).
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.validation.feasibility import verify_point

    ev = make_evaluator(m)
    lo = np.concatenate([np.asarray(v.lb, dtype=float).ravel() for v in m._variables])
    hi = np.concatenate([np.asarray(v.ub, dtype=float).ravel() for v in m._variables])
    rng = np.random.default_rng(1501)
    sign = -1.0 if maximize else 1.0
    compared = 0
    for _ in range(400):
        p = rng.uniform(lo, hi)
        if verify_point(m, p).ok:
            val = sign * float(ev.evaluate_objective(p))  # model units
            assert sign * val >= sign * r.objective - 1e-6, (p, val, info)
            compared += 1
    assert compared > 0


def test_nonmonotone_atom_is_not_lifted():
    """``-|x|`` is concave: its epigraph is not exact, so nothing is lifted and
    the route stays honest (the model still carries its nonsmooth node)."""
    from discopt._relax.nonsmooth_lift import lift_nonsmooth_atoms

    m = dm.Model("ca")
    x = m.continuous("x", lb=-3, ub=2)
    m.minimize(-dm.abs(x))
    assert lift_nonsmooth_atoms(m) is m
    r = _solve(m, "oa", "default")
    assert r.status != "infeasible"
    if r.gap_certified:
        assert r.objective == pytest.approx(-3.0, abs=1e-6)
    else:
        assert r.bound is None or r.bound <= -3.0 + 1e-6


def test_lift_is_exact_on_sampled_points():
    """At ``t = atom(x)`` every lifting row holds and the objectives agree."""
    from discopt._relax.nonsmooth_lift import complete_lifted_point, lift_nonsmooth_atoms
    from discopt._tape_nlp_evaluator import make_evaluator

    m, _ = _maxmin()
    lifted = lift_nonsmooth_atoms(m)
    assert lifted is not m and len(lifted._nsl_aux) == 2
    ev_m, ev_l = make_evaluator(m), make_evaluator(lifted)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(50):
        xo = rng.uniform(-3, 3, size=2)
        xl = complete_lifted_point(lifted, xo)
        assert ev_l.evaluate_objective(xl) == pytest.approx(ev_m.evaluate_objective(xo), abs=1e-12)
        g = np.asarray(ev_l.evaluate_constraints(xl))
        # Row 0 is the model's own ``x + y <= 1``; the lifting rows follow.
        assert np.all(g[1:] <= 1e-12)
        checked += 1
    assert checked == 50


def test_warm_start_is_completed_for_the_lifted_columns(monkeypatch):
    import discopt._relax.nonsmooth_lift as nsl

    seen = []
    real = nsl.complete_lifted_point

    def spy(lifted, x0):
        out = real(lifted, x0)
        seen.append((np.asarray(x0).size, out.size))
        return out

    monkeypatch.setattr(nsl, "complete_lifted_point", spy)
    m, _ = _abs2()
    x, y = m._variables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(
            time_limit=15,
            solver="mip-nlp",
            mip_nlp_method="oa",
            initial_solution={x: 1.5, y: -0.5},
        )
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    # Two declared columns, two aux columns (one per |.|).
    assert seen == [(2, 4)]


@pytest.mark.parametrize("method, profile", _ALL_METHODS)
@pytest.mark.parametrize("build", [_unbounded_min, _unbounded_max], ids=lambda f: f.__name__)
def test_default_box_lp_matches_the_default_route(build, method, profile):
    """A column declared with no bounds lives in the default box (+-9.999e19),
    which since #1678 (b) means "no bound": the LP is ``unbounded``, as on the
    default route (#850 had certified ``optimal`` at the corner). mip-nlp returned
    ``no_feasible_point`` (``infeasible`` before b0dd7bb)."""
    m, _ = build()
    ref = m.solve(time_limit=15)
    m2, _ = build()
    r = _solve(m2, method, profile)
    assert (r.status, r.gap_certified) == (ref.status, ref.gap_certified)
    assert r.status == "unbounded"
    assert r.objective is None and r.x is None


@pytest.mark.parametrize("method, profile", _ALL_METHODS)
def test_infinite_box_lp_is_reported_unbounded(method, profile):
    m = dm.Model("ui")
    x = m.continuous("x", lb=-np.inf, ub=np.inf)
    m.minimize(x)
    r = _solve(m, method, profile)
    assert r.status == "unbounded", (r.status, r.objective, r.bound)
    assert r.objective is None and r.x is None
