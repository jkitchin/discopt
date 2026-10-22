"""#1775 G4.2: RIC and configuration enumeration (spec §5.3, §5.4).

Before this task both ``gdp_method="ric"`` and ``"enumerate"`` were reserved and
refused. RIC is LOA with the outer approximation taken out: the lowered model's
linear rows plus no-goods as the master (a relaxation whatever the convexity),
global subproblems, and a no-good only on a certificate. Enumerate solves every
configuration globally, refuses above ``max_configurations`` instead of
truncating, and publishes an LB only once every configuration has one.
"""

from __future__ import annotations

import math
import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.enumerate import (
    Enumerate,
    configurations,
    count_configurations,
)
from discopt.gdp.algorithms.init import GDPSpace
from discopt.gdp.algorithms.ric import RIC

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _nonconvex():
    """min x + y over [x^2 + y^2 >= 9] v [x + y >= 5]; optimum 3."""
    m = dm.Model("nonconvex")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x**2 + y**2 >= 9], [x + y >= 5]], name="d")
    m.minimize(x + y)
    return m


def _nonlinear_objective(sense: str = "min"):
    m = dm.Model(f"nlobj_{sense}")
    x = m.continuous("x", lb=0.5, ub=4)
    y = m.continuous("y", lb=0.5, ub=4)
    m.either_or([[x * y >= 3, x <= 2.5], [x + 2 * y >= 4, y <= 3]], name="d")
    obj = (x - 2) ** 2 + x * y - y
    if sense == "min":
        m.minimize(obj)
    else:
        m.maximize(-obj)
    return m


def _or_model():
    """OR semantics: three configurations ({A}, {B}, {A, B}); optimum 2 at B."""
    m = dm.Model("or")
    x = m.continuous("x", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x >= 4)
    b = m.make_disjunct("B")
    b.subject_to(x <= 1)
    m.add_disjunction([a, b], name="d", semantics="or")
    m.minimize((x - 2) ** 2 + 3 * a.indicator.variable + b.indicator.variable)
    return m


def _nested():
    """[A: x >= 1, nested [A1: y >= 2] v [A2: y <= 1]] v [B: x <= 0]: three
    configurations (A+A1, A+A2, B)."""
    m = dm.Model("nested")
    x = m.continuous("x", lb=-1, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    a = m.make_disjunct("A")
    a.subject_to(x >= 1)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 2)
    a2 = m.make_disjunct("A2")
    a2.subject_to(y <= 1)
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to(x <= 0)
    m.add_disjunction([a, b], name="outer")
    m.minimize((x - 2) ** 2 + (y - 1.5) ** 2)
    return m


# ── enumerate: the configuration set ─────────────────────────────────────────


def test_counts_match_the_configurations():
    for mk, want in ((_nonconvex, 2), (_or_model, 3), (_nested, 3)):
        space = GDPSpace.from_model(mk())
        got = list(configurations(space))
        assert count_configurations(space) == len(got) == want
        assert len({tuple(id(d) for d in c) for c in got}) == want


def test_enumerate_refuses_instead_of_truncating():
    with pytest.raises(ValueError, match="more than max_configurations=1"):
        Enumerate(_nonconvex(), max_configurations=1)
    with pytest.raises(ValueError, match="max_configurations must be"):
        Enumerate(_nonconvex(), max_configurations=0)


@pytest.mark.parametrize("mk", [_nonconvex, _or_model, _nested, _nonlinear_objective])
@pytest.mark.parametrize("method", ["ric", "enumerate"])
def test_certified_agreement_with_the_spatial_bb(mk, method):
    ref = _quiet(mk().solve, gdp_method="big-m", time_limit=60)
    r = _quiet(mk().solve, gdp_method=method, time_limit=60)
    assert ref.status == "optimal" and r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)
    assert r.bound <= r.objective + 1e-9
    assert r.algorithm_route.startswith(f"gdpopt-{method}")


@pytest.mark.parametrize("method", ["ric", "enumerate"])
def test_maximize_is_in_model_units(method):
    r = _quiet(_nonlinear_objective("max").solve, gdp_method=method, time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(-0.5625, abs=1e-5)
    assert r.bound >= r.objective - 1e-9


def test_enumerate_publishes_no_bound_before_the_last_configuration():
    r = _quiet(Enumerate(_nonconvex(), time_limit=60, iteration_limit=1).run)
    assert r.bound is None and r.status == "feasible" and not r.gap_certified
    assert r.solver_stats["n_configurations_solved"] == 1


def test_enumerate_proves_an_infeasible_gdp():
    m = dm.Model("infeasible")
    x = m.continuous("x", lb=0, ub=1)
    m.either_or([[x**2 >= 4], [x >= 2]], name="d")
    m.minimize(x)
    r = _quiet(Enumerate(m, time_limit=60).run)
    assert r.status == "infeasible" and r.gap_certified


def test_an_unresolved_configuration_blocks_optimal(monkeypatch):
    """One configuration proves nothing: no bound (it has none), never optimal."""
    import discopt.gdp.algorithms.subproblem as sp
    from discopt.gdp.algorithms.subproblem import SubproblemResult

    real = sp.solve_subproblem
    calls = []

    def first_unresolved(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return SubproblemResult("unresolved", reason="test")
        return real(*a, **k)

    monkeypatch.setattr(sp, "solve_subproblem", first_unresolved)
    r = _quiet(Enumerate(_nonconvex(), time_limit=60).run)
    assert len(calls) == 2
    assert r.status == "feasible" and not r.gap_certified and r.bound is None


# ── RIC ──────────────────────────────────────────────────────────────────────


def test_ric_master_bound_needs_no_convexity():
    """The first master is the linear rows alone: a valid bound below the
    optimum, with nothing certified convex."""
    r = _quiet(RIC(_nonconvex(), time_limit=60, iteration_limit=1).run)
    assert r.bound is not None and r.bound <= 3.0 + 1e-9


def test_ric_with_a_nonlinear_objective_bounds_by_exhaustion():
    """The master is a feasibility problem: no LB until every configuration is
    excluded on a certificate, then the min of their bounds."""
    a = RIC(_nonlinear_objective(), time_limit=60)
    r = _quiet(a.run)
    assert r.status == "optimal" and r.gap_certified
    assert all(lb == -math.inf for _, lb, _, _ in r.solver_stats["trace"][:-1])
    assert r.solver_stats["stop_reason"] in ("converged", "exhausted")


def test_ric_runs_no_convexity_pass_and_no_seed(monkeypatch):
    import discopt._relax.convexity as conv
    import discopt.solvers.gdpopt_loa as gl

    def boom(*a, **k):
        raise AssertionError("RIC must not need this")

    monkeypatch.setattr(conv, "classify_oa_cut_convexity", boom)
    monkeypatch.setattr(gl, "_solve_nlp_relaxation", boom)
    monkeypatch.setattr(gl, "_add_oa_cuts", boom)
    r = _quiet(RIC(_nonconvex(), time_limit=60).run)
    assert r.status == "optimal"


def test_ric_refuses_local_subproblems():
    with pytest.raises(ValueError, match="subproblems are global"):
        RIC(_nonconvex(), subproblem="nlp")


# ── routing and options ─────────────────────────────────────────────────────


def test_gdp_options():
    r = _quiet(
        _nonconvex().solve,
        gdp_method="enumerate",
        time_limit=60,
        gdp_options={"max_configurations": 2},
    )
    assert r.status == "optimal" and r.solver_stats["n_configurations"] == 2
    with pytest.raises(ValueError, match="more than max_configurations=1"):
        _quiet(_nonconvex().solve, gdp_method="enumerate", gdp_options={"max_configurations": 1})
    with pytest.raises(ValueError, match="does not read gdp_options"):
        _nonconvex().solve(gdp_method="enumerate", gdp_options={"iterlim": 3})
    with pytest.raises(ValueError, match="max_configurations"):
        _nonconvex().solve(gdp_method="enumerate", gdp_options={"max_configurations": 0})
    r = _quiet(_nonconvex().solve, gdp_method="ric", time_limit=60, gdp_options={"iterlim": 1})
    assert r.solver_stats["iterations"] == 1
    with pytest.raises(ValueError, match="does not read gdp_options"):
        _nonconvex().solve(gdp_method="ric", gdp_options={"subproblem": "global"})


def test_ric_moves_past_an_uncertified_configuration(monkeypatch):
    """A configuration whose global solve gives a point but no certificate and no
    bound: RIC excludes it to move on, keeps it unresolved (never optimal), and
    freezes the LB at a value that still covers it -- never a later master
    bound, and never an infeasibility proof once the master is empty."""
    import discopt.gdp.algorithms.subproblem as sp
    from discopt.gdp.algorithms.subproblem import SubproblemResult

    real = sp.solve_subproblem
    calls = []

    def first_uncertified(model, selected, deadline, **k):
        calls.append(k.get("mode"))
        res = real(model, selected, deadline, **k)
        if len(calls) == 1:
            return SubproblemResult("feasible", x=res.x, objective=res.objective)
        return res

    monkeypatch.setattr(sp, "solve_subproblem", first_uncertified)
    a = RIC(_nonconvex(), time_limit=60)
    r = _quiet(a.run)
    assert a.lb_frozen and len(a.unresolved) == 1
    assert calls.count("global") == 2  # it went on to the second configuration
    assert r.status == "feasible" and not r.gap_certified
    assert r.objective == pytest.approx(3.0, abs=1e-5)
    assert r.bound is None or r.bound <= 3.0 + 1e-9
