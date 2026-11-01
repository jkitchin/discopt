"""#2082: LOA stopped "exhausted" when the master re-proposed a configuration
its global subproblem could not resolve.

A global subproblem that stops at its cap with neither a point nor a
certificate falls back to the fixed-integer NLP step, which neither bounds the
configuration nor excludes it. When the master cannot see that
configuration's cost (a nonconvex row gets no OA cut) it proposes it again, and
the #1611 C-11 re-proposal rule ended the whole run there, reported as
``stop_reason="exhausted"``. Measured on the optimization book's flowsheet GDP
(case II.B): stopped at iteration 3 of 8, at 13.8 s of a 45 s limit, on
incumbent 2276.14 against the optimum 1948.58.

The re-proposed configuration now leaves the master with ``max(its global
subproblem's bound, LB)`` in ``excluded_floor`` (``LOA._exclude_reproposed``),
which keeps ``min(master bound, excluded_floor)`` a valid LB, and the search
moves on. A stop the method still takes on a re-proposal is labelled
``"reproposal"`` / ``"unresolved_reproposal"``, not ``"exhausted"``.

The unresolved global subproblem is simulated deterministically: a spy on
``subproblem._global`` turns the chosen configuration's result into
``"unresolved"`` (no point), with or without a valid bound.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.loa import LOA

pytestmark = pytest.mark.smoke

#: "far" costs 10 sqrt(4) = 20; "near" costs 10 sqrt(1) + 5 = 15 (the optimum).
FAR, OPT = 20.0, 15.0


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _epigraph(maximize: bool = False):
    """Two configurations; their concave cost sits in a row LOA cannot cut."""
    m = dm.Model("epigraph")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=100)
    c = m.continuous("c", lb=0, ub=10)
    m.subject_to(z >= 10 * x**0.5)  # concave right-hand side: no OA cut
    m.either_or([[x >= 4, c == 0], [x >= 1, c == 5]], name="d")
    if maximize:
        m.maximize(-(z + c))
    else:
        m.minimize(z + c)
    return m


@pytest.fixture
def unresolved(monkeypatch):
    """Turn the global subproblem of the configuration whose optimum is ``FAR``
    into an ``"unresolved"`` result with no point and bound ``state["bound"]``
    (model units; ``None`` for none). ``state["nlp_fails"]`` also makes the NLP
    fallback return no point on that configuration."""
    import discopt.gdp.algorithms.subproblem as sp
    import discopt.solvers.gdpopt_loa as gl

    real_global = sp._global
    real_nlp = gl._solve_nlp_subproblem
    state = {"bound": None, "nlp_fails": False, "hits": 0, "nlp_hits": 0}

    def spy(sub, limit, tol):
        res = real_global(sub, limit, tol)
        if res.status == "optimal" and abs(abs(res.objective) - FAR) < 1e-4:
            state["hits"] += 1
            return sp.SubproblemResult("unresolved", bound=state["bound"], reason="forced")
        return res

    def nlp(evaluator, lb, ub, *a, **k):
        x = real_nlp(evaluator, lb, ub, *a, **k)
        if state["nlp_fails"] and x is not None:
            obj = abs(float(evaluator.evaluate_objective(x)))
            if abs(obj - FAR) < 1e-3:
                state["nlp_hits"] += 1
                return None
        return x

    monkeypatch.setattr(sp, "_global", spy)
    monkeypatch.setattr(gl, "_solve_nlp_subproblem", nlp)
    return state


def test_reproposed_unresolved_configuration_is_excluded_with_its_bound(unresolved):
    # Before #2082: the master proposed "far" (no cost visible), its global solve
    # was unresolved, the NLP found 20, the master re-proposed "far" and the run
    # stopped "exhausted" on incumbent 20.
    unresolved["bound"] = FAR - 0.5
    a = LOA(_epigraph(), time_limit=120)
    r = _quiet(a.run)
    assert unresolved["hits"] >= 1, "the spy never downgraded a subproblem"
    assert r.objective == pytest.approx(OPT, abs=1e-5)
    # It now ends on an empty master: every configuration excluded.
    assert r.solver_stats["stop_reason"] in ("exhausted", "converged")
    assert r.solver_stats["iterations"] >= 3
    # "far" is bounded by 19.5 > 15, so "near" is proved optimal.
    assert r.status == "optimal" and r.gap_certified
    assert r.bound <= r.objective + 1e-9


def test_without_a_bound_the_floor_is_the_lb(unresolved):
    # No bound from the capped solve: the floor is the LB at the re-proposal, so
    # the run finds the optimum but cannot certify it (the bound stays valid).
    unresolved["bound"] = None
    a = LOA(_epigraph(), time_limit=120)
    r = _quiet(a.run)
    assert unresolved["hits"] >= 1
    assert r.objective == pytest.approx(OPT, abs=1e-5)
    assert r.solver_stats["stop_reason"] not in ("reproposal", "unresolved_reproposal")
    assert r.bound_valid and r.bound is not None and r.bound <= OPT + 1e-9
    assert a.excluded_floor <= OPT + 1e-9


def test_repeat_with_no_point_is_excluded_and_still_blocks_optimal(unresolved):
    # Neither step finds a point for "far": it is retried once (a fresh global
    # budget), then leaves the master with its floor. It stays unresolved, so
    # "optimal" is never claimed even though "near" is found.
    unresolved["bound"] = FAR - 0.5
    unresolved["nlp_fails"] = True
    a = LOA(_epigraph(), time_limit=120)
    r = _quiet(a.run)
    assert unresolved["hits"] >= 2 and unresolved["nlp_hits"] >= 2
    assert r.objective == pytest.approx(OPT, abs=1e-5)
    assert r.status == "feasible" and not r.gap_certified
    assert r.solver_stats["n_unresolved"] >= 1
    assert r.bound_valid and r.bound <= OPT + 1e-9


def test_maximization_sign(unresolved):
    unresolved["bound"] = -(FAR - 0.5)  # model units: an upper bound on -(z + c)
    a = LOA(_epigraph(maximize=True), time_limit=120)
    r = _quiet(a.run)
    assert unresolved["hits"] >= 1
    assert r.objective == pytest.approx(-OPT, abs=1e-5)
    assert r.status == "optimal" and r.bound >= r.objective - 1e-9


def test_nlp_mode_reproposal_is_labelled():
    # NLP mode keeps the C-11 stop (a local point bounds nothing), but the stop
    # is a re-proposal, not an exhausted search space.
    r = _quiet(LOA(_epigraph(), time_limit=120, subproblem="nlp").run)
    assert r.solver_stats["stop_reason"] in ("reproposal", "converged")
    assert r.solver_stats["stop_reason"] != "exhausted"
