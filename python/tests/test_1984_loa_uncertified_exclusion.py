"""#1984 item 2: an uncertified global subproblem stopped LOA on its configuration.

A global subproblem stopped by ``subproblem_time_limit`` short of a certificate
was marked solved but left in the master. When the master cannot see that
configuration's cost (a nonconvex row gets no OA cut), it proposes the same
configuration again, and the re-proposal stops the run on that configuration's
incumbent. Measured on the optimization book's flowsheet GDP (case II.B,
epigraph form): stopped at iteration 2 on 2276.14, against the optimum 1945.96.

The configuration now leaves the master with ``max(subproblem bound, LB)`` in
``excluded_floor``. Both are valid lower bounds on its optimum, so the LB
stays valid and the search moves on.

The cap is simulated deterministically: a spy on ``subproblem._global``
downgrades the chosen configuration's certified result to ``"feasible"`` and
keeps a valid but weaker bound, which is what a capped solve returns.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.gloa import GLOA
from discopt.gdp.algorithms.loa import LOA

pytestmark = pytest.mark.smoke

#: Configuration "far" costs 10 sqrt(4) = 20, "near" costs 10 sqrt(1) + 5 = 15.
OPT = 15.0


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
def capped(monkeypatch):
    """Downgrade every certified-optimal global subproblem with objective in
    ``capped.objectives`` to an uncertified result with bound ``objective - 0.5``
    (model units, minimization; mirrored for maximization)."""
    import discopt.gdp.algorithms.subproblem as sp

    real = sp._global
    state = {"objectives": set(), "hits": 0}

    def spy(sub, limit, tol):
        res = real(sub, limit, tol)
        if res.status == "optimal" and any(
            abs(abs(res.objective) - o) < 1e-4 for o in state["objectives"]
        ):
            state["hits"] += 1
            shift = 0.5 if res.objective > 0 else -0.5
            return sp.SubproblemResult(
                "feasible", x=res.x, objective=res.objective, bound=res.objective - shift
            )
        return res

    monkeypatch.setattr(sp, "_global", spy)
    return state


@pytest.mark.parametrize("cls", [LOA, GLOA])
def test_uncertified_configuration_does_not_stop_the_search(cls, capped):
    # The master proposes one configuration first (LOA: "far", no cost visible;
    # GLOA: "near", by the secant). Either way the uncapped configuration is
    # proposed again, and before #1984 that re-proposal ended the run.
    capped["objectives"] = {20.0}
    r = _quiet(cls(_epigraph(), time_limit=120).run)
    assert capped["hits"] >= 1, "the spy never downgraded a subproblem"
    assert r.objective == pytest.approx(OPT, abs=1e-5)
    assert r.status == "optimal" and r.gap_certified
    assert r.bound <= r.objective + 1e-9


def test_excluded_bound_is_the_configurations_floor(capped):
    # Cap the optimal configuration: its bound 14.5 is the best the run can
    # prove, so the LB must stop there, never above.
    capped["objectives"] = {OPT}
    a = LOA(_epigraph(), time_limit=120)
    r = _quiet(a.run)
    assert capped["hits"] >= 1
    assert r.objective == pytest.approx(OPT, abs=1e-5)
    assert r.bound_valid and r.bound == pytest.approx(OPT - 0.5, abs=1e-6)
    assert r.status == "feasible" and not r.gap_certified
    assert a.excluded_floor == pytest.approx(OPT - 0.5, abs=1e-6)


def test_maximization_sign(capped):
    capped["objectives"] = {20.0}
    r = _quiet(LOA(_epigraph(maximize=True), time_limit=120).run)
    assert capped["hits"] >= 1
    assert r.objective == pytest.approx(-OPT, abs=1e-5)
    assert r.status == "optimal" and r.bound >= r.objective - 1e-9
