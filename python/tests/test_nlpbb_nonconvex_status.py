"""NLP-BB on a nonconvex model must not claim infeasibility, and the #844
fallback that runs after it must not report a time limit it never reached.

On ``nvs08``/``nvs16``/``nvs20`` (all feasible) ``solve(nlp_bb=True)`` returned
``infeasible``. The serial node loop gives a fractional nonconvex node the
bound ``-inf`` ("no bound, branch me"), and the NaN guard after it turned that
``-inf`` into the 1e30 exclusion sentinel. Children inherit their parent's bound
as a floor, so the first integer point entered the tree at 1e30 and was dropped,
and the empty tree read as a proof. Separately, a *local* NLP's
``SolveStatus.INFEASIBLE`` counted as a proof on a nonconvex node; it proves
nothing there.

On ``nvs17``/``nvs23`` the same false verdict came back in 0.3 s. ``Model.solve``
then ran the #844 LP-spatial fallback on its 35% reserve only (7 s of a 20 s
budget) although 12.7 s were unspent, so the fallback's own limit surfaced as
``time_limit`` after 7.4 s, and ``wall_time`` still read the primary's 0.3 s.
"""

from __future__ import annotations

import os
import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S
from discopt.modeling.core import from_nl
from discopt.solvers import NLPResult, SolveStatus
from discopt.solvers._convex_kernel import last_attempt_seconds

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


def _nonconvex_feasible_model():
    """Feasible nonconvex MINLP: x*y >= 1 with x, y in [0.5, 4], k integer."""
    m = dm.Model("nc")
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=4.0)
    k = m.integer("k", lb=0, ub=3)
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y + k >= 2.5)
    m.minimize((x - 2.0) ** 2 * (y - 1.0) ** 2 + x * y + k)
    return m


def _convex_feasible_model():
    m = dm.Model("cvx")
    x = m.continuous("x", lb=0.0, ub=4.0)
    k = m.integer("k", lb=0, ub=3)
    m.subject_to(x + k >= 1.5)
    m.minimize((x - 1.0) ** 2 + k)
    return m


def _force_local_infeasible(monkeypatch):
    """Every node NLP answers INFEASIBLE, as a local solver may on a nonconvex box."""
    calls = []

    def verdict(lb, ub):
        calls.append(1)
        x = 0.5 * (np.clip(lb, -1e3, 1e3) + np.clip(ub, -1e3, 1e3))
        return NLPResult(status=SolveStatus.INFEASIBLE, x=x)

    monkeypatch.setattr(S, "_solve_node_nlp", lambda ev, x0, lb, ub, *a, **k: verdict(lb, ub))
    monkeypatch.setattr(
        S, "_solve_root_node_multistart", lambda ev, lb, ub, *a, **k: verdict(lb, ub)
    )
    return calls


def _primary_only(model, **kw):
    # solve_model is NLP-BB's caller without the #844 fallback on top.
    return S.solve_model(model, time_limit=20, nlp_bb=True, **kw)


@pytest.mark.correctness
def test_local_infeasible_verdict_is_not_a_proof_on_a_nonconvex_model(monkeypatch):
    calls = _force_local_infeasible(monkeypatch)
    r = _primary_only(_nonconvex_feasible_model(), batch_size=1)
    assert calls, "the injected node solver never ran"
    assert r.status != "infeasible", r.status
    if r.objective is None:
        assert r.status == "unknown", r.status
        assert not r.gap_certified


@pytest.mark.correctness
def test_local_infeasible_verdict_still_proves_on_a_convex_model(monkeypatch):
    # Control: on a convex model a local solver's INFEASIBLE is global.
    calls = _force_local_infeasible(monkeypatch)
    r = _primary_only(_convex_feasible_model(), batch_size=1, skip_convex_check=False)
    assert calls, "the injected node solver never ran"
    assert r.status == "infeasible", r.status


@pytest.mark.correctness
@pytest.mark.parametrize("name", ["nvs08", "nvs16", "nvs20"])
def test_feasible_nonconvex_instances_are_not_reported_infeasible(name):
    r = from_nl(os.path.join(DATA, f"{name}.nl")).solve(time_limit=20, nlp_bb=True)
    assert r.status != "infeasible", f"{name}: feasible instance reported infeasible"


@pytest.mark.correctness
def test_fallback_after_an_early_primary_gets_the_unspent_budget(monkeypatch):
    """A primary that returns early with no incumbent leaves its budget unspent."""
    from discopt._relax import lp_spatial_bb as L
    from discopt.modeling.core import SolveResult

    budgets = []
    real = L.solve_lp_spatial_bb

    def spy(model, *, time_limit, **kw):
        budgets.append(time_limit)
        return real(model, time_limit=time_limit, **kw)

    def early_primary(model, **kw):
        # What NLP-BB returned on nvs17 before the fix: no incumbent, 0.3 s.
        time.sleep(0.3)
        return SolveResult(status="unknown", wall_time=0.3, python_time=0.3)

    monkeypatch.setattr(L, "solve_lp_spatial_bb", spy)
    monkeypatch.setattr(S, "solve_model", early_primary)
    limit = 6.0
    m = from_nl(os.path.join(DATA, "nvs17.nl"))
    t = time.perf_counter()
    r = m.solve(time_limit=limit)
    wall = time.perf_counter() - t
    assert len(budgets) == 1, "the #844 fallback did not run"
    # Nearly all of the budget, not the 35% reserve (2.1 s).
    assert budgets[0] > 0.85 * limit, budgets
    # wall_time covers the fallback too.
    assert abs(r.wall_time - wall) < 0.25 + 0.05 * wall, (r.wall_time, wall)
    assert r.wall_time <= r.rust_time + r.python_time + 1e-9
    if r.status == "time_limit":
        assert wall >= 0.9 * limit, (r.status, wall)


@pytest.mark.correctness
def test_fallback_keeps_its_reserve_after_a_primary_that_spent_its_budget(monkeypatch):
    """Control: the #917 sizing is unchanged when the primary used its share."""
    from discopt._relax import lp_spatial_bb as L
    from discopt.modeling.core import SolveResult

    budgets = []
    monkeypatch.setattr(
        L, "solve_lp_spatial_bb", lambda model, *, time_limit, **kw: budgets.append(time_limit)
    )

    def spent_primary(model, *, time_limit, **kw):
        time.sleep(time_limit + 0.2)
        return SolveResult(status="time_limit", wall_time=time_limit + 0.2)

    monkeypatch.setattr(S, "solve_model", spent_primary)
    m = from_nl(os.path.join(DATA, "nvs17.nl"))
    m.solve(time_limit=6.0)

    # #1346: the reserve is 35% of what the DEFAULT PATH received, which is the
    # caller's limit MINUS the convex-kernel attempt (#911 deducts it, deliberately
    # — the spec build is the convexity classification and on the instances that
    # hazard bites it is ~1 s of wall). nvs17 is not kernel-eligible, so the attempt
    # only classifies and declines; it is charged all the same.
    #
    # This asserted a flat ``0.35 * 6.0`` while DISCOPT_CONVEX_KERNEL was opt-in, when
    # nothing ran ahead of the reserve. Reading the deduction rather than widening the
    # tolerance keeps the #917 proportion pinned EXACTLY; the separate bound below is
    # what would still fail if a declined attempt ever became expensive, so relaxing
    # the arithmetic does not cost the signal.
    spent = last_attempt_seconds()
    assert spent < 0.10 * 6.0, f"a DECLINED kernel attempt cost {spent:.3f}s of a 6 s budget"
    assert budgets == [pytest.approx(0.35 * (6.0 - spent), rel=1e-6)], (budgets, spent)
