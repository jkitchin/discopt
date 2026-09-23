"""#1442 -- a multiobjective sweep must not mistake "did not answer" for
"proved infeasible", and must never return a silently incomplete front.

``epsilon_constraint`` abandoned every remaining ε cell as soon as one
subproblem returned no solution vector, whatever the reason. That shortcut is
licensed only by a *proven* ``"infeasible"``; ``error`` / ``time_limit`` /
``node_limit`` / ``iteration_limit``, and an incumbent the solver declined to
certify, prove nothing about the tighter cells that follow. The front came back
short, still tagged ``augmecon2``, with nothing to distinguish it from a
complete one.

The tests below are deterministic: the failing cell is *injected* by stubbing
``Model.solve`` for one ε value, rather than relying on the numerical accident
that first exposed this. The end of the file carries the original integration
reproduction as well.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.mo import epsilon_constraint, weighted_sum, weighted_tchebycheff
from discopt.mo.pareto import ParetoFront
from discopt.modeling.core import Model as _ModelClass

# ── the model: small enough to enumerate exhaustively ────────────────────────
# max g1 = v0 + v1 ,  min g2 = v0^2 + v1^2   over v in {0..4}^2,  v0 + v1 >= 2
TRUE_FRONT = {
    (2.0, 2.0),
    (3.0, 5.0),
    (4.0, 8.0),
    (5.0, 13.0),
    (6.0, 18.0),
    (7.0, 25.0),
    (8.0, 32.0),
}


def _build():
    m = Model("mo1442")
    v = m.integer("v", shape=(2,), lb=0, ub=4)
    m.subject_to(dm.sum(v) >= 2)
    return m, [v[0] + v[1], v[0] ** 2 + v[1] ** 2], ["max", "min"]


def _brute_force_front():
    """Independent oracle: enumerate every feasible point, filter by dominance."""
    pts = [(a, b) for a, b in itertools.product(range(5), range(5)) if a + b >= 2]
    F = np.array([[a + b, a * a + b * b] for a, b in pts], dtype=float)
    senses = np.array([-1.0, 1.0])  # max, min
    keep = []
    for i in range(len(F)):
        d_any = False
        for j in range(len(F)):
            if i == j:
                continue
            d = senses * (F[j] - F[i])
            if np.all(d <= 1e-12) and np.any(d < -1e-12):
                d_any = True
                break
        if not d_any:
            keep.append(tuple(F[i]))
    return set(keep)


def _vectors(front):
    return {tuple(np.round(np.asarray(p.objectives, float), 6)) for p in front.points}


def _spy_solve(monkeypatch, fail_at=None, status="error"):
    """Patch ``Model.solve`` to count calls and, optionally, blank exactly one.

    ``fail_at`` is a 1-based call index; that call's result is returned with
    ``x=None`` and the given ``status``, which is how a subproblem that did not
    answer reaches the sweep. Returns a ``calls`` list whose length is the total
    number of solves, so a caller can tell whether the sweep continued past the
    injected hole.

    The patch installs a plain function so normal descriptor binding applies and
    ``self`` still arrives as the model.
    """
    orig = _ModelClass.solve
    calls: list[str] = []

    def spy(self, *a, **kw):
        result = orig(self, *a, **kw)
        calls.append(str(result.status))
        if fail_at is not None and len(calls) == fail_at:
            result.x = None
            result.objective = None
            result.status = status
            calls[-1] = status
        return result

    monkeypatch.setattr(_ModelClass, "solve", spy)
    return calls


def _clean_sweep(sweep=epsilon_constraint, **kw):
    """Run *sweep* unpatched and return ``(front, n_points, n_solves)``."""
    m, objs, senses = _build()
    n = {"count": 0}
    orig = _ModelClass.solve

    def counting(self, *a, **kwargs):
        n["count"] += 1
        return orig(self, *a, **kwargs)

    _ModelClass.solve = counting
    try:
        front = sweep(m, objs, senses=senses, **kw)
    finally:
        _ModelClass.solve = orig
    return front, len(front.points), n["count"]


def test_oracle_is_the_front_we_expect():
    """Guard the oracle itself -- a wrong oracle makes every test below vacuous."""
    assert _brute_force_front() == TRUE_FRONT


# ── the core defect ──────────────────────────────────────────────────────────


EPS_KW = dict(n_points=21, bypass=False, payoff="simple", time_limit=30.0)


@pytest.mark.parametrize("status", ["error", "time_limit", "node_limit", "iteration_limit"])
def test_inconclusive_cell_does_not_abandon_the_rest_of_the_sweep(monkeypatch, status):
    """One unanswered ε cell must cost at most THAT cell, not every later one."""
    _, n_clean_points, n_clean_solves = _clean_sweep(filter=False, **EPS_KW)
    assert n_clean_points >= 10, (
        f"probe is vacuous: clean sweep recorded only {n_clean_points} cells"
    )
    fail_at = n_clean_solves // 2  # comfortably inside the sweep, not the payoff phase

    m, objs, senses = _build()
    calls = _spy_solve(monkeypatch, fail_at=fail_at, status=status)
    got = epsilon_constraint(m, objs, senses=senses, filter=False, **EPS_KW)

    assert len(calls) > fail_at, (
        f"the sweep stopped at the injected {status} cell (call {fail_at} of "
        f"{len(calls)}) -- an unanswered cell is not a proof about later cells"
    )
    assert len(got.points) == n_clean_points - 1, (
        f"{status}: expected to lose exactly the injected cell "
        f"({n_clean_points} -> {n_clean_points - 1}), got {len(got.points)}"
    )


@pytest.mark.parametrize("status", ["error", "time_limit"])
def test_an_incomplete_front_says_so(monkeypatch, status):
    """The hole must be visible: flag, recorded cell, method tag, and a warning."""
    _, _, n_clean_solves = _clean_sweep(filter=False, **EPS_KW)
    m, objs, senses = _build()
    _spy_solve(monkeypatch, fail_at=n_clean_solves // 2, status=status)

    with pytest.warns(UserWarning, match="not a proof of infeasibility"):
        front = epsilon_constraint(m, objs, senses=senses, **EPS_KW)

    assert front.incomplete
    assert len(front.incomplete_cells) == 1
    params, got_status = front.incomplete_cells[0]
    assert got_status == status
    assert "epsilon" in params, params
    assert front.method.endswith("/incomplete"), front.method


def test_a_complete_sweep_is_not_flagged():
    """The flag must mean something: a clean sweep sets neither flag nor tag."""
    m, objs, senses = _build()
    front = epsilon_constraint(m, objs, senses=senses, **EPS_KW)
    assert not front.incomplete
    assert front.incomplete_cells == []
    assert "/incomplete" not in front.method


def test_proven_infeasible_still_ends_the_inner_sweep(monkeypatch):
    """The shortcut is correct for a PROVEN infeasibility and must survive.

    Tightening ε past an infeasible cell stays infeasible, so the sweep should
    stop -- and an infeasible cell is not a hole, so nothing is flagged.
    """
    _, n_clean_points, n_clean_solves = _clean_sweep(filter=False, **EPS_KW)
    fail_at = n_clean_solves // 2

    m, objs, senses = _build()
    calls = _spy_solve(monkeypatch, fail_at=fail_at, status="infeasible")
    got = epsilon_constraint(m, objs, senses=senses, filter=False, **EPS_KW)

    assert len(calls) == fail_at, (
        f"a proven-infeasible cell must still end the inner sweep, but the sweep "
        f"ran {len(calls)} solves past call {fail_at}"
    )
    assert len(got.points) < n_clean_points
    assert not got.incomplete, "a proven infeasibility is not a hole in the front"
    assert "/incomplete" not in got.method


@pytest.mark.parametrize("sweep", [weighted_sum, weighted_tchebycheff])
def test_other_sweeps_record_their_holes(monkeypatch, sweep):
    """weighted_sum / weighted_tchebycheff already skip a cell; they must record it."""
    _, _, n_clean_solves = _clean_sweep(sweep=sweep, n_weights=9, time_limit=30.0)
    m, objs, senses = _build()
    _spy_solve(monkeypatch, fail_at=n_clean_solves // 2, status="error")

    with pytest.warns(UserWarning, match="not a proof of infeasibility"):
        front = sweep(m, objs, senses=senses, n_weights=9, time_limit=30.0)

    assert front.incomplete
    assert [st for _, st in front.incomplete_cells] == ["error"]
    assert "weights" in front.incomplete_cells[0][0]
    assert front.method.endswith("/incomplete")


def test_filtered_carries_the_incompleteness_forward():
    """Filtering points cannot fill a hole, so the record must survive it."""
    front = ParetoFront(
        points=[],
        method="augmecon2/incomplete",
        objective_names=["f1", "f2"],
        senses=["max", "min"],
        incomplete_cells=[({"epsilon": {"f2": 7.0}}, "error")],
    )
    assert front.filtered().incomplete
    assert front.filtered().incomplete_cells == front.incomplete_cells


# ── integration: the front this was found on ─────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("payoff", ["simple", "lexicographic"])
def test_epsilon_constraint_recovers_the_whole_front(payoff):
    """AUGMECON2 is documented complete for general fronts -- so hold it to that.

    Under ``payoff="lexicographic"`` (the default) one ε cell used to error and
    truncate the sweep, losing (2,2) and (3,5). Both payoff modes must now
    return every true Pareto vector, and any that are still missing must be
    accounted for by a recorded incomplete cell rather than dropped in silence.
    """
    m, objs, senses = _build()
    front = epsilon_constraint(
        m, objs, senses=senses, n_points=31, bypass=False, payoff=payoff, time_limit=60.0
    )

    # Compare on the integral points, since subproblem solutions carry
    # integrality residuals up to ~1e-6 and the raw objective vectors inherit them.
    got = {
        (float(a + b), float(a * a + b * b))
        for a, b in (
            tuple(int(round(t)) for t in np.asarray(p.x["v"], float).ravel()) for p in front.points
        )
    }
    missing = TRUE_FRONT - got
    assert not missing, (
        f"payoff={payoff}: front is missing {sorted(missing)}; "
        f"incomplete_cells={front.incomplete_cells}"
    )
    assert got <= TRUE_FRONT, (
        f"payoff={payoff}: returned dominated points {sorted(got - TRUE_FRONT)}"
    )
