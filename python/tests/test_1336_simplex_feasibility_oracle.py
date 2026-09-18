"""#1336: a POUNCE route that cannot cross-check must ask, not give up.

Since #1309/#1319 a raw Ipopt code 2 (``Infeasible_Problem_Detected``) is not
trusted on its own -- the barrier method raises it from numerical failure on
badly-conditioned problems with no infeasibility behind it -- so it is cross-
checked against an elastic Phase-1 LP. When ``_phase1_min_violation`` returns **no
point at all**, or ``_phase1_verdict`` lands ``undecided``, there is nothing to
check against, and both routes reported an honest ``error``.

Sound, but a real loss. On the QP route there is no second engine to degrade to
(#359), so ``error`` was the end of the line on a model ``main`` calls
``infeasible`` -- correctly in fact, but only by trusting exactly the unverified
code 2 the cross-check exists to stop trusting.

The constraint system is **linear even on the QP route**, so feasibility is an LP
question, and discopt ships an exact engine that answers it over the box as
declared. ``_simplex_feasibility_verdict`` asks it. The answer now comes back
``infeasible`` WITH a proof -- the Rust simplex exits ``Infeasible`` only on a
*verified* Farkas ray -- which is strictly better than both the old ``error`` and
``main``'s unchecked label.

The positive half matters too: a ``feasible`` verdict is an exhibited point,
re-checked here against the rows and the box, which is the positive feasibility
proof the #1327 round-4 review asked for so ``_settle_ambiguous_unbounded`` rests
on something rather than on "Phase-1 did not prove infeasible".
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers.lp_pounce import (
    _INF,
    PHASE1_FEASIBLE,
    PHASE1_INFEASIBLE,
    PHASE1_UNDECIDED,
)


def _oracle():
    """Imported lazily, not at module scope: a module-level import of a symbol the
    fix ADDS makes the whole file un-collectable against a pre-fix tree, so the
    end-to-end tests above would report ImportError instead of the defect they
    exist to catch (CLAUDE.md §6)."""
    from discopt.solvers.lp_pounce import _simplex_feasibility_verdict

    return _simplex_feasibility_verdict


# No marker: fast regression fences on a certification defect, like #1319's.


def _decoupled_conflict(L: float, *, quadratic: bool, satisfiable: bool = False) -> dm.Model:
    """The issue's model: an O(1) contradiction beside an unrelated row whose
    activity is ~``L``. The large row is what defeats the Phase-1 cross-check."""
    m = dm.Model("i1336")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.subject_to(a - b <= (2.0 if satisfiable else 0.0))
    m.subject_to(a - b >= 1.0)
    z = m.continuous("z", lb=L, ub=2 * L)
    v = m.continuous("v", lb=0, ub=2 * L)
    m.subject_to(z - v <= 0)
    y = m.continuous("y", lb=-1e20, ub=1e20)
    w = m.continuous("w", lb=-1, ub=1)
    m.minimize((w - 0.5) ** 2 - y if quadratic else -y)
    return m


@pytest.mark.parametrize("L", [1e13, 1e14, 1e15])
def test_the_issues_repro_is_decided_infeasible_with_a_proof(L):
    """Verbatim from #1336 (the QP default route). Pre-fix: ``error``."""
    r = _decoupled_conflict(L, quadratic=True).solve(time_limit=60)
    assert r.status == "infeasible", f"L={L:e}: got {r.status!r}"
    assert r.status != "unbounded"


@pytest.mark.parametrize("L", [1e13, 1e14, 1e15])
def test_the_lp_analogue_is_decided_infeasible(L):
    """The same model with a linear objective, so it takes the LP route."""
    r = _decoupled_conflict(L, quadratic=False).solve(time_limit=60)
    assert r.status == "infeasible", f"L={L:e}: got {r.status!r}"


@pytest.mark.parametrize("quadratic", [True, False])
def test_a_satisfiable_sibling_is_never_called_infeasible(quadratic):
    """The direction that would be a false certificate: the same shape with the
    conflict removed must never come back ``infeasible``."""
    r = _decoupled_conflict(1e14, quadratic=quadratic, satisfiable=True).solve(time_limit=60)
    assert r.status != "infeasible", (
        f"quadratic={quadratic}: certified 'infeasible' on a satisfiable model"
    )


# ── unit: the oracle itself ──────────────────────────────────────────────


def _two_sided(rows, lo, hi):
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(lo, dtype=np.float64),
        np.asarray(hi, dtype=np.float64),
    )


def test_oracle_proves_infeasibility():
    """``a - b <= 0`` and ``a - b >= 1`` over ``[0, 10]^2``: empty, by inspection."""
    A, cl, cu = _two_sided([[1.0, -1.0], [1.0, -1.0]], [-_INF, 1.0], [0.0, _INF])
    lb, ub = np.zeros(2), np.full(2, 10.0)
    assert _oracle()(A, cl, cu, lb, ub) == PHASE1_INFEASIBLE


def test_oracle_proves_feasibility():
    """The same rows made satisfiable (``1 <= a - b <= 2``)."""
    A, cl, cu = _two_sided([[1.0, -1.0], [1.0, -1.0]], [-_INF, 1.0], [2.0, _INF])
    lb, ub = np.zeros(2), np.full(2, 10.0)
    assert _oracle()(A, cl, cu, lb, ub) == PHASE1_FEASIBLE


def test_oracle_honors_the_declared_box():
    """Feasibility is a question about the box AS DECLARED: the same rows are
    satisfiable on ``[0, 10]`` and empty once the box excludes every solution."""
    A, cl, cu = _two_sided([[1.0, -1.0]], [1.0], [2.0])
    assert _oracle()(A, cl, cu, np.zeros(2), np.full(2, 10.0)) == PHASE1_FEASIBLE
    # a and b pinned equal -> a - b is 0, which is outside [1, 2].
    assert _oracle()(A, cl, cu, np.array([5.0, 5.0]), np.array([5.0, 5.0])) == PHASE1_INFEASIBLE


def test_oracle_handles_equality_rows():
    """``cl == cu`` goes to the engine as an equality rather than two inequalities."""
    A, cl, cu = _two_sided([[1.0, 1.0], [1.0, 1.0]], [0.0, 4.0], [0.0, 4.0])
    lb, ub = np.full(2, -10.0), np.full(2, 10.0)
    assert _oracle()(A, cl, cu, lb, ub) == PHASE1_INFEASIBLE
    A2, cl2, cu2 = _two_sided([[1.0, 1.0], [1.0, -1.0]], [4.0, 0.0], [4.0, 0.0])
    assert _oracle()(A2, cl2, cu2, lb, ub) == PHASE1_FEASIBLE


def test_oracle_never_guesses_when_it_cannot_decide(monkeypatch):
    """Absence of the engine is ``undecided`` -- never a verdict, and never a
    raised exception into a solve that was working."""
    import discopt.solvers.lp_simplex as LS

    monkeypatch.setattr(LS, "SIMPLEX_AVAILABLE", False)
    A, cl, cu = _two_sided([[1.0, -1.0], [1.0, -1.0]], [-_INF, 1.0], [0.0, _INF])
    assert _oracle()(A, cl, cu, np.zeros(2), np.full(2, 10.0)) == PHASE1_UNDECIDED


def test_oracle_does_not_take_a_feasible_point_on_trust(monkeypatch):
    """The positive half is a PROOF, so the exhibited point is re-checked. An
    engine returning a row-violating point labelled optimal yields ``undecided``,
    not ``feasible``."""
    from discopt.solvers import LPResult, SolveStatus

    def _mislabels(*_a, **_k):
        # a - b = 9, nowhere near the demanded [1, 2]; labelled OPTIMAL anyway.
        return LPResult(status=SolveStatus.OPTIMAL, x=np.array([9.0, 0.0]), objective=0.0)

    monkeypatch.setattr("discopt.solvers.lp_simplex.solve_lp", _mislabels)
    A, cl, cu = _two_sided([[1.0, -1.0]], [1.0], [2.0])
    verdict = _oracle()(A, cl, cu, np.zeros(2), np.full(2, 10.0))
    assert verdict == PHASE1_UNDECIDED, (
        f"a violating point must not become a feasibility proof, got {verdict!r}"
    )


def test_oracle_verdicts_are_exhaustive_and_distinct():
    """Probe-fired check (CLAUDE.md §6): every arm above returns one of the three
    verdicts and nothing else, so a silently-None return cannot read as a pass."""
    cases = [
        ([[1.0, -1.0], [1.0, -1.0]], [-_INF, 1.0], [0.0, _INF], PHASE1_INFEASIBLE),
        ([[1.0, -1.0], [1.0, -1.0]], [-_INF, 1.0], [2.0, _INF], PHASE1_FEASIBLE),
    ]
    checked = 0
    for rows, lo, hi, expected in cases:
        A, cl, cu = _two_sided(rows, lo, hi)
        got = _oracle()(A, cl, cu, np.zeros(2), np.full(2, 10.0))
        assert got in (PHASE1_INFEASIBLE, PHASE1_FEASIBLE, PHASE1_UNDECIDED)
        assert got == expected
        checked += 1
    assert checked == 2, f"probe ran {checked} cases, expected 2"
