"""Recovery re-solves respect a spent deadline (#1989).

#1989 traced a 1 s ``MccormickLPRelaxer.solve_at_node`` grant running 21 s on an
fsg perspective-hull GDP node: the node LP false-failed, and the recovery cascade
behind it -- ``MilpRelaxationModel.solve``'s equilibrated false-infeasible re-verify
(handed a fresh copy of ``time_limit``), then the C-42 pool-strip and #671 row-filter
retries in ``solve_at_node`` (never checking the node's deadline) -- each ran a full
re-solve on a budget already gone.

These tests pin the three seams with a slow fake solve, so they need no fsg model:

* the equilibrated re-verify gets what is LEFT of the budget, and is not started
  once the budget is spent -- in which case the un-re-verified ``infeasible`` is NOT
  returned (it is the possibly-false verdict the re-verify exists to check);
* ``solve_at_node`` does not start the #671 retry once its own deadline has passed,
  and ``DISCOPT_NODE_RETRY_PAST_DEADLINE=1`` restores it (positive control).
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp


class _FakeMilpResult:
    def __init__(self, status):
        self.status = status
        self.objective = None
        self.bound = None
        self.x = None


def _ill_conditioned_milp():
    """A tiny MILP-form relaxation whose coefficient spread trips the re-verify."""
    from discopt._relax.milp_relaxation import MilpRelaxationModel

    A = sp.csr_matrix(np.array([[1.0, 1e5], [-1.0, -1e5]]))
    return MilpRelaxationModel(
        c=np.array([1.0, 0.0]),
        A_ub=A,
        b_ub=np.array([1.0, -2.0]),
        bounds=[(0.0, 10.0), (0.0, 1.0)],
        obj_offset=0.0,
        # Integrality keeps the warm pure-LP path out of the way, so the generic
        # ``solve_milp`` call and its re-verify are what run.
        integrality=np.array([0, 1], dtype=np.int32),
    )


def _install_fake_solver(monkeypatch, sleep_s):
    from discopt.solvers import SolveStatus

    calls = []

    def fake_solve_milp(**kw):
        calls.append(kw.get("time_limit"))
        time.sleep(sleep_s)
        return _FakeMilpResult(SolveStatus.INFEASIBLE)

    monkeypatch.setattr(
        "discopt.solvers.lp_backend.get_milp_solver", lambda **kw: fake_solve_milp
    )
    return calls


def test_reverify_not_started_on_spent_budget(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_WARM_DEADLINE", raising=False)
    calls = _install_fake_solver(monkeypatch, sleep_s=0.3)
    res = _ill_conditioned_milp().solve(time_limit=0.1, backend="simplex")
    assert len(calls) == 1, f"the re-verify ran on a spent budget: {calls}"
    # Never the un-re-verified ``infeasible``: that would let a caller fathom on
    # the very verdict the re-verify exists to check.
    assert res.status == "time_limit"


def test_reverify_gets_the_remaining_budget(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_WARM_DEADLINE", raising=False)
    calls = _install_fake_solver(monkeypatch, sleep_s=0.2)
    _ill_conditioned_milp().solve(time_limit=5.0, backend="simplex")
    assert len(calls) == 2, f"positive control: the re-verify must run here: {calls}"
    assert calls[0] <= 5.0
    # Before #1989 the second call received a fresh ``5.0``.
    assert calls[1] < 5.0 - 0.15, calls


def test_reverify_legacy_optout_unchanged(monkeypatch):
    """``DISCOPT_LP_WARM_DEADLINE=0`` keeps the per-attempt fresh duration."""
    monkeypatch.setenv("DISCOPT_LP_WARM_DEADLINE", "0")
    calls = _install_fake_solver(monkeypatch, sleep_s=0.3)
    _ill_conditioned_milp().solve(time_limit=0.1, backend="simplex")
    assert calls == [0.1, 0.1]


def _node_retry_count(monkeypatch, *, optout: bool) -> int:
    """Run one ``solve_at_node`` whose first LP fails slowly past its deadline.

    Returns how many #671 row-filter retries were started.
    """
    import discopt._relax.mccormick_lp as mlp
    import discopt._relax.milp_relaxation as mr

    if optout:
        monkeypatch.setenv("DISCOPT_NODE_RETRY_PAST_DEADLINE", "1")
    else:
        monkeypatch.delenv("DISCOPT_NODE_RETRY_PAST_DEADLINE", raising=False)
    monkeypatch.setenv("DISCOPT_RELAX_ROW_FILTER", "1")

    state = {"solves": 0, "filters": 0}

    def slow_uncertified(self, *a, **kw):
        state["solves"] += 1
        time.sleep(0.3)
        return mr.MilpRelaxationResult(status="numerical")

    def fake_filter(milp):
        state["filters"] += 1
        return 1  # "dropped a row", so the retry would re-solve

    monkeypatch.setattr(mr.MilpRelaxationModel, "solve", slow_uncertified)
    monkeypatch.setattr(mr, "_filter_unresolvable_rows", fake_filter)

    m = dm.Model("n1989")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x * y - x)
    m.subject_to(x + y >= 1.0)
    relaxer = mlp.MccormickLPRelaxer(m)
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])
    res = relaxer.solve_at_node(lb, ub, time_limit=0.1, separate=False)
    # The node never fathoms on an uncertified verdict, retried or not.
    assert res.status != "infeasible"
    assert state["solves"] >= 1, "probe never reached the node LP (CLAUDE.md §6)"
    return state["filters"]


def test_row_filter_retry_not_started_past_node_deadline(monkeypatch):
    assert _node_retry_count(monkeypatch, optout=False) == 0


def test_row_filter_retry_optout_restores_it(monkeypatch):
    # Positive control: the same probe DOES reach the retry when re-admitted, so the
    # zero above is the gate, not a probe that never opened the branch.
    assert _node_retry_count(monkeypatch, optout=True) == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
