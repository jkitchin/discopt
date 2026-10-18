"""#1760: the root multistart must not re-solve a start that already failed.

Every start of ``_solve_root_node_multistart`` solves the same node problem, and
``_solve_node_nlp_pounce`` retries a failed start from the box midpoint and a
fixed off-center point. With four failing starts that meant the midpoint and the
off-center point were each solved four times, deterministically failing the same
way: on the optimization-book process-synthesis GDP (case study II.B) 6 of the
seed multistart's 11 POUNCE solves were such repeats, ~2.4 s of an 8 s budget,
and the 20 s run found no incumbent. The multistart now records each failure
and skips a start it has already seen fail.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import discopt.solvers.nlp_pounce as NP
import numpy as np
from discopt._tape_nlp_evaluator import build_evaluator
from discopt.solvers import NLPResult, SolveStatus


def _evaluator():
    m = dm.Model("ms1760")
    x = m.continuous("x", shape=(3,), lb=-4, ub=4)
    m.minimize(dm.sum(x**4 - 3 * x**2 + 0.5 * x))
    ev = build_evaluator(m, lambda: None)
    assert ev is not None
    return ev


def _run(monkeypatch, status):
    calls: list[np.ndarray] = []

    def fake(ev, x0, **kw):
        calls.append(np.array(x0, dtype=float))
        return NLPResult(status=status, x=np.array(x0, dtype=float), objective=1.0)

    monkeypatch.setattr(NP, "solve_nlp", fake)
    lb, ub = np.full(3, -4.0), np.full(3, 4.0)
    S._solve_root_node_multistart(
        _evaluator(), lb, ub, None, {"nlp_native": False}, "pounce", n_random=0
    )
    return calls


def _unique(points):
    out: list[np.ndarray] = []
    for p in points:
        if not any(np.array_equal(p, q) for q in out):
            out.append(p)
    return out


def test_failed_starts_are_solved_once(monkeypatch):
    calls = _run(monkeypatch, SolveStatus.ERROR)
    # midpoint, off-center retry, lower quarter, upper quarter, near-lb.
    # Before the fix: 11 calls, the midpoint and off-center point four times each.
    assert len(calls) == 5, [c.tolist() for c in calls]
    assert len(_unique(calls)) == len(calls), "a failed start was solved twice"


def test_successful_starts_are_unchanged(monkeypatch):
    """Anti-vacuity: with no failure there is nothing to skip, so every start runs."""
    calls = _run(monkeypatch, SolveStatus.OPTIMAL)
    assert len(calls) == 4


def test_time_limited_failures_are_not_memoized(monkeypatch):
    """A later attempt may have more time, so a TIME_LIMIT is retried as before."""
    calls = _run(monkeypatch, SolveStatus.TIME_LIMIT)
    assert len(calls) == 11


def test_node_nlp_without_a_record_retries_as_before(monkeypatch):
    """Callers that pass no ``failed_starts`` (every tree node) keep the full ladder."""
    calls: list[np.ndarray] = []

    def fake(ev, x0, **kw):
        calls.append(np.array(x0, dtype=float))
        return NLPResult(status=SolveStatus.ERROR, x=np.array(x0, dtype=float), objective=1.0)

    monkeypatch.setattr(NP, "solve_nlp", fake)
    lb, ub = np.full(3, -4.0), np.full(3, 4.0)
    for _ in range(2):
        S._solve_node_nlp(
            _evaluator(), np.full(3, 1.0), lb, ub, None, {"nlp_native": False}, "pounce"
        )
    assert len(calls) == 6
