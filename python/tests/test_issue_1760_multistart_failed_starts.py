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


# ---------------------------------------------------------------------------
# #1760, part 2: the disjunct-configuration constructor runs before the generic
# root heuristics. On the II.B flowsheet the pump, its NLP seed, integer local
# search and diving spent 2.0 -> 16.9 s finding nothing; the constructor then
# found an incumbent on its second sub-NLP.


def _gdp_model():
    m = dm.Model("gdp1760")
    x = m.continuous("x", lb=0.1, ub=4)
    y = m.continuous("y", lb=0.1, ub=4)
    V = [m.continuous(f"V{k}", lb=0, ub=50) for k in range(3)]
    Y = m.boolean("Y", shape=3)
    m.either_or(
        [
            [V[0] >= 3 * x * dm.exp(0.5 * x), Y.variable[0] == 1],
            [V[1] >= 2 * y**2 + x, Y.variable[1] == 1],
            [V[2] >= x * y + y, Y.variable[2] == 1],
        ],
        name="d",
    )
    for k in range(3):
        m.subject_to(V[k] <= 50 * Y.variable[k])
    m.exactly(1, [Y.variable[i] for i in range(3)])
    m.subject_to(x * y >= 1.5)
    m.minimize(sum(4 * V[k] ** 0.6 for k in range(3)) + 10 * Y.variable[0] - x - y)
    return m


def _heuristic_order(monkeypatch, flag):
    import discopt._relax.primal_heuristics as PH

    monkeypatch.setenv("DISCOPT_GDP_CONFIG_FIRST", flag)
    order: list[str] = []
    for name in ("feasibility_pump", "one_hot_config_subnlp"):
        real = getattr(PH, name)

        def spy(*a, _real=real, _name=name, **k):
            order.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(PH, name, spy)
    r = _gdp_model().solve(time_limit=30)
    assert r.objective is not None
    return order


def test_gdp_config_constructor_runs_before_the_pump(monkeypatch):
    order = _heuristic_order(monkeypatch, "1")
    assert order and order[0] == "one_hot_config_subnlp", order
    assert order.count("one_hot_config_subnlp") == 1, "the constructor ran twice"


def test_gdp_config_first_opt_out_keeps_the_legacy_order(monkeypatch):
    order = _heuristic_order(monkeypatch, "0")
    assert order and order[0] == "feasibility_pump", order
