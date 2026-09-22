"""#1435 — the root feasibility pump's projection must not depend on the clock.

The pump's fix-and-solve round minimizes the MODEL OBJECTIVE with the integers
pinned and tests only that solve's terminal iterate, so whether it returned
anything was decided by where the solve happened to stop — which
``_deadline_wall_cap`` derives from the caller's remaining wall. On
``heatexch_gen2`` that made a 10 s budget return no incumbent where a 5 s budget
sometimes returned one, i.e. a bigger budget buying a worse answer, which is
#1153's monotonicity gate.

The fix adds a second candidate per round from the same pinned subproblem: a
min-norm elastic feasibility repair, which reads no clock. These tests pin both
halves — the repair's own contract (it lands on the feasible set and never
worsens a violation), and the instance-level gate the issue was filed from.
"""

from __future__ import annotations

import time
from pathlib import Path

import discopt._relax.primal_heuristics as PH
import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver import _make_evaluator

pounce = pytest.importorskip("pounce", reason="the pump repair is a pounce routine")

_DATA = Path(__file__).resolve().parent / "data" / "minlplib_nl"


def _circle_model() -> dm.Model:
    """A curved feasible set: ``x² + y² == 1``. A tangent step from inside or
    outside lands off the circle, so one linearization is provably not enough —
    which is why the repair's outer-iteration cap is above pounce's default."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x * x + y * y == 1.0)
    m.minimize(x + y)
    return m


def _max_violation(evaluator, x: np.ndarray) -> float:
    from discopt.solvers.nlp_pounce import _infer_constraint_bounds

    cl, cu = _infer_constraint_bounds(evaluator)
    g = np.asarray(evaluator.evaluate_constraints(np.asarray(x, dtype=float)), dtype=float)
    return float(np.max(np.maximum(np.maximum(cl - g, g - cu), 0.0), initial=0.0))


def test_repair_projects_an_infeasible_point_onto_a_curved_feasible_set():
    evaluator = _make_evaluator(_circle_model())
    x0 = np.array([0.6, 0.6])
    assert not PH._check_constraint_feasibility(evaluator, x0), "premise: x0 is infeasible"

    x_rep = PH._repair_to_feasible(evaluator, x0)

    assert x_rep is not None
    assert PH._check_constraint_feasibility(evaluator, x_rep)
    lb, ub = evaluator.variable_bounds
    assert np.all(x_rep >= lb - 1e-9) and np.all(x_rep <= ub + 1e-9)
    # Min-norm, not merely feasible: the nearest circle point to (0.6, 0.6).
    np.testing.assert_allclose(x_rep, [0.5 * np.sqrt(2.0)] * 2, atol=1e-5)


@pytest.mark.parametrize("x0", [[0.0, 0.0], [1.9, 1.9], [-1.5, 0.05], [0.2, 1.4]])
def test_repair_never_worsens_the_violation(x0):
    """The safeguard contract the pump's soundness argument leans on: the
    returned point's nonlinear violation is never worse than the input's, so a
    round that adds the repair can never be worse off than one that does not."""
    evaluator = _make_evaluator(_circle_model())
    x_in = np.array(x0, dtype=float)
    before = _max_violation(evaluator, x_in)

    x_rep = PH._repair_to_feasible(evaluator, x_in)

    assert x_rep is not None
    assert _max_violation(evaluator, x_rep) <= before + 1e-12


def test_repair_opt_out_disables_it(monkeypatch):
    evaluator = _make_evaluator(_circle_model())
    monkeypatch.setenv("DISCOPT_PUMP_REPAIR", "0")
    assert PH._repair_to_feasible(evaluator, np.array([0.6, 0.6])) is None
    monkeypatch.setenv("DISCOPT_PUMP_REPAIR", "1")
    assert PH._repair_to_feasible(evaluator, np.array([0.6, 0.6])) is not None


def test_repair_respects_an_expired_deadline():
    """It is extra work inside a budgeted stage, so a stage that is already out
    of wall must not start one — otherwise the fix for a budget bug overruns the
    budget."""
    evaluator = _make_evaluator(_circle_model())
    assert PH._repair_to_feasible(evaluator, np.array([0.6, 0.6]), time.monotonic() - 1.0) is None


@pytest.mark.slow
def test_heatexch_gen2_has_an_incumbent_at_the_1153_budget():
    """The gate probe #1435 was filed from (CLAUDE.md §2: a named instance is a
    probe, never the target of the fix). Measured on ``df2bc458``, this instance
    returned no incumbent at 5 s, 10 s or 20 s, 9 runs out of 9; with the repair
    it returns 834176.34 at 10 s and 20 s, 6 out of 6, and the 20 s node count
    rises 31 -> 63 because the incumbent finally prunes."""
    path = _DATA / "heatexch_gen2.nl"
    if not path.exists():  # pragma: no cover - corpus is checked in
        pytest.skip(f"corpus instance missing: {path}")

    result = from_nl(str(path)).solve(time_limit=10.0, gap_tolerance=1e-4)

    assert result.objective is not None, "no incumbent at the budget #1153 compares"
    assert result.bound is not None and result.bound <= result.objective + 1e-6
    # Independently re-verify the incumbent rather than trusting the solve.
    fresh = from_nl(str(path))
    evaluator = _make_evaluator(fresh)
    vec = np.concatenate(
        [np.asarray(result.x[v.name], dtype=float).ravel() for v in fresh._variables]
    )
    assert PH._check_constraint_feasibility(evaluator, vec)


def test_env_default_is_on():
    """#1435 ships default-ON. The variable is an opt-*out* for a shipped
    default (CLAUDE.md §5 "Out of scope"), not a graduation gate."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("DISCOPT_PUMP_REPAIR", raising=False)
        assert PH._pump_repair_enabled() is True


def test_repair_is_attempted_at_most_once_per_pump(monkeypatch):
    """The cap that made the change affordable (#1435).

    The repair is cheap when it *succeeds* — the pump returns that round. What
    costs is a rounding that cannot be repaired: uncapped, the pump pays for a
    failed repair on every one of its ``max_rounds`` rounds and returns nothing
    anyway. Measured over a 118-instance MINLPLib panel, that cost a median -8.5%
    node throughput on the 30 instances it moved, for no primal gain — so the
    repair runs once per pump. A rounding whose repair succeeds succeeds on the
    first round, which is why one attempt keeps the whole measured gain.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    k = m.integer("k", lb=0, ub=10)
    # Infeasible for every integer k, so no round can ever produce a point and
    # the pump is forced to run all of max_rounds rounds.
    m.subject_to(x * x + k * k <= -1.0)
    m.minimize(x + k)

    calls = {"n": 0}
    real = PH._repair_to_feasible

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(PH, "_repair_to_feasible", counting)
    evaluator = _make_evaluator(m)
    rounds = 5
    out = PH.feasibility_pump(m, np.array([0.0, 0.0]), max_rounds=rounds, evaluator=evaluator)

    assert out is None, "premise: this model has no feasible point to find"
    assert rounds > 1, "premise: the pump ran more rounds than the cap allows repairs"
    assert calls["n"] == PH._PUMP_REPAIR_ATTEMPTS == 1, (
        f"repair ran {calls['n']}x over {rounds} rounds; the cap is {PH._PUMP_REPAIR_ATTEMPTS}"
    )
