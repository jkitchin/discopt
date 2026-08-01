"""#912: the deterministic work budget that replaces wall-clock "how much work" gates.

Issue #912 measured that the search tree is a function of machine speed: the
root ``integer_local_search`` bounded its own extent with a wall clock, its
descent routinely never converges, and so a faster (or less loaded) machine
explores a different tree from the same model and the same ``time_limit``. That
breaks the repo's bound-neutral verification regime at the root: "node_count
exactly unchanged" is only a meaningful assertion about a *function*.

These tests pin the two halves of the fix:

* :class:`~discopt._work_budget.WorkBudget` semantics — per-kind counting, the
  work gate, the wall-clock *backstop*, and the ``stopped_on`` attribution that
  lets a panel tell a reproducible run from one the clock cut short;
* ``integer_local_search`` under a **scaled clock**. Scaling
  ``time.perf_counter`` by ``alpha`` is exactly a machine ``alpha`` times
  slower. With the work budget on, the search must return the identical point
  after the identical amount of work at every alpha. With it off (both budgets
  0, the legacy escape hatch) the same scaling must be able to cut the search
  short — that arm is what proves the test can actually bite.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._work_budget import EVAL, NLP_SOLVE, WorkBudget  # noqa: E402

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


# --------------------------------------------------------------------------
# WorkBudget semantics
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_work_gate_is_independent_of_the_clock():
    b = WorkBudget({EVAL: 10})
    n = 0
    while not b.exhausted():
        b.charge(EVAL)
        n += 1
    assert n == 10
    assert b.spent(EVAL) == 10
    assert b.stopped_on == "work:eval"
    assert b.deterministic


@pytest.mark.unit
def test_each_kind_has_its_own_limit():
    """The whole point of per-kind counters: a cheap-operation budget must not be
    consumed by expensive operations or vice versa."""
    b = WorkBudget({EVAL: 1_000, NLP_SOLVE: 2})
    for _ in range(999):
        b.charge(EVAL)
    assert not b.exhausted()
    b.charge(NLP_SOLVE)
    assert not b.exhausted()
    b.charge(NLP_SOLVE)
    assert b.exhausted()
    assert b.stopped_on == "work:nlp_solve"
    assert b.spent(EVAL) == 999


@pytest.mark.unit
def test_unlimited_budget_still_counts():
    """The legacy path passes no limits — it must remain measurable, or the
    calibration that sized the defaults could not have been made."""
    b = WorkBudget(None)
    for _ in range(5):
        b.charge(NLP_SOLVE)
    assert b.spent(NLP_SOLVE) == 5
    assert not b.exhausted()
    assert b.stopped_on is None


@pytest.mark.unit
def test_nonpositive_limit_means_unlimited():
    for limit in (0, -1):
        b = WorkBudget({EVAL: limit})
        b.charge(EVAL, 10**9)
        assert not b.exhausted(), limit


@pytest.mark.unit
def test_deadline_is_a_backstop_not_a_work_gate():
    """A deadline that has already passed stops the loop (the user's
    ``time_limit`` is honoured) and is attributed as such."""
    b = WorkBudget({EVAL: 10**9}, deadline=time.perf_counter() - 1.0)
    assert b.exhausted()
    assert b.stopped_on == "deadline"
    assert not b.deterministic


@pytest.mark.unit
def test_work_limit_wins_the_attribution_when_both_fire():
    b = WorkBudget({EVAL: 1}, deadline=time.perf_counter() - 1.0)
    b.charge(EVAL)
    assert b.exhausted()
    assert b.stopped_on == "work:eval"


@pytest.mark.unit
def test_stopped_on_is_latched_at_the_first_stop():
    b = WorkBudget({EVAL: 1})
    b.charge(EVAL)
    assert b.exhausted() and b.stopped_on == "work:eval"
    b.deadline = time.perf_counter() - 1.0
    assert b.exhausted() and b.stopped_on == "work:eval"


@pytest.mark.unit
def test_unknown_kind_is_counted_but_never_gates():
    b = WorkBudget({EVAL: 5})
    b.charge("lp_iteration", 10**6)
    assert not b.exhausted()
    assert b.spent("lp_iteration") == 10**6


# --------------------------------------------------------------------------
# integer_local_search under a scaled clock
# --------------------------------------------------------------------------


class _ScaledClock:
    """Run the process clock ``alpha`` times faster — i.e. pretend the machine is
    ``alpha`` times slower. Patches the ``time`` module the heuristics read."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __enter__(self):
        self._orig = time.perf_counter
        t0 = self._orig()
        alpha = self.alpha
        orig = self._orig
        time.perf_counter = lambda: t0 + alpha * (orig() - t0)
        return self

    def __exit__(self, *exc):
        time.perf_counter = self._orig
        return False


def _ils_instance(name="nvs21"):
    """A real corpus instance with integers, a nonlinear body and a continuous
    block — i.e. one where ILS does sub-NLP repairs rather than no-opping."""
    from discopt.modeling.core import from_nl

    path = _NL_DIR / f"{name}.nl"
    if not path.exists():  # pragma: no cover - corpus is vendored
        pytest.skip(f"{path} missing")
    return from_nl(str(path))


def _run_ils(model, *, eval_budget, solve_budget, alpha, time_budget=5.0, deadline=None):
    """Run ILS from the box midpoint under an ``alpha``-scaled clock, returning
    the result and the budget object that gated it."""
    from discopt._jax import primal_heuristics as ph
    from discopt._jax.nlp_evaluator import cached_evaluator

    ev = cached_evaluator(model)
    lb, ub = ph._get_variable_bounds(model)
    x0 = np.clip(0.5 * (np.clip(lb, -1e3, 1e3) + np.clip(ub, -1e3, 1e3)), lb, ub)

    seen: list[WorkBudget] = []
    orig_init = WorkBudget.__init__

    def _init(self, limits=None, **kw):
        orig_init(self, limits, **kw)
        seen.append(self)

    WorkBudget.__init__ = _init  # type: ignore[method-assign]
    try:
        with _ScaledClock(alpha):
            out = ph.integer_local_search(
                model,
                x0,
                evaluator=ev,
                eval_budget=eval_budget,
                solve_budget=solve_budget,
                time_budget=time_budget,
                deadline=deadline,
            )
    finally:
        WorkBudget.__init__ = orig_init  # type: ignore[method-assign]
    assert seen, "integer_local_search did not create a work budget"
    return out, seen[0]


@pytest.mark.slow
def test_ils_is_invariant_to_machine_speed_with_a_work_budget():
    """The regression this issue exists for: same model, same budget, clock
    scaled 1x / 4x / 16x — identical work, identical point.

    The budget is deliberately set *below* what an unbounded run consumes, so the
    gate genuinely fires; with the pre-#912 wall gate that is exactly the regime
    where the returned incumbent moved with machine speed.
    """
    model = _ils_instance()
    ref_out, ref = _run_ils(model, eval_budget=2_000, solve_budget=8, alpha=1.0)
    assert ref.stopped_on is not None and ref.stopped_on.startswith("work:"), (
        "the budget must be the binding gate for this test to mean anything; "
        f"got stopped_on={ref.stopped_on!r} after {ref.used!r}"
    )
    comparisons = 0
    for alpha in (4.0, 16.0):
        out, budget = _run_ils(model, eval_budget=2_000, solve_budget=8, alpha=alpha)
        comparisons += 1
        assert budget.used == ref.used, alpha
        assert budget.stopped_on == ref.stopped_on, alpha
        assert (out is None) == (ref_out is None), alpha
        if out is not None:
            np.testing.assert_allclose(out[0], ref_out[0], rtol=0, atol=0)
            assert out[1] == ref_out[1], alpha
    assert comparisons == 2  # rule 6: the probe must prove it compared something


@pytest.mark.slow
def test_legacy_wall_gate_is_the_mechanism_the_fix_removes():
    """Fails-before evidence, kept executable: with both budgets disabled the
    extent is decided by the clock, so a scaled (slower) machine stops the search
    earlier — the property #912 measured on gear2."""
    model = _ils_instance()
    # The control's budget must comfortably cover this instance's search (measured
    # ~0.4 s) even on a loaded machine, or both arms get cut and the comparison
    # says nothing; the slow arm's scaling must cut it beyond any doubt.
    _, fast = _run_ils(model, eval_budget=0, solve_budget=0, alpha=1.0, time_budget=5.0)
    _, slow = _run_ils(model, eval_budget=0, solve_budget=0, alpha=500.0, time_budget=5.0)
    assert not fast.limits and not slow.limits, "legacy arm must be unlimited"
    assert slow.stopped_on == "deadline", (
        "the legacy arm is supposed to be cut by the clock; "
        f"got {slow.stopped_on!r} after {slow.used!r}"
    )
    assert sum(slow.used.values()) < sum(fast.used.values()), (
        "a 50x-slower machine must do strictly less work under the legacy wall "
        f"gate (that is the bug): fast={fast.used!r} slow={slow.used!r}"
    )


@pytest.mark.slow
def test_solve_deadline_still_stops_a_work_budgeted_search():
    """The clock does not disappear: it remains the backstop that honours the
    caller's ``time_limit``."""
    model = _ils_instance()
    t0 = time.perf_counter()
    _, budget = _run_ils(
        model,
        eval_budget=10**9,
        solve_budget=10**9,
        alpha=1.0,
        deadline=t0 + 0.2,
    )
    assert budget.stopped_on == "deadline"
    assert time.perf_counter() - t0 < 30.0


@pytest.mark.slow
def test_default_budgets_are_resolved_from_tuning():
    """``eval_budget=None`` must pick up ``SolverTuning``, so the shipped default
    is what a plain ``Model.solve()`` actually runs with."""
    from discopt import solver_tuning as st

    model = _ils_instance()
    _, budget = _run_ils(model, eval_budget=None, solve_budget=None, alpha=1.0)
    tuning = st.current()
    assert budget.limits == {EVAL: tuning.ils_eval_budget, NLP_SOLVE: tuning.ils_solve_budget}
