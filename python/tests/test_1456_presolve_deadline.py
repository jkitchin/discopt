"""The optional pre-solve structure passes must see the caller's deadline (#1456).

``solve_model``'s reformulation block runs five optional structure-detection
passes before branch and bound starts.  None of them could see ``time_limit``,
so the block ran to completion however long it took: measured at
``time_limit=5``, it cost ``densitymod`` 27.0 s (49 % of the solve), ``truck``
26.1 s (81 %) and ``telecomsp_metro`` 13.2 s (79 %) — before a single node.

No single pass dominates, which is why the three leaf budgets already in the
tree (#1455's ``_DISTRIBUTE_TERM_BUDGET``, #1458's ``_poly_add`` budget and PSD
support restriction) cannot close this: six individually-reasonable passes still
sum to several times the limit.  The fix is the phase-entry gate the *later*
root-setup phases have carried since #654, applied to the one stretch of
pre-solve that sat above its first call site.

Two properties are pinned, because a pass-entry gate alone was measured to be
insufficient and this file said otherwise:

1. **The aggregate.**  Once the budget is spent, no further pass *starts*, and
   the abstention is logged (#1456 item 3) with the accounting a §5 differential
   panel needs to tell a clock-decided run from a deterministic one.
2. **The outer loop.**  A pass that is already running checks the same clock
   once per constraint and abandons *wholesale* (#1456 item 2).

Correction (CLAUDE.md §11).  An earlier revision of this docstring asserted that
(2) was "deliberately NOT pinned" and that the residual was "at most one
in-flight pass ... the leaf budgets' problem, not this gate's".  A cProfile of
``truck`` at ``time_limit=10`` falsified it: the gate was consulted for
``factorable`` at t+0.60 s and not again until **t+81.52 s** — 81 seconds inside
one pass, which is not a residual but the defect itself.  The claim is withdrawn
and item 2 implemented.

The distinction the withdrawn claim was groping at survives, and is the reason
(2) is sound.  Truncating an in-flight *bound-producing* op drops a valid bound
(docs/dev/baron-gap-plan.md §8) and stays barred.  Abandoning an optional
*structure-recognition* pass does not: the pass has exactly two documented
outcomes, "rewritten" and "returned unchanged", and abandoning selects the
second — the same answer it gives when it finds nothing.  Half-rewritten is a
third state, and the callers here never produce it; they drop the partial work
and hand back the model they were given.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import logging  # noqa: E402
import time  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.presolve_deadline import (  # noqa: E402
    PresolveDeadline,
    presolve_deadline_enabled,
)

pytestmark = pytest.mark.unit

# Every pass name the reformulation block gates, in the order it gates them.
# Hardcoded rather than derived so that a pass losing its gate is a FAILURE here
# and not a silently shorter list (CLAUDE.md §6: a probe that adapts to the code
# under test measures nothing).
GATED_PASSES = (
    "canonicalize_entropy",
    "find_functionally_dependent_names",
    "binary_multilinear",
    "factorable",
    "integer_bilinear",
)


def _nonconvex_model():
    """Small, genuinely nonlinear, and not classifiable as LP/MILP at entry —
    so ``solve_model`` actually reaches the reformulation block instead of
    routing to HiGHS (#1229) or being decided as trivial (#1385)."""
    m = dm.Model("gate_probe")
    x = m.continuous("x", shape=(4,), lb=0.5, ub=2.0)
    m.subject_to(x[0] * x[1] + x[2] * x[3] <= 6.0)
    m.subject_to(x[0] * x[2] - x[1] * x[3] >= -4.0)
    m.minimize(x[0] * x[1] * x[2] + x[3] * x[3])
    return m


# --------------------------------------------------------------------------
# The gate object itself
# --------------------------------------------------------------------------


def test_gate_admits_while_budget_remains():
    gate = PresolveDeadline(lambda: False)
    admitted = [gate.afford(name) for name in GATED_PASSES]
    assert all(admitted), "a gate with budget left declined a pass"
    assert gate.ran == GATED_PASSES
    assert gate.skipped == ()
    assert gate.stopped_on is None
    assert gate.deterministic is True


def test_gate_declines_every_pass_once_the_budget_is_spent():
    gate = PresolveDeadline(lambda: True)
    admitted = [gate.afford(name) for name in GATED_PASSES]
    assert not any(admitted), "an exhausted gate admitted a pass"
    assert gate.ran == ()
    assert gate.skipped == GATED_PASSES
    assert gate.stopped_on == "deadline"


def test_gate_reports_non_determinism_the_moment_a_clock_decides():
    """``deterministic`` is what a §5 differential panel would trust to decide
    whether two runs of one model are comparable.  A gate that let the clock
    decide and still reported ``True`` would make the panel's reproducibility
    check vacuous — the exact CLAUDE.md §6 failure, in the attribute that
    matters most."""
    spent = {"yes": False}
    gate = PresolveDeadline(lambda: spent["yes"])

    assert gate.afford("canonicalize_entropy") is True
    assert gate.deterministic is True, "a run that never hit the gate is reproducible"

    spent["yes"] = True
    assert gate.afford("factorable") is False
    assert gate.deterministic is False, "clock decided a skip but the gate claims determinism"
    assert gate.stopped_on == "deadline"


def test_a_none_deadline_disables_the_gate():
    """A caller with no meaningful budget must get the old behaviour exactly,
    not a gate that trips on a missing clock."""
    gate = PresolveDeadline(None)
    assert all(gate.afford(name) for name in GATED_PASSES)
    assert gate.skipped == ()
    assert gate.deterministic is True


# --------------------------------------------------------------------------
# The gate as wired into solve_model
# --------------------------------------------------------------------------


def test_no_pass_is_skipped_when_the_budget_is_ample(caplog):
    """THE CONTROL. Without it every assertion below would pass for a gate that
    skips everything unconditionally, and the fix would read as working while
    having disabled structure detection corpus-wide."""
    m = _nonconvex_model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        m.solve(time_limit=60.0)
    gave_up = [
        r.getMessage()
        for r in caplog.records
        if "pre-solve structure passes gave up" in r.getMessage()
    ]
    assert not gave_up, f"passes abstained with a 60 s budget on a 4-variable model: {gave_up}"


def test_a_pass_that_eats_the_budget_stops_the_ones_after_it(monkeypatch, caplog):
    """THE REGRESSION. Before the fix nothing in this block checked a deadline,
    so all five passes ran however long the first one took — the mechanism
    behind ``densitymod``'s 27.0 s against a 5 s limit.

    Burning the budget inside the *first* pass is the faithful reproduction:
    the block's cost is spread across passes, so what has to be pinned is that
    the ones after the overrun do not start.
    """
    import discopt._relax.factorable_reform as fr

    real = fr.canonicalize_entropy
    calls = {"n": 0}

    def burns_the_budget(model):
        calls["n"] += 1
        time.sleep(1.5)  # > time_limit below, so the budget is gone on return
        return real(model)

    monkeypatch.setattr(fr, "canonicalize_entropy", burns_the_budget)

    m = _nonconvex_model()
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        m.solve(time_limit=1.0)

    assert calls["n"] == 1, (
        f"the burning pass ran {calls['n']}x, not once — the probe is not "
        "measuring what it claims (is the block still reached at all?)"
    )

    skipped = [r for r in caplog.records if "pre-solve structure passes gave up" in r.getMessage()]
    assert skipped, (
        "the budget was spent inside the first pass and NOTHING was skipped: "
        "the later passes are not gated"
    )
    logged = skipped[0].getMessage()
    assert "deadline" in logged, f"abstention not attributed to the clock: {logged}"
    # At least one *named* later pass must appear. Which ones depends on the
    # model's structure (a pass whose gate is `afford(...) and has_work(...)`
    # is only reached if control gets there), so the assertion is on the set
    # being non-empty and drawn from the gated names, not on an exact list.
    named = [p for p in GATED_PASSES if p in logged]
    assert named, f"summary named no gated pass: {logged}"
    assert "canonicalize_entropy" not in named, (
        "the pass that ran to completion was reported as skipped"
    )


@pytest.mark.slow
def test_the_block_cannot_outlive_the_time_limit_by_more_than_one_pass(monkeypatch):
    """END TO END: wall must track ``time_limit``, not the model.

    The bound asserted is ``time_limit`` plus a single in-flight pass, not
    ``time_limit`` itself. ``canonicalize_entropy`` is monkeypatched to sleep
    *around* the real pass, so the sleep is outside any loop the outer-loop
    check (#1456 item 2) can see — the residual pinned here is the one the
    in-pass check genuinely cannot remove, not the 81 s ``truck`` overrun,
    which it does remove (see this module's docstring).
    """
    import discopt._relax.factorable_reform as fr

    real = fr.canonicalize_entropy
    burn_s = 3.0

    def burns_the_budget(model):
        time.sleep(burn_s)
        return real(model)

    monkeypatch.setattr(fr, "canonicalize_entropy", burns_the_budget)

    m = _nonconvex_model()
    t0 = time.perf_counter()
    m.solve(time_limit=1.0)
    dt = time.perf_counter() - t0

    # One in-flight pass (3.0 s) + the limit (1.0 s) + slack for the rest of
    # the solve on a 4-variable model. Ungated, each later pass would add its
    # own cost on top with nothing stopping it.
    assert dt < burn_s + 1.0 + 25.0, f"solve ran {dt:.1f}s against a 1s time_limit"


def test_the_opt_out_restores_the_pre_1456_behaviour(monkeypatch):
    """``DISCOPT_PRESOLVE_DEADLINE=0`` is an opt-out for a shipped default, so
    the A/B #1456's verification asks for is one environment variable and not
    two checkouts. With it set, an exhausted clock must admit every pass."""
    monkeypatch.setenv("DISCOPT_PRESOLVE_DEADLINE", "0")
    gate = PresolveDeadline(lambda: True)
    admitted = [gate.afford(name) for name in GATED_PASSES]
    assert all(admitted), f"opt-out did not disable the gate: {admitted}"
    assert gate.skipped == ()
    assert gate.stopped_on is None
    assert gate.deterministic is True
    assert len(admitted) == len(GATED_PASSES) >= 5


def test_the_gate_is_on_by_default(monkeypatch):
    """The control for the test above: unset means ON. Without this, the opt-out
    test would pass against a gate that was never on in the first place."""
    monkeypatch.delenv("DISCOPT_PRESOLVE_DEADLINE", raising=False)
    assert presolve_deadline_enabled() is True
    gate = PresolveDeadline(lambda: True)
    assert gate.afford(GATED_PASSES[0]) is False
    assert gate.stopped_on == "deadline"


# --------------------------------------------------------------------------
# The outer-loop check inside a running pass (#1456 item 2)
# --------------------------------------------------------------------------


def _factorable_model(n: int = 12):
    """Has factorable work: a mixed repeated-factor product the lift rewrites.

    Sized so the scan and the rewrite both walk a loop with many iterations,
    which is what the outer-loop check is placed in.
    """
    m = dm.Model("factorable_probe")
    x = m.continuous("x", shape=(n,), lb=0.5, ub=2.0)
    for i in range(n - 1):
        m.subject_to((x[i] + x[i + 1]) * (x[i] + x[i + 1]) * x[i] <= 40.0)
    m.minimize(sum(x[i] for i in range(n)))
    return m


def test_abandon_hook_records_once_and_reports_non_determinism():
    gate = PresolveDeadline(lambda: True)
    expired = gate.abandon_hook("factorable")

    assert expired() is True
    assert expired() is True, "the hook must stay expired once the clock is spent"
    assert gate.abandoned == ("factorable",), (
        f"repeated calls recorded the pass more than once: {gate.abandoned}"
    )
    assert gate.stopped_on == "deadline"
    assert gate.deterministic is False
    assert gate.skipped == (), "an abandoned pass is not a skipped pass"


def test_abandon_hook_is_silent_while_budget_remains():
    """THE CONTROL for the test above. Without it, a hook hardwired to True
    would satisfy every abandonment assertion in this file."""
    gate = PresolveDeadline(lambda: False)
    expired = gate.abandon_hook("factorable")
    assert expired() is False
    assert gate.abandoned == ()
    assert gate.deterministic is True

    # And the opt-out reaches the in-pass check too, not only the entry gate.
    assert PresolveDeadline(None).abandon_hook("factorable")() is False


def test_factorable_scan_abstains_when_the_clock_is_spent():
    """THE REGRESSION for item 2, scan half. Before the change
    ``has_factorable_work`` took no deadline at all and ran the whole walk."""
    from discopt._relax.factorable_reform import has_factorable_work

    m = _factorable_model()
    assert has_factorable_work(m) is True, (
        "the probe model has no factorable work — it cannot measure abstention"
    )
    assert has_factorable_work(m, deadline=lambda: True) is False, (
        "the scan walked on with the budget spent"
    )


def test_factorable_rewrite_abandons_wholesale():
    """THE REGRESSION for item 2, rewrite half: the caller gets back the very
    object it passed in — not a half-lifted third state."""
    from discopt._relax.factorable_reform import factorable_reformulate

    m = _factorable_model()
    rewritten = factorable_reformulate(m)
    assert rewritten is not m, "the probe model was not rewritten at all"
    assert sum(v.size for v in rewritten._variables) > sum(v.size for v in m._variables), (
        "the lift appended no aux columns — the probe is not exercising the rewrite"
    )

    abandoned = factorable_reformulate(m, deadline=lambda: True)
    assert abandoned is m, "an abandoned rewrite returned something other than its input"


def test_the_in_pass_check_is_coarse_not_per_node():
    """#1456 item 2 says the check goes "in the outer loop over
    constraints/terms, not per node, so the check itself is not a cost".

    Pinned as a ratio against the model's DAG node count: a check that had crept
    into the recursive walk would fire orders of magnitude more often, and the
    cure would then be its own performance defect.
    """
    from discopt._relax.factorable_reform import _expr_node_count, has_factorable_work

    m = _factorable_model(n=12)
    calls = {"n": 0}

    def counting_deadline():
        calls["n"] += 1
        return False

    has_factorable_work(m, deadline=counting_deadline)

    nodes = sum(_expr_node_count(c.body) for c in m._constraints)
    assert calls["n"] > 0, "the deadline was never consulted — the probe fired nothing"
    assert calls["n"] <= len(m._constraints) + 1, (
        f"deadline consulted {calls['n']}x for {len(m._constraints)} constraints: "
        "the check is finer than the outer loop"
    )
    assert calls["n"] * 10 < nodes, (
        f"deadline consulted {calls['n']}x against {nodes} DAG nodes — too fine "
        "to be the coarse check the issue asks for"
    )


def test_solve_reports_a_pass_abandoned_mid_traversal(monkeypatch, caplog):
    """END TO END. The budget is alive when ``factorable`` starts and gone
    partway through it, which is precisely the case a pass-entry gate cannot
    see — ``truck`` spent 81.5 s there against a 10 s limit.
    """
    import discopt._relax.factorable_reform as fr

    real = fr._find_clearable_denominator
    seen = {"n": 0}

    def slow_denominator_check(expr, model):
        seen["n"] += 1
        time.sleep(0.05)  # ~1 s over the probe model's constraints
        return real(expr, model)

    monkeypatch.setattr(fr, "_find_clearable_denominator", slow_denominator_check)

    m = _factorable_model(n=40)
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        m.solve(time_limit=0.5)

    assert seen["n"] > 1, (
        f"the slowed scan ran {seen['n']}x — the block never reached the "
        "factorable pass, so this probe measures nothing"
    )
    summaries = [
        r.getMessage()
        for r in caplog.records
        if "pre-solve structure passes gave up" in r.getMessage()
    ]
    assert summaries, "a pass ran out of budget mid-traversal and nothing was logged"
    logged = summaries[0]
    assert "abandoned mid-traversal: factorable" in logged, (
        f"abandonment was not reported as such: {logged}"
    )
