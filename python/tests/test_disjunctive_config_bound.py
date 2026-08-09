"""#732 Stage 2 — the root disjunctive configuration bound
(``DISCOPT_DISJUNCTIVE_CONFIG_BOUND``, default-OFF).

Plan + entry experiments: ``docs/dev/ex1252-certification-plan.md`` Stage 2.

The pass partitions on the reform's configuration variables instead of relaxing
across them: enumerate the indicator patterns, bound each configuration box by
FBBT → OBBT → LP, unit-peel the weakest leaf on the count variables, and return
the min over leaves — a valid global lower bound by partition, anytime-valid
because children inherit their parent's bound until certified.

Measured (recorded in the plan doc): standalone root pass on ex1252 certifies
~37.9k at a 48-leaf budget (tree at 400 nodes: 16.3k); wired end-to-end at equal
solve settings the reported dual goes 0.0 → 42725. These tests pin the decline
path, soundness + the Stage-2 gate (≥ 33k), anytime validity, and the relaxer's
floor plumbing; the OFF path is untouched (attribute absent → no combine).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.disjunctive_config_bound import compute_disjunctive_config_bound
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds

pytestmark = [pytest.mark.relaxation, pytest.mark.correctness]

EX1252_OPT = 128893.74
STAGE2_GATE = 33000.0  # the plan's Stage-2 kill-criterion bar (2x the 16304 tree dual)


def _nl_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        for sub in ("python/tests/data/minlplib", "tests/data/minlplib"):
            cand = base / sub / "ex1252.nl"
            if cand.exists():
                return cand
    raise FileNotFoundError("ex1252.nl not found")


def _reformed(monkeypatch):
    monkeypatch.setenv("DISCOPT_MULTILINEAR_COUPLING_RLT", "1")
    return reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))


def test_declines_without_config_metadata():
    """A model with no reform configuration metadata: the pass declines cleanly."""
    m = dm.Model("plain")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    lb, ub = flat_variable_bounds(m)
    res = compute_disjunctive_config_bound(m, np.asarray(lb), np.asarray(ub))
    assert res.bound is None
    assert not res.infeasible
    assert res.n_processed == 0


def test_anytime_validity_tiny_budget(monkeypatch):
    """With a tiny budget the pass must stay sound: any returned bound respects
    the caller's root floor (leaves only ever inherit it or improve on it) and
    never exceeds the true optimum."""
    r = _reformed(monkeypatch)
    lb, ub = flat_variable_bounds(r)
    res = compute_disjunctive_config_bound(
        r,
        np.asarray(lb),
        np.asarray(ub),
        root_floor=5134.0,  # the pre-#707 structural floor — a known-valid bound
        max_leaf_solves=4,
    )
    if res.bound is not None:
        assert res.bound >= 5134.0 - 1e-6, "leaves may only inherit or improve the floor"
        assert res.bound <= EX1252_OPT + 1e-2, "bound above the optimum is unsound"


def test_stopped_on_reports_the_clock_when_the_deadline_fires(monkeypatch):
    """An already-expired deadline must be reported as such, not as a bound.

    Without this field the result of a clock-truncated pass is indistinguishable
    from a fully-budgeted one, so a slow runner and a bound regression produce
    the same evidence.
    """
    r = _reformed(monkeypatch)
    lb, ub = flat_variable_bounds(r)
    res = compute_disjunctive_config_bound(
        r,
        np.asarray(lb),
        np.asarray(ub),
        max_leaf_solves=120,
        deadline=time.perf_counter() - 1.0,  # already gone
    )
    assert res.stopped_on == "deadline", f"expected 'deadline', got {res.stopped_on!r}"
    assert res.n_processed == 0


def test_stopped_on_reports_the_budget_when_the_leaf_cap_fires(monkeypatch):
    """The reproducible stopping rule reports itself distinctly from the clock."""
    r = _reformed(monkeypatch)
    lb, ub = flat_variable_bounds(r)
    res = compute_disjunctive_config_bound(
        r,
        np.asarray(lb),
        np.asarray(ub),
        max_leaf_solves=2,
        deadline=time.perf_counter() + 600.0,  # generous: must not be what stops it
    )
    assert res.stopped_on == "budget", f"expected 'budget', got {res.stopped_on!r}"
    assert res.n_processed == 2


def _assert_budget_not_clock(res, budget: int, deadline_s: float) -> None:
    """The quality bars below are only readable if the LEAF BUDGET ended the pass.

    The pass stops on whichever of ``max_leaf_solves`` and ``deadline`` comes
    first. The leaf budget is reproducible: at a fixed budget the bound is a
    function of the relaxation alone (measured bit-identical across two arms and
    two budgets — 37945.427865923564 at 48 leaves, 63080.286987442756 at 120).
    The clock is not: whatever the pass has managed by then depends on the
    machine.

    So a bound below the bar means one of two completely different things, and
    the difference is not visible in the bound. This turns the ambiguity into two
    distinct failures. Do NOT "fix" a clock-bound failure by lowering the bar —
    raise the deadline, or get a faster runner.

    History: the nightly lane read 33498 (42/120 leaves) and 31567 (32/48) on a
    runner ~3x slower than the reference machine and reported both as bound
    regressions. Both were the deadline; the bounds at full budget were unchanged
    to the last digit.
    """
    assert res.stopped_on != "deadline", (
        f"CLOCK-BOUND, not a bound regression: the pass hit its {deadline_s:.0f}s "
        f"deadline after only {res.n_processed}/{budget} leaf solves, so the bound "
        f"it reports ({res.bound}) measures this runner's speed, not the "
        f"relaxation. Raise the deadline (the job's own timeout is the backstop); "
        f"do NOT lower the quality bar."
    )
    assert res.n_processed > 0, "the pass processed no leaf at all — it did not fire"


@pytest.mark.slow
def test_ex1252_spatial_recursion_clears_721_bar(monkeypatch):
    """#732 Stage 4: with continuous bisection at count-complete leaves, the pass
    breaks the ~48k plateau — the original #721 acceptance bar ("global dual
    climbs materially above ~48k") — with no incumbent and no tree.

    Measured 63080 at a 120-leaf budget (48-leaf behavior is byte-identical to
    Stage 2: counts are still splittable there, so the spatial extension only
    engages deeper).
    """
    budget, deadline_s = 120, 1500.0
    r = _reformed(monkeypatch)
    lb, ub = flat_variable_bounds(r)
    res = compute_disjunctive_config_bound(
        r,
        np.asarray(lb),
        np.asarray(ub),
        max_leaf_solves=budget,
        deadline=time.perf_counter() + deadline_s,
    )
    assert res.bound is not None
    assert res.bound <= EX1252_OPT + 1e-2, f"UNSOUND: {res.bound} > optimum {EX1252_OPT}"
    _assert_budget_not_clock(res, budget, deadline_s)
    assert res.bound >= 48000.0, (
        f"spatial recursion bound {res.bound} fell below the #721 bar (48k) at the "
        f"full {budget}-leaf budget — re-run the plan-doc Stage-4 measurement "
        "before shipping a change here"
    )


@pytest.mark.slow
def test_ex1252_pass_clears_stage2_gate(monkeypatch):
    """The full-budget root pass certifies a sound bound above the Stage-2 bar.

    This is the re-scoped Stage-2 acceptance: the branching route failed the
    >= 33k gate (12658); the disjunctive pass clears it with no incumbent and no
    tree (measured ~37.9k at this budget).
    """
    budget, deadline_s = 48, 600.0
    r = _reformed(monkeypatch)
    lb, ub = flat_variable_bounds(r)
    res = compute_disjunctive_config_bound(
        r,
        np.asarray(lb),
        np.asarray(ub),
        max_leaf_solves=budget,
        deadline=time.perf_counter() + deadline_s,
    )
    assert res.bound is not None, "the pass must certify a bound on the config class"
    assert res.bound <= EX1252_OPT + 1e-2, f"UNSOUND: {res.bound} > optimum {EX1252_OPT}"
    _assert_budget_not_clock(res, budget, deadline_s)
    assert res.bound >= STAGE2_GATE, (
        f"pass bound {res.bound} fell below the Stage-2 gate {STAGE2_GATE} at the "
        f"full {budget}-leaf budget — re-run the plan-doc entry experiments before "
        "shipping a change here"
    )


def test_relaxer_floors_node_bounds_at_disjunctive_floor(monkeypatch):
    """The solver stashes the pass result on the model; the relaxer must floor
    every optimal node bound at it (valid: a root-box bound is valid on every
    sub-box). With the attribute absent, behavior is untouched."""
    r = _reformed(monkeypatch)
    lb, ub = flat_variable_bounds(r)
    lb = np.asarray(lb, float).copy()
    ub = np.asarray(ub, float).copy()
    for ind, sel, v in ((18, 36, 1.0), (19, 37, 0.0), (20, 38, 0.0)):
        lb[ind] = ub[ind] = v
        lb[sel] = ub[sel] = v

    base = MccormickLPRelaxer(r).solve_at_node(lb.copy(), ub.copy())
    assert base.status == "optimal"

    # A floor between the raw node bound and the (known-valid) pass bound: the
    # combine must lift the node exactly to it.
    floor = float(base.lower_bound) + 1000.0
    r._disjunctive_config_floor = floor
    try:
        lifted = MccormickLPRelaxer(r).solve_at_node(lb.copy(), ub.copy())
    finally:
        del r._disjunctive_config_floor
    assert lifted.status == "optimal"
    assert lifted.lower_bound == pytest.approx(floor), (
        f"expected the node bound floored at {floor}, got {lifted.lower_bound}"
    )
