"""#1153: a ``node_limit`` exit from the native spatial kernel must be USED.

The native spatial kernel (#764) can stop for three budgeted reasons: the clock
(``time_limit``), the node cap (``node_limit``), or a proof (``optimal``). The
first and third were on the accept-list in ``_try_native_spatial_kernel``; the
second was not, so a ``node_limit`` exit was discarded wholesale and the Python
spatial fallback re-ran the search from scratch with the wall budget already
spent.

That is unsound as engineering (a complete, rigorous result thrown away) and it
presents to a user as *more time limit produces a worse answer*, because which
exit the kernel takes is itself budget-dependent — a bigger grant lets the
kernel reach ``max_nodes`` before its clock.

These tests pin the gate, not the instance: nvs19 is a probe that reaches the
node cap quickly, and the assertions are about which code path supplied the
answer and whether the certificate holds.
"""

from __future__ import annotations

import os

import discopt.solver as solver_mod
import pytest
from discopt.modeling.core import from_nl

NVS19 = os.path.join(os.path.dirname(__file__), "data", "minlplib", "nvs19.nl")


def _solve_with_kernel_spy(monkeypatch, **solve_kwargs):
    """Solve, recording every ``_try_native_spatial_kernel`` outcome."""
    outcomes: list[str] = []
    original = solver_mod._try_native_spatial_kernel

    def spy(*args, **kwargs):
        res = original(*args, **kwargs)
        outcomes.append("declined" if res is None else res.status)
        return res

    monkeypatch.setattr(solver_mod, "_try_native_spatial_kernel", spy)
    result = from_nl(NVS19).solve(**solve_kwargs)
    return result, outcomes


def test_node_limit_exit_is_accepted_not_declined(monkeypatch):
    """A kernel that stops on ``max_nodes`` must supply the reported answer.

    Before the fix this asserted-on list read ``["declined"]``: the kernel
    returned ``node_limit`` carrying obj=-1098.2 / bound=-1472.35 at 100k nodes
    and ``_try_native_spatial_kernel`` dropped it on the floor.
    """
    result, outcomes = _solve_with_kernel_spy(
        monkeypatch, time_limit=60.0, gap_tolerance=1e-4, max_nodes=2000
    )

    assert outcomes, "kernel gate never ran -- probe measured nothing (CLAUDE.md §6)"
    assert "declined" not in outcomes, (
        f"the native kernel result was discarded ({outcomes}); a node_limit exit "
        "carries a feasible incumbent and a rigorous bound and must be used (#1153)"
    )
    assert result.status == "node_limit", (
        f"expected the kernel's own node_limit status, got {result.status!r} -- "
        "a different status means the Python fallback supplied the answer"
    )


def test_accepted_node_limit_result_keeps_its_certificate(monkeypatch):
    """The accepted result must still satisfy the bound<=incumbent invariant.

    Widening the accept-list may never let an unsound certificate through: this
    is the guard that the newly-admitted status carries a valid dual bound
    (nvs19 is a minimization).
    """
    result, outcomes = _solve_with_kernel_spy(
        monkeypatch, time_limit=60.0, gap_tolerance=1e-4, max_nodes=2000
    )

    assert outcomes, "kernel gate never ran -- probe measured nothing (CLAUDE.md §6)"
    # Without this the assertions below are vacuous: they would grade the Python
    # fallback's certificate instead of the newly-admitted kernel result.
    assert "declined" not in outcomes, f"kernel result discarded ({outcomes}) -- #1153"
    assert result.objective is not None, "node_limit exit reported no incumbent"
    assert result.bound <= result.objective + 1e-6, (
        f"certificate violated: bound {result.bound} > incumbent {result.objective}"
    )
    # A node-limited exit is not a proof of optimality; it must never claim one.
    assert result.status != "optimal"


@pytest.mark.slow
def test_more_budget_does_not_produce_a_worse_incumbent(monkeypatch):
    """#1153's user-visible symptom: the 30s -> 60s incumbent inversion.

    Held at a fixed ``max_nodes`` so the comparison is between two *clock*
    budgets over the same tree size, not between two different searches. The
    larger budget must not return a worse incumbent than the smaller one.
    """
    # max_nodes=30_000 is chosen so the two arms take DIFFERENT kernel exits on
    # nvs19: at 15 s the clock binds first (~18k nodes -> ``time_limit``), at 60 s
    # the node cap binds first (-> ``node_limit``). That asymmetry is the bug's
    # trigger; with a cap both arms reach, both took the same path and the
    # inversion was invisible. Measured pre-fix: -1098.2 @15s -> -1097.6 @60s.
    small, small_outcomes = _solve_with_kernel_spy(
        monkeypatch, time_limit=15.0, gap_tolerance=1e-4, max_nodes=30_000
    )
    large, large_outcomes = _solve_with_kernel_spy(
        monkeypatch, time_limit=60.0, gap_tolerance=1e-4, max_nodes=30_000
    )

    assert small_outcomes and large_outcomes, "kernel gate never ran (CLAUDE.md §6)"
    assert small.objective is not None and large.objective is not None
    # Minimization: a larger budget must be <= (better or equal), never worse.
    assert large.objective <= small.objective + 1e-6, (
        f"incumbent got WORSE with more budget: {small.objective} @15s -> "
        f"{large.objective} @60s (#1153)"
    )
