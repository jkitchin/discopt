"""#1236 item E: what node-count determinism actually guarantees.

Issue #1236 reported its node counts as "deterministic and match the earlier
panel exactly", and used that to justify comparing `tls2`'s node count across
arms. The claim is true for a **certifying** exit and false for a **budgeted**
one, and the difference is not a nuance: measured over six solves at a 60 s
limit, `tls2` returned 197 / 227 / 357 / 359 nodes with status `feasible`. Its
node count is a wall-clock measurement, so any A/B that treats it as a fixed
number is comparing noise.

The contract this pins:

* A solve that CERTIFIES (`status == "optimal"`) explores the same tree every
  time, because nothing in it was cut short by a clock. That is what makes a
  bound-neutrality check (CLAUDE.md §5 regime 1) meaningful at all.
* A solve that exits on its BUDGET (`time_limit` / `feasible` at the limit)
  carries no such guarantee, and the suite must not assert one.

Both halves are asserted, because pinning only the first would let the second
silently acquire a guarantee it never had -- which is how #1236's framing went
wrong.
"""

from __future__ import annotations

import os

import pytest
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

#: Instances that certify quickly, one per engine family the issue exercised:
#: the MILP driver via the integer-bilinear lift, and the native spatial kernel.
CERTIFYING = ("nvs02", "nvs14", "nvs13")

REPEATS = 3


def _solve(name, time_limit=60.0):
    return from_nl(os.path.join(DATA, f"{name}.nl")).solve(time_limit=time_limit)


@pytest.mark.smoke
@pytest.mark.parametrize("name", CERTIFYING)
def test_a_certifying_solve_explores_the_same_tree_every_time(name):
    runs = [_solve(name) for _ in range(REPEATS)]

    # The probe is only meaningful on a certifying exit (CLAUDE.md §6): a
    # budgeted run would make the assertion below a statement about the clock.
    for r in runs:
        assert r.status == "optimal", (
            f"{name} exited {r.status!r}, not `optimal` -- this test asserts the "
            "CERTIFYING contract and cannot run against a budgeted exit"
        )

    nodes = {r.node_count for r in runs}
    assert len(nodes) == 1, (
        f"{name} node counts differ across {REPEATS} certifying runs: "
        f"{[r.node_count for r in runs]}"
    )
    objectives = {round(r.objective, 9) for r in runs}
    assert len(objectives) == 1, (
        f"{name} certified objectives differ across runs: {[r.objective for r in runs]}"
    )
    # The certificate invariant, sense-aware, on every repetition.
    for r in runs:
        assert r.bound is not None
        assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective))


@pytest.mark.smoke
def test_the_suite_does_not_claim_determinism_for_a_budgeted_exit():
    """`tls2` at a short budget: whatever it returns, it is not a fixed number.

    This does not assert that the node count *varies* -- that would be a flaky
    test asserting flakiness. It asserts the two properties that must hold
    regardless: the exit is budgeted (so #1236's determinism claim does not
    apply to it), and the result is still SOUND on every run.
    """
    runs = [_solve("tls2", time_limit=5.0) for _ in range(2)]
    checked = 0
    for r in runs:
        assert r.status != "optimal", (
            "tls2 certified at a 5 s budget -- pick a shorter budget or a "
            "different instance, or this test proves nothing about budgeted exits"
        )
        assert r.gap_certified is False
        if r.objective is not None and r.bound is not None:
            # Minimize model: the dual bound is a LOWER bound.
            assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective)), (
                f"budgeted exit reports bound {r.bound!r} above incumbent {r.objective!r}"
            )
        checked += 1
    assert checked == 2, "both repetitions must be inspected"
