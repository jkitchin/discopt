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
    test asserting flakiness. It asserts that the contract for whichever regime
    the solve lands in actually holds, and that the result is SOUND either way.

    It deliberately does NOT assert ``status != "optimal"`` (review finding 10).
    That shipped here and is a test that breaks the moment the solver gets FAST
    ENOUGH to certify `tls2` in 5 s -- a self-inflicted future break in the
    per-PR smoke gate, punishing exactly the improvement this repository is for.
    Nor does it skip when that happens: a skip is the §6 no-op that reads as a
    pass. It asserts the OTHER regime's contract instead, and counts which arm
    ran so neither can pass vacuously.
    """
    runs = [_solve("tls2", time_limit=5.0) for _ in range(2)]
    budgeted = 0
    certified = 0
    for r in runs:
        # Sound on every run, in every regime -- minimize model, so the dual
        # bound is a LOWER bound and may never sit above the incumbent.
        if r.objective is not None and r.bound is not None:
            assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective)), (
                f"{r.status} exit reports bound {r.bound!r} above incumbent {r.objective!r}"
            )
        if r.status == "optimal":
            # The solver got fast enough. #1236's determinism claim now APPLIES
            # to this instance, so hold it to that instead of to the budgeted
            # contract -- and a certified exit must actually be certified.
            assert r.gap_certified is True, (
                "an `optimal` exit that is not gap_certified is the #1262 defect"
            )
            certified += 1
        else:
            # Budgeted exit: #1236's determinism claim does not extend to it.
            assert r.gap_certified is False, f"a {r.status!r} exit claims gap_certified"
            budgeted += 1
    assert budgeted + certified == 2, "both repetitions must be inspected"
    if certified == 2:
        # Both runs certified: the determinism contract is in force, so pin it.
        assert runs[0].node_count == runs[1].node_count, (
            f"tls2 now certifies at 5 s but explored {runs[0].node_count} then "
            f"{runs[1].node_count} nodes -- a certifying solve must be deterministic"
        )
