"""Regression test for #827 (family D): the trivial-point primal seed finds an
incumbent on models whose optimum is an obvious point the primal never samples.

`ball_mk2_30` is 30 INTEGER variables in [-1,1]; the single ball constraint
`sum_i (x_i^2 - 0.995825 x_i) <= 0` is satisfied by an integer point only at the
origin (any x_i = +/-1 makes the sum strictly positive), so the origin is the
unique feasible point and the optimum (obj 0). discopt's B&B never samples it and
returns NO incumbent; SCIP solves it in 0.02s.

With `DISCOPT_TRIVIAL_PRIMAL=1`, `solve_model` seeds a trivial feasible point
(origin / box-center / bound corners) as the `initial_point`, which the sub-solver
re-verifies (constraint AND integer feasibility) and injects as an incumbent.

Default OFF; see the `DISCOPT_TRIVIAL_PRIMAL` row in
`docs/dev/flag-retirement-audit.md` for the §5 panel and its verdict. Flag-OFF
behavior is unchanged. Corpus-gated.

**Why these solves pin `DISCOPT_CONVEX_KERNEL=0` (#1422).** `ball_mk2_30`'s
MINLPLib form -- 30 integers, a LINEAR objective, one CONVEX ball row -- is
convex-kernel eligible, and `try_convex_solve` is granted
`min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`, i.e. the WHOLE budget for any
`time_limit <= 120`. At 8 s the kernel cannot certify this instance, so it spends
the entire budget and declines, and `solve_model` -- where the #827 seed lives --
runs with ~0 s left and takes its #654 deadline short-circuit. Measured on the
faithful reconstruction (`x**2`, the form the `.nl` `^` opcode produces; the
algebraically identical `x*x` form is not spec-eligible), 8 s budget::

    CONVEX_KERNEL=1 TRIVIAL_PRIMAL=0  time_limit  obj=None  bound=None  nodes=0
    CONVEX_KERNEL=1 TRIVIAL_PRIMAL=1  time_limit  obj=None  bound=None  nodes=0
    CONVEX_KERNEL=0 TRIVIAL_PRIMAL=0  time_limit  obj=None  bound=-28.0 nodes=717
    CONVEX_KERNEL=0 TRIVIAL_PRIMAL=1  feasible    obj=0.0   bound=-28.0 nodes=735

So with the kernel on, BOTH arms return nothing and this file measures nothing
about #827 -- the failure #1422 reported. Pinning the kernel off is isolation of
the unit under test, not a relaxed assertion: every assertion below is unchanged,
and each test additionally asserts that the search actually ran, so the file
cannot silently degrade to a no-op again. That the default configuration returns
neither incumbent nor bound on this class is a real finding about the convex
kernel's budget policy, recorded in #1422 and tracked separately -- #911 measured
and REJECTED the obvious fix (capping the attempt to a fraction of the budget),
so it is not a change to make in passing.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import pytest

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))


def _solve(inst: str, flag: str, tl: float):
    prev = {k: os.environ.get(k) for k in ("DISCOPT_TRIVIAL_PRIMAL", "DISCOPT_CONVEX_KERNEL")}
    os.environ["DISCOPT_TRIVIAL_PRIMAL"] = flag
    # See the module docstring (#1422): the convex-kernel attempt takes the whole
    # budget on this instance, so leaving it on means neither arm reaches the seed.
    os.environ["DISCOPT_CONVEX_KERNEL"] = "0"
    try:
        # #1431: ``deterministic=True`` makes the run elapsed-independent. With the
        # G2 governor retired, RENS is no longer throttled, and on this instance it
        # can reach the origin within the 8 s wall on a fast/idle machine but not a
        # loaded one -- so the OFF arm's "finds no incumbent" premise became a
        # coin-flip (observed failing 2 of 3 repeats). Under ``deterministic`` the
        # RENS sub-solve gets the caller's own time_limit rather than a slice of
        # what is LEFT on the wall (see the ``_role2_slice`` note in solver.py), and
        # both arms are stable: OFF None / ON 0.0 on 3 of 3 repeats. Every assertion
        # below is unchanged -- this removes a timing dependence, it does not relax
        # the test.
        return dm.from_nl(str(BENCH / f"{inst}.nl")).solve(time_limit=tl, deterministic=True)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "ball_mk2_30.nl").exists(),
    reason="ball_mk2_30.nl (benchmark corpus) absent",
)
def test_827_trivial_seed_finds_ball_mk2_incumbent():
    """flag OFF finds NO incumbent; flag ON finds the origin (the optimum, obj 0)."""
    off = _solve("ball_mk2_30", "0", 8.0)
    on = _solve("ball_mk2_30", "1", 8.0)
    # #1422: assert the probe fired. A solve that never opened a node measures
    # nothing about a mechanism that acts on the search, and "no incumbent" then
    # reads as a pass on the OFF arm for the wrong reason.
    assert off.node_count > 0 and on.node_count > 0, (
        f"the search never ran (OFF {off.node_count} nodes, ON {on.node_count}); "
        "this asserts nothing about the #827 seed -- see the module docstring"
    )
    assert off.objective is None, (
        f"baseline unexpectedly found an incumbent ({off.objective}); repro drifted"
    )
    assert on.objective is not None and abs(on.objective) < 1e-4, (
        f"#827: trivial seed failed to find the ball_mk2 optimum 0.0 (got {on.objective})"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (BENCH / "ex4_1_1.nl").exists(),
    reason="ex4_1_1.nl (benchmark corpus) absent",
)
def test_827_trivial_seed_does_not_regress_control():
    """A model that already solves must return the same optimum ON vs OFF (the
    trivial initial_point is only an early incumbent, never displacing the search)."""
    off = _solve("ex4_1_1", "0", 8.0)
    on = _solve("ex4_1_1", "1", 8.0)
    assert off.objective is not None and on.objective is not None
    assert abs(on.objective - off.objective) < 1e-4, (
        f"#827 trivial seed regressed a control: OFF={off.objective} ON={on.objective}"
    )


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "chimera_k64ising-01.nl").exists(),
    reason="chimera_k64ising-01.nl (benchmark corpus) absent",
)
def test_827_family_c_milp_bb_seeds_initial_point():
    """#827 family C: chimera (1192 integer vars, 0 constraints, MAXIMIZE Ising)
    routes to _solve_milp_bb, which now honors ``initial_point``. With the trivial
    seed ON it surfaces a feasible incumbent where OFF finds none.

    NOTE: chimera is severely wall-budget-bound (#814), so this is a slow test; the
    incumbent is a floor (the trivial all-zeros point), not the optimum (24.3) — a
    good Ising primal is separate. We assert only that a SOUND feasible incumbent
    is returned (for MAXIMIZE, obj must not exceed the known optimum)."""
    on = _solve("chimera_k64ising-01", "1", 5.0)
    assert on.node_count > 0 or on.objective is not None, (
        "the search never ran and no incumbent was seeded -- this asserts nothing "
        "about the #827 seed (#1422)"
    )
    assert on.objective is not None, (
        "#827 family C: _solve_milp_bb did not seed initial_point (no incumbent)"
    )
    # MAXIMIZE: a feasible incumbent can never exceed the true optimum (24.3).
    assert on.objective <= 24.3 + 1e-3, (
        f"#827 family C: unsound incumbent {on.objective} exceeds the optimum 24.3"
    )
