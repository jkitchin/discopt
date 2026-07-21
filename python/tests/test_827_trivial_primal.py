"""Regression test for #827 (family D): the trivial-point primal seed finds an
incumbent on models whose optimum is an obvious point the primal never samples.

`ball_mk2_30` is 30 INTEGER variables in [-1,1]; the single ball constraint
`sum_i (x_i^2 - 0.995825 x_i) <= 0` is satisfied by an integer point only at the
origin (any x_i = +/-1 makes the sum strictly positive), so the origin is the
unique feasible point and the optimum (obj 0). discopt's B&B never samples it and
returns NO incumbent; SCIP solves it in 0.02s.

With the trivial-primal seed on, `solve_model` seeds a trivial feasible point
(origin / box-center / bound corners) as the `initial_point`, which the sub-solver
re-verifies (constraint AND integer feasibility) and injects as an incumbent.

GRADUATED default-ON per §5 (gate: benefit 45% / regression 7%, soundness ok,
cert-neutral); `DISCOPT_TRIVIAL_PRIMAL=0` restores the legacy no-seed behavior.
Corpus-gated.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import pytest

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))


def _solve(inst: str, flag: str, tl: float):
    prev = os.environ.get("DISCOPT_TRIVIAL_PRIMAL")
    os.environ["DISCOPT_TRIVIAL_PRIMAL"] = flag
    try:
        return dm.from_nl(str(BENCH / f"{inst}.nl")).solve(time_limit=tl)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_TRIVIAL_PRIMAL", None)
        else:
            os.environ["DISCOPT_TRIVIAL_PRIMAL"] = prev


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "ball_mk2_30.nl").exists(),
    reason="ball_mk2_30.nl (benchmark corpus) absent",
)
def test_827_trivial_seed_finds_ball_mk2_incumbent():
    """opt-out (=0) finds NO incumbent; seed on (=1) finds the origin (the optimum,
    obj 0)."""
    off = _solve("ball_mk2_30", "0", 8.0)
    on = _solve("ball_mk2_30", "1", 8.0)
    assert off.objective is None, (
        f"baseline unexpectedly found an incumbent ({off.objective}); repro drifted"
    )
    assert on.objective is not None and abs(on.objective) < 1e-4, (
        f"#827: trivial seed failed to find the ball_mk2 optimum 0.0 (got {on.objective})"
    )


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(
    not (BENCH / "ball_mk2_30.nl").exists(),
    reason="ball_mk2_30.nl (benchmark corpus) absent",
)
def test_827_graduated_default_finds_ball_mk2_incumbent():
    """GRADUATED (#829): with the env UNSET (the new default-ON), ball_mk2_30 finds
    its incumbent — the seed is now the default path, not opt-in."""
    prev = os.environ.pop("DISCOPT_TRIVIAL_PRIMAL", None)
    try:
        r = dm.from_nl(str(BENCH / "ball_mk2_30.nl")).solve(time_limit=8.0)
    finally:
        if prev is not None:
            os.environ["DISCOPT_TRIVIAL_PRIMAL"] = prev
    assert r.objective is not None and abs(r.objective) < 1e-4, (
        f"#829: graduated default failed to find the ball_mk2 optimum 0.0 (got {r.objective})"
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
    assert on.objective is not None, (
        "#827 family C: _solve_milp_bb did not seed initial_point (no incumbent)"
    )
    # MAXIMIZE: a feasible incumbent can never exceed the true optimum (24.3).
    assert on.objective <= 24.3 + 1e-3, (
        f"#827 family C: unsound incumbent {on.objective} exceeds the optimum 24.3"
    )
