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

Default OFF pending the §5 panel; flag-OFF behavior is unchanged. Corpus-gated.
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
    """flag OFF finds NO incumbent; flag ON finds the origin (the optimum, obj 0)."""
    off = _solve("ball_mk2_30", "0", 8.0)
    on = _solve("ball_mk2_30", "1", 8.0)
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
