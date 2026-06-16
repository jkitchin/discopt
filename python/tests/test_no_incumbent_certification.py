"""Regression: never report ``gap_certified=True`` without an incumbent.

``SolveResult.gap_certified`` defaults to ``True``, and the no-incumbent branch
of every B&B finalizer (``solve_model``, ``_solve_nlp_bb``, ``_solve_milp_bb``,
``_solve_miqp_bb``) set ``status`` to a resource limit but did not reset
``_gap_certified``. A time-/node-limit exit that found no feasible solution
therefore claimed certified optimality with *no incumbent at all*, leaking a
leftover (phantom) tree bound — a prime-directive violation: optimality reported
where none was proven. ``ex1233`` reproduces it (it hits the time limit before
finding any feasible point).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"


@pytest.mark.correctness
def test_no_certification_without_incumbent():
    """A resource-limit exit with no incumbent must not certify optimality."""
    nl = _DATA / "ex1233.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=20, gap_tolerance=1e-4)

    # The fundamental invariant: certification requires an incumbent. If the
    # solver *did* find one within the budget the test still holds (the guard is
    # only about the no-incumbent case), so this is robust to timing.
    assert not (r.gap_certified and r.objective is None), (
        f"certified with no incumbent: status={r.status} bound={r.bound} gap={r.gap}"
    )
