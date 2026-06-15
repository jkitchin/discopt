"""Regression test for issue #127: false-infeasible on a feasible convex MIQP.

Raw ``alan`` (MINLPLib) was reported ``status=infeasible`` although it is
feasible with optimum ≈ 2.925 — a false-infeasible, the dual of a false-optimal.

Root cause was in ``_solve_miqp_bb``: it created the B&B tree directly from the
model bounds without root FBBT, so ``alan``'s ``x0..x3`` reached the node QP IPM
with infinite upper bounds (they are bounded only implicitly by the two
equalities). The IPM diverged to a NaN iterate (``converged==3``); the NaN then
failed the constraint-feasibility check and the node was pruned as *rigorously
infeasible*. With every root branch pruned, the search returned ``infeasible``.

The fix is twofold: (1) run root FBBT before tree creation (tightening
``x0..x3`` to ``[0, 1]`` / ``[0, 0.833]`` so every node QP is well-posed); and
(2) never treat a non-converged / NaN node solve as a proof of infeasibility —
re-solve it with POUNCE (which can certify infeasibility soundly) or keep it
open, instead of pruning.

Note: this solves the *raw* instance — no infinite-bound capping — which is what
distinguishes the regression from ``test_minlplib_benchmark`` (that pre-caps
infinite bounds and so never exercised the bug).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_NL = Path(__file__).parent / "data" / "minlplib" / "alan.nl"

# Global optimum from MINLPLib (BARON/SCIP/Couenne agree).
_OPT = 2.925


@pytest.mark.correctness
def test_alan_not_false_infeasible():
    """Raw alan must never be reported infeasible (issue #127).

    The headline soundness invariant: a feasible model must not get a
    definitive "no solution exists" verdict.
    """
    assert _NL.exists(), f"missing {_NL}"
    r = dm.from_nl(str(_NL)).solve(time_limit=30)
    assert r.status != "infeasible", "false-infeasible on a feasible convex MIQP (#127)"


@pytest.mark.correctness
def test_alan_solves_to_optimum_with_sound_bound():
    """Raw alan certifies to its MINLPLib optimum with a valid dual bound."""
    r = dm.from_nl(str(_NL)).solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal", f"status={r.status}"
    assert r.objective is not None
    assert abs(r.objective - _OPT) <= 1e-3, f"obj={r.objective} != {_OPT}"
    # Soundness invariant: a valid dual bound never exceeds the optimum.
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4, f"invalid dual bound {r.bound} > obj {r.objective}"
