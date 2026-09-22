"""Regression test for #1431: a solve's answer must not depend on process history.

``HeuristicGovernor`` (G2) kept a *process-lifetime* miss-streak per heuristic
class. Two solves in which RENS failed to improve latched it off for the rest of
the process (``K_DISABLE = 2``), so every later ``Model.solve()`` in that process
ran without RENS -- including models where RENS is the heuristic that solves them.

Measured before the fix, two ``ball_mk2_30`` solves followed by
``portfol_roundlot`` (60 s, ``nlp_bb=True``)::

    baseline                           time_limit  obj=None   2641 nodes
    DISCOPT_HEURISTIC_GOVERNOR=0       optimal     0.0282905     3 nodes
    governor().reset() before it       optimal     0.0282905     3 nodes

The root bound was bit-identical across arms and so was the first node NLP, so
nothing about the model or the NLP backend changed -- only whether RENS ran.

The latch was also permanent: ``record()`` is the only place clearing
``disabled``, but it is reached only when ``allowed()`` returned True, so a
disabled source could never be re-enabled despite the docstring saying otherwise.

These tests are corpus-gated and deliberately phrased as *invariants* rather than
as assertions about RENS: what #1431 fixes is that prior solves cannot change a
later solve's result, whatever heuristic is responsible.
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import pytest

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
DATA = Path(__file__).parent / "data" / "minlplib"

PORTFOL = DATA / "portfol_roundlot.nl"
BALL = BENCH / "ball_mk2_30.nl"


def _solve_portfol():
    return dm.from_nl(str(PORTFOL)).solve(time_limit=60, nlp_bb=True)


def test_1431_no_process_lifetime_heuristic_latch():
    """No module may expose a process-lifetime heuristic disable latch.

    This is the structural half of the fix and runs without the benchmark corpus:
    the retired ``discopt.heuristic_governor`` module must be gone, and the solver
    must not hold a reference to it.
    """
    import importlib

    import discopt.solver as S

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("discopt.heuristic_governor")

    assert not hasattr(S, "_get_heuristic_governor"), (
        "solver still reaches for the retired G2 governor (#1431)"
    )
    src = Path(S.__file__).read_text()
    assert "_heuristic_governor" not in src, (
        "solver.py still references the retired G2 governor (#1431)"
    )


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not BALL.exists(), reason="ball_mk2_30.nl (benchmark corpus) absent")
@pytest.mark.skipif(not PORTFOL.exists(), reason="portfol_roundlot.nl absent")
def test_1431_portfol_unaffected_by_two_prior_ball_solves():
    """The #1431 repro: two prior solves must not change a later solve's result.

    Before the fix the second arm returned ``time_limit`` with no incumbent.
    """
    prev = os.environ.get("DISCOPT_CONVEX_KERNEL")
    os.environ["DISCOPT_CONVEX_KERNEL"] = "0"
    try:
        alone = _solve_portfol()
        for _ in range(2):
            dm.from_nl(str(BALL)).solve(time_limit=8.0)
        after = _solve_portfol()
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_CONVEX_KERNEL", None)
        else:
            os.environ["DISCOPT_CONVEX_KERNEL"] = prev

    # Assert the probe fired (CLAUDE.md §6): a run that found nothing even when
    # solved alone would make the comparison below vacuous.
    assert alone.objective is not None, (
        "portfol_roundlot found no incumbent even solved alone; the repro drifted "
        "and this test measures nothing about #1431"
    )
    assert after.objective is not None, (
        "#1431: two prior ball_mk2_30 solves cost portfol_roundlot its incumbent "
        f"(alone={alone.objective}, after={after.objective}) -- a solve's result "
        "must not depend on process history"
    )
    assert abs(after.objective - alone.objective) < 1e-6, (
        f"#1431: prior solves changed the answer ({alone.objective} -> {after.objective})"
    )
