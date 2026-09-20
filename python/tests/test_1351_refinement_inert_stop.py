"""Issue #1351 defect 2: AMP must stop when partition refinement is INERT.

When the refinement pass finds zero atoms attached to any partitioned variable it
emits no piecewise rows, so a finer partition yields a bit-identical MILP and
therefore the identical bound. Iterating on that is provably wasted work: before
this change ``ex1221`` and ``nvs04`` each burned 40 iterations, and
``cvxnonsep_psig30`` 33, to re-derive the bound they already had at iteration 1.

The stop is STRUCTURAL, not "the bound stopped improving". That distinction is
load-bearing: a flat bound is not evidence that a finer partition *cannot* move
it, so a stop-after-k rule is unsound as a bound-neutrality claim for every k. A
structural test is sound for free, because a pass that emits no rows cannot
change the relaxation. The implementation is literally
``if milp_result.partition_refinable_atoms == 0: break`` (``amp.py``) -- it never
reads the bound trajectory -- so these tests assert that MECHANISM (the
"structurally inert" log line firing, or not) rather than an iteration count.

They used to assert a trajectory shape instead, and it rotted (2026-09-20): the
control pinned ``nvs11`` at ``len(lbs) > 7``, read off a run where its bound sat
flat for seven rounds before gaining 43.3. That shape turns out to be a function
of the TIME LIMIT, not of the instance -- measured here, deterministic to the
digit over two reps each:

===========  ======  =======  =========  =====================  ======
instance     limit   rounds   flat run   LB first -> last        wall
===========  ======  =======  =========  =====================  ======
nvs11        20 s    14       7          -474.975 -> -431.042   14.7 s
nvs11        60 s     4       0          -474.975 -> -431.041    6.3 s
nvs15        20 s    16       8            -0.250 ->   0.99990  14.3 s
nvs15        60 s     5       0            -0.250 ->   0.99993   4.6 s
===========  ======  =======  =========  =====================  ======

Same optimum and same ``optimal`` status either way; only the path differs, because
a larger budget lets each MILP subsolve build a stronger relaxation and the loop
then needs far fewer rounds. (Note the smaller limit costs more than twice the
wall clock -- shrinking a budget does not always shrink a test.) An iteration
count is therefore the wrong instrument here twice over: it is not stable under
the budget, and it is not what the rule reads.

Being bound-neutral by construction, this is CLAUDE.md §5 regime (a): the
certified objective and the bound must be EXACTLY unchanged; only the iteration
count may fall.
"""

import logging
import re

import pytest
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.smoke

_NL = "python/tests/data/minlplib_nl/{}.nl"
_LB_RE = re.compile(r"AMP iter (\d+): LB=(\S+?), UB=(\S+?)[,\s]")

# instance -> (pre-fix iteration count, bound that was reached at iteration 1)
_INERT = {
    "ex1221": (40, 6.73569),
    "flay02m": (13, 22.2857),
}


class _Trajectory(logging.Handler):
    """Records the bound trajectory AND whether the inert stop actually fired.

    ``inert`` is the mechanism under test: it is set only by the log line the
    ``partition_refinable_atoms == 0`` branch emits, so asserting on it pins the
    branch itself instead of a symptom (an iteration count) that other causes --
    gap certification, the time limit, ``max_iter`` -- produce just as well.
    """

    def __init__(self):
        super().__init__()
        self.lbs: list[float] = []
        self.inert = False

    def emit(self, record):
        msg = record.getMessage()
        if "structurally inert" in msg:
            self.inert = True
        m = _LB_RE.search(msg + " ")
        if m:
            self.lbs.append(float(m.group(2)))


def _solve_with_trajectory(name: str):
    # The 60 s limit is NOT slack to be trimmed: on this file it is load-bearing
    # in the cheap direction. nvs11 costs 6.3 s at time_limit=60 and 14.7 s at
    # time_limit=20 (table in the module docstring) -- the budget feeds the
    # per-iteration MILP subsolves, so cutting it buys weaker relaxations and more
    # rounds. The two inert instances stop at iteration 1 and never approach it.
    handler = _Trajectory()
    logger = logging.getLogger("discopt")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        result = from_nl(_NL.format(name)).solve(
            solver="amp", rel_gap=1e-4, max_iter=40, time_limit=60
        )
    finally:
        logger.removeHandler(handler)
    return result, handler


@pytest.mark.parametrize("name", sorted(_INERT))
def test_inert_refinement_stops_after_one_iteration(name):
    """The stalled instances stop immediately instead of burning max_iter."""
    before_iters, expected_bound = _INERT[name]
    result, traj = _solve_with_trajectory(name)
    lbs = traj.lbs

    assert lbs, f"{name}: no AMP iteration logged -- the probe measured nothing"
    # The mechanism, not the symptom: the count below would also be 1 if the loop
    # had certified the gap or timed out, which is not what this test is about.
    assert traj.inert, (
        f"{name}: the structural inert stop never fired -- it ran {len(lbs)} "
        f"iterations and stopped for some other reason: {lbs}"
    )
    assert len(lbs) < before_iters, (
        f"{name}: still ran {len(lbs)} iterations (was {before_iters} before the fix)"
    )
    # The bound is reached at iteration 1 and refinement cannot move it, so the
    # loop should stop right there.
    assert len(lbs) == 1, f"{name}: expected a single iteration, got {len(lbs)}: {lbs}"
    assert result.bound == pytest.approx(expected_bound, rel=1e-4), (
        f"{name}: bound changed -- this stop must be bound-NEUTRAL"
    )


@pytest.mark.parametrize("name", sorted(_INERT))
def test_inert_stop_is_bound_neutral(name):
    """Regime (a): stopping early must not alter the reported bound at all."""
    _, expected_bound = _INERT[name]
    result, _traj = _solve_with_trajectory(name)
    assert result.bound is not None
    assert result.bound == pytest.approx(expected_bound, rel=1e-4)
    # A provably inert refinement cannot certify, so the honest status is kept.
    assert result.status in ("feasible", "optimal", "iteration_limit", "time_limit")


def test_instance_with_refinable_atoms_still_iterates():
    """Control: the stop must NOT fire where refinement can still act.

    ``nvs11`` has refinable atoms on its partitioned variables, so the structural
    branch must never take it -- the guard has to be selective, or it is just an
    unconditional stop-at-iteration-1 that happens to keep the bound its first
    relaxation found.

    This asserts the branch did not fire and that refinement went on doing real
    work (more than one round, and the bound moved). It deliberately does NOT pin
    how many rounds: the previous version required more than 7, which holds at
    ``time_limit=20`` and not at the 60 used here (module docstring), so the count
    tracked the budget rather than the rule.
    """
    result, traj = _solve_with_trajectory("nvs11")
    lbs = traj.lbs
    assert lbs, "nvs11: no AMP iteration logged -- the probe measured nothing"
    assert not traj.inert, (
        "the inert stop fired on nvs11, which HAS refinable atoms -- the guard is "
        f"no longer selective; trajectory {lbs}"
    )
    assert len(lbs) > 1, (
        f"nvs11 stopped after one iteration ({lbs}); refinement must keep acting "
        "where atoms remain to refine"
    )
    assert lbs[-1] > lbs[0], "nvs11's bound must still improve over the run"
    assert result.bound is not None
