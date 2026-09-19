"""Issue #1351 defect 2: AMP must stop when partition refinement is INERT.

When the refinement pass finds zero atoms attached to any partitioned variable it
emits no piecewise rows, so a finer partition yields a bit-identical MILP and
therefore the identical bound. Iterating on that is provably wasted work: before
this change ``ex1221`` and ``nvs04`` each burned 40 iterations, and
``cvxnonsep_psig30`` 33, to re-derive the bound they already had at iteration 1.

The stop is STRUCTURAL, not "the bound stopped improving". That distinction is
load-bearing and measured: on the in-repo corpus ``nvs11``'s lower bound sat flat
for SEVEN consecutive rounds and then gained 43.3 (-474.975 -> -431.655, ~9%), so
any stop-after-k rule with k <= 7 discards a real improvement. A structural test
cannot, because a pass that emits no rows cannot change the relaxation.

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
    def __init__(self):
        super().__init__()
        self.lbs: list[float] = []

    def emit(self, record):
        m = _LB_RE.search(record.getMessage() + " ")
        if m:
            self.lbs.append(float(m.group(2)))


def _solve_with_trajectory(name: str):
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
    return result, handler.lbs


@pytest.mark.parametrize("name", sorted(_INERT))
def test_inert_refinement_stops_after_one_iteration(name):
    """The stalled instances stop immediately instead of burning max_iter."""
    before_iters, expected_bound = _INERT[name]
    result, lbs = _solve_with_trajectory(name)

    assert lbs, f"{name}: no AMP iteration logged -- the probe measured nothing"
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
    result, _ = _solve_with_trajectory(name)
    assert result.bound is not None
    assert result.bound == pytest.approx(expected_bound, rel=1e-4)
    # A provably inert refinement cannot certify, so the honest status is kept.
    assert result.status in ("feasible", "optimal", "iteration_limit", "time_limit")


def test_instance_with_refinable_atoms_still_iterates():
    """Control: the stop must NOT fire where refinement can still act.

    ``nvs11`` is the counterexample that rules out a plateau heuristic -- its
    bound is flat for 7 rounds and then improves -- so it must keep iterating.
    """
    result, lbs = _solve_with_trajectory("nvs11")
    assert lbs, "nvs11: no AMP iteration logged -- the probe measured nothing"
    assert len(lbs) > 7, (
        f"nvs11 stopped after {len(lbs)} iterations; it needs more than 7 to reach "
        "the improvement that rules out a stop-after-k rule"
    )
    assert lbs[-1] > lbs[0], "nvs11's bound must still improve over the run"
    assert result.bound is not None
