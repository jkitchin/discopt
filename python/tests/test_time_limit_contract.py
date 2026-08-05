"""``Model.solve(time_limit=T)`` must return within T.

Measured on ``hda`` before this test existed, discopt overran a *constant*
~6-8 s regardless of ``T`` — +7.22 s against a 2 s limit (361%), +6.19 s
against 40 s (15%) — so no limit below ~10 s was honorable at all:

    limit=  2s  wall=  9.22s   limit= 10s  wall= 18.20s
    limit=  5s  wall= 12.20s   limit= 20s  wall= 25.76s
    limit= 40s  wall= 46.19s

The dominant term was a root-relaxation fallback granted ``_ROOT_FALLBACK_FLOOR_S``
(3.0 s) *against an already-spent budget*, which then overran even that grant by
32-35%: on ``hda`` at a 5 s limit it was entered 1.95 s late and ran a further
4.25 s. It is now a pre-deadline reserve (``_ROOT_FALLBACK_RESERVE_S``) withheld
from the search only while the tree is bound-less, so the same bound recovery
happens inside the contract.

SCIP 10.0.2 on the same instances lands within 0.03-0.09 s of the limit at
every limit tested (2 s / 5 s / 10 s, 9/9 runs) while still reporting a finite,
monotonically improving dual bound — the limit is a contract, not a hint.

The overrun is not merely untidy: it silently handicaps every benchmark
comparison (in the in-repo global50 run BARON was held to 60.3 s on
``casctanks`` while discopt took 66.3 s), so any shifted-geometric-mean or
wall-ratio computed against another solver is flattering by that margin.
"""

import math
import time
from pathlib import Path

import pytest
from discopt.modeling.core import from_nl

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"

# Absolute slack for interpreter/result-marshaling teardown. SCIP's measured
# overshoot is 0.03-0.09 s; 1.5 s is generous for a Python solver and still far
# below the 6-8 s regression this test exists to catch.
_SLACK_S = 1.5
# Proportional slack, applied on top: a solver polling a deadline at node
# granularity may finish the node it is in.
_SLACK_FRAC = 0.10

# The residual overrun the reserve does NOT fix, and why it is not fixed here.
#
# ``hda`` at a 10 s limit is bimodal (5 reps, load-gated, ``scratchpad/hda10.py``):
# 10.60 / 10.78 / 12.79 / 13.03 / 12.94 s — mean 12.03 s, sd 1.23, against 12.50 s
# allowed. The two fast reps report the weak dual bound -141697, the three slow
# ones the full -64473, and that is the whole story: the #138 fallback's optional
# separated-relaxation phase is what overruns. When the fallback's grant is already
# spent by the time a first bound is in hand, ``_fb_stop`` returns the weak bound
# and the solve lands at ~10.7 s; when the grant still has room the phase starts,
# and ``solve_at_node`` then *ignores the budget it was handed* for a further ~4 s.
#
# That drop is issue #928 (``_lp_warm_deadline_enabled`` in
# ``_jax/milp_relaxation.py``): the warm pure-LP fast path discards the caller's
# ``time_limit``. It predates this test and is not the reserve's to fix. The
# ``_fb_left()`` clamp on that call site is already in place and correct; it is
# inert only because the callee drops it. (The flag's panel script and companion
# test are named ``issue917_*``/``test_917_*`` for historical reasons; #917 is a
# different, closed issue.)
#
# Turning ``DISCOPT_LP_WARM_DEADLINE=1`` makes all cases here pass (11.14 s ± 0.04
# on this one), but it is not the fix, for two independently sufficient reasons:
# it fails CLAUDE.md §5 bar 2 — a 3-rep load-gated panel over the 17 instances
# where the deadline actually binds gives ON-OFF deltas of -3.2 / +6.6 / +5.7 s,
# flipping sign inside the metric's own noise — and it buys the punctuality by
# losing the bound (-64473 -> -141697 in 5/5 reps), which is the wrong trade under
# §1. So this case is marked xfail non-strict rather than silenced or given a
# wider slack: it keeps running, it keeps reporting, and it will xpass the moment
# #917 is properly resolved.
_XFAIL_LP_DEADLINE_DROP = pytest.mark.xfail(
    reason="#928: warm pure-LP path drops the caller's time_limit, so the #138 "
    "fallback's separated-relaxation phase overruns its grant by ~4s. Bimodal: "
    "10.6-13.0s against 12.5s allowed. Not strict — it passes ~2/5 runs.",
    strict=False,
)

# (instance, time_limit) pairs. Short limits are the discriminating cases —
# a constant-overhead overrun is invisible at 60 s and catastrophic at 2 s.
_CASES = [
    ("hda", 5.0),
    pytest.param("hda", 10.0, marks=_XFAIL_LP_DEADLINE_DROP),
    ("casctanks", 5.0),
]


@pytest.mark.slow
@pytest.mark.parametrize("name,limit", _CASES)
def test_solve_returns_within_time_limit(name, limit):
    """``solve(time_limit=T)`` returns within T plus a small fixed slack."""
    model = from_nl(str(_NL_DIR / f"{name}.nl"))

    t0 = time.perf_counter()
    result = model.solve(time_limit=limit)
    wall = time.perf_counter() - t0

    budget = limit * (1.0 + _SLACK_FRAC) + _SLACK_S
    assert wall <= budget, (
        f"{name}: solve(time_limit={limit}) took {wall:.2f}s, "
        f"overran by {wall - limit:+.2f}s ({100 * (wall - limit) / limit:.0f}%); "
        f"allowed {budget:.2f}s. status={result.status} bound={result.bound}"
    )


@pytest.mark.slow
def test_dual_bound_is_anytime():
    """A longer budget must buy a bound, and never a *worse* one.

    Before the reserve fix the dual bound came from a post-deadline patch rather
    than from anything inside the budget, so ``hda`` reported the identical
    -64473.4 at 5 s, 10 s, 20 s and 40 s: eight times the budget bought zero
    improvement. Monotonicity is asserted (never worse), not strict improvement —
    a search whose root bound has genuinely converged may legitimately plateau.

    ``bound is None`` is read as -inf, which is what it means: no dual bound was
    proved. That is the honest report when the *mandatory* root work (Rust
    presolve + convexity classification + the root McCormick LP, ~7 s on ``hda``)
    outruns the whole limit, and it is exactly the case the deleted
    post-deadline grant used to paper over by spending time it had not been
    given. At the larger budget there is time, so a finite bound is required.
    """
    model_bounds = []
    for limit in (5.0, 15.0):
        model = from_nl(str(_NL_DIR / "hda.nl"))
        result = model.solve(time_limit=limit)
        model_bounds.append(-math.inf if result.bound is None else float(result.bound))

    assert len(model_bounds) == 2, "PROBE NEVER FIRED: fewer than 2 solves recorded"
    assert math.isfinite(model_bounds[1]), (
        f"no dual bound at time_limit=15.0 (got {model_bounds[1]}); "
        "15 s is well past the mandatory root work, so a bound must be proved"
    )
    # hda is a minimization: a larger lower bound is tighter.
    assert model_bounds[1] >= model_bounds[0] - 1e-6, (
        f"dual bound regressed with a larger budget: "
        f"{model_bounds[0]:.6g} at 5s -> {model_bounds[1]:.6g} at 15s"
    )
