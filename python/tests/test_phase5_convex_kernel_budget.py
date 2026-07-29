"""Phase 5.4 — a declined convex-kernel attempt must not be *additive* on the budget.

Consolidation plan Phase 5.4 (``docs/dev/consolidation-plan-2026-07-28.md``) names
one measured hazard as the thing that must be made "strictly safe" before
``DISCOPT_CONVEX_KERNEL`` can graduate default-ON: ``watercontamination0202``
classifies convex in 2.9 s and then runs 2001 s with no bound against 49 s on the
spatial path (``sota-parity-analysis-2026-07-27.md`` §3 G-C).

The entry experiment (``discopt_benchmarks/scripts/phase5_convex_kernel_budget_entry.py``)
reproduced the *mechanism* in-repo. ``Model.solve`` gave the kernel
``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`` seconds, adopted the result only
when it certified, and then called ``solve_model`` with the caller's **full**
``time_limit`` again — the attempt was never deducted. Measured on
``clay0303hfsg``, 10 s budget, arms interleaved, 2 replicates:

===========  ==========================  ==============
arm          wall                        status
===========  ==========================  ==============
kernel OFF   13.52 s (sd 0.05)           ``time_limit``
kernel ON    **25.24 s** (sd 0.12)       ``time_limit``
===========  ==========================  ==============

2.5x the stated limit, reproduced in every replicate. These tests are the standing
guard on the fix.

Two properties, and both matter:

1. **The contract.** With the kernel ON, a solve whose kernel attempt does not
   certify must still respect ``time_limit`` — the attempt comes *out of* the
   budget, not on top of it.
2. **The win is not paid for.** The fix must not be a fractional budget cap. The
   measured attempt costs (``clay0303hfsg`` 41.9 s = spec 2.34 + tree 39.55) mean a
   cap below ~93 % of a 45 s budget turns the corpus's only certification win
   (``clay0303hfsg`` OFF ``feasible`` -> ON ``optimal``) back into ``feasible``. So
   ``last_attempt_seconds()`` must report the true attempt wall and nothing must
   shrink the attempt itself.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402

_ck = pytest.importorskip("discopt.solvers._convex_kernel")

_DATA = Path(__file__).parent / "data"
_CORPUS_DIRS = (_DATA / "minlplib_nl", _DATA / "minlplib")

#: Eligible AND uncertifiable inside a small budget — the shape the hazard needs.
#: Measured: ``build_convex_spec`` accepts it, and the kernel tree needs 39.55 s.
_HAZARD_INSTANCE = "clay0303hfsg"

#: Small enough that the kernel provably cannot certify (it needs ~40 s), so the
#: attempt is guaranteed to be *declined* — which is the only case that can be
#: additive.
_BUDGET_S = 8.0

#: The solve is allowed the budget plus a fixed slice for parse + JAX compile +
#: teardown, which are outside the solver's own clock. The pre-fix behaviour was
#: ``budget + attempt + budget`` = 25.2 s on a 10 s budget, i.e. ~2.5x; this bound
#: is comfortably below anything the additive path can produce and comfortably
#: above the one-budget path.
_OVERHEAD_ALLOWANCE_S = 10.0


def _instance_path(stem: str) -> Path:
    for d in _CORPUS_DIRS:
        p = d / f"{stem}.nl"
        if p.exists():
            return p
    pytest.skip(f"{stem}.nl not in the in-repo corpus")


def test_attempt_clock_is_exactly_zero_when_the_flag_is_off(monkeypatch):
    """The deduction must be a literal zero on the default path.

    ``Model.solve`` subtracts ``last_attempt_seconds()`` from the budget it hands
    ``solve_model``. A nonzero reading with the flag off would move every
    deadline-sensitive decision in the solver and break Regime-N neutrality — the
    node-count-exactly-unchanged gate the whole consolidation plan is measured on.
    So this asserts ``== 0.0`` exactly, not "small".
    """
    import discopt.modeling as dm

    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    assert _ck.try_convex_solve(m, time_limit=5.0) is None
    assert _ck.last_attempt_seconds() == 0.0


def test_attempt_clock_measures_a_real_attempt(monkeypatch):
    """With the flag on, the clock must report the attempt — including the spec
    build, which is 2.34 s on ``clay0303hfsg`` and 1.16 s on ``cvxnonsep_psig40r``
    and is therefore not a rounding error. A clock that only timed the tree would
    under-deduct exactly on the class this fix exists for."""
    from discopt.modeling.core import from_nl

    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    m = from_nl(str(_instance_path("syn05m")))
    t0 = time.perf_counter()
    res = _ck.try_convex_solve(m, time_limit=30.0)
    wall = time.perf_counter() - t0
    attempt = _ck.last_attempt_seconds()
    print(f"[phase5-budget] syn05m attempt={attempt:.3f}s wall={wall:.3f}s res={res is not None}")
    assert attempt > 0.0, "the attempt clock never started — the deduction would be a no-op"
    assert attempt <= wall + 1e-6, "the attempt clock reports more wall than elapsed"


@pytest.mark.slow
def test_declined_kernel_attempt_does_not_double_the_budget(monkeypatch):
    """The contract: ``solve(time_limit=T)`` stays ~T even when the kernel attempt
    is declined.

    This is the standing guard on the Phase 5.4 fix. It fails on the pre-fix tree
    (measured 25.2 s against a 10 s request) and passes after.
    """
    from discopt.modeling.core import from_nl

    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    path = _instance_path(_HAZARD_INSTANCE)

    t0 = time.perf_counter()
    r = from_nl(str(path)).solve(time_limit=_BUDGET_S)
    wall = time.perf_counter() - t0
    print(
        f"[phase5-budget] {_HAZARD_INSTANCE} budget={_BUDGET_S}s wall={wall:.2f}s "
        f"status={r.status} obj={r.objective} bound={r.bound}"
    )
    # Precondition: the attempt must actually have been DECLINED, else the test
    # passes vacuously against a certified fast solve (CLAUDE.md §6).
    assert r.status != "optimal", (
        f"{_HAZARD_INSTANCE} certified within {_BUDGET_S}s — this instance no longer "
        "exercises the declined-attempt path; pick one that does rather than "
        "letting the guard pass vacuously"
    )
    assert wall <= _BUDGET_S + _OVERHEAD_ALLOWANCE_S, (
        f"convex-kernel ON took {wall:.2f}s for a {_BUDGET_S}s request — the declined "
        "attempt is additive on the budget again (Phase 5.4). Pre-fix this was "
        "25.2s against a 10s request."
    )
