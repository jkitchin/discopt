"""#1371: ``deterministic=True`` must not exempt a solve from the role-1 wall.

#1152 settled the role-1 contract — ``time_limit`` is "a hard wall with an anytime
bound", and the solve returns by the deadline — and ``_role2_budget``'s own
docstring asserted the invariant that keeps it true under ``deterministic``:

    The role-1 deadline is *not* routed through here and still stops the search.

It did not. ``_role2_deadline``/``_role2_budget``/``_role2_horizon`` returned
``None``/``math.inf`` under ``deterministic``, which removes the role-2 budget **and
the role-1 wall together**. Root OBBT then ran to its deterministic caps with
nothing polling the clock. Measured before the fix, idle machine, both modes back to
back in one process::

    casctanks   limit 60 s   deterministic=False   60.23 s (1.00x)
                             deterministic=True   600.04 s (10.00x)
    bchoco08    limit 30 s   deterministic=False   30.11 s (1.00x)
                             deterministic=True   376.28 s (12.54x)

After: ``casctanks`` 61.03 s (1.02x), ``bchoco08`` 30.86 s (1.03x).

**Why this file exists and `test_875_root_setup_budget.py` did not catch it.** The
only other test asserting the 1.25x wall is
``test_watercontamination0202_honours_its_time_limit``, which is ``@pytest.mark.slow``
*and* ``skipif`` on a ``~/Dropbox/...`` path outside the repo — skipped on every CI
runner and every fresh checkout — and it runs only the default mode, which passes.
The instances here are **vendored** (``casctanks`` 32 KB, ``bchoco07`` 12 KB,
``bchoco08`` 20 KB), so this guard runs wherever the repo does, and it covers
**both** modes because the mode that broke the contract was the untested one.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

#: #875's threshold, unchanged. Quoted from its comment: "The 1.25x threshold below
#: is still the one #875 set; it was never relaxed." A slower CI box does not need a
#: looser ratio -- the wall is a wall -- but the headroom below absorbs fixed startup
#: (parse, presolve) that does not scale with the budget.
_RATIO = 1.25
_FIXED_OVERHEAD_S = 8.0


def _wall_of(name: str, budget: float, deterministic: bool) -> tuple[float, str]:
    m = from_nl(os.path.join(DATA, f"{name}.nl"))
    t0 = time.perf_counter()
    r = m.solve(time_limit=budget, deterministic=deterministic)
    return time.perf_counter() - t0, r.status


@pytest.mark.slow
@pytest.mark.parametrize("name", ["casctanks", "bchoco08"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_solve_returns_by_its_deadline(name, deterministic):
    """#1152's contract, on both modes and on instances that are in this repo.

    ``deterministic=True`` is the arm that regressed; ``False`` is the control that
    always passed, kept so a future failure can be attributed to the mode rather
    than to the machine (the same role the opt-out arm plays in #875).
    """
    budget = 30.0
    wall, status = _wall_of(name, budget, deterministic)
    assert wall < _RATIO * budget + _FIXED_OVERHEAD_S, (
        f"{name} took {wall:.1f}s against a {budget:.0f}s time_limit "
        f"({wall / budget:.2f}x) with deterministic={deterministic}; role-1 is not "
        "bounding this solve"
    )
    assert status != "infeasible", f"{name}: FALSE-INFEASIBLE on a feasible instance"


@pytest.mark.slow
def test_determinism_survives_the_role1_floor():
    """The fix must not buy the wall back by costing reproducibility.

    #1187's own configuration: ``clay0303hfsg`` at ``max_nodes=20, time_limit=120``
    returned three different incumbents (55092.52 / 46785.55 / 41573.26) before that
    fix and reproduces bit-exactly after. Role-1 is deliberately generous, so it must
    never bind and the floor must therefore change nothing -- which is the condition
    ``deterministic`` documents for reproducibility in the first place ("a solve
    reproduces when the role-1 budget never binds").
    """
    runs = []
    for _ in range(3):
        m = from_nl(os.path.join(DATA, "clay0303hfsg.nl"))
        t0 = time.perf_counter()
        r = m.solve(time_limit=120, deterministic=True, max_nodes=20)
        runs.append((r.objective, r.bound, r.node_count, time.perf_counter() - t0))

    assert len({r[0] for r in runs}) == 1, f"objective not reproducible: {runs}"
    assert len({r[1] for r in runs}) == 1, f"dual bound not reproducible: {runs}"
    assert len({r[2] for r in runs}) == 1, f"node count not reproducible: {runs}"
    # If this ever fails the test above is no longer testing what it claims: role-1
    # would be binding, and the runs could differ for a legitimate reason.
    assert max(r[3] for r in runs) < 120.0, (
        "role-1 bound on a budget chosen so it would not; this no longer isolates "
        f"the floor from ordinary budget truncation: {runs}"
    )


def test_the_role2_helpers_fall_back_to_role1_not_to_no_clock(monkeypatch):
    """The mechanism, without a solve.

    A unit-level pin because the end-to-end tests above are ``slow`` and a corpus
    can only ever sample: the defect was that these three helpers answered "no
    clock" under ``deterministic``, and that is a property of the helpers.
    """
    import dataclasses

    from discopt import solver as S

    deadline = time.perf_counter() + 45.0
    # Capture the ORIGINAL before patching: a lambda that calls ``S._tuning()`` is
    # calling the patched name, which recurses until the stack ends.
    _orig = S._tuning
    monkeypatch.setattr(S, "_tuning", lambda: dataclasses.replace(_orig(), deterministic=True))
    token = S._ROLE1_DEADLINE.set(deadline)
    try:
        assert S._role2_deadline(deadline - 40.0) == deadline, (
            "the role-2 deadline must become the ROLE-1 deadline, not None"
        )
        budget = S._role2_budget(1.0)
        assert budget is not None and 0.0 < budget <= 45.0
        horizon = S._role2_horizon(1.0)
        assert horizon != float("inf") and 0.0 < horizon <= 45.0
    finally:
        S._ROLE1_DEADLINE.reset(token)


def test_without_a_role1_deadline_the_helpers_still_report_no_clock(monkeypatch):
    """Outside a ``solve_model`` there is no role-1 wall to fall back to, and the
    helpers must say so rather than invent one — a fabricated deadline in the past
    would read as already expired and silently disable the stage."""
    import dataclasses

    from discopt import solver as S

    # Capture the ORIGINAL before patching: a lambda that calls ``S._tuning()`` is
    # calling the patched name, which recurses until the stack ends.
    _orig = S._tuning
    monkeypatch.setattr(S, "_tuning", lambda: dataclasses.replace(_orig(), deterministic=True))
    token = S._ROLE1_DEADLINE.set(None)
    try:
        assert S._role2_deadline(123.0) is None
        assert S._role2_budget(1.0) is None
        assert S._role2_horizon(1.0) == float("inf")
    finally:
        S._ROLE1_DEADLINE.reset(token)


def test_a_nested_solve_does_not_clobber_the_outer_wall():
    """``_role2_slice`` hands a nested ``solve_model`` the caller's whole
    ``time_limit``, so solves nest. Without the ``_scoped_role1_deadline`` token the
    inner solve's nearer deadline would outlive its return and the outer solve would
    spend the rest of its run measuring against a wall that had already passed —
    every role-2 fallback reading as expired, which is the stale-deadline shape that
    made the #844 fallback degrade to its cold path.
    """
    from discopt import solver as S

    outer = time.perf_counter() + 1000.0
    token = S._ROLE1_DEADLINE.set(outer)
    try:

        @S._scoped_role1_deadline
        def _inner():
            S._ROLE1_DEADLINE.set(time.perf_counter() + 1.0)
            return "done"

        assert _inner() == "done"
        assert S._ROLE1_DEADLINE.get() == outer, (
            "a nested solve's deadline escaped its scope and overwrote the outer wall"
        )
    finally:
        S._ROLE1_DEADLINE.reset(token)
