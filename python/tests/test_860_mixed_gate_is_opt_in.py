"""#860: the mixed/MAXIMIZE scope widening must be opt-in at *every* entry point.

The widening is a real capability and it is sound (the ``gear4`` false certificate
that first appeared with it was the LP-presolve ``INF``-sentinel bug, fixed in #877,
not the widening). What it is not is *net-positive* on the default path, which is
CLAUDE.md §5 bar (2). Measured on ``gear4`` — mixed (4 integer, 2 continuous with
infinite upper bounds), so admitted only under the widening — at a 25 s budget:

===========================  ==================================================
``DISCOPT_LP_SPATIAL_MIXED``  result
===========================  ==================================================
``0`` (default)              ``optimal``, 1.6434284641, certified, 3 nodes, 1.1 s
``1``                        ``time_limit``, 17.514, uncertified, 2678 nodes, 25 s
===========================  ==================================================

The engine accepts a model the default path already certified in 3 nodes, then spends
the whole budget to return an incumbent ~10.7x worse with no certificate. Sound, but a
regression — and ``lp_spatial=True`` is a documented public kwarg, so shipping the
widened gate by default would trade a certificate for a worse incumbent.

These tests pin the *gating*, not wall-clock, so they cannot rot into machine-speed
assertions.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.lp_spatial_bb import _is_in_scope


def _mixed_minimize():
    """One integer + one continuous: in scope only under the widening."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=0, ub=5)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 2)
    return m


def _pure_integer_maximize():
    """All integer but MAXIMIZE: in scope only under the widening."""
    m = dm.Model("intmax")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.maximize(x * y)
    m.subject_to(x + y <= 6)
    return m


def _pure_integer_minimize():
    """The pre-#860 class: in scope either way (control)."""
    m = dm.Model("intmin")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(x * y)
    m.subject_to(x + y >= 2)
    return m


@pytest.mark.parametrize(
    "build,widened_only",
    [(_mixed_minimize, True), (_pure_integer_maximize, True), (_pure_integer_minimize, False)],
    ids=["mixed_minimize", "pure_integer_maximize", "pure_integer_minimize_control"],
)
def test_scope_depends_on_the_mixed_keyword(build, widened_only):
    """``mixed=True`` admits the widened classes; ``mixed=False`` is the pre-#860 gate."""
    m = build()
    assert _is_in_scope(m, mixed=True) is True, "widened gate should admit this model"
    assert _is_in_scope(m, mixed=False) is (not widened_only), (
        "pre-#860 gate admitted a widened-only class (or rejected the control)"
    )


def test_mixed_defaults_to_false_so_a_new_call_site_is_conservative():
    """The default must be the conservative gate.

    A new call site that forgets to pass ``mixed=`` must inherit the pre-#860 scope
    rather than silently shipping a widening that did not graduate — the same reason
    ``row_scan_is_anytime`` defaults to ``False``.
    """
    assert _is_in_scope(_mixed_minimize()) is False, (
        "_is_in_scope defaults to the WIDENED gate; a new caller would ship it by default"
    )
    assert _is_in_scope(_pure_integer_maximize()) is False
    # control: the default must still admit the class the engine has always served
    assert _is_in_scope(_pure_integer_minimize()) is True

    import inspect

    from discopt._relax.lp_spatial_bb import solve_lp_spatial_bb

    param = inspect.signature(solve_lp_spatial_bb).parameters["mixed"]
    assert param.default is False, (
        f"solve_lp_spatial_bb(mixed=...) defaults to {param.default!r}; it must default to "
        "False so an unflagged caller gets the pre-#860 gate"
    )


def test_both_production_call_sites_pass_the_flag():
    """Source-level guard: neither entry point may rely on the default.

    Cheap, and it catches the exact regression this test file exists to prevent — a
    call site that stops threading the flag and re-widens the default path.
    """
    import inspect

    from discopt import solver as solver_mod
    from discopt.modeling import core as core_mod

    checked = 0
    for mod, needle in (
        (solver_mod, "solve_lp_spatial_bb("),
        (core_mod, "_is_in_scope(self"),
    ):
        src = inspect.getsource(mod)
        idx = src.find(needle)
        assert idx != -1, f"{needle!r} not found in {mod.__name__} — test is stale"
        window = src[idx : idx + 600]
        assert "_lp_spatial_mixed_fallback_enabled()" in window, (
            f"{mod.__name__}: {needle!r} does not pass "
            f"mixed=_lp_spatial_mixed_fallback_enabled(); the widening would ship by default"
        )
        checked += 1
    assert checked == 2


def test_flag_helper_defaults_off(monkeypatch):
    """``DISCOPT_LP_SPATIAL_MIXED`` unset means OFF, and ``1`` turns it on.

    Deliberately does NOT ``importlib.reload`` anything. The helper reads
    ``os.environ`` directly with no caching, so a reload buys nothing — and reloading
    ``solver_tuning`` mid-session replaces the ``SolverTuning`` class object, after
    which unrelated ``isinstance(..., SolverTuning)`` assertions elsewhere in the same
    worker fail. A first draft of this test did exactly that and broke
    ``test_result_io`` and ``test_solve_daemon`` in CI.
    """
    from discopt.modeling.core import _lp_spatial_mixed_fallback_enabled

    monkeypatch.delenv("DISCOPT_LP_SPATIAL_MIXED", raising=False)
    assert _lp_spatial_mixed_fallback_enabled() is False, "the widening must default OFF"

    monkeypatch.setenv("DISCOPT_LP_SPATIAL_MIXED", "1")
    assert _lp_spatial_mixed_fallback_enabled() is True, "the opt-in must work"

    # "0" and other falsy spellings stay off — the helper's own contract.
    monkeypatch.setenv("DISCOPT_LP_SPATIAL_MIXED", "0")
    assert _lp_spatial_mixed_fallback_enabled() is False
