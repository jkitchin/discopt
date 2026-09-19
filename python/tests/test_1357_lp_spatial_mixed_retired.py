"""#1357: ``DISCOPT_LP_SPATIAL_MIXED`` is retired; the production paths use the
pre-#860 gate, and the ``mixed=`` capability survives as an explicit-caller option.

This file replaces ``test_860_mixed_gate_is_opt_in.py``, whose subject was the flag.
The guarantee #860 wanted — the widening must never reach the default path by
accident — is *stronger* now: there is no env var that can turn it on, so these tests
pin that the call sites do not widen and that the capability is still reachable and
still sound when asked for explicitly.

**Why the flag was retired.** It ran its CLAUDE.md §5 graduation panel and failed bar
(2). Cert-clean (0 certification regressions, 0 ``incumbent_verification_failed``, 0
unsound bounds) but not net-positive: ``gains=1 improved=1 lost_incumbents=2``. Two
instances lost their incumbent outright because the #844 reserve handed 35% of the
budget to a fallback that then declined the model anyway. On the explicit
``solve(lp_spatial=True)`` path ``gear4`` went from ``optimal`` 1.6434284641 certified
in 3 nodes to ``time_limit`` 17.514 uncertified in 2673 nodes. Under §5's retirement
rule a measured-harmful flag is deleted rather than left default-OFF forever.

These tests pin *gating*, never wall-clock, so they cannot rot into machine-speed
assertions.
"""

from __future__ import annotations

import inspect

import discopt.modeling as dm
from discopt._relax.lp_spatial_bb import _is_in_scope, solve_lp_spatial_bb


def _mixed_minimize():
    """One integer + one continuous: in scope only under the widening."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=0, ub=5)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 2)
    return m


def _pure_integer_minimize():
    """In scope under the pre-#860 gate too."""
    m = dm.Model("pure")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(x * y)
    m.subject_to(x + y >= 2)
    return m


# ---------------------------------------------------------------------------
# The flag is gone
# ---------------------------------------------------------------------------


def test_the_retired_flag_is_never_read_at_runtime():
    """A retirement that leaves the env read behind is not a retirement.

    Deliberately forbids the *read*, not the *name*: the comments explaining why the
    flag was retired have to keep naming it, or the next reader loses the reasoning
    along with the flag. An earlier draft of this test banned the bare name and failed
    on exactly those comments.
    """
    import pathlib
    import re

    import discopt

    pkg = pathlib.Path(discopt.__file__).parent
    read = re.compile(
        r'environ\.get\(\s*["\']DISCOPT_LP_SPATIAL_MIXED|_lp_spatial_mixed_fallback_enabled\s*\('
    )
    offenders = [
        f"{path.relative_to(pkg)}:{i}"
        for path in pkg.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if read.search(line)
    ]
    assert not offenders, f"retired flag is still READ at: {offenders}"


def test_the_gate_helper_no_longer_exists():
    from discopt.modeling import core

    assert not hasattr(core, "_lp_spatial_mixed_fallback_enabled")


# ---------------------------------------------------------------------------
# The production paths take the pre-#860 gate
# ---------------------------------------------------------------------------


def test_neither_production_call_site_widens_the_gate():
    """Source-level guard, the same shape #860's test used — inverted.

    #860 asserted both call sites *passed the flag*. With the flag retired the
    guarantee is that neither passes ``mixed=`` at all, so both inherit ``False``.
    """
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
        assert "mixed=" not in window, (
            f"{mod.__name__}: {needle!r} passes mixed=; the widening would reach the "
            "default path, which #1357 retired"
        )
        checked += 1
    assert checked == 2


def test_both_entry_points_default_to_the_pre_860_gate():
    for fn in (_is_in_scope, solve_lp_spatial_bb):
        param = inspect.signature(fn).parameters["mixed"]
        assert param.default is False, (
            f"{fn.__name__}(mixed=...) defaults to {param.default!r}; it must default "
            "to False so a new call site inherits the conservative gate"
        )


def test_a_mixed_model_is_declined_by_default():
    """The retirement's actual effect: the default gate refuses the mixed class."""
    assert _is_in_scope(_mixed_minimize()) is False
    assert solve_lp_spatial_bb(_mixed_minimize(), time_limit=5) is None


def test_a_pure_integer_model_is_still_accepted():
    """Retiring the widening must not narrow the engine's original scope."""
    assert _is_in_scope(_pure_integer_minimize()) is True


# ---------------------------------------------------------------------------
# The capability survives, on request
# ---------------------------------------------------------------------------


def test_the_widening_is_still_reachable_explicitly():
    """§5 retires the entry point, not the reusable piece.

    The docstring on ``_is_in_scope`` carries the measurement that retired the flag, so
    a future caller weighing this does not have to re-derive it.
    """
    assert _is_in_scope(_mixed_minimize(), mixed=True) is True
    assert "1357" in (_is_in_scope.__doc__ or ""), (
        "the retirement measurement must stay at the point of use"
    )
