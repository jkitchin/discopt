"""#862: depth-first plunging gives the node loop a primal source.

The LP-per-node engine can only produce an incumbent from the node loop at an **exact
leaf** (a fully-fixed box, where every nonlinear term is determined). Best-first never
reaches one on the `tln`/`nvs` family — depth 32 with ``fully_fixed = 0`` in 2000+
nodes — so the primal was left entirely to rounding heuristics reading a relaxation
that is ~250x loose (2562/2562 roundings infeasible; see
``docs/dev/lp-node-primal-quality.md``). Plunging commits to one child depth-first
until an exact leaf, a fathom, or ``_PLUNGE_MAX_DEPTH``.

Measured on #862's own metric — the #844 fallback panel, 60 s, oracles from
``minlplib.solu``:

======  ==============  ==============  =========
inst    before          after           optimum
======  ==============  ==============  =========
tln4    9.3  (+12.0%)   **8.3 (exact)** 8.3
tln5    32.2 (+212.6%)  10.8 (+4.9%)    10.3
tln6    65.3 (+326.8%)  16.2 (+5.9%)    15.3
======  ==============  ==============  =========

worst gap 0.7657 -> 0.0556, mean 0.5178 -> 0.0340; panel reports 0 cert regressions,
0 unsound, 0 false primals, 0 overshoots.

**Default: ON for the fallback, OFF for the general engine.** The fallback exists only
to find a primal (``require_incremental=True``, output is an incumbent, never a
certificate), so trading dual progress for depth is its whole job. The general engine
is asked to *prove* optimality and plunging costs it: ``gear2`` certifies in 657 nodes
/ 7.2 s best-first versus 6017 nodes / 61.3 s plunging (same answer, both ~0), which
at a 20 s budget reads as a certification regression. Scoping the default keeps the
gains where they belong and leaves the proving path bit-identical.

These tests pin the **soundness invariant and the wiring**, never wall-clock or node
counts: at a *time* limit those are not reproducible — two runs of the identical
configuration over the 58-instance in-repo corpus produced 23 differences, including
``gear4``'s objective moving 31.5 -> 17.5, so a node-count assertion here would be a
flake generator.
"""

from __future__ import annotations

import pathlib

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.lp_spatial_bb import (
    _PLUNGE_MAX_DEPTH,
    _plunge_enabled,
    solve_lp_spatial_bb,
)

# --------------------------------------------------------------------------- #
# Wiring: where the default is on, and how the env var overrides it
# --------------------------------------------------------------------------- #


def test_default_is_on_for_the_fallback_and_off_for_the_engine(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_SPATIAL_PLUNGE", raising=False)
    assert _plunge_enabled(require_incremental=True) is True, (
        "the #844 fallback's job is a primal; plunging must be on there by default"
    )
    assert _plunge_enabled(require_incremental=False) is False, (
        "the general engine must prove optimality; plunging must be off there by default"
    )
    assert _plunge_enabled() is False, "the bare default must be the conservative one"


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("0", False), ("false", False)],
)
def test_env_var_forces_both_ways_including_the_fallback(monkeypatch, raw, expected):
    """An explicit setting overrides the per-path default in BOTH directions.

    ``=0`` must switch the fallback off too, otherwise there is no way to reproduce
    the pre-#862 behaviour for a bisect.

    The empty string is deliberately NOT in this table — see
    :func:`test_empty_value_means_unset_not_off`.
    """
    monkeypatch.setenv("DISCOPT_LP_SPATIAL_PLUNGE", raw)
    assert _plunge_enabled(require_incremental=True) is expected
    assert _plunge_enabled(require_incremental=False) is expected


def test_empty_value_means_unset_not_off(monkeypatch):
    """``DISCOPT_LP_SPATIAL_PLUNGE=`` resolves to each path's own default.

    Before the flag-parse unification the empty string was an *off* spelling here,
    while three other idioms read it as *on* — the same value meant opposite things
    depending on which flag you set it on (architecture review §2.4). One table now
    maps unset and empty alike to the caller's default (CHANGELOG "Unreleased →
    Fixed"; ``docs/reference/flags.md``), so this asserts empty is indistinguishable
    from unset rather than pinning a single boolean: the two paths have *different*
    defaults, which is the whole point of the #862 scoping.
    """
    monkeypatch.delenv("DISCOPT_LP_SPATIAL_PLUNGE", raising=False)
    unset_fallback = _plunge_enabled(require_incremental=True)
    unset_engine = _plunge_enabled(require_incremental=False)
    # The defaults this flag exists to scope: ON for the #844 fallback, OFF elsewhere.
    assert unset_fallback is True
    assert unset_engine is False

    monkeypatch.setenv("DISCOPT_LP_SPATIAL_PLUNGE", "")
    assert _plunge_enabled(require_incremental=True) is unset_fallback
    assert _plunge_enabled(require_incremental=False) is unset_engine

    # ...and the =0 escape hatch still switches both off, unchanged by the unification.
    monkeypatch.setenv("DISCOPT_LP_SPATIAL_PLUNGE", "0")
    assert _plunge_enabled(require_incremental=True) is False
    assert _plunge_enabled(require_incremental=False) is False


def test_plunge_depth_cap_is_finite():
    assert 0 < _PLUNGE_MAX_DEPTH < 10_000


# --------------------------------------------------------------------------- #
# Soundness: the generalized global lower bound
# --------------------------------------------------------------------------- #


def _integer_product_model(n=6, ub=4):
    """Pure-integer product model: in scope, needs branching, has a known optimum.

    ``min sum_i (x_i * x_{i+1}) - 3*sum_i x_i`` over ``[0, ub]^n`` with a knapsack row.
    The exact optimum is found by brute force below, so the certificate can be checked
    against ground truth rather than against the engine's own opinion.
    """
    m = dm.Model(f"prod{n}")
    xs = [m.integer(f"x{i}", lb=0, ub=ub) for i in range(n)]
    obj = sum(xs[i] * xs[i + 1] for i in range(n - 1)) - 3 * sum(xs)
    m.minimize(obj)
    m.subject_to(sum(xs) <= 2 * ub)
    m.subject_to(sum(xs) >= 2)
    return m


def _brute_force(n=6, ub=4):
    import itertools

    best = float("inf")
    for pt in itertools.product(range(ub + 1), repeat=n):
        s = sum(pt)
        if not (2 <= s <= 2 * ub):
            continue
        val = sum(pt[i] * pt[i + 1] for i in range(n - 1)) - 3 * s
        best = min(best, val)
    return best


@pytest.mark.parametrize("plunge", ["0", "1"])
def test_bound_never_crosses_the_true_optimum(monkeypatch, plunge):
    """The core §1 invariant, under both node orders, against a brute-forced optimum.

    Plunging pops a node that is NOT the frontier minimum. The in-loop global lower
    bound used to be read off the popped node, which is only the minimum under
    best-first; reading it that way while plunging would report a bound ABOVE the truth
    and could certify optimality over space still on the heap. This is the test that
    would catch that.
    """
    monkeypatch.setenv("DISCOPT_LP_SPATIAL_PLUNGE", plunge)
    n, ub = 6, 4
    true_opt = _brute_force(n, ub)
    r = solve_lp_spatial_bb(_integer_product_model(n, ub), time_limit=30, gap_tolerance=1e-6)
    assert r is not None, "model should be in scope"
    checks = 0
    if r.bound is not None and np.isfinite(r.bound):
        checks += 1
        assert r.bound <= true_opt + 1e-6 * (1 + abs(true_opt)), (
            f"plunge={plunge}: dual bound {r.bound!r} exceeds the true optimum "
            f"{true_opt!r} — the global lower bound is being read off a non-minimal node"
        )
    if r.objective is not None and np.isfinite(r.objective):
        checks += 1
        assert r.objective >= true_opt - 1e-6 * (1 + abs(true_opt)), (
            f"plunge={plunge}: objective {r.objective!r} is BELOW the true optimum "
            f"{true_opt!r} — a false primal"
        )
    if r.status == "optimal":
        checks += 1
        assert r.objective == pytest.approx(true_opt, rel=1e-6, abs=1e-6), (
            f"plunge={plunge}: certified optimal at {r.objective!r}, true optimum is {true_opt!r}"
        )
    assert checks >= 1, f"plunge={plunge}: no assertion fired — the probe is vacuous"


def test_plunging_actually_changes_the_search(monkeypatch):
    """Guard against the flag being an accidental no-op.

    Without this, every other test here could pass while plunging never engaged. Uses
    ``nvs17`` — the instance the #862 diagnosis is built on — under a fixed **node**
    budget rather than a time limit, so the comparison is reproducible: at a *time*
    limit two runs of the identical configuration differ (23 differences over the
    58-instance corpus), which would make this a flake.

    A synthetic product model was tried first and solved at the root in 1 node, so
    plunging never fired and this assertion caught it — which is the point of having it.
    """
    from discopt.modeling.core import from_nl

    nl = pathlib.Path(__file__).parent / "data" / "minlplib" / "nvs17.nl"
    assert nl.exists(), f"missing fixture {nl}"
    seen = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_LP_SPATIAL_PLUNGE", flag)
        r = solve_lp_spatial_bb(
            from_nl(str(nl)), time_limit=300, max_nodes=300, gap_tolerance=1e-12
        )
        assert r is not None, "nvs17 should be in scope"
        seen[flag] = (
            r.node_count,
            None if r.bound is None else round(float(r.bound), 6),
            None if r.objective is None else round(float(r.objective), 6),
        )
    assert seen["0"] != seen["1"], (
        f"plunge on and off produced identical search state {seen} at a fixed node "
        "budget — the flag is a no-op and the rest of this file proves nothing"
    )
    # And the plunge arm must not be unsound on this instance (oracle -1100.4).
    for flag, (_nodes, bound, obj) in seen.items():
        if bound is not None:
            assert bound <= -1100.4 + 1e-4, f"plunge={flag}: bound {bound} above the optimum"
        if obj is not None:
            assert obj >= -1100.4 - 1e-4, f"plunge={flag}: objective {obj} below the optimum"
