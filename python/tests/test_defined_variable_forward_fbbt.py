"""FBBT forward-substitution for variables defined by an equality (issue: nvs/gear
unbounded auxiliaries).

A variable that appears linearly and in isolation in an equality —
``c·x_def + g(others) == rhs`` with ``x_def`` absent from ``g`` — is fully
determined: ``x_def = (rhs - g)/c``. ``DefinedVariableForwardRule`` bounds it by the
interval enclosure of the defining expression. This turns the *unbounded* division/
sqrt auxiliary slacks of the nvs05/gear4 class into finite ranges, which keeps the
per-node McCormick relaxation bounded over the whole spatial tree. Without it an
unbounded-relaxation node is sentinel-pruned (an unsound fathom), permanently
tainting the dual bound so the (already-found) global optimum can never be
certified.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.nonlinear_bound_tightening import (
    _EMPTY_INTERVAL_FEAS_TOL,
    DefinedVariableForwardRule,
    NonlinearBoundTighteningInfeasible,
    _cached_flat_metadata,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import from_nl
from discopt.solver import _extract_variable_info

_DATA = Path(__file__).parent / "data" / "minlplib"


def test_forward_substitution_bounds_division_slack():
    """A free slack ``s = c/(x*y)`` over a positive box gets a finite, *sound* range."""
    m = dm.Model("recip_slack")
    x = m.continuous("x", lb=1.0, ub=10.0)
    y = m.continuous("y", lb=2.0, ub=4.0)
    s = m.continuous("s", lb=-float("inf"), ub=float("inf"))  # defined by the equality
    m.minimize(x + y)
    m.subject_to(s == 12.0 / (x * y))

    _, lb, ub, _, _ = _extract_variable_info(m)
    s_idx = [v.name for v in m._variables].index("s")
    assert not np.isfinite(lb[s_idx]) and not np.isfinite(ub[s_idx])  # free to start

    tl, tu, stats = tighten_nonlinear_bounds(m, lb.copy(), ub.copy())
    assert np.isfinite(tl[s_idx]) and np.isfinite(tu[s_idx]), "slack still unbounded"
    # True range of 12/(x*y) over x∈[1,10], y∈[2,4] is [12/40, 12/2] = [0.3, 6.0].
    # The enclosure must be sound: contain the true range, never exclude it.
    assert tl[s_idx] <= 0.3 + 1e-9
    assert tu[s_idx] >= 6.0 - 1e-9
    assert "defined_variable_forward" in stats.applied_rules


def test_forward_substitution_chains_through_definitions():
    """``b`` defined via ``a`` resolves once ``a`` is bounded (fixpoint iteration)."""
    m = dm.Model("chain")
    x = m.continuous("x", lb=1.0, ub=4.0)
    a = m.continuous("a", lb=-float("inf"), ub=float("inf"))
    b = m.continuous("b", lb=-float("inf"), ub=float("inf"))
    m.minimize(x)
    m.subject_to(a == x * x)  # a in [1, 16]
    m.subject_to(b == a + 5.0)  # b in [6, 21], only after a is bounded

    _, lb, ub, _, _ = _extract_variable_info(m)
    names = [v.name for v in m._variables]
    tl, tu, _ = tighten_nonlinear_bounds(m, lb.copy(), ub.copy())
    for nm in ("a", "b"):
        k = names.index(nm)
        assert np.isfinite(tl[k]) and np.isfinite(tu[k]), f"{nm} unbounded (chain failed)"


def test_nvs05_aux_vars_bounded():
    """nvs05's four free auxiliaries (division/sqrt slacks) all become finite."""
    m = from_nl(str(_DATA / "nvs05.nl"))
    _, lb, ub, _, _ = _extract_variable_info(m)
    tl, tu, stats = tighten_nonlinear_bounds(m, lb.copy(), ub.copy())
    assert np.all(np.isfinite(tl)) and np.all(np.isfinite(tu)), "an nvs05 var stayed unbounded"
    assert stats.n_tightened >= 8  # four aux vars x two bounds


def _crossing_model(fixed_value: float):
    """``s == x/3`` with ``x`` fixed and ``s`` free above a declared ``lb = 1.0``.

    The forward substitution derives ``s in [fixed_value/3, fixed_value/3]`` and
    intersects it with ``[1.0, inf)``. Choosing ``fixed_value`` just below ``3.0``
    makes that intersection *cross* (``lo > hi``) by a controlled amount.
    """
    m = dm.Model("crossing")
    x = m.continuous("x", lb=fixed_value, ub=fixed_value)
    s = m.continuous("s", lb=1.0, ub=float("inf"))
    m.minimize(x)
    m.subject_to(s == x / 3.0)
    return m, [v.name for v in m._variables].index("s")


def test_sub_tolerance_crossing_snaps_instead_of_raising():
    """A rounding-scale ``lo > hi`` collapses to a degenerate box, not a ValueError.

    ``2.999999999999991 / 3 == 0.999999999999997``, so the derived range crosses the
    declared ``lb = 1.0`` by 3e-15 -- exactly the ex7_3_6 shape of #1197, where the
    ``Interval(lo, hi)`` invariant used to raise straight out of
    ``tighten_nonlinear_bounds`` and take the instance out of every panel.
    """
    m, s_idx = _crossing_model(2.999999999999991)
    _, lb, ub, _, _ = _extract_variable_info(m)
    metadata = _cached_flat_metadata(m)

    rule_lb, rule_ub = DefinedVariableForwardRule().tighten(m, lb.copy(), ub.copy(), metadata)
    # Degenerate, and snapped to the enclosure's own endpoint (outward, i.e. *below*
    # the declared lb) rather than to the tighter declared bound.
    assert rule_lb[s_idx] == rule_ub[s_idx]
    # Assert the crossing this test exists for was actually exercised: strictly
    # below 1.0, and by less than the empty-interval tolerance.
    assert 0.0 < 1.0 - rule_ub[s_idx] < _EMPTY_INTERVAL_FEAS_TOL

    tl, tu, stats = tighten_nonlinear_bounds(m, lb.copy(), ub.copy())
    assert not stats.infeasible, stats.infeasibility_reason
    assert tl[s_idx] <= tu[s_idx]
    assert tu[s_idx] == pytest.approx(1.0, abs=1e-9)


def test_beyond_tolerance_crossing_is_reported_as_infeasible():
    """A crossing far past any rounding slack is a real proof, reported as one.

    ``s == 2.7/3 == 0.9`` cannot hold with ``s >= 1``: the derived interval and the
    box are genuinely disjoint (by 0.1, four orders above ``_EMPTY_INTERVAL_FEAS_TOL``).
    That must surface through ``stats.infeasible``, not as an escaping ValueError and
    not as a silently skipped row.
    """
    m, _ = _crossing_model(2.7)
    _, lb, ub, _, _ = _extract_variable_info(m)

    _, _, stats = tighten_nonlinear_bounds(m, lb.copy(), ub.copy())
    assert stats.infeasible
    assert "defined_variable_forward" in (stats.infeasibility_reason or "")

    with pytest.raises(NonlinearBoundTighteningInfeasible):
        DefinedVariableForwardRule().tighten(m, lb.copy(), ub.copy(), _cached_flat_metadata(m))


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_nvs05_certifies():
    """With the aux vars bounded, the spatial McCormick relaxation is bounded at
    every node, so the rigorous dual bound is no longer dropped on an unbounded node
    and nvs05 *certifies* its global optimum (it never did before — it reported
    ``feasible`` with a loose bound at any time limit)."""
    r = from_nl(str(_DATA / "nvs05.nl")).solve(time_limit=180, gap_tolerance=1e-4)
    assert r.status == "optimal", f"nvs05 did not certify (status={r.status})"
    assert r.gap_certified
    assert r.objective == pytest.approx(5.4709341, abs=1e-3)
    assert r.bound is not None and r.bound <= 5.4709341 + 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
