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
from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds
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
