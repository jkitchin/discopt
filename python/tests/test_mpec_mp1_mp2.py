"""Regression tests for MPEC fixes MP-1 and MP-2.

- **MP-1** (`mpec.py:tighten_complementarity_bounds`): fixing a complementarity
  partner to 0 overwrote its lower bound as well as its upper bound. When the
  partner carried a strictly positive lower bound the pair is genuinely infeasible,
  but the old code silently collapsed the box to ``[0, 0]`` and reported
  ``n_fixed=1`` — a subsequent solve then certifies the infeasible model optimal.
  The fix intersects only the upper bound and raises on the infeasible case.
- **MP-2** (`mpec.py:solve_mpec`): a first-iteration NLP failure was swallowed by
  ``except BaseException: break`` and returned ``None`` silently. The fix narrows
  the catch and surfaces a first-iteration failure as a ``RuntimeError``.

Each fails on the pre-fix code.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.mpec import complementarity, solve_mpec, tighten_complementarity_bounds

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------- MP-1
def test_mp1_infeasible_pair_raises_not_hidden():
    """0 <= a _|_ b >= 0 with a.lb=b.lb=0.5 is infeasible -> must raise."""
    m = dm.Model("infeas")
    a = m.continuous("a", lb=0.5, ub=5.0)
    b = m.continuous("b", lb=0.5, ub=5.0)
    with pytest.raises(ValueError, match="infeasible"):
        tighten_complementarity_bounds(m, [complementarity(a, b, name="ab")])


def test_mp1_feasible_case_still_fixes_to_zero():
    """The blessed case (partner lb=0) still fixes the partner to [0, 0]."""
    m = dm.Model("feas")
    a = m.continuous("a", lb=0.5, ub=3.0)
    b = m.continuous("b", lb=0.0, ub=3.0)
    n_fixed = tighten_complementarity_bounds(m, [complementarity(a, b)])
    assert n_fixed == 1
    assert float(b.lb) == 0.0 and float(b.ub) == 0.0
    assert float(a.lb) == 0.5  # driver untouched


def test_mp1_intersects_tighter_upper_bound():
    """Fixing to zero intersects ub (never widens it) and keeps lb=0."""
    m = dm.Model("intersect")
    a = m.continuous("a", lb=1.0, ub=5.0)
    b = m.continuous("b", lb=0.0, ub=0.2)
    tighten_complementarity_bounds(m, [complementarity(a, b)])
    assert float(b.lb) == 0.0 and float(b.ub) == 0.0


def test_mp1_no_fix_when_both_sides_free():
    m = dm.Model("free")
    a = m.continuous("a", lb=0.0, ub=3.0)
    b = m.continuous("b", lb=0.0, ub=3.0)
    assert tighten_complementarity_bounds(m, [complementarity(a, b)]) == 0


# ---------------------------------------------------------------------------- MP-2
def test_mp2_first_iteration_solver_failure_surfaces(monkeypatch):
    """A first-iteration NLP failure must raise, not silently return None."""

    def _boom(*_args, **_kwargs):
        raise ValueError("backend exploded")

    def _fake_get_nlp_solver(_name):
        return _boom

    monkeypatch.setattr("discopt.solvers.nlp_backend.get_nlp_solver", _fake_get_nlp_solver)

    m = dm.Model("mp2")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize(x + y)
    with pytest.raises(RuntimeError, match="first homotopy iteration"):
        solve_mpec(m, [complementarity(x, y)], max_iter=4)
