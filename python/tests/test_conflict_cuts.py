"""Tests for conflict analysis / no-good cuts (discopt.conflict).

The correctness-critical property: a no-good cut excludes only an FBBT-proven
infeasible assignment, never a feasible point — so the feasible region's optimum
is unchanged. These tests pin that property directly.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import pytest

from discopt.conflict import add_conflict_cuts, find_conflict_cuts, no_good_cut


def _packing_model() -> tuple[dm.Model, list]:
    """max x0 + 2 x1 + 3 x2 s.t. x0+x1+x2 <= 1 — any two ones is infeasible."""
    m = dm.Model("conflict")
    x = [m.binary(f"x{i}") for i in range(3)]
    m.maximize(x[0] + 2 * x[1] + 3 * x[2])
    m.subject_to(x[0] + x[1] + x[2] <= 1)
    return m, x


# ───────────────────────── no_good_cut semantics ─────────────────────────


def test_no_good_cut_excludes_only_target():
    """Excluding (x0=1, x1=1) must drop only that combination."""
    m = dm.Model("ng")
    x0 = m.binary("x0")
    x1 = m.binary("x1")
    m.maximize(x0 + x1)  # unconstrained optimum is (1, 1) -> 2
    m.subject_to(no_good_cut([(x0, 1), (x1, 1)]))
    res = m.solve()
    assert res.status == "optimal"
    # (1,1) is excluded; the best remaining is a single one -> objective 1.
    assert abs(float(res.objective) - 1.0) < 1e-6


def test_no_good_cut_empty_assignment_raises():
    with pytest.raises(ValueError):
        no_good_cut([])


# ───────────────────────── conflict detection ─────────────────────────


def test_finds_pairwise_conflicts():
    m, _ = _packing_model()
    cuts = find_conflict_cuts(m, max_order=2)
    # The three pairs (0,1), (0,2), (1,2) set to (1,1) are each infeasible.
    assert len(cuts) == 3


def test_conflict_detection_restores_bounds():
    m, x = _packing_model()
    before = [(float(v.lb), float(v.ub)) for v in m._variables]
    find_conflict_cuts(m, max_order=2)
    after = [(float(v.lb), float(v.ub)) for v in m._variables]
    assert before == after


def test_minimal_conflicts_only():
    """An order-1 conflict must suppress its order-2 supersets."""
    m = dm.Model("min")
    x = [m.binary(f"x{i}") for i in range(3)]
    m.maximize(x[0] + x[1] + x[2])
    m.subject_to(x[0] <= 0)  # x0 = 1 is infeasible on its own
    cuts = find_conflict_cuts(m, max_order=2)
    # The minimal conflict {x0=1} is found; no superset pair conflict involving
    # x0=1 is reported (it is already covered).
    assert len(cuts) == 1


def test_no_binaries_returns_empty():
    m = dm.Model("cont")
    y = m.continuous("y", lb=0, ub=5)
    m.minimize((y - 1) ** 2)
    assert find_conflict_cuts(m) == []


# ───────────────────────── soundness: optimum preserved ─────────────────────────


def test_adding_conflict_cuts_preserves_optimum():
    m1, _ = _packing_model()
    opt_without = float(m1.solve().objective)

    m2, _ = _packing_model()
    n = add_conflict_cuts(m2, max_order=2)
    assert n == 3
    opt_with = float(m2.solve().objective)

    # No-good cuts only remove proven-infeasible assignments -> optimum identical.
    assert abs(opt_without - opt_with) < 1e-6
    assert abs(opt_with - 3.0) < 1e-6  # x2 = 1
