"""Tests for Irreducible Infeasible Subsystem (IIS) computation.

An IIS must be (a) infeasible on its own and (b) irreducible — removing any one
member restores feasibility. These tests pin both properties plus the API
contract (raises on feasible models, finds bound- and integer-driven conflicts).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest
from discopt.infeasibility import IISResult, compute_iis


def _names(iis: IISResult) -> set[str]:
    return {c.name for c in iis.constraints}


def _is_infeasible(build) -> bool:
    return build().solve(time_limit=30).status == "infeasible"


# ───────────────────────── constraint conflicts ─────────────────────────


def test_isolates_two_constraint_conflict_from_red_herrings():
    m = dm.Model("c")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x >= 5, name="x_big")
    m.subject_to(x <= 2, name="x_small")
    m.subject_to(y <= 3, name="rh1")
    m.subject_to(x + y <= 8, name="rh2")
    m.minimize(x + y)

    iis = m.compute_iis()
    assert _names(iis) == {"x_big", "x_small"}
    assert iis.proven_irreducible


def test_iis_is_infeasible_and_irreducible():
    """Property test: the IIS is infeasible, and dropping any member feasibilizes."""
    m = dm.Model("c")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x >= 5, name="a")
    m.subject_to(x <= 2, name="b")
    m.subject_to(x <= 9, name="loose")  # redundant red herring
    m.minimize(x)

    iis = compute_iis(m, include_bounds=False)
    members = list(iis.constraints)
    assert len(members) >= 2

    # (a) The IIS alone is infeasible.
    def _with(cons):
        mm = dm.Model("sub")
        xx = mm.continuous("x", lb=-10, ub=10)
        mm.minimize(xx)
        # rebuild the member constraints against the fresh variable
        mapping = {"a": xx >= 5, "b": xx <= 2, "loose": xx <= 9}
        for c in cons:
            mm.subject_to(mapping[c.name])
        return mm

    assert _is_infeasible(lambda: _with(members))
    # (b) Dropping any single member restores feasibility.
    for drop in members:
        rest = [c for c in members if c is not drop]
        assert not _is_infeasible(lambda: _with(rest)), f"dropping {drop.name} stayed infeasible"


# ───────────────────────── bound-driven conflicts ─────────────────────────


def test_finds_bound_vs_constraint_conflict():
    m = dm.Model("b")
    x = m.continuous("x", lb=5, ub=10)  # lower bound 5 is the culprit
    z = m.continuous("z", lb=-1, ub=1)
    m.subject_to(x <= 2, name="cap")
    m.subject_to(z >= -0.5, name="unrelated")
    m.minimize(x)

    iis = m.compute_iis()
    assert _names(iis) == {"cap"}
    assert [(v.name, side) for v, side in iis.variable_bounds] == [("x", "lower")]


def test_include_bounds_false_skips_bound_members():
    m = dm.Model("b")
    x = m.continuous("x", lb=5, ub=10)
    m.subject_to(x <= 2, name="cap")
    m.minimize(x)

    # Without bounds, the single constraint cannot explain the infeasibility, so
    # the (constraint-only) reduction keeps the constraint but is not minimal.
    iis = compute_iis(m, include_bounds=False)
    assert iis.variable_bounds == []
    assert _names(iis) == {"cap"}


# ───────────────────────── integer-driven conflict ─────────────────────────


def test_integer_domain_conflict():
    # 3 < y < 4 with y integer is infeasible; the two constraints form the IIS.
    m = dm.Model("i")
    y = m.integer("y", lb=0, ub=10)
    m.subject_to(y >= 3.2, name="lo")
    m.subject_to(y <= 3.8, name="hi")
    m.minimize(y)
    assert m.solve(time_limit=30).status == "infeasible"

    iis = m.compute_iis(include_bounds=False)
    assert _names(iis) == {"lo", "hi"}


# ───────────────────────── API contract ─────────────────────────


def test_raises_on_feasible_model():
    m = dm.Model("f")
    a = m.continuous("a", lb=0, ub=5)
    m.subject_to(a <= 3)
    m.minimize(a)
    with pytest.raises(ValueError, match="feasible"):
        m.compute_iis()


def test_result_len_and_bool_and_summary():
    m = dm.Model("c")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x >= 5, name="a")
    m.subject_to(x <= 2, name="b")
    m.minimize(x)
    iis = m.compute_iis(include_bounds=False)
    assert len(iis) == 2
    assert bool(iis) is True
    text = iis.summary()
    assert "Irreducible Infeasible Subsystem" in text
    assert "a" in text and "b" in text
