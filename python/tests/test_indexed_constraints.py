"""Tests for indexed constraints over named sets (Phase 7 M3)."""

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Constraint
from discopt.modeling.indexed import IndexedConstraint, Skip


class TestModelConstraint:
    def test_one_constraint_per_member(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        x = m.continuous("x", over=plants, lb=0)
        ic = m.constraint(plants, lambda p: x[p] <= 100, name="cap")
        assert isinstance(ic, IndexedConstraint)
        assert len(ic) == 3
        assert len(m._constraints) == 3

    def test_key_tuple_naming(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        x = m.continuous("x", over=plants, lb=0)
        m.constraint(plants, lambda p: x[p] <= 100, name="cap")
        names = {c.name for c in m._constraints}
        assert names == {"cap[pitt]", "cap[sf]"}

    def test_tuple_member_unpacks_into_rule(self):
        m = dm.Model()
        links = m.set("links", [("pitt", "a"), ("sf", "b")])
        x = m.continuous("x", over=links, lb=0)
        m.constraint(links, lambda p, k: x[p, k] <= 10, name="flow")
        names = {c.name for c in m._constraints}
        assert names == {"flow[pitt,a]", "flow[sf,b]"}

    def test_skip_omits_member(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3, 4])
        x = m.continuous("x", over=s, lb=0)
        ic = m.constraint(s, lambda i: (x[i] <= 1) if i % 2 == 0 else Skip, name="even")
        assert len(ic) == 2
        assert set(ic.keys()) == {2, 4}
        assert len(m._constraints) == 2

    def test_getitem_returns_constraint(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        x = m.continuous("x", over=plants, lb=0)
        ic = m.constraint(plants, lambda p: x[p] <= 100, name="cap")
        c = ic["pitt"]
        assert isinstance(c, Constraint)
        assert c.name == "cap[pitt]"

    def test_getitem_bad_key_raises(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        x = m.continuous("x", over=plants, lb=0)
        ic = m.constraint(plants, lambda p: x[p] <= 100, name="cap")
        with pytest.raises(KeyError, match="no constraint for key"):
            _ = ic["nyc"]

    def test_rule_returning_non_constraint_raises(self):
        m = dm.Model()
        s = m.set("s", [1, 2])
        with pytest.raises(TypeError, match="expected a Constraint"):
            m.constraint(s, lambda i: 42, name="bad")

    def test_aggregation_over_set_in_rule(self):
        # supply constraint: sum over markets of ship[p, k] <= cap[p]
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        markets = m.set("markets", ["a", "b", "c"])
        ship = m.continuous("ship", over=plants * markets, lb=0)
        ic = m.constraint(
            plants,
            lambda p: dm.sum(ship[p, k] for k in markets) <= 100,
            name="supply",
        )
        assert len(ic) == 2
        assert len(m._constraints) == 2


class TestSubjectToIterables:
    def test_generator_of_constraints(self):
        m = dm.Model()
        s = m.set("s", [0, 1, 2])
        x = m.continuous("x", over=s, lb=0)
        m.subject_to((x[i] <= 5 for i in s), name="ub")
        assert len(m._constraints) == 3
        assert {c.name for c in m._constraints} == {"ub_0", "ub_1", "ub_2"}

    def test_list_still_works(self):
        m = dm.Model()
        x = m.continuous("x", shape=(3,), lb=0)
        m.subject_to([x[i] <= 5 for i in range(3)], name="ub")
        assert len(m._constraints) == 3

    def test_single_constraint_still_works(self):
        m = dm.Model()
        x = m.continuous("x", shape=(2,), lb=0)
        m.subject_to(x[0] + x[1] <= 10, name="c")
        assert len(m._constraints) == 1
        assert m._constraints[0].name == "c"

    def test_bool_raises(self):
        m = dm.Model()
        with pytest.raises(TypeError, match="Did you mean to compare"):
            m.subject_to(True)

    def test_iterable_with_non_constraint_raises(self):
        m = dm.Model()
        with pytest.raises(TypeError, match="position 1"):
            m.subject_to([dm.Model().continuous("z") <= 1, 99])


class TestExports:
    def test_skip_and_indexedconstraint_exported(self):
        assert dm.Skip is Skip
        assert dm.IndexedConstraint is IndexedConstraint
