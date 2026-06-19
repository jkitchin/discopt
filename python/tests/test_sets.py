"""Tests for named index sets and set algebra (Phase 7: set & index abstractions).

Milestone M1: the :class:`~discopt.modeling.sets.Set` algebra layer and
``Model.set()`` registration. Indexed variables/constraints are exercised in
later milestones.
"""

import discopt.modeling as dm
import pytest
from discopt.modeling.sets import ProductSet, RangeSet, Set


class TestSetConstruction:
    def test_scalar_members_dimen1(self):
        s = Set("plants", ["pitt", "sf", "nyc"])
        assert s.dimen == 1
        assert len(s) == 3
        assert list(s) == ["pitt", "sf", "nyc"]

    def test_tuple_members_infer_dimen(self):
        s = Set("links", [("pitt", "a"), ("sf", "b")])
        assert s.dimen == 2
        assert ("pitt", "a") in s

    def test_lists_coerced_to_tuples(self):
        s = Set("links", [["pitt", "a"], ["sf", "b"]])
        assert s.dimen == 2
        assert ("pitt", "a") in s
        assert all(isinstance(m, tuple) for m in s)

    def test_one_tuple_unwrapped_to_scalar(self):
        s = Set("plants", [("pitt",), ("sf",)])
        assert s.dimen == 1
        assert "pitt" in s
        assert list(s) == ["pitt", "sf"]

    def test_duplicates_removed_order_preserved(self):
        s = Set("p", ["b", "a", "b", "c", "a"])
        assert list(s) == ["b", "a", "c"]

    def test_empty_defaults_dimen1(self):
        s = Set("empty", [])
        assert s.dimen == 1
        assert len(s) == 0

    def test_dimen_override(self):
        s = Set("links", [("a", "b")], dimen=2)
        assert s.dimen == 2

    def test_inconsistent_arity_raises(self):
        with pytest.raises(ValueError, match="arity"):
            Set("bad", [("a", "b"), "c"])

    def test_dimen_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimen"):
            Set("bad", [("a", "b")], dimen=3)

    def test_dimen_below_one_raises(self):
        with pytest.raises(ValueError, match="dimen must be"):
            Set("bad", [1, 2], dimen=0)

    def test_ordinal(self):
        s = Set("p", ["x", "y", "z"])
        assert s.ordinal("y") == 1
        with pytest.raises(KeyError, match="not a member"):
            s.ordinal("q")

    def test_contains_normalizes(self):
        s = Set("links", [("a", "b")])
        assert ("a", "b") in s
        # a list probe is normalized to a tuple before lookup
        assert ["a", "b"] in s
        assert ("a", "c") not in s

    def test_repr(self):
        s = Set("p", [1, 2, 3])
        assert "p" in repr(s) and "dimen=1" in repr(s)


class TestRangeSet:
    def test_single_arg_one_based(self):
        assert list(RangeSet(3)) == [1, 2, 3]

    def test_two_arg_inclusive(self):
        assert list(RangeSet(2, 5)) == [2, 3, 4, 5]

    def test_dimen_one(self):
        assert RangeSet(4).dimen == 1

    def test_is_a_set(self):
        assert isinstance(RangeSet(3), Set)
        assert 2 in RangeSet(3)


class TestSetAlgebra:
    def test_union(self):
        a = Set("a", [1, 2, 3])
        b = Set("b", [3, 4, 5])
        u = a | b
        assert list(u) == [1, 2, 3, 4, 5]

    def test_union_preserves_order_left_first(self):
        a = Set("a", ["x", "y"])
        b = Set("b", ["y", "z"])
        assert list(a | b) == ["x", "y", "z"]

    def test_intersection(self):
        a = Set("a", [1, 2, 3, 4])
        b = Set("b", [2, 4, 6])
        assert list(a & b) == [2, 4]

    def test_intersection_commutative_membership(self):
        a = Set("a", [1, 2, 3])
        b = Set("b", [2, 3, 4])
        assert set(a & b) == set(b & a)

    def test_difference(self):
        a = Set("a", [1, 2, 3, 4])
        b = Set("b", [2, 4])
        assert list(a - b) == [1, 3]

    def test_difference_law(self):
        # (A | B) - B is a subset of A
        a = Set("a", [1, 2, 3])
        b = Set("b", [3, 4, 5])
        assert set((a | b) - b) <= set(a)

    def test_algebra_dimen_mismatch_raises(self):
        a = Set("a", [1, 2])
        b = Set("b", [("x", "y")])
        with pytest.raises(ValueError, match="dimensionality"):
            _ = a | b
        with pytest.raises(ValueError, match="dimensionality"):
            _ = a & b
        with pytest.raises(ValueError, match="dimensionality"):
            _ = a - b


class TestProductSet:
    def test_product_len_and_dimen(self):
        a = Set("a", ["p", "q"])
        b = Set("b", [1, 2, 3])
        p = a * b
        assert isinstance(p, ProductSet)
        assert len(p) == 6
        assert p.dimen == 2

    def test_product_members(self):
        a = Set("a", ["p", "q"])
        b = Set("b", [1, 2])
        assert list(a * b) == [("p", 1), ("p", 2), ("q", 1), ("q", 2)]

    def test_product_membership(self):
        p = Set("a", ["p", "q"]) * Set("b", [1, 2])
        assert ("p", 2) in p
        assert ("z", 2) not in p
        assert ("p", 2, 3) not in p

    def test_triple_product_flattens(self):
        a = Set("a", ["x"])
        b = Set("b", ["y"])
        c = Set("c", ["z"])
        p = a * b * c
        assert p.dimen == 3
        assert list(p) == [("x", "y", "z")]
        # chaining does not nest factors
        assert len(p.factors) == 3

    def test_product_of_products_membership(self):
        p = (Set("a", [1, 2]) * Set("b", [3])) * Set("c", [4, 5])
        assert (1, 3, 4) in p
        assert len(p) == 4

    def test_cardinality_law(self):
        a = Set("a", range(3))
        b = Set("b", range(4))
        assert len(a * b) == len(a) * len(b)

    def test_ordinal_row_major(self):
        a = Set("a", ["p", "q"])
        b = Set("b", [1, 2, 3])
        p = a * b
        # row-major: ordinal matches position in iteration order
        for i, m in enumerate(p):
            assert p.ordinal(m) == i

    def test_ordinal_bad_member_raises(self):
        p = Set("a", ["p", "q"]) * Set("b", [1, 2])
        with pytest.raises(KeyError, match="not a member"):
            p.ordinal(("z", 1))

    def test_ordinal_nested_product(self):
        p = (Set("a", [1, 2]) * Set("b", [3, 4])) * Set("c", [5, 6])
        for i, m in enumerate(p):
            assert p.ordinal(m) == i

    def test_ordinal_accepts_nested_and_flat_keys(self):
        # arcs (dimen 2) crossed with commodities (dimen 1) -> dimen 3
        arcs = Set("arcs", [("s", "u"), ("u", "t")])
        comm = Set("comm", ["c1", "c2"])
        prod = arcs * comm
        # flat key, nested key, and fully nested key all resolve identically
        assert prod.ordinal(("s", "u", "c1")) == prod.ordinal((("s", "u"), "c1"))
        assert prod.ordinal(("u", "t", "c2")) == 3
        assert (("s", "u"), "c1") in prod
        assert ("s", "u", "c1") in prod


class TestFilter:
    def test_where_dimen1(self):
        s = Set("n", range(1, 6))
        evens = s.where(lambda i: i % 2 == 0)
        assert list(evens) == [2, 4]

    def test_where_dimen2_unpacks(self):
        p = Set("a", [1, 2, 3]) * Set("b", [1, 2, 3])
        upper = p.where(lambda i, j: i < j)
        assert list(upper) == [(1, 2), (1, 3), (2, 3)]

    def test_where_on_product_avoids_full_materialization(self):
        # Filtering a product yields a concrete Set of only matching members.
        links = (Set("p", ["pitt", "sf", "nyc"]) * Set("m", ["a", "b"])).where(
            lambda p, k: (p, k) != ("nyc", "a")
        )
        assert isinstance(links, Set)
        assert len(links) == 5
        assert ("nyc", "a") not in links

    def test_where_idempotent(self):
        s = Set("n", range(10))
        pred = lambda i: i % 2 == 0  # noqa: E731
        once = s.where(pred)
        twice = once.where(pred)
        assert list(once) == list(twice)

    def test_with_first(self):
        links = Set("l", [("p", "a"), ("p", "b"), ("q", "a")])
        assert list(links.with_first("p")) == [("p", "a"), ("p", "b")]

    def test_with_last(self):
        links = Set("l", [("p", "a"), ("q", "a"), ("q", "b")])
        assert list(links.with_last("a")) == [("p", "a"), ("q", "a")]

    def test_with_first_requires_dimen2(self):
        s = Set("n", [1, 2, 3])
        with pytest.raises(ValueError, match="dimen >= 2"):
            s.with_first(1)


class TestModelSet:
    def test_register_and_return(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        assert isinstance(plants, Set)
        assert plants in m._sets

    def test_duplicate_name_raises(self):
        m = dm.Model()
        m.set("plants", ["pitt"])
        with pytest.raises(ValueError, match="already used"):
            m.set("plants", ["sf"])

    def test_set_usable_in_algebra(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        markets = m.set("markets", ["a", "b"])
        assert len(plants * markets) == 6

    def test_top_level_exports(self):
        import discopt

        assert discopt.Set is Set
        assert discopt.RangeSet is RangeSet
