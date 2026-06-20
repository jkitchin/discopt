"""Tests for indexed variables over named sets (Phase 7 M2).

Covers ``Model.continuous/binary/integer(..., over=SET)``, the backing flat
variable, per-key bounds, key indexing, and desugaring equivalence to the
positional ``shape=`` API.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import IndexExpression, VarType
from discopt.modeling.indexed import IndexedVar


def _flat_signature(model):
    """A structural fingerprint of a model's flat variables for equivalence checks."""
    return [
        (v.name, v.var_type, v.shape, tuple(np.ravel(v.lb)), tuple(np.ravel(v.ub)))
        for v in model._variables
    ]


class TestIndexedVarBasics:
    def test_continuous_over_set_backed_by_flat(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        x = m.continuous("x", over=plants, lb=0)
        assert isinstance(x, IndexedVar)
        assert x.flat.shape == (3,)
        assert x.flat.var_type == VarType.CONTINUOUS
        assert len(m._variables) == 1  # one flat var, not three

    def test_getitem_returns_index_expression(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        x = m.continuous("x", over=plants)
        expr = x["sf"]
        assert isinstance(expr, IndexExpression)
        assert expr.base is x.flat
        assert expr.index == 1

    def test_tuple_key_indexing(self):
        m = dm.Model()
        links = m.set("links", [("pitt", "a"), ("sf", "b")])
        x = m.continuous("x", over=links, lb=0)
        assert x["pitt", "a"].index == 0
        assert x[("sf", "b")].index == 1

    def test_bad_key_raises(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        x = m.continuous("x", over=plants)
        with pytest.raises(KeyError, match="not a member"):
            _ = x["nyc"]

    def test_binary_over_product(self):
        m = dm.Model()
        workers = m.set("w", ["w1", "w2"])
        tasks = m.set("t", ["t1", "t2", "t3"])
        y = m.binary("assign", over=workers * tasks)
        assert y.flat.shape == (6,)
        assert y.flat.var_type == VarType.BINARY
        assert np.all(y.flat.lb == 0) and np.all(y.flat.ub == 1)

    def test_integer_over_set(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        n = m.integer("n", over=plants, lb=0, ub=10)
        assert n.flat.var_type == VarType.INTEGER
        assert np.all(n.flat.ub == 10)

    def test_len_iter_contains_keys(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        x = m.continuous("x", over=plants)
        assert len(x) == 3
        assert list(x) == ["pitt", "sf", "nyc"]
        assert "sf" in x and "denver" not in x
        assert x.keys() == ("pitt", "sf", "nyc")


class TestIndexedBounds:
    def test_scalar_bounds_broadcast(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, lb=0, ub=100)
        assert np.all(x.flat.lb == 0) and np.all(x.flat.ub == 100)

    def test_dict_bounds(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        cap = {"pitt": 10, "sf": 20, "nyc": 30}
        x = m.continuous("x", over=plants, lb=0, ub=cap)
        assert list(x.flat.ub) == [10, 20, 30]

    def test_dict_bounds_missing_key_raises(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        with pytest.raises(KeyError, match="no entry for member"):
            m.continuous("x", over=plants, ub={"pitt": 10})

    def test_callable_bounds_dimen1(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, ub=lambda i: 10 * i)
        assert list(x.flat.ub) == [10, 20, 30]

    def test_callable_bounds_dimen2_unpacks(self):
        m = dm.Model()
        links = m.set("links", [("a", 1), ("b", 2)])
        x = m.continuous("x", over=links, ub=lambda node, k: k * 100)
        assert list(x.flat.ub) == [100, 200]

    def test_over_and_shape_mutually_exclusive(self):
        m = dm.Model()
        s = m.set("s", [1, 2])
        with pytest.raises(ValueError, match="mutually"):
            m.continuous("x", shape=(2,), over=s)


class TestValueRetrieval:
    def test_value_maps_keys(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        x = m.continuous("x", over=plants, lb=0, ub=10)
        m.minimize(dm.sum(x.flat))

        class _FakeResult:
            def value(self, var):
                return np.array([1.0, 2.0, 3.0])

        vals = x.value(_FakeResult())
        assert vals == {"pitt": 1.0, "sf": 2.0, "nyc": 3.0}


class TestDesugarEquivalence:
    """An indexed model must produce the same flat model as the positional one."""

    def test_equivalent_to_shape_variable(self):
        # Indexed over an integer range vs. a dense shape=(n,) variable.
        m_idx = dm.Model()
        s = m_idx.set("s", [0, 1, 2, 3])
        m_idx.continuous("x", over=s, lb=0, ub=5)

        m_pos = dm.Model()
        m_pos.continuous("x", shape=(4,), lb=0, ub=5)

        assert _flat_signature(m_idx) == _flat_signature(m_pos)

    def test_dict_bounds_match_explicit_array(self):
        m_idx = dm.Model()
        plants = m_idx.set("plants", ["a", "b", "c"])
        m_idx.continuous("x", over=plants, lb=0, ub={"a": 1, "b": 2, "c": 3})

        m_pos = dm.Model()
        m_pos.continuous("x", shape=(3,), lb=0, ub=np.array([1.0, 2.0, 3.0]))

        assert _flat_signature(m_idx) == _flat_signature(m_pos)

    def test_index_expression_matches_positional(self):
        m_idx = dm.Model()
        s = m_idx.set("s", ["a", "b", "c"])
        x = m_idx.continuous("x", over=s)
        e = x["b"]

        m_pos = dm.Model()
        xp = m_pos.continuous("x", shape=(3,))
        ep = xp[1]

        assert e.index == ep.index
        assert e.base.name == ep.base.name


class TestExports:
    def test_indexedvar_exported(self):
        assert dm.IndexedVar is IndexedVar
