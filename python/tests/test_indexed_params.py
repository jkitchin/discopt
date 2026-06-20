"""Tests for indexed parameters and aggregation over named sets (Phase 7 M4)."""

import discopt.modeling as dm
from discopt.modeling.core import IndexExpression, Parameter, SumOverExpression
from discopt.modeling.indexed import IndexedParam


class TestIndexedParam:
    def test_dict_value(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        cap = m.parameter("cap", over=plants, value={"pitt": 10, "sf": 20, "nyc": 30})
        assert isinstance(cap, IndexedParam)
        assert isinstance(cap.flat, Parameter)
        assert list(cap.flat.value) == [10, 20, 30]

    def test_scalar_value_broadcast(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        p = m.parameter("p", over=s, value=5.0)
        assert list(p.flat.value) == [5, 5, 5]

    def test_callable_value(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        p = m.parameter("p", over=s, value=lambda i: i**2)
        assert list(p.flat.value) == [1, 4, 9]

    def test_getitem_is_expression(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        cap = m.parameter("cap", over=plants, value={"pitt": 10, "sf": 20})
        assert isinstance(cap["pitt"], IndexExpression)
        assert cap["pitt"].index == 0

    def test_at_returns_numeric(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        cap = m.parameter("cap", over=plants, value={"pitt": 10, "sf": 20})
        assert cap.at("sf") == 20.0

    def test_plain_parameter_still_works(self):
        m = dm.Model()
        p = m.parameter("price", value=50.0)
        assert isinstance(p, Parameter)
        assert not isinstance(p, IndexedParam)


class TestAggregationOverSets:
    def test_sum_generator_over_set(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        x = m.continuous("x", over=plants, lb=0)
        expr = dm.sum(x[p] for p in plants)
        assert isinstance(expr, SumOverExpression)

    def test_sum_callable_over_dimen1_set(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf", "nyc"])
        cap = m.parameter("cap", over=plants, value={"pitt": 1, "sf": 2, "nyc": 3})
        x = m.continuous("x", over=plants, lb=0)
        expr = dm.sum(lambda p: cap[p] * x[p], over=plants)
        assert isinstance(expr, SumOverExpression)

    def test_sum_callable_over_dimen2_unpacks(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        markets = m.set("markets", ["a", "b"])
        links = plants * markets
        ship = m.continuous("ship", over=links, lb=0)
        # rule takes two args because links has dimen 2
        expr = dm.sum(lambda p, k: ship[p, k], over=links)
        assert isinstance(expr, SumOverExpression)

    def test_prod_over_set(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, lb=1, ub=2)
        expr = dm.prod(lambda i: x[i], over=s)
        # product of three terms desugars to nested BinaryOp, not a single var
        assert expr is not None

    def test_prod_generator(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, lb=1, ub=2)
        expr = dm.prod(x[i] for i in s)
        assert expr is not None

    def test_norm_on_backing_flat(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, lb=0)
        # norm operates on the flat vector variable
        expr = dm.norm(x.flat, ord=2)
        assert expr is not None

    def test_double_sum_transportation_objective(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        markets = m.set("markets", ["a", "b", "c"])
        cost = m.parameter(
            "cost",
            over=plants * markets,
            value=lambda p, k: 1.0,
        )
        ship = m.continuous("ship", over=plants * markets, lb=0)
        obj = dm.sum(cost[p, k] * ship[p, k] for p in plants for k in markets)
        m.minimize(obj)
        assert m._objective is not None


class TestExports:
    def test_indexedparam_exported(self):
        assert dm.IndexedParam is IndexedParam
