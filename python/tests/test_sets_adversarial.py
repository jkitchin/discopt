"""Adversarial tests for the set/index abstractions (Phase 7).

A second, independent pass aimed at corners the feature-level suites do not
cover: integer labels vs. positions, hash-colliding members, reflected
comparisons, all-or-nothing fast-path fallback, empty sets, set-algebra-defined
indices, mixed builder/expression export, and three-way solve equivalence.
"""

import discopt.modeling as dm
import numpy as np
import pytest


class TestLabelsVsPositions:
    def test_integer_labels_index_by_member_not_position(self):
        # x[10] must mean the member 10 (ordinal 1), never array position 10.
        m = dm.Model()
        s = m.set("s", [5, 10, 15])
        x = m.continuous("x", over=s, lb=0, ub=100)
        assert x[5].index == 0
        assert x[10].index == 1
        assert x[15].index == 2

    def test_integer_label_dict_bounds_map_by_member(self):
        m = dm.Model()
        s = m.set("s", [5, 10, 15])
        y = m.continuous("y", over=s, lb=0, ub={5: 1.0, 10: 2.0, 15: 3.0})
        m.maximize(dm.sum(y[k] for k in s))
        v = y.value(m.solve())
        assert v[5] == pytest.approx(1.0, abs=1e-5)
        assert v[10] == pytest.approx(2.0, abs=1e-5)
        assert v[15] == pytest.approx(3.0, abs=1e-5)

    def test_bool_int_hash_collision_dedup(self):
        # True == 1 and False == 0 in Python; the set must dedup them.
        s = dm.Model().set("b", [True, False, 1, 0])
        assert len(s) == 2

    def test_duplicate_members_first_occurrence_ordinal(self):
        s = dm.Model().set("s", ["b", "a", "b", "c"])
        assert s.ordinal("b") == 0
        assert list(s) == ["b", "a", "c"]


class TestDegenerateSets:
    def test_empty_set_variable_and_constraint(self):
        m = dm.Model()
        e = m.set("e", [])
        xe = m.continuous("xe", over=e, lb=0)
        assert xe.flat.shape == (0,)
        ic = m.constraint(e, lambda i: xe[i] <= 1, name="none")
        assert len(ic) == 0
        assert len(m._constraints) == 0

    def test_where_to_empty(self):
        s = dm.Model().set("s", [1, 2, 3])
        assert len(s.where(lambda i: i > 100)) == 0

    def test_empty_aggregation_solves(self):
        m = dm.Model()
        s = m.set("s", [1, 2])
        empty = s.where(lambda i: False)
        x = m.continuous("x", over=s, lb=0, ub=5)
        # empty sum is 0, so this constraint is trivially satisfied
        m.subject_to(dm.sum(x[i] for i in empty) <= 100, name="trivial")
        m.maximize(dm.sum(x[i] for i in s))
        r = m.solve()
        assert r.status == "optimal"
        assert r.objective == pytest.approx(10.0, abs=1e-5)

    def test_rangeset_zero_is_empty(self):
        assert len(dm.RangeSet(0)) == 0


class TestIndexingErrors:
    def test_wrong_arity_indexing_raises(self):
        m = dm.Model()
        links = m.set("l", [("a", 1), ("b", 2)])
        z = m.continuous("z", over=links, lb=0)
        with pytest.raises(KeyError):
            _ = z["a"]  # dimen-2 set needs a 2-key

    def test_rule_returning_list_raises(self):
        m = dm.Model()
        s = m.set("s", [0, 1])
        x = m.continuous("x", over=s, lb=0)
        with pytest.raises(TypeError, match="expected a Constraint"):
            m.constraint(s, lambda i: [x[i] <= 1], name="bad")


class TestFastPathAdversarial:
    def test_partial_nonlinear_family_all_or_nothing(self):
        # One nonlinear member must force the whole family to the general path.
        m = dm.Model()
        s = m.set("s", [0, 1, 2])
        x = m.continuous("x", over=s, lb=0, ub=5)
        ic = m.constraint(
            s, lambda i: (x[i] ** 2 <= 4) if i == 0 else (x[i] <= 4), name="mix", fast=True
        )
        assert ic.fast is False
        assert len(m._constraints) == 3

    def test_reflected_comparison_routes_and_solves(self):
        # 5 <= x[i] uses the reflected operator; must still route + solve.
        m = dm.Model()
        s = m.set("s", [0, 1])
        x = m.continuous("x", over=s, lb=0, ub=10)
        ic = m.constraint(s, lambda i: 5 <= x[i], name="r", fast=True)
        assert ic.fast is True
        m.minimize(dm.sum(x[i] for i in s))
        v = x.value(m.solve())
        assert all(val == pytest.approx(5.0, abs=1e-4) for val in v.values())

    def test_coefficient_cancellation_fast_equals_slow(self):
        def build(fast):
            m = dm.Model()
            s = m.set("s", [0, 1])
            x = m.continuous("x", over=s, lb=0, ub=10)
            m.constraint(s, lambda i: 2 * x[i] - x[i] + 3 <= 5, name="r", fast=fast)
            m.maximize(dm.sum(x[i] for i in s))
            return m, x

        mf, xf = build(True)
        ms, xs = build(False)
        rf, rs = mf.solve(), ms.solve()
        assert rf.objective == pytest.approx(rs.objective, abs=1e-5)
        assert all(v == pytest.approx(2.0, abs=1e-4) for v in xf.value(rf).values())

    def test_multivariable_coupling_falls_back_but_correct(self):
        def build(fast):
            m = dm.Model()
            s = m.set("s", [0, 1])
            x = m.continuous("x", over=s, lb=0, ub=10)
            z = m.continuous("z", over=s, lb=0, ub=10)
            ic = m.constraint(s, lambda i: x[i] + z[i] <= 4, name="c", fast=fast)
            m.maximize(dm.sum(x[i] + z[i] for i in s))
            return m, ic

        mf, icf = build(True)
        ms, ics = build(False)
        assert icf.fast is False
        assert mf.solve().objective == pytest.approx(ms.solve().objective, abs=1e-5)

    def test_dimen3_product_constraint_routes(self):
        m = dm.Model()
        A = m.set("A", ["p", "q"])
        B = m.set("B", [1, 2])
        C = m.set("C", ["x", "y"])
        f = m.continuous("f", over=A * B * C, lb=0, ub=10)
        ic = m.constraint(A * B * C, lambda a, b, c: f[a, b, c] <= 5, name="cap", fast=True)
        assert ic.fast is True
        assert len(m._constraints) == 0


class TestExportAdversarial:
    def _roundtrip(self, m):
        import os
        import tempfile

        r0 = m.solve()
        with tempfile.NamedTemporaryFile(suffix=".nl", delete=False) as fh:
            path = fh.name
        m.to_nl(path)
        r1 = dm.from_nl(path).solve()
        os.unlink(path)
        return r0, r1

    def test_mixed_builder_and_expression_constraints_export(self):
        m = dm.Model()
        s = m.set("s", ["a", "b", "c"])
        x = m.continuous("x", over=s, lb=0, ub=10)
        m.constraint(s, lambda i: x[i] <= 8, name="ub", fast=True)  # builder
        m.subject_to(dm.sum(x[i] for i in s) >= 6, name="total")  # expression
        m.minimize(dm.sum(x[i] for i in s))
        ncons = int(m.to_nl(None).splitlines()[1].split()[1])
        assert ncons == 4  # 3 builder + 1 expression, no double count
        r0, r1 = self._roundtrip(m)
        assert r1.objective == pytest.approx(r0.objective, abs=1e-5)

    def test_builder_quadratic_objective_plus_builder_constraints(self):
        m = dm.Model()
        x = m.continuous("x", shape=(3,), lb=-5, ub=5)
        m.add_quadratic_objective(2 * np.eye(3), np.array([-2.0, -4.0, -6.0]), x, constant=1.0)
        m.add_linear_constraints(np.ones((1, 3)), x, "<=", np.array([10.0]), name="c")
        r0, r1 = self._roundtrip(m)
        assert r1.objective == pytest.approx(r0.objective, abs=1e-4)

    def test_to_nl_is_idempotent(self):
        m = dm.Model()
        s = m.set("s", ["a", "b"])
        x = m.continuous("x", over=s, lb=0, ub=5)
        m.constraint(s, lambda i: x[i] <= 3, name="c", fast=True)
        m.minimize(dm.sum(x[i] for i in s))
        n1 = int(m.to_nl(None).splitlines()[1].split()[1])
        n2 = int(m.to_nl(None).splitlines()[1].split()[1])
        assert n1 == n2 == 2


@pytest.mark.pr_correctness
class TestThreeWayEquivalence:
    """indexed-fast == indexed-slow == positional-dense, end to end."""

    C = [3.0, 2.0, 5.0, 4.0]
    W = [2.0, 1.0, 3.0, 2.0]
    CAP = 4.0

    def _indexed(self, fast):
        m = dm.Model()
        items = m.set("I", ["a", "b", "c", "d"])
        y = m.binary("y", over=items)
        val = dict(zip(items, self.C))
        wt = dict(zip(items, self.W))
        m.maximize(dm.sum(val[i] * y[i] for i in items))
        one = m.set("_cap", [0])  # singleton index for the single capacity row
        m.constraint(
            one, lambda _: dm.sum(wt[i] * y[i] for i in items) <= self.CAP, name="cap", fast=fast
        )
        return m

    def _positional(self):
        m = dm.Model()
        y = m.binary("y", shape=(4,))
        m.maximize(dm.sum(self.C[i] * y[i] for i in range(4)))
        m.subject_to(dm.sum(self.W[i] * y[i] for i in range(4)) <= self.CAP, name="cap")
        return m

    def test_all_three_agree(self):
        rf = self._indexed(fast=True).solve()
        rs = self._indexed(fast=False).solve()
        rp = self._positional().solve()
        assert rf.objective == pytest.approx(rs.objective, abs=1e-6)
        assert rf.objective == pytest.approx(rp.objective, abs=1e-6)


class TestSetAlgebraDefinedIndex:
    def test_union_indexed_model_solves(self):
        m = dm.Model()
        a = m.set("a", [1, 2, 3, 4])
        b = m.set("b", [3, 4, 5, 6])
        idx = a | b
        x = m.continuous("x", over=idx, lb=0, ub=1)
        m.maximize(dm.sum(x[i] for i in idx))
        r = m.solve()
        assert len(idx) == 6
        assert r.objective == pytest.approx(6.0, abs=1e-5)

    def test_callable_dimen2_bounds_solved(self):
        m = dm.Model()
        P = m.set("P", ["p1", "p2"])
        K = m.set("K", ["k1", "k2"])
        x = m.continuous("x", over=P * K, lb=0, ub=lambda p, k: 10 if p == "p1" else 1)
        m.maximize(dm.sum(x[p, k] for p in P for k in K))
        v = x.value(m.solve())
        assert v[("p1", "k1")] == pytest.approx(10.0, abs=1e-4)
        assert v[("p2", "k2")] == pytest.approx(1.0, abs=1e-4)
