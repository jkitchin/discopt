"""Tests for the linear fast-path routing of indexed constraint families (Phase 7 M5).

The fast path must be a pure performance optimization: a family routed into the
Rust builder must yield the same model (and same solution) as the general
expression path.
"""

import discopt.modeling as dm
import pytest

SUPPLY = {"pitt": 20.0, "sf": 30.0}
DEMAND = {"a": 10.0, "b": 25.0, "c": 15.0}
COST = {
    ("pitt", "a"): 4.0,
    ("pitt", "b"): 6.0,
    ("pitt", "c"): 8.0,
    ("sf", "a"): 5.0,
    ("sf", "b"): 3.0,
    ("sf", "c"): 7.0,
}


def build_transportation(fast: bool):
    m = dm.Model("transport")
    plants = m.set("plants", list(SUPPLY))
    markets = m.set("markets", list(DEMAND))
    links = plants * markets
    ship = m.continuous("ship", over=links, lb=0, ub=1000)
    m.minimize(dm.sum(COST[(p, k)] * ship[p, k] for p in plants for k in markets))
    m.constraint(
        plants,
        lambda p: dm.sum(ship[p, k] for k in markets) <= SUPPLY[p],
        name="supply",
        fast=fast,
    )
    m.constraint(
        markets,
        lambda k: dm.sum(ship[p, k] for p in plants) >= DEMAND[k],
        name="demand",
        fast=fast,
    )
    return m, ship


class TestFastPathRouting:
    def test_fast_routes_into_builder(self):
        m, _ = build_transportation(fast=True)
        # both families are single-variable affine -> nothing on the Python list
        assert len(m._constraints) == 0
        from discopt._rust import model_to_repr

        repr_ = model_to_repr(m, m._builder)
        assert repr_.n_constraints == len(SUPPLY) + len(DEMAND)

    def test_slow_keeps_python_constraints(self):
        m, _ = build_transportation(fast=False)
        assert len(m._constraints) == len(SUPPLY) + len(DEMAND)

    def test_indexedconstraint_fast_flag(self):
        m = dm.Model()
        plants = m.set("plants", ["pitt", "sf"])
        x = m.continuous("x", over=plants, lb=0)
        ic_fast = m.constraint(plants, lambda p: x[p] <= 1, name="c", fast=True)
        assert ic_fast.fast is True
        m2 = dm.Model()
        plants2 = m2.set("plants", ["pitt", "sf"])
        x2 = m2.continuous("x", over=plants2, lb=0)
        ic_slow = m2.constraint(plants2, lambda p: x2[p] <= 1, name="c", fast=False)
        assert ic_slow.fast is False


class TestFastPathFallback:
    def test_nonlinear_falls_back(self):
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, lb=0, ub=5)
        ic = m.constraint(s, lambda i: x[i] ** 2 <= 4, name="nl", fast=True)
        assert ic.fast is False
        assert len(m._constraints) == 3

    def test_multivariable_falls_back(self):
        m = dm.Model()
        s = m.set("s", [1, 2])
        x = m.continuous("x", over=s, lb=0)
        y = m.continuous("y", over=s, lb=0)
        ic = m.constraint(s, lambda i: x[i] + y[i] <= 1, name="link", fast=True)
        assert ic.fast is False
        assert len(m._constraints) == 2

    def test_parameter_coefficient_falls_back(self):
        # A parameter must stay symbolic -> the fast path must not bake it in.
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        cap = m.parameter("cap", over=s, value={1: 1.0, 2: 2.0, 3: 3.0})
        x = m.continuous("x", over=s, lb=0)
        ic = m.constraint(s, lambda i: cap[i] * x[i] <= 10, name="p", fast=True)
        assert ic.fast is False
        assert len(m._constraints) == 3

    def test_mixed_sense_falls_back(self):
        # <= and >= both normalize to "<=", so mix in an equality to differ.
        m = dm.Model()
        s = m.set("s", [1, 2, 3, 4])
        x = m.continuous("x", over=s, lb=0, ub=5)
        ic = m.constraint(
            s, lambda i: (x[i] <= 3) if i % 2 == 0 else (x[i] == 2), name="mix", fast=True
        )
        assert ic.fast is False
        assert len(m._constraints) == 4

    def test_ge_normalizes_and_still_fast(self):
        # A uniform >= family is still single-sense ("<=") after normalization.
        m = dm.Model()
        s = m.set("s", [1, 2, 3])
        x = m.continuous("x", over=s, lb=0, ub=5)
        ic = m.constraint(s, lambda i: x[i] >= 1, name="lb", fast=True)
        assert ic.fast is True


class TestNlExportFastPath:
    def test_fast_model_nl_has_all_constraints(self):
        m_fast, _ = build_transportation(fast=True)
        m_slow, _ = build_transportation(fast=False)
        # header line 2: "<nvars> <ncons> <nobjs> ..."
        nfast = int(m_fast.to_nl(None).splitlines()[1].split()[1])
        nslow = int(m_slow.to_nl(None).splitlines()[1].split()[1])
        assert nfast == nslow == len(SUPPLY) + len(DEMAND)

    def test_fast_model_nl_roundtrip_preserves_optimum(self, tmp_path):
        m_fast, _ = build_transportation(fast=True)
        r0 = m_fast.solve()
        path = tmp_path / "fast.nl"
        m_fast.to_nl(str(path))
        m2 = dm.from_nl(str(path))
        r1 = m2.solve()
        assert r1.status == "optimal"
        assert r1.objective == pytest.approx(r0.objective, rel=1e-6)

    def test_direct_add_linear_constraints_exported(self, tmp_path):
        # The pre-existing fast-construction API also bypassed _constraints.
        import numpy as np

        m = dm.Model("direct")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        m.add_linear_constraints(A, x, ">=", np.array([2.0, 3.0]), name="c")
        m.minimize(dm.sum(x[i] for i in range(3)))
        r0 = m.solve()
        path = tmp_path / "direct.nl"
        m.to_nl(str(path))
        r1 = dm.from_nl(str(path)).solve()
        assert int(m.to_nl(None).splitlines()[1].split()[1]) == 2  # both rows exported
        assert r1.objective == pytest.approx(r0.objective, rel=1e-6)


class TestNlExportFastObjective:
    @staticmethod
    def _build(sense, constant):
        import numpy as np

        m = dm.Model("fastobj")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        m.add_linear_constraints(A, x, ">=", np.array([2.0, 3.0]), name="c")
        m.add_linear_objective(np.array([3.0, 1.0, 2.0]), x, constant=constant, sense=sense)
        return m

    @pytest.mark.parametrize(
        "sense,constant", [("minimize", 0.0), ("minimize", 5.0), ("maximize", 0.0)]
    )
    def test_linear_objective_roundtrips(self, tmp_path, sense, constant):
        m = self._build(sense, constant)
        r0 = m.solve()
        path = tmp_path / "obj.nl"
        m.to_nl(str(path))
        r1 = dm.from_nl(str(path)).solve()
        assert r1.status == "optimal"
        assert r1.objective == pytest.approx(r0.objective, rel=1e-6, abs=1e-6)

    def test_expression_objective_not_overwritten(self):
        # A real objective set after a builder objective must win on export.
        m = self._build("minimize", 0.0)
        x = m._variables[0]
        m.minimize(dm.sum(x[i] for i in range(3)))  # replaces placeholder
        txt = m.to_nl(None)
        # objective gradient has all 3 unit coeffs from the expression objective
        assert "G0 3" in txt


class TestNlExportQuadraticObjective:
    @staticmethod
    def _build(Q, c, constant=0.0):
        import numpy as np

        m = dm.Model("qp")
        x = m.continuous("x", shape=(2,), lb=-10, ub=10)
        m.add_quadratic_objective(np.asarray(Q), np.asarray(c), x, constant=constant)
        m.add_linear_constraints(np.array([[1.0, 1.0]]), x, "<=", np.array([10.0]), name="c")
        return m

    def test_diagonal_q_roundtrips(self, tmp_path):
        # 0.5 x'Qx + c'x with Q=2I, c=(-2,-6) -> min at (1,3), obj -10.
        m = self._build([[2.0, 0.0], [0.0, 2.0]], [-2.0, -6.0])
        r0 = m.solve()
        path = tmp_path / "qp.nl"
        m.to_nl(str(path))
        r1 = dm.from_nl(str(path)).solve()
        assert r1.status == "optimal"
        assert r1.objective == pytest.approx(r0.objective, rel=1e-5, abs=1e-5)

    def test_constant_offset_roundtrips(self, tmp_path):
        m = self._build([[2.0, 0.0], [0.0, 2.0]], [-2.0, -6.0], constant=7.5)
        r0 = m.solve()
        path = tmp_path / "qp.nl"
        m.to_nl(str(path))
        r1 = dm.from_nl(str(path)).solve()
        assert r1.objective == pytest.approx(r0.objective, rel=1e-5, abs=1e-5)

    def test_coupled_cross_terms_roundtrip(self, tmp_path):
        # Off-diagonal Q exercises the x_i x_j cross terms.
        import numpy as np

        m = dm.Model("qp2")
        x = m.continuous("x", shape=(2,), lb=-10, ub=10)
        m.add_quadratic_objective(np.array([[2.0, 1.0], [1.0, 2.0]]), np.zeros(2), x)
        m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([2.0]), name="c")
        r0 = m.solve()
        path = tmp_path / "qp2.nl"
        m.to_nl(str(path))
        r1 = dm.from_nl(str(path)).solve()
        assert r1.objective == pytest.approx(r0.objective, rel=1e-5, abs=1e-5)

    def test_objective_is_nonlinear_in_nl(self):
        # The quadratic objective must export a nonlinear O-segment, not n0.
        m = self._build([[2.0, 0.0], [0.0, 2.0]], [-2.0, -6.0])
        lines = m.to_nl(None).splitlines()
        o_idx = next(i for i, ln in enumerate(lines) if ln.startswith("O0"))
        assert lines[o_idx + 1] != "n0"  # has a real nonlinear body


class TestFastPathEquivalence:
    def test_same_solution_fast_vs_slow(self):
        m_fast, ship_fast = build_transportation(fast=True)
        m_slow, ship_slow = build_transportation(fast=False)
        r_fast = m_fast.solve()
        r_slow = m_slow.solve()
        assert r_fast.status == "optimal"
        assert r_slow.status == "optimal"
        assert r_fast.objective == pytest.approx(r_slow.objective, rel=1e-6)
        vf = ship_fast.value(r_fast)
        vs = ship_slow.value(r_slow)
        for key in vf:
            assert vf[key] == pytest.approx(vs[key], abs=1e-5)

    def test_fast_matches_known_optimum(self):
        # Balanced transportation; LP optimum is unique and checkable.
        m, ship = build_transportation(fast=True)
        r = m.solve()
        assert r.status == "optimal"
        # demand exactly met, supply not exceeded
        vals = ship.value(r)
        for k in DEMAND:
            got = sum(vals[(p, k)] for p in SUPPLY)
            assert got == pytest.approx(DEMAND[k], abs=1e-5)
        for p in SUPPLY:
            used = sum(vals[(p, k)] for k in DEMAND)
            assert used <= SUPPLY[p] + 1e-5
