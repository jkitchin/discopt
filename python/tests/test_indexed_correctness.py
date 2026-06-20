"""End-to-end correctness for set/index models (Phase 7 M6).

Solves set-based models and checks them against known optima, against the
equivalent positional (``shape=``) model, and through a ``.nl`` round-trip.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.pr_correctness

# ── Assignment problem (binary, equality constraints) ──

_ACOST = {
    ("w1", "a"): 9,
    ("w1", "b"): 2,
    ("w1", "c"): 7,
    ("w2", "a"): 6,
    ("w2", "b"): 4,
    ("w2", "c"): 3,
    ("w3", "a"): 5,
    ("w3", "b"): 8,
    ("w3", "c"): 1,
}
_WORKERS = ["w1", "w2", "w3"]
_TASKS = ["a", "b", "c"]


def _assignment(fast: bool):
    m = dm.Model("assign")
    w = m.set("w", _WORKERS)
    t = m.set("t", _TASKS)
    assign = m.binary("assign", over=w * t)
    m.minimize(dm.sum(_ACOST[(i, j)] * assign[i, j] for i in w for j in t))
    m.constraint(w, lambda i: dm.sum(assign[i, j] for j in t) == 1, name="one_task", fast=fast)
    m.constraint(t, lambda j: dm.sum(assign[i, j] for i in w) == 1, name="one_worker", fast=fast)
    return m, assign


def _brute_force_assignment() -> float:
    return min(
        sum(_ACOST[(w, t)] for w, t in zip(_WORKERS, perm))
        for perm in itertools.permutations(_TASKS)
    )


class TestAssignment:
    def test_matches_brute_force_optimum(self):
        m, _ = _assignment(fast=True)
        r = m.solve()
        assert r.status == "optimal"
        assert r.objective == pytest.approx(_brute_force_assignment(), abs=1e-6)

    def test_solution_is_a_permutation(self):
        m, assign = _assignment(fast=True)
        r = m.solve()
        vals = assign.value(r)
        # each worker assigned exactly one task and vice versa
        for w in _WORKERS:
            assert sum(round(vals[(w, t)]) for t in _TASKS) == 1
        for t in _TASKS:
            assert sum(round(vals[(w, t)]) for w in _WORKERS) == 1

    def test_fast_and_slow_agree(self):
        rf = _assignment(fast=True)[0].solve()
        rs = _assignment(fast=False)[0].solve()
        assert rf.objective == pytest.approx(rs.objective, abs=1e-6)


# ── Transportation (continuous) vs equivalent positional model ──

_SUPPLY = [20.0, 30.0]
_DEMAND = [10.0, 25.0, 15.0]
_TCOST = np.array([[4.0, 6.0, 8.0], [5.0, 3.0, 7.0]])


def _transport_sets():
    m = dm.Model("t_sets")
    P = m.set("P", ["p0", "p1"])
    K = m.set("K", ["k0", "k1", "k2"])
    sup = dict(zip(P, _SUPPLY))
    dem = dict(zip(K, _DEMAND))
    cost = {(p, k): _TCOST[i, j] for i, p in enumerate(P) for j, k in enumerate(K)}
    ship = m.continuous("ship", over=P * K, lb=0, ub=1000)
    m.minimize(dm.sum(cost[(p, k)] * ship[p, k] for p in P for k in K))
    m.constraint(P, lambda p: dm.sum(ship[p, k] for k in K) <= sup[p], name="sup")
    m.constraint(K, lambda k: dm.sum(ship[p, k] for p in P) >= dem[k], name="dem")
    return m


def _transport_positional():
    m = dm.Model("t_pos")
    ship = m.continuous("ship", shape=(2, 3), lb=0, ub=1000)
    m.minimize(dm.sum(_TCOST[i, j] * ship[i, j] for i in range(2) for j in range(3)))
    for i in range(2):
        m.subject_to(dm.sum(ship[i, j] for j in range(3)) <= _SUPPLY[i], name=f"sup_{i}")
    for j in range(3):
        m.subject_to(dm.sum(ship[i, j] for i in range(2)) >= _DEMAND[j], name=f"dem_{j}")
    return m


class TestTransportationEquivalence:
    def test_sets_match_positional_objective(self):
        r_sets = _transport_sets().solve()
        r_pos = _transport_positional().solve()
        assert r_sets.status == "optimal"
        assert r_pos.status == "optimal"
        assert r_sets.objective == pytest.approx(r_pos.objective, rel=1e-6)


# ── .nl round-trip ──


class TestNlRoundTrip:
    @pytest.mark.parametrize("fast", [False, True])
    def test_roundtrip_preserves_optimum(self, tmp_path, fast):
        # .nl export recovers both general-path and fast-path (builder) rows.
        m, _ = _assignment(fast=fast)
        r0 = m.solve()
        path = tmp_path / "assign.nl"
        m.to_nl(str(path))
        m2 = dm.from_nl(str(path))
        r1 = m2.solve()
        assert r1.status == "optimal"
        assert r1.objective == pytest.approx(r0.objective, abs=1e-6)


# ── Pyomo cross-check (skipped if pyomo is unavailable) ──


class TestPyomoCrossCheck:
    def test_transportation_matches_pyomo(self):
        pyo = pytest.importorskip("pyomo.environ")
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0, 1)
        model.K = pyo.RangeSet(0, 2)
        model.ship = pyo.Var(model.P, model.K, domain=pyo.NonNegativeReals)
        model.obj = pyo.Objective(
            expr=sum(_TCOST[i, j] * model.ship[i, j] for i in model.P for j in model.K)
        )
        model.sup = pyo.Constraint(
            model.P, rule=lambda mm, i: sum(mm.ship[i, j] for j in mm.K) <= _SUPPLY[i]
        )
        model.dem = pyo.Constraint(
            model.K, rule=lambda mm, j: sum(mm.ship[i, j] for i in mm.P) >= _DEMAND[j]
        )
        dmodel = dm.from_pyomo(model)
        r = dmodel.solve()
        assert r.status == "optimal"
        assert r.objective == pytest.approx(_transport_sets().solve().objective, rel=1e-6)
