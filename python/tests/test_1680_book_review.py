"""#1680: four observations from the optimization-book review.

1. A kinetic fit started at the ``k1 == k2`` pole ran to its time limit without
   saying why its gap stayed open (``time_limit`` is honoured; the diagnostic was
   missing because a bound *did* come back).
2. The HiGHS LP/MILP route densified a bulk-row model's constraint matrix (the
   standard-form extraction, the constraint-dual matcher, the convex-kernel probe).
3. ``add_linear_constraints`` duals were keyed only per row (``name_r``).
4. ``solve_lagrangian`` cross-solved every block at every iteration (#1634) and
   ran its recovery LP on the POUNCE IPM.
"""

from __future__ import annotations

import logging
import tracemalloc

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model

pytestmark = pytest.mark.unit


# ── 1. kinetic fit at the pole ────────────────────────────────────────────────


def _abc_fit():
    """A -> B -> C, first order: B(t) = k1/(k2-k1) (exp(-k1 t) - exp(-k2 t))."""
    t = np.linspace(0.5, 10, 12)
    k1t, k2t = 1.0, 0.4
    rng = np.random.default_rng(0)
    data = k1t / (k2t - k1t) * (np.exp(-k1t * t) - np.exp(-k2t * t))
    data = data + 0.01 * rng.standard_normal(t.size)
    m = dm.Model("abc")
    k1 = m.continuous("k1", lb=0.01, ub=5.0)
    k2 = m.continuous("k2", lb=0.01, ub=5.0)
    pred = [k1 / (k2 - k1) * (dm.exp(-k1 * ti) - dm.exp(-k2 * ti)) for ti in t]
    m.minimize(sum((p - b) ** 2 for p, b in zip(pred, data)))
    return m, k1, k2


def test_kinetic_fit_at_the_pole_honours_time_limit_and_says_why(caplog):
    m, k1, k2 = _abc_fit()
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = m.solve(time_limit=5, initial_solution={k1: 0.5, k2: 0.5})
    assert r.wall_time < 60  # honoured (measured 8.2 s at time_limit=8 under load)
    # Uncertified, with the valid trivial bound and the true incumbent.
    assert not r.gap_certified
    assert r.bound is not None and r.bound <= r.objective + 1e-9
    msgs = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "discopt.solver" and "gap did not close" in rec.getMessage()
    ]
    assert len(msgs) == 1, msgs
    assert "(k2 - k1)" in msgs[0]
    assert "pole inside the variable box" in msgs[0]
    # One clause per distinct term, not one per residual (twelve identical texts).
    assert msgs[0].count("`(k1 / (k2 - k1))`") == 1


def test_certified_solve_gets_no_pole_gap_warning(caplog):
    # A pole the box excludes from below never triggers the diagnostic.
    m = dm.Model("nopole")
    x = m.continuous("x", lb=1.0, ub=3.0)
    m.minimize(1 / x + x)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert not [rec for rec in caplog.records if "gap did not close" in rec.getMessage()]


# ── 2. the HiGHS route stays sparse ──────────────────────────────────────────


def _bulk_chain(n: int) -> tuple[Model, int]:
    rng = np.random.default_rng(0)
    m = Model("bulk")
    x = m.continuous("x", shape=(n,), lb=0, ub=100)
    rows = np.repeat(np.arange(n - 1), 2)
    cols = np.stack([np.arange(n - 1), np.arange(1, n)], 1).ravel()
    A = sp.csr_matrix((np.ones(2 * (n - 1)), (rows, cols)), shape=(n - 1, n))
    m.add_linear_constraints(A, x, ">=", rng.uniform(1, 5, n - 1), name="demand")
    m.add_linear_objective(rng.uniform(1, 2, n), x)
    return m, n - 1


def test_bulk_row_lp_peak_memory_is_not_quadratic():
    # 3000 rows x 6000 standard-form columns: one dense float64 copy is 144 MB.
    # Before #1680 the peak traced here was ~155 MB already at 2000 rows (dense
    # A_eq from the extractor plus dense copies in the dual matcher).
    m, n_rows = _bulk_chain(3000)
    tracemalloc.start()
    try:
        r = m.solve()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert r.status == "optimal"
    assert peak < 100e6, f"peak {peak / 1e6:.0f} MB"
    assert r.constraint_duals is not None
    assert r.constraint_duals["demand"].shape == (n_rows,)


# ── 3. duals keyed by block name ─────────────────────────────────────────────


def _blocks_model(dup: bool = False, collide: bool = False):
    m = Model("t")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    A = sp.csr_matrix(np.array([[1.0, 1, 0], [0, 1, 1]]))
    m.add_linear_constraints(A, x, ">=", np.array([2.0, 3.0]), name="demand")
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1, 1]])), x, "<=", 8, name="cap")
    if dup:
        m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 0, 0]])), x, "<=", 9, name="cap")
    if collide:
        m.subject_to(x[0] <= 9.5, name="demand")
    m.add_linear_objective(np.array([1.0, 2.0, 1.0]), x)
    return m


def test_block_duals_keyed_by_block_name():
    r = _blocks_model().solve()
    assert r.status == "optimal"
    cd = r.constraint_duals
    d = cd["demand"]
    assert d.shape == (2,)
    # Row order, and the per-row keys are kept (backward compatible).
    assert d[0] == pytest.approx(float(cd["demand_0"]))
    assert d[1] == pytest.approx(float(cd["demand_1"]))
    assert cd["cap"].shape == (1,)
    assert np.abs(d).sum() > 0  # binding demand rows carry a nonzero dual


def test_block_duals_fail_closed_on_ambiguous_names():
    cd = _blocks_model(dup=True).solve().constraint_duals
    assert "cap" not in cd  # two blocks named "cap": no aggregate guessed
    assert cd["demand"].shape == (2,)
    cd = _blocks_model(collide=True).solve().constraint_duals
    # A Python constraint named "demand" keeps its own (scalar) dual.
    assert np.size(cd["demand"]) == 1


# ── convex-kernel probe: refuses builder/vector rows instead of IndexError ──


def test_convex_kernel_refuses_rows_that_do_not_map_onto_constraints():
    from discopt.solvers._convex_kernel import build_convex_spec

    m = Model("v")
    x = m.continuous("x", shape=(2,), lb=-5, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x <= np.array([1.0, 2.0]), name="vec")  # one constraint, two rows
    m.subject_to(x[0] + x[1] >= -3, name="ge")
    m.subject_to(dm.exp(x[0]) <= y + 1, name="nl")
    m.minimize(x[0] + x[1] + y)
    assert build_convex_spec(m) is None  # was IndexError (swallowed by Model.solve)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-3.0, abs=1e-5)


def test_convex_kernel_probe_takes_no_jacobian_of_a_continuous_model(monkeypatch):
    from discopt import _tape_nlp_evaluator as tape
    from discopt.solvers._convex_kernel import build_convex_spec

    calls = []
    real = tape.make_evaluator

    def spy(*a, **k):
        ev = real(*a, **k)
        for name in ("evaluate_jacobian", "evaluate_sparse_jacobian"):
            fn = getattr(ev, name, None)
            if fn is not None:

                def wrapped(*aa, _fn=fn, _name=name, **kk):
                    calls.append(_name)
                    return _fn(*aa, **kk)

                try:
                    setattr(ev, name, wrapped)
                except AttributeError:
                    pytest.skip("evaluator does not allow attribute patching")
        return ev

    monkeypatch.setattr(tape, "make_evaluator", spy)
    m, _ = _bulk_chain(50)
    assert build_convex_spec(m) is None
    assert calls == []


# ── 4. solve_lagrangian ───────────────────────────────────────────────────────


def _gap(seed: int = 1, K: int = 4, T: int = 16):
    rng = np.random.default_rng(seed)
    cost = rng.integers(10, 50, size=(K, T))
    w = rng.integers(5, 25, size=(K, T))
    cap = (1.3 * w.sum(1) / K).astype(int)
    m = dm.Model("gap")
    xs = [m.binary(f"x{k}", shape=(T,)) for k in range(K)]
    m.minimize(sum(int(cost[k, i]) * xs[k][i] for k in range(K) for i in range(T)))
    for k in range(K):
        m.subject_to(sum(int(w[k, i]) * xs[k][i] for i in range(T)) <= int(cap[k]))
    for i in range(T):
        c = sum(xs[k][i] for k in range(K)) == 1
        m.subject_to(c, name=f"assign{i}")
        m.mark_coupling(c)
    return m


def _count_highs_solves(monkeypatch):
    import discopt.solvers.milp_highs as H

    once = H._solve_milp_once
    n = {True: 0, False: 0}

    def spy(*a, presolve, **k):
        n[presolve] += 1
        return once(*a, presolve=presolve, **k)

    monkeypatch.setattr(H, "_solve_milp_once", spy)
    return n


def test_lagrangian_cross_solves_only_improving_iterations(monkeypatch):
    from discopt.decomposition.lagrangian.solver import solve_lagrangian

    n = _count_highs_solves(monkeypatch)
    r = solve_lagrangian(_gap(), time_limit=120)
    assert n[True] > 0, "probe did not fire"
    # Every primary used to be cross-solved (n[False] == n[True]).
    assert n[False] < n[True], n
    exact = _gap().solve(time_limit=120)
    assert exact.status == "optimal"
    assert r.bound is not None and r.bound <= exact.objective + 1e-6


def test_lagrangian_bound_is_never_an_unconfirmed_presolve_bound(monkeypatch):
    """A presolve bound that is too HIGH must never reach ``best_L``.

    Inflate every presolve-on block bound by +5 (a #1634-style false bound). The
    presolve-free cross-solve is the truth, and the confirmed bound is ``min`` of
    the two, so the reported dual bound must stay below the true optimum.
    """
    import discopt.solvers.milp_highs as H
    from discopt.decomposition.lagrangian.solver import solve_lagrangian

    once = H._solve_milp_once
    fired = []

    def lying(*a, presolve, **k):
        res = once(*a, presolve=presolve, **k)
        if presolve and res.bound is not None:
            res.bound = float(res.bound) + 5.0
            fired.append(1)
        return res

    exact = _gap().solve(time_limit=120)
    monkeypatch.setattr(H, "_solve_milp_once", lying)
    r = solve_lagrangian(_gap(), time_limit=120, max_iterations=60)
    assert fired, "probe did not fire"
    assert r.bound is not None
    assert r.bound <= exact.objective + 1e-6


def test_lagrangian_recovery_lp_is_not_routed_by_nlp_solver(monkeypatch):
    import discopt.solvers.lp_pounce as lp_pounce
    from discopt.decomposition.lagrangian.solver import solve_lagrangian

    def boom(*a, **k):
        raise AssertionError("solve_lagrangian ran an LP on the POUNCE IPM")

    monkeypatch.setattr(lp_pounce, "solve_lp", boom)
    r = solve_lagrangian(_gap(), time_limit=120, max_iterations=20)
    assert r.bound is not None
