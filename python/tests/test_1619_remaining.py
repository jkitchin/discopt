"""#1619 remaining items: C-01b (binary QP to the MILP route), E-02 (OA/SHOT stops
on a closed gap, and accepts a free column), A-22 (sparse Q on the pounce QP route)."""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


# ── C-01b ────────────────────────────────────────────────────────────────────


def _dopants():
    L, N, k = 4, 16, 4
    J1, J2, J3, hh = 0.40, -0.10, -0.25, -0.08
    Jm = np.zeros((N, N))
    for s, t in itertools.combinations(range(N), 2):
        (r1, c1), (r2, c2) = divmod(s, L), divmod(t, L)
        dr, dc = min(abs(r1 - r2), L - abs(r1 - r2)), min(abs(c1 - c2), L - abs(c1 - c2))
        Jm[s, t] = {(0, 1): J1, (1, 0): J1, (1, 1): J2, (0, 2): J3, (2, 0): J3}.get((dr, dc), 0.0)
    hv = np.array([hh if s < L else 0.0 for s in range(N)])
    pairs = [(s, t) for s, t in itertools.combinations(range(N), 2) if Jm[s, t] != 0]
    m = dm.Model("dopants_quadratic")
    x = m.binary("x", shape=(N,))
    m.minimize(
        dm.sum(
            lambda p: float(Jm[pairs[p]]) * x[pairs[p][0]] * x[pairs[p][1]], over=range(len(pairs))
        )
        + dm.sum(lambda s: hv[s] * x[s], over=range(N))
    )
    m.subject_to(dm.sum(lambda s: x[s], over=range(N)) == k)
    return m


def test_binary_qp_is_linearized_and_solved_on_the_milp_route():
    r = _dopants().solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-1.16, abs=1e-6)  # exhaustive optimum
    assert r.algorithm_route.startswith("highs-milp")


def test_binary_qp_legacy_opt_out_keeps_spatial_bb(monkeypatch):
    monkeypatch.setenv("DISCOPT_BINARY_QUADRATIC_MILP", "0")
    r = _dopants().solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.16, abs=1e-6)
    assert r.algorithm_route.startswith("spatial-bb")


# ── E-02 ─────────────────────────────────────────────────────────────────────


def _synthes1():
    m = dm.Model("synthes1")
    x1, x2 = m.continuous("x1", lb=0, ub=2), m.continuous("x2", lb=0, ub=2)
    x3, y = m.continuous("x3", lb=0, ub=1), m.binary("y", shape=(3,))
    m.minimize(
        5 * y[0] + 6 * y[1] + 8 * y[2] + 10 * x1 - 7 * x3 + 10
        - 18 * dm.log(x2 + 1) - 19.2 * dm.log(x1 - x2 + 1)
    )  # fmt: skip
    m.subject_to(0.8 * dm.log(x2 + 1) + 0.96 * dm.log(x1 - x2 + 1) - 0.8 * x3 >= 0)
    m.subject_to(dm.log(x2 + 1) + 1.2 * dm.log(x1 - x2 + 1) - x3 - 2 * y[2] >= -2)
    m.subject_to(x2 <= x1)
    m.subject_to(x2 <= 2 * y[0])
    m.subject_to(x1 - x2 <= 2 * y[1])
    m.subject_to(y[0] + y[1] <= 1)
    return m


@pytest.mark.parametrize("kw", [{}, {"cut_strategy": "esh"}])
def test_shot_profile_stops_when_the_gap_closes(kw):
    """Before: 1224 masters to the 45 s limit; the gap closed at iteration 2."""
    r = _synthes1().solve(
        solver="mip-nlp", mip_nlp_method="oa", mip_nlp_profile="shot", time_limit=45, **kw
    )
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(6.009758910746438, rel=1e-8)
    assert r.mip_nlp_trace["termination_reason"] == "gap"
    assert r.mip_count <= 10
    assert r.wall_time < 20


# ── A-22 ─────────────────────────────────────────────────────────────────────


def _chain_qp(n):
    m = dm.Model("chain")
    x = m.continuous("x", shape=(2 * n,), lb=0, ub=100)
    for i in range(n):
        m.subject_to(x[2 * i] + x[2 * i + 1] - 0.5 * x[(2 * i + 2) % (2 * n)] >= 1)
    m.minimize(dm.sum(x) + dm.sum(lambda j: x[j] ** 2, over=range(2 * n)))
    return m


def test_pounce_qp_route_keeps_q_sparse(monkeypatch):
    import discopt.solvers.convex_ipm_pounce as cvx
    import scipy.sparse as sp

    seen = []
    real = cvx.solve_qp

    def spy(**kw):
        seen.append((sp.issparse(kw["Q"]), sp.issparse(kw["A_ub"])))
        return real(**kw)

    monkeypatch.setattr(cvx, "solve_qp", spy)
    r = _chain_qp(200).solve(solver="pounce")
    assert r.status == "optimal" and r.gap_certified
    assert seen == [(True, True)]


def test_pounce_qp_route_peak_memory_is_sparse_sized():
    import tracemalloc

    m = _chain_qp(1500)
    tracemalloc.start()
    try:
        r = m.solve(solver="pounce")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert r.status == "optimal"
    # Dense Q alone is 3000^2 * 8 = 72 MB (and the route held several copies:
    # 722.7 MB at n = 2000 before #1619 A-22).
    assert peak < 150e6, peak / 1e6


@pytest.mark.parametrize("seed", range(6))
def test_certify_psd_sparse_matches_dense(seed):
    import scipy.sparse as sp
    from discopt.solvers.convex_ipm_pounce import certify_psd

    rng = np.random.default_rng(seed)
    n = 40
    B = sp.random(n, n, density=0.05, random_state=seed)
    Q = (B @ B.T).toarray()
    if seed % 2:
        Q[0, 0] -= 1.0  # break PSD on half the seeds
    Q[np.abs(Q) < 1e-300] = 0.0
    assert certify_psd(sp.csr_matrix(Q)) == certify_psd(Q)
    assert certify_psd(sp.csr_matrix(np.diag(rng.uniform(0, 1, 5000)))) is True


@pytest.mark.parametrize("seed", range(5))
def test_lagrangian_slope_sparse_path_is_bit_identical(seed):
    import scipy.sparse as sp
    from discopt.solver import _lagrangian_slope

    rng = np.random.default_rng(seed)
    m, n = 30, 50
    J = sp.random(m, n, density=0.2, random_state=seed, format="csr")
    lam = rng.normal(size=m)
    lam[rng.uniform(size=m) < 0.3] = 0.0
    # A gradient that cancels J^T lam to rounding on most columns, which is what
    # sends the slope down the exact-sum path.
    g = -np.asarray(J.T @ lam).ravel() * (1 + 1e-17 * rng.normal(size=n))
    gs, es, ms = _lagrangian_slope(g, J, lam)
    gd, ed, md = _lagrangian_slope(g, J.toarray(), lam)
    # The exact-sum fallback (what #1619 changed) gives identical slopes and error
    # bounds. ``mag`` comes from ``abs(J).T @ |lam|`` before that branch, a sparse
    # vs dense matmul whose summation order differs in the last ulp.
    assert np.array_equal(gs, gd) and np.array_equal(es[gs == 0], ed[gs == 0])
    unsure = np.flatnonzero(~(np.abs(gd) > 2.0 * 100 * np.finfo(float).eps * md))
    assert unsure.size > 0  # the probe exercised the exact-sum path
    np.testing.assert_allclose(ms, md, rtol=1e-14)


def test_shot_profile_accepts_an_unbounded_continuous_column():
    """The SHOT interior-point store refused infinite bounds and raised on every
    model with a free column (4stufen, contvar, dispatch, ... of the corpus)."""
    m = _synthes1()
    free = m.continuous("free", lb=-np.inf, ub=np.inf)
    m.subject_to(free >= 0)
    r = m.solve(solver="mip-nlp", mip_nlp_method="oa", mip_nlp_profile="shot", time_limit=45)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(6.009758910746438, rel=1e-8)


# ── D-11 ─────────────────────────────────────────────────────────────────────


def _pooling():
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "discopt_benchmarks", "scripts"))
    from _pooling import pool18

    return pool18


@pytest.mark.parametrize("flag, expect_equal", [("1", True), ("0", False)])
def test_native_kernel_branching_does_not_depend_on_units(monkeypatch, flag, expect_equal):
    """Haverly 2, q-formulation, flows in 1 / 10 / 100 barrels: one model in three
    units. Measured: 125 / 125 / 107 nodes with the absolute-width rule, 33 / 33 /
    33 scale-free."""
    monkeypatch.setenv("DISCOPT_SCALE_FREE_BRANCHING", flag)
    pool = _pooling()
    nodes, objs = [], []
    for unit in (1, 10, 100):
        m, _ = pool.build_pool(pool.in_units(pool.haverly(2), unit), "q")
        r = m.solve(time_limit=60)
        assert r.status == "optimal" and r.gap_certified
        assert r.algorithm_route.startswith("native-spatial")
        nodes.append(r.node_count)
        objs.append(r.objective)
    assert max(objs) - min(objs) <= 1e-6 * (1 + abs(objs[0]))
    assert (len(set(nodes)) == 1) == expect_equal, nodes
