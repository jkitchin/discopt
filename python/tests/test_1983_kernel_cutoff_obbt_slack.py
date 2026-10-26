"""#1983: a pooling model that certified at node 1 stopped certifying under #1871's cutoff OBBT.

* **The regression.** #1871 tightens the native kernel's root box by OBBT with the
  seed incumbent's value as an objective cutoff. When the seed is already optimal,
  ``{relaxation, objective <= incumbent}`` is about one point: on the pq pooling
  model below 30 columns ended with widths 1e-17 to 1e-6, the root LP over that box
  was numerically degenerate (HiGHS: infeasible; the kernel's simplex:
  ``Numerical``), every node inherited the slivers, and the solve ran 100,000 nodes
  uncertified where the untightened box certifies at node 1. The cutoff now sits
  one kernel acceptance gap above the incumbent (``_kernel_cutoff_slack``): a
  superset of the old region, so sound, and never collapsed below the precision
  the certificate is decided at.
* **Misreporting.** A bound-less kernel exit whose rigorous root-relaxation
  fallback (#933) met the incumbent was reported ``node_limit`` with ``gap = 0``;
  it is now re-certified ``optimal``, as the Python tree does with the same bound.
  The log line named ``time_limit`` for a node-limit exit.
* **Off-kernel warning.** It fired for a model the kernel never takes (an ``exp``
  rate law), and an explicit ``tuning`` moved a solve off the kernel silently.
"""

from __future__ import annotations

import logging
import warnings
from functools import reduce

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt.solver_tuning import SolverTuning

# ── the issue's pq pooling plant (seeded; 17 columns) ────────────────────────


def _lsum(terms):
    terms = list(terms)
    return reduce(lambda a, b: a + b, terms) if terms else 0.0


def _plant(seed=5, nI=5, nL=2, nJ=3, K=2):
    rng = np.random.default_rng(seed)
    qual = np.round(rng.uniform(0.5, 4.5, (nI, K)), 1)
    cost = np.round(18 - 2.0 * qual.mean(1) + rng.uniform(-1, 1, nI), 1)
    crude = "ABCDE"
    inputs = {
        crude[i]: dict(cost=float(cost[i]), qual=[float(v) for v in qual[i]]) for i in range(nI)
    }
    pools = {}
    for p in range(nL):
        feeds = sorted(rng.choice(nI, size=3, replace=False))
        pools[f"P{p + 1}"] = dict(
            feeds=[crude[i] for i in feeds], cap=float(np.round(rng.uniform(80, 160)))
        )
    products = {
        f"X{j + 1}": dict(
            price=float(np.round(rng.uniform(12, 18), 1)),
            demand=float(np.round(rng.uniform(50, 150))),
            qmax=[float(v) for v in np.round(rng.uniform(1.5, 3.0, K), 1)],
        )
        for j in range(nJ)
    }
    bypass = [(crude[i], j) for i in range(nI) for j in products if rng.uniform() < 0.25]
    return dict(
        inputs=inputs,
        pools=pools,
        products=products,
        bypass=bypass,
        pool_out=[(p, j) for p in pools for j in products],
    )


def pq_pooling() -> dm.Model:
    d = _plant()
    m = dm.Model("pool_pq")
    IN, POOLS, PROD = d["inputs"], d["pools"], d["products"]
    K = len(next(iter(IN.values()))["qual"])
    out = {p: [j for (q, j) in d["pool_out"] if q == p] for p in POOLS}
    cap = {p: POOLS[p]["cap"] for p in POOLS}

    def var(name, lo, hi):
        return m.continuous(name, lb=float(lo), ub=float(hi))

    y = {(p, j): var(f"y_{p}{j}", 0, min(cap[p], PROD[j]["demand"])) for (p, j) in d["pool_out"]}
    z = {(i, j): var(f"z_{i}{j}", 0, PROD[j]["demand"]) for (i, j) in d["bypass"]}
    byp_of = {j: [z[i, jj] for (i, jj) in d["bypass"] if jj == j] for j in PROD}
    sell = {j: _lsum([y[p, j] for p in POOLS if j in out[p]] + byp_of[j]) for j in PROD}
    bypassed = {i: [z[i, j] for (ii, j) in d["bypass"] if ii == i] for i in IN}
    q = {(i, p): var(f"q_{i}{p}", 0, 1) for p in POOLS for i in POOLS[p]["feeds"]}
    qy = {(i, p, j): q[i, p] * y[p, j] for p in POOLS for i in POOLS[p]["feeds"] for j in out[p]}
    for p in POOLS:
        m.subject_to(_lsum(q[i, p] for i in POOLS[p]["feeds"]) == 1)
    bought = {
        i: _lsum(
            [qy[i, p, j] for p in POOLS if i in POOLS[p]["feeds"] for j in out[p]] + bypassed[i]
        )
        for i in IN
    }
    for p in POOLS:  # the user's RLT rows (pq form)
        for j in out[p]:
            m.subject_to(_lsum(qy[i, p, j] for i in POOLS[p]["feeds"]) - y[p, j] == 0)
        for i in POOLS[p]["feeds"]:
            m.subject_to(_lsum(qy[i, p, j] for j in out[p]) - cap[p] * q[i, p] <= 0)
    for p in POOLS:
        m.subject_to(_lsum(y[p, j] for j in out[p]) <= cap[p])
    for j in PROD:
        m.subject_to(sell[j] <= PROD[j]["demand"])
        for k in range(K):
            into = [
                IN[i]["qual"][k] * qy[i, p, j]
                for p in POOLS
                if j in out[p]
                for i in POOLS[p]["feeds"]
            ]
            byp = [IN[i]["qual"][k] * z[i, jj] for (i, jj) in d["bypass"] if jj == j]
            m.subject_to(_lsum(into + byp) - PROD[j]["qmax"][k] * sell[j] <= 0)
    revenue = _lsum(PROD[j]["price"] * sell[j] for j in PROD)
    m.maximize(revenue - _lsum(IN[i]["cost"] * bought[i] for i in IN))
    return m


PQ_OPTIMUM = 402.0283839591


def _quiet_solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(**kw)


# ── the slack itself ────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_slack_is_the_kernel_acceptance_gap(monkeypatch):
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", raising=False)
    # Default tolerances: documented gap, max(abs 1e-6, 1e-4 * |published|).
    assert S._kernel_cutoff_slack(-402.0, 0.0, 1e-4, None) == pytest.approx(0.0402)
    assert S._kernel_cutoff_slack(-2.0, 0.0, 1e-4, None) == pytest.approx(2e-4)
    assert S._kernel_cutoff_slack(0.0, 0.0, 1e-4, None) == pytest.approx(1e-6)
    # The published value includes the offset.
    assert S._kernel_cutoff_slack(-2.0, 1000.0, 1e-4, None) == pytest.approx(0.0998)
    # A caller-tightened absolute tolerance is honoured (kernel takes the min).
    assert S._kernel_cutoff_slack(-402.0, 0.0, 1e-4, 1e-9) == pytest.approx(1e-9)


@pytest.mark.smoke
def test_slack_opt_out_restores_the_exact_cutoff(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", "0")
    assert S._kernel_cutoff_slack(-402.0, 0.0, 1e-4, None) == 0.0


# ── the regression ──────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.correctness
def test_pq_pooling_certifies_at_the_root_again(monkeypatch):
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", raising=False)
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT", raising=False)
    S.KERNEL_CUTOFF_OBBT_STATS.clear()
    r = _quiet_solve(pq_pooling(), max_nodes=2000, time_limit=120)
    assert S.KERNEL_CUTOFF_OBBT_STATS["fired"] >= 1, "cutoff OBBT never ran: probe is blind"
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(PQ_OPTIMUM, rel=1e-8)
    assert r.bound >= r.objective - 1e-6  # maximize: bound is an upper bound
    assert r.solver_stats["root/status"] == "bounded"
    assert r.node_count <= 5


@pytest.mark.slow
@pytest.mark.correctness
def test_fallback_bound_that_closes_the_gap_certifies(monkeypatch, caplog):
    """With the slack off the kernel stays undecided; its #933 fallback bound meets
    the incumbent, which is a proof, and the log names the real exit."""
    monkeypatch.setenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", "0")
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = _quiet_solve(pq_pooling(), max_nodes=200, time_limit=120)
    assert r.solver_stats["root/status"] == "undecided"
    assert r.bound_source == "root_relaxation"
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(PQ_OPTIMUM, rel=1e-8)
    assert r.bound >= r.objective - 1e-6
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("bound-less at node_limit" in s for s in msgs), msgs
    assert not any("bound-less at time_limit" in s for s in msgs)


# ── the off-kernel warning ──────────────────────────────────────────────────


def _bilinear():
    m = dm.Model("b")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize(-x * y + x)
    m.subject_to(x + y <= 5)
    return m


def _rate_law():
    m = dm.Model("cstr")
    T = m.continuous("T", lb=300, ub=400)
    c = m.continuous("c", lb=0, ub=1)
    m.minimize(c + 0.001 * T)
    m.subject_to(c == 1 - 0.5 * dm.exp(-1000.0 / T))
    return m


def _kernel_warnings(m, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = m.solve(time_limit=30, **kw)
    return r, [x for x in w if "native spatial branch-and-bound kernel" in str(x.message)]


@pytest.mark.smoke
def test_explicit_tuning_is_named_when_it_moves_a_solve_off_the_kernel():
    r, w = _kernel_warnings(_bilinear(), tuning=SolverTuning(root_fixpoint=False))
    assert r.status == "optimal" and "Python tree" in str(r.algorithm_route)
    assert len(w) == 1 and "tuning" in str(w[0].message)


@pytest.mark.smoke
def test_no_warning_for_a_model_the_kernel_never_takes():
    r, w = _kernel_warnings(_rate_law(), strategy="depth_first")
    assert r.status == "optimal"
    assert w == []
    # The model is outside the kernel's term families: the plain solve does not
    # reach the kernel either, so there was nothing to move.
    r0, w0 = _kernel_warnings(_rate_law())
    assert "native-spatial" not in str(r0.algorithm_route) and w0 == []


@pytest.mark.smoke
def test_a_kernel_model_still_warns_and_a_plain_one_does_not():
    r, w = _kernel_warnings(_bilinear(), strategy="depth_first")
    assert len(w) == 1 and "strategy" in str(w[0].message)
    r0, w0 = _kernel_warnings(_bilinear())
    assert "native-spatial" in str(r0.algorithm_route) and w0 == []


@pytest.mark.smoke
def test_bypass_reasons_match_the_route():
    reasons = S._native_kernel_bypass_reasons(
        mccormick_bounds="auto",
        initial_point=None,
        lazy_constraints=None,
        incumbent_callback=None,
        node_callback=None,
        kwargs={},
    )
    assert reasons == []
    reasons = S._native_kernel_bypass_reasons(
        mccormick_bounds="none",
        initial_point=np.zeros(2),
        lazy_constraints=None,
        incumbent_callback=None,
        node_callback=None,
        kwargs={"tuning": SolverTuning()},
        search_levers=["strategy"],
    )
    assert reasons == ["initial_point", "mccormick_bounds", "tuning", "strategy"]
