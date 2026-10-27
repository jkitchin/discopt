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
    # A caller-tightened absolute tolerance keeps the documented criterion with
    # its own absolute arm (#2000): max(1e-9, 1e-4 * 402).
    assert S._kernel_cutoff_slack(-402.0, 0.0, 1e-4, 1e-9) == pytest.approx(0.0402)
    # A looser one keeps #1243's min mapping.
    assert S._kernel_cutoff_slack(-402.0, 0.0, 1e-4, 1e-3) == pytest.approx(1e-4)


@pytest.mark.smoke
def test_slack_only_where_the_exact_cutoff_root_is_undecided(monkeypatch):
    """The rescue fires only on a failed root probe, so a decided exact-cutoff box
    is used unchanged (the always-on slack loosened nous1's bound, see the flag)."""
    calls = []
    monkeypatch.setattr(S, "_kernel_root_undecided", lambda spec, dl: calls.append(1) or False)
    S.KERNEL_CUTOFF_OBBT_STATS.clear()
    m = dm.Model("b")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize(-x * y + x)
    m.subject_to(x + y <= 5)
    _quiet_solve(m, time_limit=30)
    assert S.KERNEL_CUTOFF_OBBT_STATS.get("slack_adopted", 0) == 0
    assert S.KERNEL_CUTOFF_OBBT_STATS.get("root_undecided", 0) == 0


# ── the regression ──────────────────────────────────────────────────────────
#
# #2005: which arm pq_pooling's root takes is platform-dependent. On Linux CI the
# exact-cutoff root LP ends undecided (and so does the kernel with the slack off);
# on a macOS dev machine the same box fathoms by propagation and certifies with no
# slack. The real-path tests below therefore assert only what holds on either arm
# (sound certificate; the slack never adopted without a failed exact probe), and
# the two mechanisms are pinned by constructing the undecided state directly.


def _assert_pq_certified(r):
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(PQ_OPTIMUM, rel=1e-8)
    assert r.bound >= r.objective - 1e-6  # maximize: bound is an upper bound


@pytest.mark.slow
@pytest.mark.correctness
def test_pq_pooling_certifies_at_the_root_again(monkeypatch):
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", raising=False)
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT", raising=False)
    S.KERNEL_CUTOFF_OBBT_STATS.clear()
    r = _quiet_solve(pq_pooling(), max_nodes=2000, time_limit=120)
    st = S.KERNEL_CUTOFF_OBBT_STATS
    assert st["fired"] >= 1, "cutoff OBBT never ran: probe is blind"
    assert st["adopted"] >= 1
    # The rescue runs only after a failed exact-cutoff probe, and resolves once.
    assert st["root_undecided"] <= 1
    assert st["slack_adopted"] + st["kept_root_box"] == st["root_undecided"]
    _assert_pq_certified(r)
    if r.solver_stats["root/status"] == "undecided":
        # Only reachable when the exact box was kept with an undecided root (the
        # probe ran out of time): the #933 fallback carries the certificate.
        assert st["slack_adopted"] == 0
        assert r.bound_source == "root_relaxation"
    else:
        # Whichever box was adopted (exact where decided, the slack one where
        # not), its root decides and certifies the seed (#1983's regression).
        assert r.node_count <= 5


@pytest.mark.slow
@pytest.mark.correctness
def test_undecided_exact_cutoff_root_takes_the_slack_box(monkeypatch):
    """The exact-cutoff root probe is made to fail, as it does on Linux CI: the
    OBBT is redone with the cutoff lifted by the kernel's acceptance gap, that box
    is adopted, and the solve certifies at the root."""
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", raising=False)
    monkeypatch.delenv("DISCOPT_KERNEL_CUTOFF_OBBT", raising=False)
    real_probe, real_box = S._kernel_root_undecided, S._kernel_cutoff_obbt_box
    probes, cutoffs = [], []

    def probe(spec, deadline):
        probes.append(1)
        return True if len(probes) == 1 else real_probe(spec, deadline)

    def box(model, lb, ub, cutoff, deadline):
        cutoffs.append(float(cutoff))
        return real_box(model, lb, ub, cutoff, deadline)

    monkeypatch.setattr(S, "_kernel_root_undecided", probe)
    monkeypatch.setattr(S, "_kernel_cutoff_obbt_box", box)
    S.KERNEL_CUTOFF_OBBT_STATS.clear()
    r = _quiet_solve(pq_pooling(), max_nodes=2000, time_limit=120)
    st = S.KERNEL_CUTOFF_OBBT_STATS
    assert len(probes) == 2, probes  # the exact box, then the slack box
    assert len(cutoffs) == 2, cutoffs
    # Internal (minimised) units: the exact cutoff is the seed, -PQ_OPTIMUM.
    assert cutoffs[0] == pytest.approx(-PQ_OPTIMUM, rel=1e-8)
    slack = S._kernel_cutoff_slack(cutoffs[0], 0.0, S._DEFAULT_REL_GAP_TOL, None)
    assert slack == pytest.approx(1e-4 * PQ_OPTIMUM)
    assert cutoffs[1] - cutoffs[0] == pytest.approx(slack)
    assert st["root_undecided"] == 1
    assert st["slack_adopted"] == 1 and st["kept_root_box"] == 0
    assert st["adopted"] == 1
    _assert_pq_certified(r)
    assert r.node_count <= 5


@pytest.mark.slow
@pytest.mark.correctness
def test_slack_off_certifies_on_either_root_arm(monkeypatch):
    """With the slack off, the real solve is sound whichever arm the platform's root
    LP takes; a bound-less root is closed by the #933 fallback."""
    monkeypatch.setenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", "0")
    r = _quiet_solve(pq_pooling(), max_nodes=200, time_limit=120)
    _assert_pq_certified(r)
    if r.solver_stats["root/status"] == "undecided":
        assert r.bound_source == "root_relaxation"


@pytest.mark.slow
@pytest.mark.correctness
def test_fallback_bound_that_closes_the_gap_certifies(monkeypatch, caplog):
    """A bound-less ``node_limit`` kernel exit (what pq_pooling's kernel returns on
    Linux with the slack off) is constructed on every platform: the #933 fallback
    bound meets the incumbent, which is a proof, and the log names the real exit."""
    import discopt._rust as R

    monkeypatch.setenv("DISCOPT_KERNEL_CUTOFF_OBBT_SLACK", "0")
    real_tree = R.solve_spatial_tree_py
    injected = []

    def tree(**kw):
        res = real_tree(**kw)
        res["status"] = "node_limit"
        res["bound"] = float("-inf")
        res["root_bound"] = float("-inf")
        res["root_status"] = "undecided"
        injected.append(1)
        return res

    monkeypatch.setattr(R, "solve_spatial_tree_py", tree)
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = _quiet_solve(pq_pooling(), max_nodes=200, time_limit=120)
    assert injected, "the kernel was never called: injection is blind"
    assert r.solver_stats["root/status"] == "undecided"
    assert r.bound_source == "root_relaxation"
    _assert_pq_certified(r)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("bound-less at node_limit" in s for s in msgs), msgs
    assert any("node_limit -> optimal" in s for s in msgs), msgs
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
