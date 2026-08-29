"""#1066: the root cut budget every Python-driven MILP solve runs at.

``milp_simplex`` is the single funnel through which every in-house MILP reaches
the Rust driver, and until this change it handed the driver nothing — so every
solve inherited the binding's defaults (``root_cuts=16, cut_rounds=1,
cut_select=False``), set in #334 against a cost #1102 removed: back then each
round of the root loop re-derived the augmented LP from a cold slack basis, so a
second round cost a full root solve.

Simply *raising* that budget was measured and rejected (performance-plan §23):
it is a large win on a hard master (rsyn0830m 49.2 s -> 0.3 s) and a large loss
on an easy one (the tls2 masters close in 241-595 nodes at the legacy budget and
in 0.0 s; at the raised budget the extra rows derail the search and it is still
open at 60 s). Neither budget dominates, so ``solve_milp`` *measures*: it probes
at the legacy budget under a node cap and spends the strong budget only on a
MILP the probe fails to close.

These tests pin the parts that can silently rot:

* the probe runs at the legacy budget and, when it *proves* an answer, is the
  only solve that happens — no escalation overhead on the common case;
* a MILP the probe cannot close escalates exactly once, to the strong budget,
  seeded with the probe's incumbent;
* ``DISCOPT_MILP_CUT_BUDGET=0`` restores the single legacy solve exactly;
* the pure-LP short-circuit still wins under either flag value — a model with no
  integer column has nothing to cut, and a budget there is inert work on every
  relaxation LP;
* the merge is sound (bound is the better of the two, never invented), and a
  strong arm that *contradicts* a feasible point the probe holds is refused
  rather than allowed to emit a false certificate.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

rust = pytest.importorskip("discopt._rust")
if not hasattr(rust, "solve_milp_csc_py"):
    pytest.skip("simplex MILP binding not built", allow_module_level=True)

from discopt.solvers import MILPResult, SolveStatus, milp_simplex  # noqa: E402

CUT_KEYS = ("root_cuts", "cut_rounds", "cut_select")
LEGACY = {"root_cuts": 16, "cut_rounds": 1, "cut_select": False}
STRONG = {"root_cuts": 200, "cut_rounds": 10, "cut_select": True}


def _tiny_milp():
    """min -x0 - x1 s.t. 2x0 + 2x1 <= 3, x binary — one fractional root LP."""
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[2.0, 2.0]])
    b_ub = np.array([3.0])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    integrality = np.array([1, 1])
    return dict(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, integrality=integrality)


def _market_split(m=4, seed=20260828):
    """Cornuejols-Dawande market split: the canonical cut-resistant MILP.

    ``m`` equality rows over ``10*(m-1)`` binaries with a fractional-but-tight
    LP relaxation. Enumeration is exponential, so the probe cannot close it
    inside the node cap — this is the escalating class.
    """
    rng = np.random.default_rng(seed)
    n = 10 * (m - 1)
    a = rng.integers(0, 100, size=(m, n)).astype(float)
    b = np.floor(a.sum(axis=1) / 2.0)
    return dict(
        c=np.zeros(n),
        A_eq=a,
        b_eq=b,
        bounds=[(0.0, 1.0)] * n,
        integrality=np.ones(n, dtype=int),
    )


def _capture(monkeypatch, **solve_kwargs):
    """Run one ``solve_milp``; return its result and each driver call's options."""
    seen: list[dict] = []
    real = rust.solve_milp_csc_py

    def spy(*args, **kwargs):
        rec = {k: kwargs.get(k) for k in CUT_KEYS}
        # positional arg 12 is max_nodes; 14 is the seed keyword
        rec["max_nodes"] = args[12]
        rec["seeded"] = kwargs.get("initial_incumbent") is not None
        seen.append(rec)
        return real(*args, **kwargs)

    monkeypatch.setattr(rust, "solve_milp_csc_py", spy)
    res = milp_simplex.solve_milp(**solve_kwargs)
    assert seen, "the driver was never called — this test measured nothing"
    return res, seen


def test_flag_default_is_the_module_constant(monkeypatch):
    """The env var is the only thing that moves the flag off its module default."""
    monkeypatch.delenv("DISCOPT_MILP_CUT_BUDGET", raising=False)
    assert milp_simplex.milp_cut_budget_enabled() is milp_simplex._MILP_CUT_BUDGET_DEFAULT
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "0")
    assert milp_simplex.milp_cut_budget_enabled() is False
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "1")
    assert milp_simplex.milp_cut_budget_enabled() is True


def test_opt_out_is_one_legacy_solve(monkeypatch):
    """``=0`` is the pre-#1066 path: one call, legacy budget, full node budget."""
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "0")
    res, seen = _capture(monkeypatch, max_nodes=50_000, **_market_split())
    assert len(seen) == 1, f"the opt-out must not escalate, got {len(seen)} solves"
    assert {k: seen[0][k] for k in CUT_KEYS} == LEGACY
    assert seen[0]["max_nodes"] == 50_000
    assert seen[0]["seeded"] is False
    assert res.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)


def test_probe_that_proves_it_is_the_only_solve(monkeypatch):
    """The common case: the cheap budget closes it, so nothing escalates."""
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "1")
    res, seen = _capture(monkeypatch, **_tiny_milp())
    assert res.status == SolveStatus.OPTIMAL
    assert len(seen) == 1, "a MILP the probe proves must not pay for a second solve"
    assert {k: seen[0][k] for k in CUT_KEYS} == LEGACY
    assert seen[0]["max_nodes"] == milp_simplex._PROBE_MAX_NODES


def test_probe_that_stalls_escalates_once_and_is_seeded(monkeypatch):
    """The escalating class: probe at the cap, then exactly one strong solve."""
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "1")
    res, seen = _capture(monkeypatch, max_nodes=60_000, time_limit=60.0, **_market_split())
    assert len(seen) == 2, f"expected probe + escalation, got {len(seen)} solves"

    probe, strong = seen
    assert {k: probe[k] for k in CUT_KEYS} == LEGACY
    assert probe["max_nodes"] == milp_simplex._PROBE_MAX_NODES
    assert {k: strong[k] for k in CUT_KEYS} == STRONG
    assert strong["max_nodes"] < 60_000, "the probe's nodes must come off the budget"
    assert res.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_pure_lp_keeps_its_short_circuit_under_either_flag(monkeypatch, flag):
    """No integer column means nothing to cut — and never an escalation."""
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", flag)
    _, seen = _capture(
        monkeypatch,
        c=np.array([-1.0, -1.0]),
        A_ub=np.array([[2.0, 2.0]]),
        b_ub=np.array([3.0]),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        integrality=None,
    )
    assert len(seen) == 1, "an LP has no integer search to escalate"
    assert seen[0]["root_cuts"] == 0
    assert seen[0]["cut_rounds"] == 0


def test_merge_takes_the_better_bound_and_incumbent():
    """Both arms bound the same MILP, so max-bound / min-incumbent is sound."""
    probe = MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=np.array([1.0]),
        objective=10.0,
        bound=1.0,
        node_count=5,
    )
    strong = MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=np.array([2.0]),
        objective=8.0,
        bound=3.0,
        node_count=7,
    )
    merged = milp_simplex._merge_escalation(probe, strong)
    assert merged.bound == 3.0, "the tighter valid dual bound must survive"
    assert merged.objective == 8.0, "the better incumbent must survive"
    assert merged.node_count == 12, "the caller must be told what the search cost"

    # ...and symmetrically, when the probe holds the better of each.
    flipped = milp_simplex._merge_escalation(
        MILPResult(
            status=SolveStatus.ITERATION_LIMIT,
            x=np.array([1.0]),
            objective=8.0,
            bound=3.0,
            node_count=5,
        ),
        MILPResult(
            status=SolveStatus.ITERATION_LIMIT,
            x=np.array([2.0]),
            objective=10.0,
            bound=1.0,
            node_count=7,
        ),
    )
    assert flipped.bound == 3.0
    assert flipped.objective == 8.0


def test_strong_arm_contradicting_a_feasible_point_is_refused(caplog):
    """A false ``infeasible`` must never reach the caller (CLAUDE.md §1).

    If the strong arm claims infeasibility on a MILP where the probe holds a
    point, the two cut budgets disagree about the same problem. The escalation
    declines to be the arm that emits the false certificate.
    """
    probe = MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=np.array([1.0]),
        objective=10.0,
        bound=1.0,
        node_count=5,
    )
    strong = MILPResult(status=SolveStatus.INFEASIBLE, node_count=7)
    with caplog.at_level(logging.ERROR):
        merged = milp_simplex._merge_escalation(probe, strong)
    assert merged.status == SolveStatus.ITERATION_LIMIT
    assert merged.objective == 10.0
    assert merged.node_count == 12
    assert "disagree" in caplog.text, "the contradiction must be reported, not swallowed"


def test_strong_arm_claiming_a_worse_optimum_is_refused(caplog):
    """An 'optimum' above a point we already hold is a cut-off feasible region."""
    probe = MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=np.array([1.0]),
        objective=10.0,
        bound=1.0,
        node_count=5,
    )
    strong = MILPResult(
        status=SolveStatus.OPTIMAL,
        x=np.array([2.0]),
        objective=12.0,
        bound=12.0,
        node_count=7,
    )
    with caplog.at_level(logging.ERROR):
        merged = milp_simplex._merge_escalation(probe, strong)
    assert merged.status != SolveStatus.OPTIMAL, "a cut-off optimum must not be certified"
    assert merged.objective == 10.0
    assert "disagree" in caplog.text


def test_a_proof_from_the_strong_arm_is_returned_as_a_proof():
    """The normal escalation outcome: the strong arm closes what the probe could not."""
    probe = MILPResult(
        status=SolveStatus.ITERATION_LIMIT,
        x=np.array([1.0]),
        objective=10.0,
        bound=1.0,
        node_count=5,
    )
    strong = MILPResult(
        status=SolveStatus.OPTIMAL,
        x=np.array([2.0]),
        objective=9.0,
        bound=9.0,
        node_count=7,
    )
    merged = milp_simplex._merge_escalation(probe, strong)
    assert merged.status == SolveStatus.OPTIMAL
    assert merged.objective == 9.0
    assert merged.node_count == 12


MASTERS = pathlib.Path(__file__).parent / "data" / "oa_masters"


def _load_master(name):
    """Rebuild a captured OA master's ``solve_milp`` kwargs from its .npz."""
    d = np.load(MASTERS / name)
    kw = dict(
        c=d["c"],
        bounds=list(zip(d["lb"].tolist(), d["ub"].tolist())),
        integrality=d["integrality"],
    )
    for tag, mat, rhs in (("A_ub", "A_ub", "b_ub"), ("A_eq", "A_eq", "b_eq")):
        if f"{tag}_data" in d.files:
            kw[mat] = sp.csr_matrix(
                (d[f"{tag}_data"], d[f"{tag}_indices"], d[f"{tag}_indptr"]),
                shape=tuple(d[f"{tag}_shape"]),
            )
            kw[rhs] = d[rhs]
    return kw


@pytest.mark.slow
def test_easy_master_is_not_derailed_by_the_raised_budget(monkeypatch):
    """The regression that killed the static profile, pinned.

    ``tls2_master0`` closes in 241 nodes at the legacy budget. Handing it the
    raised budget unconditionally leaves it *unsolved* at 60 s — the extra rows
    derail a search that was already nearly done. The escalation must decline to
    escalate here, so the flag being on costs it nothing.
    """
    problem = _load_master("tls2_master0.npz")

    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "0")
    legacy = milp_simplex.solve_milp(max_nodes=500_000, time_limit=60.0, **problem)
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "1")
    policy = milp_simplex.solve_milp(max_nodes=500_000, time_limit=60.0, **problem)

    assert legacy.status == SolveStatus.OPTIMAL, "the captured master must be closable"
    assert policy.status == SolveStatus.OPTIMAL, (
        "the flag must not cost this master its proof — this is the tls2 regression "
        "that rejected the unconditional raised budget"
    )
    assert policy.objective == pytest.approx(legacy.objective, abs=1e-6)
    # Declining to escalate means the probe alone answered it: no second solve.
    assert policy.node_count <= legacy.node_count + 1


@pytest.mark.slow
def test_hard_master_is_closed_only_because_of_the_escalation(monkeypatch):
    """The win, pinned. ``rsyn0830m_master0`` needs 529 573 nodes / 49.2 s at the
    legacy budget and 1 197 nodes / 0.3 s at the raised one. Under a budget
    between the two, only the escalating arm closes it — which is exactly why
    ``rsyn0830m`` cannot certify at default settings without this change."""
    problem = _load_master("rsyn0830m_master0.npz")

    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "0")
    legacy = milp_simplex.solve_milp(max_nodes=500_000, time_limit=20.0, **problem)
    monkeypatch.setenv("DISCOPT_MILP_CUT_BUDGET", "1")
    policy = milp_simplex.solve_milp(max_nodes=500_000, time_limit=20.0, **problem)

    assert policy.status == SolveStatus.OPTIMAL, (
        "the escalation must close the master the legacy budget cannot"
    )
    assert legacy.status != SolveStatus.OPTIMAL, (
        "the legacy budget closing this in 20 s would mean the master is no longer "
        "the hard class this test was built from — re-measure before relaxing it"
    )
    # The escalation's bound is sound: it never crosses the optimum it proved.
    assert policy.bound == pytest.approx(policy.objective, rel=1e-6)
    if legacy.bound is not None:
        assert policy.bound >= legacy.bound - 1e-6, "escalation must not loosen the bound"
