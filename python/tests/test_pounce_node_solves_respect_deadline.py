"""Structured POUNCE node solves are bounded by the solve's deadline.

``_solve_node_lp_pounce`` and ``_structured_node_recovery`` called
``pounce.solve_qp`` with no time limit, though both sit under a caller that owns
one. A node or recovery solve started near the deadline then ran on past it: on
``test_1236``'s captured fac2 OA master (100 s budget, POUNCE main) one recovery
solve ran 12.8 s past the deadline, enough for that test to exceed its 120 s
pytest timeout on CI. Both are now given the remaining time and are not started
once none is left; a stopped or skipped solve leaves the node open on its parent
bound, which is the answer the deadline gives it anyway. The root relaxation is
the one exemption: it runs to completion, as before, because it seeds the bound.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import scipy.sparse as sp

pounce = pytest.importorskip("pounce")

SETUP_SLACK = 0.5

MASTER = os.path.join(os.path.dirname(__file__), "data", "oa_masters", "fac2_master0.npz")


def _load_master():
    d = np.load(MASTER)
    kw = dict(
        c=d["c"],
        bounds=list(zip(d["lb"].tolist(), d["ub"].tolist())),
        integrality=d["integrality"],
    )
    for mat, rhs in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        kw[mat] = sp.csr_matrix(
            (d[f"{mat}_data"], d[f"{mat}_indices"], d[f"{mat}_indptr"]),
            shape=tuple(d[f"{mat}_shape"]),
        )
        kw[rhs] = d[rhs]
    return kw


def test_every_structured_node_solve_carries_the_remaining_budget(monkeypatch):
    from discopt.solvers.milp_pounce import solve_milp

    budget = 10.0
    calls: list[tuple[float, object]] = []
    orig = pounce.solve_qp
    t0 = [0.0]

    def spy(*args, **kwargs):
        calls.append((time.perf_counter() - t0[0], kwargs.get("time_limit")))
        return orig(*args, **kwargs)

    monkeypatch.setattr(pounce, "solve_qp", spy)
    t0[0] = time.perf_counter()
    solve_milp(time_limit=budget, gap_tolerance=1e-4, max_nodes=200, **_load_master())

    # The root relaxation is exempt by design -- it seeds the dual bound, which
    # even a ``time_limit=1e-6`` solve must report -- and it is the first
    # structured solve. Every tree-node solve after it must carry a budget.
    assert calls and calls[0][1] is None, calls[:1]
    tree_calls = calls[1:]
    assert tree_calls, "no tree-node structured solve ran -- the probe did not fire"
    for started_at, limit in tree_calls:
        assert limit is not None, "a structured node solve ran with no time limit"
        assert 0.0 < float(limit) <= budget
        # Not started after the deadline. The slack covers solve_milp starting its
        # own clock a little after this test's t0 (measured ~0.35 s).
        assert started_at <= budget + SETUP_SLACK, started_at
        # Never granted more than what was left when it started.
        assert started_at + float(limit) <= budget + SETUP_SLACK, (started_at, limit)
