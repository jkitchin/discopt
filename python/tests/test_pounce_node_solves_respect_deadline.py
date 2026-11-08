"""The retired matrix "pounce" MILP backend still honours ``time_limit``.

This file used to pin the deadline plumbing of the POUNCE interior-point node solves
inside the Python MILP tree (``_solve_node_lp_pounce``: one recovery solve ran 12.8 s
past a 100 s budget on ``test_1236``'s captured fac2 OA master). That backend is
retired (Python-MILP-tree retirement, R11): ``milp_pounce.solve_milp`` is an alias
for the Rust simplex engine. What a caller relied on -- the call returns within its
budget on that master -- is checked on the alias, which is what ``backend="pounce"``
now runs.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import scipy.sparse as sp

MASTER = os.path.join(os.path.dirname(__file__), "data", "oa_masters", "fac2_master0.npz")
#: Construction, warm-up and the post-deadline wind-down; generous for a loaded runner.
SLACK = 3.0


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


def test_the_retired_pounce_backend_returns_within_its_time_limit():
    from discopt.solvers.milp_pounce import solve_milp

    budget = 2.0
    t0 = time.perf_counter()
    with pytest.warns(FutureWarning, match="retired"):
        res = solve_milp(time_limit=budget, gap_tolerance=1e-12, **_load_master())
    elapsed = time.perf_counter() - t0
    assert elapsed <= budget + SLACK, elapsed
    assert res.node_count >= 1, "the engine never ran -- the probe measured nothing"
