"""A certifying MILP solve explores the same tree whatever its time limit (#1116 role-2 clock).

The Rust engine's heuristic sub-MIPs (root RENS, in-tree RENS/RINS) used to stop at a
share of the wall clock left, ``min(0.25 x left, 2 s)``, and the root RENS was skipped
outright below 0.05 s. What a certifying solve found therefore depended on the time
limit and on machine load: ``test_1236_determinism_contract[nvs02]`` failed in #2220's
CI with node counts ``[2, 4, 2]``, and in-process GIL contention moved nvs02 from 2 to
58 nodes. HiGHS bounds its sub-MIPs by work (nodes, leaves, stall nodes) and gives
them only the parent's remaining time (``HighsPrimalHeuristics.cpp:96-111``); so does
discopt now: ``DISCOPT_MILP_SUBMIP_WORK`` (default 2.5e9 deterministic work units, about
2 s; ``=0`` restores the wall share) counts LP iterations, nodes, LU entries, exact
arithmetic and mod-k separation, and the simplex stops on it mid-LP.

Measured on 60 random multi-knapsacks (seeds 0-59, this file's generator) solved at
30 s and at 0.15 s: with the wall-share budget 31 certified at both limits with
different node counts; with the work budget, 0. The seeds below are three of the 31,
each certifying in under 0.03 s, so the 0.15 s limit is a 5x margin.
"""

import numpy as np
import pytest
from discopt.solvers.milp_simplex import solve_milp


def _knapsack(seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(20, 50))
    m = int(r.integers(3, 8))
    A = r.integers(5, 60, (m, n)).astype(float)
    b = A.sum(1) * r.uniform(0.25, 0.5)
    c = -(A.sum(0) * r.uniform(0.6, 1.4, n)).round()
    return dict(
        c=c, A_ub=A, b_ub=b, bounds=[(0, int(r.integers(1, 4)))] * n, integrality=np.ones(n, int)
    )


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [44, 47, 51])
def test_a_certifying_solve_does_not_depend_on_its_time_limit(seed):
    long, short = (solve_milp(**_knapsack(seed), time_limit=t) for t in (30.0, 0.15))
    assert long.status.value == "optimal" and short.status.value == "optimal", (
        long.status,
        short.status,
    )
    assert long.node_count == short.node_count, (long.node_count, short.node_count)
    assert long.objective == pytest.approx(short.objective, abs=1e-9)
