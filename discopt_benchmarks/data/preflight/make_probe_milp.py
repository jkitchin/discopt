"""Regenerate ``probe_milp.nl``, the LP/MILP solver-preflight probe.

A 5x5 assignment problem with one side constraint: 25 binaries and 11 linear
rows. Just over the 10-variable / 10-constraint cap of the AMPL demo licenses,
so a demo-licensed binary refuses it instead of passing the preflight.

Its optimum is not trusted from any solver: ``brute_force_optimum`` enumerates
all 120 assignments, and ``discopt_benchmarks/tests/test_solver_preflight.py``
asserts that value equals ``PREFLIGHT_PROBES["milp"].optimum``.

    python discopt_benchmarks/data/preflight/make_probe_milp.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

COST = [
    [9, 2, 7, 8, 6],
    [6, 4, 3, 7, 5],
    [5, 8, 1, 8, 4],
    [7, 6, 9, 4, 3],
    [3, 7, 5, 9, 2],
]
# Side constraint: x[0][1] + x[2][2] <= 1. Both are cheap cells, so it binds.
SIDE = [(0, 1), (2, 2)]
N = len(COST)


def brute_force_optimum() -> float:
    best = float("inf")
    for perm in itertools.permutations(range(N)):
        if sum(perm[i] == j for i, j in SIDE) > 1:
            continue
        best = min(best, sum(COST[i][perm[i]] for i in range(N)))
    return float(best)


def build_model():
    import discopt.modeling as dm

    m = dm.Model("probe_milp")
    x = [[m.binary(f"x_{i}_{j}") for j in range(N)] for i in range(N)]
    for i in range(N):
        m.subject_to(sum(x[i][j] for j in range(N)) == 1, name=f"row_{i}")
    for j in range(N):
        m.subject_to(sum(x[i][j] for i in range(N)) == 1, name=f"col_{j}")
    m.subject_to(sum(x[i][j] for i, j in SIDE) <= 1, name="side")
    m.minimize(sum(COST[i][j] * x[i][j] for i in range(N) for j in range(N)))
    return m


if __name__ == "__main__":
    out = Path(__file__).with_name("probe_milp.nl")
    build_model().to_nl(str(out))
    print(f"wrote {out}; brute-force optimum = {brute_force_optimum()}")
