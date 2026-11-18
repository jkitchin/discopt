"""#2235: the #2069 independent confirmation runs the in-house engine's batched search.

#2196 made the HiGHS-design tree search the in-house MILP driver's default. That
search is a port of HiGHS's serial default search and evaluates one node LP at a
time, so the confirmation of a HiGHS OA master gave up the batched search's
parallel node LPs. On ``portfol_classical050_1``'s master it took 212 s against 37 s
(4 threads); ``test_2059_soc_outer``'s solve went from ~74 s to ~250 s on a machine
without SCIP. ``milp_simplex.solve_milp(batched_search=True)`` selects the batched
search per call, and ``milp_highs._run_simplex_engine`` uses it. Both searches are
complete and sound; only the tree differs.

These tests pin:

* the confirmation passes ``batched_search=True``;
* the keyword reaches the driver: with it the driver runs no HiGHS-design search
  (profile counter ``HighsMipSearchRuns``), and without it the same MILPs do,
  which proves the population exercises the route; the optima agree.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import MILPResult, SolveStatus, milp_highs, milp_simplex

_RUNS = re.compile(r"^\s+HighsMipSearchRuns\s+(\d+)$", re.M)


def test_confirmation_runs_the_batched_search(monkeypatch):
    seen = []

    def fake_solve_milp(**kw):
        seen.append(kw)
        return MILPResult(status=SolveStatus.OPTIMAL, bound=1.0, node_count=3)

    monkeypatch.setattr(milp_simplex, "solve_milp", fake_solve_milp)
    arrays = dict(c=np.array([1.0]), integrality=np.array([1]))
    status, bound, nodes = milp_highs._run_simplex_engine(arrays, 5.0, 1e-6, None)
    assert (status, bound, nodes) == (SolveStatus.OPTIMAL, 1.0, 3)
    assert len(seen) == 1
    assert seen[0]["batched_search"] is True
    assert seen[0]["time_limit"] == 5.0 and seen[0]["gap_tolerance"] == 1e-6


def _knapsacks(count: int):
    """Small bounded-integer packing MILPs (maximise), several rows each."""
    rng = np.random.default_rng(2235)
    for _ in range(count):
        n = int(rng.integers(8, 15))
        m = int(rng.integers(4, 9))
        A = rng.integers(0, 8, size=(m, n)).astype(float)
        b = rng.integers(10, 31, size=m).astype(float) + 0.5
        c = -(rng.integers(1, 10, size=n).astype(float) + 0.5)
        yield c, sp.csr_matrix(A), b, [(0.0, 4.0)] * n, np.ones(n, dtype=int)


def _solve(capfd, c, A, b, bounds, integ, batched):
    capfd.readouterr()
    res = milp_simplex.solve_milp(
        c, A_ub=A, b_ub=b, bounds=bounds, integrality=integ, batched_search=batched
    )
    runs = sum(int(x) for x in _RUNS.findall(capfd.readouterr().err))
    return res, runs


def test_batched_search_keyword_reaches_the_driver(monkeypatch, capfd):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    monkeypatch.delenv("DISCOPT_HIGHS_MIP_SEARCH", raising=False)  # the default
    route_runs = compared = 0
    for c, A, b, bounds, integ in _knapsacks(12):
        h, h_runs = _solve(capfd, c, A, b, bounds, integ, batched=False)
        bt, bt_runs = _solve(capfd, c, A, b, bounds, integ, batched=True)
        assert bt_runs == 0, "batched_search=True ran the HiGHS-design search"
        route_runs += h_runs
        assert h.status == SolveStatus.OPTIMAL and bt.status == SolveStatus.OPTIMAL
        assert bt.objective == pytest.approx(h.objective, rel=1e-6, abs=1e-6)
        compared += 1
    assert compared == 12
    # Without the keyword the same MILPs take the HiGHS-design search: the zero
    # above is the option's doing, not a population the route never reaches.
    assert route_runs > 0
