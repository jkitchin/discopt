"""#1953: the native LP route certifies ``optimal`` with an NS safe bound.

``_solve_lp_matrix`` used to report any engine ``OPTIMAL`` as ``optimal`` with
``bound = objective``. A simplex that misjudged a reduced cost as noise stopped
0.25 above the optimum and certified it. The route now runs the HiGHS route's
NS ladder (``lp_milp_highs.certify_lp_dual``) on the engine's own dual. An
engine that stops early gets ``feasible`` with a valid bound, not ``optimal``.

The Rust MILP driver's node bound gets the same treatment
(``DISCOPT_MILP_NS_NODE_BOUND``, default on): the last test checks the gate
actually runs, and that ``=0`` really turns it off.

The engine below fakes that failure exactly: a feasible point that is not
optimal, labelled ``OPTIMAL``, with a dual consistent with that wrong vertex.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pytest
from discopt import Model
from discopt.solver import _solve_lp_matrix
from discopt.solvers import LPResult, SolveStatus


def _model(maximize=False):
    m = Model("ns_gate")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(x + y <= 1)
    if maximize:
        m.maximize(x + y + 3.0)
    else:
        m.minimize(-x - y + 3.0)
    return m


def _engine(x, objective, dual):
    calls = []

    def solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, time_limit):
        calls.append(1)
        return LPResult(
            status=SolveStatus.OPTIMAL,
            x=np.asarray(x, dtype=float),
            objective=objective,
            dual_values=None if dual is None else np.asarray(dual, dtype=float),
            reduced_costs=None,
            iterations=1,
        )

    return solve_lp, calls


def _solve(m, engine):
    return _solve_lp_matrix(m, time.perf_counter(), 10.0, engine, "fake")


@pytest.mark.smoke
@pytest.mark.parametrize("maximize", [False, True])
def test_early_stop_is_not_certified(maximize):
    # The point (0, 0) is feasible with value 3; the optimum is 2 (max: 4).
    # The engine's dual is that vertex's own (y = 0): its NS bound is the box bound.
    eng, calls = _engine([0.0, 0.0], 0.0, [0.0])
    r = _solve(_model(maximize), eng)
    assert calls == [1]
    assert r.status == "feasible"  # was "optimal" with bound 3.0
    assert r.gap_certified is False
    assert r.objective == pytest.approx(3.0)
    if maximize:
        assert r.bound is not None and r.bound >= 4.0  # a valid upper bound
    else:
        assert r.bound is not None and r.bound <= 2.0  # a valid lower bound
    assert r.bound_valid is True


@pytest.mark.smoke
@pytest.mark.parametrize("maximize", [False, True])
def test_true_optimum_still_certified(maximize):
    # (1, 0) is optimal; y_row = -1 makes every reduced cost exactly zero.
    eng, _ = _engine([1.0, 0.0], -1.0, [-1.0])
    r = _solve(_model(maximize), eng)
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective == pytest.approx(4.0 if maximize else 2.0)
    assert r.bound == pytest.approx(r.objective, abs=1e-9)


@pytest.mark.smoke
def test_missing_dual_is_not_certified():
    eng, _ = _engine([1.0, 0.0], -1.0, None)
    r = _solve(_model(), eng)
    assert r.status == "feasible"
    assert r.bound is None and r.bound_valid is False
    assert r.solver_stats["lp/ns_bound_missing"] == 1.0


@pytest.mark.smoke
def test_equality_and_inequality_rows_together():
    # A_ub and A_eq both present: the standard form stacks slacks for the
    # inequality rows only, and the dual is read inequality rows first.
    m = Model("mixed")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y <= 3)
    m.subject_to(x - y == 1)
    m.minimize(-x)  # optimum x = 2, y = 1, value -2
    r = m.solve(milp_backend="native")
    assert r.algorithm_route.startswith("native-lp")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-2.0)
    assert r.bound == pytest.approx(-2.0, abs=1e-6) and r.bound <= -2.0 + 1e-9


_KNAPSACK = """
import discopt.modeling as dm
from discopt import _rust
n = 40
w = [7 * j % 23 + 11 for j in range(n)]
p = [13 * j % 29 + 17 for j in range(n)]
m = dm.Model("k")
x = [m.binary(f"x{j}") for j in range(n)]
m.subject_to(dm.sum([w[j] * x[j] for j in range(n)]) <= sum(w) / 2)
m.maximize(dm.sum([p[j] * x[j] for j in range(n)]))
r = m.solve(nlp_solver="simplex", threads=1, time_limit=60)
c = dict(_rust.profile_counters_py())
print("RES", r.status, r.objective, r.bound, r.gap_certified, r.algorithm_route.split(" (")[0])
print("CTR", c["NodeNsBound"], c["NodeNsFbbt"], c["NodeNsAbstain"])
"""


def _run_knapsack(flag):
    # The flag is read once per process (OnceLock), so each arm is its own process.
    env = dict(os.environ, DISCOPT_PROFILE="1", DISCOPT_MILP_NS_NODE_BOUND=flag)
    out = subprocess.run(
        [sys.executable, "-c", _KNAPSACK], env=env, capture_output=True, text=True, check=True
    ).stdout
    res = next(ln for ln in out.splitlines() if ln.startswith("RES")).split()
    ctr = [int(v) for v in next(ln for ln in out.splitlines() if ln.startswith("CTR")).split()[1:]]
    return res, ctr


def test_rust_driver_node_gate_runs_and_opts_out():
    on, on_ctr = _run_knapsack("1")
    off, off_ctr = _run_knapsack("0")
    for res in (on, off):
        assert res[1:3] == ["optimal", "817.0"] and res[4] == "True"
        assert " ".join(res[5:]) == "native-milp: discopt monolithic Rust MILP engine"
    # The NS bound sits at or below the vertex value (here: maximize, so above).
    assert 817.0 <= float(on[3]) <= 817.0 + 1e-6
    assert on_ctr[0] > 0 and on_ctr[2] == 0  # the gate ran, and certified every node
    assert off_ctr == [0, 0, 0]  # `=0` is a real opt-out


@pytest.mark.smoke
@pytest.mark.parametrize("gap_tolerance", [None, 1e-7, 1e-4])
def test_interior_point_gap_is_held_to_the_solve_gap_tolerance(gap_tolerance):
    # An interior-point stop near the optimum: (0.99999, 0) has value 2.00001, and the
    # exact dual proves 2. The 5e-6 gap is above the HiGHS route's CERT tolerance, so
    # it certifies only under a solve gap_tolerance that covers it, and then reports
    # the NS bound and the true gap, never bound == objective.
    eng, _ = _engine([0.99999, 0.0], -0.99999, [-1.0])
    r = _solve_lp_matrix(
        _model(), time.perf_counter(), 10.0, eng, "fake", gap_tolerance=gap_tolerance
    )
    assert r.objective == pytest.approx(2.00001)
    assert r.bound == pytest.approx(2.0, abs=1e-12)
    if gap_tolerance is not None and gap_tolerance >= 1e-4:
        assert r.status == "optimal" and r.gap_certified is True
        assert r.gap == pytest.approx(1e-5 / 2.00001, rel=1e-6)
    else:
        assert r.status == "feasible" and r.gap_certified is False
