"""#1775 G6.2: under ``deterministic``, the GDP-config wave still stops at ``time_limit``.

#1187 freed the wave of ``one_hot_config_subnlp`` from the caller's slice under
``deterministic`` (its extent became its own 48-solve cap) but freed it from the
solve's own time limit too: ``wave_deadline`` was ``None``, so no plan was ever
refused and no sub-NLP got a wall budget. Measured on GDPlib ``hda`` (big-M,
``time_limit=100``, ``deterministic=True``): the solve ran 526 s, and stack samples
every 20 s from t = 101 s on were all inside the wave's ``subnlp``. #1371 settled the
rule: under ``deterministic`` a role-2 budget goes inert but the role-1 deadline
still applies. The wave now takes the role-1 deadline there.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver_tuning
from discopt._relax import primal_heuristics as PH
from discopt.solver import _ROLE1_DEADLINE
from discopt.solver_tuning import SolverTuning
from discopt.solvers import NLPResult, SolveStatus

pytestmark = pytest.mark.smoke

SLEEP = 0.15


def _many_plans():
    """Four 3-way ``sum_k y_k == 1`` groups (the big-M lowering's shape): 81
    configurations, so the wave has 48 plans."""
    m = dm.Model("many_plans")
    x = [m.continuous(f"x{i}", lb=0, ub=4) for i in range(4)]
    for i in range(4):
        y = [m.binary(f"y{i}_{k}") for k in range(3)]
        m.subject_to(y[0] + y[1] + y[2] == 1)
        m.subject_to(x[i] <= 1 + 3 * (1 - y[0]))
        m.subject_to(x[i] >= 2 - 2 * (1 - y[1]))
        m.subject_to(x[i] >= 3.5 - 3.5 * (1 - y[2]))
    m.minimize(sum((xi - 2.2) ** 2 for xi in x))
    return m


def _slow_backend(evaluator, x0, options=None):
    time.sleep(SLEEP)
    return NLPResult(status=SolveStatus.INFEASIBLE, x=None, objective=None)


def _run(deterministic: bool, role1_in: float, deadline_in: float | None):
    m = _many_plans()
    n = sum(v.size for v in m._variables)
    token = solver_tuning.enter_scope(SolverTuning(deterministic=deterministic))
    role1 = _ROLE1_DEADLINE.set(time.perf_counter() + role1_in)
    stats: dict = {}
    try:
        t0 = time.perf_counter()
        PH.one_hot_config_subnlp(
            m,
            np.full(n, 0.5),
            backend=_slow_backend,
            deadline=None if deadline_in is None else time.perf_counter() + deadline_in,
            stats=stats,
        )
        return time.perf_counter() - t0, stats["attempted"]
    finally:
        _ROLE1_DEADLINE.reset(role1)
        solver_tuning.reset_current(token)


def test_deterministic_wave_stops_at_the_role1_deadline():
    wall, attempted = _run(deterministic=True, role1_in=0.6, deadline_in=0.6)
    assert attempted >= 1  # the wave ran (rule 6)
    assert wall < 0.6 + 3 * SLEEP, (wall, attempted)  # 48 plans would be ~7 s


def test_deterministic_wave_ignores_the_shorter_role2_slice():
    """#1187 stands: the caller's slice does not set the wave's extent; only the
    solve's own deadline does."""
    _, short = _run(deterministic=True, role1_in=60.0, deadline_in=0.3)
    assert short > int(0.3 / SLEEP) + 2  # ran past the slice


def test_default_mode_keeps_the_caller_deadline():
    wall, attempted = _run(deterministic=False, role1_in=60.0, deadline_in=0.6)
    assert attempted >= 1
    assert wall < 0.6 + 3 * SLEEP
