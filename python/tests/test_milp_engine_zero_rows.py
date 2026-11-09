"""Python-MILP-tree retirement step 2: a MILP with no constraints runs on the Rust engine.

It used to defer to the Python MILP tree (`no_constraints`, the only non-stall route the
retirement census found: MIPLIB-38+M's `fixed-binary`). The optimum is closed-form: each
column at the bound its cost sign selects, integer columns rounded inward.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("sense", ["min", "max"])
def test_a_constraint_free_milp_is_solved_by_the_engine(monkeypatch, sense):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m = dm.Model("no_rows")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=-1, ub=4.5)
    z = m.continuous("z", lb=1, ub=2)
    obj = 2 * x - 3 * y + z
    if sense == "min":
        m.minimize(obj)
        want = 2 * 0 - 3 * math.floor(4.5) + 1  # -11
    else:
        m.maximize(obj)
        want = 2 * 5 - 3 * (-1) + 2  # 15
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(want, abs=1e-9)
    assert r.algorithm_route.startswith("native-milp"), r.algorithm_route
    assert "route/python_milp_tree_entry" not in (r.solver_stats or {})
