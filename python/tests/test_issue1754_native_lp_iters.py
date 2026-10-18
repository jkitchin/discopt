"""#1754: the native (Rust simplex) LP route reports ``lp/iters`` like the HiGHS route."""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _lp():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, (40, 15))
    b = rng.uniform(1, 5, 40)
    c = rng.uniform(1, 2, 15)
    m = dm.Model("lp")
    x = m.continuous("x", shape=(15,), lb=0, ub=10)
    m.subject_to(a @ x >= b)
    m.minimize(c @ x)
    return m


def test_native_route_reports_iterations():
    native = _lp().solve(milp_backend="native")
    highs = _lp().solve()
    assert native.algorithm_route.startswith("native-lp")
    assert native.solver_stats["lp/iters"] > 0
    assert highs.solver_stats["lp/iters"] > 0
    assert native.objective == pytest.approx(highs.objective, rel=1e-9)
