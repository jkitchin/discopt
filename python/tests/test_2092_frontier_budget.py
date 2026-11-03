"""#2092: the open branch-and-bound frontier has a memory budget.

A tree whose gap never closes keeps every frontier node live, so without a budget
its memory grows for as long as the clock allows (an external benchmark reported
3--8 GiB on 1-hour runs). ``DISCOPT_FRONTIER_MEMORY_MB`` (default 2048, ``0`` = no
budget) is turned into an open-node cap for the native spatial kernel and for the
Python-tree loops; reaching it ends the search like ``max_nodes`` does. These
tests pin the sizing and both exits, and that a capped exit keeps a valid bound
and claims no certificate.
"""

import logging

import discopt.modeling as dm
import discopt.solver as solver
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _bilinear_model(n=20, seed=1):
    """A nonconvex MIQCQP that needs a real tree (~50 kernel nodes at n=20)."""
    rng = np.random.default_rng(seed)
    m = dm.Model("frontier")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n)]
    y = [m.binary(f"y{i}") for i in range(n)]
    for k in range(n // 2):
        i, j, k3 = rng.choice(n, 3, replace=False)
        m.subject_to(x[i] * x[j] - 3.0 * x[k3] <= float(rng.uniform(5, 20)), name=f"bil{k}")
    for i in range(n):
        m.subject_to(x[i] <= 10.0 * y[i], name=f"bigm{i}")
    m.subject_to(sum(x) >= 0.4 * 10 * n, name="demand")
    m.subject_to(sum(y) <= int(0.7 * n), name="card")
    obj = sum(float(rng.uniform(1, 5)) * y[i] for i in range(n))
    for _ in range(n):
        i, j = rng.choice(n, 2, replace=False)
        obj = obj - float(rng.uniform(0.1, 1.0)) * x[i] * x[j]
    m.minimize(obj)
    return m


class TestBudgetSizing:
    def test_default_is_2_gib(self, monkeypatch):
        monkeypatch.delenv("DISCOPT_FRONTIER_MEMORY_MB", raising=False)
        assert solver._frontier_memory_bytes() == 2 << 30

    def test_zero_means_no_budget(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_FRONTIER_MEMORY_MB", "0")
        assert solver._frontier_memory_bytes() is None
        assert solver._tree_max_open_nodes(100) is None
        spec = {"n_cols": 10, "fixed_row_ptr": [0, 1], "term_kind": [0]}
        assert solver._kernel_max_open_nodes(spec) is None

    @pytest.mark.parametrize("raw", ["-1", "lots", "1.5"])
    def test_a_bad_value_is_refused_not_ignored(self, monkeypatch, raw):
        monkeypatch.setenv("DISCOPT_FRONTIER_MEMORY_MB", raw)
        with pytest.raises(ValueError, match="DISCOPT_FRONTIER_MEMORY_MB"):
            solver._frontier_memory_bytes()

    def test_caps_follow_the_per_node_estimates(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_FRONTIER_MEMORY_MB", "100")
        budget = 100 << 20
        # kernel: 9 n + 3 m + 256, m = fixed rows + 4 per lifted term
        spec = {"n_cols": 523, "fixed_row_ptr": list(range(301)), "term_kind": [0] * 200}
        m = 300 + 4 * 200
        assert solver._kernel_max_open_nodes(spec) == budget // (9 * 523 + 3 * m + 256)
        # Python tree: 16 n + 256
        assert solver._tree_max_open_nodes(300) == budget // (16 * 300 + 256)
        # never below 1000 nodes, so a tiny budget cannot stop a solve at the root
        monkeypatch.setenv("DISCOPT_FRONTIER_MEMORY_MB", "1")
        assert solver._tree_max_open_nodes(10**6) == 1000


def _optimum():
    res = _bilinear_model().solve(time_limit=120)
    assert res.status == "optimal", res.status
    return float(res.objective)


def test_kernel_stops_on_the_frontier_cap_with_a_valid_bound(monkeypatch):
    opt = _optimum()
    monkeypatch.setattr(solver, "_kernel_max_open_nodes", lambda spec: 2)
    res = _bilinear_model().solve(time_limit=120)
    assert res.algorithm_route.startswith("native-spatial"), res.algorithm_route
    assert res.solver_stats["tree/frontier_capped"] == 1.0
    assert res.status != "optimal" and not res.gap_certified
    assert res.bound is None or res.bound <= opt + 1e-6, (res.bound, opt)


def test_kernel_cap_absent_by_default(monkeypatch):
    monkeypatch.delenv("DISCOPT_FRONTIER_MEMORY_MB", raising=False)
    res = _bilinear_model().solve(time_limit=120)
    assert res.algorithm_route.startswith("native-spatial")
    assert res.solver_stats["tree/frontier_capped"] == 0.0
    assert res.status == "optimal"


def test_python_tree_stops_on_the_frontier_cap_with_a_valid_bound(monkeypatch, caplog):
    opt = _optimum()
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    monkeypatch.setattr(solver, "_tree_max_open_nodes", lambda n_vars: 3)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        res = _bilinear_model().solve(time_limit=120)
    assert "Python tree" in res.algorithm_route, res.algorithm_route
    assert res.termination == "node_limit", res.termination
    assert any("frontier reached its memory budget" in r.getMessage() for r in caplog.records)
    assert res.status != "optimal" and not res.gap_certified
    assert res.bound is None or res.bound <= opt + 1e-6, (res.bound, opt)
