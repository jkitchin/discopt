"""The Rust lazy master stops on the LP/NLP-BB loop's own gap criterion (#1352 class).

The engine's default stopping test is ``(inc - lb) / max(|inc|, 1) <= gap_tol``.
Below unit objective scale that is an absolute window, so the lp_nlp_bb master
could exit ``optimal`` with a relative gap the loop then refused: measured on the
n=20 perspective portfolio (optimum 0.010442), it stopped 1.6e-5 short (0.155%)
and the route reported ``feasible``. ``milp_simplex._lazy_gap_kwargs`` selects the
engine's ``max(|inc|, |lb|, 1e-10)`` denominator plus the loop's absolute arm.
The end-to-end probe is
``test_1658_pounce_tol_and_oa_master::test_issue_portfolio_certifies_on_the_route``.
"""

import discopt._rust as rust
import numpy as np
import pytest
from discopt.solvers._gap import GAP_ABS_TOL
from discopt.solvers.milp_simplex import solve_milp_with_lazy_cuts


def _knapsack():
    rng = np.random.default_rng(0)
    n = 12
    v = rng.integers(10, 100, n).astype(float)
    w = rng.integers(10, 100, n).astype(float)
    return dict(
        c=-1e-3 * v,
        A_ub=w[None, :],
        b_ub=np.array([w.sum() / 3]),
        bounds=[(0, 1)] * n,
        integrality=np.ones(n, int),
        lazy_callback=lambda x: [],
        time_limit=30,
    )


def _captured_kwargs(monkeypatch):
    seen: list[dict] = []
    real = rust.solve_milp_lazy_csc_py

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **k)

    monkeypatch.setattr(rust, "solve_milp_lazy_csc_py", spy)
    r = solve_milp_with_lazy_cuts(**_knapsack())
    assert len(seen) == 1, "the lazy binding was not called"
    return r, seen[0]


@pytest.mark.smoke
def test_the_lazy_master_gets_the_loops_gap_criterion(monkeypatch):
    monkeypatch.delenv("DISCOPT_OA_MASTER_GAP_SCALED", raising=False)
    r, kw = _captured_kwargs(monkeypatch)
    assert kw.get("rel_gap_max_abs") is True
    assert kw.get("abs_gap_tol") == GAP_ABS_TOL
    assert r.status.value == "optimal"


def test_the_opt_out_restores_the_legacy_test(monkeypatch):
    monkeypatch.setenv("DISCOPT_OA_MASTER_GAP_SCALED", "0")
    r, kw = _captured_kwargs(monkeypatch)
    assert "rel_gap_max_abs" not in kw and "abs_gap_tol" not in kw
    assert r.status.value == "optimal"
