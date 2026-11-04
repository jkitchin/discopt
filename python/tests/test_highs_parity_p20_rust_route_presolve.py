"""highs-parity P2.0 (i): the Rust pure LP/MILP route skips root presolve, as HiGHS's does.

P2.1 found discopt's root presolve the largest single piece of the Rust LP route's wall (up
to 4.8 s on an LP raw HiGHS solves in 13 ms); the HiGHS route has skipped it since #1229.
Skipping it on the Rust route left all 65 L/M/INF panel instances' status, certificate and
objective identical (P2.0). The declared box stays valid, merely looser.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import pytest


def _lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.subject_to(x[0] + 2 * x[1] - x[2] <= 4)
    m.subject_to(x[0] + x[1] + x[2] >= 2)
    m.minimize(3 * x[0] + x[1] + 2 * x[2])
    return m


@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_pure_route_skips_root_presolve(backend):
    assert S._highs_takes_pure_lp_milp(
        _lp(), solver_name=None, nlp_bb=None, nlp_solver="pounce", lagrangian_bound=False,
        has_callbacks=False, backend=backend,
    )  # fmt: skip


def test_rust_lp_solve_never_runs_root_presolve(monkeypatch):
    import discopt._relax.presolve_pipeline as pp

    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    calls = []
    real = pp.run_root_presolve
    monkeypatch.setattr(pp, "run_root_presolve", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = _lp().solve(time_limit=20)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(2.0, abs=1e-6)  # x1 = 2
    assert calls == []
