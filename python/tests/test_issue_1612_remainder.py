"""#1612 remainder (X-30d): a HiGHS MILP certificate closed only by HiGHS's feasibility
tolerance is re-solved at a tighter tolerance instead of being withdrawn downstream.

C-06 (set cover) and E-03 (knapsack) are covered by
``test_1612_highs_root_dual_ns_gap.py`` (PR #1636). X-30d is a different mechanism:
HiGHS's incumbent violates a row by its 1e-6 ``mip_feasibility_tolerance``, so its
objective -- and the bound it stops on -- sit 1e-6 beyond the true optimum 41. The
post-solve incumbent repair moves the objective back to 41, the published gap is then
``9.99999997e-7`` against ``abs_gap = 1e-6``, and the #1551 float-resolution guard
(``1.4e-14``) withdrew the certificate and dropped the bound.
"""

from __future__ import annotations

import dataclasses

import discopt.modeling as dm
import pytest
from discopt.solvers import lp_milp_highs as lmh


@pytest.fixture(autouse=True)
def _highs_milp_route(monkeypatch):
    """These tests pin the HiGHS MILP route's own mechanics. Since 2026-11-09 the
    pure-MILP default is the Rust engine on every install (PR #2196), so they name
    HiGHS explicitly, the optional ``discopt[highs]`` extra."""
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")


pytest.importorskip("highspy")

TRUE_OPT = 41.0


def _interdiction(k: int = 3) -> dm.Model:
    arcs = [
        (0, 1, 2), (1, 2, 2), (2, 3, 2), (4, 5, 5), (5, 6, 5), (6, 7, 5),
        (0, 4, 1), (4, 0, 1), (1, 5, 1), (5, 1, 1), (2, 6, 1), (6, 2, 1), (3, 7, 1), (7, 3, 1),
    ]  # fmt: skip
    n_arcs, n_nodes, s, t, delay = len(arcs), 8, 0, 3, 30.0
    m = dm.Model("interdiction")
    x = m.binary("x", shape=(n_arcs,))
    u = m.continuous("u", shape=(n_nodes,), lb=0, ub=1000.0)
    m.maximize(u[t] - u[s])
    m.subject_to(u[s] == 0)
    m.subject_to(dm.sum(lambda a: x[a], over=range(n_arcs)) <= k)
    for a, (i, j, ca) in enumerate(arcs):
        m.subject_to(u[j] - u[i] <= float(ca) + delay * x[a])
    return m


def test_x30d_tight_gap_certifies_with_a_valid_bound():
    r = _interdiction().solve(gap_tolerance=1e-9)
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective == pytest.approx(TRUE_OPT, abs=1e-9)
    # MAXIMIZE: the bound is an upper bound -- never below the true optimum.
    assert r.bound is not None
    assert TRUE_OPT <= r.bound <= TRUE_OPT + 1e-6
    assert r.solver_stats.get("milp/feas_artefact_tight_adopted") == 1.0
    assert "certificate/objective_unresolved" not in r.solver_stats


def test_x30d_default_gap_does_not_trigger_the_resolve():
    r = _interdiction().solve()
    assert r.status == "optimal" and r.gap_certified is True
    assert r.objective == pytest.approx(TRUE_OPT, abs=1e-9)
    assert r.bound is not None and r.bound >= TRUE_OPT
    # The relative arm closes with room to spare: no second solve is paid for.
    assert "milp/feas_artefact_tight_ran" not in r.solver_stats


@pytest.mark.parametrize("shift", [-1.0, +0.5], ids=["disagrees", "refuted"])
def test_untrustworthy_tight_resolve_is_not_adopted(monkeypatch, shift):
    """A tight re-solve whose bound disagrees with the primary's (beyond the #1640
    slack) or lies above a verified point is never published."""
    real = lmh._certified_milp
    calls = {"n": 0}

    def fake(sf, kw):
        out = real(sf, kw)
        calls["n"] += 1
        if kw.get("feasibility_tolerance") == lmh.CROSS_TIGHT_FEAS_TOL:
            assert out.bound is not None
            return dataclasses.replace(out, bound=out.bound + shift)
        return out

    monkeypatch.setattr(lmh, "_certified_milp", fake)
    r = _interdiction().solve(gap_tolerance=1e-9)
    assert calls["n"] == 2, "the tight re-solve did not run"
    assert r.solver_stats.get("milp/feas_artefact_tight_adopted") is None
    # Whatever is published, the corrupted bound (internal min sense -> -bound) is not.
    if r.bound is not None:
        assert r.bound >= TRUE_OPT
        assert abs(-r.bound - (-TRUE_OPT + shift)) > 1e-3
