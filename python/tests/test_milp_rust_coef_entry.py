"""Python-MILP-tree retirement: the withdrawn-certificate case (#2163) on the Rust route.

``min -x + 3z  s.t.  x <= M z,  x in [0, 10],  z binary`` has optimum -7 for every
``M >= 10`` (#1380, #1414). From ``M ~ 1e11`` the Rust driver's #1296 gate withdraws its
certificate (the LP layer cannot represent ``x``'s coefficient next to ``M``), and
``_solve_milp_simplex`` used to hand the MILP to the Python MILP tree (reason
``withdrawn_bound_2163``). ``DISCOPT_MILP_RUST_COEF_ENTRY`` applies the graduated #1654
coefficient tightening (``x <= 10 z``) to the engine's standard form before the search;
the point still passes the original rows. (It replaced ``DISCOPT_MILP_RUST_COEF_RESCUE``,
which re-solved the tightened form only after an uncertified search: 5/9 vs 9/9 on the
#1654 families' entry probe.)
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest


def _bigm(M: float) -> dm.Model:  # noqa: N803
    m = dm.Model("bigm")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.subject_to(x <= M * z)
    m.minimize(-x + 3 * z)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("M", [1e11, 1e12, 1e15])
def test_big_m_is_certified_on_the_rust_route_with_entry_tightening(monkeypatch, M):  # noqa: N803
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_MILP_RUST_COEF_ENTRY", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _bigm(M).solve(time_limit=30)
    st = r.solver_stats or {}
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(-7.0, abs=1e-9)
    assert r.bound <= -7.0 + 1e-6
    assert r.algorithm_route.startswith("native-milp"), r.algorithm_route
    assert st.get("milp/rust_coef_entry_rewritten", 0) >= 1, st  # the row was rewritten (§6)
    assert "route/python_milp_tree_entry" not in st


@pytest.mark.smoke
def test_without_the_flag_the_engine_reports_it_uncertified(monkeypatch):
    """With the flag off the engine's certificate is withdrawn (#2163) and, since the
    Python MILP tree was retired, that honest uncertified answer is what is returned:
    no bound, never a false certificate."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_MILP_RUST_COEF_ENTRY", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _bigm(1e12).solve(time_limit=30)
    assert r.algorithm_route.startswith("native-milp"), r.algorithm_route
    assert r.status == "feasible" and not r.gap_certified, r.status
    assert r.bound is None
    assert r.objective >= -7.0 - 1e-9  # a feasible point, never below the optimum
