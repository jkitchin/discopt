"""Python-MILP-tree retirement: the withdrawn-certificate case (#2163) on the Rust route.

``min -x + 3z  s.t.  x <= M z,  x in [0, 10],  z binary`` has optimum -7 for every
``M >= 10`` (#1380, #1414). From ``M ~ 1e11`` the Rust driver's #1296 gate withdraws its
certificate (the LP layer cannot represent ``x``'s coefficient next to ``M``), and
``_solve_milp_simplex`` used to hand the MILP to the Python MILP tree (reason
``withdrawn_bound_2163``). ``DISCOPT_MILP_RUST_COEF_RESCUE`` instead re-solves the
graduated #1654 coefficient-tightened form (``x <= 10 z``) on the same driver and adopts
its certificate after the point passes the original rows.
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
def test_withdrawn_certificate_is_rescued_on_the_rust_route(monkeypatch, M):  # noqa: N803
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_MILP_RUST_COEF_RESCUE", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _bigm(M).solve(time_limit=30)
    st = r.solver_stats or {}
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(-7.0, abs=1e-9)
    assert r.bound <= -7.0 + 1e-6
    assert r.algorithm_route.startswith("native-milp"), r.algorithm_route
    assert st.get("milp/rust_coef_rescue") == 1.0, st  # the rescue decided it (§6)
    assert "route/python_milp_tree_entry" not in st


@pytest.mark.smoke
def test_without_the_flag_the_case_still_reaches_the_tree(monkeypatch):
    """The legacy path, until the flag graduates: the deferral is counted."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_MILP_RUST_COEF_RESCUE", "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _bigm(1e12).solve(time_limit=30)
    st = r.solver_stats or {}
    assert r.status == "optimal" and r.objective == pytest.approx(-7.0, abs=1e-9)
    assert st.get("route/python_milp_tree_entry") == 10.0, st  # withdrawn_bound_2163
