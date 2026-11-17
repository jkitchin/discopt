"""#2225: an IIS probe on the Rust engine no longer solves the original rows twice.

The #1634 policy accepts a presolve-on Infeasible only when a presolve-free solve
agrees. The Rust MILP driver never trusts its own presolve's infeasibility claim: it
re-decides it on the original rows. Before #2225 that re-decision kept root bound
tightening (a presolve step) on, so the IIS engine still ran its own ``presolve="off"``
solve, a third solve of the same infeasible set. The driver now re-decides a presolve
claim with no presolve at all and says so (``MilpResult::presolve_free``), and the
engine skips its re-solve exactly when that flag is set.

What must hold:
* the flag is set only by a presolve-free search: on a ``presolve=False`` solve, and on a
  ``presolve=True`` solve whose presolve claimed infeasibility (the claim branch fired);
  a presolve-on Infeasible reached any other way still gets the engine's re-solve;
* the IIS is unchanged, and on the HiGHS engine (whose presolve-on verdict comes from
  the presolved model) nothing is skipped.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._iis_milp import milp_iis_arrays
from discopt.solvers.milp_handle import MilpHandle


def _odd_pair():
    """``x + y = 3``, ``x - y = 0`` over integers in [0, 5]: presolve substitutes the
    doubleton and finds ``2x = 3``, an infeasibility claim."""
    A = sp.csr_matrix([[1.0, 1.0], [1.0, -1.0]])
    return A, np.array([3.0, 0.0]), np.array([3.0, 0.0]), np.zeros(2), np.full(2, 5.0)


def test_a_presolve_claim_is_redecided_presolve_free(monkeypatch):
    from discopt import _rust

    A, lo, hi, lb, ub = _odd_pair()
    h = MilpHandle(np.zeros(2), A, lo, hi, lb, ub, np.ones(2, dtype=bool))
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    _rust.profile_reset_py()
    on = h.solve(presolve=True)
    fired = _rust.profile_counters_py()["MilpPresolveClaimRedecided"]
    assert fired >= 1, "presolve made no infeasibility claim: the test exercises nothing"
    assert on.status == "infeasible" and on.presolve_free
    off = h.solve(presolve=False)
    assert off.status == "infeasible" and off.presolve_free


def test_a_presolve_on_verdict_without_a_claim_is_not_presolve_free(monkeypatch):
    """The pigeonhole hole(4): presolve makes no infeasibility claim, so the presolve-on
    search runs with root bound tightening; its Infeasible is not presolve-free."""
    from discopt import _rust

    k, p = 4, 5
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for i in range(p):
        for j in range(k):
            rows.append(r)
            cols.append(i * k + j)
            vals.append(1.0)
        lo.append(1.0)
        hi.append(1e20)
        r += 1
    for j in range(k):
        for i1 in range(p):
            for i2 in range(i1 + 1, p):
                rows += [r, r]
                cols += [i1 * k + j, i2 * k + j]
                vals += [-1.0, -1.0]
                lo.append(-1.0)
                hi.append(1e20)
                r += 1
    n = p * k
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n))
    h = MilpHandle(np.zeros(n), A, np.array(lo), np.array(hi), np.zeros(n), np.ones(n),
                   np.ones(n, dtype=bool))  # fmt: skip
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    _rust.profile_reset_py()
    res = h.solve(presolve=True)
    assert _rust.profile_counters_py()["MilpPresolveClaimRedecided"] == 0
    assert res.status == "infeasible"
    assert not res.presolve_free


def _arrays():
    """The odd pair plus two rows that take no part in the infeasibility."""
    A = sp.csr_matrix([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    lo = np.array([3.0, 0.0, 0.0, -1e20])
    hi = np.array([3.0, 0.0, 1e20, 9.0])
    groups = {i: np.array([i]) for i in range(4)}
    return A, lo, hi, groups, np.zeros(3), np.full(3, 5.0), np.ones(3, dtype=bool)


@pytest.mark.parametrize("confirm", ["each", "final"])
def test_the_rust_engine_skips_only_the_redundant_resolve(confirm):
    res = milp_iis_arrays(_arrays(), iis_engine="rust", confirm=confirm, sat_candidate=False)
    assert sorted(res.constraints) == [0, 1]
    assert res.integrality, "the IIS needs integrality: 2x = 3 has a continuous solution"
    assert res.n_cross_inline >= 1, "no probe used the in-solve presolve-free check"
    if confirm == "each":
        # Every infeasibility claim was either re-solved or checked in-solve.
        assert res.n_cross_checks + res.n_cross_inline >= 1


def test_the_highs_engine_never_skips():
    pytest.importorskip("highspy")
    res = milp_iis_arrays(_arrays(), iis_engine="highs", confirm="each", sat_candidate=False)
    assert sorted(res.constraints) == [0, 1]
    assert res.n_cross_inline == 0
    assert res.n_cross_checks >= 1
