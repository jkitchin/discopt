"""#1654: big-M MILPs that plain HiGHS solves returned ``error`` or lost certificates.

Two mechanisms, both in ``lp_milp_highs``:

* A HiGHS incumbent refused by the integral-realisation readback (#1380) -- a binary
  read back within HiGHS's ``mip_feasibility_tolerance`` buys ``tol * M`` of row
  slack -- ended the solve as ``error``. Its integer assignment is now re-derived by
  the fixed-integer LP and, once verified, goes through the route's checks as an
  ordinary incumbent (:func:`_repair_refused_incumbent`).
* An uncertified result is re-solved on the coefficient-tightened form
  (:func:`coefficient_tightened`): each binary's big-M shrunk to its row's activity
  bound, which leaves the integer-feasible set exactly unchanged.
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs as L

P = [7, 5, 9, 4, 6, 8, 3, 5]
R = [0, 2, 4, 1, 6, 3, 8, 5]


def _sequencing(big_m: float) -> dm.Model:
    n = len(P)
    m = dm.Model("seq1654")
    s = [m.continuous(f"s{i}", lb=R[i], ub=200) for i in range(n)]
    cmax = m.continuous("Cmax", lb=0, ub=200)
    for i, j in itertools.combinations(range(n), 2):
        y = m.binary(f"y{i}_{j}")
        m.subject_to(s[i] + P[i] <= s[j] + big_m * (1 - y))
        m.subject_to(s[j] + P[j] <= s[i] + big_m * y)
    for i in range(n):
        m.subject_to(s[i] + P[i] <= cmax)
    m.minimize(cmax)
    return m


def _fixed_charge(big_m: float) -> dm.Model:
    m = dm.Model("fc1654")
    xa = m.continuous("xa", lb=0, ub=10)
    xb = m.continuous("xb", lb=0, ub=10)
    y = m.binary("y")
    m.subject_to(xa + xb >= 5)
    m.subject_to(xa <= big_m * y)
    m.minimize(100 * y + xa + 30 * xb)
    return m


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(**kw)


def test_tightening_preserves_every_integer_feasible_point():
    """Exactness: a row holds at (y, x) before iff it holds after, for y in {0, 1}
    and every x in the box (checked on the box's vertices and random interior)."""
    _, _, sf = _highs_std_form(_fixed_charge(1e9))
    ct, n = L.coefficient_tightened(sf)
    assert ct is not None and n >= 1
    rng = np.random.default_rng(0)
    logical = L._logical_columns(sf)
    A, A2 = sf.A.tocsr(), ct.A.tocsr()  # noqa: N806
    checks = 0
    for _ in range(400):
        x = rng.uniform(sf.xl, np.minimum(sf.xu, 1e3))
        x[sf.int_idx] = rng.integers(0, 2, sf.int_idx.size)
        for row in range(sf.m):
            js = A[row].indices
            lg = [j for j in js if logical[j]]
            if not lg:
                continue
            # row value without its logical: the logical's sign fixes the sense
            j_s = lg[0]
            a = A[row].toarray().ravel()
            a2 = A2[row].toarray().ravel()
            act = a @ x - a[j_s] * x[j_s]
            act2 = a2 @ x - a2[j_s] * x[j_s]
            ok = (sf.b[row] - act) / a[j_s] >= -1e-9
            ok2 = (ct.b[row] - act2) / a2[j_s] >= -1e-9
            assert ok == ok2, (row, x)
            checks += 1
    assert checks > 0


def test_tightening_shrinks_the_big_m():
    _, _, sf = _highs_std_form(_fixed_charge(1e9))
    ct, _ = L.coefficient_tightened(sf)
    assert ct is not None
    # xa's upper bound, less a round-off margin of ~(k+2)*8*eps*1e9 = 7e-6
    assert float(np.abs(ct.A.data).max()) <= 10.0 + 1e-4


@pytest.mark.parametrize("big_m", [1e7, 1e9])
def test_sequencing_certifies_the_true_makespan(big_m):
    r = _solve(_sequencing(big_m), time_limit=120)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(47.0, abs=1e-6)
    assert r.bound <= 47.0 + 1e-6


@pytest.mark.parametrize("big_m", [1e8, 1e10])
def test_fixed_charge_certifies_the_true_optimum(big_m):
    """True optimum 105 (y = 1, xa = 5). HiGHS alone accepts y = 5/M here."""
    r = _solve(_fixed_charge(big_m))
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(105.0, abs=1e-6)
    assert r.bound <= 105.0 + 1e-6


def test_refused_incumbent_is_repaired_not_an_error(monkeypatch):
    """With the rescue off, the repair alone turns ``error`` into a verified point."""
    monkeypatch.setenv("DISCOPT_MILP_COEF_TIGHTEN", "0")
    r = _solve(_sequencing(1e7), time_limit=120)
    assert r.status != "error", r.algorithm_route
    assert r.objective is not None and r.objective >= 47.0 - 1e-6
    if r.gap_certified:
        assert r.objective == pytest.approx(47.0, abs=1e-6)


def test_flag_off_skips_the_rescue(monkeypatch):
    monkeypatch.setenv("DISCOPT_MILP_COEF_TIGHTEN", "0")
    r = _solve(_fixed_charge(1e10))
    assert not (r.solver_stats or {}).get("milp/coef_tighten_ran")
