"""#1678 (c): big-M MILPs whose continuous columns sit at the default box.

Facility location with ``x`` left at the default ``9.999e19`` upper bound returned
``error`` at M = 1e10 and 1e14. HiGHS accepted every ``y`` at about 2e-9: with
M = 1e10 that is 20 units of row slack, so every facility read as closed while still
serving demand. The point fails discopt's readback, and the fixed-integer repair
(all ``y = 0``) is infeasible. The #1654 coefficient tightening, the rescue for this
class, skipped every big-M row, because ``x`` had no finite box to bound the row's
activity.

``DISCOPT_MILP_IMPLIED_BOUNDS`` (default ON; ``=0`` opts out) first gives the open
columns the finite sides the rows imply (outward-rounded FBBT). ``x_ij <= d_j``
follows from the demand equality, and the tightening rewrites ``M`` down to ``d_j``.

The tests here are the §5 bound-changing checks on generated families:

* differential bound: the tightened LP relaxation bound is >= the untightened one
  and <= the true MILP optimum, found by enumerating the binaries;
* feasible-point sampling: every sampled integer-feasible point of the model is
  feasible for the tightened form, and the reverse also holds.
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs as L
from scipy.optimize import linprog

F = [100, 120, 90]
D = [20, 30, 25, 15]
C = [[2, 4, 5, 3], [3, 1, 3, 4], [4, 3, 2, 2]]


def _issue_model(big_m: float) -> dm.Model:
    m = dm.Model("fl1678")
    y = [m.binary(f"y{i}") for i in range(3)]
    x = [[m.continuous(f"x{i}_{j}", lb=0) for j in range(4)] for i in range(3)]
    for j in range(4):
        m.subject_to(sum(x[i][j] for i in range(3)) == D[j])
    for i in range(3):
        for j in range(4):
            m.subject_to(x[i][j] <= big_m * y[i])
    m.minimize(
        sum(F[i] * y[i] for i in range(3))
        + sum(C[i][j] * x[i][j] for i in range(3) for j in range(4))
    )
    return m


def _family(seed: int, big_m: float) -> dm.Model:
    """A generated facility-location model with open ``x`` boxes. By seed it has
    demand equalities or ``>=`` rows, with or without capacities. With ``>=`` and no
    capacity, ``x`` has no implied upper bound."""
    rng = np.random.default_rng(seed)
    nf, nc = int(rng.integers(2, 4)), int(rng.integers(2, 5))
    eq, capped = bool(seed % 2), bool((seed // 2) % 2)
    dem = rng.uniform(1, 9, nc).round(3)
    cap = rng.uniform(dem.sum() / 2, dem.sum(), nf).round(3)
    m = dm.Model(f"fam{seed}")
    y = [m.binary(f"y{i}") for i in range(nf)]
    x = [[m.continuous(f"x{i}_{c}", lb=0) for c in range(nc)] for i in range(nf)]
    for c in range(nc):
        col = sum(x[i][c] for i in range(nf))
        m.subject_to(col == float(dem[c]) if eq else col >= float(dem[c]))
    for i in range(nf):
        if capped:
            m.subject_to(sum(x[i]) <= float(cap[i]))
        for c in range(nc):
            m.subject_to(x[i][c] <= big_m * y[i])
    cost = rng.uniform(1, 10, (nf, nc))
    m.minimize(
        sum(float(rng.uniform(5, 40)) * y[i] for i in range(nf))
        + sum(float(cost[i, c]) * x[i][c] for i in range(nf) for c in range(nc))
    )
    return m


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(**kw)


def _lp(sf, xl=None, xu=None, c=None):
    xl = sf.xl if xl is None else xl
    xu = sf.xu if xu is None else xu
    bounds = [
        (None if lo <= -L.READBACK_LIMIT else lo, None if hi >= L.READBACK_LIMIT else hi)
        for lo, hi in zip(xl, xu)
    ]
    return linprog(
        sf.c if c is None else c, A_eq=sf.A.toarray(), b_eq=sf.b, bounds=bounds, method="highs"
    )


def _fixed(sf, assign):
    xl, xu = sf.xl.copy(), sf.xu.copy()
    xl[sf.int_idx] = assign
    xu[sf.int_idx] = assign
    return xl, xu


def _row_ok(sf, x, tol=1e-6):
    """``x`` satisfies ``sf``'s rows and box once its logical (slack) columns are
    re-derived from their rows. The tightening changes a row's slack value, so only
    the structural part of a point carries over between the two forms."""
    x = np.array(x, dtype=np.float64)
    A = sf.A.tocsc()  # noqa: N806
    rows = sf.A.tocsr()
    logical = L._logical_columns(sf)
    for j in np.flatnonzero(logical):
        i, a = int(A.indices[A.indptr[j]]), float(A.data[A.indptr[j]])
        x[j] = 0.0
        x[j] = (sf.b[i] - float((rows[i] @ x)[0])) / a
    resid = np.abs(sf.A @ x - sf.b)
    scale = np.abs(sf.A) @ np.abs(x) + np.abs(sf.b) + 1.0
    return bool(np.all(resid <= tol * scale)) and bool(
        np.all(x >= sf.xl - tol * (1 + np.abs(sf.xl)))
        and np.all(x <= sf.xu + tol * (1 + np.abs(sf.xu)))
    )


def test_flag_off_leaves_the_open_box_rows_alone():
    """The pre-#1678 state: no row is rewritten, because ``x`` has no finite box."""
    _, _, sf = _highs_std_form(_issue_model(1e10))
    assert L.coefficient_tightened(sf) == (None, 0)


def test_default_fbbt_box_still_treats_the_default_box_as_declared():
    """Bound-neutral for the certificate callers: with the default ``open_limit`` a
    declared ``9.999e19`` side is not replaced."""
    _, _, sf = _highs_std_form(_issue_model(1e10))
    box = L.fbbt_box(sf)
    big = sf.xu >= L.READBACK_LIMIT
    assert big.any()
    assert np.array_equal(box.xu[big & (sf.xu < L.INF)], sf.xu[big & (sf.xu < L.INF)])


def test_implied_bounds_shrink_the_big_m_to_the_demand():
    _, _, sf = _highs_std_form(_issue_model(1e14))
    ct, n = L.coefficient_tightened(sf, implied_bounds=True)
    assert ct is not None and n == 12
    # Each x_ij gets the implied side d_j, widened outward by a round-off margin only.
    for k in range(12):
        j = 3 + k
        assert D[k % 4] <= ct.xu[j] <= D[k % 4] * (1 + 1e-9) + 1e-9
    # M = 1e14 shrinks to d_j plus the #1654 round-off margin, which scales with the
    # coefficient being removed: 8 (k + 2) eps 1e14 ~ 0.7.
    assert float(np.abs(ct.A.data).max()) <= max(D) + 1.0


@pytest.mark.parametrize("big_m", [1e10, 1e14])
def test_issue_repro_certifies_by_default(big_m):
    r = _solve(_issue_model(big_m), time_limit=60)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(340.0, abs=1e-6)
    assert r.bound <= 340.0 + 1e-6


def test_opt_out_restores_the_pre_1678_route(monkeypatch):
    """``=0`` keeps the legacy path reachable: no row is rewritten and the issue's
    model is not certified (it returned ``error`` on highspy 1.12)."""
    monkeypatch.setenv("DISCOPT_MILP_IMPLIED_BOUNDS", "0")
    r = _solve(_issue_model(1e10), time_limit=60)
    assert not r.gap_certified
    assert not (r.solver_stats or {}).get("milp/coef_tightened_entries")


@pytest.mark.parametrize("seed", range(16))
def test_differential_bound_and_feasible_point_sampling(seed):
    """§5 on a generated family, at a big-M where the LP is still well conditioned."""
    big_m = 1e3
    _, _, sf = _highs_std_form(_family(seed, big_m))
    ct, n = L.coefficient_tightened(sf, implied_bounds=True)
    if ct is None:
        # Only the uncapped ">=" variant may have no implied bound.
        assert seed % 4 == 0, seed
        return
    rng = np.random.default_rng(100 + seed)
    nb = sf.int_idx.size
    opt = np.inf
    checks = 0
    for assign in itertools.product((0.0, 1.0), repeat=nb):
        assign = np.array(assign)
        res = _lp(sf, *_fixed(sf, assign))
        res_ct = _lp(ct, *_fixed(ct, assign))
        # The same integer assignment is feasible before iff after, at the same value.
        assert res.status == res_ct.status, (assign, res.status, res_ct.status)
        checks += 1
        if res.status != 0:
            continue
        assert res.fun == pytest.approx(res_ct.fun, rel=1e-7, abs=1e-6)
        opt = min(opt, res.fun)
        # Feasible-point sampling: random-objective vertices of each side lie in the other.
        for _ in range(6):
            w = rng.normal(size=sf.n)
            p = _lp(sf, *_fixed(sf, assign), c=w)
            if p.status == 0:
                assert _row_ok(ct, p.x), "a feasible point of the model was cut off"
                checks += 1
            q = _lp(ct, *_fixed(ct, assign), c=w)
            if q.status == 0:
                assert _row_ok(sf, q.x), "the tightened form admitted an infeasible point"
                checks += 1
    assert np.isfinite(opt)
    lp0, lp1 = _lp(sf), _lp(ct)
    assert lp0.status == 0 and lp1.status == 0
    assert lp1.fun >= lp0.fun - 1e-7 * (1 + abs(lp0.fun))  # never looser
    assert lp1.fun <= opt + 1e-6 * (1 + abs(opt))  # never above the true optimum
    checks += 2
    assert checks > 2**nb
