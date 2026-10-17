"""#1731: integer-valued objectives in the spatial B&B tree.

``DISCOPT_SPATIAL_OBJ_LATTICE`` rounded spatial-tree node bounds up onto the
objective's lattice ``c0 + k*g``. It was RETIRED (CLAUDE.md §5): the graduation
panel over the 108 MINLPLib/QPLIB instances its detector accepted was cert-clean
but not net-positive (``docs/dev/flag-retirement-audit.md``). These fixtures
stay: the spatial tree on integer-valued objectives, judged against brute-force
enumeration, plus a soundness pin on the issue's repro, which on main stalls at
a fractional bound rather than certifying.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver_tuning import SolverTuning


def _knapsack(w, cap, *, maximize: bool, offset: float = 0.0, nl: bool = True):
    n = len(w)
    m = dm.Model("ks")
    x = m.binary("x", shape=(n,))
    m.subject_to(np.abs(w) @ x <= cap)
    if maximize:
        m.maximize(w @ x + offset)
    else:
        m.minimize(w @ x + offset)
        m.subject_to(np.abs(w) @ x >= cap / 2)
    if nl:
        m.subject_to(x[0] * x[1] <= 1)  # redundant bilinear row -> spatial B&B
    return m


def _brute(w, cap, *, maximize: bool, offset: float = 0.0):
    best = None
    for bits in itertools.product((0, 1), repeat=len(w)):
        x = np.asarray(bits, dtype=float)
        if np.abs(w) @ x > cap + 1e-9:
            continue
        if not maximize and np.abs(w) @ x < cap / 2 - 1e-9:
            continue
        v = float(w @ x) + offset
        if best is None or (v > best if maximize else v < best):
            best = v
    return best


def _assert_valid_bound(r, opt, *, maximize: bool):
    assert r.bound is not None
    tol = 1e-6 * max(1.0, abs(opt))
    if maximize:
        assert r.bound >= opt - tol  # never below the true maximum
    else:
        assert r.bound <= opt + tol  # never above the true minimum


CASES = []
_rng = np.random.default_rng(1731)
for _k in range(6):
    _w = _rng.integers(3, 40, size=9).astype(float)
    if _k % 2:
        _w[::3] *= -1.0
    CASES.append((_w, float(np.abs(_w).sum() / 2) + 0.5, bool(_k % 3), float(_k) * 0.25))


@pytest.mark.parametrize("w, cap, maximize, offset", CASES)
def test_knapsack_against_enumeration(w, cap, maximize, offset):
    opt = _brute(w, cap, maximize=maximize, offset=offset)
    assert opt is not None
    r = _knapsack(w, cap, maximize=maximize, offset=offset).solve(time_limit=30)
    assert r.objective == pytest.approx(opt, abs=1e-6)
    _assert_valid_bound(r, opt, maximize=maximize)


def _bqp(Q, c, *, maximize: bool):
    n = len(c)
    m = dm.Model("bqp")
    x = m.binary("x", shape=(n,))
    obj = sum(int(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(i, n) if Q[i, j])
    obj = obj + c @ x
    m.subject_to(x[0] * x[1] <= 1)  # redundant bilinear row -> spatial B&B
    m.subject_to(dm.sum(x) <= n - 2)
    (m.maximize if maximize else m.minimize)(obj)
    return m


def _bqp_brute(Q, c, *, maximize: bool):
    n = len(c)
    vals = [
        float(np.asarray(b) @ np.triu(Q) @ np.asarray(b) + c @ np.asarray(b))
        for b in itertools.product((0, 1), repeat=n)
        if sum(b) <= n - 2
    ]
    return max(vals) if maximize else min(vals)


QCASES = []
for _k in range(4):
    _Q = np.triu(_rng.integers(-6, 7, size=(7, 7))).astype(float) * 2.0
    QCASES.append((_Q, _rng.integers(-9, 10, size=7).astype(float) * 2.0, bool(_k % 2)))


@pytest.mark.parametrize("Q, c, maximize", QCASES)
def test_binary_quadratic_against_enumeration(Q, c, maximize):
    opt = _bqp_brute(Q, c, maximize=maximize)
    r = _bqp(Q, c, maximize=maximize).solve(time_limit=30)
    assert r.objective == pytest.approx(opt, abs=1e-6)
    _assert_valid_bound(r, opt, maximize=maximize)


def test_flag_is_retired():
    assert not hasattr(SolverTuning(), "spatial_obj_lattice")


def test_issue_repro_bound_stays_valid():
    rng = np.random.default_rng(1)
    n = 30
    w = rng.integers(10, 100, size=n).astype(float)
    W = int(w.sum() / 2) + 0.5
    m = dm.Model("ss")
    x = m.binary("x", shape=(n,))
    m.subject_to(w @ x <= W)
    m.maximize(w @ x)
    m.subject_to(x[0] * x[1] <= 1)
    r = m.solve(time_limit=10)
    if r.status == "optimal":
        assert r.objective == pytest.approx(834.0)
    if r.objective is not None:
        assert r.objective <= 834.0 + 1e-6
    _assert_valid_bound(r, 834.0, maximize=True)
