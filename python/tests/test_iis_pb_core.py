"""The SAT core candidate beyond clauses: cardinality and pseudo-Boolean rows.

``_iis_milp._binary_core`` encodes every integral-coefficient row of a
pure-binary model as a guarded BDD (Een & Sorensson 2006). Before this, one
non-clause row disabled the candidate for the whole model; on MIPcc27's
smallset that left neos-4344877-stour at 1005 elements (incomplete at 60 s,
Gurobi 126) where the candidate gives 126 in 2.2 s.

The encoding is checked exhaustively on small random rows; the engine-level
tests pin that a cardinality model now gets a candidate and still returns a
certified IIS, and that the stage declines what it cannot encode.
"""

import itertools
import random

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _iis_milp
from discopt._iis_milp import _encode_geq, _parse_order, _pb_rows, milp_iis

highspy = pytest.importorskip("highspy")
from discopt._rust import PySatSolver  # noqa: E402


@pytest.mark.smoke
def test_encoding_agrees_with_the_row_on_every_assignment():
    rng = random.Random(7)
    checks = 0
    for _ in range(150):
        n = rng.randint(1, 6)
        cols = np.arange(n)
        vals = np.array([rng.choice([-5, -3, -2, -1, 1, 2, 3, 4]) for _ in range(n)], float)
        smin, smax = vals[vals < 0].sum(), vals[vals > 0].sum()
        kind = rng.choice(["lo", "hi", "eq", "both"])
        lo = rng.uniform(smin - 1, smax + 1) if kind in ("lo", "both") else -1e20
        hi = rng.uniform(smin - 1, smax + 1) if kind in ("hi", "both") else 1e20
        if kind == "eq":
            lo = hi = float(rng.randint(int(smin), int(smax)))
        if kind == "both" and lo > hi:
            lo, hi = hi, lo
        guard = n + 1
        nv, clauses = guard, []
        for w, lits, K in _pb_rows(cols, vals, lo, hi):
            nv = _encode_geq(w, lits, K, guard, nv, clauses)
        for x in itertools.product([0, 1], repeat=n):
            act = float(np.dot(vals, x))
            want = act >= lo - 1e-9 and act <= hi + 1e-9
            s = PySatSolver(nv)
            for c in clauses:
                s.add_clause(c)
            fix = [(j + 1) if x[j] else -(j + 1) for j in range(n)]
            assert s.solve([guard] + fix) == want, (vals, lo, hi, x)
            checks += 1
    assert checks > 2500


def _pigeons_card(p, h):
    """p pigeons, h holes, p > h: 'each pigeon somewhere' and 'at most one per
    hole' as cardinality rows (not clauses), plus redundant rows."""
    m = dm.Model("php")
    x = m.binary("x", shape=(p, h))
    for i in range(p):
        m.subject_to(sum(x[i, j] for j in range(h)) >= 1, name=f"pigeon{i}")
    for j in range(h):
        m.subject_to(sum(x[i, j] for i in range(p)) <= 1, name=f"hole{j}")
    for i in range(p):
        m.subject_to(sum(2 * x[i, j] for j in range(h)) <= 2 * h, name=f"red{i}")
    m.minimize(0 * x[0, 0])
    return m


@pytest.mark.smoke
def test_cardinality_model_gets_a_candidate_and_a_certified_iis():
    m = _pigeons_card(4, 3)
    res = milp_iis(m)
    assert res.sat_rows is not None and res.sat_rows < len(m._constraints)
    assert res.complete and res.proven_irreducible
    names = {m._constraints[i].name for i in res.constraints}
    assert not any(nm.startswith("red") for nm in names)
    assert len(res.certificates) == len(res.constraints) + len(res.bounds) + len(res.integrality)


def test_member_flag_n_skips_the_candidate():
    m = _pigeons_card(4, 3)
    assert milp_iis(m, order="ICB-n").sat_rows is None
    assert _parse_order("ICB-rn") == ("ICB", True, False)
    assert _parse_order("CIB") == ("CIB", False, True)
    for bad in ("ICB-", "ICB-rr", "ICB-x"):
        with pytest.raises(ValueError, match="order"):
            _parse_order(bad)


def test_fractional_coefficients_get_no_candidate():
    m = dm.Model("frac")
    x = m.binary("x", shape=(3,))
    m.subject_to(0.5 * x[0] + 0.5 * x[1] >= 1.5, name="frac")
    m.subject_to(x[2] >= 0, name="other")
    m.minimize(0 * x[0])
    res = milp_iis(m)
    assert res.sat_rows is None
    assert [m._constraints[i].name for i in res.constraints] == ["frac"]


def test_encoding_over_the_size_cap_gets_no_candidate(monkeypatch):
    monkeypatch.setattr(_iis_milp, "_SAT_MAX_CLAUSES", 2)
    res = milp_iis(_pigeons_card(4, 3))
    assert res.sat_rows is None and res.complete


def test_candidate_stage_stays_inside_its_budget_on_pb_rows():
    """Large-coefficient equality rows (Cornuejols-Dawande market split) make
    every CDCL conflict expensive: the stage must still stop near its budget,
    not one coarse conflict slice later (it overran 0.2 s by 0.6 s)."""
    import time

    rng = np.random.default_rng(7)
    A = rng.integers(0, 100, size=(4, 36))
    rhs = A.sum(axis=1) // 2
    m = dm.Model("ms")
    x = m.binary("x", shape=(36,))
    for i in range(4):
        m.subject_to(sum(float(A[i, j]) * x[j] for j in range(36)) == float(rhs[i]))
    m.minimize(0 * x[0])
    over = []
    real = _iis_milp._binary_core

    def timed(*a, **k):
        out = real(*a, **k)
        over.append(time.perf_counter() - a[-1])
        return out

    _iis_milp._binary_core = timed
    try:
        with pytest.raises((TimeoutError, ValueError)):
            milp_iis(m, total_time_limit=2.0)
    finally:
        _iis_milp._binary_core = real
    assert len(over) == 1
    assert over[0] < 0.25, over
