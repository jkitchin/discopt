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
    assert _parse_order("ICB-rn") == ("ICB", True, False, False)
    assert _parse_order("CIB") == ("CIB", False, True, False)
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


def test_candidate_stage_stays_inside_its_budget_on_pb_rows(monkeypatch):
    """Large-coefficient equality rows (Cornuejols-Dawande market split) make
    every CDCL conflict expensive, so the stage must look at its clock often: each
    solver call is capped at ``_SAT_SLICE_CONFLICTS`` conflicts, and once the
    stage's stop time has passed it makes no further call. (At 1000 conflicts
    per slice one slice overran a 0.2 s stage budget by 0.6 s and took the base
    proof's time.) Asserted on the calls, not on wall time, so load cannot flip it.
    """
    import time

    import discopt._rust as rust

    rng = np.random.default_rng(7)
    A = rng.integers(0, 100, size=(4, 36))
    rhs = A.sum(axis=1) // 2
    m = dm.Model("ms")
    x = m.binary("x", shape=(36,))
    for i in range(4):
        m.subject_to(sum(float(A[i, j]) * x[j] for j in range(36)) == float(rhs[i]))
    m.minimize(0 * x[0])

    # The stage's own inputs, captured from a real call (the run itself is cut
    # short: only the arrays are wanted).
    captured: list[tuple] = []

    def capture(*a, **k):
        captured.append(a)
        return None, 0

    monkeypatch.setattr(_iis_milp, "_binary_core", capture)
    with pytest.raises((TimeoutError, ValueError)):
        milp_iis(m, total_time_limit=0.5)
    monkeypatch.undo()
    assert len(captured) == 1
    args = captured[0][:-1]

    calls: list[tuple[float, int]] = []  # (time of call, conflict limit)
    real_solver = rust.PySatSolver

    class Recording:
        def __init__(self, n):
            self._s = real_solver(n)

        def solve(self, assumptions, conflict_limit=None):
            calls.append((time.perf_counter(), conflict_limit))
            return self._s.solve(assumptions, conflict_limit)

        def __getattr__(self, name):
            return getattr(self._s, name)

    monkeypatch.setattr(rust, "PySatSolver", Recording)
    # A stop time the encoding (~0.15 s) fits in even under load; the search
    # cannot finish market split in it, so the stage must stop on the clock.
    stop = time.perf_counter() + 1.0
    _iis_milp._binary_core(*args, stop)
    assert len(calls) >= 1
    assert all(lim is not None and lim <= _iis_milp._SAT_SLICE_CONFLICTS for _, lim in calls)
    assert _iis_milp._SAT_SLICE_CONFLICTS <= 100
    assert sum(t >= stop for t, _ in calls) <= 1


def _knapsack_and_none(n=32, seed=3):
    """A row whose BDD has ~n*K nodes (weights up to 1e6) plus `sum x <= 0`:
    together a 2-row IIS."""
    rng = np.random.default_rng(seed)
    w = rng.integers(1, 10**6, size=n)
    m = dm.Model("knap")
    x = m.binary("x", shape=(n,))
    m.subject_to(sum(float(w[j]) * x[j] for j in range(n)) >= float(w.sum() // 2), name="knap")
    m.subject_to(sum(x[j] for j in range(n)) <= 0, name="none")
    m.minimize(0 * x[0])
    return m


def test_a_huge_bdd_row_cannot_run_the_stage_past_its_budget():
    """Review of #2112: the clock and the clause cap were checked only between
    rows, and this one row ran a 2 s budget to 46 s and TimeoutError. The
    encoding now checks both per BDD layer."""
    m = _knapsack_and_none()
    res = milp_iis(m, total_time_limit=2.0)
    assert sorted(m._constraints[i].name for i in res.constraints) == ["knap", "none"]
    assert res.complete


def test_encode_geq_aborts_before_creating_anything():
    w, lits, K = [7, 5, 3, 2], [1, 2, 3, 4], 9
    clauses: list = []
    with pytest.raises(_iis_milp._EncodeAbort, match="size"):
        _encode_geq(w, lits, K, 5, 5, clauses, max_clauses=3)
    assert clauses == []
    import time as _time

    with pytest.raises(_iis_milp._EncodeAbort, match="time"):
        _encode_geq(w, lits, K, 5, 5, clauses, stop_at=_time.perf_counter() - 1)
    assert clauses == []
    assert _encode_geq(w, lits, K, 5, 5, clauses) > 5 and clauses  # unrestricted: encodes


def test_an_oversized_row_is_left_out_and_the_others_still_give_a_core(monkeypatch):
    m = _knapsack_and_none(n=20)
    x = m._variables[0]
    m.subject_to(x[0] >= 1, name="c_on")  # with c_off: an infeasible clause pair
    m.subject_to(x[0] <= 0, name="c_off")
    monkeypatch.setattr(_iis_milp, "_SAT_MAX_CLAUSES", 50)  # the knapsack row cannot fit
    res = milp_iis(m)
    # The core comes from the three encodable rows (knap is left out), and any
    # 2-row IIS among them is right: c_on + c_off, or c_on + none.
    assert res.sat_rows is not None and res.sat_rows <= 3
    names = {m._constraints[i].name for i in res.constraints}
    assert len(names) == 2 and "knap" not in names
    assert res.complete
