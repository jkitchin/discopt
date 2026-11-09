"""highs-parity P3.1: the Rust ``MilpHandle``, a persistent MILP edited in place.

(1) A handle re-solved after each toggle equals a fresh ``solve_milp_csc_py`` call on
the same arrays bit for bit (status, x, objective, bound, nodes, LP iterations): the
handle runs the same driver entry with the same options, so any drift is a bug.
(2) The ranged-row layout (``A x - s = 0``, ``s`` in ``[row_lo, row_hi]``) is the
problem the caller meant: every toggled problem's verdict and optimum match HiGHS.
(3) ``presolve_only`` reports the driver's root presolve verdict.
The Rust-side tests (``discopt-core`` ``bnb::milp_handle``) check (1) on the
``sparse_milp_diff`` panel and 20 random toggle sequences.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers.milp_handle import RUST_INF, MilpHandle


def _random_milp(rng, m, n):
    A = sp.random(m, n, density=0.6, random_state=rng, data_rvs=lambda k: rng.integers(-3, 5, k))
    A = sp.csc_matrix(A, dtype=float)
    c = rng.integers(-6, 5, n).astype(float)
    row_hi = rng.integers(2, 12, m).astype(float)
    row_lo = np.where(rng.random(m) < 0.3, row_hi - rng.integers(1, 6, m), -np.inf)
    col_lb, col_ub = np.zeros(n), np.full(n, 3.0)
    is_int = rng.random(n) < 0.6
    return c, A, row_lo, row_hi, col_lb, col_ub, is_int


def _fresh(h: MilpHandle, c, A, **kw):
    """A fresh ``solve_milp_csc_py`` on the arrays the handle holds."""
    from discopt._rust import solve_milp_csc_py

    m, n_s = A.shape
    full = sp.hstack([A, -sp.identity(m, format="csc")], format="csc")
    full.sort_indices()
    sf = h.std_form()
    return solve_milp_csc_py(
        np.concatenate([c, np.zeros(m)]), m, n_s + m, full.indptr.astype(np.int64),
        full.indices.astype(np.int64), full.data, np.zeros(m), sf["lb"], sf["ub"],
        sf["integer_cols"], n_s, **kw,
    )  # fmt: skip


def _highs(c, A, row_lo, row_hi, lb, ub, is_int):
    highspy = pytest.importorskip("highspy")
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    n = len(c)
    for j in range(n):
        h.addVar(lb[j] if lb[j] > -RUST_INF else -inf, ub[j] if ub[j] < RUST_INF else inf)
        h.changeColCost(j, float(c[j]))
        if is_int[j]:
            h.changeColIntegrality(j, highspy.HighsVarType.kInteger)
    Ar = A.tocsr()
    for i in range(A.shape[0]):
        cols = Ar.indices[Ar.indptr[i] : Ar.indptr[i + 1]].astype(np.int32)
        vals = Ar.data[Ar.indptr[i] : Ar.indptr[i + 1]]
        lo = row_lo[i] if row_lo[i] > -RUST_INF else -inf
        hi = row_hi[i] if row_hi[i] < RUST_INF else inf
        h.addRow(lo, hi, len(cols), cols, vals)
    h.run()
    st = h.modelStatusToString(h.getModelStatus())
    return st, (h.getInfo().objective_function_value if st == "Optimal" else None)


def test_handle_matches_a_fresh_solve_and_highs_through_toggle_sequences():
    rng = np.random.default_rng(2026)
    compared = oracle_checked = 0
    statuses = set()
    for seq in range(20):
        m, n = 2 + seq % 3, 3 + seq % 4
        c, A, row_lo, row_hi, lb, ub, is_int = _random_milp(rng, m, n)
        h = MilpHandle(c, A, row_lo, row_hi, lb, ub, is_int)
        row_lo, row_hi, lb, ub, is_int = (
            row_lo.copy(), row_hi.copy(), lb.copy(), ub.copy(), is_int.copy()
        )  # fmt: skip
        for step in range(6):
            if step:
                kind = rng.integers(4)
                if kind == 0:  # drop / restore a row
                    i = int(rng.integers(m))
                    off = row_hi[i] >= RUST_INF and row_lo[i] <= -RUST_INF
                    row_lo[i], row_hi[i] = (-np.inf, float(rng.integers(2, 12))) if off else (
                        -np.inf, np.inf)  # fmt: skip
                    h.change_row_bounds(i, row_lo[i], row_hi[i])
                elif kind == 1:  # tighten / move a column box
                    j = int(rng.integers(n))
                    lb[j] = float(rng.integers(0, 3))
                    ub[j] = lb[j] + float(rng.integers(0, 2))
                    h.change_col_bounds(j, lb[j], ub[j])
                elif kind == 2:  # toggle integrality
                    j = int(rng.integers(n))
                    is_int[j] = not is_int[j]
                    h.change_integrality(j, bool(is_int[j]))
                else:  # a fractional box on an integer column
                    j = int(rng.integers(n))
                    lb[j], ub[j] = 0.4, 0.6
                    h.change_col_bounds(j, lb[j], ub[j])
            got = h.solve(max_nodes=5000)
            want = _fresh(h, c, A, max_nodes=5000)
            assert got.status == want[0] and got.nodes == want[4] and got.lp_iters == want[5]
            assert np.array_equal(got.x, np.asarray(want[1]))
            assert np.float64(got.objective).tobytes() == np.float64(want[2]).tobytes()
            assert np.float64(got.bound).tobytes() == np.float64(want[3]).tobytes()
            compared += 1
            statuses.add(got.status)
            hs, hobj = _highs(c, A, _rust(row_lo), _rust(row_hi), lb, ub, is_int)
            if hs == "Optimal":
                assert got.status == "optimal", (seq, step, got)
                assert got.objective == pytest.approx(hobj, abs=1e-6, rel=1e-6)
                assert got.bound <= hobj + 1e-6
                oracle_checked += 1
            elif hs == "Infeasible":
                assert got.status == "infeasible", (seq, step, got)
                oracle_checked += 1
    print(f"compared={compared} oracle_checked={oracle_checked} statuses={sorted(statuses)}")
    assert compared == 120 and oracle_checked >= 100
    assert {"optimal", "infeasible"} <= statuses


def _rust(v):
    return np.clip(np.asarray(v, dtype=float), -RUST_INF, RUST_INF)


def test_presolve_only_and_validation():
    A = sp.csc_matrix([[1.0, 1.0]])
    h = MilpHandle([-1, -1], A, [-np.inf], [3.0], [0, 0], [2, 2], [True, True])
    assert h.presolve_only() is False
    h.change_col_bounds(0, 2.5, 2.7)  # an integer column with no integer in its box
    assert h.presolve_only() is True
    r = h.solve()
    assert r.status == "infeasible" and r.nodes == 0
    with pytest.raises(IndexError):
        h.change_row_bounds(1, 0, 1)
    with pytest.raises(ValueError):
        h.change_col_bounds(0, np.nan, 1.0)
    with pytest.raises(ValueError):
        h.change_integrality(2, True)  # a logical, not a structural column
    with pytest.raises(ValueError):
        MilpHandle([1.0], A, [0.0], [1.0], [0, 0], [1, 1], [False, False])


def test_a_seed_is_validated_not_trusted():
    A = sp.csc_matrix([[5.0, 5.0, 5.0, 5.0]])
    c = [-10.0, -9.0, -8.0, -1.0]
    h = MilpHandle(c, A, [-np.inf], [9.0], [0] * 4, [1] * 4, [True] * 4)
    plain = h.solve()
    h.set_solution([1.0, 1.0, 1.0, 1.0])  # violates the row: ignored by the driver
    bad = h.solve()
    assert (bad.status, bad.objective) == (plain.status, plain.objective)
    h.set_solution(None)
