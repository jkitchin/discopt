"""#1229: free columns must never come back at the 1e20 INF sentinel.

On ``issue-2388.lp`` (HiGHS check instance; arrays frozen in
``data/issue1229_issue2388_milp.npz`` because ``ref/`` is not tracked) the Rust MILP
driver's slack starting basis parked zero-cost free columns ``AT_UPPER``, so eight
of them read back as ``1e20``. Paired sentinels cancel *exactly* in the rows that
pair them, so the point looked feasible and was reported ``optimal``; row ``c43``
(``x17 - x37 >= 10.2``) is violated by 10.2. HiGHS: optimal 0.0.

Feasibility is checked with a dense per-row dot product: scipy's sparse matvec
contracts to FMA and fabricates a residual on sentinel-valued entries (see #1229).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

DATA = Path(__file__).parent / "data" / "issue1229_issue2388_milp.npz"


def _instance():
    d = np.load(DATA)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["shape"]))
    return {k: d[k] for k in d.files} | {"A": A}


def _dense_max_violation(inst, x):
    x = np.asarray(x, float)
    assert np.all(np.isfinite(x))
    A = inst["A"]
    viol = max(
        float(np.max(np.maximum(inst["col_lo"] - x, 0.0))),
        float(np.max(np.maximum(x - inst["col_hi"], 0.0))),
    )
    checked_rows = 0
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        act = float(np.dot(A.data[s:e], x[A.indices[s:e]]))
        viol = max(viol, inst["row_lo"][i] - act, act - inst["row_hi"][i])
        checked_rows += 1
    assert checked_rows == A.shape[0] > 0
    xi = x[inst["is_int"]]
    return max(viol, float(np.max(np.abs(xi - np.round(xi)))))


def _ub_eq_form(inst):
    A, lo, hi = inst["A"], inst["row_lo"], inst["row_hi"]
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    up = np.isfinite(hi) & ~eq
    dn = np.isfinite(lo) & ~eq
    A_ub = sp.vstack([A[np.flatnonzero(up)], -A[np.flatnonzero(dn)]]).tocsr()
    b_ub = np.concatenate([hi[up], -lo[dn]])
    bounds = [
        (lo_j if np.isfinite(lo_j) else None, hi_j if np.isfinite(hi_j) else None)
        for lo_j, hi_j in zip(inst["col_lo"], inst["col_hi"])
    ]
    return A_ub, b_ub, A[np.flatnonzero(eq)], lo[eq], bounds


def test_fixture_has_the_free_columns_that_trigger_the_defect():
    inst = _instance()
    free = ~np.isfinite(inst["col_lo"]) & ~np.isfinite(inst["col_hi"])
    # The defect needs zero-cost free columns: 14, all continuous and cost 0.
    assert int(free.sum()) == 14
    assert np.count_nonzero(inst["c"][free]) == 0
    assert not inst["is_int"][free].any()


def test_milp_simplex_entry_returns_a_feasible_point():
    """The ``get_milp_solver("simplex")`` entry (MINLP masters, Lagrangian, Benders)."""
    from discopt.solvers import SolveStatus
    from discopt.solvers.milp_simplex import solve_milp

    inst = _instance()
    A_ub, b_ub, A_eq, b_eq, bounds = _ub_eq_form(inst)
    r = solve_milp(
        inst["c"], A_ub, b_ub, A_eq, b_eq, bounds, inst["is_int"].astype(int), time_limit=30.0
    )
    assert r.status == SolveStatus.OPTIMAL
    x = np.asarray(r.x, float)[: inst["c"].shape[0]]
    assert np.max(np.abs(x)) < 1e15, "a column came back at the INF sentinel"
    assert _dense_max_violation(inst, x) <= 1e-6
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert r.bound <= r.objective + 1e-9


def test_driver_binding_returns_a_feasible_point():
    """The raw binding with the arguments #1229 reported."""
    from discopt import _rust
    from discopt.solvers.milp_simplex import _marshal_std_form

    inst = _instance()
    A_ub, b_ub, A_eq, b_eq, bounds = _ub_eq_form(inst)
    std = _marshal_std_form(inst["c"], A_ub, b_ub, A_eq, b_eq, bounds, inst["is_int"].astype(int))
    n = inst["c"].shape[0]
    st, x, obj, bd, nodes, iters = _rust.solve_milp_csc_py(
        std.c, std.m, n + std.m, std.col_ptr, std.row_idx, std.vals,
        std.b, std.lb, std.ub, std.int_cols, n, 0.0, 1_000_000, 1e-4, 1e-9,
        root_cuts=16, cut_rounds=1, cut_select=False,
        sb_max_cands=6, sb_node_budget=48, time_limit_s=30.0,
    )  # fmt: skip
    x = np.asarray(x, float)[:n]
    assert np.max(np.abs(x)) < 1e15, "a column came back at the INF sentinel"
    assert _dense_max_violation(inst, x) <= 1e-6
    assert obj == pytest.approx(0.0, abs=1e-6)
