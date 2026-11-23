"""#2303: the Rust arm of NLP-BB's root-cut stage now separates GMI rows from the basis
its simplex ended on, not one *recovered* at the vertex (the HiGHS arm reads HiGHS's
optimal basis).

What this does NOT do is fix #2303's round count, which is set by round-off: 74-162
rounds across OpenBLAS kernel types in either arm (see ``root_gmi_lp_basis_enabled``).
So the end-to-end test asserts the mechanism and the certified answer, never a round
count or a stop reason.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._rust import gomory_cuts_py, solve_lp_warm_csc_py
from discopt.solvers import _root_cuts as RC
from discopt.solvers.lp_milp_highs import StdForm
from discopt.solvers.lp_std_engine import RustEngine


def _small_ip_lp():
    # max x0 + x1 s.t. 2 x0 + 2 x1 <= 3, x0 - x1 <= 0.5, x >= 0 (standard form with
    # slacks): the LP optimum is fractional in an integer column.
    a = np.array([[2.0, 2.0, 1.0, 0.0], [1.0, -1.0, 0.0, 1.0]])
    b = np.array([3.0, 0.5])
    c = np.array([-1.0, -1.0, 0.0, 0.0])
    lo = np.zeros(4)
    hi = np.array([10.0, 10.0, 1e20, 1e20])
    is_int = np.array([True, True, False, False])
    return a, b, c, lo, hi, is_int


def _solve(a, b, c, lo, hi):
    acsc = sp.csc_matrix(a)
    m, n = a.shape
    return solve_lp_warm_csc_py(
        c,
        m,
        n,
        acsc.indptr.astype(np.int64),
        acsc.indices.astype(np.int64),
        acsc.data.astype(np.float64),
        b,
        lo,
        hi,
        None,
        None,
    )


@pytest.mark.smoke
def test_explicit_basis_matches_recovery_at_a_nondegenerate_vertex():
    a, b, c, lo, hi, is_int = _small_ip_lp()
    st, x, _obj, _it, cs, bv, _y, _ray = _solve(a, b, c, lo, hi)
    assert st == "optimal" and len(bv) == a.shape[0]
    rec = gomory_cuts_py(x, a, b, c, lo, hi, is_int)
    own = gomory_cuts_py(
        x,
        a,
        b,
        c,
        lo,
        hi,
        is_int,
        basic_vars=np.asarray(bv, dtype=np.int64),
        col_status=np.asarray(cs, dtype=np.int8),
    )
    assert rec is not None and own is not None
    assert rec[0].shape[0] >= 1  # the probe separated something (CLAUDE.md §6)
    np.testing.assert_allclose(own[0], rec[0])
    np.testing.assert_allclose(own[1], rec[1])


def test_malformed_basis_raises():
    a, b, c, lo, hi, is_int = _small_ip_lp()
    st, x, _obj, _it, cs, bv, _y, _ray = _solve(a, b, c, lo, hi)
    bv = np.asarray(bv, dtype=np.int64)
    cs = np.asarray(cs, dtype=np.int8)
    with pytest.raises(ValueError, match="both"):
        gomory_cuts_py(x, a, b, c, lo, hi, is_int, basic_vars=bv)
    with pytest.raises(ValueError, match="out of range, repeated"):
        gomory_cuts_py(
            x, a, b, c, lo, hi, is_int, basic_vars=np.array([bv[0], bv[0]]), col_status=cs
        )
    with pytest.raises(ValueError, match="statuses"):
        gomory_cuts_py(x, a, b, c, lo, hi, is_int, basic_vars=bv, col_status=cs[:-1])


def test_rust_engine_returns_its_basis_on_optimal():
    a, b, c, lo, hi, _ = _small_ip_lp()
    raw = RustEngine().solve(
        StdForm.from_arrays(c, sp.csc_matrix(a), b, lo, hi),
        time_limit=None,
        presolve=None,
        tight_tolerances=False,
    )
    assert raw.status == "optimal"
    assert raw.basic_vars is not None and raw.col_status is not None
    assert len(raw.basic_vars) == a.shape[0] and len(raw.col_status) == a.shape[1]


def _synthesis_gdp() -> dm.Model:
    """#2303's reactor/separator synthesis (convex GDP), as the issue builds it."""
    reactors = [(22.0, 120.0, 2.0), (29.0, 260.0, 2.5), (32.0, 380.0, 2.4)]
    seps = [(0.95, 150.0, 1.2), (0.90, 60.0, 1.6)]
    m = dm.Model("synthesis_nl")
    f, p = m.continuous("F", lb=0, ub=100), m.continuous("P", lb=0, ub=100)
    s = m.continuous("S", lb=0, ub=60)
    cr, cs = m.continuous("CR", lb=0, ub=600), m.continuous("CS", lb=0, ub=300)
    yr_b, ys_b = m.boolean("YR", shape=3), m.boolean("YS", shape=2)
    yr, ys = yr_b.variable, ys_b.variable
    m.either_or(
        [
            [p <= a * dm.log(1 + f / 10), cr >= fx + v * f, yr[k] == 1]
            for k, (a, fx, v) in enumerate(reactors)
        ],
        name="reactor",
    )
    m.either_or(
        [[s == rec * p, cs >= fx + v * p, ys[k] == 1] for k, (rec, fx, v) in enumerate(seps)],
        name="separator",
    )
    m.exactly(1, [yr[k] for k in range(3)])
    m.exactly(1, [ys[k] for k in range(2)])
    m.logical(yr_b[1].implies(~ys_b[1]))
    m.maximize(30.0 * s - 9.0 * f - cr - cs)
    return m


def test_synthesis_gdp_root_stage_reads_the_simplex_basis(monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    monkeypatch.delenv("DISCOPT_NLPBB_ROOT_CUTS_ENGINE", raising=False)
    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_GMI_LP_BASIS", "1")  # a default-OFF opt-in
    got = {}
    real = RC.generate_root_cuts

    def spy(*a, **k):
        got["r"] = real(*a, **k)
        return got["r"]

    monkeypatch.setattr(RC, "generate_root_cuts", spy)
    before = dict(RC.GMI_BASIS_STATS)
    r = _synthesis_gdp().solve(gdp_method="big-m")
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(487.7362, abs=1e-3)
    assert "r" in got, "the root-cut stage never ran (CLAUDE.md §6)"
    assert RC.GMI_BASIS_STATS["lp_basis"] > before["lp_basis"]
    rc = got["r"]
    assert rc.rounds_run >= 1 and rc.stop_reason in {"no_cuts", "stall", "rounds", "budget"}


def test_supplied_basis_with_a_sentinel_bound_falls_back_to_recovery():
    """A nonbasic column whose named bound is the 1e20 sentinel (a nonbasic FREE
    column arrives as AT_LOWER) must never reach the separator: the binding falls
    back to recovery, exactly as if no basis had been supplied (#2307 review)."""
    a, b, c, lo, hi, is_int = _small_ip_lp()
    st, x, _obj, _it, cs, bv, _y, _ray = _solve(a, b, c, lo, hi)
    cs = np.asarray(cs, dtype=np.int8).copy()
    bv = np.asarray(bv, dtype=np.int64)
    nonbasic = [j for j in range(a.shape[1]) if cs[j] != 1]
    assert nonbasic, "the probe needs a nonbasic column (CLAUDE.md §6)"
    j = nonbasic[0]
    lo2 = lo.copy()
    lo2[j] = -1e20  # the column is now free below; its status still says AT_LOWER
    cs[j] = 0
    own = gomory_cuts_py(x, a, b, c, lo2, hi, is_int, basic_vars=bv, col_status=cs)
    rec = gomory_cuts_py(x, a, b, c, lo2, hi, is_int)
    if rec is None:
        assert own is None
    else:
        assert own is not None
        np.testing.assert_array_equal(own[0], rec[0])
        np.testing.assert_array_equal(own[1], rec[1])
        assert np.all(np.abs(own[0]) < 1e10)


def test_supplied_basis_off_the_vertex_falls_back_to_recovery():
    """A basis whose nonbasic column is not at the bound it names describes another
    vertex: it is ignored, not used."""
    a, b, c, lo, hi, is_int = _small_ip_lp()
    st, x, _obj, _it, cs, bv, _y, _ray = _solve(a, b, c, lo, hi)
    cs = np.asarray(cs, dtype=np.int8)
    bv = np.asarray(bv, dtype=np.int64)
    x_off = np.asarray(x, dtype=float).copy()
    j = next(j for j in range(a.shape[1]) if cs[j] != 1)
    x_off[j] += 0.25
    own = gomory_cuts_py(x_off, a, b, c, lo, hi, is_int, basic_vars=bv, col_status=cs)
    rec = gomory_cuts_py(x_off, a, b, c, lo, hi, is_int)
    assert (own is None) == (rec is None)
    if own is not None:
        np.testing.assert_array_equal(own[0], rec[0])
