"""#2303: NLP-BB's root-cut stage separated GMI rows on the Rust arm from a basis
*recovered* at the simplex's vertex instead of the basis the simplex ended on.

At a degenerate vertex ``recover_basis`` picks a basis by column order, not the
dual-feasible one whose tableau GMI is defined from (the HiGHS arm reads HiGHS's
optimal basis). Measured on the issue's reactor/separator GDP: LP objectives and
vertices agree between the engines to 5e-10 on 209 of 211 LPs, row duals differ by up
to 72 at identical ``x``, and the stage ran 162 rounds (5.4 s, ``stall``) from the
recovered basis against 38 (0.4 s, ``no_cuts``) from the simplex's own.
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
    monkeypatch.delenv("DISCOPT_NLPBB_ROOT_GMI_LP_BASIS", raising=False)
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
    # 162 rounds / stop=stall from the recovered basis (the issue's measurement).
    assert rc.rounds_run <= 80, (rc.rounds_run, rc.stop_reason)
    assert rc.stop_reason == "no_cuts"
