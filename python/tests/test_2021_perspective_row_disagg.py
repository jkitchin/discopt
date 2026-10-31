"""#2021 section 3: the disaggregated epigraph of a split quadratic ``<=`` row.

``DISCOPT_PERSPECTIVE_ROW_DISAGG`` (default ON; ``=0`` opts out) gives the
``lp_nlp_bb`` master one column ``s_i >= d_i w_i**2 / z_i`` per split coordinate of each
``PerspectiveRowTerm``, a remainder row ``R-tangent + sum_i s_i <= 0`` and
per-coordinate Frangioni-Gentile rows, in ADDITION to the aggregate split cut.
Measurement and panel: performance-plan section 103.12.

The checks are the section 5 regime-2 set: no feasible point (with ``s_i = d_i
w_i**2``) violates a pooled row; the disaggregated rows at ``r = w^`` imply the
aggregate cut, so the master is never looser; the bound is valid and strictly
tighter on a fixed reference set; and the flag is wired end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._relax import perspective as P
from discopt.solvers import oa
from test_2021_perspective_row_cut import (
    MU,
    _flat,
    _master_bound,
    _portfolio,
    _sample_feasible,
    _terms,
    _true_optimum,
)

FLAG = "DISCOPT_PERSPECTIVE_ROW_DISAGG"


def _fractional_point(m, rng):
    """A point with ``x = w`` and every ``z_i`` in ``(w_i, 1)``: an LP-like point."""
    n = sum(v.size for v in m._variables)
    p = np.zeros(n)
    w = rng.random(3)
    w = w / w.sum()
    for i in range(3):
        p[_flat(m, f"w{i}")] = w[i]
        p[_flat(m, f"x{i}")] = w[i]
        p[_flat(m, f"z{i}")] = w[i] + (1.0 - w[i]) * rng.uniform(0.05, 0.95)
    return p


@pytest.mark.smoke
def test_the_flag_is_on_by_default(monkeypatch):
    """Graduated on its panels (performance-plan section 103.12); ``=0`` opts out."""
    monkeypatch.delenv(FLAG, raising=False)
    assert P.perspective_row_disaggregation_enabled()
    assert oa._perspective_row_disagg_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not P.perspective_row_disaggregation_enabled()
    assert not oa._perspective_row_disagg_enabled()


@pytest.mark.smoke
def test_the_term_carries_its_indicator_bounds():
    m = _portfolio()
    _ev, (term,) = _terms(m)
    pos = term.split_positions()
    assert pos.size == 3
    # ``w_i - z_i <= 0``: U = 1 for every coordinate
    np.testing.assert_allclose(term.U[pos], 1.0)


@pytest.mark.smoke
def test_no_feasible_point_violates_a_pooled_row():
    """Validity: every pooled row holds at feasible points with ``s_i = d_i w_i**2``."""
    m = _portfolio(card=3)
    _ev, (term,) = _terms(m)
    n = sum(v.size for v in m._variables)
    rng = np.random.default_rng(2021)
    feas = _sample_feasible(m, rng, 200)
    epi = oa._PerspectiveRowEpigraph([term])
    for _ in range(60):
        # fractional LP-like references and arbitrary ones (infeasible too)
        epi.add_point(0, _fractional_point(m, rng), n)
        epi.add_point(0, rng.random(n) * 1.5 - 0.25, n)
    assert epi.remainder_rows > 0
    ratio_rows = sum(r > w for r, w in zip(epi._r, [0.0] * len(epi._r)))
    assert epi.term_rows > 0 and ratio_rows > 0
    n_master = n + epi.n_cols
    rows = epi.all_rows(n_master, n)
    assert len(rows) == epi.remainder_rows + epi.term_rows
    executed = 0
    for p in feas:
        full = np.concatenate([p, epi.start_values(p)])
        for a, b in rows:
            assert a @ full <= b + 1e-12, a @ full - b
            executed += 1
    assert executed == len(feas) * len(rows) > 0


@pytest.mark.smoke
def test_the_rows_at_r_equal_w_imply_the_aggregate_cut():
    """Dominance: remainder + per-coordinate rows at ``r = w^`` sum to ``term.cut``."""
    m = _portfolio()
    _ev, (term,) = _terms(m)
    n = sum(v.size for v in m._variables)
    rng = np.random.default_rng(5)
    executed = 0
    for _ in range(50):
        p = _fractional_point(m, rng)
        rem, rem_rhs = term.remainder_cut(p, n)
        agg, agg_rhs, shifted = term.cut(p, n)
        assert shifted == 3
        total = rem.copy()
        for i in term.split_positions():
            wi, d = float(p[term.W[i]]), float(term.d[i])
            # s_i >= d (2 w^ w - w^2 z): substitute for s_i in the remainder row
            total[term.W[i]] += 2.0 * d * wi
            total[term.z[i]] -= d * wi * wi
        np.testing.assert_allclose(total, agg, rtol=0.0, atol=1e-14)
        assert rem_rhs == pytest.approx(agg_rhs, abs=1e-14)
        executed += 1
    assert executed == 50


@pytest.mark.smoke
def test_the_ratio_row_is_tight_on_the_perspective_at_its_point():
    m = _portfolio()
    _ev, (term,) = _terms(m)
    rng = np.random.default_rng(11)
    executed = 0
    for _ in range(30):
        p = _fractional_point(m, rng)
        refs = dict()
        for i, r in term.perspective_references(p):
            refs.setdefault(i, []).append(r)
        for i in term.split_positions():
            w, z, d = float(p[term.W[i]]), float(p[term.z[i]]), float(term.d[i])
            r = max(refs[int(i)])
            assert r == pytest.approx(w / z, rel=1e-12)
            # the row's value at (w, z) equals the perspective there
            assert 2 * d * r * w - d * r * r * z == pytest.approx(d * w * w / z, rel=1e-10)
            executed += 1
    assert executed == 90


def _disagg_master_bound(m, term, refs):
    """``_master_bound``'s LP plus the s block and every pooled row at ``refs``."""
    from scipy.optimize import linprog

    n = sum(v.size for v in m._variables)
    epi = oa._PerspectiveRowEpigraph([term])
    for ref in refs:
        epi.add_point(0, ref, n)
    n_master = n + epi.n_cols
    w = [_flat(m, f"w{i}") for i in range(3)]
    x = [_flat(m, f"x{i}") for i in range(3)]
    z = [_flat(m, f"z{i}") for i in range(3)]
    c = np.zeros(n_master)
    c[w] = -MU
    a_eq, b_eq, a_ub, b_ub = [], [], [], []

    def row():
        return np.zeros(n_master)

    for i in range(3):
        r = row()
        r[x[i]], r[w[i]] = 1.0, -1.0
        a_eq.append(r)
        b_eq.append(0.0)
        r = row()
        r[w[i]], r[z[i]] = 1.0, -1.0
        a_ub.append(r)
        b_ub.append(0.0)
    r = row()
    r[w] = 1.0
    a_eq.append(r)
    b_eq.append(1.0)
    r = row()
    r[z] = 1.0
    a_ub.append(r)
    b_ub.append(2.0)
    for ref in refs:  # the aggregate split cut stays in the master
        coeffs, rhs, _k = term.cut(ref, n)
        r = row()
        r[:n] = coeffs
        a_ub.append(r)
        b_ub.append(rhs)
    for a, b in epi.all_rows(n_master, n):
        a_ub.append(a)
        b_ub.append(b)
    bounds = [(0.0, 1.0)] * n + [(0.0, None)] * epi.n_cols
    for i in range(3):
        bounds[x[i]] = (-1.0, 1.0)
    res = linprog(c, A_ub=np.array(a_ub), b_ub=b_ub, A_eq=np.array(a_eq), b_eq=b_eq, bounds=bounds)
    assert res.status == 0, res.message
    return -res.fun


@pytest.mark.smoke
def test_differential_master_bound():
    """Section 5: never looser than the aggregate split master, valid, sometimes tighter."""
    m = _portfolio()
    _ev, (term,) = _terms(m)
    rng = np.random.default_rng(7)
    opt = _true_optimum()
    executed = 0
    strictly = 0
    for _ in range(25):
        refs = [_fractional_point(m, rng) for _ in range(4)]
        split = _master_bound(m, term, refs, split=True)
        disagg = _disagg_master_bound(m, term, refs)
        assert disagg <= split + 1e-9
        assert disagg >= opt - 1e-7, (disagg, opt)
        strictly += disagg < split - 1e-6
        executed += 1
    assert executed == 25
    assert strictly > 0


@pytest.mark.smoke
def test_master_layout_puts_the_block_last_with_start_values():
    m = _portfolio()
    _ev, (term,) = _terms(m)
    n = sum(v.size for v in m._variables)
    epi = oa._PerspectiveRowEpigraph([term])
    epi.add_point(0, _fractional_point(m, np.random.default_rng(3)), n)
    master = oa._build_master_milp_data(
        linear_A_rows=[],
        linear_b_rows=[],
        linear_senses=[],
        oa_A_rows=[],
        oa_b_rows=[],
        n_vars=n,
        lb=np.zeros(n),
        ub=np.ones(n),
        integrality=np.zeros(n, dtype=np.int32),
        obj_is_linear=True,
        obj_coeffs=(np.zeros(n), 0.0),
        objective_bound_valid=True,
        perspective_row_epigraph=epi,
    )
    assert master.perspective_row_start == n
    assert len(master.c) == n + epi.n_cols == n + 3
    assert master.bounds[n:] == [(0.0, 1e20)] * 3
    assert not np.any(master.c[n:])
    assert master.A_ub.shape == (epi.remainder_rows + epi.term_rows, n + 3)
    start = np.zeros(n)
    start[_flat(m, "w0")] = 0.5
    full = oa._extend_master_mip_start(master, n_vars=n, mip_start=start, mip_start_objective=None)
    d0 = term.d[list(term.W).index(_flat(m, "w0"))]
    assert full[n:].sum() == pytest.approx(d0 * 0.25, rel=1e-12)


@pytest.mark.smoke
def test_the_flag_is_wired_and_keeps_the_answer(monkeypatch):
    from discopt.solvers.oa import solve_lp_nlp_bb

    # This model's risk row is a norm ball, which the #2059 outer approximation
    # (``DISCOPT_OA_SOC_OUTER``, default ON) covers, and a covered row does not get
    # this block. The test is about this block, so it runs with that one off; the
    # interaction is pinned in ``test_2059_soc_outer.py``.
    monkeypatch.setenv("DISCOPT_OA_SOC_OUTER", "0")
    monkeypatch.setenv(FLAG, "0")
    off = solve_lp_nlp_bb(_portfolio(), time_limit=60.0, milp_solver="highs")
    monkeypatch.setenv(FLAG, "1")
    on = solve_lp_nlp_bb(_portfolio(), time_limit=60.0, milp_solver="highs")
    assert off.status == "optimal" and on.status == "optimal", (off.status, on.status)
    assert on.objective == pytest.approx(_true_optimum(), abs=1e-5)
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    stats = on.solver_stats or {}
    assert stats.get("oa/perspective_row_disagg_cols") == 3.0
    # The block ran: rows were pooled and the master was separated against them.
    # (This 3-asset master certifies in its first tree, so none was violated and
    # ``..._rows`` -- rows OFFERED to the master -- is legitimately 0 here.)
    assert stats.get("oa/perspective_row_disagg_pooled", 0.0) > 0.0
    assert stats.get("oa/perspective_row_disagg_separations", 0.0) >= 1.0
    assert "oa/perspective_row_disagg_rows" in stats
    assert "oa/perspective_row_disagg_rows" not in (off.solver_stats or {})
