"""#2300: ``milp/root_dual_violation`` read 1.0 on a correctly solved root LP whose duals
came from the in-house LP engine.

``box_dual_violation`` divides each reduced cost by ``|c_j| + (|A|ᵀ|y|)_j``. On a basic
single-entry ``c_j = 0`` slack whose row dual is round-off (7.3e-15 here) numerator and
denominator are both ``|a·y_i|``, so the ratio is exactly 1.0. HiGHS returns exact zeros
there; the Rust engine returns round-off. The reported statistic now floors the scale at
the LP's own magnitude. The route's *gate* keeps the per-column test (decisions
unchanged; see ``box_dual_violation``'s docstring for why), so on Rust duals the gate
still falls through to the #1612 NS gap, which certifies.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs as H

pytestmark = [pytest.mark.smoke, pytest.mark.requires_highs]


def _set_cover(n: int = 16, rows: int = 30, seed: int = 16) -> dm.Model:
    rng = np.random.default_rng(seed)
    cover = np.zeros((rows, n), int)
    for i in range(rows):
        cover[i, rng.choice(n, 3, replace=False)] = 1
    cost = rng.integers(2, 5, n).astype(float)
    m = dm.Model("set_cover")
    s = m.binary("s", shape=(n,))
    for i in range(rows):
        m.subject_to(dm.sum(lambda j: s[j], over=[j for j in range(n) if cover[i, j]]) >= 1)
    m.minimize(dm.sum(lambda j: cost[j] * s[j], over=range(n)))
    return m


def test_reported_violation_is_round_off_on_both_engines() -> None:
    _, _, sf = _highs_std_form(_set_cover())
    tol = H.dual_violation_tolerance(sf)
    checked = 0
    for engine in ("highs", "rust"):
        lp = H.solve_lp_std(sf, engine=engine)
        assert lp.status == "optimal" and lp.x is not None and lp.row_dual is not None
        v = H.box_dual_violation(sf, lp.x, lp.row_dual, problem_scale_floor=True)
        assert v <= tol, (engine, v, tol)
        checked += 1
    # The per-column (gate) value is unchanged: on Rust duals it is still the 1.0 the
    # issue measured, which is what sends the route to the NS gap.
    lp = H.solve_lp_std(sf, engine="rust")
    assert H.box_dual_violation(sf, lp.x, lp.row_dual) > tol
    checked += 1
    assert checked == 3


def test_floor_still_flags_the_1410_missolve() -> None:
    """The floored statistic must still report HiGHS's #1410 mis-solved root pair."""
    import scipy.sparse as sp
    from test_1410_highs_dual_certificate import MISSOLVED_ROOT_X, MISSOLVED_ROOT_Y, UB, A, B, C

    m = len(B)
    mat = sp.hstack([sp.csr_matrix(np.array(A)), sp.identity(m, format="csr")], format="csc")
    sf = H.StdForm.from_arrays(
        np.concatenate([C, np.zeros(m)]),
        mat,
        np.array(B),
        np.zeros(len(C) + m),
        np.concatenate([np.array(UB, float), np.full(m, H.INF)]),
    )
    v = H.box_dual_violation(
        sf, np.array(MISSOLVED_ROOT_X), np.array(MISSOLVED_ROOT_Y), problem_scale_floor=True
    )
    assert v > H.dual_violation_tolerance(sf), v


def test_rust_engine_solve_reports_round_off_and_gate_is_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("DISCOPT_LP_STD_ENGINE", "rust")
    r = _set_cover().solve()
    st = r.solver_stats
    assert "milp/root_dual_violation" in st, "MEASURED NOTHING: the root check did not run"
    assert st["milp/root_dual_violation"] < 1e-12, st
    # Gate unchanged: the per-column test tripped, the NS gap ran and certified.
    assert st.get("milp/root_dual_certified_by_ns") == 1.0, st
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(17.0, abs=1e-6)
    assert r.bound == pytest.approx(17.0, abs=1e-6)
