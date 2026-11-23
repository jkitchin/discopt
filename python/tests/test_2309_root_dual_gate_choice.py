"""#2309: the HiGHS MILP route's root-dual gate stays the per-column ``box_dual_violation``.

The #2309 panel (``scripts/panel_2309_root_dual_gate.py``, 4420 instances x 5 arms:
highspy 1.11.0 / 1.15.1 / 1.16.0, root LP on HiGHS and on the in-house engine) measured
four gates end to end against solver-independent truth. 84 root pairs were wrong against
the exact LP optimum, all on the high side; the per-column gate refused 84/84. These
tests pin why the two cheaper-looking alternatives were rejected, so nobody gates on them:

* the scale-floored value (``problem_scale_floor=True``, reported as
  ``milp/root_dual_violation_scaled``) read BELOW tolerance on 6 of the 84 -- two of them
  are recorded below, as highspy 1.15.1 returned them;
* skipping interior single-entry ``c_j = 0`` columns (the #2300 alternative) is unsound by
  construction: a pair can be sign-correct on every structural column and carry a nonzero
  dual on a row whose slack is basic, which only that skipped column shows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import lp_milp_highs as H

pytestmark = pytest.mark.smoke

_DATA = json.loads(
    (Path(__file__).parent / "data" / "issue_2309" / "missed_by_scaled_floor.json").read_text()
)
_PAIRS = sorted(k for k in _DATA if not k.startswith("_"))


def _sf(rec) -> H.StdForm:
    s = rec["sf"]
    return H.StdForm.from_arrays(
        np.array(s["c"]),
        sp.csc_matrix(np.array(s["A"])),
        np.array(s["b"]),
        np.array(s["xl"]),
        np.array(s["xu"]),
        obj_const=s["obj_const"],
    )


def _refuses(sf, x, y, value) -> bool:
    """The route's decision (``_solve_milp_std``): gate value over tolerance and the
    NS-safe gap of the pair not within the LP certificate yardstick."""
    if value <= H.dual_violation_tolerance(sf):
        return False
    lp_obj = float(sf.c @ x) + sf.obj_const
    ns = H.root_pair_ns_gap(sf, x, y)
    return ns is None or ns > H.CERT_ABS + H.CERT_REL * abs(lp_obj)


@pytest.mark.parametrize("key", _PAIRS)
def test_recorded_wrong_pair_is_refused_by_the_gate_but_not_by_the_floor(key) -> None:
    rec = _DATA[key]
    sf, x, y = _sf(rec), np.array(rec["x"]), np.array(rec["y"])
    lp_obj = float(sf.c @ x) + sf.obj_const
    yardstick = H.CERT_ABS + H.CERT_REL * abs(rec["lp_true"])
    # The pair really is wrong: its objective is above the exact LP optimum ...
    assert lp_obj - rec["lp_true"] > yardstick, (lp_obj, rec["lp_true"])
    # ... the gate the route uses refuses it ...
    assert _refuses(sf, x, y, H.box_dual_violation(sf, x, y))
    # ... and the scale-floored value would have let it through.
    floored = H.box_dual_violation(sf, x, y, problem_scale_floor=True)
    assert floored <= H.dual_violation_tolerance(sf), floored
    assert not _refuses(sf, x, y, floored)


def test_skipping_interior_singleton_zero_cost_columns_is_unsound() -> None:
    """min -x1  s.t.  x1 <= 1,  x1 + x2 <= 3,  0 <= x1, x2 <= 2.  The pair below is
    primal feasible with objective -0.5 against the optimum -1; every structural reduced
    cost is sign-correct, and the only violation is the basic slack of row 0 pricing
    ``d = -y_0 * 1 = 1``. Skip that column and the pair passes without the NS gap."""
    A = sp.csc_matrix(np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]]))  # noqa: N806
    sf = H.StdForm.from_arrays(
        np.array([-1.0, 0.0, 0.0, 0.0]),
        A,
        np.array([1.0, 3.0]),
        np.zeros(4),
        np.array([2.0, 2.0, H.INF, H.INF]),
    )
    x, y = np.array([0.5, 0.0, 0.5, 2.5]), np.array([-1.0, 0.0])
    assert np.allclose(sf.A @ x, sf.b)
    d = sf.c - A.T @ y
    singleton_zero_cost = (np.diff(A.indptr) == 1) & (sf.c == 0.0)
    basic = (x > sf.xl) & (x < sf.xu)
    # Without the skipped column there is nothing to see:
    others = ~(singleton_zero_cost & basic)
    assert np.all(d[others & (x <= sf.xl)] >= 0.0) and np.all(d[others & basic] == 0.0)
    # The gate the route uses refuses it; the NS gap shows the 0.5 it is wrong by.
    assert _refuses(sf, x, y, H.box_dual_violation(sf, x, y))
    assert H.root_pair_ns_gap(sf, x, y) == pytest.approx(0.5)


@pytest.mark.requires_highs
@pytest.mark.parametrize("key", _PAIRS)
def test_route_refuses_the_recorded_pair(key, monkeypatch) -> None:
    """End to end: hand the route the recorded root pair; it must not certify."""
    import discopt.modeling as dm

    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    rec, want = _DATA[key], _sf(_DATA[key])
    real = H.solve_lp_std
    injected: list = []

    def spy(sf, **kw):
        out = real(sf, **kw)
        same = (
            sf.A.shape == want.A.shape
            and np.array_equal(sf.c, want.c)
            and np.array_equal(sf.A.toarray(), want.A.toarray())
            and np.array_equal(sf.xu, want.xu)
            and np.array_equal(sf.xl, want.xl)
        )
        if same and out.x is not None and out.row_dual is not None:
            out.x, out.row_dual = np.array(rec["x"]), np.array(rec["y"])
            injected.append(1)
        return out

    monkeypatch.setattr(H, "solve_lp_std", spy)
    mdl = rec["model"]
    m = dm.Model("i2309")
    xs = [m.integer(f"x{j}", lb=0, ub=mdl["ub"][j]) for j in range(len(mdl["c"]))]
    m.minimize(sum(mdl["c"][j] * xs[j] for j in range(len(xs))))
    for i in range(len(mdl["b"])):
        m.subject_to(sum(mdl["A"][i][j] * xs[j] for j in range(len(xs))) <= mdl["b"][i])
    r = m.solve(time_limit=60)
    assert injected, "MEASURED NOTHING: the route's root LP never matched the recorded form"
    st = r.solver_stats
    assert st["milp/root_dual_violation"] > H.dual_violation_tolerance(want), st
    assert st.get("milp/root_dual_unverified") == 1.0, st
    assert not r.gap_certified, (r.status, r.algorithm_route)
