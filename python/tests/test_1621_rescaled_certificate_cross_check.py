"""#1621: the #1537 logical-column rescaling certified wrong answers on big-M MILPs.

On the default pure-MILP HiGHS route a feasible gasoline-blending MILP with big-M
line-up constraints was certified ``infeasible`` at M = 1e9 and ``optimal`` with
objective 0 at M = 1e10 (``gap_certified=True`` both times); its optimum is
473,958.43 and ``x = z = n = 0`` is feasible.

Root cause. ``logical_column_scales`` rescales the route's own unit logical when it
sits below the #1295 cap of its row. In a big-M row ``x - M z + s = 0`` the logical
is small only because the user's ``x`` is: ``x``'s entry is also ``1/M`` of the row.
Rescaling ``s`` removed the one entry the #1295 guard measures (an open column),
left the user's ``1/M`` entry -- the actual trap -- in place, and the rescaled
problem was then trusted. Two fixes, each tested on its own:

* ``logical_column_scales`` rescales a logical only when every other entry of its
  row is within the cap (the #1537 premise: the row is well scaled apart from the
  route's own column);
* ``solve_milp_std`` cross-checks any certificate earned on the rescaled form
  against a solve of the unscaled form, and discards it when a verified point
  refutes it (tested by forcing the pre-#1621 rescaling).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.lp_milp_highs as L  # noqa: N812
import numpy as np
import pytest

BLEND_OPT = 473958.43121396995  # plain HiGHS on the to_mps() export, every M
FC_OPT = 105.0  # y = 1, xA = 5

_S = dict(
    avail=[3000, 6000, 9000, 12000, 4000],
    cost=[45, 70, 98, 92, 105],
    RON=[93, 70, 98, 92, 96],
    RVP=[52.0, 11.0, 3.5, 7.0, 4.6],
    S=[10.0, 5.0, 1.0, 20.0, 6.0],
    SG=[0.584, 0.664, 0.810, 0.745, 0.700],
)
_P = dict(price=[110, 122], maxd=[20000, 8000], RONmin=[91, 96], RVPmax=[9.0, 9.0], Smax=[10, 10])


def _blend(M: float) -> dm.Model:  # noqa: N803
    """The issue's first witness, verbatim."""
    F, Lmin, B = 10_000.0, 500.0, [2500.0, 1000.0]  # noqa: N806
    I, J = range(5), range(2)  # noqa: N806, E741
    m = dm.Model("blend")
    x = {(i, j): m.continuous(f"x_{i}_{j}", lb=0, ub=float(_P["maxd"][j])) for i in I for j in J}
    z = {(i, j): m.binary(f"z_{i}_{j}") for i in I for j in J}
    n = {j: m.integer(f"n_{j}", lb=0, ub=int(_P["maxd"][j] // B[j])) for j in J}
    for i in I:
        m.subject_to(sum(x[i, j] for j in J) <= _S["avail"][i])
    for j in J:
        m.subject_to(sum(x[i, j] for i in I) <= _P["maxd"][j])
        m.subject_to(sum((_S["RON"][i] - _P["RONmin"][j]) * x[i, j] for i in I) >= 0)
        m.subject_to(
            sum((_S["RVP"][i] ** 1.25 - _P["RVPmax"][j] ** 1.25) * x[i, j] for i in I) <= 0
        )
        m.subject_to(sum(_S["SG"][i] * (_S["S"][i] - _P["Smax"][j]) * x[i, j] for i in I) <= 0)
        m.subject_to(sum(x[i, j] for i in I) == B[j] * n[j])
    for i in I:
        for j in J:
            m.subject_to(x[i, j] <= M * z[i, j])
            m.subject_to(x[i, j] >= Lmin * z[i, j])
    m.maximize(
        sum((_P["price"][j] - _S["cost"][i]) * x[i, j] for i in I for j in J) - F * sum(z.values())
    )
    return m


def _fixed_charge(M: float) -> dm.Model:  # noqa: N803
    """The issue's second witness (Chapter II.20)."""
    m = dm.Model("fc")
    xa = m.continuous("xA", lb=0, ub=100)
    xb = m.continuous("xB", lb=0, ub=100)
    y = m.binary("y")
    m.subject_to(xa + xb >= 5)
    m.subject_to(xa <= M * y)
    m.minimize(100 * y + xa + 30 * xb)
    return m


def _legacy_scales(sf, n_struct):
    """``logical_column_scales`` as shipped before #1621: no check on the rest of the
    row. Used to force the rescaled path so the cross-check is tested on its own."""
    if n_struct is None or n_struct >= sf.n:
        return None
    A = L.sp.csc_matrix(sf.A)  # noqa: N806
    absA = abs(A).tocoo()  # noqa: N806
    row_max = np.zeros(sf.m)
    np.maximum.at(row_max, absA.row, absA.data)
    f = np.ones(sf.n)
    cand = L._logical_columns(sf)
    cand[:n_struct] = False
    for j in np.flatnonzero(cand):
        i, a = int(A.indices[A.indptr[j]]), abs(float(A.data[A.indptr[j]]))
        if row_max[i] <= 0.0 or a / row_max[i] >= L.UNSCALABLE_OPEN_RATIO:
            continue
        if sf.xl[j] != 0.0 or sf.xu[j] < L.INF:
            continue  # plain ``s >= 0`` slacks only: scaling them is exact
        k = float(2.0 ** np.floor(np.log2(row_max[i] / a)))
        f[j] = k / 2.0 if k * a > row_max[i] else k
    return f if np.any(f != 1.0) else None


def _assert_not_falsely_certified(r, truth: float, maximize: bool) -> None:
    assert r.status != "infeasible", f"feasible model certified infeasible ({r.status})"
    if r.gap_certified:
        assert r.status == "optimal", r.status
        assert r.objective == pytest.approx(truth, rel=1e-6), r.objective
    if r.objective is not None:
        # Any published objective is a feasible point's: it cannot beat the optimum.
        if maximize:
            assert r.objective <= truth * (1 + 1e-6), r.objective
        else:
            assert r.objective >= truth - 1e-6, r.objective


# ── the issue's witnesses, end to end ─────────────────────────────────────────


@pytest.mark.parametrize("M", [1e8, 1e9, 1e10])
def test_blend_bigm_is_never_falsely_certified(M):  # noqa: N803
    """Main: M = 1e9 -> certified ``infeasible``; M = 1e10 -> certified ``optimal`` 0."""
    r = _blend(M).solve(time_limit=60)
    st = r.solver_stats or {}
    assert st.get("route/lp_milp_backend") == 1.0, "HiGHS MILP route not taken"
    _assert_not_falsely_certified(r, BLEND_OPT, maximize=True)


@pytest.mark.parametrize("M", [1e10, 1e12])
def test_fixed_charge_bigm_is_never_falsely_certified(M):  # noqa: N803
    """The issue reported ``optimal`` 150 certified at M >= 1e10. On b936476d this
    already published 105 uncertified (the #1537 point re-verification caught the
    rescaled y = 5/M incumbent); kept as a guard."""
    r = _fixed_charge(M).solve(time_limit=60)
    assert (r.solver_stats or {}).get("route/lp_milp_backend") == 1.0
    _assert_not_falsely_certified(r, FC_OPT, maximize=False)


# ── fix 1: the rescaling rule ─────────────────────────────────────────────────


def test_bigm_row_logical_is_not_rescaled():
    """A logical whose row also holds a user entry below the cap is left alone; the
    same logical in a uniformly large row (the #1537 class) is still rescaled."""
    big = 3e6
    # cols: x0, z1 (struct), s2 logical of the big-M row, s3 logical of the big row
    A = [  # noqa: N806
        [1.0, -1e10, 1.0, 0.0],  # x0 - M z1 + s2 = 0: x0's entry is 1e-10 of the row
        [big, 2 * big, 0.0, 1.0],  # rows x 1e6: every user entry within the cap
    ]
    sf = L.StdForm.from_arrays(
        np.zeros(4),
        np.asarray(A),
        np.array([0.0, big]),
        np.zeros(4),
        np.array([10.0, 1.0, L.INF, L.INF]),
        int_idx=[1],
    )
    f = L.logical_column_scales(sf, 2)
    assert f is not None
    assert f[2] == 1.0, "big-M row's logical was rescaled past the user's 1/M entry"
    assert f[3] != 1.0, "a uniformly scaled row's logical must still be rescaled"
    assert _legacy_scales(sf, 2)[2] != 1.0, "the legacy rule did rescale it (test premise)"


def test_blend_bigm_logicals_are_not_rescaled():
    r = _blend(1e10).solve(time_limit=60)
    st = r.solver_stats or {}
    assert st.get("route/lp_milp_backend") == 1.0
    assert "milp/logicals_rescaled" not in st, st


# ── fix 2: the cross-check, under the pre-#1621 rescaling ─────────────────────


@pytest.mark.parametrize("M", [1e9, 1e10])
def test_cross_check_refutes_a_rescaled_certificate(monkeypatch, M):  # noqa: N803
    """With the rescaling forced exactly as before #1621, the rescaled form certifies
    the issue's false answers; the cross-check must refute and discard them."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    monkeypatch.setattr(L, "logical_column_scales", _legacy_scales)
    # #1634 switched HiGHS's parallel-column presolve rule off, which changes which of
    # the route's falsifiers catches M = 1e10 (the #1634 presolve cross-check does,
    # and the result is still declined). Pin the pre-#1634 presolve so this test keeps
    # exercising the #1621 cross-check it is about.
    monkeypatch.setattr(L, "MILP_PRESOLVE_RULE_OFF", 0)
    r = _blend(M).solve(time_limit=60)
    st = r.solver_stats or {}
    assert st.get("route/lp_milp_backend") == 1.0
    assert st.get("milp/rescale_certificate_refuted") == 1.0, st
    _assert_not_falsely_certified(r, BLEND_OPT, maximize=True)


def test_cross_check_keeps_a_sound_rescaled_certificate():
    """``max 3 x0 + 2 x1, k (x0 + x1) + s = 4.5 k``: the rescaled certificate is
    right, the cross-check runs, finds nothing below it, and it stands."""
    k = 4e6
    sf = L.StdForm.from_arrays(
        np.array([-3.0, -2.0, 0.0]),
        np.array([[k, k, 1.0]]),
        np.array([4.5 * k]),
        np.zeros(3),
        np.array([3.0, 3.0, L.INF]),
        int_idx=[0, 1],
    )
    out = L.solve_milp_std(sf, time_limit=30, gap_tolerance=1e-4, max_nodes=1000, n_struct=2)
    assert out.stats.get("milp/logicals_rescaled") == 1.0
    assert out.stats.get("milp/rescale_cross_check_ran") == 1.0
    assert "milp/rescale_certificate_refuted" not in out.stats
    assert out.status == "optimal" and out.gap_certified
    assert out.objective == pytest.approx(-11.0)


@pytest.mark.parametrize("status", ["optimal", "infeasible"])
def test_cross_check_without_budget_withdraws_the_certificate(status):
    """A cross-check that cannot run is a safety net that never ran (#1309)."""
    sf = L.StdForm.from_arrays(
        np.array([1.0]), np.array([[1.0]]), np.array([1.0]), np.zeros(1), np.array([2.0])
    )
    x = np.array([1.0]) if status == "optimal" else None
    out = L.HighsOutcome(status, x=x, objective=1.0 if x is not None else None, gap_certified=True)
    out.bound = 1.0 if x is not None else None
    kw = dict(
        time_limit=0.0,
        gap_tolerance=1e-4,
        abs_gap_tolerance=None,
        max_nodes=10,
        initial_point=None,
        n_struct=1,
        root_check=True,
    )
    res = L._cross_check_rescaled(sf, out, kw, t0=0.0)
    assert not res.gap_certified
    assert res.stats.get("milp/rescale_cross_check_skipped") == 1.0
    assert res.status == ("feasible" if status == "optimal" else "error")


def test_rescaled_point_is_still_reverified_on_the_callers_form(monkeypatch):
    """#1414's class through the forced rescaling (the scenario
    ``test_1537_...::test_rescaled_logical_point_is_reverified_on_the_callers_form``
    used to reach unforced): ``x = 10, z = 0`` passes the scaled slack's tolerance and
    must be refused on the original form."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    monkeypatch.setattr(L, "logical_column_scales", _legacy_scales)
    m = dm.Model("bigm")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.subject_to(x - 1e8 * z <= 0)
    m.minimize(-x + 3 * z)
    r = m.solve(time_limit=30)
    st = r.solver_stats or {}
    assert st.get("milp/logicals_rescale_refused") == 1.0, st
    assert r.objective is None or r.objective >= -7.0 - 1e-6, (r.status, r.objective)
