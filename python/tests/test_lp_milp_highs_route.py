"""#1229 HiGHS LP/MILP route: verified contract (plan ``lp-milp-highs-routing-plan.md`` §3).

Every HiGHS label is backed by a check discopt runs itself: a readback guard, a
feasibility check on the model's standard form, an NS-safe bound for an LP optimum,
a Farkas ray for an LP infeasible, a primal ray plus a feasible point for an LP
unbounded. These tests pin each status branch and the guards, and compare the route
against the Rust route on instances where both certify.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

# highspy becomes a core dependency in P3 of the plan; until then an install
# without the extra skips, loudly, rather than failing on import.
pytest.importorskip("highspy")

from discopt.solvers import lp_milp_highs as H  # noqa: E402

INF = H.INF
DATA = Path(__file__).parent / "data" / "issue1229_issue2388_milp.npz"


@pytest.fixture
def highs(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")


@pytest.fixture
def rust(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")


def _on_highs_route(res) -> bool:
    return (res.solver_stats or {}).get("route/lp_milp_backend") == 1.0


def _scalar(d, name):
    return float(np.asarray(d[name]).reshape(()))


# ── verification kernels ────────────────────────────────────────────────


def test_farkas_ray_is_verified_in_either_orientation_and_refused_when_unsound():
    infeas = H.StdForm.from_arrays([1.0, 1.0], [[1.0, 1.0]], [30.0], [0, 0], [10, 10])
    assert H.farkas_verified(np.array([1.0]), infeas)
    assert H.farkas_verified(np.array([-1.0]), infeas)
    feas = H.StdForm.from_arrays([1.0, 1.0], [[1.0, 1.0]], [15.0], [0, 0], [10, 10])
    assert not H.farkas_verified(np.array([1.0]), feas)
    # An open side makes the box supremum infinite: no proof, whatever HiGHS says.
    open_side = H.StdForm.from_arrays([1.0, 1.0], [[1.0, 1.0]], [30.0], [0, 0], [10, INF])
    assert not H.farkas_verified(np.array([1.0]), open_side)


def test_farkas_refuses_a_ray_whose_contradiction_is_only_roundoff():
    """``r = Aᵀy`` is a floating-point column dot. At ``2**53`` the ``+1`` of column 0 is
    absorbed, so the computed ``Aᵀy`` is ``[-1, 1, 1]`` where the exact one is ``[0, 1, 1]``.
    That spurious ``-1`` dropped the box supremum from 4 to 3, and the old relative margin
    (``RAY_REL`` times the *rounded* total, 1.1e-8 here) was cleared by a gap the arithmetic
    had manufactured -- on a problem whose ``x* = (1, 2, 2)`` satisfies every row exactly.

    This is the LP ``infeasible`` label's only gate, and the ``farkas-root-lp`` promotion
    in ``solve_milp_std`` turns it into a certified MILP ``infeasible``, so a wrong accept
    here is a wrong certificate.
    """
    P = 2.0**53
    sf = H.StdForm.from_arrays(
        [0.0, 0.0, 0.0],
        [[P, 1.0, 0.0], [1.0, 0.0, 0.0], [-P, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [P + 2.0, 1.0, -P + 2.0, -1.0],
        [1.0, 2.0, 2.0], [1.0, 2.0, 2.0],
    )  # fmt: skip
    xstar = np.array([1.0, 2.0, 2.0])
    # The premise: x* is feasible exactly, with no tolerance doing any work.
    assert np.array_equal(sf.A @ xstar, sf.b)
    assert H.feasibility_problem(xstar, sf, check_integrality=False) is None
    assert not H.farkas_verified(np.array([1.0, 1.0, 1.0, 1.0]), sf)


def test_primal_ray_must_lie_in_the_recession_cone_and_descend():
    sf = H.StdForm.from_arrays([-1.0, -1.0], [[1.0, -1.0]], [0.0], [0, 0], [INF, INF])
    assert H.primal_ray_verified(np.array([1.0, 1.0]), sf)
    assert not H.primal_ray_verified(np.array([1.0, 0.0]), sf)  # A d != 0
    assert not H.primal_ray_verified(np.array([-1.0, -1.0]), sf)  # through a finite bound
    ascent = H.StdForm.from_arrays([1.0, 1.0], [[1.0, -1.0]], [0.0], [0, 0], [INF, INF])
    assert not H.primal_ray_verified(np.array([1.0, 1.0]), ascent)  # cᵀd > 0


def test_readback_refuses_the_sentinel_but_accepts_a_declared_huge_finite_bound():
    free = H.StdForm.from_arrays([0.0], np.zeros((0, 1)), [], [-INF], [INF])
    assert H.readback_problem(np.array([1e20]), free) is not None  # the #1229 class
    assert H.readback_problem(np.array([np.nan]), free) is not None
    assert H.readback_problem(np.array([1.0]), free, row_dual=np.array([1e16])) is not None
    box = H.StdForm.from_arrays([-1.0], np.zeros((0, 1)), [], [-9.999e19], [9.999e19])
    assert H.readback_problem(np.array([9.999e19]), box) is None
    assert H.readback_problem(np.array([5e19]), box) is not None  # huge but not at a bound


def test_feasibility_check_enforces_integrality_only_when_asked():
    sf = H.StdForm.from_arrays([1.0], [[1.0]], [0.5], [0], [1], int_idx=[0])
    assert H.feasibility_problem(np.array([0.5]), sf, check_integrality=True) is not None
    assert H.feasibility_problem(np.array([0.5]), sf, check_integrality=False) is None


def test_rows_at_the_huge_box_are_checked_exactly():
    """Float residuals cannot see a violation at ``9.999e19`` scale; exact ones can."""
    B = 9.999e19
    # x + y - s1 = 2, x + y + s2 = 1, s >= 0 is infeasible; the corner cancels in float.
    infeas = H.StdForm.from_arrays(
        [1.0, -1.0, 0.0, 0.0], [[1, 1, -1, 0], [1, 1, 0, 1]], [2.0, 1.0],
        [-B, -B, 0.0, 0.0], [B, B, INF, INF],
    )  # fmt: skip
    x = np.array([-B, B, 0.0, 0.0])
    assert float(np.abs(infeas.A @ x - infeas.b).max()) == 2.0  # visible, but forgiven by 1e-9·2e20
    assert H.feasibility_problem(x, infeas, check_integrality=False) is not None
    # x + y - s = 0, s >= 0 at x = B, y = 1: s = B + 1 is not a double, but it exists.
    corner = H.StdForm.from_arrays([-1.0, -1.0, 0.0], [[1, 1, -1]], [0.0], [-B, 0, 0], [B, 1, INF])
    xc = np.array([B, 1.0, B])
    assert H.feasibility_problem(xc, corner, check_integrality=False) is None
    # A costed column is not re-derivable (the objective depends on it): residual 1.
    costed = H.StdForm.from_arrays([-1.0, -1.0, 1.0], corner.A, [0.0], corner.xl, corner.xu)
    assert H.feasibility_problem(xc, costed, check_integrality=False) is not None


def test_huge_box_masks_each_side_separately_and_keeps_declared_bounds():
    """``_relax_huge_box`` re-derived the side to open from the *sign* of the bound rather
    than from a per-side mask, so a column with one sentinel-magnitude side had its other,
    ordinary declared side opened too: ``lb=-5`` beside a default ``ub`` became ``-inf``,
    HiGHS saw a free column and the MILP route answered ``error``. ``lb=0`` survived only
    because ``0 < 0`` is false. Opening a side is still a relaxation, so the fully free
    column (the F2 case) must keep opening both.
    """
    B = 9.999e19
    #                                free   lb=-5/default ub   default lb/ub=5   finite
    sf = H.StdForm.from_arrays(
        [0.0, 0.0, 0.0, 0.0], np.zeros((0, 4)), [],
        [-B, -5.0, -B, 0.0], [B, B, 5.0, 10.0],
    )  # fmt: skip
    lo, hi = H._huge_box(sf)
    assert lo.tolist() == [True, False, True, False]
    assert hi.tolist() == [True, True, False, False]
    r = H._relax_huge_box(sf, lo, hi)
    assert r.xl.tolist() == [-INF, -5.0, -INF, 0.0]  # the declared -5 and 0 are kept
    assert r.xu.tolist() == [INF, INF, 5.0, 10.0]  # the declared 5 and 10 are kept
    # Every side that moved, moved outward -- the result is a relaxation of ``sf``.
    assert np.all(r.xl <= sf.xl) and np.all(r.xu >= sf.xu)


def test_ns_bound_is_valid_from_an_arbitrary_dual():
    # min x0 + 2 x1, x0 + x1 = 3, 0 <= x <= 10: optimum 3. Any y gives a lower bound.
    #
    # The bound carries `refine.rs::NS_MARGIN_REL * S` of slack, `S = 1 + |bᵀy| +
    # Σ|contrib|` (#1230: without it the evaluation lands *above* the optimum on
    # ~21% of LPs). So the soundness assertions below take NO tolerance, and the
    # sharpness assertions allow exactly the margin.
    from discopt._rust import NS_MARGIN_REL

    sf = H.StdForm.from_arrays([1.0, 2.0], [[1.0, 1.0]], [3.0], [0, 0], [10, 10])
    # y=1: bᵀy = 3, rc = (0, 1) -> contrib 1*0 = 0, so S = 4.
    assert H.ns_bound(np.array([1.0]), sf) == pytest.approx(3.0, abs=4.0 * NS_MARGIN_REL)
    for y in (-5.0, 0.0, 0.3, 4.0):
        g = H.ns_bound(np.array([y]), sf)
        assert g is not None and g <= 3.0
    open_sf = H.StdForm.from_arrays([1.0, 2.0], [[1.0, 1.0]], [3.0], [-INF, 0], [10, 10])
    # y=4: rc = (-3, -2) < 0 puts x0 on its finite upper side, so the bound is finite.
    # bᵀy = 12, contribs -30 and -20, so S = 63.
    assert H.ns_bound(np.array([4.0]), open_sf) == pytest.approx(
        12.0 - 30.0 - 20.0, abs=63.0 * NS_MARGIN_REL
    )
    assert H.ns_bound(np.array([4.0]), open_sf) <= 12.0 - 30.0 - 20.0  # sound, no slack
    assert H.ns_bound(np.array([0.0]), open_sf) is None  # rc0 = 1 > 0 on an open lower side


def test_ns_binding_rejects_a_malformed_csc():
    from discopt._rust import ns_safe_bound_csc_py

    f = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        ns_safe_bound_csc_py(
            np.array([1.0]), f, 1, 2,
            np.array([0, 1], dtype=np.int64),  # length n, not n+1
            np.array([0], dtype=np.int64), np.array([1.0]),
            np.array([3.0]), np.zeros(2), np.full(2, 10.0),
        )  # fmt: skip


def _recession_lp():
    # min x0  s.t.  x0 + 0.1 x1 - 0.1 x2 = 1, x0 in [0, 10], x1, x2 >= 0: optimum 0.
    # d = (0, 1, 1) is a zero-cost recession direction, so every optimal dual has
    # rc1 = rc2 = 0 exactly and any roundoff in y puts one of them on its open side.
    return H.StdForm.from_arrays(
        [1.0, 0.0, 0.0], [[1.0, 0.1, -0.1]], [1.0], [0.0, 0.0, 0.0], [10.0, INF, INF]
    )


def test_exact_dual_correction_recovers_a_bound_float_ns_cannot():
    sf = _recession_lp()
    y = np.array([1e-17])
    assert H.ns_bound(y, sf) is None
    assert H.ns_bound(y, H.fbbt_box(sf)) is None  # x1, x2 stay open: FBBT cannot help
    g, why = H.exact_ns_bound(y, sf)
    assert why == "" and g == 0.0


def test_exact_dual_correction_is_a_valid_bound_from_any_dual():
    sf = _recession_lp()
    rng = np.random.default_rng(0)
    finite = 0
    for y0 in np.r_[rng.normal(0.0, 1.0, 40), rng.normal(0.0, 1e-12, 40), 20.0, -3.0]:
        g, why = H.exact_ns_bound(np.array([y0]), sf)
        assert (g is None) == bool(why)
        if g is not None:
            finite += 1
            assert g <= 0.0
    assert finite > 0
    assert H.exact_ns_bound(np.array([np.nan]), sf)[0] is None


def test_exact_dual_correction_zeroes_a_sentinel_scale_side():
    """A finite ±9.999e19 side is as useless as an open one for a bound: roundoff
    reduced cost 1e-18 times 9.999e19 costs ~100 of bound (nlp_cvx_001_010 lost 1.9e3
    this way and stayed uncertified). The correction must zero it too."""
    B = 9.999e19
    sf = H.StdForm.from_arrays(
        [1.0, 0.0, 0.0], [[1.0, 0.1, -0.1]], [1.0], [0.0, 0.0, 0.0], [10.0, B, B]
    )
    y = np.array([1e-17])
    g_float = H.ns_bound(y, sf)
    assert g_float is not None and g_float < -50.0  # finite, valid, and far from 0
    g, why = H.exact_ns_bound(y, sf)
    assert why == "" and g == 0.0


def test_exact_elimination_is_capped_by_work_not_wall(monkeypatch):
    from fractions import Fraction as F

    M = [[F(2), F(1), F(1)], [F(1), F(3), F(2)], [F(1), F(0), F(0)]]
    rhs = [F(4), F(5), F(6)]
    z = H._exact_solve(M, rhs, None)
    assert z is not None
    assert [sum(M[i][k] * z[k] for k in range(3)) for i in range(3)] == rhs
    monkeypatch.setattr(H, "EXACT_MAX_WORK", 10)
    assert H._exact_solve(M, rhs, None) is None  # same input, same refusal, any machine


def test_recession_ray_is_verified_and_absent_for_a_bounded_lp():
    # min -x0 - x1 with no rows over x >= 0: HiGHS may report unbounded with no ray.
    sf = H.StdForm.from_arrays([-1.0, -1.0], np.zeros((0, 2)), [], [0.0, 0.0], [INF, INF])
    d = H.recession_ray(sf, time_limit=None)
    assert d is not None and H.primal_ray_verified(d, sf)
    out = H.solve_lp_std(sf)
    assert out.status == "unbounded"
    bounded = H.StdForm.from_arrays([-1.0], np.zeros((0, 1)), [], [0.0], [5.0])
    assert H.recession_ray(bounded, time_limit=None) is None


def test_fbbt_box_contains_the_feasible_set_under_huge_term_cancellation():
    # x0 + x1 - x2 = 1 with x0 up to 1e19 and x1, x2 open: a derived side of x1 subtracts
    # terms of size 1e19, where plain float cancellation is off by far more than 1e-9.
    B = 1e19
    sf = H.StdForm.from_arrays(
        [0.0, 0.0, 0.0], [[1.0, 1.0, -1.0], [0.0, 1.0, 0.0]], [1.0, 3.0],
        [0.0, -INF, 0.0], [B, INF, INF],
    )  # fmt: skip
    box = H.fbbt_box(sf)
    assert np.isfinite(box.xl[1]) and box.xl[1] < INF  # x1 = 3 from row 2
    for x0 in (0.0, 1.0, B):
        x = np.array([x0, 3.0, x0 + 2.0])
        assert np.all(box.xl <= x) and np.all(x <= box.xu)
    assert box.xu[2] >= B + 2.0


def _certificate_instance(name):
    d = np.load(Path(__file__).parent / "data" / "lp_highs_certificate_instances.npz")
    g = lambda k: d[f"{name}__{k}"]  # noqa: E731
    A = sp.csc_matrix((g("data"), g("indices"), g("indptr")), shape=tuple(g("shape")))
    return H.StdForm.from_arrays(
        g("c"), A, g("b"), g("xl"), g("xu"), _scalar(d, f"{name}__obj_const")
    )


def test_zero_cost_recession_lp_certifies_by_exact_dual_correction():
    """e226 (netlib): HiGHS's float dual leaves NS at -inf even over the FBBT box."""
    out = H.solve_lp_std(_certificate_instance("e226"))
    assert out.status == "optimal" and out.gap_certified
    assert out.labels["lp/bound_provenance"] == "ns-exact-dual-correction"
    assert out.bound <= out.objective
    assert out.objective == pytest.approx(-11.638929066370533, abs=1e-6)


@pytest.mark.parametrize(
    "name, provenance", [("klein1", "phase1-exact-dual-correction"), ("forest6", "farkas-fbbt-box")]
)
def test_infeasible_lp_without_a_declared_box_farkas_ray_is_proved(name, provenance):
    out = H.solve_lp_std(_certificate_instance(name))
    assert out.status == "infeasible" and out.gap_certified
    assert out.labels["lp/infeasible_provenance"] == provenance


# ── LP route through Model.solve ────────────────────────────────────────


def _mixed_lp():
    m = dm.Model("mixed_lp")
    x = m.continuous("x", lb=0.0, ub=8.0)
    y = m.continuous("y", lb=0.0, ub=8.0)
    z = m.continuous("z", lb=0.0, ub=8.0)
    m.maximize(3 * x + 2 * y + z)
    m.subject_to(x + y + z <= 10, name="cap")
    m.subject_to(x - y >= -2, name="bal")
    m.subject_to(x + z == 6, name="link")
    return m


def test_lp_optimal_is_certified_and_matches_the_rust_route(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    ref = _mixed_lp().solve()
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    res = _mixed_lp().solve()
    assert ref.status == "optimal" and not _on_highs_route(ref)
    assert res.status == "optimal" and _on_highs_route(res)
    assert res.gap_certified
    assert res.objective == pytest.approx(ref.objective, abs=1e-7)
    assert res.bound is not None and res.bound >= res.objective - 1e-9  # max sense
    assert res.solver_stats["lp/ns_gap"] <= 1e-6
    for name in ("x", "y", "z"):
        assert _scalar(res.x, name) == pytest.approx(_scalar(ref.x, name), abs=1e-7)
    checked = 0
    for fam in ("constraint_duals", "bound_duals_lower", "bound_duals_upper"):
        got, want = getattr(res, fam), getattr(ref, fam)
        assert got is not None and want is not None and set(got) == set(want)
        for k in want:
            assert _scalar(got, k) == pytest.approx(_scalar(want, k), abs=1e-7), (fam, k)
            checked += 1
    assert checked == 9


def test_lp_infeasible_needs_a_verified_farkas_ray(highs):
    # Free columns: bound propagation cannot tighten them, so the contradiction
    # between the two rows reaches the route instead of being caught by presolve.
    m = dm.Model("lp_infeasible")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x - y)
    m.subject_to(x + y >= 2, name="lo")
    m.subject_to(x + y <= 1, name="hi")
    res = m.solve()
    assert res.status == "infeasible" and _on_highs_route(res)


def test_lp_unbounded_needs_a_verified_ray_and_point(highs):
    m = dm.Model("lp_unbounded")
    x = m.continuous("x", lb=0.0, ub=np.inf)
    y = m.continuous("y", lb=0.0, ub=np.inf)
    m.minimize(-x - y)
    m.subject_to(x - y == 0, name="c")
    res = m.solve()
    assert res.status == "unbounded" and _on_highs_route(res)


def test_lp_default_box_corner_is_optimal_not_error(highs):
    """``min -x`` on the default ±9.999e19 box (#850/#937) survives the readback guard."""
    m = dm.Model("corner")
    x = m.continuous("x")
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(-x - y)
    m.subject_to(x + y >= 0, name="c")
    res = m.solve()
    assert _on_highs_route(res)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-9.999e19 - 1.0, rel=1e-12)


@pytest.mark.parametrize("build", [_mixed_lp, lambda: _knapsack()], ids=["lp", "milp"])
def test_highs_route_skips_discopt_root_presolve(monkeypatch, build):
    """Plan §12 H5: HiGHS presolves itself; discopt's root presolve is pure wall there."""
    import discopt._relax.presolve_pipeline as pp

    calls = []
    real = pp.run_root_presolve

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pp, "run_root_presolve", counting)
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    ref = build().solve()
    rust_calls = len(calls)
    assert rust_calls > 0  # the probe fires on the route that keeps presolve
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    res = build().solve()
    assert _on_highs_route(res) and res.status == "optimal" and res.gap_certified
    assert len(calls) == rust_calls
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)


@pytest.mark.parametrize("build", [_mixed_lp, lambda: _knapsack()], ids=["lp", "milp"])
def test_highs_route_runs_after_another_highs_user_in_the_process(highs, build):
    """HiGHS keeps one process-wide thread scheduler, sized by its first run.

    The OA/GDP paths and user code run highspy with default options. A route that
    pins a different ``threads`` value is refused by HiGHS for the rest of the
    process ("global scheduler has already been initialized"), so a later LP/MILP
    solve returned ``error``.
    """
    import highspy

    highspy.Highs.resetGlobalScheduler(True)
    other = highspy.Highs()
    other.setOptionValue("output_flag", False)
    other.setOptionValue("threads", 2)  # a scheduler size the route did not choose
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = 1, 0
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = np.array([1.0]), np.array([0.0]), np.array([1.0])
    lp.integrality_ = [highspy.HighsVarType.kInteger]
    other.passModel(lp)
    assert other.run() == highspy.HighsStatus.kOk
    del other
    res = build().solve()
    assert _on_highs_route(res)
    assert res.status == "optimal" and res.gap_certified


def test_unknown_backend_value_is_refused(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "higs")
    with pytest.raises(ValueError, match="DISCOPT_LP_MILP_BACKEND"):
        _mixed_lp().solve()


# ── MILP route ──────────────────────────────────────────────────────────


def _knapsack(n=12, seed=3):
    rng = np.random.default_rng(seed)
    w = rng.integers(5, 40, size=n)
    v = rng.integers(5, 60, size=n)
    m = dm.Model("knap")
    xs = [m.binary(f"b{i}") for i in range(n)]
    m.maximize(sum(int(v[i]) * xs[i] for i in range(n)))
    m.subject_to(sum(int(w[i]) * xs[i] for i in range(n)) <= int(w.sum() // 2), name="cap")
    return m


def test_milp_optimal_matches_the_rust_route(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    ref = _knapsack().solve()
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    res = _knapsack().solve()
    assert ref.status == "optimal" and res.status == "optimal"
    assert _on_highs_route(res) and not _on_highs_route(ref)
    assert res.gap_certified
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.bound >= res.objective - 1e-6  # max sense
    assert res.solver_stats["milp/bound_provenance_highs_fp"] == 1.0
    assert res.solver_stats["milp/root_check_ran"] == 1.0
    assert res.root_bound is not None and res.root_bound >= res.bound - 1e-6
    assert "bound_provenance=highs-fp" in res.algorithm_route


def test_milp_node_limit_is_never_certified(highs):
    rng = np.random.default_rng(11)
    n, k = 40, 5
    W = rng.integers(10, 100, size=(k, n))
    v = rng.integers(10, 100, size=n)
    m = dm.Model("mknap")
    xs = [m.binary(f"b{i}") for i in range(n)]
    m.maximize(sum(int(v[i]) * xs[i] for i in range(n)))
    for r in range(k):
        m.subject_to(sum(int(W[r, i]) * xs[i] for i in range(n)) <= int(W[r].sum() // 3))
    res = m.solve(max_nodes=1)
    assert _on_highs_route(res)
    if res.status == "optimal":
        pytest.skip("HiGHS closed this instance at the root; no limit branch exercised")
    assert res.status in ("feasible", "node_limit")
    assert not res.gap_certified
    if res.objective is not None and res.bound is not None:
        assert res.bound >= res.objective - 1e-6


def test_milp_infeasible_by_the_root_lp_carries_farkas_provenance(highs):
    m = dm.Model("milp_lp_infeasible")
    a = m.integer("a", lb=-1000, ub=1000)
    b = m.integer("b", lb=-1000, ub=1000)
    m.minimize(a - b)
    m.subject_to(a + b >= 2, name="lo")
    m.subject_to(a + b <= 1, name="hi")
    res = m.solve()
    assert res.status == "infeasible" and _on_highs_route(res)
    assert res.solver_stats.get("milp/infeasible_provenance_farkas") == 1.0


def test_milp_on_the_huge_box_gets_a_farkas_proof_not_a_bare_label(highs):
    """The MILP analogue of the huge-box LP: free columns, contradictory rows.

    HiGHS's MILP presolve answers ``kInfeasible`` by itself here (probed on four
    variants). What the exact check changes is the proof: the root LP used to come
    back "optimal" at the cancelled corner, leaving the claim on HiGHS's label; now
    the root LP yields a Farkas ray discopt verifies.
    """
    m = dm.Model("milp_huge_infeasible")
    x = m.continuous("x")
    y = m.continuous("y")
    b = m.binary("b")
    m.minimize(x - y + b)
    m.subject_to(x + y >= 2 + b, name="lo")
    m.subject_to(x + y <= 1, name="hi")
    res = m.solve()
    assert _on_highs_route(res)
    assert res.status == "infeasible" and res.gap_certified
    assert res.solver_stats.get("milp/infeasible_provenance_farkas") == 1.0


def test_milp_incumbent_that_fails_verification_is_an_error(highs, monkeypatch):
    """An incumbent discopt cannot verify is never returned. With a feasible root LP
    nothing is proven either way, so the answer is ``error``, not a certificate."""
    real = H.feasibility_problem
    refused = []

    def refuse_integer_points(x, sf, check_integrality):
        if check_integrality:
            refused.append(1)
            return "refused by the test"
        return real(x, sf, check_integrality)

    monkeypatch.setattr(H, "feasibility_problem", refuse_integer_points)
    res = _knapsack().solve()
    assert refused, "the incumbent check never ran"
    assert _on_highs_route(res)
    assert res.status == "error" and not res.gap_certified
    assert res.objective is None
    assert res.solver_stats.get("milp/root_check_ran") == 1.0


def test_milp_integer_infeasible_is_certified_with_highs_provenance(highs):
    m = dm.Model("milp_int_infeasible")
    x = m.integer("x", lb=0, ub=5)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x + y)
    m.subject_to(2 * x == 1, name="odd")
    res = m.solve()
    assert res.status == "infeasible" and _on_highs_route(res)
    assert res.gap_certified
    assert res.solver_stats.get("milp/infeasible_provenance_highs") == 1.0
    assert "milp/infeasible_provenance_farkas" not in res.solver_stats


def test_milp_on_the_default_box_is_not_falsely_infeasible(highs):
    """HiGHS MIP given the finite ±9.999e19 box of the free column ``y`` answered
    ``kInfeasible``, and the route certified it. Feasible: optimum 4.0 at x=(-1, 2, 1,
    3, 3), y=0 (scipy and the Rust route agree). Found by adversarial testing."""
    m = dm.Model("milp_default_box")
    lo, hi = [-1, 1, 0, 0, 3], [-1, 8, 2, 4, 3]
    x = [m.integer(f"x{j}", lb=lo[j], ub=hi[j]) for j in range(5)]
    y = m.continuous("y")
    m.minimize(-4 * x[0] + 4 * x[1] - 4 * x[2] - x[3] + 5 * y - 1)
    m.subject_to(-3 * x[2] + 2 * x[3] + 5 * x[4] - 5 * y == 18)
    m.subject_to(-x[0] + 5 * x[1] + x[2] + 4 * x[3] + x[4] <= 27)
    m.subject_to(-3 * x[0] - x[1] + 3 * x[2] + x[3] - x[4] + y >= 4)
    m.subject_to(2 * x[0] - 3 * x[1] + 5 * x[2] <= -2.8685039722160535)
    m.subject_to(x[0] - 2 * x[1] - 3 * x[2] - 3 * x[4] - y <= -17)
    m.subject_to(4 * x[0] + 3 * x[1] + 5 * x[2] - 5 * x[3] - 3 * x[4] <= -17)
    res = m.solve(time_limit=30)
    assert _on_highs_route(res)
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(4.0, abs=1e-6)
    assert res.solver_stats.get("milp/huge_box_relaxed") == 1.0


def test_milp_declared_lower_bound_beside_a_default_side_survives(highs):
    """A column with one sentinel-magnitude side and one *ordinary declared* side had the
    declared side opened too: ``_huge_box`` flagged the column for its default ``ub`` and
    ``_relax_huge_box`` re-derived the side from the sign, so ``x >= -5`` was discarded.
    HiGHS then saw a free column, returned ``kUnboundedOrInfeasible``, and the route
    answered ``error`` where the declared box has an optimum (the Rust route says -5.0).

    The F2 test above cannot catch this: its column ``y`` is fully free, which is exactly
    the case where opening both sides is harmless. ``lb=0`` escaped on ``0 < 0``.
    """
    m = dm.Model("milp_one_sided_lb")
    x = m.continuous("x", lb=-5.0)
    z = m.binary("z")
    m.subject_to(x + 3 * z <= 10, name="cap")
    m.minimize(x)
    res = m.solve(time_limit=20)
    assert _on_highs_route(res)
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(-5.0, abs=1e-6)


def test_milp_declared_upper_bound_beside_a_default_side_survives(highs):
    """Mirror of the lower-bound case: a finite declared ``ub`` beside a default ``lb``."""
    m = dm.Model("milp_one_sided_ub")
    x = m.continuous("x", ub=5.0)
    z = m.binary("z")
    m.subject_to(x + 3 * z >= -10, name="cap")
    m.maximize(x)
    res = m.solve(time_limit=20)
    assert _on_highs_route(res)
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(5.0, abs=1e-6)


def test_lp_coefficient_at_highs_drop_threshold_is_kept(highs):
    """HiGHS drops ``|a| <= small_matrix_value`` (default 1e-9) and passModel only warns;
    the route raised on that warning. ``1e-9 x >= 1e-9`` is ``x >= 1``."""
    m = dm.Model("lp_tiny_coef")
    x = m.continuous("x", lb=0.0)
    m.subject_to(1e-9 * x >= 1e-9)
    m.minimize(x)
    res = m.solve(time_limit=20)
    assert _on_highs_route(res)
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(1.0, abs=1e-6)


def test_row_rhs_at_the_infinity_sentinel_is_an_error_not_a_crash(highs):
    """HiGHS refuses a row bound >= 1e20 (passModel kError); the route raised
    ``RuntimeError`` out of ``solve``. It is now an uncertified ``error`` result."""
    m = dm.Model("lp_rhs_sentinel")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(2 * x <= 1.94849311702961e20)
    m.minimize(-x)
    res = m.solve(time_limit=20)
    assert _on_highs_route(res)
    assert res.status == "error" and not res.gap_certified
    assert res.objective is None


def test_milp_highs_would_perturb_gets_no_answer_from_it(highs):
    """Below ``small_matrix_value`` HiGHS solves a different MILP, and neither its
    infeasible label nor its tree bound can be re-derived for the real one."""
    m = dm.Model("milp_sub_threshold_coef")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + 1e-13 * y >= 1)
    m.minimize(x + y)
    res = m.solve(time_limit=20)
    assert _on_highs_route(res)
    assert res.status == "error" and not res.gap_certified
    assert res.objective is None


def test_issue_1229_instance_returns_a_feasible_point_on_the_highs_route():
    """The #1229 regression guard on the HiGHS route (plan §11)."""
    d = np.load(DATA)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["shape"]))
    m_rows, n = A.shape

    def _sent(v, s):
        return np.where(np.isfinite(v), v, s * INF)

    sf = H.StdForm.from_arrays(
        np.concatenate([d["c"], np.zeros(m_rows)]),
        sp.hstack([A, -sp.identity(m_rows)]).tocsc(),
        np.zeros(m_rows),
        np.concatenate([_sent(d["col_lo"], -1), _sent(d["row_lo"], -1)]),
        np.concatenate([_sent(d["col_hi"], 1), _sent(d["row_hi"], 1)]),
        int_idx=np.flatnonzero(d["is_int"]),
    )
    out = H.solve_milp_std(sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=10**6)
    assert out.status == "optimal", out.message
    x = out.x[:n]
    assert np.max(np.abs(x)) < 1e15
    rows = 0
    for i in range(m_rows):
        s, e = A.indptr[i], A.indptr[i + 1]
        act = float(np.dot(A.data[s:e], x[A.indices[s:e]]))
        assert d["row_lo"][i] - 1e-6 <= act <= d["row_hi"][i] + 1e-6
        rows += 1
    assert rows == m_rows > 0
    assert out.objective == pytest.approx(0.0, abs=1e-6)


_DEFAULT_ROUTE_PROBE = """
import sys
import discopt.modeling as dm
m = dm.Model("probe")
kind = sys.argv[1]
x = m.continuous("x", lb=0.5, ub=4)
y = m.integer("y", lb=0, ub=3)
if kind == "minlp":
    m.minimize(x * y - x)
    m.subject_to(x * y >= 1)
else:
    m.minimize(x - 2 * y)
    m.subject_to(x + y <= 4)
r = m.solve(time_limit=30)
route = (r.solver_stats or {}).get("route/lp_milp_backend")
print("RESULT", r.status, route, "highspy" in sys.modules)
"""


@pytest.mark.parametrize("kind", ["minlp", "milp"])
def test_default_route_imports_highspy_only_for_pure_lp_milp(kind):
    """Plan §4: a default MINLP solve never imports highspy; a pure MILP takes HiGHS.

    A fresh interpreter with the flag unset, so the graduated default is what runs and
    nothing earlier in the test process has imported highspy.
    """
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_LP_MILP_BACKEND"}
    out = subprocess.run(
        [sys.executable, "-c", _DEFAULT_ROUTE_PROBE, kind],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    lines = [ln.split() for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, out.stdout + out.stderr
    _, status, route, imported = lines[0]
    assert status == "optimal"
    if kind == "minlp":
        assert (route, imported) == ("None", "False")
    else:
        assert (route, imported) == ("1.0", "True")
