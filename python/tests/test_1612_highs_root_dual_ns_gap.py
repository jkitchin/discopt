"""#1612: HiGHS kOptimal downgraded to ``feasible`` by the #1410 root-dual check.

The #1410 guard measures each reduced cost's violation relative to its own scale
``|c_j| + (|A|ᵀ|y|)_j``. On a slack column (``c_j = 0``) whose row dual is round-off zero
that scale IS round-off, so a ``1e-14`` wrong-signed dual reads as a relative violation of
``1.0`` and a perfectly solved set-cover LP voids the MILP certificate. A knapsack's
``6.8e-14`` violation tripped the same check against a ``4.3e-14`` round-off bound.

The fix keeps the per-column test as a sufficient condition and, when it fails, asks the
rigorous question instead: does HiGHS's own root pair certify the root LP? The
Neumaier--Shcherbina bound from HiGHS's duals over the declared box (after zeroing a
wrong-signed reduced cost on an open singleton column, which NS cannot charge) must meet
HiGHS's primal objective within the route's LP certificate yardstick. NS is a valid
bound for any dual, so the check cannot be fooled by the repair; on the #1410 matrix,
which HiGHS mis-solves, the gap is ``0.022`` and the certificate is still refused.
"""

import discopt.modeling as dm
import discopt.solvers.lp_milp_highs as H
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(autouse=True)
def _pin_highs_engine(monkeypatch):
    """These tests exercise the HiGHS engine behind ``solve_lp_std``.
    Since highs-parity P6.4 (2026-11-07) the default engine is the in-house
    one, so these HiGHS-path tests pin ``DISCOPT_LP_STD_ENGINE=highs`` (a test that sets it still
    overrides)."""
    monkeypatch.setenv("DISCOPT_LP_STD_ENGINE", "highs")


pytest.importorskip("highspy")


def _set_cover(seed: int):
    rng = np.random.default_rng(seed)
    cover = np.zeros((120, 60), int)
    for i in range(120):
        cover[i, rng.choice(60, 3, replace=False)] = 1
    w = rng.integers(2, 5, 60).astype(float)
    m = dm.Model("sc")
    y = m.binary("y", shape=(60,))
    for i in range(120):
        m.subject_to(dm.sum(y[j] for j in range(60) if cover[i, j]) >= 1)
    m.minimize(dm.sum(w[j] * y[j] for j in range(60)))
    return m, cover, w


def _set_cover_truth(cover, w) -> float:
    from scipy.optimize import Bounds, LinearConstraint, milp

    r = milp(
        w, constraints=LinearConstraint(cover, lb=1), integrality=np.ones(60), bounds=Bounds(0, 1)
    )
    assert r.status == 0
    return float(r.fun)


_U = [0.47368421052634424, 0.5263157894736549, 0.18421052631579893, 0.5263157894736558,
      0.3684210526315981, 0.39473684210530025, 0.44736842105265995, 0.3157894736841994,
      0.6315789473684019, 0.21052631578945913, 0.31578947368420496, 0.36842105263161784,
      0.421052631578969, 0.36842105263160474, 0.2631578947368435, 0.21052631578950132,
      0.18421052631579915, 0.21052631578950987, 0.26315789473683326, 0.6315789473684019,
      0.26315789473680895, 0.47368421052634513, 0.5263157894737045, 0.2631578947367854,
      0.31578947368418187, 0.3684210526316132, 0.2894736842104966, 0.6052631578946998,
      0.2631578947368375, 0.578947368421031]  # fmt: skip
_WT = [47, 53, 20, 53, 39, 41, 45, 31, 60, 22, 31, 35, 43, 36, 25, 21, 20, 21, 26, 60,
       27, 46, 50, 29, 31, 37, 30, 59, 27, 56]  # fmt: skip


def _knapsack():
    m = dm.Model("knap")
    a = m.binary("a", shape=(30,))
    m.maximize(dm.sum(lambda j: _U[j] * a[j], over=range(30)))
    m.subject_to(dm.sum(lambda j: float(_WT[j]) * a[j], over=range(30)) <= 100.0)
    m.subject_to(a[0] + a[1] <= 1)
    return m


def _knapsack_truth() -> float:
    """Exact DP over the integer weights; the side row ``a0 + a1 <= 1`` by cases."""

    def dp(items):
        best = np.full(101, -np.inf)
        best[0] = 0.0
        for j in items:
            for w in range(100, _WT[j] - 1, -1):
                best[w] = max(best[w], best[w - _WT[j]] + _U[j])
        return float(best.max())

    return max(dp([j for j in range(30) if j != 1]), dp([j for j in range(30) if j != 0]))


@pytest.mark.parametrize("seed", [2, 7])
def test_set_cover_witness_certifies_with_a_valid_bound(seed: int) -> None:
    m, cover, w = _set_cover(seed)
    truth = _set_cover_truth(cover, w)
    r = m.solve()
    st = r.solver_stats
    # The witness still trips the per-column test -- that is the case under test.
    assert st["milp/root_dual_violation"] > H._DUAL_CERT_SAFETY * 4 * np.finfo(float).eps
    assert st.get("milp/root_dual_certified_by_ns") == 1.0, st
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.bound <= truth + 1e-6, (r.bound, truth)
    assert r.objective == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("gap", [None, 1e-9])
def test_knapsack_witness_certifies_with_a_valid_bound(gap) -> None:
    truth = _knapsack_truth()
    r = _knapsack().solve(**({} if gap is None else {"gap_tolerance": gap}))
    assert r.solver_stats.get("milp/root_dual_certified_by_ns") == 1.0, r.solver_stats
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    # maximisation: the bound is an UPPER bound on the optimum
    assert r.bound >= truth - 1e-6, (r.bound, truth)
    assert r.objective == pytest.approx(truth, abs=1e-6)


def _slacked_lp(c, a, b, xu) -> "H.StdForm":
    m = a.shape[0]
    mat = sp.hstack([sp.csr_matrix(a), sp.identity(m, format="csr")], format="csc")
    return H.StdForm.from_arrays(
        np.concatenate([c, np.zeros(m)]),
        mat,
        b,
        np.zeros(a.shape[1] + m),
        np.concatenate([xu, np.full(m, H.INF)]),
    )


def _solved_lps(n_want: int):
    rng = np.random.default_rng(1612)
    got = 0
    while got < n_want:
        n, m = 6, 3
        c = rng.normal(size=n)
        a = rng.normal(size=(m, n))
        xu = rng.integers(1, 5, size=n).astype(float)
        b = a @ (xu / 2) + np.abs(a) @ (xu * 0.3)
        sf = _slacked_lp(c, a, b, xu)
        out = H.solve_lp_std(sf, time_limit=30.0)
        if out.status == "optimal" and out.x is not None and out.row_dual is not None:
            got += 1
            yield sf, out


def test_ns_gap_is_round_off_on_correct_pairs_and_refuses_perturbed_duals() -> None:
    """(b) A REAL dual violation is charged in full and refused; round-off is not."""
    checks = 0
    for sf, out in _solved_lps(20):
        obj = float(sf.c @ out.x) + sf.obj_const
        thr = H.CERT_ABS + H.CERT_REL * abs(obj)
        g = H.root_pair_ns_gap(sf, out.x, out.row_dual)
        assert g is not None and -1e-9 <= g <= thr, g
        checks += 1
        # Perturbed duals: shift every row dual by -1e-3. That is the direction the
        # slack repair cannot undo (a ``+e_i`` slack prices ``-y_i``, so a smaller
        # ``y_i`` keeps it sign-feasible); a ``+`` shift on an inactive row would be
        # repaired straight back to the correct dual, which is right but tests nothing.
        # Every structural column now carries a reduced-cost error charged over its box,
        # and every slack row a complementarity error.
        y_bad = np.array(out.row_dual, dtype=float) - 1e-3
        g_bad = H.root_pair_ns_gap(sf, out.x, y_bad)
        assert g_bad is None or g_bad > thr, (g_bad, thr)
        checks += 1
    assert checks == 40, "MEASURED NOTHING"


def test_ns_gap_refuses_a_wrong_optimum() -> None:
    """(b) A feasible but non-optimal primal point cannot pass, whatever dual it carries:
    the NS bound is a valid lower bound, so the gap is at least the suboptimality."""
    checks = 0
    for sf, out in _solved_lps(10):
        obj = float(sf.c @ out.x)
        n = sf.n - sf.m
        x0 = np.zeros(sf.n)  # structural 0, slacks = b (b >= 0 by construction here)
        x0[n:] = sf.b
        if np.any(x0[n:] < 0):
            continue
        sub = float(sf.c @ x0) - obj
        if sub <= 1e-3:
            continue
        g = H.root_pair_ns_gap(sf, x0, out.row_dual)
        assert g is None or g >= sub - 1e-9, (g, sub)
        checks += 1
    assert checks > 0, "MEASURED NOTHING"


def test_open_singleton_round_off_is_repaired_not_charged_as_infinite() -> None:
    """A slack whose dual is wrong-signed at 1e-14 makes the raw NS bound -inf; the
    repair zeroes that reduced cost and the gap stays at round-off."""
    checks = 0
    for sf, out in _solved_lps(10):
        n = sf.n - sf.m
        y = np.array(out.row_dual, dtype=float)
        # a slack column is ``+e_i`` with ``xu = inf``: d = -y_i must be >= 0, so a
        # positive y_i is the wrong sign. Inject it on a row whose dual is zero.
        zero_rows = np.flatnonzero(np.abs(y) < 1e-12)
        if zero_rows.size == 0:
            continue
        y[zero_rows[0]] = 1e-14
        assert H.ns_bound(y, sf) is None  # the raw bound cannot charge an open side
        obj = float(sf.c @ out.x)
        g = H.root_pair_ns_gap(sf, out.x, y)
        assert g is not None and g <= H.CERT_ABS + H.CERT_REL * abs(obj), g
        assert sf.c[n + zero_rows[0]] == 0.0
        checks += 1
    assert checks > 0, "MEASURED NOTHING"


def test_route_refuses_when_root_duals_are_perturbed(monkeypatch) -> None:
    """(b) End to end: corrupt the root LP's duals the route inspects; the witness that
    now certifies must be refused again, with the HiGHS tree bound withheld."""
    real = H.solve_lp_std
    fired = []

    def corrupt(sf, **kw):
        out = real(sf, **kw)
        if out.row_dual is not None and out.x is not None:
            y = np.array(out.row_dual, dtype=float)
            y[:] += 0.5  # a real violation on every binary column
            out.row_dual = y
            fired.append(1)
        return out

    monkeypatch.setattr(H, "solve_lp_std", corrupt)
    m, cover, w = _set_cover(2)
    truth = _set_cover_truth(cover, w)
    r = m.solve()
    assert fired, "MEASURED NOTHING: the root LP was never intercepted"
    assert r.solver_stats.get("milp/root_dual_unverified") == 1.0, r.solver_stats
    assert not r.gap_certified and r.status != "optimal"
    assert r.bound is None or r.bound <= truth + 1e-6


def test_1410_matrix_is_still_refused() -> None:
    """The class #1410 guards: HiGHS mis-solves the LP, and the NS gap says so."""
    from test_1410_highs_dual_certificate import _build, _enumerated_optimum

    truth = _enumerated_optimum()
    r = _build().solve(time_limit=120)
    st = r.solver_stats
    assert st.get("milp/root_dual_unverified") == 1.0, st
    assert st["milp/root_dual_ns_gap"] > 1e-3
    assert not r.gap_certified
    assert r.bound is None or r.bound <= truth + 1e-7
