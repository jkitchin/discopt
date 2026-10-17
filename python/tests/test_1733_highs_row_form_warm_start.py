"""#1733: the HiGHS MILP route certified a warm start as optimal (a false bound), and
lost the optimum on the same MILP cold.

The model is an epsilon-constraint step: a facility-location MILP whose emissions are
held at their minimum plus 1e-6 (``em <= e* + 1e-6``), so the feasible set is the
emission-optimal assignments. The enumerated cost optimum is 363.3962. On main the
cold solve returned 403.3962 (uncertified), and a solve warm-started from the
emission-optimal point certified 433.3962 with ``bound == incumbent``.

Root cause: HiGHS's MIP solver mis-solves the route's explicit-slack standard form on
this class. Replayed on the route's ``StdForm``, HiGHS returns ``kOptimal`` 403.3962
with the route's presolve rules and ``kInfeasible`` with presolve off; the same model
with the slacks passed as row bounds solves to 363.3962 both ways. On 100 seeds of
the generator (shift 1e-6), slack form was wrong on 25 with presolve and 8 without;
row form on none (600 runs). The warm start then turned the wrong ``kOptimal`` into a
certificate, because the #1634 presolve-free cross-check took the same seed and
agreed with it.

Two fixes: the route folds logical columns into row bounds
(``lp_milp_highs._eliminate_logicals``, ``DISCOPT_MILP_ROW_FORM``, default ON), and
the cross-check no longer receives the user's warm start.
"""

from __future__ import annotations

import itertools

import discopt as do
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import lp_milp_highs as L

OPT = 363.396200211421


def _model():
    rng = np.random.default_rng(7)
    nS, nC = 5, 7
    sites, cust = rng.uniform(0, 10, (nS, 2)), rng.uniform(0, 10, (nC, 2))
    dist = np.linalg.norm(sites[:, None] - cust[None], axis=2)
    d = rng.integers(5, 15, nC).astype(float)
    phi = np.array([30.0, 40.0, 55.0, 70.0, 90.0])
    e = np.array([3.0, 2.4, 1.6, 1.0, 0.4])
    u = np.full(nS, 35.0)
    m = do.Model("fl")
    y = m.binary("y", shape=(nS,))
    x = m.binary("x", shape=(nS, nC))
    for i in range(nC):
        m.subject_to(sum(x[j, i] for j in range(nS)) == 1)
    for j in range(nS):
        m.subject_to(sum(d[i] * x[j, i] for i in range(nC)) <= u[j] * y[j])
    em = sum(e[j] * d[i] * x[j, i] for j in range(nS) for i in range(nC))
    m.minimize(em)
    r0 = m.solve()
    assert r0.status == "optimal" and abs(r0.objective - 45.0) < 1e-6
    m.subject_to(em <= r0.objective + 1e-6)
    m.minimize(
        sum(phi[j] * y[j] for j in range(nS))
        + sum(0.5 * dist[j, i] * d[i] * x[j, i] for j in range(nS) for i in range(nC))
    )
    return m, {y: r0.x["y"], x: r0.x["x"]}


def _enumerated_optimum():
    """The oracle, independent of any solver: every assignment of 7 customers to 5
    sites, emissions in integer units of 0.2."""
    rng = np.random.default_rng(7)
    sites, cust = rng.uniform(0, 10, (5, 2)), rng.uniform(0, 10, (7, 2))
    dist = np.linalg.norm(sites[:, None] - cust[None], axis=2)
    d = rng.integers(5, 15, 7)
    phi = np.array([30.0, 40.0, 55.0, 70.0, 90.0])
    e5 = np.array([15, 12, 8, 5, 2])
    best = {}
    for a in itertools.product(range(5), repeat=7):
        load = np.bincount(a, weights=d, minlength=5)
        if (load > 35).any():
            continue
        em5 = int(sum(e5[a[i]] * d[i] for i in range(7)))
        cost = phi[load > 0].sum() + sum(0.5 * dist[a[i], i] * d[i] for i in range(7))
        best[em5] = min(best.get(em5, np.inf), cost)
    return best[min(best)]


@pytest.mark.slow
def test_oracle_is_the_pinned_optimum():
    assert abs(_enumerated_optimum() - OPT) < 1e-9


@pytest.mark.parametrize("warm", [False, True])
def test_issue_repro_certifies_the_true_optimum(warm):
    m, ws = _model()
    r = m.solve(initial_solution=ws) if warm else m.solve()
    if r.solver_stats["highs/version"] < L.ROW_FORM_MIN_HIGHS:
        # HiGHS 1.10/1.11 keep the slack form (see ROW_FORM_MIN_HIGHS); there the
        # contract is no false certificate, not a certificate. Measured: 1.10 certifies
        # 363.3962; 1.11 labels the model infeasible, which the cross-form check
        # refutes, so the optimum is reported uncertified.
        assert r.bound is None or r.bound <= OPT + 1e-6 * OPT, (r.status, r.bound)
        if r.status == "optimal":
            assert abs(r.objective - OPT) <= 1e-6 * OPT
        assert r.solver_stats.get("milp/row_form_skipped_old_highs") == 1.0
        return
    assert r.status == "optimal", (r.status, r.objective, r.bound, r.algorithm_route)
    assert abs(r.objective - OPT) <= 1e-6 * OPT
    assert r.bound <= OPT + 1e-6 * OPT
    assert r.solver_stats.get("milp/logicals_as_row_bounds") == 6.0


def test_warm_start_cannot_certify_on_the_slack_form(monkeypatch):
    """The opt-out reaches the HiGHS behaviour the issue measured; the cross-check,
    no longer handed the seed, must then decline rather than confirm 433.3962."""
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    m, ws = _model()
    r = m.solve(initial_solution=ws)
    assert "milp/logicals_as_row_bounds" not in (r.solver_stats or {})
    assert r.bound is None or r.bound <= OPT + 1e-6 * OPT, (r.status, r.objective, r.bound)
    if r.status == "optimal":
        assert abs(r.objective - OPT) <= 1e-6 * OPT


def _sf():
    # rows: x0 + 2 x1 + s0 = 4 (s0 in [0, inf)); x0 - x1 - 3 s1 = -1 (s1 in [-1, 2]);
    # x0 + x1 + s2 + s3 = 3 (two logicals in one row: neither is eliminated).
    A = sp.csc_matrix(
        np.array(
            [
                [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, -3.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )
    )
    return L.StdForm.from_arrays(
        c=[1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        A=A,
        b=[4.0, -1.0, 3.0],
        xl=[0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        xu=[5.0, 5.0, np.inf, 2.0, 1.0, 1.0],
        int_idx=[0, 1],
    )


def test_eliminate_logicals_is_exact():
    sf = _sf()
    red, lo, hi, rf = L._eliminate_logicals(sf, sf.b.copy(), sf.b.copy(), n_struct=2)
    assert rf is not None
    assert rf.cols.tolist() == [2, 3] and rf.rows.tolist() == [0, 1]
    assert red.n == 4 and red.int_idx.tolist() == [0, 1]
    # x0 + 2 x1 <= 4; x0 - x1 = -1 + 3 s1 in [-4, 5]; row 2 stays an equality
    assert lo[0] == -np.inf and hi[0] == 4.0
    # Nonzero subtracted terms are rounded outward, by one ulp.
    assert -4.0 - 1e-12 < lo[1] <= -4.0 and 5.0 <= hi[1] < 5.0 + 1e-12
    assert lo[2] == hi[2] == 3.0
    # Exactness: for every integer point, the ranged rows admit it iff some value of
    # the eliminated logicals satisfies sf; the lifted point then is sf-feasible.
    n_checked = 0
    for x0, x1 in itertools.product(range(6), repeat=2):
        for s2, s3 in itertools.product((0.0, 0.5, 1.0), repeat=2):
            xk = np.array([x0, x1, s2, s3], dtype=float)
            r = red.A @ xk
            in_range = bool(np.all(r >= lo - 1e-12) & np.all(r <= hi + 1e-12))
            full = L._lift_logicals(xk, sf, rf)
            feasible = L.feasibility_problem(full, sf, check_integrality=True) is None
            assert in_range == feasible, (x0, x1, s2, s3, full)
            n_checked += 1
    assert n_checked == 324


def test_eliminate_logicals_keeps_structural_columns():
    sf = _sf()
    # Every column a candidate when n_struct is None; with n_struct=6 none is.
    assert L._eliminate_logicals(sf, sf.b, sf.b, n_struct=6)[3] is None
    # With n_struct=None the integer model columns are still not logicals, and the
    # row holding two logicals keeps both.
    red, *_ = L._eliminate_logicals(sf, sf.b, sf.b, n_struct=None)
    assert red.n == 4


def test_falsifier_witnesses_are_right_on_the_row_form():
    """The #1640 and #1621 falsifier tests now pin the slack form (their witnesses
    need HiGHS to be wrong first). On the default row form the same models must come
    out right without any falsifier firing."""
    import sys
    from pathlib import Path

    import discopt.modeling as dm

    sys.path.insert(0, str(Path(__file__).parent))
    from test_1640_cross_check_tolerance_bound import _log_max

    m, truth = _log_max()
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.objective == pytest.approx(truth, abs=1e-6)
    assert r.bound >= truth - 1e-9  # maximize

    m = dm.Model("bigm")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.subject_to(x - 1e8 * z <= 0)
    m.minimize(-x + 3 * z)
    r = m.solve(time_limit=30)
    assert r.status == "optimal" and r.objective == pytest.approx(-7.0, abs=1e-6)
    assert r.bound <= -7.0 + 1e-6


@pytest.mark.parametrize("coef", [0.5, -0.5, 2.0, -2.0])
def test_sentinel_side_opens_the_row_end(coef):
    """The open-bound sentinel is the finite 1e20: ``0.5 * 1e20`` must not become a
    finite row end (CLAUDE.md: test unboundedness on the bound, not the product)."""
    A = sp.csc_matrix(np.array([[1.0, 1.0, coef]]))
    sf = L.StdForm.from_arrays(
        c=[1.0, 1.0, 0.0], A=A, b=[3.0], xl=[0.0, 0.0, 0.0], xu=[5.0, 5.0, L.INF],
        int_idx=[0, 1],
    )  # fmt: skip
    red, lo, hi, rf = L._eliminate_logicals(sf, sf.b.copy(), sf.b.copy(), n_struct=2)
    assert rf is not None and red.n == 2
    # coef * s in [0, +open) for coef > 0, (-open, 0] for coef < 0.
    if coef > 0:
        assert lo[0] == -np.inf and hi[0] == 3.0
    else:
        assert lo[0] == 3.0 and hi[0] == np.inf


def test_old_highs_keeps_the_slack_form(monkeypatch):
    """HiGHS 1.10/1.11 return a false kInfeasible on the #1229 row form (fixed in
    1.12), so older versions are handed the slack form."""
    monkeypatch.setattr(L, "_version_number", lambda h: 11100.0)
    m, _ = _model()
    r = m.solve()
    st = r.solver_stats
    assert st.get("milp/row_form_skipped_old_highs") == 1.0
    assert "milp/logicals_as_row_bounds" not in st
    assert r.bound is None or r.bound <= OPT + 1e-6 * OPT


def test_infeasible_label_is_refuted_by_the_other_form(monkeypatch):
    """A HiGHS-only infeasible label is held against the other form: a verified point
    there refutes it (#1733; measured on HiGHS 1.10/1.11, forced here on any version
    by making the row-form solve report kInfeasible with no point)."""
    real = L._solve_milp_std
    calls = []

    def lying(sf, **kw):
        out = real(sf, **kw)
        calls.append(kw.get("row_form"))
        if kw.get("row_form") is None and out.status == "optimal":
            out = L.HighsOutcome(
                "infeasible", gap_certified=True, highs_status="kInfeasible",
                root_bound=out.root_bound, stats=dict(out.stats),
                labels={**out.labels, "milp/infeasible_provenance": "highs"},
            )  # fmt: skip
        return out

    monkeypatch.setattr(L, "_solve_milp_std", lying)
    m, _ = _model()
    r = m.solve()
    assert {True, False} & set(calls), calls  # the other form was solved as the falsifier
    st = r.solver_stats
    assert st.get("milp/infeasible_label_refuted") == 1.0
    assert r.status in ("optimal", "feasible") and r.objective is not None
    assert r.objective >= OPT - 1e-6 * OPT
    assert r.bound is None or r.bound <= OPT + 1e-6 * OPT


def test_tolerance_shaded_incumbent_is_rederived(monkeypatch):
    """HiGHS (random_seed 16) published y = f(x) + 1.0e-6 on test_piecewise's incremental
    graph: its objective beat the LP over its own integers by row tolerance alone. The
    route now publishes the LP's verified point, which is exact (CI saw the same point
    on the default seed, a CPU-dependent search path)."""
    import discopt.modeling as dm

    real = L._new_highs

    def seeded(highspy, opts):
        h = real(highspy, opts)
        h.setOptionValue("random_seed", 16)
        return h

    monkeypatch.setattr(L, "_new_highs", seeded)
    b = [0.0, 1.0, 2.5, 3.0, 4.0, 6.0]
    v = [1.0, 3.0, -0.5, 2.0, 2.0, -1.0]
    t = 1.83822
    m = dm.Model()
    x = m.continuous("x", lb=t, ub=t)
    y = m.piecewise(x, b, v, method="incremental")
    m.maximize(y)
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert float(r.value(y)) == pytest.approx(float(np.interp(t, b, v)), abs=1e-8)
