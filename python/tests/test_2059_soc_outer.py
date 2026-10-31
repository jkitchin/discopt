"""#2059: Ben-Tal--Nemirovski outer approximation of convex norm-ball rows.

``_relax/soc_outer.py`` recognises a convex quadratic row whose linear part lies on
its quadratic support as a ball ``||L x + v|| <= rho`` and gives the HiGHS
``lp_nlp_bb`` master a lifted polyhedral OUTER approximation of it
(``DISCOPT_OA_SOC_OUTER``, default ON). These tests pin:

* soundness of the block: every point of the ball extends to a point satisfying
  every row (random points, several dimensions, off-centre balls);
* its accuracy: the LP reach of the polyhedron is within ``1/cos(pi/2**(nu+1))``
  per tower level of the ball;
* the recogniser's gates (convex, ball-shaped, scalar, at least two terms);
* the master wiring on a small cardinality portfolio, and the issue's instance
  (``slow``): portfol_classical050_1 certifies at 120 s.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import soc_outer as SO

FLAG = "DISCOPT_OA_SOC_OUTER"
DATA = Path(__file__).parent / "data" / "issue_2059"


def _ball(n, rng, v_scale=0.1):
    A = rng.normal(size=(n, n))
    v = rng.normal(size=n) * v_scale
    row = SO.NormBallRow(
        row=0, L_idx=[np.arange(n)] * n, L_val=[A[k] for k in range(n)], v=v, rho=1.0
    )
    return A, v, SO.SocOuterBlock([row])


def _worst_violation(block, x):
    vals = block.start_values(x)
    worst = -np.inf
    for xt, bt, rhs in block.ub_rows:
        lhs = sum(c * x[i] for i, c in xt.items()) + sum(c * vals[i] for i, c in bt.items())
        worst = max(worst, lhs - rhs)
    return worst, vals


@pytest.mark.parametrize("n", [2, 3, 5, 16])
def test_every_point_of_the_ball_extends_to_the_polyhedron(n):
    rng = np.random.default_rng(n)
    A, v, block = _ball(n, rng)
    checked = 0
    for _ in range(300):
        d = rng.normal(size=n)
        u = d / np.linalg.norm(d) * rng.uniform(0.0, 1.0)
        x = np.linalg.solve(A, u - v)
        worst, vals = _worst_violation(block, x)
        assert worst <= 1e-9, worst
        assert np.all(vals >= -1e-15)
        checked += 1
    # Points ON the sphere are the tight case.
    for _ in range(100):
        d = rng.normal(size=n)
        x = np.linalg.solve(A, d / np.linalg.norm(d) - v)
        assert _worst_violation(block, x)[0] <= 1e-9
        checked += 1
    assert checked == 400


@pytest.mark.parametrize("n", [2, 7, 16])
def test_polyhedron_reach_matches_the_ben_tal_nemirovski_accuracy(n):
    highspy = pytest.importorskip("highspy")
    rng = np.random.default_rng(100 + n)
    A, v, block = _ball(n, rng)
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    N = n + block.n_cols
    h.addVars(N, np.r_[np.full(n, -inf), np.zeros(block.n_cols)], np.full(N, inf))
    for xt, bt, rhs in block.ub_rows:
        idx = list(xt) + [n + c for c in bt]
        val = list(xt.values()) + list(bt.values())
        h.addRow(-inf, rhs, len(idx), np.array(idx, np.int32), np.array(val))
    depth = math.ceil(math.log2(n))
    limit = 1.0 / math.cos(math.pi / 2 ** (SO.LEVELS + 1)) ** depth
    reach = 0.0
    for _ in range(30):
        d = rng.normal(size=n)
        d /= np.linalg.norm(d)
        h.changeColsCost(n, np.arange(n, dtype=np.int32), -(A.T @ d))
        h.run()
        x = np.array(h.getSolution().col_value)[:n]
        reach = max(reach, float(np.linalg.norm(A @ x + v)))
    assert 1.0 <= reach <= limit + 1e-9, (reach, limit)


def _rows(m):
    from discopt._tape_nlp_evaluator import make_evaluator

    ev = make_evaluator(m)
    n = sum(v.size for v in m._variables)
    lb = np.concatenate([np.ravel(v.lb) for v in m._variables]).astype(float)
    ub = np.concatenate([np.ravel(v.ub) for v in m._variables]).astype(float)
    assert lb.size == n
    return SO.norm_ball_rows(m, ev.constraint_row_map(), lb, ub)


def _model():
    m = dm.Model("balls")
    x = m.continuous("x", shape=(3,), lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x[0] + x[1] + x[2] + y)
    return m, x, y


def test_recogniser_accepts_balls_and_completes_the_square():
    m, x, y = _model()
    m.subject_to(x[0] ** 2 + 2 * x[1] ** 2 <= 4.0)  # ellipse at the origin
    # Off-centre ball (x0 - 1)**2 + (x2 + 2)**2 <= 1, written expanded: the sparse
    # extractor is exact-or-abstain and does not expand a power of a sum.
    m.subject_to(x[0] * x[0] - 2 * x[0] + x[2] * x[2] + 4 * x[2] + 4.0 <= 0.0)
    m.subject_to(-(x[1] ** 2) - x[2] ** 2 >= -9.0)  # ">=" form
    rows = _rows(m)
    assert len(rows) == 3
    assert rows[0].rho == pytest.approx(2.0, rel=1e-9)
    assert rows[1].rho == pytest.approx(1.0, rel=1e-6)
    assert rows[2].rho == pytest.approx(3.0, rel=1e-9)
    # The completed square reproduces the ball: ||L x + v|| at the centre is 0.
    r = rows[1]
    centre = np.array([1.0, 0.0, -2.0, 0.0])
    u = [float(val @ centre[idx] + r.v[k]) for k, (idx, val) in enumerate(zip(r.L_idx, r.L_val))]
    assert np.linalg.norm(u) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x[0] ** 2 - x[1] ** 2 <= 1.0,  # nonconvex
        lambda x, y: x[0] ** 2 + x[1] ** 2 + y <= 1.0,  # linear part off the support
        lambda x, y: x[0] ** 2 <= 1.0,  # one term: the plain tangent again
        lambda x, y: x[0] ** 2 + x[1] ** 2 == 1.0,  # equality
        lambda x, y: x[0] ** 2 + x[1] ** 2 >= 1.0,  # reverse-convex
        lambda x, y: x[0] ** 2 + x[1] ** 2 <= -1.0,  # empty ball
        lambda x, y: (x[0] - 1) ** 2 + x[1] ** 2 <= 1.0,  # the extractor abstains
    ],
    ids=["nonconvex", "off_support", "one_term", "equality", "reverse", "empty", "unexpanded"],
)
def test_recogniser_declines_rows_that_are_not_convex_balls(build):
    m, x, y = _model()
    m.subject_to(build(x, y))
    assert _rows(m) == []


@pytest.mark.parametrize(
    "raw, expected",
    [(None, True), ("0", False), ("1", True), ("on", True)],
)
def test_flag_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, raw)
    assert SO.soc_outer_enabled() is expected


def _card_portfolio(n=8, K=3, seed=4):
    rng = np.random.default_rng(seed)
    mu = np.round(rng.uniform(0.02, 0.15, n), 4)
    S = np.round(np.eye(n) * rng.uniform(0.2, 0.45, n) + rng.normal(0, 0.02, (n, n)), 4)
    m = dm.Model("card_ball")
    w = m.continuous("w", shape=(n,), lb=0.0, ub=1.0)
    z = m.binary("z", shape=(n,))
    x = m.continuous("x", shape=(n,), lb=-10.0, ub=10.0)
    for k in range(n):
        m.subject_to(x[k] == dm.sum(lambda j: float(S[k, j]) * w[j], over=range(n)))
    m.subject_to(dm.sum(lambda k: x[k] ** 2, over=range(n)) <= 0.04)
    m.subject_to(dm.sum(lambda j: w[j], over=range(n)) == 1.0)
    for j in range(n):
        m.subject_to(w[j] <= z[j])
    m.subject_to(dm.sum(lambda j: z[j], over=range(n)) <= K)
    m.maximize(dm.sum(lambda j: float(mu[j]) * w[j], over=range(n)))
    return m


def test_master_carries_the_block_and_certifies(monkeypatch):
    pytest.importorskip("highspy")
    from discopt.solvers.mip_nlp import solve_mip_nlp

    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[flag] = solve_mip_nlp(
                _card_portfolio(), method="lp_nlp_bb", milp_solver="highs", time_limit=60
            )
    off, on = out["0"], out["1"]
    stats = on.solver_stats or {}
    assert stats.get("oa/soc_outer_rows") == 1.0, stats  # the probe fired (section 6)
    assert stats.get("oa/soc_outer_cols", 0) > 0
    # Linear objective: master points inside the approximation are accepted.
    assert stats.get("oa/soc_outer_accepted", 0) > 0, stats
    assert "oa/soc_outer_rows" not in (off.solver_stats or {})
    for r in (off, on):
        assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert on.objective == pytest.approx(off.objective, rel=1e-6)
    # Maximisation: a dual bound is an upper bound and may not pass the optimum.
    assert on.bound >= on.objective - 1e-9


@pytest.mark.slow
@pytest.mark.correctness
def test_issue_2059_portfol_classical050_1_certifies_at_120s(monkeypatch):
    """The issue's instance: uncertified at 120 s before #2059 (route bound
    -0.096817 withdrawn, fallback -0.097767; optimum -0.0947601179)."""
    pytest.importorskip("highspy")
    from discopt.modeling.core import from_nl
    from discopt.validation.feasibility import verify_point

    monkeypatch.delenv(FLAG, raising=False)  # the default
    nl = DATA / "portfol_classical050_1.nl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = from_nl(str(nl)).solve(time_limit=120)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(-0.0947601179, rel=1e-4)
    assert r.bound <= -0.0947601179 + 1e-6
    fresh = from_nl(str(nl))
    flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in fresh._variables])
    assert verify_point(fresh, flat).ok


def test_a_covered_row_does_not_also_get_the_split_row_epigraph(monkeypatch):
    """#2059 x #2021: with both blocks on, the outer approximation supersedes the
    #2021 split-row epigraph on the rows it covers (performance-plan 103.15), and
    the answer is unchanged."""
    pytest.importorskip("highspy")
    from discopt.solvers.mip_nlp import solve_mip_nlp

    monkeypatch.setenv("DISCOPT_PERSPECTIVE_ROW_DISAGG", "1")
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[flag] = solve_mip_nlp(
                _card_portfolio(), method="lp_nlp_bb", milp_solver="highs", time_limit=60
            )
    off, on = out["0"], out["1"]
    assert (off.solver_stats or {}).get("oa/perspective_row_disagg_cols", 0) > 0
    on_stats = on.solver_stats or {}
    assert on_stats.get("oa/soc_outer_rows") == 1.0
    assert "oa/perspective_row_disagg_cols" not in on_stats
    assert "oa/soc_outer_accepted" in on_stats  # the acceptance counter is exported
    for r in (off, on):
        assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert on.objective == pytest.approx(off.objective, rel=1e-6)


def test_no_master_point_is_accepted_under_a_nonlinear_objective(monkeypatch):
    """With an objective epigraph the master's value under-estimates the objective
    at a point, so accepting it would let HiGHS prune with a value nothing attains
    (measured on nvs12: 0.6 -> 5.1 s and a fallback). The block is still built."""
    pytest.importorskip("highspy")
    from discopt.solvers.mip_nlp import solve_mip_nlp

    monkeypatch.setenv(FLAG, "1")
    m = _card_portfolio()
    w = next(v for v in m._variables if v.name == "w")
    m.minimize(dm.sum(lambda j: (w[j] - 0.3) ** 2, over=range(w.size)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(m, method="lp_nlp_bb", milp_solver="highs", time_limit=60)
    stats = r.solver_stats or {}
    assert stats.get("oa/soc_outer_rows") == 1.0, stats
    assert stats.get("oa/soc_outer_accepted") == 0.0, stats
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)


@pytest.mark.parametrize(
    "Q",
    [
        np.diag([1.0, 1.0, -1e-10]),  # review of #2064: a slightly negative diagonal
        np.array([[1.0, 0.5], [0.5, -1e-3]]),  # clearly indefinite
    ],
    ids=["diag_negative", "indefinite"],
)
def test_factor_declines_what_it_cannot_prove(Q):
    """The first version dropped eigenvalues in ``[-margin, 0)`` uncharged, and the
    ball of ``x1**2 + x2**2 - 1e-10 x3**2 <= 1`` cut off the feasible point
    ``(0.9, 0.9, 1e5)``."""
    assert SO._factor(Q, 1e-9 * max(1.0, float(np.abs(Q).max()))) is None


def _L(rows, n):
    L = np.zeros((len(rows), n))
    for t, (idx, val) in enumerate(rows):
        L[t, idx] = val
    return L


@pytest.mark.parametrize(
    "Q, shifted",
    [
        (np.array([[2.0, 0.5], [0.5, 1.0]]), False),  # well-conditioned PD: no shift
        (np.array([[1.0, 1.0], [1.0, 1.0]]), True),  # singular PSD (rank one)
        (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1e-5], [0.0, 1e-5, -1e-12]]), True),
    ],
    ids=["pd", "singular", "slightly_indefinite"],
)
def test_factor_proves_l_transpose_l_below_q_plus_shift(Q, shifted):
    margin = 1e-9 * max(1.0, float(np.abs(Q).max()))
    out = SO._factor(Q, margin)
    assert out is not None
    rows, shift = out
    assert (shift > 0.0) is shifted
    L = _L(rows, Q.shape[0])
    assert np.linalg.eigvalsh(Q + shift * np.eye(Q.shape[0]) - L.T @ L).min() > 0.0


def test_factor_proves_l_transpose_l_below_q_on_random_pd():
    rng = np.random.default_rng(7)
    checked = 0
    for n in (2, 5, 9):
        for _ in range(40):
            A = rng.normal(size=(n, n))
            Q = A.T @ A + 1e-3 * np.eye(n)
            out = SO._factor(Q, 1e-9 * max(1.0, float(np.abs(Q).max())))
            assert out is not None and out[1] == 0.0
            L = _L(out[0], n)
            assert np.linalg.eigvalsh(Q - L.T @ L).min() > 0.0
            checked += 1
    assert checked == 120


def _rank_one_model(ub):
    m = dm.Model("factor_model")
    x = m.continuous("x", shape=(3,), lb=-ub, ub=ub)
    m.minimize(-x[0] - x[1] - x[2])
    # (x0 + x1 + x2)**2 + x2**2 <= 4, written expanded (the extractor does not
    # expand a product of sums): Q = [[1,1,1],[1,1,1],[1,1,2]], singular along x0 - x1.
    s01, s02, s12 = x[0] * x[1], x[0] * x[2], x[1] * x[2]
    m.subject_to(
        x[0] * x[0] + x[1] * x[1] + 2.0 * x[2] * x[2] + 2.0 * s01 + 2.0 * s02 + 2.0 * s12 <= 4.0
    )
    return m


def test_a_rank_deficient_row_needs_a_bounded_box_and_stays_sound():
    """Review of #2064, item 1: a singular PSD row keeps the block when its support
    is bounded, the shift charged to the radius; never on an unbounded one."""
    assert _rows(_rank_one_model(1e20)) == []
    rows = _rows(_rank_one_model(3.0))
    assert len(rows) == 1
    r = rows[0]
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(4000):
        x = rng.uniform(-3.0, 3.0, 3)
        if (x.sum()) ** 2 + x[2] ** 2 > 4.0:
            continue  # not feasible for the row
        u = [float(val @ x[idx] + r.v[k]) for k, (idx, val) in enumerate(zip(r.L_idx, r.L_val))]
        assert np.linalg.norm(u) <= r.rho * (1.0 + 1e-12), (x, np.linalg.norm(u), r.rho)
        checked += 1
    assert checked > 500


def test_no_point_is_accepted_when_the_fixed_nlp_fails(monkeypatch):
    """Review of #2064: acceptance must not keep an assignment the fixed NLP did
    not realise, or HiGHS prunes against a value no point attains."""
    pytest.importorskip("highspy")
    from discopt.solvers import oa
    from discopt.solvers.mip_nlp import solve_mip_nlp

    real = oa._solve_fixed_nlp_subproblem_attempt
    calls = []

    def failing(*a, **k):
        r = real(*a, **k)
        calls.append(1)
        r.x, r.objective = None, None
        return r

    monkeypatch.setattr(oa, "_solve_fixed_nlp_subproblem_attempt", failing)
    monkeypatch.setenv(FLAG, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(_card_portfolio(), method="lp_nlp_bb", milp_solver="highs", time_limit=20)
    stats = r.solver_stats or {}
    assert calls, "the fixed NLP was never called (section 6)"
    assert stats.get("oa/soc_outer_rows") == 1.0, stats
    assert stats.get("oa/soc_outer_accepted") == 0.0, stats


@pytest.mark.parametrize("path", ["rejection", "lp_rounds"])
def test_rows_held_at_an_accepted_point_reach_the_next_tree(monkeypatch, path):
    """Review of #2064, item 3: rows an accepted point would have returned are held,
    then delivered with the next rejection (or the LP rounds before the next tree).

    A scripted master drives ``lazy_callback`` with two points of the cardinality
    portfolio: A sits just outside the risk ball with every linear row exact (so it
    is accepted), B puts all weight on one asset, far outside the ball (rejected).
    """
    pytest.importorskip("highspy")
    from discopt.solvers import MILPResult, SolveStatus, milp_highs
    from discopt.solvers.mip_nlp import solve_mip_nlp

    monkeypatch.setenv(FLAG, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = solve_mip_nlp(_card_portfolio(), method="lp_nlp_bb", milp_solver="highs")
    assert ref.status == "optimal"
    rng = np.random.default_rng(4)  # the _card_portfolio seed: rebuild S and mu
    n = 8
    rng.uniform(0.02, 0.15, n)
    S = np.round(np.eye(n) * rng.uniform(0.2, 0.45, n) + rng.normal(0, 0.02, (n, n)), 4)
    w_opt = np.clip(np.asarray(ref.x["w"], dtype=np.float64), 0.0, None)  # NLP roundoff
    w_opt /= w_opt.sum()
    z_opt = np.round(np.asarray(ref.x["z"], dtype=np.float64))
    support = np.flatnonzero(w_opt > 1e-6)
    assert support.size >= 2
    i, j = int(support[0]), int(support[1])

    def risk(w):
        return float(np.sum((S @ w) ** 2))

    d = np.zeros(n)
    d[i], d[j] = 1.0, -1.0
    sign = 1.0 if risk(w_opt + 1e-6 * d) > risk(w_opt) else -1.0
    lo, hi = 0.0, 1e-2
    for _ in range(60):  # risk just above 0.04, inside the outer approximation
        mid = 0.5 * (lo + hi)
        if risk(w_opt + sign * mid * d) < 0.04 * (1.0 + 2e-5):
            lo = mid
        else:
            hi = mid
    w_a = w_opt + sign * hi * d
    assert 0.04 < risk(w_a) < 0.04 * (1.0 + 1e-4) and np.all(w_a >= 0.0)
    k = int(np.argmax(np.sum(S**2, axis=0)))
    w_b = np.zeros(n)
    w_b[k] = 1.0
    z_b = np.zeros(n)
    z_b[k] = 1.0
    assert risk(w_b) > 0.08

    returns: list = []
    node_returns: list = []

    def point(w, z, n_master):
        x = np.zeros(n_master)
        x[:n], x[n : 2 * n], x[2 * n : 3 * n] = w, z, S @ w
        return x

    def scripted(**kw):
        n_master = len(kw["c"])
        returns.append(list(kw["lazy_callback"](point(w_a, z_opt, n_master))))
        if path == "lp_rounds":
            assert kw.get("node_callback") is not None, "the LP-round path is not wired"
            # At the optimum itself no row of its own is violated, so whatever the LP
            # rounds receive here is exactly the held rows.
            node_returns.append(list(kw["node_callback"](point(w_opt, z_opt, n_master))))
        returns.append(list(kw["lazy_callback"](point(w_b, z_b, n_master))))
        return MILPResult(status=SolveStatus.TIME_LIMIT, x=None, objective=None, bound=None)

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", scripted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(_card_portfolio(), method="lp_nlp_bb", milp_solver="highs")
    stats = r.solver_stats or {}
    assert stats.get("oa/soc_outer_accepted") == 1.0, stats
    held = int(stats["oa/soc_outer_deferred_held"])
    assert held > 0 and returns[0] == []  # A accepted, its rows held
    assert stats.get("oa/soc_outer_deferred_flushed") == float(held)
    if path == "lp_rounds":
        # Delivered before the next tree by the LP rounds, so B no longer carries them.
        assert len(node_returns[0]) == held
    else:
        # B's rejection carries A's held rows ahead of its own.
        assert len(returns[1]) > held
