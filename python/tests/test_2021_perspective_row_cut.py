"""#2021 section 3: the diagonal-perspective split of a coupled quadratic ``<=`` row.

A risk row ``x' Sigma x <= sigma2`` whose ``x`` is defined by linear ``==`` rows over
semicontinuous weights ``w`` (``0 <= w_i <= U z_i``, ``z_i`` binary) splits as
``M = (M - D) + D`` with ``M - D`` PSD; the OA tangent of the row then gains
``d_i w_i**2 (1 - z_i)`` (``_relax/perspective.perspective_row_terms``). Default
ON; ``DISCOPT_PERSPECTIVE_ROW_CUT=0`` opts out. Entry experiment and panel: performance-plan
section 103.10.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import perspective as P
from discopt.solvers import oa

FLAG = "DISCOPT_PERSPECTIVE_ROW_CUT"
SIGMA = np.array([[0.09, 0.03, 0.02], [0.03, 0.06, 0.025], [0.02, 0.025, 0.08]])
MU = np.array([0.12, 0.10, 0.08])
SIGMA2 = 0.06  # binds: asset 0 alone has variance 0.09


def _portfolio(w_lb=0.0, indicator=True, risk="quadratic", card=2):
    """Three assets, a cardinality budget, and a dense risk row over ``x = w``."""
    m = dm.Model("card3")
    w = [m.continuous(f"w{i}", lb=w_lb, ub=1.0) for i in range(3)]
    z = [m.binary(f"z{i}") for i in range(3)]
    x = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(3)]
    for i in range(3):
        m.subject_to(x[i] - w[i] == 0.0)
        if indicator:
            m.subject_to(w[i] - z[i] <= 0.0)
    m.subject_to(sum(w) == 1.0)
    m.subject_to(sum(z) <= card)
    quad = sum(SIGMA[i, j] * x[i] * x[j] for i in range(3) for j in range(3))
    if risk == "quadratic":
        m.subject_to(quad <= SIGMA2)
    else:
        m.subject_to(dm.exp(x[0]) + quad <= SIGMA2 + 1.0)
    m.maximize(sum(MU[i] * w[i] for i in range(3)))
    return m


def _terms(m):
    from discopt._tape_nlp_evaluator import make_evaluator

    ev = make_evaluator(m)
    return ev, P.perspective_row_terms(m, ev.constraint_row_map())


def _flat(m, name):
    from discopt._relax.term_classifier import _compute_var_offset

    v = next(v for v in m._variables if v.name == name)
    return _compute_var_offset(v, m)


@pytest.mark.smoke
def test_the_flag_is_on_by_default(monkeypatch):
    """Graduated on its panels (performance-plan section 103.10); ``=0`` opts out."""
    monkeypatch.delenv(FLAG, raising=False)
    assert P.perspective_row_cut_enabled()
    assert oa._perspective_row_cut_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not P.perspective_row_cut_enabled()
    assert not oa._perspective_row_cut_enabled()


@pytest.mark.smoke
def test_detector_returns_the_certified_diagonal_split():
    m = _portfolio()
    ev, terms = _terms(m)
    assert len(terms) == 1
    (t,) = terms
    # the risk row is the last scalar constraint
    risk_row = [s for s, _e, c in ev.constraint_row_map() if c is m._constraints[-1]]
    assert t.row == risk_row[0]
    # x = w exactly, so M = 0.5 F' (2 Sigma) F = Sigma (up to the column order of W)
    order = np.argsort([_flat(m, f"w{i}") for i in range(3)])
    assert [int(c) for c in t.W] == sorted(_flat(m, f"w{i}") for i in range(3))
    sig = SIGMA[np.ix_(order, order)]
    dg = np.diag(sig)
    # largest t with Sigma - t diag(Sigma) PSD is the smallest generalized eigenvalue
    t_star = float(np.linalg.eigvalsh(sig / np.sqrt(np.outer(dg, dg)))[0])
    assert t.t == pytest.approx(0.999 * t_star, rel=1e-6)
    np.testing.assert_allclose(t.d, t.t * dg, rtol=1e-12)
    assert np.linalg.eigvalsh(sig - np.diag(t.d))[0] >= 0.0
    assert sorted(int(c) for c in t.z) == sorted(_flat(m, f"z{i}") for i in range(3))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kwargs",
    [
        {"w_lb": -1.0},  # w can go negative: z = 0 no longer pins it to 0
        {"indicator": False},  # no switch-off row
        {"risk": "exp"},  # the row is not quadratic
    ],
)
def test_detector_declines(kwargs):
    _ev, terms = _terms(_portfolio(**kwargs))
    assert terms == []


def _sample_feasible(m, rng, k):
    """``k`` feasible points of the 3-asset model (``card = 3`` so ``z`` is free)."""
    pts = []
    n = sum(v.size for v in m._variables)
    for _ in range(100 * k):
        if len(pts) == k:
            break
        z = rng.integers(0, 2, size=3)
        if not z.any():
            continue
        w = rng.random(3) * z
        w = w / w.sum()
        if w @ SIGMA @ w > SIGMA2 + 1e-12:
            continue
        p = np.zeros(n)
        for i in range(3):
            p[_flat(m, f"w{i}")] = w[i]
            p[_flat(m, f"x{i}")] = w[i]
            p[_flat(m, f"z{i}")] = z[i]
        pts.append(p)
    assert len(pts) == k, len(pts)
    return pts


@pytest.mark.smoke
def test_no_feasible_point_violates_a_split_cut():
    m = _portfolio(card=3)
    _ev, (term,) = _terms(m)
    n = sum(v.size for v in m._variables)
    rng = np.random.default_rng(2021)
    feas = _sample_feasible(m, rng, 300)
    executed = 0
    shifted_cuts = 0
    for _ in range(200):
        # references anywhere in the box, including infeasible ones
        ref = rng.random(n) * 1.5 - 0.25
        coeffs, rhs, k = term.cut(ref, n)
        shifted_cuts += bool(k)
        for p in feas:
            assert coeffs @ p <= rhs + 1e-12, coeffs @ p - rhs
            executed += 1
    assert executed == 200 * 300
    assert shifted_cuts > 0


def _master_bound(m, term, refs, split):
    """min -mu'w over the linear rows with z relaxed to [0, 1], plus the risk row's
    cut at each reference: the plain tangent, or the split cut."""
    from scipy.optimize import linprog

    n = sum(v.size for v in m._variables)
    w = [_flat(m, f"w{i}") for i in range(3)]
    x = [_flat(m, f"x{i}") for i in range(3)]
    z = [_flat(m, f"z{i}") for i in range(3)]
    c = np.zeros(n)
    c[w] = -MU
    a_eq, b_eq, a_ub, b_ub = [], [], [], []
    for i in range(3):
        r = np.zeros(n)
        r[x[i]], r[w[i]] = 1.0, -1.0
        a_eq.append(r)
        b_eq.append(0.0)
        r = np.zeros(n)
        r[w[i]], r[z[i]] = 1.0, -1.0
        a_ub.append(r)
        b_ub.append(0.0)
    r = np.zeros(n)
    r[w] = 1.0
    a_eq.append(r)
    b_eq.append(1.0)
    r = np.zeros(n)
    r[z] = 1.0
    a_ub.append(r)
    b_ub.append(2.0)
    for ref in refs:
        if split:
            coeffs, rhs, _k = term.cut(ref, n)
        else:
            coeffs = term.plain_tangent(ref, n)
            xs = ref[term.S]
            rhs = -term.k + 0.5 * float(xs @ term.QSS @ xs)
        a_ub.append(coeffs)
        b_ub.append(rhs)
    bounds = [(0.0, 1.0)] * n
    for i in range(3):
        bounds[x[i]] = (-1.0, 1.0)
    res = linprog(c, A_ub=np.array(a_ub), b_ub=b_ub, A_eq=np.array(a_eq), b_eq=b_eq, bounds=bounds)
    assert res.status == 0, res.message
    return -res.fun  # an upper bound on max mu'w


def _true_optimum():
    from scipy.optimize import minimize

    best = -np.inf
    executed = 0
    for zz in itertools.product((0, 1), repeat=3):
        if not 1 <= sum(zz) <= 2:
            continue
        act = [i for i in range(3) if zz[i]]
        r = minimize(
            lambda v: -MU[act] @ v,
            np.full(len(act), 1.0 / len(act)),
            constraints=[
                {"type": "eq", "fun": lambda v: v.sum() - 1.0},
                {"type": "ineq", "fun": lambda v: SIGMA2 - v @ SIGMA[np.ix_(act, act)] @ v},
            ],
            bounds=[(0.0, 1.0)] * len(act),
            method="SLSQP",
        )
        executed += 1
        if r.success and SIGMA2 - r.x @ SIGMA[np.ix_(act, act)] @ r.x >= -1e-6:
            best = max(best, float(MU[act] @ r.x))
    assert executed == 6
    # (w0, w1) = (2/3, 1/3) is the closed-form optimum: the risk row binds at w0 = 2/3
    assert best == pytest.approx(0.12 * 2 / 3 + 0.10 / 3, abs=1e-6)
    return best


@pytest.mark.smoke
def test_differential_master_bound():
    """Section 5: with the split the master bound is no looser, and still valid."""
    m = _portfolio()
    _ev, (term,) = _terms(m)
    n = sum(v.size for v in m._variables)
    rng = np.random.default_rng(7)
    opt = _true_optimum()
    executed = 0
    strictly = 0
    for _ in range(25):
        refs = [rng.random(n) for _ in range(4)]
        for ref in refs:
            # reference points that satisfy x = w, so the plain tangent is the
            # row's tangent at the same x^ the split uses
            for i in range(3):
                ref[_flat(m, f"x{i}")] = ref[_flat(m, f"w{i}")]
        plain = _master_bound(m, term, refs, split=False)
        split = _master_bound(m, term, refs, split=True)
        assert split <= plain + 1e-9
        assert split >= opt - 1e-7, (split, opt)
        strictly += split < plain - 1e-6
        executed += 1
    assert executed == 25
    assert strictly > 0


def _oa_rows(m, x_star, apply_identity, monkeypatch):
    if apply_identity:
        monkeypatch.setattr(
            oa,
            "_apply_perspective_row_cut",
            lambda ev, cid, coeffs, rhs, support, n_vars: (coeffs, rhs),
        )
    d = oa._decompose_model(m)
    rows_a: list = []
    rows_b: list = []
    senses = [c.sense for _s, _e, c in d.evaluator.constraint_row_map()]
    oa._add_oa_cuts(
        d.evaluator,
        x_star,
        d.n_vars,
        d.n_cons,
        senses,
        rows_a,
        rows_b,
        True,
        [True] * d.n_cons,
        True,
    )
    return d, rows_a, rows_b


@pytest.mark.smoke
def test_flag_off_rows_are_byte_identical(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    m = _portfolio()
    n = sum(v.size for v in m._variables)
    x_star = np.random.default_rng(3).random(n)
    d0, a0, b0 = _oa_rows(m, x_star, False, monkeypatch)
    assert not hasattr(d0.evaluator, "_perspective_row_terms")
    _d1, a1, b1 = _oa_rows(_portfolio(), x_star, True, monkeypatch)
    assert len(a0) == len(a1) > 0
    for r0, r1 in zip(a0, a1):
        assert np.asarray(r0).tobytes() == np.asarray(r1).tobytes()
    assert [float(b) for b in b0] == [float(b) for b in b1]


@pytest.mark.smoke
def test_flag_on_replaces_the_risk_row_tangent(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m = _portfolio()
    n = sum(v.size for v in m._variables)
    x_star = np.full(n, 0.5)
    d, a, b = _oa_rows(m, x_star, False, monkeypatch)
    state = d.evaluator._perspective_row_terms
    assert state.applied == 1 and state.id_mismatch == 0
    (term,) = state.by_row.values()
    coeffs, rhs, _k = term.cut(x_star, n)
    assert any(np.array_equal(r, coeffs) and float(rr) == rhs for r, rr in zip(a, b))


@pytest.mark.smoke
def test_a_row_id_that_names_another_row_is_refused():
    m = _portfolio()
    _ev, (term,) = _terms(m)
    n = sum(v.size for v in m._variables)
    state = oa._PerspectiveRowState([term])

    class _Ev:
        _perspective_row_terms = state

    x = np.full(n, 0.5)
    wrong = np.zeros(n)
    wrong[0] = 1.0
    out, rhs = oa._apply_perspective_row_cut(_Ev(), term.row, wrong, 3.0, x, n)
    assert out is wrong and rhs == 3.0
    assert state.id_mismatch == 1 and state.applied == 0
    out, _rhs = oa._apply_perspective_row_cut(_Ev(), term.row, term.plain_tangent(x, n), 0.0, x, n)
    assert state.applied == 1
    assert not np.array_equal(out, term.plain_tangent(x, n))


@pytest.mark.smoke
def test_the_split_is_counted_in_solver_stats_and_keeps_the_answer(monkeypatch):
    from discopt.solvers.oa import solve_lp_nlp_bb

    monkeypatch.setenv(FLAG, "0")
    off = solve_lp_nlp_bb(_portfolio(), time_limit=60.0, milp_solver="highs")
    monkeypatch.setenv(FLAG, "1")
    on = solve_lp_nlp_bb(_portfolio(), time_limit=60.0, milp_solver="highs")
    assert off.status == "optimal" and on.status == "optimal", (off.status, on.status)
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    assert on.objective == pytest.approx(_true_optimum(), abs=1e-5)
    assert (on.solver_stats or {}).get("oa/perspective_row_cuts", 0.0) > 0.0
    assert "oa/perspective_row_cuts" not in (off.solver_stats or {})
