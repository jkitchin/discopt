"""Regression tests for the remaining #1611 items: X-30c, D-30, D-31.

X-30c: ``AffineDecisionRule`` + budget set returned the static value for every
budget. The vector bound rows ``y0 + Y xi <= ub`` the rule adds were robustified
as ONE row, so the polyhedral counterpart gave all elements one shared dual set
and pinned every policy column to a common value. Each element must get its own
adversary.

D-30: a fully linear master with ``lazy_constraints`` (routed to spatial B&B) lost
its certificate when one node's NLP solve failed: the failure became a
non-rigorous fathom whose floor capped the dual bound (status ``feasible``, bound
354.7 vs optimum 410.3). Such a node is now re-solved as the verified HiGHS LP it
is: a Farkas certificate fathoms it rigorously, a safe bound replaces the sentinel.

D-31: GBD on a convex reactor MINLP stopped ``iteration_limit`` with bound None.
The master bound had in fact converged; the bound was withheld because the
convexity classifier could not prove ``-c*F/(F + a)`` convex. The collinear
linear-fractional quotient ``(alpha*u + beta)/u`` is now classified exactly.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.convexity.lattice import Curvature
from discopt._relax.convexity.rules import classify_expr, classify_oa_cut_convexity

# ─────────────────────────── X-30c ───────────────────────────

_CS = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
_FZ = np.array([2.0, 2.5])
_D = np.ones(3)
_DH = np.full(3, 0.3)
_PEN = 200.0


def _facility(gamma: float, adr: bool) -> float:
    from discopt.ro import AffineDecisionRule, RobustCounterpart, budget_uncertainty_set

    m = dm.Model("x30c")
    z = m.continuous("z", shape=(2,), lb=0, ub=5.0)
    x = m.continuous("x", shape=(6,), lb=0, ub=5.0)
    s = m.continuous("s", shape=(3,), lb=0, ub=5.0)
    d = m.parameter("d", value=_D)
    eta = m.continuous("eta", lb=0, ub=500.0)
    m.minimize(_FZ[0] * z[0] + _FZ[1] * z[1] + eta)
    m.subject_to(
        dm.sum(lambda q: float(_CS.ravel()[q]) * x[q], over=range(6))
        + dm.sum(lambda j: _PEN / 10 * s[j], over=range(3))
        <= eta
    )
    for j in range(3):
        m.subject_to(x[j] + x[3 + j] + s[j] >= d[j])
    for i in range(2):
        m.subject_to(dm.sum(lambda j: x[3 * i + j], over=range(3)) <= z[i])
    if adr:
        AffineDecisionRule(x, uncertain_params=d, prefix="ax").apply()
        AffineDecisionRule(s, uncertain_params=d, prefix="as").apply()
    RobustCounterpart(m, budget_uncertainty_set(d, delta=_DH, gamma=gamma)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    return float(r.objective)


@pytest.mark.parametrize(("gamma", "fully_adjustable"), [(1.0, 11.775), (2.0, 12.75)])
def test_x30c_affine_rule_adapts_under_budget_set(gamma, fully_adjustable):
    static = _facility(gamma, adr=False)
    affine = _facility(gamma, adr=True)
    assert static == pytest.approx(13.65, abs=1e-5)
    # The affine rule is a restriction of full adjustability and a relaxation of
    # the static policy; on this instance it attains the fully adjustable value
    # (checked by scenario enumeration of the budget set's vertices).
    assert affine == pytest.approx(fully_adjustable, abs=1e-5)
    assert affine < static - 0.5


def test_x30c_vector_uncertain_row_splits_per_element():
    from discopt.ro.counterpart import _split_vector_uncertain_rows

    m = dm.Model("split")
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    d = m.parameter("d", value=np.ones(3))
    m.subject_to(y + d <= np.array([2.0, 3.0, 4.0]), name="v")
    m.subject_to(dm.sum(y) <= 2.0, name="certain")
    out = _split_vector_uncertain_rows(m._constraints, {"d"})
    names = [c.name for c in out]
    assert names == ["v[0]", "v[1]", "v[2]", "certain"]
    # each element is its own scalar row with the matching right-hand side
    for con in out[:3]:
        assert con.sense == "<="
        assert np.ndim(con.rhs) == 0
        assert con.body.shape == ()
    rhs_vec = np.asarray(m._constraints[0].rhs, dtype=float)
    assert [float(c.rhs) for c in out[:3]] == [float(v) for v in np.broadcast_to(rhs_vec, (3,))]


# ─────────────────────────── D-30 ───────────────────────────


class _Pool:
    """Duck-typed CutPool: ``len`` and ``to_constraint_arrays``."""

    def __init__(self, A, b, senses):
        self._A, self._b, self._s = np.asarray(A, float), np.asarray(b, float), list(senses)

    def __len__(self):
        return len(self._s)

    def to_constraint_arrays(self):
        return self._A, self._b, self._s


def _box_lp():
    from discopt.solvers.lp_milp_highs import StdForm

    # min x0 + 2 x1  over 0 <= x <= 10, no equality rows.
    return StdForm.from_arrays(
        np.array([1.0, 2.0]), sp.csc_matrix((0, 2)), np.zeros(0), np.zeros(2), np.full(2, 10.0)
    )


def test_d30_rescue_bound_includes_cuts_and_node_box():
    from discopt.solver import _linear_node_lp_rescue

    pool = _Pool([[1.0, 1.0]], [3.0], [">="])
    kind, lb, x = _linear_node_lp_rescue(
        _box_lp(), 2, pool, np.array([0.0, 0.5]), np.array([10.0, 10.0]), time_limit=30.0
    )
    assert kind == "bound"
    # optimum x = (2.5, 0.5): 2.5 + 1.0
    assert lb == pytest.approx(3.5, abs=1e-6)
    assert lb <= 3.5 + 1e-9  # a safe bound never exceeds the true LP value
    assert x.shape == (2,)


def test_d30_rescue_certifies_infeasible_node():
    from discopt.solver import _linear_node_lp_rescue

    pool = _Pool([[1.0, 1.0]], [3.0], [">="])
    kind, lb, x = _linear_node_lp_rescue(
        _box_lp(), 2, pool, np.zeros(2), np.ones(2), time_limit=30.0
    )
    assert (kind, lb, x) == ("infeasible", None, None)


def test_d30_rescue_not_applicable_to_nonlinear_model():
    from discopt.solver import _linear_rescue_std_form

    m = dm.Model("nl")
    x = m.continuous("x", lb=0, ub=2)
    m.minimize(x * x)
    assert _linear_rescue_std_form(m, 1) is False

    lin = dm.Model("lin")
    y = lin.continuous("y", shape=(2,), lb=0, ub=2)
    lin.minimize(y[0] + y[1])
    lin.subject_to(y[0] + y[1] >= 1)
    assert _linear_rescue_std_form(lin, 2) is not False
    assert _linear_rescue_std_form(lin, 3) is False  # lifted layout: not this LP


@pytest.mark.slow
def test_d30_lazy_benders_master_certifies():
    from discopt.callbacks import CutResult

    rng = np.random.default_rng(22)
    n_w, n_c, n_s = 8, 15, 4
    xy_w, xy_c = rng.uniform(0, 100, (n_w, 2)), rng.uniform(0, 100, (n_c, 2))
    dist = np.sqrt(((xy_w[:, None] - xy_c[None]) ** 2).sum(-1))
    base = rng.uniform(10, 40, n_c)
    ang = np.arctan2(xy_c[:, 1] - 50, xy_c[:, 0] - 50)
    d = base * (1 + 0.8 * np.cos(ang[None, :] - 2 * np.pi * np.arange(n_s)[:, None] / n_s))
    cap, f = rng.uniform(150, 300, n_w), rng.uniform(40, 80, n_w)
    c = 13 * 0.5 * dist / 1000
    x_ub = 2 * float(d.max())

    def transport_lp(y_hat, s):
        t = dm.Model("t")
        x = t.continuous("x", shape=(n_w, n_c), lb=0, ub=x_ub)
        t.minimize(
            dm.sum(
                lambda i: dm.sum(lambda j: float(c[i, j]) * x[i, j], over=range(n_c)),
                over=range(n_w),
            )
        )
        for j in range(n_c):
            t.subject_to(dm.sum(lambda i: x[i, j], over=range(n_w)) >= float(d[s, j]), name=f"d{j}")
        for i in range(n_w):
            t.subject_to(
                dm.sum(lambda j: x[i, j], over=range(n_c)) <= float(cap[i] * y_hat[i]),
                name=f"c{i}",
            )
            for j in range(n_c):
                t.subject_to(x[i, j] <= float(d[s, j] * y_hat[i]), name=f"l{i}_{j}")
        r = t.solve()
        if r.status != "optimal":
            return r, None
        L = r.constraint_duals
        u = np.array([float(L[f"d{j}"]) for j in range(n_c)])
        v = np.array([float(L[f"c{i}"]) for i in range(n_w)])
        w = np.array([[float(L[f"l{i}_{j}"]) for j in range(n_c)] for i in range(n_w)])
        return r, (u, v, w)

    m = dm.Model("master")
    y = m.binary("y", shape=(n_w,))
    eta = m.continuous("eta", shape=(n_s,), lb=0, ub=1000.0)
    m.minimize(
        dm.sum(lambda i: float(f[i]) * y[i], over=range(n_w))
        + dm.sum(lambda s: eta[s], over=range(n_s))
    )
    m.subject_to(dm.sum(lambda i: float(cap[i]) * y[i], over=range(n_w)) >= float(d.sum(1).max()))

    def cb(ctx, model):
        xr = np.asarray(ctx.x_relaxation).ravel()
        y_hat, eta_hat = np.round(xr[:n_w]), xr[n_w : n_w + n_s]
        out = []
        for s in range(n_s):
            r, du = transport_lp(y_hat, s)
            if du is not None and r.objective > eta_hat[s] + 1e-6 * max(1.0, r.objective):
                u, v, w = du
                g = -(cap * v + w @ d[s])
                out.append(
                    CutResult(
                        terms=[(eta[s], 1.0)] + [(y[i], float(-g[i])) for i in range(n_w)],
                        sense=">=",
                        rhs=float(d[s] @ u),
                    )
                )
        return out

    r = m.solve(lazy_constraints=cb, time_limit=600)
    assert r.status == "optimal"
    assert r.gap_certified is True
    assert r.objective == pytest.approx(410.298876602442, rel=1e-6)
    assert r.bound <= r.objective + 1e-6
    assert r.bound == pytest.approx(r.objective, rel=1e-4)


# ─────────────────────────── D-31 ───────────────────────────


def _probe_model():
    m = dm.Model("lf")
    F = m.continuous("F", shape=(2,), lb=0, ub=16.8)
    G = m.continuous("G", lb=-5, ub=-1)
    H = m.continuous("H", lb=-3, ub=3)
    return m, F, G, H


def _numeric_curvature(f, lo, hi):
    """Sign of the second difference of a 1-D function on a grid over [lo, hi]."""
    t = np.linspace(lo, hi, 201)
    v = f(t)
    d2 = v[:-2] - 2 * v[1:-1] + v[2:]
    return d2


@pytest.mark.parametrize(
    ("build", "fn", "lo", "hi", "expected"),
    [
        # c*F/(F + a) = c - c*a/(F + a): concave on F >= 0 (saturating rate).
        (lambda F, G: 6.4 * F[0] / (F[0] + 4.8), lambda t: 6.4 * t / (t + 4.8), 0, 16.8, "concave"),
        (
            lambda F, G: -(6.4 * F[0] / (F[0] + 4.8)),
            lambda t: -6.4 * t / (t + 4.8),
            0,
            16.8,
            "convex",
        ),
        (
            lambda F, G: (2 * F[0] + 3.0) / (F[0] + 1.0),
            lambda t: (2 * t + 3) / (t + 1),
            0,
            16.8,
            "convex",
        ),
        # strictly negative denominator: verdicts swap.
        (lambda F, G: G / (G - 1.0), lambda t: t / (t - 1), -5, -1, "concave"),
    ],
)
def test_d31_collinear_linear_fractional_curvature(build, fn, lo, hi, expected):
    m, F, G, _ = _probe_model()
    verdict = classify_expr(build(F, G), m, {})
    want = Curvature.CONVEX if expected == "convex" else Curvature.CONCAVE
    assert verdict == want
    d2 = _numeric_curvature(fn, lo, hi)
    assert d2.size > 0
    assert np.all(d2 >= -1e-12) if expected == "convex" else np.all(d2 <= 1e-12)


@pytest.mark.parametrize(
    "build",
    [
        lambda F, G, H: (F[0] + 1.0) / (F[1] + 1.0),  # different variables: quasi-linear only
        lambda F, G, H: F[0] / (F[0] + G),  # numerator not collinear with denominator
        lambda F, G, H: (H + 1.0) / (H + 4.0 + 0.0 * F[0] - 3.5),  # denominator spans zero
        lambda F, G, H: (2.0 * F[0] + F[1]) / (F[0] + F[1] + 1.0),  # coefficients not proportional
    ],
)
def test_d31_non_collinear_quotients_get_no_claim(build):
    m, F, G, H = _probe_model()
    assert classify_expr(build(F, G, H), m, {}) == Curvature.UNKNOWN


def _reactor():
    V = np.array([4.0, 6.0, 9.0, 14.0])
    kr, CA0, pA = 0.8, 2.0, 5e-3
    Fmax = 1.2 * V
    Fp = np.array([10.0, 16.0, 22.0])
    hp = np.array([3000.0, 3000.0, 2000.0])
    cr = np.array([45.0, 60.0, 80.0, 105.0])
    m = dm.Model("rt")
    y = m.binary("y", shape=(4,))
    F = m.continuous("F", shape=(3, 4), lb=0, ub=float(Fmax.max()))
    m.minimize(
        dm.sum(lambda k: float(cr[k]) * y[k], over=range(4))
        + dm.sum(
            lambda t: (
                float(pA * hp[t])
                * (
                    float(CA0 * Fp[t])
                    - dm.sum(
                        lambda k: float(CA0 * kr * V[k]) * F[t, k] / (F[t, k] + float(kr * V[k])),
                        over=range(4),
                    )
                )
            ),
            over=range(3),
        )
    )
    for t in range(3):
        m.subject_to(dm.sum(lambda k: F[t, k], over=range(4)) == float(Fp[t]))
        for k in range(4):
            m.subject_to(F[t, k] <= float(Fmax[k]) * y[k])
    m.subject_to(dm.sum(lambda k: float(Fmax[k]) * y[k], over=range(4)) >= float(Fp.max()))
    m.first_stage(y)
    return m


def test_d31_reactor_objective_classified_convex():
    assert classify_oa_cut_convexity(_reactor()).objective_is_convex is True


def test_d31_gbd_certifies_convex_reactor():
    r = _reactor().solve(decomposition="benders")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(745.439, abs=1e-3)
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-6
    assert r.bound == pytest.approx(r.objective, rel=1e-4)
