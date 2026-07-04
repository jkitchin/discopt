"""Bilevel Phase 2: strong-duality reformulation + convex-QP lower levels.

Validates, without the global solver, that (a) the strong-duality reformulation's
aggregate equality `Σ μ_i g_i == 0` is satisfied exactly at the follower optimum
and is equivalent to the per-row KKT complementarity for an LP lower level, and
(b) the lifted convexity gate accepts convex-QP lower levels (with KKT still
correct there) while refusing nonconvex / non-quadratic / nonlinear-equality
lower levels. Needs JAX + scipy; no Rust extension.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
scipy_opt = pytest.importorskip("scipy.optimize")
import jax.numpy as jnp  # noqa: E402
from discopt._jax.dag_compiler import compile_expression_params  # noqa: E402
from discopt.bilevel import BilevelProblem  # noqa: E402
from discopt.modeling.core import FunctionCall, Model  # noqa: E402

pytestmark = pytest.mark.smoke


def _eval(body, model, values):
    flat = [float(values.get(v, 0.0)) for v in model._variables]
    params = tuple(jnp.asarray(p.value) for p in model._parameters)
    return float(compile_expression_params(body, model)(jnp.asarray(flat), params))


def _lp_follower(d, A, b, bounds):
    d, A, b = np.asarray(d, float), np.asarray(A, float), np.asarray(b, float)
    res = scipy_opt.linprog(c=d, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    assert res.success, res.message
    y = res.x
    active = np.where(np.abs(b - A @ y) < 1e-7)[0]
    mu = np.zeros(len(b))
    if len(active):
        mu[active] = np.linalg.lstsq(A[active].T, -d, rcond=None)[0]
    return y, mu


# ---------------------------------------------------------------------------
# 1. Strong-duality reformulation.
# ---------------------------------------------------------------------------


def _lp_instance(method):
    """Follower: min_y y  s.t.  x + y >= 3,  y <= 2x,  y in [0,10]."""
    m = Model("sd")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - 4 * y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[x + y >= 3, y <= 2 * x],
    )
    bl.formulate(method=method)
    d, A = [1.0], [[-1.0], [1.0]]
    b_of_x = lambda xv: [xv - 3.0, 2.0 * xv]  # noqa: E731
    return m, x, y, bl, d, A, b_of_x


def test_strong_duality_structure():
    m, x, y, bl, *_ = _lp_instance("strong_duality")
    assert bl.strong_duality is not None
    assert bl.kkt is None
    # 2 user constraints + 2 finite follower bounds (y in [0,10])
    assert len(bl.strong_duality.multipliers) == 4
    assert len(bl.strong_duality.stationarity) == 1
    assert bl.strong_duality.strong_duality.name.endswith("strong_duality")
    # no disjunction / either-or constraints were emitted (single bilinear equality)
    assert not any("either" in (c.name or "") for c in m._constraints)


def test_strong_duality_satisfied_at_follower_optimum():
    m, x, y, bl, d, A, b_of_x = _lp_instance("strong_duality")
    for xv in [1.2, 1.8, 2.5]:
        y_star, mu = _lp_follower(d, A, b_of_x(xv), bounds=[(0.0, 10.0)])
        assign = {x: xv, y: float(y_star[0])}
        for k, muv in enumerate(mu):  # user-constraint duals; bound duals are 0 here
            assign[bl.strong_duality.multipliers[k]] = float(muv)
        # stationarity == 0 and the aggregate strong-duality equality == 0
        assert abs(_eval(bl.strong_duality.stationarity[0].body, m, assign)) < 1e-7
        assert abs(_eval(bl.strong_duality.strong_duality.body, m, assign)) < 1e-7
        assert np.all(mu >= -1e-9)


def test_kkt_and_strong_duality_agree_on_lp():
    """Both encodings are satisfied at the same follower optimum (LP)."""
    m_k, xk, yk, bl_k, d, A, b_of_x = _lp_instance("kkt")
    m_s, xs, ys, bl_s, *_ = _lp_instance("strong_duality")
    xv = 1.7
    y_star, mu = _lp_follower(d, A, b_of_x(xv), bounds=[(0.0, 10.0)])

    ak = {xk: xv, yk: float(y_star[0])}
    for i, muv in enumerate(mu):
        ak[bl_k.kkt.multipliers[i]] = float(muv)
    for p in bl_k.kkt.comp_pairs:  # per-row complementarity ~ 0 (bound duals = 0)
        assert abs(_eval(p.f, m_k, ak) * _eval(p.g, m_k, ak)) < 1e-7

    as_ = {xs: xv, ys: float(y_star[0])}
    for i, muv in enumerate(mu):
        as_[bl_s.strong_duality.multipliers[i]] = float(muv)
    assert abs(_eval(bl_s.strong_duality.strong_duality.body, m_s, as_)) < 1e-7  # aggregate ~ 0


# ---------------------------------------------------------------------------
# 2. Convex-QP lower level (the lifted gate) — KKT still correct.
# ---------------------------------------------------------------------------


def test_convex_qp_lower_level_accepted_and_kkt_correct():
    # Follower: min_y  y^2 - 2 x y   s.t.  y <= 1,  y in [0,10].
    # Unconstrained min is y=x; for x>1 the bound y<=1 binds -> y*=1, μ=2x-2>0.
    m = Model("qp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-y - x)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y * y - 2 * x * y,  # convex in y (Hessian = 2)
        lower_constraints=[y <= 1],
    )
    bl.formulate(method="kkt")  # gate must accept convex-QP
    assert bl.kkt is not None and len(bl.kkt.stationarity) == 1

    for xv in [1.5, 2.0, 3.0]:
        y_star = 1.0  # bound binds for x>1
        mu = 2.0 * xv - 2.0 * y_star  # from stationarity 2y - 2x + μ = 0
        assign = {x: xv, y: y_star, bl.kkt.multipliers[0]: mu}
        # stationarity 2y - 2x + μ == 0
        assert abs(_eval(bl.kkt.stationarity[0].body, m, assign)) < 1e-7
        # complementarity μ ⊥ (1 - y): 1 - y* = 0 here, so satisfied
        p = bl.kkt.comp_pairs[0]
        assert abs(_eval(p.f, m, assign) * _eval(p.g, m, assign)) < 1e-7
        assert mu >= -1e-9


def test_convex_qp_via_strong_duality_accepted():
    m = Model("qp2")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y * y - 2 * x * y,
        lower_constraints=[y <= 1],
    )
    bl.formulate(method="strong_duality")
    assert bl.strong_duality is not None


# ---------------------------------------------------------------------------
# 2b. Follower variable bounds enter the KKT (correctness at an active bound).
# ---------------------------------------------------------------------------


def test_active_follower_bound_gets_its_multiplier():
    # Follower: min_y y  s.t.  y <= 8,  y in [2, 10]. Optimum y*=2 sits on the
    # lower bound, so the reformulation MUST carry a bound multiplier or it would
    # exclude the true follower optimum.
    m = Model("bnd")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=2, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m, upper_vars=[x], lower_vars=[y], lower_objective=y, lower_constraints=[y <= 8]
    )
    bl.formulate(method="kkt")
    # constraints: [y<=8 (user), 2-y<=0 (lb), y-10<=0 (ub)]
    assert len(bl.lower_constraints_full) == 3
    # at y*=2 the lower bound is active: μ_lb = 1, others 0 -> stationarity 1+μ0-μ_lb+μ_ub = 0
    mu_lb = bl.kkt.multipliers[1]  # order: user, lb, ub
    assign = {x: 5.0, y: 2.0, mu_lb: 1.0}
    assert abs(_eval(bl.kkt.stationarity[0].body, m, assign)) < 1e-9
    # its complementarity 0 <= μ_lb ⊥ (y - 2) >= 0 holds (y-2 = 0)
    lb_pair = bl.kkt.comp_pairs[1]
    assert abs(_eval(lb_pair.f, m, assign) * _eval(lb_pair.g, m, assign)) < 1e-9


def test_without_bound_handling_active_bound_optimum_is_excluded():
    # The same instance with the fix OFF: stationarity 1 + μ0 == 0 has no μ0 >= 0
    # solution, so y*=2 would be (wrongly) infeasible in the reformulation.
    m = Model("bnd_off")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=2, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[y <= 8],
        include_follower_bounds=False,
    )
    bl.formulate(method="kkt")
    assert len(bl.lower_constraints_full) == 1
    # stationarity is 1 + μ0; with the only multiplier μ0 >= 0 it can never be 0.
    for mu0 in (0.0, 5.0):
        assign = {x: 5.0, y: 2.0, bl.kkt.multipliers[0]: mu0}
        assert _eval(bl.kkt.stationarity[0].body, m, assign) >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# 3. Lifted-gate refusals.
# ---------------------------------------------------------------------------


def test_nonconvex_qp_refused():
    # min_{y1,y2} y1*y2 has an indefinite Hessian [[0,1],[1,0]] -> nonconvex.
    m = Model("ncx")
    x = m.continuous("x", lb=0, ub=10)
    y1 = m.continuous("y1", lb=0, ub=10)
    y2 = m.continuous("y2", lb=0, ub=10)
    m.minimize(x - y1 - y2)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y1, y2],
        lower_objective=y1 * y2,  # indefinite in (y1,y2)
        lower_constraints=[y1 + y2 <= 4],
    )
    with pytest.raises(NotImplementedError, match="nonconvex"):
        bl.formulate(method="kkt")


def test_nonquadratic_lower_refused_pending_certifier():
    m = Model("nq")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0.1, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=FunctionCall("exp", y),  # exp(y): convex but non-quadratic
        lower_constraints=[y <= 5],
    )
    with pytest.raises(NotImplementedError, match="not-quadratic|certifier"):
        bl.formulate(method="kkt")


def test_nonlinear_equality_refused():
    m = Model("nle")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x - y)
    bl = BilevelProblem(
        m,
        upper_vars=[x],
        lower_vars=[y],
        lower_objective=y,
        lower_constraints=[y * y == x],  # nonlinear equality -> nonconvex feasible set
    )
    with pytest.raises(NotImplementedError, match="affine in the follower"):
        bl.formulate(method="kkt")
